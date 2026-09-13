"""Train ACT on a folder of demonstrations, on this machine.

    .venv/bin/python scripts/train_act.py --data out/demos/ball --out out/act/ball
    .venv/bin/python scripts/train_act.py --data out/demos/ball --out out/act/ball --steps 20000

Runs to completion or until killed, and picks up where it left off: every
`--ckpt-every` steps the model, optimiser, step count and normalisation are
written to `last.pt` (atomically, so a kill mid-write leaves the previous one
intact), and starting again with the same `--out` resumes from it.  The best
validation loss so far is kept as `best.pt`.  `train.log` gets a line per log
interval; `progress.json` the latest numbers.

Sized for an Apple-silicon laptop: frames at 128 px in memory, a 256-wide
transformer, batch 8, float32 on the MPS backend.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlbot.act import (ACT, Demos, act_loss, load_checkpoint, prep_images,   # noqa: E402
                       random_shift, save_checkpoint, write_json)


def device_name() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def evaluate(model, demos: Demos, norm, device, rng, chunk, batch, batches=8):
    model.eval()
    l1s = []
    with torch.no_grad():
        for _ in range(batches):
            imgs, state, acts, pads = demos.batch(rng, batch, chunk)
            state = ((state - norm["state_mean"]) / norm["state_std"]).to(device)
            acts_n = ((acts - norm["action_mean"]) / norm["action_std"]).to(device)
            pred, _, _ = model(prep_images(imgs, device), state)
            l1 = (pred - acts_n).abs().mean(-1)
            l1s.append(float((l1 * ~pads.to(device)).sum() / (~pads).sum()))
    model.train()
    return float(np.mean(l1s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[str(REPO / "out" / "demos" / "ball")],
                    help="one or more demo folders with a manifest.json")
    ap.add_argument("--out", default=str(REPO / "out" / "act" / "ball"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=32, help="actions per prediction")
    ap.add_argument("--img", type=int, default=128, help="frame size held in memory")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--kl", type=float, default=10.0, help="KL weight, as in ACT")
    ap.add_argument("--val", type=int, default=8, help="episodes held out")
    ap.add_argument("--shift", type=int, default=6, help="px of random image shift; 0 off")
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "train.log", "a")

    def log(msg: str) -> None:
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    device = torch.device(device_name())
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    items, task = [], None
    for root in map(Path, args.data):
        manifest = json.loads((root / "manifest.json").read_text())
        task = task or manifest["task"]
        items += [(root, e) for e in manifest["episodes"]]
    items.sort(key=lambda it: it[1]["episode"])
    train_items, val_items = items[:-args.val], items[-args.val:]
    log(f"device {device}; {len(train_items)} train / {len(val_items)} val episodes "
        f"from {', '.join(args.data)}; task: {task!r}")
    train = Demos(train_items, args.img, log=log)
    val = Demos(val_items, args.img, log=log)
    log(f"{train.rows} train rows, {val.rows} val rows, frames at {args.img} px")

    config = dict(chunk=args.chunk, img=args.img, kl=args.kl, batch=args.batch)
    model = ACT(chunk=args.chunk).to(device)
    backbone = [p for n, p in model.named_parameters() if n.startswith("backbone")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    optim = torch.optim.AdamW([{"params": rest, "lr": args.lr},
                               {"params": backbone, "lr": args.lr_backbone}],
                              weight_decay=1e-4)
    norm = train.norm
    step, best_val = 0, float("inf")
    last = out / "last.pt"
    if last.exists():
        ck = load_checkpoint(last, device)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        step, best_val, norm = ck["step"], ck["best_val"], ck["norm"]
        log(f"resumed from {last} at step {step} (best val {best_val:.4f})")
    write_json(out / "config.json", dict(config, data=args.data, steps=args.steps,
                                          shift=args.shift,
                                          train=[e["episode"] for _, e in train_items],
                                          val=[e["episode"] for _, e in val_items]))

    norm_dev = {k: v.to(device) for k, v in norm.items()}
    model.train()
    clock, since = time.time(), step
    running = {"loss": 0.0, "l1": 0.0, "kl": 0.0, "n": 0}
    while step < args.steps:
        imgs, state, acts, pads = train.batch(rng, args.batch, args.chunk)
        if args.shift:
            imgs = random_shift(imgs, args.shift, rng)
        state = (state.to(device) - norm_dev["state_mean"]) / norm_dev["state_std"]
        acts = (acts.to(device) - norm_dev["action_mean"]) / norm_dev["action_std"]
        pads = pads.to(device)
        pred, mu, logvar = model(prep_images(imgs, device), state, acts)
        loss, l1, kl = act_loss(pred, acts, pads, mu, logvar, args.kl)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        step += 1
        for k, v in (("loss", loss), ("l1", l1), ("kl", kl)):
            running[k] += float(v)
        running["n"] += 1

        if step % args.log_every == 0:
            rate = (time.time() - clock) / max(1, step - since)
            eta = rate * (args.steps - step)
            log(f"step {step:6d}  loss {running['loss']/running['n']:.4f}  "
                f"l1 {running['l1']/running['n']:.4f}  kl {running['kl']/running['n']:.3f}  "
                f"{rate:.2f}s/step  eta {eta/3600:.1f}h")
            running = {"loss": 0.0, "l1": 0.0, "kl": 0.0, "n": 0}
        if step % args.ckpt_every == 0 or step == args.steps:
            v = evaluate(model, val, norm, device, np.random.default_rng(1),
                         args.chunk, args.batch)
            improved = v < best_val
            best_val = min(best_val, v)
            save_checkpoint(last, model, optim, step, norm, config, best_val)
            if improved:
                save_checkpoint(out / "best.pt", model, optim, step, norm, config, best_val)
            write_json(out / "progress.json", dict(step=step, steps=args.steps,
                                                    val_l1=v, best_val_l1=best_val,
                                                    sec_per_step=(time.time() - clock) / max(1, step - since)))
            log(f"  val l1 {v:.4f} (best {best_val:.4f}){'  *' if improved else ''}  "
                f"-> {last.name}")
    log("done")


if __name__ == "__main__":
    main()
