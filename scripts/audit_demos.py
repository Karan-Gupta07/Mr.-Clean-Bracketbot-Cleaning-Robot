"""Audit a folder of collected demonstrations and write its manifest.

    .venv/bin/python scripts/audit_demos.py out/demos/ball_more
    .venv/bin/python scripts/audit_demos.py out/demos/ball --keep 80

Every episode must have its frames file with one image per row per camera.
Episodes are then cut for the things that made bad demonstrations in the
first sets: hovering for over 35 s, the object losing contact with the pads
and coming back, the object nudged over 1.5 cm before the grasp, a joint
jumping faster than 3 rad/s, or the working wrist camera not seeing the
object at the start.  Cut episodes move to `rejected/` with a `why.json`.
With `--keep`, the shortest N survivors are kept and the rest go the same
way, marked surplus.  The survivors are listed in `manifest.json`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlbot.teleop import load_demo                                   # noqa: E402

RED = lambda im: int(((im[..., 0] > 150) & (im[..., 1] < 110) & (im[..., 2] < 110)).sum())  # noqa: E731


def audit(ep: Path) -> tuple[dict, list[str]]:
    meta, a = load_demo(ep)
    fr = ep.with_name(ep.stem + "_frames.npz")
    why = []
    if not fr.exists():
        return meta, ["no frames"]
    with np.load(fr) as z:
        if not all(z[c].shape[0] == meta["rows"] for c in z["cameras"]):
            why.append("frames mismatch")
        wrist0 = RED(z[f"wrist_{meta['arm']}_cam"][0].astype(int))
    held = a["held"] >= 0
    first = int(np.argmax(held)) if held.any() else 0
    obj = a["obj_pos"][:, 0]
    arm = a["arm_right"] if meta["arm"] == "right" else a["arm_left"]
    if meta["seconds"] > 35:
        why.append("long")
    if max(0, int(np.sum(np.diff(held.astype(int)) == 1)) - 1) > 0:
        why.append("regrasp")
    if np.linalg.norm(obj[first, :2] - obj[0, :2]) * 100 > 1.5:
        why.append("pushed")
    if np.abs(np.diff(arm, axis=0)).max() * meta["control_hz"] > 3.0:
        why.append("swing")
    if wrist0 < 30:
        why.append("object not in wrist camera")
    return meta, why


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--keep", type=int, default=None, help="keep the N shortest survivors")
    args = ap.parse_args()
    root = Path(args.folder)
    rejected = root / "rejected"
    rejected.mkdir(exist_ok=True)

    kept, cut = [], []
    for ep in sorted(Path(p) for p in glob.glob(str(root / "ep_*.npz"))
                     if not p.endswith("_frames.npz")):
        meta, why = audit(ep)
        (cut if why else kept).append((ep, meta, why))
    kept.sort(key=lambda k: k[1]["seconds"])
    if args.keep is not None and len(kept) > args.keep:
        cut += [(ep, meta, ["surplus, longest"]) for ep, meta, _ in kept[args.keep:]]
        kept = kept[:args.keep]

    for ep, _, _ in cut:
        for f in (ep, ep.with_name(ep.stem + "_frames.npz")):
            if f.exists():
                shutil.move(str(f), rejected / f.name)
    old = json.loads((rejected / "why.json").read_text()) if (rejected / "why.json").exists() else []
    (rejected / "why.json").write_text(json.dumps(old + [(ep.name, why) for ep, _, why in cut], indent=1))

    episodes = [dict(episode=ep.name, frames=ep.stem + "_frames.npz", task=meta["task"],
                     arm=meta["arm"], rows=meta["rows"], seconds=round(meta["seconds"], 2),
                     layout=meta["layout"]) for ep, meta, _ in kept]
    manifest = dict(task=episodes[0]["task"] if episodes else "", table=kept[0][1]["table"] if kept else "",
                    control_hz=kept[0][1]["control_hz"] if kept else 20,
                    cameras=["head_cam", "wrist_right_cam", "wrist_left_cam"], size=224,
                    episodes=episodes)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    reasons = {}
    for _, _, why in cut:
        for w in why:
            reasons[w] = reasons.get(w, 0) + 1
    print(f"{root}: kept {len(kept)} (right {sum(e['arm'] == 'right' for e in episodes)}, "
          f"left {sum(e['arm'] == 'left' for e in episodes)}), cut {len(cut)}: "
          + ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
