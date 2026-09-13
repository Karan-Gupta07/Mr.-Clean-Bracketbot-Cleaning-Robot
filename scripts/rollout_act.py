"""Run a trained ACT policy in the simulator and score it.

    .venv/bin/python scripts/rollout_act.py --episodes 10                # headless, scored
    .venv/bin/mjpython scripts/rollout_act.py --episodes 3 --view        # watch it live
    .venv/bin/python scripts/rollout_act.py --ckpt out/act/ball/last.pt

Closed loop, at the recorder's 20 Hz.  Each tick renders the three cameras
the policy was trained on, reads the 16 joint states, and asks the policy for
a chunk of future commands; the command applied is the temporal ensemble of
every chunk that has predicted this tick, newest weighted least, as in the
ACT paper.  The layout is randomised the way the collector randomises it, so
these are layouts the policy has never seen.

Every rollout is written next to the checkpoint as an episode file that
`scripts/replay_demo.py` and `scripts/render_demos.py` understand, so a run
can be watched again or turned into frames.  Success is the collector's own
test: the ball resting on the crate floor, clear of the walls, out of the hand.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from rlbot.act import ACT, CAMERAS, load_checkpoint, prep_images            # noqa: E402
from rlbot.arm import GRIPPER                                                # noqa: E402
from rlbot.grasp import hold_everything, in_hand                             # noqa: E402
from rlbot.skills import Robot                                               # noqa: E402
from rlbot.teleop import Jog, task_text                                      # noqa: E402
from collect_demos import Layout                                             # noqa: E402

RENDER_PX = 224            # what the training frames were rendered at
ENSEMBLE_M = 0.01          # ACT's temporal-ensemble decay
MAX_SECONDS = 25.0
FPS = 60


class Policy:
    """The checkpoint, plus everything between the sim and its tensors."""

    def __init__(self, ckpt: Path, model: mujoco.MjModel, device, mode: str = "ensemble"):
        self.mode = mode          # ensemble | newest | open-loop
        ck = load_checkpoint(ckpt, device)
        self.config, self.norm = ck["config"], {k: v.to(device) for k, v in ck["norm"].items()}
        self.chunk = self.config["chunk"]
        self.net = ACT(chunk=self.chunk).to(device)
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.device = device
        self.step_trained = ck["step"]
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, RENDER_PX)
        model.vis.global_.offheight = max(model.vis.global_.offheight, RENDER_PX)
        self.renderer = mujoco.Renderer(model, RENDER_PX, RENDER_PX)
        self.history: dict[int, list[tuple[int, torch.Tensor]]] = {}
        self.plan, self.plan_at = None, 0

    def observe(self, data) -> torch.Tensor:
        """(1, cam, 3, img, img) uint8, downsized the way training did it."""
        frames = []
        for cam in CAMERAS:
            self.renderer.update_scene(data, camera=cam)
            frames.append(torch.from_numpy(self.renderer.render().copy()))
        x = torch.stack(frames).permute(0, 3, 1, 2).float()          # (cam, 3, H, W)
        img = self.config["img"]
        x = F.interpolate(x, size=(img, img), mode="area").round().to(torch.uint8)
        return x[None]

    @torch.no_grad()
    def act(self, tick: int, imgs, state: np.ndarray) -> np.ndarray:
        if self.mode == "open-loop" and self.plan is not None and tick < self.plan_at + self.chunk:
            return self.plan[tick - self.plan_at].numpy()     # ride the chunk out
        s = (torch.from_numpy(state).float().to(self.device)[None] - self.norm["state_mean"]) / self.norm["state_std"]
        pred, _, _ = self.net(prep_images(imgs, self.device), s)
        chunk = (pred[0] * self.norm["action_std"] + self.norm["action_mean"]).cpu()
        if self.mode != "ensemble":
            self.plan, self.plan_at = chunk, tick
            return chunk[0].numpy()
        for k in range(self.chunk):
            self.history.setdefault(tick + k, []).append((tick, chunk[k]))
        # everything that has an opinion about this tick, newest weighted least
        preds = self.history.pop(tick)
        w = torch.tensor([math.exp(-ENSEMBLE_M * (tick - t0)) for t0, _ in preds])
        acts = torch.stack([a for _, a in preds])
        return ((w[:, None] * acts).sum(0) / w.sum()).numpy()

    def reset(self) -> None:
        self.history.clear()
        self.plan, self.plan_at = None, 0


def read_state(robot: Robot) -> np.ndarray:
    d, m = robot.data, robot.model
    out = []
    for side in ("right", "left"):
        out.append(d.qpos[robot.arms[side].ik.qadr])
        out.append([d.qpos[m.jnt_qposadr[m.joint(GRIPPER[side]).id]]])
    return np.concatenate(out).astype(np.float32)


def apply_action(robot: Robot, a: np.ndarray) -> None:
    d, m = robot.data, robot.model
    for i, side in enumerate(("right", "left")):
        arm = robot.arms[side]
        cmd = a[8 * i:8 * i + 7]
        lo, hi = m.actuator_ctrlrange[arm.acts].T
        d.ctrl[arm.acts] = np.clip(cmd, lo, hi)
        d.ctrl[arm.grip_act] = float(np.clip(a[8 * i + 7], 0.0, 1.0))


def settled_in_crate(robot: Robot, obj: str) -> bool:
    m, d = robot.model, robot.data
    if not robot.inside_crate(d.body(obj).xpos):
        return False
    if any(in_hand(m, d, obj, s, robot.hands[s]) for s in ("right", "left")):
        return False
    crate = m.body(robot.crate).id
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == crate]
    obj_geom = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(obj).id)
    fromto = np.zeros(6)
    dist = [mujoco.mj_geomDistance(m, d, obj_geom, g, 0.2, fromto) for g in geoms]
    return dist[0] < 0.002 and all(w > 0.0 for w in dist[1:])


def rollout(robot: Robot, policy: Policy, layout: Layout, obj: str, hz: int,
            viewer=None, speed: float = 1.0) -> tuple[bool, dict]:
    m, d = robot.model, robot.data
    mujoco.mj_resetData(m, d)
    where = layout.apply(*layout.sample())
    mujoco.mj_forward(m, d)
    hold_everything(m, d)
    robot.rig.seconds(0.4)
    # Start where the demonstrations start.  Every recording begins after the
    # collector's ready move, with the working arm hovering over the table;
    # handed the rest pose instead, the policy swings the arm through the
    # ball on its way to somewhere it recognises.
    side = "left" if robot.across(obj) > 0 else "right"
    jog = Jog(robot, side, robot.items[obj].width)
    jog.ready()
    while jog.busy:
        for _ in range(25):
            jog.step()
            robot.rig.step()
    robot.rig.seconds(0.3)
    policy.reset()

    every = max(1, round(1 / (hz * m.opt.timestep)))
    sync_every = max(1, round(1 / (FPS * m.opt.timestep)))
    rows = {"time": [], "qpos": [], "obj_pos": [], "obj_quat": [], "action": [], "state": []}
    bodies = [m.body(n).id for n in robot.items]
    clock, tick, step = time.time(), 0, 0
    while d.time < MAX_SECONDS:
        if step % every == 0:
            state = read_state(robot)
            a = policy.act(tick, policy.observe(d), state)
            apply_action(robot, a)
            rows["time"].append(float(d.time)); rows["qpos"].append(d.qpos.copy())
            rows["obj_pos"].append(np.array([d.xpos[b] for b in bodies]))
            rows["obj_quat"].append(np.array([d.xquat[b] for b in bodies]))
            rows["action"].append(a); rows["state"].append(state)
            tick += 1
            if d.body(obj).xpos[2] < 0.4:
                break                                   # on the floor
        robot.rig.step()
        step += 1
        if viewer is not None and step % sync_every == 0:
            if not viewer.is_running():
                break
            viewer.sync()
            ahead = sync_every * m.opt.timestep / speed - (time.time() - clock)
            if ahead > 0:
                time.sleep(ahead)
            clock = time.time()
    ok = settled_in_crate(robot, obj)
    note = "in the crate" if ok else robot.where(obj)
    return ok, dict(layout=where, note=note, rows={k: np.array(v) for k, v in rows.items()})


def save_rollout(path: Path, robot: Robot, policy: Policy, obj: str, ok: bool,
                 result: dict, hz: int, joints: dict) -> None:
    rows = result["rows"]
    meta = dict(task=task_text(obj), target=obj, arm="policy", table=robot.table.name,
                balancing=robot.balancing, control_hz=hz, objects=list(robot.items),
                joints=joints, timestep=robot.model.opt.timestep, success=bool(ok),
                rows=len(rows["time"]), seconds=float(rows["time"][-1] - rows["time"][0]),
                layout=result["layout"], note=result["note"], operator="act",
                checkpoint_step=policy.step_trained, events=[])
    np.savez_compressed(path, meta=json.dumps(meta), **rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "out" / "act" / "ball" / "best.pt"))
    ap.add_argument("--table", default="ball")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--hz", type=int, default=20)
    ap.add_argument("--view", action="store_true", help="live viewer (mjpython)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="where rollouts go; default beside the checkpoint")
    ap.add_argument("--mode", default="ensemble", choices=["ensemble", "newest", "open-loop"],
                    help="temporal ensemble (ACT), first action of the newest chunk, "
                         "or run each chunk out before re-planning")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    robot = Robot(args.table)
    obj = next(n for n, i in robot.items.items() if i.graspable)
    policy = Policy(Path(args.ckpt), robot.model, device, args.mode)
    rng = np.random.default_rng(args.seed)
    layout = Layout(robot, obj, rng)
    out = Path(args.out or Path(args.ckpt).parent / "rollouts")
    out.mkdir(parents=True, exist_ok=True)
    m = robot.model
    bodies = [m.body(n).id for n in robot.items]
    joints = {m.joint(j).name: (int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), int(m.jnt_type[j]))
              for j in range(m.njnt) if m.joint(j).name and m.jnt_bodyid[j] not in bodies}
    print(f"{Path(args.ckpt).name} at step {policy.step_trained}; {args.episodes} rollouts on "
          f"{robot.table.name}, seed {args.seed}", flush=True)

    def run(viewer=None):
        wins = 0
        for i in range(args.episodes):
            ok, result = rollout(robot, policy, layout, obj, args.hz, viewer, args.speed)
            wins += ok
            path = out / f"rollout_{args.mode}_{args.seed}_{i:03d}.npz"
            save_rollout(path, robot, policy, obj, ok, result, args.hz, joints)
            lay = result["layout"]
            print(f"  {i + 1:3d}  {'ok  ' if ok else 'FAIL'} {result['note']:<14} "
                  f"ball ({lay['object_at'][0]:+.2f}, {lay['object_at'][1]:+.2f}) "
                  f"crate ({lay['crate_at'][0]:+.2f}, {lay['crate_at'][1]:+.2f})  "
                  f"{result['rows']['time'][-1]:.1f}s  -> {path.name}", flush=True)
            if viewer is not None and not viewer.is_running():
                break
        print(f"\n{wins}/{args.episodes} in the crate", flush=True)

    if args.view:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(m, robot.data) as viewer:
            rot = robot.data.body("root").xmat.reshape(3, 3)
            viewer.cam.lookat[:] = robot.data.body("root").xpos + rot[:, 0] * 0.45 + np.array([0, 0, 0.3])
            viewer.cam.distance, viewer.cam.elevation = 1.6, -25
            viewer.cam.azimuth = math.degrees(math.atan2(rot[1, 0], rot[0, 0])) + 150
            run(viewer)
            print("done - close the window to finish", flush=True)
            while viewer.is_running():
                time.sleep(0.1)
    else:
        run()


if __name__ == "__main__":
    main()
