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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from rlbot.act import MAX_SECONDS, Policy, device, run_act                 # noqa: E402
from rlbot.grasp import hold_everything                                      # noqa: E402
from rlbot.skills import Robot                                               # noqa: E402
from rlbot.teleop import task_text                                           # noqa: E402
from collect_demos import Layout                                             # noqa: E402

FPS = 60


def rollout(robot: Robot, policy: Policy, layout: Layout, obj: str, hz: int,
            viewer=None, speed: float = 1.0) -> tuple[bool, dict]:
    """A fresh random layout, then `rlbot.act.run_act`, paced for the viewer if there is one."""
    m, d = robot.model, robot.data
    mujoco.mj_resetData(m, d)
    where = layout.apply(*layout.sample())
    mujoco.mj_forward(m, d)
    hold_everything(m, d)
    robot.rig.seconds(0.4)

    on_step = None
    if viewer is not None:
        sync_every = max(1, round(1 / (FPS * m.opt.timestep)))
        clock = [time.time()]

        def on_step(step: int) -> bool:
            if step % sync_every:
                return True
            if not viewer.is_running():
                return False
            viewer.sync()
            ahead = sync_every * m.opt.timestep / speed - (time.time() - clock[0])
            if ahead > 0:
                time.sleep(ahead)
            clock[0] = time.time()
            return True

    ok, result = run_act(robot, policy, obj, hz, MAX_SECONDS, on_step)
    return ok, dict(layout=where, **result)


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

    robot = Robot(args.table)
    obj = next(n for n, i in robot.items.items() if i.graspable)
    policy = Policy(Path(args.ckpt), robot.model, device(), args.mode,
                    hide=[g for hand in robot.hands.values() for g in hand.pads])
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
