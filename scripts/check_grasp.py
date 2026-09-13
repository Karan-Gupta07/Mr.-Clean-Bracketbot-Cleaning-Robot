"""Close the fingers on every object in the room and try to lift it.

    .venv/bin/python scripts/check_grasp.py                 # all three tables
    .venv/bin/python scripts/check_grasp.py --table pick
    .venv/bin/python scripts/check_grasp.py --balance       # on the wheels, balancing

`build_room.py` proves the arm can be *driven* to each object.  That is
kinematics, and kinematics has never held onto anything.  This runs the grasp:
approach from 0.12 m above, descend, close until the pads load up, lift, and
check the object came up with the hand and is still in the jaw afterwards.

By default the base is welded to the floor at the docking pose, so a failure is
the grasp's fault and not the balancer's.  `--balance` puts it back on its wheels
with the station keeper running, which is the honest version of the same test.

The motions themselves live in `src/rlbot/grasp.py`, shared with the agent.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlbot.arm import GRIPPER, Arm, Gripper, MeshGripper              # noqa: E402
from rlbot.grasp import (HOLD_SECONDS, LIFTED, Rig, StationDriver,   # noqa: E402
                         hold_everything, in_hand, move, plan_waypoints,
                         squeeze, welded_at)
from rlbot.gripper_pads import bare_grippers
from rlbot.robot import ROOM                                         # noqa: E402
from rlbot.room import TABLES, grasp_pose                            # noqa: E402


@dataclass
class Result:
    item: str
    side: str
    rose: float        # m the object actually came up
    held: bool

    @property
    def ok(self) -> bool:
        return self.held and self.rose > LIFTED


def attempt(item, table, balance: bool, verbose=True, bare=False) -> Result:
    x, y, yaw = table.dock
    if balance:
        spec = mujoco.MjSpec.from_file(str(ROOM))
        spec.option.impratio = 200
        model = bare_grippers(spec).compile() if bare else spec.compile()
        data = mujoco.MjData(model)
        key = model.key(f"dock_{table.name.split('_')[1]}").id
        mujoco.mj_resetDataKeyframe(model, data, key)
    else:
        model = welded_at(x, y, yaw, bare=bare)
        data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hand_type = MeshGripper if bare else Gripper

    plan = plan_waypoints(model, mujoco.MjData(model), grasp_pose(item, table),
                          item.width, item.yaws, yaw, data.qpos.copy(),
                          hand_type=hand_type)
    if plan is None:
        return Result(item.name, "-", 0.0, False)
    _, side, wrist, opening, (above, on, up) = plan

    hold_everything(model, data)
    rig = Rig(model, data, driver=StationDriver(model, data) if balance else None)
    arm = Arm(model, data, side)
    arm.grip(opening)          # only as wide as this object needs
    rig.seconds(0.5)

    start_z = float(data.body(item.name).xpos[2])
    move(rig, arm, above.qpos, 1.6)        # over the object, fingers open
    move(rig, arm, on.qpos, 1.0)           # down around it
    squeeze(rig, arm)                       # close until the pads load up
    move(rig, arm, up.qpos, 1.2)           # lift
    rig.seconds(HOLD_SECONDS)

    hand = hand_type(model, side)
    rose = float(data.body(item.name).xpos[2]) - start_z
    held = in_hand(model, data, item.name, side, hand)
    span = float(np.interp(data.qpos[model.jnt_qposadr[model.joint(GRIPPER[side]).id]],
                           hand.q, hand.gap))
    if verbose:
        flag = "ok  " if (held and rose > LIFTED) else "FAIL"
        print(f"  {flag} {item.name:<12} {side:>5} arm, wrist "
              f"{math.degrees(wrist):3.0f} deg   rose {rose * 1000:+6.1f} mm   "
              f"jaw {span * 1000:5.1f} mm   {'holding' if held else 'empty'}")
    return Result(item.name, side, rose, held)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="all",
                    choices=["all"] + [t.name.split("_")[1] for t in TABLES])
    ap.add_argument("--item", help="just this object, by name")
    ap.add_argument("--bare", action="store_true",
                    help="Test the supplied blades with no contact pads")
    ap.add_argument("--balance", action="store_true",
                    help="stand on the wheels with the station keeper running")
    args = ap.parse_args()

    tables = [t for t in TABLES if args.table in ("all", t.name.split("_")[1])]
    print(f"grasp check: "
          f"{'balancing on the wheels' if args.balance else 'base welded'}")

    results = []
    for table in tables:
        picked = [i for i in table.items
                  if i.graspable and args.item in (None, i.name)]
        if not picked:
            continue
        print(f"\n{table.name} - docked at "
              f"({table.dock[0]:+.2f}, {table.dock[1]:+.2f})")
        for item in picked:
            results.append(attempt(item, table, args.balance, bare=args.bare))

    if not results:
        ap.error("no matching graspable objects")
    good = [r for r in results if r.ok]
    print(f"\n{len(good)}/{len(results)} lifted and held")
    if len(good) < len(results):
        raise SystemExit("failed: " + ", ".join(r.item for r in results if not r.ok))


if __name__ == "__main__":
    main()
