"""Close the fingers on every object in the room and try to lift it.

    .venv/bin/python scripts/check_grasp.py                 # all three tables
    .venv/bin/python scripts/check_grasp.py --table ware    # just the crockery
    .venv/bin/python scripts/check_grasp.py --balance       # on the wheels, balancing

`build_room.py` proves the arm can be *driven* to each object.  That is
kinematics, and kinematics has never held onto anything.  This runs the grasp:
approach from 0.12 m above, descend, close, lift, and check the object came up
with the hand and is still between the fingers 1.5 s later.

By default the base is welded to the floor at the docking pose, so a failure is
the grasp's fault and not the balancer's.  `--balance` puts it back on its wheels
with the PD controller running, which is the honest version of the same test and
the one that matters for the real robot.
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

from rlbot import BalanceController, Gains                        # noqa: E402
from rlbot.arm import GRIPPER, SHUT, Arm, ArmIK, Gripper, down_quat  # noqa: E402
from rlbot.robot import ROOM, State                               # noqa: E402
from rlbot.room import GRASP_YAWS, TABLES, grasp_pose             # noqa: E402

APPROACH = 0.12        # m above the grasp point to start from
LIFT = 0.15            # m to raise the object
HOLD = 1.5             # s to keep holding it before believing the grasp
LIFTED = 0.05          # m the object has to rise to count
APERTURE = 0.195       # m between the fingertips at full open, measured


@dataclass
class Result:
    item: str
    side: str
    ik_err: float
    rose: float        # m the object actually came up
    held: bool

    @property
    def ok(self) -> bool:
        return self.held and self.rose > LIFTED


def welded_at(x, y, yaw):
    """The room with the robot bolted to the floor at a docking pose.

    Deleting the free joint takes 7 qpos out of the model, so the room's
    keyframes no longer fit and have to go with it.
    """
    spec = mujoco.MjSpec.from_file(str(ROOM))
    for key in list(spec.keys):
        spec.delete(key)
    root = spec.body("root")
    for joint in list(root.joints):
        spec.delete(joint)
    root.pos = [x, y, 0.0]
    root.quat = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    return spec.compile()


def floating_at(x, y, yaw):
    """The room as built, with the robot stood at a docking pose on its wheels."""
    model = mujoco.MjModel.from_xml_path(str(ROOM))
    return model


class Rig:
    """One attempt: a model, a clock, and whatever has to run every step."""

    def __init__(self, model, data, balance=False):
        self.model, self.data = model, data
        self.ctrl = None
        if balance:
            self.ctrl = BalanceController(Gains.for_bracketbot())
            self.wheels = [model.actuator(f"wheel_{s}").id for s in ("left", "right")]
            self._gyro = model.sensor("gyro").adr[0]
            self._vel = [model.sensor(f"vel_{s}").adr[0] for s in ("left", "right")]

    def state(self) -> State:
        d = self.data
        rot = d.body("root").xmat.reshape(3, 3)
        gyro = d.sensordata[self._gyro : self._gyro + 3]
        return State(
            pitch=math.atan2(-rot[2, 0], math.hypot(rot[0, 0], rot[1, 0])),
            pitch_rate=float(gyro[1]),
            yaw=math.atan2(rot[1, 0], rot[0, 0]),
            yaw_rate=float(gyro[2]),
            wheel_speed=float(sum(d.sensordata[a] for a in self._vel) / 2),
            forward_speed=float(rot[:, 0] @ d.qvel[0:3]),
            height=float(d.body("root").xpos[2]),
        )

    def step(self, n=1):
        for _ in range(n):
            if self.ctrl is not None:
                left, right = self.ctrl(self.state())
                self.data.ctrl[self.wheels] = (left, right)
            mujoco.mj_step(self.model, self.data)

    def seconds(self, t):
        self.step(int(t / self.model.opt.timestep))


def move(rig: Rig, arm: Arm, target, seconds=1.2, tol=0.01, patience=3.0):
    """Walk the servos to a pose, then wait until the arm has actually got there.

    Two separate problems.  Commanding a far pose outright swings the arm at
    whatever speed the force limit allows, which on a balancing robot is a good
    way to fall over - hence the ramp.  But these servos also carry kv = 0.1 kp,
    so a hinge moving 0.5 rad/s spends 10 of its 17 N.m budget fighting its own
    damping: at the end of a ramp the arm is still a long way behind the command.
    The first version of this script closed the gripper there, 187 mm short of
    the cube, and batted it off the table.  So ramp, then settle.

    Returns how far the arm still is from the command when it gives up waiting.
    """
    start = arm.data.qpos[arm.ik.qadr].copy()
    steps = max(1, int(seconds / rig.model.opt.timestep))
    for i in range(steps):
        arm.hold(start + (target - start) * (i + 1) / steps)
        rig.step()

    arm.hold(target)
    for _ in range(int(patience / rig.model.opt.timestep)):
        gap = float(np.abs(arm.data.qpos[arm.ik.qadr] - target).max())
        if gap < tol:
            return gap
        rig.step()
    return float(np.abs(arm.data.qpos[arm.ik.qadr] - target).max())


# A 7 DOF arm reaches most poses several ways, and IK restarted from scratch is
# free to answer with a different one each time.  Ramping the servos between two
# such answers swings the whole arm through the workspace on the way - which is
# how the first version of this script batted a cube off the table on its way
# down to grasp it.  So only the first waypoint is solved freely; the rest
# continue from it, one restart, and a candidate is thrown out if a waypoint
# still lands more than CONTINUITY away in joint space.
CONTINUITY = 0.8       # rad, summed over the arm's seven joints


def squeeze(rig: Rig, arm: Arm, target: float, seconds=0.8, settle=0.6):
    """Close the fingers at a speed the object can survive.

    The gripper is a position servo too, and commanding it shut in one go is a
    1.0 rad step: it swings the blades in at whatever the force limit allows and
    punts the object across the room - a 55 mm cube left at 1 m/s and landed by
    the far wall.  Ramp the command instead and let the servo stall against the
    object, where its residual error becomes the grip force.
    """
    start = float(arm.data.ctrl[arm.grip_act])
    steps = max(1, int(seconds / rig.model.opt.timestep))
    for i in range(steps):
        arm.grip(start + (target - start) * (i + 1) / steps)
        rig.step()
    rig.seconds(settle)


def plan_grasp(model, data, item, table, seed_from):
    """Best (arm, wrist yaw, opening, pre-grasp, grasp, lift) for one object.

    Targets are the *jaws*, not the grip site: `Gripper` knows where the tip
    midpoint sits for a given opening and corrects the site target, so the object
    ends up centred between the blades instead of 25 mm to one side of them.
    """
    _, _, dock_yaw = table.dock
    jaws = grasp_pose(item, table)
    best = None
    for side in ("right", "left"):
        ik, hand = ArmIK(model, side), Gripper(model, side)
        opening = hand.opening_for(item.width)
        for yaw in GRASP_YAWS:
            quat = down_quat(dock_yaw + yaw)
            waypoint = [hand.site_target(jaws + np.array([0, 0, dz]), quat, opening)
                        for dz in (APPROACH, 0.0, LIFT)]
            data.qpos[:] = seed_from
            mujoco.mj_kinematics(model, data)

            above = ik.solve(data, waypoint[0], quat)
            if not above.ok:
                continue
            chain, previous, broke = [above], above, False
            for target in waypoint[1:]:
                got = ik.solve(data, target, quat, seed=previous.qpos, restarts=1)
                if not got.ok or np.abs(got.qpos - previous.qpos).sum() > CONTINUITY:
                    broke = True
                    break
                chain.append(got)
                previous = got
            if broke:
                continue

            score = sum(c.pos_err for c in chain)
            if best is None or score < best[0]:
                best = (score, side, yaw, opening, *chain)
    return best


def attempt(item, table, balance: bool, verbose=True) -> Result:
    x, y, yaw = table.dock
    model = floating_at(x, y, yaw) if balance else welded_at(x, y, yaw)
    data = mujoco.MjData(model)

    if balance:
        key = model.key(f"dock_{table.name.split('_')[1]}").id
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    rig = Rig(model, data, balance=balance)
    plan = plan_grasp(model, mujoco.MjData(model), item, table, data.qpos.copy())
    if plan is None:
        return Result(item.name, "-", float("nan"), 0.0, False)
    score, side, wrist, opening, above, on, up = plan

    arm = Arm(model, data, side)
    # hold every servo where it already is, so the other arm does not sag
    for act in range(model.nu):
        joint = model.actuator(act).trnid[0]
        data.ctrl[act] = data.qpos[model.jnt_qposadr[joint]]
    arm.grip(opening)          # only as wide as this object needs
    rig.seconds(0.5)

    start_z = float(data.body(item.name).xpos[2])
    move(rig, arm, above.qpos, 1.6)        # over the object, fingers open
    gap = move(rig, arm, on.qpos, 1.0)     # down around it
    squeeze(rig, arm, SHUT)                 # let the fingers load up
    move(rig, arm, up.qpos, 1.2)           # lift
    rig.seconds(HOLD)

    rose = float(data.body(item.name).xpos[2]) - start_z
    held = grasped(model, data, item.name, side)
    grip = model.joint(GRIPPER[side]).id
    span = float(data.qpos[model.jnt_qposadr[grip]]) * APERTURE
    if verbose:
        flag = "ok  " if (held and rose > LIFTED) else "FAIL"
        print(f"  {flag} {item.name:<12} {side:>5} arm, wrist "
              f"{math.degrees(wrist):3.0f} deg   rose {rose * 1000:+6.1f} mm   "
              f"fingers {span * 1000:5.1f} mm apart   "
              f"{'holding' if held else 'empty'}")
    return Result(item.name, side, score, rose, held)


def grasped(model, data, body_name: str, side: str) -> bool:
    """Is the object still in contact with both of that hand's fingers?"""
    prefix = "" if side == "right" else "l_"
    fingers = {model.body(f"{prefix}{f}_finger__{f}_finger").id
               for f in ("left", "right")}
    target = model.body(body_name).id
    touching = set()
    for c in range(data.ncon):
        pair = {model.geom_bodyid[data.contact[c].geom1],
                model.geom_bodyid[data.contact[c].geom2]}
        if target in pair:
            touching |= pair & fingers
    return len(touching) == 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="all",
                    choices=["all"] + [t.name.split("_")[1] for t in TABLES])
    ap.add_argument("--item", help="just this object, by name")
    ap.add_argument("--balance", action="store_true",
                    help="stand on the wheels with the PD balancer running")
    args = ap.parse_args()

    tables = [t for t in TABLES
              if args.table in ("all", t.name.split("_")[1])]
    print(f"grasp check: {'balancing on the wheels' if args.balance else 'base welded'}")

    results = []
    for table in tables:
        picked = [i for i in table.items if args.item in (None, i.name)]
        if not picked:
            continue
        print(f"\n{table.name} - docked at "
              f"({table.dock[0]:+.2f}, {table.dock[1]:+.2f})")
        for item in picked:
            results.append(attempt(item, table, args.balance))

    good = [r for r in results if r.ok]
    print(f"\n{len(good)}/{len(results)} lifted and held")
    if len(good) < len(results):
        raise SystemExit("failed: " + ", ".join(r.item for r in results if not r.ok))


if __name__ == "__main__":
    main()
