"""The motions a pick is made of: plan waypoints, move to them, close, lift.

Factored out of `scripts/check_grasp.py` so the grasp harness and anything that
drives the robot - an agent, a policy - run the same code rather than two
implementations that drift apart.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .arm import GRIP_SITE, GRIPPER, SHUT, Arm, ArmIK, Gripper, down_quat
from .control import Gains, StationKeeper
from .robot import ROOM, State

APPROACH = 0.12        # m above the grasp point to start from
LIFT = 0.15            # m to raise the object
LIFTED = 0.05          # m the object has to rise to count as picked up
HELD_WITHIN = 0.06     # m the object has to stay of the jaw to count as held
HOLD_SECONDS = 1.5     # s to keep holding something before believing the grasp

# Closing is stall-detected, not a fixed amount.  These blades swing rather than
# slide, so how far past first contact the command has to go before the pads are
# really loaded depends on the object: 0.10 rad holds a 42 mm cube and drops a
# mug, 0.35 rad holds neither.  Walk the command shut until the joint stops
# following it - that is the pads meeting something - then push GRIP_BITE
# further and hold, which is a grip force of about kp * GRIP_BITE.
STALL = 0.04           # rad of servo tracking error that counts as contact
GRIP_BITE = 0.20       # rad of command past contact, for the squeeze

# A 7 DOF arm reaches most poses several ways, and IK restarted from scratch is
# free to answer with a different one each time.  Ramping the servos between two
# such answers swings the whole arm through the workspace on the way - which is
# how the first version of this batted a cube off the table on its way down to
# grasp it.  Only the first waypoint is solved freely; the rest continue from
# it, one restart, and a candidate is thrown out if a waypoint still lands more
# than CONTINUITY away in joint space.
CONTINUITY = 0.8       # rad, summed over the arm's seven joints


def welded_at(x, y, yaw, bare=False):
    """The room with the robot bolted to the floor at a docking pose.

    Deleting the free joint takes 7 qpos out of the model, so the room's
    keyframes no longer fit and have to go with it.
    """
    spec = mujoco.MjSpec.from_file(str(ROOM))
    spec.option.impratio = 200
    for key in list(spec.keys):
        spec.delete(key)
    root = spec.body("root")
    for joint in list(root.joints):
        spec.delete(joint)
    root.pos = [x, y, 0.0]
    root.quat = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    if bare:
        from .gripper_pads import bare_grippers
        bare_grippers(spec)
    return spec.compile()


def read_state(model, data, sensors) -> State:
    gyro, vel_l, vel_r = sensors
    rot = data.body("root").xmat.reshape(3, 3)
    return State(
        pitch=math.atan2(-rot[2, 0], math.hypot(rot[0, 0], rot[1, 0])),
        pitch_rate=float(data.sensordata[gyro + 1]),
        yaw=math.atan2(rot[1, 0], rot[0, 0]),
        yaw_rate=float(data.sensordata[gyro + 2]),
        wheel_speed=float((data.sensordata[vel_l] + data.sensordata[vel_r]) / 2),
        forward_speed=float(rot[:, 0] @ data.qvel[0:3]),
        height=float(data.body("root").xpos[2]),
    )


class StationDriver:
    """Keeps the robot upright and over its own feet, every step.

    This is the whole of "navigation" for a manipulation task: the wheels are
    not driven anywhere, they are only allowed to roll as much as balancing
    demands and are then wound back.
    """

    def __init__(self, model, data, gains=None):
        self.model = model
        self.control = StationKeeper(gains or Gains.for_bracketbot())
        self.wheels = [model.actuator(f"wheel_{s}").id for s in ("left", "right")]
        self.qadr = [model.jnt_qposadr[model.joint(f"wheel_{s}").id]
                     for s in ("left", "right")]
        self.sensors = (model.sensor("gyro").adr[0],
                        model.sensor("vel_left").adr[0],
                        model.sensor("vel_right").adr[0])
        self.control.reset(self.angle(data))

    def angle(self, data) -> float:
        return float(sum(data.qpos[a] for a in self.qadr) / 2)

    def com_lean(self, data) -> float:
        """How far the whole robot's centre of mass leans over the axle.

        The balancer was written against chassis pitch, which is the right
        signal for a rigid robot and the wrong one for this: reaching an arm out
        moves mass half a metre forward without tilting the mast at all, so the
        chassis reads level right up until the whole thing goes over.

        This goes into the *setpoint*, not the measurement.  Substituting it for
        the measured pitch breaks the PD - the D term still comes from the gyro,
        which is now measuring something else - and the robot falls over faster.
        As a reference it says where the chassis has to sit for the total mass to
        end up over the axle: lean back by exactly this much.
        """
        rot = data.body("root").xmat.reshape(3, 3)
        offset = rot.T @ (data.subtree_com[1] - data.body("root").xpos)
        return math.atan2(offset[0], max(offset[2], 1e-3))

    def __call__(self, data) -> None:
        state = read_state(self.model, data, self.sensors)
        data.ctrl[self.wheels] = self.control(state, self.angle(data),
                                              self.com_lean(data))


class Rig:
    """A model, a clock, and whatever has to run every step."""

    def __init__(self, model, data, driver=None, on_step=None):
        self.model, self.data = model, data
        self.driver = driver          # what holds the robot up, or None if welded
        self.on_step = on_step        # e.g. a video recorder

    def step(self, n=1):
        for _ in range(n):
            if self.driver is not None:
                self.driver(self.data)
            mujoco.mj_step(self.model, self.data)
            if self.on_step is not None:
                self.on_step(self.data)

    def seconds(self, t):
        self.step(int(t / self.model.opt.timestep))


def set_const(model, data=None) -> None:
    """`mj_setConst` that keeps the room's declared `statistic`.

    Besides the qpos0 constants it recomputes `stat.extent` and `stat.center`
    from the geometry - 4 m becomes 11.4 m in this room - and every camera's
    near plane is `extent * znear`: 80 mm becomes 230 mm, which clips the
    jaws and a ball between them out of the wrist views.
    """
    extent, center = float(model.stat.extent), model.stat.center.copy()
    mujoco.mj_setConst(model, mujoco.MjData(model) if data is None else data)
    model.stat.extent = extent
    model.stat.center[:] = center


def hold_everything(model, data) -> None:
    """Point every position servo at the pose it is already in.

    Servos default to a command of zero, which for the mast carriages means
    'slam to the top of the rail'.
    """
    for act in range(model.nu):
        if model.actuator(act).name.startswith("wheel_"):
            continue                      # the balancer owns those
        joint = model.actuator(act).trnid[0]
        data.ctrl[act] = data.qpos[model.jnt_qposadr[joint]]


def move(rig: Rig, arm: Arm, target, seconds=1.2, tol=0.01, patience=3.0) -> float:
    """Walk the servos to a pose, then wait until the arm has actually got there.

    Two separate problems.  Commanding a far pose outright swings the arm at
    whatever speed the force limit allows, which on a balancing robot is a good
    way to fall over - hence the ramp.  But these servos also carry kv = 0.1 kp,
    so a hinge moving 0.5 rad/s spends 10 of its 17 N.m budget fighting its own
    damping: at the end of a ramp the arm is still a long way behind the command.
    Ramp, then settle.
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


def squeeze(rig: Rig, arm: Arm, floor: float = SHUT, rate: float = 0.6,
            settle: float = 0.6) -> float:
    """Close the fingers until they are loaded, then hold there.

    Commanding the gripper shut in one go is a 1 rad step: it swings the blades
    in at whatever the force limit allows and punts the object across the room.
    Walk the command in at `rate` rad/s, watch the joint fall behind it, and
    stop GRIP_BITE past where that starts.
    """
    model, data = rig.model, arm.data
    joint = model.jnt_qposadr[model.joint(GRIPPER[arm.side]).id]
    command = float(data.ctrl[arm.grip_act])
    step = rate * model.opt.timestep
    bite = None

    while command > floor:
        command = max(floor, command - step)
        arm.grip(command)
        rig.step()
        if bite is None and data.qpos[joint] - command > STALL:
            bite = command - GRIP_BITE
        if bite is not None and command <= bite:
            break

    rig.seconds(settle)
    return float(data.qpos[joint])


def plan_waypoints(model, scratch, jaws, width, yaws, dock_yaw, seed_from,
                   sides=("right", "left"), hand_type=Gripper):
    """Best (score, side, wrist, opening, [above, on, up]) for a top-down grasp.

    Targets are the *jaws*, not the grip site: `Gripper` knows where the tip
    midpoint sits for a given opening and corrects the site target, so the object
    ends up centred between the blades instead of 25 mm to one side of them.
    """
    # First fit, in the order given - not the lowest IK residual.  Scanning all
    # the wrist angles and keeping the neatest one makes `yaws` order
    # meaningless, which quietly turns "retry with a different wrist" into
    # "retry with exactly the same wrist".  The caller orders these; honour it.
    for side in sides:
        ik, hand = ArmIK(model, side), hand_type(model, side)
        opening = hand.opening_for(width)
        if opening is None:            # wider than the jaw opens
            continue
        for yaw in yaws:
            quat = down_quat(dock_yaw + yaw)
            targets = [hand.site_target(jaws + np.array([0, 0, dz]), quat, opening)
                       for dz in (APPROACH, 0.0, LIFT)]
            scratch.qpos[:] = seed_from
            mujoco.mj_kinematics(model, scratch)

            above = ik.solve(scratch, targets[0], quat)
            if not above.ok:
                continue
            chain, previous, broke = [above], above, False
            for target in targets[1:]:
                got = ik.solve(scratch, target, quat, seed=previous.qpos,
                               restarts=1)
                if not got.ok or np.abs(got.qpos - previous.qpos).sum() > CONTINUITY:
                    broke = True
                    break
                chain.append(got)
                previous = got
            if broke:
                continue

            return (sum(c.pos_err for c in chain), side, yaw, opening, chain)
    return None


def in_hand(model, data, body_name: str, side: str, hand: Gripper) -> bool:
    """Is the object still in that hand?

    Not "is it touching both pads": these blades are not symmetric, so a good
    grasp often has two contacts on one pad and one on the other, and the contact
    set flickers between steps.  What settles it is where the object is - if it
    is still in the jaw, the hand has it.
    """
    prefix = "" if side == "right" else "l_"
    fingers = {model.body(f"{prefix}{f}_finger__{f}_finger").id
               for f in ("left", "right")}
    target = model.body(body_name).id
    touching = False
    for c in range(data.ncon):
        pair = {model.geom_bodyid[data.contact[c].geom1],
                model.geom_bodyid[data.contact[c].geom2]}
        if target in pair and pair & fingers:
            touching = True
            break
    if not touching:
        return False

    site = model.site(GRIP_SITE[side]).id
    opening = float(data.qpos[model.jnt_qposadr[model.joint(GRIPPER[side]).id]])
    jaw = (data.site_xpos[site]
           + data.site_xmat[site].reshape(3, 3) @ hand.jaw_offset(opening))
    return bool(np.linalg.norm(data.body(body_name).xpos - jaw) < HELD_WITHIN)
