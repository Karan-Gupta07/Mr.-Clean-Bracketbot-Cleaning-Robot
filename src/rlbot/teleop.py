"""Drive one arm's jaws around by hand, and record what happened.

This is the data side of the VLA plan: an operator jogs the jaws with the
keyboard, picks something up, puts it in the crate, and every step of it is
written down at a control rate a policy can be trained at.

The operator does not command joints.  They command where the *jaws* are and
which way the wrist is turned, the hand always pointing down, and each nudge is
one damped-least-squares step continued from the servos' current command -
seeded from the command and not the measured pose, so lag never accumulates
into the target.  A nudge that IK cannot follow, or that would need a different
arm configuration to reach, is refused and the target stays put.

Nothing here opens a window.  `scripts/teleop.py` does that; this is the part
that can be tested without one.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .arm import GRIPPER, SHUT, down_quat
from .grasp import GRIP_BITE, STALL, in_hand
from .skills import Robot

NUDGE = 0.01               # m the jaws move per key press
TURN = math.radians(5)     # wrist yaw per key press
NUDGE_RANGE = (0.0025, 0.04)   # how fine and how coarse the step can be set
CLOSE_RATE = 0.6           # rad/s the gripper command walks in at, as `squeeze` does
OPEN_RATE = 0.3            # rad/s it walks back out: faster is a flick, and the
                           # blades throw a 50 g cube clear of the crate
RAMP = 1.5                 # s to ramp the servos to a far pose (reset, ready)
CONTINUITY = 0.4           # rad, summed over the arm; more than this on a nudge
                           # means IK jumped to another branch, and the arm would
                           # swing through the table to get there
CONTROL_HZ = 20            # rows per second in a demonstration

# A nudge moves the target a step, and the command *chases* the target at a
# set speed rather than jumping to it.  Two reasons.  Held keys repeat at about
# 30 a second, which at 1 cm a press is 0.3 m/s, and the servos cannot follow
# that: the arm lags its command by 2 cm and carries on for those 2 cm after
# the operator lets go - into the cube.  And what keeps a cube in this hand is
# two pads and friction: a command that steps 1 cm shakes it, and a carry made
# of thirty such steps a second loses it every time.  So the command ramps, at
# a speed that drops with something in the hand, and a key held down queues at
# most a couple of steps ahead of where the arm has got to.
SPEED = 0.15               # m/s the jaws move, hand empty
SPEED_HOLDING = 0.05       # m/s with something in the hand
TURN_RATE = math.radians(45)   # rad/s the wrist turns
QUEUE = 2                  # nudges that may be pending before the next is refused

# The servos sag.  They carry kv = 0.1 kp and 10 N.m at the wrist, so a hand
# held out over the table settles 2-5 mm below and short of where it was sent,
# by an amount that depends on the arm's configuration.  A cube leaves 6 mm of
# daylight either side of the pads, so that is the difference between a grasp
# and a pad landing on the cube's top edge.  While the hand is open and has
# come to rest, measure the miss and shift the command the other way - slowly,
# and only so far, so that a hand pressing on something does not wind itself
# up.  At rest matters: a moving arm lags its command by 2 cm, and a trim that
# takes that for sag overshoots by the same amount the moment the operator
# stops, which is how a hand arriving over a cube knocks it onto the floor.
TRIM_EVERY = 0.1           # s between corrections
TRIM_GAIN = 0.5            # of the measured miss taken up per correction
TRIM_MAX = 0.010           # m the command may be shifted from the target
TRIM_DEADBAND = 0.001      # m of miss not worth correcting
STILL = 0.05               # rad/s; every arm joint under this means at rest

# Where the jaws start: ahead of the mast, off to the arm's own side, and well
# above everything on the table.  The near row of objects is 0.33 m out, the
# cubes sit at +-0.14 and +-0.26 m across, and nothing is taller than 80 mm.
READY = (0.30, 0.20, 0.20)     # (ahead, across toward the arm's side, above the top)

COLOUR = {"cube_s": "orange", "cube_m": "blue", "cube_l": "green",
          "cube_xl": "purple", "ball": "red", "bowl": "white", "cup": "grey"}


class Jog:
    """One arm under keyboard control: a jaw target, a wrist yaw, a gripper."""

    def __init__(self, robot: Robot, side: str, width: float):
        self.robot, self.side = robot, side
        self.model, self.data = robot.model, robot.data
        self.arm, self.hand = robot.arms[side], robot.hands[side]
        self.yaw0 = robot.dock_yaw          # nudges are in the docking frame
        self._grip_adr = self.model.jnt_qposadr[
            self.model.joint(GRIPPER[side]).id]
        self.set_width(width)

        self.step_size = NUDGE
        self.closed = False
        self._bite: float | None = None
        self._ramp: deque[np.ndarray] = deque()   # servo commands still to send
        self._segments: deque[int] = deque()      # how many of those per nudge
        self._going = False                       # a `go` is in progress
        self.target = self.jaws.copy()      # world, where the jaws are sent
        self.yaw = math.pi / 2              # relative to the dock: across the table
        self.trim = np.zeros(3)             # what the command is shifted by, for sag
        self.refused = 0                    # nudges IK could not follow
        self._trim_every = max(1, round(TRIM_EVERY / self.model.opt.timestep))
        self._count = 0

    # ---- frames ----------------------------------------------------------
    def local(self, ahead: float, across: float, up: float) -> np.ndarray:
        """A point given in the docking frame, as world coordinates."""
        root = self.data.body("root").xpos
        c, s = math.cos(self.yaw0), math.sin(self.yaw0)
        return np.array([root[0] + c * ahead - s * across,
                         root[1] + s * ahead + c * across,
                         0.70 + up])            # the table top

    @property
    def quat(self) -> np.ndarray:
        return down_quat(self.yaw0 + self.yaw)

    @property
    def opening(self) -> float:
        """The gripper command as it stands - what the jaw offset answers to."""
        return float(self.data.ctrl[self.arm.grip_act])

    @property
    def jaws(self) -> np.ndarray:
        """Where the jaw centre actually is, in the world."""
        site = self.arm.site
        return (self.data.site_xpos[site]
                + self.data.site_xmat[site].reshape(3, 3)
                @ self.hand.jaw_offset(float(self.data.qpos[self._grip_adr])))

    @property
    def held(self) -> str | None:
        """What this hand is holding, by asking the sim rather than remembering."""
        for name, item in self.robot.items.items():
            if item.graspable and in_hand(self.model, self.data, name,
                                          self.side, self.hand):
                return name
        return None

    def set_width(self, width: float) -> None:
        """Open only as wide as the thing being reached for needs, as the grasp
        harness does: the blades splay as they open, and a hand open past the
        object's width closes on it at a worse angle."""
        opening = self.hand.opening_for(width)
        self.open_to = opening if opening is not None else float(
            self.hand.q[int(np.argmax(self.hand.gap))])

    # ---- moving ----------------------------------------------------------
    def solve(self, target, yaw: float, seed, **kw):
        """IK for the jaws at `target` plus the sag trim, on a scratch state."""
        quat = down_quat(self.yaw0 + yaw)
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        mujoco.mj_kinematics(self.model, scratch)
        site = self.hand.site_target(target + self.trim, quat, self.opening)
        return self.arm.ik.solve(scratch, site, quat, seed=seed, **kw)

    @property
    def still(self) -> bool:
        return bool(np.abs(self.data.qvel[self.arm.ik.dofs]).max() < STILL)

    def _trim(self) -> None:
        """One correction of the command toward where the jaws actually are."""
        miss = self.target - self.jaws
        if np.linalg.norm(miss) < TRIM_DEADBAND:
            return
        trim = self.trim + TRIM_GAIN * miss
        if np.linalg.norm(trim) > TRIM_MAX:
            trim *= TRIM_MAX / np.linalg.norm(trim)
        seed = self.data.ctrl[self.arm.acts].copy()
        was, self.trim = self.trim, trim
        got = self.solve(self.target, self.yaw, seed, restarts=1, iters=60)
        if got.ok and float(np.abs(got.qpos - seed).sum()) <= CONTINUITY:
            self.arm.hold(got.qpos)
        else:
            self.trim = was

    def nudge(self, ahead=0.0, across=0.0, up=0.0, turn=0.0) -> str | None:
        """Move the target one step in the docking frame.

        Returns None when it went through, "lag" when the arm is still working
        through earlier nudges (try again in a moment), or "reach" when IK
        cannot get there without swinging the arm through another configuration.
        """
        if self._going or len(self._segments) >= QUEUE:
            return "lag"
        c, s = math.cos(self.yaw0), math.sin(self.yaw0)
        delta = self.step_size * np.array([c * ahead - s * across,
                                           s * ahead + c * across, up])
        target = self.target + delta
        yaw = self.yaw + turn * TURN
        # continue from where the command will be, not where it is now
        seed = (self._ramp[-1] if self._ramp
                else self.data.ctrl[self.arm.acts]).copy()
        got = self.solve(target, yaw, seed, restarts=1, iters=60)
        if not got.ok or float(np.abs(got.qpos - seed).sum()) > CONTINUITY:
            self.refused += 1
            return "reach"
        speed = SPEED_HOLDING if self.closed else SPEED
        seconds = (float(np.linalg.norm(delta)) / speed if np.any(delta)
                   else abs(turn) * TURN / TURN_RATE)
        self._queue(seed, got.qpos, seconds)
        self.target, self.yaw = target, yaw
        return None

    def _queue(self, start, end, seconds: float) -> None:
        n = max(1, int(seconds / self.model.opt.timestep))
        self._ramp.extend(start + (end - start) * (i + 1) / n for i in range(n))
        self._segments.append(n)

    def go(self, target, yaw: float, seconds: float | None = None,
           **kw) -> bool:
        """Send the arm somewhere far in one ramped move.

        Solved from the current command - freely by default, or with
        `restarts=1` to stay in the arm's present configuration, which is what
        a carry wants - and ramped over `seconds`, or if that is not given over
        long enough that no joint has to hurry: the harness found 2.5 s per
        radian of travel is what keeps a held object in the hand.
        """
        seed = self.data.ctrl[self.arm.acts].copy()
        self.trim[:] = 0.0
        got = self.solve(np.asarray(target, dtype=float), yaw, seed, **kw)
        if not got.ok:
            return False
        travel = float(np.abs(got.qpos - seed).sum())
        if seconds is None:
            seconds = max(RAMP, 2.5 * travel)
        self._ramp.clear()
        self._segments.clear()
        self._queue(seed, got.qpos, seconds)
        self._going = True
        self.target, self.yaw = np.asarray(target, dtype=float), yaw
        return True

    def ready(self) -> bool:
        ahead, across, up = READY
        return self.go(self.local(ahead, across if self.side == "left"
                                  else -across, up), math.pi / 2, RAMP)

    def toggle_grip(self) -> None:
        self.closed = not self.closed

    def step(self) -> None:
        """Once per sim step, before `mj_step`: advance the ramp and the hand.

        The gripper is the `squeeze` logic spread over time so the loop never
        blocks: walk the command in at CLOSE_RATE, notice the joint falling
        behind it - that is the pads meeting something - and stop GRIP_BITE
        past there.  Opening walks back out to the width set for the object.
        """
        if self._ramp:
            self.arm.hold(self._ramp.popleft())
            self._segments[0] -= 1
            if self._segments[0] == 0:
                self._segments.popleft()
            if not self._ramp:
                self._going = False
        self._count += 1
        if (not self._ramp and not self.closed and self.still
                and self._count % self._trim_every == 0):
            self._trim()

        cmd, dt = self.opening, self.model.opt.timestep
        if self.closed:
            if self._bite is None and self.data.qpos[self._grip_adr] - cmd > STALL:
                self._bite = max(SHUT, cmd - GRIP_BITE)
            floor = SHUT if self._bite is None else self._bite
            cmd = max(floor, cmd - CLOSE_RATE * dt)
        else:
            self._bite = None
            cmd = min(self.open_to, cmd + OPEN_RATE * dt)
        self.arm.grip(cmd)

    @property
    def busy(self) -> bool:
        """Mid-`go`: nudges are refused until the arm has arrived."""
        return self._going


@dataclass
class Demo:
    """One episode as it is being recorded: rows at the control rate."""

    meta: dict
    rows: list[dict] = field(default_factory=list)
    events: list[tuple[float, str]] = field(default_factory=list)


class DemoRecorder:
    """Writes what the operator did, one compressed .npz per episode.

    Raw state, not a chosen action space.  Everything a trainer might want as
    the action - the jaw target, the wrist yaw, the gripper command, the full
    servo command, the joint positions - is in every row, and the deltas fall
    out at training time.  Deciding that here would bake one policy's input
    format into every demonstration collected before the policy existed.

    Camera frames are not here either: they are a pure function of `qpos`, so
    `scripts/render_demos.py` re-renders them afterwards at whatever size and
    from whichever cameras the model turns out to want.
    """

    def __init__(self, robot: Robot, out_dir: Path, hz: int = CONTROL_HZ):
        self.robot = robot
        self.out_dir = Path(out_dir)
        self.every = max(1, round(1 / (hz * robot.model.opt.timestep)))
        self.hz = hz
        self.objects = list(robot.items)
        self.bodies = [robot.model.body(n).id for n in self.objects]
        self.demo: Demo | None = None
        self.count = 0

    @property
    def active(self) -> bool:
        return self.demo is not None

    def start(self, **meta) -> None:
        self.demo = Demo(dict(meta, table=self.robot.table.name,
                              balancing=self.robot.balancing,
                              control_hz=self.hz, objects=self.objects,
                              timestep=self.robot.model.opt.timestep))
        self.count = 0

    def event(self, name: str) -> None:
        if self.demo is not None:
            self.demo.events.append((float(self.robot.data.time), name))

    def step(self, jog: Jog) -> None:
        """Once per sim step; keeps a row every `every` steps."""
        if self.demo is None:
            return
        self.count += 1
        if self.count % self.every:
            return
        d, arms = self.robot.data, self.robot.arms
        site = jog.arm.site
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, d.site_xmat[site])
        held = jog.held
        self.demo.rows.append({
            "time": float(d.time),
            "qpos": d.qpos.copy(),
            "qvel": d.qvel.copy(),
            "ctrl": d.ctrl.copy(),
            "arm_right": d.qpos[arms["right"].ik.qadr].copy(),
            "arm_left": d.qpos[arms["left"].ik.qadr].copy(),
            "grip_right": float(d.ctrl[arms["right"].grip_act]),
            "grip_left": float(d.ctrl[arms["left"].grip_act]),
            "active_arm": 0 if jog.side == "right" else 1,
            "ee_pos": jog.jaws,
            "ee_quat": quat,
            "target_pos": jog.target.copy(),
            "target_yaw": float(jog.yaw),
            "grip_closed": int(jog.closed),
            "held": self.objects.index(held) if held else -1,
            "obj_pos": np.array([d.xpos[b] for b in self.bodies]),
            "obj_quat": np.array([d.xquat[b] for b in self.bodies]),
        })

    def discard(self) -> int:
        n = len(self.demo.rows) if self.demo else 0
        self.demo = None
        return n

    def stop(self, success: bool) -> Path | None:
        """Write the episode out.  Returns the path, or None if it was empty."""
        demo, self.demo = self.demo, None
        if demo is None or not demo.rows:
            return None
        arrays = {k: np.array([r[k] for r in demo.rows]) for k in demo.rows[0]}
        meta = dict(demo.meta, success=bool(success), rows=len(demo.rows),
                    seconds=float(demo.rows[-1]["time"] - demo.rows[0]["time"]),
                    events=demo.events)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / time.strftime("ep_%Y%m%d_%H%M%S.npz")
        np.savez_compressed(path, meta=json.dumps(meta), **arrays)
        return path


def load_demo(path) -> tuple[dict, dict[str, np.ndarray]]:
    """A recorded episode: (meta, arrays)."""
    with np.load(path) as f:
        meta = json.loads(str(f["meta"]))
        arrays = {k: f[k] for k in f.files if k != "meta"}
    return meta, arrays


def task_text(obj: str) -> str:
    """The language instruction a demonstration answers to."""
    colour = COLOUR.get(obj)
    kind = obj.split("_")[0]
    return f"put the {colour} {kind} in the crate" if colour else \
        f"put the {kind} in the crate"
