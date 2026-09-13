"""Collect pick-and-place demonstrations without an operator.

    .venv/bin/python scripts/collect_demos.py --table ball --episodes 200
    .venv/bin/python scripts/collect_demos.py --table ware --object bowl --episodes 200
    .venv/bin/python scripts/collect_demos.py --table ball --episodes 3 --seed 1

The same controller `scripts/teleop.py` puts under the keyboard, driven by a
script instead of a hand: over the object, down, close, lift, carry to a clear
spot in the crate, let go, back off.  Each episode is scored the way the
operator's are - object inside the crate and out of the hand - and successes
are written with the same recorder, in the same format, to the same place.

Variety comes from the layout.  Every episode moves both the object and the
crate: the object anywhere in the band either arm can reach, the crate
anywhere near the middle of the table that does not overlap it.  The crate is
furniture with no joint, so it is moved by editing its body position in the
compiled model rather than through `qpos`.

Failures are kept too, under `failed/`, because they cost nothing and a policy
trained only on clean runs never sees what a miss looks like.  Only successes
count toward `--episodes`.

Every `--viz-every` successes, the episode just kept is also rendered to an
animated GIF under `viz/` - an over-the-shoulder view beside the wrist camera
that did the work - so a run can be checked on without a window.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlbot.grasp import APPROACH, LIFT, hold_everything                 # noqa: E402
from rlbot.room import TABLE_H, Table                                   # noqa: E402
from rlbot.skills import Robot                                          # noqa: E402
from rlbot.teleop import DemoRecorder, Jog, load_demo, task_text        # noqa: E402

# Where things may land, in the table's own frame: `along` its long axis and
# `depth` toward the robot, both from the object's laid-out position.  Sweeping
# the cubes found nothing within 0.12 m of the centreline can be picked - the
# arm reaches the table from one side and that is where the wrist runs out -
# and that past 0.28 m the outer wrist joint is at its limit.  So the object
# goes in the band between, on either side, and the crate stays in the middle.
OBJECT_ALONG = (0.14, 0.28)    # m from the centreline, either side
OBJECT_DEPTH = 0.04            # m either way from the laid-out row
CRATE_ALONG = 0.08             # m either way from the centre
CRATE_DEPTH = 0.05
CLEARANCE = 0.02               # m between the crate's wall and the object

KEY_EVERY = 0.033              # s between scripted key presses: held-key rate
SETTLE = 0.6                   # s to let the sag trim finish before closing
CLOSE_WAIT = 2.0               # s for the squeeze to load up
RELEASE_WAIT = 2.0             # s for the fingers to open and the object to land
RETREAT = 0.12                 # m to lift the open hand clear of the crate
PRE_GRASP = 0.06               # m above the grasp height: pads just clear of the rim
LIFTED = 0.05                  # m above the table an object still in the hand sits

# Wrist yaw for the grasp, relative to the docking frame, per kind of thing.
# Sweeping the bowl over heights, openings and angles: 45 degrees holds it
# most often, and it is the tapered wall meeting the pads off-square that
# does it.  Everything else takes the across-the-table grasp the harness uses.
GRASP_YAW = {"bowl": 45.0, "cup": 45.0}
DEFAULT_YAW = 90.0
# How much higher than the layout's grasp height to close, per kind.  The
# ball's grasp point is its centre, which puts the pads a hair off the table;
# 6 mm up doubles the pick rate.
GRASP_LIFT = {"ball": 0.006}

VIZ_FPS = 10                   # the recorder runs at 20 Hz; every other row
VIZ_SIZE = (240, 320)          # rows, columns per panel


class Viz:
    """Renders a kept episode to a GIF, from the model as it stands.

    Done straight after the episode and before the next reset on purpose: the
    crate's position lives in the model, not in `qpos`, so replaying the rows
    later would draw it back in the middle of the table.
    """

    def __init__(self, robot: Robot, out_dir: Path):
        self.robot = robot
        self.out_dir = out_dir
        self.renderer = None
        self.cam = mujoco.MjvCamera()

    def __call__(self, path: Path, side: str, hz: int) -> Path:
        from PIL import Image        # in the venv; only this preview needs it
        model, data = self.robot.model, self.robot.data
        if self.renderer is None:
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, VIZ_SIZE[1])
            model.vis.global_.offheight = max(model.vis.global_.offheight, VIZ_SIZE[0])
            self.renderer = mujoco.Renderer(model, *VIZ_SIZE)
        _, arrays = load_demo(path)
        keep = data.qpos.copy()
        wrist = model.camera(f"wrist_{side}_cam").id
        frames = []
        for q in arrays["qpos"][::max(1, hz // VIZ_FPS)]:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            root = data.body("root")
            rot = root.xmat.reshape(3, 3)
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.cam.lookat[:] = root.xpos + rot[:, 0] * 0.45 + np.array([0, 0, 0.35])
            self.cam.distance = 1.6
            self.cam.elevation = -25
            self.cam.azimuth = math.degrees(math.atan2(rot[1, 0], rot[0, 0])) + 145
            self.renderer.update_scene(data, self.cam)
            over = self.renderer.render().copy()
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = wrist
            self.renderer.update_scene(data, self.cam)
            frames.append(Image.fromarray(np.concatenate(
                [over, self.renderer.render()], axis=1)))
        data.qpos[:] = keep
        mujoco.mj_forward(model, data)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        out = self.out_dir / (path.stem + ".gif")
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=1000 // VIZ_FPS, loop=0)
        return out


class Layout:
    """Puts the object and the crate somewhere new before each episode."""

    def __init__(self, robot: Robot, obj: str, rng: np.random.Generator):
        self.robot, self.obj, self.rng = robot, obj, rng
        model = robot.model
        self.table: Table = robot.table
        self.item = robot.items[obj]
        self.crate_item = robot.items[robot.crate]
        self.crate_body = model.body(robot.crate).id
        self.obj_adr = model.jnt_qposadr[model.body(obj).jntadr[0]]
        self.obj_z = float(model.body(obj).pos[2])      # as laid out, on the table

    def sample(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """(object, crate) offsets as (along, depth), non-overlapping."""
        size = self.crate_item.size
        half_l, half_w = size["l"] / 2, size["w"] / 2
        reach = self.item.width / 2 + CLEARANCE
        for _ in range(100):
            side = self.rng.choice((-1.0, 1.0))
            obj = (side * self.rng.uniform(*OBJECT_ALONG),
                   self.rng.uniform(-OBJECT_DEPTH, OBJECT_DEPTH))
            crate = (self.rng.uniform(-CRATE_ALONG, CRATE_ALONG),
                     self.rng.uniform(-CRATE_DEPTH, CRATE_DEPTH))
            if (abs(obj[0] - crate[0]) > half_l + reach
                    or abs(obj[1] - crate[1]) > half_w + reach):
                return obj, crate
        raise RuntimeError("could not place the crate clear of the object")

    def apply(self, obj_at, crate_at) -> dict:
        """Move them.  Returns the layout for the episode's metadata."""
        model, data = self.robot.model, self.robot.data
        obj_pos = self.table.place(dataclasses.replace(self.item, at=obj_at))
        crate_pos = self.table.place(dataclasses.replace(self.crate_item,
                                                         at=crate_at))
        data.qpos[self.obj_adr:self.obj_adr + 3] = [obj_pos[0], obj_pos[1],
                                                    self.obj_z]
        data.qpos[self.obj_adr + 3:self.obj_adr + 7] = [1, 0, 0, 0]
        model.body_pos[self.crate_body][:2] = crate_pos[:2]
        return {"object_at": [float(v) for v in obj_at],
                "crate_at": [float(v) for v in crate_at],
                "object_pos": [float(v) for v in obj_pos[:2]],
                "crate_pos": [float(v) for v in crate_pos[:2]]}


class Operator:
    """The scripted hand on the keys: the same moves, at a held-key rate."""

    def __init__(self, robot: Robot, recorder: DemoRecorder,
                 yaw: float | None = None):
        self.robot, self.recorder = robot, recorder
        self.yaw = yaw                  # degrees; None picks by object kind
        self.jog: Jog | None = None
        self.dock_c = math.cos(robot.dock_yaw)
        self.dock_s = math.sin(robot.dock_yaw)

    def run(self, seconds: float) -> None:
        for _ in range(int(seconds / self.robot.model.opt.timestep)):
            self.jog.step()
            self.robot.rig.step()
            self.recorder.step(self.jog)

    def walk(self, goal, straight: bool = False, limit: float = 30.0) -> bool:
        """Nudge the target to `goal` as a key held down would: one axis at a
        time, or `straight` along the line to it - which is how something in
        the hand travels, because a carry that turns corners shakes it out.
        False if IK refuses before it gets there."""
        c, s = self.dock_c, self.dock_s
        jog, refused, t0 = self.jog, 0, self.robot.data.time
        while self.robot.data.time - t0 < limit:
            d = goal - jog.target
            if np.linalg.norm(d) < jog.step_size * 0.6:
                return True
            ahead, across, up = c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]
            if straight:
                unit = np.array([ahead, across, up]) / np.linalg.norm(d)
                why = jog.nudge(*unit)
                if why is None:
                    self.recorder.event("carry")
            else:
                axis, value = max(((abs(ahead), "ahead", ahead),
                                   (abs(across), "across", across),
                                   (abs(up), "up", up)))[1:]
                why = jog.nudge(**{axis: float(np.sign(value))})
                if why is None:
                    self.recorder.event(axis + ("+" if value > 0 else "-"))
            if why == "reach":
                refused += 1
                if refused > 5:
                    return False
            self.run(KEY_EVERY)
        return False

    def go(self, goal, **kw) -> bool:
        """A big move the way the harness makes them: one ramped move to an IK
        solution, rather than crept up on a centimetre at a time."""
        if not self.jog.go(goal, self.jog.yaw, **kw):
            return False
        self.recorder.event("go")
        while self.jog.busy:
            self.run(0.05)
        self.settle()
        return True

    def settle(self, patience: float = 3.0) -> None:
        """Wait for the arm to stop.  The ramp ends before the servos do -
        they carry kv = 0.1 kp and lag it - and letting go of a bowl while the
        hand is still on its way to the crate puts the bowl on the table."""
        t0 = self.robot.data.time
        self.run(0.2)
        while not self.jog.still and self.robot.data.time - t0 < patience:
            self.run(0.1)

    def grip(self, close: bool, wait: float) -> None:
        self.jog.toggle_grip()
        self.recorder.event("close" if close else "open")
        self.run(wait)

    def episode(self, obj: str) -> tuple[bool, str]:
        """One pick-and-place.  (success, what happened)."""
        robot = self.robot
        side = "left" if robot.across(obj) > 0 else "right"
        self.jog = Jog(robot, side, robot.items[obj].width)
        self.jog.ready()
        while self.jog.busy:
            self.run(0.05)
        yaw = self.yaw if self.yaw is not None else GRASP_YAW.get(
            robot.items[obj].kind, DEFAULT_YAW)
        self.jog.yaw = math.radians(yaw)      # after ready(), which resets it
        self.run(0.3)

        self.recorder.start(task=task_text(obj), target=obj, arm=side,
                            operator="scripted", grasp_yaw_deg=yaw)
        # Creep up on the object as an operator would.  Creeping keeps the arm
        # in the configuration it started in, and from the ready pose that
        # runs out a few centimetres short of the outer band on the arm's own
        # side - so if it is refused, get there the way the harness does, with
        # one free solve and a ramp.
        # Settle twice on the way down.  The trim only centres the hand while
        # it is at rest, and a hand that starts its descent 18 mm off lands a
        # pad on the bowl's rim.  The second stop is with the pads just above
        # the rim, so the last few centimetres go in straight.
        on = robot.jaw_target(obj) + [0, 0, GRASP_LIFT.get(robot.items[obj].kind, 0.0)]
        above = on + [0, 0, APPROACH]
        if not (self.walk(above) or self.go(above)):
            return False, "unreachable"
        self.run(SETTLE)
        if not self.walk(on + [0, 0, PRE_GRASP]):
            return False, "unreachable"
        self.run(SETTLE)
        if not self.walk(on):
            return False, "unreachable"
        self.run(SETTLE)
        self.grip(True, CLOSE_WAIT)
        if self.jog.held != obj:
            return False, "grasp_failed"
        if not self.walk(on + [0, 0, LIFT]):
            return False, "lift_refused"
        # Carry as one slow ramp in joint space, staying in this configuration.
        # A Cartesian carry - straight or cornered, fast or slow - rolls a ball
        # and tips a bowl out of the pads; the harness found this is the move
        # that keeps them, and it keeps them here too.
        drop = robot.free_spot_in(robot.crate)
        if not (self.go(drop, restarts=1) or self.go(drop)):
            return False, "carry_refused"
        self.run(0.5)
        # Still in the air is still in the hand.  The contact check flickers
        # on a ball held by two pads, and a flicker is not a drop.
        if robot.data.body(obj).xpos[2] < TABLE_H + LIFTED:
            return False, "dropped_on_carry"
        self.grip(False, RELEASE_WAIT)
        self.walk(self.jog.target + [0, 0, RETREAT])
        self.run(0.5)
        if self.settled_in_crate(obj):
            return True, "in the crate"
        return False, "missed: " + robot.where(obj)

    def settled_in_crate(self, obj: str) -> bool:
        """In the crate, on its floor, clear of its walls, and not moving.

        `inside_crate` answers a footprint test, which a ball bouncing off
        the rim passes on its way out.  This waits a moment, then asks the
        collision engine: touching the crate's floor, a positive distance to
        every wall, and still.
        """
        robot, m, d = self.robot, self.robot.model, self.robot.data
        before = d.body(obj).xpos.copy()
        self.run(0.5)
        if np.linalg.norm(d.body(obj).xpos - before) > 0.005:
            return False                      # still rolling
        if self.jog.held == obj or not robot.inside_crate(d.body(obj).xpos):
            return False
        crate = m.body(robot.crate).id
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == crate]
        obj_geom = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(obj).id)
        fromto = np.zeros(6)
        dist = [mujoco.mj_geomDistance(m, d, obj_geom, g, 0.2, fromto) for g in geoms]
        floor, walls = dist[0], dist[1:]
        return floor < 0.002 and all(w > 0.0 for w in walls)


def reset(robot: Robot, layout: Layout, obj_at, crate_at) -> dict:
    model, data = robot.model, robot.data
    mujoco.mj_resetData(model, data)
    where = layout.apply(obj_at, crate_at)
    mujoco.mj_forward(model, data)
    hold_everything(model, data)
    robot.rig.seconds(0.4)
    return where


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="ball")
    ap.add_argument("--object", help="what to pick; default the first thing "
                                     "on the table that can be")
    ap.add_argument("--episodes", type=int, default=200,
                    help="successful episodes to collect")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hz", type=int, default=20)
    ap.add_argument("--out", default=str(REPO / "out" / "demos"))
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="give up after this many episodes tried (default 3x)")
    ap.add_argument("--yaw", type=float, default=None,
                    help="wrist yaw for the grasp in degrees; default by object")
    ap.add_argument("--viz-every", type=int, default=10,
                    help="render every Nth kept episode to a GIF; 0 for none")
    args = ap.parse_args()

    robot = Robot(args.table)
    obj = args.object or next(n for n, i in robot.items.items() if i.graspable)
    if obj not in robot.items or not robot.items[obj].graspable:
        raise SystemExit(f"nothing called {obj!r} to pick on {robot.table.name}")
    out = Path(args.out) / args.table
    good = DemoRecorder(robot, out, args.hz)
    bad = DemoRecorder(robot, out / "failed", args.hz)
    rng = np.random.default_rng(args.seed)
    layout = Layout(robot, obj, rng)
    viz = Viz(robot, out / "viz")
    cap = args.max_attempts or 3 * args.episodes

    print(f"{robot.table.name}: {task_text(obj)}, {args.episodes} episodes, "
          f"seed {args.seed} -> {out}")
    done, attempts, reasons, started = 0, 0, {}, time.time()
    while done < args.episodes and attempts < cap:
        attempts += 1
        obj_at, crate_at = layout.sample()
        where = reset(robot, layout, obj_at, crate_at)
        operator = Operator(robot, good, args.yaw)
        ok, note = operator.episode(obj)
        meta = dict(layout=where, note=note, attempt=attempts)
        if ok:
            good.demo.meta.update(meta)
            path = good.stop(True)
            done += 1
            if args.viz_every and done % args.viz_every == 0:
                gif = viz(path, operator.jog.side, args.hz)
                print(f"  viz -> {gif}", flush=True)
        else:
            bad.demo, good.demo = good.demo, None
            bad.demo.meta.update(meta)
            path = bad.stop(False)
            reasons[note] = reasons.get(note, 0) + 1
        rate = (time.time() - started) / attempts
        print(f"  {attempts:4d}  {'ok  ' if ok else 'FAIL'} {note:<22} "
              f"object ({obj_at[0]:+.2f}, {obj_at[1]:+.2f}) crate "
              f"({crate_at[0]:+.2f}, {crate_at[1]:+.2f})  {done}/{args.episodes} "
              f"kept  {rate:.0f}s/ep", flush=True)

    print(f"\n{done} episodes in {out} from {attempts} attempts, "
          f"{time.time() - started:.0f}s")
    if reasons:
        print("failures: " + ", ".join(f"{k} x{v}" for k, v in
                                       sorted(reasons.items(), key=lambda kv: -kv[1])))
    if done < args.episodes:
        raise SystemExit(f"stopped short: {done} of {args.episodes}")


if __name__ == "__main__":
    main()
