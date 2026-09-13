"""Drive one arm by hand and record demonstrations of the box pick-and-place.

    .venv/bin/mjpython scripts/teleop.py                  # right arm, blue cube, base welded
    .venv/bin/mjpython scripts/teleop.py --cube l          # the green cube instead
    .venv/bin/mjpython scripts/teleop.py --balance         # on the wheels, station keeping
    .venv/bin/mjpython scripts/teleop.py --jitter 0.02     # scatter the cubes +-2 cm on each reset

    W / S      jaws forward / back (toward the table)    SPACE   close / open the hand
    A / D      jaws left / right                         1-4     which cube the task is about
    R / F      jaws up / down                            X       swap arms
    Q / E      wrist counter-clockwise / clockwise       H       home: reset the scene
    [ / ]      finer / coarser steps                     ENTER   start / stop recording
    C          cancel the recording                      P       print where things are
    A press moves 1 cm or 5 degrees; hold the key and it repeats.

mjpython, not python: on macOS the window has to be created on the process main
thread.  Keys arrive on that thread and the sim runs on this one, so presses go
through a queue and are applied between steps rather than during them.

Episodes land in out/demos/<table>/ as .npz files.  They hold the state, not the
pictures: run scripts/render_demos.py afterwards to get camera frames.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlbot.grasp import hold_everything                              # noqa: E402
from rlbot.skills import Robot                                       # noqa: E402
from rlbot.teleop import NUDGE_RANGE, DemoRecorder, Jog, task_text  # noqa: E402

FPS = 60                   # window refreshes per second
ENTER, SPACE = 257, 32     # GLFW key codes the viewer hands back

NUDGES = {                 # key -> (ahead, across, up, turn)
    "W": (1, 0, 0, 0), "S": (-1, 0, 0, 0),
    "A": (0, 1, 0, 0), "D": (0, -1, 0, 0),
    "R": (0, 0, 1, 0), "F": (0, 0, -1, 0),
    "Q": (0, 0, 0, 1), "E": (0, 0, 0, -1),
}


class Session:
    """The scene, the arm under control, and what the keys do to them."""

    def __init__(self, robot: Robot, side: str, cube: str, out_dir: Path,
                 hz: int, jitter: float):
        self.robot = robot
        self.cube = cube
        self.jitter = jitter
        self.rng = np.random.default_rng()
        self.cubes = [n for n, i in robot.items.items() if i.graspable]
        self.recorder = DemoRecorder(robot, out_dir, hz)
        self.keys: deque[int] = deque()
        self.jog = Jog(robot, side, robot.items[cube].width)
        if robot.balancing:
            self.home_qpos = robot.data.qpos.copy()
        self.reset()

    def key(self, code: int) -> None:
        """Runs on the window's thread: just remember it."""
        self.keys.append(code)

    # ---- the scene ---------------------------------------------------------
    def reset(self) -> None:
        if self.recorder.active:
            print(f"  reset: dropped {self.recorder.discard()} rows")
        model, data = self.robot.model, self.robot.data
        mujoco.mj_resetData(model, data)
        if self.robot.balancing:
            data.qpos[:] = self.home_qpos
        for name in self.cubes:
            if self.jitter:
                adr = model.jnt_qposadr[model.body(name).jntadr[0]]
                data.qpos[adr:adr + 2] += self.rng.uniform(-self.jitter,
                                                           self.jitter, 2)
        mujoco.mj_forward(model, data)
        hold_everything(model, data)
        if self.robot.rig.driver is not None:
            driver = self.robot.rig.driver
            driver.control.reset(driver.angle(data))
        self.robot.rig.seconds(0.4)
        self.jog.closed = False
        self.jog.target = self.jog.jaws.copy()
        if not self.jog.ready():
            print("  cannot reach the ready pose from here")

    def swap_arm(self) -> None:
        side = "left" if self.jog.side == "right" else "right"
        self.jog = Jog(self.robot, side, self.robot.items[self.cube].width)
        self.jog.ready()
        print(f"  {side} arm")

    def choose(self, cube: str) -> None:
        self.cube = cube
        self.jog.set_width(self.robot.items[cube].width)
        print(f"  task: {task_text(cube)}")

    def status(self) -> None:
        j = self.jog
        pos = j.target
        print(f"  {j.side} arm  jaws at ({pos[0]:+.3f}, {pos[1]:+.3f}, "
              f"{pos[2]:.3f})  wrist {math.degrees(j.yaw):+.0f} deg  "
              f"hand {'closed' if j.closed else 'open'}"
              f"{' on ' + j.held if j.held else ''}  step {j.step_size * 100:.2g} cm")
        print("  " + self.robot.scene().replace("\n", "\n  "))

    # ---- recording ---------------------------------------------------------
    def toggle_recording(self) -> None:
        if not self.recorder.active:
            self.recorder.start(task=task_text(self.cube), target=self.cube,
                                arm=self.jog.side)
            print(f"  recording: {task_text(self.cube)}")
            return
        success = (self.robot.inside_crate(self.robot.data.body(self.cube).xpos)
                   and self.jog.held != self.cube)
        rows = len(self.recorder.demo.rows)
        path = self.recorder.stop(success)
        print(f"  {'success' if success else 'FAILED'}: {rows} rows -> {path}")

    def cancel(self) -> None:
        print(f"  cancelled: dropped {self.recorder.discard()} rows")

    # ---- keys, applied between sim steps ------------------------------------
    def apply(self) -> None:
        while self.keys:
            code = self.keys.popleft()
            name = chr(code).upper() if 0 < code < 128 else ""
            if name in NUDGES and not self.jog.busy:
                why = self.jog.nudge(*NUDGES[name])
                if why == "reach":
                    print("  refused: cannot reach that")
                elif why is None:
                    self.recorder.event(name)
            elif code == SPACE:
                self.jog.toggle_grip()
                self.recorder.event("close" if self.jog.closed else "open")
            elif name and name in "1234" and int(name) <= len(self.cubes):
                self.choose(self.cubes[int(name) - 1])
            elif name == "X":
                self.swap_arm()
            elif name == "H":
                self.reset()
            elif name == "[":
                self.jog.step_size = max(NUDGE_RANGE[0], self.jog.step_size / 2)
            elif name == "]":
                self.jog.step_size = min(NUDGE_RANGE[1], self.jog.step_size * 2)
            elif code == ENTER:
                self.toggle_recording()
            elif name == "C":
                self.cancel()
            elif name == "P":
                self.status()


def draw_target(scene, jog: Jog) -> None:
    """A dot where the jaws are being sent, red once the hand is closing."""
    scene.ngeom = 0
    colour = (0.9, 0.2, 0.2, 0.8) if jog.closed else (0.2, 0.9, 0.4, 0.8)
    mujoco.mjv_initGeom(scene.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.008, 0, 0]), jog.target, np.eye(3).flatten(),
                        np.array(colour, dtype=np.float32))
    scene.ngeom = 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="cubes")
    ap.add_argument("--cube", default="m", choices=["s", "m", "l", "xl"],
                    help="which cube the task is about")
    ap.add_argument("--arm", default="right", choices=["right", "left"])
    ap.add_argument("--balance", action="store_true",
                    help="stand on the wheels with the station keeper running")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="m to scatter the cubes by on each reset")
    ap.add_argument("--hz", type=int, default=20, help="rows per second recorded")
    ap.add_argument("--out", default=str(REPO / "out" / "demos"))
    args = ap.parse_args()

    robot = Robot(args.table, balancing=args.balance)
    cube = f"cube_{args.cube}" if args.table == "cubes" else next(
        n for n, i in robot.items.items() if i.graspable)
    session = Session(robot, args.arm, cube, Path(args.out) / args.table,
                      args.hz, args.jitter)
    print(__doc__.split("\n\n")[2])
    print(f"\n  task: {task_text(cube)}   ({'balancing' if args.balance else 'base welded'})")

    model, data = robot.model, robot.data
    every = max(1, round(1 / (FPS * model.opt.timestep)))
    step = 0
    with mujoco.viewer.launch_passive(model, data,
                                      key_callback=session.key) as viewer:
        rot = data.body("root").xmat.reshape(3, 3)
        viewer.cam.lookat[:] = (data.body("root").xpos
                                + rot[:, 0] * 0.45 + np.array([0, 0, 0.3]))
        viewer.cam.distance = 1.6
        viewer.cam.elevation = -25
        viewer.cam.azimuth = math.degrees(math.atan2(rot[1, 0], rot[0, 0])) + 150
        clock = time.time()
        while viewer.is_running():
            session.apply()
            session.jog.step()
            robot.rig.step()
            session.recorder.step(session.jog)
            step += 1
            if step % every == 0:
                draw_target(viewer.user_scn, session.jog)
                viewer.sync()
                ahead = every * model.opt.timestep - (time.time() - clock)
                if ahead > 0:
                    time.sleep(ahead)
                clock = time.time()

    if session.recorder.active:
        print(f"  window closed: dropped {session.recorder.discard()} rows")


if __name__ == "__main__":
    main()
