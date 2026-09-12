"""Drive the BracketBot around the room it has to work in.

    .venv/bin/mjpython scripts/room.py              # start in the middle
    .venv/bin/mjpython scripts/room.py --at cubes   # parked at a table
    .venv/bin/mjpython scripts/room.py --no-lidar   # without the scan overlay

mjpython, not python: on macOS the window has to be created on the process main
thread.  launch_passive routes it there; mujoco.viewer.launch() does not.

    W / S      drive forward / back        1 2 3   jump to a table's docking pose
    A / D      turn left / right           R       back to the middle of the room
    SPACE      stop                        L       lidar overlay on / off
    [ / ]      arms down the rail / up     P       print the pose estimate

The balancer is running the whole time, so this is the robot as it actually is:
every command goes through a thing that is falling over and catching itself.
Driving forward means asking it to lean, and the lidar ring leans with it.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlbot import BalanceController, Gains, Lidar, WheelOdometry, true_pose  # noqa: E402
from rlbot.arm import ARM_JOINTS                                            # noqa: E402
from rlbot.robot import ROOM, State                                         # noqa: E402
from rlbot.room import TABLES                                               # noqa: E402

# A balancing robot cannot be told to change speed; it can only be told to lean.
# Stepping the wheel-speed target from 6 rad/s to 0 asks for the whole change at
# once, and the PD baseline - tuned to stand still and survive a shove, not to
# drive - puts itself on the floor.  Keep the targets modest and slew them.
DRIVE = 3.0        # rad/s of wheel speed asked for by one press of W (0.25 m/s)
TURN = 0.6         # rad/s of yaw asked for by A or D (1.2 tips it over)
SLEW = 2.0         # rad/s per second the wheel-speed target may change
RAIL_STEP = 0.08   # m the arm carriages move per keypress
SCAN_HZ = 10       # the lidar's own rate, not the solver's


class Console:
    """Keyboard state, and everything the keys drive."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.speed = 0.0        # what the keys ask for
        self.turn = 0.0
        self.applied = 0.0      # what the controller is actually given
        self.show_lidar = True
        self.rail = 0.0
        self.rail_acts = [model.actuator(f"{s}j0").id for s in ("r", "l")]
        self.docks = {ord(str(i + 1)): f"dock_{t.name.split('_')[1]}"
                      for i, t in enumerate(TABLES)}

    def key(self, code: int) -> None:
        if code in (ord("W"), ord("w")):
            self.speed += DRIVE
        elif code in (ord("S"), ord("s")):
            self.speed -= DRIVE
        elif code in (ord("A"), ord("a")):
            self.turn += TURN
        elif code in (ord("D"), ord("d")):
            self.turn -= TURN
        elif code == ord(" "):
            self.speed = self.turn = 0.0
        elif code in (ord("L"), ord("l")):
            self.show_lidar = not self.show_lidar
        elif code == ord("["):
            self.set_rail(self.rail - RAIL_STEP)
        elif code == ord("]"):
            self.set_rail(self.rail + RAIL_STEP)
        elif code in (ord("R"), ord("r")):
            self.jump("start")
        elif code in self.docks:
            self.jump(self.docks[code])

    def slew(self, dt: float) -> float:
        """Walk the applied speed target toward what the keys asked for."""
        step = SLEW * dt
        self.applied += float(np.clip(self.speed - self.applied, -step, step))
        return self.applied

    def set_rail(self, value: float) -> None:
        lo, hi = self.model.actuator_ctrlrange[self.rail_acts[0]]
        self.rail = float(np.clip(value, lo, hi))
        for act in self.rail_acts:
            self.data.ctrl[act] = self.rail

    def jump(self, keyframe: str) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data,
                                    self.model.key(keyframe).id)
        hold_arms(self.model, self.data)
        self.speed = self.turn = self.applied = 0.0
        self.rail = float(self.data.ctrl[self.rail_acts[0]])
        print(f"  -> {keyframe}")


def hold_arms(model, data) -> None:
    """Point every arm servo at the pose the arm is already in.

    Position servos default to a command of zero, which for the mast carriages
    means 'slam to the top of the rail'.  Whatever else is going on, the arms
    should start by staying where they are.
    """
    for side in ARM_JOINTS.values():
        for name in side:
            act = model.actuator(name).id
            joint = model.joint(name).id
            data.ctrl[act] = data.qpos[model.jnt_qposadr[joint]]
    for hand in ("right_left_gripper", "left_left_gripper"):
        act = model.actuator(hand).id
        data.ctrl[act] = data.qpos[model.jnt_qposadr[model.joint(hand).id]]


def read_state(model, data, sensors) -> State:
    gyro, vel_l, vel_r = sensors
    rot = data.body("root").xmat.reshape(3, 3)
    return State(
        pitch=math.atan2(rot[0, 2], rot[2, 2]),
        pitch_rate=float(data.sensordata[gyro + 1]),
        yaw=math.atan2(rot[1, 0], rot[0, 0]),
        yaw_rate=float(data.sensordata[gyro + 2]),
        wheel_speed=float((data.sensordata[vel_l] + data.sensordata[vel_r]) / 2),
        forward_speed=float(rot[:, 0] @ data.qvel[0:3]),
        height=float(data.body("root").xpos[2]),
    )


def draw_scan(scene, points) -> None:
    """Put the scan into the viewer's own scratch scene, one dot per return."""
    scene.ngeom = 0
    for point in points[: scene.maxgeom]:
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.022, 0, 0]), point, np.eye(3).flatten(),
            np.array([0.15, 0.9, 0.45, 0.85], dtype=np.float32))
        scene.ngeom += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", default="start",
                    choices=["start"] + [t.name.split("_")[1] for t in TABLES],
                    help="start parked at a table instead of mid-room")
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--beams", type=int, default=72)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(ROOM))
    data = mujoco.MjData(model)
    key = "start" if args.at == "start" else f"dock_{args.at}"
    mujoco.mj_resetDataKeyframe(model, data, model.key(key).id)
    mujoco.mj_forward(model, data)

    sensors = (model.sensor("gyro").adr[0], model.sensor("vel_left").adr[0],
               model.sensor("vel_right").adr[0])
    wheels = [model.actuator(f"wheel_{s}").id for s in ("left", "right")]
    control = BalanceController(Gains.for_bracketbot())
    lidar = Lidar(model, beams=args.beams)
    odom = WheelOdometry()
    odom.reset(*true_pose(data).as_array())

    console = Console(model, data)
    console.show_lidar = not args.no_lidar
    hold_arms(model, data)
    console.rail = float(data.ctrl[console.rail_acts[0]])

    print(__doc__.split("\n\n")[2])       # the key map
    scan_every = max(1, int(1 / (SCAN_HZ * model.opt.timestep)))
    step = 0

    with mujoco.viewer.launch_passive(model, data,
                                      key_callback=console.key) as viewer:
        while viewer.is_running():
            tick = time.time()
            state = read_state(model, data, sensors)
            left, right = control(state, yaw_rate_ref=console.turn,
                                  speed_ref=console.slew(model.opt.timestep))
            data.ctrl[wheels] = (left, right)
            mujoco.mj_step(model, data)

            odom.update(float(data.sensordata[sensors[1]]),
                        float(data.sensordata[sensors[2]]),
                        model.opt.timestep, yaw_rate=state.yaw_rate)

            if step % scan_every == 0:
                if console.show_lidar:
                    draw_scan(viewer.user_scn, lidar.points(data))
                elif viewer.user_scn.ngeom:
                    viewer.user_scn.ngeom = 0
            step += 1

            viewer.sync()
            slack = model.opt.timestep - (time.time() - tick)
            if slack > 0:
                time.sleep(slack)

    truth, guess = true_pose(data), odom.pose
    print(f"\nfinished at ({truth.x:+.2f}, {truth.y:+.2f}, "
          f"{math.degrees(truth.yaw):+.0f} deg)")
    print(f"wheel odometry thought ({guess.x:+.2f}, {guess.y:+.2f}, "
          f"{math.degrees(guess.yaw):+.0f} deg) - "
          f"{math.hypot(truth.x - guess.x, truth.y - guess.y) * 100:.0f} cm of drift")


if __name__ == "__main__":
    main()
