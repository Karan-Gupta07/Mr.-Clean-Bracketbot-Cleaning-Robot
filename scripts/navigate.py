"""Drive the robot to each table dock in the sim, on the schedule, and check it arrives.

    .venv/bin/python scripts/navigate.py                      # all 9 routes, headless
    .venv/bin/python scripts/navigate.py --route start-cubes
    .venv/bin/mjpython scripts/navigate.py --route start-cubes --view

The balancer and the DriveController run exactly as the ROS bridge runs them:
500 Hz physics, commands at 50 Hz, and the navigator ticked at --nav-hz (10 Hz,
the node's rate). The navigator gets the sim's true pose in
place of the SLAM estimate and replans from it. A route passes if the robot
ends within 10 cm and 5 degrees of the dock, never falls, and never touches the
furniture. It also reports how long the chassis spent tilted past the 2 degrees
the bridge's scan gate allows at a stretch, because that is the number that
says whether driving will starve SLAM of scans.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlbot import Balancer, Gains                                  # noqa: E402
from rlbot.control import DriveController                          # noqa: E402
from rlbot.navigate import Navigator, true_pose                    # noqa: E402
from rlbot.navmap import OccupancyGrid                             # noqa: E402
from rlbot.planner import goal_for                                 # noqa: E402
from rlbot.robot import ROOM                                       # noqa: E402
from rlbot.room import TABLES                                      # noqa: E402

COMMAND_HZ = 50            # the bridge ticks its controller at 50 Hz
TILT_GATE = math.radians(2)
ARRIVE_XY, ARRIVE_YAW = 0.10, 5.0
_SPATIAL_EPS = 1e-9


@dataclass
class Result:
    name: str
    err_xy: float = math.nan
    err_yaw: float = math.nan       # degrees
    seconds: float = 0.0
    fell: bool = False
    touched: set = field(default_factory=set)
    replans: int = 0
    max_off: float = 0.0
    tilt_streak: float = 0.0        # longest stretch past the bridge's tilt gate, s
    failed: str | None = None

    @property
    def ok(self) -> bool:
        return (not self.fell and not self.touched and self.failed is None
                and self.err_xy <= ARRIVE_XY + _SPATIAL_EPS
                and self.err_yaw <= ARRIVE_YAW + math.degrees(_SPATIAL_EPS))


def routes():
    docks = {t.name.split("_")[1]: t.dock for t in TABLES}
    out = [(f"start-{n}", "start", n) for n in docks]
    out += [(f"{a}-{b}", f"dock_{a}", b) for a in docks for b in docks if a != b]
    return out


def furniture_contacts(model, data) -> set:
    """Names of room geoms the robot is touching. Wheels on the floor do not count."""
    root = model.body("root").id
    names = set()
    for k in range(data.ncon):
        con = data.contact[k]
        roots = [model.body_rootid[model.geom_bodyid[g]] for g in (con.geom1, con.geom2)]
        if (roots[0] == root) == (roots[1] == root):
            continue
        other = con.geom2 if roots[0] == root else con.geom1
        if model.geom_type[other] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        names.add(model.geom(other).name or f"geom{other}")
    return names


def drive(name: str, keyframe: str, table: str, grid, nav_hz: float, view: bool = False) -> Result:
    bot = Balancer(ROOM, chassis="root", keyframe=keyframe)
    model, data = bot.model, bot.data
    radius = float(model.geom("wheel_left_collision").size[0])
    control = DriveController(radius, Gains.for_bracketbot())
    nav = Navigator(grid)
    goal = goal_for(table)
    result = Result(name, fell=bot.has_fallen(), touched=furniture_contacts(model, data))
    nav.go_to(goal, true_pose(data), data.time)
    budget = nav.traj.duration * 5 + 15 if nav.traj is not None else 0.0
    every = max(1, int(round(1 / (COMMAND_HZ * bot.dt))))
    think = max(1, int(round(1 / (nav_hz * bot.dt))))
    command = (0.0, 0.0)
    streak, step = 0.0, 0
    window = (mujoco.viewer.launch_passive(model, data)
              if view and nav.traj is not None else contextlib.nullcontext())
    with window as viewer:
        while data.time < budget and not nav.done and not nav.failed and not result.fell:
            tick = time.monotonic()
            if viewer is not None and not viewer.is_running():
                result.failed = "viewer closed before the route finished"
                break
            if step % think == 0:
                command = nav.step(true_pose(data), data.time)
            if step % every == 0:
                control.command(*command, data.time)
            bot.step(*control(bot.state(), data.time, bot.dt))
            mujoco.mj_forward(model, data)
            step += 1
            result.fell = bot.has_fallen(bot.state())
            result.touched |= furniture_contacts(model, data)
            tilt = math.acos(float(np.clip(data.body("root").xmat.reshape(3, 3)[2, 2], -1.0, 1.0)))
            streak = streak + bot.dt if tilt > TILT_GATE else 0.0
            result.tilt_streak = max(result.tilt_streak, streak)
            if viewer is not None:
                viewer.sync()
                slack = bot.dt - (time.monotonic() - tick)
                if slack > 0:
                    time.sleep(slack)
    x, y, yaw = true_pose(data)
    result.err_xy = math.hypot(x - goal[0], y - goal[1])
    result.err_yaw = math.degrees(abs(math.atan2(math.sin(yaw - goal[2]), math.cos(yaw - goal[2]))))
    result.seconds = float(data.time)
    result.replans = len(nav.replans)
    result.max_off = max((off for _, off in nav.replans), default=0.0)
    result.failed = nav.failed or result.failed
    if not nav.done and result.failed is None and not result.fell:
        result.failed = f"ran out of time after {budget:.0f} s"
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--route", help="one route, e.g. start-cubes or ware-ball")
    ap.add_argument("--view", action="store_true", help="watch it (needs mjpython)")
    ap.add_argument("--nav-hz", type=float, default=10, help="navigator rate; the ROS node runs 10 Hz")
    args = ap.parse_args()
    chosen = [r for r in routes() if args.route is None or r[0] == args.route]
    if not chosen:
        ap.error(f"no route {args.route!r}; one of " + ", ".join(r[0] for r in routes()))
    if not 0 < args.nav_hz <= COMMAND_HZ:
        ap.error(f"--nav-hz must be above 0 and at most the {COMMAND_HZ} Hz command rate")
    grid = OccupancyGrid.from_room()
    print(f"  {'route':<12} {'arrive':>14}  {'time':>6}  {'replans':>7}  {'max off':>7}  {'tilt>2deg':>9}")
    failures = []
    for name, keyframe, table in chosen:
        r = drive(name, keyframe, table, grid, args.nav_hz, view=args.view)
        verdict = "ok  " if r.ok else "FAIL"
        why = ("fell" if r.fell else ", ".join(sorted(r.touched)) if r.touched else r.failed or "")
        print(f"  {verdict} {name:<12} {r.err_xy * 100:5.1f} cm {r.err_yaw:4.1f} deg  "
              f"{r.seconds:5.0f} s  {r.replans:7d}  {r.max_off * 100:5.0f} cm  "
              f"{r.tilt_streak:6.2f} s   {why}", flush=True)
        if not r.ok:
            failures.append(name)
    if failures:
        print(f"\n{len(failures)} of {len(chosen)} routes failed: " + ", ".join(failures))
        return 1
    print(f"\n{len(chosen)} routes arrived within {ARRIVE_XY * 100:.0f} cm and {ARRIVE_YAW:.0f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
