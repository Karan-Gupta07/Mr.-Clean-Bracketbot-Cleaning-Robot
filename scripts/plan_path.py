"""Plan every route between the start pose and the three table docks, and check them.

    .venv/bin/python scripts/plan_path.py
    .venv/bin/python scripts/plan_path.py --route cubes-ware
    .venv/bin/python scripts/plan_path.py --quiet          # no maps, just the verdicts

Draws each route on the room's occupancy map and refuses routes that clip the
furniture, break the drive limits, or do not arrive at the dock pose pointing
the right way. Pure geometry: nothing here moves the robot. `navigate.py` is
where the schedule meets the physics.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlbot.navmap import OccupancyGrid                  # noqa: E402
from rlbot.planner import Limits, NoPath, plan, profile  # noqa: E402
from rlbot.room import TABLES                           # noqa: E402

START = (0.0, 0.0, 0.0)        # the room's 'start' keyframe


def routes():
    """(name, start pose, goal pose): the start to each dock, and dock to dock."""
    docks = {t.name.split("_")[1]: t.dock for t in TABLES}
    out = [(f"start-{n}", START, p) for n, p in docks.items()]
    out += [(f"{a}-{b}", pa, pb)
            for a, pa in docks.items() for b, pb in docks.items() if a != b]
    return out


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def check_route(grid, name, start, goal, limits, verbose=True) -> list[str]:
    try:
        path = plan(grid, start, goal, limits)
        traj = profile(path, limits)
        if not all(np.isfinite(a).all() for a in (traj.t, traj.xy, traj.yaw, traj.v, traj.omega)):
            raise NoPath("trajectory contains nonfinite values")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            dt = np.diff(traj.t)
            if not np.isfinite(dt).all() or (dt <= 0).any():
                raise NoPath("trajectory time must have finite, strictly positive intervals")
            accel = np.abs(np.diff(traj.v) / dt)
            alpha = np.abs(np.diff(traj.omega) / dt)
        if not np.isfinite(accel).all() or not np.isfinite(alpha).all():
            raise NoPath("trajectory has nonfinite linear or angular acceleration")
    except NoPath as error:
        print(f"  MISS {name:<12} {error}")
        return [f"{name}: {error}"]
    clear = float(grid.clearance(path.xy).min()) if len(path.xy) else math.inf
    # clearance() measures to cell centres; the surface is at most half a diagonal further out
    need = limits.radius + limits.margin - grid.resolution * math.sqrt(2) / 2
    end_xy = math.hypot(traj.xy[-1][0] - goal[0], traj.xy[-1][1] - goal[1])
    end_yaw = math.degrees(abs(wrap(traj.yaw[-1] - goal[2])))
    start_yaw = math.degrees(abs(wrap(traj.yaw[0] - start[2])))
    problems = []
    if clear < need:
        problems.append(f"{name}: curve passes {clear * 100:.0f} cm from furniture, needs {need * 100:.0f}")
    if np.abs(traj.v).max() > limits.v_max + 1e-6:
        problems.append(f"{name}: speed {np.abs(traj.v).max():.3f} over {limits.v_max}")
    if np.abs(traj.omega).max() > limits.omega_max + 1e-6:
        problems.append(f"{name}: yaw rate {np.abs(traj.omega).max():.3f} over {limits.omega_max}")
    if accel.max(initial=0.0) > limits.a_max + 1e-6:
        problems.append(f"{name}: acceleration {accel.max():.3f} over {limits.a_max}")
    if end_xy > 0.01 or end_yaw > 1.0:
        problems.append(f"{name}: ends {end_xy * 100:.1f} cm and {end_yaw:.1f} deg off the goal")
    if start_yaw > 1.0:
        problems.append(f"{name}: starts {start_yaw:.1f} deg off the heading")
    if verbose:
        print(grid.ascii(path.xy, marks=[start[:2], goal[:2]]))
    flag = "ok  " if not problems else "MISS"
    print(f"  {flag} {name:<12} exit {path.exit:+.2f} m, turn {math.degrees(path.turn):+4.0f} deg, "
          f"curve {path.length:.2f} m, dock {path.dock:+.2f} m   {traj.duration:5.1f} s, "
          f"min radius {path.min_radius:.2f} m, clearance {clear * 100:.0f} cm, "
          f"peak alpha {alpha.max(initial=0.0):.2f} rad/s2")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--route", help="one route, e.g. start-cubes or ware-ball")
    ap.add_argument("--quiet", action="store_true", help="verdicts only, no maps")
    args = ap.parse_args()
    chosen = [r for r in routes() if args.route is None or r[0] == args.route]
    if not chosen:
        ap.error(f"no route {args.route!r}; one of " + ", ".join(r[0] for r in routes()))
    grid = OccupancyGrid.from_room()
    limits = Limits()
    problems = []
    for name, start, goal in chosen:
        problems += check_route(grid, name, start, goal, limits, verbose=not args.quiet)
    if problems:
        print("\nnot driveable as planned:\n  " + "\n  ".join(problems))
        return 1
    print(f"\n{len(chosen)} routes planned within the limits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
