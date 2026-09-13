"""Unit-level checks for the navigation stack, on grids small enough to reason about.

    .venv/bin/python scripts/check_navigation.py

No sim, no ROS, a few seconds.  `plan_path.py` runs the real room; `navigate.py`
runs the physics.  This is the one to run after touching navmap, planner or
navigate, because when it fails it says which line of which module.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from rlbot.navmap import OccupancyGrid          # noqa: E402
from rlbot.navigate import Navigator, true_pose  # noqa: E402
from rlbot.planner import (   # noqa: E402
    Limits, NoPath, Path as NavPath, Trajectory, _clamped_spline, _fit, _resample, _sample_spline,
    _wrap, astar, goal_for, plan, profile, shortcut,
)


def small_grid():
    """A 1 x 1 m room at 10 cm cells with a 30 cm post in the middle."""
    occupied = np.zeros((11, 11), dtype=bool)
    occupied[4:7, 4:7] = True
    return OccupancyGrid(occupied, 0.10, (0.0, 0.0))


def check_grid():
    g = small_grid()
    assert g.cell(0.0, 0.0) == (0, 0) and g.cell(0.51, 0.49) == (5, 5)
    assert g.world(5, 5) == (0.5, 0.5)
    assert g.free(0.1, 0.1) and not g.free(0.5, 0.5)
    assert not g.free(-0.2, 0.0), "outside the grid is not free"
    assert g.line_free((0.1, 0.1), (0.9, 0.1))
    assert not g.line_free((0.1, 0.5), (0.9, 0.5)), "a line through the post is blocked"
    np.testing.assert_array_equal(g.free_at([(0.1, 0.1), (0.5, 0.5), (2.0, 0.0)]),
                                  [True, False, False])

    fat = g.inflate(0.15)
    assert fat.occupied.sum() > g.occupied.sum()
    assert not fat.free(0.3, 0.5) and fat.free(0.2, 0.2)
    assert not fat.free(0.0, 0.5), "the border inflates inward"
    assert not g.inflate(0.10).free(0.3, 0.5), "inflation reaches exactly its radius"

    assert g.rect_free(0.2, 0.2, 0.0, 0.05, 0.05)
    assert not g.rect_free(0.32, 0.5, 0.0, 0.10, 0.05), "a rectangle reaching the post"
    assert g.rect_free(0.32, 0.5, math.pi / 2, 0.10, 0.05), "the same rectangle turned side-on"

    np.testing.assert_allclose(g.clearance([(0.2, 0.5), (0.9, 0.9)]),
                               [0.2, math.hypot(0.3, 0.3)], atol=1e-9)

    art = g.ascii([(0.1, 0.1)], marks=[(0.9, 0.9)])
    assert "*" in art and "#" in art and "+" in art

    room = OccupancyGrid.from_room()
    assert room.occupied.shape == (121, 91) and room.resolution == 0.05
    assert room.free(0.0, 0.0), "the middle of the room is open"
    assert not room.free(2.25, -1.10), "the ball table blocks"
    assert not room.free(1.15, 1.30), "the pillar blocks"
    assert not room.free(0.60, 1.55), "the divider blocks"
    assert not room.free(3.0, 0.0), "the wall blocks"
    print("grid ok")


def check_pgm():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image = np.full((4, 6), 254, dtype=np.uint8)        # 254 is free in a saved map
        image[0, :] = 0                                     # top row occupied: that is max y
        image[2, 1] = 205                                   # one unknown cell
        (tmp / 'map.pgm').write_bytes(b'P5\n# made by a test\n6 4\n255\n' + image.tobytes())
        (tmp / 'map.yaml').write_text(
            'image: map.pgm\nmode: trinary\nresolution: 0.5\norigin: [-1.0, -2.0, 0.0]\n'
            'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
        g = OccupancyGrid.from_pgm(tmp / 'map.yaml')
        assert g.occupied.shape == (6, 4) and g.resolution == 0.5
        assert g.origin == (-0.75, -1.75), 'cell (0, 0) is half a cell in from the yaml origin'
        assert g.occupied[:, 3].all(), 'the top image row is the highest y row'
        assert not g.occupied[:, 0].any()
        assert g.occupied[1, 1], 'unknown counts as occupied'
        assert g.occupied.sum() == 6 + 1
        assert g.occupied.dtype == np.bool_ and g.occupied.flags.c_contiguous
        boxed = OccupancyGrid.from_pgm(tmp / 'map.yaml', extra_boxes=[(0.25, -0.75, 0.0, 0.1, 0.1)])
        assert boxed.occupied.sum() == 6 + 1 + 1 and boxed.occupied[2, 2]
        assert g.occupied.sum() == 6 + 1, 'overlays do not affect another loaded grid'
    print('pgm ok')


def check_astar():
    g = small_grid()
    cells = astar(g, (1, 5), (9, 5))
    assert cells[0] == (1, 5) and cells[-1] == (9, 5)
    assert all(not g.occupied[c] for c in cells)
    assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 for a, b in zip(cells, cells[1:]))
    assert sum(math.dist(a, b) for a, b in zip(cells, cells[1:])) > 8, "has to go round the post"
    points = np.array([g.world(i, j) for i, j in cells])
    keep = shortcut(g, points)
    assert keep[0] == 0 and keep[-1] == len(points) - 1 and len(keep) < len(points)
    assert all(g.line_free(points[a], points[b]) for a, b in zip(keep, keep[1:]))
    assert all(a < b for a, b in zip(keep, keep[1:]))
    assert all(not g.line_free(points[a], p) for a, b in zip(keep, keep[1:]) for p in points[b + 1:])
    with np.testing.assert_raises(NoPath):
        astar(g, (5, 5), (9, 5))
    walled = small_grid()
    walled.occupied[5, :] = True
    with np.testing.assert_raises(NoPath):
        astar(walled, (1, 5), (9, 5))
    print("astar ok")


def check_astar_edges():
    g = small_grid()
    assert astar(g, (1, 1), (1, 1)) == [(1, 1)]
    for invalid in ((-1, 5), (11, 5), (5, -1), (5, 11), (5, 5)):
        with np.testing.assert_raises(NoPath):
            astar(g, invalid, (1, 1))
        with np.testing.assert_raises(NoPath):
            astar(g, (1, 1), invalid)
    with np.testing.assert_raises(NoPath):
        astar(g, (5, 5), (5, 5))

    for start in ((0, 0), (0, 1), (1, 0), (1, 1)):
        goal = (1 - start[0], 1 - start[1])
        sides = ((goal[0], start[1]), (start[0], goal[1]))
        for blocked in ((), (sides[0],), (sides[1],), sides):
            corner = OccupancyGrid(np.zeros((2, 2), dtype=bool), 0.5, (-2.0, 3.0))
            for cell in blocked:
                corner.occupied[cell] = True
            before = corner.occupied.copy()
            if len(blocked) == 2:
                with np.testing.assert_raises(NoPath):
                    astar(corner, start, goal)
            else:
                cells = astar(corner, start, goal)
                assert cells[0] == start and cells[-1] == goal and len(cells) == 2 + len(blocked)
                points = np.array([corner.world(*c) for c in cells])
                assert all(corner.line_free(a, b) for a, b in zip(points, points[1:]))
                assert shortcut(corner, points) == list(range(len(points)))
            np.testing.assert_array_equal(corner.occupied, before)
    print("astar edges ok")


def check_shortcut():
    g = small_grid()
    points = np.array([g.world(i, 1) for i in (1, 2, 3)])
    before = points.copy()
    assert shortcut(g, points[:1]) == [0]
    assert shortcut(g, points[:2]) == [0, 1]
    assert shortcut(g, points) == [0, 2]
    np.testing.assert_array_equal(points, before)
    for blocked in (
        [(0.1, 0.5), (0.9, 0.5)],
        [(0.1, 0.5), (0.9, 0.5), (0.9, 0.6)],
    ):
        with np.testing.assert_raises(NoPath):
            shortcut(g, blocked)
    print("shortcut ok")


def check_spline():
    knots = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    coef = _clamped_spline(knots, np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    p, yaw, kappa, s = _sample_spline(coef, 0.01)
    np.testing.assert_allclose(p[0], knots[0], atol=1e-9)
    np.testing.assert_allclose(p[-1], knots[-1], atol=1e-9)
    assert abs(yaw[0]) < 1e-6 and abs(yaw[-1] - math.pi / 2) < 1e-6, "clamped to the end tangents"
    assert np.all(np.diff(s) > 0)
    assert np.min(np.linalg.norm(p - knots[1], axis=1)) < 0.011, "passes through the middle knot"
    assert np.abs(kappa).max() < 20
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    lim = Limits(radius=0.1, margin=0.05)
    path = plan(g, (0.3, 0.3, 0.0), (1.7, 1.2, math.pi / 2), lim)
    assert path.exit == 0 and path.dock == 0 and path.turn == 0
    np.testing.assert_allclose(path.xy[0], [0.3, 0.3], atol=1e-6)
    np.testing.assert_allclose(path.xy[-1], [1.7, 1.2], atol=1e-6)
    assert abs(path.yaw[0]) < 1e-3, "leaves along the heading"
    assert abs(path.yaw[-1] - math.pi / 2) < 1e-3, "arrives along the goal yaw"
    assert np.allclose(np.diff(path.s), path.s[1] - path.s[0], atol=1e-9), "resampled by arc length"
    assert g.inflate(0.15).free_at(path.xy).all()
    assert path.length > 1.5 and path.min_radius > 0.2
    print("spline ok")


def check_dock():
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    g.fill_box(1.0, 1.7, 0.0, 0.5, 0.3)              # a table top: x 0.5..1.5, y 1.4..2.0
    lim = Limits(radius=0.1, margin=0.05, half_depth=0.05, half_width=0.09)
    goal = (1.0, 1.28, math.pi / 2)                  # 12 cm off the edge, facing it
    assert not g.inflate(lim.radius + lim.margin).free(*goal[:2]), "the dock is inside the inflation"
    path = plan(g, (0.3, 0.3, math.radians(50)), goal, lim)
    assert path.exit == 0 and path.turn == 0
    assert 0.05 < path.dock < 0.15, path.dock
    np.testing.assert_allclose(path.xy[-1], [1.0, 1.28 - path.dock], atol=1e-6)
    assert abs(path.yaw[-1] - math.pi / 2) < 1e-3
    back = plan(g, goal, (0.3, 0.3, math.pi), lim)
    assert back.exit < 0, "leaves a dock in reverse"
    assert abs(back.turn) > math.radians(45), "then turns to face the corridor"
    np.testing.assert_allclose(back.xy[0], [1.0, 1.28 + back.exit], atol=1e-6)
    assert abs(_wrap(back.yaw[0] - (math.pi / 2 + back.turn))) < 1e-3
    with np.testing.assert_raises(NoPath):
        plan(g, (0.3, 0.3, 0.0), (1.0, 1.7, 0.0), lim)
    print("dock ok")


def check_straight_out():
    from rlbot.planner import BACKOFF_MAX, _straight_out

    g = OccupancyGrid(np.zeros((121, 41), dtype=bool), 0.05, (-3.0, -1.0))
    half = (0.01, 0.01)
    pose = (0.0, 0.0, 0.0)
    assert _straight_out(g, g.inflate(0.1), pose, half) == 0
    g.fill_box(-0.15, 0.1, 0.0, 0.25, 0.0)
    before = g.occupied.copy()
    inflated = g.inflate(0.1)
    assert inflated.line_free((0.15, 0.0), (0.15, 0.0)), "a nearer forward anchor exists"
    distance = _straight_out(g, inflated, pose, half)
    np.testing.assert_allclose(distance, -0.45, atol=1e-12)
    assert g.rect_free(distance / 2, 0.0, 0.0, half[0] + abs(distance) / 2, half[1])
    assert inflated.line_free((distance, 0.0), (distance, 0.0))
    np.testing.assert_array_equal(g.occupied, before)

    rear = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    rear.occupied[4, 20] = True
    lim = Limits(radius=0.1, margin=0.0, half_depth=half[0], half_width=half[1])
    start, goal = (0.3, 1.0, 0.0), (0.8, 1.0, 0.0)
    path = plan(rear, start, goal, lim)
    np.testing.assert_allclose(path.exit, 0.05, atol=1e-12)
    assert path.turn == path.dock == 0, "blocked reverse falls back to a forward exit"
    np.testing.assert_allclose(path.xy[0], (0.35, 1.0), atol=1e-12)
    assert rear.rect_free(0.3 + path.exit / 2, 1.0, 0.0, half[0] + path.exit / 2, half[1])
    approach = plan(rear, (0.35, 1.0, 0.0), start, lim)
    np.testing.assert_allclose(approach.dock, -0.05, atol=1e-12)
    assert approach.exit == approach.turn == approach.length == 0 and not len(approach.xy)
    np.testing.assert_allclose(approach.waypoints, [(0.35, 1.0)] * 2, atol=1e-12)
    back = plan(rear, (0.8, 1.0, math.pi), (0.3, 1.0, math.pi), lim)
    assert back.exit == back.turn == 0 and back.dock > 0
    np.testing.assert_allclose(back.xy[-1], (0.3 + back.dock, 1.0), atol=1e-12)

    coarse = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.2, (-4.0, -4.0))
    coarse.occupied[19, 20] = coarse.occupied[21, 20] = True
    tiny = (0.005, 0.005)
    trapped = (0.03, 0.0, 0.0)
    for sign in (-1, 1):
        assert all(coarse.rect_free(0.03 + sign * k * 0.1, 0.0, 0.0, *tiny)
                   for k in range(11)), "endpoint samples miss both blocking cell centres"
        assert coarse.inflate(0.2).free(0.03 + sign, 0.0)
    with np.testing.assert_raises(NoPath):
        _straight_out(coarse, coarse.inflate(0.2), trapped, tiny)
    lim = Limits(radius=0.2, margin=0.0, half_depth=tiny[0], half_width=tiny[1])
    for start, goal in ((trapped, (1.0, 0.0, 0.0)), ((1.0, 0.0, math.pi), trapped)):
        with np.testing.assert_raises(NoPath):
            plan(coarse, start, goal, lim)

    edge = OccupancyGrid(np.zeros((21, 21), dtype=bool), 0.1, (0.0, 0.0))
    np.testing.assert_allclose(_straight_out(edge, edge.inflate(0.1), (0.02, 1.0, 0.0), (0.04, 0.04)),
                               0.05, atol=1e-12)
    for invalid in ((-0.02, 1.0, 0.0), (0.02, 1.0, math.pi / 2)):
        with np.testing.assert_raises(NoPath):
            _straight_out(edge, edge.inflate(0.1), invalid, (0.04, 0.04))

    long = OccupancyGrid(np.zeros((121, 41), dtype=bool), 0.05, (-3.0, -1.0))
    long.fill_box(0.0, 0.1, 0.0, 0.95, 0.0)
    assert _straight_out(long, long.inflate(0.1), pose, half) == -BACKOFF_MAX == -1.0
    long.fill_box(0.0, 0.1, 0.0, 1.0, 0.0)
    assert long.inflate(0.1).line_free((-1.05, 0.0), (-1.05, 0.0))
    with np.testing.assert_raises_regex(NoPath, "no room within 1.0 m"):
        _straight_out(long, long.inflate(0.1), pose, half)
    print("straight out ok")


def check_dock_stationary_margin():
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    g.fill_box(1.0, 1.7, 0.0, 0.5, 0.3)
    lim = Limits(radius=0.1, margin=0.05, half_depth=0.05, half_width=0.09)
    pose = (1.0, 1.28, math.pi / 2)
    path = plan(g, pose, pose, lim)
    assert path.start == path.goal == pose
    assert path.exit == path.turn == path.dock == path.length == 0
    assert path.xy.shape == (0, 2) and path.yaw.shape == path.kappa.shape == path.s.shape == (0,)
    np.testing.assert_array_equal(path.waypoints, [pose[:2], pose[:2]])
    for x, y in ((1.1, 1.0), (1.0, 1.1)):
        blocked = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
        blocked.occupied[blocked.cell(x, y)] = True
        pose = (1.0, 1.0, 0.0)
        assert blocked.rect_free(*pose, 0.06, 0.06), "the bare footprint fits"
        lim = Limits(radius=0.1, margin=0.05, half_depth=0.06, half_width=0.06)
        for start, goal in ((pose, pose), ((0.3, 0.3, 0.0), pose), (pose, (0.3, 0.3, 0.0))):
            with np.testing.assert_raises_regex(NoPath, "does not fit"):
                plan(blocked, start, goal, lim)
    print("dock stationary margin ok")


def check_profile():
    lim = Limits()
    n = 51
    straight = NavPath(start=(0.0, 0.0, 0.0), goal=(1.0, 0.0, 0.0), exit=0.0, turn=0.0,
                       xy=np.column_stack([np.linspace(0, 1, n), np.zeros(n)]),
                       yaw=np.zeros(n), kappa=np.zeros(n), s=np.linspace(0, 1, n),
                       dock=0.0, waypoints=np.array([[0.0, 0.0], [1.0, 0.0]]))
    traj = profile(straight, lim)
    assert isinstance(traj, Trajectory)
    assert traj.v[0] == 0 and traj.v[-1] == 0 and traj.v.max() <= lim.v_max + 1e-9
    assert np.all(np.diff(traj.t) > 0)
    accel = np.abs(np.diff(traj.v) / np.diff(traj.t))
    assert accel.max() <= lim.a_max * 1.1, accel.max()
    trapezoid = 1.0 / lim.v_max + lim.v_max / lim.a_max
    assert abs(traj.duration - trapezoid) < 0.3, (traj.duration, trapezoid)
    xy, yaw, v, omega = traj.at(traj.duration / 2)
    assert 0.3 < xy[0] < 0.7 and abs(v - lim.v_max) < 1e-9 and omega == 0
    assert traj.at(1e9)[2] == 0, "clamped past the end"
    theta = np.linspace(0, math.pi / 2, 40)                      # a 0.2 m quarter circle
    bend = NavPath(start=(0.2, 0.0, math.pi / 2), goal=(0.0, 0.2, math.pi), exit=0.0, turn=0.0,
                   xy=np.column_stack([0.2 * np.cos(theta), 0.2 * np.sin(theta)]),
                   yaw=theta + math.pi / 2, kappa=np.full(40, 5.0), s=0.2 * theta,
                   dock=0.0, waypoints=np.zeros((2, 2)))
    traj = profile(bend, lim)
    assert np.abs(traj.omega).max() <= lim.omega_max + 1e-9
    assert traj.v.max() <= lim.omega_max / 5 + 1e-9, "slowed for the bend"
    assert traj.v[1:-1].min() > 0, "never parks mid-arc"
    spin = NavPath(start=(0.0, 0.0, 0.0), goal=(0.0, 0.0, math.pi / 2), exit=0.0, turn=math.pi / 2,
                   xy=np.zeros((0, 2)), yaw=np.zeros(0), kappa=np.zeros(0), s=np.zeros(0),
                   dock=0.0, waypoints=np.zeros((2, 2)))
    traj = profile(spin, lim)
    assert np.all(traj.v == 0) and np.abs(traj.omega).max() <= lim.omega_max + 1e-9
    assert abs(traj.yaw[-1] - math.pi / 2) < 1e-9 and np.allclose(traj.xy, 0)
    alpha = np.abs(np.diff(traj.omega) / np.diff(traj.t))
    assert alpha.max() <= lim.alpha_max * 1.1
    out_and_in = NavPath(start=(0.0, 0.0, 0.0), goal=(0.1, 0.0, 0.0), exit=-0.1, turn=0.0,
                         xy=np.zeros((0, 2)), yaw=np.zeros(0), kappa=np.zeros(0), s=np.zeros(0),
                         dock=0.2, waypoints=np.zeros((2, 2)))
    traj = profile(out_and_in, lim)
    assert traj.v.min() < 0, "the exit is driven in reverse"
    np.testing.assert_allclose(traj.xy[-1], [0.1, 0.0], atol=1e-9)
    assert np.all(traj.yaw == 0)
    print("profile ok")


def check_profile_caps():
    theta = np.linspace(0, math.pi / 2, 40)
    for radius, sign, lim in (
        (0.1, 1, Limits()), (0.1, -1, Limits()), (0.2, 1, Limits(v_max=0.02)),
    ):
        path = NavPath(start=(radius, 0.0, sign * math.pi / 2),
                       goal=(0.0, sign * radius, sign * math.pi), exit=0.0, turn=0.0,
                       xy=np.column_stack([radius * np.cos(theta), sign * radius * np.sin(theta)]),
                       yaw=sign * (theta + math.pi / 2), kappa=np.full(40, sign / radius),
                       s=radius * theta, dock=0.0, waypoints=np.zeros((2, 2)))
        before = [a.copy() for a in (path.xy, path.yaw, path.kappa, path.s)]
        traj = profile(path, lim)
        assert np.abs(traj.omega).max() <= lim.omega_max + 1e-12, "curve cap must not exceed omega_max"
        assert traj.v.max() <= lim.v_max + 1e-12, "curve cap must not exceed v_max"
        assert traj.v[0] == traj.v[-1] == traj.omega[0] == traj.omega[-1] == 0
        assert np.all(traj.v[1:-1] > 0) and np.all(sign * traj.omega[1:-1] > 0)
        assert np.all(np.diff(traj.t) > 0) and np.isfinite(traj.t).all()
        assert np.abs(np.diff(traj.v) / np.diff(traj.t)).max() <= lim.a_max + 1e-12
        np.testing.assert_allclose(traj.omega, traj.v * path.kappa, atol=1e-12)
        np.testing.assert_allclose(np.diff(traj.t) * (traj.v[1:] + traj.v[:-1]) / 2,
                                   np.diff(path.s), rtol=1e-12, atol=1e-14)
        for actual, expected in zip((path.xy, path.yaw, path.kappa, path.s), before):
            np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(traj.xy, path.xy) and not np.shares_memory(traj.yaw, path.yaw)
    print("profile caps ok")


def check_profile_trims():
    lim = Limits(v_max=0.03, alpha_max=0.001)
    theta = np.linspace(0, 0.02, 3)
    path = NavPath(start=(0.2, 0.0, math.pi / 2),
                   goal=(0.2 * math.cos(0.02), 0.2 * math.sin(0.02), math.pi / 2 + 0.02),
                   exit=0.0, turn=0.0,
                   xy=np.column_stack([0.2 * np.cos(theta), 0.2 * np.sin(theta)]),
                   yaw=theta + math.pi / 2, kappa=np.full(3, 5.0), s=0.2 * theta,
                   dock=0.0, waypoints=np.zeros((2, 2)))
    traj = profile(path, lim)
    expected = lim.v_max * 0.9 ** 10
    np.testing.assert_allclose(traj.v, [0.0, expected, 0.0], rtol=1e-12, atol=0,
                               err_msg="the tenth trim must affect the returned schedule")
    np.testing.assert_allclose(traj.t, [0.0, 0.004 / expected, 0.008 / expected], rtol=1e-12)
    np.testing.assert_allclose(traj.omega, traj.v * path.kappa, atol=1e-12)
    alpha = np.abs(np.diff(traj.omega) / np.diff(traj.t))
    print(f"profile trims ok (curve alpha {alpha.max():.6f} rad/s^2; preferred limit {lim.alpha_max:g})")


def check_profile_phases():
    from rlbot.planner import DS, DTH

    lim = Limits()
    theta = np.linspace(0, math.pi / 2, 40)
    path = NavPath(start=(0.0, 0.0, math.pi / 2), goal=(0.2, 0.0, math.pi / 2),
                   exit=-0.1, turn=-math.pi / 2,
                   xy=np.column_stack([0.2 * np.sin(theta), 0.1 - 0.2 * np.cos(theta)]),
                   yaw=theta, kappa=np.full(40, 5.0), s=0.2 * theta,
                   dock=-0.1, waypoints=np.zeros((2, 2)))
    traj = profile(path, lim)
    counts = [max(3, int(np.ceil(abs(path.exit) / DS)) + 1),
              max(3, int(np.ceil(abs(path.turn) / DTH)) + 1), len(theta),
              max(3, int(np.ceil(abs(path.dock) / DS)) + 1)]
    stops = np.r_[0, np.cumsum(np.array(counts) - 1)]
    assert len(traj.t) == stops[-1] + 1, "joins do not duplicate the at-rest sample"
    np.testing.assert_array_equal(np.flatnonzero((traj.v == 0) & (traj.omega == 0)), stops)
    np.testing.assert_allclose(traj.xy[stops], [[0, 0], [0, -0.1], [0, -0.1], [0.2, 0.1], [0.2, 0]],
                               atol=1e-12)
    np.testing.assert_allclose(traj.yaw[stops], [math.pi / 2, math.pi / 2, 0, math.pi / 2, math.pi / 2],
                               atol=1e-12)
    assert traj.t[0] == 0 and np.all(np.diff(traj.t) > 0)
    assert np.abs(traj.v).max() <= lim.v_max + 1e-12
    assert np.abs(traj.omega).max() <= lim.omega_max + 1e-12
    assert np.abs(np.diff(traj.v) / np.diff(traj.t)).max() <= lim.a_max + 1e-12
    for phase, (a, b) in enumerate(zip(stops, stops[1:])):
        interior = slice(a + 1, b)
        if phase == 1:
            assert np.all(traj.v[a:b + 1] == 0) and np.all(traj.omega[interior] < 0)
            np.testing.assert_allclose(traj.xy[a:b + 1], np.tile(traj.xy[a], (b - a + 1, 1)), atol=1e-12)
            dt = np.diff(traj.t[a:b + 1])
            assert np.abs(np.diff(traj.omega[a:b + 1]) / dt).max() <= lim.alpha_max + 1e-12
            np.testing.assert_allclose(dt * (traj.omega[a + 1:b + 1] + traj.omega[a:b]) / 2,
                                       np.diff(traj.yaw[a:b + 1]), atol=1e-12)
        else:
            sign = 1 if phase == 2 else -1
            assert np.all(sign * traj.v[interior] > 0)
            if phase != 2:
                assert np.all(traj.omega[a:b + 1] == 0)
                np.testing.assert_allclose(traj.yaw[a:b + 1], math.pi / 2, atol=1e-12)
        for index in (a, b):
            xy, yaw, v, omega = traj.at(traj.t[index])
            np.testing.assert_allclose(xy, traj.xy[index], atol=1e-12)
            assert abs(_wrap(yaw - traj.yaw[index])) < 1e-12 and v == omega == 0
    for time, index in ((-1e9, 0), (1e9, -1)):
        xy, yaw, v, omega = traj.at(time)
        np.testing.assert_allclose(xy, traj.xy[index], atol=1e-12)
        assert abs(_wrap(yaw - traj.yaw[index])) < 1e-12 and v == omega == 0
    print("profile phases ok")


def check_profile_edges():
    from dataclasses import replace

    pose = (0.0, 0.0, 0.0)
    empty = NavPath(start=pose, goal=pose, exit=0.0, turn=0.0,
                    xy=np.zeros((0, 2)), yaw=np.zeros(0), kappa=np.zeros(0), s=np.zeros(0),
                    dock=0.0, waypoints=np.zeros((2, 2)))
    still = profile(empty)
    assert profile.__defaults__ == (None,), "each call gets fresh default limits"
    assert still.duration == 0 and still.t.shape == still.v.shape == still.omega.shape == (1,)
    for time in (-1e9, 0.0, 1e9):
        xy, yaw, v, omega = still.at(time)
        np.testing.assert_array_equal(xy, pose[:2])
        assert yaw == v == omega == 0

    lim = Limits()
    for distance in (0.01, 1e-6, 1e-8, 1e-20):
        for sign in (-1, 1):
            amount = sign * distance
            for phase in ("exit", "dock", "turn"):
                goal = (0.0, 0.0, amount) if phase == "turn" else (amount, 0.0, 0.0)
                path = replace(empty, goal=goal, **{phase: amount})
                traj = profile(path, lim)
                assert len(traj.t) == 3 and traj.duration > 0, "small nonzero phases must not disappear"
                assert np.all(np.diff(traj.t) > 0)
                accel = lim.alpha_max if phase == "turn" else lim.a_max
                np.testing.assert_allclose(traj.duration, 2 * math.sqrt(distance / accel), rtol=1e-12, atol=0)
                speed = traj.omega if phase == "turn" else traj.v
                assert speed[0] == speed[-1] == 0 and sign * speed[1] > 0
                np.testing.assert_allclose(np.diff(traj.t) * (speed[1:] + speed[:-1]) / 2,
                                           np.full(2, amount / 2), rtol=1e-12, atol=0)
                np.testing.assert_allclose(traj.xy[-1], goal[:2], rtol=1e-12, atol=0)
                np.testing.assert_allclose(traj.yaw[-1], goal[2], rtol=1e-12, atol=0)
            s = np.linspace(0, distance, 3)
            tiny = replace(empty, goal=(distance, 0.0, 0.0), xy=np.column_stack([s, np.zeros(3)]),
                           yaw=np.zeros(3), kappa=np.zeros(3), s=s)
            traj = profile(tiny, lim)
            assert len(traj.t) == 3 and traj.v[1] > 0
            np.testing.assert_allclose(traj.duration, 2 * math.sqrt(distance / lim.a_max), rtol=1e-12, atol=0)
            np.testing.assert_allclose(traj.xy[-1], tiny.goal[:2], rtol=1e-12, atol=0)

    wrapped = Trajectory(np.array([0.0, 1.0]), np.array([[0.0, 0.0], [2.0, 4.0]]),
                         np.radians([179.0, -179.0]), np.array([0.0, 0.1]), np.array([-0.1, 0.1]))
    xy, yaw, v, omega = wrapped.at(0.5)
    np.testing.assert_allclose(xy, [1.0, 2.0], atol=1e-12)
    assert abs(abs(yaw) - math.pi) < 1e-12 and abs(v - 0.05) < 1e-12 and abs(omega) < 1e-12
    for time, index in ((-1.0, 0), (2.0, -1)):
        xy, yaw, v, omega = wrapped.at(time)
        np.testing.assert_array_equal(xy, wrapped.xy[index])
        assert abs(_wrap(yaw - wrapped.yaw[index])) < 1e-12
        assert v == wrapped.v[index] and omega == wrapped.omega[index]
    print("profile edges ok")


def check_plan_edges():
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    inflated = g.inflate(0.26)
    pose = (0.5, 0.5, 0.0)
    path = plan(g, pose, pose)
    assert path.start == pose and path.goal == pose
    assert path.exit == path.turn == path.dock == path.length == 0
    assert path.min_radius == math.inf
    assert path.xy.shape == (0, 2)
    assert path.yaw.shape == path.kappa.shape == path.s.shape == (0,)
    np.testing.assert_array_equal(path.waypoints, [pose[:2], pose[:2]])
    for syaw, gyaw, turn in (
        (0.0, math.pi / 2, math.pi / 2),
        (math.radians(179), math.radians(-179), math.radians(2)),
        (math.radians(-179), math.radians(179), math.radians(-2)),
        (0.0, math.radians(0.5), 0.0),
    ):
        path = plan(g, (0.5, 0.5, syaw), (0.5, 0.5, gyaw))
        assert path.exit == path.dock == path.length == 0 and not len(path.xy)
        np.testing.assert_allclose(path.turn, turn, atol=1e-12)
    for heading, turn in ((math.pi / 2, -math.pi / 2), (math.pi / 4, 0.0)):
        path = plan(g, (0.5, 0.5, heading), (1.5, 0.5, 0.0))
        assert path.turn == turn, "turn first only above 45 degrees"
        np.testing.assert_allclose(path.yaw[[0, -1]], [heading + turn, 0.0], atol=1e-12)
        assert path.exit == path.dock == 0
        assert all(inflated.line_free(a, b) for a, b in zip(path.xy, path.xy[1:]))
    print("plan edges ok")


def check_plan_short_distance():
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    start = (0.5, 0.5, 0.0)
    for distance in (1e-6, 0.01, 0.019, 0.02):
        goal = (0.5 + distance, 0.5, 0.0)
        path = plan(g, start, goal)
        assert len(path.xy) >= 3, "a sub-DS translation must not disappear"
        assert path.exit == path.turn == path.dock == 0
        assert path.length > 0 and path.min_radius == math.inf
        np.testing.assert_allclose(path.xy[[0, -1]], [start[:2], goal[:2]], rtol=0, atol=1e-12)
        np.testing.assert_allclose(path.length, distance, rtol=0, atol=1e-12)
        assert np.all(np.diff(path.s) > 0)
        np.testing.assert_allclose(np.diff(path.s), path.s[1], rtol=0, atol=1e-12)
    print("plan short distance ok")


def check_spline_regularity():
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    for gyaw in (math.pi, -math.pi, math.pi - 1e-9, -math.pi + 1e-9):
        with np.testing.assert_raises_regex(NoPath, "nonregular"):
            plan(g, (0.5, 1.0, 0.0), (1.5, 1.0, gyaw))
    root = 0.37123
    axis = np.array([0.6, 0.8])
    coef = (np.array([0.0, 1.0]), np.zeros((1, 2)),
            np.array([root ** 2 * axis]), np.array([-root * axis]), np.array([axis / 3]))
    with np.testing.assert_raises_regex(NoPath, "nonregular"):
        _sample_spline(coef, 0.1)
    straight = plan(g, (0.5, 1.0, 0.0), (1.5, 1.0, 0.0))
    assert straight.min_radius == math.inf
    np.testing.assert_allclose(straight.length, 1.0, atol=1e-12)
    upper = plan(g, (0.5, 1.0, 0.0), (1.5, 1.0, math.pi / 2))
    lower = plan(g, (0.5, 1.0, 0.0), (1.5, 1.0, -math.pi / 2))
    np.testing.assert_allclose(upper.xy[:, 0], lower.xy[:, 0], atol=1e-12)
    np.testing.assert_allclose(upper.xy[:, 1], 2 - lower.xy[:, 1], atol=1e-12)
    np.testing.assert_allclose(upper.yaw, -lower.yaw, atol=1e-12)
    np.testing.assert_allclose(upper.kappa, -lower.kappa, atol=1e-12)
    assert np.max(np.abs(np.diff(np.unwrap(upper.yaw)))) < 0.2
    print("spline regularity ok")


def check_plan_short_curve():
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    inflated = g.inflate(0.26)
    for gyaw in (math.pi / 2, -math.pi / 2):
        start, goal = (0.5, 1.0, 0.0), (0.51, 1.0, gyaw)
        path = plan(g, start, goal)
        coef = _clamped_spline(np.array([start[:2], goal[:2]]), [1, 0], [0, math.sin(gyaw)])
        u, a, b, c, d = coef
        t = np.linspace(0, u[-1], 20001)[:, None]
        p = a[0] + b[0] * t + c[0] * t ** 2 + d[0] * t ** 3
        dp = b[0] + 2 * c[0] * t + 3 * d[0] * t ** 2
        yaw = np.unwrap(np.arctan2(dp[:, 1], dp[:, 0]))
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        ss = np.linspace(0, s[-1], len(path.xy))
        expected = np.column_stack([np.interp(ss, s, p[:, 0]), np.interp(ss, s, p[:, 1])])
        assert len(path.xy) >= 3 and path.exit == path.turn == path.dock == 0
        np.testing.assert_allclose(path.length, s[-1], rtol=1e-4, atol=1e-12)
        np.testing.assert_allclose(path.xy, expected, rtol=0, atol=2e-6)
        np.testing.assert_allclose(path.yaw, np.interp(ss, s, yaw), rtol=0, atol=2e-3)
        np.testing.assert_allclose(path.xy[[0, -1]], [start[:2], goal[:2]], atol=1e-12)
        np.testing.assert_allclose(path.yaw[[0, -1]], [0, gyaw], atol=1e-12)
        np.testing.assert_allclose(np.diff(path.s), path.s[1], atol=1e-12)
        assert np.max(np.abs(path.xy[:, 1] - 1.0)) > 0.001, "retain the interior bend"
        assert np.isfinite(path.kappa).all()
        assert all(inflated.line_free(a, b) for a, b in zip(path.xy, path.xy[1:]))
    print("plan short curve ok")


def check_plan_endpoints():
    g = small_grid()
    before = g.occupied.copy()
    lim = Limits(radius=0.1, margin=0.0, half_depth=0.06, half_width=0.08)
    inflated = g.inflate(0.1)
    half = (lim.half_depth, lim.half_width)
    assert g.free(0.3, 0.5) and not inflated.free(0.3, 0.5)
    assert inflated.free(0.25, 0.5), "a free rounded cell can still touch an obstacle"
    assert not inflated.line_free((0.25, 0.5), (0.25, 0.5))
    for x, y in ((0.5, 0.5), (-0.1, 0.2), (0.0, 0.2)):
        blocked = (x, y, 0.0)
        for other in ((x, y, 0.0), (x, y, math.pi / 2), (x + 0.01, y, 0.0), (0.1, 0.1, 0.0)):
            for start, goal in ((blocked, other), (other, blocked)):
                with np.testing.assert_raises(NoPath):
                    plan(g, start, goal, lim)
    for x in (0.3, 0.25):
        pose = (x, 0.5, 0.0)
        still = plan(g, pose, pose, lim)
        assert still.exit == still.turn == still.dock == still.length == 0 and not len(still.xy)
        approach = plan(g, (0.15, 0.5, 0.0), pose, lim)
        departure = plan(g, pose, (0.15, 0.5, math.pi), lim)
        assert approach.exit == approach.turn == departure.dock == 0
        np.testing.assert_allclose(approach.dock, x - 0.2, atol=1e-12)
        np.testing.assert_allclose(departure.exit, 0.2 - x, atol=1e-12)
        assert abs(departure.turn) == math.pi
        np.testing.assert_allclose(approach.xy[-1], (0.2, 0.5), atol=1e-12)
        np.testing.assert_allclose(departure.xy[0], (0.2, 0.5), atol=1e-12)
        distance = approach.dock
        assert g.rect_free(x - distance / 2, 0.5, 0.0, half[0] + distance / 2, half[1])
        for path in (approach, departure, plan(g, pose, (x + 0.01, 0.5, 0.0), lim)):
            assert all(inflated.line_free(a, b) for a, b in zip(path.xy, path.xy[1:]))
            assert path.length > 0
            for endpoint, distance in ((path.start, path.exit), (path.goal, -path.dock)):
                px, py, yaw = endpoint
                c, s = math.cos(yaw), math.sin(yaw)
                anchor = (px + distance * c, py + distance * s)
                assert inflated.line_free(anchor, anchor)
                assert g.rect_free(px + distance * c / 2, py + distance * s / 2, yaw,
                                   half[0] + abs(distance) / 2, half[1])
    assert g.rect_free(0.33, 0.5, 0.0, *half)
    assert not g.rect_free(0.33, 0.5, math.pi / 2, *half)
    for start, goal in (((0.33, 0.5, 0.0), (0.33, 0.5, math.pi / 2)),
                        ((0.33, 0.5, math.pi / 2), (0.33, 0.5, 0.0))):
        with np.testing.assert_raises(NoPath):
            plan(g, start, goal, lim)
    np.testing.assert_array_equal(g.occupied, before)
    print("plan endpoints ok")


def check_fit_retry():
    g = small_grid()
    before = g.occupied.copy()
    corridor = np.array([g.world(*c) for c in astar(g, (1, 5), (9, 5))])
    original = corridor.copy()
    keep = [0, len(corridor) - 1]
    tangent = np.array([1.0, 0.0])
    p, *_ = _sample_spline(_clamped_spline(corridor[keep], tangent, tangent), g.resolution / 2)
    assert not g.free_at(p).all(), "the initial fit crosses the post"
    xy, yaw, kappa, s, knots = _fit(g, corridor, keep, tangent, tangent)
    assert len(knots) > len(keep), "a clipped fit restores corridor knots"
    assert all(any(np.array_equal(knot, point) for point in corridor) for knot in knots)
    np.testing.assert_allclose(xy[[0, -1]], corridor[[0, -1]], atol=1e-12)
    np.testing.assert_allclose(yaw[[0, -1]], [0, 0], atol=1e-12)
    assert g.free_at(xy).all() and np.isfinite(kappa).all() and np.all(np.diff(s) > 0)
    assert all(g.line_free(a, b) for a, b in zip(xy, xy[1:]))
    assert keep == [0, len(corridor) - 1]
    np.testing.assert_array_equal(corridor, original)
    np.testing.assert_array_equal(g.occupied, before)
    with np.testing.assert_raises(NoPath):
        _fit(g, corridor[[0, -1]], [0, 1], tangent, tangent)
    print("fit retry ok")


def check_fit_corner_contact():
    g = OccupancyGrid(np.zeros((11, 11), dtype=bool), 0.1, (0.0, 0.0))
    g.occupied[3, 2] = True
    corridor = np.array([[0.2, 0.2], [0.3, 0.3]])
    tangent = np.array([1.0, 1.0]) / math.sqrt(2)
    raw = _sample_spline(_clamped_spline(corridor, tangent, tangent), g.resolution / 2)
    xy, *_ = _resample(*raw, 0.02)
    assert g.free_at(raw[0]).all() and g.free_at(xy).all()
    assert any(not g.line_free(a, b) for a, b in zip(xy, xy[1:])), "samples miss corner contact"
    with np.testing.assert_raises(NoPath):
        _fit(g, corridor, [0, 1], tangent, tangent)
    print("fit corner contact ok")


def check_fit_resampled_corner():
    g = OccupancyGrid(np.zeros((11, 11), dtype=bool), 0.01, (0.0, 0.0))
    g.occupied[3, 2] = True
    corridor = np.array([[0.019, 0.005], [0.026, 0.026]])
    t0, tn = np.array([0.0, 1.0]), np.array([1.0, 0.0])
    raw = _sample_spline(_clamped_spline(corridor, t0, tn), g.resolution / 2)
    p = raw[0]
    assert all(g.line_free(a, b) for a, b in zip(p, p[1:])), "the dense polyline is clear"
    xy, *_ = _resample(*raw, 0.02)
    assert g.free_at(xy).all()
    assert any(not g.line_free(a, b) for a, b in zip(xy, xy[1:])), "resampling can cut the corner"
    with np.testing.assert_raises(NoPath):
        _fit(g, corridor, [0, 1], t0, tn)
    print("fit resampled corner ok")


def check_goal_for():
    from rlbot.room import TABLES

    assert {table.name.split('_')[1] for table in TABLES} == {'ball', 'cubes', 'ware'}
    for table in TABLES:
        assert goal_for(table.name.split('_')[1]) == table.dock
    with np.testing.assert_raises_regex(KeyError, "no table called 'missing'"):
        goal_for('missing')
    print("goal_for ok")


def check_pgm_formats():
    from rlbot.navmap import _read_pgm

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'map.pgm'
        samples = np.array([[0, 3, 6], [9, 12, 15]], dtype=np.uint8)
        for raw in (
            b'P2\n# header\n3 # width\n2\n15\n0 3# pixel\n6\n9 12 15\n# end',
            b'P5\n# header\n3 # width\n2\n15\n' + samples.tobytes(),
        ):
            path.write_bytes(raw)
            np.testing.assert_array_equal(_read_pgm(path), samples.astype(float) * 17)

        for maxval, samples in (
            (1023, np.array([[0, 205], [512, 1023]], dtype='>u2')),
            (65535, np.array([[0, 205 * 257], [254 * 257, 65535]], dtype='>u2')),
        ):
            for magic in (b'P2', b'P5'):
                body = (b' '.join(str(int(v)).encode() for v in samples.flat)
                        if magic == b'P2' else samples.tobytes())
                path.write_bytes(magic + f'\n2 2\n{maxval}\n'.encode() + body)
                np.testing.assert_allclose(_read_pgm(path), samples.astype(float) * 255 / maxval)
                if maxval == 65535:
                    assert _read_pgm(path)[0, 1] == 205, 'normalized unknown retains its exact shade'
                    yaml_path = Path(tmp) / 'map.yaml'
                    yaml_path.write_text('image: map.pgm\nresolution: 1\norigin: [0, 0, 0]\n')
                    np.testing.assert_array_equal(OccupancyGrid.from_pgm(yaml_path).occupied,
                                                  [[False, True], [False, True]])

        for separator in (b'\n', b'\r\n', b' ', b'\t', b'\r', b'\v', b'\f'):
            for leading in (0, 9, 10, 11, 12, 13, 32, 35, 255):
                body = bytes([leading, 254, 205, 0])
                path.write_bytes(b'P5\n2 2\n255' + separator + body)
                np.testing.assert_array_equal(_read_pgm(path), [[leading, 254], [205, 0]])
    print('pgm formats ok')


def check_pgm_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'saved # map.pgm').write_bytes(b'P2\n7 1\n255\n0 63 64 128 191 205 255\n')
        yaml_path = tmp / 'map.yaml'
        for options, expected in (
            ('', [True, True, True, True, True, True, False]),
            ('free_thresh: 0.5\n', [True, True, True, False, False, True, False]),
            ('negate: 1\n', [False, False, True, True, True, True, True]),
            ('negate: 1\nfree_thresh: 0.9\noccupied_thresh: 0.95\n',
             [False, False, False, False, False, True, True]),
            (f'negate: 1\nfree_thresh: {64 / 255}\n',
             [False, False, True, True, True, True, True]),
        ):
            yaml_path.write_text(
                '# saved map\nimage: "saved # map.pgm" # relative to the YAML\n'
                'resolution: 0.5 # metres\norigin: [-1.0, 2.0, 0.0] # corner\n' + options)
            g = OccupancyGrid.from_pgm(yaml_path)
            assert g.origin == (-0.75, 2.25)
            np.testing.assert_array_equal(g.occupied[:, 0], expected)
    print('pgm metadata ok')


def check_pgm_thresholds():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        yaml_path = tmp / 'map.yaml'
        for maxval, values in (
            (255, (203, 204, 206)),
            (65535, (52427, 52428, 52429)),
            (1023, (819, 820, 821)),
            (65535, (52428, 52429, 52430)),
        ):
            threshold = (maxval - values[1]) / maxval
            for negate in (0, 1):
                samples = [maxval - v if negate else v for v in values]
                yaml_path.write_text(
                    'image: map.pgm\nresolution: 1\norigin: [0, 0, 0]\n'
                    f'free_thresh: {threshold}\noccupied_thresh: 0.65\n'
                    + ('negate: 1\n' if negate else ''))
                for magic in (b'P2', b'P5'):
                    body = (b' '.join(str(v).encode() for v in samples) if magic == b'P2'
                            else np.array(samples, dtype=np.uint8 if maxval < 256 else '>u2').tobytes())
                    (tmp / 'map.pgm').write_bytes(magic + f'\n3 1\n{maxval}\n'.encode() + body)
                    g = OccupancyGrid.from_pgm(yaml_path)
                    np.testing.assert_array_equal(
                        g.occupied[:, 0], [True, True, False],
                        err_msg=f'{magic!r}, maxval={maxval}, negate={negate}, threshold={threshold}')
    print('pgm thresholds ok')


def check_pgm_invalid():
    from rlbot.navmap import _read_pgm

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path = tmp / 'map.pgm'
        for raw in (
            b'', b'# no image', b'P5', b'P5\n1', b'P5\n1 1\n# no maxval',
            b'P3\n1 1\n255\n\x00', b'P5\nx 1\n255\n\x00',
            b'P5\n0 1\n255\n', b'P2\n1 -1\n255\n',
            b'P5\n1 1\n0\n\x00', b'P5\n1 1\n65536\n\x00\x00',
            b'P5\n1 1\n255', b'P5\n2 1\n255\n\x00', b'P5\n1 1\n65535\n\x00',
            b'P5\n1 1\n10\n\x0b',
            b'P2\n2 1\n255\n0 # missing pixel', b'P2\n1 1\n255\nnan',
            b'P2\n1 1\n255\n-1', b'P2\n1 1\n255\n256',
            b'P2\n1 1\n255\n1.5', b'P2\n1 1\n255\n0 1',
            b'P2\n1 1\n255\n1_0', b'P5\n1 1\n2_55\n\x00',
            b'P5\n1_0 1\n255\n' + bytes(10),
        ):
            path.write_bytes(raw)
            with np.testing.assert_raises(ValueError):
                _read_pgm(path)

        path.write_bytes(b'P5\n1 1\n255\n\xfe')
        yaml_path = tmp / 'map.yaml'
        base = {'image': 'map.pgm', 'resolution': '0.5', 'origin': '[0, 0, 0]'}
        for key, value in (
            ('origin', '[0, 0, 0.1]'), ('origin', '[0, 0, -1e-12]'),
            ('origin', '[0, 0, nan]'), ('origin', '[0, inf, 0]'),
            ('origin', '[0, 0]'), ('origin', '0'), ('origin', '[0, 0, 0'),
            ('resolution', '0'), ('resolution', '-1'), ('resolution', 'nan'), ('resolution', 'inf'),
            ('free_thresh', '-0.1'), ('free_thresh', '1.1'), ('free_thresh', 'nan'),
            ('free_thresh', '0.8'), ('occupied_thresh', '0.1'), ('occupied_thresh', 'inf'),
            ('occupied_thresh', '1.2'), ('negate', '2'), ('negate', '0.5'),
            ('mode', 'raw'), ('mode', 'scale'), ('image', ''),
        ):
            meta = dict(base, **{key: value})
            yaml_path.write_text(''.join(f'{k}: {v}\n' for k, v in meta.items()))
            with np.testing.assert_raises_regex(ValueError, key):
                OccupancyGrid.from_pgm(yaml_path)

        for key in base:
            meta = {k: v for k, v in base.items() if k != key}
            yaml_path.write_text(''.join(f'{k}: {v}\n' for k, v in meta.items()))
            with np.testing.assert_raises_regex(ValueError, key):
                OccupancyGrid.from_pgm(yaml_path)
    print('pgm invalid input ok')


def check_line_free():
    g = OccupancyGrid(np.zeros((11, 11), dtype=bool), 0.10, (0.0, 0.0))
    g.occupied[3, 2] = True
    assert g.cell(0.26, 0.245) == (3, 2) and not g.free(0.26, 0.245)
    blocked = (
        ((0.2, 0.2), (0.6, 0.5)),
        ((0.2, 0.2), (0.3, 0.3)),
        ((0.25, 0.1), (0.25, 0.4)),
        ((0.1, 0.25), (0.4, 0.25)),
        ((0.3, 0.1), (0.3, 0.4)),
        ((0.1, 0.2), (0.5, 0.2)),
        ((0.2, 0.2), (0.25, 0.2)),
        ((0.3, 0.2), (0.3, 0.2)),
        ((0.25, 0.25), (0.25, 0.25)),
        ((-0.051, 0.8), (0.5, 0.8)),
        ((0.5, 0.8), (1.051, 0.8)),
        ((0.8, -0.051), (0.8, 0.5)),
        ((0.8, 0.5), (0.8, 1.051)),
        ((-0.2, 0.8), (-0.1, 0.8)),
        ((-0.051, 0.8), (-0.051, 0.8)),
    )
    clear = (
        ((0.1, 0.1), (0.6, 0.1)),
        ((0.1, 0.1), (0.1, 0.7)),
        ((0.1, 0.1), (0.1, 0.1)),
        ((0.249, 0.1), (0.249, 0.4)),
        ((0.1, 0.251), (0.4, 0.251)),
        ((-0.05, -0.05), (1.05, -0.05)),
        ((1.05, -0.05), (1.05, 1.05)),
        ((-0.05, -0.05), (-0.05, -0.05)),
    )
    for expected, segments in ((False, blocked), (True, clear)):
        for p, q in segments:
            for start, end in ((p, q), (q, p)):
                assert g.line_free(start, end) == expected, (start, end, expected)

    shifted = OccupancyGrid(np.zeros((4, 6), dtype=bool), 0.5, (-2.0, 3.0))
    assert shifted.line_free((-2.25, 2.75), (-0.25, 5.75)), "physical map edges are inside"
    assert shifted.line_free((-0.25, 5.75), (-2.25, 2.75))
    print("line_free ok")


def check_rect_bounds():
    g = OccupancyGrid(np.zeros((21, 21), dtype=bool), 0.05, (0.0, 0.0))
    half = (0.0935995, 0.18575)
    assert not g.rect_free(0.05, 0.5, 0.0, *half), "the footprint protrudes past x = -0.025"
    lower, upper = -0.025, 1.025
    for yaw in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, -math.pi / 4):
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        rx, ry = c * half[0] + s * half[1], s * half[0] + c * half[1]
        assert g.rect_free(0.5, 0.5, yaw, *half)
        for x, y, dx, dy in (
            (lower + rx, 0.5, -1e-6, 0.0),
            (upper - rx, 0.5, 1e-6, 0.0),
            (0.5, lower + ry, 0.0, -1e-6),
            (0.5, upper - ry, 0.0, 1e-6),
        ):
            assert g.rect_free(x, y, yaw, *half), "a footprint may touch the map boundary"
            assert g.rect_free(x - dx, y - dy, yaw, *half), "just inside stays free"
            assert not g.rect_free(x + dx, y + dy, yaw, *half), "a corner is outside"
        for x, y in ((-1.0, 0.5), (2.0, 0.5), (0.5, -1.0), (0.5, 2.0)):
            assert not g.rect_free(x, y, yaw, *half), "wholly outside is blocked"

    g.occupied[10, 10] = True
    assert not g.rect_free(0.5, 0.5, math.pi / 4, *half)
    assert g.rect_free(0.5 - half[0] - 0.001, 0.5, 0.0, *half), "only occupied centres block"
    shifted = OccupancyGrid(np.zeros((4, 6), dtype=bool), 0.5, (-2.0, 3.0))
    assert shifted.rect_free(-1.25, 4.25, 0.0, 1.0, 1.5)
    assert not shifted.rect_free(-1.250001, 4.25, 0.0, 1.0, 1.5)
    print("rect bounds ok")


def check_inflate_radius():
    g = OccupancyGrid(np.zeros((21, 21), dtype=bool), 0.05, (0.0, 0.0))
    g.occupied[10, 10] = True
    fat = g.inflate(0.15)
    for di, dj in ((3, 0), (-3, 0), (0, 3), (0, -3)):
        assert fat.occupied[10 + di, 10 + dj], "include occupied cells exactly 0.15 m away"
    assert not fat.occupied[13, 11] and not fat.occupied[14, 10], "do not exceed the radius"
    np.testing.assert_array_equal(g.inflate(0.0).occupied, g.occupied)
    assert g.occupied.sum() == 1, "inflation leaves the source unchanged"
    print("inflate radius ok")


def check_inflate_border():
    g = OccupancyGrid(np.zeros((21, 21), dtype=bool), 0.05, (0.0, 0.0))
    expected = np.ones_like(g.occupied)
    expected[3:-3, 3:-3] = False
    np.testing.assert_array_equal(g.inflate(0.15).occupied, expected,
                                  err_msg="the third inward border cell is exactly 0.15 m away")
    print("inflate border ok")


def check_plan_path_invalid():
    """Nonfinite samples must not mask drive limits or endpoint errors."""
    from contextlib import redirect_stdout
    from dataclasses import replace
    from io import StringIO
    from unittest.mock import patch

    from scripts import plan_path as cli

    g = OccupancyGrid.from_room()
    lim = Limits()
    name, start, goal = next(r for r in cli.routes() if r[0] == "start-ball")
    path = plan(g, start, goal, lim)
    traj = profile(path, lim)
    masked = replace(traj, v=traj.v.copy())
    masked.v[1:3] = [1.5, np.nan]
    cases = [("masked speed", masked, "nonfinite")]
    for field, index in (("t", 1), ("v", 1), ("omega", 1),
                         ("xy", (-1, 0)), ("xy", (-1, 1)), ("yaw", 0), ("yaw", -1)):
        for value in (np.nan, np.inf, -np.inf):
            samples = getattr(traj, field).copy()
            samples[index] = value
            cases.append((f"{field}[{index}] = {value}", replace(traj, **{field: samples}), "nonfinite"))
    for time in (traj.t[0], traj.t[0] - 1):
        times = traj.t.copy()
        times[1] = time
        cases.append((f"time {time}", replace(traj, t=times), "time"))
    for field in ("v", "omega"):
        times = traj.t.copy()
        times[1] = 1e-310
        invalid = replace(traj, t=times, v=np.zeros_like(traj.v), omega=np.zeros_like(traj.omega))
        getattr(invalid, field)[1] = 0.1
        cases.append((f"derived {field} overflow", invalid, "nonfinite"))
    for label, invalid, expected in cases:
        out = StringIO()
        with patch.object(cli, "plan", return_value=path), \
             patch.object(cli, "profile", return_value=invalid), \
             patch.object(cli.OccupancyGrid, "from_room", return_value=g), \
             patch.object(sys, "argv", ["plan_path.py", "--route", name, "--quiet"]), \
             np.errstate(all="raise"), redirect_stdout(out):
            problems = cli.check_route(g, name, start, goal, lim, verbose=False)
            code = cli.main()
        assert problems and all(p.startswith(f"{name}:") for p in problems), (label, out.getvalue())
        assert any(expected in p for p in problems), (label, problems)
        assert code == 1 and "MISS" in out.getvalue(), (label, code, out.getvalue())
        assert "  ok" not in out.getvalue() and "routes planned within the limits" not in out.getvalue()
    print("plan_path invalid trajectories ok")


def check_plan_path_valid():
    """Keep stationary schedules, infinite clearance and finite alpha residuals valid."""
    from contextlib import redirect_stdout
    from io import StringIO
    from unittest.mock import patch

    from scripts import plan_path as cli

    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    lim = Limits()
    start = (0.5, 0.5, 0.0)
    out = StringIO()
    with redirect_stdout(out):
        assert cli.check_route(g, "clear", start, (1.5, 0.5, 0.0), lim, verbose=False) == []
    assert "clearance inf" in out.getvalue() and "  ok" in out.getvalue()
    g.occupied[-1, -1] = True
    out = StringIO()
    with redirect_stdout(out), np.errstate(all="raise"):
        assert cli.check_route(g, "stationary", start, start, lim, verbose=False) == []
    assert "clearance inf" in out.getvalue() and "peak alpha 0.00" in out.getvalue()

    theta = np.linspace(0, 0.0005, 3)
    xy = np.column_stack([1 + 0.2 * np.cos(theta), 1 + 0.2 * np.sin(theta)])
    arc = NavPath(start=(*xy[0], math.pi / 2), goal=(*xy[-1], math.pi / 2 + theta[-1]),
                  exit=0.0, turn=0.0, xy=xy, yaw=theta + math.pi / 2, kappa=np.full(3, 5.0),
                  s=0.2 * theta, dock=0.0, waypoints=xy[[0, -1]])
    traj = profile(arc, lim)
    alpha = np.abs(np.diff(traj.omega) / np.diff(traj.t)).max()
    assert alpha > lim.alpha_max
    out = StringIO()
    with patch.object(cli, "plan", return_value=arc), redirect_stdout(out):
        assert cli.check_route(g, "alpha-only", arc.start, arc.goal, lim, verbose=False) == []
    assert "peak alpha 0.50" in out.getvalue() and "  ok" in out.getvalue()
    print("plan_path valid trajectories ok")


def check_navigator():
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    lim = Limits(radius=0.1, margin=0.05)
    nav = Navigator(g, lim, off_path=0.10, pose_timeout=0.5)
    start, goal = (0.3, 0.3, 0.0), (1.7, 1.2, 0.0)
    traj = nav.go_to(goal, start, now=10.0)
    assert traj is nav.traj and not nav.done and nav.failed is None and nav.replans == []
    assert nav.step(start, 10.0) == (0.0, 0.0), "starts from rest"
    assert nav.step(None, 10.5) == traj.at(0.5)[2:]
    for elapsed in (1.5, 2.0, 2.1):
        xy, yaw, _, _ = traj.at(elapsed)
        np.testing.assert_allclose(nav.step((*xy, yaw), 10.0 + elapsed), traj.at(elapsed)[2:])
    assert nav.replans == [] and nav.t0 == 10.0 and nav.traj is traj
    pose = (xy[0] + 0.2, xy[1], yaw)
    assert nav.step(pose, 12.1) == (0.0, 0.0) and nav.replans == []
    settle(nav, pose, 12.1)
    assert len(nav.replans) == 1 and abs(nav.replans[0][1] - 0.2) < 1e-6
    np.testing.assert_allclose(nav.traj.xy[0], pose[:2], atol=1e-12)
    now = nav.t0 + 1.0
    assert nav.step(goal, now) == (0.0, 0.0) and not nav.done
    settle(nav, goal, now)
    assert nav.done and nav.step(None, now + 100) == (0.0, 0.0)
    blocked = OccupancyGrid(np.ones((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    stuck = Navigator(blocked, lim)
    assert stuck.go_to(goal, start, 0.0) is None and stuck.failed
    assert stuck.step(start, 0.0) == (0.0, 0.0)
    print("navigator ok")


def check_navigator_defaults():
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    first, second, explicit_none = Navigator(g), Navigator(g), Navigator(g, None)
    assert first.limits == second.limits == explicit_none.limits == Limits()
    first.limits.v_max = 0.01
    assert second.limits == explicit_none.limits == Limits(), "default limits must not be shared"
    explicit_none.limits.margin = 0.2
    assert second.limits == Navigator(g).limits == Limits()
    lim = Limits(v_max=0.02)
    nav = Navigator(g, lim)
    assert nav.grid is g and nav.limits is lim
    assert nav.off_path == 0.15 and nav.pose_timeout == 0.5
    assert nav.arrive_xy == 0.10 and nav.arrive_yaw == math.radians(5) and nav.max_replans == 60
    assert nav.goal is nav.traj is nav.failed is None and not nav.done and nav.replans == []
    for now, pose in ((0.0, None), (1.0, (0.5, 0.5, 0.0)), (100.0, None)):
        assert nav.step(pose, now) == (0.0, 0.0)
    assert nav.goal is nav.traj is nav.failed is None and not nav.done and nav.replans == []
    assert nav.t0 == 0.0
    print("navigator defaults and idle ok")


def check_navigator_timeout():
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    nav = Navigator(g)
    start, goal = (0.5, 0.5, 0.0), (1.5, 0.5, 0.0)
    traj = nav.go_to(goal, start, 8.0)
    assert nav.step(None, 8.5) == traj.at(0.5)[2:] and traj.at(0.5)[2] > 0
    assert nav.step(None, np.nextafter(8.5, math.inf)) == (0.0, 0.0)
    assert nav.traj is traj and nav.t0 == 8.0 and not nav.done and nav.failed is None
    pose = (0.55, 0.5, 0.0)
    assert nav.step(pose, 9.0) == (0.0, 0.0), "recovery must not resume the old clock"
    assert nav.step(None, 9.5) == (0.0, 0.0) and not nav._samples
    now = settle(nav, pose, 9.5)
    assert len(nav.replans) == 1 and nav.t0 == now
    np.testing.assert_allclose(nav.traj.xy[0], pose[:2], atol=1e-12)
    assert nav.step(None, now + 0.5) == nav.traj.at(0.5)[2:]
    assert nav.step(None, now + 0.500001) == (0.0, 0.0)
    nav.step(None, now + 40.0)
    assert "no pose for 30 s" in nav.failed and not nav.done and len(nav.replans) == 1

    nav = Navigator(g, pose_timeout=20.0)
    traj = nav.go_to(goal, start, 0.0)
    assert nav.step(None, 1.0) == traj.at(1.0)[2:] and nav.replans == []
    assert nav.step(None, traj.duration) == (0.0, 0.0) and not nav.done
    assert nav.step(goal, traj.duration) == (0.0, 0.0) and not nav.done
    settle(nav, goal, traj.duration)
    assert nav.done
    print("navigator timeout ok")


def check_navigator_replans():
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    nav = Navigator(g, off_path=0.125)
    start, goal = (0.5, 0.5, 0.0), (1.5, 0.5, 0.0)
    traj = nav.go_to(goal, start, 0.0)
    xy, yaw, _, _ = traj.at(0.5)
    assert nav.step((xy[0], xy[1] + 0.125, yaw + 0.4), 0.5) == traj.at(0.5)[2:]
    assert nav.replans == [] and nav.traj is traj, "no tracking correction at the drift boundary"
    xy, yaw, _, _ = traj.at(0.75)
    pose = (xy[0], xy[1] + 0.125001, yaw)
    assert nav.step(pose, 0.75) == (0.0, 0.0)
    assert nav.t0 == 0.0 and nav.replans == [] and nav.traj is traj
    now = settle(nav, pose, 0.75)
    np.testing.assert_allclose(nav.replans, [(now, 0.125001)], atol=1e-12)
    np.testing.assert_allclose(nav.traj.xy[0], pose[:2], atol=1e-12)
    assert abs(_wrap(nav.traj.yaw[0] - pose[2])) < 1e-12
    assert nav._hold_since is None and nav.traj is not traj and nav.t0 == now

    nav = Navigator(g, off_path=1.0)
    traj = nav.go_to(goal, start, 0.0)
    missed = (goal[0] - 0.125, goal[1], goal[2])
    assert nav.step(missed, traj.duration) == (0.0, 0.0)
    assert not nav.done and nav.failed is None and nav.replans == []
    now = settle(nav, missed, traj.duration)
    assert nav.replans == [(now, 0.125)] and nav.t0 == now
    np.testing.assert_allclose(nav.traj.xy[0], missed[:2], atol=1e-12)
    print("navigator replan boundaries ok")


def check_navigator_failures():
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    start, goal = (0.5, 0.5, 0.0), (1.5, 0.5, 0.0)
    nav = Navigator(g)
    traj = nav.go_to(goal, start, 0.0)
    missed = (1.3, 0.5, 0.0)
    g.occupied[:] = True
    assert nav.step(missed, traj.duration) == (0.0, 0.0)
    assert nav.traj is traj and nav.failed is None and not nav.replans
    now = settle(nav, missed, traj.duration)
    assert nav.traj is None and "does not fit" in nav.failed and not nav.done
    assert len(nav.replans) == 1 and nav.replans[0][0] == now and nav.t0 == 0.0
    failed, history = nav.failed, list(nav.replans)
    g.occupied[:] = False
    for pose, stamp in ((start, now + 1), (None, now + 2), (goal, now + 100)):
        assert nav.step(pose, stamp) == (0.0, 0.0)
    assert nav.failed == failed and nav.traj is None and nav.replans == history
    nav.go_to(goal, start, now + 101)
    assert nav.failed is None and not nav.done and nav.replans == []
    print("navigator collision failure and reuse ok")


def check_navigator_deadlines():
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    start, goal = (0.5, 0.5, 0.0), (1.5, 0.5, 0.0)
    nav = Navigator(g, pose_timeout=0.1)
    traj = nav.go_to(goal, start, 1.0)
    deadline = nav.t0 + nav.pose_timeout
    assert nav.step(None, deadline) == traj.at(deadline - nav.t0)[2:], "inclusive timeout boundary"
    assert nav.step(None, np.nextafter(deadline, math.inf)) == (0.0, 0.0)

    start, goal = (0.75, 0.75, 0.0), (1.5, 1.0, 0.0)
    for pose, arrived in ((goal, True), ((1.375, 1.0, 0.0), False)):
        nav = Navigator(g, off_path=1.0)
        traj = nav.go_to(goal, start, 20.0)
        deadline = nav.t0 + traj.duration
        before = np.nextafter(deadline, -math.inf)
        expected = (0.0, 0.0) if arrived else traj.at(before - nav.t0)[2:]
        assert nav.step(pose, before) == expected
        assert not nav.done and nav.replans == [] and nav.t0 == 20.0
        assert nav.step(pose, deadline) == (0.0, 0.0) and not nav.done
        now = settle(nav, pose, deadline)
        assert nav.done == arrived and nav.failed is None
        if arrived:
            assert nav.replans == [] and nav.t0 == 20.0
        else:
            assert nav.replans == [(now, 0.125)] and nav.t0 == now
    print("navigator exact deadlines ok")


def check_navigator_reuse():
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    nav = Navigator(g)
    start, goal = (0.5, 0.5, 0.0), (1.5, 0.5, 0.0)
    requested = list(goal)
    assert nav.go_to(requested, start, 10.0) is not None and nav.goal == goal
    requested[0] = 0.0
    assert nav.goal == goal and all(type(v) is float for v in nav.goal)
    assert nav.step(start, 10.5) == nav.traj.at(0.5)[2:]
    assert nav.t0 == 10.0 and not nav.replans
    nav.step((0.5, 0.7, 0.0), 10.6)
    assert nav._hold_since == 10.6
    settle(nav, (0.5, 0.7, 0.0), 10.6)
    assert len(nav.replans) == 1
    history = nav.replans
    pose, new_goal = (0.75, 0.75, 0.0), (1.5, 1.0, 0.0)
    traj = nav.go_to(np.array(new_goal), pose, 20.0)
    assert traj is nav.traj and nav.goal == new_goal and nav.t0 == 20.0
    assert nav.replans == [] and nav.replans is not history and len(history) == 1
    assert not nav.done and nav.failed is None
    np.testing.assert_allclose(traj.xy[[0, -1]], [pose[:2], new_goal[:2]], atol=1e-12)
    assert nav._hold_since is None and not nav._samples and nav._turn_goal is None
    assert nav.step(None, 20.5) == traj.at(0.5)[2:] and traj.at(0.5)[2] > 0
    assert nav.step(new_goal, 20.0 + traj.duration) == (0.0, 0.0) and not nav.done
    settle(nav, new_goal, 20.0 + traj.duration)
    assert nav.done
    assert nav.step(start, 90.0) == nav.step(None, 91.0) == (0.0, 0.0)
    assert nav.traj is traj and nav.replans == [] and nav.failed is None and nav.done
    assert nav.go_to(goal, start, 100.0) is nav.traj and nav.traj is not traj
    assert not nav.done and nav.failed is None and nav.t0 == 100.0 and nav.replans == []
    assert nav.step(start, 100.0) == (0.0, 0.0)
    g.occupied[:] = True
    assert nav.go_to(new_goal, pose, 101.0) is None
    assert nav.traj is None and nav.failed and not nav.done and nav.replans == []
    g.occupied[:] = False
    traj = nav.go_to(goal, start, 102.0)
    assert traj is nav.traj and nav.failed is None and not nav.done and nav.replans == []
    assert nav.t0 == 102.0 and nav.goal == goal
    assert nav.step(None, 102.5) == traj.at(0.5)[2:] and traj.at(0.5)[2] > 0
    print("navigator reuse ok")


def check_navigator_arrival():
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    start, goal = (0.5, 1.0, 0.0), (1.0, 1.0, 0.0)
    nav = Navigator(g, arrive_xy=0.125, arrive_yaw=0.125)
    traj = nav.go_to(goal, start, 0.0)
    boundary = (1.125, 1.0, 0.125)
    assert nav.step(boundary, 1.0) == (0.0, 0.0) and not nav.done
    assert nav.traj is traj and nav.replans == [], "arrival first commands zero, not a new route"
    settle(nav, boundary, 1.0)
    assert nav.done
    for pose in ((1.125001, 1.0, 0.0), (1.0, 1.0, 0.125001), (1.1, 1.1, 0.0)):
        nav = Navigator(g, arrive_xy=0.125, arrive_yaw=0.125, max_replans=0)
        traj = nav.go_to(goal, start, 0.0)
        assert nav.step(pose, traj.duration) == (0.0, 0.0) and not nav.done
        settle(nav, pose, traj.duration)
        assert not nav.done and nav.failed and nav.replans == [], "both arrival tolerances must hold"
    for goal_yaw, pose_yaw in ((179, -179), (-179, 179), (0, 360)):
        goal = (1.0, 1.0, math.radians(goal_yaw))
        nav = Navigator(g)
        traj = nav.go_to(goal, goal, 0.0)
        assert traj.duration == 0 and not nav.done
        pose = (1.01, 1.0, math.radians(pose_yaw))
        assert nav.step(pose, 0.0) == (0.0, 0.0) and not nav.done
        settle(nav, pose, 0.0)
        assert nav.done
    nav = Navigator(g)
    assert nav.go_to((1.0, 1.0, math.radians(179)), (1.0, 1.0, math.radians(179)), 0.0) is not None
    assert not nav._arrived((1.0, 1.0, math.radians(-174)))
    assert not nav._arrived((1.100001, 1.0, math.radians(179)))
    print("navigator arrival ok")


def check_navigator_spatial_roundoff():
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    start, goal = (0.5, 1.0, 0.0), (1.0, 1.0, 0.0)
    for extra, inside in ((0.0, True), (5e-10, True), (1e-8, False)):
        for dx, dy in ((0.10 + extra, 0.0), (0.0, 0.10 + extra)):
            nav = Navigator(g)
            traj = nav.go_to(goal, start, 0.0)
            pose = (goal[0] + dx, goal[1] + dy, 0.0)
            assert nav.step(pose, traj.duration) == (0.0, 0.0) and not nav.done
            settle(nav, pose, traj.duration)
            assert nav.done == inside and len(nav.replans) == int(not inside), "10 cm arrival boundary"
            if inside:
                assert nav.failed is None and nav.traj is traj

        for sign in (-1, 1):
            wrapped_goal = (1.0, 1.0, sign * math.radians(179))
            pose = (1.0, 1.0, sign * (math.radians(-176) + extra))
            nav = Navigator(g)
            traj = nav.go_to(wrapped_goal, wrapped_goal, 0.0)
            assert nav.step(pose, traj.duration) == (0.0, 0.0) and not nav.done
            settle(nav, pose, traj.duration)
            assert nav.done == inside and len(nav.replans) == int(not inside), "wrapped 5 degree boundary"
            assert nav.failed is None
            if inside:
                assert nav.traj is traj

        nav = Navigator(g, off_path=0.10)
        traj = nav.go_to(goal, start, 0.0)
        xy, yaw, _, _ = traj.at(0.5)
        command = nav.step((xy[0], xy[1] + 0.10 + extra, yaw), 0.5)
        assert nav.replans == [] and nav.failed is None and not nav.done
        if inside:
            assert command == traj.at(0.5)[2:] and nav.traj is traj and nav.t0 == 0.0
            assert nav._hold_since is None
        else:
            assert command == (0.0, 0.0) and nav.traj is traj and nav._hold_since == 0.5
            now = settle(nav, (xy[0], xy[1] + 0.10 + extra, yaw), 0.5)
            assert len(nav.replans) == 1 and nav.traj is not traj and nav.t0 == now
            assert abs(nav.replans[0][1] - (0.10 + extra)) < 1e-12
    print("navigator spatial roundoff ok")


def settle(nav, pose, now):
    count = len(nav.replans)
    for tick in range(1, 251):
        stamp = now + tick * 0.02
        command = nav.step(pose, stamp)
        if nav.done or nav.failed or len(nav.replans) != count:
            return stamp
        assert command == (0.0, 0.0), "settling never issues a moving command"
    raise AssertionError("settling must finish or fail within five seconds")


def check_terminal_contract():
    from rlbot.planner import terminal_plan, turn_free

    g, lim = OccupancyGrid.from_room(), Limits()
    goal = goal_for("cubes")
    pose = (-0.206684, -1.368994, math.radians(-123.383))
    for dx in (-0.001, 0.0, 0.001):
        start = (pose[0] + dx, pose[1], pose[2])
        path = terminal_plan(g, start, goal, lim, 0.10)
        assert path.exit == path.dock == path.length == 0
        assert abs(math.degrees(path.turn) - 33.383) < 1e-9
        assert path.goal[:2] == start[:2] and path.goal[2] == goal[2]
        assert turn_free(g, start, goal[2], lim)
    assert terminal_plan(g, (0.0, 0.0, 0.0), goal, lim, 0.10) is None
    for start in ((-0.2, -1.374, math.pi / 2), (-0.2, -1.41, math.pi / 2)):
        assert g.rect_free(*start, lim.half_depth + lim.margin, lim.half_width + lim.margin)
        assert not turn_free(g, start, goal[2], lim), "end rectangles cannot certify a rotation"
        path = terminal_plan(g, start, goal, lim, 0.10)
        assert path.exit != 0 and path.dock == path.length == 0
        end = path.goal
        assert math.dist(end[:2], goal[:2]) <= 0.10
        assert g.rect_free((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, start[2],
                           lim.half_depth + lim.margin + abs(path.exit) / 2,
                           lim.half_width + lim.margin)
        assert turn_free(g, (*end[:2], start[2]), end[2], lim)
    with np.testing.assert_raises(NoPath):
        terminal_plan(g, (-0.2, -1.41, math.pi / 2), goal, lim, 0.005)
    blocked = OccupancyGrid(np.ones((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    with np.testing.assert_raises(NoPath):
        terminal_plan(blocked, (1.0, 1.0, 0.0), (1.0, 1.0, 0.1), lim, 0.10)
    print("terminal planning: short wrapped turns, jitter, guarded escape and blocked finish ok")


def check_terminal_drift():
    from rlbot.planner import terminal_plan, turn_free

    grid, lim = OccupancyGrid.from_room(), Limits()
    goal = goal_for("cubes")
    for pose in ((-0.14002, -1.33723, math.radians(-73.97)),
                 (-0.1418, -1.30913, math.radians(-88.54))):
        path = terminal_plan(grid, pose, goal, lim, 0.10)
        assert path is not None and path.exit > 0 and abs(path.turn) < math.radians(20)
        assert math.dist(path.goal[:2], goal[:2]) < 0.08
        assert turn_free(grid, (*path.goal[:2], pose[2]), goal[2], lim)
        assert grid.rect_free((pose[0] + path.goal[0]) / 2, (pose[1] + path.goal[1]) / 2, pose[2],
                              lim.half_depth + lim.margin + path.exit / 2, lim.half_width + lim.margin)
    pose = (-1.61564, 0.98995, math.radians(152.42))
    goal = goal_for("ware")
    path = terminal_plan(grid, pose, goal, lim, 0.10)
    assert path is not None and path.exit == 0 and 0 < path.turn < math.pi / 2
    assert abs(_wrap(path.goal[2] - math.atan2(goal[1] - pose[1], goal[0] - pose[0]))) < 1e-9
    print("terminal drift: guarded inward reserve and direct capture heading avoid anchor detours ok")


def check_phase_metadata():
    from scripts.plan_path import routes

    grid = OccupancyGrid.from_room()
    for _, start, goal in routes():
        path = plan(grid, start, goal)
        traj = profile(path)
        expected = [kind for kind, present in (("exit", path.exit), ("turn", path.turn),
                                               ("curve", len(path.xy)), ("dock", path.dock)) if present]
        assert [kind for kind, _ in traj.phases] == expected
        ends = [index for _, index in traj.phases]
        assert ends[-1] == len(traj.t) - 1 and all(a < b for a, b in zip(ends, ends[1:]))
        assert np.all(traj.v[ends] == 0) and np.all(traj.omega[ends] == 0)
    print("trajectory metadata ok")


def check_phase_contract():
    g = OccupancyGrid.from_room()
    nav = Navigator(g)
    traj = nav.go_to(goal_for("cubes"), (0.0, 0.0, 0.0), 0.0)
    xy, yaw, _, _ = traj.at(2.0)
    assert nav.step((*xy, yaw), 2.0) == traj.at(2.0)[2:]
    assert nav.traj is traj and nav.t0 == 0.0 and nav.replans == [], "do not restart a running phase"
    kind, end = traj.phases[0]
    assert kind == "turn" and traj.v[end] == traj.omega[end] == 0
    target = float(traj.yaw[end])
    undershot = (*traj.xy[end], target + math.radians(20))
    stamp = float(traj.t[end]) + 0.2
    assert nav.step(undershot, stamp) == (0.0, 0.0), "overshoot cannot enter the curve"
    assert not nav.done and nav.replans == []
    stamp = settle(nav, undershot, stamp)
    assert nav.failed is None and len(nav.replans) == 1
    assert nav.traj.phases[0][0] == "turn" and np.all(nav.traj.v == 0)
    assert abs(_wrap(nav.traj.yaw[-1] - target)) < 1e-9, "correct even below TURN_FIRST"
    assert nav.step(undershot, stamp + 0.2)[0] == 0
    fresh = nav.traj
    end = fresh.phases[0][1]
    reached = (*fresh.xy[end], float(fresh.yaw[end]))
    stamp = nav.t0 + fresh.t[end]
    assert nav.step(reached, stamp) == (0.0, 0.0)
    settle(nav, reached, stamp)
    assert nav.failed is None and len(nav.replans) == 2
    assert nav.traj.phases[0][0] == "curve"
    print("phase contract: uninterrupted phases, overshoot stop and actual heading correction ok")


def check_settle_contract():
    g = OccupancyGrid(np.zeros((81, 81), dtype=bool), 0.05, (-2.0, -2.0))
    goal = (0.5, 0.0, 0.0)
    nav = Navigator(g)
    traj = nav.go_to(goal, (-1.0, 0.0, 0.0), 0.0)
    assert nav.step(goal, 1.0) == (0.0, 0.0) and not nav.done
    for tick in range(1, 76):
        moving = (0.5 + 0.025 * math.sin(tick * 0.1), 0.0, 0.0)
        assert nav.step(moving, 1.0 + tick * 0.02) == (0.0, 0.0)
        assert not nav.done, "crossing goal tolerance is not a stopped arrival"
    stamp = settle(nav, goal, 2.5)
    assert nav.done and stamp < traj.duration and not nav.replans
    assert nav.step(None, stamp + 100) == (0.0, 0.0)
    nav.go_to(goal, goal, 10.0)
    assert not nav.done and nav.step(goal, 10.0) == (0.0, 0.0)
    for stamp in np.arange(10.1, 11.5, 0.1):
        nav.step(goal, float(stamp))
    nav.step(None, 11.5)
    nav.step(goal, 11.6)
    assert not nav.done
    nav.step(goal, 11.9)
    assert not nav.done, "missing pose resets the stable window"
    nav.step(goal, 12.11)
    assert nav.done
    nav.go_to(goal, goal, 20.0)
    nav.step(goal, 20.0)
    for tick in range(1, 160):
        moving = (goal[0] + 0.025 * math.sin(tick * 0.5), goal[1], goal[2])
        assert nav.step(moving, 20.0 + tick * 0.05) == (0.0, 0.0)
        assert not nav.done and nav.failed is None
    assert nav.step(goal, 28.0) == (0.0, 0.0), "the settle deadline is 8 s"
    assert nav.done and not nav.replans, "unstable but arrived still finishes at the deadline"
    print("settling: moving crossings rejected, old tail ignored, fresh stable evidence required ok")


def check_settle_deadline():
    from rlbot.navigate import _LOST_MAX

    g = OccupancyGrid(np.zeros((81, 81), dtype=bool), 0.05, (-2.0, -2.0))
    start, goal, off = (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.5, 0.3, 0.0)
    for quota, replans in ((60, 1), (0, 0)):
        nav = Navigator(g, max_replans=quota)
        assert nav._settle_max == 8.0
        nav.go_to(goal, start, 0.0)
        assert nav.step(off, 1.0) == (0.0, 0.0) and nav._hold_since == 1.0
        for tick in range(1, 160):                  # jitter far too wide to ever settle
            jittered = (off[0] + 0.02 * math.sin(tick * 0.5), off[1], off[2])
            assert nav.step(jittered, 1.0 + tick * 0.05) == (0.0, 0.0)
            assert nav.failed is None and not nav.done and not nav.replans
        assert nav.step(off, 9.0) == (0.0, 0.0)
        assert len(nav.replans) == replans and not nav.done
        if quota:
            assert nav.failed is None and nav._hold_since is None and nav.t0 == 9.0
            np.testing.assert_allclose(nav.traj.xy[0], off[:2], atol=1e-12)
        else:
            assert "gave up after 0 replans" in nav.failed, nav.failed

    nav = Navigator(g)
    nav.go_to(goal, start, 0.0)
    assert nav.step(off, 1.0) == (0.0, 0.0)         # holding, then the pose stops arriving
    for stamp in (9.0, 20.0, 1.0 + _LOST_MAX - 0.1):
        assert nav.step(None, stamp) == (0.0, 0.0) and nav.failed is None
    assert nav.step(None, 1.0 + _LOST_MAX) == (0.0, 0.0)
    assert nav.failed == "no pose for 30 s while holding zero" and not nav.done
    print("settle deadline: unstable poses re-arm and replan, only a lost pose latches ok")


def check_recovery_contract():
    g = OccupancyGrid(np.zeros((81, 81), dtype=bool), 0.05, (-2.0, -2.0))
    start, goal = (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    nav = Navigator(g)
    old = nav.go_to(goal, start, 0.0)
    assert nav.step(None, 0.5) == old.at(0.5)[2:]
    assert nav.step(None, 0.500001) == (0.0, 0.0)
    recovered = (-0.8, 0.0, 0.0)
    assert nav.step(recovered, 0.6) == (0.0, 0.0) and not nav.replans
    settle(nav, recovered, 0.6)
    assert len(nav.replans) == 1 and nav.traj is not old and nav.failed is None
    np.testing.assert_allclose(nav.traj.xy[0], recovered[:2], atol=1e-12)
    for quota in (0, 1):
        nav = Navigator(g, max_replans=quota)
        nav.go_to(goal, start, 0.0)
        for attempt in range(quota + 1):
            now = nav.t0 + 0.5
            xy, yaw, _, _ = nav.traj.at(0.5)
            pose = (xy[0], xy[1] + 0.2, yaw)
            assert nav.step(pose, now) == (0.0, 0.0)
            count = len(nav.replans)
            assert count == attempt
            settle(nav, pose, now)
        assert nav.failed and len(nav.replans) == quota and not nav.done
        failed = nav.failed
        assert nav.step(goal, 100.0) == nav.step(None, 101.0) == (0.0, 0.0)
        assert nav.failed == failed
        nav.go_to(goal, start, 102.0)
        assert nav.failed is None and not nav.replans and not nav.done
        assert nav.step(start, 102.0) == (0.0, 0.0)
    print("recovery: stale clock discarded after settling, actual plans counted, quota latches ok")


def check_navigator_reset_safety():
    from unittest.mock import patch

    class BrokenNumber:
        def __float__(self):
            raise RuntimeError("conversion bug")

    grid = OccupancyGrid(np.zeros((81, 81), dtype=bool), 0.05, (-2.0, -2.0))
    start, goal, replacement = (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    invalid = [(replacement, None, 0.5), (None, start, 0.5), ((0.0, 1.0), start, 0.5),
               ((0.0, 1.0, 0.0, 0.0), start, 0.5), ((math.nan, 1.0, 0.0), start, 0.5),
               (("invalid", 1.0, 0.0), start, 0.5), (replacement, (0.0, 1.0), 0.5),
               (replacement, (0.0, 1.0, 0.0, 0.0), 0.5), (replacement, (math.inf, 0.0, 0.0), 0.5),
               (replacement, start, math.nan), (replacement, start, -0.5),
               (replacement, start, "invalid")]
    for in_flight in (False, True):
        for request in invalid:
            nav = Navigator(grid)
            old = nav.go_to(goal, start, 0.0) if in_flight else None
            pose = (*old.at(0.5)[0], old.at(0.5)[1]) if old else start
            if old:
                assert nav.step(pose, 0.5)[0] > 0
            assert nav.go_to(*request) is None
            assert nav.traj is None and nav.goal is None and nav.failed and not nav.done
            if request[1] is None:
                assert nav.failed == "a fresh pose is required to plan"
            failure = nav.failed
            assert nav.step(pose, 0.5) == nav.step(None, 1.0) == nav.step(start, 10.0) == (0.0, 0.0)
            assert nav.failed == failure and not nav.replans
            fresh = nav.go_to(replacement, start, 11.0)
            assert fresh is nav.traj and fresh is not old and fresh is not None
            assert nav.failed is None and nav.goal == replacement and nav.replans == []
            xy, yaw, _, _ = fresh.at(0.2)
            np.testing.assert_allclose(nav.step((*xy, yaw), 11.2), fresh.at(0.2)[2:], atol=1e-12)

        nav = Navigator(grid)
        if in_flight:
            nav.go_to(goal, start, 0.0)
        with np.testing.assert_raises_regex(RuntimeError, "conversion bug"):
            nav.go_to((BrokenNumber(), 1.0, 0.0), start, 0.5)
        assert nav.traj is None and nav.goal is None and nav.failed
        assert nav.step(start, 1.0) == (0.0, 0.0)
        assert nav.go_to(goal, start, 2.0) is not None and nav.failed is None

        for function in ("plan", "profile"):
            nav = Navigator(grid)
            old = nav.go_to(goal, start, 0.0) if in_flight else None
            with patch(f"rlbot.navigate.{function}", side_effect=RuntimeError("planner bug")):
                with np.testing.assert_raises_regex(RuntimeError, "planner bug"):
                    nav.go_to(replacement, start, 0.5)
            assert nav.traj is None and nav.failed and not nav.done
            assert nav.step(start, 1.0) == nav.step(None, 2.0) == (0.0, 0.0)
            assert nav.go_to(goal, start, 3.0) is not None and nav.failed is None
    print("navigator reset safety: invalid requests latch zero, planner bugs propagate safely, fresh reset recovers ok")


def check_navigator_runtime_inputs():
    class BrokenNumber:
        def __float__(self):
            raise RuntimeError("conversion bug")

    grid = OccupancyGrid(np.zeros((81, 81), dtype=bool), 0.05, (-2.0, -2.0))
    start, goal = (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    invalid = [(pose, 0.4, False) for pose in
               ((math.nan,) * 3, (0.0, 0.0, math.inf), (math.inf, 0.0, 0.0),
                (0.0, math.nan, 0.0), (), (0.0, 0.0), (0.0,) * 4, 7, ("invalid", 0.0, 0.0))]
    invalid += [(None, stamp, False) for stamp in (math.nan, math.inf, -math.inf, -1, 0.1, None, "invalid")]
    invalid += [((BrokenNumber(), 0.0, 0.0), 0.4, True), (None, BrokenNumber(), True)]
    for holding in (False, True):
        for pose, stamp, unexpected in invalid:
            nav = Navigator(grid)
            old = nav.go_to(goal, start, 0.0)
            xy, yaw, _, _ = old.at(0.2)
            assert nav.step((*xy, yaw), 0.2)[0] > 0
            if holding:
                assert nav.step(goal, 0.3) == (0.0, 0.0) and nav._hold_since == 0.3
            last_pose = nav._last_pose
            if unexpected:
                with np.testing.assert_raises_regex(RuntimeError, "conversion bug"):
                    nav.step(pose, stamp)
            else:
                assert nav.step(pose, stamp) == (0.0, 0.0)
            assert nav.traj is None and nav.failed and not nav.done
            assert nav._last_pose == last_pose, "invalid input cannot refresh pose freshness"
            failure = nav.failed
            assert nav.step(None, 0.5) == nav.step(start, 1.0) == (0.0, 0.0)
            assert nav.failed == failure and not nav.replans
            fresh = nav.go_to(goal, start, 0.0)
            assert fresh is not None and fresh is not old and nav.failed is None
            xy, yaw, _, _ = fresh.at(0.2)
            assert nav.step((*xy, yaw), 0.2) == fresh.at(0.2)[2:]
            assert nav.step((*xy, yaw), 0.2) == fresh.at(0.2)[2:], "equal timestamps remain valid"
            assert nav.step(None, 0.4) == fresh.at(0.4)[2:], "None retains its intentional grace period"
    nav = Navigator(grid)
    nav.go_to(goal, start, 0.0)
    for stamp in (0.2, 0.6, 1.0, 2.0):
        assert nav.step((math.nan,) * 3, stamp) == (0.0, 0.0)
    assert nav._last_pose == 0.0 and nav.failed
    fresh = nav.go_to(goal, start, 0.0)
    assert nav.step(None, 0.4) == fresh.at(0.4)[2:]
    assert nav.step(start, 0.3) == (0.0, 0.0) and "backwards" in nav.failed
    assert nav._last_pose == 0.0, "clock ordering includes missing-pose updates"
    nav.go_to(goal, start, 10.0)
    nav.step(None, 10.2)
    fresh = nav.go_to(goal, start, 0.0)
    assert nav.step(None, 0.2) == fresh.at(0.2)[2:] and nav.failed is None
    print("navigator runtime inputs: invalid poses/clocks latch zero, bugs propagate safely, fresh clock resets ok")


def check_true_pose():
    from types import SimpleNamespace

    left, right = np.array([1.0, 2.0, 3.0]), np.array([3.0, 6.0, 5.0])
    for yaw in (0.0, math.pi / 2, math.radians(-179)):
        c, s = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(0.3), math.sin(0.3)
        rot = np.array([[c * cp, -s, c * sp], [s * cp, c, s * sp], [-sp, 0.0, cp]])
        bodies = {"wheel_left": SimpleNamespace(xpos=left), "wheel_right": SimpleNamespace(xpos=right),
                  "root": SimpleNamespace(xpos=np.array([9.0, 10.0, 11.0]), xmat=rot.ravel().copy())}
        data = SimpleNamespace(body=bodies.__getitem__)
        pose = true_pose(data)
        assert type(pose) is tuple and all(type(v) is float for v in pose)
        np.testing.assert_allclose(pose, (2.0, 4.0, yaw), atol=1e-12)
        np.testing.assert_array_equal(left, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(right, [3.0, 6.0, 5.0])
        np.testing.assert_array_equal(bodies["root"].xmat, rot.ravel())
    print("true_pose axle convention ok")


def check_navigation_exports():
    import rlbot
    from rlbot import arm, control, robot

    expected = {name: getattr(robot, name) for name in ("Balancer", "State", "TOY", "BRACKETBOT", "ROOM")}
    expected.update({name: getattr(control, name) for name in ("BalanceController", "Gains")})
    expected.update({name: getattr(arm, name)
                     for name in ("Arm", "ArmIK", "Gripper", "Solution", "down_quat", "OPEN", "SHUT")})
    expected.update(OccupancyGrid=OccupancyGrid, Limits=Limits, NoPath=NoPath, Path=NavPath,
                    Trajectory=Trajectory, goal_for=goal_for, plan=plan, profile=profile,
                    Navigator=Navigator, true_pose=true_pose)
    assert set(rlbot.__all__) == expected.keys() and len(rlbot.__all__) == len(expected)
    for name, value in expected.items():
        assert getattr(rlbot, name) is value, name
    print("navigation exports ok")


if __name__ == "__main__":
    check_grid()
    check_pgm()
    check_pgm_formats()
    check_pgm_metadata()
    check_pgm_thresholds()
    check_pgm_invalid()
    check_line_free()
    check_rect_bounds()
    check_inflate_radius()
    check_inflate_border()
    check_astar()
    check_astar_edges()
    check_shortcut()
    check_spline()
    check_dock()
    check_straight_out()
    check_dock_stationary_margin()
    check_profile()
    check_profile_caps()
    check_profile_trims()
    check_profile_phases()
    check_profile_edges()
    check_plan_edges()
    check_plan_short_distance()
    check_spline_regularity()
    check_plan_short_curve()
    check_plan_endpoints()
    check_fit_retry()
    check_fit_corner_contact()
    check_fit_resampled_corner()
    check_goal_for()
    check_plan_path_invalid()
    check_plan_path_valid()
    check_navigator()
    check_navigator_defaults()
    check_navigator_timeout()
    check_navigator_replans()
    check_navigator_failures()
    check_navigator_deadlines()
    check_navigator_reuse()
    check_navigator_arrival()
    check_navigator_spatial_roundoff()
    check_terminal_contract()
    check_terminal_drift()
    check_phase_metadata()
    check_phase_contract()
    check_settle_contract()
    check_settle_deadline()
    check_recovery_contract()
    check_navigator_reset_safety()
    check_navigator_runtime_inputs()
    check_true_pose()
    check_navigation_exports()
    print("all ok")
