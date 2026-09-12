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
sys.path.insert(0, str(REPO / "src"))

from rlbot.navmap import OccupancyGrid          # noqa: E402


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

    assert g.rect_free(0.2, 0.2, 0.0, 0.05, 0.05)
    assert not g.rect_free(0.35, 0.5, 0.0, 0.10, 0.05), "a rectangle reaching the post"
    assert g.rect_free(0.25, 0.5, math.pi / 2, 0.10, 0.05), "the same rectangle turned side-on"

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


if __name__ == "__main__":
    check_grid()
    print("all ok")
