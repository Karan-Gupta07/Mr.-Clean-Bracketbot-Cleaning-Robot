# Path Planning and Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive the BracketBot from anywhere in the room to a table's docking pose on a curved, speed-limited schedule, replanning from the localised pose, with the same code running against the sim's true pose locally and the SLAM pose in Docker.

**Architecture:** An occupancy grid (`navmap.py`) from the known room or a saved SLAM map; a planner (`planner.py`) that runs A*, shortcuts, fits a clamped cubic spline, and profiles it into a `(speed, yaw_rate)(t)` schedule under the `DriveController`'s limits; a navigator (`navigate.py`) that plays the schedule open loop and replans when the pose drifts; one thin rclpy node that feeds it TF and publishes `/cmd_vel`. Docks sit inside the robot's inflation radius, so every route is: straight out of the start (in reverse if need be), a turn in place only if the corridor leaves more than 45° off the heading, the curve, and a straight run along the dock axis into the goal.

**Tech Stack:** Python 3.12, numpy, mujoco 3.13 (already in `.venv`). No scipy, no pytest: checks are assert-based scripts like the rest of the repo. ROS 2 Jazzy only inside `ros2_ws/` and only in Docker.

**Spec:** `docs/superpowers/specs/2026-09-12-path-planning-navigation-design.md`. Task 0 amends it for the dock segments; both docs are deleted in the last task, per the user.

**Branch:** `path-planning-controls`, based on `feat/slam-cv-camera-inspection`. Commit after every task. Commit messages end with the line `Claude-Session: https://claude.ai/code/session_01CQtc6f3qAqzNU3TbMU9DkB`.

---

## File structure

| File | Responsibility |
| --- | --- |
| `src/rlbot/navmap.py` (new) | `OccupancyGrid`: from the room or a PGM, inflate, point/line/rectangle queries, ASCII. `robot_footprint`, `table_tops`. |
| `src/rlbot/planner.py` (new) | `Limits`, `Path`, `Trajectory`, `NoPath`. `astar`, `shortcut`, the spline, `plan`, `profile`, `goal_for`. Pure geometry. |
| `src/rlbot/navigate.py` (new) | `Navigator`: play a schedule, replan on drift or timer, timeout, done/failed. `true_pose` for the sim stub. |
| `scripts/check_navigation.py` (new) | Unit-level assert checks for the three modules on synthetic grids. Runs in seconds. |
| `scripts/plan_path.py` (new) | The 9 room routes: ASCII maps, metrics, asserts. Geometry only. |
| `scripts/navigate.py` (new) | The 9 routes driven in the sim through `DriveController`. Arrival, falls, contacts, tilt. `--view`. |
| `scripts/build_room.py` (modify) | Import the grid from `navmap` instead of carrying its own copy. |
| `ros2_ws/src/rlbot_bridge/rlbot_bridge/navigate.py` (new) | rclpy node: TF in, `/cmd_vel` out. |
| `ros2_ws/src/rlbot_bridge/setup.py` (modify) | `navigate` console script. |
| `README.md` (modify) | Status row, what works, how to run, folder layout, next steps, Docker paragraph. |

Every module starts with a docstring that says what it is for and what it is not, in the voice of the existing ones. Comments explain why, not what.

---

### Task 0: Amend the spec for the dock segments

The spec says the curve arrives "with no turn in place at either end". That cannot hold: `Table.dock` is 0.24 m from the table and the inflation radius is 0.26 m, so the dock cell is occupied on the inflated grid and A* would refuse it. The fix is straight segments along the dock axis, checked with the robot's rectangle rather than its circle, and a turn in place when leaving a dock.

**Files:**
- Modify: `docs/superpowers/specs/2026-09-12-path-planning-navigation-design.md`

- [ ] **Step 1: Append the amendment**

Append to the end of the spec file:

```markdown

## Amendments during planning

1. **Docks are inside the inflation.** `Table.dock` puts the mast axis 0.24 m
   from the table edge; the planner inflates by 0.26 m. So a route has up to
   four segments: a straight run out of the start along its heading (in
   reverse when the free space is behind), a turn in place if the corridor
   leaves more than 45° off the heading, the curve, and a straight run along
   the goal's axis into the goal. The straight segments are checked against
   the robot's oriented rectangle on the raw grid, the curve against its
   circle on the inflated grid. The robot stops between segments; it does
   not stop anywhere along the curve.
2. **Angular acceleration is reported, not asserted.** `omega = v * kappa`,
   so at the start of a bend `d omega / dt = a * kappa`, which exceeds
   0.3 rad/s² whenever the bend is tighter than 3 m at full acceleration.
   The profile scales the speed cap down at those samples, up to ten times;
   whatever residual is left, the `DriveController` ramp absorbs, and the
   next replan corrects. `plan_path.py` prints the peak.
3. **No `map_offset` parameter on the ROS node.** The bridge always spawns
   at the `start` keyframe, so the map frame is the world frame. The
   parameter can be added the day someone maps from somewhere else.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-09-12-path-planning-navigation-design.md
git commit -m "Amend the navigation spec for dock approach segments

Claude-Session: https://claude.ai/code/session_01CQtc6f3qAqzNU3TbMU9DkB"
```

---

### Task 1: The occupancy grid from the known room

**Files:**
- Create: `src/rlbot/navmap.py`
- Create: `scripts/check_navigation.py`

- [ ] **Step 1: Write the failing check**

Create `scripts/check_navigation.py`:

```python
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
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/bin/python scripts/check_navigation.py`
Expected: `ModuleNotFoundError: No module named 'rlbot.navmap'`

- [ ] **Step 3: Write navmap.py**

Create `src/rlbot/navmap.py`:

```python
"""Where the robot can and cannot stand: an occupancy grid of the room.

Two ways to get one.  `OccupancyGrid.from_room()` rasterises the known room in
`models/room.xml`, and is what the room builder's clearance check and the local
navigation check use.  `OccupancyGrid.from_pgm()` reads the map SLAM Toolbox
saves, which is what the robot navigates on once it has mapped the room itself.

Either way a cell is occupied if geometry covers its centre.  That is the raw
grid; `inflate()` grows it by the robot's radius so a planner can treat the
robot as a point, and `rect_free()` tests the robot's actual rectangle against
the raw grid for the few moves - docking, backing out - where the circle is too
pessimistic to allow them at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .room import ROOM_X, ROOM_Y, TABLE_L, TABLE_W, TABLES

MODELS = Path(__file__).resolve().parents[2] / "models"
ROBOT_MODEL = MODELS / "bracketbot.xml"
ROOM_MODEL = MODELS / "room.xml"

GRID = 0.05      # m, default cell size; SLAM Toolbox is configured to the same
MARGIN = 0.05    # m of daylight the robot should have on top of its own radius


def robot_footprint():
    """The radius and height of the robot, read off its collision boxes.

    Not a number typed in here: `build_mjcf.py` measures three boxes over the
    chassis off the meshes, and this is what they come to - a 0.19 x 0.37 m
    footprint, 1.70 m tall.  Returns (radius, height, half-extents).
    """
    model = mujoco.MjModel.from_xml_path(str(ROBOT_MODEL))
    half = np.zeros(2)
    height = 0.0
    for name in ("chassis_drive", "chassis_mast", "chassis_head"):
        geom = model.geom(name)
        half = np.maximum(half, geom.size[:2])
        height = max(height, float(geom.pos[2] + geom.size[2]))
    return float(math.hypot(*half)), height, half


def table_tops():
    """The three table tops as (x, y, yaw, half_x, half_y), for from_pgm.

    A lidar at 0.32 m sees the legs and not the top, so a map built from it
    has open floor under each table that a 1.7 m robot cannot use.
    """
    return [(t.centre[0], t.centre[1], t.yaw, TABLE_L / 2, TABLE_W / 2)
            for t in TABLES]


@dataclass
class OccupancyGrid:
    occupied: np.ndarray          # bool, (nx, ny); True where the robot cannot be
    resolution: float             # m per cell
    origin: tuple[float, float]   # world (x, y) of the centre of cell (0, 0)

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_room(cls, resolution: float = GRID, height: float | None = None):
        """Rasterise models/room.xml.

        Anything reaching into the robot's height band blocks, which is why
        table tops block even though the robot could drive between the legs.
        Loose objects on the tables are not architecture and are skipped.
        """
        if height is None:
            height = robot_footprint()[1]
        model = mujoco.MjModel.from_xml_path(str(ROOM_MODEL))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        xs = np.arange(-ROOM_X / 2, ROOM_X / 2 + resolution / 2, resolution)
        ys = np.arange(-ROOM_Y / 2, ROOM_Y / 2 + resolution / 2, resolution)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        occupied = np.zeros(gx.shape, dtype=bool)

        for g in range(model.ngeom):
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            if model.geom_bodyid[g] and model.body_jntnum[model.geom_bodyid[g]]:
                continue
            pos, size = data.geom_xpos[g], model.geom_size[g]
            rot = data.geom_xmat[g].reshape(3, 3)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                reach = float(size[1])
                inside = np.hypot(gx - pos[0], gy - pos[1]) <= size[0] + 1e-9
            else:                                   # box, yaw in the plane
                reach = float(np.abs(rot[2]) @ size[:3])
                dx, dy = gx - pos[0], gy - pos[1]
                local_x = rot[0, 0] * dx + rot[1, 0] * dy
                local_y = rot[0, 1] * dx + rot[1, 1] * dy
                inside = ((np.abs(local_x) <= size[0] + 1e-9)
                          & (np.abs(local_y) <= size[1] + 1e-9))
            if pos[2] - reach > height or pos[2] + reach < 0:
                continue
            occupied |= inside
        return cls(occupied, resolution, (float(xs[0]), float(ys[0])))

    def inflate(self, radius: float) -> "OccupancyGrid":
        """Grow every occupied cell by `radius`.

        Beyond the edge of the grid counts as occupied, so the border inflates
        inward too: a SLAM map ends where the lidar stopped seeing, not where
        the floor does.
        """
        r = int(math.ceil(radius / self.resolution))
        offsets = [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
                   if math.hypot(di, dj) * self.resolution <= radius]
        nx, ny = self.occupied.shape
        src = np.pad(self.occupied, r, constant_values=True)
        out = np.zeros_like(self.occupied)
        for di, dj in offsets:
            out |= src[r + di : r + di + nx, r + dj : r + dj + ny]
        return OccupancyGrid(out, self.resolution, self.origin)

    def fill_box(self, x, y, yaw, half_x, half_y) -> None:
        """Mark an oriented rectangle occupied, in place."""
        gx, gy = self._centres()
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = gx - x, gy - y
        local_x, local_y = c * dx + s * dy, -s * dx + c * dy
        self.occupied |= (np.abs(local_x) <= half_x) & (np.abs(local_y) <= half_y)

    # ---- queries ---------------------------------------------------------
    def cell(self, x, y) -> tuple[int, int]:
        return (int(round((x - self.origin[0]) / self.resolution)),
                int(round((y - self.origin[1]) / self.resolution)))

    def world(self, i, j) -> tuple[float, float]:
        return (self.origin[0] + i * self.resolution,
                self.origin[1] + j * self.resolution)

    def inside(self, i, j) -> bool:
        nx, ny = self.occupied.shape
        return 0 <= i < nx and 0 <= j < ny

    def free(self, x, y) -> bool:
        i, j = self.cell(x, y)
        return self.inside(i, j) and not self.occupied[i, j]

    def free_at(self, points) -> np.ndarray:
        """`free` for many points at once."""
        pts = np.asarray(points, float).reshape(-1, 2)
        ij = np.rint((pts - self.origin) / self.resolution).astype(int)
        nx, ny = self.occupied.shape
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
        out = np.zeros(len(pts), dtype=bool)
        out[inside] = ~self.occupied[ij[inside, 0], ij[inside, 1]]
        return out

    def line_free(self, p, q) -> bool:
        """True if the straight segment p -> q stays in free cells.
        Sampled at half the cell size, which cannot skip a cell."""
        p, q = np.asarray(p, float)[:2], np.asarray(q, float)[:2]
        n = max(2, int(np.ceil(np.linalg.norm(q - p) / (self.resolution / 2))) + 1)
        return bool(self.free_at(p + np.linspace(0, 1, n)[:, None] * (q - p)).all())

    def rect_free(self, x, y, yaw, half_x, half_y) -> bool:
        """True if no occupied cell centre lies inside the oriented rectangle.

        The robot's own footprint at a pose, on the raw grid.  This is what
        lets it park 0.24 m from a table when its circle would want 0.26.
        """
        gx, gy = self._centres()
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = gx - x, gy - y
        local_x, local_y = c * dx + s * dy, -s * dx + c * dy
        inside = (np.abs(local_x) <= half_x) & (np.abs(local_y) <= half_y)
        return not bool((inside & self.occupied).any())

    def clearance(self, points) -> np.ndarray:
        """Distance from each point to the nearest occupied cell centre, m."""
        pts = np.asarray(points, float).reshape(-1, 2)
        occ = np.argwhere(self.occupied) * self.resolution + self.origin
        if not len(occ):
            return np.full(len(pts), np.inf)
        d = np.linalg.norm(pts[:, None, :] - occ[None, :, :], axis=2)
        return d.min(axis=1)

    def ascii(self, path=None, marks=()) -> str:
        """The map as text, y up, every other row so it is not twice too tall.
        '#' occupied, '.' free, '*' the path, '+' a mark."""
        nx, ny = self.occupied.shape
        layer = np.zeros((nx, ny), dtype=np.int8)      # 0 free, 1 occupied, 2 path, 3 mark
        layer[self.occupied] = 1
        for value, points in ((2, path), (3, marks)):
            if points is None:
                continue
            for x, y in np.asarray(points, float).reshape(-1, 2):
                i, j = self.cell(x, y)
                if self.inside(i, j):
                    layer[i, j] = value
        glyph = ".#*+"
        rows = []
        for j in range(ny - 1, -1, -2):
            pair = layer[:, j] if j == 0 else np.maximum(layer[:, j], layer[:, j - 1])
            rows.append("".join(glyph[v] for v in pair))
        return "\n".join(rows)

    def _centres(self):
        nx, ny = self.occupied.shape
        xs = self.origin[0] + self.resolution * np.arange(nx)
        ys = self.origin[1] + self.resolution * np.arange(ny)
        return np.meshgrid(xs, ys, indexing="ij")
```

- [ ] **Step 4: Run the check**

Run: `.venv/bin/python scripts/check_navigation.py`
Expected:
```
grid ok
all ok
```

If `room.occupied.shape` is not `(121, 91)`, the `np.arange` end is off by a rounding: it must include `+resolution / 2` on the stop so `3.0` is the last x.

- [ ] **Step 5: Commit**

```bash
git add src/rlbot/navmap.py scripts/check_navigation.py
git commit -m "Add the occupancy grid the planner will run on

Claude-Session: https://claude.ai/code/session_01CQtc6f3qAqzNU3TbMU9DkB"
```

---

### Task 2: Read the map SLAM Toolbox saves

**Files:**
- Modify: `src/rlbot/navmap.py`
- Modify: `scripts/check_navigation.py`

- [ ] **Step 1: Write the failing check**

Add to `scripts/check_navigation.py`, after `check_grid`:

```python
def check_pgm():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image = np.full((4, 6), 254, dtype=np.uint8)        # 254 is free in a saved map
        image[0, :] = 0                                     # top row occupied: that is max y
        image[2, 1] = 205                                   # one unknown cell
        (tmp / "map.pgm").write_bytes(b"P5\n# made by a test\n6 4\n255\n" + image.tobytes())
        (tmp / "map.yaml").write_text(
            "image: map.pgm\nmode: trinary\nresolution: 0.5\norigin: [-1.0, -2.0, 0.0]\n"
            "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")

        g = OccupancyGrid.from_pgm(tmp / "map.yaml")
        assert g.occupied.shape == (6, 4) and g.resolution == 0.5
        assert g.origin == (-0.75, -1.75), "cell (0, 0) is half a cell in from the yaml origin"
        assert g.occupied[:, 3].all(), "the top image row is the highest y row"
        assert not g.occupied[:, 0].any()
        assert g.occupied[1, 1], "unknown counts as occupied"
        assert g.occupied.sum() == 6 + 1

        boxed = OccupancyGrid.from_pgm(tmp / "map.yaml",
                                       extra_boxes=[(0.25, -0.75, 0.0, 0.1, 0.1)])
        assert boxed.occupied.sum() == 6 + 1 + 1 and boxed.occupied[2, 2]
    print("pgm ok")
```

And call it from `__main__`:

```python
if __name__ == "__main__":
    check_grid()
    check_pgm()
    print("all ok")
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/bin/python scripts/check_navigation.py`
Expected: `AttributeError: type object 'OccupancyGrid' has no attribute 'from_pgm'`

- [ ] **Step 3: Add from_pgm and the two readers**

In `src/rlbot/navmap.py`, add inside `OccupancyGrid`, right after `from_room`:

```python
    @classmethod
    def from_pgm(cls, yaml_path, extra_boxes=()):
        """A map as SLAM Toolbox's save_map writes it: a PGM image and a YAML
        with resolution, origin and thresholds.

        Cells the mapper never saw count as occupied - the saver writes them
        as 205, which with its own default free_thresh of 0.25 would reload
        as free, and a planner that treats the unseen as open floor drives
        into it.  `extra_boxes` are (x, y, yaw, half_x, half_y) rectangles
        marked occupied on top: the table tops the lidar cannot see.
        """
        yaml_path = Path(yaml_path)
        meta = _read_yaml(yaml_path)
        image = _read_pgm(yaml_path.parent / meta["image"])     # (rows, cols), row 0 at the top
        resolution = float(meta["resolution"])
        ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
        shade = image / 255.0
        occ = shade if int(meta.get("negate", 0)) else 1.0 - shade
        free = (occ < float(meta.get("free_thresh", 0.25))) & (image != 205)
        occupied = ~free.T[:, ::-1]        # (cols, rows), row axis flipped so index 0 is min y
        grid = cls(np.ascontiguousarray(occupied), resolution,
                   (ox + resolution / 2, oy + resolution / 2))
        for box in extra_boxes:
            grid.fill_box(*box)
        return grid
```

And add at module level, after `table_tops`:

```python
def _read_yaml(path) -> dict:
    """The handful of `key: value` lines a map_server YAML has.

    ponytail: flat keys plus one list is all map_saver ever writes; reach for
    a real YAML parser the day it does not parse.
    """
    meta = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if value.startswith("["):
            meta[key] = [float(v) for v in value.strip("[]").split(",")]
        else:
            meta[key] = value.strip("'\"")
    return meta


def _read_pgm(path) -> np.ndarray:
    """A binary (P5) or ASCII (P2) PGM as a float array scaled to 0..255."""
    raw = Path(path).read_bytes()
    tokens, i = [], 0
    while len(tokens) < 4:
        while i < len(raw) and raw[i : i + 1].isspace():
            i += 1
        if raw[i : i + 1] == b"#":
            while i < len(raw) and raw[i : i + 1] != b"\n":
                i += 1
            continue
        j = i
        while j < len(raw) and not raw[j : j + 1].isspace():
            j += 1
        tokens.append(raw[i:j])
        i = j
    magic, width, height, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    i += 1                                       # the single whitespace after maxval
    if magic == b"P5":
        dtype = np.uint8 if maxval < 256 else ">u2"
        pixels = np.frombuffer(raw, dtype=dtype, count=width * height, offset=i)
    elif magic == b"P2":
        pixels = np.array(raw[i:].split()[: width * height], dtype=float)
    else:
        raise ValueError(f"{path}: not a PGM (magic {magic!r})")
    return pixels.reshape(height, width).astype(float) * (255.0 / maxval)
```

- [ ] **Step 4: Run the check**

Run: `.venv/bin/python scripts/check_navigation.py`
Expected:
```
grid ok
pgm ok
all ok
```

- [ ] **Step 5: Commit**

```bash
git add src/rlbot/navmap.py scripts/check_navigation.py
git commit -m "Read the occupancy map SLAM Toolbox saves

Claude-Session: https://claude.ai/code/session_01CQtc6f3qAqzNU3TbMU9DkB"
```

---

### Task 3: Point build_room.py at the shared grid

**Files:**
- Modify: `scripts/build_room.py`

- [ ] **Step 1: Record the current output**

Run: `.venv/bin/python scripts/build_room.py > /tmp/build_room_before.txt; git status --short models/`
Expected: `git status` prints nothing (the committed XML is what the script writes). Keep `/tmp/build_room_before.txt`.

- [ ] **Step 2: Replace the local grid code with imports**

In `scripts/build_room.py`:

Replace the import block line

```python
from rlbot.arm import ArmIK, down_quat                            # noqa: E402
```

with

```python
from rlbot.arm import ArmIK, down_quat                            # noqa: E402
from rlbot.navmap import MARGIN, OccupancyGrid, robot_footprint   # noqa: E402
```

Delete these, which now live in `navmap.py`: the lines

```python
ROBOT_MODEL = REPO / "models" / "bracketbot.xml"
GRID = 0.05            # m, cell size of the clearance map
MARGIN = 0.05          # m of daylight the robot should have on top of its own size
```

the whole `def robot_footprint():` function, and the whole `def occupancy(model, radius, height):` function.

Replace the body of `check_clearance` from its first line through the `print(f"    floor it can stand on: ...` statement with:

```python
def check_clearance(model) -> list[str]:
    """Is there room for this robot to stand, turn and get to every table?"""
    radius, height, half = robot_footprint()
    grid = OccupancyGrid.from_room().inflate(radius + MARGIN)
    free = ~grid.occupied
    xs = grid.origin[0] + grid.resolution * np.arange(free.shape[0])
    ys = grid.origin[1] + grid.resolution * np.arange(free.shape[1])
    seen = reachable(xs, ys, free, (0.0, 0.0))
    problems = []

    print(f"\n  clearance for a {2 * radius * 100:.0f} cm wide, "
          f"{height * 100:.0f} cm tall robot, plus {MARGIN * 100:.0f} cm")
    print(f"    floor it can stand on: {free.sum() * grid.resolution ** 2:.1f} m2 of "
          f"{ROOM_X * ROOM_Y:.1f} m2, and {seen.sum() * grid.resolution ** 2:.1f} m2 of that "
          f"reachable from the middle")
```

The rest of `check_clearance` (the per-table loop, the ASCII print, the return) is unchanged.

- [ ] **Step 3: Run it and compare**

Run: `.venv/bin/python scripts/build_room.py > /tmp/build_room_after.txt; echo exit=$?; git status --short models/; diff /tmp/build_room_before.txt /tmp/build_room_after.txt`
Expected: `exit=0`, no changes under `models/`, and the diff shows at most the two floor-area numbers moving by a few tenths of a m² (raster inflation versus the old exact-distance test). Every table must still print `ok`. If any prints `MISS`, the inflation is wrong: check `inflate` pads with `True` and uses `<=` on the radius.

- [ ] **Step 4: Update the README number if it moved**

If the "floor it can stand on" number changed, edit the sentence in `README.md` line 30 that reads `16.2 of the 27 m2 of floor is standable` to the new number.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_room.py README.md
git commit -m "Have the room builder use the shared occupancy grid

Claude-Session: https://claude.ai/code/session_01CQtc6f3qAqzNU3TbMU9DkB"
```

---

### Task 4: A* and the shortcut

**Files:**
- Create: `src/rlbot/planner.py`
- Modify: `scripts/check_navigation.py`

- [ ] **Step 1: Write the failing check**

Add to `scripts/check_navigation.py`. Import line, next to the navmap import:

```python
from rlbot.planner import NoPath, astar, shortcut   # noqa: E402
```

Check function, after `check_pgm`:

```python
def check_astar():
    g = small_grid()
    cells = astar(g, (1, 5), (9, 5))
    assert cells[0] == (1, 5) and cells[-1] == (9, 5)
    assert all(not g.occupied[c] for c in cells)
    assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 for a, b in zip(cells, cells[1:]))
    assert len(cells) >= 10, "has to go round the post"

    points = np.array([g.world(i, j) for i, j in cells])
    keep = shortcut(g, points)
    assert keep[0] == 0 and keep[-1] == len(points) - 1 and len(keep) < len(points)
    assert all(g.line_free(points[a], points[b]) for a, b in zip(keep, keep[1:]))

    with np.testing.assert_raises(NoPath):
        astar(g, (5, 5), (9, 5))
    walled = small_grid()
    walled.occupied[5, :] = True
    with np.testing.assert_raises(NoPath):
        astar(walled, (1, 5), (9, 5))
    print("astar ok")
```

Add `check_astar()` to `__main__` after `check_pgm()`.

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/bin/python scripts/check_navigation.py`
Expected: `ModuleNotFoundError: No module named 'rlbot.planner'`

- [ ] **Step 3: Write planner.py with the search**

Create `src/rlbot/planner.py`:

```python
"""From a pose to a pose across the room: a curved path, and a schedule to drive it.

`plan` finds the route.  A* on the inflated grid gives the corridor; a
line-of-sight shortcut takes the staircase corners out of it; a clamped cubic
spline through what is left gives a curve that leaves tangent to the robot's
heading and arrives tangent to the dock.  Because a dock sits inside the
robot's own inflation radius, the curve is bracketed by straight runs along
the start's and the goal's axes that are checked against the robot's
rectangle rather than its circle, and preceded by a turn in place only when
the corridor leaves more than 45 degrees off the heading.

`profile` turns that into a time-stamped schedule of speed and yaw rate under
the balancer's limits: slow through bends so the yaw rate stays under what it
survives, ramps no steeper than its DriveController will pass.

Nothing here touches the sim.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .navmap import OccupancyGrid
from .room import TABLES


class NoPath(RuntimeError):
    """There is no route the planner is willing to drive."""


# ---- search --------------------------------------------------------------
def astar(grid: OccupancyGrid, start, goal) -> list[tuple[int, int]]:
    """8-connected A* between two cells.  Returns the cells, start to goal.

    Diagonals may not cut a corner: both the cells they pass between have to
    be free, or the robot's circle would clip the corner they cut.
    """
    start, goal = tuple(start), tuple(goal)
    if not grid.inside(*start) or grid.occupied[start]:
        raise NoPath(f"start cell {start} at {grid.world(*start)} is occupied")
    if not grid.inside(*goal) or grid.occupied[goal]:
        raise NoPath(f"goal cell {goal} at {grid.world(*goal)} is occupied")

    moves = [(di, dj, math.hypot(di, dj))
             for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]

    def h(c):
        return math.hypot(c[0] - goal[0], c[1] - goal[1])

    best = {start: 0.0}
    came: dict = {}
    heap = [(h(start), 0.0, start)]
    while heap:
        _, g, cur = heapq.heappop(heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if g > best.get(cur, math.inf):
            continue
        for di, dj, cost in moves:
            nxt = (cur[0] + di, cur[1] + dj)
            if not grid.inside(*nxt) or grid.occupied[nxt]:
                continue
            if di and dj and (grid.occupied[cur[0] + di, cur[1]]
                              or grid.occupied[cur[0], cur[1] + dj]):
                continue
            ng = g + cost
            if ng < best.get(nxt, math.inf):
                best[nxt] = ng
                came[nxt] = cur
                heapq.heappush(heap, (ng + h(nxt), ng, nxt))
    raise NoPath(f"no route on the grid from {grid.world(*start)} to {grid.world(*goal)}")


def shortcut(grid: OccupancyGrid, points) -> list[int]:
    """Indices of the waypoints a straight line cannot see past.  Greedy:
    from each kept point, jump to the farthest one still in line of sight."""
    points = np.asarray(points, float)
    keep, i = [0], 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not grid.line_free(points[i], points[j]):
            j -= 1
        keep.append(j)
        i = j
    return keep
```

- [ ] **Step 4: Run the check**

Run: `.venv/bin/python scripts/check_navigation.py`
Expected:
```
grid ok
pgm ok
astar ok
all ok
```

- [ ] **Step 5: Commit**

```bash
git add src/rlbot/planner.py scripts/check_navigation.py
git commit -m "Add A* and a line-of-sight shortcut on the grid

Claude-Session: https://claude.ai/code/session_01CQtc6f3qAqzNU3TbMU9DkB"
```

---

### Task 5: The spline and `plan` on open floor

**Files:**
- Modify: `src/rlbot/planner.py`
- Modify: `scripts/check_navigation.py`

- [ ] **Step 1: Write the failing check**

Update the planner import in `scripts/check_navigation.py`:

```python
from rlbot.planner import (Limits, NoPath, _clamped_spline, _sample_spline,   # noqa: E402
                           astar, plan, shortcut)
```

Add after `check_astar`:

```python
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

    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))     # 2 x 2 m, empty
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
```

Add `check_spline()` to `__main__` after `check_astar()`.

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/bin/python scripts/check_navigation.py`
Expected: `ImportError: cannot import name 'Limits' from 'rlbot.planner'`

- [ ] **Step 3: Add the types, the spline, and a first `plan`**

In `src/rlbot/planner.py`, add after the `NoPath` class:

```python
DS = 0.02                          # m between samples of a finished path
TURN_FIRST = math.radians(45)      # heading error above which the robot turns in place before the curve
BACKOFF_MAX = 1.0                  # m to search along a pose's axis for a spot the circle fits


@dataclass
class Limits:
    v_max: float = 0.15        # m/s.       DriveController's max_speed
    omega_max: float = 0.3     # rad/s.     DriveController's max_yaw_rate
    a_max: float = 0.1         # m/s^2.     DriveController's linear_accel
    alpha_max: float = 0.3     # rad/s^2.   DriveController's angular_accel
    v_min: float = 0.05        # m/s floor on tight curves, so it never parks mid-arc
    radius: float = 0.21       # m, half the 0.42 m driving footprint (navmap.robot_footprint)
    half_depth: float = 0.094  # m, chassis half-extent along the heading
    half_width: float = 0.186  # m, chassis half-extent across it
    margin: float = 0.05       # m of daylight on top of the radius


@dataclass
class Path:
    """A route in four parts, any of which may be empty.

    `exit` metres straight along the start heading (negative is reverse),
    `turn` radians in place, the curve (`xy`, `yaw`, `kappa`, `s`, sampled
    every DS metres of arc), then `dock` metres straight along the goal's
    yaw.  `waypoints` are the spline's knots, for drawing.
    """

    start: tuple[float, float, float]
    goal: tuple[float, float, float]
    exit: float
    turn: float
    xy: np.ndarray
    yaw: np.ndarray
    kappa: np.ndarray
    s: np.ndarray
    dock: float
    waypoints: np.ndarray

    @property
    def length(self) -> float:
        """The curve alone, m."""
        return float(self.s[-1]) if len(self.s) else 0.0

    @property
    def min_radius(self) -> float:
        k = float(np.abs(self.kappa).max()) if len(self.kappa) else 0.0
        return math.inf if k == 0 else 1.0 / k


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


# ---- the curve -----------------------------------------------------------
def _clamped_spline(knots, d0, dn):
    """Coefficients of a clamped cubic spline through `knots` (m, 2), with
    first derivatives d0 and dn at the ends, parametrised by chord length.

    Burden & Faires' clamped spline: solve the tridiagonal system for the
    quadratic coefficients c, then b and d follow.  Returns (u, a, b, c, d)
    with one row per segment.
    """
    a = np.asarray(knots, float)
    n = len(a) - 1
    h = np.linalg.norm(np.diff(a, axis=0), axis=1)
    if n < 1 or (h <= 0).any():
        raise ValueError("a spline needs at least two distinct knots")
    u = np.concatenate([[0.0], np.cumsum(h)])
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros((n + 1, 2))
    A[0, 0], A[0, 1] = 2 * h[0], h[0]
    rhs[0] = 3 * (a[1] - a[0]) / h[0] - 3 * np.asarray(d0, float)
    A[n, n - 1], A[n, n] = h[n - 1], 2 * h[n - 1]
    rhs[n] = 3 * np.asarray(dn, float) - 3 * (a[n] - a[n - 1]) / h[n - 1]
    for i in range(1, n):
        A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        rhs[i] = 3 * (a[i + 1] - a[i]) / h[i] - 3 * (a[i] - a[i - 1]) / h[i - 1]
    c = np.linalg.solve(A, rhs)
    b = (a[1:] - a[:-1]) / h[:, None] - h[:, None] * (c[1:] + 2 * c[:-1]) / 3
    d = (c[1:] - c[:-1]) / (3 * h[:, None])
    return u, a[:-1], b, c[:-1], d


def _sample_spline(coef, step):
    """Points, headings, signed curvatures and arc lengths along the spline,
    about `step` apart in the parameter."""
    u, a, b, c, d = coef
    p, dp, ddp = [], [], []
    last = len(a) - 1
    for i in range(len(a)):
        n = max(2, int(np.ceil((u[i + 1] - u[i]) / step)) + 1)
        t = np.linspace(0, u[i + 1] - u[i], n, endpoint=(i == last))[:, None]
        p.append(a[i] + b[i] * t + c[i] * t ** 2 + d[i] * t ** 3)
        dp.append(b[i] + 2 * c[i] * t + 3 * d[i] * t ** 2)
        ddp.append(2 * c[i] + 6 * d[i] * t)
    p, dp, ddp = np.vstack(p), np.vstack(dp), np.vstack(ddp)
    yaw = np.arctan2(dp[:, 1], dp[:, 0])
    speed = np.hypot(dp[:, 0], dp[:, 1])
    kappa = (dp[:, 0] * ddp[:, 1] - dp[:, 1] * ddp[:, 0]) / np.maximum(speed ** 3, 1e-9)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    return p, yaw, kappa, s


def _resample(p, yaw, kappa, s, ds):
    """The same curve at a uniform arc-length spacing, at least three samples."""
    n = max(3, int(np.ceil(s[-1] / ds)) + 1)
    ss = np.linspace(0, s[-1], n)
    x, y = np.interp(ss, s, p[:, 0]), np.interp(ss, s, p[:, 1])
    yw = np.arctan2(np.interp(ss, s, np.sin(yaw)), np.interp(ss, s, np.cos(yaw)))
    return np.column_stack([x, y]), yw, np.interp(ss, s, kappa), ss


def _fit(inflated: OccupancyGrid, corridor, keep, t0, tn):
    """A spline through the kept corridor points that stays on free cells.

    If it clips, put back the corridor point nearest the first clipped sample
    and go again.  Each retry keeps strictly more points, so this terminates;
    a spline through every corridor cell hugs the polyline to within a cell.
    """
    keep = list(keep)
    while True:
        knots = corridor[keep]
        p, yaw, kappa, s = _sample_spline(_clamped_spline(knots, t0, tn),
                                          inflated.resolution / 2)
        bad = np.flatnonzero(~inflated.free_at(p))
        if not len(bad):
            return (*_resample(p, yaw, kappa, s, DS), knots)
        dist = np.linalg.norm(corridor - p[bad[0]], axis=1)
        dist[keep] = np.inf
        if not np.isfinite(dist).any():
            raise NoPath("the curve clips an obstacle even through every corridor cell; "
                         "the margin is too small for this map")
        keep = sorted(keep + [int(np.argmin(dist))])


# ---- the whole route -----------------------------------------------------
def plan(grid: OccupancyGrid, start, goal, limits: Limits = Limits()) -> Path:
    """The route from `start` to `goal`, both (x, y, yaw) in the grid's frame."""
    inflated = grid.inflate(limits.radius + limits.margin)
    start, goal = tuple(map(float, start)), tuple(map(float, goal))
    sx, sy, syaw = start
    gx, gy, gyaw = goal
    a = np.array([sx, sy])
    b = np.array([gx, gy])
    exit_len, d_goal, turn = 0.0, 0.0, 0.0

    curve = None
    if np.linalg.norm(b - a) > DS:
        cells = astar(inflated, inflated.cell(*a), inflated.cell(*b))
        corridor = np.array([tuple(a)] + [inflated.world(i, j) for i, j in cells[1:-1]] + [tuple(b)])
        keep = shortcut(inflated, corridor)
        first = corridor[keep[1]] - a
        heading = math.atan2(first[1], first[0])
        if abs(_wrap(heading - syaw)) > TURN_FIRST:
            turn = _wrap(heading - syaw)
            t0 = first / np.linalg.norm(first)
        else:
            t0 = np.array([math.cos(syaw), math.sin(syaw)])
        tn = np.array([math.cos(gyaw), math.sin(gyaw)])
        curve = _fit(inflated, corridor, keep, t0, tn)
    elif abs(_wrap(gyaw - syaw)) > math.radians(1):
        turn = _wrap(gyaw - syaw)

    if curve is None:
        xy, yaw, kappa, s = np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0)
        knots = np.vstack([a, b])
    else:
        xy, yaw, kappa, s, knots = curve
    return Path(start, goal, exit_len, turn, xy, yaw, kappa, s, -d_goal, knots)


def goal_for(name: str):
    """A table's docking pose (x, y, yaw) by short name: 'ball', 'cubes', 'ware'."""
    for table in TABLES:
        if table.name.split("_")[1] == name:
            return table.dock
    raise KeyError(f"no table called {name!r}; one of "
                   + ", ".join(t.name.split("_")[1] for t in TABLES))
```

- [ ] **Step 4: Run the check**

Run: `.venv/bin/python scripts/check_navigation.py`
Expected:
```
grid ok
pgm ok
astar ok
spline ok
all ok
```

- [ ] **Step 5: Commit**

```bash
git add src/rlbot/planner.py scripts/check_navigation.py
git commit -m "Fit a clamped cubic spline through the shortcut corridor

Claude-Session: https://claude.ai/code/session_01CQtc6f3qAqzNU3TbMU9DkB"
```

---

### Task 6: Straight runs into and out of a dock

**Files:**
- Modify: `src/rlbot/planner.py`
- Modify: `scripts/check_navigation.py`

- [ ] **Step 1: Write the failing check**

Add after `check_spline` in `scripts/check_navigation.py`:

```python
def check_dock():
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    g.fill_box(1.0, 1.7, 0.0, 0.5, 0.3)              # a table top: x 0.5..1.5, y 1.4..2.0
    lim = Limits(radius=0.1, margin=0.05, half_depth=0.05, half_width=0.09)
    goal = (1.0, 1.28, math.pi / 2)                  # 12 cm off the edge, facing it
    assert not g.inflate(lim.radius + lim.margin).free(*goal[:2]), "the dock is inside the inflation"

    path = plan(g, (0.3, 0.3, math.radians(50)), goal, lim)     # within 45 deg of the corridor
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
        plan(g, (0.3, 0.3, 0.0), (1.0, 1.7, 0.0), lim)      # buried in the table
    print("dock ok")
```

Add `_wrap` to the planner import line, and `check_dock()` to `__main__` after `check_spline()`.

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/bin/python scripts/check_navigation.py`
Expected: `rlbot.planner.NoPath: goal cell (20, 26) at (1.0, 1.3) is occupied`

- [ ] **Step 3: Add `_straight_out` and use it in `plan`**

In `src/rlbot/planner.py`, add before `plan`:

```python
def _straight_out(grid: OccupancyGrid, inflated: OccupancyGrid, pose, half) -> float:
    """Signed distance along the pose's heading to the nearest point where the
    robot's circle is clear.  Backwards first, because that is how you leave
    a table.  Every pose on the way has to fit the robot's rectangle on the
    raw grid, so it cannot back through a table leg to get there."""
    x, y, yaw = pose
    if not grid.rect_free(x, y, yaw, *half):
        raise NoPath(f"the robot does not fit at ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg)")
    if inflated.free(x, y):
        return 0.0
    c, s = math.cos(yaw), math.sin(yaw)
    step = grid.resolution / 2
    for sign in (-1, 1):
        for k in range(1, int(BACKOFF_MAX / step) + 1):
            d = sign * k * step
            px, py = x + d * c, y + d * s
            if not grid.rect_free(px, py, yaw, *half):
                break
            if inflated.free(px, py):
                return d
    raise NoPath(f"no room within {BACKOFF_MAX} m along the axis of "
                 f"({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg)")
```

In `plan`, replace

```python
    a = np.array([sx, sy])
    b = np.array([gx, gy])
    exit_len, d_goal, turn = 0.0, 0.0, 0.0
```

with

```python
    half = (limits.half_depth + limits.margin, limits.half_width + limits.margin)
    exit_len = _straight_out(grid, inflated, start, half)
    d_goal = _straight_out(grid, inflated, goal, half)
    a = np.array([sx + exit_len * math.cos(syaw), sy + exit_len * math.sin(syaw)])
    b = np.array([gx + d_goal * math.cos(gyaw), gy + d_goal * math.sin(gyaw)])
    turn = 0.0
```

- [ ] **Step 4: Run the check**

Run: `.venv/bin/python scripts/check_navigation.py`
Expected:
```
grid ok
pgm ok
astar ok
spline ok
dock ok
all ok
```

- [ ] **Step 5: Commit**

```bash
git add src/rlbot/planner.py scripts/check_navigation.py
git commit -m "Bracket the curve with straight runs so docks are reachable

Claude-Session: https://claude.ai/code/session_01CQtc6f3qAqzNU3TbMU9DkB"
```

---

### Task 7: The speed profile

**Files:**
- Modify: `src/rlbot/planner.py`
- Modify: `scripts/check_navigation.py`

- [ ] **Step 1: Write the failing check**

Add `Path, Trajectory, profile` to the planner import in `scripts/check_navigation.py`, and after `check_dock`:

```python
def check_profile():
    lim = Limits()
    n = 51
    straight = Path(start=(0.0, 0.0, 0.0), goal=(1.0, 0.0, 0.0), exit=0.0, turn=0.0,
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
    bend = Path(start=(0.2, 0.0, math.pi / 2), goal=(0.0, 0.2, math.pi), exit=0.0, turn=0.0,
                xy=np.column_stack([0.2 * np.cos(theta), 0.2 * np.sin(theta)]),
                yaw=theta + math.pi / 2, kappa=np.full(40, 5.0), s=0.2 * theta,
                dock=0.0, waypoints=np.zeros((2, 2)))
    traj = profile(bend, lim)
    assert np.abs(traj.omega).max() <= lim.omega_max + 1e-9
    assert traj.v.max() <= lim.omega_max / 5 + 1e-9, "slowed for the bend"
    assert traj.v[1:-1].min() > 0, "never parks mid-arc"

    spin = Path(start=(0.0, 0.0, 0.0), goal=(0.0, 0.0, math.pi / 2), exit=0.0, turn=math.pi / 2,
                xy=np.zeros((0, 2)), yaw=np.zeros(0), kappa=np.zeros(0), s=np.zeros(0),
                dock=0.0, waypoints=np.zeros((2, 2)))
    traj = profile(spin, lim)
    assert np.all(traj.v == 0) and np.abs(traj.omega).max() <= lim.omega_max + 1e-9
    assert abs(traj.yaw[-1] - math.pi / 2) < 1e-9 and np.allclose(traj.xy, 0)
    alpha = np.abs(np.diff(traj.omega) / np.diff(traj.t))
    assert alpha.max() <= lim.alpha_max * 1.1

    out_and_in = Path(start=(0.0, 0.0, 0.0), goal=(0.1, 0.0, 0.0), exit=-0.1, turn=0.0,
                      xy=np.zeros((0, 2)), yaw=np.zeros(0), kappa=np.zeros(0), s=np.zeros(0),
                      dock=0.2, waypoints=np.zeros((2, 2)))
    traj = profile(out_and_in, lim)
    assert traj.v.min() < 0, "the exit is driven in reverse"
    np.testing.assert_allclose(traj.xy[-1], [0.1, 0.0], atol=1e-9)
    assert np.all(traj.yaw == 0)
    print("profile ok")
```

Add `check_profile()` to `__main__` after `check_dock()`.

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/bin/python scripts/check_navigation.py`
Expected: `ImportError: cannot import name 'Trajectory' from 'rlbot.planner'`

- [ ] **Step 3: Add `Trajectory` and `profile`**

In `src/rlbot/planner.py`, add after the `Path` class:

```python
@dataclass
class Trajectory:
    """A schedule: where the robot should be and what it should be doing at t."""

    t: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray
    v: np.ndarray              # m/s along the heading, negative in reverse
    omega: np.ndarray          # rad/s

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def at(self, t: float):
        """(xy, yaw, v, omega) at time t, interpolated, clamped to the ends."""
        t = min(max(float(t), 0.0), self.duration)
        xy = np.array([np.interp(t, self.t, self.xy[:, 0]), np.interp(t, self.t, self.xy[:, 1])])
        yaw = math.atan2(np.interp(t, self.t, np.sin(self.yaw)),
                         np.interp(t, self.t, np.cos(self.yaw)))
        return xy, yaw, float(np.interp(t, self.t, self.v)), float(np.interp(t, self.t, self.omega))
```

And add at the end of the file, before `goal_for`:

```python
# ---- the schedule --------------------------------------------------------
DTH = 0.02      # rad between samples of a turn in place


def _ramp(n: int, ds: float, cap, accel: float):
    """Speeds at n samples ds apart: under `cap`, zero at both ends, and never
    changing faster than `accel` per unit time.  Forward pass accelerates,
    backward pass leaves room to stop.  Returns (v, t)."""
    cap = np.broadcast_to(np.asarray(cap, float), n).copy()
    v = np.zeros(n)
    for i in range(1, n):
        v[i] = min(cap[i], math.sqrt(v[i - 1] ** 2 + 2 * accel * ds))
    v[-1] = 0.0
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * accel * ds))
    avg = np.maximum((v[1:] + v[:-1]) / 2, 1e-9)
    return v, np.concatenate([[0.0], np.cumsum(ds / avg)])


def _straight(p0, yaw: float, length: float, limits: Limits) -> Trajectory:
    n = max(3, int(np.ceil(abs(length) / DS)) + 1)
    ds = abs(length) / (n - 1)
    v, t = _ramp(n, ds, limits.v_max, limits.a_max)
    sign = 1.0 if length >= 0 else -1.0
    along = sign * np.linspace(0, abs(length), n)
    xy = np.asarray(p0, float) + np.outer(along, [math.cos(yaw), math.sin(yaw)])
    return Trajectory(t, xy, np.full(n, yaw), sign * v, np.zeros(n))


def _turn(p, yaw0: float, dyaw: float, limits: Limits) -> Trajectory:
    n = max(3, int(np.ceil(abs(dyaw) / DTH)) + 1)
    ds = abs(dyaw) / (n - 1)
    w, t = _ramp(n, ds, limits.omega_max, limits.alpha_max)
    sign = 1.0 if dyaw >= 0 else -1.0
    yaw = yaw0 + sign * np.linspace(0, abs(dyaw), n)
    return Trajectory(t, np.tile(np.asarray(p, float), (n, 1)), yaw, np.zeros(n), sign * w)


def _curve(path: Path, limits: Limits) -> Trajectory:
    """Slow through bends so omega = v * kappa stays under omega_max, then
    trim the speed cap where d omega / dt still exceeds alpha_max.

    ponytail: ten rounds of 10 % trims.  At the start of a bend
    d omega / dt = a * kappa no matter the cap, so a residual can remain;
    the DriveController's ramp absorbs it and the next replan corrects.
    """
    s, kappa = path.s, path.kappa
    n, ds = len(s), float(s[1] - s[0])
    cap = np.minimum(limits.v_max, limits.omega_max / np.maximum(np.abs(kappa), 1e-9))
    cap = np.maximum(cap, limits.v_min)
    for _ in range(10):
        v, t = _ramp(n, ds, cap, limits.a_max)
        omega = v * kappa
        alpha = np.abs(np.diff(omega)) / np.maximum(np.diff(t), 1e-9)
        hot = np.flatnonzero(alpha > limits.alpha_max)
        if not len(hot):
            break
        cap[np.unique(np.concatenate([hot, hot + 1]))] *= 0.9
    return Trajectory(t, path.xy.copy(), path.yaw.copy(), v, omega)


def _join(parts: list[Trajectory]) -> Trajectory:
    """Concatenate, dropping each part's first sample: it repeats the last of
    the part before, at rest in the same place."""
    t, xy, yaw, v, w = [parts[0].t], [parts[0].xy], [parts[0].yaw], [parts[0].v], [parts[0].omega]
    for p in parts[1:]:
        t.append(p.t[1:] + t[-1][-1])
        xy.append(p.xy[1:])
        yaw.append(p.yaw[1:])
        v.append(p.v[1:])
        w.append(p.omega[1:])
    return Trajectory(np.concatenate(t), np.vstack(xy), np.concatenate(yaw),
                      np.concatenate(v), np.concatenate(w))


def profile(path: Path, limits: Limits = Limits()) -> Trajectory:
    """Time-stamp the route: straight out, turn, curve, straight in.  The
    robot is at rest between parts and never stops along the curve."""
    sx, sy, syaw = path.start
    gyaw = path.goal[2]
    parts = []
    px, py = sx + path.exit * math.cos(syaw), sy + path.exit * math.sin(syaw)
    if abs(path.exit) > 1e-6:
        parts.append(_straight((sx, sy), syaw, path.exit, limits))
    if abs(path.turn) > 1e-6:
        parts.append(_turn((px, py), syaw, path.turn, limits))
    if len(path.xy) >= 2:
        parts.append(_curve(path, limits))
        px, py = path.xy[-1]
    if abs(path.dock) > 1e-6:
        parts.append(_straight((px, py), gyaw, path.dock, limits))
    if not parts:
        return Trajectory(np.zeros(1), np.array([[sx, sy]]), np.array([syaw]),
                          np.zeros(1), np.zeros(1))
    return _join(parts)
```

- [ ] **Step 4: Run the check**

Run: `.venv/bin/python scripts/check_navigation.py`
Expected:
```
grid ok
pgm ok
astar ok
spline ok
dock ok
profile ok
all ok
```

If the trapezoid duration is off by more than 0.3 s, `_ramp` is using `accel * ds` where it needs `2 * accel * ds` (v² = v₀² + 2 a s).

- [ ] **Step 5: Commit**

```bash
git add src/rlbot/planner.py scripts/check_navigation.py
git commit -m "Profile a route into a speed and yaw-rate schedule

Claude-Session: https://claude.ai/code/session_01CQtc6f3qAqzNU3TbMU9DkB"
```

---

### Task 8: The nine room routes

**Files:**
- Create: `scripts/plan_path.py`

- [ ] **Step 1: Write the script**

Create `scripts/plan_path.py`:

```python
"""Plan every route between the start pose and the three table docks, and check them.

    .venv/bin/python scripts/plan_path.py
    .venv/bin/python scripts/plan_path.py --route cubes-ware
    .venv/bin/python scripts/plan_path.py --quiet          # no maps, just the verdicts

Draws each route on the room's occupancy map and refuses routes that clip the
furniture, break the drive limits, or do not arrive at the dock pose pointing
the right way.  Pure geometry: nothing here moves the robot.  `navigate.py` is
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

from rlbot.navmap import OccupancyGrid                             # noqa: E402
from rlbot.planner import Limits, NoPath, plan, profile             # noqa: E402
from rlbot.room import TABLES                                       # noqa: E402

START = (0.0, 0.0, 0.0)        # the room's 'start' keyframe


def routes():
    """(name, start pose, goal pose): the start to each dock, and dock to dock."""
    docks = {t.name.split("_")[1]: t.dock for t in TABLES}
    out = [(f"start-{n}", START, p) for n, p in docks.items()]
    out += [(f"{a}-{b}", pa, pb) for a, pa in docks.items() for b, pb in docks.items() if a != b]
    return out


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def check_route(grid, name, start, goal, limits, verbose=True) -> list[str]:
    try:
        path = plan(grid, start, goal, limits)
    except NoPath as error:
        print(f"  MISS {name:<12} {error}")
        return [f"{name}: {error}"]
    traj = profile(path, limits)

    clear = float(grid.clearance(path.xy).min()) if len(path.xy) else math.inf
    need = limits.radius + limits.margin - grid.resolution
    accel = np.abs(np.diff(traj.v) / np.diff(traj.t))
    alpha = np.abs(np.diff(traj.omega) / np.diff(traj.t))
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
    if accel.max() > limits.a_max * 1.1:
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
          f"peak alpha {alpha.max():.2f} rad/s2")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--route", help="one route, e.g. start-cubes or ware-ball")
    ap.add_argument("--quiet", action="store_true", help="verdicts only, no maps")
    args = ap.parse_args()

    grid = OccupancyGrid.from_room()
    limits = Limits()
    chosen = [r for r in routes() if not args.route or r[0] == args.route]
    if not chosen:
        ap.error(f"no route {args.route!r}; one of " + ", ".join(r[0] for r in routes()))

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
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python scripts/plan_path.py --quiet`
Expected: nine `ok` lines and `9 routes planned within the limits`, exit 0. Each `start-*` route has `exit +0.00` and a positive `dock`; each dock-to-dock route has a negative `exit`, a `turn` of well over 45°, and a positive `dock`. Peak alpha will be above 0.3 on most routes; that is reported, not failed.

If a route reports `MISS ... the curve clips an obstacle even through every corridor cell`, lower `TURN_FIRST` in `planner.py` to `math.radians(30)` and rerun: the clamped start tangent was pulling the curve into the furniture.

If a route reports `no route on the grid`, run without `--quiet` and look at the map: the approach point (`+` at the goal end) must sit on a `.` cell.

- [ ] **Step 3: Look at one map**

Run: `.venv/bin/python scripts/plan_path.py --route cubes-ware`
Expected: an ASCII room with `#` furniture, the `*` curve leaving the cubes dock, rounding the divider or pillar side, and ending at the ware dock; `+` at both ends.

- [ ] **Step 4: Commit**

```bash
git add scripts/plan_path.py
git commit -m "Plan and check the nine routes between the start and the docks

Claude-Session: https://claude.ai/code/session_01CQtc6f3qAqzNU3TbMU9DkB"
```

---

### Task 9: The navigator

**Files:**
- Create: `src/rlbot/navigate.py`
- Modify: `scripts/check_navigation.py`
- Modify: `src/rlbot/__init__.py`

- [ ] **Step 1: Write the failing check**

Add to the imports in `scripts/check_navigation.py`:

```python
from rlbot.navigate import Navigator                # noqa: E402
```

Add after `check_profile`:

```python
def check_navigator():
    g = OccupancyGrid(np.zeros((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    lim = Limits(radius=0.1, margin=0.05)
    nav = Navigator(g, lim, replan_every=2.0, off_path=0.10, pose_timeout=0.5)
    start, goal = (0.3, 0.3, 0.0), (1.7, 1.2, 0.0)

    traj = nav.go_to(goal, start, now=10.0)
    assert traj is nav.traj and not nav.done and nav.failed is None and nav.replans == []
    assert nav.step(start, 10.0) == (0.0, 0.0), "starts from rest"
    v, _ = nav.step(None, 10.5)
    assert v > 0, "no pose within the timeout: keeps playing the schedule"
    assert nav.step(None, 11.1) == (0.0, 0.0), "pose timed out: hold"

    xy, yaw, _, _ = traj.at(1.5)
    nav.step((xy[0], xy[1], yaw), 11.5)
    assert nav.replans == [], "on schedule, inside the replan period: no replan"

    xy, yaw, _, _ = traj.at(2.1)
    nav.step((xy[0], xy[1], yaw), 12.1)
    assert len(nav.replans) == 1 and nav.replans[0][1] < 0.01, "periodic replan"
    assert nav.t0 == 12.1

    xy, yaw, _, _ = nav.traj.at(0.5)
    nav.step((xy[0] + 0.2, xy[1], yaw), nav.t0 + 0.5)
    assert len(nav.replans) == 2 and abs(nav.replans[1][1] - 0.2) < 1e-6, "knocked off: replan now"

    nav.step(goal, nav.t0 + nav.traj.duration + 0.1)
    assert nav.done and nav.step(goal, nav.t0 + 100) == (0.0, 0.0)

    blocked = OccupancyGrid(np.ones((41, 41), dtype=bool), 0.05, (0.0, 0.0))
    stuck = Navigator(blocked, lim)
    assert stuck.go_to(goal, start, 0.0) is None and stuck.failed
    assert stuck.step(start, 0.0) == (0.0, 0.0)
    print("navigator ok")
```

Add `check_navigator()` to `__main__` after `check_profile()`.

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/bin/python scripts/check_navigation.py`
Expected: `ModuleNotFoundError: No module named 'rlbot.navigate'`

- [ ] **Step 3: Write navigate.py**

Create `src/rlbot/navigate.py`:

```python
"""Drive a planned schedule, and replan from the localised pose when it drifts.

Open loop on purpose: the schedule already respects every limit the
DriveController enforces, so the navigator just reads (speed, yaw_rate) off it
at the current time.  What corrects the drift a balancing robot accumulates is
a slow outer loop, not a tracking controller: every `replan_every` seconds, or
as soon as the pose is more than `off_path` from where the schedule says it
should be, it plans again from where the robot actually is.

It never touches the sim, the controller or ROS.  The caller hands it a pose
and a time and passes what comes back to `DriveController.command`, or to a
`/cmd_vel` publisher; it is the same code either way.
"""

from __future__ import annotations

import math

import numpy as np

from .navmap import OccupancyGrid
from .planner import Limits, NoPath, Trajectory, plan, profile


class Navigator:
    def __init__(self, grid: OccupancyGrid, limits: Limits = Limits(),
                 replan_every: float = 2.0, off_path: float = 0.10,
                 pose_timeout: float = 0.5, arrive_xy: float = 0.10,
                 arrive_yaw: float = math.radians(5), max_replans: int = 30):
        self.grid, self.limits = grid, limits
        self.replan_every, self.off_path = replan_every, off_path
        self.pose_timeout = pose_timeout
        self.arrive_xy, self.arrive_yaw = arrive_xy, arrive_yaw
        self.max_replans = max_replans
        self.goal = None
        self.traj: Trajectory | None = None
        self.t0 = 0.0
        self.done = False
        self.failed: str | None = None
        self.replans: list[tuple[float, float]] = []     # (time, metres off schedule)
        self._last_pose = 0.0

    def go_to(self, goal, pose_now, now: float) -> Trajectory | None:
        """Plan from where the robot is and start the clock.  None if it cannot."""
        self.goal = tuple(map(float, goal))
        self.done, self.failed, self.replans = False, None, []
        self._plan(pose_now, now)
        return self.traj

    def step(self, pose_now, now: float) -> tuple[float, float]:
        """(speed m/s, yaw_rate rad/s) to command right now.

        `pose_now` is (x, y, yaw), or None when no estimate is available; the
        schedule keeps playing without one for `pose_timeout` seconds, then
        holds.  Returns (0, 0) once done or after giving up.
        """
        if self.done or self.failed or self.traj is None:
            return 0.0, 0.0
        if pose_now is None:
            if now - self._last_pose > self.pose_timeout:
                return 0.0, 0.0
        else:
            self._last_pose = now
            elapsed = now - self.t0
            if self._arrived(pose_now):
                if elapsed >= self.traj.duration:
                    self.done = True
                    return 0.0, 0.0
            else:
                expected = self.traj.at(elapsed)[0]
                off = float(np.hypot(pose_now[0] - expected[0], pose_now[1] - expected[1]))
                if off > self.off_path or elapsed >= self.replan_every \
                        or elapsed >= self.traj.duration:
                    if len(self.replans) >= self.max_replans:
                        self.failed = (f"gave up after {self.max_replans} replans, "
                                       f"{off * 100:.0f} cm off schedule")
                        return 0.0, 0.0
                    self.replans.append((now, off))
                    self._plan(pose_now, now)
                    if self.traj is None:
                        return 0.0, 0.0
        _, _, v, omega = self.traj.at(now - self.t0)
        return v, omega

    def _plan(self, pose_now, now: float) -> None:
        try:
            self.traj = profile(plan(self.grid, pose_now, self.goal, self.limits), self.limits)
        except NoPath as error:
            self.traj, self.failed = None, str(error)
            return
        self.t0 = now
        self._last_pose = now

    def _arrived(self, pose) -> bool:
        dx, dy = pose[0] - self.goal[0], pose[1] - self.goal[1]
        dyaw = math.atan2(math.sin(pose[2] - self.goal[2]), math.cos(pose[2] - self.goal[2]))
        return math.hypot(dx, dy) <= self.arrive_xy and abs(dyaw) <= self.arrive_yaw


def true_pose(data) -> tuple[float, float, float]:
    """The wheel-axle midpoint and the chassis heading, from the sim.

    The same point the ROS bridge calls base_footprint, so the local check
    and the real thing measure arrival the same way.  For scoring and for the
    SLAM stand-in only: nothing that runs on the robot may read this.
    """
    axle = (data.body("wheel_left").xpos + data.body("wheel_right").xpos) / 2
    rot = data.body("root").xmat.reshape(3, 3)
    return float(axle[0]), float(axle[1]), math.atan2(rot[1, 0], rot[0, 0])
```

Then in `src/rlbot/__init__.py`, add the imports and names:

```python
from .navmap import OccupancyGrid
from .planner import Limits, NoPath, Path, Trajectory, goal_for, plan, profile
from .navigate import Navigator, true_pose
```

and extend `__all__` with:

```python
    "OccupancyGrid", "Limits", "NoPath", "Path", "Trajectory", "goal_for", "plan", "profile",
    "Navigator", "true_pose",
```

- [ ] **Step 4: Run the check**

Run: `.venv/bin/python scripts/check_navigation.py`
Expected:
```
grid ok
pgm ok
astar ok
spline ok
dock ok
profile ok
navigator ok
all ok
```

- [ ] **Step 5: Commit**

```bash
git add src/rlbot/navigate.py src/rlbot/__init__.py scripts/check_navigation.py
git commit -m "Add the navigator: play the schedule, replan on drift

Claude-Session: https://claude.ai/code/session_01CQtc6f3qAqzNU3TbMU9DkB"
```

---

### Task 10: Drive the routes in the sim

This is where the design meets the physics. Expect to iterate.

**Files:**
- Create: `scripts/navigate.py`

- [ ] **Step 1: Write the script**

Create `scripts/navigate.py`:

```python
"""Drive the robot to each table dock in the sim, on the schedule, and check it arrives.

    .venv/bin/python scripts/navigate.py                      # all 9 routes, headless
    .venv/bin/python scripts/navigate.py --route start-cubes
    .venv/bin/mjpython scripts/navigate.py --route start-cubes --view

The balancer and the DriveController run exactly as the ROS bridge runs them:
500 Hz physics, commands at 50 Hz.  The navigator gets the sim's true pose in
place of the SLAM estimate and replans from it.  A route passes if the robot
ends within 10 cm and 5 degrees of the dock, never falls, and never touches the
furniture.  It also reports how long the chassis spent tilted past the 2 degrees
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
                and self.err_xy <= ARRIVE_XY and self.err_yaw <= ARRIVE_YAW)


def routes():
    docks = {t.name.split("_")[1]: t.dock for t in TABLES}
    out = [(f"start-{n}", "start", n) for n in docks]
    out += [(f"{a}-{b}", f"dock_{a}", b) for a in docks for b in docks if a != b]
    return out


def furniture_contacts(model, data) -> set:
    """Names of room geoms the robot is touching.  Wheels on the floor do not count."""
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


def drive(name: str, keyframe: str, table: str, grid, view: bool = False) -> Result:
    bot = Balancer(ROOM, chassis="root", keyframe=keyframe)
    model, data = bot.model, bot.data
    radius = float(model.geom("wheel_left_collision").size[0])
    control = DriveController(radius, Gains.for_bracketbot())
    nav = Navigator(grid)
    goal = goal_for(table)
    result = Result(name)

    nav.go_to(goal, true_pose(data), data.time)
    if nav.traj is None:
        result.failed = nav.failed
        return result
    budget = nav.traj.duration * 3 + 15
    every = max(1, int(round(1 / (COMMAND_HZ * bot.dt))))
    streak, step = 0.0, 0

    window = (mujoco.viewer.launch_passive(model, data) if view else contextlib.nullcontext())
    with window as viewer:
        while data.time < budget and not nav.done and not nav.failed:
            tick = time.time()
            if step % every == 0:
                v, omega = nav.step(true_pose(data), data.time)
                control.command(v, omega, data.time)
            state = bot.step(*control(bot.state(), data.time, bot.dt))
            step += 1

            if bot.has_fallen(state):
                result.fell = True
                break
            result.touched |= furniture_contacts(model, data)
            tilt = math.acos(min(1.0, data.body("root").xmat.reshape(3, 3)[2, 2]))
            streak = streak + bot.dt if tilt > TILT_GATE else 0.0
            result.tilt_streak = max(result.tilt_streak, streak)

            if viewer is not None:
                viewer.sync()
                slack = bot.dt - (time.time() - tick)
                if slack > 0:
                    time.sleep(slack)

    x, y, yaw = true_pose(data)
    result.err_xy = math.hypot(x - goal[0], y - goal[1])
    result.err_yaw = math.degrees(abs(math.atan2(math.sin(yaw - goal[2]), math.cos(yaw - goal[2]))))
    result.seconds = float(data.time)
    result.replans = len(nav.replans)
    result.max_off = max((off for _, off in nav.replans), default=0.0)
    result.failed = nav.failed
    if not nav.done and result.failed is None and not result.fell:
        result.failed = f"ran out of time after {budget:.0f} s"
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--route", help="one route, e.g. start-cubes or ware-ball")
    ap.add_argument("--view", action="store_true", help="watch it (needs mjpython)")
    args = ap.parse_args()

    chosen = [r for r in routes() if not args.route or r[0] == args.route]
    if not chosen:
        ap.error(f"no route {args.route!r}; one of " + ", ".join(r[0] for r in routes()))
    grid = OccupancyGrid.from_room()

    print(f"  {'route':<12} {'arrive':>14}  {'time':>6}  {'replans':>7}  {'max off':>7}  {'tilt>2deg':>9}")
    failures = []
    for name, keyframe, table in chosen:
        r = drive(name, keyframe, table, grid, view=args.view)
        verdict = "ok  " if r.ok else "FAIL"
        why = (f"fell" if r.fell else ", ".join(sorted(r.touched)) if r.touched
               else r.failed or "")
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
```

- [ ] **Step 2: Run one route**

Run: `.venv/bin/python scripts/navigate.py --route start-cubes`
Expected: one line, `ok`, arrival under 10 cm and 5 deg, a handful of replans, and a `tilt>2deg` stretch. Takes about a minute of wall time.

- [ ] **Step 3: Watch it once**

Run: `.venv/bin/mjpython scripts/navigate.py --route cubes-ware --view`
Expected: the robot backs off the cubes table, turns in place, drives a curve to the ware table, and pulls in facing it. It must not clip the divider.

- [ ] **Step 4: Run all nine**

Run: `.venv/bin/python scripts/navigate.py`
Expected: nine `ok` lines and `9 routes arrived within 10 cm and 5 deg`, exit 0. Budget several minutes.

If routes fail, in this order:

1. **Arrival over 10 cm, few replans.** The open loop runs ahead or behind the robot. Lower `replan_every` to `1.0` in `Navigator`'s defaults, rerun.
2. **Arrival over 10 cm, many replans, max off small.** The final approach keeps missing sideways: the exit from the last replan started off-axis. Lower `off_path` to `0.05`, rerun.
3. **`fell` on a dock-to-dock route.** The turn in place or the reverse is too aggressive for the balancer. Lower `omega_max` to `0.2` and `alpha_max` to `0.2` in `Limits`, rerun; if it still falls, lower `a_max` to `0.05`.
4. **Touches the divider or pillar.** The curve is within margin but the balancer overshoots the bend. Raise `margin` in `Limits` to `0.08`, rerun `plan_path.py` then `navigate.py`.
5. **`ran out of time`.** The robot is oscillating between replans. Raise `off_path` to `0.15`.

Record what was changed and why in the commit message. If after these the nine routes still do not pass, stop and report the table: that is the evidence a tracking loop is needed, which is the next design, not this one.

- [ ] **Step 5: Commit**

```bash
git add scripts/navigate.py src/rlbot/navigate.py src/rlbot/planner.py
git commit -m "Drive the nine routes in the sim and check arrival

Claude-Session: https://claude.ai/code/session_01CQtc6f3qAqzNU3TbMU9DkB"
```

---

### Task 11: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Status row**

Replace line 22 (the `2. SLAM navigation` row) with:

```markdown
| 2. SLAM navigation | ROS 2 Jazzy / SLAM Toolbox mapping, map saving and localization restart pass in Ubuntu Docker. The robot plans a curved route to any table and drives it on the balancer in the sim, replanning from its pose. Not yet run on the SLAM pose in Docker. |
```

- [ ] **Step 2: What works**

After the `- **A room the robot actually fits in.**` bullet (line 30), insert:

```markdown
- **Driving to a table.** `scripts/navigate.py` plans a curved route from anywhere in the room to a table's docking pose and drives it on the balancing robot: A* on an occupancy grid for the corridor, a spline through it for the curve, a speed schedule under the drive limits, played open loop and replanned every couple of seconds from the robot's pose. All nine routes between the start and the three tables arrive within 10 cm and 5 degrees without touching the furniture. The grid comes from the known room today and from the saved SLAM map next; the code is the same either way.
```

- [ ] **Step 3: How to run**

After the `.venv/bin/python scripts/check_arm_clearance.py` line and its comment (line 73), insert:

```bash

# Plan the nine routes between the start pose and the three tables, draw them, check them.
.venv/bin/python scripts/plan_path.py

# Drive those routes in the sim on the balancer and check the robot arrives.
.venv/bin/python scripts/navigate.py
.venv/bin/mjpython scripts/navigate.py --route cubes-ware --view

# Unit checks for the grid, planner and navigator on small synthetic maps.
.venv/bin/python scripts/check_navigation.py
```

- [ ] **Step 4: Folder layout**

After the `scripts/check_arm_clearance.py` line (175), insert:

```
scripts/plan_path.py        Plans and draws the nine routes between the start and the table docks.
scripts/navigate.py         Drives those routes in the sim and checks arrival, falls and contacts.
scripts/check_navigation.py Unit checks for navmap, planner and navigate on small grids.
```

After the `src/rlbot/hybrid_ik.py` line (181), insert:

```
src/rlbot/navmap.py         The occupancy grid: from the known room, or from the map SLAM Toolbox saves.
src/rlbot/planner.py        A* + shortcut + spline for the route; the speed and yaw-rate schedule to drive it.
src/rlbot/navigate.py       Plays the schedule open loop and replans from the pose when it drifts.
```

- [ ] **Step 5: Next steps**

In "Plan for the next steps", Step 2, replace the two bullets

```markdown
- Add a path planner so the robot can drive to a target spot while it keeps its balance.
- Add a "dock at a table" move so the robot ends up in a good spot for the arms to reach.
```

with

```markdown
- Run the navigator on the SLAM pose inside Docker: `ros2 run rlbot_bridge navigate`.
- Union the known table tops into the SLAM map, or raise the lidar, so a lidar-built map does not show open floor under the tables.
- Check whether driving trips the bridge's 2 degree tilt gate often enough to starve SLAM of scans, and if so gate the command on localisation loss instead.
- Dock off a perceived table edge rather than the pose in `room.py`.
```

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "Document navigation in the README

Claude-Session: https://claude.ai/code/session_01CQtc6f3qAqzNU3TbMU9DkB"
```

---

### Task 12: The ROS node

Cannot be run on the Mac outside Docker. Write it, wire it, then hand-test in the image.

**Files:**
- Create: `ros2_ws/src/rlbot_bridge/rlbot_bridge/navigate.py`
- Modify: `ros2_ws/src/rlbot_bridge/setup.py`
- Modify: `README.md`

- [ ] **Step 1: Write the node**

Create `ros2_ws/src/rlbot_bridge/rlbot_bridge/navigate.py`:

```python
"""Drive the robot to a table on the SLAM pose.

    ros2 run rlbot_bridge navigate --ros-args -p use_sim_time:=true \
        -p map_yaml:=/artifacts/mapping_check_1/saved/map.yaml -p to:=cubes
    ros2 run rlbot_bridge navigate --ros-args -p use_sim_time:=true \
        -p map_yaml:=... -p to:="1.71 -1.10 0.0"

Runs beside `mapping.launch.py mode:=localization`.  Ten times a second it
looks up map -> base_footprint, hands it to the Navigator, and publishes what
comes back on /cmd_vel.  Exits 0 on arrival, 1 if the navigator gives up.

The map frame is taken to be the room's world frame, because the bridge spawns
the robot at the room's `start` keyframe and the map is built from there.  The
lidar cannot see table tops, so the known ones are unioned into the map.
"""

import math
import os
import sys
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

REPO = Path(os.environ.get("RLBOT_REPO", Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(REPO / "src"))

from rlbot.navigate import Navigator                  # noqa: E402
from rlbot.navmap import OccupancyGrid, table_tops    # noqa: E402
from rlbot.planner import goal_for                    # noqa: E402


class NavigateNode(Node):
    def __init__(self):
        super().__init__("rlbot_navigate")
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("to", "")
        map_yaml = self.get_parameter("map_yaml").value
        to = self.get_parameter("to").value.split()
        if not map_yaml or not to:
            raise ValueError("need -p map_yaml:=/path/map.yaml and -p to:=<table | x y yaw>")
        self.goal = goal_for(to[0]) if len(to) == 1 else tuple(float(v) for v in to)
        grid = OccupancyGrid.from_pgm(map_yaml, extra_boxes=table_tops())
        self.navigator = Navigator(grid)
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.timer = self.create_timer(0.1, self.tick)
        self.started = False
        self.finished = None
        self.get_logger().info(f"driving to {self.goal}")

    def pose(self):
        try:
            tf = self.tf.lookup_transform("map", "base_footprint", Time())
        except TransformException as error:
            self.get_logger().warning(f"no map->base_footprint yet: {error}",
                                      throttle_duration_sec=2.0)
            return None
        q = tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        return tf.transform.translation.x, tf.transform.translation.y, yaw

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        pose = self.pose()
        if not self.started:
            if pose is None:
                return
            traj = self.navigator.go_to(self.goal, pose, now)
            self.started = True
            if traj is not None:
                self.get_logger().info(f"planned a {traj.duration:.0f} s route from "
                                       f"({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f} deg)")
        v, omega = self.navigator.step(pose, now)
        message = Twist()
        message.linear.x, message.angular.z = float(v), float(omega)
        self.publisher.publish(message)
        if self.navigator.done:
            self.get_logger().info(f"arrived after {len(self.navigator.replans)} replans")
            self.finished = 0
        elif self.navigator.failed:
            self.get_logger().error(self.navigator.failed)
            self.finished = 1


def main():
    rclpy.init()
    node = None
    code = 1
    try:
        node = NavigateNode()
        while rclpy.ok() and node.finished is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        code = node.finished if node.finished is not None else 1
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.publisher.publish(Twist())
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register the console script**

In `ros2_ws/src/rlbot_bridge/setup.py`, change the `console_scripts` list to:

```python
    entry_points={"console_scripts": [
        "simulation = rlbot_bridge.simulation:main",
        "save_map = rlbot_bridge.map_session:main",
        "navigate = rlbot_bridge.navigate:main",
    ]},
```

- [ ] **Step 3: Syntax-check it locally**

Run: `.venv/bin/python -m py_compile ros2_ws/src/rlbot_bridge/rlbot_bridge/navigate.py && echo compiled`
Expected: `compiled`. (The imports need ROS; only the syntax is checked here.)

- [ ] **Step 4: README paragraph**

In `README.md`, in the "ROS 2 Jazzy mapping in Docker" section, after the paragraph that begins `The initial map-pose guess defaults to x/y/yaw = 0.`, insert three things in order. First this paragraph:

```markdown
With the stack in localization mode, drive to a table on the SLAM pose. It reads the saved map, unions in the known table tops the lidar cannot see, looks up `map -> base_footprint`, and publishes `/cmd_vel` until it arrives within 10 cm and 5 degrees:
```

Then this fenced bash block:

```bash
docker --context colima-rlbot exec -it rlbot-localize /opt/rlbot/docker/entrypoint.sh ros2 run rlbot_bridge navigate --ros-args -p use_sim_time:=true -p map_yaml:=/artifacts/room_1/map.yaml -p to:=cubes
```

Then this paragraph:

```markdown
`to` takes a table name (`ball`, `cubes`, `ware`) or `"x y yaw"`. It is the same navigator `scripts/navigate.py` runs locally on the sim's true pose.
```

- [ ] **Step 5: Hand-test in Docker**

Docker Desktop must already be running (start it yourself; launching it from a session revokes file access). On this Mac there is no colima, so drop `--context colima-rlbot` from the README commands.

```bash
docker build -t rlbot:jazzy .
docker run --rm -v "$PWD/out:/artifacts" rlbot:jazzy python scripts/check_ros_mapping.py --output /artifacts/mapping_check_1
```

Expected: ends with `ROS mapping, loop traversal, save and localization restart passed.` and leaves `out/mapping_check_1/saved/map.yaml`. Several minutes on the first build.

Then, in one terminal:

```bash
docker run --rm -it --name rlbot-localize -v "$PWD/out:/artifacts" rlbot:jazzy ros2 launch rlbot_bridge mapping.launch.py mode:=localization map_file:=/artifacts/mapping_check_1/saved/map
```

and in another:

```bash
docker exec -it rlbot-localize /opt/rlbot/docker/entrypoint.sh ros2 run rlbot_bridge navigate --ros-args -p use_sim_time:=true -p map_yaml:=/artifacts/mapping_check_1/saved/map.yaml -p to:=cubes
```

Expected: `driving to (-0.2, -1.41, -1.57...)`, `planned a NN s route from (0.00, 0.00, 0 deg)`, then after a minute or two `arrived after N replans`, exit 0. In the first terminal the bridge logs nothing new; if it logs `robot fell`, the tilt gate or the limits are the problem, see the spec's handoff section.

If the node exits 1 with `start cell ... is occupied`, the SLAM map's unknown border is inflating over the start: open `out/mapping_check_1/saved/map.pgm` and check the robot's start is on white, not grey.

- [ ] **Step 6: Commit**

```bash
git add ros2_ws/src/rlbot_bridge/rlbot_bridge/navigate.py ros2_ws/src/rlbot_bridge/setup.py README.md
git commit -m "Add a ROS node that drives the navigator on the SLAM pose

Claude-Session: https://claude.ai/code/session_01CQtc6f3qAqzNU3TbMU9DkB"
```

---

### Task 13: Remove the planning docs and push

The user wants the spec and plan gone once the work is in. The README is the record.

**Files:**
- Delete: `docs/superpowers/specs/2026-09-12-path-planning-navigation-design.md`
- Delete: `docs/superpowers/plans/2026-09-12-path-planning-navigation.md`

- [ ] **Step 1: Run everything once more**

```bash
.venv/bin/python scripts/check_navigation.py && .venv/bin/python scripts/plan_path.py --quiet && .venv/bin/python scripts/navigate.py && .venv/bin/python scripts/evaluate.py && .venv/bin/python scripts/check_slam_inputs.py
```

Expected: every script exits 0. `evaluate.py` and `check_slam_inputs.py` prove the balancer and Rababb's checks still pass with the shared grid change.

- [ ] **Step 2: Delete the docs**

```bash
git rm -r docs/superpowers
rmdir docs 2>/dev/null || true
git commit -m "Remove the navigation planning docs now the work is in

Claude-Session: https://claude.ai/code/session_01CQtc6f3qAqzNU3TbMU9DkB"
git push
```

Note `docs/slam_cv_plan.md` is Rababb's and stays; `rmdir docs` only succeeds if the directory is empty, which it will not be.

- [ ] **Step 3: Hand off**

The branch is based on `feat/slam-cv-camera-inspection`, which is not merged. A PR against `main` would carry Rababb's commit; open it against `feat/slam-cv-camera-inspection` instead, or wait for theirs to merge and rebase. Tell the user which, and tell Karan and Rababb about the `control.py` and lidar-module conflict between PR #5 and the SLAM branch.
