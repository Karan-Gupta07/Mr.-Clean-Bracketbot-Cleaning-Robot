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
