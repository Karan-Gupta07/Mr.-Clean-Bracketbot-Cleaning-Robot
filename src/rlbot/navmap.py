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
    footprint, 1.61 m tall.  Returns (radius, height, half-extents).
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


def _read_yaml(path) -> dict:
    """The handful of `key: value` lines a map_server YAML has.

    ponytail: flat keys plus one list is all map_saver ever writes; reach for
    a real YAML parser the day it does not parse.
    """
    meta = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' not in line:
            raise ValueError(f'{path}: expected key: value in map metadata')
        key, value = (part.strip() for part in line.split(':', 1))
        if value.startswith(('\'', '"')):
            end = value.find(value[0], 1)
            if end < 0 or (value[end + 1:].strip() and not value[end + 1:].lstrip().startswith('#')):
                raise ValueError(f'{path}: invalid quoted {key}')
            meta[key] = value[1:end]
        else:
            value = value.split('#', 1)[0].strip()
            if value.startswith('['):
                if not value.endswith(']'):
                    raise ValueError(f'{path}: invalid {key} list')
                try:
                    meta[key] = [float(v) for v in value[1:-1].split(',')]
                except ValueError as exc:
                    raise ValueError(f'{path}: invalid {key} list') from exc
            else:
                meta[key] = value
    return meta


def _read_pgm(path) -> np.ndarray:
    """A binary (P5) or ASCII (P2) PGM as a float array scaled to 0..255."""
    raw = Path(path).read_bytes()
    i = 0

    def token():
        nonlocal i
        while i < len(raw):
            if raw[i:i+1].isspace():
                i += 1
            elif raw[i:i+1] == b'#':
                while i < len(raw) and raw[i:i+1] not in (b'\r', b'\n'):
                    i += 1
            else:
                break
        start = i
        while i < len(raw) and not raw[i:i+1].isspace() and raw[i:i+1] != b'#':
            i += 1
        return raw[start:i]

    def integer():
        value = token()
        if not value.isdigit():
            raise ValueError(f'{path}: expected a decimal PGM integer')
        return int(value)

    magic = token()
    if magic not in (b'P5', b'P2'):
        raise ValueError(f'{path}: not a PGM (magic {magic!r})')
    try:
        width, height, maxval = (integer() for _ in range(3))
    except ValueError as exc:
        raise ValueError(f'{path}: invalid or truncated PGM header') from exc
    if width <= 0 or height <= 0 or not 1 <= maxval <= 65535:
        raise ValueError(f'{path}: PGM dimensions must be positive and maxval must be in 1..65535')
    count = width * height
    if magic == b'P5':
        if i >= len(raw) or not raw[i:i+1].isspace():
            raise ValueError(f'{path}: missing PGM raster separator')
        dtype = np.dtype(np.uint8 if maxval < 256 else '>u2')
        size = count * dtype.itemsize
        i += 2 if raw[i:i+2] == b'\r\n' and len(raw) - i >= size + 2 else 1
        if len(raw) - i < size:
            raise ValueError(f'{path}: PGM raster size does not match its dimensions')
        pixels = np.frombuffer(raw, dtype=dtype, count=count, offset=i)
    else:
        try:
            pixels = np.array([integer() for _ in range(count)], dtype=np.int64)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f'{path}: invalid or truncated PGM pixels') from exc
        if token():
            raise ValueError(f'{path}: too many PGM pixels')
    if ((pixels < 0) | (pixels > maxval)).any():
        raise ValueError(f'{path}: PGM pixel outside 0..{maxval}')
    return pixels.reshape(height, width).astype(float) * 255.0 / maxval


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
            cylinder = model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
            reach = float(size[1]) if cylinder else float(np.abs(rot[2]) @ size[:3])
            if pos[2] - reach > height or pos[2] + reach < 0:
                continue
            if cylinder:
                inside = np.hypot(gx - pos[0], gy - pos[1]) <= size[0] + 1e-9
            else:                                   # box, yaw in the plane
                dx, dy = gx - pos[0], gy - pos[1]
                local_x = rot[0, 0] * dx + rot[1, 0] * dy
                local_y = rot[0, 1] * dx + rot[1, 1] * dy
                inside = ((np.abs(local_x) <= size[0] + 1e-9)
                          & (np.abs(local_y) <= size[1] + 1e-9))
            occupied |= inside
        return cls(occupied, resolution, (float(xs[0]), float(ys[0])))

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
        for key in ('image', 'resolution', 'origin'):
            if key not in meta:
                raise ValueError(f'{yaml_path}: missing {key} in map metadata')
        if not isinstance(meta['image'], str) or not meta['image']:
            raise ValueError(f'{yaml_path}: image must be a nonempty path')
        if meta.get('mode', 'trinary') != 'trinary':
            raise ValueError(f'{yaml_path}: only trinary map mode is supported')

        def number(key, default=None):
            try:
                value = float(meta.get(key, default))
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{yaml_path}: invalid {key}') from exc
            if not math.isfinite(value):
                raise ValueError(f'{yaml_path}: {key} must be finite')
            return value

        resolution = number('resolution')
        if resolution <= 0:
            raise ValueError(f'{yaml_path}: resolution must be positive')
        origin = meta['origin']
        if not isinstance(origin, list) or len(origin) != 3 or not all(map(math.isfinite, origin)):
            raise ValueError(f'{yaml_path}: origin must be a finite [x, y, yaw] list')
        ox, oy, yaw = origin
        if yaw != 0:
            raise ValueError(f'{yaml_path}: nonzero origin yaw is unsupported by an axis-aligned grid')
        negate = meta.get('negate', '0')
        if negate not in ('0', '1'):
            raise ValueError(f'{yaml_path}: negate must be 0 or 1')
        free_thresh = number('free_thresh', 0.25)
        occupied_thresh = number('occupied_thresh', 0.65)
        if not 0 <= free_thresh < occupied_thresh <= 1:
            raise ValueError(f'{yaml_path}: thresholds must satisfy 0 <= free_thresh < occupied_thresh <= 1')

        image = _read_pgm(yaml_path.parent / meta['image'])     # (rows, cols), row 0 at the top
        shade = image if int(negate) else 255.0 - image
        occ = shade / 255.0
        on_threshold = np.isclose(occ, free_thresh, rtol=0.0, atol=np.finfo(float).eps)
        free = (occ < free_thresh) & ~on_threshold & (image != 205)
        occupied = ~free.T[:, ::-1]        # (cols, rows), row axis flipped so index 0 is min y
        grid = cls(np.ascontiguousarray(occupied), resolution, (ox + resolution / 2, oy + resolution / 2))
        for box in extra_boxes:
            grid.fill_box(*box)
        return grid

    def inflate(self, radius: float) -> "OccupancyGrid":
        """Grow every occupied cell by `radius`.

        Beyond the edge of the grid counts as occupied, so the border inflates
        inward too: a SLAM map ends where the lidar stopped seeing, not where
        the floor does.
        """
        r = int(math.ceil(radius / self.resolution))
        offsets = [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
                   if math.hypot(di, dj) * self.resolution <= radius + 1e-9]
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
        self.occupied |= (np.abs(local_x) <= half_x + 1e-9) & (np.abs(local_y) <= half_y + 1e-9)

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
        Test the closed segment against cell boxes, including corner contact."""
        p, q = np.asarray(p, float)[:2], np.asarray(q, float)[:2]
        lower = np.asarray(self.origin) - self.resolution / 2
        upper = lower + np.asarray(self.occupied.shape) * self.resolution
        if (not np.isfinite((p, q)).all()
                or (np.minimum(p, q) < lower - 1e-9).any()
                or (np.maximum(p, q) > upper + 1e-9).any()):
            return False

        cell_lower = np.argwhere(self.occupied) * self.resolution + lower - 1e-9
        cell_upper = cell_lower + self.resolution + 2e-9
        enter, leave = np.zeros(len(cell_lower)), np.ones(len(cell_lower))
        for axis in range(2):
            delta = q[axis] - p[axis]
            if delta == 0:
                enter[(p[axis] < cell_lower[:, axis])
                      | (p[axis] > cell_upper[:, axis])] = np.inf
            else:
                t0 = (cell_lower[:, axis] - p[axis]) / delta
                t1 = (cell_upper[:, axis] - p[axis]) / delta
                enter = np.maximum(enter, np.minimum(t0, t1))
                leave = np.minimum(leave, np.maximum(t0, t1))
        return not bool((enter <= leave).any())

    def rect_free(self, x, y, yaw, half_x, half_y) -> bool:
        """True if the rectangle stays in bounds and covers no occupied cell centre.

        The robot's own footprint at a pose, on the raw grid.  This is what
        lets it park 0.24 m from a table when its circle would want 0.26.
        """
        c, s = math.cos(yaw), math.sin(yaw)
        extent = np.array((abs(c) * half_x + abs(s) * half_y,
                           abs(s) * half_x + abs(c) * half_y))
        centre = np.array((x, y))
        lower = np.asarray(self.origin) - self.resolution / 2
        upper = lower + np.asarray(self.occupied.shape) * self.resolution
        if ((centre - extent < lower - 1e-9).any()
                or (centre + extent > upper + 1e-9).any()):
            return False

        gx, gy = self._centres()
        dx, dy = gx - x, gy - y
        local_x, local_y = c * dx + s * dy, -s * dx + c * dy
        inside = (np.abs(local_x) <= half_x + 1e-9) & (np.abs(local_y) <= half_y + 1e-9)
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
