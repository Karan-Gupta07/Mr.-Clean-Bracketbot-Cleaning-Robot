"""What is in the room: three tables, and the things on them.

Shared by the builder that writes the MJCF and by anything that has to reason
about the objects afterwards - where to dock, where to grasp, how wide each
thing is - so the layout is stated once.

Every dimension answers to a measurement taken off the robot in
`models/bracketbot.xml`:

  * the gripper opens to 195 mm at the fingertips, so nothing here is wider than
    150 mm across its grasp axis;
  * a top-down grasp runs out about 0.36 m from the mast axis, so the robot
    docks 0.24 m off each table edge and objects sit 0.09 m in from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

# ---- the room ------------------------------------------------------------
ROOM_X, ROOM_Y = 6.0, 4.5      # m, inside faces of the walls
WALL_H, WALL_T = 2.5, 0.10

TABLE_H = 0.70                 # m, top surface - the hands rest at 0.715 m
TABLE_L, TABLE_W = 1.00, 0.60  # m, top
TOP_T, LEG = 0.04, 0.05        # top thickness, square leg

DOCK_GAP = 0.24                # m, mast axis to table edge when parked at it
EDGE_INSET = 0.09              # m, table edge to the near row of objects
# 0.24 + 0.09 = 0.33 m from the mast axis to an object, against a top-down
# envelope that runs out around 0.36 m.  The chassis is 0.094 m deep, so the
# robot still parks with 0.15 m of daylight between itself and the table.

MAX_GRASP_WIDTH = 0.150        # m, across the grasp axis (gripper opens 0.195)


@dataclass
class Item:
    """Something on a table.  `width` is the span the fingers have to close over."""

    name: str
    kind: str
    at: tuple[float, float]    # m, offset along (table length, table depth) from centre
    width: float
    mass: float
    rgba: tuple[float, float, float, float]
    size: dict = field(default_factory=dict)


@dataclass
class Table:
    name: str
    centre: tuple[float, float]
    yaw: float                 # rad; the near edge faces the robot
    items: list[Item]

    @property
    def frame(self):
        """(origin, length axis, depth axis) in world - depth points at the robot."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        along = np.array([c, s, 0.0])          # table's long axis
        toward = np.array([-s, c, 0.0])        # from the table's centre to the robot
        return np.array([*self.centre, 0.0]), along, toward

    @property
    def dock(self):
        """Where the robot parks: (x, y, yaw), facing the table."""
        origin, _, toward = self.frame
        p = origin + toward * (TABLE_W / 2 + DOCK_GAP)
        yaw = math.atan2(-toward[1], -toward[0])   # +x points at the table
        return float(p[0]), float(p[1]), yaw

    def place(self, item: Item):
        """World position of an item's base, from its (along, inset) offset."""
        origin, along, toward = self.frame
        inset = TABLE_W / 2 - EDGE_INSET - item.at[1]
        return origin + along * item.at[0] + toward * inset + np.array([0, 0, TABLE_H])


def ball(name, at, d=0.070, mass=0.12, rgba=(0.85, 0.25, 0.2, 1)):
    return Item(name, "ball", at, d, mass, rgba, {"r": d / 2})


def cube(name, at, s, mass, rgba):
    return Item(name, "cube", at, s, mass, rgba, {"s": s})


def crate(name, at, mass=0.35, rgba=(0.55, 0.42, 0.28, 1)):
    """The open box each table's objects are meant to end up in.  Its outside is
    140 mm across, so the gripper can also straddle and carry the crate itself."""
    return Item(name, "crate", at, 0.140, mass, rgba,
                {"l": 0.180, "w": 0.140, "h": 0.090, "t": 0.008})


def bowl(name, at, mass=0.25, rgba=(0.92, 0.92, 0.88, 1)):
    return Item(name, "bowl", at, 0.150, mass, rgba,
                {"r_base": 0.045, "r_rim": 0.075, "h": 0.055, "t": 0.005})


def plate(name, at, mass=0.20, rgba=(0.80, 0.84, 0.90, 1)):
    """A rim, not a disc: a flat disc gives a parallel gripper nothing to hold."""
    return Item(name, "plate", at, 0.145, mass, rgba,
                {"r_base": 0.056, "r_rim": 0.0725, "h": 0.018, "t": 0.005})


TABLES = [
    Table("table_ball", (2.25, -1.10), math.radians(90), [
        ball("ball", (-0.20, 0.0)),
        crate("crate_ball", (0.22, 0.0)),
    ]),
    Table("table_cubes", (-0.20, -1.95), math.radians(0), [
        cube("cube_s", (-0.28, 0.03), 0.035, 0.04, (0.90, 0.55, 0.15, 1)),
        cube("cube_m", (-0.16, -0.03), 0.045, 0.07, (0.25, 0.60, 0.85, 1)),
        cube("cube_l", (-0.04, 0.03), 0.055, 0.11, (0.35, 0.70, 0.35, 1)),
        cube("cube_xl", (0.08, -0.03), 0.065, 0.16, (0.75, 0.30, 0.65, 1)),
        crate("crate_cubes", (0.27, 0.0)),
    ]),
    Table("table_ware", (-2.25, 0.90), math.radians(-90), [
        bowl("bowl", (-0.26, 0.0)),
        plate("plate", (-0.03, 0.0)),
        crate("crate_ware", (0.24, 0.0)),
    ]),
]

# ---- fixed furniture, so the map is not four bare walls -------------------
PILLAR = (1.15, 1.30, 0.14)        # x, y, radius
DIVIDER = (0.60, 1.55, 0.06, 0.70)  # x, y centre, half-thickness, half-length


# ---- geometry helpers ----------------------------------------------------
def cone_shell(r_base, r_rim, z0, z1, thickness, segments=24):
    """A ring of tilted slabs approximating an open cone - a bowl, or a plate rim.

    MuJoCo primitives are all convex, so a vessel has to be built out of its own
    wall.  The slabs overlap by 12%, which seals the wall against a 35 mm cube
    squeezing between two of them without making the rim look like a crown.
    """
    slant = math.hypot(r_rim - r_base, z1 - z0)
    tilt = math.atan2(r_rim - r_base, z1 - z0)
    r_mid, z_mid = (r_base + r_rim) / 2, (z0 + z1) / 2
    half_w = 1.12 * math.pi * r_mid / segments

    out = []
    for i in range(segments):
        th = 2 * math.pi * i / segments
        quat_z = np.array([math.cos(th / 2), 0, 0, math.sin(th / 2)])
        quat_tilt = np.array([math.cos(tilt / 2), math.sin(tilt / 2), 0, 0])
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, quat_z, quat_tilt)
        out.append({
            "pos": (r_mid * math.cos(th), r_mid * math.sin(th), z_mid),
            "quat": tuple(quat),
            "size": (thickness / 2, half_w, slant / 2),
        })
    return out


def item_geoms(item: Item):
    """(type, size, pos, quat, is_the_grasp_surface) for one item, base at z=0."""
    s = item.size
    if item.kind == "ball":
        return [("sphere", (s["r"],), (0, 0, s["r"]), None)]
    if item.kind == "cube":
        h = s["s"] / 2
        return [("box", (h, h, h), (0, 0, h), None)]
    if item.kind == "crate":
        l, w, h, t = s["l"] / 2, s["w"] / 2, s["h"], s["t"] / 2
        return [
            ("box", (l, w, t), (0, 0, t), None),
            ("box", (l, t, h / 2), (0, w - t, h / 2), None),
            ("box", (l, t, h / 2), (0, -(w - t), h / 2), None),
            ("box", (t, w - 2 * t, h / 2), (l - t, 0, h / 2), None),
            ("box", (t, w - 2 * t, h / 2), (-(l - t), 0, h / 2), None),
        ]
    if item.kind in ("bowl", "plate"):
        base_h = s["t"]
        geoms = [("cylinder", (s["r_base"], base_h / 2), (0, 0, base_h / 2), None)]
        for slab in cone_shell(s["r_base"], s["r_rim"], base_h, s["h"], s["t"]):
            geoms.append(("box", slab["size"], slab["pos"], slab["quat"]))
        return geoms
    raise ValueError(item.kind)


def lowest_point(geoms) -> float:
    """The z of the lowest point of an item's geoms, in its own frame.

    A bowl's wall slabs are tilted, so their corners hang below the flat base
    they are built on.  Spawning the body with that corner exactly on the table
    top starts the sim with a 3 mm penetration, which the solver then pushes out
    - the crockery visibly hops on the first frame.  Measure instead.
    """
    lo = 0.0
    for kind, size, pos, quat in geoms:
        if kind == "sphere":
            lo = min(lo, pos[2] - size[0])
        elif kind == "cylinder":
            lo = min(lo, pos[2] - size[1])
        else:                                   # box, possibly rotated
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, np.asarray(quat if quat is not None
                                                else [1.0, 0, 0, 0]))
            reach = sum(abs(rot.reshape(3, 3)[2, i]) * size[i] for i in range(3))
            lo = min(lo, pos[2] - reach)
    return lo


GRASP_YAWS = [math.radians(a) for a in (0, 30, 60, 90, 120, 150)]


FINGERTIP_CLEARANCE = 0.008    # m the fingertips stop above the table


def grasp_pose(item: Item, table: Table):
    """Where the grip site has to be to take this item.

    The grip site is the *fingertip* midpoint, and the fingers are 134 mm blades
    hinged at the hand.  Aim the site at an object's waist and the tips close on
    it at their very ends, where the blades are thinnest and furthest from their
    pivots: a 55 mm cube shot 200 mm sideways out of the jaws.  Aim it just above
    the table instead and the object sits back along the pads, between the flats,
    where closing squeezes it rather than flicking it.

    Everything in this room is at most 90 mm tall, so a fixed height clears the
    table for all of it.
    """
    return table.place(item) + np.array([0, 0, FINGERTIP_CLEARANCE])
