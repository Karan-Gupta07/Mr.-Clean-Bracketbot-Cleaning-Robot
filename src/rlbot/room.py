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

# What the hand can actually hold, measured by closing it on test blocks rather
# than read off the fingertips.  The fingertips part by 195 mm, but the blades
# splay as they open: past about half travel the two gripping faces stop facing
# each other, and the usable jaw is 40 to 60 mm.  Everything on these tables is
# sized to that.
MAX_GRASP_WIDTH = 0.060        # m, across the grasp axis


# Wrist angles to try when reaching for something, relative to the table, best
# first.  Across the table beats along it, and not by a little: closing the jaws
# along the robot's own forward axis fails on objects that the same hand at the
# same spot holds easily turned 90 degrees.  The arm reaches the table from one
# side, so at 0 degrees the wrist is folded back on itself and the blades come
# together at a worse angle than the pad fit assumes.
GRASP_YAWS = tuple(math.radians(a) for a in (90, 60, 120, 30, 150, 0))
SQUARE_YAWS = (math.pi / 2, 0.0)


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

    @property
    def yaws(self):
        """Wrist angles worth trying, relative to the table.

        `width` is measured across a face.  Close on a box across its diagonal
        and the jaws meet 1.4 times that - 102 mm on a 72 mm cube, past what the
        hand opens - so square objects only get square approaches.  Round ones
        do not care, and get the full sweep.
        """
        return SQUARE_YAWS if self.kind in ("cube", "crate") else GRASP_YAWS

    @property
    def fixed(self) -> bool:
        """Furniture is bolted down.  A crate light enough for the solver to
        shove is a crate the arm nudges out from under the object it is about
        to drop, and then the placement misses something that was in the right
        place when it was planned."""
        return self.kind == "crate"

    @property
    def graspable(self) -> bool:
        """Crates are furniture: the robot puts things in them, not carries
        them.  Everything else on a table is meant to be picked up."""
        return self.kind != "crate"

    @property
    def height(self) -> float:
        if self.kind == "ball":
            return 2 * self.size["r"]
        if self.kind == "cube":
            return self.size["s"]
        return self.size["h"]


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


def ball(name, at, d=0.055, mass=0.09, rgba=(0.85, 0.25, 0.2, 1)):
    """A ball rolls away from a gripper closing on it, so this one is rubber:
    rolling friction, not the frictionless marble a bare sphere geom would be."""
    return Item(name, "ball", at, d, mass, rgba, {"r": d / 2, "condim": 6,
                                                  "friction": (1.2, 0.05, 0.02)})


def cube(name, at, s, mass, rgba):
    return Item(name, "cube", at, s, mass, rgba, {"s": s})


def crate(name, at, mass=0.40, rgba=(0.55, 0.42, 0.28, 1)):
    """The open box each table's objects are meant to end up in.

    Sized to what has to go in it, not to what the hand can carry: 208 x 148 mm
    of clear interior takes any of these objects with room to miss by a couple
    of centimetres.  An earlier version was 55 mm across the outside so the
    gripper could straddle the crate itself - which left a 43 mm interior, too
    narrow for three of the four cubes meant to go in it.  The crate is
    furniture; it does not need to be pickable.
    """
    return Item(name, "crate", at, 0.220, mass, rgba,
                {"l": 0.220, "w": 0.160, "h": 0.080, "t": 0.006})


def bowl(name, at, mass=0.14, rgba=(0.92, 0.92, 0.88, 1)):
    """Tapered, and 56 mm tall because that is what the hand can hold.

    At 50 mm - the obvious dinner-service proportion - the pads take it 38 mm up
    instead of 43, and it is picked up every time and dropped every time.  The
    band either side of 56 mm is narrow: 54, 55 and 58 mm all fail.
    """
    return Item(name, "bowl", at, 0.058, mass, rgba,
                {"r_base": 0.020, "r_rim": 0.029, "h": 0.056, "t": 0.004})


def cup(name, at, mass=0.13, rgba=(0.80, 0.84, 0.90, 1)):
    """Crockery the hand can actually take.

    Two shapes were tried here and dropped.  A plate is 26 mm tall, and the pads
    reach 22 mm below the middle of the jaw, so closing on a plate means closing
    on the table.  A straight-sided mug is worse: a tall, round, thin-walled
    tube touches two flat pads at two points on a curve, and rolled out of the
    jaw on every carry.  Grip heights from 24 to 58 mm, four taper-and-height
    combinations, carries from 2 to 8 seconds and more grip force all failed it.

    A tapered cup works, because the pads close under the flare rather than on a
    parallel wall.  These proportions are the ones that survived the sweep.
    """
    return Item(name, "cup", at, 0.056, mass, rgba,
                {"r_base": 0.019, "r_rim": 0.028, "h": 0.055, "t": 0.004})


TABLES = [
    Table("table_ball", (2.25, -1.10), math.radians(90), [
        ball("ball", (-0.24, 0.0)),
        crate("crate_ball", (0.00, 0.0)),
    ]),
    Table("table_cubes", (-0.20, -1.95), math.radians(0), [
        # Sizes and positions are both measured, not chosen.  Sweeping a cube
        # from 54 to 58 mm across four positions: 54 and 55 mm are picked at
        # some spots and not others, 56, 57 and 58 mm are picked and crated at
        # every one of them.  Sweeping position: nothing within 0.12 m of the
        # centreline works at all, which is fine, because that is where the
        # crate is.  So: 56-58 mm, out at +-0.14 and +-0.26 m.
        cube("cube_s", (-0.26, 0.0), 0.056, 0.11, (0.90, 0.55, 0.15, 1)),
        cube("cube_m", (-0.14, 0.0), 0.057, 0.12, (0.25, 0.60, 0.85, 1)),
        cube("cube_l", (0.14, 0.0), 0.058, 0.13, (0.35, 0.70, 0.35, 1)),
        cube("cube_xl", (0.26, 0.0), 0.057, 0.12, (0.75, 0.30, 0.65, 1)),
        crate("crate_cubes", (0.00, 0.0)),
    ]),
    Table("table_ware", (-2.25, 0.90), math.radians(-90), [
        bowl("bowl", (-0.22, 0.0)),
        # 0.22 m out, not 0.14.  Sat next to the crate it is picked up fine on
        # an empty table and not at all once the bowl is in the crate: the hand
        # has to come down 30 mm from a crate wall that now has something
        # standing in it.  Whether an object is reachable depends on what has
        # already been put away.
        cup("cup", (0.22, 0.0)),
        crate("crate_ware", (0.00, 0.0)),
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
    if item.kind in ("bowl", "cup"):
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


PAD_REACH = 0.022      # m the pads extend below the jaw centre, plus a margin
RIM_GRIP = 0.012       # m below the rim to take a bowl or a mug


def grasp_pose(item: Item, table: Table):
    """Where the jaws have to be to take this item.

    Two heights fight here.  The pads reach PAD_REACH below the middle of the
    jaw, so any lower and they close on the table before they close on the
    object.  Any higher and the jaw is above a short object altogether.

    Above that floor, where to grip depends on the shape:

      * A cube or a ball: halfway up, which is where closing squeezes it rather
        than levering it over.
      * A bowl or a mug: just under the rim.  These are tapered, and `width` is
        measured across the rim - grip them at their waist and the jaw opens for
        a diameter the object does not have there.  It is also where a tapered
        wall gives the pads a lip to close under, which is the difference
        between carrying a mug and knocking it over.
    """
    return table.place(item) + np.array([0, 0, max(PAD_REACH, item.height / 2)])
