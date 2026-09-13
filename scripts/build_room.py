"""Write the manipulation room: three tables of things to pick up, and walls to map.

    .venv/bin/python scripts/build_room.py

The layout itself lives in `src/rlbot/room.py`, so the grasp checker and any
controller can read the same table of what is where.  This script turns it into
MJCF and then argues with it: it solves IK for a top-down grasp on every object
from its table's docking pose, with both arms and a spread of wrist angles.  It
writes `models/room.xml` and `models/room_scene.xml` first, then runs the
clearance check and the arm-reach check, and exits nonzero if either fails,
with the files already written.

That is a kinematic test, not a proof of grasp - `scripts/check_grasp.py` closes
the fingers on each object and lifts it.

Output: models/room_scene.xml (committed, so running this is optional).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlbot.arm import ArmIK, down_quat                            # noqa: E402
from rlbot.navmap import MARGIN, OccupancyGrid, robot_footprint   # noqa: E402
from rlbot.room import (                                          # noqa: E402
    DIVIDER, Item, LEG, MAX_GRASP_WIDTH, PILLAR, ROOM_X, ROOM_Y,
    TABLE_H, TABLE_L, TABLE_W, TABLES, TOP_T, Table, WALL_H, WALL_T,
    grasp_pose, item_geoms, lowest_point,
)

ROOM_ONLY = REPO / "models" / "room.xml"          # the environment by itself
OUT = REPO / "models" / "room_scene.xml"          # the same room with the robot in it


# ---- XML emission --------------------------------------------------------
def fmt(vals):
    return " ".join(f"{v:.6g}" for v in vals)


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


def xml_item(item: Item, table: Table, indent="    "):
    geoms = item_geoms(item)
    pos = table.place(item) - np.array([0, 0, lowest_point(geoms)])
    # mass is the item's, shared out by volume-free equal split: these are thin
    # shells, and what matters for a grasp is the total the fingers have to hold
    each = item.mass / len(geoms)
    lines = [f'{indent}<body name="{item.name}" pos="{fmt(pos)}">',
             f'{indent}  <freejoint/>']
    for i, (kind, size, gpos, quat) in enumerate(geoms):
        q = f' quat="{fmt(quat)}"' if quat is not None else ""
        lines.append(
            f'{indent}  <geom name="{item.name}_{i}" type="{kind}" '
            f'size="{fmt(size)}" pos="{fmt(gpos)}"{q} mass="{each:.6g}" '
            f'condim="{item.size.get("condim", 4)}" '
            f'friction="{fmt(item.size.get("friction", (1, 0.02, 0.001)))}" '
            f'solimp="0.95 0.99 0.001" material="mat_{item.name}"/>')
    lines.append(f"{indent}</body>")
    return "\n".join(lines)


def xml_table(table: Table, indent="    "):
    origin, _, _ = table.frame
    half_l, half_w = TABLE_L / 2, TABLE_W / 2
    top_z = TABLE_H - TOP_T / 2
    quat = (math.cos(table.yaw / 2), 0, 0, math.sin(table.yaw / 2))
    lines = [f'{indent}<body name="{table.name}" pos="{fmt(origin)}" '
             f'quat="{fmt(quat)}">',
             f'{indent}  <geom name="{table.name}_top" type="box" '
             f'size="{fmt((half_l, half_w, TOP_T / 2))}" pos="0 0 {top_z:.4f}" '
             f'material="mat_wood" friction="1 0.01 0.001" condim="4"/>']
    inset = LEG / 2 + 0.03
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = sx * (half_l - inset), sy * (half_w - inset)
            lines.append(
                f'{indent}  <geom name="{table.name}_leg{sx}{sy}" type="box" '
                f'size="{fmt((LEG / 2, LEG / 2, (TABLE_H - TOP_T) / 2))}" '
                f'pos="{x:.4f} {y:.4f} {(TABLE_H - TOP_T) / 2:.4f}" '
                f'material="mat_wood"/>')
    lines.append(f"{indent}</body>")
    return "\n".join(lines)


def xml_walls(indent="    "):
    hx, hy = ROOM_X / 2, ROOM_Y / 2
    t, h = WALL_T / 2, WALL_H / 2
    walls = [
        ("wall_px", (hx + t, 0, h), (t, hy + 2 * t, h), "mat_wall_a"),
        ("wall_nx", (-hx - t, 0, h), (t, hy + 2 * t, h), "mat_wall_b"),
        ("wall_py", (0, hy + t, h), (hx + 2 * t, t, h), "mat_wall_c"),
        ("wall_ny", (0, -hy - t, h), (hx + 2 * t, t, h), "mat_wall_d"),
    ]
    lines = []
    for name, pos, size, mat in walls:
        lines.append(f'{indent}<geom name="{name}" type="box" size="{fmt(size)}" '
                     f'pos="{fmt(pos)}" material="{mat}"/>')
    px, py, pr = PILLAR
    lines.append(f'{indent}<geom name="pillar" type="cylinder" '
                 f'size="{pr:.4g} {WALL_H / 2:.4g}" pos="{px} {py} {WALL_H / 2}" '
                 f'material="mat_wall_b"/>')
    dx, dy, dt, dl = DIVIDER
    lines.append(f'{indent}<geom name="divider" type="box" '
                 f'size="{dt:.4g} {dl:.4g} {WALL_H / 2:.4g}" '
                 f'pos="{dx} {dy} {WALL_H / 2}" material="mat_wall_c"/>')
    return "\n".join(lines)


def materials():
    mats = ['    <texture name="tex_floor" type="2d" builtin="checker" '
            'rgb1="0.38 0.40 0.44" rgb2="0.30 0.32 0.36" width="512" height="512"/>',
            '    <material name="mat_floor" texture="tex_floor" texuniform="true" '
            'texrepeat="1 1" reflectance="0.05"/>',
            '    <material name="mat_wood" rgba="0.62 0.46 0.31 1" specular="0.2"/>']
    # distinct wall shades: a visual SLAM front end needs the four walls to be
    # tellable apart, and a lidar one needs the corners between them
    for tag, rgb in (("a", "0.72 0.74 0.78"), ("b", "0.66 0.70 0.75"),
                     ("c", "0.75 0.72 0.68"), ("d", "0.70 0.73 0.70")):
        mats.append(f'    <texture name="tex_wall_{tag}" type="2d" builtin="checker" '
                    f'rgb1="{rgb}" rgb2="0.58 0.60 0.64" width="256" height="256"/>')
        mats.append(f'    <material name="mat_wall_{tag}" texture="tex_wall_{tag}" '
                    f'texuniform="false" texrepeat="12 5"/>')
    for table in TABLES:
        for item in table.items:
            mats.append(f'    <material name="mat_{item.name}" '
                        f'rgba="{fmt(item.rgba)}" specular="0.3" shininess="0.4"/>')
    return "\n".join(mats)


HEADER = """<mujoco model="{name}">
  <!-- Generated by scripts/build_room.py - edit the spec there, not this file. -->
{include}
  <statistic center="0 0 0.8" extent="4"/>

  <option impratio="10"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>
    <map znear="0.02"/>
    <quality shadowsize="4096"/>
    <global azimuth="150" elevation="-20" offwidth="1280" offheight="960"/>
  </visual>
"""


def scene_xml(keyframes: str = "", robot: bool = True) -> str:
    """The room.  With the robot included, or on its own as a plain environment."""
    header = HEADER.format(
        name="bracketbot room" if robot else "room",
        include='  <include file="bracketbot.xml"/>\n' if robot else "")
    body = [header,
            "  <asset>", materials(), "  </asset>", "",
            "  <worldbody>",
            *[f'    <light name="ceiling_{i}" pos="{x} {y} 2.45" dir="0 0 -1" '
              f'diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05" '
              f'castshadow="{"true" if i == 0 else "false"}"/>'
              for i, (x, y) in enumerate([(-1.6, -1.0), (1.6, -1.0),
                                          (-1.6, 1.0), (1.6, 1.0)])],
            f'    <geom name="floor" type="plane" size="{ROOM_X / 2 + 0.5:.4g} '
            f'{ROOM_Y / 2 + 0.5:.4g} 0.05" material="mat_floor" condim="3" '
            'friction="1.5 0.005 0.0001"/>',
            xml_walls()]
    for table in TABLES:
        body += ["", xml_table(table)]
        for item in table.items:
            body.append(xml_item(item, table))
    body += ["  </worldbody>"]
    if keyframes:
        body += ["", keyframes]
    body += ["</mujoco>", ""]
    return "\n".join(body)


# ---- keyframes -----------------------------------------------------------
def robot_qpos(x, y, yaw, rail=0.0, model=None):
    """The 27 robot qpos: free root, two wheels, then 2 x (rail + 6 arm + gripper)."""
    q = np.zeros(27)
    q[0:3] = [x, y, 0.0]
    q[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    q[9] = rail          # rj0
    q[18] = rail         # lj0
    return q


def keyframe_block(model) -> str:
    """Poses worth starting from.  Object qpos comes straight off the compiled
    model, so a keyframe never has to restate where the crockery lives."""
    rest = model.qpos0[27:].copy()

    poses = [("start", 0.0, 0.0, 0.0, 0.0),
             ("drive", 0.0, 0.0, 0.0, -0.35)]
    for table in TABLES:
        x, y, yaw = table.dock
        poses.append((f"dock_{table.name.split('_')[1]}", x, y, yaw, 0.0))

    lines = ["  <keyframe>",
             "    <!-- 'drive' drops the arm carriages 0.35 m: the hands hang at",
             "         0.715 m, which is table height, and 0.365 m is not. -->"]
    for name, x, y, yaw, rail in poses:
        q = np.concatenate([robot_qpos(x, y, yaw, rail), rest])
        lines.append(f'    <key name="{name}" qpos="{fmt(q)}"/>')
    lines.append("  </keyframe>")
    return "\n".join(lines)


# ---- build-time reachability check ---------------------------------------
def grasp_pose(item: Item, table: Table):
    """Where the grip site has to be to take this item, and how high off the table.

    Everything here is grasped top-down around its waist: the fingers are 134 mm
    blades, so they straddle the object and close, rather than pinching its top.
    """
    base = table.place(item)
    s = item.size
    if item.kind == "ball":
        height = s["r"]                      # the equator
    elif item.kind == "cube":
        height = s["s"] / 2
    elif item.kind == "crate":
        height = s["h"] - 0.025              # just under the rim
    else:
        height = s["h"] - 0.010              # bowl and plate: on the rim
    return base + np.array([0, 0, height])


def check_reach(model, verbose=True) -> list[str]:
    """Solve IK for a top-down grasp on every object, from its docking pose."""
    data = mujoco.MjData(model)
    solvers = {side: ArmIK(model, side) for side in ("right", "left")}
    problems = []

    if verbose:
        print("\n  top-down IK from the docking pose")
    for table in TABLES:
        dx, dy, yaw = table.dock
        for item in table.items:
            target = grasp_pose(item, table)
            data.qpos[:] = model.qpos0
            data.qpos[0:3] = [dx, dy, 0.0]
            data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

            best, best_side, best_yaw = None, None, None
            for side, ik in solvers.items():
                for g in item.yaws:
                    got = ik.solve(data, target, down_quat(yaw + g),
                                   seed=model.qpos0[ik.qadr])
                    if best is None or (got.pos_err, got.rot_err) < (best.pos_err,
                                                                     best.rot_err):
                        best, best_side, best_yaw = got, side, g
                    if best.ok:
                        break
                if best.ok:
                    break

            if verbose:
                flag = "ok  " if best.ok else "MISS"
                print(f"    {flag} {item.name:<12} {best_side:>5} arm, wrist "
                      f"{math.degrees(best_yaw):3.0f} deg   "
                      f"{best.pos_err * 1000:5.1f} mm, "
                      f"{math.degrees(best.rot_err):4.1f} deg")
            if not best.ok:
                problems.append(f"{item.name}: best IK leaves the grip site "
                                f"{best.pos_err * 1000:.0f} mm and "
                                f"{math.degrees(best.rot_err):.0f} deg off")
            if item.width > MAX_GRASP_WIDTH:
                problems.append(f"{item.name}: {item.width * 1000:.0f} mm across "
                                f"exceeds the {MAX_GRASP_WIDTH * 1000:.0f} mm budget")
    return problems


def _rect(x, y, yaw, half):
    """The four corners of an oriented rectangle, and its two axes."""
    c, s = math.cos(yaw), math.sin(yaw)
    axes = np.array([[c, s], [-s, c]])
    corners = np.array([[sx * half[0], sy * half[1]]
                        for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    return corners @ axes + np.array([x, y]), axes


def rect_gap(a_corners, a_axes, b_corners, b_axes) -> float:
    """Separation between two oriented rectangles; negative if they overlap.

    Separating axis theorem: two convex shapes are apart iff some axis exists on
    which their projections do not overlap, and the widest such gap is the
    distance between them.
    """
    widest = -np.inf
    for axis in np.vstack([a_axes, b_axes]):
        a = a_corners @ axis
        b = b_corners @ axis
        widest = max(widest, max(b.min() - a.max(), a.min() - b.max()))
    return float(widest)


def dock_gap(model, x, y, yaw, half) -> float:
    """How much daylight the robot has when parked facing a table.

    The clearance map treats the robot as a 0.42 m circle, because when it is
    driving it could be pointing anywhere.  Parked at a table it is not: it is
    0.19 m deep and that is the side facing the table, so the circle understates
    the gap by 13 cm and calls a perfectly good docking pose blocked.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    corners, axes = _rect(x, y, yaw, half)

    gap = np.inf
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if model.geom_bodyid[g] and model.body_jntnum[model.geom_bodyid[g]]:
            continue
        pos, size = data.geom_xpos[g], model.geom_size[g]
        rot = data.geom_xmat[g].reshape(3, 3)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            other, other_axes = _rect(pos[0], pos[1], 0.0, (size[0], size[0]))
        else:
            yaw_g = math.atan2(rot[1, 0], rot[0, 0])
            other, other_axes = _rect(pos[0], pos[1], yaw_g, size[:2])
        gap = min(gap, rect_gap(corners, axes, other, other_axes))
    return gap


def reachable(xs, ys, free, start):
    """Flood fill from the robot's start cell."""
    seen = np.zeros_like(free)
    i = int(np.clip(np.searchsorted(xs, start[0]), 0, len(xs) - 1))
    j = int(np.clip(np.searchsorted(ys, start[1]), 0, len(ys) - 1))
    if not free[i, j]:
        return seen
    stack = [(i, j)]
    seen[i, j] = True
    while stack:
        a, b = stack.pop()
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            m, n = a + da, b + db
            if 0 <= m < free.shape[0] and 0 <= n < free.shape[1] \
                    and free[m, n] and not seen[m, n]:
                seen[m, n] = True
                stack.append((m, n))
    return seen


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

    for table in TABLES:
        x, y, _ = table.dock
        i = int(np.clip(np.searchsorted(xs, x), 0, len(xs) - 1))
        j = int(np.clip(np.searchsorted(ys, y), 0, len(ys) - 1))
        gap = dock_gap(model, x, y, table.dock[2], half)
        near = seen[max(0, i - 6):i + 7, max(0, j - 6):j + 7].any()
        if gap > 0.02 and near:
            print(f"    ok   {table.name:<12} parks with {gap * 100:4.1f} cm to spare, "
                  f"and open floor behind it")
        else:
            why = ("overlaps the furniture" if gap <= 0.02
                   else "has no reachable floor behind it")
            print(f"    MISS {table.name:<12} docking pose {why}")
            problems.append(f"{table.name}: docking pose {why}")

    print()
    for j in range(len(ys) - 1, -1, -2):
        row = "".join("#" if not free[i, j] else ("." if seen[i, j] else ",")
                      for i in range(len(xs)))
        print("    " + row)
    print("    # wall or furniture, . floor the robot can reach, , floor it cannot")
    return problems


def build(check: bool = True) -> None:
    ROOM_ONLY.write_text(scene_xml(robot=False))
    mujoco.MjModel.from_xml_path(str(ROOM_ONLY))     # it has to compile on its own

    OUT.write_text(scene_xml())
    model = mujoco.MjModel.from_xml_path(str(OUT))
    OUT.write_text(scene_xml(keyframe_block(model)))
    model = mujoco.MjModel.from_xml_path(str(OUT))

    items = [i for t in TABLES for i in t.items]
    print(f"wrote {ROOM_ONLY.relative_to(REPO)} (the room on its own) "
          f"and {OUT.relative_to(REPO)} (with the robot)")
    print(f"  room {ROOM_X} x {ROOM_Y} m   tables {len(TABLES)}   "
          f"objects {len(items)}   bodies {model.nbody}   dof {model.nv}")
    for table in TABLES:
        x, y, yaw = table.dock
        print(f"  {table.name:<12} at {table.centre}  dock ({x:+.2f}, {y:+.2f}, "
              f"{math.degrees(yaw):+.0f} deg)  " +
              ", ".join(i.name for i in table.items))

    if not check:
        return
    # the clearance map is of the room itself; the robot is what it is measured
    # against, not part of the furniture
    problems = check_clearance(mujoco.MjModel.from_xml_path(str(ROOM_ONLY)))
    problems += check_reach(model)
    if problems:
        raise SystemExit("\nnot workable as laid out:\n  " + "\n  ".join(problems))
    print("\n  every object can be reached top-down from its docking pose")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-check", action="store_true",
                    help="skip the clearance and arm-reach checks")
    build(check=not ap.parse_args().no_check)
