"""How close does the arm from joint 3 outward get to the robot's own body?

    .venv/bin/python scripts/check_arm_clearance.py

The robot's collision geoms are filtered so they never collide with each other
(contype 2 / conaffinity 1), so the sim will happily drive the forearm through
the mast and say nothing.  The real arm is torque-controlled: striking the
chassis loads the servo, it fights the contact, and the arm jerks back out.
So measure the clearance directly with mj_geomDistance, which ignores the
filter, between every arm collider from joint 3 outward (forearm, hand, both
fingers) and the three chassis boxes.

Three views:
  1. every grasp waypoint the planner actually commands, for all ten objects;
  2. joint 3 swept through its full range with the rest of the arm at the
     sponsors' home pose;
  3. random reachable poses, reporting the closest approach.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from rlbot.arm import ARM_JOINTS, ArmIK          # noqa: E402
from rlbot.room import TABLES                    # noqa: E402
from check_grasp import plan_grasp, welded_at    # noqa: E402

# joint 3 outward: bicep is joint 2's link, so it is deliberately left out
J3_LINKS = {
    "right": ("forearm__forearm", "hand__hand",
              "left_finger__left_finger", "right_finger__right_finger"),
    "left": ("l_forearm__forearm", "l_hand__hand",
             "l_left_finger__left_finger", "l_right_finger__right_finger"),
}
WARN = 0.010   # m; the arm rests 15 mm off the mast covers at HOME, so 10 mm is a real dip
PATH_SAMPLES = 20   # points along each joint-space leg between waypoints
# Where the real arm starts a reach: the sponsors' daemon homes to this and
# their IK centres on it (constants.py: home j3 = 0.25 turn = 90 deg).  Not
# the URDF zero, which hangs the forearm alongside the mast.
HOME = np.array([0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 0.0])


def chassis_geoms(model):
    """Every mesh on the rigid part of the robot (mast, covers, head, drive
    unit) plus the wheel cylinders.  Not the three chassis_* boxes: those hug
    the widest vertex over a 1.2 m band and sit 10-16 mm proud of the covers,
    which turns the arm's own rest pose into a false alarm."""
    arm_root = model.body("arm_base").id

    def in_arm(b):
        while b != 0:
            if b == arm_root:
                return True
            b = model.body_parentid[b]
        return False

    out = []
    for g in range(model.ngeom):
        body = model.body(model.geom_bodyid[g]).name
        if body.startswith("wheel_"):
            if model.geom(g).name.endswith("_collision"):
                out.append((g, body))
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and not in_arm(model.geom_bodyid[g]):
            out.append((g, model.mesh(model.geom_dataid[g]).name.split("__")[0]))
    return out


def clearance(model, data, side: str):
    """(min distance, arm link, chassis part) at the current qpos."""
    mujoco.mj_kinematics(model, data)
    best = (float("inf"), "", "")
    for link in J3_LINKS[side]:
        g = model.geom(f"{link}_collision").id
        for c, name in chassis_geoms(model):
            d = mujoco.mj_geomDistance(model, data, g, c, 1.0, None)
            if d < best[0]:
                best = (float(d), link, name)
    return best


def grasp_waypoints(model):
    """Clearance at every commanded waypoint for every object, both arms."""
    data = mujoco.MjData(model)
    worst = (float("inf"), "")
    print("1. grasp waypoints (above / on / lift), min clearance to chassis")
    for table in TABLES:
        x, y, yaw = table.dock
        for item in table.items:
            m = welded_at(x, y, yaw)
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            # seed the planner from the sponsors' home pose, both arms
            seed = m.qpos0.copy()
            for arm in ("right", "left"):
                seed[ArmIK(m, arm).qadr] = HOME
            plan = plan_grasp(m, mujoco.MjData(m), item, table, seed)
            if plan is None:
                print(f"   {item.name:<12} no plan")
                continue
            _, side, _, _, *chain = plan
            ik = ArmIK(m, side)
            # check_grasp.move() ramps the servos linearly in joint space, so
            # the path is a straight line between consecutive waypoints; the
            # first leg starts from the arm's home pose
            legs = [HOME] + [sol.qpos for sol in chain]
            out, dists = [], []
            for tag, q0, q1 in zip(("above", "on", "lift"), legs, legs[1:]):
                leg_min = (float("inf"), "", "")
                for s in np.linspace(0.0, 1.0, PATH_SAMPLES):
                    d.qpos[:] = m.qpos0
                    d.qpos[ik.qadr] = q0 + (q1 - q0) * s
                    got = clearance(m, d, side)
                    if got[0] < leg_min[0]:
                        leg_min = got
                dist, link, c = leg_min
                out.append(f"{tag} {dist*1000:5.0f} mm")
                dists.append(dist)
                if dist < worst[0]:
                    worst = (dist, f"{item.name} ->{tag}: {link} vs {c}")
            flag = "NEAR" if min(dists) < WARN else "    "
            print(f"   {flag} {item.name:<12} {side:>5}   " + "   ".join(out))
    print(f"   worst: {worst[0]*1000:.0f} mm at {worst[1]}")
    return worst[0]


def joint3_sweep(model, side: str, steps: int = 25):
    """Joint 3 through its range, everything else at HOME, rail at top."""
    data = mujoco.MjData(model)
    ik = ArmIK(model, side)
    j3 = ik.jids[3]
    lo, hi = model.jnt_range[j3]
    print(f"2. {side} arm: joint 3 ({ARM_JOINTS[side][3]}) swept {math.degrees(lo):.0f}..{math.degrees(hi):.0f} deg, others at HOME")
    worst = (float("inf"), 0.0, "", "")
    row = []
    for q in np.linspace(lo, hi, steps):
        data.qpos[:] = model.qpos0
        data.qpos[ik.qadr] = HOME
        data.qpos[model.jnt_qposadr[j3]] = q
        dist, link, c = clearance(model, data, side)
        row.append(f"{math.degrees(q):+4.0f}:{dist*1000:3.0f}")
        if dist < worst[0]:
            worst = (dist, q, link, c)
    print("   deg:mm  " + "  ".join(row))
    print(f"   closest {worst[0]*1000:.0f} mm at {math.degrees(worst[1]):+.0f} deg "
          f"({worst[2]} vs {worst[3]})")
    return worst[0]


def random_poses(model, side: str, n: int = 400, seed: int = 0):
    """Closest approach over random poses inside the joint limits."""
    data = mujoco.MjData(model)
    ik = ArmIK(model, side)
    rng = np.random.default_rng(seed)
    near, touching, worst = 0, 0, (float("inf"), None, "", "")
    for _ in range(n):
        q = rng.uniform(ik.lo, ik.hi)
        data.qpos[:] = model.qpos0
        data.qpos[ik.qadr] = q
        dist, link, c = clearance(model, data, side)
        near += dist < WARN
        touching += dist <= 0.0
        if dist < worst[0]:
            worst = (dist, q, link, c)
    print(f"3. {side} arm: {n} random reachable poses -> "
          f"{touching} penetrate the chassis, {near} within {WARN*1000:.0f} mm")
    q = worst[1]
    print(f"   closest {worst[0]*1000:.0f} mm ({worst[2]} vs {worst[3]}) at joints "
          + " ".join(f"{name}={v:+.2f}" for name, v in zip(ARM_JOINTS[side], q)))
    return touching


def main() -> None:
    from rlbot.robot import BRACKETBOT
    model = mujoco.MjModel.from_xml_path(str(BRACKETBOT))
    data = mujoco.MjData(model)
    home = []
    for side in ("right", "left"):
        ik = ArmIK(model, side)
        data.qpos[:] = model.qpos0
        data.qpos[ik.qadr] = HOME
        dist, link, c = clearance(model, data, side)
        home.append(f"{side} {dist*1000:.0f} mm ({link} vs {c})")
    print("0. home pose (elbow at 90 deg), min clearance to chassis")
    print("   HOME pose: " + ", ".join(home))
    print()
    grasp_min = grasp_waypoints(model)
    print()
    for side in ("right", "left"):
        joint3_sweep(model, side)
        print()
    hits = sum(random_poses(model, side) for side in ("right", "left"))
    print()
    if grasp_min < WARN:
        print(f"FAIL: a commanded grasp waypoint is within {WARN*1000:.0f} mm of the chassis")
        sys.exit(1)
    print(f"ok: every commanded grasp waypoint clears the chassis by >= {WARN*1000:.0f} mm"
          + (f"  (but {hits} random poses penetrate it: the workspace itself is not safe,"
             " the planner has to keep it out)" if hits else ""))


if __name__ == "__main__":
    main()
