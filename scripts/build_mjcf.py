"""Compile the BracketBot URDF into a simulatable MJCF.

    .venv/bin/python scripts/build_mjcf.py

The URDF is a geometry export, not a physics model.  Three things have to be
repaired before MuJoCo can do anything with it, and each is applied here rather
than by hand-editing so the source URDF stays pristine and every fix is
reviewable:

  1. No wheel joints.  Both wheels are welded into a chain of fixed joints, so
     the robot is one rigid lump.  We lift the four wheel geoms out into two new
     bodies hinged about the axle.
  2. No collision geometry at all - 50 visuals, 0 collisions.  Visual meshes are
     marked non-colliding and primitives are added for the parts that touch the
     world.
  3. Broken inertials.  The export totals 0.29 kg with 95% of it in one link,
     and inertias down at 1e-9.  We discard them and recompute from mesh volume
     at a uniform density scaled to hit TOTAL_MASS.

Output: models/bracketbot.xml (committed, so running this is optional).
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
URDF = REPO / "models" / "bracketbot" / "chopped_urdf_v2.urdf"
OUT = REPO / "models" / "bracketbot.xml"

# PLACEHOLDER until the real robot is weighed - see the module docstring.
TOTAL_MASS = 12.0          # kg, whole robot
WHEEL_MASS_FRACTION = 0.06  # each wheel, of the total

# URDF <mimic>: follower -> leader.  MuJoCo's URDF parser already emits the
# equality constraint; what it cannot know is that the follower must not also get
# a servo.  Give it one and the servo fights the constraint, pinning the gripper
# shut - commanded fully open it reached -0.10 rad instead of +1.0.
MIMIC = {
    "right_right_gripper": "right_left_gripper",
    "left_right_gripper": "left_left_gripper",
}

URDF_EFFORT = 10.0          # every joint in the URDF carries effort=10 - boilerplate
SERVO_MARGIN = 2.5          # headroom over the worst-case gravity load

WHEEL_MESHES = {
    "left": ("Left_Wheel_Tire__Left_Wheel_Tire", "Left_wheel_cap__Left_wheel_cap"),
    "right": ("Right_Wheel_Tire__Right_Wheel_Tire", "Right_Wheel_Cap__Right_Wheel_Cap"),
}

MUJOCO_BLOCK = """<robot name="chopped_urdf_v2">
    <mujoco>
        <compiler meshdir="bracketbot/meshes/" strippath="true" discardvisual="false"
                  balanceinertia="true" fusestatic="false"/>
    </mujoco>"""


def _compose(pos_a, quat_a, pos_b, quat_b):
    """Frame A applied to frame B, both (pos, quat)."""
    rotated = np.zeros(3)
    mujoco.mju_rotVecQuat(rotated, np.asarray(pos_b, dtype=float), np.asarray(quat_a, dtype=float))
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, np.asarray(quat_a, dtype=float), np.asarray(quat_b, dtype=float))
    return np.asarray(pos_a, dtype=float) + rotated, quat


def world_pose(body):
    """Walk a spec body up to the world, composing frames.

    The wheels sit 14 bodies deep under `base_plate__base_plate`, which carries a
    -90 degree rotation about x.  Every other body in that chain is identity, so
    dropping this one transform is easy to do and lands the wheels with y and z
    swapped - which is exactly what it looks like.
    """
    pos, quat = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    chain = []
    node = body
    while node is not None and node.name != "world":
        chain.append(node)
        node = node.parent
    for node in reversed(chain):
        pos, quat = _compose(pos, quat, node.pos, node.quat)
    return pos, quat


def _urdf_with_compiler(tmpdir: Path) -> Path:
    """Copy the URDF with a <mujoco> compiler block injected (a URDF extension
    MuJoCo defines for exactly this).  The vendored file stays untouched."""
    text = URDF.read_text()
    text = text.replace('<robot name="chopped_urdf_v2">', MUJOCO_BLOCK, 1)
    path = tmpdir / "chopped_urdf_v2.urdf"
    path.write_text(text)
    shutil.copytree(URDF.parent / "meshes", tmpdir / "bracketbot" / "meshes")
    return path


def measure(path: Path) -> dict:
    """World-frame geometry of the wheels, read off the compiled meshes."""
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    out = {}
    for side, (tire, _cap) in WHEEL_MESHES.items():
        for i in range(model.ngeom):
            mid = model.geom(i).dataid[0]
            if mid < 0 or model.mesh(mid).name != tire:
                continue
            adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            verts = model.mesh_vert[adr : adr + num]
            world = verts @ data.geom_xmat[i].reshape(3, 3).T + data.geom_xpos[i]
            lo, hi = world.min(0), world.max(0)
            out[side] = {
                "center": (lo + hi) / 2,
                "radius": float((hi[2] - lo[2]) / 2),
                "half_width": float((hi[1] - lo[1]) / 2),
            }
    return out


def build(total_mass: float = TOTAL_MASS) -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        urdf = _urdf_with_compiler(tmp)
        wheels = measure(urdf)
        spec = mujoco.MjSpec.from_file(str(urdf))

        spec.modelname = "bracketbot"
        spec.option.timestep = 0.002
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE

        root = spec.body("root")
        root.add_freejoint(name="root")

        # ---- 1. wheels: lift the geoms out into hinged bodies -----------------
        by_mesh = {}
        for body in spec.bodies:
            for geom in body.geoms:
                if geom.meshname:
                    by_mesh.setdefault(geom.meshname, []).append((body, geom))

        for side, meshes in WHEEL_MESHES.items():
            geom = wheels[side]
            centre = geom["center"]
            wheel = root.add_body(
                name=f"wheel_{side}",
                pos=[0.0, float(centre[1]), float(centre[2])],
            )
            wheel.add_joint(
                name=f"wheel_{side}",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=[0, 1, 0],
                damping=0.01,
                armature=0.005,
            )
            axle = np.array([0.0, centre[1], centre[2]])
            for mesh_name in meshes:
                for parent, old in by_mesh.get(mesh_name, []):
                    # the geom's pose in the world, then expressed in the new
                    # (axis-aligned) wheel body - not the parent-relative pose,
                    # which silently loses the base_plate rotation
                    gpos, gquat = _compose(*world_pose(parent), old.pos, old.quat)
                    wheel.add_geom(
                        name=f"{mesh_name}_visual",
                        type=mujoco.mjtGeom.mjGEOM_MESH,
                        meshname=mesh_name,
                        pos=gpos - axle,
                        quat=gquat,
                        rgba=old.rgba,
                        contype=0,
                        conaffinity=0,
                        group=2,
                        mass=0.0,           # visual only; wheel mass is on the cylinder
                    )
                    spec.delete(old)

            # the only part of the robot that touches the ground
            wheel.add_geom(
                name=f"wheel_{side}_collision",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[geom["radius"], geom["half_width"], 0],
                zaxis=[0, 1, 0],
                mass=total_mass * WHEEL_MASS_FRACTION,
                friction=[1.5, 0.005, 0.0001],
                condim=3,
                group=3,
                rgba=[0.1, 0.1, 0.12, 1],
            )

        # ---- 2. every remaining mesh becomes visual-only ----------------------
        chassis_geoms = []
        for body in spec.bodies:
            if body.name.startswith("wheel_"):
                continue
            for g in body.geoms:
                if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                    g.contype, g.conaffinity, g.group = 0, 0, 2
                    chassis_geoms.append(g)

        # ---- 3. mass: discard the export's inertials, recompute from volume ---
        for body in spec.bodies:
            body.explicitinertial = False
        chassis_mass = total_mass * (1 - 2 * WHEEL_MASS_FRACTION)
        for g in chassis_geoms:
            g.density = 1000.0          # provisional; rescaled below
            g.mass = float("nan")       # nan = "derive mass from density x volume"

        # ---- imu, actuators, sensors ----------------------------------------
        root.add_site(name="imu", pos=[0, 0, 0.2], size=[0.01, 0, 0])

        for side in ("left", "right"):
            spec.add_actuator(
                name=f"wheel_{side}",
                target=f"wheel_{side}",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gear=[1, 0, 0, 0, 0, 0],
                ctrlrange=[-8, 8],
                ctrllimited=1,
            )

        # position servos on every arm joint, travel limits straight from the URDF
        arm_joints = [
            j
            for b in spec.bodies
            for j in b.joints
            if j.name
            and not j.name.startswith("wheel_")
            and j.name not in MIMIC          # driven by its leader, not a servo
            and j.type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ]
        for joint in arm_joints:
            lo, hi = float(joint.range[0]), float(joint.range[1])
            # a P servo settles at load/kp: the 17 N carriage on kp=2000 hangs
            # 8.6 mm low, so the slides get an order more stiffness than the hinges
            kp = 200.0 if joint.type == mujoco.mjtJoint.mjJNT_HINGE else 20000.0
            spec.add_actuator(
                name=joint.name,
                target=joint.name,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gainprm=[kp, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                biasprm=[0, -kp, -0.1 * kp, 0, 0, 0, 0, 0, 0, 0],
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                forcerange=[-10, 10],      # URDF effort limit
                forcelimited=1,
                ctrlrange=[lo, hi],
                ctrllimited=1,
            )

        spec.add_sensor(
            name="gyro", type=mujoco.mjtSensor.mjSENS_GYRO,
            objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu",
        )
        spec.add_sensor(
            name="accel", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
            objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu",
        )
        spec.add_sensor(
            name="orient", type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
            objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu",
        )
        for side in ("left", "right"):
            spec.add_sensor(
                name=f"vel_{side}", type=mujoco.mjtSensor.mjSENS_JOINTVEL,
                objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=f"wheel_{side}",
            )

        # No <mimic> handling needed here: MuJoCo's URDF parser already turns each
        # one into a joint equality constraint.  Adding our own duplicates it and
        # makes the solver satisfy the same constraint twice.  What the parser
        # does *not* do is stop us putting a servo on the follower - see MIMIC.

        # scale density so the whole robot weighs TOTAL_MASS
        model = spec.compile()
        got = sum(model.body_mass) - 2 * total_mass * WHEEL_MASS_FRACTION
        if got > 1e-9:
            scale = chassis_mass / got
            for g in chassis_geoms:
                g.density = 1000.0 * scale
        model = spec.compile()

        # Size the servos to hold their own limb up.  The URDF gives every
        # single joint effort=10 and velocity=10 - boilerplate, not a spec - and
        # 10 N cannot hold the 17 N mast carriage, so the arms slide down the
        # rail on the first step.  Size from the gravity load instead and keep
        # the URDF number as a floor.
        limits = gravity_loads(model, [j.name for j in arm_joints])
        for act in spec.actuators:
            need = limits.get(act.name)
            if need is None:
                continue
            limit = max(URDF_EFFORT, SERVO_MARGIN * need)
            act.forcerange = [-limit, limit]
        # The URDF effort also lands on the *joint's* actuatorfrcrange, which
        # clamps qfrc_actuator no matter how big the actuator's forcerange is.
        # Raising one without the other leaves the servo asking for 43 N and
        # getting 10, and the arms slide down the mast anyway.
        for joint in arm_joints:
            need = limits.get(joint.name)
            if need is None:
                continue
            limit = max(URDF_EFFORT, SERVO_MARGIN * need)
            joint.actfrcrange = [-limit, limit]
            joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        model = spec.compile()

        xml = spec.to_xml()
        OUT.write_text(xml)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    com = data.subtree_com[1]
    print(f"wrote {OUT.relative_to(REPO)}")
    print(f"  bodies {model.nbody}  dof {model.nv}  actuators {model.nu}  "
          f"meshes {model.nmesh}  sensors {model.nsensor}")
    print(f"  total mass {sum(model.body_mass):.3f} kg   CoM height {com[2]:.3f} m")
    for side in ("left", "right"):
        w = measure_wheel(model, side)
        print(f"  wheel_{side}: r={w[0]:.4f} m  half-width={w[1]:.4f} m  "
              f"axle at y={w[2]:+.4f} z={w[3]:.4f}")


def gravity_loads(model, joint_names: list[str]) -> dict[str, float]:
    """Worst-case gravity load each arm joint has to hold.

    For a slide joint that is the weight of everything below it; for a hinge,
    that weight times the longest lever arm in its subtree.  An upper bound, which
    is what we want for sizing a limit.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    out = {}
    for name in joint_names:
        jid = model.joint(name).id
        bid = model.jnt_bodyid[jid]
        weight = float(model.body_subtreemass[bid]) * 9.81
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE:
            out[name] = weight
            continue
        anchor = data.xanchor[jid]
        lever = 0.0
        for b in range(model.nbody):
            root = b
            while root != 0 and root != bid:
                root = model.body_parentid[root]
            if root == bid and model.body_mass[b] > 0:
                lever = max(lever, float(np.linalg.norm(data.xipos[b] - anchor)))
        out[name] = weight * lever
    return out


def measure_wheel(model, side):
    gid = model.geom(f"wheel_{side}_collision").id
    bid = model.geom(gid).bodyid[0]
    return (
        float(model.geom_size[gid][0]),
        float(model.geom_size[gid][1]),
        float(model.body_pos[bid][1]),
        float(model.body_pos[bid][2]),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--total-mass", type=float, default=TOTAL_MASS)
    build(p.parse_args().total_mass)
