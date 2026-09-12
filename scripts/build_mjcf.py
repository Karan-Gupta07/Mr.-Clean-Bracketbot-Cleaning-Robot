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

WHEEL_MESHES = {
    "left": ("Left_Wheel_Tire__Left_Wheel_Tire", "Left_wheel_cap__Left_wheel_cap"),
    "right": ("Right_Wheel_Tire__Right_Wheel_Tire", "Right_Wheel_Cap__Right_Wheel_Cap"),
}

MUJOCO_BLOCK = """<robot name="chopped_urdf_v2">
    <mujoco>
        <compiler meshdir="bracketbot/meshes/" strippath="true" discardvisual="false"
                  balanceinertia="true" fusestatic="false"/>
    </mujoco>"""


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
            for mesh_name in meshes:
                for _body, old in by_mesh.get(mesh_name, []):
                    wheel.add_geom(
                        name=f"{mesh_name}_visual",
                        type=mujoco.mjtGeom.mjGEOM_MESH,
                        meshname=mesh_name,
                        pos=[p - c for p, c in zip(old.pos, [0.0, centre[1], centre[2]])],
                        quat=old.quat,
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
            and j.type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ]
        for joint in arm_joints:
            lo, hi = float(joint.range[0]), float(joint.range[1])
            kp = 200.0 if joint.type == mujoco.mjtJoint.mjJNT_HINGE else 2000.0
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

        # ---- mimic joints: URDF <mimic> has no MJCF equivalent ---------------
        for follower, leader in (
            ("right_right_gripper", "right_left_gripper"),
            ("left_right_gripper", "left_left_gripper"),
        ):
            names = {j.name for j in arm_joints}
            if follower in names and leader in names:
                eq = spec.add_equality(
                    name=f"{follower}_mimic",
                    type=mujoco.mjtEq.mjEQ_JOINT,
                    name1=follower,
                    name2=leader,
                )
                eq.data[:5] = [0, 1, 0, 0, 0]

        # scale density so the whole robot weighs TOTAL_MASS
        model = spec.compile()
        got = sum(model.body_mass) - 2 * total_mass * WHEEL_MASS_FRACTION
        if got > 1e-9:
            scale = chassis_mass / got
            for g in chassis_geoms:
                g.density = 1000.0 * scale
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
