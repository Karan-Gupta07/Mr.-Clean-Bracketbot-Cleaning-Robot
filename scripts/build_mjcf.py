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

A balancing robot only ever needed wheels-on-floor contact, so (2) stopped at
two wheel cylinders.  Reaching for something on a table needs more:

  4. Nothing but the wheels could touch anything.  The fingers passed straight
     through whatever they closed on and the mast drove through walls.  Hands,
     fingers, forearms and biceps get a collision copy of their visual mesh, and
     the chassis gets three boxes measured off its own meshes.
  5. Nowhere to aim.  There was no frame between the fingertips to drive an arm
     to, and no mount for the sensors a map needs.  Both hands get a `grip_*`
     site, and the chassis gets a lidar site and a head camera.

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

# Robot collision geometry is contype 2 / conaffinity 1, so it collides with the
# world (1/1) but never with itself.  The URDF's joint limits are boilerplate
# +-120 degrees on every hinge, which lets the arms fold into the mast; a
# self-contact there would be an artefact of bad limits, not of the real robot,
# and the balancer would have to fight it.
# Collision masks.  The world is 1/1.  The two arms get their own bits so that
# they collide with the world and *with each other*, but never with themselves -
# the URDF's joint limits are boilerplate +-120 degrees on every hinge, so a
# link touching its own neighbour is an artefact of bad limits, while one arm
# swinging through the other is a real collision that a real robot would have.
# The chassis has a third bit so the arms can fold against the mast without
# fighting it.
WORLD = 1
RIGHT_ARM, LEFT_ARM, CHASSIS = 2, 4, 8
ARM_MASK = {"right": (RIGHT_ARM, WORLD | LEFT_ARM),
            "left": (LEFT_ARM, WORLD | RIGHT_ARM)}
ROBOT_CONTYPE, ROBOT_CONAFFINITY = CHASSIS, WORLD

# Links that get a collision copy of their visual mesh.  Grasping happens on the
# fingers and the palm between them; the arm links are here so a badly aimed
# reach stops at the table instead of sweeping through it.
COLLIDING_LINKS = (
    "hand__hand", "forearm__forearm", "bicep__bicep",
    "l_hand__hand", "l_forearm__forearm", "l_bicep__bicep",
)

# The fingers are the exception.  MuJoCo collides a mesh as its convex hull, and
# each blade is a 131 x 67 x 37 mm hook with a concave inner face: hulled, the
# two of them fill the jaw solid.  A 55 mm cube placed dead centre between blades
# 139 mm apart was already in contact with both, and closing shot it out.  So
# each blade gets a flat pad fitted to its real inner face instead, measured off
# the mesh at PAD_REF_OPEN.
# One slab per blade, on the last 46 mm before the tip.  These blades are hooks:
# measured face to face, the throat down by the pivot never opens past about
# 45 mm however wide the hand goes, while the mouth reaches 130 mm.  A pad that
# spans the whole blade therefore has its deep corners jutting into the jaw,
# and an object entering the mouth wedges on them and squirts back out.  Pad the
# mouth only, where the two faces stay roughly parallel, and the hand grips the
# way its shape says it should: near the fingertips.
PAD_REF_OPEN = 0.25        # rad, gripper angle the pads are measured at -
                           # near closed, which is where this claw grips
PAD_BANDS = ((-0.044, -0.004),)    # m, depth along the blade
PAD_SKIN = 0.004           # m, how far in from a slice's extreme counts as face
PAD_SLICES = 12            # slices along the blade used to trace that face
PAD_HALF = (0.014, 0.003, 0.018)   # m, half sizes: across, through, along
# Compliant rubber pads, in the only way a rigid-body solver can have them:
# all three friction dimensions, on condim 6 contacts.
#
# Sliding friction stops the object creeping out of the pinch.  At mu = 1.2 the
# solver let a 50 g cube slide 80 mm out of a 10 N grip in three seconds -
# twenty times the friction it needed on paper, lost to the way MuJoCo trades
# normal impedance against friction impedance.  That and the scene's impratio
# fixed the sliding.
#
# What it did not fix was rolling.  A round object between two flat rigid pads
# touches each at a point, and nothing resists it turning about the line between
# them: the mug tilted 84 degrees in the first fifth of every carry and the ball
# could not be picked up at all.  That is exactly what a soft pad prevents in
# real life - it deforms around the curve - and `condim 6` with a rolling
# friction term is how you say so here.  With it, both of them are carried.
PAD_CONDIM = 6
GRIP_FRICTION = [3.0, 0.05, 0.015]   # sliding, torsional, rolling
ARM_FRICTION = [0.6, 0.005, 0.0001]

# Three boxes spanning the chassis, each sized to the meshes inside its own
# height band: drive unit, mast, head.  CLEARANCE lifts the lowest box off the
# floor - the wheels are what the robot stands on, and a chassis box that
# scraped the ground would quietly hold the robot up and flatter the balancer.
CHASSIS_BANDS = (
    ("chassis_drive", 0.00, 0.22),
    ("chassis_mast", 0.22, 1.44),
    ("chassis_head", 1.44, 1.75),
)
CHASSIS_CLEARANCE = 0.05    # m, floor to the bottom of the lowest box

GRIP_OPEN = 0.5             # gripper angle the grip frame is measured at, rad
LIDAR_HEIGHT = 0.32         # m up the mast: clears the wheels, under the arms

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

        # ---- 4. arm collision: a copy of each mesh that can actually touch ----
        # A copy, not a flag flip: the visual keeps the density that carries the
        # link's mass, and the collider is weightless, so adding contact cannot
        # move the CoM this model's gains are tuned around.
        for name in COLLIDING_LINKS:
            body = spec.body(name)
            finger = "finger" in name or name.endswith("hand__hand")
            contype, conaffinity = ARM_MASK["left" if name.startswith("l_")
                                            else "right"]
            visuals = [g for g in body.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH]
            for n, g in enumerate(visuals):
                body.add_geom(
                    name=f"{name}_collision" + (f"_{n}" if len(visuals) > 1 else ""),
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=g.meshname,
                    pos=g.pos,
                    quat=g.quat,
                    mass=0.0,
                    contype=contype,
                    conaffinity=conaffinity,
                    group=3,
                    condim=4 if finger else 3,
                    friction=GRIP_FRICTION if finger else ARM_FRICTION,
                    rgba=[0.9, 0.4, 0.2, 0.35],
                )

        # the wheels join the same scheme, so a wheel cannot collide with an arm
        for side in ("left", "right"):
            wheel = spec.body(f"wheel_{side}").geoms[-1]
            wheel.contype, wheel.conaffinity = ROBOT_CONTYPE, ROBOT_CONAFFINITY

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

        # ---- 5. frames to aim at, and mounts to see from ---------------------
        # Everything from the wheels up to the head is one rigid chain, so the
        # lidar and the head camera go on `root`: same pose as bolting them to
        # the head, but in a frame that is axis-aligned with the world instead
        # of the -90 degrees about x that base_plate carries.
        root.add_site(name="lidar", pos=[0, 0, LIDAR_HEIGHT], size=[0.012, 0, 0],
                      rgba=[0.2, 0.8, 0.3, 0.6])
        root.add_camera(
            name="head_cam", pos=[0.075, 0, 1.575],
            # a MuJoCo camera looks down its own -z with +y up.  Level, it
            # never saw the table: from 1.575 m up the table top is 46 to 75
            # degrees below the horizon, outside a 58 degree view.  So pitch
            # it 62 degrees down - straight at the middle of a docked table -
            # by tilting the up axis back by the same angle.
            xyaxes=[0, -1, 0, 0.882948, 0, 0.469472], fovy=58,
        )

        model = spec.compile()
        for name, (centre, half) in chassis_boxes(model).items():
            root.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=centre, size=half, mass=0.0,
                contype=ROBOT_CONTYPE, conaffinity=ROBOT_CONAFFINITY,
                group=3, rgba=[0.9, 0.4, 0.2, 0.25],
            )

        for side, (pos, quat) in grip_frames(model).items():
            hand = spec.body("hand__hand" if side == "right" else "l_hand__hand")
            hand.add_site(name=f"grip_{side}", pos=pos, quat=quat,
                          size=[0.008, 0, 0], rgba=[0.2, 0.9, 0.9, 0.7])
            # the site's +z is the approach direction and a camera looks down
            # its own -z, so the camera is the grip frame spun 180 deg about x
            flipped = np.zeros(4)
            mujoco.mju_mulQuat(flipped, quat, np.array([0.0, 1.0, 0.0, 0.0]))
            hand.add_camera(name=f"wrist_{side}_cam", pos=pos, quat=flipped,
                            fovy=70)


        # the pads are measured in the grip site's frame, so that has to exist
        model = spec.compile()
        for tag, (pos, quat, half) in finger_pads(model).items():
            finger, n = tag.split("#")
            pad_type, pad_aff = ARM_MASK["left" if finger.startswith("l_")
                                         else "right"]
            spec.body(finger).add_geom(
                name=f"{finger}_pad{n}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos, quat=quat, size=half, mass=0.0,
                contype=pad_type, conaffinity=pad_aff,
                group=2, condim=PAD_CONDIM, friction=GRIP_FRICTION,
                solimp=[0.97, 0.99, 0.001, 0.5, 2],
                rgba=next(g.rgba for g in spec.body(finger).geoms
                          if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.group == 2),
            )

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
        # No rotor inertia on the blade joints here.  ACT's demonstrations
        # were recorded with armature 0.005 on all four (it stops the follower
        # ringing on the ball), but with it on the skills place 0 of 4 cubes
        # against 4 of 4 without: rlbot.act applies BLADE_ARMATURE for its own
        # episodes and puts it back.
        model = spec.compile()

        xml = spec.to_xml()
        OUT.write_text(xml, newline="\n")

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


def _mesh_points(model, data, body_ids):
    """World-frame vertices of every mesh geom on the given bodies."""
    out = []
    for i in range(model.ngeom):
        mid = model.geom_dataid[i]
        if mid < 0 or model.geom_bodyid[i] not in body_ids:
            continue
        adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = model.mesh_vert[adr : adr + num]
        out.append(verts @ data.geom_xmat[i].reshape(3, 3).T + data.geom_xpos[i])
    return np.vstack(out) if out else np.empty((0, 3))


def _subtree(model, root_id):
    ids = set()
    for b in range(model.nbody):
        node = b
        while node != 0:
            if node == root_id:
                ids.add(b)
                break
            node = model.body_parentid[node]
    return ids


def chassis_boxes(model):
    """Boxes over the parts of the robot that are not wheels and not arms.

    Sized to the meshes in each height band rather than typed in, so they track
    the URDF.  The robot stands 1.61 m tall on a 0.19 x 0.37 m footprint, all of
    it rigid below the shoulders - three boxes describe it closely enough for
    navigation, and no mesh has to be tested against a wall.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    skip = _subtree(model, model.body("arm_base").id)
    skip |= {model.body(f"wheel_{s}").id for s in ("left", "right")}
    pts = _mesh_points(model, data, set(range(model.nbody)) - skip)

    boxes = {}
    for i, (name, lo, hi) in enumerate(CHASSIS_BANDS):
        band = pts[(pts[:, 2] >= lo) & (pts[:, 2] < hi)]
        if not len(band):
            continue
        p_lo, p_hi = band.min(0), band.max(0)
        # In x and y the box hugs the vertices.  In z it spans its whole band -
        # the mast extrusion only has vertices at its two ends, so trusting them
        # would leave a 1.2 m gap - except at the two open ends, where the real
        # geometry is what matters: the floor below and the top of the head.
        p_lo[2] = CHASSIS_CLEARANCE if i == 0 else lo
        p_hi[2] = p_hi[2] if i == len(CHASSIS_BANDS) - 1 else hi
        boxes[name] = ((p_lo + p_hi) / 2, (p_hi - p_lo) / 2)
    return boxes


def grip_frames(model):
    """A frame between the fingertips of each hand, in that hand's body frame.

    +z points out along the fingers (the approach direction), +y is the closing
    axis.  Drive this site to where you want the object and the object ends up
    between the pads.  Measured at GRIP_OPEN, half way through the gripper's
    travel, which is where a grasp actually starts.
    """
    data = mujoco.MjData(model)
    hands = {
        "right": ("hand__hand", "left_finger__left_finger",
                  "right_finger__right_finger", "right_left_gripper"),
        "left": ("l_hand__hand", "l_left_finger__left_finger",
                 "l_right_finger__right_finger", "left_left_gripper"),
    }
    for _, _, _, leader in hands.values():
        data.qpos[model.jnt_qposadr[model.joint(leader).id]] = GRIP_OPEN
    mujoco.mj_forward(model, data)

    out = {}
    for side, (hand, finger_a, finger_b, _) in hands.items():
        origin = data.body(hand).xpos
        rot = data.body(hand).xmat.reshape(3, 3)

        tips, pivots = [], []
        for finger in (finger_a, finger_b):
            cloud = _mesh_points(model, data, {model.body(finger).id})
            far = np.argsort(np.linalg.norm(cloud - origin, axis=1))[-40:]
            tips.append(cloud[far].mean(0))
            pivots.append(data.body(finger).xpos)

        centre = (tips[0] + tips[1]) / 2
        approach = centre - (pivots[0] + pivots[1]) / 2
        approach /= np.linalg.norm(approach)
        closing = tips[1] - tips[0]
        closing -= approach * (closing @ approach)      # orthogonalise
        closing /= np.linalg.norm(closing)

        axes = np.column_stack([np.cross(closing, approach), closing, approach])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (rot.T @ axes).flatten())
        out[side] = (rot.T @ (centre - origin), quat)
    return out


def finger_pads(model):
    """A flat pad on the inner face of each blade, in that finger's own frame.

    Measured in the grip site's frame, where +z runs out along the blades and +y
    is the closing axis: take the vertices in the part of the blade that does the
    gripping, find the face pointing at the other blade, fit the plane they lie
    on, and lay a slab on it.  Returns (pos, quat, half sizes) per pad.
    """
    data = mujoco.MjData(model)
    hands = {
        "right": ("hand__hand", "grip_right",
                  ("left_finger__left_finger", "right_finger__right_finger"),
                  ("right_left_gripper", "right_right_gripper")),
        "left": ("l_hand__hand", "grip_left",
                 ("l_left_finger__left_finger", "l_right_finger__right_finger"),
                 ("left_left_gripper", "left_right_gripper")),
    }
    data.qpos[:] = model.qpos0
    for _, _, _, joints in hands.values():
        for joint in joints:          # the follower has no servo; set it by hand
            data.qpos[model.jnt_qposadr[model.joint(joint).id]] = PAD_REF_OPEN
    mujoco.mj_kinematics(model, data)

    out = {}
    for _side, (_hand, site_name, fingers, _joints) in hands.items():
        site = model.site(site_name).id
        origin, rot = data.site_xpos[site], data.site_xmat[site].reshape(3, 3)
        clouds = {f: (_mesh_points(model, data, {model.body(f).id}) - origin) @ rot
                  for f in fingers}
        inner_side = {f: 1 if c[:, 1].mean() < np.mean(
            [clouds[g][:, 1].mean() for g in fingers]) else -1
            for f, c in clouds.items()}

        for finger, cloud in clouds.items():
            towards = inner_side[finger]          # +1 if this blade faces +y
            body = data.body(finger)
            body_rot = body.xmat.reshape(3, 3)

            for n, (lo, hi) in enumerate(PAD_BANDS):
                # Trace the inner profile slice by slice.  Taking the single
                # most-inward vertex over the whole blade picks the one nearest
                # the pivot, where the two blades almost touch, and fits the pad
                # to a 20 mm patch down in the crook of the hand.
                band = cloud[(cloud[:, 2] > lo) & (cloud[:, 2] < hi)]
                face = []
                for edge in np.linspace(lo, hi, PAD_SLICES + 1)[:-1]:
                    step = (hi - lo) / PAD_SLICES
                    slab = band[(band[:, 2] >= edge) & (band[:, 2] < edge + step)]
                    if len(slab) < 4:
                        continue
                    inner = slab[:, 1].max() if towards > 0 else slab[:, 1].min()
                    face.append(slab[np.abs(slab[:, 1] - inner) < PAD_SKIN])
                face = np.vstack(face)

                # The gripping face is a tilted plane, not a slab parallel to the
                # approach axis: the blades pivot at the hand, so the face leans
                # in towards the tips.  Fit the plane the points actually lie on.
                middle = face.mean(0)
                _, _, axes = np.linalg.svd(face - middle, full_matrices=False)
                normal = axes[2] * (1.0 if axes[2][1] * towards > 0 else -1.0)
                along = axes[0]
                across = np.cross(normal, along)

                frame = np.column_stack([across, normal, along])
                extent = (face - middle) @ frame
                half = [float(np.clip(np.abs(extent[:, 0]).max(), 0.004, PAD_HALF[0])),
                        PAD_HALF[1],
                        float(np.clip(np.abs(extent[:, 2]).max(), 0.010, PAD_HALF[2]))]

                centre = middle - normal * PAD_HALF[1]   # slab just inside the face
                world = origin + rot @ centre
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, (body_rot.T @ rot @ frame).flatten())
                out[f"{finger}#{n}"] = (body_rot.T @ (world - body.xpos), quat, half)
    return out


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
