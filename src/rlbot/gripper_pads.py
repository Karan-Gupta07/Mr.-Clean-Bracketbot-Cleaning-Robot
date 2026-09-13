"""Contact pads for the supplied hooked gripper.

The CAD blades stay exactly as the URDF exports them, and they stay the visuals,
so the hand still looks like the original. What changes is what the simulator
collides. MuJoCo collides a mesh geom as one convex body, and each blade's hull
measures 147.5 cm3 against the mesh's own 46.1 cm3 - a factor of 3.2 - so the
hook the design relies on is filled in solid and the blades meet an object as
fat wedges. Closing them on the 48 mm cube lifts it 0.0 mm.

So the blade meshes stop colliding and a pad on each inner face does the
gripping: a flat, high-friction face, the way a rubber pad glued inside a
metal jaw would work. The pads are fitted to the blades they sit on, not to any
particular object, and they are drawn in the blade's own colour.

The pad thickness, friction and contact softness are prototype values chosen so
this task works in simulation. They are not measured from hardware, and no
padding has been fitted to the real robot.
"""

import mujoco
import numpy as np

from .arm import GRIPPER, FOLLOWER, GRIP_SITE, Gripper
from .parallel_gripper import visual_rgba

# Jaw command the pads are fitted at: both blades are near the object here, so
# the pad lands on the part of the blade that actually does the gripping.
FIT_COMMAND = 0.25
THICKNESS = 0.006      # m the pad stands proud of the blade, per jaw
HALF_SPAN = (0.013, THICKNESS / 2, 0.0165)  # half-extents: across, thick, along approach
# A 26 x 33 mm face, picked by sweeping pad size against all four cubes. A
# smaller face loses the 42 mm cube, which the blades grip above its centre
# and which then rolls out during the carry; a larger one fouls the 54 mm and
# 58 mm cubes on the way in.
FRICTION = (5.0, 0.02, 0.002)
SOLREF = (0.008, 1.0)  # slightly soft, so a pad settles onto a face instead of skidding


def _blade_surface(model, data, body_id, site_id, target):
    """The blade's inner face near the object, in the grip-site frame.

    Returns (points, inward) where `inward` is +1 when the blade sits on the
    low-y side of the jaw target and has to grow towards +y to reach it.
    """
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body_id or model.geom_dataid[geom] < 0:
            continue
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        world = (model.mesh_vert[start:start + count]
                 @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
        local = (world - data.site_xpos[site_id]) @ data.site_xmat[site_id].reshape(3, 3)
        band = local[(np.abs(local[:, 2] - target[2]) < 0.022)
                     & (np.abs(local[:, 0] - target[0]) < 0.016)]
        if not len(band):
            raise ValueError(f"blade {body_id} has no surface facing the jaw target")
        return band, 1.0 if band[:, 1].mean() < target[1] else -1.0
    raise ValueError(f"no mesh geom on body {body_id}")


def add_pads(spec, thickness=THICKNESS, friction=FRICTION, span=HALF_SPAN):
    """Give each supplied blade a contact pad and stop colliding the blade mesh."""
    probe = spec.compile()
    data = mujoco.MjData(probe)
    placements = []

    for side in ("right", "left"):
        prefix = "" if side == "right" else "l_"
        site_id = probe.site(GRIP_SITE[side]).id
        joints = [probe.jnt_qposadr[probe.joint(name).id]
                  for name in (GRIPPER[side], FOLLOWER[side])]
        data.qpos[:] = probe.qpos0
        data.qpos[joints] = FIT_COMMAND
        mujoco.mj_kinematics(probe, data)

        # where the jaws meet at this command: the midpoint of the two blade tips
        tips = []
        for finger in ("left", "right"):
            body = probe.body(f"{prefix}{finger}_finger__{finger}_finger").id
            for geom in range(probe.ngeom):
                if probe.geom_bodyid[geom] == body and probe.geom_dataid[geom] >= 0:
                    mesh = probe.geom_dataid[geom]
                    start, count = probe.mesh_vertadr[mesh], probe.mesh_vertnum[mesh]
                    world = (probe.mesh_vert[start:start + count]
                             @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
                    local = (world - data.site_xpos[site_id]) @ data.site_xmat[site_id].reshape(3, 3)
                    tips.append(local[np.argsort(local[:, 2])[-30:]].mean(0))
                    break
        target = (tips[0] + tips[1]) / 2

        for finger in ("left", "right"):
            name = f"{prefix}{finger}_finger__{finger}_finger"
            body_id = probe.body(name).id
            band, inward = _blade_surface(probe, data, body_id, site_id, target)
            face = band[:, 1].max() if inward > 0 else band[:, 1].min()
            centre = np.array([float(np.median(band[:, 0])),
                               face + inward * thickness / 2,
                               float(np.median(band[:, 2]))])
            # site frame -> that blade's own frame, so the pad rides with the jaw
            rotation = data.site_xmat[site_id].reshape(3, 3)
            world = data.site_xpos[site_id] + rotation @ centre
            body_rotation = data.xmat[body_id].reshape(3, 3)
            pos = body_rotation.T @ (world - data.xpos[body_id])
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (body_rotation.T @ rotation).ravel())
            placements.append((name, pos, quat))

    for name, pos, quat in placements:
        body = spec.body(name)
        for geom in list(body.geoms):
            # the CAD mesh stays as the visual; only its collision copy retires
            if geom.contype or geom.conaffinity:
                geom.contype = geom.conaffinity = 0
        body.add_geom(name=f"{name}_pad", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=pos, quat=quat,
                      size=[span[0], thickness / 2, span[2]],
                      mass=0.004, contype=2, conaffinity=1, condim=4,
                      friction=list(friction), solref=list(SOLREF),
                      rgba=visual_rgba(body), group=2)
    return spec


class PaddedGripper(Gripper):
    """Aperture measured between the pads, since the pads are what touch an object.

    The base class measures the blade tips. Once pads are fitted those tips are
    no longer the contact surface, and the blades are a pincer: commanded fully
    shut they scissor past each other rather than meeting, so a grip has to be
    commanded to the object's width instead of to zero.
    """

    def __init__(self, model, side, samples=21):
        super().__init__(model, side, samples)
        prefix = "" if side == "right" else "l_"
        pads = [model.geom(f"{prefix}{f}_finger__{f}_finger_pad").id for f in ("left", "right")]
        half = [model.geom_size[pad][1] for pad in pads]
        data = mujoco.MjData(model)
        adr = [model.jnt_qposadr[model.joint(GRIPPER[side]).id],
               model.jnt_qposadr[model.joint(FOLLOWER[side]).id]]
        for i, q in enumerate(self.q):
            data.qpos[:] = model.qpos0
            data.qpos[adr] = q
            mujoco.mj_kinematics(model, data)
            origin, rot = data.site_xpos[self.site], data.site_xmat[self.site].reshape(3, 3)
            centres = [(data.geom_xpos[pad] - origin) @ rot for pad in pads]
            # each pad's face is the one pointing at the other pad, not at y = 0:
            # the grip site does not sit between these blades.
            toward = 1.0 if centres[0][1] < centres[1][1] else -1.0
            faces = []
            for k, (centre, pad, thickness) in enumerate(zip(centres, pads, half)):
                across = (data.geom_xmat[pad].reshape(3, 3)[:, 1]) @ rot
                face = centre.copy()
                face[1] += (toward if k == 0 else -toward) * thickness * abs(across[1])
                faces.append(face)
            self.offset[i] = (faces[0] + faces[1]) / 2
            # negative once the blades have scissored past each other
            self.gap[i] = (faces[1][1] - faces[0][1]) * toward

    def grip_command(self, width, squeeze=0.004):
        """The command that closes the pads onto an object `width` across.

        Commanding shut would drive the blades past each other and flick the
        object out, so stop where the pads are `squeeze` inside its faces and
        let the force-limited servo press.
        """
        wanted = max(0.0, width - squeeze)
        # the gap only grows with the command above the point where the blades
        # cross, so interpolate on that rising stretch alone
        rising = np.argmax(self.gap)
        usable = slice(int(np.argmin(self.gap[:rising + 1])), rising + 1)
        return float(np.clip(np.interp(wanted, self.gap[usable], self.q[usable]), self.lo, self.hi))
