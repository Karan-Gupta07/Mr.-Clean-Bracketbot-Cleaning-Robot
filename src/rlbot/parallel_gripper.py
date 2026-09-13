"""Explicit parallel-jaw simulation variant; not the supplied hooked gripper."""
import mujoco
import numpy as np
from .arm import GRIPPER, FOLLOWER, GRIP_SITE, Gripper


def visual_rgba(body, fallback=(1., 1., 1., 1.)):
    """The colour of a supplied link, so a substitute part can be drawn to match."""
    for geom in body.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.group == 2:
            return np.array(geom.rgba, dtype=float)
    return np.array(fallback, dtype=float)


def replace_grippers(spec):
    """Keep arms and grip frames; replace the unreliable CAD jaws with slides.

    The 100 mm aperture, 18 x 8 x 50 mm finger pads and 8 N drive limit are
    prototype parameters. The substitute parts take their colour from the
    supplied finger and hand meshes, so renders show the original gripper's
    appearance; the geometry is still a substitution, and nothing on screen
    distinguishes it. Report the replacement in text rather than relying on it
    being visible.
    """
    for side in ("right", "left"):
        prefix = "" if side == "right" else "l_"
        hand = spec.body(prefix + "hand__hand")
        site = spec.site(GRIP_SITE[side])
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, site.quat)
        rotation = rotation.reshape(3,3)
        # Read the supplied colours before the CAD finger bodies are deleted.
        finger_rgba = visual_rgba(spec.body(prefix + "left_finger__left_finger"))
        hand_rgba = visual_rgba(hand)
        for f, sign, name in (("left", -1, GRIPPER[side]), ("right", 1, FOLLOWER[side])):
            old_name = prefix + f + "_finger__" + f + "_finger"
            spec.delete(spec.body(old_name))
            body = hand.add_body(name=old_name, pos=site.pos+rotation@np.array([0,sign*.004,0]), quat=site.quat)
            body.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0,sign,0],
                           range=[0,.05], limited=True, damping=1.0, armature=.002)
            body.add_geom(name=f"{side}_{f}_pad", type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[.009,.004,.025], mass=.03, contype=2, conaffinity=1, condim=4,
                          friction=[1.2,.02,.002], rgba=finger_rgba, group=2)
            body.add_geom(name=f"{side}_{f}_stem", type=mujoco.mjtGeom.mjGEOM_BOX,
                          pos=[0,0,-.045], size=[.006,.004,.035], mass=.011,
                          contype=2, conaffinity=1, rgba=finger_rgba, group=2)
        hand.add_geom(name=f"{side}_gripper_rail", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=site.pos+rotation@np.array([0,0,-.08]), quat=site.quat,
                      size=[.014,.06,.008], mass=0, contype=2, conaffinity=1,
                      rgba=hand_rgba, group=2)
        actuator = spec.add_actuator(name=GRIPPER[side], target=GRIPPER[side],
                                    trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                    biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                                    ctrllimited=True, forcelimited=True)
        actuator.ctrlrange = [0,.05]
        actuator.forcerange = [-8,8]
        actuator.gainprm = [800,0,0,0,0,0,0,0,0,0]
        actuator.biasprm = [0,-800,-8,0,0,0,0,0,0,0]
        spec.add_equality(name=f"{side}_parallel_mimic", type=mujoco.mjtEq.mjEQ_JOINT,
                          name1=FOLLOWER[side], name2=GRIPPER[side],
                          data=[0,1,0,0,0,0,0,0,0,0,1], solref=[.004,1])
    return spec


class ParallelGripper(Gripper):
    def __init__(self, model, side):
        self.model, self.side = model, side
        self.lo, self.hi = 0.0, 0.05
        self.q = np.array([0.0, .05])
        self.gap = 2*self.q
        self.offset = np.zeros((2,3))
