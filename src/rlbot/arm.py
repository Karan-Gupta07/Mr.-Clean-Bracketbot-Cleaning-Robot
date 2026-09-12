"""Drive one arm's grip site to a pose: inverse kinematics, and a grasp sequence.

The BracketBot's arm is 7 DOF - a 1.03 m vertical rail plus six hinges - so most
targets have a continuum of solutions and a plain Jacobian step wanders into a
joint limit.  Damped least squares with a limit clamp and a few random restarts
is enough here, and it is honest about failing: `solve` returns the residual, so
a caller can tell "reached it" from "got within 4 cm and stopped".
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import mujoco
import numpy as np

# rail + six hinges, per side; the gripper is not part of the reach
ARM_JOINTS = {
    "right": ["rj0", "rj1", "rj2", "rj3", "rj4", "rj5", "rj6"],
    "left": ["lj0", "lj1", "lj2", "lj3", "lj4", "lj5", "lj6"],
}
GRIPPER = {"right": "right_left_gripper", "left": "left_left_gripper"}
# the other blade of each hand.  It carries no servo - the URDF <mimic> became an
# equality constraint, which the solver enforces during dynamics but mj_kinematics
# does not, so anything posing the hand by hand has to set both.
FOLLOWER = {"right": "right_right_gripper", "left": "left_right_gripper"}
GRIP_SITE = {"right": "grip_right", "left": "grip_left"}

OPEN, SHUT = 1.0, 0.0          # gripper command: 195 mm at the tips, or closed


def down_quat(yaw: float = 0.0) -> np.ndarray:
    """A grip frame pointing straight down, fingers closing across `yaw`.

    The site's +z is the approach direction, so 'down' is +z -> -z world: a 180
    degree turn about x, then `yaw` about the world z to aim the closing axis.
    """
    flip = np.array([0.0, 1.0, 0.0, 0.0])
    spin = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, spin, flip)
    return out


@dataclass
class Solution:
    qpos: np.ndarray       # the arm's joint values, in ARM_JOINTS order
    pos_err: float         # m
    rot_err: float         # rad

    @property
    def ok(self) -> bool:
        return self.pos_err < 0.01 and self.rot_err < 0.25


class ArmIK:
    """Damped least squares on one arm, targeting its grip site."""

    def __init__(self, model, side: str, pos_weight: float = 1.0,
                 rot_weight: float = 0.35):
        self.model = model
        self.side = side
        self.names = ARM_JOINTS[side]
        self.jids = [model.joint(n).id for n in self.names]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.dofs = np.array([model.jnt_dofadr[j] for j in self.jids])
        self.lo = np.array([model.jnt_range[j][0] for j in self.jids])
        self.hi = np.array([model.jnt_range[j][1] for j in self.jids])
        self.site = model.site(GRIP_SITE[side]).id
        self.w = np.array([pos_weight] * 3 + [rot_weight] * 3)

    def error(self, data, pos, quat):
        """[position, orientation] error of the grip site, in world axes."""
        e = np.zeros(6)
        e[:3] = np.asarray(pos) - data.site_xpos[self.site]
        cur = np.zeros(4)
        mujoco.mju_mat2Quat(cur, data.site_xmat[self.site])
        neg, diff = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(neg, cur)
        mujoco.mju_mulQuat(diff, np.asarray(quat, dtype=float), neg)
        mujoco.mju_quat2Vel(e[3:], diff, 1.0)
        return e

    def solve(self, data, pos, quat, seed=None, iters: int = 120,
              damping: float = 0.08, restarts: int = 12, rng=None) -> Solution:
        """Joint values that put the grip site at (pos, quat).

        Restarts from a fresh random pose whenever a run stalls, and keeps the
        best result rather than the last one - a stalled run can be a long way
        off, and silently returning it is how an arm ends up inside a table.
        """
        rng = rng or np.random.default_rng(0)
        model = self.model
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        start = np.array(data.qpos[self.qadr]) if seed is None else np.asarray(seed)
        best = None

        for attempt in range(restarts):
            q = start if attempt == 0 else rng.uniform(self.lo, self.hi)
            q = np.clip(q, self.lo, self.hi)
            for _ in range(iters):
                data.qpos[self.qadr] = q
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                err = self.error(data, pos, quat) * self.w

                mujoco.mj_jacSite(model, data, jacp, jacr, self.site)
                jac = np.vstack([jacp, jacr])[:, self.dofs] * self.w[:, None]

                # dq = J^T (J J^T + lambda^2 I)^-1 e
                jjt = jac @ jac.T + (damping ** 2) * np.eye(6)
                step = jac.T @ np.linalg.solve(jjt, err)
                q = np.clip(q + step, self.lo, self.hi)

                raw = self.error(data, pos, quat)
                if np.linalg.norm(raw[:3]) < 1e-4 and np.linalg.norm(raw[3:]) < 1e-3:
                    break

            data.qpos[self.qadr] = q
            mujoco.mj_kinematics(model, data)
            raw = self.error(data, pos, quat)
            got = Solution(q.copy(), float(np.linalg.norm(raw[:3])),
                           float(np.linalg.norm(raw[3:])))
            if best is None or (got.pos_err, got.rot_err) < (best.pos_err, best.rot_err):
                best = got
            if best.ok:
                break

        data.qpos[self.qadr] = best.qpos
        mujoco.mj_kinematics(model, data)
        return best


class Gripper:
    """Where the jaws are, as a function of the gripper command.

    The blades are not symmetric about the grip site and they do not swing in a
    plane: opening from shut to wide slides the jaw centre sideways and pulls it
    back up the approach axis.  Aim the site itself at an object and the object
    lands off-centre in a jaw shallower than it looks, and one blade reaches it
    first and flicks it away.

    So measure, once, off the pads the build script laid on the blades: for each
    gripper command, where the two gripping faces are in the site's own frame.
    The planner aims the *jaws* and this corrects the site target.
    """

    def __init__(self, model, side: str, samples: int = 25):
        self.model, self.side = model, side
        self.site = model.site(GRIP_SITE[side]).id
        data = mujoco.MjData(model)
        joint = model.joint(GRIPPER[side]).id
        adr = [model.jnt_qposadr[joint],
               model.jnt_qposadr[model.joint(FOLLOWER[side]).id]]
        self.lo, self.hi = model.jnt_range[joint]

        prefix = "" if side == "right" else "l_"
        self.pads = [model.geom(f"{prefix}{f}_finger__{f}_finger_pad0").id
                     for f in ("left", "right")]
        self.q = np.linspace(self.lo, self.hi, samples)
        self.centre = np.zeros((samples, 3))   # jaw centre, in the site frame
        self.gap = np.zeros(samples)           # between the pad faces, m
        self.splay = np.zeros(samples)         # how far off facing each other, rad

        for i, q in enumerate(self.q):
            data.qpos[:] = model.qpos0
            data.qpos[adr] = q
            mujoco.mj_kinematics(model, data)
            faces = self._faces(model, data)
            self.centre[i] = faces.mean(0)
            self.gap[i] = self._narrowest(model, data)
            self.splay[i] = self._splay(model, data)

    def _faces(self, model, data):
        """The two pads' gripping faces, in the site frame."""
        origin = data.site_xpos[self.site]
        rot = data.site_xmat[self.site].reshape(3, 3)
        centres = [rot.T @ (data.geom_xpos[pad] - origin) for pad in self.pads]
        out = []
        for pad, centre, other in zip(self.pads, centres, centres[::-1]):
            # the pad's own y is its thin axis; take the face pointing at the
            # opposite pad, not at the site, which is not between them
            normal = rot.T @ data.geom_xmat[pad].reshape(3, 3)[:, 1]
            normal = normal * (1.0 if normal @ (other - centre) > 0 else -1.0)
            out.append(centre + normal * model.geom_size[pad][1])
        return np.array(sorted(out, key=lambda f: f[1]))

    def _splay(self, model, data) -> float:
        """Angle between the two pad faces.

        These blades do not swing in a plane.  Open the hand wide and the faces
        turn outward until they are no longer looking at each other at all - at
        full travel they are 65 degrees apart, and an object between them is not
        between anything.  A grasp only means something while this stays small.
        """
        normals = [data.geom_xmat[pad].reshape(3, 3)[:, 1] for pad in self.pads]
        return float(math.pi - math.acos(np.clip(-abs(normals[0] @ normals[1]),
                                                 -1.0, 1.0)))

    def _narrowest(self, model, data) -> float:
        """Closest approach of the two pads, corner to corner.

        The pads are not parallel - they lean in towards the throat - so the gap
        between their centres overstates what fits between them by about a
        quarter.  What an object has to clear is the narrowest point.
        """
        boxes = []
        for pad in self.pads:
            half = model.geom_size[pad]
            rot = data.geom_xmat[pad].reshape(3, 3)
            signs = np.array(list(itertools.product((-1, 1), repeat=3)))
            boxes.append(data.geom_xpos[pad] + (signs * half) @ rot.T)
        return float(np.linalg.norm(boxes[0][:, None, :] - boxes[1][None, :, :],
                                    axis=-1).min())

    MAX_SPLAY = math.radians(35)   # past this the faces are not opposed enough

    @property
    def widest(self) -> float:
        """The most this hand can hold with its faces still opposed."""
        usable = self.gap[self.splay <= self.MAX_SPLAY]
        return float(usable.max()) if len(usable) else 0.0

    def opening_for(self, width: float, clearance: float = 0.012):
        """The command that leaves `clearance` of daylight around an object.

        Wider is not always more open: the blades splay as they go, so the gap
        peaks and the faces stop opposing.  Take the smallest command that fits,
        and refuse anything that only fits with the hand splayed open.
        """
        wanted = width + clearance
        peak = int(np.argmax(self.gap))
        if wanted > self.gap[peak]:
            return None
        opening = float(np.interp(wanted, self.gap[: peak + 1], self.q[: peak + 1]))
        if np.interp(opening, self.q, self.splay) > self.MAX_SPLAY:
            return None
        return opening

    def jaw_offset(self, opening: float) -> np.ndarray:
        """Jaw centre at that opening, in the site frame."""
        return np.array([np.interp(opening, self.q, self.centre[:, k])
                         for k in range(3)])

    def site_target(self, jaw_target, quat, opening: float) -> np.ndarray:
        """Where to send the site so the jaws land on `jaw_target`."""
        world = np.zeros(3)
        mujoco.mju_rotVecQuat(world, self.jaw_offset(opening),
                              np.asarray(quat, dtype=float))
        return np.asarray(jaw_target, dtype=float) - world


class Arm:
    """An arm plugged into a running simulation: hold a pose, work a gripper."""

    def __init__(self, model, data, side: str):
        self.model, self.data, self.side = model, data, side
        self.ik = ArmIK(model, side)
        self.acts = np.array([model.actuator(n).id for n in ARM_JOINTS[side]])
        self.grip_act = model.actuator(GRIPPER[side]).id
        self.site = model.site(GRIP_SITE[side]).id

    @property
    def grip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.site].copy()

    def hold(self, qpos) -> None:
        """Command the servos to a pose (they are position servos, so this is
        a target, not a teleport)."""
        self.data.ctrl[self.acts] = qpos

    def grip(self, amount: float) -> None:
        self.data.ctrl[self.grip_act] = amount

    def plan(self, pos, quat, seed=None, **kw) -> Solution:
        """IK against a scratch copy of the state, leaving the sim untouched."""
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        mujoco.mj_kinematics(self.model, scratch)
        return self.ik.solve(scratch, pos, quat, seed=seed, **kw)
