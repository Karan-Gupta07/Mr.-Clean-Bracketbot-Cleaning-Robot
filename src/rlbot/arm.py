"""Drive one arm's grip site to a pose: inverse kinematics, and a grasp sequence.

The BracketBot's arm is 7 DOF - a 1.03 m vertical rail plus six hinges - so most
targets have a continuum of solutions and a plain Jacobian step wanders into a
joint limit.  Damped least squares with a limit clamp and a few random restarts
is enough here, and it is honest about failing: `solve` returns the residual, so
a caller can tell "reached it" from "got within 4 cm and stopped".
"""

from __future__ import annotations

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
              damping: float = 0.08, restarts: int = 6, rng=None) -> Solution:
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
    """Where the jaws actually are, as a function of the gripper command.

    The two blades are not symmetric about the grip site and they do not swing
    in a plane: opening from shut to wide slides the midpoint of the fingertips
    25 mm sideways and pulls it 38 mm back up the approach axis.  Aim the site
    itself at an object and the object ends up off-centre in a jaw that is not
    as deep as it looks - one blade reaches it first and flicks it away.

    So measure, once, off the model: for each gripper command, where the tip
    midpoint sits in the site's own frame and how far apart the tips are.  The
    planner then aims the *jaws* and lets this correct the site target.
    """

    def __init__(self, model, side: str, samples: int = 21):
        self.model, self.side = model, side
        self.site = model.site(GRIP_SITE[side]).id
        data = mujoco.MjData(model)
        joint = model.joint(GRIPPER[side]).id
        adr = [model.jnt_qposadr[joint],
               model.jnt_qposadr[model.joint(FOLLOWER[side]).id]]
        self.lo, self.hi = model.jnt_range[joint]

        prefix = "" if side == "right" else "l_"
        fingers = [model.body(f"{prefix}{f}_finger__{f}_finger").id
                   for f in ("left", "right")]
        self.q = np.linspace(self.lo, self.hi, samples)
        self.offset = np.zeros((samples, 3))    # tip midpoint, in the site frame
        self.gap = np.zeros(samples)            # between the tips, m

        for i, q in enumerate(self.q):
            data.qpos[:] = model.qpos0
            data.qpos[adr] = q
            mujoco.mj_kinematics(model, data)
            origin = data.site_xpos[self.site]
            rot = data.site_xmat[self.site].reshape(3, 3)
            tips = [self._tip(model, data, body, origin, rot) for body in fingers]
            self.offset[i] = (tips[0] + tips[1]) / 2
            self.gap[i] = abs(tips[0][1] - tips[1][1])

    @staticmethod
    def _tip(model, data, body_id, origin, rot, take: int = 30):
        """The blade's far end, in the site frame: +z is the approach axis."""
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != body_id or model.geom_dataid[g] < 0:
                continue
            mesh = model.geom_dataid[g]
            adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            verts = model.mesh_vert[adr : adr + num]
            world = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            local = (world - origin) @ rot
            return local[np.argsort(local[:, 2])[-take:]].mean(0)
        raise ValueError(f"no mesh geom on body {body_id}")

    def opening_for(self, width: float, clearance: float = 0.030) -> float:
        """The command that leaves `clearance` of daylight around an object."""
        wanted = width + clearance
        if wanted >= self.gap[-1]:
            return float(self.hi)
        return float(np.interp(wanted, self.gap, self.q))

    def jaw_offset(self, opening: float) -> np.ndarray:
        """Tip midpoint at that opening, in the site frame."""
        return np.array([np.interp(opening, self.q, self.offset[:, k])
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
