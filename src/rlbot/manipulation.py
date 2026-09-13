"""Contact-based pick and place with a fixed base and an IK/servo controller.

This is the manipulation baseline. No object attachment or state teleportation
is used during a rollout. The navigation neural checkpoint does not control arms.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import mujoco
import numpy as np

from .arm import Arm, ArmIK, GRIPPER, OPEN, Gripper, down_quat
from .gripper_pads import add_pads, PaddedGripper
from .parallel_gripper import replace_grippers, ParallelGripper
from .robot import ROOM
from .room import TABLES

# "padded" is the supplied hooked gripper with a contact pad on each blade: the
# CAD meshes are untouched and still the visuals, so the hand looks the same.
# "urdf" is that same gripper with nothing added, which does not grasp.
# "parallel" is the sliding-jaw substitution, kept only so the recorded RL
# results in docs/arm_rl.md stay reproducible; it is not the robot's hardware.
GRIPPERS = ("padded", "urdf", "parallel")


def hand_for(model, side, gripper="padded"):
    """The jaw-geometry model matching however this model's gripper was built."""
    return {"parallel": ParallelGripper, "padded": PaddedGripper}.get(gripper, Gripper)(model, side)


def make_model(item_name="cube_m", gripper="padded"):
    if gripper not in GRIPPERS:
        raise ValueError(f"gripper must be one of {GRIPPERS}")
    table = TABLES[1]
    spec = mujoco.MjSpec.from_file(str(ROOM))
    for key in list(spec.keys):
        spec.delete(key)
    root = spec.body("root")
    for joint in list(root.joints):
        spec.delete(joint)
    x, y, yaw = table.dock
    root.pos = [x, y, 0]
    root.quat = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    # Clear the destination on this table; other furniture remains unchanged.
    for item in table.items:
        if item.name != item_name:
            spec.delete(spec.body(item.name))
    if gripper == "parallel":
        replace_grippers(spec)
    elif gripper == "padded":
        add_pads(spec)
    return spec.compile()


@dataclass
class Phase:
    name: str
    q: np.ndarray
    grip: float
    seconds: float


class PickPlace:
    def __init__(self, seed=0, model=None, grasp_offset=(0, 0, 0), grip_force=None,
                 item_name="cube_m", gripper="padded"):
        self.gripper = gripper
        self.model = make_model(item_name, gripper) if model is None else model
        self.data = mujoco.MjData(self.model)
        self.seed = seed
        self.grasp_offset = np.asarray(grasp_offset, dtype=float)
        if grip_force is not None:
            # An explicit servo limit from the caller; None keeps the URDF's own.
            for name in GRIPPER.values():
                self.model.actuator_forcerange[self.model.actuator(name).id] = [-grip_force, grip_force]
        if gripper == "parallel":
            # Drive and couple the substitute slides; the URDF gripper keeps its
            # exported servo gains and its own <mimic> equality untouched.
            for name in GRIPPER.values():
                act = self.model.actuator(name).id
                self.model.actuator_gainprm[act, 0] = 800.0
                self.model.actuator_biasprm[act, 1:3] = [-800.0, -8.0]
            for side in ("left", "right"):
                equality = self.model.equality(f"{side}_parallel_mimic").id
                self.model.eq_solref[equality] = [0.004, 1.0]
                self.model.eq_solimp[equality] = [0.99, 0.999, 0.001, 0.5, 2]
        self.rng = np.random.default_rng(seed)
        self.object_name = item_name
        self.item = next(item for item in TABLES[1].items if item.name == item_name)
        self.object_id = self.model.body(self.object_name).id
        self.object_qadr = self.model.jnt_qposadr[self.model.body_jntadr[self.object_id]]
        self.start = np.array([-0.34, -1.71, 0.70])
        self.start[:2] += self.rng.uniform(-0.008, 0.008, 2)
        self.goal = self.start + [0.17, 0, 0]
        self.data.qpos[self.object_qadr:self.object_qadr+3] = self.start
        for side in ("right", "left"):
            self.data.qpos[ArmIK(self.model, side).qadr[3]] = math.pi / 2
        mujoco.mj_forward(self.model, self.data)
        for act in range(self.model.nu):
            joint = self.model.actuator_trnid[act, 0]
            self.data.ctrl[act] = self.data.qpos[self.model.jnt_qposadr[joint]]
        self.phases = self.plan()
        self.phase_index = 0
        self.phase_elapsed = 0.0
        self.phase_start = self.data.ctrl[self.arm.acts].copy()
        self.grip_start = float(self.data.ctrl[self.arm.grip_act])
        self.max_lift = 0.0
        self.lift_contact = False
        self.held_lift_seconds = 0.0
        self.stable_place_seconds = 0.0
        self.finished = False

    def plan(self):
        best = None
        initial = self.data.qpos.copy()
        for side in ("right", "left"):
            hand = hand_for(self.model, side, self.gripper)
            opening = hand.opening_for(self.item.width, clearance=0.035)
            for yaw in (0, math.pi / 2):
                quat = down_quat(TABLES[1].dock[2] + yaw)
                jaw_height = max(0.032, self.item.height / 2)
                jaws = self.start + [0, 0, jaw_height] + self.grasp_offset
                destination = self.goal + [0, 0, jaw_height] + self.grasp_offset
                targets = [jaws + [0,0,.12], jaws, jaws + [0,0,.15],
                           destination + [0,0,.15], destination, destination + [0,0,.12]]
                ik = ArmIK(self.model, side)
                scratch = mujoco.MjData(self.model)
                scratch.qpos[:] = initial
                mujoco.mj_kinematics(self.model, scratch)
                chain = []
                for index, target in enumerate(targets):
                    solution = ik.solve(scratch, hand.site_target(target, quat, opening), quat,
                                        seed=chain[-1].qpos if chain else None,
                                        restarts=1 if chain else 12)
                    if not solution.ok:
                        break
                    if chain and np.abs(solution.qpos - chain[-1].qpos).max() > 0.8:
                        break
                    chain.append(solution)
                if len(chain) != len(targets):
                    continue
                score = sum(x.pos_err for x in chain)
                if best is None or score < best[0]:
                    best = (score, side, opening, quat, chain)
        if best is None:
            raise RuntimeError("No continuous IK pick/place path")
        self.ik_error, self.side, self.opening, self.quat, chain = best
        self.arm = Arm(self.model, self.data, self.side)
        self.hand = hand_for(self.model, self.side, self.gripper)
        a, b, c, d, e, f = [x.qpos for x in chain]
        # Parallel jaws stall on the object when commanded shut. The supplied
        # blades are a pincer and would scissor past it, so grip to its width.
        shut = (self.hand.grip_command(self.item.width)
                if hasattr(self.hand, "grip_command") else 0.0)
        self.shut = shut
        return [Phase("approach", a, self.opening, 3.5),
                Phase("descend", b, self.opening, 2.0),
                Phase("grasp", b, shut, 1.5),
                Phase("lift", c, shut, 2.5),
                Phase("transfer", d, shut, 2.5),
                Phase("lower", e, shut, 2.0),
                Phase("release", e, self.opening, 1.5),
                Phase("retreat", f, self.opening, 2.0),
                Phase("verify", f, self.opening, 1.0)]

    @property
    def phase(self):
        return self.phases[self.phase_index].name

    def contacts(self):
        prefix = "" if self.side == "right" else "l_"
        fingers = {self.model.body(prefix + f + "_finger__" + f + "_finger").id for f in ("left", "right")}
        touching = set()
        for contact in self.data.contact:
            bodies = {self.model.geom_bodyid[g] for g in (contact.geom1, contact.geom2)}
            if self.object_id in bodies:
                touching |= bodies & fingers
        return len(touching)

    def step(self):
        if self.finished:
            return
        phase = self.phases[self.phase_index]
        self.phase_elapsed += self.model.opt.timestep
        # Finish the ramp early and give the real servos time to settle.
        fraction = min(1.0, self.phase_elapsed / (phase.seconds * 0.65))
        fraction = fraction * fraction * (3 - 2 * fraction)
        self.arm.hold(self.phase_start + (phase.q - self.phase_start) * fraction)
        self.arm.grip(self.grip_start + (phase.grip - self.grip_start) * fraction)
        mujoco.mj_step(self.model, self.data)
        height = float(self.data.xpos[self.object_id, 2] - self.start[2])
        self.max_lift = max(self.max_lift, height)
        if height > 0.05 and self.contacts() == 2:
            self.lift_contact = True
            self.held_lift_seconds += self.model.opt.timestep
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.object_id, velocity, 0)
        position = self.data.xpos[self.object_id]
        stable = (self.phase == "verify" and self.contacts() == 0
                  and np.linalg.norm(position[:2] - self.goal[:2]) < 0.03
                  and abs(position[2] - self.goal[2]) < 0.008
                  and np.linalg.norm(velocity[3:]) < 0.025)
        self.stable_place_seconds = self.stable_place_seconds + self.model.opt.timestep if stable else 0.0
        if self.phase_elapsed + 1e-9 >= phase.seconds:
            if self.phase_index + 1 == len(self.phases):
                self.finished = True
            else:
                self.phase_index += 1
                self.phase_elapsed = 0
                self.phase_start = self.data.ctrl[self.arm.acts].copy()
                self.grip_start = float(self.data.ctrl[self.arm.grip_act])

    def report(self):
        pos = self.data.xpos[self.object_id].copy()
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.object_id, velocity, 0)
        distance = float(np.linalg.norm(pos[:2] - self.goal[:2]))
        on_table = abs(float(pos[2] - self.goal[2])) < 0.008
        released = self.contacts() == 0
        stopped = np.linalg.norm(velocity[3:]) < 0.025
        success = (self.finished and self.max_lift > 0.08 and self.held_lift_seconds > 0.5
                   and self.stable_place_seconds >= 0.5 and distance < 0.03 and on_table and released and stopped)
        return {"success": bool(success), "seed": self.seed, "phase": self.phase,
                "time": float(self.data.time), "controller": "scripted IK and joint servos",
                "base": "fixed at docking pose", "side": self.side,
                "gripper": {"padded": "supplied hooked gripper with contact pads",
                            "urdf": "supplied hooked gripper, unmodified",
                            "parallel": "parallel-jaw simulation replacement"}[self.gripper],
                "object_width": self.item.width,
                "object": self.object_name, "start": self.start.tolist(), "goal": self.goal.tolist(),
                "position": pos.tolist(), "distance": distance, "max_lift": self.max_lift,
                "two_finger_contact_during_lift": self.lift_contact, "released": released,
                "held_lift_seconds": self.held_lift_seconds, "stable_place_seconds": self.stable_place_seconds,
                "on_table": on_table, "stopped": bool(stopped), "speed": float(np.linalg.norm(velocity[3:])),
                "finger_contacts": self.contacts(), "ik_position_error_sum": self.ik_error}
