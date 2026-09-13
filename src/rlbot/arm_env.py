"""Continuous arm control. No teacher, clock or phase is used to select actions."""
import math
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from .manipulation import make_model, hand_for
from .arm import Arm, ArmIK, down_quat, GRIPPER, FOLLOWER, OPEN
from .room import TABLES

# Fully-open jaw command, and the factor that normalizes it for the observation.
# "urdf" is the supplied gripper as exported. "parallel" reproduces the sliding-jaw
# substitution exactly, including its 3.0 pad friction, so the checkpoint trained
# against it still runs; those numbers are prototype values, not measured hardware.
JAWS = {"padded": (OPEN, 1.0 / OPEN), "urdf": (OPEN, 1.0 / OPEN), "parallel": (.045, 20.0)}


class ArmEnv(gym.Env):
    def __init__(self, gripper='padded', station='pick'):
        self.gripper = gripper
        self.station = station
        table_index = 0 if station == 'pick' else 1
        self.object_name = 'pick_cube' if station == 'pick' else 'cube_m'
        table = TABLES[table_index]
        self.frame_yaw = table.yaw
        c, s = math.cos(table.yaw), math.sin(table.yaw)
        self.frame_rotation = np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        self.frame_origin = np.array([*table.centre,0.])
        self.reference_origin = np.array([-.20,-1.95,0.])
        self.model = make_model(item_name=self.object_name,gripper=gripper,table_index=table_index)
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)
        self.arm = Arm(self.model, self.data, 'right')
        self.quat = down_quat(self.frame_yaw)
        self.object_id = self.model.body(self.object_name).id
        self.objq = self.model.jnt_qposadr[self.model.body_jntadr[self.object_id]]
        self.objgeom = self.model.body_geomadr[self.object_id]
        self.fingers = {self.model.body(f + '_finger__' + f + '_finger').id for f in ('left', 'right')}
        self.jaw_open, self.jaw_scale = JAWS[gripper]
        if gripper == 'parallel':
            for geom in range(self.model.ngeom):
                if self.model.geom_bodyid[geom] in self.fingers:
                    self.model.geom_friction[geom, 0] = 3.0
            for side in ('left', 'right'):
                eq = self.model.equality(side + '_parallel_mimic').id
                self.model.eq_solref[eq] = [.004, 1]
                self.model.eq_solimp[eq] = [.99, .999, .001, .5, 2]
        self.gripq = self.model.jnt_qposadr[self.model.joint(GRIPPER['right']).id]
        # What "closed on the cube" is for this gripper, as a normalized action.
        # Parallel jaws stall when commanded shut; the supplied blades are a
        # pincer and would scissor past the cube, so they stop at its width.
        self.cube_width = 0.048
        self.hand = hand_for(self.model, 'right', gripper)
        closed = (self.hand.grip_command(self.cube_width)
                  if hasattr(self.hand, 'grip_command') else 0.0)
        if hasattr(self.hand, 'grip_command'):
            # Cap the command at the approach opening rather than wide open. The
            # supplied blades are a pincer: past about 0.6 their midpoint runs away
            # from the wrist (35 mm at full open) and the arm would have to drive
            # the hand through the tabletop to put the jaws on a cube.
            self.jaw_open = self.hand.opening_for(self.cube_width, clearance=0.035)
            self.jaw_scale = 1.0 / self.jaw_open
        self.closed_action = float(np.clip(2 * closed / self.jaw_open - 1, -1, 1))
        # A bounded command range prevents pincer blades crossing through the cube.
        # These are controller bounds, not altered URDF joints or force limits.
        self.jaw_closed = closed if gripper == 'padded' else 0.0
        if gripper == 'padded':
            self.closed_action = -1.0
        self.released_qpos = 0.82 * self.jaw_open
        # Fraction of the full 80 mm/s the demonstration teacher may use per phase.
        # The supplied blades grip at an angle and flick the cube out sideways if the
        # carry starts at full rate; the parallel jaws clamp square and do not care.
        self.carry_speed = ((1., 1., 1., .2, 1., 1., 1.) if gripper != 'parallel'
                            else (1., 1., 1., 1., 1., 1., 1.))
        # Jaw centre minus grip site, in world axes, fixed at the gripping command -
        # the pose the cube has to be square in. Over the working range the offset
        # barely moves (23.0 to 21.3 mm across the jaw), so one value serves, and
        # chasing it live would walk the arm while it grips. Zero for the parallel
        # substitute, which leaves that path untouched.
        shift = np.zeros(3)
        mujoco.mju_rotVecQuat(shift, np.asarray(self.hand.jaw_offset(closed), dtype=float),
                              self.quat)
        self.jaw_shift = shift
        self.action_space = spaces.Box(-1., 1., (4,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (20,), np.float32)
        self.horizon = 400

    def to_world(self, point):
        return self.frame_origin + self.frame_rotation @ (np.asarray(point)-self.reference_origin)

    def to_task(self, point):
        return self.reference_origin + self.frame_rotation.T @ (np.asarray(point)-self.frame_origin)

    def contacts(self):
        touching = set()
        for c in self.data.contact:
            bodies = {self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]}
            if self.object_id in bodies:
                touching |= bodies & self.fingers
        return len(touching)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.start = np.array([-.34, -1.71, .70])
        self.start[:2] += self.np_random.uniform(-.008, .008, 2)
        self.goal = self.start + [.17, 0, .024]
        self.data.qpos[self.objq:self.objq+3] = self.to_world(self.start)
        self.data.qpos[self.objq+3:self.objq+7] = [math.cos(self.frame_yaw/2),0,0,math.sin(self.frame_yaw/2)]
        for side in ('right', 'left'):
            self.data.qpos[ArmIK(self.model, side).qadr[3]] = math.pi / 2
        # open the jaws first: the jaw offset this pose is solved for depends on them
        for name in (GRIPPER['right'], FOLLOWER['right']):
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = self.jaw_open
        mujoco.mj_forward(self.model, self.data)
        self.target = self.start + [0, 0, .152]
        self.scratch.qpos[:] = self.data.qpos
        solution = self.arm.ik.solve(self.scratch, self.site_for_jaws(self.target), self.quat)
        if not solution.ok:
            raise RuntimeError('Reset pose unreachable')
        self.data.qpos[self.arm.ik.qadr] = solution.qpos
        for act in range(self.model.nu):
            j = self.model.actuator_trnid[act, 0]
            self.data.ctrl[act] = self.data.qpos[self.model.jnt_qposadr[j]]
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.max_lift = self.held = self.stable = 0.
        self.previous = np.zeros(4)
        self.potential = self._potential()
        return self._obs(), {}

    @property
    def cube(self):
        return self.to_task(self.data.geom_xpos[self.objgeom])

    @property
    def grip_point(self):
        """Where the jaws actually close, which is what has to reach the cube."""
        return self.to_task(self.arm.grip_pos + self.jaw_shift)

    def site_for_jaws(self, jaw_target):
        """The site command that puts the jaws on `jaw_target`."""
        return self.to_world(jaw_target) - self.jaw_shift

    def _obs(self):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.object_id, velocity, 0)
        grip = self.grip_point
        return np.concatenate([(grip - self.start)*10,
            (self.cube-grip)*10, (self.goal-self.cube)*10,
            (self.frame_rotation.T @ velocity[3:])*10, [self.data.qpos[self.gripq]*self.jaw_scale, self.contacts()/2],
            (self.target-grip)*10, self.previous[:3]]).astype(np.float32)

    def _potential(self):
        reach = np.linalg.norm(self.cube + [0,0,.008] - self.grip_point)
        lift = np.clip((self.cube[2]-.724)/.12, 0, 1)
        distance = np.linalg.norm(self.cube[:2]-self.goal[:2])
        return -3*reach + 2*lift + 4*(.17-distance)

    def step(self, action):
        action = np.clip(np.asarray(action), -1, 1)
        # the box bounds where the jaws may go, so it means the same for any gripper
        self.target = np.clip(self.target + action[:3]*.004,
                              [-.42,-1.79,.731], [-.09,-1.63,.94])
        self.scratch.qpos[:] = self.data.qpos
        solution = self.arm.ik.solve(self.scratch, self.site_for_jaws(self.target), self.quat,
                                    seed=self.data.ctrl[self.arm.acts], iters=12, restarts=1)
        self.arm.hold(solution.qpos)
        jaw_target = self.jaw_closed + (action[3]+1)/2*(self.jaw_open-self.jaw_closed)
        if self.gripper == 'padded':
            # Rate-limit the requested angle, like a real motor command ramp.
            current = self.data.ctrl[self.arm.grip_act]
            jaw_target = np.clip(jaw_target, current-.015, current+.015)
        self.arm.grip(jaw_target)
        for _ in range(25):
            mujoco.mj_step(self.model, self.data)
            lift = float(self.cube[2]-.724)
            self.max_lift = max(self.max_lift, lift)
            if lift > .05 and self.contacts() == 2:
                self.held += self.model.opt.timestep
        self.steps += 1
        self.previous = action.copy()
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.object_id, velocity, 0)
        distance = float(np.linalg.norm(self.cube[:2]-self.goal[:2]))
        stable = (distance < .025 and abs(self.cube[2]-.724)<.008 and self.contacts()==0
                  and np.linalg.norm(velocity[3:])<.025 and self.grip_point[2]>.80)
        self.stable = self.stable+.05 if stable else 0.
        success = self.stable>=.5 and self.max_lift>.08 and self.held>.5
        potential = self._potential()
        reward = 10*(potential-self.potential)-.01 + (30 if success else 0)
        self.potential = potential
        failed = self.cube[2]<.65
        info = dict(success=bool(success), distance=distance, max_lift=self.max_lift,
                    held_seconds=self.held, stable_seconds=self.stable, steps=self.steps)
        return self._obs(), float(reward), bool(success or failed), self.steps>=self.horizon, info
