"""Continuous arm control. No teacher, clock or phase is used to select actions."""
import math
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from .manipulation import make_model
from .arm import Arm, ArmIK, down_quat, GRIPPER, FOLLOWER


class ArmEnv(gym.Env):
    def __init__(self):
        self.model = make_model()
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)
        self.arm = Arm(self.model, self.data, 'right')
        self.quat = down_quat(0)
        self.object_id = self.model.body('cube_m').id
        self.objq = self.model.jnt_qposadr[self.model.body_jntadr[self.object_id]]
        self.objgeom = self.model.body_geomadr[self.object_id]
        self.fingers = {self.model.body(f + '_finger__' + f + '_finger').id for f in ('left', 'right')}
        # Prototype high-friction pads; this coefficient is not measured hardware data.
        for geom in range(self.model.ngeom):
            if self.model.geom_bodyid[geom] in self.fingers:
                self.model.geom_friction[geom, 0] = 3.0
        self.gripq = self.model.jnt_qposadr[self.model.joint(GRIPPER['right']).id]
        for side in ('left', 'right'):
            eq = self.model.equality(side + '_parallel_mimic').id
            self.model.eq_solref[eq] = [.004, 1]
            self.model.eq_solimp[eq] = [.99, .999, .001, .5, 2]
        self.action_space = spaces.Box(-1., 1., (4,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (20,), np.float32)
        self.horizon = 400

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
        self.data.qpos[self.objq:self.objq+3] = self.start
        for side in ('right', 'left'):
            self.data.qpos[ArmIK(self.model, side).qadr[3]] = math.pi / 2
        self.target = self.start + [0, 0, .152]
        self.scratch.qpos[:] = self.data.qpos
        solution = self.arm.ik.solve(self.scratch, self.target, self.quat)
        if not solution.ok:
            raise RuntimeError('Reset pose unreachable')
        self.data.qpos[self.arm.ik.qadr] = solution.qpos
        for name in (GRIPPER['right'], FOLLOWER['right']):
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = .045
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
        return self.data.geom_xpos[self.objgeom].copy()

    def _obs(self):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.object_id, velocity, 0)
        return np.concatenate([(self.arm.grip_pos - self.start)*10,
            (self.cube-self.arm.grip_pos)*10, (self.goal-self.cube)*10,
            velocity[3:]*10, [self.data.qpos[self.gripq]*20, self.contacts()/2],
            (self.target-self.arm.grip_pos)*10, self.previous[:3]]).astype(np.float32)

    def _potential(self):
        reach = np.linalg.norm(self.cube + [0,0,.008] - self.arm.grip_pos)
        lift = np.clip((self.cube[2]-.724)/.12, 0, 1)
        distance = np.linalg.norm(self.cube[:2]-self.goal[:2])
        return -3*reach + 2*lift + 4*(.17-distance)

    def step(self, action):
        action = np.clip(np.asarray(action), -1, 1)
        self.target = np.clip(self.target + action[:3]*.004,
                              [-.42,-1.79,.731], [-.09,-1.63,.94])
        self.scratch.qpos[:] = self.data.qpos
        solution = self.arm.ik.solve(self.scratch, self.target, self.quat,
                                    seed=self.data.ctrl[self.arm.acts], iters=12, restarts=1)
        self.arm.hold(solution.qpos)
        self.arm.grip((action[3]+1)*.0225)
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
                  and np.linalg.norm(velocity[3:])<.025 and self.arm.grip_pos[2]>.80)
        self.stable = self.stable+.05 if stable else 0.
        success = self.stable>=.5 and self.max_lift>.08 and self.held>.5
        potential = self._potential()
        reward = 10*(potential-self.potential)-.01 + (30 if success else 0)
        self.potential = potential
        failed = self.cube[2]<.65
        info = dict(success=bool(success), distance=distance, max_lift=self.max_lift,
                    held_seconds=self.held, stable_seconds=self.stable, steps=self.steps)
        return self._obs(), float(reward), bool(success or failed), self.steps>=self.horizon, info
