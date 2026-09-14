"""Flybrain's arm environment, run on the shared live simulation.

`ArmEnv` builds its own room: the base welded at the docking pose, pads glued
onto the blades, and every episode starting from `mj_resetData` with the cube
teleported to a fixed spot. None of that is available once one model is driven
from table to table - there is one `MjData`, the robot got there by driving,
and resetting it would throw away the demo.

So this keeps `ArmEnv`'s observation and action semantics exactly and replaces
everything that assumed a private model: the cube where it actually is instead
of where reset put it, a ramped move into the start pose instead of writing
`qpos`, and the fitted pads `LiveSim` switches on at the pick table - the same
pads, and the same jaw numbers, as `ArmEnv(gripper='padded')`.

The checkpoint still meets a base held by a weld rather than deleted joints and
a cube wherever the drive left it, and `prepare_flybrain_live` says so out loud
rather than pretending its fixed-base evaluation carries over.
"""
from collections.abc import Callable
import math
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .arm import Arm, GRIPPER, down_quat
from .arm_env import ArmEnv, jaw_limits
from .grasp import move
from .manipulation import hand_for
from .orchestration import ToolResult
from .room import TABLES


class LiveArmEnv(ArmEnv):
    """ArmEnv's observation/action semantics on the shared live sim, no teleports."""

    def __init__(self, sim, history=1, motion_deadband=0.):
        # deliberately not ArmEnv.__init__: it compiles a model of its own
        self.sim = sim
        self.history_length = history
        self.motion_deadband = float(motion_deadband)
        self.station = 'pick'
        table = next(t for t in TABLES if t.name == 'table_pick')
        self.frame_yaw = table.yaw
        c, s = math.cos(table.yaw), math.sin(table.yaw)
        self.frame_rotation = np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        self.frame_origin = np.array([*table.centre,0.])
        self.reference_origin = np.array([-.20,-1.95,0.])
        self.model, self.data = sim.model, sim.data
        self.scratch = mujoco.MjData(self.model)
        self.arm = Arm(self.model, self.data, 'right')
        self.quat = down_quat(self.frame_yaw)
        self.object_name = 'pick_cube'
        self.object_id = self.model.body(self.object_name).id
        self.objq = self.model.jnt_qposadr[self.model.body_jntadr[self.object_id]]
        self.objgeom = self.model.body_geomadr[self.object_id]
        self.fingers = {self.model.body(f + '_finger__' + f + '_finger').id for f in ('left', 'right')}
        self.gripq = self.model.jnt_qposadr[self.model.joint(GRIPPER['right']).id]
        self.cube_width = float(2*self.model.geom_size[self.objgeom,0])
        self.rest_height = .70 + self.cube_width/2
        # the fitted pads: LiveSim switches them on at this table, and they
        # are what ArmEnv(gripper='padded') trains on, so the jaw numbers match.
        if getattr(sim, 'pads', 'padded') != 'padded':
            raise RuntimeError('the live room has the stock pads on; park at the pick table first')
        self.gripper = 'padded'
        self.hand = hand_for(self.model, 'right', 'padded')
        self.jaw_open, self.jaw_closed = jaw_limits(self.hand, self.cube_width)
        self.jaw_scale = 1.0/self.jaw_open
        self.closed_action = -1.0
        self.released_qpos = 0.82*self.jaw_open
        shift = np.zeros(3)
        mujoco.mju_rotVecQuat(shift, np.asarray(self.hand.jaw_offset(self.jaw_closed), dtype=float),
                              self.quat)
        self.jaw_shift = shift
        self.carry_speed = (1., 1., 1., .2, 1., 1., 1.)
        self.action_space = spaces.Box(-1., 1., (4,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (23*history,), np.float32)
        self.horizon = 480

    def _open_jaws(self, seconds=1.):
        """Ramp the jaw command open instead of stepping it.

        A 0.38 rad step swings the blades in at whatever the force limit allows,
        which on a robot held up by a brake still shakes the arm around.
        """
        start = float(self.data.ctrl[self.arm.grip_act])
        steps = max(1, int(seconds/self.model.opt.timestep))
        for i in range(steps):
            self.arm.grip(start + (self.jaw_open-start)*(i+1)/steps)
            self.sim.rig.step()

    def reset(self, *, seed=None, options=None):
        # gym.Env.reset, not ArmEnv's: nothing here may reset or write state.
        gym.Env.reset(self, seed=seed)
        self.start = self.to_task(self.data.body(self.object_name).xpos)
        self.start[2] = .70                     # the cube's base, as ArmEnv defines start
        self.goal = self.start + [.17, 0, self.cube_width/2]
        self.target = self.start + [0, 0, .152]
        # open the jaws first: the jaw offset this pose is solved for depends on them
        self._open_jaws()
        # The arm has to end up on the IK branch the policy trained in, not
        # merely at the same site pose: this 7-DOF arm reaches a pose several
        # ways, and the elbow-out branch loads the pincer differently enough
        # that the follower blade scissors past closed on the cube.  So solve
        # the start pose exactly as ArmEnv.reset does - same seed config, same
        # target, same deterministic solver - and only then work out a point
        # above it on that branch to come in from, because a straight joint
        # ramp from the folded arm sweeps the forearm across the table.
        seed = np.zeros(len(self.arm.ik.qadr))
        seed[3] = math.pi / 2
        self.scratch.qpos[:] = self.data.qpos
        solution = self.arm.ik.solve(self.scratch, self.site_for_jaws(self.target), self.quat, seed=seed)
        if not solution.ok:
            raise RuntimeError('Reset pose unreachable from the parked pose')
        self.scratch.qpos[:] = self.data.qpos
        above = self.arm.ik.solve(self.scratch, self.site_for_jaws(self.target + [0, 0, .10]),
                                  self.quat, seed=solution.qpos, restarts=1)
        if above.ok and np.abs(above.qpos - solution.qpos).max() < 0.8:
            move(self.sim.rig, self.arm, above.qpos, 2.5)
            move(self.sim.rig, self.arm, solution.qpos, 1.5)
        else:
            move(self.sim.rig, self.arm, solution.qpos, 2.5)
        self.steps = 0
        self.max_lift = self.held = self.stable = 0.
        self.previous = np.zeros(4)
        self.potential = self._potential()
        return self._record_observation(reset=True), {}

    def _advance(self):
        for _ in range(25):
            self.sim.rig.step()
            lift = float(self.cube[2]-self.rest_height)
            self.max_lift = max(self.max_lift, lift)
            if lift > .05 and self.contacts() == 2:
                self.held += self.model.opt.timestep

    def close(self):
        """No-op: the sim outlives the episode, and the demo drives on from here."""


def prepare_flybrain_live(sim, checkpoint, seed=3000) -> Callable[[], ToolResult]:
    """Run the arm policy on the live sim, with the validation gate deliberately off.

    `scripts/orchestrate.py` refuses to run this checkpoint unless a recorded
    twenty-for-twenty evaluation matches its hash, and that evaluation is of the
    fixed-base padded model. There is no such report for the live sim and there
    cannot be one yet, so this runs the best checkpoint there is and labels the
    result as not evidence.
    """
    if not sim.parked or sim.station != 'pick':
        raise ValueError(
            "Flybrain runs at station 'pick' on a parked robot; the live sim is "
            f'parked={bool(sim.parked)} at station={sim.station!r}. Nothing was moved.'
        )
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise RuntimeError(f'Flybrain checkpoint missing: {checkpoint}; run scripts/train_arm.py first')
    # lazily, and after the cheap checks: the learning stack is a slow import
    # and is not installed everywhere this module is
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(2)
    policy = PPO.load(checkpoint, device='cpu')
    config = getattr(policy, 'arm_config', None)
    if not isinstance(config, dict):
        config = {}
    history = config.get('history_length', 1)
    motion_deadband = config.get('motion_deadband', 0.)
    trained_on = config.get('gripper', 'unknown')
    print(f'Flybrain live: UNVALIDATED — checkpoint {checkpoint} (gripper={trained_on!r}, fixed base) '
          'running on the live room with the parking brake; results are not evidence.'
          + ('' if trained_on == 'padded' else
             f" The live room switches the fitted pads on at this table; this checkpoint "
             f"was trained on {trained_on!r} contacts."))

    def execute():
        env = LiveArmEnv(sim, history, motion_deadband)
        observation, _ = env.reset(seed=seed)
        for _ in range(env.horizon):
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        return ToolResult(info['success'], dict(tool='run_flybrain', validated=False, seed=seed,
                                                checkpoint=str(checkpoint), **info))
    return execute
