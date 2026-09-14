from contextlib import redirect_stdout
from io import StringIO
import sys
import tempfile
import types
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from rlbot.live import LiveSim
from rlbot.live_arm import LiveArmEnv, prepare_flybrain_live
from rlbot.orchestration import ToolResult


def parked_sim():
    """A LiveSim parked at the pick dock: brake on, the fitted pads switched in."""
    sim = LiveSim(keyframe='dock_pick')
    sim.station = 'pick'
    sim.park()
    return sim


class LiveArmEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.sim = parked_sim()

    def test_reset_drives_to_the_start_pose_without_resetting_or_teleporting(self):
        env = LiveArmEnv(self.sim)
        cube = self.sim.data.qpos[env.objq:env.objq+7].copy()
        pose = self.sim.data.qpos[env.arm.ik.qadr].copy()
        with patch('mujoco.mj_resetData') as reset_data:
            observation, info = env.reset(seed=3000)
        reset_data.assert_not_called()
        # micrometres of settling under gravity, not a teleport to a fixed start
        np.testing.assert_allclose(self.sim.data.qpos[env.objq:env.objq+7], cube, atol=1e-4)
        self.assertEqual(info, {})
        self.assertEqual(observation.shape, (23,))
        # the arm got to the start pose by being driven there, not by assignment
        self.assertGreater(np.abs(self.sim.data.qpos[env.arm.ik.qadr]-pose).max(), .1)

    def test_start_and_goal_come_from_where_the_cube_actually_is(self):
        env = LiveArmEnv(self.sim)
        # stand the cube somewhere ArmEnv's fixed start is not; the motion is
        # stubbed out so this is about what reset reads, not what it drives
        self.sim.data.qpos[env.objq] += .012
        mujoco.mj_forward(self.sim.model, self.sim.data)
        live = env.to_task(self.sim.data.body('pick_cube').xpos)
        with patch.object(LiveArmEnv, '_open_jaws'), patch('rlbot.live_arm.move'):
            env.reset(seed=3000)
        np.testing.assert_allclose(env.start, [*live[:2], .70], atol=1e-12)
        self.assertFalse(np.allclose(env.start, [-.34, -1.71, .70], atol=1e-3))
        np.testing.assert_allclose(env.goal, env.start + [.17, 0, env.cube_width/2])
        np.testing.assert_allclose(env.target, env.start + [0, 0, .152])

    def test_observation_width_follows_the_requested_history(self):
        env = LiveArmEnv(self.sim, history=4)
        self.assertEqual(env.observation_space.shape, (4*23,))
        observation, _ = env.reset(seed=3000)
        self.assertEqual(observation.shape, (4*23,))
        np.testing.assert_array_equal(observation.reshape(4, 23), np.tile(env._obs(), (4, 1)))

    def test_step_advances_the_shared_rig_rather_than_stepping_physics_itself(self):
        env = LiveArmEnv(self.sim)
        env.reset(seed=3000)
        self.sim.rig.on_step = Mock()
        started = self.sim.data.time
        with patch('mujoco.mj_step', wraps=mujoco.mj_step) as stepped:
            env.step(np.zeros(4))
        self.assertEqual(self.sim.rig.on_step.call_count, 25)
        self.assertEqual(stepped.call_count, 25)
        self.assertAlmostEqual(self.sim.data.time-started, 25*self.sim.model.opt.timestep)

class FlybrainLiveTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.checkpoint = Path(directory.name)/'arm.zip'
        self.checkpoint.write_bytes(b'not a trained policy')
        self.missing = Path(directory.name)/'missing.zip'
        self.sim = parked_sim()

    def fake_stack(self, policy):
        stable_baselines3 = types.ModuleType('stable_baselines3')
        stable_baselines3.PPO = Mock(load=Mock(return_value=policy))
        torch = types.ModuleType('torch')
        torch.set_num_threads = Mock()
        return patch.dict(sys.modules, {'stable_baselines3': stable_baselines3, 'torch': torch})

    def test_wrong_station_or_unparked_refuses_before_anything_moves(self):
        for parked, station in ((False, 'pick'), (True, 'ball'), (True, None)):
            with self.subTest(parked=parked, station=station):
                sim = types.SimpleNamespace(parked=parked, station=station, rig=Mock())
                with self.assertRaisesRegex(ValueError, 'Nothing was moved'):
                    prepare_flybrain_live(sim, self.checkpoint)
                self.assertEqual(sim.rig.mock_calls, [])

    def test_missing_checkpoint_fails_before_importing_the_learning_stack(self):
        # a None entry makes `import torch` raise, so the check has to come first
        with patch.dict(sys.modules, {'torch': None, 'stable_baselines3': None}):
            with self.assertRaisesRegex(RuntimeError, 'Flybrain checkpoint missing'):
                prepare_flybrain_live(self.sim, self.missing)

    def test_execute_runs_the_policy_and_reports_itself_unvalidated(self):
        class OneStep(LiveArmEnv):
            """The real environment, stopped after one action to keep this quick."""
            def step(self, action):
                observation, reward, _, truncated, info = super().step(action)
                return observation, reward, True, truncated, info

        policy = Mock(arm_config={'history_length': 1, 'motion_deadband': 0.})
        policy.predict.return_value = (np.zeros(4, np.float32), None)
        with self.fake_stack(policy), redirect_stdout(StringIO()):
            execute = prepare_flybrain_live(self.sim, self.checkpoint, seed=7)
        with patch('rlbot.live_arm.LiveArmEnv', OneStep):
            result = execute()
        policy.predict.assert_called_once()
        self.assertEqual(policy.predict.call_args.args[0].shape, (23,))
        self.assertIsInstance(result, ToolResult)
        self.assertFalse(result.ok)
        self.assertEqual(result.details['tool'], 'run_flybrain')
        self.assertFalse(result.details['validated'])
        self.assertEqual(result.details['seed'], 7)
        self.assertEqual(result.details['checkpoint'], str(self.checkpoint))
        self.assertEqual(result.details['steps'], 1)

    def test_preparing_announces_that_the_validation_gate_is_off(self):
        policy = Mock(arm_config={})
        with self.fake_stack(policy), patch('builtins.print') as printed:
            prepare_flybrain_live(self.sim, self.checkpoint)
        message = printed.call_args.args[0]
        self.assertIn('UNVALIDATED', message)
        self.assertIn(str(self.checkpoint), message)
        self.assertIn('results are not evidence', message)


if __name__ == '__main__':
    unittest.main()


class FittedPadParityTests(unittest.TestCase):
    """The live env at the pick table is ArmEnv(gripper='padded') without the weld."""

    def test_live_jaw_numbers_match_the_training_env(self):
        from rlbot.arm_env import ArmEnv
        from rlbot.gripper_pads import PaddedGripper
        env = ArmEnv(gripper='padded')
        live = LiveArmEnv(parked_sim())
        self.assertIsInstance(live.hand, PaddedGripper)
        self.assertEqual(live.gripper, 'padded')
        self.assertAlmostEqual(live.jaw_open, env.jaw_open, places=6)
        self.assertAlmostEqual(live.jaw_closed, env.jaw_closed, places=6)
        np.testing.assert_allclose(live.jaw_shift, env.jaw_shift, atol=1e-6)

    def test_refuses_the_stock_pads(self):
        sim = LiveSim(keyframe='dock_cubes')
        sim.station = 'cubes'
        sim.park()
        with self.assertRaisesRegex(RuntimeError, 'stock pads'):
            LiveArmEnv(sim)


if __name__ == '__main__':
    unittest.main()
