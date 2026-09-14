import hashlib
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from rlbot.arm_env import ArmEnv, validate_arm_config
from rlbot.gripper_pads import add_pads, bare_grippers
from rlbot.robot import ROOM, Balancer
from rlbot.grasp import read_state
from train_arm import Teacher


class ArmCompatibilityTests(unittest.TestCase):
    def test_urdf_is_byte_identical(self):
        path = ROOT / 'models/bracketbot/chopped_urdf_v2.urdf'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         '4ea3652f2a19141952f93eb1de162d36c95aa851ec59bb942a3dcebcdd3a7d57')

    def test_metadata_rejects_missing_stale_and_wrong_station(self):
        env = ArmEnv()
        validate_arm_config(env.configuration, env.configuration)
        for bad in (None, {}, dict(env.configuration, control_version=2),
                    dict(env.configuration, station='cubes'),
                    dict(env.configuration, gripper='parallel'),
                    dict(env.configuration, robot_sha256='stale')):
            with self.subTest(config=bad), self.assertRaises(ValueError):
                validate_arm_config(bad, env.configuration)

    def test_history_metadata_cannot_be_used_with_a_single_frame_checkpoint(self):
        single = ArmEnv()
        history = ArmEnv(history=16)
        self.assertEqual(single.configuration['control_version'], 4)
        self.assertEqual(history.configuration['control_version'], 5)
        self.assertEqual(history.configuration['history_length'], 16)
        self.assertEqual(history.configuration['observation_size'], 368)
        for key in ('physics_sha256','robot_sha256','scene_sha256'):
            self.assertEqual(single.configuration[key], history.configuration[key])
        with self.assertRaises(ValueError):
            validate_arm_config(single.configuration, history.configuration)
        with self.assertRaises(ValueError):
            validate_arm_config(history.configuration, single.configuration)
        calibrated = ArmEnv(history=16,motion_deadband=.05)
        with self.assertRaises(ValueError):
            validate_arm_config(history.configuration, calibrated.configuration)
        self.assertEqual(history.configuration['physics_sha256'],calibrated.configuration['physics_sha256'])

    def test_deadband_matches_external_command_filtering(self):
        reference = ArmEnv(history=16)
        env = ArmEnv(history=16,motion_deadband=.05)
        reference.reset(seed=99)
        env.reset(seed=99)
        for _ in range(20):
            action = np.array([.04,.06,-.03,-.5])
            filtered = action.copy()
            filtered[:3] = np.where(np.abs(filtered[:3])<.05,0,filtered[:3])
            expected, *expected_result = reference.step(filtered)
            actual, *actual_result = env.step(action)
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(actual_result,expected_result)

    def test_history_matches_gym_frame_stacking(self):
        import gymnasium as gym
        reference = gym.wrappers.FlattenObservation(gym.wrappers.FrameStackObservation(ArmEnv(), 16))
        env = ArmEnv(history=16)
        expected, _ = reference.reset(seed=99)
        actual, _ = env.reset(seed=99)
        np.testing.assert_array_equal(actual, expected)
        for _ in range(20):
            expected, *expected_result = reference.step([0,0,-.1,1])
            actual, *actual_result = env.step([0,0,-.1,1])
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual_result, expected_result)

    def test_history_distinguishes_the_teacher_open_close_transition(self):
        env = ArmEnv(history=16)
        observation, _ = env.reset(seed=1000)
        teacher = Teacher()
        records = []
        for step in range(39):
            action = teacher.action(env)
            if step in (37, 38):
                records.append((observation.copy(), action.copy()))
            observation, *_ = env.step(action)
        before, after = records
        self.assertEqual(before[1][3], 1)
        self.assertEqual(after[1][3], -1)
        self.assertLess(np.linalg.norm(before[0][-23:]-after[0][-23:]), .002)
        self.assertGreater(np.linalg.norm(before[0]-after[0]), 1)

    def test_jaw_calibration_only_scales_the_learned_jaw_output(self):
        from types import SimpleNamespace
        import torch
        from train_arm import calibrate_jaw
        actor = torch.nn.Linear(3,4)
        model = SimpleNamespace(policy=SimpleNamespace(action_net=actor))
        observations = torch.arange(15,dtype=torch.float32).reshape(5,3)/10
        before = actor(observations).detach().clone()
        calibrate_jaw(model,1.25)
        after = actor(observations).detach()
        torch.testing.assert_close(after[:,:3],before[:,:3])
        torch.testing.assert_close(after[:,3],before[:,3]*1.25)
        self.assertEqual(model.arm_calibration,dict(jaw_output_gain=1.25))
        for bad in (0,-1,np.nan,np.inf):
            with self.subTest(gain=bad), self.assertRaises(ValueError):
                calibrate_jaw(model,bad)
            torch.testing.assert_close(actor(observations).detach(),after)

    def test_jaw_recalibration_preserves_deadband_unless_explicitly_changed(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import torch
        import train_arm
        original = ArmEnv(history=16,motion_deadband=.05).configuration
        with tempfile.TemporaryDirectory() as output:
            for arguments, expected in (([],.05),(['--motion-deadband','0.03'],.03)):
                policy = SimpleNamespace(action_net=torch.nn.Linear(3,4),
                                         features_extractor=SimpleNamespace(graph_sha256='test graph'))
                model = SimpleNamespace(policy=policy,arm_config=original.copy(),
                                        num_timesteps=0,save=Mock())
                argv = ['train_arm.py','--history','16','--resume','test.zip',
                        '--calibrate-jaw','1.25','--output',output,*arguments]
                with patch.object(sys,'argv',argv), patch.object(train_arm.PPO,'load',return_value=model), \
                        patch.object(train_arm,'evaluate',return_value=dict(successes=0,episodes=10)), \
                        patch('builtins.print'):
                    train_arm.main()
                self.assertEqual(model.arm_config['motion_deadband'],expected)
                self.assertEqual(model.arm_config['history_length'],16)
                self.assertEqual(model.arm_calibration,dict(jaw_output_gain=1.25))
                model.save.assert_called_once()

    def test_release_correction_seats_before_opening_and_waits_for_clearance(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from train_arm import ReleaseCorrection
        env = SimpleNamespace(cube=np.array([0.,0.,.740]),goal=np.array([0.,0.,.724]),
                              rest_height=.724,max_lift=0.,held=0.,closed_action=-1.,
                              grip_point=np.array([0.,0.,.748]),target=np.array([0.,0.,.748]),
                              data=SimpleNamespace(qpos=np.array([.515])),gripq=0,
                              released_qpos=.42,contacts=Mock(return_value=2))
        teacher = ReleaseCorrection()
        self.assertIsNone(teacher.action(env))
        env.max_lift=.15; env.held=1.
        action=teacher.action(env)
        np.testing.assert_array_equal(action,[0,0,-1,-1])
        env.target[2]=env.grip_point[2]=.732
        for _ in range(15):
            self.assertEqual(teacher.action(env)[3],-1)
        env.cube[2]=.724
        for _ in range(11):
            self.assertEqual(teacher.action(env)[3],-1)
        np.testing.assert_array_equal(teacher.action(env),[0,0,0,1])
        for _ in range(25):
            self.assertEqual(teacher.action(env)[2],0)
        env.contacts.return_value=0
        for _ in range(7):
            self.assertEqual(teacher.action(env)[2],0)
        env.contacts.return_value=1
        self.assertEqual(teacher.action(env)[2],0)
        env.contacts.return_value=0
        for _ in range(7):
            self.assertEqual(teacher.action(env)[2],0)
        np.testing.assert_array_equal(teacher.action(env),[0,0,.5,1])
        env.data.qpos[0]=.3
        self.assertEqual(teacher.action(env)[2],0)
        self.assertFalse(ReleaseCorrection().active)

    def test_learned_evaluation_never_constructs_a_teacher_or_release_correction(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import train_arm
        observation=np.zeros(23,dtype=np.float32)
        action=np.array([.1,.2,.3,-.4],dtype=np.float32)
        env=SimpleNamespace(horizon=1,reset=Mock(return_value=(observation,{})),
                            step=Mock(return_value=(observation,0,True,False,dict(success=True))))
        model=SimpleNamespace(predict=Mock(return_value=(action,None)))
        with patch.object(train_arm,'Teacher',side_effect=AssertionError('teacher at inference')), \
                patch.object(train_arm,'ReleaseCorrection',side_effect=AssertionError('correction at inference')):
            score=train_arm.evaluate(model,env,1)
        self.assertEqual(score['successes'],1)
        np.testing.assert_array_equal(env.step.call_args.args[0],action)
        model.predict.assert_called_once_with(observation,deterministic=True)

    def test_release_collection_discards_failures_and_preserves_raw_prefix_outputs(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import torch
        import train_arm
        distribution=SimpleNamespace(distribution=SimpleNamespace(mean=torch.tensor([[.1,.2,.3,-1.25]])))
        policy=SimpleNamespace(obs_to_tensor=lambda state:(torch.tensor(state),False),
                               get_distribution=lambda tensor:distribution)
        env=SimpleNamespace(horizon=2,reset=Mock(side_effect=[(np.array([30.],dtype=np.float32),{}),
                                                            (np.array([31.],dtype=np.float32),{})]),
                            step=Mock(side_effect=[(np.array([30.]),0,False,False,dict(success=False)),
                                                   (np.array([30.]),0,True,False,dict(success=False)),
                                                   (np.array([31.]),0,False,False,dict(success=False)),
                                                   (np.array([31.]),0,True,False,dict(success=True))]))
        teacher=SimpleNamespace(action=Mock(side_effect=[None,np.array([0.,0.,0.,1.]),
                                                        None,np.array([0.,0.,0.,1.])]))
        with patch.object(train_arm,'ReleaseCorrection',return_value=teacher), patch('builtins.print'):
            observations,actions,corrected,runs=train_arm.collect_release_corrections(
                SimpleNamespace(policy=policy),env,2)
        np.testing.assert_array_equal(observations,[[31.],[31.]])
        np.testing.assert_array_equal(corrected,[False,True])
        self.assertEqual(actions[0,3],-1.25)
        self.assertEqual(actions[1,3],1.)
        self.assertEqual(observations.dtype,np.float32)
        self.assertEqual(actions.dtype,np.float32)
        self.assertEqual([run['success'] for run in runs],[False,True])

    def test_release_distillation_does_not_modify_graph_or_value_parameters(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        import train_arm
        extractor=torch.nn.Linear(3,5)
        extractor.graph_sha256='unit graph'
        actor=torch.nn.Sequential(torch.nn.Linear(5,4),torch.nn.Tanh())
        critic=torch.nn.Linear(5,1)
        policy=SimpleNamespace(features_extractor=extractor,extract_features=extractor,
                               mlp_extractor=SimpleNamespace(policy_net=actor,forward_actor=actor),
                               action_net=torch.nn.Linear(4,4),value_net=critic)
        model=SimpleNamespace(policy=policy,num_timesteps=0,arm_calibration=dict(jaw_output_gain=1.25),
                              save=lambda path:Path(str(path)+'.zip').write_bytes(b'unit checkpoint'))
        observations=np.arange(24,dtype=np.float32).reshape(8,3)/24
        actions=np.ones((8,4),dtype=np.float32)
        corrected=np.array([False]*4+[True]*4)
        frozen=[parameter.detach().clone() for module in (extractor,critic) for parameter in module.parameters()]
        before=policy.action_net.weight.detach().clone()
        with tempfile.TemporaryDirectory() as output, \
                patch.object(train_arm,'collect_release_corrections',return_value=(observations,actions,corrected,[])), \
                patch.object(train_arm,'evaluate',return_value=dict(successes=0,episodes=10)), \
                patch('builtins.print'):
            report=train_arm.distill_release(model,SimpleNamespace(configuration={}),Path(output),1,3,.001)
        for expected,actual in zip(frozen,[parameter for module in (extractor,critic) for parameter in module.parameters()]):
            torch.testing.assert_close(actual,expected)
            self.assertIsNone(actual.grad)
        self.assertFalse(torch.equal(before,policy.action_net.weight))
        self.assertEqual(report['ppo_steps'],0)
        self.assertTrue(report['feature_extractor_frozen'])
        self.assertEqual(report['corrected_examples'],4)

    def test_rate_limiter_command_is_visible_to_the_policy(self):
        env = ArmEnv()
        observation, _ = env.reset(seed=99)
        env.data.ctrl[env.arm.grip_act] -= .05
        self.assertFalse(np.array_equal(observation, env._obs()))

    def test_invalid_actions_do_not_mutate_the_simulator(self):
        env = ArmEnv()
        env.reset(seed=99)
        qpos, ctrl = env.data.qpos.copy(), env.data.ctrl.copy()
        for action in ([0, 0], [0, 0, 0, np.nan], [0, np.inf, 0, 0]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)
            np.testing.assert_array_equal(env.data.qpos, qpos)
            np.testing.assert_array_equal(env.data.ctrl, ctrl)

    def test_pad_selection_does_not_stack_contacts(self):
        spec = mujoco.MjSpec.from_file(str(ROOM))
        add_pads(spec)
        first = spec.compile()
        add_pads(spec)
        second = spec.compile()
        self.assertEqual(first.ngeom, second.ngeom)
        bare = bare_grippers(spec).compile()
        for prefix in ('', 'l_'):
            for finger in ('left', 'right'):
                name = f'{prefix}{finger}_finger__{finger}_finger'
                body = bare.body(name).id
                self.assertTrue(all(bare.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                                    for g in range(bare.ngeom) if bare.geom_bodyid[g] == body))

    def test_station_balance_pitch_agrees_with_navigation(self):
        bot = Balancer(ROOM, chassis='root', keyframe='dock_pick')
        yaw, pitch = 1.8, .1
        spin = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
        lean = np.array([np.cos(pitch/2), 0, np.sin(pitch/2), 0])
        mujoco.mju_mulQuat(bot.data.qpos[3:7], spin, lean)
        mujoco.mj_forward(bot.model, bot.data)
        sensors = tuple(bot.model.sensor(n).adr[0] for n in ('gyro','vel_left','vel_right'))
        self.assertAlmostEqual(read_state(bot.model, bot.data, sensors).pitch, bot.state().pitch)

    def test_teacher_can_release_and_withdraw_before_timeout(self):
        env = ArmEnv()
        env.reset(seed=1000)
        teacher = Teacher()
        for _ in range(env.horizon):
            _, _, terminated, truncated, info = env.step(teacher.action(env))
            if terminated or truncated:
                break
        self.assertTrue(info['success'], info)
        self.assertEqual(teacher.phase, 6)
        self.assertEqual(env.contacts(), 0)


if __name__ == '__main__':
    unittest.main()
