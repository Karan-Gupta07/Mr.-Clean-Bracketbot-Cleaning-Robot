from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import live_demo
from rlbot.orchestration import ToolResult

CUBES = ['--station', 'cubes', '--recognized', 'colored_cubes']
PICK = ['--station', 'pick', '--recognized', 'blue_cube_rectangle']
BALL = ['--station', 'ball', '--recognized', 'red_ball_box']


class LiveDemoTests(unittest.TestCase):
    def setUp(self):
        self.real_fable = live_demo.orchestrate.prepare_fable
        self.real_flybrain = live_demo.orchestrate.prepare_flybrain
        self.execute = Mock(return_value=ToolResult(True, {'placed': 1}))
        self.navigate = self.enterContext(patch.object(
            live_demo.orchestrate, 'navigate_to', return_value=ToolResult(True, {'arrived': True})))
        self.fable = self.enterContext(patch.object(
            live_demo.orchestrate, 'prepare_fable', return_value=self.execute))
        self.flybrain = self.enterContext(patch.object(
            live_demo.orchestrate, 'prepare_flybrain', return_value=self.execute))
        self.stdout, self.stderr = io.StringIO(), io.StringIO()

    def run_demo(self, args):
        with redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            return live_demo.main(args)

    def assert_stopped(self, args, message):
        with self.assertRaises(SystemExit) as raised:
            self.run_demo(args)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(message, self.stderr.getvalue())
        self.navigate.assert_not_called()
        self.execute.assert_not_called()

    def assert_no_preflight(self):
        self.fable.assert_not_called()
        self.flybrain.assert_not_called()

    def test_operator_must_supply_both_station_and_recognition_label(self):
        for args, missing in (([], '--station'), (['--station', 'cubes'], '--recognized'),
                              (['--recognized', 'colored_cubes'], '--station')):
            with self.subTest(args=args):
                self.assert_stopped(args, missing)
        self.assert_no_preflight()

    def test_mismatched_or_low_confidence_label_never_prepares_or_moves(self):
        self.assert_stopped(['--station', 'cubes', '--recognized', 'red_ball_box'], 'does not match')
        self.assert_stopped([*CUBES, '--confidence', '0.5'], 'must be confident')
        self.assert_no_preflight()

    def test_dry_run_is_labelled_as_route_only_and_offline_when_sweep_selected(self):
        self.assertEqual(self.run_demo([*CUBES, '--planner', 'sweep', '--dry-run']), 0)
        output = self.stdout.getvalue()
        for text in ('"tool": "run_fable"', 'user_provided_label', 'NOT camera recognition',
                     'simulator truth', 'SEPARATE SIMULATION STAGES', 'fixed-base', 'NOT a continuous handoff',
                     'OFFLINE skill validation', 'NOT live Fable API', 'readiness NOT checked'):
            self.assertIn(text, output)
        self.assert_no_preflight()
        self.navigate.assert_not_called()
        self.execute.assert_not_called()

    def test_live_fable_is_default_and_preflight_precedes_navigation_and_execution(self):
        events = Mock()
        events.attach_mock(self.fable, 'prepare')
        events.attach_mock(self.navigate, 'navigate')
        events.attach_mock(self.execute, 'execute')
        self.assertEqual(self.run_demo(CUBES), 0)
        self.assertEqual(events.mock_calls, [call.prepare('fable', view=True),
                                           call.navigate('cubes', view=True), call.execute()])
        self.flybrain.assert_not_called()
        self.assertIn('Fable API (requires', self.stdout.getvalue())
        self.assertIn('cubes window stays open', self.stdout.getvalue())

    def test_explicit_sweep_still_uses_live_viewers_but_not_the_api(self):
        self.assertEqual(self.run_demo([*CUBES, '--planner', 'sweep']), 0)
        self.fable.assert_called_once_with('sweep', view=True)
        self.navigate.assert_called_once_with('cubes', view=True)
        self.execute.assert_called_once_with()
        self.flybrain.assert_not_called()
        self.assertIn('OFFLINE skill validation; NOT live Fable API', self.stdout.getvalue())

    def test_missing_local_api_key_uses_real_preflight_and_never_moves_or_falls_back(self):
        self.fable.side_effect = self.real_fable
        with patch.object(live_demo.orchestrate.os, 'environ', {}), \
                patch.object(live_demo.orchestrate.importlib.util, 'find_spec', return_value=object()):
            self.assert_stopped(CUBES, 'Set ANTHROPIC_API_KEY locally')
        self.fable.assert_called_once_with('fable', view=True)
        self.flybrain.assert_not_called()

    def test_secure_prompt_is_console_only_and_restores_the_environment(self):
        with patch.object(live_demo.sys.stdin,'isatty',return_value=True), \
                patch.object(live_demo.getpass,'getpass',return_value='synthetic-session-key'), \
                patch.object(live_demo.os,'environ',{}):
            def prepared(planner, *, view):
                self.assertEqual(live_demo.os.environ['ANTHROPIC_API_KEY'],'synthetic-session-key')
                return self.execute
            self.fable.side_effect = prepared
            self.assertEqual(self.run_demo([*CUBES,'--prompt-api-key']),0)
            self.assertNotIn('ANTHROPIC_API_KEY',live_demo.os.environ)
        self.assertNotIn('synthetic-session-key',self.stdout.getvalue()+self.stderr.getvalue())

    def test_secure_prompt_refuses_non_console_input_without_reading_a_key(self):
        with patch.object(live_demo.sys.stdin,'isatty',return_value=False), \
                patch.object(live_demo.getpass,'getpass') as prompt:
            self.assert_stopped([*CUBES,'--prompt-api-key'],'interactive console')
            prompt.assert_not_called()
        self.assert_no_preflight()

    def test_report_records_result_without_recording_the_prompted_key(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        report = Path(directory)/'demo.json'
        with patch.object(live_demo.sys.stdin,'isatty',return_value=True), \
                patch.object(live_demo.getpass,'getpass',return_value='synthetic-session-key'), \
                patch.object(live_demo.os,'environ',{}):
            self.assertEqual(self.run_demo([*CUBES,'--prompt-api-key','--report',str(report)]),0)
        text = report.read_text(encoding='utf-8')
        self.assertNotIn('synthetic-session-key',text)
        result = json.loads(text)
        self.assertTrue(result['ok'])
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['planner'],'fable')

    def test_prompt_failure_restores_old_environment_and_redacts_secret(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        report = Path(directory)/'failed.json'
        self.fable.side_effect = ValueError('synthetic-session-key must never be reported')
        with patch.object(live_demo.sys.stdin,'isatty',return_value=True), \
                patch.object(live_demo.getpass,'getpass',return_value='synthetic-session-key'), \
                patch.object(live_demo.os,'environ',{'ANTHROPIC_API_KEY':'previous-test-value'}):
            self.assert_stopped([*CUBES,'--prompt-api-key','--report',str(report)],'redacted')
            self.assertEqual(live_demo.os.environ['ANTHROPIC_API_KEY'],'previous-test-value')
        self.assertNotIn('synthetic-session-key',report.read_text(encoding='utf-8'))
        self.assertNotIn('synthetic-session-key',self.stdout.getvalue()+self.stderr.getvalue())

    def test_dry_run_never_prompts_for_a_key(self):
        with patch.object(live_demo.getpass,'getpass') as prompt:
            self.assertEqual(self.run_demo([*CUBES,'--prompt-api-key','--dry-run']),0)
            prompt.assert_not_called()
        self.assert_no_preflight()

    def test_flybrain_forwards_selected_checkpoint_and_seed_without_control_overrides(self):
        checkpoint = Path('out/selected candidate/policy.zip')
        events = Mock()
        events.attach_mock(self.flybrain, 'prepare')
        events.attach_mock(self.navigate, 'navigate')
        events.attach_mock(self.execute, 'execute')
        self.assertEqual(self.run_demo([*PICK, '--checkpoint', str(checkpoint), '--seed', '3042']), 0)
        self.assertEqual(events.mock_calls, [call.prepare(checkpoint, 3042, view=True),
                                           call.navigate('pick', view=True), call.execute()])
        self.fable.assert_not_called()
        self.assertIn('window closes automatically when the episode ends', self.stdout.getvalue())

    def test_pick_dry_run_does_not_load_or_validate_the_checkpoint(self):
        self.assertEqual(self.run_demo([*PICK, '--checkpoint', 'not-a-real-policy.zip', '--dry-run']), 0)
        self.assertIn('"tool": "run_flybrain"', self.stdout.getvalue())
        self.assertIn('readiness NOT checked', self.stdout.getvalue())
        self.assert_no_preflight()
        self.navigate.assert_not_called()

    def test_no_implicit_checkpoint_or_force_or_sweep_fallback_for_pick(self):
        self.assert_stopped(PICK, 'explicitly selected --checkpoint')
        self.assert_stopped([*PICK, '--checkpoint', 'policy.zip', '--force'], 'unrecognized arguments')
        self.assert_stopped([*PICK, '--checkpoint', 'policy.zip', '--planner', 'sweep'], 'not a controller fallback')
        self.assert_stopped([*CUBES, '--checkpoint', 'policy.zip'], 'only used by the pick/Flybrain route')
        self.assert_no_preflight()

    def test_failed_preflight_never_navigates_or_tries_another_controller(self):
        self.flybrain.side_effect = RuntimeError('checkpoint not ready')
        self.assert_stopped([*PICK, '--checkpoint', 'policy.zip'], 'checkpoint not ready')
        self.flybrain.assert_called_once_with(Path('policy.zip'), 3000, view=True)
        self.fable.assert_not_called()

    def test_navigation_failure_stops_before_manipulation(self):
        self.navigate.return_value = ToolResult(False, {'reason': 'viewer closed'})
        self.assertEqual(self.run_demo([*CUBES, '--planner', 'sweep']), 1)
        self.fable.assert_called_once_with('sweep', view=True)
        self.execute.assert_not_called()
        self.assertIn('"manipulation": "not started"', self.stdout.getvalue())

    def test_manipulation_failure_is_not_reported_as_success(self):
        self.execute.return_value = ToolResult(False, {'success': False})
        self.assertEqual(self.run_demo([*CUBES, '--planner', 'sweep']), 1)
        self.flybrain.assert_not_called()

    def test_act_preflights_the_checkpoint_then_navigates_and_runs_one_episode(self):
        with patch.object(live_demo.orchestrate, 'prepare_act', return_value=self.execute) as act:
            self.assertEqual(self.run_demo(BALL), 0)
        act.assert_called_once_with(None, view=True)
        self.navigate.assert_called_once()
        self.execute.assert_called_once_with()
        self.assert_no_preflight()
        self.assertIn('ACT checkpoint', self.stdout.getvalue())

    def test_act_with_a_missing_checkpoint_stops_before_navigation(self):
        self.assert_stopped([*BALL, '--act-checkpoint', 'nowhere.pt'], 'ACT checkpoint unavailable: nowhere.pt')
        self.assert_no_preflight()

    def test_act_dry_run_can_report_the_route_without_claiming_it_is_ready(self):
        self.assertEqual(self.run_demo([*BALL, '--dry-run']), 0)
        self.assertIn('"tool": "run_act"', self.stdout.getvalue())
        self.assertIn('ACT checkpoint', self.stdout.getvalue())
        self.assertIn('readiness NOT checked', self.stdout.getvalue())
        self.assert_no_preflight()
        self.navigate.assert_not_called()

    def checkpoint_fixture(self, successes, config=None):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        checkpoint = Path(directory) / 'policy.zip'
        checkpoint.write_bytes(b'synthetic test checkpoint, never a real policy')
        report_path = checkpoint.parent / 'validation' / 'report.json'
        report_path.parent.mkdir()
        report_path.write_text(json.dumps(dict(
            episodes=20, successes=successes, environment=config,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())), encoding='utf-8')
        return checkpoint

    def test_real_flybrain_gate_rejects_18_of_20_before_policy_load_navigation_or_viewer(self):
        checkpoint = self.checkpoint_fixture(18)
        self.flybrain.side_effect = self.real_flybrain
        with patch('stable_baselines3.PPO.load') as load, \
                patch('rlbot.arm_env.ArmEnv') as environment, \
                patch('mujoco.viewer.launch_passive') as viewer:
            self.assert_stopped([*PICK, '--checkpoint', str(checkpoint)], 'has not passed its recorded evaluation')
        load.assert_not_called()
        environment.assert_not_called()
        viewer.assert_not_called()
        self.fable.assert_not_called()

    def test_real_preflight_uses_checkpoint_metadata_and_launches_mocked_live_viewer(self):
        config = dict(history_length=8, motion_deadband=.03, test_metadata='checkpoint owned')
        checkpoint = self.checkpoint_fixture(20, config)
        self.flybrain.side_effect = self.real_flybrain
        policy = Mock(arm_config=config)
        policy.predict.return_value = ([0, 0, 0, 0], None)
        with patch('stable_baselines3.PPO.load', return_value=policy) as load, \
                patch('rlbot.arm_env.ArmEnv') as environment, \
                patch('mujoco.viewer.launch_passive') as viewer:
            env = environment.return_value
            env.configuration, env.horizon, env.data.time = config, 1, 0.
            env.frame_yaw, env.model.stat.extent = 0., 1.
            env.to_world.return_value = [0., 0., 1.]
            env.reset.return_value = ([0], {})
            env.step.return_value = ([0], 0., True, False, {'success': True})

            def navigated(station, *, view):
                environment.assert_called_once_with(gripper='padded', station='pick', history=8, motion_deadband=.03)
                load.assert_called_once_with(checkpoint, device='cpu')
                viewer.assert_not_called()
                policy.predict.assert_not_called()
                return ToolResult(True, {'arrived': True})

            self.navigate.side_effect = navigated
            self.assertEqual(self.run_demo([*PICK, '--checkpoint', str(checkpoint), '--seed', '3017']), 0)
            viewer.assert_called_once_with(env.model, env.data)
            viewer.return_value.__enter__.return_value.sync.assert_called_once_with()
            env.reset.assert_called_once_with(seed=3017)
            env.close.assert_called_once_with()
        self.fable.assert_not_called()


if __name__ == '__main__':
    unittest.main()
