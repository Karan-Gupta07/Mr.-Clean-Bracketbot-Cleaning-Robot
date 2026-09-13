from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

from rlbot.act import ACTUnavailableError, prepare_act
from rlbot.orchestration import Dispatcher, Recognition, Target
import run_act


class ACTPreflightTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.checkpoint = self.directory / 'act.ckpt'
        self.checkpoint.write_bytes(b'not a trained ACT checkpoint')
        self.missing = self.directory / 'missing.ckpt'

    def test_missing_checkpoint_reports_actionable_scaffold_status(self):
        cases = [(None, 'No ACT checkpoint was supplied'),
                 (self.missing, f'ACT checkpoint unavailable: {self.missing}'),
                 (self.directory, f'ACT checkpoint unavailable: {self.directory}')]
        for checkpoint, expected in cases:
            with self.subTest(checkpoint=checkpoint), self.assertRaises(ACTUnavailableError) as raised:
                prepare_act(checkpoint)
            message = str(raised.exception)
            self.assertIsInstance(raised.exception, RuntimeError)
            for text in (expected, 'Action Chunking with Transformers', "station='ball'",
                         'Scaffold only', 'ACT backend is not implemented',
                         'Supply the real ACT implementation and a compatible checkpoint',
                         'rlbot.act.prepare_act', 'no VLA, scripted, Fable, or Flybrain fallback'):
                self.assertIn(text, message)

    def test_existing_checkpoint_never_enables_or_loads_a_placeholder_backend(self):
        with patch.object(Path, 'read_bytes') as read_bytes, patch.object(Path, 'open') as open_file:
            for checkpoint in (self.checkpoint, str(self.checkpoint)):
                for view in (False, True):
                    with self.subTest(checkpoint=checkpoint, view=view), self.assertRaisesRegex(
                            ACTUnavailableError, 'cannot be used without the real ACT backend'):
                        prepare_act(checkpoint, view)
            read_bytes.assert_not_called()
            open_file.assert_not_called()

    def test_preflight_never_constructs_moves_or_invokes_another_controller(self):
        targets = ('rlbot.arm_env.ArmEnv', 'rlbot.skills.Robot', 'rlbot.robot.Balancer',
                   'rlbot.navigate.Navigator', 'agent.fable_planner', 'agent.sweep_planner',
                   'mujoco.viewer.launch_passive', 'mujoco.MjModel', 'mujoco.MjData',
                   'mujoco.mj_step')
        with ExitStack() as stack:
            guards = [stack.enter_context(patch(target)) for target in targets]
            for checkpoint in (None, self.missing, self.directory, self.checkpoint):
                for view in (False, True):
                    with self.subTest(checkpoint=checkpoint, view=view):
                        navigate = Mock()
                        others = {name: Mock() for name in ('run_fable', 'run_flybrain',
                                                           'run_vla', 'run_scripted')}
                        prepare = Mock(side_effect=lambda: prepare_act(checkpoint, view))
                        dispatcher = Dispatcher(navigate, dict(others, run_act=prepare))
                        with self.assertRaises(ACTUnavailableError):
                            dispatcher.run('ball', Recognition(Target.RED_BALL_BOX, 1, 10), now=10)
                        prepare.assert_called_once_with()
                        navigate.assert_not_called()
                        for other in others.values():
                            self.assertEqual(other.mock_calls, [])
            for target, guard in zip(targets, guards):
                self.assertEqual(guard.mock_calls, [], target)


class ACTCLITests(unittest.TestCase):
    def test_describe_is_explicitly_not_readiness_or_execution_success(self):
        output = StringIO()
        with patch.object(run_act, 'prepare_act') as prepare, redirect_stdout(output):
            self.assertEqual(run_act.main(['--describe']), 0)
        description = json.loads(output.getvalue())
        self.assertEqual(description['tool'], 'run_act')
        self.assertEqual(description['station'], 'ball')
        self.assertEqual(description['status'], 'scaffold')
        self.assertIn('Action Chunking with Transformers', description['description'])
        self.assertIn('not VLA', description['description'])
        self.assertFalse(description['ready'])
        self.assertFalse(description['backend_implemented'])
        self.assertFalse(description['execution_started'])
        self.assertIsNone(description['fallback'])
        prepare.assert_not_called()

    def test_check_and_execution_fail_without_success_output(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'act.ckpt'
            checkpoint.write_bytes(b'not a trained ACT checkpoint')
            for mode in ([], ['--check']):
                for options in ([], ['--checkpoint', str(checkpoint)],
                                ['--checkpoint', str(checkpoint.with_name('missing.ckpt')), '--view']):
                    with self.subTest(mode=mode, options=options):
                        output, errors = StringIO(), StringIO()
                        with redirect_stdout(output), redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
                            run_act.main(mode + options)
                        self.assertEqual(raised.exception.code, 2)
                        self.assertEqual(output.getvalue(), '')
                        self.assertIn('ACT backend is not implemented', errors.getvalue())

    def test_cli_process_returns_nonzero_for_missing_backend(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts' / 'run_act.py'), '--check'],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, '')
        self.assertIn('ACT backend is not implemented', result.stderr)
        self.assertIn('No ACT checkpoint was supplied', result.stderr)


if __name__ == '__main__':
    unittest.main()
