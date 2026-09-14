from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

from rlbot.act import (ACTUnavailableError, DEFAULT_CHECKPOINT, checkpoint_path,
                       prepare_act, prepare_act_live)
from rlbot.orchestration import Dispatcher, Recognition, Target, ToolResult
import run_act


def parked_sim(station='ball', parked=True):
    return SimpleNamespace(parked=parked, station=station, rig=Mock(), model=Mock(),
                           robot=Mock(return_value=Mock(name='robot')))


class CheckpointTests(unittest.TestCase):
    def test_none_means_the_shipped_checkpoint(self):
        self.assertEqual(checkpoint_path(None), DEFAULT_CHECKPOINT)
        self.assertTrue(DEFAULT_CHECKPOINT.is_file(), DEFAULT_CHECKPOINT)

    def test_a_missing_checkpoint_is_unavailable_not_a_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'missing.pt'
            for checkpoint in (missing, str(missing), Path(directory)):
                with self.subTest(checkpoint=checkpoint), self.assertRaises(ACTUnavailableError) as raised:
                    checkpoint_path(checkpoint)
                message = str(raised.exception)
                for text in (f'ACT checkpoint unavailable: {checkpoint}', "station='ball'",
                             'no VLA, scripted, Fable, or Flybrain fallback'):
                    self.assertIn(text, message)


class LiveTests(unittest.TestCase):
    """prepare_act_live: everything that can fail does so before the robot moves."""

    def test_wrong_station_or_unparked_refuses_before_loading_anything(self):
        for parked, station in ((False, 'ball'), (True, 'cubes'), (True, 'pick'), (True, None)):
            with self.subTest(parked=parked, station=station), \
                    patch('rlbot.act.Policy') as policy:
                sim = parked_sim(station, parked)
                with self.assertRaises(ValueError) as raised:
                    prepare_act_live(sim, DEFAULT_CHECKPOINT)
                self.assertNotIsInstance(raised.exception, ACTUnavailableError)
                self.assertIn('Nothing was moved', str(raised.exception))
                policy.assert_not_called()
                sim.robot.assert_not_called()
                self.assertEqual(sim.rig.mock_calls, [])

    def test_a_missing_checkpoint_refuses_before_loading_anything(self):
        sim = parked_sim()
        with patch('rlbot.act.Policy') as policy, self.assertRaises(ACTUnavailableError):
            prepare_act_live(sim, Path('nowhere.pt'))
        policy.assert_not_called()
        sim.robot.assert_not_called()

    def test_prepare_loads_the_policy_once_and_execute_runs_one_episode(self):
        sim = parked_sim()
        robot = sim.robot.return_value
        robot.items = {'ball': SimpleNamespace(graspable=True),
                       'crate_ball': SimpleNamespace(graspable=False)}
        robot.hands = {'right': SimpleNamespace(pads=[3, 4]), 'left': SimpleNamespace(pads=[7, 8])}
        outcome = (True, dict(note='in the crate', side='right', ticks=200, seconds=10.0,
                              retreated=True, rows={'time': [0.0]}))
        with patch('rlbot.act.Policy') as policy, patch('rlbot.act.device', return_value='cpu'), \
                patch('rlbot.act.run_act', return_value=outcome) as run:
            policy.return_value.step_trained, policy.return_value.mode = 19750, 'ensemble'
            execute = prepare_act_live(sim)
            policy.assert_called_once_with(DEFAULT_CHECKPOINT, sim.model, 'cpu')
            sim.robot.assert_not_called()            # nothing touches the sim until execute
            result = execute()
        sim.robot.assert_called_once_with()
        run.assert_called_once_with(robot, policy.return_value, 'ball', on_step=None)
        self.assertEqual(policy.return_value.hide, [3, 4, 7, 8])
        policy.return_value.close.assert_called_once_with()
        self.assertEqual(result, ToolResult(True, dict(
            tool='run_act', checkpoint=str(DEFAULT_CHECKPOINT), step=19750, mode='ensemble',
            note='in the crate', side='right', ticks=200, seconds=10.0, retreated=True)))

    def test_a_failed_episode_is_a_false_result_not_an_exception(self):
        sim = parked_sim()
        sim.robot.return_value.items = {'ball': SimpleNamespace(graspable=True)}
        sim.robot.return_value.hands = {}
        outcome = (False, dict(note='on the table', side='left', ticks=500, seconds=25.0, rows={}))
        with patch('rlbot.act.Policy'), patch('rlbot.act.device'), \
                patch('rlbot.act.run_act', return_value=outcome):
            result = prepare_act_live(sim)()
        self.assertFalse(result.ok)
        self.assertEqual(result.details['note'], 'on the table')


class FixedBaseTests(unittest.TestCase):
    def test_prepare_builds_nothing_and_execute_builds_the_ball_room(self):
        with patch('rlbot.skills.Robot') as robot_class, patch('rlbot.act.Policy') as policy, \
                patch('rlbot.act.device'), patch('rlbot.act._episode') as episode:
            execute = prepare_act(None, False)
            robot_class.assert_not_called()
            policy.assert_not_called()
            execute()
        robot_class.assert_called_once_with('ball')
        episode.assert_called_once_with(robot_class.return_value, policy.return_value,
                                        DEFAULT_CHECKPOINT)

    def test_the_dispatcher_sees_a_missing_checkpoint_before_anything_else_runs(self):
        navigate = Mock()
        others = {name: Mock() for name in ('run_fable', 'run_flybrain', 'run_vla', 'run_scripted')}
        prepare = Mock(side_effect=lambda: prepare_act(Path('nowhere.pt')))
        dispatcher = Dispatcher(navigate, dict(others, run_act=prepare))
        with self.assertRaises(ACTUnavailableError):
            dispatcher.run('ball', Recognition(Target.RED_BALL_BOX, 1, 10), now=10)
        prepare.assert_called_once_with()
        navigate.assert_not_called()
        for other in others.values():
            self.assertEqual(other.mock_calls, [])


class CLITests(unittest.TestCase):
    def test_describe_reports_the_shipped_checkpoint_without_running(self):
        output = StringIO()
        with patch.object(run_act, 'prepare_act') as prepare, redirect_stdout(output):
            self.assertEqual(run_act.main(['--describe']), 0)
        description = json.loads(output.getvalue())
        self.assertEqual(description['tool'], 'run_act')
        self.assertEqual(description['station'], 'ball')
        self.assertEqual(description['status'], 'ready')
        self.assertEqual(description['checkpoint'], str(DEFAULT_CHECKPOINT))
        self.assertTrue(description['backend_implemented'])
        self.assertTrue(description['ready'])
        self.assertFalse(description['execution_started'])
        self.assertIsNone(description['fallback'])
        prepare.assert_not_called()

    def test_check_passes_on_the_shipped_checkpoint_and_fails_on_a_missing_one(self):
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(run_act.main(['--check']), 0)
        self.assertEqual(output.getvalue(), '')
        with redirect_stdout(output), redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
            run_act.main(['--check', '--checkpoint', 'nowhere.pt'])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(output.getvalue(), '')
        self.assertIn('ACT checkpoint unavailable: nowhere.pt', errors.getvalue())

    def test_cli_process_returns_nonzero_for_a_missing_checkpoint(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts' / 'run_act.py'),
                                 '--check', '--checkpoint', 'nowhere.pt'],
                                cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, '')
        self.assertIn('ACT checkpoint unavailable', result.stderr)


if __name__ == '__main__':
    unittest.main()
