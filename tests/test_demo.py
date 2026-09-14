from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import demo
from rlbot.act import ACTUnavailableError
from rlbot.orchestration import ToolResult


def arrival(station, ok=True, failed=None):
    """What LiveSim.drive_to hands back, without building a room to get it."""
    return dict(ok=ok, station=station, err_xy=.012, err_yaw=.4, seconds=48.,
                replans=1, fell=False, touched=[], failed=failed)


def cubes_robot():
    robot = Mock()
    robot.items = {'cube_s': SimpleNamespace(graspable=True),
                   'crate_cubes': SimpleNamespace(graspable=False)}
    robot.where.return_value = 'in the crate'
    return robot


class FakeSim:
    """A LiveSim that parks where it is told, with no MuJoCo model behind it."""

    def __init__(self, arrives=True):
        self.arrives = arrives
        self.station = None
        self.parked = False
        self.driven = []
        self.robot = Mock(return_value=cubes_robot())

    def drive_to(self, station):
        self.driven.append(station)
        if self.arrives:
            self.station, self.parked = station, True
        return arrival(station, self.arrives, None if self.arrives else 'blocked by the pillar')


class Stream:
    def __init__(self, reply):
        self.reply = reply

    def __enter__(self):
        return self

    def __exit__(self, *error):
        return False

    def get_final_message(self):
        return self.reply


def use(identifier, name, **arguments):
    return SimpleNamespace(type='tool_use', id=identifier, name=name, input=arguments)


def reply(*blocks, stop_reason='tool_use'):
    return SimpleNamespace(stop_reason=stop_reason, stop_details=None, content=list(blocks))


class RoutePromptTests(unittest.TestCase):
    def test_words_pick_out_stations_in_the_order_they_are_said(self):
        for prompt, expected in (
                ('tidy the cubes', ['cubes']),
                ('go to the ACT table', ['ball']),
                ('pick up the blue cube', ['pick']),
                ('clean the cubes then the ball then cubes again', ['cubes', 'ball', 'cubes']),
                ('clean the cubes then pick the blue cube', ['cubes', 'pick']),
                ('hello', [])):
            with self.subTest(prompt=prompt):
                self.assertEqual(demo.route_prompt(prompt), expected)


class CommanderTests(unittest.TestCase):
    def setUp(self):
        self.lines = []

    def commander(self, sim, **options):
        return demo.Commander(sim, log=self.lines.append, **options)

    def tools(self):
        return [line.strip()[3:].split('(')[0]
                for line in self.lines if line.startswith('  -> ')]

    def test_sweep_drives_to_every_named_station_and_works_each_one(self):
        sim = FakeSim()
        commander = self.commander(sim, planner='sweep')
        with patch.object(demo, 'Harness') as harness, \
                patch.object(demo, 'sweep_planner') as planner, \
                patch.object(demo, 'prepare_act_live',
                             side_effect=ACTUnavailableError('ACT backend is not implemented')) as act:
            demo.sweep_commander(commander, 'clean the cubes then the ball')
        self.assertEqual(sim.driven, ['cubes', 'ball'])
        self.assertEqual(self.tools(),
                         ['go_to', 'manipulate', 'go_to', 'manipulate', 'finished'])
        planner.assert_called_once_with(harness.return_value)
        act.assert_called_once_with(sim, None)
        self.assertTrue(commander.done)

    def test_sweep_does_not_manipulate_where_it_never_arrived(self):
        sim = FakeSim(arrives=False)
        commander = self.commander(sim, planner='sweep')
        with patch.object(demo, 'Harness') as harness, \
                patch.object(demo, 'sweep_planner') as planner:
            demo.sweep_commander(commander, 'tidy the cubes')
        self.assertEqual(self.tools(), ['go_to', 'finished'])
        harness.assert_not_called()
        planner.assert_not_called()
        self.assertFalse(commander.results[0]['ok'])

    def test_a_prompt_naming_no_table_lists_the_words_and_stays_put(self):
        sim = FakeSim()
        commander = self.commander(sim, planner='sweep')
        demo.sweep_commander(commander, 'hello')
        self.assertEqual(sim.driven, [])
        self.assertEqual(self.tools(), ['finished'])
        self.assertIn('blue cube', '\n'.join(self.lines))

    def test_manipulate_before_parking_refuses_and_starts_no_backend(self):
        commander = self.commander(FakeSim(), checkpoint=Path('policy.zip'))
        with patch.object(demo, 'Harness') as harness, \
                patch.object(demo, 'prepare_flybrain_live') as flybrain, \
                patch.object(demo, 'prepare_act_live') as act:
            out = commander.run('manipulate', {})
        self.assertIn('not parked', out)
        for guard in (harness, flybrain, act):
            guard.assert_not_called()
        self.assertEqual(commander.results, [])

    def test_the_ball_table_reports_act_unavailable_and_tries_nothing_else(self):
        sim = FakeSim()
        sim.station, sim.parked = 'ball', True
        commander = self.commander(sim, checkpoint=Path('policy.zip'))
        with patch.object(demo, 'prepare_act_live',
                          side_effect=ACTUnavailableError('ACT backend is not implemented')) as act, \
                patch.object(demo, 'prepare_flybrain_live') as flybrain, \
                patch.object(demo, 'Harness') as harness:
            out = commander.run('manipulate', {})
        self.assertIn('ACT backend is not implemented', out)
        act.assert_called_once_with(sim, None)
        flybrain.assert_not_called()
        harness.assert_not_called()
        self.assertEqual(commander.results, [dict(
            tool='run_act', station='ball', ok=False,
            details=dict(reason='ACT backend is not implemented'))])

    def test_the_pick_table_without_a_checkpoint_refuses_instead_of_falling_back(self):
        sim = FakeSim()
        sim.station, sim.parked = 'pick', True
        commander = self.commander(sim)
        with patch.object(demo, 'prepare_flybrain_live') as flybrain, \
                patch.object(demo, 'Harness') as harness:
            out = commander.run('manipulate', {})
        self.assertIn('--checkpoint', out)
        flybrain.assert_not_called()
        harness.assert_not_called()

    def test_the_pick_table_with_a_checkpoint_runs_flybrain_once(self):
        sim = FakeSim()
        sim.station, sim.parked = 'pick', True
        execute = Mock(return_value=ToolResult(True, {'success': True, 'seed': 3042}))
        commander = self.commander(sim, checkpoint=Path('policy.zip'), seed=3042)
        with patch.object(demo, 'prepare_flybrain_live', return_value=execute) as flybrain, \
                patch.object(demo, 'prepare_act_live') as act:
            out = commander.run('manipulate', {})
        flybrain.assert_called_once_with(sim, Path('policy.zip'), 3042)
        execute.assert_called_once_with()
        act.assert_not_called()
        self.assertEqual(json.loads(out),
                         {'ok': True, 'details': {'success': True, 'seed': 3042}})

    def test_a_route_already_driven_is_not_driven_again(self):
        sim = FakeSim()
        commander = self.commander(sim, planner='sweep')
        commander.run('go_to', {'station': 'cubes'})
        self.assertIn('already parked', commander.run('go_to', {'station': 'cubes'}))
        self.assertEqual(sim.driven, ['cubes'])


class FableCommanderTests(unittest.TestCase):
    def test_the_model_drives_the_commander_and_gets_its_results_back(self):
        sim, lines, sent = FakeSim(), [], []
        commander = demo.Commander(sim, planner='fable', log=lines.append)
        replies = [reply(use('t1', 'go_to', station='cubes')),
                   reply(use('t2', 'manipulate')),
                   reply(use('t3', 'finished', summary='cubes crated'))]

        def stream(**kwargs):
            sent.append(list(kwargs['messages']))
            return Stream(replies[len(sent) - 1])

        client = Mock()
        client.beta.messages.stream.side_effect = stream
        with patch('anthropic.Anthropic', return_value=client), \
                patch.object(demo, 'Harness'), patch.object(demo, 'fable_planner') as planner:
            demo.fable_commander(commander, 'tidy the cubes table', 'high')

        self.assertEqual([line.strip()[3:].split('(')[0] for line in lines
                          if line.startswith('  -> ')],
                         ['go_to', 'manipulate', 'finished'])
        planner.assert_called_once_with(ANY, 'cubes', 'high')
        self.assertEqual(sim.driven, ['cubes'])
        self.assertTrue(commander.done)
        self.assertEqual(commander.summary, 'cubes crated')

        self.assertEqual(len(sent), 3)
        self.assertEqual(sent[0], [{'role': 'user', 'content': 'tidy the cubes table'}])
        self.assertEqual(sent[1][-1], {'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': 't1', 'content': ANY}]})
        self.assertIn('parked at the cubes table', sent[1][-1]['content'][0]['content'])
        self.assertEqual(sent[2][-1]['content'][0]['tool_use_id'], 't2')

        options = client.beta.messages.stream.call_args_list[0].kwargs
        self.assertEqual(options['model'], demo.MODEL)
        self.assertEqual(options['system'], demo.TOP_SYSTEM)
        self.assertEqual(options['tools'], demo.TOP_TOOLS)
        self.assertEqual(options['output_config'], {'effort': 'high'})
        self.assertNotIn('thinking', options)

    def test_a_refusal_stops_the_loop_without_calling_a_tool(self):
        sim, lines = FakeSim(), []
        commander = demo.Commander(sim, planner='fable', log=lines.append)
        client = Mock()
        client.beta.messages.stream.return_value = Stream(reply(stop_reason='refusal'))
        with patch('anthropic.Anthropic', return_value=client):
            demo.fable_commander(commander, 'tidy the cubes table', 'high')
        self.assertEqual(sim.driven, [])
        self.assertEqual(commander.calls, 0)


class MainTests(unittest.TestCase):
    def test_the_fable_planner_without_a_key_stops_before_building_a_sim(self):
        errors = io.StringIO()
        with patch.dict(demo.os.environ, {}, clear=True), \
                patch.object(demo, 'LiveSim') as sim, redirect_stderr(errors), \
                self.assertRaises(SystemExit) as raised:
            demo.main(['tidy the cubes table'])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('ANTHROPIC_API_KEY is not set', errors.getvalue())
        sim.assert_not_called()

    def test_a_sweep_run_writes_its_status_and_exits_zero_without_a_key(self):
        sim = FakeSim()
        report = Path(self.enterContext(tempfile.TemporaryDirectory())) / 'demo.json'
        output = io.StringIO()
        with patch.dict(demo.os.environ, {}, clear=True), \
                patch.object(demo, 'LiveSim', return_value=sim), \
                patch.object(demo, 'Harness'), patch.object(demo, 'sweep_planner'), \
                redirect_stdout(output):
            code = demo.main(['--planner', 'sweep', '--report', str(report),
                              'tidy the cubes table'])
        self.assertEqual(code, 0)
        status = json.loads(report.read_text(encoding='utf-8'))
        self.assertEqual(status['prompts'], ['tidy the cubes table'])
        self.assertEqual(status['stations'], ['cubes'])
        self.assertEqual([r['tool'] for r in status['results']], ['go_to', 'run_fable'])
        self.assertIn('parking brake', output.getvalue())

    def test_a_table_that_refuses_is_a_failed_run(self):
        sim = FakeSim()
        output = io.StringIO()
        with patch.dict(demo.os.environ, {}, clear=True), \
                patch.object(demo, 'LiveSim', return_value=sim), \
                patch.object(demo, 'prepare_act_live',
                             side_effect=ACTUnavailableError('ACT backend is not implemented')), \
                redirect_stdout(output):
            code = demo.main(['--planner', 'sweep', 'go to the ACT table'])
        self.assertEqual(code, 1)
        self.assertIn('ball runs ACT', output.getvalue())


if __name__ == '__main__':
    unittest.main()
