from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rlbot.orchestration import (Dispatcher, Recognition, Target, ToolResult,
                                 classify_scene, route_for, station_at)
from rlbot.room import TABLES


class RoutingTests(unittest.TestCase):
    def test_each_recognized_scene_selects_exactly_one_controller(self):
        cases = [(Target.COLORED_CUBES, 'cubes', 'run_fable'),
                 (Target.RED_BALL_BOX, 'ball', 'run_act'),
                 (Target.BLUE_CUBE_RECTANGLE, 'pick', 'run_flybrain')]
        for target, station, tool in cases:
            with self.subTest(target=target):
                call = route_for(station, Recognition(target, .95, 10), now=10.1)
                self.assertEqual((call.station, call.tool), (station, tool))
                self.assertNotIn('vla', call.tool.lower())

    def test_shapes_and_container_disambiguate_blue_cube(self):
        self.assertEqual(classify_scene([('cube','blue')], 'rectangle'), Target.BLUE_CUBE_RECTANGLE)
        self.assertEqual(classify_scene([('ball','red')], 'box'), Target.RED_BALL_BOX)
        self.assertEqual(classify_scene([('cube','blue'),('cube','green')], 'box'), Target.COLORED_CUBES)
        for objects, container in [([('cube','blue')], 'box'),
                                    ([('ball','blue')], 'box'),
                                    ([('ball','red'),('cube','blue')], 'rectangle')]:
            with self.assertRaises(ValueError):
                classify_scene(objects, container)

    def test_ambiguous_stale_and_mismatched_requests_do_not_move(self):
        recognition = Recognition(Target.COLORED_CUBES, .95, 10)
        bad = [replace(recognition, confidence=.5), replace(recognition, confidence=math.nan),
               replace(recognition, confidence=1.1), replace(recognition, stamp=5),
               replace(recognition, stamp=12), replace(recognition, target='vla'),
               replace(recognition, target=Target.RED_BALL_BOX)]
        navigate, prepare = Mock(), Mock()
        dispatcher = Dispatcher(navigate, {'run_fable':prepare})
        for item in bad:
            with self.subTest(item=item), self.assertRaises(ValueError):
                dispatcher.run('cubes', item, now=10.1)
        navigate.assert_not_called()
        prepare.assert_not_called()

    def test_missing_act_never_falls_back_or_navigates(self):
        navigate, fable = Mock(), Mock()
        dispatcher = Dispatcher(navigate, {'run_fable':fable})
        with self.assertRaisesRegex(RuntimeError, 'run_act'):
            dispatcher.run('ball', Recognition(Target.RED_BALL_BOX, 1, 10), now=10)
        navigate.assert_not_called()
        fable.assert_not_called()

    def test_backend_preflight_and_navigation_must_pass(self):
        execute = Mock(return_value=ToolResult(True, {'placed':4}))
        navigate = Mock(return_value=ToolResult(False, {'reason':'not docked'}))
        prepare = Mock(return_value=execute)
        dispatcher = Dispatcher(navigate, {'run_fable':prepare})
        result = dispatcher.run('cubes', Recognition(Target.COLORED_CUBES, 1, 10), now=10)
        self.assertFalse(result.ok)
        execute.assert_not_called()
        prepare.side_effect = RuntimeError('checkpoint missing')
        navigate.reset_mock()
        with self.assertRaisesRegex(RuntimeError, 'checkpoint missing'):
            dispatcher.run('cubes', Recognition(Target.COLORED_CUBES, 1, 10), now=10)
        navigate.assert_not_called()

    def test_dispatch_invokes_only_selected_tool_once(self):
        execute = Mock(return_value=ToolResult(True, {'placed':1}))
        other = Mock()
        dispatcher = Dispatcher(lambda _: ToolResult(True, {}),
                                {'run_flybrain':lambda: execute, 'run_act':other})
        result = dispatcher.run('pick', Recognition(Target.BLUE_CUBE_RECTANGLE, 1, 10), now=10)
        self.assertTrue(result.ok)
        execute.assert_called_once_with()
        other.assert_not_called()

    def test_spots_resolve_table_centres_and_docks_not_unknown_floor(self):
        for table in TABLES:
            station = table.name.removeprefix('table_')
            self.assertEqual(station_at(table.centre), station)
            self.assertEqual(station_at(table.dock[:2]), station)
        for point in ((0,0), (math.nan,0), (1,), (99,99)):
            with self.assertRaises(ValueError):
                station_at(point)


if __name__ == '__main__':
    unittest.main()
