from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from agent import Pacer, watch


class AgentViewerTests(unittest.TestCase):
    def test_closed_viewer_stops_before_the_next_refresh_interval(self):
        viewer = Mock()
        viewer.is_running.return_value = False
        pacer = Pacer(viewer, SimpleNamespace(opt=SimpleNamespace(timestep=.002)))
        with self.assertRaisesRegex(RuntimeError, 'viewer closed'):
            pacer(None)
        viewer.sync.assert_not_called()

    def test_pacer_rejects_invalid_speed(self):
        for speed in (0, -1, float('nan'), float('inf')):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                Pacer(Mock(), SimpleNamespace(opt=SimpleNamespace(timestep=.002)), speed)

    def test_watch_restores_the_step_callback_on_completion_and_failure(self):
        import mujoco.viewer
        for failure in (False, True):
            previous = Mock()
            root = SimpleNamespace(xmat=np.eye(3), xpos=np.zeros(3))
            robot = SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(timestep=.002)),
                                    data=SimpleNamespace(body=lambda _: root),
                                    rig=SimpleNamespace(on_step=previous))
            viewer = Mock()
            viewer.cam = SimpleNamespace(lookat=np.zeros(3))
            viewer.is_running.side_effect = [True, False]
            window = Mock()
            window.__enter__ = Mock(return_value=viewer)
            window.__exit__ = Mock(return_value=False)
            job = Mock(side_effect=RuntimeError('planner failed') if failure else None)
            with self.subTest(failure=failure), patch.object(mujoco.viewer,'launch_passive',return_value=window), patch('builtins.print'):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, 'planner failed'):
                        watch(robot, job, 1.)
                else:
                    watch(robot, job, 1.)
                self.assertIs(robot.rig.on_step, previous)
                job.assert_called_once()
                window.__exit__.assert_called_once()


if __name__ == '__main__':
    unittest.main()
