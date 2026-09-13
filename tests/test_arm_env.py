import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rlbot.arm_env import ArmEnv


class ArmEnvironmentTests(unittest.TestCase):
    def test_policy_actions_select_motion_without_scripted_planner(self):
        with patch('rlbot.manipulation.PickPlace.plan',side_effect=AssertionError('script invoked')):
            env=ArmEnv()
            env.reset(seed=99)
            initial=env.target.copy()
            env.step(np.array([1.,0.,0.,1.]))
            self.assertAlmostEqual(env.target[0]-initial[0],.004)
            self.assertFalse(hasattr(env,'phase'))
            self.assertFalse(hasattr(env,'teacher'))

    def test_open_stationary_gripper_cannot_pass(self):
        env=ArmEnv()
        env.reset(seed=99)
        for _ in range(env.horizon):
            _,_,terminated,truncated,info=env.step([0,0,0,1])
            if terminated or truncated: break
        self.assertFalse(info['success'])
        self.assertLess(info['max_lift'],.001)
        self.assertEqual(info['held_seconds'],0)


if __name__=='__main__': unittest.main()
