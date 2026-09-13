import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rlbot.arm_env import ArmEnv


class ArmEnvironmentTests(unittest.TestCase):
    def test_pick_station_replaces_crockery_and_uses_original_hinges(self):
        import mujoco
        from rlbot.arm import GRIPPER, FOLLOWER
        env=ArmEnv()
        self.assertEqual(env.object_name,'pick_cube')
        self.assertEqual(env.model.body('table_pick').name,'table_pick')
        self.assertGreater(mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,'ball'),0)
        self.assertGreater(mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,'crate_ball'),0)
        self.assertGreater(mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,'cube_m'),0)
        self.assertEqual(mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,'table_ware'),-1)
        np.testing.assert_allclose(env.model.body('table_pick').pos[:2],[-2.25,.90])
        for joint in (*GRIPPER.values(),*FOLLOWER.values()):
            self.assertEqual(env.model.joint(joint).type[0],mujoco.mjtJoint.mjJNT_HINGE)
        point=np.array([-.34,-1.71,.7])
        np.testing.assert_allclose(env.to_task(env.to_world(point)),point,atol=1e-12)
        obs,_=env.reset(seed=99)
        self.assertEqual(obs.shape,(23,))
        self.assertLess(np.linalg.norm(env.cube[:2]-env.start[:2]),1e-8)

    def test_grasp_width_matches_the_selected_station_cube(self):
        for station in ('pick','cubes'):
            env=ArmEnv(station=station)
            self.assertAlmostEqual(env.cube_width,2*env.model.geom_size[env.objgeom,0])

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
