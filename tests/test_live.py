"""The parking brake: what holds the base still between driving and manipulating.

A whole route is a minute of simulation, so it is not run here - `scripts/demo.py`
and `scripts/navigate.py` drive.  What this pins down is the handover: the brake
holds where it was switched on, lets go again, and the skills API works on the
same model on the far side of it.
"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from rlbot.live import LiveSim


class ParkingBrakeTests(unittest.TestCase):
    def test_brake_holds_the_base_still_and_gives_it_back(self):
        sim = LiveSim(keyframe='dock_cubes')
        sim.park()
        was = sim.data.body('root').xpos.copy()
        sim.rig.seconds(1)                      # no balancer, no wheel torque
        self.assertLess(float(np.linalg.norm(sim.data.body('root').xpos - was)), .001)
        self.assertEqual(sim.data.eq_active[sim.brake], 1)
        self.assertEqual(sim.model.opt.impratio, 200)
        np.testing.assert_array_equal(sim.data.ctrl[sim.wheels], 0)
        self.assertTrue(sim.parked)

        sim.unpark()
        self.assertEqual(sim.data.eq_active[sim.brake], 0)
        self.assertEqual(sim.model.opt.impratio, 10)
        self.assertFalse(sim.parked)

    def test_skills_run_on_the_parked_model_at_a_table_with_no_crate(self):
        sim = LiveSim(keyframe='dock_pick')
        with self.assertRaises(RuntimeError):
            sim.robot()                         # nothing holds the base yet
        sim.park()
        sim.station = 'pick'                    # what an arrival would have set
        self.assertIn('pick_cube', sim.robot().scene())
        sim.home()

    def test_an_unknown_station_is_refused_before_the_brake_comes_off(self):
        sim = LiveSim(keyframe='dock_cubes')
        sim.park()
        sim.station = 'cubes'
        with self.assertRaises(KeyError):
            sim.drive_to('nowhere')
        self.assertTrue(sim.parked)
        self.assertEqual(sim.station, 'cubes')


if __name__ == '__main__':
    unittest.main()


class PadSwapTests(unittest.TestCase):
    """Each table's controller gets the pads it was tuned on, rewritten in place."""

    def test_pads_follow_the_station(self):
        import mujoco
        from rlbot.arm_env import ArmEnv
        from rlbot.robot import ROOM
        stock, padded = mujoco.MjModel.from_xml_path(str(ROOM)), ArmEnv(gripper='padded').model
        sim = LiveSim(keyframe='dock_pick')
        blade, pad = sim.model.body('right_finger__right_finger').id, sim.model.geom('right_finger__right_finger_pad').id
        hinge = sim.model.jnt_dofadr[sim.model.joint('right_right_gripper').id]

        def like(model, suffix):
            b, g = model.body('right_finger__right_finger').id, model.geom('right_finger__right_finger' + suffix).id
            np.testing.assert_allclose(sim.model.geom_size[pad], model.geom_size[g])
            np.testing.assert_allclose(sim.model.geom_pos[pad], model.geom_pos[g])
            np.testing.assert_allclose(sim.model.geom_friction[pad], model.geom_friction[g])
            self.assertEqual(sim.model.geom_condim[pad], model.geom_condim[g])
            self.assertAlmostEqual(sim.model.body_mass[blade], model.body_mass[b], places=6)
            self.assertAlmostEqual(sim.model.dof_M0[hinge], model.dof_M0[model.jnt_dofadr[model.joint('right_right_gripper').id]], places=7)
            np.testing.assert_allclose(sim.model.bvh_aabb[sim.model.body_bvhadr[blade]], model.bvh_aabb[model.body_bvhadr[b]])

        self.assertEqual(sim.pads, 'stock')
        like(stock, '_pad0')
        sim.station = 'pick'
        sim.park()
        self.assertEqual(sim.pads, 'padded')
        like(padded, '_pad')
        sim.unpark()
        self.assertEqual(sim.pads, 'stock')
        like(stock, '_pad0')
        # the live state survived mj_setConst
        self.assertAlmostEqual(float(sim.data.body('root').xpos[0]), -1.71, places=2)
        # and so did the room's declared extent: mj_setConst alone would make it
        # 11.4 m and push every camera's near plane from 80 mm to 230 mm
        self.assertAlmostEqual(float(sim.model.stat.extent), float(stock.stat.extent), places=6)
        np.testing.assert_allclose(sim.model.stat.center, stock.stat.center)


class RestOtherArmTests(unittest.TestCase):
    """The idle arm folds before the working arm moves in."""

    def test_idle_arm_out_over_the_table_is_folded_before_the_other_works(self):
        from rlbot.grasp import move
        from rlbot.skills import Robot
        robot = Robot('cubes')
        out = np.zeros(7)
        out[0], out[1], out[3] = -0.30, 0.6, 1.0          # rail down, reaching forward
        move(robot.rig, robot.arms['right'], out, 1.0)
        self.assertFalse(robot.at_rest('right'))
        robot.rest_other('left')
        self.assertTrue(robot.at_rest('right'))
        # a full hand is never folded, and a resting arm is not moved again
        robot.holding['right'] = 'cube_m'
        move(robot.rig, robot.arms['right'], out, 1.0)
        with patch.object(Robot, 'home') as home:
            robot.rest_other('left')
            robot.holding['right'] = None
            robot.rest_other('right')                       # left is already at zero
            home.assert_not_called()

    def test_pick_and_place_fold_the_other_arm_first(self):
        from rlbot.skills import Robot
        robot = Robot('cubes')
        with patch.object(Robot, 'rest_other') as rest:
            out = robot.pick('cube_l')
            self.assertTrue(out.ok, out.note)
            rest.assert_called_with('left')
            out = robot.place()
            self.assertTrue(out.ok, out.note)
            rest.assert_called_with('left')
