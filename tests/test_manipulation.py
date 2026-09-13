"""Physical lift/release regressions and a no-grip negative control."""
from pathlib import Path
import sys
import unittest
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlbot.manipulation import PickPlace, make_model
from rlbot.robot import ROOM
from rlbot.arm import GRIPPER


class ManipulationTests(unittest.TestCase):
    def test_free_object_and_explicit_gripper_variant(self):
        model = make_model()
        body = model.body("cube_m").id
        self.assertEqual(model.jnt_type[model.body_jntadr[body]], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(model.body_jntnum[model.body("root").id], 0)
        self.assertTrue(all(kind == mujoco.mjtEq.mjEQ_JOINT for kind in model.eq_type))
        original = mujoco.MjModel.from_xml_path(str(ROOM))
        for joint in GRIPPER.values():
            self.assertEqual(model.joint(joint).type[0], mujoco.mjtJoint.mjJNT_SLIDE)
            self.assertEqual(original.joint(joint).type[0], mujoco.mjtJoint.mjJNT_HINGE)

    def test_lift_transfer_and_release(self):
        task = PickPlace(seed=23)
        while not task.finished:
            task.step()
        result = task.report()
        self.assertTrue(result["success"], result)
        self.assertGreater(result["max_lift"], .08)
        self.assertGreater(result["held_lift_seconds"], .5)
        self.assertTrue(result["released"])
        self.assertLess(result["distance"], .03)

    def test_without_grip_force_does_not_succeed(self):
        task = PickPlace(seed=23, grip_force=0.0)
        while not task.finished:
            task.step()
        self.assertFalse(task.report()["success"])


if __name__ == "__main__":
    unittest.main()
