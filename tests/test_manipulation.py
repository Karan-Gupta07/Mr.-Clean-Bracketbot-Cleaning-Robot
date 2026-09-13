"""Physical lift/release regressions, and which gripper the model is built with."""
from pathlib import Path
import sys
import unittest
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlbot.manipulation import PickPlace, make_model
from rlbot.robot import ROOM
from rlbot.arm import GRIPPER, FOLLOWER


class GripperModelTests(unittest.TestCase):
    def test_supplied_gripper_blades_are_untouched(self):
        """Both supplied-gripper builds keep the CAD blades; only contact differs."""
        original = mujoco.MjModel.from_xml_path(str(ROOM))
        for variant in ("urdf", "padded"):
            model = make_model(gripper=variant)
            for joint in (*GRIPPER.values(), *FOLLOWER.values()):
                self.assertEqual(model.joint(joint).type[0], mujoco.mjtJoint.mjJNT_HINGE, variant)
                np.testing.assert_allclose(model.joint(joint).range,
                                           original.joint(joint).range, err_msg=variant)
            self.assertEqual(model.neq, original.neq, variant)
            for name in GRIPPER.values():
                np.testing.assert_allclose(model.actuator(name).forcerange,
                                           original.actuator(name).forcerange, err_msg=variant)
            # the CAD mesh is still the visual, at the colour the export gave it
            for prefix in ("", "l_"):
                for finger in ("left", "right"):
                    body = model.body(f"{prefix}{finger}_finger__{finger}_finger").id
                    meshes = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                              and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
                    self.assertTrue(meshes, variant)

    def test_pads_collide_instead_of_the_inflated_blade_hull(self):
        model = make_model(gripper="padded")
        for prefix in ("", "l_"):
            for finger in ("left", "right"):
                name = f"{prefix}{finger}_finger__{finger}_finger"
                body = model.body(name).id
                colliding = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                             and (model.geom_contype[g] or model.geom_conaffinity[g])]
                self.assertEqual([model.geom(g).name for g in colliding], [f"{name}_pad"])
                self.assertEqual(model.geom_type[colliding[0]], mujoco.mjtGeom.mjGEOM_BOX)
        # nothing collides on the untouched build's blades being replaced
        plain = make_model(gripper="urdf")
        self.assertGreater(sum(plain.geom_contype[g] or plain.geom_conaffinity[g]
                               for g in range(plain.ngeom)
                               if plain.geom_bodyid[g] == plain.body("left_finger__left_finger").id), 0)

    def test_default_model_keeps_the_supplied_gripper_unmodified(self):
        model = make_model(gripper="urdf")
        original = mujoco.MjModel.from_xml_path(str(ROOM))
        for joint in (*GRIPPER.values(), *FOLLOWER.values()):
            self.assertEqual(model.joint(joint).type[0], mujoco.mjtJoint.mjJNT_HINGE)
            self.assertEqual(model.joint(joint).type[0], original.joint(joint).type[0])
            np.testing.assert_allclose(model.joint(joint).range, original.joint(joint).range)
        # the exported <mimic> equalities, and only those, survive
        self.assertEqual(model.neq, original.neq)
        self.assertTrue(all(kind == mujoco.mjtEq.mjEQ_JOINT for kind in model.eq_type))
        # the jaws are still the CAD finger meshes, not substitute primitives
        for prefix in ("", "l_"):
            for finger in ("left", "right"):
                body = model.body(f"{prefix}{finger}_finger__{finger}_finger").id
                kinds = {model.geom_type[g] for g in range(model.ngeom)
                         if model.geom_bodyid[g] == body}
                self.assertEqual(kinds, {mujoco.mjtGeom.mjGEOM_MESH})
        # and the gripper servo keeps the force limit the export gave it
        for name in GRIPPER.values():
            np.testing.assert_allclose(model.actuator(name).forcerange,
                                       original.actuator(name).forcerange)

    def test_free_object_and_opt_in_parallel_substitute(self):
        model = make_model()
        self.assertEqual(model.geom("left_finger__left_finger_pad").type[0],
                         mujoco.mjtGeom.mjGEOM_BOX)   # padded is the default
        body = model.body("cube_m").id
        self.assertEqual(model.jnt_type[model.body_jntadr[body]], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(model.body_jntnum[model.body("root").id], 0)
        substitute = make_model(gripper="parallel")
        for joint in GRIPPER.values():
            self.assertEqual(substitute.joint(joint).type[0], mujoco.mjtJoint.mjJNT_SLIDE)
        with self.assertRaises(ValueError):
            make_model(gripper="nonexistent")


class ManipulationTests(unittest.TestCase):
    def test_padded_supplied_gripper_lifts_transfers_and_releases(self):
        task = PickPlace(seed=23)
        while not task.finished:
            task.step()
        result = task.report()
        self.assertTrue(result["success"], result)
        self.assertEqual(result["gripper"], "supplied hooked gripper with contact pads")
        self.assertGreater(result["max_lift"], .08)
        self.assertGreater(result["held_lift_seconds"], .5)
        self.assertTrue(result["released"])
        self.assertLess(result["distance"], .03)

    @unittest.expectedFailure
    def test_unpadded_supplied_gripper_cannot_hold_the_cube(self):
        """Known gap, not a flaky test: the bare CAD blades do not hold the cube.

        MuJoCo collides a mesh geom as its convex hull, 3.2x the blade's real
        volume, which fills in the hook the design relies on. `check_grasp.py
        --item cube_m` reports the same directly: 0/1 lifted, 0.0 mm risen.
        The padded build above is the fix; this records what it fixes.
        """
        task = PickPlace(seed=23, gripper="urdf")
        while not task.finished:
            task.step()
        self.assertTrue(task.report()["success"], task.report())

    def test_parallel_substitute_lifts_transfers_and_releases(self):
        task = PickPlace(seed=23, gripper="parallel")
        while not task.finished:
            task.step()
        result = task.report()
        self.assertTrue(result["success"], result)
        self.assertGreater(result["max_lift"], .08)
        self.assertGreater(result["held_lift_seconds"], .5)
        self.assertTrue(result["released"])
        self.assertLess(result["distance"], .03)

    def test_without_grip_force_does_not_succeed(self):
        task = PickPlace(seed=23, gripper="parallel", grip_force=0.0)
        while not task.finished:
            task.step()
        self.assertFalse(task.report()["success"])


if __name__ == "__main__":
    unittest.main()
