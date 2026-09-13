import hashlib
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from rlbot.arm_env import ArmEnv, validate_arm_config
from rlbot.gripper_pads import add_pads, bare_grippers
from rlbot.robot import ROOM, Balancer
from rlbot.grasp import read_state
from train_arm import Teacher


class ArmCompatibilityTests(unittest.TestCase):
    def test_urdf_is_byte_identical(self):
        path = ROOT / 'models/bracketbot/chopped_urdf_v2.urdf'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         '4ea3652f2a19141952f93eb1de162d36c95aa851ec59bb942a3dcebcdd3a7d57')

    def test_metadata_rejects_missing_stale_and_wrong_station(self):
        env = ArmEnv()
        validate_arm_config(env.configuration, env.configuration)
        for bad in (None, {}, dict(env.configuration, control_version=2),
                    dict(env.configuration, station='cubes'),
                    dict(env.configuration, gripper='parallel'),
                    dict(env.configuration, robot_sha256='stale')):
            with self.subTest(config=bad), self.assertRaises(ValueError):
                validate_arm_config(bad, env.configuration)

    def test_rate_limiter_command_is_visible_to_the_policy(self):
        env = ArmEnv()
        observation, _ = env.reset(seed=99)
        env.data.ctrl[env.arm.grip_act] -= .05
        self.assertFalse(np.array_equal(observation, env._obs()))

    def test_invalid_actions_do_not_mutate_the_simulator(self):
        env = ArmEnv()
        env.reset(seed=99)
        qpos, ctrl = env.data.qpos.copy(), env.data.ctrl.copy()
        for action in ([0, 0], [0, 0, 0, np.nan], [0, np.inf, 0, 0]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)
            np.testing.assert_array_equal(env.data.qpos, qpos)
            np.testing.assert_array_equal(env.data.ctrl, ctrl)

    def test_pad_selection_does_not_stack_contacts(self):
        spec = mujoco.MjSpec.from_file(str(ROOM))
        add_pads(spec)
        first = spec.compile()
        add_pads(spec)
        second = spec.compile()
        self.assertEqual(first.ngeom, second.ngeom)
        bare = bare_grippers(spec).compile()
        for prefix in ('', 'l_'):
            for finger in ('left', 'right'):
                name = f'{prefix}{finger}_finger__{finger}_finger'
                body = bare.body(name).id
                self.assertTrue(all(bare.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                                    for g in range(bare.ngeom) if bare.geom_bodyid[g] == body))

    def test_station_balance_pitch_agrees_with_navigation(self):
        bot = Balancer(ROOM, chassis='root', keyframe='dock_pick')
        yaw, pitch = 1.8, .1
        spin = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
        lean = np.array([np.cos(pitch/2), 0, np.sin(pitch/2), 0])
        mujoco.mju_mulQuat(bot.data.qpos[3:7], spin, lean)
        mujoco.mj_forward(bot.model, bot.data)
        sensors = tuple(bot.model.sensor(n).adr[0] for n in ('gyro','vel_left','vel_right'))
        self.assertAlmostEqual(read_state(bot.model, bot.data, sensors).pitch, bot.state().pitch)

    def test_replay_javascript_syntax(self):
        import shutil
        import subprocess
        node = shutil.which('node')
        if node is None:
            self.skipTest('Node is optional for the replay syntax check')
        template = (ROOT/'demo/arm_rl.html').read_text(encoding='utf-8')
        script = template.split('<script>',1)[1].split('</script>',1)[0]
        result = subprocess.run([node,'--check'],input=script,encoding='utf-8',capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_replay_has_targets_for_recorded_gripper_metadata(self):
        from html.parser import HTMLParser
        class Elements(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = set()
            def handle_starttag(self, tag, attrs):
                self.ids.add(dict(attrs).get('id'))
        parser = Elements()
        parser.feed((ROOT/'demo/arm_rl.html').read_text(encoding='utf-8'))
        self.assertTrue({'jawScale','jawMapping','beforeScore','afterScore',
                         'gripperScope','gripperNote','trainingSteps','observationSize'} <= parser.ids)

    def test_teacher_can_release_and_withdraw_before_timeout(self):
        env = ArmEnv()
        env.reset(seed=1000)
        teacher = Teacher()
        for _ in range(env.horizon):
            _, _, terminated, truncated, info = env.step(teacher.action(env))
            if terminated or truncated:
                break
        self.assertTrue(info['success'], info)
        self.assertEqual(teacher.phase, 6)
        self.assertEqual(env.contacts(), 0)


if __name__ == '__main__':
    unittest.main()
