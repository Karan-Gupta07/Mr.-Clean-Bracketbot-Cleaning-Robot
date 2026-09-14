"""Regressions for robot turning and the real graph policy boundary."""

import math
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rlbot.robot import Balancer
from rlbot.navigation import NavigationEnv
from rlbot.connectome import ConnectomeFeatures


class NavigationTests(unittest.TestCase):
    def test_pitch_is_heading_invariant(self):
        bot = Balancer.bracketbot(keyframe="home")
        for yaw in (0, math.pi / 2, math.pi, -math.pi / 2):
            pitch = 0.08
            yaw_quat = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
            pitch_quat = np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0])
            mujoco.mju_mulQuat(bot.data.qpos[3:7], yaw_quat, pitch_quat)
            mujoco.mj_forward(bot.model, bot.data)
            self.assertAlmostEqual(bot.state().pitch, pitch, places=6)

    def test_turn_sign_stability_and_reset(self):
        env = NavigationEnv()
        for sign in (-1, 1):
            env.reset(seed=0, options={"yaw": 0, "goal": [1, 0]})
            for _ in range(40):
                _, _, done, _, info = env.step([0, sign * 0.3])
                self.assertFalse(done)
            self.assertGreater(sign * env.bot.state().yaw_rate, 0.05)
            self.assertLess(abs(info["pitch_deg"]), 5)
        a, _ = env.reset(seed=123)
        env.step([0.5, 0.2])
        b, _ = env.reset(seed=123)
        np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            env.step([float("nan"), 0])

    def test_teacher_reaches_goal(self):
        env = NavigationEnv()
        env.reset(seed=25)
        for _ in range(env.horizon):
            _, _, done, timeout, info = env.step(env.teacher())
            if done or timeout:
                break
        self.assertTrue(info["success"])
        self.assertFalse(info["fell"])

    @unittest.skipUnless((REPO / "checkpoints/graph_512.npz").exists(), "Download graph first")
    def test_graph_gradient_and_stateless_batching(self):
        torch.set_num_threads(1)
        env = NavigationEnv()
        features = ConnectomeFeatures(env.observation_space, str(REPO / "checkpoints/graph_512.npz"))
        observations = torch.randn(3, 12)
        batched = features(observations)
        individual = torch.cat([features(row[None]) for row in observations])
        torch.testing.assert_close(batched, individual)
        batched.square().mean().backward()
        self.assertGreater(features.gain.grad.abs().sum().item(), 0)
        self.assertGreater(features.encoder.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(batched).all())
        self.assertEqual(features.last_activity.shape, (1, features.n))


if __name__ == "__main__":
    unittest.main()
