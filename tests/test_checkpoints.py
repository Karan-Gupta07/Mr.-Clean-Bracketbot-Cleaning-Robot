"""The shipped weights a fresh checkout needs: present, and the graph is the one the policy was trained on."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from gymnasium.spaces import Box
import numpy as np

import demo
from rlbot.act import DEFAULT_CHECKPOINT
from rlbot.connectome import ConnectomeFeatures

CHECKPOINTS = ROOT / 'checkpoints'


class ShippedCheckpointTests(unittest.TestCase):
    def test_the_demo_defaults_point_at_tracked_files(self):
        self.assertTrue(DEFAULT_CHECKPOINT.is_file(), DEFAULT_CHECKPOINT)
        self.assertTrue(demo.FLYBRAIN_CHECKPOINT.is_file(), demo.FLYBRAIN_CHECKPOINT)
        self.assertEqual(demo.FLYBRAIN_CHECKPOINT.parent, CHECKPOINTS)

    def test_the_tracked_graph_matches_its_manifest(self):
        manifest = json.loads((CHECKPOINTS / 'graph_512.json').read_text())
        digest = hashlib.sha256((CHECKPOINTS / 'graph_512.npz').read_bytes()).hexdigest()
        self.assertEqual(digest, manifest['graph_sha256'])

    def test_the_policy_finds_its_graph_without_out_flywire(self):
        # A bare name that does not exist relative to the cwd exercises the fallback chain.
        space = Box(-1., 1., shape=(8,), dtype=np.float32)
        features = ConnectomeFeatures(space, graph_path='graph_512.npz')
        manifest = json.loads((CHECKPOINTS / 'graph_512.json').read_text())
        self.assertEqual(features.graph_sha256, manifest['graph_sha256'])
        self.assertEqual(features.n, manifest['neurons'])


if __name__ == '__main__':
    unittest.main()
