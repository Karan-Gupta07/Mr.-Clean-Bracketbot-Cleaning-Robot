"""Differentiable controller constrained by a measured FlyWire subgraph.

Finite message-passing steps per observation; no hidden state persists between
environment steps. Activities are artificial rate-like states, not LIF spikes.
"""

from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ConnectomeFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space, graph_path="out/flywire/graph_512.npz", width=4, rounds=4):
        graph_path = Path(graph_path)
        if not graph_path.exists():
            # Allow a locally produced checkpoint to move with its graph to a new checkout:
            # first the ignored out/flywire, then the tracked copy next to the shipped policy.
            root = Path(__file__).resolve().parents[2]
            for candidate in (root / "out/flywire" / graph_path.name, root / "checkpoints" / graph_path.name):
                if candidate.exists():
                    graph_path = candidate
                    break
        with np.load(graph_path, allow_pickle=False) as graph:
            inputs = graph["inputs"].copy()
            outputs = graph["outputs"].copy()
            n = len(graph["ids"])
            indices = torch.from_numpy(np.stack([graph["post"], graph["pre"]]).astype(np.int64))
            adjacency = torch.sparse_coo_tensor(indices, torch.from_numpy(graph["weight"].copy()), (n, n), check_invariants=True).coalesce()
        super().__init__(observation_space, features_dim=len(outputs) * width)
        self.graph_path = str(Path(graph_path))
        self.graph_sha256 = hashlib.sha256(Path(graph_path).read_bytes()).hexdigest()
        self.width, self.rounds, self.n = width, rounds, n
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("inputs", torch.tensor(inputs, dtype=torch.long))
        self.register_buffer("outputs", torch.tensor(outputs, dtype=torch.long))
        self.encoder = nn.Linear(observation_space.shape[0], len(inputs) * width)
        self.gain = nn.Parameter(torch.full((n, width), 1.5))
        self.bias = nn.Parameter(torch.zeros(n, width))
        self.channel_mix = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.channel_mix.weight)
        self.last_activity = None

    def forward(self, observations):
        batch = observations.shape[0]
        injection = observations.new_zeros(batch, self.n, self.width)
        encoded = torch.tanh(self.encoder(observations)).reshape(batch, len(self.inputs), self.width)
        injection = injection.index_copy(1, self.inputs, encoded)
        hidden = injection
        for _ in range(self.rounds):
            flattened = hidden.permute(1, 0, 2).reshape(self.n, -1)
            messages = torch.sparse.mm(self.adjacency, flattened).reshape(self.n, batch, self.width).permute(1, 0, 2)
            hidden = torch.tanh(0.35 * hidden + self.gain * self.channel_mix(messages) + injection + self.bias)
        self.last_activity = hidden.detach().abs().mean(dim=-1)
        return hidden[:, self.outputs, :].reshape(batch, -1)
