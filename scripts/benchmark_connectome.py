"""Measure a graph's CPU forward/backward cost; does not train a controller."""

import argparse
import json
from pathlib import Path
import sys
import time

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlbot.connectome import ConnectomeFeatures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="out/flywire/graph_full.npz")
    parser.add_argument("--output", type=Path, default=Path("out/rl/full_graph_benchmark.json"))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = ConnectomeFeatures(gym.spaces.Box(-10, 10, (12,), dtype=np.float32), args.graph)
    observation = torch.randn(1, 12)
    forward, backward = [], []
    for i in range(args.repeats + 1):
        model.zero_grad(set_to_none=True)
        start = time.perf_counter()
        result = model(observation)
        middle = time.perf_counter()
        result.square().mean().backward()
        end = time.perf_counter()
        if i:
            forward.append(middle - start)
            backward.append(end - middle)
    report = {"graph": args.graph, "graph_sha256": model.graph_sha256,
              "neurons": model.n, "edges": model.adjacency._nnz(),
              "trainable_feature_parameters": sum(p.numel() for p in model.parameters()),
              "batch_size": 1, "cpu_threads": 2, "rounds": model.rounds, "width": model.width,
              "forward_ms": float(np.mean(forward) * 1000),
              "backward_ms": float(np.mean(backward) * 1000),
              "finite_output": bool(torch.isfinite(result).all()),
              "input_gradient_norm": float(model.encoder.weight.grad.norm()),
              "gain_gradient_norm": float(model.gain.grad.norm()),
              "torch": torch.__version__, "trained": False,
              "note": "Random initialized feature extractor only; excludes policy head and physics. Batch 1 does not establish training throughput."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
