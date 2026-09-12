"""Evaluate saved navigation policies on an explicit, separate set of seeds."""

import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO
from train_connectome import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("out/rl/heldout.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    reports = {}
    for checkpoint in args.checkpoints:
        model = PPO.load(checkpoint, device="cpu")
        report = evaluate(model, episodes=args.episodes, seed_start=args.seed_start)
        reports[str(checkpoint)] = report
        print(f"{checkpoint}: {report['successes']}/{report['episodes']} goals, {report['falls']} falls", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
