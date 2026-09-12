"""Imitate a robot waypoint teacher, then run PPO on an MLP or fly graph.

This trains on privileged simulator state in an obstacle-free scene. The graph
has finite message passing per action; the PD loop supplies stabilization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rlbot.navigation import NavigationEnv
from rlbot.connectome import ConnectomeFeatures


def collect(path, episodes=40):
    env = NavigationEnv()
    observations, actions, returns = [], [], []
    for seed in range(episodes):
        obs, _ = env.reset(seed=seed)
        episode_rewards = []
        for _ in range(env.horizon):
            action = env.teacher()
            observations.append(obs.copy())
            actions.append(action.copy())
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_rewards.append(reward)
            if terminated or truncated:
                break
        value = 0
        ep_returns = []
        for reward in reversed(episode_rewards):
            value = reward + 0.99 * value
            ep_returns.append(value)
        returns.extend(reversed(ep_returns))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, observations=np.array(observations), actions=np.array(actions), returns=np.array(returns, dtype=np.float32))
    print(f"Collected {len(observations)} teacher transitions from seeds 0..{episodes - 1}", flush=True)


def evaluate(model, episodes=20, seed_start=1000):
    env = NavigationEnv()
    results = []
    for seed in range(seed_start, seed_start + episodes):
        obs, _ = env.reset(seed=seed)
        total = 0.0
        max_pitch = 0.0
        for _ in range(env.horizon):
            action = env.teacher() if model is None else model.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            max_pitch = max(max_pitch, abs(info["pitch_deg"]))
            if terminated or truncated:
                break
        results.append({"seed": seed, "success": info["success"], "fell": info["fell"],
                        "distance": info["distance"], "return": total,
                        "max_pitch_deg": max_pitch, "seconds": info["sim_time"]})
    return {"episodes": episodes, "successes": sum(r["success"] for r in results),
            "falls": sum(r["fell"] for r in results), "mean_return": float(np.mean([r["return"] for r in results])),
            "mean_distance": float(np.mean([r["distance"] for r in results])), "runs": results}


class Progress(BaseCallback):
    def __init__(self):
        super().__init__()
        self.started = time.monotonic()

    def _on_step(self):
        if self.num_timesteps % 1024 == 0:
            print(f"PPO steps={self.num_timesteps} elapsed={time.monotonic() - self.started:.1f}s", flush=True)
        return True


class RehearsalPPO(PPO):
    """PPO followed by a small demonstration rehearsal to limit forgetting."""

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["demo_observations", "demo_actions", "rehearsal_optimizer"]

    def train(self):
        super().train()
        if not getattr(self, "rehearsal_updates", 0):
            return
        for _ in range(self.rehearsal_updates):
            idx = torch.randint(len(self.demo_observations), (128,))
            mean = self.policy.get_distribution(self.demo_observations[idx]).distribution.mean
            loss = (mean - self.demo_actions[idx]).square().mean()
            self.rehearsal_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.rehearsal_optimizer.step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["mlp", "connectome"], default="connectome")
    parser.add_argument("--graph", default="out/flywire/graph_512.npz")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--imitation-updates", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--demonstrations", type=Path, default=Path("out/rl/teacher.npz"))
    parser.add_argument("--resume-imitation", type=Path, help="Start this experiment from a saved imitation checkpoint")
    parser.add_argument("--rehearsal-updates", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(2)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = args.output or Path(f"out/rl/{args.policy}")
    output.mkdir(parents=True, exist_ok=True)
    if not args.demonstrations.exists():
        collect(args.demonstrations)
    kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), ortho_init=False)
    if args.policy == "connectome":
        kwargs.update(features_extractor_class=ConnectomeFeatures,
                      features_extractor_kwargs=dict(graph_path=str(Path(args.graph))))
    env = NavigationEnv()
    model = RehearsalPPO("MlpPolicy", env, policy_kwargs=kwargs, learning_rate=3e-5, n_steps=256,
                batch_size=128, n_epochs=5, gamma=0.99, ent_coef=0.001, vf_coef=0.02,
                seed=args.seed, device="cpu", verbose=0)
    if args.resume_imitation:
        previous = PPO.load(args.resume_imitation, device="cpu")
        if args.policy == "connectome" and getattr(previous.policy.features_extractor, "graph_sha256", None) != model.policy.features_extractor.graph_sha256:
            raise ValueError("Imitation checkpoint and requested graph differ")
        model.policy.load_state_dict(previous.policy.state_dict())
    if args.policy == "connectome":
        model.graph_sha256 = model.policy.features_extractor.graph_sha256
    with torch.no_grad():
        model.policy.log_std.fill_(-2)
    with np.load(args.demonstrations, allow_pickle=False) as data:
        observations = torch.tensor(data["observations"], dtype=torch.float32)
        actions = torch.tensor(data["actions"], dtype=torch.float32)
        returns = torch.tensor(data["returns"], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    started = time.monotonic()
    for update in range(0 if args.resume_imitation else args.imitation_updates):
        indices = torch.randint(len(observations), (128,))
        distribution = model.policy.get_distribution(observations[indices])
        prediction = distribution.distribution.mean
        action_loss = torch.square(prediction - actions[indices]).mean()
        value_loss = torch.square(model.policy.predict_values(observations[indices]).flatten() - returns[indices]).mean()
        loss = action_loss + 0.002 * value_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
        optimizer.step()
        if (update + 1) % 200 == 0:
            print(f"Imitation {update + 1}: action MSE={action_loss.item():.6f}, elapsed={time.monotonic() - started:.1f}s", flush=True)
    model.save(output / "imitation")
    before = evaluate(model, args.eval_episodes)
    print(f"Before PPO: {before['successes']}/{before['episodes']} successes", flush=True)
    model.demo_observations = observations
    model.demo_actions = actions
    model.rehearsal_updates = args.rehearsal_updates
    model.rehearsal_optimizer = torch.optim.Adam(model.policy.parameters(), lr=3e-4)
    internal_before = None
    if args.policy == "connectome":
        internal_before = model.policy.features_extractor.gain.detach().clone()
    model.learn(total_timesteps=args.steps, callback=Progress())
    model.save(output / "policy")
    after = evaluate(model, args.eval_episodes)
    graph_delta = float((model.policy.features_extractor.gain.detach() - internal_before).norm()) if internal_before is not None else None
    report = {"policy": args.policy, "seed": args.seed, "ppo_steps": model.num_timesteps,
              "imitation_updates": args.imitation_updates, "graph": str(args.graph) if args.policy == "connectome" else None,
              "resume_imitation": str(args.resume_imitation) if args.resume_imitation else None,
              "rehearsal_updates_per_ppo_rollout": args.rehearsal_updates,
              "graph_sha256": model.policy.features_extractor.graph_sha256 if args.policy == "connectome" else None,
              "graph_gain_change_during_ppo_l2": graph_delta, "before_ppo": before, "after_ppo": after,
              "elapsed_seconds": time.monotonic() - started, "torch": torch.__version__,
              "limitations": ["One training seed", "Privileged MuJoCo state", "Obstacle-free point goals",
                              "PD stabilization", "Finite graph propagation, not biological spikes or cross-step neural memory"]}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"After PPO: {after['successes']}/{after['episodes']} successes; {after['falls']} falls; report={output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
