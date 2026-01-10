#!/usr/bin/env python
"""Evaluate a trained DQN or PPO model on the TSC environment.

Usage:
    uv run python -m src.training.eval \
        --model-path runs/dqn/smoke_test/models/final.zip \
        --algo dqn \
        --config configs/smoke_dqn.yaml

    uv run python -m src.training.eval \
        --model-path runs/ppo/smoke_test/models/final.zip \
        --algo ppo \
        --config configs/single_int_ppo.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from stable_baselines3 import DQN, PPO

from src.envs.tsc_env import SingleIntersectionTSCEnv
from src.training.metrics import aggregate_episode_metrics, run_episode


def make_env(cfg: dict) -> SingleIntersectionTSCEnv:
    """Create a TSC environment from config."""
    env_cfg = cfg["env"]
    return SingleIntersectionTSCEnv(
        sumocfg_path=env_cfg["sumocfg_path"],
        decision_interval_s=env_cfg["decision_interval_s"],
        min_green_s=env_cfg["min_green_s"],
        max_sim_time_s=env_cfg.get("max_sim_time_s", 3600),
        seed=env_cfg["seed"],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained TSC model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model (.zip file)",
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["dqn", "ppo"],
        help="Algorithm used (dqn or ppo)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate trained model."""
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load model
    model_path = Path(args.model_path)
    if args.algo == "dqn":
        model = DQN.load(model_path)
    else:
        model = PPO.load(model_path)

    # Create env once (reuse across episodes for seed variation via episode_count)
    env = make_env(cfg)

    print("=" * 60)
    print(f"Evaluating {args.algo.upper()} model")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Seed: {cfg['env']['seed']}")
    print(f"Episodes: {args.n_episodes}")
    print("=" * 60)

    # Define action function for the model
    def act_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    # Run evaluation episodes using shared metrics
    episodes = []
    for ep in range(args.n_episodes):
        metrics = run_episode(env, act_fn)
        episodes.append(metrics)

        in_network = metrics["injected"] - metrics["completed"]
        print(
            f"Episode {ep + 1}: "
            f"reward={metrics['episode_reward']:.2f}, "
            f"completed={metrics['completed']}, "
            f"injected={metrics['injected']}, "
            f"in_network={in_network}, "
            f"mean_queue={metrics['mean_queue']:.2f}"
        )

    env.close()

    # Aggregate results
    results = aggregate_episode_metrics(episodes)
    results["model_path"] = str(model_path)
    results["algorithm"] = args.algo

    print("=" * 60)
    print("Results:")
    print(f"  Mean episode reward: {results['mean_episode_reward']:.2f} +/- {results['std_episode_reward']:.2f}")
    print(f"  Completed (throughput): {results['mean_completed']:.1f} +/- {results['std_completed']:.1f} vehicles/episode")
    print(f"  Injected: {results['mean_injected']:.1f} +/- {results['std_injected']:.1f} vehicles/episode")
    print(f"  In network at end: {results['mean_in_network_end']:.1f} vehicles")
    print(f"  Mean queue length: {results['mean_queue']:.2f} +/- {results['std_queue']:.2f} vehicles")
    print(f"  Episode length: {results['mean_n_steps']:.1f} steps")
    print("=" * 60)

    # Save results
    eval_path = model_path.parent.parent / "eval.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {eval_path}")


if __name__ == "__main__":
    main()
