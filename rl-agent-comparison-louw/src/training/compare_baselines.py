#!/usr/bin/env python
"""Compare trained RL agents against baselines.

Usage:
    uv run python -m src.training.compare_baselines \
        --model-path runs/dqn/smoke_test/models/final.zip \
        --algo dqn \
        --config configs/smoke_dqn.yaml \
        --n-episodes 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from stable_baselines3 import DQN, PPO

from src.baselines.agents import FixedCycleAgent, MaxQueueAgent, RandomAgent
from src.envs.tsc_env import SingleIntersectionTSCEnv
from src.training.metrics import aggregate_episode_metrics, run_episode


def make_env(cfg: dict, seed: int | None = None) -> SingleIntersectionTSCEnv:
    """Create environment from config."""
    env_cfg = cfg["env"]
    return SingleIntersectionTSCEnv(
        sumocfg_path=env_cfg["sumocfg_path"],
        decision_interval_s=env_cfg["decision_interval_s"],
        min_green_s=env_cfg["min_green_s"],
        max_sim_time_s=env_cfg.get("max_sim_time_s", 3600),
        seed=seed,
    )


def evaluate_agent(
    agent,
    env: SingleIntersectionTSCEnv,
    n_episodes: int,
    deterministic: bool = True,
    log_phases: bool = False,
) -> dict:
    """Run evaluation episodes and collect metrics using shared run_episode.

    Args:
        agent: Agent with predict(obs, deterministic) method.
        env: Environment to evaluate on.
        n_episodes: Number of episodes to run.
        deterministic: Use deterministic actions.
        log_phases: If True, log phase usage (for debugging fixed-cycle).

    Returns:
        Dict with aggregated metrics.
    """
    # Reset agent if it has a reset method (for FixedCycleAgent)
    if hasattr(agent, "reset"):
        agent.reset()

    # Define action function for this agent
    def act_fn(obs):
        action, _ = agent.predict(obs, deterministic=deterministic)
        return action

    # Run episodes using shared metrics
    episodes = []
    for _ep in range(n_episodes):
        # Reset agent at start of each episode if needed
        if hasattr(agent, "reset"):
            agent.reset()

        metrics = run_episode(env, act_fn, log_phases=log_phases)
        episodes.append(metrics)

    # Aggregate and return
    result = aggregate_episode_metrics(episodes)

    # Include phase_log from first episode if present
    if log_phases and episodes and "phase_log" in episodes[0]:
        result["phase_log"] = episodes[0]["phase_log"]

    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare RL agents against baselines")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model .zip file",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["dqn", "ppo"],
        required=True,
        help="Algorithm used for training",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML used for training",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes per agent",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Base seed for evaluation. If not set, uses config seed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: same dir as model)",
    )
    return parser.parse_args()


def main() -> None:
    """Compare trained model against baselines."""
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Use config seed if not specified (same default as eval.py)
    eval_seed = args.eval_seed if args.eval_seed is not None else cfg["env"]["seed"]

    # Create env for evaluations (metadata discovered in __init__, no reset needed)
    env = make_env(cfg, seed=eval_seed)
    n_actions = env.action_space.n
    tls_id = env.tls_id

    print("=" * 60)
    print("Baseline Comparison")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Config: {args.config}")
    print(f"Eval seed: {eval_seed}")
    print(f"TLS ID: {tls_id}")
    print(f"Phases: {n_actions}")
    print(f"Episodes per agent: {args.n_episodes}")
    print("=" * 60)

    results = {}

    def print_metrics(name: str, m: dict) -> None:
        """Print standardized metrics for an agent."""
        print(f"  Reward: {m['mean_episode_reward']:.2f} +/- {m['std_episode_reward']:.2f}")
        print(f"  Completed: {m['mean_completed']:.1f}, Injected: {m['mean_injected']:.1f}, In-network: {m['mean_in_network_end']:.1f}")
        print(f"  Mean queue: {m['mean_queue']:.2f}")
        if "phase_log" in m:
            pl = m["phase_log"]
            print(f"  Phase counts: {pl['phase_counts']}, switches: {pl['n_switches']}")
            print(f"  Phase sequence (first 30): {pl['phase_sequence_sample']}")

    # 1. Trained model
    print("\nEvaluating trained model...")
    if args.algo == "dqn":
        model = DQN.load(args.model_path)
    else:
        model = PPO.load(args.model_path)
    results["trained"] = evaluate_agent(model, env, args.n_episodes)
    print_metrics("trained", results["trained"])

    # 2. Random baseline
    print("\nEvaluating random agent...")
    random_agent = RandomAgent(n_actions, seed=eval_seed)
    results["random"] = evaluate_agent(random_agent, env, args.n_episodes)
    print_metrics("random", results["random"])

    # 3. Fixed cycle (alternating phases 0,2 for 4-phase, matching default ~40s per phase)
    print("\nEvaluating fixed cycle agent...")
    # For 4-phase intersection, alternate between opposing directions
    # With decision_interval=5s, steps_per_phase=8 gives 40s per phase (close to default 42s)
    if n_actions == 4:
        cycle_phases = [0, 2]  # N-S green then E-W green
    else:
        cycle_phases = [0, 1]  # Simple 2-phase
    fixed_agent = FixedCycleAgent(n_actions, cycle_phases=cycle_phases, steps_per_phase=8)
    results["fixed_cycle"] = evaluate_agent(fixed_agent, env, args.n_episodes, log_phases=True)
    print_metrics("fixed_cycle", results["fixed_cycle"])

    # 4. Max queue greedy
    print("\nEvaluating max-queue greedy agent...")
    greedy_agent = MaxQueueAgent(n_actions)
    results["max_queue"] = evaluate_agent(greedy_agent, env, args.n_episodes)
    print_metrics("max_queue", results["max_queue"])

    # Summary
    print("\n" + "=" * 60)
    print("Summary (higher reward = better)")
    print("=" * 60)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["mean_episode_reward"], reverse=True)
    for rank, (name, metrics) in enumerate(sorted_results, 1):
        print(
            f"{rank}. {name:15} reward={metrics['mean_episode_reward']:8.2f}  "
            f"completed={metrics['mean_completed']:5.1f}  queue={metrics['mean_queue']:.2f}"
        )

    # Check if trained beats baselines
    trained_reward = results["trained"]["mean_episode_reward"]
    random_reward = results["random"]["mean_episode_reward"]
    fixed_reward = results["fixed_cycle"]["mean_episode_reward"]

    print("\n" + "-" * 60)
    if trained_reward > random_reward:
        print("PASS: Trained model beats random baseline")
    else:
        print("FAIL: Trained model does NOT beat random baseline!")

    if trained_reward > fixed_reward:
        print("PASS: Trained model beats fixed cycle baseline")
    else:
        print("WARN: Trained model does NOT beat fixed cycle baseline")
    print("-" * 60)

    # Save results
    output_path = args.output
    if output_path is None:
        model_dir = Path(args.model_path).parent
        output_path = model_dir / "baseline_comparison.json"

    with open(output_path, "w") as f:
        json.dump(
            {
                "config": args.config,
                "model_path": args.model_path,
                "algo": args.algo,
                "n_episodes": args.n_episodes,
                "eval_seed": eval_seed,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")

    env.close()


if __name__ == "__main__":
    main()
