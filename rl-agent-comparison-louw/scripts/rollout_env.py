#!/usr/bin/env python
"""Rollout script to test the TSC environment with random actions.

Usage:
    uv run python scripts/rollout_env.py
    uv run python scripts/rollout_env.py --n-steps 50 --max-sim-time 600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.tsc_env import SingleIntersectionTSCEnv


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run TSC environment rollout with random actions."
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=20,
        help="Number of environment steps to run (default: 20)",
    )
    parser.add_argument(
        "--decision-interval",
        type=float,
        default=1.0,
        help="Seconds per agent decision step (default: 1.0)",
    )
    parser.add_argument(
        "--min-green",
        type=float,
        default=5.0,
        help="Minimum green time in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--max-sim-time",
        type=int,
        default=300,
        help="Maximum simulation time in seconds (default: 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    """Run a rollout with random actions and print step info."""
    args = parse_args()

    # Path to scenario config (relative to project root)
    sumocfg_path = project_root / "scenarios" / "single_int" / "sim.sumocfg"

    print("=" * 80)
    print("TSC Environment Rollout")
    print("=" * 80)
    print(f"Config:           {sumocfg_path}")
    print(f"Steps:            {args.n_steps}")
    print(f"Decision interval: {args.decision_interval}s")
    print(f"Min green:        {args.min_green}s")
    print(f"Max sim time:     {args.max_sim_time}s")
    print(f"Seed:             {args.seed}")
    print("=" * 80)

    env = SingleIntersectionTSCEnv(
        sumocfg_path=str(sumocfg_path),
        decision_interval_s=args.decision_interval,
        min_green_s=args.min_green,
        max_sim_time_s=args.max_sim_time,
        seed=args.seed,
    )

    try:
        obs, info = env.reset()
        print(f"TLS: {info['tls_id']}, Lanes: {info['controlled_lanes']}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space} (phases: {info['n_phases']})")
        print("-" * 80)
        print(f"{'Step':>4} | {'Time':>7} | {'Act':>3} | {'Phase':>5} | {'Observation':<24} | {'Reward':>10}")
        print("-" * 80)

        for step in range(args.n_steps):
            # Sample random action
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            obs_str = "[" + ", ".join(f"{v:5.1f}" for v in obs) + "]"
            print(
                f"{step:4d} | {info['sim_time']:7.1f} | {action:3d} | {info['phase']:5d} | {obs_str:<24} | {reward:10.2f}"
            )

            if terminated or truncated:
                print(f"\nSimulation ended at step {step}")
                break

        print("-" * 80)
        print("Rollout complete.")

    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()
