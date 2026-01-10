#!/usr/bin/env python
"""Train a PPO agent on the SingleIntersectionTSCEnv.

Usage:
    uv run python -m src.training.train_ppo --config configs/single_int_ppo.yaml
    uv run python -m src.training.train_ppo --config configs/single_int_ppo.yaml --run-name my_run
"""

from __future__ import annotations

import argparse
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.tsc_env import SingleIntersectionTSCEnv


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(
    cfg: dict, max_sim_time_s: int | None = None, seed: int | None = None
) -> SingleIntersectionTSCEnv:
    """Create a TSC environment from config.

    Args:
        cfg: Configuration dict.
        max_sim_time_s: Override max simulation time.
        seed: Override seed. If None, uses config seed. Pass -1 for stochastic (no seed).
    """
    env_cfg = cfg["env"]
    # Determine seed: -1 means stochastic (None to SUMO), None means use config
    if seed == -1:
        env_seed = None
    elif seed is not None:
        env_seed = seed
    else:
        env_seed = env_cfg["seed"]

    return SingleIntersectionTSCEnv(
        sumocfg_path=env_cfg["sumocfg_path"],
        decision_interval_s=env_cfg["decision_interval_s"],
        min_green_s=env_cfg["min_green_s"],
        max_sim_time_s=max_sim_time_s or env_cfg["max_sim_time_s"],
        seed=env_seed,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO on TSC environment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name (default: timestamp)",
    )
    return parser.parse_args()


def main() -> None:
    """Train PPO agent."""
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Set seed
    seed = cfg["env"]["seed"]
    set_seed(seed)

    # Create run directory
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / "ppo" / run_name
    models_dir = run_dir / "models"
    tb_dir = run_dir / "tb"
    models_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    shutil.copy(args.config, run_dir / "config.yaml")

    # Create training environment
    # Note: We need to reset once to set correct observation space before wrapping
    def make_train_env():
        env = make_env(cfg)
        env.reset()  # Sets correct observation_space
        return Monitor(env)

    train_env = DummyVecEnv([make_train_env])

    # Create eval environment with shorter sim time and offset seed
    # Default eval seed is training seed + 10000 to avoid overlap
    eval_max_sim = cfg["train"].get("eval_max_sim_time_s", cfg["env"]["max_sim_time_s"])
    eval_seed = cfg["train"].get("eval_seed", seed + 10000)

    def make_eval_env():
        env = make_env(cfg, max_sim_time_s=eval_max_sim, seed=eval_seed)
        env.reset()  # Sets correct observation_space
        return Monitor(env)

    eval_env = DummyVecEnv([make_eval_env])

    # Print metadata
    temp_env = make_env(cfg)
    obs, info = temp_env.reset()
    print("=" * 60)
    print("PPO Training")
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    print(f"TLS ID: {info['tls_id']}")
    print(f"Controlled lanes: {info['controlled_lanes']}")
    print(f"Phases: {info['n_phases']}")
    print(f"Observation shape: {obs.shape}")
    print(f"Total timesteps: {cfg['train']['total_timesteps']}")
    print("=" * 60)
    temp_env.close()

    # Build PPO hyperparams
    sb3_cfg = cfg["sb3"]
    policy_kwargs = sb3_cfg.get("policy_kwargs", {})

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=sb3_cfg["learning_rate"],
        n_steps=sb3_cfg["n_steps"],
        batch_size=sb3_cfg["batch_size"],
        n_epochs=sb3_cfg["n_epochs"],
        gamma=sb3_cfg["gamma"],
        gae_lambda=sb3_cfg["gae_lambda"],
        clip_range=sb3_cfg["clip_range"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tb_dir),
        seed=seed,
        verbose=1,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg["train"]["save_freq_steps"],
        save_path=str(models_dir),
        name_prefix="rl_model",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir),
        log_path=str(run_dir),
        eval_freq=cfg["train"]["eval_freq_steps"],
        n_eval_episodes=cfg["train"]["n_eval_episodes"],
        deterministic=True,
    )

    # Train
    model.learn(
        total_timesteps=cfg["train"]["total_timesteps"],
        callback=[checkpoint_callback, eval_callback],
        log_interval=cfg["train"]["log_interval"],
    )

    # Save final model
    model.save(models_dir / "final")
    print(f"\nTraining complete. Model saved to {models_dir / 'final.zip'}")

    # Cleanup
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
