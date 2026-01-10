"""Shared evaluation metrics for RL agents.

This module provides a single source of truth for episode evaluation,
ensuring eval.py and compare_baselines.py report consistent metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypedDict

import numpy as np

if TYPE_CHECKING:
    from src.envs.tsc_env import SingleIntersectionTSCEnv


class PhaseLog(TypedDict):
    """Phase logging data for debugging fixed-cycle and other agents."""

    phase_counts: dict[int, int]  # phase_idx -> number of steps in that phase
    n_switches: int  # number of phase changes
    phase_sequence_sample: list[int]  # first 30 phase indices


class EpisodeMetricsRequired(TypedDict):
    """Required metrics returned by run_episode."""

    episode_reward: float  # sum of step rewards (negative waiting time)
    completed: int  # vehicles that finished trip (SUMO arrived)
    injected: int  # vehicles that entered network (SUMO departed)
    in_network_end: int  # injected - completed (vehicles still in network at end)
    mean_queue: float  # mean of total halting count per step (sum across all lanes)
    n_steps: int  # number of decision steps in episode


class EpisodeMetrics(EpisodeMetricsRequired, total=False):
    """Metrics returned by run_episode.

    Required keys are always present. Optional keys (phase_log) only when requested.
    """

    phase_log: PhaseLog  # only present when log_phases=True


def run_episode(
    env: "SingleIntersectionTSCEnv",
    act_fn: Callable[[np.ndarray], int],
    log_phases: bool = False,
) -> EpisodeMetrics:
    """Run a single evaluation episode and collect metrics.

    Args:
        env: The TSC environment. Should already be created with desired seed.
             The env internally increments episode_count on each reset(),
             producing seed variation as: effective_seed = base_seed + episode_count.
        act_fn: Function that takes observation array and returns action (int).
                For SB3 models: lambda obs: model.predict(obs, deterministic=True)[0]
                For baseline agents: lambda obs: agent.predict(obs, deterministic=True)[0]
        log_phases: If True, include phase_log in returned metrics.

    Returns:
        EpisodeMetrics dict with standardized keys.

    Note on mean_queue:
        This is the mean of (sum of halting vehicles across all lanes) per step.
        NOT the mean per-lane queue. This gives a single number representing
        total intersection congestion at each decision point.
    """
    obs, _info = env.reset()

    episode_reward = 0.0
    completed = 0
    injected = 0
    queue_samples: list[int] = []  # total halting count per step
    phases: list[int] = []  # for phase logging
    n_steps = 0

    done = False
    while not done:
        action = act_fn(obs)

        # Handle numpy array actions (from SB3 predict)
        if isinstance(action, np.ndarray):
            action = int(action.item()) if action.size == 1 else int(action[0])

        if log_phases:
            phases.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += float(reward)
        completed += info.get("arrived", 0)
        injected += info.get("departed", 0)
        queue_samples.append(sum(info.get("halting", [])))
        n_steps += 1

    mean_queue = float(np.mean(queue_samples)) if queue_samples else 0.0

    result: EpisodeMetrics = {
        "episode_reward": episode_reward,
        "completed": completed,
        "injected": injected,
        "in_network_end": injected - completed,
        "mean_queue": mean_queue,
        "n_steps": n_steps,
    }

    if log_phases and phases:
        phase_counts: dict[int, int] = {}
        for p in phases:
            phase_counts[p] = phase_counts.get(p, 0) + 1
        n_switches = sum(1 for i in range(1, len(phases)) if phases[i] != phases[i - 1])
        result["phase_log"] = {
            "phase_counts": phase_counts,
            "n_switches": n_switches,
            "phase_sequence_sample": phases[:30],
        }

    return result


def aggregate_episode_metrics(episodes: list[EpisodeMetrics]) -> dict[str, float | int]:
    """Aggregate metrics across multiple episodes.

    Args:
        episodes: List of EpisodeMetrics from run_episode calls.

    Returns:
        Dict with mean/std for each metric, plus episode count.
    """
    rewards = [ep["episode_reward"] for ep in episodes]
    completed = [ep["completed"] for ep in episodes]
    injected = [ep["injected"] for ep in episodes]
    queues = [ep["mean_queue"] for ep in episodes]
    lengths = [ep["n_steps"] for ep in episodes]

    mean_completed = float(np.mean(completed))
    mean_injected = float(np.mean(injected))

    return {
        "mean_episode_reward": float(np.mean(rewards)),
        "std_episode_reward": float(np.std(rewards)),
        "mean_completed": mean_completed,
        "std_completed": float(np.std(completed)),
        "mean_injected": mean_injected,
        "std_injected": float(np.std(injected)),
        "mean_in_network_end": mean_injected - mean_completed,
        "mean_queue": float(np.mean(queues)),
        "std_queue": float(np.std(queues)),
        "mean_n_steps": float(np.mean(lengths)),
        "n_episodes": len(episodes),
    }
