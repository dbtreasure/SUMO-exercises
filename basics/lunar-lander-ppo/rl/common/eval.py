from typing import Any, Protocol

import gymnasium as gym
import numpy as np


class Actor(Protocol):
    """Protocol for agents that can act in an environment."""

    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[int, dict[str, Any]]: ...


def evaluate_agent(
    agent: Actor,
    env_id: str,
    n_episodes: int,
    seed: int | None = None,
) -> tuple[float, float]:
    """Run evaluation episodes with deterministic policy.

    Args:
        agent: Agent with an act() method.
        env_id: Gymnasium environment ID.
        n_episodes: Number of episodes to run.
        seed: Optional base seed for reproducibility. Each episode uses seed + episode_num.

    Returns:
        (mean_return, std_return)
    """
    env = gym.make(env_id)
    returns: list[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=(seed + ep) if seed else None)
        episode_return = 0.0
        done = False

        while not done:
            action, _ = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            done = terminated or truncated

        returns.append(episode_return)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))
