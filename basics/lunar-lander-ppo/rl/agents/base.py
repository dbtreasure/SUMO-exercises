from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Agent(ABC):
    """Abstract base class for all agents."""

    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[int, dict[str, Any]]:
        """Select an action given an observation

        :param obs - Observation from the environment
        :param deterministic - Whether to select a deterministic action
        :return (action, info) - Tuple of action and additional information.
            This lets different agents return different info without changing the signature.
            REINFORCE returns {"log_prob": ...}, PPO might add {"value": ...}.
        """
        ...

    @abstractmethod
    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Store a transition in the agent's buffer
        Even though REINFORCE doesn't use next_obs, we include it for consistency.
        A2C/PPO will need it for bootstrapping.

        :param obs - Observation from the environment
        :param action - Action taken by the agent
        :param reward - Reward received from the environment
        :param next_obs - Next observation from the environment
        :param terminated - Whether the episode has terminated
        :param truncated - Whether the episode has been truncated
        :param info - Additional information from the environment
        """
        ...

    @abstractmethod
    def update(self, last_obs: np.ndarray | None = None) -> dict[str, float]:
        """Perform a learning update.

        update() vs on_episode_end(): This is the key flexibility.
        REINFORCE calls update() inside on_episode_end() (needs full episode).
        A2C/PPO call update() every N steps with last_obs for bootstrapping.

        :param last_obs - For A2C/PPO: the observation after the rollout, used
            to bootstrap V(s_n). REINFORCE ignores this parameter.
        :return (losses, etc.) - Dictionary of losses and other metrics
        """
        ...

    def on_episode_end(self) -> dict[str, float]:
        """Called at the end of each episode. Override if needed.

        REINFORCE updates here (full episode needed).
        PPO/A2C update based on steps, not episodes, so they don't use this.

        :return (losses, etc.) - Dictionary of losses and other metrics
        """
        return {}

    def save(self, path: str) -> None:
        """Save agent state to disk.

        :param path - Path to save the agent state to
        """
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load agent state from disk.

        :param path - Path to load the agent state from
        """
        raise NotImplementedError
