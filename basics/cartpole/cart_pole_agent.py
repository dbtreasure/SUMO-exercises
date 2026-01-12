from collections import defaultdict
from typing import Literal, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class CartPoleAgent:
    def __init__(
        self,
        env: gym.Env[NDArray[np.float32], int],
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize the cartpole agent.

        Args:
            env: The environment to train on.
            learning_rate: The learning rate for the agent.
            initial_epsilon: The initial epsilon value for epsilon-greedy exploration.
            epsilon_decay: The decay rate for epsilon-greedy exploration.
            final_epsilon: The final epsilon value for epsilon-greedy exploration.
            discount_factor: The discount factor for future rewards.
        """
        self.env = env

        self.action_space: gym.spaces.Discrete = env.action_space  # type: ignore[assignment]
        self.n_actions = self.action_space.n

        # Discretization bins for continuous observation space
        # 20 bins per dimension = 20^4 = 160,000 possible states
        self.bins = [
            np.linspace(-2.4, 2.4, 20),      # position
            np.linspace(-3.0, 3.0, 20),      # velocity
            np.linspace(-0.42, 0.42, 20),    # angle
            np.linspace(-3.0, 3.0, 20),      # angular velocity
        ]

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values: defaultdict[tuple[float, ...], NDArray[np.float64]] = (
            defaultdict(lambda: np.zeros(self.n_actions))
        )
        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def discretize(self, obs: NDArray[np.float32]) -> tuple[int, int, int, int]:
        """Convert continuous observation to discrete bin indices."""
        state = []
        for i, val in enumerate(obs):
            # np.digitize returns which bin the value falls into
            bin_index = np.digitize(val, self.bins[i]) - 1
            # clamp to valid range (0 to num_bins - 1)
            bin_index = max(0, min(bin_index, len(self.bins[i]) - 1))
            state.append(bin_index)
        return tuple(state)

    def get_action(self, obs: NDArray[np.float32]) -> Literal[0, 1]:
        """Select an action given the current observation.
        Returns:
            action: 0 (left) or 1 (right)
        """
        state = self.discretize(obs)

        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return cast(Literal[0, 1], self.env.action_space.sample())

        # With probability (1-epsilon): exploit (best known action)
        return cast(Literal[0, 1], int(np.argmax(self.q_values[state])))

    def update(
        self,
        obs: NDArray[np.float32],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: NDArray[np.float32],
    ) -> None:
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        state = self.discretize(obs)
        next_state = self.discretize(next_obs)

        # Q-learning formula: Q(s,a) += lr * (reward + γ * max(Q(s')) - Q(s,a))
        future_q = 0.0 if terminated else float(np.max(self.q_values[next_state]))
        current_q = self.q_values[state][action]

        td_error = reward + self.discount_factor * future_q - current_q
        self.q_values[state][action] += self.lr * td_error

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
