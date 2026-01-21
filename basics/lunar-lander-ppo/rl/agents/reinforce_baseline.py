from typing import Any

import numpy as np
import torch

from rl.agents.base import Agent
from rl.common.buffers import RolloutBuffer
from rl.common.config import Config
from rl.common.nets import CategoricalPolicy, Critic
from rl.common.utils import normalize


class ReinforceBaselineAgent(Agent):
    """REINFORCE with baseline (value function) for variance reduction."""

    def __init__(self, obs_dim: int, act_dim: int, config: Config) -> None:
        """Initialize the agent.

        :param obs_dim - Dimension of the observation space
        :param act_dim - Dimension of the action space
        :param config - Configuration object
        """
        # Discount factor (0.99). How much to weight future rewards.
        self.gamma = config.algo.gamma
        # Whether to normalize advantages before computing gradients. Helps stability.
        self.normalize_returns = config.algo.normalize_returns
        # Weight for entropy bonus (0.0 for vanilla REINFORCE, can add later for exploration).
        self.entropy_coef = config.algo.entropy_coef
        # For gradient clipping. None means no clipping.
        self.max_grad_norm = config.algo.max_grad_norm

        # Our neural network that maps obs -> action distribution
        self.policy = CategoricalPolicy(obs_dim, act_dim, config.model.hidden_sizes)
        # Adam optimizer. Adam is the default - it adapts learning rates per-parameter and handles
        # sparse gradients well.
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.algo.lr_actor)

        # The critic (value network) estimates V(s) - expected return from each state.
        # This is our "baseline" that reduces variance in policy gradients.
        self.critic = Critic(obs_dim, config.model.hidden_sizes)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.algo.lr_critic)

        # Where we store the episode as it unfolds
        self.buffer = RolloutBuffer()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[int, dict[str, Any]]:
        """Select an action given an observation.

        :param obs - Observation
        :param deterministic - Whether to select the action deterministically
        :return action - Selected action
        :return info - Additional information about the action
        """

        # We're not training here, just running inference. This tells Pytorch not to
        # track gradients, which saves memory and compute, during the actual update,
        # we'll recompute with gradients enabled.
        with torch.no_grad():
            # Convert numpy array to Pytorch tensor. The policy expects tensors.
            # unsqueeze(0) - Add a batch dimension. Our policy expects shape (batch_size, obs_dim)
            # A single observationis (8,), so we make it (1, 8)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            # Returns (action, log_prob) as Python scalars. We wrote this in nets.py
            action, log_prob = self.policy.get_action(obs_tensor, deterministic)

            # Get value estimate for this state - used to compute advantage later
            value = self.critic(obs_tensor).item()

        # The training loop will pass this to observe() so we can store it in the buffer.
        return action, {"log_prob": log_prob, "value": value}

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
        """Store a transition in the buffer.

        :param obs: Observation
        :param action: Action taken
        :param reward: Reward received
        :param next_obs: Next observation
        :param terminated: Whether the episode terminated
        :param truncated: Whether the episode was truncated
        :param info: Additional information about the transition
        """

        done = terminated or truncated
        self.buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            log_prob=info["log_prob"],
            value=info["value"],
            done=done,
        )

    def update(self) -> dict[str, float]:
        """Perform REINFORCE + baseline update.

        Key difference from vanilla REINFORCE: we use advantage (return - value)
        instead of raw returns. This dramatically reduces variance because we're
        asking "how much better than expected?" rather than "how good absolutely?"
        """

        # Early exit, don't update if there's nothing in the buffer
        if len(self.buffer) == 0:
            return {}

        # Compute Monte Carlo returns G_t = sum of discounted future rewards
        returns = self.buffer.compute_returns(self.gamma)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        data = self.buffer.get()
        values_tensor = data["values"]

        # Advantage A(s,a) = G_t - V(s)
        # "How much better was this action than what we expected from this state?"
        # This is the key insight of the baseline: center the learning signal around
        # the expected value, so we only reinforce actions that beat expectations.
        advantages = returns_tensor - values_tensor

        if self.normalize_returns:
            advantages = normalize(advantages)

        # Recompute log_probs and entropy with current policy
        log_probs, entropy = self.policy.evaluate_actions(data["obs"], data["actions"])

        # Policy gradient loss using advantage instead of raw returns
        # Formula: loss = -E[log π(a|s) * A(s,a)] where A = G_t - V(s)
        # - Positive advantage -> action was better than expected, reinforce it
        # - Negative advantage -> action was worse than expected, discourage it
        # - The minus sign because we're doing gradient _descent_ but want
        #   to _maximize_ expected return
        policy_loss = -(log_probs * advantages).mean()

        # Entropy bonus (encourages exploration)
        # Higher entropy = more exploration. We want to _maximize_ entropy, so we negate it
        # (minimizing negative entropy = maximizing entropy). With entropy_coef=0.0 this has no effect.
        entropy_loss = -entropy.mean()

        # Value loss: train critic to predict returns
        # MSE between predicted V(s) and actual returns
        value_loss = ((data["values"] - returns_tensor) ** 2).mean()

        # Total loss
        # The 0.5 coefficient is a common choice - it balances the critic learning rate.
        # Some people make this configurable, but 0.5 is standard.
        loss = policy_loss + self.entropy_coef * entropy_loss + 0.5 * value_loss

        # Gradient descent - one backward pass, step both optimizers
        self.optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.critic_optimizer.step()

        # Clear buffer (on-policy: data is stale after update)
        self.buffer.clear()

        return {
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item(),
            "loss/entropy": entropy_loss.item(),
            "misc/entropy": entropy.mean().item(),
            "misc/return_mean": returns.mean(),
        }

    def on_episode_end(self) -> dict[str, float]:
        """Perform update at the end of each episode.

        Why not just call update() directly in the training loop?
        The training loop doesn't know when to call update - that's
        algorithm specific:
            - REINFORCE: Update per episode (here)
            - A2C/PPO: Update every N steps (in the main loop, not in on_episode_end)
        """
        return self.update()

    def save(self, path: str) -> None:
        """Save policy and critic weights to disk."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "policy_optimizer_state_dict": self.optimizer.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load policy and critic weights from disk."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
