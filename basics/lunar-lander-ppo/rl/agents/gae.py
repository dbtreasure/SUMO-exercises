from typing import Any

import numpy as np
import torch

from rl.agents.base import Agent
from rl.common.buffers import RolloutBuffer
from rl.common.config import Config
from rl.common.nets import CategoricalPolicy, Critic
from rl.common.utils import normalize


class GAEAgent(Agent):
    """Actor-Critic with Generalized Advantage Estimation (GAE)."""

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
        # Number of steps to look ahead for n-step returns.
        self.n_steps = config.algo.n_steps
        # GAE Lambda
        self.gae_lambda = config.algo.gae_lambda

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

    def update(self, last_obs: np.ndarray | None = None) -> dict[str, float]:
        """Perform GAE update with ALL possible horizons with exponential decay.

        :param last_obs - The observation after the rollout (s_{t+n}), used to
            get V(s_{t+n}) for computing n-step returns.
        """

        # GAE requires last_obs for bootstrapping
        if last_obs is None:
            raise ValueError("GAE.update() requires last_obs for bootstrapping")

        # Early exit, don't update if there's nothing in the buffer
        if len(self.buffer) == 0:
            return {}

        # Get V(s_n) for bootstrapping - the critic's estimate of the state after rollout
        with torch.no_grad():
            last_obs_tensor = torch.tensor(last_obs, dtype=torch.float32).unsqueeze(0)
            last_value = self.critic(last_obs_tensor).item()

        # GAE: compute advantages as exponentially-weighted blend of all n-step estimates
        # A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        advantages = self.buffer.compute_returns_gae(self.gamma, self.gae_lambda, last_value)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)

        data = self.buffer.get()
        values_tensor = data["values"]

        # Recover returns for value loss target: G_t = A_t + V(s_t)
        returns_tensor = advantages_tensor + values_tensor

        if self.normalize_returns:
            advantages_tensor = normalize(advantages_tensor)

        # Recompute log_probs and entropy with current policy
        log_probs, entropy = self.policy.evaluate_actions(data["obs"], data["actions"])

        # Policy gradient loss using advantage instead of raw returns
        # Formula: loss = -E[log π(a|s) * A(s,a)] where A = G_t - V(s)
        # - Positive advantage -> action was better than expected, reinforce it
        # - Negative advantage -> action was worse than expected, discourage it
        # - The minus sign because we're doing gradient _descent_ but want
        #   to _maximize_ expected return
        policy_loss = -(log_probs * advantages_tensor.detach()).mean()

        # Entropy bonus (encourages exploration)
        # Higher entropy = more exploration. We want to _maximize_ entropy, so we negate it
        # (minimizing negative entropy = maximizing entropy). With entropy_coef=0.0 this has no effect.
        entropy_loss = -entropy.mean()

        # Value loss: train critic to predict returns
        # IMPORTANT: We must recompute values WITH gradients here!
        # The stored values from act() were computed under torch.no_grad() and have
        # no gradient connection to the critic. We need fresh predictions.
        values_pred = self.critic(data["obs"]).squeeze(-1)
        value_loss = ((values_pred - returns_tensor) ** 2).mean()

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
            "misc/return_mean": returns_tensor.mean().item(),
        }

    def on_episode_end(self) -> dict[str, float]:
        """GAE updates every n_steps, not per episode. Returns empty dict."""
        return {}

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
