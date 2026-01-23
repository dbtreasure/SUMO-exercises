import numpy as np
import torch


class RolloutBuffer:
    """On-policy buffer for storing trajectories."""

    def __init__(self) -> None:
        """
        Why lists? We don't know the length of the episode upfront. Lists let us append() freely.
        obs - states we saw

        :param actions - what we did
        :param rewards - what we got (used to compute returns)
        :param log_probs - policy's log probability at collection time
        :param values - critic's estimate (unused in vanilla REINFORCE, but we include it for less refactoring later)
        :param dones - episode boundaries
        :return None
        """
        self.obs: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float = 0.0,
        done: bool = False,
    ) -> None:
        """
        Add a single timestep to the buffer.
        Called once per environment step in the training loop. The main loop does:
            action, info = agent.act(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            # agent internally calls buffer.add(next_obs, action, reward, log_prob, value, done)

        :param obs - state we saw
        :param action - action we took
        :param reward - reward we received
        :param log_prob - log probability of the action at collection time
        :param value = 0.0 - critic's estimate of the state value. defaults to 0.0
            because vanilla REINFORCE doesn't use a critic. When we add the baseline
            in following stages we'll pass real values.
        :param done - whether the episode ended. Comes last because it's derived from terminated
            or truncated.
        :return - None
        """
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self) -> None:
        """Clear all stored data after an update."""
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        """Return the number of timesteps stored in the buffer."""
        return len(self.obs)

    def compute_returns(self, gamma: float) -> np.ndarray:
        """
        Compute Monte Carlos returns: G_t = Σ γ^k r_{t+k}

        Works backwards through the buffer, resetting at episode boundaries.
        If the episode ended at step t, there are no future rewards.
        running_return = 0.0 ensures we don't bleed rewards across episode
        boundaries.

        :param gamma - discount factor. Typically 0.99. Makes future rewards
            worth slightly less than immediate rewards. γ=1.0 means "all
            rewards equal", γ=0 means "only care about immediate reward.
        """

        returns = np.zeros(len(self.obs), dtype=np.float32)
        running_return = 0.0

        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                running_return = 0.0
            running_return = self.rewards[t] + gamma * running_return
            returns[t] = running_return

        return returns

    def compute_returns_td(self, gamma: float, last_value: float) -> np.ndarray:
        """
        Compute n-step TD returns with bootstrapping.

        Like compute_returns, but instead of waiting for episode end,
        we bootstrap from the critic's estimate of the final state.

        G_t = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n * V(s_{t+n})

        :param gamma - discount factor
        :param last_value - V(s_{t+n}), critic's estimate of state after rollout
        :return - n-step TD returns
        """
        returns = np.zeros(len(self.obs), dtype=np.float32)
        running_return = last_value

        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                running_return = 0.0  # Terminal state has no future value
            running_return = self.rewards[t] + gamma * running_return
            returns[t] = running_return

        return returns

    def compute_returns_gae(self, gamma: float, gae_lambda: float, last_value: float) -> np.ndarray:
        """
        GAE: Blend ALL possible horizons with exponential decay.

        :param gamma - discount factor
        :param gae_lambda - lambda for GAE (0=TD, 1=MC)
        :param last_value - V(s_{t+n}), critic's estimate of state after rollout
        :return - GAE advantages
        """
        advantages = np.zeros(len(self.obs), dtype=np.float32)
        running_advantage = 0.0

        for t in reversed(range(len(self.rewards))):
            # The TD Residual (delta):
            # - reward I got + discounted value of where I ended up - value of where I started
            is_last_timestep = t == len(self.rewards) - 1

            # We need next_value to compute delta, but "next" means different things:
            #   - Episode ended → no next state exists → 0.0
            #   - Last step in our buffer → we don't have the value stored → ask critic via last_value
            #   - Normal case → we stored it → self.values[t + 1]
            if self.dones[t]:
                next_value = 0.0
            elif is_last_timestep:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            # This is the 1-step advantage estimate:
            # "I was in state s_t, expected value V(s_t).
            # I took an action, got reward r_t, ended up in s_{t+1} worth V(s_{t+1}).
            # Was that good?"
            # - If delta > 0: "Better than expected" (reinforce this action)
            # - If delta < 0: "Worse than expected" (discourage this action)
            delta = self.rewards[t] + gamma * next_value - self.values[t]

            # This is where GAE differs from A2C.
            # Instead of just using delta (1-step) or waiting for the full return (Monte Carlo),
            #  we're blending:
            #   A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
            # Because we loop backwards, running_advantage already contains the weighted sum of all future deltas.
            # We just add our current delta and decay what came before by γλ.
            #   - λ = 0: Only use δ_t (pure 1-step TD, high bias)
            #   - λ = 1: Sum all deltas (equivalent to Monte Carlo, high variance)
            #   - λ = 0.95: Nearby deltas matter most, distant ones fade out
            running_advantage = delta + gamma * gae_lambda * running_advantage
            advantages[t] = running_advantage

            if self.dones[t]:
                running_advantage = 0.0  # Terminal state has no future value

        # Why "Advantages" Not "Returns"?
        #   Notice we're returning advantages, not returns. In A2C we computed returns then
        #   subtracted baseline:
        #       advantages = returns - values
        #   With GAE, we compute advantages directly - the delta already has - V(s_t) baked in.
        #   The agent will use these advantages directly in the policy loss, no subtraction needed.
        return advantages

    def get(self) -> dict[str, torch.Tensor]:
        """Return all data as tensors for the update step."""

        # Notice what's missing: rewards and dones. By the time we call get(),
        # we've already used them in compute_returns(). THe update step doesn't
        # need raw rewards - it needs the computed returns (which we'll pass
        # separately).
        return {
            "obs": torch.tensor(np.array(self.obs), dtype=torch.float32),
            "actions": torch.tensor(self.actions, dtype=torch.int64),
            "log_probs": torch.tensor(self.log_probs, dtype=torch.float32),
            "values": torch.tensor(self.values, dtype=torch.float32),
        }
