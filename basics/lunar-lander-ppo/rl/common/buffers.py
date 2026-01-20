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
