import random
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


@dataclass
class UpdateResult:
    """Result from a single update step."""

    loss: float
    grad_norm_preclip: float
    grad_clipped: bool


def clip_and_get_norm(
    parameters, max_grad_norm: float | None
) -> tuple[float, bool]:
    """Clip gradients by global norm and return pre-clip norm.

    Args:
        parameters: Model parameters to clip
        max_grad_norm: Maximum gradient norm. If None or <= 0, no clipping.

    Returns:
        (preclip_norm, did_clip): The gradient norm before clipping and whether clipping occurred.
    """
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0, False

    # Compute total norm before clipping
    total_norm = torch.nn.utils.clip_grad_norm_(
        params,
        max_norm=float("inf"),  # Don't clip yet, just compute norm
    )
    preclip_norm = total_norm.item()

    # Apply clipping if enabled
    did_clip = False
    if max_grad_norm is not None and max_grad_norm > 0:
        if preclip_norm > max_grad_norm:
            did_clip = True
            # Scale gradients to have norm = max_grad_norm
            clip_coef = max_grad_norm / (preclip_norm + 1e-6)
            for p in params:
                p.grad.detach().mul_(clip_coef)

    return preclip_norm, did_clip


class DQN(nn.Module):
    def __init__(self, obs_size: int, n_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: NDArray,
        action: int,
        reward: float,
        next_state: NDArray,
        done: bool,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> list[tuple[NDArray, int, float, NDArray, bool]]:
        return random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class TrainingLogger:
    def __init__(self, window_size: int = 100):
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_losses: list[float] = []
        self.episode_grad_norms: list[float] = []
        self.episode_clip_fractions: list[float] = []
        self.window_size = window_size

    def log_episode(
        self,
        reward: float,
        length: int,
        avg_loss: float,
        avg_grad_norm: float = 0.0,
        clip_fraction: float = 0.0,
    ):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_losses.append(avg_loss)
        self.episode_grad_norms.append(avg_grad_norm)
        self.episode_clip_fractions.append(clip_fraction)

    def get_stats(self) -> dict:
        recent_rewards = self.episode_rewards[-self.window_size :]
        recent_lengths = self.episode_lengths[-self.window_size :]
        recent_losses = self.episode_losses[-self.window_size :]
        recent_grad_norms = self.episode_grad_norms[-self.window_size :]
        recent_clip_fractions = self.episode_clip_fractions[-self.window_size :]
        return {
            "mean_reward": np.mean(recent_rewards) if recent_rewards else 0,
            "std_reward": np.std(recent_rewards) if recent_rewards else 0,
            "mean_length": np.mean(recent_lengths) if recent_lengths else 0,
            "mean_loss": np.mean(recent_losses) if recent_losses else 0,
            "mean_grad_norm": np.mean(recent_grad_norms) if recent_grad_norms else 0,
            "clip_fraction": np.mean(recent_clip_fractions) if recent_clip_fractions else 0,
            "episodes": len(self.episode_rewards),
        }


class DQNAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.0001,
        gamma: float = 0.999,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        tau: float = 0.005,
        max_grad_norm: float | None = 10.0,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.max_grad_norm = max_grad_norm

        assert env.observation_space.shape is not None
        assert isinstance(env.action_space, gym.spaces.Discrete)
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # Policy network (the one we train)
        self.policy_net = DQN(obs_size, int(n_actions))
        # Target network (stable copy for computing targets)
        self.target_net = DQN(obs_size, int(n_actions))
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.replay_buffer = ReplayBuffer(buffer_size)

    def get_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(q_values.argmax().item())

    def update(self) -> UpdateResult | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(np.array([exp[0] for exp in batch]))
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor(np.array([exp[3] for exp in batch]))
        dones = torch.FloatTensor([exp[4] for exp in batch])

        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: policy net selects action, target net evaluates
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm, did_clip = clip_and_get_norm(
            self.policy_net.parameters(), self.max_grad_norm
        )

        self.optimizer.step()

        return UpdateResult(
            loss=loss.item(),
            grad_norm_preclip=grad_norm,
            grad_clipped=did_clip,
        )

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def soft_update_target(self) -> None:
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )

    def sync_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_checkpoint(self, path: str, episode: int, total_steps: int) -> None:
        torch.save(
            {
                "episode": episode,
                "total_steps": total_steps,
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> dict:
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        return {
            "episode": checkpoint["episode"],
            "total_steps": checkpoint["total_steps"],
        }
