import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


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

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        assert env.observation_space.shape is not None
        assert isinstance(env.action_space, gym.spaces.Discrete)
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # Policy network (the one we train)
        self.policy_net = DQN(obs_size, int(n_actions))
        # Target network (stable copy for computing tarets)
        self.target_net = DQN(obs_size, int(n_actions))
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.replay_buffer = ReplayBuffer(buffer_size)

    def get_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        # convert state to tensor, add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # No gradient needef for action selection
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(q_values.argmax().item())

    def update(self):
        # Don't train until we have enough experiences
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample random batch of experiences
        batch = self.replay_buffer.sample(self.batch_size)

        # Unpack batch into separate arrays
        states = torch.FloatTensor(np.array([exp[0] for exp in batch]))
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor(np.array([exp[3] for exp in batch]))
        dones = torch.FloatTensor([exp[4] for exp in batch])

        # Current Q-values: what does our network predict for the actions we took?
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: reward + gamma * max Q(next_state) from target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss: how far off were our predictions?
        loss = nn.MSELoss()(current_q, target_q)

        # Backprop and update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def sync_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
