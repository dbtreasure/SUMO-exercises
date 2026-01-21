# Basics

Foundational exercises to learn Gymnasium and reinforcement learning concepts.

## Projects

### CartPole

Classic control problem: balance a pole on a moving cart.

Two agent implementations:
- **Tabular Q-Learning**: Discretizes continuous state space into bins, uses Q-table
- **DQN (Deep Q-Network)**: Neural network approximates Q-values, handles continuous states natively

See [cartpole/README.md](cartpole/README.md) for details.

### LunarLander (DQN)

Deep Q-Network implementation for LunarLander-v2. Value-based approach with experience replay and target networks.

See [lunar-lander/README.md](lunar-lander/README.md) for details.

### LunarLander (Policy Gradient → PPO)

Learning policy gradient methods by implementing an incremental path:

| Stage | Algorithm | Status |
|-------|-----------|--------|
| 0 | REINFORCE | Done |
| 1 | REINFORCE + baseline | Done |
| 2 | A2C | Next |
| 3 | GAE | Planned |
| 4 | PPO | Planned |

See [lunar-lander-ppo/README.md](lunar-lander-ppo/README.md) for details.
