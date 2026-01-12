# CartPole

Balance a pole on a moving cart using reinforcement learning.

## Environment

- **Observation**: 4 continuous values (position, velocity, pole angle, angular velocity)
- **Actions**: 2 discrete (push left, push right)
- **Reward**: +1 per timestep the pole stays upright
- **Max reward**: 500 (episode truncates)

## Agents

### Tabular Q-Learning (`cart_pole_agent.py`)

Traditional Q-learning with state discretization.

- Bins continuous observations into discrete states (20 bins per dimension = 160,000 states)
- Q-table maps state-action pairs to expected rewards
- Epsilon-greedy exploration with linear decay

**Results**: After 30,000 episodes, achieves ~260 average reward with occasional 500s.

```bash
uv run python main.py
```

### DQN (`cart_pole_agent_dqn.py`)

Deep Q-Network using PyTorch.

- Neural network (4 -> 128 -> 128 -> 2) approximates Q-values
- Experience replay buffer for stable training
- Target network updated every 10 episodes
- Epsilon-greedy exploration with exponential decay

**Results**: After 1,000 episodes, achieves consistent 500 reward (solved).

```bash
uv run python main_dqn.py
```

## Comparison

| Metric | Tabular Q-Learning | DQN |
|--------|-------------------|-----|
| Episodes to train | 30,000 | 1,000 |
| Eval average | ~260 | 500 |
| State handling | Discretized | Continuous |

## Dependencies

```bash
uv add "gymnasium[classic-control]" numpy torch
```
