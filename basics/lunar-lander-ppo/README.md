# Policy Gradient Learning Path - LunarLander-v2

An incremental implementation of policy gradient algorithms for learning purposes.

## Algorithms

| Stage | Algorithm | Key Concept | Status |
|-------|-----------|-------------|--------|
| 0 | REINFORCE | Monte Carlo returns, policy gradient theorem | Done |
| 1 | REINFORCE + baseline | Variance reduction with value function | Next |
| 2 | A2C | TD bootstrapping, online learning | Planned |
| 3 | GAE | Lambda-weighted advantage estimation | Planned |
| 4 | PPO | Clipped objective, minibatches, multiple epochs | Planned |

## Quick Start

```bash
# Install dependencies
uv sync

# Run training
uv run python main.py --config configs/reinforce.yaml

# Monitor with TensorBoard
uv run tensorboard --logdir runs/

# Results saved to runs/<run_id>/
```

## Project Structure

```
main.py              # Training entrypoint
configs/             # YAML configs per algorithm
  reinforce.yaml     # 500k steps, vanilla REINFORCE
  reinforce_long.yaml # 2M steps with entropy bonus
rl/
  common/            # Shared utilities
    buffers.py       # RolloutBuffer (on-policy storage)
    config.py        # Pydantic config loading
    envs.py          # Environment factory
    eval.py          # Evaluation loop
    logger.py        # TensorBoard + CSV logging
    nets.py          # CategoricalPolicy network
    plot.py          # Training curve visualization
    seeding.py       # Reproducibility
    utils.py         # Normalization helpers
  agents/
    base.py          # Agent ABC
    reinforce.py     # Stage 0 implementation
runs/                # Training outputs (metrics, plots, checkpoints)
```

## Stage 0 Results: REINFORCE

Trained for 500k steps on LunarLander-v2:

**Performance:**
- Start: ~-350 eval return (crashing immediately)
- End: ~-50 eval return (hovering, occasional landing)
- High variance throughout - classic REINFORCE behavior

**What we learned:**
- Policy gradient formula: `∇J = E[log π(a|s) * G_t]`
- Monte Carlo returns computed backwards for O(n) efficiency
- Loss oscillates around zero (normal for RL, unlike supervised learning)
- Entropy stayed healthy (~1.0) - policy didn't collapse

**Why it doesn't solve LunarLander:**
REINFORCE uses raw returns as the learning signal. Episode A might get +100, Episode B might get -200, even with similar policies. The gradient is too noisy.

## Expected Performance

| Algorithm | Expected Return | Notes |
|-----------|-----------------|-------|
| REINFORCE | -50 to +50 | High variance, doesn't solve |
| REINFORCE+baseline | 100-180 | Reduced variance |
| A2C | 150-200 | Faster learning |
| PPO | 200+ | Stable, sample efficient |

## References

- [Policy Gradient Methods (Sutton & Barto Ch. 13)](http://incompleteideas.net/book/the-book.html)
- [PPO Paper (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347)
- [GAE Paper (Schulman et al. 2015)](https://arxiv.org/abs/1506.02438)
