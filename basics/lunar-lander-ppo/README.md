# Policy Gradient Learning Path - LunarLander-v2

An incremental implementation of policy gradient algorithms for learning purposes.

## Algorithms

| Stage | Algorithm | Key Concept | Status |
|-------|-----------|-------------|--------|
| 0 | REINFORCE | Monte Carlo returns, policy gradient theorem | Done |
| 1 | REINFORCE + baseline | Variance reduction with value function | Done |
| 2 | A2C | TD bootstrapping, online learning | Next |
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
  reinforce_baseline.yaml      # 500k steps, with value baseline
  reinforce_baseline_long.yaml # 2M steps, with value baseline
rl/
  common/            # Shared utilities
    buffers.py       # RolloutBuffer (on-policy storage)
    config.py        # Pydantic config loading
    envs.py          # Environment factory
    eval.py          # Evaluation loop
    logger.py        # TensorBoard + CSV logging
    nets.py          # CategoricalPolicy, Critic networks
    plot.py          # Training curve visualization
    seeding.py       # Reproducibility
    utils.py         # Normalization helpers
  agents/
    base.py          # Agent ABC
    reinforce.py     # Stage 0 implementation
    reinforce_baseline.py # Stage 1 implementation
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

## Stage 1 Results: REINFORCE + Baseline

Added a Critic network that estimates V(s). Instead of raw returns, we use advantage: `A(s,a) = G_t - V(s)`.

**Performance (2M steps):**
- Peak: +258 eval return at 750k steps (crossed solved threshold)
- End: -64 eval return (collapsed in late training)
- Still high variance - policy "forgot" good behavior

**What we learned:**
- Advantage formula: `A(s,a) = G_t - V(s)` ("how much better than expected?")
- Value loss: MSE between critic's prediction and actual returns
- One backward pass, two optimizer steps (actor + critic)
- Baseline helps early learning but doesn't fix late-training instability

**Why it still struggles:**
Monte Carlo returns (waiting for episode end) have inherent variance. Even with a baseline, the learning signal is noisy. The fix: TD bootstrapping (A2C) - use critic's estimate instead of waiting for full returns.

## Expected Performance

| Algorithm | Expected Return | Notes |
|-----------|-----------------|-------|
| REINFORCE | -50 to +50 | High variance, doesn't solve |
| REINFORCE+baseline | 100-180 | Reduced variance |
| A2C | 150-200 | Faster learning |
| PPO | 200+ | Stable, sample efficient |

## Historical Context

### REINFORCE (1992)

Ronald J. Williams (1945-2024) published "Simple statistical gradient-following algorithms for connectionist reinforcement learning" in Machine Learning journal. The key insight: you can compute gradients of expected reward using only samples - no model of the environment needed.

The paper presents algorithms that "make weight adjustments in a direction that lies along the gradient of expected reinforcement... without explicitly computing gradient estimates." This is the `∇J = E[log π(a|s) * G_t]` formula - sampling from your policy and weighting by returns naturally produces an unbiased gradient estimate.

Williams was also part of the landmark 1986 backpropagation paper with Rumelhart and Hinton.

### Actor-Critic Origins (1983)

Barto, Sutton, and Anderson's "Neuronlike adaptive elements that can solve difficult learning control problems" introduced two cooperating components:

- **ASE (Associative Search Element)** - the actor, learns the policy
- **ACE (Adaptive Critic Element)** - the critic, learns to evaluate states

The ACE's purpose: "It adaptively develops an evaluation function that is more informative than the one directly available from the environment. This reduces the uncertainty under which the ASE will learn."

This is exactly what our baseline does - transform sparse episode returns into dense, informative learning signals. The term "critic" predates this paper - Klopf used "learning with a critic" in 1973 to distinguish RL from supervised "learning with a teacher."

### TD Bootstrapping (1988)

Richard Sutton's "Learning to Predict by the Methods of Temporal Differences" formalized a key idea: instead of waiting for episode end (Monte Carlo), update value estimates based on *other value estimates*:

- **Monte Carlo**: `V(s) ← actual return G_t` (wait for episode end)
- **TD(0)**: `V(s) ← r + γV(s')` (bootstrap from next state's estimate)

Sutton proved this converges and is more sample-efficient. The idea traces back to Samuel's 1959 checker player, but Sutton's 1988 paper made it rigorous. TD is the foundation for Q-learning, DQN, and all actor-critic methods.

### A3C and A2C (2016-2017)

**Volodymyr Mnih** (PhD under Hinton, lead author of DQN) and the DeepMind team published "Asynchronous Methods for Deep Reinforcement Learning" (2016), introducing **A3C** - multiple parallel workers running environments asynchronously, sending gradients to a shared parameter server. The parallelism decorrelates training data, eliminating the need for replay buffers.

**OpenAI** later released **A2C** in their Baselines library - a synchronous variant that waits for all workers before updating. Their finding: "We have not seen any evidence that the noise introduced by asynchrony provides any performance benefit." A2C is simpler and uses GPUs more effectively.

The key innovation over REINFORCE + baseline: **n-step TD returns** instead of Monte Carlo. Instead of waiting for episode end:

```
advantage = r_0 + γr_1 + ... + γ^(n-1)r_{n-1} + γ^n V(s_n) - V(s_0)
```

Look ahead n steps, then bootstrap from the critic. This reduces variance at the cost of some bias.

## References

- [Williams 1992 - REINFORCE](https://link.springer.com/article/10.1007/BF00992696)
- [Barto, Sutton, Anderson 1983 - Actor-Critic](https://ieeexplore.ieee.org/document/6313077)
- [Sutton 1988 - TD Learning](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)
- [Mnih et al. 2016 - A3C](https://arxiv.org/abs/1602.01783)
- [OpenAI Baselines: A2C](https://openai.com/index/openai-baselines-acktr-a2c/)
- [Policy Gradient Methods (Sutton & Barto Ch. 13)](http://incompleteideas.net/book/the-book.html)
- [PPO Paper (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347)
- [GAE Paper (Schulman et al. 2015)](https://arxiv.org/abs/1506.02438)
