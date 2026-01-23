# Lunar Lander PPO - Policy Gradient Learning Path

## Project Goal

Learn policy gradient methods by implementing an incremental path on Gymnasium LunarLander-v2:

1. **REINFORCE** - Monte Carlo policy gradient (Stage 0)
2. **REINFORCE + baseline** - Value function variance reduction (Stage 1)
3. **A2C** - Actor-Critic with TD bootstrapping (Stage 2)
4. **GAE** - Generalized Advantage Estimation (Stage 3)
5. **PPO** - Clipped surrogate objective, minibatches, multiple epochs (Stage 4)

## Architecture

```
main.py                    # Single entrypoint
configs/*.yaml             # Algorithm configs
rl/
  common/                  # Shared utilities
    config.py              # YAML loading, dataclasses
    seeding.py             # Deterministic seeding
    envs.py                # Environment creation
    logger.py              # CSV metrics logging
    eval.py                # Evaluation loop
    plot.py                # Training curves
    buffers.py             # RolloutBuffer (on-policy)
    nets.py                # MLP, policy/value networks
    utils.py               # Normalization, etc.
  agents/
    base.py                # Agent ABC
    reinforce.py           # Stage 0
    reinforce_baseline.py  # Stage 1
    a2c.py                 # Stage 2
    ppo.py                 # Stages 3-4
runs/                      # Training artifacts
```

## Agent API (stable across all algorithms)

```python
class Agent(ABC):
    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[int, dict]
    def observe(self, obs, action, reward, next_obs, terminated, truncated) -> None
    def update(self) -> dict[str, float]
    def on_episode_end(self) -> dict[str, float]
    def save(self, path: str) -> None
    def load(self, path: str) -> None
```

## Running

```bash
uv run python main.py --config configs/reinforce.yaml
```

Outputs to `runs/<run_id>/`:
- `metrics.csv` - Training/eval metrics
- `plot.png` - Eval return vs step
- `checkpoints/` - Model weights

## Design Principles

- **Grug:** Minimal abstractions, one main.py, small diffs between stages
- **Incremental:** Each algorithm builds on the previous
- **Configurable:** YAML configs for all hyperparameters
- **Deterministic:** Seed everything for reproducibility

## LunarLander-v2 Notes

- **Observation:** 8D continuous (position, velocity, angle, leg contact)
- **Action:** Discrete(4) - nothing, left engine, main engine, right engine
- **Target:** Mean eval return >= 200 (PPO should achieve this)
- **REINFORCE baseline:** Expect noisy learning, ~50-150 return

## Pair Programming Style

This is a learning project. Claude acts as a senior RL engineer mentoring a fullstack dev new to RL:

- **Be pedagogical:** Explain architectural decisions, code style choices, and RL concepts
- **Give context:** How does this file fit into the bigger picture? What are the alternatives?
- **Be Socratic:** Ask what the user thinks we should do sometimes, don't just dictate
- **Teach tradeoffs:** "We could do X or Y, here's why I'd pick X for this project"
- **No rushing:** Take time to explain, don't just dump code

## Progress

### Stage 0: REINFORCE (Complete)

Implemented vanilla REINFORCE (Monte Carlo policy gradient):

- **Files created:**
  - `rl/common/` - seeding, config (pydantic), envs, logger (TensorBoard + CSV), eval, plot, buffers, nets, utils
  - `rl/agents/base.py` - Agent ABC with act/observe/update interface
  - `rl/agents/reinforce.py` - REINFORCE implementation
  - `configs/reinforce.yaml` - hyperparameters
  - `main.py` - training loop

- **Training results (500k steps):**
  - Started: ~-350 eval return (crashing)
  - Ended: ~-50 eval return (hovering, occasional landing)
  - High variance throughout - classic REINFORCE behavior

- **Key learnings:**
  - Policy gradient: `∇J = E[log π(a|s) * G_t]`
  - Monte Carlo returns computed backwards for O(n) efficiency
  - REINFORCE has high variance because no baseline to compare against
  - Entropy stayed healthy (~1.0), policy loss oscillates around zero (normal)

### Stage 1: REINFORCE + Baseline (Complete)

Added a Critic network to reduce variance via advantage estimation:

- **Files added/modified:**
  - `rl/common/nets.py` - Added `Critic` class (value network)
  - `rl/common/config.py` - Added `lr_critic` to AlgoConfig
  - `rl/agents/reinforce_baseline.py` - REINFORCE + baseline implementation
  - `configs/reinforce_baseline.yaml` - 500k steps config
  - `configs/reinforce_baseline_long.yaml` - 2M steps config
  - `main.py` - Added `loss/value` to CSV columns

- **Training results (2M steps):**
  - Peak: +258 eval return at 750k steps (crossed solved threshold)
  - End: -64 eval return (collapsed in late training)
  - Still high variance - policy "forgot" good behavior

- **Key learnings:**
  - Advantage: `A(s,a) = G_t - V(s)` ("how much better than expected?")
  - Value loss: MSE(V(s), G_t) trains critic to predict returns
  - One backward pass, two optimizer steps (actor + critic)
  - Baseline helps early learning but doesn't fix MC variance problem

### Stage 2: A2C (Complete)

Replaced Monte Carlo returns with n-step TD bootstrapping:

- **Files added/modified:**
  - `rl/common/buffers.py` - Added `compute_returns_td()` for n-step returns
  - `rl/common/config.py` - Added `n_steps` to AlgoConfig
  - `rl/agents/a2c.py` - A2C implementation with step-based updates
  - `configs/a2c.yaml`, `configs/a2c_long.yaml` - A2C configs
  - `main.py` - Added step-based update mode

- **Training results (2M steps):**
  - Peak: 255.9 +/- 9.5 (entropy=0.01), 241.5 +/- 47.4 (entropy=0.05)
  - End: 29.8 (entropy=0.01), 132.0 (entropy=0.05)
  - Entropy coefficient critical for stability

- **Key learnings:**
  - N-step TD: `G_t = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})`
  - Bootstrapping trades variance for bias
  - Step-based updates (every n_steps) vs episode-based (REINFORCE)

### Stage 3: GAE (Complete)

Replaced fixed n-step returns with Generalized Advantage Estimation:

- **Files added/modified:**
  - `rl/common/buffers.py` - Added `compute_returns_gae()` for λ-weighted advantages
  - `rl/common/config.py` - Added `gae_lambda` to AlgoConfig
  - `rl/agents/gae.py` - GAE agent implementation
  - `configs/gae.yaml`, `configs/gae_long.yaml` - GAE configs
  - `main.py` - Added "gae" to step-based update list

- **Training results (2M steps):**
  - Peak: 241.0 +/- 31.1 at 1.85M steps
  - Times > 200: 8
  - Final: 84.7 +/- 123.7 (still unstable)

- **Key learnings:**
  - GAE formula: `A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...`
  - λ=0 is TD (high bias), λ=1 is MC (high variance), λ=0.95 is the sweet spot
  - Computes advantages directly - no separate `returns - values` step
  - Smoother advantages but doesn't fix policy instability

### Next: Stage 4 (PPO)

Add clipped surrogate objective to prevent large policy updates:
- Clip ratio `r = π_new/π_old` to [1-ε, 1+ε]
- Multiple epochs over same batch
- Minibatch updates for efficiency
