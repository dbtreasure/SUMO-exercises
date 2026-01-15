# LunarLander-v2 DQN Experiments

Custom DQN implementation for the LunarLander-v2 environment with YAML-based configuration system for running controlled experiments.

## Project Structure

```
lunar-lander/
├── main.py                    # Training loop with config support
├── lunar_lander_dqn_agent.py  # DQN agent, replay buffer, network
├── plot_comparison.py         # Generate training comparison plots
├── configs/
│   ├── default.yaml           # Default config (warmup + train_freq=4)
│   ├── baseline.yaml          # Original DQN (no warmup, train_freq=1)
│   ├── balanced.yaml          # Middle ground (train_freq=2)
│   ├── warmup_only.yaml       # Isolate warmup effect (train_freq=1)
│   ├── grad_steps_test.yaml   # Decorrelated sampling (4x4)
│   ├── abl_huber_*.yaml       # Stability ablation configs
│   └── abl_mse_*.yaml         # LR ablation configs
├── checkpoints/               # Saved model checkpoints
├── reports/                   # Experiment reports
│   └── stability_ablation_report.md
└── scripts/
    └── run_ablation.sh        # Helper script for running ablations
```

## Running Experiments

```bash
# Default config (warmup + reduced update frequency)
uv run python main.py

# Specific config
uv run python main.py --config configs/baseline.yaml
```

## Experiment: Training Frequency and Warmup Period

### Background

Standard DQN updates the network after every environment step. This experiment explores two modifications:

1. **`learning_starts`** (warmup period): Collect N experiences before any gradient updates. This allows the replay buffer to fill with diverse experiences before learning begins.

2. **`train_freq`**: Update the network every N environment steps instead of every step. This trades sample efficiency for training speed.

3. **`gradient_steps`**: Number of consecutive minibatch updates per update event. This decouples update volume from update frequency, allowing decorrelated sampling.

### Configurations Tested

| Config | learning_starts | train_freq | gradient_steps | Updates/1M steps |
|--------|-----------------|------------|----------------|------------------|
| baseline | 0 | 1 | 1 | ~1,000,000 |
| warmup_only | 20,000 | 1 | 1 | ~980,000 |
| balanced | 20,000 | 2 | 1 | ~490,000 |
| warmup (default) | 20,000 | 4 | 1 | ~245,000 |
| grad_steps_test | 20,000 | 4 | 4 | ~980,000 |
| decorrelated_4x4_nowarmup | 0 | 4 | 4 | ~1,000,000 |
| decorrelated_4x8_nowarmup | 0 | 4 | 8 | ~2,000,000 |

**Clean test configs** (`decorrelated_*_nowarmup`): These configs remove the warmup period to isolate the effect of the decorrelated update schedule. The 4x4 config is compute-matched to baseline (~1M updates), while 4x8 doubles the update budget.

### Results

| Config | Wall Time | Best EVAL | Best Score | Final (20ep) | Final Score |
|--------|-----------|-----------|------------|--------------|-------------|
| baseline | 22.1m | 241.8 +/- 20.9 | **220.9** | 200.8 +/- 50.7 | 150.1 |
| warmup_only | 22.5m | 239.4 +/- 40.2 | 199.2 | 209.1 +/- 61.9 | 147.3 |
| grad_steps_test | 20.3m | 240.8 +/- 35.9 | 204.9 | 188.3 +/- 90.6 | 97.7 |
| balanced | 11.1m | 211.5 +/- 46.3 | 165.2 | 195.7 +/- 65.5 | 130.2 |
| warmup | 6.1m | 212.1 +/- 70.5 | 141.6 | 158.4 +/- 87.4 | 71.0 |

*Score = mean - std (used for ranking)*

### Key Findings

#### 1. Warmup period alone is neutral
Comparing baseline (no warmup) vs warmup_only (20k warmup, same train_freq=1):
- Nearly identical performance curves
- Both reach ~240 peak eval, ~200 final mean
- Warmup doesn't hurt, but doesn't significantly help either
- The replay buffer fills quickly enough during early random exploration

#### 2. Reduced train_freq hurts performance
Comparing configs with different train_freq values:
- baseline (train_freq=1): Best final score
- balanced (train_freq=2): 2x faster wall-clock, ~10% lower final score
- warmup (train_freq=4): 4x faster wall-clock, significantly worse final score

The update budget matters more than sample decorrelation for this environment.

#### 3. Gradient steps can compensate for reduced train_freq
The grad_steps_test config (train_freq=4, gradient_steps=4) achieves:
- Similar peak performance to baseline (~240)
- Same total update count (~980k updates)
- BUT higher final variance (std=90.6 vs std=50.7)

Decorrelated sampling helps reach peak performance but doesn't reduce final variance.

#### 4. Wall-clock vs sample efficiency tradeoff
| Config | Time | Quality |
|--------|------|---------|
| warmup | 6.1m | Poor |
| balanced | 11.1m | Acceptable |
| baseline | 22.1m | Best |

For quick iteration, balanced provides reasonable quality at half the time.

### Training Dynamics Observed

#### Q-value stability
All configs showed similar Q-value dynamics:
- Q-values stabilize around mean=25, max=60-80
- No Q-value explosion observed (max_grad_norm=10.0 helps)
- Target network hard updates every 10k steps work well

#### Epsilon decay
Using episode-based exponential decay (0.9995 per episode):
- Reaches minimum epsilon (~0.02) around episode 1500
- All configs have similar episode counts (~800-1100) at 1M steps

#### Gradient clipping
With max_grad_norm=10.0:
- Clip fraction typically 0-5%
- Gradient norms stabilize around 1-3 during training
- Higher clip fractions early in training when TD errors are large

### Environment Observations

#### Post-landing hovering behavior
The trained agent sometimes continues firing thrusters after landing. This happens because:
1. Episode only terminates when both legs contact AND body is "asleep" (no movement)
2. Continued thruster firing keeps the body "awake"
3. Agent hasn't learned that "do nothing" (action 0) is optimal after contact

This is a training artifact, not a bug. More training or reward shaping could address it.

### Future Experiments

1. **Prioritized Experience Replay**: Sample important transitions more often
2. **Dueling DQN**: Separate value and advantage streams
3. **Noisy Networks**: Replace epsilon-greedy with parametric noise
4. **Multi-seed evaluation**: Run each config 3-5 times for statistical significance
5. **Learning rate scheduling**: Anneal LR during training
6. **Larger networks**: Current [128, 128] may be underfitting
7. **Soft target updates (tau)**: Try tau=0.005 instead of hard updates

### Open Questions

1. Why does decorrelated sampling (grad_steps) increase final variance?
2. Would warmup help more with prioritized replay or larger batch sizes?
3. Is there an optimal train_freq for this environment given wall-clock constraints?
4. Does the post-landing hovering behavior affect evaluation scores significantly?

---

## Experiment: Gradient Stability & Loss Function Ablations

### Background

Our baseline DQN exhibited **99-100% gradient clipping** with pre-clip norms of 150-200 (vs threshold of 10). When we tried a 4x8 update schedule (8 gradient steps every 4 env steps), training collapsed completely.

We hypothesized that reducing gradient magnitude via:
1. **Huber loss** (smooth_l1_loss) - clips large TD errors
2. **Lower learning rate** - reduces gradient magnitude
3. **Higher max_grad_norm threshold** - allows larger updates

### Code Changes

Added configurable loss function to the agent:
```yaml
loss_type: "mse" | "huber"  # Default: mse
huber_delta: 1.0            # Huber loss threshold
```

### Configurations Tested

| Config | Loss | LR | max_grad_norm | Schedule |
|--------|------|----|---------------|----------|
| abl_huber_lr1e4_gn10 | Huber | 1e-4 | 10 | 1x1 |
| abl_mse_lr5e5_gn10 | MSE | 5e-5 | 10 | 1x1 |
| abl_huber_lr5e5_gn20 | Huber | 5e-5 | 20 | 1x1 |
| abl_huber_lr5e5_gn20_4x8 | Huber | 5e-5 | 20 | 4x8 |

### Results

| Config | Final Mean ± Std | Score | Clip % | Status |
|--------|------------------|-------|--------|--------|
| **MSE + LR 5e-5 + GN 10** | 266.94 ± 29.29 | **237.65** | 100% | PASS |
| Huber + LR 5e-5 + GN 20 | 244.63 ± 39.16 | **205.47** | 7% | PASS |
| Huber + LR 1e-4 + GN 10 | 233.02 ± 44.50 | 188.52 | 96% | FAIL |
| Huber + LR 5e-5 + GN 20 (4x8) | 222.12 ± 72.23 | 149.89 | 22% | FAIL |

### Key Findings

#### 1. Lower learning rate is the primary factor
Reducing LR from 1e-4 → 5e-5 improved final score significantly. The MSE + LR 5e-5 config achieved best score (237.65) despite 100% gradient clipping.

#### 2. Huber loss reduces gradient magnitude but doesn't improve score
Huber + higher grad norm (20) reduced clipping from 99% → 7%, but didn't translate to better performance. The gradient clipping at GN=10 with MSE loss appears to act as an effective learning rate limiter.

#### 3. Stabilization enabled 4x8 without collapse
The previously-collapsing 4x8 schedule (2M gradient updates) now runs to completion with Huber + LR 5e-5 + GN 20. However, it didn't outperform the simpler 1x1 baseline.

#### 4. More updates ≠ better performance
The 4x8 schedule performed 2M gradient updates (vs 1M for 1x1) but achieved worse final score (149.89 vs 237.65) with higher variance.

### Recommendations

- **Use MSE + LR 5e-5 + GN 10 as baseline** - best score, simplest config
- **Abandon aggressive update schedules** - no benefit despite 2x compute cost
- **Don't fear high clip fractions** - 100% clipping with lower LR works well

See `reports/stability_ablation_report.md` for the full report.

## Implementation Details

### Double DQN
Uses policy network to select actions, target network to evaluate Q-values:
```python
next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
target_q = rewards + self.gamma * next_q * (1 - dones)
```

### Target Network Updates
Hard updates every 10,000 steps (original DQN paper approach):
```python
if total_steps % config["target_update_every"] == 0:
    agent.sync_target_network()
```

### Gradient Clipping
Global norm clipping to prevent exploding gradients:
```python
grad_norm, did_clip = clip_and_get_norm(
    self.policy_net.parameters(), self.max_grad_norm
)
```

### Logging
Eval every 25k steps with diagnostics:
```
>>> EVAL @ 500,000 steps: 215.3 +/- 45.2 | Updates: 492,500 (9,850/10k) | Q(mean/max): 24.8/67.3
```

## Visualization

Generate comparison plot:
```bash
uv run python plot_comparison.py
```

This creates `training_comparison.png` showing eval reward curves with error bands for all configs.
