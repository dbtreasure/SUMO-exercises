# DQN Stability & Update Frequency Ablation Report

**Project:** LunarLander-v2 DQN Agent Optimization
**Date:** 2026-01-14
**Conducted by:** Claude (Opus 4.5) via Claude Code CLI

---

## Executive Summary

We investigated gradient instability in our DQN agent (99-100% gradient clipping) and tested whether increased update frequency (4x8 schedule) could improve sample efficiency. The stabilization effort succeeded—we reduced clipping from 99% to 7-22%—but the 4x8 schedule did not outperform the simpler 1x1 baseline despite 2x the gradient updates.

**Best configuration:** MSE loss + LR 5e-5 + 1x1 schedule → Score 237.65

---

## Problem Statement

Our baseline DQN agent exhibited:
- **99-100% gradient clipping** (pre-clip norms of 150-200 vs threshold of 10)
- **Training collapse** when using a 4x8 update schedule (train every 4 steps, 8 gradient steps per update)

We hypothesized that reducing gradient magnitude via Huber loss and/or lower learning rate would stabilize training and enable more aggressive update schedules.

---

## Methodology

### Code Changes Implemented

1. **Selectable loss function** (`loss_type` config parameter)
   - `"mse"`: Standard MSE loss (default, backward compatible)
   - `"huber"`: Smooth L1 loss with configurable `huber_delta` (default 1.0)

2. **Configurable `max_grad_norm`** threshold (was hardcoded at 10)

3. **Enhanced logging** showing loss_type, huber_delta, learning_rate, max_grad_norm at startup

### Experiments Run

All experiments: 1M steps, eval every 25k, 20-episode final evaluation.

| Config | Loss | LR | max_grad_norm | Schedule | Updates/10k |
|--------|------|----|---------------|----------|-------------|
| Baseline reference | MSE | 1e-4 | 10 | 1x1 | 10,000 |
| abl_huber_lr1e4_gn10 | Huber | 1e-4 | 10 | 1x1 | 10,000 |
| abl_mse_lr5e5_gn10 | MSE | 5e-5 | 10 | 1x1 | 10,000 |
| abl_huber_lr5e5_gn20 | Huber | 5e-5 | 20 | 1x1 | 10,000 |
| abl_huber_lr5e5_gn20_4x8 | Huber | 5e-5 | 20 | 4x8 | 20,000 |

---

## Results

### Ablation Results (1x1 Schedule)

| Config | Final Mean ± Std | Score (mean-std) | Clip % | Status |
|--------|------------------|------------------|--------|--------|
| **MSE + LR 5e-5 + GN 10** | 266.94 ± 29.29 | **237.65** | 100% | PASS |
| Huber + LR 5e-5 + GN 20 | 244.63 ± 39.16 | **205.47** | 7% | PASS |
| Huber + LR 1e-4 + GN 10 | 233.02 ± 44.50 | 188.52 | 96% | FAIL |

### Stabilized 4x8 Results

| Config | Final Mean ± Std | Score (mean-std) | Clip % | Status |
|--------|------------------|------------------|--------|--------|
| Huber + LR 5e-5 + GN 20 (4x8) | 222.12 ± 72.23 | 149.89 | 22% | FAIL |

**Training stability:** The 4x8 run completed without collapse. Eval scores during training were competitive (peak 254.3 at 800k steps), but final evaluation showed high variance.

---

## Key Findings

### 1. Lower Learning Rate is the Primary Factor
Reducing LR from 1e-4 → 5e-5 improved final score significantly, regardless of loss function. The MSE + LR 5e-5 config achieved the best score (237.65) despite 100% gradient clipping.

### 2. Huber Loss Reduces Gradient Magnitude
Huber loss with higher grad norm threshold (20) reduced clipping from 99% → 7%, confirming it successfully dampens TD-error outliers. However, this didn't translate to better final performance.

### 3. Gradient Clipping Acts as Implicit LR Scaling
The MSE + LR 5e-5 config clips 100% of the time but performs best. This suggests `max_grad_norm=10` is effectively acting as a learning rate limiter, and the explicit LR reduction compounds beneficially with it.

### 4. More Updates ≠ Better Performance
The 4x8 schedule performed 2M gradient updates (vs 1M for 1x1) but achieved worse final score (149.89 vs 237.65). Possible explanations:
- Decorrelated sampling every 4 steps may reduce data diversity per update batch
- 8 consecutive gradient steps on similar data may cause overfitting to recent experience
- The 2x wall-clock time cost (45min vs 23min) provides no benefit

### 5. Variance Matters for Scoring
The 4x8 run had competitive mean reward (222) but high std (72), yielding a poor score. The 1x1 runs had lower variance (29-44 std), which is critical since score = mean - std.

---

## Recommendations

### Immediate
1. **Use MSE + LR 5e-5 + GN 10 as the new baseline** — best score, simplest config
2. **Abandon the 4x8 schedule** — no benefit despite 2x compute cost

### Future Investigation
1. **Target network update frequency** — currently every 10k steps; may interact with update frequency
2. **Prioritized Experience Replay (PER)** — could help with sample efficiency more than raw update count
3. **Epsilon schedule tuning** — current decay may not be optimal for the lower LR regime
4. **Multi-seed evaluation** — run best config with 5-10 seeds to get reliable score estimate

---

## Artifacts

### Code Changes
- `lunar_lander_dqn_agent.py`: Added `loss_type`, `huber_delta` parameters
- `main.py`: Config passthrough and enhanced startup logging

### Config Files Created
```
configs/abl_huber_lr1e4_gn10.yaml
configs/abl_mse_lr5e5_gn10.yaml
configs/abl_mse_lr2p5e5_gn10.yaml
configs/abl_huber_lr5e5_gn10.yaml
configs/abl_huber_lr5e5_gn20.yaml
configs/abl_huber_lr5e5_gn20_decorrelated_4x8.yaml
```

### Log Files
```
abl_huber_lr1e4_gn10.log
abl_mse_lr5e5_gn10.log
abl_huber_lr5e5_gn20.log
abl_huber_lr5e5_gn20_decorrelated_4x8.log
```

---

## Conclusion

The stabilization effort was technically successful—we can now run aggressive update schedules without collapse. However, the hypothesis that "more gradient updates = better sample efficiency" was not supported. The simplest configuration (MSE loss, lower LR, standard 1x1 schedule) remains optimal.

The project should proceed with MSE + LR 5e-5 as the foundation for future improvements.

---

*Report generated by Claude (Opus 4.5) via Claude Code CLI*
