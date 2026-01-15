# LunarLander-v2 RL Exercise

## Goal
Train an agent to land a lunar lander on the moon. Target score: **>= 200** (mean_reward - std_reward). Leaderboard top scores reach 370+.

## Learning Path
1. DQN from scratch (learning)
2. PPO from scratch (learning)
3. SB3 baseline (get 200+ quickly)
4. Optuna hyperparameter sweep
5. Variance reduction techniques
6. Final push (multiple seeds, submit best)

## Environment Details (Gymnasium v0.29.0)

Classic rocket trajectory optimization problem. The lander must land safely on a pad at coordinates (0,0). Infinite fuel available for learning across multiple attempts.

### Observation Space
`Box(8,)` with bounds varying per dimension:

| Index | Value | Range |
|-------|-------|-------|
| 0 | x coordinate | -1.5 to 1.5 |
| 1 | y coordinate | -1.5 to 1.5 |
| 2 | x velocity | -5 to 5 |
| 3 | y velocity | -5 to 5 |
| 4 | angle (radians) | -π to π |
| 5 | angular velocity | -5 to 5 |
| 6 | left leg contact | 0 or 1 |
| 7 | right leg contact | 0 or 1 |

### Action Space
`Discrete(4)`:
- 0: Do nothing
- 1: Fire left orientation engine
- 2: Fire main engine
- 3: Fire right orientation engine

### Rewards
Per-step rewards:
- Proximity bonus/penalty relative to landing pad
- Velocity bonus/penalty (slower is better)
- Tilt penalty (angle deviation from horizontal)
- +10 points per leg contacting ground
- -0.03 per frame for side engine firing
- -0.3 per frame for main engine firing
- +100 for safe landing
- -100 for crash

### Starting State
Lander starts at top center of viewport with random initial force applied to center of mass.

### Episode Termination
Episodes end when:
1. Lander crashes (body contacts moon)
2. Lander exits viewport (x > 1)
3. Lander is not awake (no movement/collisions)

### Environment Arguments
```python
gym.make(
    "LunarLander-v2",
    continuous=False,      # Use discrete actions (default)
    gravity=-10.0,         # Gravitational constant (-12 to 0)
    enable_wind=False,     # Apply wind effects
    wind_power=15.0,       # Max linear wind magnitude
    turbulence_power=1.5,  # Max rotational wind magnitude
)
```

## Hyperparameter Tuning

### Priority Order (tune these first)
1. **learning_rate** - most sensitive, search 1e-4 to 3e-3 (log-uniform)
2. **gamma** - 0.99 to 0.9999, higher values for longer-term rewards
3. **gae_lambda** - 0.9 to 0.99, balances bias-variance in advantage estimates

### PPO Hyperparameters

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| learning_rate | 1e-4 to 3e-3 | 3e-4 | Most impactful; consider annealing |
| gamma | 0.99 to 0.9999 | 0.999 | Higher = longer horizon |
| gae_lambda | 0.9 to 0.99 | 0.98 | Interacts with gamma |
| n_steps | 256 to 4096 | 1024 | Steps per rollout |
| batch_size | 64 to 2048 | 64 | Must divide n_steps * n_envs |
| n_epochs | 3 to 10 | 4 | Passes over collected data |
| ent_coef | 0 to 0.02 | 0.01 | Exploration; consider decay |
| clip_range | 0.1 to 0.3 | 0.2 | Policy update constraint |
| n_envs | 8 to 64 | 16 | Parallel environments |

### DQN Hyperparameters

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| learning_rate | **5e-5 to 1e-4** | **5e-5** | **Lower is better with grad clipping** |
| buffer_size | 100k to 1M | 100k | Memory vs diversity |
| batch_size | 32 to 128 | 64 | Larger = more stable |
| tau | 0.005 to 0.01 | 0.005 | Soft update coefficient |
| target_update_interval | 5k to 20k | 10k | If using hard updates |
| exploration_fraction | 0.1 to 0.5 | 0.1 | Epsilon decay duration |
| exploration_final_eps | 0.01 to 0.1 | 0.02 | Final epsilon |
| max_grad_norm | 10 to 20 | 10 | Gradient clipping threshold |
| loss_type | mse, huber | mse | MSE works best with lower LR |

**Empirical finding:** LR 5e-5 + max_grad_norm 10 achieves score 237.65. The constant gradient clipping (100% clip rate) acts as an effective learning rate limiter. Huber loss reduces clipping but doesn't improve final score.

## Architecture

### Network Sizing
- **Baseline**: [64, 64] - SB3 default, good starting point
- **Wider**: [128, 128] or [256, 256] - try if underfitting
- **Deeper**: [64, 64, 64] - sometimes helps
- **Asymmetric**: actor [64, 64], critic [256, 256] - research shows critic benefits more from capacity

### Key Techniques
- **Orthogonal initialization** with tanh activation - stabilizes training
- **Smaller final layer weights** (scale by 0.01) - prevents large initial outputs
- **Penultimate normalization** - normalize second-to-last layer features to unit norm, reduces variance by 3x

### CPU vs GPU
- GPU only faster when hidden units > ~1024
- For [64, 64] or [128, 128] networks, CPU is often faster
- Data transfer overhead outweighs computation gains for small networks

## Variance Reduction (Critical for Leaderboard)

Ranking is by (mean - std), so reducing variance matters as much as increasing mean:

1. **Lower learning rate** - LR 5e-5 reduces variance significantly vs 1e-4
2. **Gradient clipping** - max_grad_norm 10 (100% clipping is fine with low LR)
3. **Simple update schedule** - 1x1 (every step) beats aggressive schedules like 4x8
4. **Observation normalization** - use VecNormalize wrapper
5. **More eval episodes** - 20-50 episodes for reliable estimates
6. **Deterministic evaluation** - no exploration noise
7. **Multiple seeds** - train 5-10 agents, submit best (mean - std)

**Empirical finding:** The 4x8 schedule (2x gradient updates) had mean 222 but std 72, yielding poor score 149. The 1x1 schedule with LR 5e-5 had mean 267 and std 29, yielding score 237.

## Tools

### Hyperparameter Optimization
- **Optuna** - recommended first choice, easy parallelization, good samplers (TPE, CMA-ES)
- **RL-Baselines3 Zoo** - integrates Optuna via `--optimize` flag
- **W&B Sweeps** - good visualization, Bayesian search
- **Ray Tune** - powerful but complex, supports PBT

## SB3 Baseline

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("LunarLander-v2", n_envs=16)

model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
)

model.learn(total_timesteps=1_000_000)
```

## Evaluation

```python
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

eval_env = Monitor(gym.make("LunarLander-v2"))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
```

## Local Implementation Patterns

### Episode Monitor
Track episode statistics during training:
```python
class EpisodeMonitor:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []

    def log_episode(self, reward, length):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

    def get_stats(self, last_n=100):
        rewards = self.episode_rewards[-last_n:]
        return np.mean(rewards), np.std(rewards)
```

### Evaluation Function
Run evaluation with no exploration:
```python
def evaluate(agent, env, n_episodes=10):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.get_action(obs, deterministic=True)  # No exploration
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)
```

### Checkpointing
Save and load model state:
```python
def save_checkpoint(agent, path):
    torch.save({
        'policy_net': agent.policy_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
    }, path)

def load_checkpoint(agent, path):
    checkpoint = torch.load(path)
    agent.policy_net.load_state_dict(checkpoint['policy_net'])
    agent.target_net.load_state_dict(checkpoint['target_net'])
    agent.optimizer.load_state_dict(checkpoint['optimizer'])
    agent.epsilon = checkpoint['epsilon']
```

### Gradient Clipping
Clip gradients by global norm to stabilize training and prevent exploding losses:
```python
def clip_and_get_norm(parameters, max_grad_norm: float | None) -> tuple[float, bool]:
    """Clip gradients by global norm and return pre-clip norm."""
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0, False

    # Compute total norm
    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))
    preclip_norm = total_norm.item()

    # Apply clipping if enabled
    did_clip = False
    if max_grad_norm is not None and max_grad_norm > 0:
        if preclip_norm > max_grad_norm:
            did_clip = True
            clip_coef = max_grad_norm / (preclip_norm + 1e-6)
            for p in params:
                p.grad.detach().mul_(clip_coef)

    return preclip_norm, did_clip
```

Usage in DQNAgent:
- `max_grad_norm` parameter (default: 10.0)
- Call between `loss.backward()` and `optimizer.step()`
- Returns `UpdateResult` with loss, grad_norm_preclip, and grad_clipped

### Loss Function Selection
The agent supports configurable loss functions via YAML config:
```yaml
loss_type: "mse"    # or "huber"
huber_delta: 1.0    # threshold for Huber loss (default 1.0)
```

**MSE loss** (default): Standard squared error. With LR 5e-5 and max_grad_norm 10, achieves best results despite 100% gradient clipping.

**Huber loss** (smooth_l1_loss): Clips large TD errors. Reduces gradient clipping from 99% to 7% but doesn't improve final score. Use if you want cleaner gradient diagnostics.

### Video Recording
For recording evaluation runs:
```python
# Live viewing (Mac)
env = gym.make("LunarLander-v2", render_mode="human")

# For recording frames
env = gym.make("LunarLander-v2", render_mode="rgb_array")
frames = []
obs, _ = env.reset()
while not done:
    frames.append(env.render())
    # ... step logic
```

### SB3 Monitor Wrapper
Wraps env to log episode stats automatically:
```python
from stable_baselines3.common.monitor import Monitor
eval_env = Monitor(gym.make("LunarLander-v2"))
```

### Vectorized Environments
Multiple parallel envs (for SB3/later):
```python
from stable_baselines3.common.env_util import make_vec_env
env = make_vec_env("LunarLander-v2", n_envs=16)
```

## Notes

- Using gymnasium 0.29.0 for LunarLander-v2 compatibility
- stable-baselines3 required for PPO/DQN production runs
- 16 parallel envs recommended for diverse training experience
- ~1M timesteps sufficient for >= 200, 2-5M for optimization
- Leaderboard top scores use nearly identical configs - gains are marginal and variance-dependent

## Best Known DQN Configuration

Based on stability ablation experiments (see `reports/stability_ablation_report.md`):

```yaml
# Best config: MSE + LR 5e-5 → Score 237.65
learning_rate: 0.00005      # Lower than typical 1e-4
loss_type: "mse"            # Standard MSE, not Huber
max_grad_norm: 10.0         # 100% clipping is fine
train_freq: 1               # Update every step
gradient_steps: 1           # One gradient step per update
gamma: 0.999
buffer_size: 100_000
batch_size: 64
target_update_every: 10_000
epsilon_start: 1.0
epsilon_end: 0.02
epsilon_decay: 0.9995
```

**Key insight:** High gradient clipping (100%) with low LR works better than trying to reduce clipping via Huber loss or higher grad norm threshold. The clipping acts as an effective learning rate limiter.

**What doesn't help:**
- Huber loss (reduces clipping but not score)
- Aggressive update schedules like 4x8 (more variance, worse score)
- Higher max_grad_norm (reduces clipping but not score)
