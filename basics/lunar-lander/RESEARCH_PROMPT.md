# Reinforcement Learning Hyperparameter & Architecture Research

## Context

I'm working on the Hugging Face Deep RL Course LunarLander-v2 challenge. The goal is to train an agent that achieves a score of at least 200 (mean_reward - std_reward), but I want to compete for the top of the leaderboard where scores reach 370+.


I'm specifically using lunar-lander v2 from this version of Gymnasium, https://gymnasium.farama.org/v0.29.0/environments/box2d/lunar_lander/

### Environment: LunarLander-v2

- **Observation space**: 8-dimensional continuous (x, y, velocities, angle, angular velocity, leg contacts)
- **Action space**: Discrete(4) - do nothing, fire left, fire main, fire right
- **Reward structure**: +100 landing, -100 crash, proximity/velocity/tilt bonuses, engine costs (-0.03 side, -0.3 main)
- **Episode termination**: crash, exit viewport, or successful landing
- **Target**: Maximize (mean_reward - std_reward) over evaluation episodes

### Current Leaderboard Top Performers

| Rank | Score (mean ± std) | Algorithm | Notable Config |
|------|-------------------|-----------|----------------|
| #1 | 370.24 ± 8.90 | PPO | Default HF course params, 1M steps |
| #2 | 369.40 ± 20.43 | PPO | Unknown details |
| #3 | 330.99 ± 20.23 | PPO | MLP policy |

Key observation: Top models use nearly identical configurations. Variance (std) significantly impacts ranking.

### Algorithms I'm Considering

1. **DQN** (Deep Q-Network) - for learning purposes
2. **PPO** (Proximal Policy Optimization) - what the top models use
3. Potentially: A2C, SAC, TD3

### My Learning Progression

1. Implement DQN from scratch (learning)
2. Implement PPO from scratch (learning)
3. Use Stable-Baselines3 with tuned hyperparameters
4. Full optimization push for leaderboard

---

## Research Questions

### 1. Hyperparameter Optimization Tools & Frameworks

What are the best tools for sweeping RL hyperparameters? I'm specifically interested in:

- **Optuna** - How well does it work with RL? What samplers work best (TPE, CMA-ES)?
- **Weights & Biases Sweeps** - Integration with SB3?
- **Ray Tune** - Is the complexity worth it for a single-environment problem?
- **RL Baselines3 Zoo** - Does it have built-in hyperparameter optimization?
- **Population Based Training (PBT)** - Is this overkill for LunarLander?

For each tool, I want to understand:
- Setup complexity
- Sample efficiency (how many trials needed?)
- Parallelization options
- Integration with Stable-Baselines3

### 2. PPO Hyperparameters Deep Dive

For PPO specifically, what does research and community experience say about:

#### Learning Rate
- What range should I search? (1e-5 to 1e-2?)
- Linear decay vs constant vs other schedules?
- Learning rate for actor vs critic (if separate)?

#### GAE Parameters
- `gamma` (discount factor): 0.99 vs 0.999 vs 0.9999 for episodic tasks?
- `gae_lambda`: Typical ranges? Interaction with gamma?

#### PPO-Specific
- `clip_range`: 0.1 vs 0.2 vs 0.3? Does it matter much?
- `ent_coef` (entropy coefficient): How to balance exploration? Start high and decay?
- `vf_coef` (value function coefficient): Typical values?
- `n_steps`: Relationship to episode length? LunarLander episodes are ~200-1000 steps
- `batch_size` and `n_epochs`: Interactions between these?
- `max_grad_norm`: Gradient clipping values?

#### Training Scale
- How many timesteps are actually needed? 1M? 5M? 10M?
- Diminishing returns curve?
- How many parallel environments (n_envs)? 4? 16? 64?

### 3. DQN Hyperparameters (for my learning phase)

- `learning_rate`: Typical ranges?
- `buffer_size`: How big? Memory vs performance tradeoff?
- `learning_starts`: How many steps before training begins?
- `batch_size`: 32? 64? 128?
- `tau` (soft update coefficient): Hard updates vs soft updates?
- `target_update_interval`: If using hard updates, how often?
- `train_freq`: Train every step or every N steps?
- `gradient_steps`: Multiple gradient steps per environment step?
- `exploration_fraction` and `exploration_final_eps`: Epsilon decay schedules?

#### DQN Variants Worth Trying
- Double DQN (reduces overestimation)
- Dueling DQN (separate value/advantage streams)
- Prioritized Experience Replay
- N-step returns
- Noisy Networks (for exploration instead of epsilon-greedy)

### 4. Network Architecture

#### MLP Architecture for 8-dim input
- Depth: 2 layers? 3 layers? Deeper?
- Width: 64? 128? 256? 512?
- Activation functions: ReLU vs Tanh vs LeakyReLU vs GELU?
- Layer normalization? Batch normalization?
- Separate networks for actor/critic vs shared backbone?
- Orthogonal initialization? Xavier? He?

#### Architecture Search
- Are there papers on optimal architectures for low-dimensional continuous control?
- Is there a relationship between observation dimensionality and optimal network size?

### 5. Reducing Variance (Critical for Leaderboard)

Since ranking is by (mean - std), reducing variance is as important as increasing mean:

- More evaluation episodes? How many?
- Deterministic vs stochastic evaluation?
- Seed selection / multiple training runs?
- Ensemble methods?
- Observation normalization (running mean/std)?
- Reward normalization/scaling?

### 6. Environment-Specific Tricks

For LunarLander specifically:

- Frame stacking? (Probably not needed with velocity in observation)
- Action repeat?
- Reward shaping? (Beyond default rewards)
- Curriculum learning? (Start with easier gravity?)
- Is there domain knowledge that could help? (Physics-informed?)

### 7. Training Stability & Debugging

- How to detect if training is going wrong early?
- Key metrics to monitor (beyond reward)?
- Common failure modes in PPO/DQN?
- When to early stop vs continue training?

### 8. Reproducibility & Seeds

- How much does random seed affect final performance?
- Should I train N models and pick the best?
- What's considered "fair" for leaderboard submission?

### 9. Compute Efficiency

I have access to:
- Local Mac with Apple Silicon (MPS)
- Modal.com cloud GPU credits

Questions:
- Is GPU actually faster for small MLPs or is CPU fine?
- How to efficiently parallelize hyperparameter search?
- Cost-effective strategies for cloud compute?

### 10. State of the Art & Papers

What papers should I read for:
- PPO best practices?
- Hyperparameter sensitivity analysis in RL?
- LunarLander or similar continuous control benchmarks?
- General RL optimization strategies?

Notable papers/resources I'm aware of:
- "Implementation Matters in Deep RL" (Engstrom et al.)
- "What Matters In On-Policy Reinforcement Learning?" (Andrychowicz et al.)
- RL Baselines3 Zoo benchmarks

What else should I look at?

---

## Specific Experiments I'm Planning

1. **Baseline**: Train with default HF course params, establish my baseline score
2. **Extended training**: Same params but 2M, 5M, 10M steps
3. **Network size sweep**: [64,64], [128,128], [256,256], [64,64,64]
4. **Learning rate sweep**: 1e-4, 3e-4, 1e-3, 3e-3
5. **Gamma sweep**: 0.99, 0.995, 0.999, 0.9999
6. **n_envs sweep**: 4, 8, 16, 32, 64
7. **Full Optuna optimization**: Once I understand the basics

---

## Deliverables I'm Looking For

1. **Practical hyperparameter search strategy** - What to tune first, what to leave default
2. **Tool recommendations** - Best tool for my scale (single environment, want top leaderboard)
3. **Architecture guidelines** - Rules of thumb for network sizing
4. **Variance reduction techniques** - Practical methods to tighten std
5. **Efficient experimentation workflow** - How to iterate quickly

---

## Technical Constraints

- Using gymnasium 0.29.0 (for LunarLander-v2 compatibility)
- Stable-Baselines3 for production runs
- Python 3.13
- PyTorch backend
- Will implement DQN/PPO from scratch first for learning

---

## Output Format Request

Please structure your research findings as:

1. **Executive Summary** - Top 3-5 actionable recommendations
2. **Tool Comparison Table** - Hyperparameter optimization tools with pros/cons
3. **Hyperparameter Priority List** - Ordered by impact, with suggested search ranges
4. **Architecture Recommendations** - Specific to 8-dim input, discrete action RL
5. **Experiment Roadmap** - Sequenced experiments with expected insights
6. **Resource Links** - Papers, blog posts, code examples worth reading
