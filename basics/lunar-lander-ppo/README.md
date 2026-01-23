# Policy Gradient Learning Path - LunarLander-v2

An incremental implementation of policy gradient algorithms for learning purposes.

## Algorithms

| Stage | Algorithm | Key Concept | Status |
|-------|-----------|-------------|--------|
| 0 | REINFORCE | Monte Carlo returns, policy gradient theorem | Done |
| 1 | REINFORCE + baseline | Variance reduction with value function | Done |
| 2 | A2C | TD bootstrapping, n-step returns | Done |
| 3 | GAE | Lambda-weighted advantage estimation | Done |
| 4 | PPO | Clipped objective, minibatches, multiple epochs | Next |

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
  a2c.yaml           # 500k steps, n-step TD
  a2c_long.yaml      # 2M steps, n-step TD
  a2c_entropy05.yaml # 2M steps, higher entropy for stability
  gae.yaml           # 500k steps, GAE
  gae_long.yaml      # 2M steps, GAE
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
    a2c.py           # Stage 2 implementation
    gae.py           # Stage 3 implementation
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

## Stage 2 Results: A2C

Replaced Monte Carlo returns with n-step TD bootstrapping. Instead of waiting for episode end, we look ahead n steps then bootstrap from the critic's estimate.

**Performance (2M steps):**

| Config | Peak Eval | Final Eval | Times > 200 |
|--------|-----------|------------|-------------|
| entropy=0.01 | 255.9 +/- 9.5 | 29.8 +/- 119.3 | 4 |
| entropy=0.05 | 241.5 +/- 47.4 | 132.0 +/- 102.0 | 9 |

**Key finding: Entropy coefficient matters.** With entropy=0.01, A2C learns fast but collapses hard - the policy becomes too deterministic (entropy drops to ~0.3) and "forgets" good behavior. With entropy=0.05, the policy maintains more exploration and is 4x more stable at end of training.

**What we learned:**
- N-step TD formula: `G_t = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})`
- Bootstrapping trades variance for bias - faster learning but relies on critic accuracy
- A2C updates every n_steps (5) vs. per-episode for REINFORCE
- More frequent updates → policy commits faster → needs entropy bonus to maintain exploration

**Why it still struggles:**
All algorithms so far can reach 200+ but can't stay there. The policy takes large gradient steps that overshoot, then struggles to recover. PPO's clipped objective limits how much the policy can change per update, preventing these collapses.

## Stage 3 Results: GAE

Replaced A2C's fixed n-step returns with Generalized Advantage Estimation - an exponentially-weighted blend of all possible n-step estimates.

**Performance (2M steps):**

| Config | Peak Eval | Final Eval | Times > 200 |
|--------|-----------|------------|-------------|
| λ=0.95, entropy=0.01 | 241.0 +/- 31.1 | 84.7 +/- 123.7 | 8 |

**What we learned:**
- GAE formula: `A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...` where `δ_t = r_t + γV(s_{t+1}) - V(s_t)`
- λ=0 gives pure TD (high bias, low variance), λ=1 gives Monte Carlo (low bias, high variance)
- λ=0.95 is the standard choice - smooth blend of all horizons
- GAE computes advantages directly (no separate `returns - values` step)

**Comparison with A2C:**
- Peak performance similar (241 vs 242-256)
- More times above 200 than A2C with entropy=0.01 (8 vs 4)
- Still suffers from late-training instability

**Why it still struggles:**
GAE gives smoother advantage estimates but doesn't limit *how much* the policy can change. Large gradient steps still cause the policy to overshoot and "forget" good behavior. That's what PPO's clipped objective addresses.

## Expected Performance

| Algorithm | Expected Return | Notes |
|-----------|-----------------|-------|
| REINFORCE | -50 to +50 | High variance, doesn't solve |
| REINFORCE+baseline | 50-130 | Reduced variance, still unstable |
| A2C | 130-170 | Faster learning, entropy tuning helps |
| GAE | 130-170 | Smoother advantages, similar stability to A2C |
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

### GAE: Generalized Advantage Estimation (2015)

**The Paper & Authors**

"High-Dimensional Continuous Control Using Generalized Advantage Estimation" was submitted to arXiv in June 2015 and published at ICLR 2016. The authors:

- **John Schulman** (lead author) - PhD student at Berkeley under Pieter Abbeel
- **Philipp Moritz** - Berkeley, later co-created Ray/RLlib
- **Sergey Levine** - Berkeley robotics, now professor
- **Michael I. Jordan** - Berkeley ML legend
- **Pieter Abbeel** - Schulman's advisor, robotics/RL pioneer

**John Schulman's Journey**

Schulman's path is remarkable: Caltech physics (2010) → Berkeley PhD in robotics/RL → co-founded OpenAI in December 2015 (while still finishing his PhD) → invented TRPO, GAE, and PPO → led RLHF development for ChatGPT → joined Anthropic (August 2024) → now Chief Scientist at Thinking Machines Lab (February 2025).

He's often called the "architect of ChatGPT" - his PPO algorithm became the backbone of RLHF training. GAE was a stepping stone: TRPO (2015) → GAE (2015) → PPO (2017).

**The Core Insight: λ-Weighted Advantage**

GAE builds directly on Richard Sutton's TD(λ) from 1988. Sutton showed you can interpolate between:
- **TD(0)**: Bootstrap immediately from V(s') - low variance, high bias
- **Monte Carlo**: Wait for full return - high variance, low bias

GAE applies this to advantage estimation:

```
λ=0:    A_t = r_t + γV(s_{t+1}) - V(s_t)    # 1-step TD (like A2C with n=1)
λ=1:    A_t = G_t - V(s_t)                   # Monte Carlo (like REINFORCE+baseline)
λ=0.95: Exponentially-weighted blend of all n-step estimates
```

The formula:
```
A_t^GAE = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

**What Problem It Solved**

The paper tackled high-dimensional continuous control - robotic locomotion tasks like walking, running, swimming. These have:
- Long horizons (thousands of steps)
- Dense rewards
- Need for sample efficiency

With λ=0.95-0.99, GAE gave stable learning on tasks that pure MC or TD struggled with.

**Still Used Today?**

Absolutely. GAE is the default in virtually every modern PPO implementation:
- Stable Baselines3: `gae_lambda=0.95` default
- CleanRL, RLlib, TRL (Hugging Face)
- All RLHF systems for LLMs use PPO+GAE

Even newer algorithms like GRPO (DeepSeek) are compared against PPO+GAE as the baseline.

**How It Builds on Our A2C**

Our A2C uses fixed n-step returns (n=5):
```
G_t = r_t + γr_{t+1} + ... + γ^4 r_{t+4} + γ^5 V(s_{t+5})
```

GAE instead computes a weighted average of ALL possible n-step estimates:
```
A_t^GAE = (1-λ)(A_t^(1) + λA_t^(2) + λ²A_t^(3) + ...)
```

This smooths out the advantage estimates - instead of hard-cutoff at n steps, you get exponential decay.

**Predicted Impact on Our Results**

| Issue | A2C Problem | GAE Solution |
|-------|-------------|--------------|
| Noisy advantages | Fixed n=5 cutoff is arbitrary | Smooth λ-weighted blend |
| Variance spikes | Single n-step estimate | Averaged over all horizons |
| Bias from critic | Only bootstraps at step n | Balances bootstrap vs MC |

Expected: More stable learning curves, less dramatic collapses. The λ parameter gives us a tuning knob (like entropy_coef did). Typical values: λ=0.95-0.99.

GAE alone won't solve our stability problem (that's PPO's clipped objective), but it should give smoother advantage estimates that help the policy make more consistent updates.

## Bugs & Lessons Learned

### The Silent Critic Bug (Fixed after commit `8cd23ea`)

**Symptom:** A2C training showed eval returns of -700 to -900 instead of learning. Policy loss was near zero.

**Root Cause:** The critic had zero gradient flow. We computed values during `act()` under `torch.no_grad()` and stored them as Python scalars:

```python
# In act() - values computed with no gradient tracking
with torch.no_grad():
    value = self.critic(obs_tensor).item()  # Scalar, no grad history

# In update() - using stored values for loss
value_loss = ((data["values"] - returns_tensor) ** 2).mean()
# ^^^ This tensor has NO connection to self.critic!
# loss.backward() computes nothing for the critic
```

The value loss was computed but `loss.backward()` produced no gradients for the critic network. The critic never learned, so advantages were meaningless noise.

**The Fix:** Recompute fresh value predictions during `update()`:

```python
# In update() - recompute values WITH gradients
values_pred = self.critic(data["obs"]).squeeze(-1)  # Fresh forward pass
value_loss = ((values_pred - returns_tensor) ** 2).mean()
# ^^^ Now gradients flow back through self.critic
```

**Lesson:** When debugging RL, check that gradients actually flow where you expect. A loss being computed doesn't mean anything is learning. The stored values are still useful for computing advantage (no gradients needed there), but the value loss needs fresh predictions.

**Results after fix:**
- A2C: Episodes reaching 200+ returns within 300k steps
- REINFORCE+baseline: Eval improved from -722 → +11 by 450k steps

## References

- [Williams 1992 - REINFORCE](https://link.springer.com/article/10.1007/BF00992696)
- [Barto, Sutton, Anderson 1983 - Actor-Critic](https://ieeexplore.ieee.org/document/6313077)
- [Sutton 1988 - TD Learning](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)
- [Mnih et al. 2016 - A3C](https://arxiv.org/abs/1602.01783)
- [OpenAI Baselines: A2C](https://openai.com/index/openai-baselines-acktr-a2c/)
- [Policy Gradient Methods (Sutton & Barto Ch. 13)](http://incompleteideas.net/book/the-book.html)
- [PPO Paper (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347)
- [GAE Paper (Schulman et al. 2015)](https://arxiv.org/abs/1506.02438)
