# Hyperparameter & Architecture Research for LunarLander‑v2

## Executive summary

- **Start from a strong PPO baseline** –  The RL‑Baselines3‑Zoo trained agent uses a simple multilayer perceptron (MLP) with two hidden layers of 64 units and default PPO hyperparameters; this setup achieves ~233 mean reward with a standard deviation of 53 and 1 million training steps【321436788545711†L70-L82】.  To reach leaderboard scores (mean−std ≥ 200) you’ll need to tune the learning rate, discount factor (`gamma`), generalized advantage estimation (`gae_lambda`), batch size, number of environments (`n_envs`) and network width.  Increase training timesteps (2–10 million) and parallel environments (16–64) to improve sample efficiency.

- **Use automated hyperparameter optimization** –  Modern tools reduce the cost of searching large hyperparameter spaces.  Optuna provides easy parallelization, state‑of‑the‑art samplers and pruners and requires only a few lines of code【92142108303660†L95-L112】; RL‑Baselines3‑Zoo integrates Optuna and shows how to tune PPO with random samplers and median pruners across multiple trials【593228567116626†L166-L175】.  Ray Tune supports advanced schedulers like Population‑Based Training (PBT) that mutate hyperparameters during training【274772747070616†L859-L899】.  Weights & Biases (W&B) Sweeps offer Bayesian and random search with built‑in visualization and parallel execution【922107774318047†L91-L140】.  Use these frameworks to explore learning rate (1e‑5 – 1e‑3), `gamma` (0.99 – 0.9999), `clip_range` (0.1 – 0.3), entropy coefficient, batch size and network size.

- **Prioritize learning rate, discount factor and number of steps** –  Empirical studies show that the learning rate and `gamma` are the most sensitive hyperparameters for PPO and DQN.  The paper *What matters in on‑policy RL?* recommends tuning `gamma` around 0.99 and using generalized advantage estimation with `gae_lambda≈0.9`【500994576558689†L81-L166】.  The RL‑Zoo uses `gamma=0.999` and `gae_lambda=0.98` for LunarLander【321436788545711†L70-L82】—increasing `gamma` improves long‑term rewards but may slow learning.  Adjust `n_steps` (number of steps per rollout) and `n_envs` together to control the batch size; large `n_steps*n_envs` lead to better gradient estimates but longer update intervals.

- **Design the network carefully** –  For 1‑D observations, Stable‑Baselines3 defaults to two fully connected layers of 64 units for PPO/A2C/DQN【161592480443648†L123-L133】.  Consider widening the critic network (128–512 units) while keeping the actor smaller; research shows that actor networks can often be reduced by up to 99 % without harming performance【362968749460804†L23-L116】.  Use `tanh` or `ReLU` activations with orthogonal initialization and separate actor/critic networks; deeper networks (3 layers) sometimes help, but bigger isn’t always better.

- **Reduce variance and ensure reproducibility** –  RL results vary widely across random seeds; the SB3 tips page recommends running several seeds and averaging results【192912671246389†L125-L136】.  Evaluate the trained agent on a separate test environment for 5–20 episodes【192912671246389†L185-L188】 and use deterministic actions during evaluation to reduce noise【192912671246389†L185-L197】.  Normalize observations (with `VecNormalize`), clip returns and advantages, apply orthogonal initialization and gradient clipping【513231685660795†L16-L80】.  Penultimate normalization (scaling penultimate layer features to unit norm) has been shown to reduce variance by more than a factor of three across tasks【541930901020993†L24-L106】.

## Tool comparison table

| Tool/framework | Setup complexity | Sample efficiency & algorithm support | Parallelization & scheduling | Integration with Stable‑Baselines3 | Notes |
| --- | --- | --- | --- | --- | --- |
| **Optuna** | Minimal; define an objective function and search space, install `optuna` | Uses state‑of‑the‑art samplers (TPE, CMA‑ES) and pruning (median, Wilcoxon); easy to parallelize; supports early stopping【92142108303660†L95-L112】 | Supports distributed optimization and can be combined with Dask or joblib; Optuna dashboard visualizes trials | SB3‑Zoo integrates Optuna via the `train.py --optimize` flag; you can also call Optuna manually in custom code【593228567116626†L166-L175】 | Good first choice; handles continuous and discrete parameters; robust pruners reduce wasted trials |
| **Ray Tune** | Moderate; requires Ray installation and definition of a `trainable` function | Supports random search, Bayesian Optimization (HyperOpt, Ax), ASHA/HyperBand and Population‑Based Training; can pause and resume trials | Ray handles distributed execution across CPUs/GPUs; PBT mutates hyperparameters during training【274772747070616†L859-L899】 | Works with SB3 via custom training loops or RLlib; not as plug‑and‑play as Optuna | Ideal when running many parallel experiments or using advanced schedulers; overhead may be overkill for a single environment |
| **Weights & Biases Sweeps** | Low; define sweep configuration (parameters and search strategy) and call `wandb.agent` | Supports random, grid and Bayesian search; provides live visualization of metrics and hyperparameter influence【922107774318047†L91-L140】 | Parallel agents can run across machines; integrates with W&B dashboard for monitoring | SB3 users can wrap training loops to log metrics; W&B callback exists; Sweeps require W&B account | Excellent for collaborative experiments and tracking; external service; no built‑in pruning |
| **RL‑Baselines3 Zoo** | Very easy; training script with `--optimize` uses Optuna; includes tuned hyperparameters for many environments【593228567116626†L166-L175】 | Performs random search with a pruner; default budgets (e.g., 1000 trials) can be large; uses a subset of hyperparameters | Uses multiprocessing across CPU cores; cannot distribute across cluster by default | Built‑in; you can load tuned parameters from YAML files; model loading, evaluation and video generation included | Good baseline; limited to predetermined hyperparameter spaces; less flexible than custom Optuna/Ray setups |
| **Population‑Based Training (PBT)** (Ray Tune scheduler) | More complex; requires Ray and definition of hyperparameter mutation logic | Combines exploration and exploitation: periodically transfers weights from top‑performing trials and mutates hyperparameters【274772747070616†L859-L899】; adapts schedules on the fly | Runs many trials in parallel; Ray orchestrates exploitation/mutation steps | Not directly supported in SB3 but can be implemented via Ray Tune `Trainable` classes | Suitable when hyperparameters need dynamic schedules; high compute cost and complexity may be unnecessary for LunarLander |

## Hyperparameter priority list

### Proximal Policy Optimization (PPO)

| Hyperparameter | Importance & recommended ranges | Discussion |
|---|---|---|
| **Learning rate** | **Very high** – search between **1e‑4 and 3e‑3** (log‑uniform).  Use a linear or exponential schedule (anneal to zero) as recommended by implementation‑matters guidelines【513231685660795†L16-L80】.  Start around **3e‑4** (RL‑Zoo default) and adjust; high learning rates speed up early learning but may cause instability.  Consider separate learning rates for actor and critic (e.g., actor 1e‑4, critic 3e‑4) once stable. | Influences both sample efficiency and final performance; interacts with batch size and number of epochs. |
| **Discount factor (`gamma`)** | **High** – tune from **0.99 to 0.9999**.  For episodic tasks like LunarLander (max ~1000 steps), higher values (≥0.999) encourage longer‑term rewards but increase variance.  The RL‑Zoo uses 0.999【321436788545711†L70-L82】; the on‑policy RL study recommends starting at 0.99【500994576558689†L81-L166】.  Evaluate effect on landing precision and fuel consumption. | Higher `gamma` values delay reward discounting; may require smaller learning rate and larger batch sizes. |
| **Generalized advantage estimation (`gae_lambda`)** | **High** – tune **0.9–1.0**.  Lower values reduce variance at the cost of more bias; 0.95–0.98 are common.  The RL‑Zoo uses 0.98【321436788545711†L70-L82】; the on‑policy RL study suggests ~0.9【500994576558689†L81-L166】. | Balances bias–variance in advantage estimates; interacts strongly with `gamma`. |
| **Clip range (`clip_range`)** | **Medium** – typical values **0.1–0.3**.  A study recommends ~0.25【500994576558689†L81-L166】; RL‑Zoo uses default 0.2.  Too small values hinder learning; too large values allow large policy updates and may increase variance. | Consider decaying the clip range during training. |
| **Entropy coefficient (`ent_coef`)** | **Medium** – tune **0.0–0.02**.  Encourages exploration; RL‑Zoo uses 0.01【321436788545711†L70-L82】.  Higher values may reduce final performance but lower standard deviation; try decaying entropic regularization over time. | Reduces premature convergence; interacts with exploration noise. |
| **Value function coefficient (`vf_coef`)** | **Medium** – adjust **0.5–1.0**.  Balances policy and value losses; default 0.5 works for many tasks.  Higher values emphasize value learning and can stabilise training but may slow policy improvement. | Monitor the ratio of policy and value losses; tune in conjunction with learning rate. |
| **Number of steps per update (`n_steps`)** | **Medium** – LunarLander episodes range 200–1000 steps.  Tune total batch size `n_steps * n_envs` between **2048 and 65536**.  RL‑Zoo uses 1024 steps with 16 envs (=16384 batch)【321436788545711†L70-L82】.  More steps improve gradient estimates but increase delay between updates; adjust `batch_size` accordingly. | Increase `n_steps` and `n_envs` together to keep batch size manageable. |
| **Batch size** | **Medium** – tune **64–2048**.  Should divide `n_steps * n_envs`.  Smaller batches give noisier updates; larger batches require more memory.  Combine with `n_epochs` (number of minibatch passes). | Keeping batch size near 64–256 often works well; RL‑Zoo uses 64【321436788545711†L70-L82】. |
| **Number of epochs (`n_epochs`)** | **Lower** – typical values **3–10**.  More epochs reuse data more but risk overfitting; RL‑Zoo uses 4【321436788545711†L70-L82】.  Increase when using small batches. | Monitor KL divergence to avoid over‑updating. |
| **Number of environments (`n_envs`)** | **High** for wall‑clock speed – tune **8–64** depending on hardware.  More environments generate diverse experiences and reduce variance but increase CPU overhead.  For Apple M‑series CPU consider 8–16; for cloud GPU with many cores try 32–64. | `n_envs` multiplies with `n_steps` to define batch size; ensure your network fits in memory. |
| **Max gradient norm (`max_grad_norm`)** | **Low** – default **0.5** or **1.0**.  Controls gradient clipping; prevents exploding gradients【513231685660795†L16-L80】.  Adjust only when training becomes unstable. |
| **Activation and initialization** | **Important** – use `tanh` or `ReLU` with **orthogonal initialization**; this combination stabilizes training and improves performance【513231685660795†L16-L80】.  Initialize final layer weights smaller and use Softplus for standard deviation outputs【500994576558689†L81-L166】. | Implementation details matter; follow best practices from Implementation Matters. |

### Deep Q‑Network (DQN)

| Hyperparameter | Recommended ranges & comments | Discussion |
|---|---|---|
| **Learning rate** | **0.0001–0.001**.  SB3 default is 1e‑4【409894009475792†L235-L241】; research exploring 0.0001–0.01 found best results near **0.001**【367633168928146†L320-L340】.  Use schedules if training diverges. | Too high a rate leads to oscillations; too low slows convergence. |
| **Replay buffer size (`buffer_size`)** | **100 k – 1 M** transitions.  SB3 default is 1 million【409894009475792†L235-L241】.  Larger buffers improve data diversity but use more memory. | For LunarLander, 100k–500k may suffice; monitor memory usage. |
| **Learning starts (`learning_starts`)** | **1 k – 10 k** steps.  SB3 default is 100 steps【409894009475792†L235-L241】; increasing to 500–1000 ensures the replay buffer contains diverse experiences before updates. | Avoid starting updates too early to prevent overfitting to poor trajectories. |
| **Batch size** | **32–128**.  SB3 default is 32【409894009475792†L235-L241】; experiments found good results with 64【367633168928146†L320-L340】.  Larger batches improve stability but increase memory. |
| **Target network update** | Soft update coefficient `tau` **0.005–0.01** (for Polyak averaging) or hard update every **1–5 k** steps (SB3 default 10 000【409894009475792†L235-L241】).  Soft updates often yield smoother learning; tune `tau` if using Polyak. |
| **Train frequency (`train_freq`)** | Train every **1–4** steps.  SB3 default is every 4 steps【409894009475792†L235-L241】.  More frequent updates speed learning but may overfit; adjust with `gradient_steps`. |
| **Gradient steps per update (`gradient_steps`)** | **1–8**.  SB3 default is 1; multiple gradient steps can improve sample efficiency at the cost of computation. | Use `-1` in SB3 to match the number of gradient steps to collected steps. |
| **Exploration schedule** | Initial epsilon 1.0 decaying to 0.05 over **0.1–0.5** of training steps【409894009475792†L235-L241】.  Use `epsilon-greedy` or Noisy Networks for exploration; slower decays improve long‑term exploration. |
| **Network architecture** | Two fully connected layers of **64** units with ReLU (SB3 default)【161592480443648†L123-L133】.  Try wider layers (128–256), dueling architecture and Double DQN to reduce over‑estimation. | Increase width if the agent struggles to learn; deeper networks can overfit for small state spaces. |
| **Prioritized replay & n‑step returns** | Use Prioritized Experience Replay (PER) and **n‑step returns** to improve sample efficiency; tune PER’s alpha (0.4–0.6) and beta (annealed to 1.0). | These extensions reduce variance and focus learning on important transitions. |

## Architecture recommendations

1. **Baseline architecture** –  For LunarLander’s 8‑dimensional observation space and discrete action space (4 actions), start with the default **[64, 64]** network used by SB3 (two layers of 64 units)【161592480443648†L123-L133】.  This architecture is simple and efficient.

2. **Increase critic capacity** –  Research shows that the critic benefits more from capacity than the actor.  In the **Honey I Shrunk the Actor** study, actor networks were reduced by up to **99 %** without hurting performance【362968749460804†L23-L116】.  Therefore, try wider critics (128–512 units) while keeping the actor at 64–128 units.  SB3 allows specifying separate sizes via `policy_kwargs=dict(net_arch=[dict(pi=[sizes], vf=[sizes])])`.

3. **Depth & width** –  Experiment with **3‑layer networks** such as [64, 64, 64], [128, 128, 64] or [256, 256, 128].  Deeper networks may learn more complex value functions but can be harder to train; use layer normalization if training becomes unstable.  For DQN, keep depth ≤3 to avoid overfitting.

4. **Activation functions** –  Use **Tanh** or **ReLU** activations.  Implementation‑Matters highlights that **tanh** combined with **orthogonal initialization** stabilizes PPO training【513231685660795†L16-L80】.  Avoid batch normalization; instead, consider **layer normalization** in recurrent settings.

5. **Orthogonal initialization and smaller final layers** –  Initialize weights orthogonally and scale final layer weights down (e.g., multiply by 0.01) to prevent large initial outputs【500994576558689†L81-L166】.

6. **Structured architectures** –  Consider **Structured Control Nets**, which split the policy into linear and nonlinear modules and improve sample efficiency while reducing parameter count【713955018304670†L8-L24】.  This may help when widening networks doesn’t increase performance.

7. **Observation normalization** –  Always normalize observations.  Use SB3’s `VecNormalize` wrapper or implement running mean/variance using Welford’s algorithm【192912671246389†L139-L143】.  Normalize rewards if using DQN.

## Experiment roadmap

1. **Baseline & reproducibility**
   - Train a PPO agent with RL‑Zoo default hyperparameters (two layers of 64, `gamma=0.999`, `gae_lambda=0.98`, learning rate 3e‑4) for **1 million steps** and evaluate on **20** episodes using deterministic actions.  Record mean and standard deviation of rewards and multiple random seeds (at least 3)【192912671246389†L125-L136】.
   - Ensure the environment uses `VecNormalize` for observation scaling and wrap it with a `Monitor` to record episode returns【192912671246389†L139-L144】.

2. **Extended training**
   - Increase training timesteps to **2 M**, **5 M** and **10 M** while keeping other hyperparameters fixed.  Plot mean reward and standard deviation over time to assess diminishing returns.

3. **Network size sweep**
   - Evaluate [64, 64], [128, 128], [256, 256], [64, 64, 64] and asymmetric architectures (actor [64, 64], critic [128, 128] or [256, 256]).  Train each configuration for 2–5 M steps with 16 environments and measure mean ± std.
   - Compare performance and compute cost; note that larger networks may require CPU or GPU decisions (see compute efficiency below).

4. **Learning rate & gamma sweep**
   - Use Optuna or W&B Sweeps to search learning rate (1e‑5 – 3e‑3), `gamma` (0.99 – 0.9999) and `gae_lambda` (0.9 – 0.99).  Limit trial budget (~50–100) with a median pruner; evaluate on 5 episodes during tuning.  Expect learning rate and `gamma` to be most sensitive.

5. **Batch size & steps sweep**
   - Vary `n_steps` (256–4096) and `n_envs` (8–64), keeping batch size manageable (e.g., total transitions between 8k and 32k).  Evaluate how larger rollouts affect variance and wall‑clock time.  Use vector environments to parallelize across CPU cores; for each configuration, adjust `batch_size` and `n_epochs` accordingly.

6. **Entropy & clip range sweep**
   - Tune `ent_coef` (0–0.02) and `clip_range` (0.1–0.3).  Consider decaying `ent_coef` over training.  Monitor the KL divergence and adjust `clip_range` if updates are too conservative or too aggressive.

7. **DQN baseline & variants**
   - Implement a vanilla DQN with default SB3 hyperparameters: learning rate 1e‑4, buffer size 1 M, target update every 10 k steps, batch size 32【409894009475792†L235-L241】.  Train for 1 M steps and evaluate.
   - Explore Double DQN, Dueling DQN, Prioritized Replay and Noisy Networks.  Tune learning rate (1e‑4–1e‑3), buffer size (100k–1 M), batch size (32–128) and target update frequency (5 k–20 k).  Evaluate after 2–5 M steps.

8. **Variance reduction**
   - Implement **penultimate normalization**: normalize the penultimate layer’s feature vector to unit norm before the output layer.  This technique reduces variance across seeds by preventing saturating activations【541930901020993†L24-L106】.
   - Clip rewards to a bounded range (e.g., ±1) and normalize advantages (zero mean, unit variance).  Use gradient clipping and orthogonal initialization【513231685660795†L16-L80】.
   - Evaluate each candidate over more episodes (20–50) to compute mean and standard deviation; sample multiple seeds and average results to reduce noise【192912671246389†L125-L136】.

9. **Compute efficiency & hardware**
   - **CPU vs GPU** –  Deep RL is often CPU‑bound because environments generate experience sequentially; small networks may run faster on CPU.  A NUS study showed that training small recurrent networks (hidden units 128–1024) was faster on CPUs than GPUs; the overhead of transferring tensors to the GPU outweighed computation benefits【332173762009436†L158-L166】.  Only when the network exceeded ~1024 units did the GPU become faster【332173762009436†L168-L172】.
   - **Parallel environments** –  Use multiple CPU cores to run vectorized environments; this reduces wall‑clock time and stabilizes gradients.  On local Apple Silicon, 8–16 environments typically saturate CPU cores; on cloud GPUs with more cores, consider 32–64.
   - **Trial and logging** –  Run a few trials to determine whether CPU or GPU is faster for your network size【332173762009436†L176-L181】.  Generate logs to identify bottlenecks, especially when transferring tensors between CPU and GPU【332173762009436†L188-L195】.

10. **Final push & leaderboard submission**
    - After identifying strong hyperparameters, train **multiple agents** (e.g., 5–10 seeds) and select the model with the highest **mean – std** on the evaluation episodes.  This exploits variance across runs while adhering to the leaderboard’s ranking metric.
    - Evaluate the final model on at least **50** episodes to obtain a reliable estimate of mean and standard deviation; use deterministic actions; record seeds and hyperparameters for reproducibility【192912671246389†L185-L197】.
    - Optionally explore PBT or dynamic hyperparameter schedules if compute allows; these methods can adapt learning rate and exploration coefficients during training and may yield additional gains【274772747070616†L859-L899】.

## Resource links

- **SB3 Reinforcement Learning Tips and Tricks** – guidance on evaluation, seeds, normalization and hyperparameter tuning【192912671246389†L125-L136】【192912671246389†L185-L197】.
- **Stable‑Baselines3 Policy Networks** – default network architectures for PPO/DQN and guidelines for custom architectures【161592480443648†L123-L133】.
- **RL‑Baselines3 Zoo documentation** – information on training scripts, hyperparameter files and Optuna‑based optimization【593228567116626†L166-L175】.
- **Hugging Face SB3 PPO LunarLander model** – lists the RL‑Zoo hyperparameters used (batch size 64, `gamma=0.999`, `gae_lambda=0.98`, 16 environments, 1 M timesteps) and evaluation results【321436788545711†L70-L82】.
- **Implementation Matters in Deep RL** – details code‑level tricks like orthogonal initialization, learning rate annealing, reward clipping and gradient clipping that significantly influence PPO performance【513231685660795†L16-L80】.
- **What matters in on‑policy RL?** – recommends tuning `gamma` (~0.99), using `gae_lambda≈0.9`, separate actor/critic networks, small final layer initialization and `tanh` activations【500994576558689†L81-L166】.
- **Honey, I Shrunk the Actor** – shows actor networks can be reduced drastically without loss, suggesting prioritizing critic capacity【362968749460804†L23-L116】.
- **Structured Control Nets** – proposes splitting policy networks into linear and nonlinear modules to improve sample efficiency and generalization【713955018304670†L8-L24】.
- **Hyperparameters in Reinforcement Learning and How To Tune Them** – highlights the importance of formal hyperparameter optimization and recommends separating tuning and testing seeds; shows HPO methods outperform hand‑tuning with less compute【849103764380465†L10-L24】【849103764380465†L90-L103】.
- **NUS report on minimizing processing time in deep RL** – explains why RL is often CPU‑bound and shows that CPUs are faster than GPUs for small networks (128–1024 hidden units)【332173762009436†L158-L166】, recommending running trials to determine whether GPU is beneficial【332173762009436†L176-L181】.
- **Variance reduction case study (ICLR 2022)** – identifies unstable network parametrization as a cause of high variance and recommends **penultimate normalization**, which reduced average standard deviation by a factor >3 across tasks【541930901020993†L24-L106】.

