# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

Reproduction of a 2022 SUMO User Conference paper on applying **DQN vs PPO** to **traffic signal optimization** in **SUMO**, with a minimal, readable implementation.

### Primary Goals
1. Reproduce the paper's **Section 4 (Methodology)** faithfully:
   - Observation: **queue length per incoming lane** (halting vehicles; speed ≤ 0.1 m/s)
   - Reward: **negative sum of waiting times** of halting vehicles
   - Action: **phase changes** with a **minimum green time** constraint
   - Train: **DQN** and **PPO**
2. Keep the implementation **simple**, **educational**, and **deterministic** (seeded)
3. After reproduction, modernize iteratively (2026 add-ons), but **not yet**

### Non-Goals (for now)
- Multi-agent / multi-intersection control
- Publishing a poster/paper
- Heavy infra, distributed training, or exotic RL libraries

---

## Commands

```bash
# Environment setup
uv venv && source .venv/bin/activate
uv sync

# Run training (target commands - not yet implemented)
uv run python -m src.training.train_dqn --config configs/single_int_dqn.yaml
uv run python -m src.training.train_ppo --config configs/single_int_ppo.yaml

# Test SUMO scenario runs
sumo -c scenarios/single_int/sim.sumocfg --quit-on-end
```

---

### Planned File Structure (create if missing)

    scenarios/
      single_int/
        plain/
          nodes.nod.xml
          edges.edg.xml
        routes.rou.xml
        sim.sumocfg
        net.net.xml
    scripts/
      build_single_int_network.sh
      smoke_traci.py
    src/
      envs/
        tsc_env.py          # Gymnasium env wrapping SUMO+TraCI
        obs.py              # queue-length observation
        reward.py           # waiting-time reward
        tls.py              # phase / TLS utilities
        sumo_utils.py       # SUMO start/stop helpers
      training/
        train_dqn.py
        train_ppo.py
        train_ppo.py
        eval.py
    configs/
      single_int_dqn.yaml
      single_int_ppo.yaml

---

## Implementation Specifications

### Environment API
Gymnasium env with:
- `reset(seed=...) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`
- `decision_interval_s`: seconds of SUMO simulation per agent step (e.g., 5s)
- `min_green_s`: minimum green time before allowing phase change (e.g., 10-15s)

### Observation (paper-faithful)
- For each **incoming lane** to the intersection: queue = number of vehicles with speed ≤ 0.1 m/s (halting)
- Observation vector size = number of incoming lanes
- Keep lane ordering stable and explicit

### Reward (paper-faithful)
- `R_t = - sum(waiting_time(vehicle_i))` for vehicles halting on controlled incoming lanes
- Use SUMO waiting time semantics via TraCI

### Action Space
- Discrete `K` actions = pick among **valid green phases**
- Enforce phase changes only when `time_since_phase_start >= min_green_s`
- Insert yellow/all-red transitions if needed when switching

### Eval Metrics
- Mean waiting time
- Mean delay/travel time through intersection
- Mean + max queue length
- Throughput (vehicles completed)
- Stops (optional)

---

## Repo Principles

1. **Minimal diffs, tight scope**: prefer small, incremental PR-sized changes
2. **Reproducibility**: every training run should be seedable and produce consistent logs
3. **Readability > cleverness**: plain Python, explicit types where helpful, docstrings
4. **Fail fast with good errors**: validate SUMO paths, scenario files, TLS id presence, lane list size
5. **Avoid overfitting to one machine**: don't hardcode absolute paths; read from config/env

---

## SUMO Environment Assumptions

- SUMO installed (via .pkg, brew tap, or PyPI). `sumo`, `sumo-gui`, and `netedit` should be runnable.
- `SUMO_HOME` points to SUMO installation root containing `tools/` directory
- Python bindings are in `$SUMO_HOME/tools`

---

## Current Phase: Task 1 & 2

### Task 1: Create `scenarios/single_int` minimal runnable SUMO config
- Create a tiny 4-way intersection with TLS
- Simple flows in `routes.rou.xml` (two perpendicular directions)
- Ensure `sumo -c scenarios/single_int/sim.sumocfg --quit-on-end` runs

### Task 2: TraCI smoke test script
- Starts SUMO via TraCI
- Steps 100 ticks
- Prints simulation time
- Prints one lane's halting count

---

## Definition of Done (Reproduction Phase)

We consider Phase 1 "reproduced" when:
- Both `train_dqn` and `train_ppo` commands run successfully
- Both produce training curves (reward over time) and evaluation metrics
- The environment implements: queue-length obs, waiting-time reward, phase selection with min green
- README explains install, training/eval, and results location
