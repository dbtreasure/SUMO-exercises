# SUMO RL Agent Comparison

Reproduction of a 2022 SUMO User Conference paper comparing DQN vs PPO for traffic signal optimization.

Louw, C., Labuschagne, L. ., & Woodley, T. . (2022). A Comparison of Reinforcement Learning Agents Applied to Traffic Signal Optimisation. SUMO Conference Proceedings, 3, 15–43. https://doi.org/10.52825/scp.v3i.116

https://www.tib-op.org/ojs/index.php/scp/article/view/116

## Setup

```bash
uv venv && source .venv/bin/activate
uv sync
```

## Build Network

```bash
./scripts/build_single_int_network.sh
```

## Run Simulation (headless)

```bash
sumo -c scenarios/single_int/sim.sumocfg --quit-on-end
```
