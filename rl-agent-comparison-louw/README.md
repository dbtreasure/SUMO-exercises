# SUMO RL Agent Comparison

Reproduction of a 2022 SUMO User Conference paper comparing DQN vs PPO for traffic signal optimization.

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
