#!/usr/bin/env python
"""Visualize agents in SUMO GUI for qualitative comparison.

Usage:
    # Watch trained DQN
    uv run python -m src.training.visualize \
        --agent trained \
        --model-path runs/dqn/smoke_test/models/final.zip \
        --algo dqn \
        --config configs/smoke_dqn.yaml

    # Watch fixed cycle baseline
    uv run python -m src.training.visualize \
        --agent fixed_cycle \
        --config configs/smoke_dqn.yaml

    # Watch random baseline
    uv run python -m src.training.visualize \
        --agent random \
        --config configs/smoke_dqn.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import DQN, PPO

from src.baselines.agents import FixedCycleAgent, RandomAgent
from src.envs.sumo_utils import close_sumo, get_sumo_tools_path, start_sumo


def make_gui_env(cfg: dict, seed: int = 42, delay: int = 200):
    """Create environment config for GUI mode (returns components, not full env)."""
    env_cfg = cfg["env"]
    return {
        "sumocfg_path": str(Path(env_cfg["sumocfg_path"]).resolve()),
        "decision_interval_s": env_cfg["decision_interval_s"],
        "min_green_s": env_cfg["min_green_s"],
        "max_sim_time_s": env_cfg.get("max_sim_time_s", 300),
        "seed": seed,
        "delay": delay,
    }


def run_gui_episode(
    agent,
    env_cfg: dict,
    agent_name: str,
) -> None:
    """Run a single episode with SUMO GUI.

    Args:
        agent: Agent with predict(obs, deterministic) method.
        env_cfg: Environment configuration dict.
        agent_name: Name of agent for display.
    """
    get_sumo_tools_path()
    import traci

    print(f"\nStarting SUMO GUI with {agent_name}...")
    print("Use GUI controls: Play/Pause, Step, or set Delay slider.")
    print("Close the SUMO GUI window when done.\n")

    # Start SUMO with GUI
    label = start_sumo(
        env_cfg["sumocfg_path"],
        gui=True,
        seed=env_cfg["seed"],
        override_end_s=env_cfg["max_sim_time_s"],
        auto_start=True,
        gui_delay=env_cfg.get("delay", 200),
    )
    traci.switch(label)

    # Discover TLS
    tls_ids = traci.trafficlight.getIDList()
    if not tls_ids:
        raise RuntimeError("No traffic lights found")
    tls_id = tls_ids[0]

    # Get controlled lanes
    all_lanes = traci.trafficlight.getControlledLanes(tls_id)
    controlled_lanes = sorted(set(lane for lane in all_lanes if not lane.startswith(":")))

    # Get number of phases
    logics = traci.trafficlight.getAllProgramLogics(tls_id)
    n_phases = len(logics[0].phases)

    print(f"TLS ID: {tls_id}")
    print(f"Controlled lanes: {controlled_lanes}")
    print(f"Phases: {n_phases}")

    # Initialize phase tracking
    current_phase = traci.trafficlight.getPhase(tls_id)
    time_since_phase_start = 0.0
    min_green_s = env_cfg["min_green_s"]
    decision_interval_s = env_cfg["decision_interval_s"]

    # Reset agent if needed
    if hasattr(agent, "reset"):
        agent.reset()

    step_count = 0
    total_reward = 0.0

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            sim_time = traci.simulation.getTime()
            if sim_time >= env_cfg["max_sim_time_s"]:
                break

            # Get observation (halting counts per lane)
            obs = np.array(
                [traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes],
                dtype=np.float32,
            )

            # Get action from agent
            action, _ = agent.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action.item() if action.size == 1 else action[0]

            # Apply action with min green constraint
            if time_since_phase_start >= min_green_s:
                if action != current_phase:
                    traci.trafficlight.setPhase(tls_id, action)
                    current_phase = action
                    time_since_phase_start = 0.0

            # Step simulation
            for _ in range(int(decision_interval_s)):
                traci.simulationStep()
                time_since_phase_start += 1.0

            # Compute reward (halting vehicles only)
            reward = 0.0
            for lane in controlled_lanes:
                for vid in traci.lane.getLastStepVehicleIDs(lane):
                    if traci.vehicle.getSpeed(vid) <= 0.1:
                        reward -= traci.vehicle.getWaitingTime(vid)

            total_reward += reward
            step_count += 1

            # Print periodic status
            if step_count % 10 == 0:
                print(
                    f"  t={sim_time:6.1f}s  phase={current_phase}  "
                    f"queue={obs.sum():.0f}  reward={reward:.1f}"
                )

    except traci.exceptions.FatalTraCIError:
        print("SUMO GUI closed by user.")
    finally:
        close_sumo(label=label)

    print(f"\nEpisode complete: {step_count} steps, total reward: {total_reward:.2f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize agents in SUMO GUI")
    parser.add_argument(
        "--agent",
        type=str,
        choices=["trained", "random", "fixed_cycle"],
        required=True,
        help="Agent type to visualize",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (required if agent=trained)",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["dqn", "ppo"],
        default="dqn",
        help="Algorithm (required if agent=trained)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for simulation",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=200,
        help="GUI delay in ms between steps (higher=slower, 0=max speed)",
    )
    return parser.parse_args()


def main() -> None:
    """Run visualization."""
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = make_gui_env(cfg, seed=args.seed, delay=args.delay)

    # We need to know n_phases for baseline agents
    # Quick probe to get it
    get_sumo_tools_path()
    import traci

    probe_label = start_sumo(
        env_cfg["sumocfg_path"],
        gui=False,
        seed=0,
        override_end_s=10,
    )
    traci.switch(probe_label)
    tls_id = traci.trafficlight.getIDList()[0]
    logics = traci.trafficlight.getAllProgramLogics(tls_id)
    n_phases = len(logics[0].phases)
    close_sumo(label=probe_label)

    # Create agent
    if args.agent == "trained":
        if args.model_path is None:
            raise ValueError("--model-path required for trained agent")
        if args.algo == "dqn":
            agent = DQN.load(args.model_path)
        else:
            agent = PPO.load(args.model_path)
        agent_name = f"Trained {args.algo.upper()}"
    elif args.agent == "random":
        agent = RandomAgent(n_phases, seed=args.seed)
        agent_name = "Random"
    elif args.agent == "fixed_cycle":
        if n_phases == 4:
            cycle_phases = [0, 2]
        else:
            cycle_phases = [0, 1]
        agent = FixedCycleAgent(n_phases, cycle_phases=cycle_phases, steps_per_phase=8)
        agent_name = "Fixed Cycle"
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    print("=" * 60)
    print(f"SUMO GUI Visualization: {agent_name}")
    print("=" * 60)

    run_gui_episode(agent, env_cfg, agent_name)


if __name__ == "__main__":
    main()
