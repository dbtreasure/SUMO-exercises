"""Gymnasium environment for single-intersection traffic signal control.

This environment wraps a SUMO simulation via TraCI to enable RL agent training
for traffic signal optimization.

Paper-faithful implementation (2022 SUMO User Conference):
- Observation: Queue length per incoming lane (halting vehicles, speed <= 0.1 m/s)
- Reward: Negative sum of waiting times of halting vehicles
- Action: Phase selection with minimum green time constraint
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.sumo_utils import close_sumo, get_sumo_tools_path, start_sumo


class SingleIntersectionTSCEnv(gym.Env):
    """Gymnasium environment for traffic signal control at a single intersection.

    Implements the methodology from the 2022 SUMO User Conference paper:
    - Observation: Queue length (halting count) per controlled incoming lane
    - Reward: Negative sum of waiting times for vehicles on controlled lanes
    - Action: Select traffic light phase, subject to minimum green constraint

    Attributes:
        sumocfg_path: Path to SUMO configuration file.
        decision_interval_s: Seconds of simulation per agent step.
        min_green_s: Minimum green time before phase change is allowed.
        tls_id: Traffic light system ID (auto-discovered if None).
        seed: Random seed for reproducibility.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sumocfg_path: str,
        decision_interval_s: float = 5.0,
        min_green_s: float = 10.0,
        max_sim_time_s: int = 300,
        tls_id: str | None = None,
        seed: int | None = 0,
        green_phase_indices: list[int] | None = None,
        yellow_phase_map: dict[tuple[int, int], int] | None = None,
        yellow_duration_s: int = 3,
    ) -> None:
        """Initialize the TSC environment.

        Args:
            sumocfg_path: Path to .sumocfg file.
            decision_interval_s: Simulation seconds per agent decision step.
            min_green_s: Minimum seconds a phase must be active before switching.
            max_sim_time_s: Maximum simulation time in seconds (default 300 for dev).
            tls_id: Traffic light ID. If None, auto-discover first TLS in network.
            seed: Random seed for SUMO simulation reproducibility. None for stochastic.
            green_phase_indices: List of SUMO phase indices that are green phases.
                Default: [0, 2] for standard 4-phase TLS.
            yellow_phase_map: Mapping from (from_green, to_green) to yellow phase index.
                Default: {(0, 2): 1, (2, 0): 3} for standard 4-phase TLS.
            yellow_duration_s: Duration of yellow phase in seconds. Default: 3.
        """
        super().__init__()

        self.sumocfg_path = str(Path(sumocfg_path).resolve())
        self.decision_interval_s = decision_interval_s
        self.min_green_s = min_green_s
        self.max_sim_time_s = max_sim_time_s
        self._tls_id_config = tls_id
        self._base_seed: int | None = seed
        self._episode_count: int = 0

        # Validate config file exists
        if not Path(self.sumocfg_path).exists():
            raise FileNotFoundError(f"SUMO config not found: {self.sumocfg_path}")

        # Runtime state (set during reset)
        self._end_time: float = 3600.0
        self._time_since_phase_start: float = 0.0
        self._current_phase: int = 0
        self._traci = None
        self._connection_label: str | None = None

        # Probe SUMO to discover TLS metadata and set proper spaces
        self.tls_id, self._controlled_lanes, self._n_phases = self._probe_metadata()

        # Initialize green phase configuration
        self._green_phase_indices = (
            green_phase_indices if green_phase_indices is not None else [0, 2]
        )
        self._yellow_phase_map = (
            yellow_phase_map if yellow_phase_map is not None else {(0, 2): 1, (2, 0): 3}
        )
        self._yellow_duration_s = yellow_duration_s

        # Validate green phase configuration
        for gp in self._green_phase_indices:
            if gp < 0 or gp >= self._n_phases:
                raise ValueError(
                    f"Invalid green phase index {gp}. Must be in [0, {self._n_phases - 1}]."
                )
        for (from_g, to_g), yellow in self._yellow_phase_map.items():
            if from_g not in self._green_phase_indices:
                raise ValueError(f"Yellow map key {from_g} not in green_phase_indices.")
            if to_g not in self._green_phase_indices:
                raise ValueError(f"Yellow map key {to_g} not in green_phase_indices.")
            if yellow < 0 or yellow >= self._n_phases:
                raise ValueError(f"Invalid yellow phase index {yellow}.")
        if self._yellow_duration_s > self.decision_interval_s:
            raise ValueError(
                f"yellow_duration_s ({self._yellow_duration_s}) must be <= "
                f"decision_interval_s ({self.decision_interval_s})."
            )

        # Track current green phase (agent-space index)
        self._current_green_idx: int = 0

        # Set proper spaces based on discovered metadata
        n_lanes = len(self._controlled_lanes)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(n_lanes,), dtype=np.float32
        )
        # Action space is now green phases only (not all SUMO phases)
        self._n_green_actions = len(self._green_phase_indices)
        self.action_space = spaces.Discrete(self._n_green_actions)

    def _import_traci(self):
        """Import TraCI module."""
        get_sumo_tools_path()
        import traci

        return traci

    def _probe_metadata(self) -> tuple[str, list[str], int]:
        """Probe SUMO briefly to discover TLS metadata.

        This starts SUMO, queries TLS/lane information, then closes.
        Called during __init__ to set action_space and observation_space
        before any reset() call.

        Returns:
            Tuple of (tls_id, controlled_lanes, n_phases).
        """
        traci = self._import_traci()

        # Start SUMO briefly (seed doesn't matter for metadata probe)
        label = start_sumo(
            self.sumocfg_path,
            gui=False,
            seed=0,
            override_end_s=self.max_sim_time_s,
        )
        traci.switch(label)

        try:
            # Discover TLS
            tls_id = self._discover_tls(traci)

            # Get controlled lanes
            all_lanes = traci.trafficlight.getControlledLanes(tls_id)
            controlled_lanes = sorted(
                set(lane for lane in all_lanes if not lane.startswith(":"))
            )

            # Get number of phases
            logics = traci.trafficlight.getAllProgramLogics(tls_id)
            if not logics:
                raise RuntimeError(f"No program logics found for TLS '{tls_id}'.")
            n_phases = len(logics[0].phases)

            return tls_id, controlled_lanes, n_phases
        finally:
            # Always close the probe connection
            close_sumo(label=label)

    def _discover_tls(self, traci) -> str:
        """Discover the TLS ID from the network.

        Args:
            traci: TraCI module.

        Returns:
            TLS ID string.

        Raises:
            RuntimeError: If no TLS found or specified TLS doesn't exist.
        """
        tls_ids = traci.trafficlight.getIDList()
        if not tls_ids:
            raise RuntimeError("No traffic lights found in network.")

        if self._tls_id_config is not None:
            if self._tls_id_config not in tls_ids:
                raise RuntimeError(
                    f"TLS '{self._tls_id_config}' not found. "
                    f"Available: {list(tls_ids)}"
                )
            return self._tls_id_config

        return tls_ids[0]

    def _get_controlled_lanes(self, traci) -> list[str]:
        """Get unique incoming lanes controlled by the TLS.

        Filters out internal lanes (starting with ':') and returns unique lanes
        in deterministic order.

        Args:
            traci: TraCI module.

        Returns:
            List of incoming lane IDs in sorted order.
        """
        all_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        # Filter out internal lanes and get unique
        incoming = sorted(set(lane for lane in all_lanes if not lane.startswith(":")))
        return incoming

    def _get_n_phases(self, traci) -> int:
        """Get number of phases in the TLS program.

        Args:
            traci: TraCI module.

        Returns:
            Number of phases.
        """
        logics = traci.trafficlight.getAllProgramLogics(self.tls_id)
        if not logics:
            raise RuntimeError(f"No program logics found for TLS '{self.tls_id}'.")
        return len(logics[0].phases)

    def _action_to_green_phase(self, action: int) -> int:
        """Map agent action index to SUMO green phase index.

        Args:
            action: Agent action (0 to n_green_actions - 1).

        Returns:
            SUMO phase index (e.g., 0 or 2 for green phases).
        """
        return self._green_phase_indices[action]

    def _yellow_to_next_green(self, yellow_phase: int) -> int:
        """Given a yellow phase, return the green phase that follows it.

        For this TLS: yellow 1 follows green 0 and leads to green 2,
                      yellow 3 follows green 2 and leads to green 0.

        Args:
            yellow_phase: SUMO yellow phase index.

        Returns:
            SUMO green phase index that follows the yellow.
        """
        for (from_green, to_green), yellow in self._yellow_phase_map.items():
            if yellow == yellow_phase:
                return to_green
        # Fallback: return first green
        return self._green_phase_indices[0]

    def _get_observation(self, traci) -> np.ndarray:
        """Compute observation: halting vehicle count per controlled lane.

        Paper-faithful: Queue = number of vehicles with speed <= 0.1 m/s.
        SUMO's getLastStepHaltingNumber uses this threshold by default.

        Args:
            traci: TraCI module.

        Returns:
            Array of shape (n_lanes,) with halting counts.
        """
        halting = [
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in self._controlled_lanes
        ]
        return np.array(halting, dtype=np.float32)

    def _get_reward(self, traci) -> float:
        """Compute reward: negative sum of waiting times for halting vehicles.

        Paper-faithful: R_t = -sum(waiting_time(veh_i)) for halting vehicles
        on controlled lanes. Halting = speed <= 0.1 m/s (SUMO default threshold).

        Args:
            traci: TraCI module.

        Returns:
            Negative total waiting time (more negative = worse).
        """
        total_wait = 0.0
        for lane in self._controlled_lanes:
            for vid in traci.lane.getLastStepVehicleIDs(lane):
                speed = traci.vehicle.getSpeed(vid)
                if speed <= 0.1:  # Halting threshold (m/s)
                    total_wait += traci.vehicle.getWaitingTime(vid)
        return -total_wait

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and start a new SUMO simulation.

        Args:
            seed: Random seed (overrides __init__ seed if provided).
            options: Additional options (unused).

        Returns:
            Tuple of (observation, info dict).
        """
        super().reset(seed=seed)

        # Close any existing simulation
        self.close()

        # Determine simulation seed:
        # - If seed provided to reset(), use it directly
        # - Otherwise use base_seed + episode_count for variation per episode
        # - If base_seed is None, sim_seed stays None (SUMO uses random)
        if seed is not None:
            sim_seed = seed
        elif self._base_seed is not None:
            sim_seed = self._base_seed + self._episode_count
        else:
            sim_seed = None

        self._episode_count += 1

        # Start SUMO with configured end time
        traci = self._import_traci()
        self._connection_label = start_sumo(
            self.sumocfg_path,
            gui=False,
            seed=sim_seed,
            override_end_s=self.max_sim_time_s,
        )
        # Switch to our connection
        traci.switch(self._connection_label)
        self._traci = traci

        # Set configured end time
        self._end_time = float(self.max_sim_time_s)

        # Initialize phase tracking (TLS metadata already discovered in __init__)
        initial_phase = traci.trafficlight.getPhase(self.tls_id)

        # If starting in a yellow phase, advance to the next green
        if initial_phase not in self._green_phase_indices:
            next_green = self._yellow_to_next_green(initial_phase)
            traci.trafficlight.setPhase(self.tls_id, next_green)
            initial_phase = next_green

        # Set current phase tracking
        self._current_phase = initial_phase
        self._current_green_idx = self._green_phase_indices.index(initial_phase)
        self._time_since_phase_start = 0.0

        # Get initial observation
        obs = self._get_observation(traci)

        info = {
            "sim_time": traci.simulation.getTime(),
            "tls_id": self.tls_id,
            "controlled_lanes": self._controlled_lanes,
            "n_phases": self._n_phases,
            "n_green_actions": self._n_green_actions,
            "green_phase_indices": self._green_phase_indices,
            "phase": self._current_phase,
            "current_green": self._green_phase_indices[self._current_green_idx],
            "current_green_action": self._current_green_idx,
        }

        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one agent step in the environment with automatic yellow insertion.

        The agent selects a green phase action (0 to n_green_actions-1). If a phase
        change is requested and allowed (min_green satisfied), the environment
        automatically inserts the appropriate yellow transition before switching
        to the target green phase.

        Args:
            action: Green phase action index (0 to n_green_actions-1).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if self._traci is None or self._connection_label is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        traci = self._traci
        # Switch to our connection (important with multiple envs)
        traci.switch(self._connection_label)

        # Convert agent action to target SUMO green phase
        target_green = self._action_to_green_phase(action)
        current_green = self._green_phase_indices[self._current_green_idx]

        # Track what happens this step
        phase_changed = False
        yellow_used = False
        remaining_steps = int(self.decision_interval_s)

        # Check if phase change is requested and allowed (min green constraint)
        if target_green != current_green and self._time_since_phase_start >= self.min_green_s:
            # Phase change needed - insert yellow first
            yellow_phase = self._yellow_phase_map.get((current_green, target_green))

            if yellow_phase is not None:
                # Set yellow phase
                traci.trafficlight.setPhase(self.tls_id, yellow_phase)
                yellow_used = True

                # Step through yellow duration
                yellow_steps = min(self._yellow_duration_s, remaining_steps)
                for _ in range(yellow_steps):
                    traci.simulationStep()
                    remaining_steps -= 1

                # Now set the target green phase
                traci.trafficlight.setPhase(self.tls_id, target_green)
                self._current_green_idx = self._green_phase_indices.index(target_green)
                self._current_phase = target_green
                self._time_since_phase_start = 0.0
                phase_changed = True
            else:
                # No yellow mapping exists - direct switch (shouldn't happen with proper config)
                traci.trafficlight.setPhase(self.tls_id, target_green)
                self._current_green_idx = self._green_phase_indices.index(target_green)
                self._current_phase = target_green
                self._time_since_phase_start = 0.0
                phase_changed = True

        # Step remaining simulation time in current green phase
        # Accumulate arrived/departed across sub-steps
        step_arrived = 0
        step_departed = 0
        for _ in range(remaining_steps):
            traci.simulationStep()
            self._time_since_phase_start += 1.0
            step_arrived += traci.simulation.getArrivedNumber()
            step_departed += traci.simulation.getDepartedNumber()

        # Get current simulation time
        sim_time = traci.simulation.getTime()

        # Compute observation and reward
        obs = self._get_observation(traci)
        reward = self._get_reward(traci)

        # Check termination
        terminated = sim_time >= self._end_time
        if not terminated:
            # Also check if no more vehicles expected
            min_expected = traci.simulation.getMinExpectedNumber()
            terminated = min_expected == 0 and sim_time > 100  # Grace period

        truncated = False

        # Build info dict with green phase tracking
        info = {
            "sim_time": sim_time,
            "phase": self._current_phase,
            "current_green": self._green_phase_indices[self._current_green_idx],
            "current_green_action": self._current_green_idx,
            "target_green": target_green,
            "phase_changed": phase_changed,
            "yellow_used": yellow_used,
            "time_since_phase_start": self._time_since_phase_start,
            "halting": obs.tolist(),
            "total_wait": -reward,  # Positive waiting time for logging
            "arrived": step_arrived,
            "departed": step_departed,
        }

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the SUMO simulation."""
        if self._traci is not None and self._connection_label is not None:
            close_sumo(label=self._connection_label)
            self._traci = None
            self._connection_label = None
