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
        seed: int = 0,
    ) -> None:
        """Initialize the TSC environment.

        Args:
            sumocfg_path: Path to .sumocfg file.
            decision_interval_s: Simulation seconds per agent decision step.
            min_green_s: Minimum seconds a phase must be active before switching.
            max_sim_time_s: Maximum simulation time in seconds (default 300 for dev).
            tls_id: Traffic light ID. If None, auto-discover first TLS in network.
            seed: Random seed for SUMO simulation reproducibility.
        """
        super().__init__()

        self.sumocfg_path = str(Path(sumocfg_path).resolve())
        self.decision_interval_s = decision_interval_s
        self.min_green_s = min_green_s
        self.max_sim_time_s = max_sim_time_s
        self._tls_id_config = tls_id
        self._seed = seed

        # Will be set after reset()
        self.tls_id: str = ""
        self._controlled_lanes: list[str] = []
        self._n_phases: int = 0
        self._end_time: float = 3600.0
        self._time_since_phase_start: float = 0.0
        self._current_phase: int = 0
        self._traci = None

        # Placeholder spaces (updated after first reset)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(1)

        # Validate config file exists
        if not Path(self.sumocfg_path).exists():
            raise FileNotFoundError(f"SUMO config not found: {self.sumocfg_path}")

    def _import_traci(self):
        """Import TraCI module."""
        get_sumo_tools_path()
        import traci

        return traci

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
        """Compute reward: negative sum of waiting times.

        Paper-faithful: R_t = -sum(waiting_time(veh_i)) for vehicles on controlled lanes.

        Args:
            traci: TraCI module.

        Returns:
            Negative total waiting time (more negative = worse).
        """
        # Collect unique vehicle IDs across all controlled lanes
        vehicle_ids: set[str] = set()
        for lane in self._controlled_lanes:
            vehicle_ids.update(traci.lane.getLastStepVehicleIDs(lane))

        # Sum waiting times
        total_wait = sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
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

        # Use provided seed or fall back to init seed
        sim_seed = seed if seed is not None else self._seed

        # Start SUMO with configured end time
        traci = self._import_traci()
        start_sumo(
            self.sumocfg_path,
            gui=False,
            seed=sim_seed,
            override_end_s=self.max_sim_time_s,
        )
        self._traci = traci

        # Discover TLS and lanes
        self.tls_id = self._discover_tls(traci)
        self._controlled_lanes = self._get_controlled_lanes(traci)
        self._n_phases = self._get_n_phases(traci)

        # Use configured end time
        self._end_time = float(self.max_sim_time_s)

        # Update spaces now that we know dimensions
        n_lanes = len(self._controlled_lanes)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(n_lanes,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self._n_phases)

        # Initialize phase tracking
        self._current_phase = traci.trafficlight.getPhase(self.tls_id)
        self._time_since_phase_start = 0.0

        # Get initial observation
        obs = self._get_observation(traci)

        info = {
            "sim_time": traci.simulation.getTime(),
            "tls_id": self.tls_id,
            "controlled_lanes": self._controlled_lanes,
            "n_phases": self._n_phases,
            "phase": self._current_phase,
        }

        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one agent step in the environment.

        Args:
            action: Phase index to switch to (0 to n_phases-1).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if self._traci is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        traci = self._traci

        # Check if phase change is allowed (min green constraint)
        phase_changed = False
        if self._time_since_phase_start >= self.min_green_s:
            if action != self._current_phase:
                traci.trafficlight.setPhase(self.tls_id, action)
                self._current_phase = action
                self._time_since_phase_start = 0.0
                phase_changed = True

        # Advance simulation by decision_interval_s (1-second steps)
        steps = int(self.decision_interval_s)
        for _ in range(steps):
            traci.simulationStep()
            self._time_since_phase_start += 1.0

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

        # Build info dict
        info = {
            "sim_time": sim_time,
            "phase": self._current_phase,
            "phase_changed": phase_changed,
            "time_since_phase_start": self._time_since_phase_start,
            "halting": obs.tolist(),
            "total_wait": -reward,  # Positive waiting time for logging
        }

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the SUMO simulation."""
        if self._traci is not None:
            close_sumo()
            self._traci = None
