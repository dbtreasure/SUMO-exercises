"""Baseline agents for sanity checking RL performance.

These simple agents help verify that trained models actually learn
something useful and aren't just exploiting reward bugs.
"""

from __future__ import annotations

import numpy as np


class RandomAgent:
    """Agent that selects random phases uniformly.

    This is the simplest baseline - if a trained agent can't beat this,
    something is fundamentally wrong.
    """

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        """Initialize random agent.

        Args:
            n_actions: Number of possible actions (phases).
            seed: Random seed for reproducibility.
        """
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        """Select a random action.

        Args:
            observation: Current observation (ignored).
            deterministic: Ignored for random agent.

        Returns:
            Tuple of (action array, None) to match SB3 interface.
        """
        action = self.rng.integers(0, self.n_actions)
        return np.array([action]), None


class FixedCycleAgent:
    """Agent that cycles through phases in fixed order.

    Simulates a simple pre-timed signal that doesn't adapt to traffic.
    Common baseline for traffic signal control.
    """

    def __init__(
        self,
        n_actions: int,
        cycle_phases: list[int] | None = None,
        steps_per_phase: int = 1,
    ) -> None:
        """Initialize fixed cycle agent.

        Args:
            n_actions: Number of possible actions (phases).
            cycle_phases: Ordered list of phases to cycle through.
                         If None, cycles through all phases [0, 1, ..., n-1].
            steps_per_phase: Number of steps to hold each phase before switching.
        """
        self.n_actions = n_actions
        self.cycle_phases = cycle_phases if cycle_phases is not None else list(range(n_actions))
        self.steps_per_phase = steps_per_phase
        self._step_count = 0
        self._phase_index = 0

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        """Select the next phase in the cycle.

        Args:
            observation: Current observation (ignored).
            deterministic: Ignored for fixed cycle.

        Returns:
            Tuple of (action array, None) to match SB3 interface.
        """
        action = self.cycle_phases[self._phase_index]

        self._step_count += 1
        if self._step_count >= self.steps_per_phase:
            self._step_count = 0
            self._phase_index = (self._phase_index + 1) % len(self.cycle_phases)

        return np.array([action]), None

    def reset(self) -> None:
        """Reset cycle to beginning."""
        self._step_count = 0
        self._phase_index = 0


class MaxQueueAgent:
    """Agent that always gives green to the lane with the longest queue.

    A simple greedy heuristic that should perform reasonably well.
    If a trained agent can't beat this, the learning may not be effective.
    """

    def __init__(
        self, n_actions: int, lane_to_green_action_map: dict[int, int] | None = None
    ) -> None:
        """Initialize max queue agent.

        Args:
            n_actions: Number of possible green actions (not total SUMO phases).
            lane_to_green_action_map: Maps lane index to green action that serves it.
                If None, assumes lane i maps to action i % n_actions.
        """
        self.n_actions = n_actions
        self.lane_to_green_action_map = lane_to_green_action_map

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        """Select green action for lane with longest queue.

        Args:
            observation: Queue lengths per lane.
            deterministic: Ignored.

        Returns:
            Tuple of (action array, None) to match SB3 interface.
        """
        max_queue_lane = int(np.argmax(observation))

        if self.lane_to_green_action_map is not None:
            action = self.lane_to_green_action_map.get(max_queue_lane, 0)
        else:
            action = max_queue_lane % self.n_actions

        return np.array([action]), None
