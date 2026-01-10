"""SUMO/TraCI utility functions for starting and stopping simulations.

This module provides helpers to:
- Locate SUMO_HOME and import TraCI
- Start SUMO with deterministic seeding
- Safely close TraCI connections
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Store the SUMO subprocess handles for reliable cleanup (keyed by label)
_sumo_processes: dict[str, subprocess.Popen[bytes] | None] = {}

# Counter for generating unique labels
_label_counter = 0


def generate_unique_label() -> str:
    """Generate a unique TraCI connection label."""
    global _label_counter
    _label_counter += 1
    return f"env_{_label_counter}"


def get_sumo_tools_path() -> str:
    """Get the path to SUMO tools directory and ensure TraCI is importable.

    Returns:
        Path to SUMO_HOME/tools directory.

    Raises:
        EnvironmentError: If SUMO_HOME is not set or tools directory doesn't exist.
    """
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise EnvironmentError(
            "SUMO_HOME environment variable is not set. "
            "Please set it to your SUMO installation directory, e.g.:\n"
            '  export SUMO_HOME="/opt/homebrew/share/sumo"'
        )

    tools_path = Path(sumo_home) / "tools"
    if not tools_path.exists():
        raise EnvironmentError(
            f"SUMO tools directory not found at {tools_path}. "
            f"Check that SUMO_HOME={sumo_home} is correct."
        )

    # Add to sys.path if not already there
    tools_str = str(tools_path)
    if tools_str not in sys.path:
        sys.path.insert(0, tools_str)

    return tools_str


def _get_traci():
    """Import and return the traci module.

    Raises:
        ImportError: If TraCI cannot be imported.
    """
    get_sumo_tools_path()  # Ensure path is set up
    try:
        import traci

        return traci
    except ImportError as e:
        raise ImportError(
            f"Failed to import TraCI: {e}. "
            "Ensure SUMO is installed and SUMO_HOME is set correctly."
        ) from e


def start_sumo(
    sumocfg_path: str,
    gui: bool = False,
    seed: int | None = None,
    label: str | None = None,
    override_end_s: int | None = None,
    auto_start: bool = True,
    gui_delay: int | None = None,
) -> str:
    """Start a SUMO simulation via TraCI.

    Args:
        sumocfg_path: Path to .sumocfg file (relative or absolute).
        gui: If True, use sumo-gui; otherwise use headless sumo.
        seed: Random seed for reproducibility. If None, SUMO uses its default.
        label: TraCI connection label. If None, generates unique label.
        override_end_s: If set, override simulation end time (seconds).
        auto_start: If True, simulation starts immediately. If False (GUI mode),
                   user controls play/pause/step manually.
        gui_delay: Delay between simulation steps in ms (GUI mode only).
                  Higher = slower. 0 = max speed. 100-500 good for watching.

    Returns:
        The connection label used.

    Raises:
        FileNotFoundError: If sumocfg_path doesn't exist.
        EnvironmentError: If SUMO binary cannot be found.
    """
    global _sumo_processes
    traci = _get_traci()

    # Generate unique label if not provided
    if label is None:
        label = generate_unique_label()

    # Validate config file exists
    cfg_path = Path(sumocfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"SUMO config file not found: {sumocfg_path}")

    # Determine SUMO binary
    sumo_binary = "sumo-gui" if gui else "sumo"

    # Build command
    cmd = [
        sumo_binary,
        "-c",
        str(cfg_path.resolve()),
        "--quit-on-end",  # Exit when simulation ends
        # Suppress logging for faster/quieter runs
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
        "--no-warnings",
        "true",
    ]

    if auto_start:
        cmd.append("--start")  # Start simulation immediately

    if gui and gui_delay is not None:
        cmd.extend(["--delay", str(gui_delay)])

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    if override_end_s is not None:
        cmd.extend(["--end", str(override_end_s)])

    # Start TraCI connection and store process handle
    # traci.start returns (port, sumoProcess) tuple
    result = traci.start(cmd, label=label)
    if isinstance(result, tuple) and len(result) >= 2:
        _sumo_processes[label] = result[1]
    else:
        _sumo_processes[label] = None

    return label


def close_sumo(label: str | None = None, timeout: float = 0.5) -> None:
    """Safely close the TraCI connection and terminate SUMO subprocess.

    Args:
        label: TraCI connection label to close. If None, closes current connection.
        timeout: Seconds to wait for graceful shutdown before force-killing.
    """
    global _sumo_processes
    traci = _get_traci()

    # Try graceful TraCI close
    try:
        if label is not None:
            traci.switch(label)
        traci.close()
    except Exception:
        pass  # Ignore errors on close

    # Force-kill subprocess if still alive
    if label is not None and label in _sumo_processes:
        proc = _sumo_processes.get(label)
        if proc is not None and hasattr(proc, "wait"):
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass  # Process already exited
        del _sumo_processes[label]
