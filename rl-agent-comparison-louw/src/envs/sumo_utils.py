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

# Store the SUMO subprocess handle for reliable cleanup
_sumo_process: subprocess.Popen[bytes] | None = None


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
    label: str = "default",
    override_end_s: int | None = None,
) -> None:
    """Start a SUMO simulation via TraCI.

    Args:
        sumocfg_path: Path to .sumocfg file (relative or absolute).
        gui: If True, use sumo-gui; otherwise use headless sumo.
        seed: Random seed for reproducibility. If None, SUMO uses its default.
        label: TraCI connection label (for multiple connections).
        override_end_s: If set, override simulation end time (seconds).

    Raises:
        FileNotFoundError: If sumocfg_path doesn't exist.
        EnvironmentError: If SUMO binary cannot be found.
    """
    global _sumo_process
    traci = _get_traci()

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
        "--start",  # Start simulation immediately (for GUI)
        "--quit-on-end",  # Exit when simulation ends
        # Suppress logging for faster/quieter runs
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
        "--no-warnings",
        "true",
    ]

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    if override_end_s is not None:
        cmd.extend(["--end", str(override_end_s)])

    # Start TraCI connection and store process handle
    # traci.start returns (port, sumoProcess) tuple
    result = traci.start(cmd, label=label)
    if isinstance(result, tuple) and len(result) >= 2:
        _sumo_process = result[1]
    else:
        _sumo_process = None


def close_sumo(timeout: float = 0.5) -> None:
    """Safely close the TraCI connection and terminate SUMO subprocess.

    Args:
        timeout: Seconds to wait for graceful shutdown before force-killing.
    """
    global _sumo_process
    traci = _get_traci()

    # Try graceful TraCI close
    try:
        traci.close()
    except Exception:
        pass  # Ignore errors on close

    # Force-kill subprocess if still alive
    if _sumo_process is not None and hasattr(_sumo_process, "wait"):
        try:
            _sumo_process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _sumo_process.terminate()
            try:
                _sumo_process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                _sumo_process.kill()
        except Exception:
            pass  # Process already exited
        _sumo_process = None
