import csv
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Dual logging to TensorBoard and CSV."""

    def __init__(self, log_dir: Path, csv_columns: list[str] | None = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.tb_writer = SummaryWriter(self.log_dir)

        # CSV - can be initialized with known columns, or lazily on first log
        self.csv_path = self.log_dir / "metrics.csv"
        self._csv_file: Any = None
        self._csv_writer: Any = None
        self._csv_columns: list[str] | None = None

        if csv_columns is not None:
            self._init_csv(["step"] + csv_columns)

    def _init_csv(self, columns: list[str]) -> None:
        """Initialize CSV with given columns."""
        self._csv_columns = columns
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=self._csv_columns, extrasaction="ignore"
        )
        self._csv_writer.writeheader()

    def log(self, metrics: dict[str, float | None], step: int) -> None:
        """Log metrics to TensorBoard and CSV.

        Args:
            metrics: Dict of metric names to values. Use None for missing values.
            step: Training step number.
        """
        # TensorBoard: each metric as a scalar (skip None values)
        for key, value in metrics.items():
            if value is not None:
                self.tb_writer.add_scalar(key, value, step)

        # CSV: lazy init on first call if not pre-initialized
        if self._csv_writer is None:
            self._init_csv(["step"] + list(metrics.keys()))

        # Build row with all columns, None for missing
        row: dict[str, Any] = {"step": step}
        assert self._csv_columns is not None
        for col in self._csv_columns:
            if col != "step":
                row[col] = metrics.get(col)

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        """Close file handles."""
        self.tb_writer.close()
        if self._csv_file:
            self._csv_file.close()
