from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training(log_dir: Path) -> None:
    """Generate training plot from metrics CSV.

    Reads metrics.csv from log_dir, plots eval return vs step,
    and saves to plot.png. Falls back to training returns if no eval data.
    """
    csv_path = log_dir / "metrics.csv"
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Try eval data first, fall back to training returns
    if "eval/mean_return" in df.columns:
        eval_df = df[df["eval/mean_return"].notna()]
        if not eval_df.empty:
            steps = eval_df["step"]
            mean_ret = eval_df["eval/mean_return"]
            std_ret = eval_df["eval/std_return"]
            ax.plot(steps, mean_ret, linewidth=2, label="Eval return")
            ax.fill_between(steps, mean_ret - std_ret, mean_ret + std_ret, alpha=0.3)

    # Also plot training returns if available
    if "train/episode_return" in df.columns:
        train_df = df[df["train/episode_return"].notna()]
        if not train_df.empty:
            ax.plot(
                train_df["step"],
                train_df["train/episode_return"],
                alpha=0.3,
                linewidth=1,
                label="Episode return",
            )

    ax.axhline(y=200, color="green", linestyle="--", label="Solved (200)")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Return")
    ax.set_title("Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(log_dir / "plot.png", dpi=150, bbox_inches="tight")
    plt.close()
