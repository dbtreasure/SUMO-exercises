"""Plot comparison of training runs from log files."""

import re
from pathlib import Path

import matplotlib.pyplot as plt

# Parse eval results from log files
def parse_evals(log_content: str) -> tuple[list[int], list[float], list[float]]:
    """Extract steps, mean rewards, and std from EVAL lines."""
    pattern = r">>> EVAL @ ([\d,]+) steps: ([-\d.]+) \+/- ([\d.]+)"
    steps = []
    means = []
    stds = []
    for match in re.finditer(pattern, log_content):
        step = int(match.group(1).replace(",", ""))
        mean = float(match.group(2))
        std = float(match.group(3))
        steps.append(step)
        means.append(mean)
        stds.append(std)
    return steps, means, stds


# Read log files
warmup_log = Path("/var/folders/jf/vj8kqlw909qdl292y5bs9f7h0000gn/T/claude/-Users-dan-Development-SUMO-exercises-basics/tasks/bae66e4.output").read_text()
baseline_log = Path("/var/folders/jf/vj8kqlw909qdl292y5bs9f7h0000gn/T/claude/-Users-dan-Development-SUMO-exercises-basics/tasks/b783eaa.output").read_text()
balanced_log = Path("/var/folders/jf/vj8kqlw909qdl292y5bs9f7h0000gn/T/claude/-Users-dan-Development-SUMO-exercises-basics/tasks/b3c5c5b.output").read_text()

warmup_only_log = Path("/var/folders/jf/vj8kqlw909qdl292y5bs9f7h0000gn/T/claude/-Users-dan-Development-SUMO-exercises-basics/tasks/bc947e4.output").read_text()
grad_steps_log = Path("/var/folders/jf/vj8kqlw909qdl292y5bs9f7h0000gn/T/claude/-Users-dan-Development-SUMO-exercises-basics/tasks/b7d98f0.output").read_text()

warmup_steps, warmup_means, warmup_stds = parse_evals(warmup_log)
baseline_steps, baseline_means, baseline_stds = parse_evals(baseline_log)
balanced_steps, balanced_means, balanced_stds = parse_evals(balanced_log)
warmup_only_steps, warmup_only_means, warmup_only_stds = parse_evals(warmup_only_log)
grad_steps_steps, grad_steps_means, grad_steps_stds = parse_evals(grad_steps_log)

# Create plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot with error bands
ax.plot(warmup_steps, warmup_means, label="Warmup (train_freq=4)", color="tab:blue", linewidth=2)
ax.fill_between(warmup_steps,
                [m - s for m, s in zip(warmup_means, warmup_stds)],
                [m + s for m, s in zip(warmup_means, warmup_stds)],
                alpha=0.2, color="tab:blue")

ax.plot(baseline_steps, baseline_means, label="Baseline (train_freq=1)", color="tab:orange", linewidth=2)
ax.fill_between(baseline_steps,
                [m - s for m, s in zip(baseline_means, baseline_stds)],
                [m + s for m, s in zip(baseline_means, baseline_stds)],
                alpha=0.2, color="tab:orange")

ax.plot(balanced_steps, balanced_means, label="Balanced (train_freq=2)", color="tab:green", linewidth=2)
ax.fill_between(balanced_steps,
                [m - s for m, s in zip(balanced_means, balanced_stds)],
                [m + s for m, s in zip(balanced_means, balanced_stds)],
                alpha=0.2, color="tab:green")

ax.plot(warmup_only_steps, warmup_only_means, label="Warmup only (20k wait)", color="tab:purple", linewidth=2)
ax.fill_between(warmup_only_steps,
                [m - s for m, s in zip(warmup_only_means, warmup_only_stds)],
                [m + s for m, s in zip(warmup_only_means, warmup_only_stds)],
                alpha=0.2, color="tab:purple")

ax.plot(grad_steps_steps, grad_steps_means, label="Grad steps (4x4 decorr)", color="tab:red", linewidth=2)
ax.fill_between(grad_steps_steps,
                [m - s for m, s in zip(grad_steps_means, grad_steps_stds)],
                [m + s for m, s in zip(grad_steps_means, grad_steps_stds)],
                alpha=0.2, color="tab:red")

# Add passing threshold line
ax.axhline(y=200, color="black", linestyle="--", linewidth=1, label="Passing threshold (200)")

ax.set_xlabel("Environment Steps", fontsize=12)
ax.set_ylabel("Eval Reward (mean +/- std)", fontsize=12)
ax.set_title("DQN Training: train_freq Comparison", fontsize=14)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1_000_000)

plt.tight_layout()
plt.savefig("training_comparison.png", dpi=150)
print("Saved: training_comparison.png")
plt.show()
