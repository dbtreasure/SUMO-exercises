import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

from lunar_lander_dqn_agent import DQNAgent, TrainingLogger


def load_config(config_path: str | None = None) -> dict:
    """Load config from YAML file, defaulting to configs/default.yaml"""
    default_path = Path(__file__).parent / "configs" / "default.yaml"
    path = Path(config_path) if config_path else default_path
    with open(path) as f:
        return yaml.safe_load(f)


def format_time(seconds: float) -> str:
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def evaluate(agent: DQNAgent, n_episodes: int = 10) -> tuple[float, float]:
    """Run evaluation episodes with no exploration."""
    eval_env = gym.make("LunarLander-v2")
    rewards = []
    original_epsilon = agent.epsilon
    agent.epsilon = 0

    for _ in range(n_episodes):
        state, _ = eval_env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
        rewards.append(total_reward)

    eval_env.close()
    agent.epsilon = original_epsilon
    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    config_path = Path(args.config).resolve() if args.config else Path(__file__).parent / "configs" / "default.yaml"
    print(f"Config: {config_path}")
    print(f"  checkpoint_dir: {config['checkpoint_dir']}")
    print(f"  learning_starts: {config['learning_starts']:,}")
    print(f"  train_freq: {config['train_freq']}")
    print(f"  gradient_steps: {config['gradient_steps']}")
    print(f"  total_steps: {config['total_steps']:,}")
    print(f"  loss_type: {config.get('loss_type', 'mse')}")
    print(f"  huber_delta: {config.get('huber_delta', 1.0)}")
    print(f"  learning_rate: {config['learning_rate']}")
    print(f"  max_grad_norm: {config['max_grad_norm']}")

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("LunarLander-v2")
    agent = DQNAgent(
        env=env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        max_grad_norm=config["max_grad_norm"],
        loss_type=config.get("loss_type", "mse"),
        huber_delta=config.get("huber_delta", 1.0),
    )
    logger = TrainingLogger(window_size=100)

    total_steps = 0
    updates_performed = 0
    episode = 0
    best_mean_reward = -float("inf")
    start_time = time.time()

    print("Starting training...", flush=True)
    print(f"Target: {config['total_steps']:,} steps", flush=True)
    print("-" * 80, flush=True)

    while total_steps < config["total_steps"]:
        state, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_losses: list[float] = []
        episode_grad_norms: list[float] = []
        episode_clips: list[bool] = []
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            done_flag = terminated or truncated
            agent.replay_buffer.push(
                state, action, float(reward), next_state, done_flag
            )

            # Only update if: past warmup, on correct cadence, buffer has enough samples
            if (
                total_steps >= config["learning_starts"]
                and total_steps % config["train_freq"] == 0
                and len(agent.replay_buffer) >= config["batch_size"]
            ):
                step_losses = []
                updates_this_event = 0
                for _ in range(config["gradient_steps"]):
                    result = agent.update()
                    if result is not None:
                        step_losses.append(result.loss)
                        episode_grad_norms.append(result.grad_norm_preclip)
                        episode_clips.append(result.grad_clipped)
                        updates_this_event += 1
                updates_performed += updates_this_event
                if step_losses:
                    episode_losses.append(sum(step_losses) / len(step_losses))

            state = next_state
            episode_reward += float(reward)
            episode_length += 1
            total_steps += 1
            done = terminated or truncated

            # Hard target network update (original DQN paper approach)
            if total_steps % config["target_update_every"] == 0:
                agent.sync_target_network()

            # Check for periodic actions mid-episode
            if total_steps % config["eval_every"] == 0:
                mean_r, std_r = evaluate(agent, n_episodes=10)
                updates_per_10k = (updates_performed / total_steps) * 10_000
                q_stats = agent.get_q_stats()
                q_str = f"Q(mean/max): {q_stats[0]:.1f}/{q_stats[1]:.1f}" if q_stats else "Q: N/A"
                print(
                    f"\n>>> EVAL @ {total_steps:,} steps: {mean_r:.1f} +/- {std_r:.1f} | "
                    f"Updates: {updates_performed:,} ({updates_per_10k:.0f}/10k) | {q_str}",
                    flush=True,
                )
                if mean_r > best_mean_reward:
                    best_mean_reward = mean_r
                    agent.save_checkpoint(
                        str(checkpoint_dir / "best.pt"), episode, total_steps
                    )
                    print("    New best! Saved checkpoint.", flush=True)

            if total_steps % config["checkpoint_every"] == 0:
                agent.save_checkpoint(
                    str(checkpoint_dir / f"step_{total_steps}.pt"), episode, total_steps
                )

            if total_steps >= config["total_steps"]:
                break

        # End of episode
        agent.decay_epsilon()
        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        avg_grad_norm = (
            float(np.mean(episode_grad_norms)) if episode_grad_norms else 0.0
        )
        clip_fraction = (
            sum(episode_clips) / len(episode_clips) if episode_clips else 0.0
        )
        logger.log_episode(
            episode_reward,
            episode_length,
            avg_loss,
            avg_grad_norm,
            clip_fraction,
        )
        episode += 1

        # Periodic logging
        if episode % config["log_every"] == 0:
            stats = logger.get_stats()
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
            remaining_steps = config["total_steps"] - total_steps
            eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            progress = (total_steps / config["total_steps"]) * 100

            print(
                f"Ep {episode:4d} | {progress:5.1f}% | Steps: {total_steps:7,} | "
                f"Reward: {episode_reward:7.1f} | "
                f"Mean(100): {stats['mean_reward']:6.1f} +/- {stats['std_reward']:5.1f} | "
                f"Loss: {stats['mean_loss']:.4f} | "
                f"GradNorm: {stats['mean_grad_norm']:.2f} | "
                f"Clip: {stats['clip_fraction']:.1%} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"ETA: {format_time(eta)}",
                flush=True,
            )

    env.close()

    # Save final checkpoint
    agent.save_checkpoint(str(checkpoint_dir / "final.pt"), episode, total_steps)
    total_time = time.time() - start_time
    print("-" * 80)
    print("Training complete!")
    print(f"Total episodes: {episode}")
    print(f"Total steps: {total_steps:,}")
    print(f"Total time: {format_time(total_time)}")
    print(f"Avg steps/sec: {total_steps / total_time:.1f}")

    # Final evaluation
    print("\nFinal evaluation (20 episodes)...")
    mean_reward, std_reward = evaluate(agent, n_episodes=20)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Score (mean - std): {mean_reward - std_reward:.2f}")

    if mean_reward - std_reward >= 200:
        print("PASSED! Score >= 200")
    else:
        print("Not yet passing. Consider more training or tuning.")

    # Visual evaluation
    print("\nVisual evaluation (5 episodes)...")
    agent.epsilon = 0
    eval_env = gym.make("LunarLander-v2", render_mode="human")

    for ep in range(5):
        state, _ = eval_env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        print(f"  Episode {ep + 1}: {total_reward:.1f}")

    eval_env.close()


if __name__ == "__main__":
    main()
