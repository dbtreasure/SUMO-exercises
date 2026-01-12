import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from lunar_lander_dqn_agent import DQNAgent, TrainingLogger


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


# Training config
TOTAL_STEPS = 5_000_000
EVAL_EVERY = 25_000
LOG_EVERY = 50
CHECKPOINT_EVERY = 100_000
TARGET_UPDATE_EVERY = 10_000  # Hard target network update (original DQN paper)
CHECKPOINT_DIR = Path("checkpoints")


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
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    env = gym.make("LunarLander-v2")
    agent = DQNAgent(env=env)
    logger = TrainingLogger(window_size=100)

    total_steps = 0
    episode = 0
    best_mean_reward = -float("inf")
    start_time = time.time()

    print("Starting training...", flush=True)
    print(f"Target: {TOTAL_STEPS:,} steps", flush=True)
    print("-" * 80, flush=True)

    while total_steps < TOTAL_STEPS:
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

            agent.replay_buffer.push(
                state, action, float(reward), next_state, terminated
            )

            result = agent.update()
            if result is not None:
                episode_losses.append(result.loss)
                episode_grad_norms.append(result.grad_norm_preclip)
                episode_clips.append(result.grad_clipped)

            state = next_state
            episode_reward += float(reward)
            episode_length += 1
            total_steps += 1
            done = terminated or truncated

            # Hard target network update (original DQN paper approach)
            if total_steps % TARGET_UPDATE_EVERY == 0:
                agent.sync_target_network()

            # Check for periodic actions mid-episode
            if total_steps % EVAL_EVERY == 0:
                mean_r, std_r = evaluate(agent, n_episodes=10)
                print(
                    f"\n>>> EVAL @ {total_steps:,} steps: {mean_r:.1f} +/- {std_r:.1f}",
                    flush=True,
                )
                if mean_r > best_mean_reward:
                    best_mean_reward = mean_r
                    agent.save_checkpoint(
                        str(CHECKPOINT_DIR / "best.pt"), episode, total_steps
                    )
                    print("    New best! Saved checkpoint.", flush=True)

            if total_steps % CHECKPOINT_EVERY == 0:
                agent.save_checkpoint(
                    str(CHECKPOINT_DIR / f"step_{total_steps}.pt"), episode, total_steps
                )

            if total_steps >= TOTAL_STEPS:
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
        if episode % LOG_EVERY == 0:
            stats = logger.get_stats()
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
            remaining_steps = TOTAL_STEPS - total_steps
            eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            progress = (total_steps / TOTAL_STEPS) * 100

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
    agent.save_checkpoint(str(CHECKPOINT_DIR / "final.pt"), episode, total_steps)
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
