import argparse
from datetime import datetime
from pathlib import Path

from rl.agents import AGENTS
from rl.common.config import load_config
from rl.common.envs import make_env
from rl.common.eval import evaluate_agent
from rl.common.logger import Logger
from rl.common.plot import plot_training
from rl.common.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RL agent")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Create a run directory: runs/<timestamp>_<algo_name>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{timestamp}_{config.algo.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create environments
    env = make_env(config.env.id)
    eval_env = make_env(config.env.id)

    # Seed everything
    seed_everything(config.seed, env)

    # Get dimensions from environment
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Create agent
    agent_cls = AGENTS[config.algo.name]
    agent = agent_cls(obs_dim, act_dim, config)

    # Create logger with all expected columns
    csv_columns = [
        "eval/mean_return",
        "eval/std_return",
        "train/episode_return",
        "train/episode_length",
        "loss/policy",
        "loss/value",
        "loss/entropy",
        "misc/entropy",
        "misc/return_mean",
    ]
    logger = Logger(run_dir, csv_columns=csv_columns)

    # Training state
    total_steps = 0
    episode = 0

    # Determine update mode: step-based (A2C/PPO) vs episode-based (REINFORCE)
    step_based_update = config.algo.name in ["a2c", "ppo"]
    rollout_step = 0  # Counter for step-based updates

    print(f"Starting training: {config.algo.name}")
    print(f"Run directory: {run_dir}")
    print(f"Total steps: {config.train.total_steps:,}")
    print("-" * 50)

    # Main training loop
    while total_steps < config.train.total_steps:  # Keep going until we've done enough steps
        obs, _ = env.reset(seed=config.seed + episode if episode == 0 else None)
        episode_return = 0.0
        episode_length = 0
        done = False

        # Episode loop
        while not done:  # Run one episode step by step
            # Agent selects action
            action, info = agent.act(obs)  # get action and info (log_prob)

            # Environment step
            next_obs, reward, terminated, truncated, _ = env.step(
                action
            )  # get next state, reward, done
            done = terminated or truncated

            # Store transition
            agent.observe(obs, action, float(reward), next_obs, terminated, truncated, info)

            # Step-based update (A2C/PPO): update every n_steps
            if step_based_update:
                rollout_step += 1
                if rollout_step >= config.algo.n_steps:
                    # next_obs is the state after the rollout - use for bootstrapping
                    update_info = agent.update(next_obs)
                    rollout_step = 0
                    # Log update (but not too frequently)
                    if total_steps % config.train.log_every_steps < config.algo.n_steps:
                        logger.log(update_info, step=total_steps)

            # Track episode stats
            obs = next_obs
            episode_return += float(reward)
            episode_length += 1
            total_steps += 1

            # Periodic evaluation
            if total_steps % config.train.eval_every_steps == 0:
                mean_ret, std_ret = evaluate_agent(
                    agent, config.env.id, config.train.eval_episodes, seed=config.seed
                )
                print(f"Step {total_steps:,} | Eval return: {mean_ret:.1f} +/- {std_ret:.1f}")
                logger.log(
                    {"eval/mean_return": mean_ret, "eval/std_return": std_ret},
                    step=total_steps,
                )

            # Check if we've hit the step limit
            if total_steps >= config.train.total_steps:
                break

        # Episode finished
        episode += 1

        # For step-based agents: update with remaining data if episode ended mid-rollout
        if step_based_update and rollout_step > 0:
            update_info = agent.update(obs)  # obs is final state
            rollout_step = 0
        else:
            # Episode-based agents (REINFORCE): update at episode end
            update_info = agent.on_episode_end()

        # Log episode stats
        if episode % 10 == 0:  # Log every 10 episodes to reduce noise
            log_data = {
                "train/episode_return": episode_return,
                "train/episode_length": episode_length,
                **update_info,
            }
            logger.log(log_data, step=total_steps)
            print(
                f"Episode {episode} | Steps: {total_steps:,} | Return: {episode_return:.1f} | "
                f"Loss: {update_info.get('loss/policy', 0):.4f}"
            )

    # Training complete
    print("-" * 50)
    print("Training complete!")

    # Final evaluation
    mean_ret, std_ret = evaluate_agent(agent, config.env.id, n_episodes=20, seed=config.seed)
    print(f"Final eval ({20} episodes): {mean_ret:.1f} +/- {std_ret:.1f}")

    # Save final model
    agent.save(str(run_dir / "final_model.pt"))
    print(f"Model saved to {run_dir / 'final_model.pt'}")

    # Generate plot
    plot_training(run_dir)
    print(f"Plot saved to {run_dir / 'plot.png'}")

    # Cleanup
    logger.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
