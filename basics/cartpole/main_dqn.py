import gymnasium as gym
import numpy as np

from cart_pole_agent_dqn import DQNAgent

# Training Config
N_EPISODES = 1000
SYNC_EVERY = 10  # Sync target network every N episodes

env = gym.make("CartPole-v1")

agent = DQNAgent(
    env=env,
    learning_rate=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    buffer_size=10000,
    batch_size=64,
)

for episode in range(N_EPISODES):
    state, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        # Store experience
        agent.replay_buffer.push(state, action, float(reward), next_state, terminated)

        # Train on a batch
        agent.update()

        state = next_state
        total_reward += float(reward)

    agent.decay_epsilon()

    # Sync target network periodically
    if episode % SYNC_EVERY == 0:
        agent.sync_target_network()
        print(
            f"Episode {episode}: Total Reward = {total_reward}, epsilon = {agent.epsilon:.3f}"
        )

env.close()
print("Training complete!")

# Evaluation - watch the trained agent
print("\nEvaluating trained agent...")
agent.epsilon = 0  # No random actions during evaluation
eval_env = gym.make("CartPole-v1", render_mode="human")

for ep in range(5):
    state, info = eval_env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.get_action(state)
        state, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += float(reward)

    print(f"Eval episode {ep}: reward={total_reward}")

eval_env.close()
