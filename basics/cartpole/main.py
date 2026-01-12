import gymnasium as gym
import numpy as np

from cart_pole_agent import CartPoleAgent

# Training config
N_EPISODES = 30000
RENDER_EVERY = 100

env = gym.make("CartPole-v1")

agent = CartPoleAgent(
    env=env,
    learning_rate=0.1,
    initial_epsilon=1.0,
    epsilon_decay=0.00005,  # Decay over ~20000 episodes
    final_epsilon=0.01,
)

for episode in range(N_EPISODES):
    observation, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        agent.update(observation, action, float(reward), terminated, next_observation)
        observation = next_observation
        total_reward += float(reward)

    agent.decay_epsilon()

    if episode % RENDER_EVERY == 0:
        print(f"Episode {episode}: reward={total_reward}, epsilon={agent.epsilon:.3f}")

env.close()
print("Training complete!")

# Evaluation - watch the trained agent
print("\nEvaluating trained agent...")
eval_env = gym.make("CartPole-v1", render_mode="human")

for ep in range(5):
    obs, info = eval_env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Pure exploitation - no random actions
        state = agent.discretize(obs)
        action = int(np.argmax(agent.q_values[state]))
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += float(reward)

    print(f"Eval episode {ep}: reward={total_reward}")

eval_env.close()
