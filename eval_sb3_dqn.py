import numpy as np
from stable_baselines3 import DQN

from cityflow_sb3_env import CityFlowSB3Env


def evaluate(num_episodes=20):
    # Create env with same settings
    env = CityFlowSB3Env(
        config_path="cityflow_scenario/config.json",
        intersection_id="intersection_1_1",
        action_duration=10,
        max_episode_steps=300,
    )

    # Load trained SB3 model
    model = DQN.load("sb3_dqn_cityflow")

    episode_rewards = []

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Deterministic (greedy) policy for evaluation
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"[SB3 EVAL] Episode {ep}/{num_episodes} total_reward={total_reward:.2f}")

    avg_reward = float(np.mean(episode_rewards))
    print(f"\nAverage SB3 DQN eval reward over {num_episodes} episodes: {avg_reward:.2f}")

    env.close()


if __name__ == "__main__":
    evaluate(num_episodes=20)
