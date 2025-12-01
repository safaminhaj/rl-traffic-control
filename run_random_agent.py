import numpy as np
from cityflow_env import CityFlowSingleJunctionEnv
import random


def main(num_episodes=20):
    env = CityFlowSingleJunctionEnv(
        config_path="cityflow_scenario/config.json",
        intersection_id="intersection_1_1",
        action_duration=10,
        max_episode_steps=300
    )

    episode_rewards = []

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        print(f"\n=== RANDOM Episode {ep} ===")
        while not done:
            action = random.randint(0, env.action_space_n - 1)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            step += 1

        episode_rewards.append(total_reward)
        print(f"RANDOM Episode {ep} finished, total_reward={total_reward:.2f}")

    avg_reward = float(np.mean(episode_rewards))
    print(f"\nAverage RANDOM reward over {num_episodes} episodes: {avg_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main(num_episodes=20)
