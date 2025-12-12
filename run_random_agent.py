import numpy as np
from cityflow_env import CityFlowSingleJunctionEnv
import random

"""
numpy: used to compute average reward at the end
CityFlowSingleJunctionEnv: your RL environment wrapper
random: Python's built-in RNG for choosing random actions
This agent does not learn; it just samples random actions.

"""


def main(num_episodes=20):
    env = CityFlowSingleJunctionEnv(
        config_path="cityflow_scenario/config.json",
        intersection_id="intersection_1_1",
        action_duration=10,
        max_episode_steps=300,
    )
    """
    Creates an environment identical to the other scripts:
    10-second (10-step) actions
    300 simulation steps per episode
    Therefore each episode has approximately 30 RL actions
    """

    episode_rewards = (
        []
    )  # Will store one number per episode: the total reward accumulated in that episode.

    """
    During each episode:
    env.reset() resets CityFlow and returns the initial state.
    done marks termination (when simulation time reaches 300 steps).
    total_reward accumulates rewards.
    step counts number of actions taken (not simulation steps).
    """
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


"""
Compute and print the mean reward from all episodes.
That average becomes the performance benchmark for a random agent.
The RL agent should score better than random, meaning:
rewards should be less negative
fewer cars waiting in total
"""

if __name__ == "__main__":
    main(num_episodes=20)
