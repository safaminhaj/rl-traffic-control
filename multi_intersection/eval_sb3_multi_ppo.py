import numpy as np
from stable_baselines3 import PPO

from cityflow_multi_sb3_env import CityFlowMultiSB3Env


def evaluate_multi_ppo(num_episodes=20):
    """
    Evaluate a trained multi-intersection PPO agent on the CityFlow simulator.

    - Uses the same environment configuration as training.
    - Loads the model saved as 'sb3_ppo_cityflow_multi.zip'.
    - Runs num_episodes episodes with a greedy (deterministic) policy.
    - Prints per-episode total reward and the average over all episodes.
    """

    # IMPORTANT: this must match the intersection_ids used in training
    intersection_ids = [
        "intersection_1_1",  # central 4-way intersection
        "intersection_1_0",  # extra signalized intersection to the south
    ]

    # Create the multi-junction environment
    env = CityFlowMultiSB3Env(
        config_path="cityflow_scenario/config.json",
        intersection_ids=intersection_ids,
        action_duration=10,
        max_episode_steps=300,
    )

    # Load the trained PPO model
    model = PPO.load("sb3_ppo_cityflow_multi")

    episode_rewards = []

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Deterministic (greedy) policy for evaluation
            # action will be a MultiDiscrete vector:
            # e.g. [phase_for_intersection_1_1, phase_for_intersection_1_0]
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)
        print(
            f"[SB3 MULTI-PPO EVAL] Episode {ep}/{num_episodes} "
            f"total_reward={total_reward:.2f}"
        )

    avg_reward = float(np.mean(episode_rewards))
    print(
        f"\nAverage SB3 multi-intersection PPO reward over "
        f"{num_episodes} episodes: {avg_reward:.2f}"
    )

    env.close()


if __name__ == "__main__":
    evaluate_multi_ppo(num_episodes=20)
