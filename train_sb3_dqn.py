from stable_baselines3 import DQN

from cityflow_sb3_env import CityFlowSB3Env


def main():
    # Create a single instance of the environment
    env = CityFlowSB3Env(
        config_path="cityflow_scenario/config.json",
        intersection_id="intersection_1_1",
        action_duration=10,
        max_episode_steps=300,
    )

    # Create DQN model with some reasonable hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=2_000,
        train_freq=1,
        gradient_steps=1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.3,  # fraction of total timesteps over which epsilon decays
        verbose=1
    )

    # Number of env steps (each step = one action → 10 sim seconds in our env)
    total_timesteps = 200_000

    model.learn(total_timesteps=total_timesteps)

    # Save the trained model
    model.save("sb3_dqn_cityflow")

    env.close()
    print("SB3 DQN training finished, model saved as sb3_dqn_cityflow.zip")


if __name__ == "__main__":
    main()
