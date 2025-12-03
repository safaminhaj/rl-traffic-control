from stable_baselines3 import PPO

from cityflow_sb3_env import CityFlowSB3Env


def main():
    # Create env (reuses your SB3 wrapper)
    env = CityFlowSB3Env(
        config_path="cityflow_scenario/config.json",
        intersection_id="intersection_1_1",
        action_duration=10,
        max_episode_steps=300,
    )

    # PPO model with reasonable hyperparameters for this kind of task
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,           # rollouts per update (can be tuned)
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # small exploration bonus
        verbose=1,
        # tensorboard_log=None,  # add a path if you later want TB logs
    )

    # Number of environment steps (each step = 10 sim seconds in your env)
    total_timesteps = 200_000

    model.learn(total_timesteps=total_timesteps)

    # Save trained PPO model
    model.save("sb3_ppo_cityflow")

    env.close()
    print("SB3 PPO training finished, model saved as sb3_ppo_cityflow.zip")


if __name__ == "__main__":
    main()
