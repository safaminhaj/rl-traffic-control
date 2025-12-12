# # train_sb3_multi_ppo.py

# from stable_baselines3 import PPO
# from cityflow_multi_sb3_env import CityFlowMultiSB3Env

# def main():
#     # control 4 intersections, for example
#     intersection_ids = [
#         "intersection_1_1",
#         "intersection_1_0",
#     ]

#     env = CityFlowMultiSB3Env(
#         config_path="cityflow_scenario/config.json",
#         intersection_ids=intersection_ids,
#         action_duration=10,
#         max_episode_steps=300,
#     )

#     model = PPO(
#         "MlpPolicy",
#         env,
#         learning_rate=3e-4,
#         n_steps=512,
#         batch_size=64,
#         n_epochs=10,
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         ent_coef=0.01,
#         verbose=1,
#     )

#     model.learn(total_timesteps=200_000)
#     model.save("sb3_ppo_cityflow_multi")
#     env.close()

# if __name__ == "__main__":
#     main()
