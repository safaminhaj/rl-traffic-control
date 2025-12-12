# import gym
# from gym import spaces
# import numpy as np

# from cityflow_multi_env import CityFlowMultiJunctionEnv


# class CityFlowMultiSB3Env(gym.Env):
#     """
#     Gym-compatible wrapper around CityFlowMultiJunctionEnv so we can use
#     Stable-Baselines3 (PPO, DQN, etc.) to control MULTIPLE intersections.

#     - Observation space: Box, flattened vector of lane waiting counts.
#     - Action space: MultiDiscrete, one discrete phase index per intersection.
#     """

#     metadata = {"render.modes": []}

#     def __init__(
#         self,
#         config_path="cityflow_scenario/config.json",
#         intersection_ids=None,
#         action_duration=10,
#         max_episode_steps=300,
#     ):
#         super().__init__()

#         if intersection_ids is None:
#             raise ValueError("You must pass a list of intersection_ids.")

#         # Underlying multi-junction env
#         self.env = CityFlowMultiJunctionEnv(
#             config_path=config_path,
#             intersection_ids=intersection_ids,
#             action_duration=action_duration,
#             max_episode_steps=max_episode_steps,
#         )

#         # Build observation_space from a sample state
#         sample_state = self.env.reset()
#         obs_dim = sample_state.shape[0]

#         # Waiting counts are >= 0, no strict upper bound
#         self.observation_space = spaces.Box(
#             low=0.0,
#             high=np.inf,
#             shape=(obs_dim,),
#             dtype=np.float32,
#         )

#         # MultiDiscrete: one phase index per intersection
#         nvec = self.env.action_space_nvec  # list of ints
#         self.action_space = spaces.MultiDiscrete(nvec)

#     # ------------- Gym core API ------------- #

#     def reset(self):
#         """
#         Gym-style reset: returns initial observation.
#         """
#         obs = self.env.reset()
#         return obs

#     def step(self, action):
#         """
#         Gym-style step: returns (obs, reward, done, info).
#         'action' is expected to be a sequence compatible with MultiDiscrete:
#         e.g. np.ndarray of shape (num_intersections,).
#         """
#         obs, reward, done, info = self.env.step(action)
#         return obs, reward, done, info

#     def close(self):
#         self.env.close()
