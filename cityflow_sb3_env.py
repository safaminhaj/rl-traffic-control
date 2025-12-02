import gym
from gym import spaces
import numpy as np

from cityflow_env import CityFlowSingleJunctionEnv


class CityFlowSB3Env(gym.Env):
    """
    Gym-compatible wrapper around CityFlowSingleJunctionEnv
    so it can be used with Stable-Baselines3.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        config_path="cityflow_scenario/config.json",
        intersection_id="intersection_1_1",
        action_duration=10,
        max_episode_steps=300,
    ):
        super().__init__()

        # Underlying environment (your original one)
        self.env = CityFlowSingleJunctionEnv(
            config_path=config_path,
            intersection_id=intersection_id,
            action_duration=action_duration,
            max_episode_steps=max_episode_steps,
        )

        # Build observation_space from a sample state
        sample_state = self.env.reset()
        obs_dim = sample_state.shape[0]

        # Waiting vehicle counts are >= 0, upper bound is not hard-limited
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Discrete action: phase index
        self.action_space = spaces.Discrete(self.env.action_space_n)

    def reset(self):
        """
        Gym-style reset: returns initial observation.
        (No info dict for classic Gym / SB3.)
        """
        obs = self.env.reset()
        return obs

    def step(self, action):
        """
        Gym-style step: returns (obs, reward, done, info).
        """
        obs, reward, done, info = self.env.step(int(action))
        return obs, reward, done, info

    def close(self):
        self.env.close()
