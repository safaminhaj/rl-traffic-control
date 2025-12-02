import os
import json
import numpy as np
import cityflow


class CityFlowSingleJunctionEnv:
    """
    Very simple RL wrapper around CityFlow for ONE intersection.
    Conceptually it’s similar to a Gym-like env:
    - Action: integer phase_id (0, 1, 2, ...).
    - State: vector of lane waiting vehicle counts.
    - Reward: negative total waiting vehicles (we want to MINIMIZE waiting).
    Action: choose which traffic light phase is active (integer index).
    State: a numeric vector describing how many vehicles are waiting on each lane.
    Reward: negative total number of waiting vehicles (so maximizing reward means reducing waiting).
    """

    def __init__(
        self,
        config_path="cityflow_scenario/config.json",
        intersection_id="intersection_1_1",
        action_duration=10,  # how many sim seconds each action lasts
        max_episode_steps=300,  # episode length in seconds
    ):
        self.config_path = config_path
        self.intersection_id = intersection_id
        self.action_duration = int(action_duration)
        self.max_episode_steps = int(max_episode_steps)

        """
        config_path: path to CityFlow config JSON.
        intersection_id: which intersection in the road network to control.
        action_duration: how long one chosen phase stays active before the agent can act again (in simulation steps, assumed to represent seconds).
        max_episode_steps: how long the episode lasts (in simulation steps / seconds).
        """

        # Create engine
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.eng = cityflow.Engine(self.config_path, thread_num=1)
        # Uses our config file.
        # thread_num=1 means one simulation thread (simpler and deterministic for RL).

        # Load scenario info from config/roadnet
        with open(self.config_path) as f:
            cfg = json.load(f)

        self.interval = float(cfg.get("interval", 1.0))
        roadnet_path = os.path.join(
            os.path.dirname(self.config_path), cfg["roadnetFile"]
        )
        with open(roadnet_path) as f:
            roadnet = json.load(f)

        # Find number of phases for our intersection
        self.num_phases = self._get_num_phases(roadnet, self.intersection_id)
        if self.num_phases == 0:
            raise ValueError(
                f"No traffic light phases found for {self.intersection_id}"
            )

        # We'll use a fixed, sorted lane order for the state vector
        # (all lanes in the network for now – fine for a small toy grid)
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        self.lanes = sorted(lane_waiting.keys())

        self.current_step = 0

    @staticmethod
    # This searches the road network JSON to find the intersection with id == intersection_id
    # Otherwise, counts the number of entries in lightphases – that’s the number of discrete phases the agent can choose from.
    def _get_num_phases(roadnet, intersection_id):
        for inter in roadnet["intersections"]:
            if inter["id"] == intersection_id:
                tl = inter.get("trafficLight")
                if tl is None:
                    return 0
                return len(tl.get("lightphases", []))
        return 0

    # ------------ Core RL API ------------ #

    def reset(self):
        """Reset simulation and return initial state."""
        self.eng.reset()
        self.current_step = 0

        # Optional: set an initial phase (0)
        self.eng.set_tl_phase(self.intersection_id, 0)

        return self._get_state()

    def step(self, action):
        """
        Apply action (= phase_id), advance simulation, and compute reward.

        Returns:
            next_state (np.ndarray)
            reward (float)
            done (bool)
            info (dict)
        """
        phase_id = int(action)
        if phase_id < 0 or phase_id >= self.num_phases:
            raise ValueError(
                f"Invalid phase {phase_id}, should be in [0, {self.num_phases-1}]"
            )

        # Set the traffic light phase
        self.eng.set_tl_phase(self.intersection_id, phase_id)

        # Advance simulation for 'action_duration' seconds
        for _ in range(self.action_duration):
            self.eng.next_step()
            self.current_step += 1

        # Compute reward and next state
        reward = self._compute_reward()
        next_state = self._get_state()
        done = self.current_step >= self.max_episode_steps

        info = {
            "sim_time": self.eng.get_current_time(),
            "average_travel_time": self.eng.get_average_travel_time(),
        }

        return next_state, reward, done, info

    # ------------ Helpers ------------ #

    def _get_state(self):
        """State: waiting vehicle count on each lane (fixed order)."""
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        state = np.array([lane_waiting.get(l, 0) for l in self.lanes], dtype=np.float32)
        return state

    def _compute_reward(self):
        """Reward: negative total waiting vehicles (we want to minimize it)."""
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        total_waiting = sum(lane_waiting.values())
        return -float(total_waiting)

    @property
    def action_space_n(self):
        """Number of discrete actions (phases)."""
        return self.num_phases

    def close(self):
        # CityFlow Engine doesn't require explicit close, but we keep this for API symmetry
        del self.eng
