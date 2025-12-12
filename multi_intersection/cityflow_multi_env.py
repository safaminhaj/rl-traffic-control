import os
import json
import numpy as np
import cityflow


class CityFlowMultiJunctionEnv:
    """
    CityFlow environment that controls MULTIPLE signalized intersections
    in the same network.

    - intersections: a list of intersection IDs (strings) that must each
      have a trafficLight object defined in the roadnet file.

    - Action: vector of phase indices, one per controlled intersection.
      Example:
          intersection_ids = ["intersection_1_1", "intersection_1_2"]
          action = [0, 3]  # phase 0 for 1_1, phase 3 for 1_2

    - State: global vector of lane waiting vehicle counts (all lanes in the
      network, sorted by lane ID).

    - Reward: negative total waiting vehicles over all lanes (global reward).
    """

    def __init__(
        self,
        config_path="cityflow_scenario/config.json",
        intersection_ids=None,
        action_duration=10,    # simulation seconds per action
        max_episode_steps=300  # episode length in seconds
    ):
        if intersection_ids is None:
            raise ValueError("You must provide a list of intersection_ids.")

        self.config_path = config_path
        self.intersection_ids = list(intersection_ids)
        self.action_duration = int(action_duration)
        self.max_episode_steps = int(max_episode_steps)

        # --- Create CityFlow engine ---
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.eng = cityflow.Engine(self.config_path, thread_num=1)

        # --- Load config & roadnet info ---
        with open(self.config_path) as f:
            cfg = json.load(f)

        self.interval = float(cfg.get("interval", 1.0))
        roadnet_path = os.path.join(os.path.dirname(self.config_path),
                                    cfg["roadnetFile"])
        with open(roadnet_path) as f:
            roadnet = json.load(f)

        # --- Number of phases per controlled intersection ---
        self.num_phases = {}
        for iid in self.intersection_ids:
            n_ph = self._get_num_phases(roadnet, iid)
            if n_ph == 0:
                raise ValueError(
                    f"No traffic light phases found for intersection '{iid}'. "
                    "Check that 'trafficLight' is defined in the roadnet file."
                )
            self.num_phases[iid] = n_ph

        # --- Lane order for state representation ---
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        self.lanes = sorted(lane_waiting.keys())

        self.current_step = 0

    # ------------------------------------------------------------------ #
    #                             Helpers                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_num_phases(roadnet, intersection_id):
        """Return the number of signal phases for a given intersection ID."""
        for inter in roadnet["intersections"]:
            if inter["id"] == intersection_id:
                tl = inter.get("trafficLight")
                if tl is None:
                    return 0
                return len(tl.get("lightphases", []))
        return 0

    def _get_state(self):
        """
        Global state: waiting vehicle count on each lane (fixed lane order).
        """
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        state = np.array(
            [lane_waiting.get(lane_id, 0) for lane_id in self.lanes],
            dtype=np.float32
        )
        return state

    def _compute_reward(self):
        """
        Global reward: negative total waiting vehicles (we want to minimize it).
        """
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        total_waiting = sum(lane_waiting.values())
        return -float(total_waiting)

    # ------------------------------------------------------------------ #
    #                             RL API                                 #
    # ------------------------------------------------------------------ #

    def reset(self):
        """
        Reset the simulation and return the initial global state.
        """
        self.eng.reset()
        self.current_step = 0

        # Optional: set an initial phase (0) for all controlled intersections.
        for iid in self.intersection_ids:
            self.eng.set_tl_phase(iid, 0)

        return self._get_state()

    def step(self, action):
        """
        Apply a multi-intersection action and advance the simulation.

        Parameters
        ----------
        action : sequence (list, tuple, np.ndarray)
            A vector of integers with length == len(intersection_ids).
            Each entry is a phase index for the corresponding intersection.

        Returns
        -------
        next_state : np.ndarray
        reward : float
        done : bool
        info : dict
        """
        # --- Validate & normalize action ---
        if not hasattr(action, "__len__"):
            raise ValueError(
                "For multi-junction env, 'action' must be a sequence "
                f"of length {len(self.intersection_ids)}."
            )

        if len(action) != len(self.intersection_ids):
            raise ValueError(
                f"Expected action of length {len(self.intersection_ids)}, "
                f"got {len(action)}."
            )

        # Convert to list of ints
        action = [int(a) for a in action]

        # Bounds check for each intersection
        for iid, phase in zip(self.intersection_ids, action):
            n_phases = self.num_phases[iid]
            if phase < 0 or phase >= n_phases:
                raise ValueError(
                    f"Invalid phase {phase} for intersection '{iid}', "
                    f"should be in [0, {n_phases - 1}]."
                )

        # --- Apply all phases ---
        for iid, phase in zip(self.intersection_ids, action):
            self.eng.set_tl_phase(iid, phase)

        # --- Advance simulation ---
        for _ in range(self.action_duration):
            self.eng.next_step()
            self.current_step += 1

        # --- Compute reward & next state ---
        reward = self._compute_reward()
        next_state = self._get_state()
        done = self.current_step >= self.max_episode_steps

        info = {
            "sim_time": self.eng.get_current_time(),
            "average_travel_time": self.eng.get_average_travel_time(),
        }

        return next_state, reward, done, info

    # ------------------------------------------------------------------ #
    #                          Convenience                               #
    # ------------------------------------------------------------------ #

    @property
    def num_intersections(self):
        """Number of intersections being controlled."""
        return len(self.intersection_ids)

    @property
    def action_space_nvec(self):
        """
        Number of discrete actions (phases) for each intersection,
        in the same order as self.intersection_ids.
        """
        return [self.num_phases[iid] for iid in self.intersection_ids]

    def close(self):
        """
        CityFlow engine does not require explicit 'close', but we keep this
        method for API symmetry with other RL environments.
        """
        del self.eng
