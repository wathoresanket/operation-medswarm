"""
medswarm_env.py

This is the "game" the RL agent plays.

Think of it like a board game:
- The board = Connaught Place map with 12 disaster zones
- The players = ground ambulance + medical drone
- Each turn = pick where each agent goes next
- Score = reward (positive for helping, negative for distance + failures)

The environment follows the Gymnasium API, which is the standard interface
for RL environments (like how all video game controllers have the same buttons).
"""

import pickle
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MedSwarmEnv(gym.Env):
    """
    Multi-agent disaster triage environment.

    Two agents:
      - Ambulance: moves on real roads, no range limit
      - Drone: flies straight line, limited to max_battery meters total
    
    Goal: Reach all 12 triage zones with minimum total travel distance.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, data_path="data/medswarm_data.pkl", config=None, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # load default config if none provided
        if config is None:
            config = {
                "max_battery": 3000.0,
                "max_steps": 100,
                "reward": {
                    "distance_penalty": -0.1,
                    "zone_stabilized": 100.0,
                    "battery_failure": -2000.0,
                    "mission_complete": 5000.0,
                },
            }

        self.max_battery = config["max_battery"]
        self.max_steps = config.get("max_steps", 100)
        self.reward_cfg = config["reward"]

        # load map data from the pickle file prepared by data_prep.py
        with open(data_path, "rb") as f:
            map_data = pickle.load(f)

        self.hospital_idx = map_data["hospital_idx"]
        self.zone_indices = map_data["zone_indices"]
        self.num_zones = len(self.zone_indices)
        self.num_nodes = len(map_data["nodes"])  # hospital + zones = 13 nodes

        # distance matrices
        # ambulance uses road_dist (real roads), drone uses euclidean_dist (straight line)
        self.road_dist = map_data["road_dist"].astype(np.float32)
        self.euclidean_dist = map_data["euclidean_dist"].astype(np.float32)

        # store coords for rendering
        self.coords = map_data.get("coords", {})
        self.nodes = map_data["nodes"]

        # ---- Action Space ----
        # Each agent picks which node to go to next (hospital = 0, zones = 1-12)
        # Total nodes = 13, so 13 choices per agent
        # MultiDiscrete([13, 13]) means: pick a number 0-12 for each agent
        self.action_space = spaces.MultiDiscrete([self.num_nodes, self.num_nodes])

        # ---- Observation Space ----
        # What the agent "sees" each step:
        # [ambulance_pos, drone_pos, drone_battery_normalized, zone_1_done, ..., zone_12_done]
        # Total: 1 + 1 + 1 + 12 = 15 values
        obs_low  = np.zeros(3 + self.num_zones, dtype=np.float32)
        obs_high = np.ones(3 + self.num_zones, dtype=np.float32)
        obs_high[0] = self.num_nodes - 1  # ambulance position
        obs_high[1] = self.num_nodes - 1  # drone position
        obs_high[2] = 1.0                 # battery as fraction (0.0 to 1.0)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # internal state (initialized properly in reset())
        self._ambulance_pos = self.hospital_idx
        self._drone_pos = self.hospital_idx
        self._battery_left = self.max_battery
        self._zones_done = np.zeros(self.num_zones, dtype=np.float32)
        self._step_count = 0

    # ------------------------------------------------------------------ #
    #  RESET — start a fresh episode                                       #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # both agents start at the hospital
        self._ambulance_pos = self.hospital_idx
        self._drone_pos = self.hospital_idx
        self._battery_left = self.max_battery
        self._zones_done = np.zeros(self.num_zones, dtype=np.float32)
        self._step_count = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    # ------------------------------------------------------------------ #
    #  STEP — one action, one timestep                                     #
    # ------------------------------------------------------------------ #

    def step(self, action):
        amb_target, drone_target = int(action[0]), int(action[1])

        reward = 0.0
        terminated = False  # episode ended naturally (success or failure)
        truncated = False   # episode cut off by step limit

        # ---- Move Ambulance ----
        amb_dist = self.road_dist[self._ambulance_pos][amb_target]
        reward += amb_dist * self.reward_cfg["distance_penalty"]
        self._ambulance_pos = amb_target

        # check if ambulance just stabilized a zone
        if amb_target in self.zone_indices:
            zone_i = self.zone_indices.index(amb_target)
            if self._zones_done[zone_i] == 0.0:
                self._zones_done[zone_i] = 1.0
                reward += self.reward_cfg["zone_stabilized"]

        # ---- Move Drone ----
        drone_dist = self.euclidean_dist[self._drone_pos][drone_target]

        # check if drone has enough battery for this move
        if self._battery_left - drone_dist < 0:
            # drone ran out of battery — mission fails
            reward += self.reward_cfg["battery_failure"]
            terminated = True
        else:
            self._battery_left -= drone_dist
            self._drone_pos = drone_target

            # check if drone just stabilized a zone
            if drone_target in self.zone_indices:
                zone_i = self.zone_indices.index(drone_target)
                if self._zones_done[zone_i] == 0.0:
                    self._zones_done[zone_i] = 1.0
                    reward += self.reward_cfg["zone_stabilized"]

        # ---- Check Mission Complete ----
        if np.all(self._zones_done == 1.0):
            reward += self.reward_cfg["mission_complete"]
            terminated = True

        # ---- Step limit ----
        self._step_count += 1
        if self._step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()

        # extra info for logging/debugging
        info = {
            "ambulance_pos": self._ambulance_pos,
            "drone_pos": self._drone_pos,
            "battery_left": self._battery_left,
            "zones_done": int(np.sum(self._zones_done)),
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  OBSERVATION — pack state into a flat numpy array                    #
    # ------------------------------------------------------------------ #

    def _get_obs(self):
        obs = np.zeros(3 + self.num_zones, dtype=np.float32)
        obs[0] = float(self._ambulance_pos)
        obs[1] = float(self._drone_pos)
        obs[2] = self._battery_left / self.max_battery  # normalize to 0-1
        obs[3:] = self._zones_done
        return obs

    # ------------------------------------------------------------------ #
    #  RENDER — print current state to terminal                            #
    # ------------------------------------------------------------------ #

    def render(self):
        zones_done = int(np.sum(self._zones_done))
        battery_pct = (self._battery_left / self.max_battery) * 100

        print(f"\n--- Step {self._step_count} ---")
        print(f"  Ambulance @ node {self._ambulance_pos}")
        print(f"  Drone     @ node {self._drone_pos}  |  Battery: {battery_pct:.1f}%")
        print(f"  Zones stabilized: {zones_done} / {self.num_zones}")
        print(f"  Zone status: {self._zones_done.astype(int)}")

    def close(self):
        pass