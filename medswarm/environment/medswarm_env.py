"""
MedSwarm Gymnasium Environment
===============================
Custom environment for multi-agent medical emergency triage simulation.
"""

import pickle
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class RewardConfig:
    """Configuration for reward shaping."""
    distance_penalty: float = -0.1
    zone_stabilized: float = 100.0
    battery_failure: float = -2000.0
    mission_complete: float = 5000.0


class MedSwarmEnv(gym.Env):
    """
    Multi-Agent Medical Emergency Response Environment.
    
    This environment simulates a disaster triage scenario where two agents
    (a ground ambulance and a triage drone) must cooperatively deliver
    medical supplies to multiple triage zones.
    
    Attributes:
        amb_matrix: Pre-computed ambulance (road network) distances
        drone_matrix: Pre-computed drone (Euclidean) distances
        max_battery: Maximum drone battery capacity in meters
        num_nodes: Total number of nodes (1 base + N triage zones)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        data_path: str = "data/medswarm_data.pkl",
        max_battery: float = 3000.0,
        reward_config: Optional[RewardConfig] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the MedSwarm environment.
        
        Args:
            data_path: Path to the pre-computed data pickle file
            max_battery: Maximum drone flight range in meters
            reward_config: Custom reward configuration
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()
        
        # Load pre-computed distance matrices
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            
        self.amb_matrix = data['amb_matrix']
        self.drone_matrix = data['drone_matrix']
        self.node_coords = data.get('node_coords', {})
        self.num_nodes = len(data['nodes'])
        self.num_zones = self.num_nodes - 1  # Exclude base hospital
        
        # Environment parameters
        self.max_battery = max_battery
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode
        
        # Action Space: [Ambulance target (0 to num_nodes-1), Drone target (0 to num_nodes-1)]
        self.action_space = spaces.MultiDiscrete([self.num_nodes, self.num_nodes])
        
        # Observation Space components:
        # - amb_loc: Current ambulance location (node index)
        # - drone_status: 0 = docked on ambulance, 1 = deployed
        # - drone_loc: Current drone location (node index)
        # - battery: Remaining battery capacity
        # - triage_zones: Binary vector (1 = needs help, 0 = stabilized)
        obs_dim = 4 + self.num_zones
        low = np.array([0, 0, 0, 0.0] + [0] * self.num_zones, dtype=np.float32)
        high = np.array([self.num_nodes - 1, 1, self.num_nodes - 1, self.max_battery] + 
                        [1] * self.num_zones, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        # Internal state
        self._reset_state()
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_history = []
        
    def _reset_state(self):
        """Reset internal state variables."""
        self.amb_loc = 0  # Start at base hospital
        self.drone_status = 0  # Docked
        self.drone_loc = 0
        self.battery = self.max_battery
        self.triage_zones = np.ones(self.num_zones, dtype=np.float32)
        
    def _get_obs(self) -> np.ndarray:
        """Construct the observation vector."""
        return np.concatenate([
            [self.amb_loc, self.drone_status, self.drone_loc, self.battery],
            self.triage_zones
        ]).astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary information."""
        return {
            "amb_loc": self.amb_loc,
            "drone_loc": self.drone_loc,
            "drone_status": "deployed" if self.drone_status else "docked",
            "battery_remaining": self.battery,
            "zones_remaining": int(np.sum(self.triage_zones)),
            "episode_reward": self.episode_reward,
            "episode_steps": self.episode_steps
        }
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        self._reset_state()
        
        # Reset episode tracking
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_history = []
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: [ambulance_target, drone_target] node indices
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        amb_action, drone_action = int(action[0]), int(action[1])
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # Record state for history
        step_record = {
            "step": self.episode_steps,
            "amb_from": self.amb_loc,
            "amb_to": amb_action,
            "drone_from": self.drone_loc,
            "drone_to": drone_action,
            "battery_before": self.battery
        }
        
        # 1. AMBULANCE MOVEMENT
        amb_dist = self.amb_matrix[self.amb_loc][amb_action]
        self.amb_loc = amb_action
        reward += amb_dist * self.reward_config.distance_penalty
        
        # 2. DRONE MOVEMENT & BATTERY LOGIC
        if drone_action != 0 and drone_action != self.amb_loc:
            # Drone is deployed to a different location
            self.drone_status = 1
            drone_dist = self.drone_matrix[self.drone_loc][drone_action]
            self.drone_loc = drone_action
            self.battery -= drone_dist
            step_record["drone_dist"] = drone_dist
        else:
            # Drone returns to or stays at ambulance (recharge)
            self.drone_status = 0
            self.drone_loc = self.amb_loc
            self.battery = self.max_battery
            step_record["drone_dist"] = 0
            
        step_record["battery_after"] = self.battery
        
        # 3. CHECK BATTERY FAILURE
        if self.battery < 0:
            reward += self.reward_config.battery_failure
            terminated = True
            info["termination_reason"] = "battery_depleted"
            step_record["battery_failed"] = True
        else:
            step_record["battery_failed"] = False
            
            # 4. DELIVERIES / TRIAGE STABILIZATION
            zones_stabilized = 0
            
            # Ambulance delivery (node 1-N maps to triage_zones index 0 to N-1)
            if self.amb_loc > 0 and self.triage_zones[self.amb_loc - 1] == 1:
                self.triage_zones[self.amb_loc - 1] = 0
                reward += self.reward_config.zone_stabilized
                zones_stabilized += 1
                
            # Drone delivery
            if self.drone_loc > 0 and self.triage_zones[self.drone_loc - 1] == 1:
                self.triage_zones[self.drone_loc - 1] = 0
                reward += self.reward_config.zone_stabilized
                zones_stabilized += 1
                
            step_record["zones_stabilized"] = zones_stabilized
            
            # 5. CHECK WIN CONDITION
            if np.sum(self.triage_zones) == 0:
                reward += self.reward_config.mission_complete
                terminated = True
                info["termination_reason"] = "mission_complete"
                
        # Update episode tracking
        self.episode_reward += reward
        self.episode_steps += 1
        step_record["reward"] = reward
        self.episode_history.append(step_record)
        
        info.update(self._get_info())
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (placeholder for visualization)."""
        if self.render_mode == "human":
            zones_status = "".join(["●" if z else "○" for z in self.triage_zones])
            print(f"Step {self.episode_steps}: Amb@{self.amb_loc} | "
                  f"Drone@{self.drone_loc}({'D' if self.drone_status else 'A'}) | "
                  f"Battery:{self.battery:.0f}m | Zones:[{zones_status}]")
    
    def get_episode_history(self):
        """Return the complete episode history for visualization."""
        return self.episode_history.copy()
    
    def close(self):
        """Clean up resources."""
        pass
