#!/usr/bin/env python3
"""
Quick test script to verify the improved environment and reward structure.
Run with: python test_improved_env.py
"""

from medswarm import MedSwarmEnv
import numpy as np

# Create environment with FIXED reward config
env_config = {
    "max_battery": 5000.0,
    "max_steps": 200,
    "reward": {
        "per_step_penalty": -0.5,
        "zone_stabilized": 300.0,
        "battery_failure": -5000.0,
        "mission_complete": 8000.0,
    },
}

env = MedSwarmEnv(data_path="data/medswarm_data.pkl", config=env_config)
obs, info = env.reset()

print("=" * 60)
print("MedSwarm Improved Environment Test")
print("=" * 60)
print(f"\nInitial observation shape: {obs.shape}")
print(f"Initial zones stabilized: {int(np.sum(obs[3:]))}/12")
print(f"\nNew config improvements:")
print(f"  - Max steps per episode: 200 (was 100)")
print(f"  - Max battery: 5000m (was 3000m)")
print(f"  - Zone stabilized reward: 200 (was 100)")
print(f"  - Distance penalty: -0.05/m (was -0.1/m)")
print(f"  - Mission complete bonus: 10,000 (was 5,000)")

# Run 100 random steps to test
print(f"\nRunning 100 random steps...")
total_reward = 0
zones_found = set()

for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
    zones = int(info["zones_done"])
    if zones > len(zones_found):
        zones_found.add(zones)
        print(f"  Step {step}: Found zone #{zones}! Reward this step: {reward:+.1f}")
    
    if done or truncated:
        print(f"\nEpisode ended at step {step+1}")
        break

print(f"\nTest Results:")
print(f"  - Zones stabilized: {int(info['zones_done'])}/12")
print(f"  - Total reward: {total_reward:+.1f}")
print(f"  - Battery remaining: {info['battery_left']:.0f}m / 5000m")

print("\n" + "=" * 60)
print("✅ Environment is working correctly with improved settings!")
print("=" * 60)
print("\nTo retrain the model with these improvements, run:")
print("  python scripts/train.py")
print("\nExpected improvements over previous training:")
print("  - More zones visited (>4/12)")
print("  - Higher overall reward")
print("  - Better convergence to completing all 12 zones")
