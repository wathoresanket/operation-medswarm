"""
Operation MedSwarm: Hybrid RL for Urban Disaster Triage
========================================================

A Multi-Agent Reinforcement Learning system for optimizing emergency
medical logistics using heterogeneous agents (ground ambulance + triage drone).

Modules:
    - data: Geographic data preparation and distance matrix computation
    - environment: Custom Gymnasium environment for the MedSwarm simulation
    - training: PPO training pipeline with callbacks and logging
    - visualization: Interactive Streamlit dashboard
    - utils: Helper functions and utilities
"""

__version__ = "1.0.0"
__author__ = "MedSwarm Team"

from medswarm.environment.medswarm_env import MedSwarmEnv

__all__ = ["MedSwarmEnv", "__version__"]
