"""
helpers.py

Small utility functions used across the project.
Nothing fancy here — just stuff that keeps the other files clean.
"""

import os
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_config(config_path="config/config.yaml"):
    """Load the YAML config file and return it as a dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_map_data(data_path="data/medswarm_data.pkl"):
    """Load the prepared map data from the pickle file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Run 'python scripts/prepare_data.py' first!"
        )
    with open(data_path, "rb") as f:
        return pickle.load(f)


def ensure_dir(path):
    """Create a directory if it doesn't exist. Doesn't crash if it already exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def meters_to_km(meters):
    """Convert meters to kilometers, rounded to 2 decimal places."""
    return round(meters / 1000, 2)


def normalize_battery(battery_left, max_battery):
    """Battery as a fraction 0.0 to 1.0."""
    return battery_left / max_battery


def plot_map(data, save_path=None, title="MedSwarm — Mission Map"):
    """
    Quick static map plot showing hospital + all triage zones.
    Uses fake lat/lon coords if OSM data not available.
    
    Saves to save_path if provided, else just shows the plot.
    """
    coords = data["coords"]
    nodes = data["nodes"]
    hospital_idx = data["hospital_idx"]
    zone_indices = data["zone_indices"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#f0f0f0")

    # plot zones
    for i, idx in enumerate(nodes):
        lat, lon = coords[idx]
        if idx == nodes[hospital_idx]:
            # hospital — big red cross
            ax.scatter(lon, lat, s=300, color="red", zorder=5, marker="+", linewidths=3)
            ax.annotate("Hospital", (lon, lat), textcoords="offset points",
                        xytext=(5, 5), fontsize=9, color="red", fontweight="bold")
        else:
            # triage zone — orange circle
            zone_num = zone_indices.index(hospital_idx + i) + 1 if idx != nodes[hospital_idx] else None
            ax.scatter(lon, lat, s=100, color="darkorange", zorder=4, edgecolors="black", linewidths=0.5)
            ax.annotate(f"Z{i}", (lon, lat), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color="black")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)

    source = data.get("source", "unknown")
    ax.text(0.01, 0.01, f"Data source: {source}", transform=ax.transAxes,
            fontsize=8, color="gray")

    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Map saved to {save_path}")
    else:
        plt.show()

    plt.close()
    return fig


def plot_training_curve(log_path, save_path=None):
    """
    Plot the training reward curve from Stable-Baselines3 evaluation logs.
    
    The eval callback saves results to evaluations.npz in the log folder.
    """
    eval_file = os.path.join(log_path, "evaluations.npz")

    if not os.path.exists(eval_file):
        print(f"  No evaluations.npz found in {log_path}")
        print("  Train the model first, then check back.")
        return None

    data = np.load(eval_file)
    timesteps = data["timesteps"]
    results = data["results"]  # shape: (n_evals, n_eval_episodes)
    mean_rewards = results.mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timesteps, mean_rewards, color="steelblue", linewidth=2)
    ax.fill_between(timesteps,
                    mean_rewards - results.std(axis=1),
                    mean_rewards + results.std(axis=1),
                    alpha=0.2, color="steelblue")

    ax.set_title("PPO Training Curve — Mean Episode Reward", fontsize=13, fontweight="bold")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Training curve saved to {save_path}")
    else:
        plt.show()

    plt.close()
    return fig


def run_one_episode(model, env, deterministic=True, render=False):
    """
    Run one full episode with a trained model.
    Returns a list of (obs, action, reward, info) tuples for replay.
    """
    obs, _ = env.reset()
    trajectory = []
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        trajectory.append({
            "obs": obs.copy(),
            "action": action.copy(),
            "reward": reward,
            "info": info.copy(),
        })

        if render:
            env.render()

        obs = next_obs
        done = terminated or truncated

    return trajectory, total_reward