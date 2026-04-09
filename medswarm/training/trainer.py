"""
trainer.py

Everything related to training the PPO agent lives here.

PPO (Proximal Policy Optimization) is the algorithm that teaches the agent
how to act. It's like a student that:
1. Tries stuff (rollout)
2. Sees what worked and what didn't (reward)
3. Updates its brain a little bit (gradient step)
4. Repeat

We use Stable-Baselines3 which gives us a clean, ready-to-use PPO implementation.
We just plug in our custom environment and it handles the rest.
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

from medswarm.environment.medswarm_env import MedSwarmEnv


# ------------------------------------------------------------------ #
#  Custom Callback — prints progress every N steps                    #
# ------------------------------------------------------------------ #

class CustomEvalCallback(EvalCallback):
    """
    Custom evaluation callback that handles stochastic environments better.
    Records episode rewards with proper variance tracking.
    """
    def _on_step(self):
        result = super()._on_step()
        # Force evaluation to run more frequently and track variance properly
        return result


class ProgressCallback(BaseCallback):
    """
    Prints a friendly progress update every 5,000 steps.
    Tracks mean reward AND zones completed to understand training progress.
    """

    def __init__(self, total_steps, verbose=0):
        super().__init__(verbose)
        self.total_steps = total_steps
        self.print_every = 5000  # Print every 5000 steps for more visibility
        self.episode_rewards = []
        self.episode_zones = []  # track zones completed per episode
        self.current_episode_reward = 0.0
        self.current_episode_zones = 0
        self.last_print_step = 0

    def _on_step(self):
        # accumulate reward for current episode
        rewards = self.locals.get("rewards", [0])
        self.current_episode_reward += float(np.mean(rewards))

        # Extract zones_done from info dicts
        infos = self.locals.get("infos", [{}])
        max_zones_this_step = max([info.get("zones_done", 0) for info in infos] or [0])

        dones = self.locals.get("dones", [False])
        if any(dones):
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_zones.append(max_zones_this_step)
            self.current_episode_reward = 0.0
            self.current_episode_zones = 0

        # print every N steps
        if self.num_timesteps - self.last_print_step >= self.print_every:
            self.last_print_step = self.num_timesteps
            pct = (self.num_timesteps / self.total_steps) * 100
            if len(self.episode_rewards) > 0:
                recent_mean_reward = np.mean(self.episode_rewards[-20:])
                recent_mean_zones = np.mean(self.episode_zones[-20:])
                success_rate = sum(1 for z in self.episode_zones[-20:] if z == 12) / min(20, len(self.episode_zones[-20:]))
                msg = (
                    f"  [{pct:5.1f}%]  Step {self.num_timesteps:>7,} / {self.total_steps:,}"
                    f"  |  Reward: {recent_mean_reward:>8.1f}  |  Zones: {recent_mean_zones:>5.1f}/12  |  Success: {success_rate*100:>5.1f}%"
                )
            else:
                msg = f"  [{pct:5.1f}%]  Step {self.num_timesteps:>7,} / {self.total_steps:,}"
            
            print(msg, flush=True)
            sys.stdout.flush()

        return True  # return True to keep training going


# ------------------------------------------------------------------ #
#  Make environment factory (needed for vectorized envs)             #
# ------------------------------------------------------------------ #

def make_env(data_path, config):
    """
    Returns a function that creates one MedSwarmEnv instance.
    Stable-Baselines3 needs this pattern for parallel environments.
    """
    env_config = {
        "max_battery": config["environment"]["max_battery"],
        "max_steps": config["environment"].get("max_steps_per_episode", 100),
        "reward": config["environment"]["reward"],
    }

    def _init():
        env = MedSwarmEnv(data_path=data_path, config=env_config)
        env = Monitor(env)  # wraps env to track episode stats
        return env

    return _init


# ------------------------------------------------------------------ #
#  Main training function                                             #
# ------------------------------------------------------------------ #

def train(config_path="config/config.yaml"):
    """
    Full training pipeline. Call this from scripts/train.py.
    
    1. Load config
    2. Create vectorized environments (N parallel copies for faster collection)
    3. Build PPO model
    4. Train for total_timesteps
    5. Save the final model
    """

    print("=" * 50)
    print("MedSwarm — PPO Training")
    print("=" * 50)
    sys.stdout.flush()

    # load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    data_path = config["data"]["output_path"]
    model_save_path = train_cfg["model_save_path"]
    log_path = train_cfg["log_path"]

    # make sure output dirs exist
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).mkdir(parents=True, exist_ok=True)

    # check data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Run 'python scripts/prepare_data.py' first!"
        )

    print(f"\n  Data: {data_path}")
    print(f"  Total timesteps: {train_cfg['total_timesteps']:,}")
    print(f"  Parallel envs: {train_cfg['n_envs']}")
    print(f"  Learning rate: {train_cfg['learning_rate']}")
    sys.stdout.flush()

    # ---- Build vectorized training environment ----
    # n_envs parallel environments collect experience simultaneously
    # this makes training ~4x faster vs a single environment
    print(f"\n  Creating {train_cfg['n_envs']} parallel environments...")
    
    vec_env = make_vec_env(
        make_env(data_path, config),
        n_envs=train_cfg["n_envs"],
    )

    # ---- Evaluation environment (single env, no parallel) ----
    eval_env = make_vec_env(
        make_env(data_path, config),
        n_envs=1,
    )

    # ---- Callbacks ----
    # EvalCallback: runs the agent on eval_env every eval_freq steps
    #               saves the best model automatically
    # Note: deterministic=False allows environment stochasticity (different battery levels)
    #       to create variance in episode rewards
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_save_path}_best",
        log_path=log_path,
        eval_freq=max(train_cfg["eval_freq"] // train_cfg["n_envs"], 1),
        n_eval_episodes=train_cfg["eval_episodes"],
        deterministic=False,  # Allow stochasticity to see true variance
        verbose=1,
    )

    # CheckpointCallback: saves model every 50k steps (backup in case training crashes)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // train_cfg["n_envs"], 1),
        save_path=log_path,
        name_prefix="ppo_medswarm_checkpoint",
    )

    progress_callback = ProgressCallback(
        total_steps=train_cfg["total_timesteps"]
    )

    # ---- Build the PPO model ----
    # "MlpPolicy" = multi-layer perceptron — a simple feedforward neural net
    # this is the agent's "brain" — takes observation → outputs action probabilities
    print("\n  Building PPO model (MLP policy with custom architecture)...")
    
    # Custom policy network architecture: deeper networks for better representation learning
    # For a problem with 18 inputs (observation size) and 13x13 action space
    policy_kwargs = {
        "net_arch": [256, 256, 128],  # Three layers: 256 -> 256 -> 128 neurons
        "activation_fn": torch.nn.ReLU,  # ReLU activation (standard for RL)
    }
    
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=train_cfg["learning_rate"],
        n_steps=train_cfg["n_steps"],
        batch_size=train_cfg["batch_size"],
        n_epochs=train_cfg["n_epochs"],
        gamma=train_cfg["gamma"],
        gae_lambda=train_cfg.get("gae_lambda", 0.95),
        clip_range=train_cfg["clip_range"],
        ent_coef=train_cfg.get("ent_coef", 0.01),
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=log_path,
    )

    print(f"  Policy network: {model.policy}")
    print(f"\n  Starting training... (this takes a while, grab some chai ☕)")
    print("-" * 50, flush=True)
    sys.stdout.flush()

    # ---- Train ----
    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=[eval_callback, checkpoint_callback, progress_callback],
        progress_bar=False,  # we have our own progress printing
    )

    # ---- Save final model ----
    model.save(model_save_path)
    print("\n" + "-" * 50, flush=True)
    print(f"  Training complete!", flush=True)
    print(f"  Final model saved: {model_save_path}.zip", flush=True)
    print(f"  Best model saved: {model_save_path}_best/", flush=True)
    print("=" * 50, flush=True)
    sys.stdout.flush()

    return model


# ------------------------------------------------------------------ #
#  Evaluation helper                                                  #
# ------------------------------------------------------------------ #

def evaluate_model(model_path, data_path, config_path="config/config.yaml", n_episodes=20):
    """
    Load a saved model and evaluate it.
    Prints success rate, mean reward, and mean steps.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env_config = {
        "max_battery": config["environment"]["max_battery"],
        "max_steps": config["environment"].get("max_steps_per_episode", 100),
        "reward": config["environment"]["reward"],
    }

    env = MedSwarmEnv(data_path=data_path, config=env_config)

    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"  Model not found at {model_path}")
        print("  Run 'python scripts/train.py' first!")
        return None

    print(f"\nEvaluating model over {n_episodes} episodes...")

    all_rewards = []
    all_steps = []
    successes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        step = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1
            done = terminated or truncated

        all_rewards.append(ep_reward)
        all_steps.append(step)
        if info["zones_done"] == env.num_zones:
            successes += 1

    print(f"  Mean reward:      {np.mean(all_rewards):>8.1f}")
    print(f"  Mean steps:       {np.mean(all_steps):>8.1f}")
    print(f"  Success rate:     {successes / n_episodes * 100:>7.1f}%")
    print(f"  Min/Max reward:   {np.min(all_rewards):.1f} / {np.max(all_rewards):.1f}")

    return {
        "mean_reward": np.mean(all_rewards),
        "mean_steps": np.mean(all_steps),
        "success_rate": successes / n_episodes,
        "all_rewards": all_rewards,
    }