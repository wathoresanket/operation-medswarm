"""
MedSwarm Training Module
=========================
PPO training pipeline with comprehensive logging and callbacks.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, 
    EvalCallback, 
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from medswarm.environment.medswarm_env import MedSwarmEnv


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # PPO Hyperparameters
    algorithm: str = "PPO"
    total_timesteps: int = 300000
    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    
    # Environment settings
    n_envs: int = 4
    data_path: str = "data/medswarm_data.pkl"
    
    # Evaluation settings
    eval_freq: int = 5000
    n_eval_episodes: int = 10
    
    # Logging settings
    log_path: str = "logs/"
    model_path: str = "models/"
    save_freq: int = 10000
    verbose: int = 1


class MetricsCallback(BaseCallback):
    """
    Custom callback for tracking and saving detailed training metrics.
    """
    
    def __init__(self, save_path: str, save_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.metrics = {
            "timesteps": [],
            "episodes": [],
            "mean_reward": [],
            "std_reward": [],
            "mean_length": [],
            "success_rate": [],
            "battery_failures": [],
            "mission_completes": []
        }
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_outcomes = []
        
    def _on_step(self) -> bool:
        # Collect episode info from the Monitor wrapper
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(ep_info.get('r', 0))
            self.episode_lengths.append(ep_info.get('l', 0))
            
        # Save metrics periodically
        if self.n_calls % self.save_freq == 0 and len(self.episode_rewards) > 0:
            self.metrics["timesteps"].append(self.num_timesteps)
            self.metrics["episodes"].append(len(self.episode_rewards))
            self.metrics["mean_reward"].append(float(np.mean(self.episode_rewards[-100:])))
            self.metrics["std_reward"].append(float(np.std(self.episode_rewards[-100:])))
            self.metrics["mean_length"].append(float(np.mean(self.episode_lengths[-100:])))
            
            # Save to JSON
            self._save_metrics()
            
        return True
    
    def _save_metrics(self):
        """Save metrics to JSON file."""
        os.makedirs(self.save_path, exist_ok=True)
        metrics_file = os.path.join(self.save_path, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def _on_training_end(self):
        """Save final metrics on training end."""
        self._save_metrics()


class MedSwarmTrainer:
    """
    Comprehensive trainer for the MedSwarm environment.
    
    Handles environment setup, model initialization, training with callbacks,
    and logging of all relevant metrics.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration (uses defaults if not provided)
        """
        self.config = config or TrainingConfig()
        self.model: Optional[PPO] = None
        self.env = None
        self.eval_env = None
        
        # Create directories
        os.makedirs(self.config.log_path, exist_ok=True)
        os.makedirs(self.config.model_path, exist_ok=True)
        
        # Save config
        self._save_config()
        
    def _save_config(self):
        """Save training configuration to JSON."""
        config_path = os.path.join(self.config.log_path, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
            
    def setup_environment(self):
        """Set up training and evaluation environments."""
        print("🏥 Setting up MedSwarm environments...")
        
        # Create vectorized training environment
        def make_env():
            env = MedSwarmEnv(data_path=self.config.data_path)
            return Monitor(env)
        
        self.env = make_vec_env(make_env, n_envs=self.config.n_envs)
        
        # Create evaluation environment
        self.eval_env = Monitor(
            MedSwarmEnv(data_path=self.config.data_path),
            filename=os.path.join(self.config.log_path, "eval")
        )
        
        print(f"✅ Environments ready ({self.config.n_envs} parallel training envs)")
        
    def setup_model(self):
        """Initialize the PPO model."""
        print("🤖 Initializing PPO agent...")
        
        # Configure logger
        logger = configure(self.config.log_path, ["stdout", "csv", "tensorboard"])
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            verbose=self.config.verbose,
            tensorboard_log=os.path.join(self.config.log_path, "tensorboard")
        )
        self.model.set_logger(logger)
        
        print("✅ PPO agent initialized")
        
    def setup_callbacks(self) -> CallbackList:
        """Set up training callbacks."""
        callbacks = []
        
        # Evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=self.config.model_path,
            log_path=self.config.log_path,
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=self.config.model_path,
            name_prefix="medswarm_checkpoint"
        )
        callbacks.append(checkpoint_callback)
        
        # Custom metrics callback
        metrics_callback = MetricsCallback(
            save_path=self.config.log_path,
            save_freq=1000,
            verbose=self.config.verbose
        )
        callbacks.append(metrics_callback)
        
        return CallbackList(callbacks)
    
    def train(self) -> PPO:
        """
        Run the complete training pipeline.
        
        Returns:
            Trained PPO model
        """
        print("\n" + "="*60)
        print("🚀 OPERATION MEDSWARM - TRAINING INITIATED")
        print("="*60 + "\n")
        
        # Setup
        self.setup_environment()
        self.setup_model()
        callbacks = self.setup_callbacks()
        
        # Training
        print(f"\n📈 Starting training for {self.config.total_timesteps:,} timesteps...")
        start_time = datetime.now()
        
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        training_time = datetime.now() - start_time
        
        # Save final model
        final_model_path = os.path.join(self.config.model_path, "medswarm_final")
        self.model.save(final_model_path)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE")
        print(f"   Duration: {training_time}")
        print(f"   Final model saved: {final_model_path}")
        print("="*60 + "\n")
        
        return self.model
    
    def evaluate(self, n_episodes: int = 10, render: bool = False) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render during evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        print(f"\n📊 Evaluating model for {n_episodes} episodes...")
        
        env = MedSwarmEnv(data_path=self.config.data_path, render_mode="human" if render else None)
        
        rewards = []
        lengths = []
        outcomes = []
        
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
                if render:
                    env.render()
                    
            rewards.append(episode_reward)
            lengths.append(episode_length)
            outcomes.append(info.get("termination_reason", "unknown"))
            
        env.close()
        
        results = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
            "success_rate": outcomes.count("mission_complete") / n_episodes,
            "battery_failure_rate": outcomes.count("battery_depleted") / n_episodes
        }
        
        print(f"   Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"   Success Rate: {results['success_rate']*100:.1f}%")
        
        return results
    
    def load_model(self, model_path: str):
        """Load a pre-trained model."""
        self.model = PPO.load(model_path)
        print(f"✅ Model loaded from: {model_path}")


def main():
    """CLI entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MedSwarm RL agent")
    parser.add_argument("--timesteps", type=int, default=300000,
                        help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=0.0003,
                        help="Learning rate")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--eval-freq", type=int, default=5000,
                        help="Evaluation frequency")
    parser.add_argument("--data-path", default="data/medswarm_data.pkl",
                        help="Path to data file")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        data_path=args.data_path
    )
    
    trainer = MedSwarmTrainer(config)
    trainer.train()
    trainer.evaluate(n_episodes=10)


if __name__ == "__main__":
    main()
