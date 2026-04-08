#!/usr/bin/env python
"""
CLI Script for Training
========================
Train the MedSwarm PPO agent.
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medswarm.training.trainer import MedSwarmTrainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train the MedSwarm PPO reinforcement learning agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py
  python scripts/train.py --timesteps 500000 --lr 0.0001
  python scripts/train.py --n-envs 8 --eval-freq 10000
        """
    )
    
    # Training parameters
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=300000,
        help="Total training timesteps (default: 300000)"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.0003,
        help="Learning rate (default: 0.0003)"
    )
    parser.add_argument(
        "--n-envs", 
        type=int, 
        default=4,
        help="Number of parallel environments (default: 4)"
    )
    parser.add_argument(
        "--n-steps", 
        type=int, 
        default=2048,
        help="Number of steps per rollout (default: 2048)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=64,
        help="Minibatch size (default: 64)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--eval-freq", 
        type=int, 
        default=5000,
        help="Evaluation frequency in timesteps (default: 5000)"
    )
    parser.add_argument(
        "--eval-episodes", 
        type=int, 
        default=10,
        help="Number of evaluation episodes (default: 10)"
    )
    
    # Paths
    parser.add_argument(
        "--data-path", 
        default="data/medswarm_data.pkl",
        help="Path to data file (default: data/medswarm_data.pkl)"
    )
    parser.add_argument(
        "--log-path", 
        default="logs/",
        help="Path for logs (default: logs/)"
    )
    parser.add_argument(
        "--model-path", 
        default="models/",
        help="Path for models (default: models/)"
    )
    
    # Options
    parser.add_argument(
        "--evaluate-only", 
        action="store_true",
        help="Only evaluate, don't train"
    )
    parser.add_argument(
        "--load-model", 
        type=str,
        help="Path to pre-trained model to load"
    )
    parser.add_argument(
        "--verbose", 
        type=int, 
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(args.data_path):
        print(f"\n❌ Data file not found: {args.data_path}")
        print("   Run 'python scripts/prepare_data.py' first to prepare the data.\n")
        sys.exit(1)
    
    # Create config
    config = TrainingConfig(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        data_path=args.data_path,
        log_path=args.log_path,
        model_path=args.model_path,
        verbose=args.verbose
    )
    
    # Initialize trainer
    trainer = MedSwarmTrainer(config)
    
    # Load model if specified
    if args.load_model:
        trainer.load_model(args.load_model)
    
    # Train or evaluate
    if args.evaluate_only:
        if trainer.model is None:
            print("\n❌ No model loaded. Use --load-model to specify a model.\n")
            sys.exit(1)
        trainer.evaluate(n_episodes=args.eval_episodes, render=True)
    else:
        trainer.train()
        trainer.evaluate(n_episodes=args.eval_episodes)
    
    print("\n✅ Done!\n")


if __name__ == "__main__":
    main()
