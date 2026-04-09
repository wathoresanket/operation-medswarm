"""
train.py

Step 2 of the pipeline.

Trains the PPO agent on the MedSwarm environment.

This will take ~10-20 minutes depending on your machine.
The agent starts out terrible (random actions), and gradually learns
to coordinate the ambulance and drone to cover all 12 zones efficiently.

Usage:
    python scripts/train.py
    python scripts/train.py --config config/config.yaml
    python scripts/train.py --evaluate   # skip training, just evaluate existing model
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medswarm.training.trainer import train, evaluate_model
from medswarm.utils.helpers import load_config, plot_training_curve


def main():
    parser = argparse.ArgumentParser(description="Train MedSwarm PPO agent")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Skip training, just evaluate the saved model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of eval episodes (default: 20)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = config["data"]["output_path"]
    model_path = config["training"]["model_save_path"]
    log_path = config["training"]["log_path"]

    if not args.evaluate:
        # run training
        model = train(config_path=args.config)

        # evaluate right after training
        print("\nRunning post-training evaluation...")
        results = evaluate_model(model_path, data_path, args.config, n_episodes=args.episodes)

        # save training curve as a PNG
        plot_training_curve(log_path, save_path="logs/training_curve.png")
        print("\nTraining curve saved to logs/training_curve.png")
        print("Now run: python scripts/run_dashboard.py")

    else:
        # just evaluate existing model
        print("Skipping training — evaluating existing model...")
        results = evaluate_model(model_path, data_path, args.config, n_episodes=args.episodes)
        if results is None:
            print("No model found. Train first with: python scripts/train.py")


if __name__ == "__main__":
    main()