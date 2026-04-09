"""
prepare_data.py

Step 1 of the pipeline.

Downloads the Connaught Place road network from OpenStreetMap
and builds the distance matrices that the RL environment needs.

If you have no internet, it auto-falls back to synthetic data.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --config config/config.yaml
"""

import sys
import os
import argparse

# make sure we can import from the medswarm package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medswarm.data.data_prep import prepare_data


def main():
    parser = argparse.ArgumentParser(description="Prepare MedSwarm map data")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)",
    )
    args = parser.parse_args()

    data = prepare_data(config_path=args.config)

    print("\nQuick sanity check:")
    print(f"  Hospital index: {data['hospital_idx']}")
    print(f"  Zone indices: {data['zone_indices']}")
    print(f"  Road dist matrix shape: {data['road_dist'].shape}")
    print(f"  Sample road dist (hospital → zone 1): {data['road_dist'][0][1]:.0f}m")
    print(f"  Sample drone dist (hospital → zone 1): {data['euclidean_dist'][0][1]:.0f}m")
    print("\nAll good! Now run: python scripts/train.py")


if __name__ == "__main__":
    main()