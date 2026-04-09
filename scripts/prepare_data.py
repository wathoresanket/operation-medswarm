"""
prepare_data.py

Step 1 of the pipeline.

Downloads the Connaught Place road network from OpenStreetMap
and builds the distance matrices that the RL environment needs.

No CLI args — config is loaded internally.

Usage:
    python scripts/prepare_data.py
"""

import sys
import os
import yaml

# make sure we can import from the medswarm package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medswarm.data.data_prep import prepare_data


def load_config():
    config_path = "config/config.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def main():
    print("Loading config...")
    config = load_config()

    print("Preparing data...")
    data = prepare_data(config)   # ✅ pass dict, NOT path

    print("\nQuick sanity check:")
    print(f"  Hospital index: {data['hospital_node']}")
    print(f"  Zone indices: {data['zone_nodes']}")
    print(f"  Road dist matrix shape: {data['ambulance_matrix'].shape}")
    print(f"  Sample road dist (hospital → zone 1): {data['ambulance_matrix'][0][1]:.0f}m")
    print(f"  Sample drone dist (hospital → zone 1): {data['drone_matrix'][0][1]:.0f}m")

    print("\n✅ All good! Now run: python scripts/train.py")


if __name__ == "__main__":
    main()