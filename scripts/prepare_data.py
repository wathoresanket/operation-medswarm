#!/usr/bin/env python
"""
CLI Script for Data Preparation
================================
Downloads OpenStreetMap data and prepares distance matrices.
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medswarm.data.data_prep import DataPreparation


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MedSwarm geographic data from OpenStreetMap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/prepare_data.py
  python scripts/prepare_data.py --location "Manhattan, New York, USA" --zones 20
  python scripts/prepare_data.py --output data/custom_data.pkl
        """
    )
    
    parser.add_argument(
        "--location", 
        default="Connaught Place, New Delhi, India",
        help="Location name for OpenStreetMap query (default: Connaught Place, New Delhi)"
    )
    parser.add_argument(
        "--zones", 
        type=int, 
        default=12,
        help="Number of triage zones to create (default: 12)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducible node sampling (default: 42)"
    )
    parser.add_argument(
        "--output", 
        default="data/medswarm_data.pkl",
        help="Output path for the pickle file (default: data/medswarm_data.pkl)"
    )
    parser.add_argument(
        "--network-type",
        default="drive",
        choices=["drive", "walk", "bike", "all"],
        help="Type of street network to download (default: drive)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🗺️  MEDSWARM DATA PREPARATION")
    print("="*60)
    print(f"   Location: {args.location}")
    print(f"   Triage Zones: {args.zones}")
    print(f"   Random Seed: {args.seed}")
    print(f"   Output: {args.output}")
    print("="*60 + "\n")
    
    try:
        prep = DataPreparation(
            location=args.location,
            network_type=args.network_type,
            num_zones=args.zones,
            random_seed=args.seed
        )
        data = prep.prepare_and_save(args.output)
        
        print("\n" + "="*60)
        print("✅ DATA PREPARATION COMPLETE")
        print(f"   Nodes: {len(data.nodes)}")
        print(f"   Matrix Size: {data.amb_matrix.shape}")
        print(f"   Output: {args.output}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
