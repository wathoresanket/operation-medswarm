"""
run_dashboard.py

Step 3 of the pipeline — the fun part.

Launches the Streamlit dashboard in your browser.

The dashboard has 3 tabs:
  - Map: shows the hospital and all 12 triage zones on a real map
  - Training Curves: reward over time as the agent learned
  - Mission Replay: run the trained agent live and watch it move step-by-step

Usage:
    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --port 8502
"""

import sys
import os
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medswarm.utils.helpers import load_config


def main():
    parser = argparse.ArgumentParser(description="Launch MedSwarm Streamlit dashboard")
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run dashboard on (default: 8501)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
    )
    args = parser.parse_args()

    # path to the dashboard file
    dashboard_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "medswarm",
        "visualization",
        "dashboard.py",
    )

    print(f"Starting dashboard on http://localhost:{args.port}")
    print("Press Ctrl+C to stop.\n")

    cmd = [
        "streamlit", "run",
        dashboard_path,
        "--server.port", str(args.port),
        "--server.headless", "false",
    ]

    subprocess.run(cmd)


if __name__ == "__main__":
    main()