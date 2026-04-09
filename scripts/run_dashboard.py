"""
run_dashboard.py

Step 3 of the pipeline — the fun part.

Launches the Gradio dashboard in your browser.

The dashboard has 4 tabs:
  - Mission Replay: run the trained agent live and watch it move step-by-step
  - Training Progress: reward over time as the agent learned
  - Environment Info: configuration and problem setup details
  - Help: quick start guide and troubleshooting

Usage:
    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --port 7860
    
Gradio benefits:
  - Lightweight and fast
  - No server overhead
  - Direct Python integration
  - Better reliability for live demos
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medswarm.utils.helpers import load_config


def main():
    parser = argparse.ArgumentParser(description="Launch MedSwarm Gradio dashboard")
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run dashboard on (default: 7860)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link (requires internet)",
    )
    args = parser.parse_args()

    # Import dashboard module
    dashboard_module = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "medswarm",
        "visualization",
        "dashboard.py",
    )
    
    print(f"""
╔════════════════════════════════════════════╗
║  🚑 Operation MedSwarm Dashboard (Gradio)  ║
╚════════════════════════════════════════════╝

🌐 Dashboard running at: http://localhost:{args.port}

📋 Features:
  • Mission Replay — Watch trained agent in action
  • Training Progress — View metrics and convergence
  • Environment Info — Detailed configuration
  • Help — Quick start guide & troubleshooting

⏹️  Press Ctrl+C to stop.
""")

    # Import and run the Gradio demo directly
    spec = __import__('importlib.util').util.spec_from_file_location("dashboard", dashboard_module)
    dashboard = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(dashboard)
    
    # Launch the Gradio app
    dashboard.demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()