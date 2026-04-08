#!/usr/bin/env python
"""
CLI Script for Dashboard
=========================
Launch the MedSwarm visualization dashboard.
"""

import argparse
import sys
import os
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Launch the MedSwarm visualization dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_dashboard.py
  python scripts/run_dashboard.py --port 8502
  python scripts/run_dashboard.py --theme light
        """
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501,
        help="Port to run the dashboard on (default: 8501)"
    )
    parser.add_argument(
        "--theme", 
        default="dark",
        choices=["dark", "light"],
        help="Dashboard theme (default: dark)"
    )
    parser.add_argument(
        "--browser", 
        action="store_true",
        default=True,
        help="Open browser automatically (default: True)"
    )
    parser.add_argument(
        "--no-browser", 
        action="store_true",
        help="Don't open browser automatically"
    )
    
    args = parser.parse_args()
    
    # Get the dashboard path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    dashboard_path = os.path.join(project_dir, "medswarm", "visualization", "dashboard.py")
    
    if not os.path.exists(dashboard_path):
        print(f"\n❌ Dashboard not found: {dashboard_path}\n")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🏥 MEDSWARM DASHBOARD")
    print("="*60)
    print(f"   Port: {args.port}")
    print(f"   Theme: {args.theme}")
    print(f"   URL: http://localhost:{args.port}")
    print("="*60 + "\n")
    
    # Build streamlit command
    cmd = [
        "streamlit", "run", dashboard_path,
        "--server.port", str(args.port),
        "--theme.base", args.theme,
    ]
    
    if args.no_browser:
        cmd.extend(["--server.headless", "true"])
    
    # Change to project directory and run
    os.chdir(project_dir)
    
    try:
        subprocess.run(cmd)
    except FileNotFoundError:
        print("\n❌ Streamlit not found. Install it with: pip install streamlit\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped.\n")


if __name__ == "__main__":
    main()
