"""
dashboard.py - Gradio Dashboard for Operation MedSwarm

Interactive visualization of:
  1. Live mission replay (trained agent in action)
  2. Training metrics (reward, zones, success rate)
  3. Environment configuration

Run with: python scripts/run_dashboard.py
"""

import os
import sys
import numpy as np
import gradio as gr
import plotly.graph_objects as go
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from medswarm.environment.medswarm_env import MedSwarmEnv
from medswarm.utils.helpers import load_config

# ---- Load Configuration ----
try:
    config = load_config("config/config.yaml")
    DATA_PATH = config["data"]["output_path"]
    MODEL_PATH = config["training"]["model_save_path"]
    LOG_PATH = config["training"]["log_path"]
except Exception as e:
    print(f"Error loading config: {e}")
    sys.exit(1)

# ---- Check if trained model exists ----
MODEL_BEST_PATH = f"{MODEL_PATH}_best/best_model"
MODEL_EXISTS = Path(MODEL_BEST_PATH + ".zip").exists()

# ---- Load trained model (if exists) ----
if MODEL_EXISTS:
    try:
        from stable_baselines3 import PPO
        trained_model = PPO.load(MODEL_BEST_PATH)
    except Exception:
        trained_model = None
        print("Warning: Could not load model")
else:
    trained_model = None
    print("Note: No trained model found. Run training first.")

# ---- Load environment ----
try:
    env_config = {
        "max_battery": config["environment"]["max_battery"],
        "max_steps": config["environment"].get("max_steps_per_episode", 200),
        "reward": config["environment"]["reward"],
    }
    env = MedSwarmEnv(data_path=DATA_PATH, config=env_config)
except Exception:
    print("Error loading environment. Check config and data path.")
    env = None


# ============================================================================
# FUNCTION 1: Live Mission Replay
# ============================================================================

def run_mission_replay(num_steps: int = 200) -> tuple:
    """
    Run a single episode with the trained agent and return visualization.
    
    Returns:
        (status_text, positions_plot, rewards_plot)
    """
    if env is None:
        return "Error: Environment not loaded", None, None
    
    if trained_model is None:
        return (
            "❌ No trained model found. Run training first:\n"
            "python scripts/train.py",
            None,
            None
        )
    
    # Reset environment
    obs, _ = env.reset()
    
    # Track trajectory
    amb_positions = [env._ambulance_pos]
    drone_positions = [env._drone_pos]
    episode_rewards = []
    zones_done_over_time = [0]
    
    total_reward = 0
    zones_completed = 0
    battery_depleted = False
    step = 0
    
    for step in range(num_steps):
        # Get action from trained model (stochastic, matching training)
        action, _ = trained_model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Track metrics
        total_reward += reward
        episode_rewards.append(reward)
        zones_completed = info["zones_done"]
        zones_done_over_time.append(zones_completed)
        
        amb_positions.append(env._ambulance_pos)
        drone_positions.append(env._drone_pos)
        
        if info["battery_left"] <= 0:
            battery_depleted = True
        
        if terminated or truncated:
            break
    
    # Build status text
    success = "✅ SUCCESS!" if zones_completed == env.num_zones else "❌ INCOMPLETE"
    status = f"""
{success}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Episode Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Zones Completed: {zones_completed}/{env.num_zones}
💰 Total Reward: {total_reward:.1f}
⏱️  Steps Taken: {step+1}/{num_steps}
🔋 Battery Status: {"DEPLETED ❌" if battery_depleted else f"{env._battery_left:.0f}m remaining ✅"}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    
    # Create positions plot
    fig_pos = go.Figure()
    fig_pos.add_trace(go.Scatter(
        y=amb_positions,
        mode='lines+markers',
        name='Ambulance Position',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    fig_pos.add_trace(go.Scatter(
        y=drone_positions,
        mode='lines+markers',
        name='Drone Position',
        line=dict(color='red', width=2),
        marker=dict(size=6)
    ))
    fig_pos.update_layout(
        title="Agent Positions Over Time",
        xaxis_title="Step",
        yaxis_title="Node ID (0=Hospital, 1-12=Zones)",
        hovermode='x unified',
        height=400
    )
    
    # Create rewards plot
    fig_reward = go.Figure()
    fig_reward.add_trace(go.Scatter(
        y=np.cumsum(episode_rewards),
        mode='lines',
        name='Cumulative Reward',
        line=dict(color='green', width=2),
        fill='tozeroy'
    ))
    fig_reward.update_layout(
        title="Cumulative Reward Over Time",
        xaxis_title="Step",
        yaxis_title="Cumulative Reward",
        hovermode='x',
        height=400
    )
    
    return status, fig_pos, fig_reward


# ============================================================================
# FUNCTION 2: Training Metrics
# ============================================================================

def load_training_metrics() -> str:
    """
    Load and display training metrics from logs.
    """
    if not Path(LOG_PATH).exists():
        return "⚠️ No training logs found yet. Run training first:\npython scripts/train.py"
    
    try:
        # Look for evaluations file
        eval_file = Path(LOG_PATH) / "evaluations.npz"
        if eval_file.exists():
            data = np.load(eval_file, allow_pickle=True)
            timesteps = data['timesteps']
            results = data['results']
            
            if len(timesteps) > 0:
                latest_idx = -1
                latest_timestep = timesteps[latest_idx]
                latest_reward = results[latest_idx][0] if len(results[latest_idx]) > 0 else 0
                
                metrics = f"""
📈 Training Metrics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Latest Metrics (Step {latest_timestep:,})
  • Mean Reward: {latest_reward:.1f}
  • Training Progress: {(latest_timestep / 300000) * 100:.1f}%
  
✅ Expected Performance:
  • 50k steps: 5-10% success rate
  • 100k steps: 20-30% success rate
  • 200k steps: 60-75% success rate
  • 300k steps: 80-85% success rate
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                """
                return metrics
    except Exception:
        pass
    
    return "ℹ️ Training in progress or no evaluation data yet."


# ============================================================================
# FUNCTION 3: Environment Info
# ============================================================================

def get_environment_info() -> str:
    """
    Display environment configuration and statistics.
    """
    if env is None:
        return "Error: Environment not loaded"
    
    info = f"""
🗺️  Environment Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📍 Location: {config['data']['location']}
🏥 Hospital: Node 0 (starting point)
🚨 Disaster Zones: 12 zones (nodes 1-12)
🚑 Total Nodes: 13

🚑 Ambulance
  • Type: Ground vehicle
  • Movement: Real road networks (shortest path)
  • Range: Unlimited
  • Speed: Varies by road distance

🚁 Drone
  • Type: Aerial vehicle
  • Movement: Straight-line flight (Euclidean)
  • Battery: {config['environment']['max_battery']:.0f}m max
  • Critical: Battery depletion = mission failure

⏱️  Episode Settings
  • Max steps: {config['environment']['max_steps_per_episode']}
  • Parallel envs: {config['training']['n_envs']}

💰 Reward Structure
  • Per-step penalty: {config['environment']['reward']['per_step_penalty']:.1f}
  • Zone stabilized: +{config['environment']['reward']['zone_stabilized']:.1f}
  • Battery failure: {config['environment']['reward']['battery_failure']:.1f}
  • Mission complete: +{config['environment']['reward']['mission_complete']:.1f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    return info


# ============================================================================
# Build Gradio Interface
# ============================================================================

with gr.Blocks(title="Operation MedSwarm", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🚑 Operation MedSwarm
    ### Multi-Agent RL for Urban Disaster Triage
    """)
    
    with gr.Tabs():
        
        # ---- TAB 1: Mission Replay ----
        with gr.Tab("🎬 Mission Replay"):
            gr.Markdown("Watch the trained agent solve a disaster response mission in real-time.")
            
            with gr.Row():
                replay_btn = gr.Button("▶️ Run Mission", size="lg")
            
            status_output = gr.Textbox(
                label="Mission Status",
                lines=12,
                interactive=False,
                value="Click 'Run Mission' to start..."
            )
            
            with gr.Row():
                pos_plot = gr.Plot(label="Agent Positions")
                reward_plot = gr.Plot(label="Cumulative Reward")
            
            replay_btn.click(
                fn=run_mission_replay,
                inputs=[],
                outputs=[status_output, pos_plot, reward_plot]
            )
        
        # ---- TAB 2: Training Progress ----
        with gr.Tab("📊 Training Progress"):
            gr.Markdown("View training metrics and convergence.")
            
            refresh_btn = gr.Button("🔄 Refresh Metrics")
            metrics_output = gr.Textbox(
                label="Training Metrics",
                lines=15,
                interactive=False,
                value=load_training_metrics()
            )
            
            refresh_btn.click(
                fn=load_training_metrics,
                inputs=[],
                outputs=[metrics_output]
            )
        
        # ---- TAB 3: Environment Info ----
        with gr.Tab("ℹ️ Environment Info"):
            gr.Markdown("Detailed information about the problem setup and configuration.")
            
            env_info = gr.Textbox(
                label="Environment Configuration",
                lines=20,
                interactive=False,
                value=get_environment_info()
            )
        
        # ---- TAB 4: Help ----
        with gr.Tab("❓ Help"):
            gr.Markdown("""
            ## Quick Start
            
            ### 1. Prepare Data
            ```bash
            python scripts/prepare_data.py
            ```
            
            ### 2. Train Agent
            ```bash
            python scripts/train.py  # ~30 min on CPU
            ```
            
            ### 3. View Dashboard
            ```bash
            python scripts/run_dashboard.py  # Already running!
            ```
            
            ## Expected Results
            - **50k steps**: 5-10% success rate
            - **100k steps**: 20-30% success rate
            - **200k steps**: 60-75% success rate
            - **300k steps**: 80-85% success rate
            
            ## Troubleshooting
            
            **Q: Dashboard shows "No trained model"**
            - A: Run `python scripts/train.py` first
            
            **Q: Mission replay not working**
            - A: Ensure model is fully trained (300k steps)
            
            **Q: Zones not being completed**
            - A: Training might be incomplete. Check progress in training tab.
            """)


# ============================================================================
# Launch
# ============================================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )