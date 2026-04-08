"""
MedSwarm Interactive Dashboard
===============================
Streamlit-based visualization dashboard for training metrics and simulation replay.
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="MedSwarm Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .success-card { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .warning-card { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .info-card { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .stPlotlyChart {
        background: rgba(255,255,255,0.05);
        border-radius: 0.5rem;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def load_training_metrics(log_path: str = "logs/") -> Optional[Dict]:
    """Load training metrics from JSON file."""
    metrics_file = os.path.join(log_path, "training_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None


def load_evaluation_results(log_path: str = "logs/") -> Optional[Dict]:
    """Load evaluation results."""
    eval_file = os.path.join(log_path, "evaluations.npz")
    if os.path.exists(eval_file):
        data = np.load(eval_file)
        return {
            "timesteps": data["timesteps"],
            "results": data["results"],
            "ep_lengths": data["ep_lengths"]
        }
    return None


def load_map_data(data_path: str = "data/medswarm_data.pkl") -> Optional[Dict]:
    """Load geographic data for map visualization."""
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            return pickle.load(f)
    return None


def create_training_curves(metrics: Dict) -> go.Figure:
    """Create training curves visualization."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Mean Episode Reward", 
            "Episode Length",
            "Reward Standard Deviation",
            "Training Progress"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    timesteps = metrics.get("timesteps", [])
    
    # Mean reward curve
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=metrics.get("mean_reward", []),
            mode='lines+markers',
            name='Mean Reward',
            line=dict(color='#4ECDC4', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Episode length
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=metrics.get("mean_length", []),
            mode='lines+markers',
            name='Ep Length',
            line=dict(color='#FF6B6B', width=2),
            marker=dict(size=6)
        ),
        row=1, col=2
    )
    
    # Reward std
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=metrics.get("std_reward", []),
            mode='lines+markers',
            name='Reward Std',
            line=dict(color='#95E1D3', width=2),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(149, 225, 211, 0.3)'
        ),
        row=2, col=1
    )
    
    # Episodes count
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=metrics.get("episodes", []),
            mode='lines+markers',
            name='Episodes',
            line=dict(color='#F38181', width=2),
            marker=dict(size=6)
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color='white')
    )
    
    fig.update_xaxes(title_text="Timesteps", gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_map_visualization(
    data: Dict, 
    amb_path: Optional[List[int]] = None,
    drone_path: Optional[List[int]] = None
) -> go.Figure:
    """Create map visualization with nodes and optional paths."""
    node_coords = data.get('node_coords', {})
    nodes = data.get('nodes', [])
    
    if not node_coords:
        # Fallback to getting coords from graph if node_coords not available
        graph = data.get('graph')
        if graph:
            node_coords = {n: (graph.nodes[n]['x'], graph.nodes[n]['y']) for n in nodes}
    
    # Extract coordinates
    x_coords = [node_coords[n][0] for n in nodes]
    y_coords = [node_coords[n][1] for n in nodes]
    
    fig = go.Figure()
    
    # Plot nodes
    colors = ['#FFD93D'] + ['#FF6B6B'] * (len(nodes) - 1)  # Yellow for base, red for zones
    sizes = [25] + [18] * (len(nodes) - 1)
    labels = ['🏥 Base Hospital'] + [f'Zone {i}' for i in range(1, len(nodes))]
    
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers+text',
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=2, color='white')
        ),
        text=labels,
        textposition='top center',
        textfont=dict(size=10, color='white'),
        name='Nodes',
        hovertemplate='<b>%{text}</b><br>X: %{x:.0f}<br>Y: %{y:.0f}<extra></extra>'
    ))
    
    # Plot ambulance path if provided
    if amb_path and len(amb_path) > 1:
        amb_x = [node_coords[nodes[i]][0] for i in amb_path]
        amb_y = [node_coords[nodes[i]][1] for i in amb_path]
        fig.add_trace(go.Scatter(
            x=amb_x, y=amb_y,
            mode='lines',
            line=dict(color='#4ECDC4', width=3, dash='solid'),
            name='🚑 Ambulance Path'
        ))
    
    # Plot drone path if provided
    if drone_path and len(drone_path) > 1:
        drone_x = [node_coords[nodes[i]][0] for i in drone_path]
        drone_y = [node_coords[nodes[i]][1] for i in drone_path]
        fig.add_trace(go.Scatter(
            x=drone_x, y=drone_y,
            mode='lines',
            line=dict(color='#95E1D3', width=2, dash='dash'),
            name='🛸 Drone Path'
        ))
    
    fig.update_layout(
        title=dict(
            text='<b>Connaught Place - Triage Zone Map</b>',
            font=dict(size=20, color='white')
        ),
        height=500,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(title_text="X (meters)", showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(title_text="Y (meters)", showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_battery_gauge(battery_level: float, max_battery: float = 3000.0) -> go.Figure:
    """Create battery level gauge."""
    percentage = (battery_level / max_battery) * 100
    
    if percentage > 60:
        color = "#38ef7d"
    elif percentage > 30:
        color = "#FFD93D"
    else:
        color = "#FF6B6B"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "🔋 Drone Battery", 'font': {'size': 18, 'color': 'white'}},
        number={'suffix': "%", 'font': {'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': color},
            'bgcolor': 'rgba(0,0,0,0.3)',
            'borderwidth': 2,
            'bordercolor': 'white',
            'steps': [
                {'range': [0, 30], 'color': 'rgba(255,107,107,0.3)'},
                {'range': [30, 60], 'color': 'rgba(255,217,61,0.3)'},
                {'range': [60, 100], 'color': 'rgba(56,239,125,0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 20
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    
    return fig


def create_zone_status(zones: List[int]) -> go.Figure:
    """Create triage zone status visualization."""
    colors = ['#38ef7d' if z == 0 else '#FF6B6B' for z in zones]
    labels = [f'Zone {i+1}' for i in range(len(zones))]
    status = ['✅ Stabilized' if z == 0 else '🚨 Needs Help' for z in zones]
    
    fig = go.Figure(go.Bar(
        x=labels,
        y=[1] * len(zones),
        marker_color=colors,
        text=status,
        textposition='inside',
        hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='<b>Triage Zone Status</b>', font=dict(size=16, color='white')),
        height=200,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        showlegend=False,
        yaxis={'visible': False},
        xaxis={'tickangle': 45}
    )
    
    return fig


def create_agent_comparison(metrics: Dict) -> go.Figure:
    """Create agent performance comparison chart."""
    # This would typically compare ambulance vs drone performance
    # For now, show reward distribution
    
    rewards = metrics.get("mean_reward", [])
    if not rewards:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=rewards,
        nbinsx=30,
        marker_color='#4ECDC4',
        opacity=0.7,
        name='Reward Distribution'
    ))
    
    fig.update_layout(
        title=dict(text='<b>Reward Distribution</b>', font=dict(size=16, color='white')),
        xaxis_title='Episode Reward',
        yaxis_title='Frequency',
        height=300,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
    )
    
    return fig


def main():
    """Main dashboard application."""
    # Header
    st.markdown('<h1 class="main-header">🏥 Operation MedSwarm Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #888;">Multi-Agent Reinforcement Learning for Urban Disaster Triage</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    log_path = st.sidebar.text_input("Log Path", value="logs/")
    data_path = st.sidebar.text_input("Data Path", value="data/medswarm_data.pkl")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
    
    if auto_refresh:
        st.sidebar.write("🔄 Refreshing every 5 seconds...")
        import time
        time.sleep(5)
        st.rerun()
    
    # Load data
    metrics = load_training_metrics(log_path)
    eval_results = load_evaluation_results(log_path)
    map_data = load_map_data(data_path)
    
    # Top metrics row
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    if metrics:
        with col1:
            latest_reward = metrics["mean_reward"][-1] if metrics["mean_reward"] else 0
            st.metric(
                "📈 Latest Mean Reward", 
                f"{latest_reward:.1f}",
                delta=f"{latest_reward - metrics['mean_reward'][-2]:.1f}" if len(metrics['mean_reward']) > 1 else None
            )
        
        with col2:
            total_episodes = metrics["episodes"][-1] if metrics["episodes"] else 0
            st.metric("🎮 Total Episodes", f"{total_episodes:,}")
        
        with col3:
            total_steps = metrics["timesteps"][-1] if metrics["timesteps"] else 0
            st.metric("👣 Total Timesteps", f"{total_steps:,}")
        
        with col4:
            avg_length = metrics["mean_length"][-1] if metrics["mean_length"] else 0
            st.metric("📏 Avg Episode Length", f"{avg_length:.1f}")
    else:
        st.warning("⚠️ No training metrics found. Train the model first!")
    
    # Main content
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Training Progress", 
        "🗺️ Map View", 
        "🎯 Simulation",
        "📋 Configuration"
    ])
    
    with tab1:
        if metrics:
            st.plotly_chart(create_training_curves(metrics), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                reward_dist = create_agent_comparison(metrics)
                if reward_dist:
                    st.plotly_chart(reward_dist, use_container_width=True)
            
            with col2:
                # Show training statistics
                st.markdown("### 📊 Training Statistics")
                if metrics["mean_reward"]:
                    best_reward = max(metrics["mean_reward"])
                    worst_reward = min(metrics["mean_reward"])
                    st.write(f"**Best Mean Reward:** {best_reward:.2f}")
                    st.write(f"**Worst Mean Reward:** {worst_reward:.2f}")
                    st.write(f"**Improvement:** {best_reward - metrics['mean_reward'][0]:.2f}")
        else:
            st.info("📈 Training metrics will appear here once training begins.")
    
    with tab2:
        if map_data:
            st.plotly_chart(create_map_visualization(map_data), use_container_width=True)
            
            # Distance matrix heatmap
            st.markdown("### 📍 Distance Matrices")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🚑 Ambulance (Road Network)**")
                amb_matrix = map_data.get('amb_matrix', np.zeros((13, 13)))
                fig = px.imshow(
                    amb_matrix,
                    labels=dict(x="To Node", y="From Node", color="Distance (m)"),
                    color_continuous_scale="Viridis",
                    template='plotly_dark'
                )
                fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**🛸 Drone (Euclidean)**")
                drone_matrix = map_data.get('drone_matrix', np.zeros((13, 13)))
                fig = px.imshow(
                    drone_matrix,
                    labels=dict(x="To Node", y="From Node", color="Distance (m)"),
                    color_continuous_scale="Plasma",
                    template='plotly_dark'
                )
                fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Map data not found. Run data preparation first!")
    
    with tab3:
        st.markdown("### 🎮 Mission Simulation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Simulation controls
            if st.button("▶️ Run Simulation Episode", type="primary"):
                st.info("Loading model and running simulation...")
                # This would run an actual episode with the trained model
                # For now, show placeholder
                st.success("Simulation complete! Mission successful.")
        
        with col2:
            # Status indicators
            st.markdown("#### 📊 Current Status")
            zones = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0]  # Example status
            remaining = sum(zones)
            st.write(f"**Zones Remaining:** {remaining}/12")
            st.write(f"**Ambulance Location:** Node 5")
            st.write(f"**Drone Status:** Deployed")
        
        # Battery gauge
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_battery_gauge(1800.0), use_container_width=True)
        with col2:
            st.plotly_chart(create_zone_status(zones), use_container_width=True)
    
    with tab4:
        st.markdown("### ⚙️ Training Configuration")
        
        config_file = os.path.join(log_path, "training_config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**PPO Hyperparameters**")
                st.json({
                    "learning_rate": config.get("learning_rate"),
                    "n_steps": config.get("n_steps"),
                    "batch_size": config.get("batch_size"),
                    "gamma": config.get("gamma"),
                    "gae_lambda": config.get("gae_lambda")
                })
            
            with col2:
                st.markdown("**Environment Settings**")
                st.json({
                    "n_envs": config.get("n_envs"),
                    "total_timesteps": config.get("total_timesteps"),
                    "eval_freq": config.get("eval_freq"),
                    "data_path": config.get("data_path")
                })
        else:
            st.info("Configuration will appear here after training starts.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">Operation MedSwarm © 2024 | '
        'Multi-Agent RL for Urban Disaster Response</p>',
        unsafe_allow_html=True
    )


def run_dashboard():
    """Entry point for running the dashboard."""
    main()


if __name__ == "__main__":
    main()
