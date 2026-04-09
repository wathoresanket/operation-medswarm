"""
dashboard.py

Streamlit dashboard with 3 tabs:
  1. Map — shows the mission map (hospital + zones)
  2. Training Curves — reward over time during training
  3. Mission Replay — run trained agent live and watch it move

Run with: python scripts/run_dashboard.py
Or directly: streamlit run medswarm/visualization/dashboard.py
"""

import os
import sys
import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# make sure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from medswarm.environment.medswarm_env import MedSwarmEnv
from medswarm.utils.helpers import load_config, load_map_data

# ---- Page Config ----
st.set_page_config(
    page_title="Operation MedSwarm",
    page_icon="🚑",
    layout="wide",
)

# ---- Load Config ----
CONFIG_PATH = "config/config.yaml"
try:
    config = load_config(CONFIG_PATH)
    DATA_PATH = config["data"]["output_path"]
    MODEL_PATH = config["training"]["model_save_path"]
    LOG_PATH = config["training"]["log_path"]
except Exception as e:
    st.error(f"Could not load config: {e}")
    st.stop()


# ---- Sidebar ----
st.sidebar.title("🚑 MedSwarm")
st.sidebar.markdown("**Urban Disaster Triage**")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Data:** `{DATA_PATH}`")
st.sidebar.markdown(f"**Model:** `{MODEL_PATH}`")
st.sidebar.markdown(f"**Logs:** `{LOG_PATH}`")

data_exists = os.path.exists(DATA_PATH)
model_exists = os.path.exists(MODEL_PATH + ".zip") or os.path.exists(MODEL_PATH)

st.sidebar.markdown("---")
st.sidebar.markdown("**Status:**")
st.sidebar.markdown(f"{'✅' if data_exists else '❌'} Map data")
st.sidebar.markdown(f"{'✅' if model_exists else '❌'} Trained model")

if not data_exists:
    st.sidebar.warning("Run `python scripts/prepare_data.py` first")
if not model_exists:
    st.sidebar.info("Run `python scripts/train.py` to train")


# ---- Header ----
st.title("🚑 Operation MedSwarm")
st.markdown(
    "**Hybrid Multi-Agent RL for Urban Disaster Triage** — "
    "CONVOKE 8.0 · ML Engineering Track · Learning Under Constraints"
)
st.markdown("---")

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["🗺️ Mission Map", "📈 Training Curves", "▶️ Mission Replay"])


# ====================================================
# TAB 1 — MAP
# ====================================================
with tab1:
    st.subheader("Mission Map — Connaught Place, New Delhi")

    if not data_exists:
        st.warning("No map data found. Run `python scripts/prepare_data.py` first.")
    else:
        try:
            map_data = load_map_data(DATA_PATH)
            coords = map_data["coords"]
            nodes = map_data["nodes"]
            hospital_idx = map_data["hospital_idx"]
            zone_indices = map_data["zone_indices"]

            lats, lons, names, colors, sizes = [], [], [], [], []

            for i, node in enumerate(nodes):
                lat, lon = coords[node]
                lats.append(lat)
                lons.append(lon)

                if i == hospital_idx:
                    names.append("🏥 Hospital (Base)")
                    colors.append("red")
                    sizes.append(20)
                else:
                    zone_num = zone_indices.index(node) + 1
                    names.append(f"Zone {zone_num}")
                    colors.append("orange")
                    sizes.append(12)

            fig = go.Figure()

            # add zones
            fig.add_trace(go.Scattermapbox(
                lat=[lats[i] for i in range(len(nodes)) if i != hospital_idx],
                lon=[lons[i] for i in range(len(nodes)) if i != hospital_idx],
                mode="markers+text",
                marker=dict(size=14, color="darkorange"),
                text=[names[i] for i in range(len(nodes)) if i != hospital_idx],
                textposition="top right",
                name="Triage Zones",
            ))

            # add hospital
            fig.add_trace(go.Scattermapbox(
                lat=[lats[hospital_idx]],
                lon=[lons[hospital_idx]],
                mode="markers+text",
                marker=dict(size=18, color="red", symbol="hospital"),
                text=["🏥 Hospital"],
                textposition="top right",
                name="Hospital",
            ))

            center_lat = np.mean(lats)
            center_lon = np.mean(lons)

            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=13),
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                height=500,
                legend=dict(x=0, y=1),
            )

            st.plotly_chart(fig, use_container_width=True)

            # stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Triage Zones", len(zone_indices))
            col2.metric("Max Drone Range", f"{config['environment']['max_battery']/1000:.1f} km")
            col3.metric("Data Source", map_data.get("source", "unknown").upper())

        except Exception as e:
            st.error(f"Error loading map: {e}")


# ====================================================
# TAB 2 — TRAINING CURVES
# ====================================================
with tab2:
    st.subheader("PPO Training Progress")

    eval_file = os.path.join(LOG_PATH, "PPO_1", "evaluations.npz")
    # also check directly in log_path
    if not os.path.exists(eval_file):
        eval_file = os.path.join(LOG_PATH, "evaluations.npz")

    if not os.path.exists(eval_file):
        st.info("No training logs found yet.")
        st.markdown("Train the model first:")
        st.code("python scripts/train.py", language="bash")

        # show what the curve should look like (demo)
        st.markdown("**What to expect after training:**")
        demo_steps = np.linspace(0, 300000, 30)
        # simulated learning curve (starts bad, improves over time)
        demo_rewards = -500 + 3000 * (1 - np.exp(-demo_steps / 80000)) + np.random.randn(30) * 100
        fig = px.line(x=demo_steps, y=demo_rewards,
                      labels={"x": "Timesteps", "y": "Mean Reward"},
                      title="Expected Training Curve (Demo — train to see real data)")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    else:
        data = np.load(eval_file)
        timesteps = data["timesteps"]
        results = data["results"]
        mean_r = results.mean(axis=1)
        std_r = results.std(axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.concatenate([timesteps, timesteps[::-1]]),
            y=np.concatenate([mean_r + std_r, (mean_r - std_r)[::-1]]),
            fill="toself",
            fillcolor="rgba(70, 130, 180, 0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name="±1 std dev",
        ))
        fig.add_trace(go.Scatter(
            x=timesteps, y=mean_r,
            line=dict(color="steelblue", width=2.5),
            name="Mean reward",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="PPO Training Curve — Mean Episode Reward",
            xaxis_title="Timesteps",
            yaxis_title="Mean Reward",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Final Mean Reward", f"{mean_r[-1]:.0f}")
        col2.metric("Peak Mean Reward", f"{mean_r.max():.0f}")
        col3.metric("Total Timesteps", f"{timesteps[-1]:,}")


# ====================================================
# TAB 3 — MISSION REPLAY
# ====================================================
with tab3:
    st.subheader("Live Mission Replay")

    if not data_exists:
        st.warning("Run data preparation first.")
    elif not model_exists:
        st.info("Train the model first, then come back here to see it in action.")
        st.code("python scripts/train.py", language="bash")
    else:
        st.markdown("Click the button to run one episode with the trained agent and watch it move.")

        use_random = st.checkbox("Use random agent instead (for comparison)", value=False)

        if st.button("▶️ Run Episode"):
            try:
                from stable_baselines3 import PPO

                env_config = {
                    "max_battery": config["environment"]["max_battery"],
                    "max_steps": config["environment"].get("max_steps_per_episode", 100),
                    "reward": config["environment"]["reward"],
                }

                env = MedSwarmEnv(data_path=DATA_PATH, config=env_config)
                map_data = load_map_data(DATA_PATH)

                if not use_random:
                    model = PPO.load(MODEL_PATH)

                obs, _ = env.reset()
                step_log = []
                done = False
                total_reward = 0.0

                while not done:
                    if use_random:
                        action = env.action_space.sample()
                    else:
                        action, _ = model.predict(obs, deterministic=True)

                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    step_log.append({
                        "Step": info["step"],
                        "Ambulance → Node": int(action[0]),
                        "Drone → Node": int(action[1]),
                        "Battery Left (m)": f"{info['battery_left']:.0f}",
                        "Zones Done": info["zones_done"],
                        "Step Reward": f"{reward:.1f}",
                    })
                    done = terminated or truncated

                # show results
                zones_done = info["zones_done"]
                success = zones_done == env.num_zones

                if success:
                    st.success(f"✅ Mission complete in {info['step']} steps! Total reward: {total_reward:.0f}")
                else:
                    st.error(f"❌ Mission failed. Zones stabilized: {zones_done}/12. Reward: {total_reward:.0f}")

                st.metric("Total Reward", f"{total_reward:.0f}")

                import pandas as pd
                st.dataframe(pd.DataFrame(step_log), use_container_width=True)

            except Exception as e:
                st.error(f"Error during replay: {e}")
                st.exception(e)