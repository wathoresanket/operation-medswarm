# Operation MedSwarm
### Hybrid Multi-Agent Reinforcement Learning for Urban Disaster Triage

> Built for CONVOKE 8.0 — KnowledgeQuarry, CIC University of Delhi  
> Problem 02: Learning Under Constraints (ML Engineering Track)

---

## What is this project?

Imagine a massive disaster in **Connaught Place, New Delhi**. Roads are jammed, people need medical help fast. You have two vehicles:

- A **ground ambulance** — carries unlimited supplies but is stuck on roads
- A **medical drone** — can fly anywhere but has a **3km battery limit**

The question is: *how do you coordinate these two to reach all 12 disaster zones as fast as possible?*

This project trains an AI using **Reinforcement Learning (RL)** to figure out the best coordination strategy. It starts by knowing nothing, makes a ton of mistakes, and slowly learns what works — exactly like how you'd get better at a game the more you play.

---

## The Problem (plain English)

| | |
|---|---|
| **Scenario** | Urban disaster, 12 zones need medical supplies |
| **Location** | Connaught Place, New Delhi (real map data from OpenStreetMap) |
| **Agent 1** | Ground Ambulance — uses real roads, no range limit |
| **Agent 2** | Medical Drone — straight-line flight, max 3km battery |
| **Goal** | Stabilize all 12 zones with minimum total distance traveled |
| **Hard constraint** | Drone dies if battery runs out — mission fails |

Maps directly to the competition constraints:
- **C-01** Two agent types with different capabilities ✓  
- **C-02** Energy/distance limits modeled explicitly ✓  
- **C-03** Optimizes tasks completed + total cost ✓  
- **C-04** PPO learns and improves over 300,000 steps ✓  

---

## How it works

### Step 1 — Build the map
Download the real road network of Connaught Place from OpenStreetMap using `osmnx`. Then:
- Pick 1 node as the hospital (starting base)
- Pick 12 nodes as triage zones
- Pre-compute a **road distance matrix** for the ambulance (shortest path on roads)
- Pre-compute a **Euclidean distance matrix** for the drone (straight-line flight)

If there's no internet, it automatically generates realistic synthetic data.

### Step 2 — Define the RL environment
Model the problem as a **Markov Decision Process (MDP)**:

```
State  →  [ambulance_pos, drone_pos, battery_left (normalized), zone_1_done, ..., zone_12_done]
Action →  [which node ambulance goes to, which node drone goes to]  — 13 choices each
Reward →  -0.1 per meter traveled  +100 per zone stabilized  -2000 if drone battery dies  +5000 mission complete
```

### Step 3 — Train with PPO
Use **Proximal Policy Optimization (PPO)** from Stable-Baselines3. The agent runs thousands of simulated missions. After each one it tweaks its strategy a little to get a better reward next time.

After ~300K timesteps, it goes from failing 95% of missions to completing 85%+ successfully.

### Step 4 — Visualize
A Streamlit dashboard shows training curves, the map, and live mission replays.

---

## Project Structure

```
medswarm/
│
├── README.md                        ← you're reading this
├── requirements.txt                 ← pip packages
├── setup.py                         ← install medswarm as a local package
├── .gitignore
│
├── config/
│   └── config.yaml                  ← all settings in one place
│
├── medswarm/                        ← the actual Python package
│   ├── data/
│   │   └── data_prep.py             ← downloads OSM map, builds distance matrices
│   ├── environment/
│   │   └── medswarm_env.py          ← the Gymnasium RL environment (the "game")
│   ├── training/
│   │   └── trainer.py               ← PPO training loop + callbacks
│   ├── utils/
│   │   └── helpers.py               ← small utility functions
│   └── visualization/
│       └── dashboard.py             ← Streamlit dashboard
│
├── scripts/
│   ├── prepare_data.py              ← run this first
│   ├── train.py                     ← run this second
│   └── run_dashboard.py             ← run this to see results
│
├── data/                            ← auto-generated (gitignored)
├── models/                          ← saved models (gitignored)
└── logs/                            ← training logs (gitignored)
```

---

## Setup & Running

### Requirements
- Python 3.9 or newer
- ~4GB RAM
- Internet (optional — for real map download in step 1)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/medswarm.git
cd medswarm

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the medswarm package itself (editable mode)
pip install -e .
```

### Running the full pipeline

```bash
# Step 1: Download map data and build distance matrices (~1 min, needs internet)
# Falls back to synthetic data automatically if offline
python scripts/prepare_data.py

# Step 2: Train the PPO agent (~10-20 min depending on your machine)
python scripts/train.py

# Step 3: Launch the dashboard
python scripts/run_dashboard.py
# then open: http://localhost:8501
```

### Quick environment test (no training needed)

```python
from medswarm import MedSwarmEnv
import numpy as np

env = MedSwarmEnv(data_path="data/medswarm_data.pkl")
obs, info = env.reset()
print("Initial obs:", obs)

# random agent — just to verify the env works
for step in range(50):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        print(f"Episode ended at step {step+1}, reward: {reward:.1f}")
        break
```

---

## Results

| Metric | Random Agent | Trained Agent (300K steps) |
|--------|-------------|--------------------------|
| Mean Episode Reward | ~-500 | ~+2500 |
| Mission Success Rate | ~5% | ~85% |
| Avg Steps to Complete | 50+ | ~18 |
| Battery Failure Rate | ~40% | ~2% |

The agent learns to:
1. Send the drone to nearby zones first (battery conservation)
2. Use the ambulance for distant zones the drone can't reach
3. Coordinate so both agents are always useful — no idle time

---

## Configuration

Edit `config/config.yaml` to change anything. No need to touch the code.

```yaml
data:
  location: "Connaught Place, New Delhi, India"
  num_triage_zones: 12
  random_seed: 42

environment:
  max_battery: 3000.0       # meters — drone dies beyond this
  reward:
    distance_penalty: -0.1
    zone_stabilized: 100.0
    battery_failure: -2000.0
    mission_complete: 5000.0

training:
  total_timesteps: 300000
  learning_rate: 0.0003
  n_envs: 4
```

---

## Dependencies

| Package | What it's for |
|---------|--------------|
| `gymnasium` | Standard RL environment API |
| `stable-baselines3` | PPO implementation |
| `osmnx` | Download OpenStreetMap road networks |
| `networkx` | Shortest path computation |
| `numpy`, `pandas` | Data handling |
| `matplotlib`, `plotly` | Plotting |
| `streamlit` | Interactive dashboard |
| `pyyaml` | Config file parsing |
| `torch` | Neural network backend for PPO |

---

## License

MIT — do whatever you want with it.

---

*Made for CONVOKE 8.0 · KnowledgeQuarry · CIC, University of Delhi*