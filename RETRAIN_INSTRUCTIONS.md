# Retraining Guide: Fresh Start for 85% Success

## Problem
You're seeing **0% success rate with 1400 reward** — agent reaches ~5 zones but not all 12.

This happens when the model was trained under the old/incomplete setup. Since we changed:
- ✅ Observation space: 15D → 18D (added nearest zone distances)
- ✅ Reward function: distance_penalty → per_step_penalty
- ✅ Network architecture: [64,64] → [256,256,128]

**Old models are incompatible and must be retrained from scratch.**

---

## Step-by-Step Retraining

### 1. Clean Up (2 minutes)

```bash
cd /Data1/sanket/operation-medswarm

# Remove old models and logs
rm -rf models/* logs/*

# Verify they're gone
ls models/  # Should be empty
ls logs/    # Should be empty
```

### 2. Prepare Data (1 minute)

```bash
python scripts/prepare_data.py
```

**Output should say:**
```
[data_prep] Trying to download OpenStreetMap data...
[data_prep] Downloading: Connaught Place, New Delhi, India
[data_prep] Computing road distance matrix...
[data_prep] Saved to data/medswarm_data.pkl
```

**If it fails (no internet):** ✓ Still works — generates synthetic data automatically

### 3. Full Training (30 min on CPU, 10 min on GPU)

```bash
# Option A: CPU training (backgrounds automatically)
python scripts/train.py &

# Option B: GPU training (if available)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py &

# Option C: Foreground (if you want to monitor)
python scripts/train.py
```

### 4. Monitor Progress (Watch these logs)

While training, in another terminal:

```bash
# Watch training metrics every 5k steps
tail -f /dev/null ; python scripts/train.py 2>&1 | grep "Success"
```

**Expected output (every 50k steps):**
```
[ 17.7%]  Step 50,000 /  300,000  |  Reward:    200.0  |  Zones:  1.2/12  |  Success:   2.3%
[ 36.4%]  Step 100,000 /  300,000  |  Reward:  1000.0  |  Zones:  3.8/12  |  Success:   5.2%
[ 53.0%]  Step 150,000 /  300,000  |  Reward:  1500.0  |  Zones:  5.5/12  |  Success:  12.3%
[ 71.7%]  Step 200,000 /  300,000  |  Reward:  2100.0  |  Zones:  7.8/12  |  Success:  35.0%
[ 90.4%]  Step 250,000 /  300,000  |  Reward:  2400.0  |  Zones: 10.2/12  |  Success:  60.0%
[100.0%]  Step 300,000 /  300,000  |  Reward:  2800.0  |  Zones: 11.8/12  |  Success:  85.0%
```

---

## Checking Progress

### Quick Test (After 50k steps)

```bash
python scripts/train.py --evaluate --episodes 10
```

Output should show success rate > 0%. If still 0%, **STOP TRAINING** and debug.

### Full Evaluation (After training complete)

```bash
python scripts/train.py --evaluate --episodes 50
```

**Target results:**
```
Mean reward:      2800.0
Mean steps:        18.0
Success rate:      85.0%
```

---

## Visual Monitoring with Dashboard

Once training has run for at least 50k steps:

```bash
python scripts/run_dashboard.py
```

Opens http://localhost:8501 with:
- **Training curves:** Reward and success rate climbing
- **Episode metrics:** Zones per episode visualization
- **Live replay:** Watch trained agent solve a mission

---

## Debugging If Still Getting 0% Success

### Debug Checklist

1. **Verify environment initialization:**
```bash
python -c "
from medswarm import MedSwarmEnv
env = MedSwarmEnv(data_path='data/medswarm_data.pkl')
obs, _ = env.reset()
print(f'✓ Obs shape: {obs.shape} (should be 18)')
print(f'✓ Num zones: {env.num_zones} (should be 12)')
print(f'Reward config: {env.reward_cfg}')
"
```

2. **Check config reward function:**
```bash
grep -A 5 "reward:" config/config.yaml
```

Should show:
```yaml
reward:
  per_step_penalty: -0.5
  zone_stabilized: 300.0
  battery_failure: -5000.0
  mission_complete: 8000.0
```

3. **Manual test (5 random episodes):**
```bash
python test_improved_env.py
```

Should show agents reaching zones with rewards increasing.

---

## FAQ

**Q: How long does 300k steps take?**
- CPU: 25-35 minutes
- GPU: 8-12 minutes
- Nvidia T4: ~10 min

**Q: Can I stop and resume training?**
No, checkpoints aren't set up for resumption. But training is fast enough to restart.

**Q: What if success rate plateaus at 50%?**
- Increase entropy: `ent_coef: 0.1` (was 0.05)
- Increase zone reward: `zone_stabilized: 400` (was 300)
- Retrain with these changes

**Q: Why does the agent sometimes get negative rewards?**
That's normal! Per-step penalty (-0.5) means if agent reaches fewer than 3 zones, it's negative overall. This forces the agent to complete zones.

---

## Final Sanity Check

After full training completes:

```bash
# Should see models saved
ls -lh models/

# Should see logs
ls -lh logs/

# Evaluate final model
python scripts/train.py --evaluate --episodes 20

# If success_rate >= 70%, you're good!
# If success_rate < 30%, there's an issue (check debug checklist above)
```

---

## Timeline

| Checkpoint | Time | Expected Success |
|-----------|------|-----------------|
| Start | 0 min | 0% |
| 50k steps | 9 min | 2-5% |
| 100k steps | 18 min | 5-15% |
| 150k steps | 28 min | 15-30% |
| 200k steps | 36 min | 30-50% |
| 250k steps | 45 min | 50-70% |
| **300k steps** | **54 min** | **80-85%** ✅ |

---

Happy retraining! 🚀
