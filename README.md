# ICU Sepsis Treatment RL Agent

---

##  Overview
The agent acts as an AI doctor in a simulated ICU, observing a sepsis patient's
vital signs every timestep and choosing a treatment action to bring all vitals
into the safe recovery range. Unlike video game RL agents that move around a
screen, this agent is invisible and has no on-screen avatar. It represents an
AI clinical decision system acting behind the monitor, where the "mission" is
patient stabilisation rather than character movement.
The "game board" is the patient monitor, and winning means the patient survives.

---

##  Project Structure

```
Relebohile_RL_Summative/
├── environment/
│   ├── custom_env.py        # ICU Sepsis Gymnasium environment
│   └── rendering.py         # Pygame ICU patient monitor dashboard
├── training/
│   ├── dqn_training.py      # DQN — 10 hyperparameter runs
│   └── pg_training.py       # PPO + REINFORCE — 10 runs each
├── models/
│   ├── dqn/                 # Saved DQN models (run01–run10)
│   └── pg/
│       ├── ppo/             # Saved PPO models
│       └── reinforce/       # Saved REINFORCE models
├── logs/
│   ├── dqn/                 # dqn_results.json + dqn_comparison.png
│   └── pg/                  # pg_results.json + pg_comparison.png
├── analysis.ipynb           # Reward curves and model comparison plots

├── main.py                  # Run best model with Pygame GUI + terminal
├── random_agent.py          # Random agent demo (no model, no training)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/relebohile_pheko/Relebohile_RL_Summative.git
cd Relebohile_RL_Summative
pip install -r requirements.txt
```

---

##  How to Run

### 1. Watch random agent (no training needed)
```bash
python random_agent.py
```

### 2. Train DQN (10 hyperparameter runs)
```bash
python training/dqn_training.py
```

### 3. Train PPO + REINFORCE (10 runs each)
```bash
python training/pg_training.py
```

### 4. Run best agent with Pygame GUI
```bash
python main.py
```

### 5. Run specific model
```bash
python main.py --algo dqn --run 8
python main.py --algo ppo --run 5
python main.py --algo reinforce --run 6
python main.py --episodes 5
python main.py --no-render    # terminal only, no GUI
```

---

##  Environment Details

| Component | Detail |
|-----------|--------|
| Observation space | 6 normalised vitals: HR, BP, O2, Lactate, Infection, Time |
| Action space | 5 discrete treatment actions |
| Max steps per episode | 150 |
| Success condition | All 5 vitals in healthy range simultaneously (+200 reward) |
| Failure condition | BP < 60, O2 < 80, or Lactate > 9 (−100 reward) |
| BP fix | BP drifts down naturally above 130 mmHg to prevent overshoot |

### Actions
| ID | Treatment | Effect |
|----|-----------|--------|
| 0 | Do Nothing | Disease progresses; BP drifts down if above 130 |
| 1 | IV Fluids | Raises BP, reduces Lactate, lowers HR slightly |
| 2 | Antibiotics | Reduces Infection and Lactate, lowers HR |
| 3 | Oxygen Therapy | Raises O2 saturation, lowers HR slightly |
| 4 | Vasopressors | Strongly raises BP, slight HR increase |

### Reward Structure
| Event | Reward |
|-------|--------|
| Progress toward target vitals | +1.5 × improvement per step |
| Each vital in healthy range | +2.0 per vital per step |
| Step living penalty | −0.5 |
| Critical BP (< 75 mmHg) | −0.3 × (75 − BP) |
| Critical O2 (< 88%) | −0.5 × (88 − O2) |
| Patient fully recovers | +200 (terminal) |
| Patient dies | −100 (terminal) |

---

##  Algorithms and Results

| Algorithm | Type | Runs | Best Run | Mean Reward | Std |
|-----------|------|------|----------|-------------|-----|
| **PPO** | Policy Gradient | 10 | Run 5 | **672.01** | 57.89 |
| **DQN** | Value-Based | 10 | Run 8 | 598.55 | 297.48 |
| **REINFORCE** | Policy Gradient | 10 | Run 6 | 586.91 | 124.11 |

**DQN Best Hyperparameters (Run 8):**
- Learning Rate: 5e-4 | Gamma: 0.99 | Batch: 128 | Buffer: 200k | Exploration: 0.15

**REINFORCE Best Hyperparameters (Run 6):**
- Learning Rate: 1e-3 | Gamma: 0.99 | n_steps: 100 | Entropy: 0.05

**PPO Best Hyperparameters (Run 5):**
- Learning Rate: 3e-4 | Gamma: 0.99 | Clip: 0.2 | Entropy: 0.05 | n_steps: 256

---

##  Key Findings

- **PPO performed best overall**: Run 5 reached the top reward (672.01), with Run 6 close behind (669.35)
- **DQN remained strong but less stable**: best DQN reached 598.55, with larger variance across runs
- **REINFORCE achieved high reward but was highly sensitive**: best run reached 586.91, while some settings were negative
- **Entropy mattered for policy-gradient methods in this setup**: PPO Run 5 and REINFORCE Run 6 both used higher entropy regularisation (0.05)
- **Latest rollout check shows a policy-stalling pattern for PPO Run 5**: agent maintained safety but timed out in all 3 test episodes
- **The trained agent shows deliberate clinical behaviour**: oxygen therapy first, then antibiotics to clear infection, then IV fluids to raise BP — mirroring real ICU protocols

---

##  Live Simulation Example (PPO Run 5)

```
Episode 1: TIMEOUT | Steps: 150 | Reward: 619.6
Episode 2: TIMEOUT | Steps: 150 | Reward: 723.2
Episode 3: TIMEOUT | Steps: 150 | Reward: 618.0
Average: 653.59 ± 49.24 | Recovered: 0/3 | Deaths: 0/3
```

##  Random Baseline Simulation (Latest Run)

```
Episode 1: RECOVERED | Steps: 40 | Reward: 541.1
Episode 2: RECOVERED | Steps: 37 | Reward: 488.3
Episode 3: RECOVERED | Steps: 20 | Reward: 370.3
Average: 466.57 | Recovered: 3/3 | Deaths: 0/3
```

This side-by-side rollout evidence is included for transparency and helps motivate further reward-shaping/termination tuning.

---

##  Requirements

```
gymnasium>=0.29.0
stable-baselines3>=2.3.0
pygame>=2.5.0
numpy>=1.24.0
matplotlib>=3.7.0
torch>=2.0.0
tensorboard>=2.13.0
jupyter>=1.0.0
notebook>=7.0.0
```

Install: `pip install -r requirements.txt`

---

