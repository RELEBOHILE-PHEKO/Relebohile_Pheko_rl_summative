# ICU Sepsis Treatment RL Agent

---

##  Mission
The agent acts as an AI doctor in a simulated ICU, observing a sepsis patient's
vital signs every timestep and choosing a treatment action to bring all vitals
into the safe recovery range. Unlike video game RL agents that move around a
screen, this agent is invisible; it represents an AI clinical decision system.
The "game board" is the patient monitor, and winning means the patient survives.

---

##  Project Structure

```
Relebohile_RL_Summative/
├── environment/
│   ├── custom_env.py        # ICU Sepsis Gymnasium environment
│   └── rendering.py         # Pygame ICU patient monitor dashboard
├── training/
│   ├── train_dqn.py         # DQN — 10 hyperparameter runs
│   └── train_pg.py          # PPO + REINFORCE — 10 runs each
├── models/
│   ├── dqn/                 # Saved DQN models (run01–run10)
│   └── pg/
│       ├── ppo/             # Saved PPO models
│       └── reinforce/       # Saved REINFORCE models
├── logs/
│   ├── dqn/                 # dqn_results.json + dqn_comparison.png
│   └── pg/                  # pg_results.json + pg_comparison.png
├── notebooks/
│   ├── analysis.ipynb       # Reward curves and model comparison plots
│   
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
python training/train_dqn.py
```

### 3. Train PPO + REINFORCE (10 runs each)
```bash
python training/train_pg.py
```

### 4. Run best agent with Pygame GUI
```bash
python main.py
```

### 5. Run specific model
```bash
python main.py --algo dqn --run 1
python main.py --algo ppo --run 4
python main.py --algo reinforce --run 9
python main.py --episodes 5
python main.py --no-render    # terminal only, no GUI
```

---

##  Environment Details

| Component | Detail |
|-----------|--------|
| Observation space | 6 normalised vitals: HR, BP, O2, Lactate, Infection, Time |
| Action space | 5 discrete treatment actions |
| Max steps per episode | 100 |
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
| **DQN** | Value-Based | 10 | Run 1 | **570.13** | 79.64 |
| **REINFORCE** | Policy Gradient | 10 | Run 9 | 495.07 | 119.98 |
| **PPO** | Policy Gradient | 10 | Run 4 | 351.19 | 59.32 |

**DQN Best Hyperparameters (Run 1):**
- Learning Rate: 1e-4 | Gamma: 0.99 | Batch: 64 | Buffer: 100k | Exploration: 0.15

**REINFORCE Best Hyperparameters (Run 9):**
- Learning Rate: 2e-3 | Gamma: 0.99 | n_steps: 50 | Entropy: 0.01

**PPO Best Hyperparameters (Run 4):**
- Learning Rate: 3e-4 | Gamma: 0.95 | Clip: 0.2 | Entropy: 0.01 | n_steps: 256

---

##  Key Findings

- **DQN performed best**: experience replay and target networks provide stable learning
- **REINFORCE surprised** : Run 9 with shorter rollouts (n_steps=50) beat all PPO runs
- **PPO underperformed**:conservative updates were too slow within 100k timesteps
- **Lower gamma (0.95)** improved both PPO and REINFORCE: immediate vital stabilisation matters more than long-term planning in sepsis
- **The trained agent shows deliberate clinical behaviour**: oxygen therapy first, then antibiotics to clear infection, then IV fluids to raise BP — mirroring real ICU protocols

---

##  Live Simulation Results (DQN Run 1)

```
Episode 1: TIMEOUT  | Steps: 100 | Reward: 674.4
Episode 2: RECOVERED | Steps:  62 | Reward: 642.7 
Episode 3: RECOVERED | Steps:  36 | Reward: 498.4 
Average: 605.18 ± 76.57 | Recovered: 2/3 | Deaths: 0/3
```

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

