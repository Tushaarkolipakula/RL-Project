# RL Dynamic Pricing for Financial Products

Reinforcement learning agents that learn to price personal loans dynamically. The pricing decision is modelled as a Markov Decision Process: at each step the agent observes demand, customer credit score, market interest rate, and recent acceptance history, then adjusts the offered rate by one of five discrete increments. Reward is risk-adjusted profit on originated loans.

Five methods are implemented and compared:

| Method | Type | Description |
|---|---|---|
| Fixed Pricing | Baseline | Constant 8.5% regardless of customer or market |
| Rule-Based | Baseline | Credit-tier heuristic: 10.5% / 8.5% / 6.5% for poor / fair / good credit |
| Q-Learning | RL | Tabular Q-learning with Dyna-style replay planning |
| Policy Gradient | RL | REINFORCE with a two-layer softmax network (64 → 32 → 5) |
| DQN | RL | Double Deep Q-Network with experience replay and target network |

All three RL agents are pure NumPy — no external deep-learning library is required.

---

## Project Structure

```
.
├── configs/
│   └── default.json
├── outputs/                        # created on first run
│   └── full/
│       ├── results.json
│       ├── results_plot.png
│       └── data/
│           ├── synthetic_dataset.csv
│           └── dataset_summary.json
├── rl_pricing/
│   ├── __init__.py
│   ├── agents.py                   # Q-Learning, Policy Gradient
│   ├── config.py                   # all hyperparameters in one place
│   ├── dataset.py                  # synthetic dataset generation
│   ├── dqn_agent.py                # DQN (Double DQN, pure NumPy)
│   ├── environment.py              # LoanPricingEnv
│   ├── evaluation.py               # metrics and summary table
│   ├── mdp.py                      # acceptance model, reward, MDP helpers
│   ├── plotting.py                 # learning curves and comparison plots
│   ├── policies.py                 # Fixed, Rule-Based, Profit-Greedy, Balanced
│   ├── state.py                    # feature encoding and discretisation
│   └── training.py                 # training loops for all three RL agents
└── scripts/
    ├── evaluate.py                 # evaluate saved models without retraining
    ├── generate_dataset.py         # generate synthetic_dataset.csv standalone
    ├── smoke_test.py               # fast contract check
    └── train.py                    # main entry point
```

---

## Setup

```bash
pip install -r requirements.txt
```

Only NumPy is required for training and evaluation. Matplotlib is used for plots when installed; the code writes an SVG fallback if it is not.

---

## Running Experiments

### Quick smoke test
Verifies the environment interface and all agents run without errors:
```bash
python scripts/smoke_test.py
```

### Quick experiment (2 seeds, 500 episodes)
```bash
python scripts/train.py \
    --seeds 0 1 \
    --train-episodes 500 \
    --eval-episodes 50 \
    --episode-length 100 \
    --output-dir outputs/quick \
    --no-plot
```

### Paper-scale experiment (5 seeds, 5000 episodes)
```bash
python scripts/train.py \
    --seeds 0 1 2 3 4 \
    --train-episodes 5000 \
    --eval-episodes 500 \
    --episode-length 200 \
    --output-dir outputs/full
```

### Original two-agent experiment (skip DQN)
```bash
python scripts/train.py --no-dqn
```

### Evaluate saved models without retraining
```bash
# Baselines only
python scripts/evaluate.py

# All five methods
python scripts/evaluate.py \
    --q-model   outputs/full/q_learning_seed0.pkl \
    --pg-model  outputs/full/policy_gradient_seed0.pkl \
    --dqn-model outputs/full/dqn_seed0.pkl

# Custom episode count and seeds
python scripts/evaluate.py \
    --dqn-model outputs/full/dqn_seed0.pkl \
    --episodes 200 \
    --seeds 0 1 2
```

### Generate the synthetic dataset separately
```bash
python scripts/generate_dataset.py --episodes 500 --output-dir outputs/data
```

---

## MDP Formulation

**State** `sₜ = (demand, credit, market_rate, rolling_acceptance)`

| Field | Values | Description |
|---|---|---|
| `demand` | 0, 1, 2 | Low / medium / high demand level |
| `credit` | 1, 2, 3 | Poor / fair / good credit score category |
| `market_rate` | [0.03, 0.10] | Mean-reverting market interest rate |
| `rolling_acceptance` | [0, 1] | Fraction of loans accepted in the last 50 steps |

**Actions** — five discrete rate adjustments applied to the current offered rate:

```
{−0.50%, −0.25%, 0%, +0.25%, +0.50%}
```

The offered rate is clipped to [3%, 15%] after each adjustment. Because actions adjust the *current* rate rather than setting it directly, all learning agents track the rate internally — this keeps the environment interface clean while preserving the Markov property.

**Reward** — risk-adjusted profit on originated loans only:

```
Rₜ = Aₜ · (rₜ − c) · L  −  Aₜ · λ · D(creditₜ)
```

where `Aₜ ∈ {0,1}` is loan acceptance, `rₜ` is the offered rate, `c = 3%` is the cost of capital, `L = 1.0` is the normalised loan amount, and `D(credit)` is the default probability for the customer's credit tier. Risk cost is charged only on accepted loans.

**Customer acceptance** follows a calibrated logistic model:

```
P(accept) = sigmoid(α − β·rₜ + δ·credit + demand_shift)
```

with `α = 1.138`, `β = 0.55`, `δ = 0.5`. This gives approximately 73% acceptance at 3% for a good-credit customer.

---

## Agents

### Q-Learning (`rl_pricing/agents.py`)
Tabular Q-learning with Dyna-style replay planning. The state is discretised into a 5-tuple `(demand, credit, market_bucket, acceptance_bucket, rate_bucket)`. Q-values are initialised using the expected one-step reward as a prior, which dramatically accelerates early learning.

Key hyperparameters: `α = 0.12`, `ε₀ = 1.0`, `ε_decay = 0.998`, `ε_min = 0.05`, `γ = 0.95`, `planning_steps = 12`.

### Policy Gradient (`rl_pricing/agents.py`)
REINFORCE with a two-layer softmax network. The input is a 5-dimensional normalised feature vector from `encode_features()`. Full-episode Monte Carlo returns are used; returns are normalised per episode for gradient stability. Optimised with Adam.

Key hyperparameters: `lr = 2×10⁻³`, hidden layers 64 → 32, `γ = 0.95`.

### DQN (`rl_pricing/dqn_agent.py`)
Double Deep Q-Network. Uses the same 5-dimensional feature vector as Policy Gradient. Key components:

- **Double DQN** — the online network selects the next action; the target network scores it. This removes the maximisation bias that causes plain DQN to overestimate Q-values.
- **Experience replay** — a circular buffer of 20,000 transitions. Mini-batches of 64 break temporal correlations in the gradient signal.
- **Target network** — hard-copied from the online network every 20 episodes, stabilising the bootstrapped TD targets.
- **Huber loss** — quadratic for small TD errors, linear for large ones. Appropriate here because accepted high-rate loans create occasional large positive reward spikes.
- **Adam optimiser** — consistent with the Policy Gradient agent.

Key hyperparameters: `lr = 1×10⁻³`, hidden layers 64 → 32, `ε₀ = 1.0`, `ε_decay = 0.998`, `ε_min = 0.05`, `γ = 0.95`, `batch_size = 64`, `target_update = 20` episodes, `warmup_steps = 500`.

---

## The Synthetic Dataset

Running `train.py` (or `generate_dataset.py`) creates `outputs/data/synthetic_dataset.csv`. Each row is one loan application — one environment timestep — and records the **customer and market side only**:

| Column | Description |
|---|---|
| `episode`, `step`, `seed` | Position in the simulation |
| `demand`, `credit` | Customer characteristics |
| `market_rate` | Prevailing market rate at this step |
| `rolling_accept` | Rolling acceptance rate the agent would observe |
| `optimal_rate` | Rate that maximises expected reward for this customer |
| `p_accept_at_optimal` | Acceptance probability at the optimal rate |
| `default_prob` | Default probability for this credit tier |

The dataset deliberately **omits the agent's offered rate and loan acceptance outcome** because those depend on the policy. It is a characterisation of the environment population, useful for reporting dataset statistics, offline analysis, and reproducibility. Seeds for dataset generation start at 10,000 and never overlap with training seeds (0–4).

All three RL agents learn entirely through **online interaction with `LoanPricingEnv`** — the CSV is not used during training.

---

## Configuration

All hyperparameters live in `rl_pricing/config.py` as a frozen dataclass `PricingConfig`. Nothing is hardcoded elsewhere. To change a hyperparameter, either modify the dataclass default or pass a custom instance:

```python
from rl_pricing.config import PricingConfig
from rl_pricing.training import run_full_experiment

config = PricingConfig(
    train_episodes=2000,
    dqn_lr=5e-4,
    dqn_target_update=10,
)
run_full_experiment(config=config, output_dir="outputs/custom")
```

---

## Output Files

After a full training run, `outputs/full/` contains:

| File | Contents |
|---|---|
| `results.json` | Config, per-seed training rewards, and evaluation summary for all methods |
| `results_plot.png` | Learning curves and comparison bar charts |
| `q_learning_seed{N}.pkl` | Saved Q-table for seed N |
| `policy_gradient_seed{N}.pkl` | Saved PG network weights for seed N |
| `dqn_seed{N}.pkl` | Saved DQN network weights for seed N |
| `data/synthetic_dataset.csv` | 100,000-row dataset (500 episodes × 200 steps) |
| `data/dataset_summary.json` | Distribution statistics for the dataset |

---

## Requirements

```
numpy
matplotlib   # optional — SVG fallback used if not installed
```

Python 3.10 or later is required (uses `match` syntax and `X | Y` type unions).

---

