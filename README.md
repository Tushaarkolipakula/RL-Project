# RL Dynamic Pricing for Financial Products:

Reinforcement learning agents that learn to price personal loans dynamically. The pricing decision is modelled as a Markov Decision Process: at each step the agent observes demand, customer credit score, market interest rate, and recent acceptance history, then adjusts the offered rate by one of five discrete increments. Reward is risk-adjusted profit on originated loans.

Five methods are implemented and compared:

| Method | Type | Description |
|---|---|---|
| Fixed Pricing | Baseline | Constant 8.5% regardless of customer or market |
| Rule-Based | Baseline | Credit-tier heuristic: 10.5% / 8.5% / 6.5% for poor / fair / good credit |
| Q-Learning | RL | Tabular Q-learning with Dyna-style replay planning |
| Policy Gradient | RL | REINFORCE with a two-layer softmax network (64 → 32 → 5) |
| DQN | RL | Double Deep Q-Network with experience replay and soft target updates |

All three RL agents are pure NumPy — no external deep-learning library is required.

---

## Results

Evaluated over 5 random seeds, 500 episodes each:

| Method | Profit/step | Accept Rate | Avg Rate |
|---|---|---|---|
| Fixed Pricing | +0.00417 ± 0.0010 | 8.1% | 8.46% |
| Rule-Based | +0.00434 ± 0.0010 | 8.9% | 8.44% |
| Q-Learning | +0.00658 ± 0.0008 | 23.5% | 6.15% |
| Policy Gradient | +0.00658 ± 0.0008 | 26.0% | 5.88% |
| DQN | **+0.00659 ± 0.0008** | 25.9% | 5.92% |

All three RL agents improve profit/step by ~58% over the fixed-rate baseline by discovering the optimal pricing zone (~6% offered rate, ~25% acceptance rate).

---

## Project Structure

```
.
├── outputs/                        # created on first run
│   ├── results.json
│   ├── results_plot.png
│   ├── data/
│   │   ├── synthetic_dataset.csv
│   │   └── dataset_summary.json
│   └── ablations/                  # created by ablation scripts
│       ├── ablation_sample_efficiency.json
│       ├── ablation_sample_efficiency.png
│       ├── ablation_components.json
│       └── ablation_components.png
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
│   ├── policies.py                 # Fixed, Rule-Based
│   ├── state.py                    # feature encoding and discretisation
│   └── training.py                 # training loops for all three RL agents
└── scripts/
    ├── ablation_components.py      # component ablation study
    ├── ablation_sample_efficiency.py  # sample efficiency ablation study
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

### Full experiment (5 seeds, 5000 episodes)
```bash
python scripts/train.py \
    --seeds 0 1 2 3 4 \
    --train-episodes 5000 \
    --eval-episodes 500 \
    --episode-length 200 \
    --output-dir outputs/
```

### Two-agent experiment (skip DQN)
```bash
python scripts/train.py --no-dqn
```

### Evaluate saved models without retraining
```bash
# Baselines only
python scripts/evaluate.py

# All five methods
python scripts/evaluate.py \
    --q-model   outputs/q_learning_seed0.pkl \
    --pg-model  outputs/policy_gradient_seed0.pkl \
    --dqn-model outputs/dqn_seed0.pkl

# Custom episode count and seeds
python scripts/evaluate.py \
    --dqn-model outputs/dqn_seed0.pkl \
    --episodes 200 \
    --seeds 0 1 2
```

### Generate the synthetic dataset separately
```bash
python scripts/generate_dataset.py --episodes 500 --output-dir outputs/
```

---

## Ablation Studies

Both ablation scripts are crash-safe: results are saved to a per-seed JSON file after every seed completes. Re-running the script automatically skips already-completed seeds and resumes from where it left off.

### Ablation 1 — Sample Efficiency

Evaluates each agent at episode checkpoints (100, 250, 500, 1000, 2000, 5000) to show how quickly each algorithm learns.

```bash
mkdir -p outputs/ablations

# Full scale (5 seeds, ~2 hours)
python scripts/ablation_sample_efficiency.py \
    --seeds 0 1 2 3 4 \
    --output-dir outputs/ablations

# Quick test (1 seed, fewer checkpoints)
python scripts/ablation_sample_efficiency.py \
    --seeds 0 \
    --checkpoints 100 500 1000 5000 \
    --output-dir outputs/ablations
```

**Key findings:**
- **DQN** converges immediately at episode 100 (profit/step = 0.00670, std = 0.00006 — tightest of any agent at any checkpoint). Experience replay and Polyak updates provide a stable learning signal from the start.
- **Q-Learning** improves steadily from 0.00609 at episode 100 to 0.00664 at episode 5000. The Dyna planning steps provide early sample efficiency.
- **Policy Gradient** shows the highest early variance (std = 0.00101 at episode 100, driven by a single seed reaching only 0.00392) before stabilising by episode 2000. REINFORCE's full Monte Carlo returns make it sensitive to unlucky early episodes.

### Ablation 3 — Component Ablations

Isolates the contribution of each architectural feature by removing it:

| Agent | Variant | Profit/step | Δ vs Full |
|---|---|---|---|
| Q-Learning | Full | 0.006575 ± 0.000075 | — |
| Q-Learning | No Dyna replay | 0.006637 ± 0.000063 | +0.000062 |
| Policy Gradient | Full | 0.006501 ± 0.000212 | — |
| Policy Gradient | No return normalisation | 0.006075 ± 0.000356 | −0.000426 (−6.5%) |
| DQN | Full | 0.006581 ± 0.000082 | — |
| DQN | Single network | 0.006582 ± 0.000121 | +0.000001 |
| DQN | n_step = 1 | 0.005853 ± 0.000154 | −0.000728 (−11.1%) |

```bash
# Full scale (5 seeds, 5000 episodes, ~1 hour)
python scripts/ablation_components.py \
    --seeds 0 1 2 3 4 \
    --train-episodes 5000 \
    --output-dir outputs/ablations

# Recommended (2 seeds, 2000 episodes, ~15 min)
python scripts/ablation_components.py \
    --seeds 0 1 \
    --train-episodes 2000 \
    --output-dir outputs/ablations

# Quick smoke-test
python scripts/ablation_components.py \
    --seeds 0 \
    --train-episodes 500 \
    --output-dir outputs/ablations
```

---

## MDP Formulation

**State** `sₜ = (demand, credit, market_rate, rolling_acceptance)`

| Field | Values | Description |
|---|---|---|
| `demand` | 0, 1, 2 | Low / medium / high demand level |
| `credit` | 1, 2, 3 | Poor / fair / good credit score category |
| `market_rate` | [0.03, 0.10] | Exogenous mean-reverting market benchmark rate |
| `rolling_acceptance` | [0, 1] | Fraction of loans accepted in the last 50 steps |

> **Note:** `market_rate` is an observed environmental signal (e.g. the central-bank policy rate), not the agent's offered rate. The agent's offered rate `rₜ ∈ [0.03, 0.15]` is a separate decision variable with a wider feasible range to allow credit-risk spreads above the market benchmark. The rule-based strategy's 10.5% poor-credit rate exceeds the market-rate ceiling (10%) but is within the offered-rate range — this is intentional and consistent with standard consumer-lending practice.

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

with `α = 1.138`, `β = 0.55`, `δ = 0.5`. This gives approximately 73% acceptance at 3% for a good-credit customer under medium demand.

---

## Agents

### Q-Learning (`rl_pricing/agents.py`)
Tabular Q-learning with Dyna-style replay planning. The state is discretised into a 5-tuple `(demand, credit, market_bucket, acceptance_bucket, rate_bucket)` giving 1,125 cells. Q-values are warm-started with the expected one-step reward as a prior, accelerating early learning.

Key hyperparameters: `α = 0.12`, `ε₀ = 1.0`, `ε_decay = 0.998`, `ε_min = 0.05`, `γ = 0.95`, `planning_steps = 12`, `replay_capacity = 50,000`.

### Policy Gradient (`rl_pricing/agents.py`)
REINFORCE with a two-layer softmax network. Input is a 5-dimensional normalised feature vector from `encode_features()`. Full-episode Monte Carlo returns are used; returns are normalised per episode for gradient stability. Ablation confirms this normalisation is critical — removing it drops performance by 6.5% and increases variance by 68%. Optimised with Adam.

Key hyperparameters: `lr = 2×10⁻³`, hidden layers 64 → 32, `γ = 0.95`.

### DQN (`rl_pricing/dqn_agent.py`)
Double Deep Q-Network with the following components:

- **Double DQN** — online network selects the next action; target network scores it. Removes maximisation bias from plain DQN. Ablation shows negligible effect on this simple MDP (Δ = +0.000001).
- **Experience replay** — circular buffer of 20,000 transitions. Mini-batches of 128 break temporal correlations in the gradient signal.
- **Soft target updates (Polyak)** — `φ⁻ ← τφ + (1−τ)φ⁻` with `τ = 0.005`, applied every step. Produces smoother target trajectories than periodic hard copies.
- **n-step returns** — 5-step returns before bootstrapping. Ablation shows this is the most critical DQN component — removing it (n=1) drops performance by 11.1%.
- **Huber loss** — quadratic for small TD errors, linear for large ones. Handles reward spikes from high-rate accepted loans.
- **Warmup** — 500 steps before gradient updates begin.
- **Adam optimiser** — consistent with the Policy Gradient agent.

Key hyperparameters: `lr = 1×10⁻³`, hidden layers 64 → 32, `ε₀ = 1.0`, `ε_decay = 0.997`, `ε_min = 0.05`, `γ = 0.95`, `batch_size = 128`, `τ = 0.005`, `n_step = 5`, `warmup_steps = 500`.

---

## The Synthetic Dataset

Running `train.py` (or `generate_dataset.py`) creates `outputs/data/synthetic_dataset.csv`. Each row is one loan application — one environment timestep — and records the **customer and market side only**:

| Column | Description |
|---|---|
| `episode`, `step`, `seed` | Position in the simulation |
| `demand`, `credit` | Customer characteristics |
| `market_rate` | Prevailing market benchmark rate at this step |
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
    dqn_tau=0.01,
    dqn_n_step=3,
)
run_full_experiment(config=config, output_dir="outputs/custom")
```

---

## Output Files

After a full training run, `outputs/` contains:

| File | Contents |
|---|---|
| `results.json` | Config, per-seed training rewards, and evaluation summary for all methods |
| `results_plot.png` | Learning curves and comparison bar charts |
| `q_learning_seed{N}.pkl` | Saved Q-table for seed N |
| `policy_gradient_seed{N}.pkl` | Saved PG network weights for seed N |
| `dqn_seed{N}.pkl` | Saved DQN network weights for seed N |
| `data/synthetic_dataset.csv` | 100,000-row dataset (500 episodes × 200 steps) |
| `data/dataset_summary.json` | Distribution statistics for the dataset |
| `ablations/ablation_sample_efficiency.json` | Per-seed results at each training checkpoint |
| `ablations/ablation_sample_efficiency.png` | Sample efficiency learning curves |
| `ablations/ablation_components.json` | Per-seed results for all seven ablated variants |
| `ablations/ablation_components.png` | Component ablation bar charts |

---

## Requirements

```
numpy
matplotlib   # optional — SVG fallback used if not installed
```

Python 3.10 or later is required (uses `match` syntax and `X | Y` type unions).
