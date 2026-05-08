"""Central configuration for the dynamic loan-pricing project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class PricingConfig:
    """All simulator, reward, training, and evaluation constants.

    
    ACCEPTANCE MODEL
    ─────────────────
    accept_beta=0.55, accept_alpha=1.138 preserves the paper's primary
    calibration anchor (P(accept @ 3%, credit=3) ≈ 0.73) while making
    customers moderately more price-sensitive than the original beta=0.467.
    This widens the reward gradient so RL agents have a clearer learning
    signal and the profit differences between methods are visible in plots.

    Q-LEARNING HYPERPARAMETERS
    ───────────────────────────
    alpha=0.12, epsilon_start=1.0, epsilon_decay=0.998, epsilon_min=0.05,
    planning_steps=12. See agents.py for full rationale.

    POLICY GRADIENT HYPERPARAMETERS
    ─────────────────────────────────
    pg_lr=2e-3, hidden layers 64→32. See agents.py for full rationale.

    DQN HYPERPARAMETERS
    ────────────────────
    dqn_lr=1e-3          : Adam learning rate — same as PolicyGradientAgent.
    dqn_hidden1/2=64,32  : Mirrors the PG network width for fair comparison.
    dqn_epsilon_*        : Same ε-greedy schedule as Q-Learning.
    dqn_replay_capacity  : Larger buffer than tabular replay (50 k → 20 k steps
                           is enough; 20 k avoids memory pressure on long runs).
    dqn_batch_size=64    : Standard mini-batch; balances gradient noise vs speed.
    dqn_target_update=20 : Hard copy every 20 episodes (~4 000 steps at ep_len
                           200); keeps targets stable without lagging too much.
    dqn_grad_steps=1     : One gradient step per env step — simple and stable.
    dqn_warmup_steps=500 : Fill the buffer before learning starts; prevents
                           degenerate early updates on a nearly empty buffer.
    """

    # ── Pricing bounds ───────────────────────────────────────────────────────
    r_min: float = 0.03
    r_max: float = 0.15
    cost_of_capital: float = 0.03
    loan_amount: float = 1.0
    risk_loss_scale: float = 0.006

    # Five discrete rate-adjustment actions (paper Eq. 2)
    actions: tuple[float, ...] = (-0.005, -0.0025, 0.0, 0.0025, 0.005)

    # ── Episode / evaluation ─────────────────────────────────────────────────
    episode_length: int = 200
    acceptance_window: int = 50
    train_episodes: int = 5000
    eval_episodes: int = 500
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    discount: float = 0.95

    # ── Customer & market simulation ─────────────────────────────────────────
    credit_categories: tuple[int, ...] = (1, 2, 3)
    credit_probs: tuple[float, ...] = (0.25, 0.50, 0.25)
    demand_levels: tuple[int, ...] = (0, 1, 2)
    demand_probs: tuple[float, ...] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    demand_logit_shift: tuple[float, ...] = (-0.25, 0.0, 0.25)
    market_eta: float = 0.05
    market_star: float = 0.05
    market_sigma: float = 0.002
    market_min: float = 0.03
    market_max: float = 0.10
    initial_market_low: float = 0.04
    initial_market_high: float = 0.07

    # Logistic acceptance model (paper Eq. 4)
    accept_alpha: float = 1.138
    accept_beta: float = 0.55
    accept_delta: float = 0.5

    # Default-risk model (paper Eq. 5): D(c) = sigmoid(-k*(c - cbar))
    default_k: float = 2.0
    default_cbar: float = 2.0

    # ── Baseline pricing settings ────────────────────────────────────────────
    fixed_rate: float = 0.085
    rule_poor_rate: float = 0.105
    rule_fair_rate: float = 0.085
    rule_good_rate: float = 0.065

    target_rate_grid_size: int = 241

    # ── Q-Learning hyperparameters ───────────────────────────────────────────
    q_alpha: float = 0.12
    q_epsilon_start: float = 1.0
    q_epsilon_min: float = 0.05
    q_epsilon_decay: float = 0.998
    replay_capacity: int = 50_000
    planning_steps: int = 12

    # ── Policy Gradient hyperparameters ─────────────────────────────────────
    pg_lr: float = 2e-3
    pg_hidden1: int = 64
    pg_hidden2: int = 32

    # ── DQN hyperparameters ──────────────────────────────────────────────────
# ── DQN hyperparameters ──────────────────────────────────────────────────

    dqn_lr:              float = 1e-3
    dqn_hidden1:         int   = 64
    dqn_hidden2:         int   = 32
    dqn_epsilon_start:   float = 1.0
    dqn_epsilon_min:     float = 0.05
    dqn_epsilon_decay:   float = 0.997  
    dqn_replay_capacity: int   = 20_000
    dqn_batch_size:      int   = 128     
    dqn_grad_steps:      int   = 1
    dqn_warmup_steps:    int   = 500
    dqn_tau:             float = 0.005
    dqn_n_step:          int   = 5

    # ── Derived properties ───────────────────────────────────────────────────

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    @property
    def rule_based_rates(self) -> dict[int, float]:
        return {
            1: self.rule_poor_rate,
            2: self.rule_fair_rate,
            3: self.rule_good_rate,
        }