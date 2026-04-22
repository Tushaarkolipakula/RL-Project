"""Synthetic dataset generation for the dynamic loan-pricing project.

WHY THIS EXISTS
───────────────
The environment generates customer loan applications on the fly during
training. Saving a fixed dataset serves three purposes:

  1. Reproducibility — evaluation always runs on the same customers,
     so results.json numbers are stable across re-runs.
  2. Separation of concerns — training data and test data are distinct.
     Agents train on episodes with seeds 0-4; the held-out test set uses
     a different seed range (10000+) that was never seen during training.
  3. Report credibility — you can state "evaluated on 100 000 synthetic
     loan applications" and show the dataset statistics.

DATASET STRUCTURE
──────────────────
Each row represents one loan application — one timestep in the environment.
The agent is not involved; this is purely the customer/market side.

Columns:
  episode          int    which evaluation episode (0-based)
  step             int    timestep within the episode (0 to episode_length-1)
  seed             int    RNG seed used for this episode
  demand           int    0=low, 1=medium, 2=high
  credit           int    1=poor, 2=fair, 3=good
  market_rate      float  prevailing market interest rate
  rolling_accept   float  rolling acceptance rate over last 50 steps
  p_accept_at_optimal  float  acceptance prob if agent offered the optimal rate
  optimal_rate     float  rate that maximises expected reward for this customer
  default_prob     float  probability this customer defaults

The dataset intentionally does NOT include the agent's offered rate or
whether the loan was accepted — those depend on the policy being evaluated
and belong in results.json, not the dataset.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from rl_pricing.config import DEFAULT_OUTPUT_DIR, PricingConfig
from rl_pricing.mdp import (
    acceptance_probability,
    default_probability,
    expected_reward,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core generation
# ─────────────────────────────────────────────────────────────────────────────

def _optimal_rate(credit: int, demand: int, cfg: PricingConfig) -> float:
    """Rate that maximises expected one-step reward for this customer."""
    rates = np.linspace(cfg.r_min, cfg.r_max, 241)
    rewards = [expected_reward(r, credit, demand, cfg).expected_reward for r in rates]
    return float(rates[int(np.argmax(rewards))])


def generate_dataset(
    config: PricingConfig | None = None,
    n_episodes: int = 500,
    seed_offset: int = 10_000,
) -> list[dict]:
    """Generate a list of loan-application records.

    Parameters
    ----------
    config:
        Pricing configuration.  Uses default if None.
    n_episodes:
        Number of evaluation episodes to simulate.
    seed_offset:
        RNG seeds start at seed_offset so the test set never overlaps with
        training seeds (which run from 0 upward).

    Returns
    -------
    List of dicts, one per timestep across all episodes.
    """
    cfg = config or PricingConfig()
    records: list[dict] = []

    for ep in range(n_episodes):
        seed = seed_offset + ep
        rng = np.random.default_rng(seed)

        # Episode-level initialisation (mirrors LoanPricingEnv.reset)
        market_rate = float(rng.uniform(cfg.initial_market_low, cfg.initial_market_high))
        demand = int(rng.choice(cfg.demand_levels, p=cfg.demand_probs))
        credit = int(rng.choice(cfg.credit_categories, p=cfg.credit_probs))
        acceptance_history: list[int] = []

        for step in range(cfg.episode_length):
            # Rolling acceptance (what the agent would observe)
            window = acceptance_history[-cfg.acceptance_window:]
            rolling = float(np.mean(window)) if window else 0.5

            # Customer-side quantities (independent of agent's rate choice)
            opt_r = _optimal_rate(credit, demand, cfg)
            p_opt = acceptance_probability(opt_r, credit, demand, cfg)
            d_prob = default_probability(credit, cfg)

            records.append({
                "episode":            ep,
                "step":               step,
                "seed":               seed,
                "demand":             demand,
                "credit":             credit,
                "market_rate":        round(market_rate, 6),
                "rolling_accept":     round(rolling, 4),
                "optimal_rate":       round(opt_r, 4),
                "p_accept_at_optimal": round(p_opt, 4),
                "default_prob":       round(d_prob, 4),
            })

            # Placeholder acceptance for rolling window
            # Use p_accept at optimal rate as a neutral reference
            acceptance_history.append(int(rng.random() < p_opt))

            # Market and customer transition (mirrors LoanPricingEnv._transition)
            noise = float(rng.normal(0.0, cfg.market_sigma))
            market_rate = float(np.clip(
                market_rate + cfg.market_eta * (cfg.market_star - market_rate) + noise,
                cfg.market_min, cfg.market_max,
            ))
            demand = int(rng.choice(cfg.demand_levels, p=cfg.demand_probs))
            credit = int(rng.choice(cfg.credit_categories, p=cfg.credit_probs))

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Saving helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(records: list[dict], path: str | Path) -> Path:
    """Write the dataset to a CSV file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        out.write_text("")
        return out
    fieldnames = list(records[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return out


def save_summary(records: list[dict], path: str | Path) -> Path:
    """Write a human-readable JSON summary of dataset statistics."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    demands = [r["demand"] for r in records]
    credits = [r["credit"] for r in records]
    market_rates = [r["market_rate"] for r in records]
    opt_rates = [r["optimal_rate"] for r in records]
    default_probs = [r["default_prob"] for r in records]
    p_opts = [r["p_accept_at_optimal"] for r in records]

    n_episodes = max(r["episode"] for r in records) + 1
    ep_length = max(r["step"] for r in records) + 1

    summary = {
        "total_applications": len(records),
        "episodes": n_episodes,
        "episode_length": ep_length,
        "demand_distribution": {
            "low (0)":    round(demands.count(0) / len(demands), 3),
            "medium (1)": round(demands.count(1) / len(demands), 3),
            "high (2)":   round(demands.count(2) / len(demands), 3),
        },
        "credit_distribution": {
            "poor (1)":  round(credits.count(1) / len(credits), 3),
            "fair (2)":  round(credits.count(2) / len(credits), 3),
            "good (3)":  round(credits.count(3) / len(credits), 3),
        },
        "market_rate": {
            "mean":  round(float(np.mean(market_rates)), 4),
            "std":   round(float(np.std(market_rates)), 4),
            "min":   round(float(np.min(market_rates)), 4),
            "max":   round(float(np.max(market_rates)), 4),
        },
        "optimal_rate_pct": {
            "mean":  round(float(np.mean(opt_rates)) * 100, 2),
            "std":   round(float(np.std(opt_rates)) * 100, 2),
            "min":   round(float(np.min(opt_rates)) * 100, 2),
            "max":   round(float(np.max(opt_rates)) * 100, 2),
        },
        "default_probability": {
            "mean":  round(float(np.mean(default_probs)), 4),
            "std":   round(float(np.std(default_probs)), 4),
        },
        "acceptance_at_optimal_rate": {
            "mean":  round(float(np.mean(p_opts)), 4),
            "std":   round(float(np.std(p_opts)), 4),
        },
    }

    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Convenience entry point (called from scripts/generate_dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_and_save(
    output_dir: str | Path | None = None,
    n_episodes: int = 500,
    config: PricingConfig | None = None,
) -> tuple[Path, Path]:
    """Generate the dataset and write both CSV and summary JSON.

    Returns (csv_path, summary_path).
    """
    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR) / "data"
    cfg = config or PricingConfig()

    print(f"Generating {n_episodes} episodes × {cfg.episode_length} steps "
          f"= {n_episodes * cfg.episode_length:,} loan applications …")

    records = generate_dataset(cfg, n_episodes=n_episodes)

    csv_path = save_csv(records, out_dir / "synthetic_dataset.csv")
    summary_path = save_summary(records, out_dir / "dataset_summary.json")

    print(f"Saved dataset  → {csv_path}  ({len(records):,} rows)")
    print(f"Saved summary  → {summary_path}")
    return csv_path, summary_path