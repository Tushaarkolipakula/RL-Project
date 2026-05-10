"""Ablation 1 — Sample Efficiency.

Trains Q-Learning, Policy Gradient, and DQN agents and evaluates each at
fixed episode checkpoints. Shows how quickly each agent learns.

Crash-safe: results are saved to a per-seed JSON file after every seed
completes. If the script crashes, re-running it automatically skips seeds
that already have a saved file and picks up from where it left off.

Checkpoints: 100, 250, 500, 1000, 2000, 5000 episodes.

Usage
-----
# Default (2 seeds, all checkpoints)
python scripts/ablation_sample_efficiency.py

# Full paper-scale
python scripts/ablation_sample_efficiency.py --seeds 0 1 2 3 4

# Faster run (1 seed, fewer checkpoints)
python scripts/ablation_sample_efficiency.py --seeds 0 --checkpoints 100 500 1000 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_pricing.agents import PolicyGradientAgent, ReplayQLearningAgent
from rl_pricing.config import PricingConfig
from rl_pricing.dqn_agent import DQNAgent
from rl_pricing.environment import LoanPricingEnv
from rl_pricing.evaluation import evaluate_agent
from rl_pricing.policies import FixedRatePolicy, RuleBasedPolicy

DEFAULT_CHECKPOINTS = [100, 250, 500, 1000, 2000, 5000]
DEFAULT_SEEDS       = [0, 1]
EVAL_EPISODES       = 200
EPISODE_LENGTH      = 200


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers — one JSON file per seed
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(output_dir: Path, seed: int) -> Path:
    return output_dir / f"_se_seed{seed}.json"


def _load_cache(output_dir: Path, seed: int) -> dict | None:
    p = _cache_path(output_dir, seed)
    if p.exists():
        with p.open() as f:
            return json.load(f)
    return None


def _save_cache(output_dir: Path, seed: int, data: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with _cache_path(output_dir, seed).open("w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _train_steps(agent, env, n_episodes: int, decay_epsilon: bool) -> None:
    for _ in range(n_episodes):
        obs = env.reset()
        agent.start_episode(obs)
        done = False
        while not done:
            action = agent.act(obs, greedy=False)
            next_obs, reward, done = env.step(action)
            agent.observe(obs, action, reward, next_obs, done)
            obs = next_obs
        if decay_epsilon:
            agent.decay_epsilon()


def _eval_profit(agent, config: PricingConfig, seed: int) -> float:
    summary, _ = evaluate_agent(
        method="eval",
        factory=lambda s, a=agent: a,
        config=config,
        seeds=[seed],
        episodes=EVAL_EPISODES,
        greedy=True,
    )
    return summary.profit_mean


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_seed(seed: int, checkpoints: list[int], config: PricingConfig) -> dict:
    """Train all three agents incrementally through checkpoints for one seed.

    Each agent is created fresh and trained cumulatively up to each checkpoint.
    Agents are deleted after use to free memory before the next one starts.

    Returns { method: { str(checkpoint): profit } }
    """
    seed_results: dict[str, dict[str, float]] = {
        "Q-Learning":      {},
        "Policy Gradient": {},
        "DQN":             {},
    }

    # ── Q-Learning ────────────────────────────────────────────────────────────
    print("  Q-Learning …")
    env   = LoanPricingEnv(config=config, seed=seed)
    agent = ReplayQLearningAgent(config=config, seed=seed)
    prev  = 0
    for ckpt in checkpoints:
        _train_steps(agent, env, ckpt - prev, decay_epsilon=True)
        profit = _eval_profit(agent, config, seed)
        seed_results["Q-Learning"][str(ckpt)] = profit
        print(f"    ep={ckpt:>5}  profit/step={profit:+.5f}")
        prev = ckpt
    del agent, env

    # ── Policy Gradient ───────────────────────────────────────────────────────
    print("  Policy Gradient …")
    env   = LoanPricingEnv(config=config, seed=seed)
    agent = PolicyGradientAgent(config=config, seed=seed)
    prev  = 0
    for ckpt in checkpoints:
        _train_steps(agent, env, ckpt - prev, decay_epsilon=False)
        profit = _eval_profit(agent, config, seed)
        seed_results["Policy Gradient"][str(ckpt)] = profit
        print(f"    ep={ckpt:>5}  profit/step={profit:+.5f}")
        prev = ckpt
    del agent, env

    # ── DQN ───────────────────────────────────────────────────────────────────
    print("  DQN …")
    env   = LoanPricingEnv(config=config, seed=seed)
    agent = DQNAgent(config=config, seed=seed)
    prev  = 0
    for ckpt in checkpoints:
        _train_steps(agent, env, ckpt - prev, decay_epsilon=True)
        profit = _eval_profit(agent, config, seed)
        seed_results["DQN"][str(ckpt)] = profit
        print(f"    ep={ckpt:>5}  profit/step={profit:+.5f}")
        prev = ckpt
    del agent, env

    return seed_results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_sample_efficiency(
    seeds: list[int],
    checkpoints: list[int],
    config: PricingConfig,
    output_dir: Path,
) -> dict:
    checkpoints = sorted(checkpoints)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Run (or resume) each seed ─────────────────────────────────────────────
    all_seed_results: dict[int, dict] = {}
    for seed in seeds:
        cached = _load_cache(output_dir, seed)
        if cached is not None:
            print(f"\n── Seed {seed} — resuming from cache ✓")
            all_seed_results[seed] = cached
        else:
            print(f"\n── Seed {seed} ──────────────────────────────────────────")
            seed_results = _run_seed(seed, checkpoints, config)
            _save_cache(output_dir, seed, seed_results)
            print(f"  └─ seed {seed} saved to cache ✓")
            all_seed_results[seed] = seed_results

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("\nEvaluating baselines …")
    baselines: dict[str, float] = {}
    for name, factory in [
        ("Fixed Pricing", lambda s: FixedRatePolicy(config=config)),
        ("Rule-Based",    lambda s: RuleBasedPolicy(config=config)),
    ]:
        summary, _ = evaluate_agent(
            method=name, factory=factory, config=config,
            seeds=seeds, episodes=EVAL_EPISODES, greedy=True,
        )
        baselines[name] = summary.profit_mean
        print(f"  {name}: profit/step={summary.profit_mean:+.5f}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    methods = ["Q-Learning", "Policy Gradient", "DQN"]
    aggregated: dict[str, dict] = {}
    for method in methods:
        per_ckpt: dict[int, list[float]] = {c: [] for c in checkpoints}
        for seed in seeds:
            for ckpt in checkpoints:
                per_ckpt[ckpt].append(all_seed_results[seed][method][str(ckpt)])
        aggregated[method] = {
            "checkpoints": checkpoints,
            "mean": [float(np.mean(per_ckpt[c])) for c in checkpoints],
            "std":  [float(np.std(per_ckpt[c]))  for c in checkpoints],
            "per_seed": {str(c): per_ckpt[c]      for c in checkpoints},
        }

    output: dict = {
        "checkpoints": checkpoints,
        "seeds":       seeds,
        "baselines":   baselines,
        "results":     aggregated,
    }

    out_path = output_dir / "ablation_sample_efficiency.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFinal results → {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    col_w = 22
    header = f"{'Episodes':>10}" + "".join(f"{m:>{col_w}}" for m in methods)
    sep    = "─" * len(header)
    print(f"\n{sep}")
    print("Sample Efficiency — Mean Profit/step (± std) across seeds")
    print(sep)
    print(header)
    print(sep)
    for i, ckpt in enumerate(checkpoints):
        row = f"{ckpt:>10}"
        for method in methods:
            m = aggregated[method]["mean"][i]
            s = aggregated[method]["std"][i]
            row += f"  {m:+.5f} ± {s:.4f}    "
        print(row)
    print(sep)
    print(f"  Fixed Pricing baseline : {baselines['Fixed Pricing']:+.5f}")
    print(f"  Rule-Based    baseline : {baselines['Rule-Based']:+.5f}")

    _plot(aggregated, baselines, checkpoints, output_dir)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def _plot(aggregated, baselines, checkpoints, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = {"Q-Learning": "#2563eb", "Policy Gradient": "#16a34a", "DQN": "#dc2626"}

    for method, colour in colours.items():
        means = aggregated[method]["mean"]
        stds  = aggregated[method]["std"]
        ax.plot(checkpoints, means, marker="o", label=method,
                color=colour, linewidth=2)
        ax.fill_between(
            checkpoints,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15, color=colour,
        )

    for name, val in baselines.items():
        ax.axhline(val, linestyle="--", linewidth=1.2,
                   color="gray", label=f"{name} (baseline)")

    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Profit / Step")
    ax.set_title("Sample Efficiency: Profit vs Training Episodes")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = output_dir / "ablation_sample_efficiency.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Plot   → {plot_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation 1 — Sample Efficiency")
    p.add_argument("--seeds",          type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--checkpoints",    type=int, nargs="+", default=DEFAULT_CHECKPOINTS)
    p.add_argument("--episode-length", type=int, default=EPISODE_LENGTH)
    p.add_argument("--output-dir",     type=Path, default=Path("outputs/ablations"))
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    config = PricingConfig(episode_length=args.episode_length)
    run_sample_efficiency(
        seeds=args.seeds,
        checkpoints=args.checkpoints,
        config=config,
        output_dir=args.output_dir,
    )