"""Ablation 1 — Sample Efficiency.

Trains Q-Learning, Policy Gradient, and DQN agents and evaluates each at
fixed episode checkpoints. Shows how quickly each agent learns.

Checkpoints: 100, 250, 500, 1000, 2000, 5000 episodes.

Usage
-----
# Default (2 seeds, all checkpoints)
python scripts/ablation_sample_efficiency.py

# Faster run (1 seed, fewer checkpoints)
python scripts/ablation_sample_efficiency.py --seeds 0 --checkpoints 100 500 1000 5000

# Full paper-scale
python scripts/ablation_sample_efficiency.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import replace
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
EVAL_EPISODES       = 200   # kept short so checkpoints are fast
EPISODE_LENGTH      = 200


# ─────────────────────────────────────────────────────────────────────────────
# Generic checkpoint trainer
# ─────────────────────────────────────────────────────────────────────────────

def _train_to_checkpoint(agent, env, n_episodes: int, decay_epsilon: bool) -> float:
    """Run `n_episodes` episodes; return mean per-step reward of last 50 episodes."""
    recent: list[float] = []
    for ep in range(n_episodes):
        obs = env.reset()
        agent.start_episode(obs)
        done   = False
        ep_rew = 0.0
        steps  = 0
        while not done:
            action = agent.act(obs, greedy=False)
            next_obs, reward, done = env.step(action)
            agent.observe(obs, action, reward, next_obs, done)
            ep_rew += reward
            steps  += 1
            obs     = next_obs
        if decay_epsilon:
            agent.decay_epsilon()
        if ep >= n_episodes - 50:
            recent.append(ep_rew / max(steps, 1))
    return float(np.mean(recent)) if recent else 0.0


def run_sample_efficiency(
    seeds: list[int],
    checkpoints: list[int],
    config: PricingConfig,
    output_dir: Path,
) -> dict:
    checkpoints = sorted(checkpoints)
    results: dict[str, dict[int, list[float]]] = {
        "Q-Learning":      {c: [] for c in checkpoints},
        "Policy Gradient": {c: [] for c in checkpoints},
        "DQN":             {c: [] for c in checkpoints},
    }

    for seed in seeds:
        print(f"\n── Seed {seed} ──────────────────────────────────────────")

        # ── Q-Learning ────────────────────────────────────────────────────────
        print("  Q-Learning checkpoints …")
        q_env   = LoanPricingEnv(config=config, seed=seed)
        q_agent = ReplayQLearningAgent(config=config, seed=seed)
        prev    = 0
        for ckpt in checkpoints:
            delta = ckpt - prev
            _train_to_checkpoint(q_agent, q_env, delta, decay_epsilon=True)
            # Greedy evaluation
            summary, _ = evaluate_agent(
                method="Q-Learning",
                factory=lambda s, a=q_agent: a,
                config=config,
                seeds=[seed],
                episodes=EVAL_EPISODES,
                greedy=True,
            )
            results["Q-Learning"][ckpt].append(summary.profit_mean)
            print(f"    ep={ckpt:>5}  profit/step={summary.profit_mean:+.5f}")
            prev = ckpt

        # ── Policy Gradient ───────────────────────────────────────────────────
        print("  Policy Gradient checkpoints …")
        pg_env   = LoanPricingEnv(config=config, seed=seed)
        pg_agent = PolicyGradientAgent(config=config, seed=seed)
        prev     = 0
        for ckpt in checkpoints:
            delta = ckpt - prev
            _train_to_checkpoint(pg_agent, pg_env, delta, decay_epsilon=False)
            summary, _ = evaluate_agent(
                method="Policy Gradient",
                factory=lambda s, a=pg_agent: a,
                config=config,
                seeds=[seed],
                episodes=EVAL_EPISODES,
                greedy=True,
            )
            results["Policy Gradient"][ckpt].append(summary.profit_mean)
            print(f"    ep={ckpt:>5}  profit/step={summary.profit_mean:+.5f}")
            prev = ckpt

        # ── DQN ───────────────────────────────────────────────────────────────
        print("  DQN checkpoints …")
        dqn_env   = LoanPricingEnv(config=config, seed=seed)
        dqn_agent = DQNAgent(config=config, seed=seed)
        prev      = 0
        for ckpt in checkpoints:
            delta = ckpt - prev
            _train_to_checkpoint(dqn_agent, dqn_env, delta, decay_epsilon=True)
            summary, _ = evaluate_agent(
                method="DQN",
                factory=lambda s, a=dqn_agent: a,
                config=config,
                seeds=[seed],
                episodes=EVAL_EPISODES,
                greedy=True,
            )
            results["DQN"][ckpt].append(summary.profit_mean)
            print(f"    ep={ckpt:>5}  profit/step={summary.profit_mean:+.5f}")
            prev = ckpt

    # ── Baseline references (episode-independent) ─────────────────────────────
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
        print(f"\nBaseline {name}: profit/step={summary.profit_mean:+.5f}")

    # ── Aggregate across seeds ────────────────────────────────────────────────
    aggregated: dict[str, dict[str, list]] = {}
    for method, ckpt_data in results.items():
        aggregated[method] = {
            "checkpoints": checkpoints,
            "mean":  [float(np.mean(ckpt_data[c])) for c in checkpoints],
            "std":   [float(np.std(ckpt_data[c]))  for c in checkpoints],
            "seeds": {str(c): ckpt_data[c]          for c in checkpoints},
        }

    output: dict = {
        "checkpoints": checkpoints,
        "seeds":       seeds,
        "baselines":   baselines,
        "results":     aggregated,
    }

    out_path = output_dir / "ablation_sample_efficiency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    col_w = 14
    header = f"{'Episodes':>10}" + "".join(f"{'Q-Learning':>{col_w}}{'PG':>{col_w}}{'DQN':>{col_w}}")
    print("\n" + "─" * len(header))
    print("Sample Efficiency — Mean Profit/step across seeds")
    print("─" * len(header))
    print(header)
    print("─" * len(header))
    for i, ckpt in enumerate(checkpoints):
        row = f"{ckpt:>10}"
        for method in ["Q-Learning", "Policy Gradient", "DQN"]:
            m = aggregated[method]["mean"][i]
            s = aggregated[method]["std"][i]
            row += f"  {m:+.5f}±{s:.4f}"
        print(row)
    print("─" * len(header))
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
        ax.plot(checkpoints, means, marker="o", label=method, color=colour, linewidth=2)
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
    p.add_argument("--seeds",       type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--checkpoints", type=int, nargs="+", default=DEFAULT_CHECKPOINTS)
    p.add_argument("--episode-length", type=int, default=EPISODE_LENGTH)
    p.add_argument("--output-dir",  type=Path, default=Path("outputs/ablations"))
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
