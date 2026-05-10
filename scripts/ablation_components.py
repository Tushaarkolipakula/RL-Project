"""Ablation 3 — Component Ablations.

Tests what each agent's key architectural component contributes by removing it:

  Q-Learning:
    - Full (planning_steps=12)           ← default
    - No Dyna replay (planning_steps=0)  ← ablated

  Policy Gradient:
    - Full (return normalisation ON)     ← default
    - No normalisation                   ← ablated

  DQN:
    - Full (Double DQN + n_step=5)       ← default
    - Single DQN (online net for target) ← ablated
    - No n-step  (n_step=1)              ← ablated

Crash-safe: results are saved to a per-seed JSON file after every seed
completes. Re-running the script skips already-completed seeds automatically.

Usage
-----
# Quick smoke-test
python scripts/ablation_components.py --seeds 0 --train-episodes 500

# Recommended for report (2 seeds, 2000 episodes)
python scripts/ablation_components.py --seeds 0 1 --train-episodes 2000

# Full scale (5 seeds, 5000 episodes)
python scripts/ablation_components.py --seeds 0 1 2 3 4 --train-episodes 5000
"""

from __future__ import annotations

import argparse
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

DEFAULT_SEEDS          = [0, 1]
DEFAULT_TRAIN_EPISODES = 2000
DEFAULT_EVAL_EPISODES  = 200
EPISODE_LENGTH         = 200


# ─────────────────────────────────────────────────────────────────────────────
# Patched Policy Gradient — disables return normalisation
# ─────────────────────────────────────────────────────────────────────────────

class _PGNoNorm(PolicyGradientAgent):
    """PolicyGradientAgent with per-episode return normalisation disabled.

    Only _reinforce_update() is overridden — the 3-line normalisation block
    is removed. Buffering, backprop, and Adam are all inherited unchanged.
    """

    def _reinforce_update(self) -> None:
        T       = len(self._ep_rewards)
        rewards = np.array(self._ep_rewards, dtype=np.float64)

        # Discounted returns — identical to parent
        returns = np.zeros(T, dtype=np.float64)
        G = 0.0
        for t in range(T - 1, -1, -1):
            G          = rewards[t] + self.gamma * G
            returns[t] = G

        # ── NO normalisation (ablated) ────────────────────────────────────────
        # Parent: std = returns.std()
        #         if std > 1e-8: returns = (returns - returns.mean()) / (std + 1e-8)

        dW1 = np.zeros_like(self.W1); db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2); db2 = np.zeros_like(self.b2)
        dW3 = np.zeros_like(self.W3); db3 = np.zeros_like(self.b3)

        for t in range(T):
            s   = self._ep_states[t]
            a   = self._ep_actions[t]
            G_t = returns[t]

            h1, h2, probs = self._forward(s)

            d_logits     = -probs.copy()
            d_logits[a] += 1.0
            d_logits    *= G_t

            dW3 += np.outer(h2, d_logits);  db3 += d_logits
            d_h2 = d_logits @ self.W3.T * (h2 > 0)
            dW2 += np.outer(h1, d_h2);      db2 += d_h2
            d_h1 = d_h2 @ self.W2.T * (h1 > 0)
            dW1 += np.outer(s,  d_h1);      db1 += d_h1

        # Adam ascent — identical to parent
        self._t += 1
        grads  = {"W1": dW1, "b1": db1, "W2": dW2,
                  "b2": db2, "W3": dW3, "b3": db3}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for name, param in self._params():
            g             = grads[name]
            self._m[name] = beta1 * self._m[name] + (1 - beta1) * g
            self._v[name] = beta2 * self._v[name] + (1 - beta2) * g * g
            m_hat         = self._m[name] / (1.0 - beta1 ** self._t)
            v_hat         = self._v[name] / (1.0 - beta2 ** self._t)
            param        += self.lr * m_hat / (np.sqrt(v_hat) + eps)


# ─────────────────────────────────────────────────────────────────────────────
# Patched DQN — Single DQN (no Double-DQN)
# ─────────────────────────────────────────────────────────────────────────────

class _DQNSingle(DQNAgent):
    """DQNAgent with Double-DQN disabled.

    Uses the online network to both select and score next actions,
    reintroducing the maximisation bias that Double DQN was designed to fix.
    """

    def _learn(self) -> None:
        batch = self.replay.sample(self.batch_size)
        B     = len(batch)

        S      = np.stack([t.state      for t in batch])
        A      = np.array([t.action     for t in batch], dtype=np.int64)
        R      = np.array([t.ret        for t in batch])
        S_next = np.stack([t.next_state for t in batch])
        Done   = np.array([t.done       for t in batch], dtype=np.float64)

        r_std = R.std()
        if r_std > 1e-8:
            R = (R - R.mean()) / (r_std + 1e-8)

        Q_online, cache = self.online_net.forward(S)

        # ── Single DQN: online net selects AND scores (ablated) ───────────────
        # Double DQN: online selects, target scores → removes maximisation bias
        # Single DQN: online does both → bias reintroduced
        Q_next        = self.online_net.predict(S_next)
        next_actions  = np.argmax(Q_next, axis=1)
        Q_next_val    = Q_next[np.arange(B), next_actions]

        gamma_n = self.gamma ** self.n_step
        target  = R + gamma_n * Q_next_val * (1.0 - Done)

        Q_taken = Q_online[np.arange(B), A]
        td_err  = Q_taken - target
        huber_g = np.where(np.abs(td_err) <= 1.0, td_err, np.sign(td_err))

        dQ = np.zeros_like(Q_online)
        dQ[np.arange(B), A] = huber_g / B
        self.online_net.update(cache, dQ, lr=self.lr)


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers — one JSON file per seed
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(output_dir: Path, seed: int) -> Path:
    return output_dir / f"_comp_seed{seed}.json"


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
# Per-seed runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_seed(
    seed: int,
    variants: list[tuple[str, callable, bool]],
    config: PricingConfig,
    eval_episodes: int,
) -> dict[str, float]:
    """Train every variant for one seed; return { label: profit }."""
    seed_results: dict[str, float] = {}

    for label, factory, decay_epsilon in variants:
        agent = factory(seed)
        env   = LoanPricingEnv(config=config, seed=seed)

        for _ in range(config.train_episodes):
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

        summary, _ = evaluate_agent(
            method=label,
            factory=lambda s, a=agent: a,
            config=config,
            seeds=[seed],
            episodes=eval_episodes,
            greedy=True,
        )
        profit = summary.profit_mean
        seed_results[label] = profit
        print(f"  {label:<35}  profit/step={profit:+.5f}")

        del agent, env   # free memory before next variant

    return seed_results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_component_ablations(
    seeds: list[int],
    config: PricingConfig,
    eval_episodes: int,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, callable, bool]] = [
        ("Q-Learning (full)",
         lambda seed: ReplayQLearningAgent(config=config, seed=seed),
         True),
        ("Q-Learning (no Dyna replay)",
         lambda seed: ReplayQLearningAgent(
             config=replace(config, planning_steps=0), seed=seed),
         True),
        ("PG (full)",
         lambda seed: PolicyGradientAgent(config=config, seed=seed),
         False),
        ("PG (no return norm)",
         lambda seed: _PGNoNorm(config=config, seed=seed),
         False),
        ("DQN (full)",
         lambda seed: DQNAgent(config=config, seed=seed),
         True),
        ("DQN (single net)",
         lambda seed: _DQNSingle(config=config, seed=seed),
         True),
        ("DQN (n_step=1)",
         lambda seed: DQNAgent(config=replace(config, dqn_n_step=1), seed=seed),
         True),
    ]
    labels = [v[0] for v in variants]

    # ── Run (or resume) each seed ─────────────────────────────────────────────
    all_seed_results: dict[int, dict[str, float]] = {}
    for seed in seeds:
        cached = _load_cache(output_dir, seed)
        if cached is not None:
            print(f"\n── Seed {seed} — resuming from cache ✓")
            all_seed_results[seed] = cached
        else:
            print(f"\n── Seed {seed} ──────────────────────────────────────────")
            seed_results = _run_seed(seed, variants, config, eval_episodes)
            _save_cache(output_dir, seed, seed_results)
            print(f"  └─ seed {seed} saved to cache ✓")
            all_seed_results[seed] = seed_results

    # ── Aggregate ─────────────────────────────────────────────────────────────
    aggregated: dict[str, dict] = {}
    for label in labels:
        vals = [all_seed_results[s][label] for s in seeds]
        aggregated[label] = {
            "mean":     float(np.mean(vals)),
            "std":      float(np.std(vals)),
            "per_seed": vals,
        }

    output = {
        "seeds":          seeds,
        "train_episodes": config.train_episodes,
        "eval_episodes":  eval_episodes,
        "results":        aggregated,
    }

    out_path = output_dir / "ablation_components.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFinal results → {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    groups = [
        ("Q-Learning",     ["Q-Learning (full)", "Q-Learning (no Dyna replay)"]),
        ("Policy Gradient",["PG (full)",          "PG (no return norm)"]),
        ("DQN",            ["DQN (full)",          "DQN (single net)", "DQN (n_step=1)"]),
    ]
    print("\n" + "─" * 65)
    print("Component Ablations — Mean Profit/step (± std) across seeds")
    print("─" * 65)
    for group_name, grp_labels in groups:
        print(f"\n  {group_name}")
        full_mean = aggregated[grp_labels[0]]["mean"]
        for lbl in grp_labels:
            m    = aggregated[lbl]["mean"]
            s    = aggregated[lbl]["std"]
            diff = m - full_mean
            tag  = "" if lbl == grp_labels[0] else f"  (Δ {diff:+.5f})"
            print(f"    {lbl:<35}  {m:+.5f} ± {s:.4f}{tag}")
    print("─" * 65)

    _plot(aggregated, groups, output_dir)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def _plot(aggregated, groups, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
    group_colours = {
        "Q-Learning":     ["#2563eb", "#93c5fd"],
        "Policy Gradient":["#16a34a", "#86efac"],
        "DQN":            ["#dc2626", "#fca5a5", "#fb923c"],
    }

    for ax, (group_name, grp_labels) in zip(axes, groups):
        colours = group_colours[group_name]
        means   = [aggregated[l]["mean"] for l in grp_labels]
        stds    = [aggregated[l]["std"]  for l in grp_labels]
        short   = [l.split("(")[-1].rstrip(")") if "(" in l else "full"
                   for l in grp_labels]

        bars = ax.bar(short, means, color=colours,
                      edgecolor="white", linewidth=0.8, width=0.55)
        ax.errorbar(range(len(grp_labels)), means, yerr=stds,
                    fmt="none", color="black", capsize=5, linewidth=1.5)

        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.15,
                    f"{m:+.5f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(group_name, fontweight="bold")
        ax.set_ylabel("Profit / Step" if ax is axes[0] else "")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        for i in range(1, len(bars)):
            bars[i].set_alpha(0.7)

    fig.suptitle("Component Ablations: Contribution of Each Architectural Feature",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()

    plot_path = output_dir / "ablation_components.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Plot   → {plot_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation 3 — Component Ablations")
    p.add_argument("--seeds",          type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    p.add_argument("--eval-episodes",  type=int, default=DEFAULT_EVAL_EPISODES)
    p.add_argument("--episode-length", type=int, default=EPISODE_LENGTH)
    p.add_argument("--output-dir",     type=Path, default=Path("outputs/ablations"))
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    config = PricingConfig(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        episode_length=args.episode_length,
    )
    run_component_ablations(
        seeds=args.seeds,
        config=config,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
    )