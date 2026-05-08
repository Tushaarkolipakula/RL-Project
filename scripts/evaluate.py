"""Evaluate saved or built-in pricing policies without retraining.

Five methods evaluated (matching the paper + DQN extension):
  1. Fixed Pricing
  2. Rule-Based
  3. Q-Learning       (requires --q-model)
  4. Policy Gradient  (requires --pg-model)
  5. DQN              (requires --dqn-model)

Examples
--------
# Baselines only
python scripts/evaluate.py

# All five methods
python scripts/evaluate.py \
    --q-model   outputs/full/q_learning_seed0.pkl \
    --pg-model  outputs/full/policy_gradient_seed0.pkl \
    --dqn-model outputs/full/dqn_seed0.pkl

# Backward-compatible single-model alias
python scripts/evaluate.py --model outputs/full/q_learning_seed0.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_pricing.agents import PolicyGradientAgent, ReplayQLearningAgent
from rl_pricing.config import PricingConfig
from rl_pricing.dqn_agent import DQNAgent
from rl_pricing.evaluation import evaluate_agent, format_summary_table
from rl_pricing.policies import FixedRatePolicy, RuleBasedPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pricing policies")
    parser.add_argument("--episodes",       type=int,  default=PricingConfig.eval_episodes)
    parser.add_argument("--seeds",          type=int,  nargs="*", default=list(PricingConfig.seeds))
    parser.add_argument("--episode-length", type=int,  default=PricingConfig.episode_length)
    # Saved model paths (optional — baselines always run)
    parser.add_argument("--q-model",   type=Path, default=None,
                        help="Path to a saved Q-Learning .pkl model")
    parser.add_argument("--pg-model",  type=Path, default=None,
                        help="Path to a saved Policy Gradient .pkl model")
    parser.add_argument("--dqn-model", type=Path, default=None,
                        help="Path to a saved DQN .pkl model")
    # Backward-compatible alias
    parser.add_argument("--model", type=Path, default=None,
                        help="Alias for --q-model (backward compatibility)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --model is a backward-compatible alias for --q-model
    q_model_path = args.q_model or args.model

    config = PricingConfig(
        episode_length=args.episode_length,
        eval_episodes=args.episodes,
        seeds=tuple(args.seeds),
    )

    specs = [
        ("Fixed Pricing", lambda seed: FixedRatePolicy(config=config)),
        ("Rule-Based",    lambda seed: RuleBasedPolicy(config=config)),
    ]

    if q_model_path is not None:
        specs.append((
            "Q-Learning",
            lambda seed: ReplayQLearningAgent.load(q_model_path, seed=seed),
        ))

    if args.pg_model is not None:
        specs.append((
            "Policy Gradient",
            lambda seed: PolicyGradientAgent.load(args.pg_model, seed=seed),
        ))

    if args.dqn_model is not None:
        specs.append((
            "DQN",
            lambda seed: DQNAgent.load(args.dqn_model, seed=seed),
        ))

    summaries = []
    for method, factory in specs:
        summary, _ = evaluate_agent(
            method=method,
            factory=factory,
            config=config,
            seeds=list(args.seeds),
            episodes=args.episodes,
            greedy=True,
        )
        summaries.append(summary)

    print(format_summary_table(summaries))


if __name__ == "__main__":
    main()