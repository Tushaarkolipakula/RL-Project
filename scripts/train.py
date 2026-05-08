"""Train and evaluate the complete dynamic-pricing project."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_pricing.config import DEFAULT_OUTPUT_DIR, PricingConfig
from rl_pricing.evaluation import MetricSummary, format_summary_table
from rl_pricing.training import run_full_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL dynamic pricing experiment")
    parser.add_argument("--train-episodes", type=int, default=PricingConfig.train_episodes)
    parser.add_argument("--eval-episodes",  type=int, default=PricingConfig.eval_episodes)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(PricingConfig.seeds))
    parser.add_argument("--episode-length", type=int, default=PricingConfig.episode_length)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip generating the results plot.")
    parser.add_argument("--no-dqn", action="store_true",
                        help="Skip DQN training and reproduce the original "
                             "two-agent (Q-Learning + Policy Gradient) experiment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PricingConfig(
        episode_length=args.episode_length,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        seeds=tuple(args.seeds),
    )

    results = run_full_experiment(
        config=config,
        seeds=list(args.seeds),
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        include_dqn=not args.no_dqn,
    )

    summaries = [MetricSummary(**item) for item in results["summaries"]]
    print(format_summary_table(summaries))
    print(f"\nSaved results to {results['results_path']}")

    if not args.no_plot:
        from rl_pricing.plotting import plot_experiment
        plot_path = args.output_dir / "results_plot.png"
        saved_plot = plot_experiment(results, plot_path)
        print(f"Saved plot to {saved_plot}")


if __name__ == "__main__":
    main()