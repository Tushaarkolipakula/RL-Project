"""Generate and save the synthetic loan-application dataset.

This script is separate from training so the dataset can be created once
and reused. Run it before training:

    python scripts/generate_dataset.py

Or with custom options:

    python scripts/generate_dataset.py --episodes 1000 --output-dir outputs/

Output files (both written to --output-dir):
    synthetic_dataset.csv    one row per loan application
    dataset_summary.json     statistics: distributions, rate ranges, etc.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_pricing.config import DEFAULT_OUTPUT_DIR, PricingConfig
from rl_pricing.dataset import build_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic loan-application dataset"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of episodes to simulate (default: 500 → 100 000 rows)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "data",
        help="Directory to write CSV and summary JSON (default: outputs/data/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PricingConfig()
    build_and_save(
        output_dir=args.output_dir,
        n_episodes=args.episodes,
        config=config,
    )


if __name__ == "__main__":
    main()