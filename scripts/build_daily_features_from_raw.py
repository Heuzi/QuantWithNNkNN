from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.massive_stage1 import compute_daily_features, load_daily_bars_csv, write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed daily features from a dataset folder containing raw/daily_market_bars.csv."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset folder that contains raw/daily_market_bars.csv and will receive processed/daily_features.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    raw_bars_path = dataset_root / "raw" / "daily_market_bars.csv"
    processed_dir = dataset_root / "processed"
    processed_path = processed_dir / "daily_features.csv"

    if not raw_bars_path.exists():
        raise SystemExit(f"Raw bars file not found: {raw_bars_path}")

    bars = load_daily_bars_csv(raw_bars_path)
    features = compute_daily_features(bars)
    write_csv(processed_path, features)
    print(f"Wrote {len(features)} daily feature rows to {processed_path}")


if __name__ == "__main__":
    main()
