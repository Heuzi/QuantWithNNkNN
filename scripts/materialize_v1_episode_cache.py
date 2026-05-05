from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.episode_eligibility import EpisodeEligibilityConfig, parse_allowed_exchanges  # noqa: E402
from src.data.v1_dataset import (  # noqa: E402
    DEFAULT_BENCHMARK_TICKER,
    DEFAULT_CLASSIFICATION_HORIZON,
    DEFAULT_CLASSIFICATION_THRESHOLD,
    DEFAULT_HORIZONS,
    DEFAULT_WINDOW_LENGTH,
    parse_horizons,
)
from src.data.v1_episode_cache import build_episode_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize an episode-level V1 training cache with float32 tabular arrays "
            "and memmapped sequence arrays. This is the out-of-core training input for "
            "full EODHD runs."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--feature-sets", required=True, help="Comma-separated feature sets to cache.")
    parser.add_argument("--horizons", default=",".join(str(value) for value in DEFAULT_HORIZONS))
    parser.add_argument("--window-length", type=int, default=DEFAULT_WINDOW_LENGTH)
    parser.add_argument("--benchmark-ticker", default=DEFAULT_BENCHMARK_TICKER)
    parser.add_argument("--max-episodes", type=int, default=0, help="Keep the most recent N eligible episodes; 0 keeps all.")
    parser.add_argument("--classification-horizon", type=int, default=DEFAULT_CLASSIFICATION_HORIZON)
    parser.add_argument("--classification-threshold", type=float, default=DEFAULT_CLASSIFICATION_THRESHOLD)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument(
        "--disable-episode-eligibility-filter",
        action="store_true",
        help="Disable as-of common-stock/history/liquidity/price/exchange episode filtering.",
    )
    parser.add_argument("--eligibility-min-history-days", type=int, default=0)
    parser.add_argument("--eligibility-valid-ohlcv-lookback", type=int, default=60)
    parser.add_argument("--eligibility-min-valid-ohlcv-days", type=int, default=55)
    parser.add_argument("--eligibility-dollar-volume-lookback", type=int, default=60)
    parser.add_argument("--eligibility-min-avg-dollar-volume", type=float, default=100_000.0)
    parser.add_argument("--eligibility-min-price", type=float, default=1.0)
    parser.add_argument("--eligibility-allowed-exchanges", default="NYSE,NASDAQ,AMEX,BATS")
    return parser.parse_args()


def _episode_eligibility_config(args: argparse.Namespace) -> EpisodeEligibilityConfig | None:
    if args.disable_episode_eligibility_filter:
        return None
    return EpisodeEligibilityConfig(
        min_history_days=args.eligibility_min_history_days or args.window_length,
        valid_ohlcv_lookback=args.eligibility_valid_ohlcv_lookback,
        min_valid_ohlcv_days=args.eligibility_min_valid_ohlcv_days,
        dollar_volume_lookback=args.eligibility_dollar_volume_lookback,
        min_avg_dollar_volume=args.eligibility_min_avg_dollar_volume,
        min_price=args.eligibility_min_price,
        allowed_exchanges=parse_allowed_exchanges(args.eligibility_allowed_exchanges),
    )


def main() -> None:
    args = parse_args()
    feature_sets = [item.strip() for item in args.feature_sets.split(",") if item.strip()]
    if not feature_sets:
        raise SystemExit("--feature-sets must name at least one feature set.")
    manifest = build_episode_cache(
        dataset_root=args.dataset_root,
        cache_dir=args.cache_dir,
        feature_sets=feature_sets,
        horizons=parse_horizons(args.horizons),
        window_length=args.window_length,
        benchmark_ticker=args.benchmark_ticker,
        max_episodes=args.max_episodes or None,
        classification_horizon=args.classification_horizon,
        classification_threshold=args.classification_threshold,
        eligibility_config=_episode_eligibility_config(args),
        force=args.force,
        progress_every=args.progress_every,
    )
    print(f"Episode cache written to {Path(args.cache_dir).resolve()}")
    print(f"Episodes: {manifest['episode_count']}; stock rows: {manifest['stock_row_count']}")


if __name__ == "__main__":
    main()

