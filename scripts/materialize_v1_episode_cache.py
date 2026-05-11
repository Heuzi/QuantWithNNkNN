from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.episode_eligibility import EpisodeEligibilityConfig, parse_allowed_exchanges  # noqa: E402
from src.data.research_universe import ConservativeResearchUniverseConfig  # noqa: E402
from src.data.v1_dataset import (  # noqa: E402
    DEFAULT_BENCHMARK_TICKER,
    DEFAULT_CLASSIFICATION_EVENT_TYPE,
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
    parser.add_argument(
        "--classification-event-type",
        default=DEFAULT_CLASSIFICATION_EVENT_TYPE,
        help=(
            "Classification label semantics. Default keeps existing anytime pathwise "
            "outperformance labels; sustained_pathwise_outperform and path_5pct_20d are opt-in."
        ),
    )
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
    parser.add_argument(
        "--disable-conservative-research-universe",
        action="store_true",
        help="Disable the shared strategy-universe filter for train/test/cache construction.",
    )
    parser.add_argument("--research-common-stocks-only", action="store_true", default=True)
    parser.add_argument("--research-allowed-exchanges", default="NYSE,NASDAQ,AMEX")
    parser.add_argument("--research-min-price", type=float, default=10.0)
    parser.add_argument("--research-min-history-days", type=int, default=252)
    parser.add_argument("--research-min-median-dollar-volume-20d", type=float, default=10_000_000.0)
    parser.add_argument("--research-min-median-dollar-volume-60d", type=float, default=10_000_000.0)
    parser.add_argument("--research-max-zero-volume-day-ratio-60d", type=float, default=0.02)
    parser.add_argument("--research-min-current-dollar-volume-vs-median-20d", type=float, default=0.20)
    parser.add_argument("--research-liquidity-short-lookback-days", type=int, default=20)
    parser.add_argument("--research-liquidity-long-lookback-days", type=int, default=60)
    parser.add_argument("--research-trend-lookback-days", type=int, default=252)
    parser.add_argument("--research-return-6m-lookback-days", type=int, default=126)
    parser.add_argument("--research-sma-short-lookback-days", type=int, default=50)
    parser.add_argument("--research-sma-long-lookback-days", type=int, default=200)
    parser.add_argument("--research-min-return-6m", type=float, default=-0.15)
    parser.add_argument("--research-max-drawdown-from-252d-high-pct", type=float, default=35.0)
    parser.add_argument("--research-disable-close-above-sma200", action="store_true")
    parser.add_argument("--research-disable-sma50-above-sma200", action="store_true")
    parser.add_argument("--research-disable-spike-filter", action="store_true")
    parser.add_argument("--research-spike-lookback-days", type=int, default=60)
    parser.add_argument("--research-max-abs-return-1d-60d-pct", type=float, default=25.0)
    parser.add_argument("--research-max-true-range-60d-pct", type=float, default=25.0)
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


def _research_universe_config(args: argparse.Namespace) -> ConservativeResearchUniverseConfig | None:
    if args.disable_conservative_research_universe:
        return None
    return ConservativeResearchUniverseConfig(
        common_stocks_only=bool(args.research_common_stocks_only),
        allowed_exchanges=parse_allowed_exchanges(args.research_allowed_exchanges),
        min_price=args.research_min_price,
        min_history_days=args.research_min_history_days,
        liquidity_short_lookback=args.research_liquidity_short_lookback_days,
        liquidity_long_lookback=args.research_liquidity_long_lookback_days,
        min_median_dollar_volume_20d=args.research_min_median_dollar_volume_20d,
        min_median_dollar_volume_60d=args.research_min_median_dollar_volume_60d,
        max_zero_volume_day_ratio_60d=args.research_max_zero_volume_day_ratio_60d,
        min_current_dollar_volume_vs_median_20d=args.research_min_current_dollar_volume_vs_median_20d,
        trend_lookback_days=args.research_trend_lookback_days,
        return_6m_lookback_days=args.research_return_6m_lookback_days,
        sma_short_lookback_days=args.research_sma_short_lookback_days,
        sma_long_lookback_days=args.research_sma_long_lookback_days,
        min_return_6m=args.research_min_return_6m,
        max_drawdown_from_252d_high=args.research_max_drawdown_from_252d_high_pct / 100.0,
        require_close_above_sma200=not bool(args.research_disable_close_above_sma200),
        require_sma50_above_sma200=not bool(args.research_disable_sma50_above_sma200),
        spike_filter_enabled=not bool(args.research_disable_spike_filter),
        spike_lookback_days=args.research_spike_lookback_days,
        max_abs_return_1d_60d=args.research_max_abs_return_1d_60d_pct / 100.0,
        max_true_range_pct_60d=args.research_max_true_range_60d_pct / 100.0,
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
        classification_event_type=args.classification_event_type,
        eligibility_config=_episode_eligibility_config(args),
        research_config=_research_universe_config(args),
        force=args.force,
        progress_every=args.progress_every,
    )
    print(f"Episode cache written to {Path(args.cache_dir).resolve()}")
    print(f"Episodes: {manifest['episode_count']}; stock rows: {manifest['stock_row_count']}")
    if manifest.get("labeling"):
        labeling = manifest["labeling"]
        print(
            "Labeling: "
            f"mode={labeling.get('mode')}; "
            f"labeled={labeling.get('labeled_row_count')}; "
            f"missing_forward_window={labeling.get('unlabeled_missing_forward_window_rows')}; "
            f"invalid_price={labeling.get('unlabeled_invalid_price_rows')}"
        )


if __name__ == "__main__":
    main()
