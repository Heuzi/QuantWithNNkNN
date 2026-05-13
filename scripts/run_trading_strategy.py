from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.predict_v1_supervised_baselines import (  # noqa: E402
    _add_anchor_close,
    _aligned_features,
    _episode_eligibility_config_from_run,
    _research_universe_config_from_run,
    _resolve_artifact_path,
    _resolve_model_index_path,
)
from src.data.episode_eligibility import (  # noqa: E402
    add_episode_eligibility_columns,
    eligibility_metadata_columns,
    parse_allowed_exchanges,
)
from src.data.research_universe import (  # noqa: E402
    ConservativeResearchUniverseConfig,
    conservative_research_universe_summary,
    latest_research_universe_diagnostics,
)
from src.data.eodhd_enrichment import (  # noqa: E402
    add_fundamental_features,
    add_sentiment_features,
    load_fundamental_feature_rows,
    load_sentiment_rows,
)
from src.data.eodhd_stage1 import (  # noqa: E402
    EODHDAPIError,
    EODHDError,
    EODHDRateLimitError,
    EODHDRESTClient,
    EODHD_DAILY_BAR_HEADERS,
    RateLimiter,
    eodhd_symbol_for_code,
    load_eodhd_credentials,
    normalize_eodhd_bulk_eod_rows,
    normalize_eodhd_eod_rows,
)
from src.data.massive_stage1 import compute_daily_features, write_csv, write_json  # noqa: E402
from src.data.v1_dataset import (  # noqa: E402
    MARKET_CONTEXT_TICKERS,
    SECTOR_ETF_BY_GICS,
    STATIC_CATEGORICAL_COLUMNS,
    build_sequence_feature_store,
    add_context_relative_return_features,
    add_priority_a_ohlcv_features,
    add_window_summaries,
    encode_static_categories,
    load_market_context_features,
    select_augmented_stock_feature_columns,
    select_context_feature_columns,
    preferred_stock_feature_path,
    validate_model_feature_columns,
    _filtered_stock_universe,
    _merge_context,
)
from src.models.v1_baselines import load_model_bundle, prediction_frame  # noqa: E402


DEFAULT_RUN_DIRS = (
    "artifacts/v1_baselines/eodhd_true_full_xgboost",
    "artifacts/v1_baselines/eodhd_true_full_torch_mlp",
    "artifacts/v1_baselines/eodhd_true_full_torch_seq_static",
)
LATEST_INFERENCE_DIRNAME = "latest_inference"
STOCK_UPDATE_FILENAME = "eodhd_stock_bars_daily_updates.csv"
CONTEXT_UPDATE_FILENAME = "market_context_bars_daily_updates.csv"
RECENT_STOCK_BARS_FILENAME = "recent_stock_bars.csv"
RECENT_CONTEXT_BARS_FILENAME = "recent_context_bars.csv"
LATEST_STOCK_FEATURES_FILENAME = "latest_daily_features.csv"
LATEST_CONTEXT_FEATURES_FILENAME = "latest_market_context_features.csv"
STOCK_FEATURE_UPDATE_FILENAME = "daily_features_incremental_updates.csv"
CONTEXT_FEATURE_UPDATE_FILENAME = "market_context_features_incremental_updates.csv"
FEATURE_UPDATE_MANIFEST_FILENAME = "incremental_feature_updates_manifest.json"

POSITION_REVIEW_COLUMNS = (
    "run_date",
    "ticker",
    "entry_date",
    "entry_price",
    "shares",
    "current_price",
    "current_return_pct",
    "days_held",
    "latest_rank_bucket",
    "agreement_status",
    "review_action",
    "reason",
    "current_status",
    "notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run current production V1 classifiers and generate trading review reports."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/eodhd_us_equities_30y",
        help="EODHD dataset root containing raw bars and production metadata.",
    )
    parser.add_argument("--credentials-path", default="EODHD_api_key")
    parser.add_argument("--exchange", default="US", help="EODHD bulk EOD exchange code.")
    parser.add_argument("--skip-fetch", action="store_true", help="Use local latest-inference cache/raw data only.")
    parser.add_argument("--fetch-start-date", default="", help="Optional YYYY-MM-DD override for missing EOD fetch start.")
    parser.add_argument("--fetch-end-date", default="", help="Optional YYYY-MM-DD override for missing EOD fetch end.")
    parser.add_argument(
        "--recent-raw-rows-per-ticker",
        type=int,
        default=320,
        help="Raw bars retained per ticker before computing latest inference features.",
    )
    parser.add_argument("--max-tickers", type=int, default=0, help="Optional smoke cap for stock tickers.")
    parser.add_argument("--skip-fundamentals", action="store_true", help="Do not join saved fundamentals.")
    parser.add_argument("--skip-sentiment", action="store_true", help="Do not join saved sentiment.")
    parser.add_argument("--include-test-symbols", action="store_true", help="Include exchange test symbols if present.")
    parser.add_argument(
        "--force-rebuild-latest-inference",
        action="store_true",
        help="Rebuild latest inference features even when a current cache exists.",
    )
    parser.add_argument(
        "--disable-persist-latest-feature-updates",
        action="store_true",
        help=(
            "Do not persist newly computed latest-inference feature rows into processed incremental update files "
            "for future retrain consolidation."
        ),
    )
    parser.add_argument(
        "--latest-inference-dir",
        default="",
        help="Optional latest-inference output/cache directory. Defaults under dataset_root/processed.",
    )
    parser.add_argument("--progress-every-rows", type=int, default=2_000_000)
    parser.add_argument("--feature-progress-every-tickers", type=int, default=500)
    parser.add_argument("--progress-bar-width", type=int, default=30)
    parser.add_argument("--rate-limit-calls", type=int, default=200)
    parser.add_argument("--rate-limit-period-seconds", type=float, default=60.0)
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Production model run directory. Repeat for multiple models. Defaults to the current three-model set.",
    )
    parser.add_argument("--anchor-date", default="", help="Optional YYYY-MM-DD cutoff for predictions.")
    parser.add_argument(
        "--max-anchor-lag-days",
        type=int,
        default=3,
        help="Keep latest ticker windows no more than this many calendar days behind the newest anchor date.",
    )
    parser.add_argument("--output-root", default="artifacts/production_reports")
    parser.add_argument("--report-name", default="", help="Optional report folder name. Defaults to UTC timestamp.")
    parser.add_argument("--position-ledger", default="data/open_positions.csv")
    parser.add_argument("--entry-top-percent", type=float, default=5.0)
    parser.add_argument("--watchlist-top-percent", type=float, default=10.0)
    parser.add_argument(
        "--disable-conservative-research-universe",
        action="store_true",
        help="Disable the shared strategy-universe filter for latest recommendations.",
    )
    parser.add_argument("--research-allowed-exchanges", default="")
    parser.add_argument("--research-min-price", type=float, default=None)
    parser.add_argument("--research-min-history-days", type=int, default=None)
    parser.add_argument("--research-min-median-dollar-volume-20d", type=float, default=None)
    parser.add_argument("--research-min-median-dollar-volume-60d", type=float, default=None)
    parser.add_argument("--research-max-zero-volume-day-ratio-60d", type=float, default=None)
    parser.add_argument("--research-min-current-dollar-volume-vs-median-20d", type=float, default=None)
    parser.add_argument("--research-liquidity-short-lookback-days", type=int, default=None)
    parser.add_argument("--research-liquidity-long-lookback-days", type=int, default=None)
    parser.add_argument("--research-trend-lookback-days", type=int, default=None)
    parser.add_argument("--research-return-6m-lookback-days", type=int, default=None)
    parser.add_argument("--research-sma-short-lookback-days", type=int, default=None)
    parser.add_argument("--research-sma-long-lookback-days", type=int, default=None)
    parser.add_argument("--research-min-return-6m", type=float, default=None)
    parser.add_argument("--research-max-drawdown-from-252d-high-pct", type=float, default=None)
    parser.add_argument("--research-disable-close-above-sma200", action="store_true")
    parser.add_argument("--research-disable-sma50-above-sma200", action="store_true")
    parser.add_argument("--research-disable-spike-filter", action="store_true")
    parser.add_argument("--research-spike-lookback-days", type=int, default=None)
    parser.add_argument("--research-max-abs-return-1d-60d-pct", type=float, default=None)
    parser.add_argument("--research-max-true-range-60d-pct", type=float, default=None)
    parser.add_argument("--review-rank-threshold-percent", type=float, default=20.0)
    parser.add_argument("--strong-review-rank-threshold-percent", type=float, default=30.0)
    parser.add_argument("--target-profit-pct", type=float, default=5.0)
    parser.add_argument("--stop-loss-pct", type=float, default=-5.0)
    parser.add_argument("--max-hold-days", type=int, default=20)
    parser.add_argument("--min-model-agreement-count", type=int, default=2)
    return parser.parse_args()


def _utc_now_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _run_date() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _research_universe_config(
    args: argparse.Namespace,
    *,
    base_config: ConservativeResearchUniverseConfig | None = None,
) -> ConservativeResearchUniverseConfig:
    base = base_config or ConservativeResearchUniverseConfig()
    return ConservativeResearchUniverseConfig(
        enabled=not bool(args.disable_conservative_research_universe),
        common_stocks_only=base.common_stocks_only,
        allowed_exchanges=parse_allowed_exchanges(args.research_allowed_exchanges) if args.research_allowed_exchanges else base.allowed_exchanges,
        min_price=args.research_min_price if args.research_min_price is not None else base.min_price,
        min_history_days=args.research_min_history_days if args.research_min_history_days is not None else base.min_history_days,
        liquidity_short_lookback=(
            args.research_liquidity_short_lookback_days
            if args.research_liquidity_short_lookback_days is not None
            else base.liquidity_short_lookback
        ),
        liquidity_long_lookback=(
            args.research_liquidity_long_lookback_days
            if args.research_liquidity_long_lookback_days is not None
            else base.liquidity_long_lookback
        ),
        min_median_dollar_volume_20d=(
            args.research_min_median_dollar_volume_20d
            if args.research_min_median_dollar_volume_20d is not None
            else base.min_median_dollar_volume_20d
        ),
        min_median_dollar_volume_60d=(
            args.research_min_median_dollar_volume_60d
            if args.research_min_median_dollar_volume_60d is not None
            else base.min_median_dollar_volume_60d
        ),
        max_zero_volume_day_ratio_60d=(
            args.research_max_zero_volume_day_ratio_60d
            if args.research_max_zero_volume_day_ratio_60d is not None
            else base.max_zero_volume_day_ratio_60d
        ),
        min_current_dollar_volume_vs_median_20d=(
            args.research_min_current_dollar_volume_vs_median_20d
            if args.research_min_current_dollar_volume_vs_median_20d is not None
            else base.min_current_dollar_volume_vs_median_20d
        ),
        trend_lookback_days=(
            args.research_trend_lookback_days
            if args.research_trend_lookback_days is not None
            else base.trend_lookback_days
        ),
        return_6m_lookback_days=(
            args.research_return_6m_lookback_days
            if args.research_return_6m_lookback_days is not None
            else base.return_6m_lookback_days
        ),
        sma_short_lookback_days=(
            args.research_sma_short_lookback_days
            if args.research_sma_short_lookback_days is not None
            else base.sma_short_lookback_days
        ),
        sma_long_lookback_days=(
            args.research_sma_long_lookback_days
            if args.research_sma_long_lookback_days is not None
            else base.sma_long_lookback_days
        ),
        min_return_6m=args.research_min_return_6m if args.research_min_return_6m is not None else base.min_return_6m,
        max_drawdown_from_252d_high=(
            args.research_max_drawdown_from_252d_high_pct / 100.0
            if args.research_max_drawdown_from_252d_high_pct is not None
            else base.max_drawdown_from_252d_high
        ),
        require_close_above_sma200=(
            False if args.research_disable_close_above_sma200 else base.require_close_above_sma200
        ),
        require_sma50_above_sma200=(
            False if args.research_disable_sma50_above_sma200 else base.require_sma50_above_sma200
        ),
        spike_filter_enabled=False if args.research_disable_spike_filter else base.spike_filter_enabled,
        spike_lookback_days=(
            args.research_spike_lookback_days
            if args.research_spike_lookback_days is not None
            else base.spike_lookback_days
        ),
        max_abs_return_1d_60d=(
            args.research_max_abs_return_1d_60d_pct / 100.0
            if args.research_max_abs_return_1d_60d_pct is not None
            else base.max_abs_return_1d_60d
        ),
        max_true_range_pct_60d=(
            args.research_max_true_range_60d_pct / 100.0
            if args.research_max_true_range_60d_pct is not None
            else base.max_true_range_pct_60d
        ),
    )


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} | {message}", flush=True)


def _progress_bar(current: int, total: int | None, *, width: int = 30) -> str:
    if not total or total <= 0:
        return f"[{'?' * width}]"
    current = max(min(int(current), int(total)), 0)
    filled = int(round(width * current / total))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def _write_progress(
    path: Path | None,
    *,
    phase: str,
    current: int = 0,
    total: int | None = None,
    detail: str = "",
    extra: dict[str, object] | None = None,
) -> None:
    if path is None:
        return
    payload: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "pid": os.getpid(),
        "phase": phase,
        "current": int(current),
        "total": int(total) if total is not None else None,
        "detail": detail,
    }
    if total:
        payload["percent"] = round(100.0 * min(max(current, 0), total) / total, 2)
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def _report_progress(
    path: Path | None,
    *,
    phase: str,
    current: int = 0,
    total: int | None = None,
    detail: str = "",
    width: int = 30,
    extra: dict[str, object] | None = None,
) -> None:
    _write_progress(path, phase=phase, current=current, total=total, detail=detail, extra=extra)
    if total:
        percent = 100.0 * min(max(current, 0), total) / total
        _log(f"{_progress_bar(current, total, width=width)} {percent:5.1f}% | {phase} | {detail}")
    else:
        _log(f"{_progress_bar(current, total, width=width)}       | {phase} | {detail}")


def _daily_features_path(dataset_root: Path) -> Path:
    return preferred_stock_feature_path(dataset_root)


def _add_static_metadata(stock_features: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    required_columns = {
        "gics_sector",
        "gics_sub_industry",
        "sector",
        "industry",
        "type",
        "isin",
        "is_delisted",
        "delisted_date",
        "metadata_source",
    }
    missing_columns = [column for column in required_columns if column not in stock_features.columns]
    if not missing_columns:
        return stock_features
    metadata_path = dataset_root / "raw" / "eodhd_equity_metadata.csv"
    if not metadata_path.exists():
        out = stock_features.copy()
        out["gics_sector"] = out.get("gics_sector", "Unknown")
        out["gics_sub_industry"] = out.get("gics_sub_industry", "Unknown")
        return out
    header = pd.read_csv(metadata_path, nrows=0).columns.tolist()
    usecols = [column for column in ["ticker", *sorted(required_columns)] if column in header]
    metadata = pd.read_csv(metadata_path, usecols=usecols)
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
    metadata = metadata.drop_duplicates("ticker", keep="last")
    merge_columns = ["ticker", *[column for column in missing_columns if column in metadata.columns]]
    out = stock_features.merge(metadata[merge_columns], on="ticker", how="left")
    if "gics_sector" not in out.columns:
        out["gics_sector"] = "Unknown"
    if "gics_sub_industry" not in out.columns:
        out["gics_sub_industry"] = "Unknown"
    if "sector" in out.columns:
        out["gics_sector"] = out["gics_sector"].fillna(out["sector"]).replace("", "Unknown")
    if "industry" in out.columns:
        out["gics_sub_industry"] = out["gics_sub_industry"].fillna(out["industry"]).replace("", "Unknown")
    out["gics_sector"] = out["gics_sector"].fillna("Unknown").replace("", "Unknown")
    out["gics_sub_industry"] = out["gics_sub_industry"].fillna("Unknown").replace("", "Unknown")
    return out


NUMERIC_BAR_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "raw_close",
    "adjusted_close",
    "volume",
    "vwap",
    "transactions",
    "timestamp_ms",
    "adjustment_factor",
    "dollar_volume",
}


def _parse_iso_date(value: str) -> date:
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _weekday_dates(start_date: date, end_date: date) -> list[date]:
    out: list[date] = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            out.append(current)
        current += timedelta(days=1)
    return out


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _latest_inference_dir(dataset_root: Path, override: str) -> Path:
    return Path(override) if override else dataset_root / "processed" / LATEST_INFERENCE_DIRNAME


def _local_data_end_date(dataset_root: Path, latest_dir: Path) -> str | None:
    latest_manifest = _load_json(latest_dir / "run_manifest.json")
    for key in ("local_data_end_date", "data_max_date"):
        value = latest_manifest.get(key)
        if value:
            return str(value)[:10]
    raw_manifest = _load_json(dataset_root / "raw" / "eodhd_fetch_manifest.json")
    value = raw_manifest.get("end_date")
    return str(value)[:10] if value else None


def _parse_bar_row(row: dict[str, str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in row.items():
        if value == "":
            parsed[key] = None
        elif key in NUMERIC_BAR_FIELDS:
            parsed[key] = float(value)
        elif key == "adjusted":
            parsed[key] = str(value).strip().lower() in {"true", "1", "yes"}
        elif key in {"ticker", "eodhd_symbol", "exchange"}:
            parsed[key] = str(value).upper() if value is not None else value
        else:
            parsed[key] = value
    return parsed


def _bar_key(row: dict[str, object]) -> tuple[str, str, bool]:
    return (str(row.get("ticker") or "").upper(), str(row.get("date") or "")[:10], bool(row.get("adjusted", True)))


def _load_bar_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(_parse_bar_row(row))
    return rows


def _write_bar_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EODHD_DAILY_BAR_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in EODHD_DAILY_BAR_HEADERS})


def _merge_bar_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[str, str, bool], dict[str, object]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        date_value = str(row.get("date") or "")[:10]
        if not ticker or not date_value:
            continue
        normalized = dict(row)
        normalized["ticker"] = ticker
        normalized["date"] = date_value
        merged[_bar_key(normalized)] = normalized
    return sorted(merged.values(), key=lambda item: (str(item["ticker"]), str(item["date"])))


def _tail_bar_rows(rows: Sequence[dict[str, object]], rows_per_ticker: int) -> list[dict[str, object]]:
    rows_per_ticker = max(int(rows_per_ticker), 1)
    out: list[dict[str, object]] = []
    merged = _merge_bar_rows(rows)
    if not merged:
        return out
    for _, group in pd.DataFrame(merged).groupby("ticker", sort=False):
        out.extend(group.sort_values("date").tail(rows_per_ticker).to_dict("records"))
    return sorted(out, key=lambda item: (str(item["ticker"]), str(item["date"])))


def _max_rows_per_ticker(rows: Sequence[dict[str, object]]) -> int:
    if not rows:
        return 0
    counts: dict[str, int] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        if ticker:
            counts[ticker] = counts.get(ticker, 0) + 1
    return max(counts.values()) if counts else 0


def _stream_recent_bar_rows(
    path: Path,
    *,
    rows_per_ticker: int,
    wanted_tickers: set[str] | None = None,
    progress_every_rows: int = 2_000_000,
    progress_path: Path | None = None,
    progress_width: int = 30,
) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    tails: dict[str, deque[dict[str, str]]] = {}
    rows_seen = 0
    start_time = time.monotonic()
    _log(f"bootstrapping latest raw cache from {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_seen += 1
            ticker = str(row.get("ticker") or "").upper()
            if not ticker or (wanted_tickers is not None and ticker not in wanted_tickers):
                continue
            if ticker not in tails:
                tails[ticker] = deque(maxlen=rows_per_ticker)
            tails[ticker].append(row)
            if progress_every_rows and rows_seen % progress_every_rows == 0:
                _report_progress(
                    progress_path,
                    phase="bootstrap_recent_raw_cache",
                    current=rows_seen,
                    total=None,
                    detail=(
                        f"read {rows_seen:,} raw rows; retained {sum(len(items) for items in tails.values()):,} "
                        f"rows for {len(tails):,} tickers in {time.monotonic() - start_time:.1f}s"
                    ),
                    width=progress_width,
                )
    parsed = [_parse_bar_row(row) for items in tails.values() for row in items]
    parsed = _merge_bar_rows(parsed)
    _log(f"raw cache bootstrap retained {len(parsed):,} rows for {len(tails):,} tickers")
    return parsed


def _looks_like_test_symbol(row: dict[str, str]) -> bool:
    ticker = str(row.get("ticker") or row.get("symbol") or "").upper()
    name = str(row.get("name") or "").upper()
    if ticker in {"ZVZZT", "ZWZZT", "NTEST"}:
        return True
    return "TEST" in name and ("STOCK" in name or "SYMBOL" in name or "ISSUE" in name)


def _load_universe(
    dataset_root: Path,
    max_tickers: int = 0,
    *,
    include_test_symbols: bool = False,
) -> tuple[list[dict[str, str]], set[str], set[str], dict[str, str]]:
    path = dataset_root / "raw" / "eodhd_common_stock_universe.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ticker = str(row.get("ticker") or row.get("symbol") or "").upper()
            symbol = str(row.get("eodhd_symbol") or "").upper()
            if not ticker or not symbol:
                continue
            row = dict(row)
            row["ticker"] = ticker
            row["eodhd_symbol"] = symbol
            if not include_test_symbols and _looks_like_test_symbol(row):
                continue
            rows.append(row)
            if max_tickers and len(rows) >= max_tickers:
                break
    stock_tickers = {row["ticker"] for row in rows}
    context_tickers = {str(ticker).upper() for ticker in MARKET_CONTEXT_TICKERS}
    exchange_by_ticker = {
        row["ticker"]: str(row.get("exchange") or "").upper()
        for row in rows
        if row.get("exchange")
    }
    return rows, stock_tickers, context_tickers, exchange_by_ticker


def _max_date_from_rows(rows: Sequence[dict[str, object]]) -> str | None:
    dates = [str(row.get("date") or "")[:10] for row in rows if row.get("date")]
    return max(dates) if dates else None


def _filter_bulk_rows(
    rows: Sequence[dict[str, object]],
    *,
    exchange: str,
    stock_tickers: set[str],
    context_tickers: set[str],
    exchange_by_ticker: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    normalized = normalize_eodhd_bulk_eod_rows(rows, exchange=exchange, adjusted=True)
    stock_rows = [row for row in normalized if str(row.get("ticker") or "").upper() in stock_tickers]
    stock_rows = _apply_exchange_by_ticker(stock_rows, exchange_by_ticker)
    context_rows = [row for row in normalized if str(row.get("ticker") or "").upper() in context_tickers]
    return stock_rows, context_rows


def _apply_exchange_by_ticker(
    rows: Sequence[dict[str, object]],
    exchange_by_ticker: dict[str, str],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        normalized = dict(row)
        ticker = str(normalized.get("ticker") or "").upper()
        if ticker in exchange_by_ticker:
            normalized["exchange"] = exchange_by_ticker[ticker]
        out.append(normalized)
    return out


def _fetch_missing_eod_rows(
    *,
    client: EODHDRESTClient,
    dataset_root: Path,
    latest_dir: Path,
    exchange: str,
    fetch_start_date: str,
    fetch_end_date: str,
    skip_fetch: bool,
    stock_tickers: set[str],
    context_tickers: set[str],
    exchange_by_ticker: dict[str, str],
    progress_path: Path | None = None,
    progress_width: int = 30,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    local_end = _local_data_end_date(dataset_root, latest_dir)
    latest_probe_rows: list[dict[str, object]] = []
    latest_probe_date: str | None = None
    api_calls_estimate = 0

    if skip_fetch:
        _report_progress(progress_path, phase="fetch_eod", current=1, total=1, detail="skipped", width=progress_width)
        return [], [], {
            "skipped_fetch": True,
            "local_data_end_date_before_fetch": local_end,
            "planned_fetch_dates": [],
            "estimated_bulk_api_call_units": 0,
            "fetch_status": "skipped",
        }

    if not fetch_end_date:
        _log(f"probing latest EODHD bulk EOD date for {exchange}")
        latest_probe_rows = client.get_bulk_eod_last_day(exchange)
        latest_stock_probe, latest_context_probe = _filter_bulk_rows(
            latest_probe_rows,
            exchange=exchange,
            stock_tickers=stock_tickers,
            context_tickers=context_tickers,
            exchange_by_ticker=exchange_by_ticker,
        )
        latest_probe_date = _max_date_from_rows([*latest_stock_probe, *latest_context_probe])
        api_calls_estimate += 100
        if not latest_probe_date:
            raise SystemExit("EODHD latest bulk EOD probe returned no usable rows.")
        fetch_end_date = latest_probe_date

    if fetch_start_date:
        start = _parse_iso_date(fetch_start_date)
    elif local_end:
        start = _parse_iso_date(local_end) + timedelta(days=1)
    else:
        start = _parse_iso_date(fetch_end_date)
    end = _parse_iso_date(fetch_end_date)
    planned_dates = _weekday_dates(start, end) if start <= end else []
    if not planned_dates:
        _report_progress(
            progress_path,
            phase="fetch_eod",
            current=1,
            total=1,
            detail=f"already current through {fetch_end_date}",
            width=progress_width,
        )
        return [], [], {
            "skipped_fetch": False,
            "local_data_end_date_before_fetch": local_end,
            "latest_eodhd_bulk_date": latest_probe_date or fetch_end_date,
            "planned_fetch_dates": [],
            "estimated_bulk_api_call_units": api_calls_estimate,
            "fetch_status": "already_current",
        }

    if not latest_probe_rows or (latest_probe_date and latest_probe_date not in {item.isoformat() for item in planned_dates}):
        api_calls_estimate += 100 * len(planned_dates)
    else:
        api_calls_estimate += 100 * max(len(planned_dates) - 1, 0)
    _log(
        "planned EODHD bulk fetch: "
        f"{planned_dates[0].isoformat()} to {planned_dates[-1].isoformat()} "
        f"({len(planned_dates)} weekdays), estimated {api_calls_estimate} bulk call units"
    )

    stock_rows_all: list[dict[str, object]] = []
    context_rows_all: list[dict[str, object]] = []
    per_date: list[dict[str, object]] = []
    latest_rows_by_date = {latest_probe_date: latest_probe_rows} if latest_probe_date and latest_probe_rows else {}
    for planned_date in planned_dates:
        date_index = planned_dates.index(planned_date) + 1
        planned_date_str = planned_date.isoformat()
        try:
            raw_rows = latest_rows_by_date.get(planned_date_str)
            source = "latest_probe" if raw_rows is not None else "dated_bulk"
            if raw_rows is None:
                raw_rows = client.get_bulk_eod_last_day(exchange, date=planned_date_str)
            stock_rows, context_rows = _filter_bulk_rows(
                raw_rows,
                exchange=exchange,
                stock_tickers=stock_tickers,
                context_tickers=context_tickers,
                exchange_by_ticker=exchange_by_ticker,
            )
            stock_rows_all.extend(stock_rows)
            context_rows_all.extend(context_rows)
            per_date.append(
                {
                    "date": planned_date_str,
                    "source": source,
                    "raw_rows": len(raw_rows),
                    "stock_rows": len(stock_rows),
                    "context_rows": len(context_rows),
                    "status": "ok" if raw_rows else "empty",
                }
            )
            _log(
                f"bulk EOD {planned_date_str}: raw={len(raw_rows):,}, "
                f"stocks={len(stock_rows):,}, context={len(context_rows):,}"
            )
            _report_progress(
                progress_path,
                phase="fetch_bulk_eod",
                current=date_index,
                total=len(planned_dates),
                detail=f"{planned_date_str} stocks={len(stock_rows):,} context={len(context_rows):,}",
                width=progress_width,
            )
        except EODHDRateLimitError:
            raise
        except EODHDAPIError as exc:
            per_date.append({"date": planned_date_str, "status": "error", "error": str(exc)[:500]})
            _log(f"bulk EOD {planned_date_str} failed: {exc}")
            _report_progress(
                progress_path,
                phase="fetch_bulk_eod",
                current=date_index,
                total=len(planned_dates),
                detail=f"{planned_date_str} error={str(exc)[:120]}",
                width=progress_width,
            )

    existing_context_keys = {(str(row["ticker"]).upper(), str(row["date"])[:10]) for row in context_rows_all}
    fallback_rows: list[dict[str, object]] = []
    for planned_date in planned_dates:
        planned_date_str = planned_date.isoformat()
        missing_context = [ticker for ticker in sorted(context_tickers) if (ticker, planned_date_str) not in existing_context_keys]
        if not missing_context:
            continue
        _log(f"context fallback for {planned_date_str}: {len(missing_context)} tickers")
        for ticker in missing_context:
            symbol = eodhd_symbol_for_code(ticker, default_exchange=exchange)
            raw_rows = client.get_eod(symbol, from_date=planned_date_str, to_date=planned_date_str)
            fallback_rows.extend(
                normalize_eodhd_eod_rows(raw_rows, symbol=symbol, ticker=ticker, exchange=exchange, adjusted=True)
            )
    if fallback_rows:
        context_rows_all.extend(fallback_rows)

    return stock_rows_all, context_rows_all, {
        "skipped_fetch": False,
        "local_data_end_date_before_fetch": local_end,
        "latest_eodhd_bulk_date": latest_probe_date or fetch_end_date,
        "planned_fetch_dates": [item.isoformat() for item in planned_dates],
        "estimated_bulk_api_call_units": api_calls_estimate,
        "stock_rows_fetched": len(stock_rows_all),
        "context_rows_fetched": len(context_rows_all),
        "per_date": per_date,
        "fetch_status": "ok",
    }


def _update_daily_update_file(path: Path, incoming_rows: Sequence[dict[str, object]]) -> None:
    if not incoming_rows:
        return
    existing = _load_bar_csv(path)
    merged = _merge_bar_rows([*existing, *incoming_rows])
    _write_bar_csv(path, merged)


def _build_recent_raw_cache(
    *,
    dataset_root: Path,
    latest_dir: Path,
    stock_tickers: set[str],
    context_tickers: set[str],
    exchange_by_ticker: dict[str, str],
    fetched_stock_rows: Sequence[dict[str, object]],
    fetched_context_rows: Sequence[dict[str, object]],
    rows_per_ticker: int,
    progress_every_rows: int,
    progress_path: Path | None = None,
    progress_width: int = 30,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_dir = dataset_root / "raw"
    stock_cache_path = latest_dir / RECENT_STOCK_BARS_FILENAME
    context_cache_path = latest_dir / RECENT_CONTEXT_BARS_FILENAME
    stock_update_path = raw_dir / STOCK_UPDATE_FILENAME
    context_update_path = raw_dir / CONTEXT_UPDATE_FILENAME

    _update_daily_update_file(stock_update_path, fetched_stock_rows)
    _update_daily_update_file(context_update_path, fetched_context_rows)

    if stock_cache_path.exists():
        _report_progress(
            progress_path,
            phase="load_recent_raw_cache",
            current=1,
            total=3,
            detail=f"loading existing {stock_cache_path.name}",
            width=progress_width,
        )
        stock_source = _load_bar_csv(stock_cache_path)
        if _max_rows_per_ticker(stock_source) < rows_per_ticker:
            _log(
                f"{stock_cache_path.name} has fewer than {rows_per_ticker} rows/ticker; "
                "rebuilding from full raw bars"
            )
            stock_source = _stream_recent_bar_rows(
                raw_dir / "eodhd_stock_bars.csv",
                rows_per_ticker=rows_per_ticker,
                wanted_tickers=stock_tickers,
                progress_every_rows=progress_every_rows,
                progress_path=progress_path,
                progress_width=progress_width,
            )
            stock_source.extend(_load_bar_csv(stock_update_path))
    else:
        stock_source = _stream_recent_bar_rows(
            raw_dir / "eodhd_stock_bars.csv",
            rows_per_ticker=rows_per_ticker,
            wanted_tickers=stock_tickers,
            progress_every_rows=progress_every_rows,
            progress_path=progress_path,
            progress_width=progress_width,
        )
        stock_source.extend(_load_bar_csv(stock_update_path))
    stock_source.extend(fetched_stock_rows)
    stock_recent = _apply_exchange_by_ticker(_tail_bar_rows(stock_source, rows_per_ticker), exchange_by_ticker)

    if context_cache_path.exists():
        _report_progress(
            progress_path,
            phase="load_recent_raw_cache",
            current=2,
            total=3,
            detail=f"loading existing {context_cache_path.name}",
            width=progress_width,
        )
        context_source = _load_bar_csv(context_cache_path)
        if _max_rows_per_ticker(context_source) < rows_per_ticker:
            _log(
                f"{context_cache_path.name} has fewer than {rows_per_ticker} rows/ticker; "
                "rebuilding from full raw bars"
            )
            _report_progress(
                progress_path,
                phase="load_recent_raw_cache",
                current=2,
                total=3,
                detail=f"loading {raw_dir / 'market_context_bars.csv'}",
                width=progress_width,
            )
            context_source = [
                row
                for row in _load_bar_csv(raw_dir / "market_context_bars.csv")
                if str(row.get("ticker") or "").upper() in context_tickers
            ]
            context_source.extend(_load_bar_csv(context_update_path))
    else:
        _report_progress(
            progress_path,
            phase="load_recent_raw_cache",
            current=2,
            total=3,
            detail=f"loading {raw_dir / 'market_context_bars.csv'}",
            width=progress_width,
        )
        context_source = [
            row
            for row in _load_bar_csv(raw_dir / "market_context_bars.csv")
            if str(row.get("ticker") or "").upper() in context_tickers
        ]
        context_source.extend(_load_bar_csv(context_update_path))
    context_source.extend(fetched_context_rows)
    context_recent = _tail_bar_rows(context_source, rows_per_ticker)

    _write_bar_csv(stock_cache_path, stock_recent)
    _write_bar_csv(context_cache_path, context_recent)
    _report_progress(
        progress_path,
        phase="load_recent_raw_cache",
        current=3,
        total=3,
        detail=f"stock_rows={len(stock_recent):,} context_rows={len(context_recent):,}",
        width=progress_width,
    )
    return stock_recent, context_recent


def _compute_daily_features_with_progress(
    rows: Sequence[dict[str, object]],
    *,
    label: str,
    progress_path: Path | None = None,
    progress_every_tickers: int = 500,
    progress_width: int = 30,
) -> list[dict[str, object]]:
    if not rows:
        _report_progress(progress_path, phase=f"compute_{label}_features", current=1, total=1, detail="no rows", width=progress_width)
        return []
    sorted_rows = sorted(rows, key=lambda item: (str(item.get("ticker") or ""), str(item.get("date") or "")))
    total_tickers = len({str(row.get("ticker") or "").upper() for row in sorted_rows if row.get("ticker")})
    feature_rows: list[dict[str, object]] = []
    group: list[dict[str, object]] = []
    current_ticker = ""
    processed = 0
    start_time = time.monotonic()

    def flush_group() -> None:
        nonlocal group, current_ticker, processed
        if not group:
            return
        feature_rows.extend(compute_daily_features(group))
        processed += 1
        if processed == 1 or processed % max(progress_every_tickers, 1) == 0 or processed == total_tickers:
            _report_progress(
                progress_path,
                phase=f"compute_{label}_features",
                current=processed,
                total=total_tickers,
                detail=(
                    f"ticker={current_ticker} features={len(feature_rows):,} "
                    f"elapsed={time.monotonic() - start_time:.1f}s"
                ),
                width=progress_width,
            )
        group = []

    for row in sorted_rows:
        ticker = str(row.get("ticker") or "").upper()
        if group and ticker != current_ticker:
            flush_group()
        current_ticker = ticker
        group.append(dict(row))
    flush_group()
    return sorted(feature_rows, key=lambda item: (str(item["ticker"]), str(item["date"])))


def _materialize_latest_inference_features(
    *,
    dataset_root: Path,
    latest_dir: Path,
    stock_bars: Sequence[dict[str, object]],
    context_bars: Sequence[dict[str, object]],
    skip_fundamentals: bool,
    skip_sentiment: bool,
    progress_path: Path | None = None,
    progress_width: int = 30,
    feature_progress_every_tickers: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = dataset_root / "raw"
    latest_dir.mkdir(parents=True, exist_ok=True)
    _log(f"computing latest stock features from {len(stock_bars):,} bounded raw rows")
    stock_features = _compute_daily_features_with_progress(
        stock_bars,
        label="stock",
        progress_path=progress_path,
        progress_every_tickers=feature_progress_every_tickers,
        progress_width=progress_width,
    )
    stock_frame = _add_static_metadata(pd.DataFrame(stock_features), dataset_root)
    stock_features = stock_frame.to_dict("records")

    if not skip_fundamentals:
        _report_progress(
            progress_path,
            phase="join_fundamentals",
            current=0,
            total=None,
            detail="loading local fundamental feature rows",
            width=progress_width,
        )
        symbols = sorted({str(row.get("eodhd_symbol") or "").upper() for row in stock_bars if row.get("eodhd_symbol")})
        _log(f"joining saved fundamentals for {len(symbols):,} symbols")
        fundamental_rows = load_fundamental_feature_rows(raw_dir / "eodhd_fundamentals_raw", symbols=symbols)
        stock_features = add_fundamental_features(stock_features, fundamental_rows)
    if not skip_sentiment:
        _report_progress(
            progress_path,
            phase="join_sentiment",
            current=0,
            total=None,
            detail="loading local sentiment rows",
            width=progress_width,
        )
        wanted = {str(row.get("ticker") or "").upper() for row in stock_bars if row.get("ticker")}
        min_date = min(str(row.get("date"))[:10] for row in stock_bars if row.get("date"))
        _log("joining saved sentiment rows")
        sentiment_rows = [
            row
            for row in load_sentiment_rows(raw_dir / "eodhd_sentiment_daily.csv")
            if str(row.get("ticker") or "").upper() in wanted and str(row.get("date") or "")[:10] >= min_date
        ]
        stock_features = add_sentiment_features(stock_features, sentiment_rows)

    context_features = _compute_daily_features_with_progress(
        context_bars,
        label="context",
        progress_path=progress_path,
        progress_every_tickers=max(min(feature_progress_every_tickers, 10), 1),
        progress_width=progress_width,
    )
    stock_frame = pd.DataFrame(stock_features)
    context_frame = pd.DataFrame(context_features)
    if not stock_frame.empty:
        stock_frame["ticker"] = stock_frame["ticker"].astype(str).str.upper()
        stock_frame["date"] = stock_frame["date"].astype(str)
    if not context_frame.empty:
        context_frame["ticker"] = context_frame["ticker"].astype(str).str.upper()
        context_frame["date"] = context_frame["date"].astype(str)

    stock_path = latest_dir / LATEST_STOCK_FEATURES_FILENAME
    context_path = latest_dir / LATEST_CONTEXT_FEATURES_FILENAME
    stock_frame.to_csv(stock_path, index=False)
    context_frame.to_csv(context_path, index=False)
    prediction_windows = []
    if not stock_frame.empty:
        latest = stock_frame.sort_values(["ticker", "date"]).groupby("ticker", sort=False).tail(1)
        prediction_windows = latest[["ticker", "date"]].rename(columns={"date": "anchor_date"}).to_dict("records")
    pd.DataFrame(prediction_windows).to_csv(latest_dir / "prediction_windows.csv", index=False)
    _report_progress(
        progress_path,
        phase="materialize_latest_inference",
        current=1,
        total=1,
        detail=f"stock_features={len(stock_frame):,} context_features={len(context_frame):,}",
        width=progress_width,
    )
    return stock_frame, context_frame


def _merge_feature_update_frame(
    path: Path,
    incoming: pd.DataFrame,
    *,
    min_date: str | None,
) -> int:
    if incoming.empty or not min_date:
        return 0
    if not {"ticker", "date"}.issubset(incoming.columns):
        return 0
    updates = incoming.copy()
    updates["ticker"] = updates["ticker"].astype(str).str.upper()
    updates["date"] = updates["date"].astype(str)
    updates = updates[updates["date"] >= str(min_date)].copy()
    if updates.empty:
        return 0

    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path, low_memory=False)
        if not existing.empty and {"ticker", "date"}.issubset(existing.columns):
            existing["ticker"] = existing["ticker"].astype(str).str.upper()
            existing["date"] = existing["date"].astype(str)
            updates = pd.concat([existing, updates], ignore_index=True, sort=False)

    updates = updates.drop_duplicates(["ticker", "date"], keep="last")
    updates = updates.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    updates.to_csv(temp_path, index=False)
    temp_path.replace(path)
    return int(len(updates))


def _persist_incremental_feature_updates(
    *,
    dataset_root: Path,
    latest_dir: Path,
    stock_features: pd.DataFrame,
    context_features: pd.DataFrame,
    min_update_date: str | None,
    progress_path: Path | None,
    progress_width: int,
) -> dict[str, object]:
    processed_dir = dataset_root / "processed"
    if not min_update_date:
        manifest = {
            "enabled": True,
            "persisted": False,
            "reason": "no fetched update dates",
            "stock_update_rows": 0,
            "context_update_rows": 0,
        }
        write_json(latest_dir / FEATURE_UPDATE_MANIFEST_FILENAME, manifest)
        return manifest
    _report_progress(
        progress_path,
        phase="persist_incremental_feature_updates",
        current=0,
        total=None,
        detail=f"min_update_date={min_update_date}",
        width=progress_width,
    )
    stock_path = processed_dir / STOCK_FEATURE_UPDATE_FILENAME
    context_path = processed_dir / CONTEXT_FEATURE_UPDATE_FILENAME
    stock_rows = _merge_feature_update_frame(stock_path, stock_features, min_date=min_update_date)
    context_rows = _merge_feature_update_frame(context_path, context_features, min_date=min_update_date)
    manifest = {
        "enabled": True,
        "persisted": bool(stock_rows or context_rows),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "mode": "latest_inference_incremental_feature_updates",
        "dataset_root": str(dataset_root.resolve()),
        "latest_inference_dir": str(latest_dir.resolve()),
        "min_update_date": min_update_date,
        "stock_update_file": str(stock_path.resolve()),
        "context_update_file": str(context_path.resolve()),
        "stock_update_rows": stock_rows,
        "context_update_rows": context_rows,
        "notes": [
            "These sidecar files are retrain inputs, not production prediction outputs.",
            "The true-full retrain wrapper consolidates them into processed/daily_features.csv and processed/market_context_features.csv before normalization.",
        ],
    }
    write_json(latest_dir / FEATURE_UPDATE_MANIFEST_FILENAME, manifest)
    write_json(processed_dir / FEATURE_UPDATE_MANIFEST_FILENAME, manifest)
    _report_progress(
        progress_path,
        phase="persist_incremental_feature_updates",
        current=1,
        total=1,
        detail=f"stock_update_rows={stock_rows:,} context_update_rows={context_rows:,}",
        width=progress_width,
    )
    return manifest


def _load_latest_feature_cache(
    latest_dir: Path,
    *,
    stock_tickers: set[str],
    context_tickers: set[str],
    min_rows_per_ticker: int = 0,
    progress_path: Path | None = None,
    progress_width: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    stock_path = latest_dir / LATEST_STOCK_FEATURES_FILENAME
    context_path = latest_dir / LATEST_CONTEXT_FEATURES_FILENAME
    if not stock_path.exists() or not context_path.exists():
        return None
    _report_progress(
        progress_path,
        phase="load_latest_feature_cache",
        current=0,
        total=None,
        detail=f"loading {stock_path.name} and {context_path.name}",
        width=progress_width,
    )
    stock_features = pd.read_csv(stock_path, low_memory=False)
    context_features = pd.read_csv(context_path, low_memory=False)
    if stock_features.empty or context_features.empty:
        return None
    stock_features["ticker"] = stock_features["ticker"].astype(str).str.upper()
    stock_features["date"] = stock_features["date"].astype(str)
    context_features["ticker"] = context_features["ticker"].astype(str).str.upper()
    context_features["date"] = context_features["date"].astype(str)
    stock_features = stock_features[stock_features["ticker"].isin(stock_tickers)].copy()
    context_features = context_features[context_features["ticker"].isin(context_tickers)].copy()
    if min_rows_per_ticker > 0:
        stock_max_rows = int(stock_features.groupby("ticker").size().max()) if not stock_features.empty else 0
        context_max_rows = int(context_features.groupby("ticker").size().max()) if not context_features.empty else 0
        if min(stock_max_rows, context_max_rows) < int(min_rows_per_ticker):
            _log(
                f"latest feature cache has fewer than {min_rows_per_ticker} rows/ticker "
                f"(stock max={stock_max_rows}, context max={context_max_rows}); rebuilding"
            )
            return None
    _report_progress(
        progress_path,
        phase="load_latest_feature_cache",
        current=1,
        total=1,
        detail=f"stock_features={len(stock_features):,} context_features={len(context_features):,}",
        width=progress_width,
    )
    return stock_features, context_features


def _refresh_latest_inference_dataset(
    args: argparse.Namespace,
    run_dirs: Sequence[Path],
    *,
    research_config: ConservativeResearchUniverseConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], Path]:
    dataset_root = Path(args.dataset_root)
    latest_dir = _latest_inference_dir(dataset_root, args.latest_inference_dir)
    required_rows_per_ticker = (
        research_config.required_recent_rows_per_ticker if research_config.enabled else 0
    )
    requested_rows_per_ticker = max(int(args.recent_raw_rows_per_ticker), 125, required_rows_per_ticker)
    latest_dir.mkdir(parents=True, exist_ok=True)
    progress_path = latest_dir / "progress.json"
    _report_progress(
        progress_path,
        phase="preflight",
        current=0,
        total=None,
        detail="loading universe and model indexes",
        width=args.progress_bar_width,
    )
    universe_rows, stock_tickers, context_tickers, exchange_by_ticker = _load_universe(
        dataset_root,
        max_tickers=args.max_tickers,
        include_test_symbols=args.include_test_symbols,
    )
    credentials = load_eodhd_credentials(args.credentials_path)
    if not args.skip_fetch and not credentials.api_key:
        raise SystemExit("EODHD_API_KEY is missing. Set it in EODHD_api_key or the environment.")
    client = EODHDRESTClient(
        credentials,
        rate_limiter=RateLimiter(
            max_calls=args.rate_limit_calls,
            period_seconds=args.rate_limit_period_seconds,
        ),
    )
    for run_dir in run_dirs:
        _resolve_model_index_path(run_dir)
    stock_rows, context_rows, fetch_manifest = _fetch_missing_eod_rows(
        client=client,
        dataset_root=dataset_root,
        latest_dir=latest_dir,
        exchange=args.exchange.upper(),
        fetch_start_date=args.fetch_start_date,
        fetch_end_date=args.fetch_end_date,
        skip_fetch=args.skip_fetch,
        stock_tickers=stock_tickers,
        context_tickers=context_tickers,
        exchange_by_ticker=exchange_by_ticker,
        progress_path=progress_path,
        progress_width=args.progress_bar_width,
    )
    if not args.force_rebuild_latest_inference and not stock_rows and not context_rows:
        cached = _load_latest_feature_cache(
            latest_dir,
            stock_tickers=stock_tickers,
            context_tickers=context_tickers,
            min_rows_per_ticker=required_rows_per_ticker,
            progress_path=progress_path,
            progress_width=args.progress_bar_width,
        )
        if cached is not None:
            stock_features, context_features = cached
            local_max = max(
                str(stock_features["date"].max()) if not stock_features.empty else "",
                str(context_features["date"].max()) if not context_features.empty else "",
            )
            manifest = {
                "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "mode": "latest_inference_daily_refresh",
                "dataset_root": str(dataset_root.resolve()),
                "latest_inference_dir": str(latest_dir.resolve()),
                "universe_rows_loaded": len(universe_rows),
                "stock_tickers": len(stock_tickers),
                "context_tickers": len(context_tickers),
                "recent_raw_rows_per_ticker": requested_rows_per_ticker,
                "latest_stock_feature_rows": len(stock_features),
                "latest_context_feature_rows": len(context_features),
                "local_data_end_date": local_max or None,
                "cache_reused": True,
                "skip_fundamentals": bool(args.skip_fundamentals),
                "skip_sentiment": bool(args.skip_sentiment),
                "include_test_symbols": bool(args.include_test_symbols),
                "fetch": fetch_manifest,
            }
            write_json(latest_dir / "run_manifest.json", manifest)
            _report_progress(
                progress_path,
                phase="latest_inference_ready",
                current=1,
                total=1,
                detail=f"reused cache local_data_end_date={local_max}",
                width=args.progress_bar_width,
            )
            return stock_features, context_features, manifest, latest_dir
    raw_stock, raw_context = _build_recent_raw_cache(
        dataset_root=dataset_root,
        latest_dir=latest_dir,
        stock_tickers=stock_tickers,
        context_tickers=context_tickers,
        exchange_by_ticker=exchange_by_ticker,
        fetched_stock_rows=stock_rows,
        fetched_context_rows=context_rows,
        rows_per_ticker=requested_rows_per_ticker,
        progress_every_rows=args.progress_every_rows,
        progress_path=progress_path,
        progress_width=args.progress_bar_width,
    )
    stock_features, context_features = _materialize_latest_inference_features(
        dataset_root=dataset_root,
        latest_dir=latest_dir,
        stock_bars=raw_stock,
        context_bars=raw_context,
        skip_fundamentals=args.skip_fundamentals,
        skip_sentiment=args.skip_sentiment,
        progress_path=progress_path,
        progress_width=args.progress_bar_width,
        feature_progress_every_tickers=args.feature_progress_every_tickers,
    )
    planned_dates = [str(value)[:10] for value in fetch_manifest.get("planned_fetch_dates", []) if value]
    min_update_date = min(planned_dates) if planned_dates else None
    feature_update_manifest = (
        {"enabled": False, "persisted": False, "reason": "disabled by CLI"}
        if args.disable_persist_latest_feature_updates
        else _persist_incremental_feature_updates(
            dataset_root=dataset_root,
            latest_dir=latest_dir,
            stock_features=stock_features,
            context_features=context_features,
            min_update_date=min_update_date,
            progress_path=progress_path,
            progress_width=args.progress_bar_width,
        )
    )
    local_max = _max_date_from_rows([*raw_stock, *raw_context])
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "mode": "latest_inference_daily_refresh",
        "dataset_root": str(dataset_root.resolve()),
        "latest_inference_dir": str(latest_dir.resolve()),
        "universe_rows_loaded": len(universe_rows),
        "stock_tickers": len(stock_tickers),
        "context_tickers": len(context_tickers),
        "recent_raw_rows_per_ticker": requested_rows_per_ticker,
        "raw_stock_rows": len(raw_stock),
        "raw_context_rows": len(raw_context),
        "latest_stock_feature_rows": len(stock_features),
        "latest_context_feature_rows": len(context_features),
        "local_data_end_date": local_max,
        "cache_reused": False,
        "skip_fundamentals": bool(args.skip_fundamentals),
        "skip_sentiment": bool(args.skip_sentiment),
        "include_test_symbols": bool(args.include_test_symbols),
        "fetch": fetch_manifest,
        "incremental_feature_updates": feature_update_manifest,
    }
    write_json(latest_dir / "run_manifest.json", manifest)
    _report_progress(
        progress_path,
        phase="latest_inference_ready",
        current=1,
        total=1,
        detail=f"local_data_end_date={local_max}",
        width=args.progress_bar_width,
    )
    return stock_features, context_features, manifest, latest_dir


def _load_recent_stock_features(
    dataset_root: Path,
    *,
    anchor_date: str | None,
    rows_per_ticker: int,
    chunksize: int,
) -> pd.DataFrame:
    path = _daily_features_path(dataset_root)
    if not path.exists():
        raise FileNotFoundError(path)
    rows_per_ticker = max(int(rows_per_ticker), 60)
    chunksize = max(int(chunksize), 1)
    tails = pd.DataFrame()
    total_rows = 0
    kept_rows = 0
    _log(f"streaming latest {rows_per_ticker} rows/ticker from {path}")
    for chunk_index, chunk in enumerate(pd.read_csv(path, chunksize=chunksize), start=1):
        total_rows += len(chunk)
        if chunk.empty:
            continue
        chunk["ticker"] = chunk["ticker"].astype(str).str.upper()
        chunk["date"] = chunk["date"].astype(str)
        if anchor_date:
            chunk = chunk[chunk["date"] <= anchor_date]
        if chunk.empty:
            continue
        combined = chunk if tails.empty else pd.concat([tails, chunk], ignore_index=True)
        combined = combined.sort_values(["ticker", "date"], kind="mergesort")
        tails = combined.groupby("ticker", sort=False).tail(rows_per_ticker).reset_index(drop=True)
        kept_rows = len(tails)
        if chunk_index == 1 or chunk_index % 5 == 0:
            _log(f"read {total_rows:,} source rows; kept {kept_rows:,} recent rows")
    if tails.empty:
        raise SystemExit("No stock feature rows available for the requested anchor date.")
    tails = tails.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    tails = _add_static_metadata(tails, dataset_root)
    _log(f"recent stock feature frame ready: {len(tails):,} rows, {tails['ticker'].nunique():,} tickers")
    return tails


def _load_model_records(run_dirs: Sequence[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for run_dir in run_dirs:
        model_index_path = _resolve_model_index_path(run_dir)
        model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
        for record in model_index.get("models", []):
            records.append({"run_dir": run_dir, "model_index_path": model_index_path, **record})
    if not records:
        raise SystemExit("No models found in the requested run directories.")
    return records


def _leaderboard_for_run(run_dir: Path, task_type: str) -> pd.DataFrame:
    if task_type == "classification":
        candidates = [run_dir / "classification_oos_leaderboard.csv", run_dir / "classification_leaderboard.csv"]
    else:
        candidates = [run_dir / "oos_leaderboard.csv", run_dir / "leaderboard.csv"]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def _build_latest_feature_sets_for_records(
    stock_features: pd.DataFrame,
    context_features: pd.DataFrame,
    *,
    records: Sequence[dict[str, object]],
    window_length: int,
    benchmark_ticker: str,
    anchor_date: str | None,
    max_anchor_lag_days: int,
    eligibility_config: object | None,
    progress_path: Path | None = None,
    progress_width: int = 30,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, list[str]]]:
    requested_tabular = {
        str(record["feature_set"])
        for record in records
        if not str(record.get("input_layout") or "").startswith("sequence")
        and not str(record["feature_set"]).endswith("_sequence")
    }
    cutoff = anchor_date
    stocks = stock_features.copy()
    context = context_features.copy()
    if cutoff:
        stocks = stocks[stocks["date"] <= cutoff]
        context = context[context["date"] <= cutoff]

    _report_progress(
        progress_path,
        phase="build_feature_sets",
        current=0,
        total=None,
        detail="adding market/sector relative return features",
        width=progress_width,
    )
    stocks = add_context_relative_return_features(
        stocks,
        context,
        benchmark_ticker=benchmark_ticker,
    )
    context = add_priority_a_ohlcv_features(context)

    stocks = _filtered_stock_universe(stocks, benchmark_ticker=benchmark_ticker)
    if eligibility_config is not None:
        stocks = add_episode_eligibility_columns(
            stocks,
            eligibility_config,
            benchmark_ticker=benchmark_ticker,
        )
    latest_idx = stocks.groupby("ticker")["date"].idxmax()
    latest = stocks.loc[latest_idx].copy()
    latest = latest[latest["window_row_count"] >= window_length]
    if eligibility_config is not None and "episode_eligible" in latest.columns:
        latest = latest[latest["episode_eligible"]]
    if max_anchor_lag_days >= 0 and not latest.empty:
        latest_dates = pd.to_datetime(latest["date"].astype(str), errors="coerce")
        newest_anchor = latest_dates.max()
        min_anchor = newest_anchor - pd.Timedelta(days=int(max_anchor_lag_days))
        latest = latest[latest_dates >= min_anchor].copy()
    latest = latest.rename(columns={"date": "anchor_date"})
    latest["sector_etf"] = latest["gics_sector"].map(SECTOR_ETF_BY_GICS)
    metadata_columns = [
        "ticker",
        "anchor_date",
        "gics_sector",
        "gics_sub_industry",
        "sector_etf",
        "window_row_count",
        *eligibility_metadata_columns(latest),
    ]
    metadata = latest[metadata_columns].reset_index(drop=True)

    base = metadata.copy()
    feature_sets: dict[str, pd.DataFrame] = {}
    feature_columns: dict[str, list[str]] = {}
    context_summary_cache: dict[bool, pd.DataFrame] = {}
    non_features = {
        "ticker",
        "anchor_date",
        "gics_sector",
        "gics_sub_industry",
        "sector_etf",
        "window_row_count",
        *eligibility_metadata_columns(metadata),
    }
    feature_set_names = sorted(requested_tabular)
    for feature_index, feature_set in enumerate(feature_set_names, start=1):
        parts = set(feature_set.split("_"))
        compact = "compact" in parts
        include_relative = "relative" in parts
        include_market = "market" in parts
        include_sector = "sector" in parts
        include_fundamentals = "fundamentals" in parts
        include_sentiment = "sentiment" in parts
        _log(f"building tabular feature set {feature_set}")
        _report_progress(
            progress_path,
            phase="build_feature_sets",
            current=feature_index,
            total=max(len(feature_set_names), 1),
            detail=f"tabular {feature_set}",
            width=progress_width,
        )
        stock_cols = select_augmented_stock_feature_columns(
            stocks,
            include_relative=include_relative,
            compact=compact,
            include_sentiment=include_sentiment,
            include_fundamentals=include_fundamentals,
        )
        stock_summary = add_window_summaries(
            stocks,
            feature_cols=stock_cols,
            prefix="stock_",
            window_length=window_length,
        )
        frame = base.merge(
            stock_summary,
            left_on=["ticker", "anchor_date"],
            right_on=["ticker", "date"],
            how="left",
        ).drop(columns=["date"], errors="ignore")
        if include_market or include_sector:
            if compact not in context_summary_cache:
                context_cols = select_context_feature_columns(context, compact=compact)
                context_summary_cache[compact] = add_window_summaries(
                    context,
                    feature_cols=context_cols,
                    prefix="context_",
                    window_length=window_length,
                )
            context_summary = context_summary_cache[compact]
            if include_market:
                frame = _merge_context(
                    frame,
                    context_summary,
                    ticker=benchmark_ticker.upper(),
                    ticker_column=None,
                    prefix="market_context_",
                )
            if include_sector:
                frame = _merge_context(
                    frame,
                    context_summary,
                    ticker=None,
                    ticker_column="sector_etf",
                    prefix="sector_context_",
                )
        numeric = [
            col
            for col in frame.columns
            if col not in non_features and pd.api.types.is_numeric_dtype(frame[col])
        ]
        validate_model_feature_columns(numeric, feature_set=feature_set)
        feature_sets[feature_set] = frame[["ticker", "anchor_date", *numeric]].copy()
        feature_columns[feature_set] = numeric
    return metadata, feature_sets, feature_columns


def _prediction_column(frame: pd.DataFrame) -> str:
    score_candidates = [col for col in frame.columns if col.startswith("pred_score_")]
    if len(score_candidates) == 1:
        return score_candidates[0]
    candidates = [col for col in frame.columns if col.startswith("pred_prob_")]
    alias_candidates = [
        col
        for col in candidates
        if "_class_" not in col or not col.rsplit("_class_", 1)[1].isdigit()
    ]
    if len(alias_candidates) == 1:
        return alias_candidates[0]
    if len(candidates) == 1:
        return candidates[0]
    class_2_candidates = [col for col in candidates if col.endswith("_class_2")]
    if len(class_2_candidates) == 1:
        return class_2_candidates[0]
    raise ValueError(f"Expected one classification probability ranking column, found {candidates}")


def _score_models(
    *,
    records: list[dict[str, object]],
    metadata: pd.DataFrame,
    feature_sets: dict[str, pd.DataFrame],
    stock_features: pd.DataFrame,
    context_features: pd.DataFrame,
    progress_path: Path | None = None,
    progress_width: int = 30,
) -> pd.DataFrame:
    first_model = records[0]
    sequence_stores: dict[tuple[str, tuple[str, ...]], object] = {}
    prediction_frames: list[pd.DataFrame] = []
    for record_index, record in enumerate(records, start=1):
        run_dir = Path(record["run_dir"])
        task_type = str(record.get("task_type") or "classification")
        if task_type != "classification":
            _log(f"skipping non-classification model {record.get('model_name')}")
            continue
        model_name = str(record["model_name"])
        feature_set = str(record["feature_set"])
        _report_progress(
            progress_path,
            phase="score_models",
            current=record_index,
            total=len(records),
            detail=f"loading/scoring {model_name}",
            width=progress_width,
        )
        _log(f"loading {model_name} from {run_dir}")
        bundle = load_model_bundle(_resolve_artifact_path(run_dir, str(record["artifact_path"])))
        model = bundle["model"]
        bundle_metadata = bundle.get("metadata", {})
        input_layout = str(record.get("input_layout") or bundle_metadata.get("input_layout") or "tabular")
        if input_layout == "sequence_static":
            sequence_feature_columns = list(
                record.get("sequence_feature_columns")
                or bundle_metadata.get("sequence_feature_columns")
                or []
            )
            static_columns = list(
                record.get("static_categorical_columns")
                or bundle_metadata.get("static_categorical_columns")
                or STATIC_CATEGORICAL_COLUMNS
            )
            static_vocabularies = dict(
                record.get("static_vocabularies")
                or bundle_metadata.get("static_vocabularies")
                or {}
            )
            benchmark_ticker = str(
                record.get("benchmark_ticker")
                or bundle_metadata.get("benchmark_ticker")
                or first_model["benchmark_ticker"]
            )
            cache_key = (feature_set, tuple(sequence_feature_columns))
            if cache_key not in sequence_stores:
                _log(f"building sequence store for {feature_set}")
                sequence_stores[cache_key] = build_sequence_feature_store(
                    stock_features,
                    feature_set,
                    context_features=context_features,
                    benchmark_ticker=benchmark_ticker,
                    feature_columns=sequence_feature_columns,
                )
            x = {
                "store": sequence_stores[cache_key],
                "metadata": metadata.reset_index(drop=True),
                "static_categorical": encode_static_categories(
                    metadata.reset_index(drop=True),
                    static_vocabularies,
                    columns=static_columns,
                ),
            }
        else:
            feature_columns = list(record.get("feature_columns") or bundle_metadata.get("feature_columns") or [])
            x = _aligned_features(feature_sets[feature_set], feature_columns)
        _log(f"scoring {model_name} on {len(metadata):,} latest windows")
        pred = model.predict(x)
        leaderboard = _leaderboard_for_run(run_dir, task_type)
        leaderboard_rank = None
        recommended = False
        if not leaderboard.empty:
            match = leaderboard[
                (leaderboard["model_name"] == model_name) & (leaderboard["feature_set"] == feature_set)
            ]
            if not match.empty:
                leaderboard_rank = int(match.iloc[0]["leaderboard_rank"])
                recommended = bool(match.iloc[0]["recommended"])
        prediction_frames.append(
            prediction_frame(
                metadata,
                pred,
                target_columns=list(record["target_columns"]),
                model_name=model_name,
                feature_set=feature_set,
                leaderboard_rank=leaderboard_rank,
                recommended=recommended,
                task_type=task_type,
            )
        )
    if not prediction_frames:
        raise SystemExit("No classification predictions were generated.")
    return pd.concat(prediction_frames, ignore_index=True)


def _rank_bucket(percentile: float) -> str:
    if percentile <= 3:
        return "TOP_3_PERCENT"
    if percentile <= 5:
        return "TOP_5_PERCENT"
    if percentile <= 10:
        return "TOP_DECILE"
    if percentile <= 20:
        return "TOP_20_PERCENT"
    if percentile <= 30:
        return "TOP_30_PERCENT"
    return "BELOW_TOP_30_PERCENT"


def _agreement_status(top5_count: int, top10_count: int, model_count: int, ensemble_percentile: float) -> str:
    if top5_count >= 2 or ensemble_percentile <= 3:
        return f"VERY STRONG ({top10_count}/{model_count} top-decile)"
    if top10_count >= 2:
        return f"STRONG ({top10_count}/{model_count} top-decile)"
    if top10_count == 1 or ensemble_percentile <= 10:
        return f"WEAK ({top10_count}/{model_count} top-decile)"
    return f"NONE ({top10_count}/{model_count} top-decile)"


def _action_reason(action: str, percentile: float, top10_count: int, model_count: int) -> str:
    if action == "ENTRY CANDIDATE":
        return f"Top {percentile:.2f}% with {top10_count}/{model_count} models in top decile"
    if action == "WATCHLIST":
        if top10_count >= 2:
            return f"Top decile with {top10_count}/{model_count} model agreement"
        return "Top decile but weak model agreement"
    return "Below top decile"


def _build_ranked_signals(
    predictions: pd.DataFrame,
    *,
    run_date: str,
    entry_top_percent: float,
    watchlist_top_percent: float,
    min_model_agreement_count: int,
) -> pd.DataFrame:
    prob_col = _prediction_column(predictions)
    scored = predictions.copy()
    scored["prediction_score"] = pd.to_numeric(scored[prob_col], errors="coerce")
    scored["model_key"] = scored["model_name"].astype(str) + "::" + scored["feature_set"].astype(str)
    scored["model_rank"] = scored.groupby("model_key")["prediction_score"].rank(method="first", ascending=False)
    scored["model_count_for_rank"] = scored.groupby("model_key")["prediction_score"].transform("count")
    scored["model_percentile"] = 100.0 * scored["model_rank"] / scored["model_count_for_rank"]
    scored["model_top5"] = scored["model_percentile"] <= 5.0
    scored["model_top10"] = scored["model_percentile"] <= 10.0

    grouped = scored.groupby(["ticker", "anchor_date"], dropna=False)
    ranked = grouped.agg(
        current_price=("anchor_close", "first"),
        ensemble_score=("prediction_score", "mean"),
        model_count=("model_key", "nunique"),
        top_5pct_model_count=("model_top5", "sum"),
        top_decile_model_count=("model_top10", "sum"),
        best_model_percentile=("model_percentile", "min"),
        best_model_score=("prediction_score", "max"),
    ).reset_index()
    ranked = ranked.sort_values(["ensemble_score", "top_decile_model_count", "best_model_score"], ascending=False)
    ranked["ensemble_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    ranked["ensemble_percentile"] = 100.0 * ranked["ensemble_rank"] / max(len(ranked), 1)
    ranked["rank_bucket"] = ranked["ensemble_percentile"].map(_rank_bucket)
    ranked["agreement_status"] = ranked.apply(
        lambda row: _agreement_status(
            int(row["top_5pct_model_count"]),
            int(row["top_decile_model_count"]),
            int(row["model_count"]),
            float(row["ensemble_percentile"]),
        ),
        axis=1,
    )

    def suggested_action(row: pd.Series) -> str:
        percentile = float(row["ensemble_percentile"])
        agreement = int(row["top_decile_model_count"])
        if percentile <= entry_top_percent and agreement >= min_model_agreement_count:
            return "ENTRY CANDIDATE"
        if percentile <= watchlist_top_percent:
            return "WATCHLIST"
        return "IGNORE"

    ranked["suggested_action"] = ranked.apply(suggested_action, axis=1)
    ranked["reason"] = ranked.apply(
        lambda row: _action_reason(
            str(row["suggested_action"]),
            float(row["ensemble_percentile"]),
            int(row["top_decile_model_count"]),
            int(row["model_count"]),
        ),
        axis=1,
    )
    ranked.insert(0, "run_date", run_date)
    ranked = ranked.rename(columns={"anchor_date": "prediction_date"})
    return ranked[
        [
            "run_date",
            "prediction_date",
            "ticker",
            "rank_bucket",
            "agreement_status",
            "suggested_action",
            "reason",
            "current_price",
            "ensemble_score",
            "ensemble_rank",
            "ensemble_percentile",
            "model_count",
            "top_decile_model_count",
            "top_5pct_model_count",
            "best_model_score",
            "best_model_percentile",
        ]
    ]


def _read_position_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    ledger = pd.read_csv(path)
    ledger.columns = [str(col).strip() for col in ledger.columns]
    required = {"ticker", "entry_date", "entry_price", "shares", "current_status"}
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise ValueError(f"Position ledger {path} is missing required columns: {missing}")
    ledger["ticker"] = ledger["ticker"].astype(str).str.upper()
    return ledger


def _business_days_held(entry_date: object, prediction_date: object) -> int | float:
    try:
        start = np.datetime64(str(entry_date)[:10], "D")
        end = np.datetime64(str(prediction_date)[:10], "D")
        return int(np.busday_count(start, end)) + 1
    except Exception:
        return np.nan


def _position_action(row: pd.Series, *, review_threshold: float, strong_review_threshold: float) -> tuple[str, str]:
    if pd.isna(row.get("current_price")):
        return "REVIEW", "No current prediction/price found for ticker"
    current_return = row.get("current_return_pct")
    if pd.isna(current_return):
        return "REVIEW", "Current return could not be calculated"
    if float(current_return) >= float(row["target_profit_pct"]):
        return "SELL - TARGET", "Current return reached target profit"
    if float(current_return) <= float(row["stop_loss_pct"]):
        return "SELL - STOP", "Current return reached stop loss"
    days_held = row.get("days_held")
    if pd.notna(days_held) and int(days_held) >= int(row["max_hold_days"]):
        return "SELL - TIME", "Max holding period reached"
    percentile = float(row["ensemble_percentile"])
    agreement = int(row["top_decile_model_count"])
    if percentile > strong_review_threshold:
        return "STRONG REVIEW", "Latest rank fell below top 30%"
    if percentile > review_threshold:
        return "REVIEW", "Latest rank fell below top 20%"
    if agreement == 0 and float(current_return) <= 0:
        return "CONSIDER EXIT", "Model agreement disappeared and return is flat/negative"
    if agreement == 0 and pd.notna(days_held) and int(days_held) > 10:
        return "CONSIDER EXIT", "Model agreement disappeared after more than 10 held days"
    return "HOLD", "No hard exit or review trigger"


def _optional_numeric_column(frame: pd.DataFrame, column: str, default: float | int) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _build_position_review(
    ledger: pd.DataFrame,
    ranked_signals: pd.DataFrame,
    *,
    run_date: str,
    target_profit_pct: float,
    stop_loss_pct: float,
    max_hold_days: int,
    review_threshold: float,
    strong_review_threshold: float,
) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=POSITION_REVIEW_COLUMNS)
    open_mask = ~ledger["current_status"].fillna("").astype(str).str.upper().isin({"CLOSED", "SOLD"})
    open_positions = ledger.loc[open_mask].copy()
    if open_positions.empty:
        return pd.DataFrame(columns=POSITION_REVIEW_COLUMNS)
    signal_cols = [
        "ticker",
        "prediction_date",
        "current_price",
        "rank_bucket",
        "agreement_status",
        "ensemble_percentile",
        "top_decile_model_count",
    ]
    review = open_positions.merge(ranked_signals[signal_cols], on="ticker", how="left")
    review["run_date"] = run_date
    review["entry_price"] = pd.to_numeric(review["entry_price"], errors="coerce")
    review["shares"] = pd.to_numeric(review["shares"], errors="coerce")
    review["current_price"] = pd.to_numeric(review["current_price"], errors="coerce")
    review["current_return_pct"] = (review["current_price"] / review["entry_price"] - 1.0) * 100.0
    review["days_held"] = review.apply(lambda row: _business_days_held(row["entry_date"], row["prediction_date"]), axis=1)
    review["target_profit_pct"] = _optional_numeric_column(review, "target_profit_pct", target_profit_pct)
    review["stop_loss_pct"] = _optional_numeric_column(review, "stop_loss_pct", stop_loss_pct)
    review["max_hold_days"] = _optional_numeric_column(review, "max_hold_days", max_hold_days)
    review["latest_rank_bucket"] = review["rank_bucket"].fillna("NO_CURRENT_SIGNAL")
    review["agreement_status"] = review["agreement_status"].fillna("NO_CURRENT_SIGNAL")
    actions = review.apply(
        lambda row: _position_action(
            row,
            review_threshold=review_threshold,
            strong_review_threshold=strong_review_threshold,
        ),
        axis=1,
    )
    review["review_action"] = [action for action, _ in actions]
    review["reason"] = [reason for _, reason in actions]
    if "notes" not in review.columns:
        review["notes"] = ""
    return review[list(POSITION_REVIEW_COLUMNS)]


def _write_summary(
    path: Path,
    *,
    run_date: str,
    dataset_root: Path,
    stock_features: pd.DataFrame,
    predictions: pd.DataFrame,
    ranked_signals: pd.DataFrame,
    position_review: pd.DataFrame,
    ledger_path: Path,
    research_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    summary = {
        "run_date": run_date,
        "dataset_root": str(dataset_root),
        "data_min_date": str(stock_features["date"].min()),
        "data_max_date": str(stock_features["date"].max()),
        "recent_feature_rows": int(len(stock_features)),
        "recent_tickers": int(stock_features["ticker"].nunique()),
        "prediction_rows": int(len(predictions)),
        "ranked_signal_rows": int(len(ranked_signals)),
        "entry_candidate_count": int((ranked_signals["suggested_action"] == "ENTRY CANDIDATE").sum()),
        "watchlist_count": int((ranked_signals["suggested_action"] == "WATCHLIST").sum()),
        "position_ledger": str(ledger_path),
        "position_ledger_exists": bool(ledger_path.exists()),
        "open_position_review_rows": int(len(position_review)),
        "hold_count": int((position_review.get("review_action", pd.Series(dtype=str)) == "HOLD").sum()),
        "review_count": int(position_review.get("review_action", pd.Series(dtype=str)).isin(["REVIEW", "STRONG REVIEW"]).sum()),
        "consider_exit_count": int((position_review.get("review_action", pd.Series(dtype=str)) == "CONSIDER EXIT").sum()),
        "sell_signal_count": int(position_review.get("review_action", pd.Series(dtype=str)).astype(str).str.startswith("SELL").sum()),
    }
    if research_summary:
        summary.update({key: value for key, value in research_summary.items() if key != "research_universe_config"})
    (path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(path / "summary.csv", index=False)
    lines = [
        "# Trading Strategy Run Summary",
        "",
        f"- Run date: `{run_date}`",
        f"- Dataset root: `{dataset_root}`",
        f"- Data date range in recent frame: `{summary['data_min_date']}` to `{summary['data_max_date']}`",
        f"- Conservative research universe: `{summary.get('research_universe_passed_rows', summary['ranked_signal_rows'])}` of `{summary.get('research_universe_input_rows', summary['ranked_signal_rows'])}` broad eligible windows passed",
        f"- Ranked signals: `{summary['ranked_signal_rows']}`",
        f"- Entry candidates: `{summary['entry_candidate_count']}`",
        f"- Watchlist rows: `{summary['watchlist_count']}`",
        f"- Open-position review rows: `{summary['open_position_review_rows']}`",
        f"- Ledger found: `{summary['position_ledger_exists']}`",
        "",
        "This report suggests actions only. It does not place trades.",
    ]
    (path / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    run_date = _run_date()
    dataset_root = Path(args.dataset_root)
    run_dirs = [Path(path) for path in (args.run_dir or DEFAULT_RUN_DIRS)]
    run_research_config = None
    for run_dir in run_dirs:
        run_research_config = _research_universe_config_from_run(run_dir)
        if run_research_config is not None:
            break
    research_config = _research_universe_config(args, base_config=run_research_config)
    report_name = args.report_name or _utc_now_label()
    output_dir = Path(args.output_root) / report_name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_progress_path = output_dir / "progress.json"
    _report_progress(
        report_progress_path,
        phase="start",
        current=0,
        total=None,
        detail="starting trading strategy run",
        width=args.progress_bar_width,
    )

    records = _load_model_records(run_dirs)
    benchmark_ticker = str(records[0].get("benchmark_ticker") or "SPY")
    window_length = max(int(record.get("window_length") or 60) for record in records)
    eligibility_config = None
    for run_dir in run_dirs:
        eligibility_config = _episode_eligibility_config_from_run(run_dir)
        if eligibility_config is not None:
            break

    stock_features, context_features, refresh_manifest, latest_dir = _refresh_latest_inference_dataset(
        args,
        run_dirs,
        research_config=research_config,
    )
    if args.anchor_date:
        stock_features = stock_features[stock_features["date"].astype(str) <= args.anchor_date].copy()
        context_features = context_features[context_features["date"].astype(str) <= args.anchor_date].copy()
    if args.anchor_date and not context_features.empty:
        context_features = context_features[context_features["date"].astype(str) <= args.anchor_date].copy()
    if context_features.empty:
        raise SystemExit("Market context features are missing.")

    _log("building needed latest feature sets")
    metadata, feature_sets, _ = _build_latest_feature_sets_for_records(
        stock_features,
        context_features,
        records=records,
        window_length=window_length,
        benchmark_ticker=benchmark_ticker,
        anchor_date=args.anchor_date or None,
        max_anchor_lag_days=args.max_anchor_lag_days,
        eligibility_config=eligibility_config,
        progress_path=report_progress_path,
        progress_width=args.progress_bar_width,
    )
    metadata = _add_anchor_close(metadata, stock_features)
    if metadata.empty:
        raise SystemExit("No eligible latest prediction windows were produced.")

    _log("applying conservative research universe filter")
    research_diagnostics = latest_research_universe_diagnostics(
        stock_features,
        metadata,
        research_config,
        benchmark_ticker=benchmark_ticker,
    )
    research_diagnostics.to_csv(output_dir / "research_universe_diagnostics.csv", index=False)
    research_summary = conservative_research_universe_summary(research_diagnostics, research_config)
    research_mask = research_diagnostics["research_universe_ok"].fillna(False).astype(bool)
    if research_config.enabled:
        kept = int(research_mask.sum())
        _log(
            "conservative research universe kept "
            f"{kept:,}/{len(research_diagnostics):,} broad eligible windows"
        )
    metadata = research_diagnostics.loc[research_mask].reset_index(drop=True)
    for feature_set, frame in list(feature_sets.items()):
        feature_sets[feature_set] = frame.loc[research_mask.to_numpy(dtype=bool)].reset_index(drop=True)
    if metadata.empty:
        raise SystemExit("No latest prediction windows passed the conservative research universe filter.")
    _log(f"eligible latest windows after research filter: {len(metadata):,}")
    scoring_tickers = set(metadata["ticker"].astype(str).str.upper())
    scoring_stock_features = stock_features[stock_features["ticker"].astype(str).str.upper().isin(scoring_tickers)].copy()

    predictions = _score_models(
        records=records,
        metadata=metadata,
        feature_sets=feature_sets,
        stock_features=scoring_stock_features,
        context_features=context_features,
        progress_path=report_progress_path,
        progress_width=args.progress_bar_width,
    )
    predictions.to_csv(output_dir / "all_model_predictions.csv", index=False)
    predictions.to_csv(output_dir / "all_ranked_predictions.csv", index=False)
    for model_name, model_predictions in predictions.groupby("model_name", sort=False):
        safe_name = str(model_name).replace("/", "_").replace("\\", "_")
        model_predictions.to_csv(output_dir / f"predictions_{safe_name}.csv", index=False)

    ranked_signals = _build_ranked_signals(
        predictions,
        run_date=run_date,
        entry_top_percent=args.entry_top_percent,
        watchlist_top_percent=args.watchlist_top_percent,
        min_model_agreement_count=args.min_model_agreement_count,
    )
    ranked_signals.to_csv(output_dir / "ranked_signals.csv", index=False)
    ranked_signals.to_csv(output_dir / "all_ranked_predictions.csv", index=False)
    entry_report = ranked_signals[ranked_signals["suggested_action"] == "ENTRY CANDIDATE"].copy()
    watchlist_report = ranked_signals[ranked_signals["suggested_action"] == "WATCHLIST"].copy()
    entry_report.to_csv(output_dir / "entry_candidates.csv", index=False)
    watchlist_report.to_csv(output_dir / "watchlist.csv", index=False)
    model_agreement = ranked_signals[
        [
            "run_date",
            "prediction_date",
            "ticker",
            "agreement_status",
            "model_count",
            "top_decile_model_count",
            "top_5pct_model_count",
            "ensemble_score",
            "ensemble_rank",
            "ensemble_percentile",
            "suggested_action",
        ]
    ].copy()
    model_agreement.to_csv(output_dir / "model_agreement_summary.csv", index=False)

    ledger_path = Path(args.position_ledger)
    ledger = _read_position_ledger(ledger_path)
    position_review = _build_position_review(
        ledger,
        ranked_signals,
        run_date=run_date,
        target_profit_pct=args.target_profit_pct,
        stop_loss_pct=args.stop_loss_pct,
        max_hold_days=args.max_hold_days,
        review_threshold=args.review_rank_threshold_percent,
        strong_review_threshold=args.strong_review_rank_threshold_percent,
    )
    position_review.to_csv(output_dir / "open_position_review.csv", index=False)
    position_review.to_csv(output_dir / "position_review.csv", index=False)
    if not ledger_path.exists():
        template = pd.DataFrame(
            columns=[
                "ticker",
                "entry_date",
                "entry_price",
                "shares",
                "current_status",
                "notes",
                "target_profit_pct",
                "stop_loss_pct",
                "max_hold_days",
            ]
        )
        template.to_csv(output_dir / "open_positions_template.csv", index=False)

    summary = _write_summary(
        output_dir,
        run_date=run_date,
        dataset_root=dataset_root,
        stock_features=stock_features,
        predictions=predictions,
        ranked_signals=ranked_signals,
        position_review=position_review,
        ledger_path=ledger_path,
        research_summary=research_summary,
    )
    run_manifest = {
        **summary,
        "research_universe_config": research_config.to_dict(),
        "latest_inference_dir": str(latest_dir.resolve()),
        "latest_inference_manifest": refresh_manifest,
        "run_dirs": [str(path) for path in run_dirs],
        "model_count": len(records),
        "anchor_date_arg": args.anchor_date or None,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _report_progress(
        report_progress_path,
        phase="complete",
        current=1,
        total=1,
        detail=f"reports written to {output_dir.resolve()}",
        width=args.progress_bar_width,
    )
    _log(f"reports written to {output_dir.resolve()}")
    _log(
        "summary: "
        f"{summary['entry_candidate_count']} entry candidates, "
        f"{summary['watchlist_count']} watchlist rows, "
        f"{summary['open_position_review_rows']} open positions reviewed"
    )


if __name__ == "__main__":
    main()
