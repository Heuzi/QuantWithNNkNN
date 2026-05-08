from __future__ import annotations

import argparse
import csv
import json
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
    _resolve_artifact_path,
    _resolve_model_index_path,
)
from src.data.episode_eligibility import (  # noqa: E402
    add_episode_eligibility_columns,
    eligibility_metadata_columns,
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
        default=140,
        help="Raw bars retained per ticker before computing latest inference features.",
    )
    parser.add_argument("--max-tickers", type=int, default=0, help="Optional smoke cap for stock tickers.")
    parser.add_argument("--skip-fundamentals", action="store_true", help="Do not join saved fundamentals.")
    parser.add_argument("--skip-sentiment", action="store_true", help="Do not join saved sentiment.")
    parser.add_argument(
        "--latest-inference-dir",
        default="",
        help="Optional latest-inference output/cache directory. Defaults under dataset_root/processed.",
    )
    parser.add_argument("--progress-every-rows", type=int, default=2_000_000)
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


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} | {message}", flush=True)


def _daily_features_path(dataset_root: Path) -> Path:
    normalized = dataset_root / "processed" / "daily_features_normalized.csv"
    processed = dataset_root / "processed" / "daily_features.csv"
    return normalized if normalized.exists() else processed


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


def _stream_recent_bar_rows(
    path: Path,
    *,
    rows_per_ticker: int,
    wanted_tickers: set[str] | None = None,
    progress_every_rows: int = 2_000_000,
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
                _log(
                    f"read {rows_seen:,} raw rows; retained {sum(len(items) for items in tails.values()):,} "
                    f"rows for {len(tails):,} tickers in {time.monotonic() - start_time:.1f}s"
                )
    parsed = [_parse_bar_row(row) for items in tails.values() for row in items]
    parsed = _merge_bar_rows(parsed)
    _log(f"raw cache bootstrap retained {len(parsed):,} rows for {len(tails):,} tickers")
    return parsed


def _load_universe(dataset_root: Path, max_tickers: int = 0) -> tuple[list[dict[str, str]], set[str], set[str]]:
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
            rows.append(row)
            if max_tickers and len(rows) >= max_tickers:
                break
    stock_tickers = {row["ticker"] for row in rows}
    context_tickers = {str(ticker).upper() for ticker in MARKET_CONTEXT_TICKERS}
    return rows, stock_tickers, context_tickers


def _max_date_from_rows(rows: Sequence[dict[str, object]]) -> str | None:
    dates = [str(row.get("date") or "")[:10] for row in rows if row.get("date")]
    return max(dates) if dates else None


def _filter_bulk_rows(
    rows: Sequence[dict[str, object]],
    *,
    exchange: str,
    stock_tickers: set[str],
    context_tickers: set[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    normalized = normalize_eodhd_bulk_eod_rows(rows, exchange=exchange, adjusted=True)
    stock_rows = [row for row in normalized if str(row.get("ticker") or "").upper() in stock_tickers]
    context_rows = [row for row in normalized if str(row.get("ticker") or "").upper() in context_tickers]
    return stock_rows, context_rows


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
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    local_end = _local_data_end_date(dataset_root, latest_dir)
    latest_probe_rows: list[dict[str, object]] = []
    latest_probe_date: str | None = None
    api_calls_estimate = 0

    if skip_fetch:
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
        except EODHDRateLimitError:
            raise
        except EODHDAPIError as exc:
            per_date.append({"date": planned_date_str, "status": "error", "error": str(exc)[:500]})
            _log(f"bulk EOD {planned_date_str} failed: {exc}")

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
    fetched_stock_rows: Sequence[dict[str, object]],
    fetched_context_rows: Sequence[dict[str, object]],
    rows_per_ticker: int,
    progress_every_rows: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_dir = dataset_root / "raw"
    stock_cache_path = latest_dir / RECENT_STOCK_BARS_FILENAME
    context_cache_path = latest_dir / RECENT_CONTEXT_BARS_FILENAME
    stock_update_path = raw_dir / STOCK_UPDATE_FILENAME
    context_update_path = raw_dir / CONTEXT_UPDATE_FILENAME

    _update_daily_update_file(stock_update_path, fetched_stock_rows)
    _update_daily_update_file(context_update_path, fetched_context_rows)

    if stock_cache_path.exists():
        stock_source = _load_bar_csv(stock_cache_path)
    else:
        stock_source = _stream_recent_bar_rows(
            raw_dir / "eodhd_stock_bars.csv",
            rows_per_ticker=rows_per_ticker,
            wanted_tickers=stock_tickers,
            progress_every_rows=progress_every_rows,
        )
        stock_source.extend(_load_bar_csv(stock_update_path))
    stock_source.extend(fetched_stock_rows)
    stock_recent = _tail_bar_rows(stock_source, rows_per_ticker)

    if context_cache_path.exists():
        context_source = _load_bar_csv(context_cache_path)
    else:
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
    return stock_recent, context_recent


def _materialize_latest_inference_features(
    *,
    dataset_root: Path,
    latest_dir: Path,
    stock_bars: Sequence[dict[str, object]],
    context_bars: Sequence[dict[str, object]],
    skip_fundamentals: bool,
    skip_sentiment: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = dataset_root / "raw"
    latest_dir.mkdir(parents=True, exist_ok=True)
    _log(f"computing latest stock features from {len(stock_bars):,} bounded raw rows")
    stock_features = compute_daily_features(stock_bars)
    stock_frame = _add_static_metadata(pd.DataFrame(stock_features), dataset_root)
    stock_features = stock_frame.to_dict("records")

    if not skip_fundamentals:
        symbols = sorted({str(row.get("eodhd_symbol") or "").upper() for row in stock_bars if row.get("eodhd_symbol")})
        _log(f"joining saved fundamentals for {len(symbols):,} symbols")
        fundamental_rows = load_fundamental_feature_rows(raw_dir / "eodhd_fundamentals_raw", symbols=symbols)
        stock_features = add_fundamental_features(stock_features, fundamental_rows)
    if not skip_sentiment:
        wanted = {str(row.get("ticker") or "").upper() for row in stock_bars if row.get("ticker")}
        min_date = min(str(row.get("date"))[:10] for row in stock_bars if row.get("date"))
        _log("joining saved sentiment rows")
        sentiment_rows = [
            row
            for row in load_sentiment_rows(raw_dir / "eodhd_sentiment_daily.csv")
            if str(row.get("ticker") or "").upper() in wanted and str(row.get("date") or "")[:10] >= min_date
        ]
        stock_features = add_sentiment_features(stock_features, sentiment_rows)

    context_features = compute_daily_features(context_bars)
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
    return stock_frame, context_frame


def _refresh_latest_inference_dataset(args: argparse.Namespace, run_dirs: Sequence[Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], Path]:
    dataset_root = Path(args.dataset_root)
    latest_dir = _latest_inference_dir(dataset_root, args.latest_inference_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    universe_rows, stock_tickers, context_tickers = _load_universe(dataset_root, max_tickers=args.max_tickers)
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
    )
    raw_stock, raw_context = _build_recent_raw_cache(
        dataset_root=dataset_root,
        latest_dir=latest_dir,
        stock_tickers=stock_tickers,
        context_tickers=context_tickers,
        fetched_stock_rows=stock_rows,
        fetched_context_rows=context_rows,
        rows_per_ticker=max(int(args.recent_raw_rows_per_ticker), 125),
        progress_every_rows=args.progress_every_rows,
    )
    stock_features, context_features = _materialize_latest_inference_features(
        dataset_root=dataset_root,
        latest_dir=latest_dir,
        stock_bars=raw_stock,
        context_bars=raw_context,
        skip_fundamentals=args.skip_fundamentals,
        skip_sentiment=args.skip_sentiment,
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
        "recent_raw_rows_per_ticker": max(int(args.recent_raw_rows_per_ticker), 125),
        "raw_stock_rows": len(raw_stock),
        "raw_context_rows": len(raw_context),
        "latest_stock_feature_rows": len(stock_features),
        "latest_context_feature_rows": len(context_features),
        "local_data_end_date": local_max,
        "skip_fundamentals": bool(args.skip_fundamentals),
        "skip_sentiment": bool(args.skip_sentiment),
        "fetch": fetch_manifest,
    }
    write_json(latest_dir / "run_manifest.json", manifest)
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

    _log("adding market/sector relative return features")
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
    for feature_set in sorted(requested_tabular):
        parts = set(feature_set.split("_"))
        compact = "compact" in parts
        include_relative = "relative" in parts
        include_market = "market" in parts
        include_sector = "sector" in parts
        include_fundamentals = "fundamentals" in parts
        include_sentiment = "sentiment" in parts
        _log(f"building tabular feature set {feature_set}")
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
    candidates = [col for col in frame.columns if col.startswith("pred_prob_")]
    if len(candidates) != 1:
        raise ValueError(f"Expected one classification probability column, found {candidates}")
    return candidates[0]


def _score_models(
    *,
    records: list[dict[str, object]],
    metadata: pd.DataFrame,
    feature_sets: dict[str, pd.DataFrame],
    stock_features: pd.DataFrame,
    context_features: pd.DataFrame,
) -> pd.DataFrame:
    first_model = records[0]
    sequence_stores: dict[tuple[str, tuple[str, ...]], object] = {}
    prediction_frames: list[pd.DataFrame] = []
    for record in records:
        run_dir = Path(record["run_dir"])
        task_type = str(record.get("task_type") or "classification")
        if task_type != "classification":
            _log(f"skipping non-classification model {record.get('model_name')}")
            continue
        model_name = str(record["model_name"])
        feature_set = str(record["feature_set"])
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
    (path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(path / "summary.csv", index=False)
    lines = [
        "# Trading Strategy Run Summary",
        "",
        f"- Run date: `{run_date}`",
        f"- Dataset root: `{dataset_root}`",
        f"- Data date range in recent frame: `{summary['data_min_date']}` to `{summary['data_max_date']}`",
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
    report_name = args.report_name or _utc_now_label()
    output_dir = Path(args.output_root) / report_name
    output_dir.mkdir(parents=True, exist_ok=True)

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
    )
    metadata = _add_anchor_close(metadata, stock_features)
    if metadata.empty:
        raise SystemExit("No eligible latest prediction windows were produced.")
    _log(f"eligible latest windows: {len(metadata):,}")

    predictions = _score_models(
        records=records,
        metadata=metadata,
        feature_sets=feature_sets,
        stock_features=stock_features,
        context_features=context_features,
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
    )
    run_manifest = {
        **summary,
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
    _log(f"reports written to {output_dir.resolve()}")
    _log(
        "summary: "
        f"{summary['entry_candidate_count']} entry candidates, "
        f"{summary['watchlist_count']} watchlist rows, "
        f"{summary['open_position_review_rows']} open positions reviewed"
    )


if __name__ == "__main__":
    main()
