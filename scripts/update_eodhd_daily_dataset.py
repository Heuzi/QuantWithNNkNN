from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.episode_eligibility import (  # noqa: E402
    EpisodeEligibilityConfig,
    add_episode_eligibility_columns,
    episode_eligibility_summary,
    parse_allowed_exchanges,
)
from src.data.eodhd_stage1 import (  # noqa: E402
    DEFAULT_BENCHMARK_TICKER,
    DEFAULT_EXCHANGES,
    EODHDAPIError,
    EODHDError,
    EODHDRateLimitError,
    EODHDRESTClient,
    EODHD_DAILY_BAR_HEADERS,
    EODHD_METADATA_HEADERS,
    EODHD_UNIVERSE_HEADERS,
    RateLimiter,
    build_metadata_row,
    eodhd_symbol_for_code,
    load_eodhd_credentials,
    merge_universe_rows,
    normalize_eodhd_eod_rows,
    normalize_exchange_symbol_rows,
)
from src.data.incremental_update import (  # noqa: E402
    compute_latest_prediction_windows,
    max_bar_date,
    merge_daily_bar_rows,
    min_bar_date,
)
from src.data.massive_stage1 import (  # noqa: E402
    append_csv,
    compute_daily_features,
    compute_episode_index,
    load_daily_bars_csv,
    write_csv,
    write_json,
)
from src.data.eodhd_enrichment import (  # noqa: E402
    FUNDAMENTAL_RAW_FILTER,
    add_fundamental_features,
    add_sentiment_features,
    fundamental_payload_path,
    load_fundamental_feature_rows,
    load_sentiment_rows,
    merge_sentiment_rows,
    normalize_sentiment_response,
    read_fundamental_payload,
)
from src.data.normalization import (  # noqa: E402
    build_normalized_manifest,
    compute_normalized_feature_rows,
    load_equity_metadata,
)
from src.data.v1_dataset import MARKET_CONTEXT_TICKERS  # noqa: E402


PREDICTION_WINDOW_HEADERS = [
    "ticker",
    "anchor_date",
    "window_start_date",
    "window_end_date",
    "window_length",
    "available_window_rows",
    "target_horizon_days",
    "target_status",
    "target_return",
    "market_adjusted_target_return",
    "benchmark_ticker",
    "anchor_selection",
    "inference_ready",
]

EPISODE_INDEX_HEADERS = [
    "ticker",
    "anchor_date",
    "window_start_date",
    "window_end_date",
    "target_horizon_days",
    "target_return",
    "market_adjusted_target_return",
    "benchmark_ticker",
    "benchmark_anchor_date",
    "benchmark_future_date",
    "available_window_rows",
]

STATUS_HEADERS = [
    "role",
    "eodhd_symbol",
    "ticker",
    "status",
    "row_count",
    "start_date",
    "end_date",
    "error",
    "finished_utc",
]

SENTIMENT_HEADERS = [
    "ticker",
    "eodhd_symbol",
    "date",
    "sentiment_count_raw",
    "sentiment_normalized_raw",
]

PILOT_PREFERRED_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "JPM",
    "XOM",
    "UNH",
    "WMT",
]


def parse_args() -> argparse.Namespace:
    today = date.today()
    default_end = today - timedelta(days=1)
    parser = argparse.ArgumentParser(
        description="Build or refresh the EODHD-backed 30-year U.S. equities dataset."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/eodhd_us_equities_30y",
        help="Dataset folder containing raw/ and processed/ subfolders.",
    )
    parser.add_argument("--credentials-path", default="EODHD_api_key", help="Path to local EODHD API key file.")
    parser.add_argument("--start-date", default="1995-01-01", help="Inclusive YYYY-MM-DD.")
    parser.add_argument("--end-date", default=default_end.isoformat(), help="Inclusive YYYY-MM-DD.")
    parser.add_argument(
        "--tickers",
        default="",
        help="Optional comma-separated common-stock tickers for smoke/pilot runs. Defaults to EODHD exchange lists.",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=0,
        help="Limit stock universe after filtering. Use for smoke/pilot runs. 0 means no limit.",
    )
    parser.add_argument(
        "--exchanges",
        default=",".join(DEFAULT_EXCHANGES),
        help="Comma-separated listed U.S. exchanges to query for common-stock universe.",
    )
    parser.add_argument(
        "--include-otc",
        action="store_true",
        help="Include OTC/PINK rows if they appear in symbol lists. Default excludes them.",
    )
    parser.add_argument("--benchmark-ticker", default=DEFAULT_BENCHMARK_TICKER)
    parser.add_argument("--window-length", type=int, default=60)
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--prediction-anchor-date", default="")
    parser.add_argument("--skip-fetch", action="store_true", help="Rebuild processed artifacts from existing raw bars.")
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Fetch raw EODHD universe/fundamentals/bars/sentiment and stop before in-memory feature rebuild.",
    )
    parser.add_argument(
        "--skip-fundamentals",
        action="store_true",
        help="Use symbol-list metadata only. Missing sector/industry becomes Unknown.",
    )
    parser.add_argument(
        "--fundamentals-filter",
        default=FUNDAMENTAL_RAW_FILTER,
        help="EODHD v1.1 fundamentals filter to store as raw JSON.",
    )
    parser.add_argument("--skip-sentiment", action="store_true", help="Skip EODHD sentiment fetch and feature join.")
    parser.add_argument("--sentiment-batch-size", type=int, default=25)
    parser.add_argument("--skip-normalization", action="store_true")
    parser.add_argument("--force-refetch", action="store_true", help="Ignore checkpoint status and refetch symbols.")
    parser.add_argument("--rate-limit-calls", type=int, default=200)
    parser.add_argument("--rate-limit-period-seconds", type=float, default=60.0)
    parser.add_argument(
        "--fetch-workers",
        type=int,
        default=1,
        help="Concurrent EODHD fetch workers. The shared rate limiter still caps total request rate.",
    )
    parser.add_argument(
        "--disable-episode-eligibility-filter",
        action="store_true",
        help="Disable as-of common-stock/history/liquidity/price/exchange filtering for episode and prediction windows.",
    )
    parser.add_argument("--eligibility-min-history-days", type=int, default=0)
    parser.add_argument("--eligibility-valid-ohlcv-lookback", type=int, default=60)
    parser.add_argument("--eligibility-min-valid-ohlcv-days", type=int, default=55)
    parser.add_argument("--eligibility-dollar-volume-lookback", type=int, default=60)
    parser.add_argument("--eligibility-min-avg-dollar-volume", type=float, default=100_000.0)
    parser.add_argument("--eligibility-min-price", type=float, default=1.0)
    parser.add_argument(
        "--eligibility-allowed-exchanges",
        default="NYSE,NASDAQ,AMEX,BATS",
        help="Comma-separated exchange allowlist. AMEX also matches EODHD NYSE MKT / NYSE American.",
    )
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


def _write_csv_with_headers(path: Path, rows: list[dict[str, object]], headers: Sequence[str]) -> None:
    if rows:
        write_csv(path, rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()


def _load_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _append_status(path: Path, row: dict[str, object]) -> None:
    append_csv(path, [row], headers=STATUS_HEADERS)


def _csv_data_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        next(handle, None)
        return sum(1 for _ in handle)


def _status_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for row in _load_csv_rows(path):
        key = f"{row.get('role') or 'unknown'}::{row.get('status') or 'unknown'}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _write_raw_fetch_manifest(
    *,
    dataset_root: Path,
    raw_dir: Path,
    universe_path: Path,
    metadata_path: Path,
    fundamentals_raw_dir: Path,
    stock_bars_path: Path,
    context_bars_path: Path,
    sentiment_path: Path,
    status_path: Path,
    start_date: str,
    end_date: str,
    universe_rows: int,
    fetch_workers: int,
    mode: str,
    error: str | None = None,
) -> dict[str, object]:
    manifest = {
        "vendor": "eodhd",
        "mode": mode,
        "dataset_root": str(dataset_root.resolve()),
        "start_date": start_date,
        "end_date": end_date,
        "universe_rows": universe_rows,
        "fetch_workers": fetch_workers,
        "status_counts": _status_counts(status_path),
        "stock_bar_rows": _csv_data_row_count(stock_bars_path),
        "context_bar_rows": _csv_data_row_count(context_bars_path),
        "sentiment_rows": _csv_data_row_count(sentiment_path),
        "raw_files": {
            "universe": str(universe_path),
            "metadata": str(metadata_path),
            "fundamentals_raw_dir": str(fundamentals_raw_dir),
            "stock_bars": str(stock_bars_path),
            "context_bars": str(context_bars_path),
            "sentiment_daily": str(sentiment_path),
            "status": str(status_path),
        },
    }
    if error:
        manifest["error"] = error
    write_json(raw_dir / "eodhd_fetch_manifest.json", manifest)
    return manifest


def _run_fetch_jobs(
    pending: Sequence[dict[str, object]],
    *,
    fetch_workers: int,
    fetch_one,
    record_result,
) -> None:
    if fetch_workers <= 1:
        for item in pending:
            record_result(fetch_one(item))
        return

    iterator = iter(pending)
    executor = ThreadPoolExecutor(max_workers=max(1, fetch_workers))
    futures = {}
    try:
        for _ in range(max(1, fetch_workers)):
            try:
                item = next(iterator)
            except StopIteration:
                break
            futures[executor.submit(fetch_one, item)] = item

        while futures:
            future = next(as_completed(futures))
            futures.pop(future)
            record_result(future.result())
            try:
                item = next(iterator)
            except StopIteration:
                continue
            futures[executor.submit(fetch_one, item)] = item
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def _completed_symbols(*paths: Path, status_path: Path | None = None, role: str | None = None) -> set[str]:
    completed: set[str] = set()
    if status_path and status_path.exists():
        for row in _load_csv_rows(status_path):
            if role and row.get("role") != role:
                continue
            if row.get("status") in {"ok", "empty"} and row.get("eodhd_symbol"):
                completed.add(str(row["eodhd_symbol"]).upper())
        if completed:
            return completed
    for path in paths:
        if not path.exists():
            continue
        for row in load_daily_bars_csv(path):
            symbol = row.get("eodhd_symbol")
            if symbol:
                completed.add(str(symbol).upper())
    return completed


def _explicit_universe(tickers: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in [ticker.strip().upper() for ticker in tickers.split(",") if ticker.strip()]:
        symbol = eodhd_symbol_for_code(item)
        rows.append(
            {
                "symbol": item,
                "ticker": item,
                "eodhd_symbol": symbol,
                "name": None,
                "country": "USA",
                "exchange": "US",
                "currency": "USD",
                "type": "Common Stock",
                "isin": None,
                "is_delisted": None,
            }
        )
    return rows


def _limit_universe_for_pilot(universe: list[dict[str, object]], max_tickers: int) -> list[dict[str, object]]:
    if max_tickers <= 0 or len(universe) <= max_tickers:
        return universe
    rows_by_ticker = {str(row.get("ticker") or row.get("symbol") or "").upper(): row for row in universe}
    selected: list[dict[str, object]] = []
    selected_tickers: set[str] = set()
    for ticker in PILOT_PREFERRED_TICKERS:
        row = rows_by_ticker.get(ticker)
        if row is not None:
            selected.append(row)
            selected_tickers.add(ticker)
        if len(selected) >= max_tickers:
            return selected

    def add_rows(rows: list[dict[str, object]]) -> None:
        for row in rows:
            ticker = str(row.get("ticker") or row.get("symbol") or "").upper()
            if ticker in selected_tickers:
                continue
            selected.append(row)
            selected_tickers.add(ticker)
            if len(selected) >= max_tickers:
                return

    current_rows = [row for row in universe if row.get("is_delisted") is not True]
    delisted_rows = [row for row in universe if row.get("is_delisted") is True]
    add_rows(current_rows)
    if len(selected) < max_tickers:
        add_rows(delisted_rows)
    return selected


def _collect_universe(client: EODHDRESTClient, args: argparse.Namespace) -> list[dict[str, object]]:
    if args.tickers.strip():
        universe = _explicit_universe(args.tickers)
        if args.max_tickers and args.max_tickers > 0:
            universe = universe[: args.max_tickers]
    else:
        rows_by_exchange: list[list[dict[str, object]]] = []
        for exchange in [item.strip().upper() for item in args.exchanges.split(",") if item.strip()]:
            # EODHD's delisted=1 symbol-list view is delisted-only, so collect both views.
            raw_rows = client.get_exchange_symbol_list(exchange, symbol_type="common_stock", include_delisted=False)
            rows_by_exchange.append(
                normalize_exchange_symbol_rows(
                    raw_rows,
                    exchange=exchange,
                    include_otc=args.include_otc,
                    is_delisted=False,
                )
            )
            delisted_rows = client.get_exchange_symbol_list(exchange, symbol_type="common_stock", include_delisted=True)
            rows_by_exchange.append(
                normalize_exchange_symbol_rows(
                    delisted_rows,
                    exchange=exchange,
                    include_otc=args.include_otc,
                    is_delisted=True,
                )
            )
        universe = merge_universe_rows(rows_by_exchange)
        if args.max_tickers and args.max_tickers > 0:
            universe = _limit_universe_for_pilot(universe, args.max_tickers)
    return universe


def _metadata_by_ticker_from_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or row.get("symbol") or "").upper()
        if not ticker:
            continue
        metadata[ticker] = {
            "gics_sector": str(row.get("gics_sector") or "Unknown"),
            "gics_sub_industry": str(row.get("gics_sub_industry") or "Unknown"),
        }
    return metadata


def _merge_universe_with_metadata(
    universe: list[dict[str, object]],
    metadata_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    metadata_by_symbol = {str(row.get("eodhd_symbol") or "").upper(): row for row in metadata_rows}
    merged: list[dict[str, object]] = []
    for row in universe:
        out = dict(row)
        metadata = metadata_by_symbol.get(str(row.get("eodhd_symbol") or "").upper(), {})
        for key in ("exchange", "currency", "type", "isin", "is_delisted"):
            if metadata.get(key) not in (None, ""):
                out[key] = metadata[key]
        merged.append(out)
    return merged


def _fetch_fundamentals_to_raw_dir(
    *,
    client: EODHDRESTClient,
    universe: list[dict[str, object]],
    raw_dir: Path,
    status_path: Path,
    filters: str,
    force_refetch: bool,
    fetch_workers: int,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    completed = set() if force_refetch else _completed_symbols(status_path=status_path, role="fundamentals")
    pending: list[dict[str, object]] = []
    for row in universe:
        symbol = str(row["eodhd_symbol"]).upper()
        path = fundamental_payload_path(raw_dir, symbol)
        if symbol in completed and path.exists():
            continue
        pending.append(row)

    def fetch_one(row: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
        symbol = str(row["eodhd_symbol"]).upper()
        ticker = str(row.get("ticker") or row.get("symbol") or symbol).upper()
        path = fundamental_payload_path(raw_dir, symbol)
        status = {
            "role": "fundamentals",
            "eodhd_symbol": symbol,
            "ticker": ticker,
            "status": "ok",
            "row_count": 1,
            "start_date": "",
            "end_date": "",
            "error": "",
            "finished_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        try:
            payload = client.get_fundamentals(symbol, filters=filters or None)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            return status, {"step": "fetched_fundamentals", "symbol": symbol}
        except EODHDRateLimitError:
            raise
        except EODHDAPIError as exc:
            status.update({"status": "error", "row_count": 0, "error": str(exc)[:500]})
            return status, {"step": "fundamentals_error", "symbol": symbol, "error": str(exc)[:200]}

    def record_result(result: tuple[dict[str, object], dict[str, object]]) -> None:
        status, message = result
        _append_status(status_path, status)
        print(json.dumps(message), flush=True)

    _run_fetch_jobs(pending, fetch_workers=fetch_workers, fetch_one=fetch_one, record_result=record_result)


def _collect_metadata(
    client: EODHDRESTClient,
    universe: list[dict[str, object]],
    *,
    skip_fundamentals: bool,
    fundamentals_raw_dir: Path | None = None,
) -> list[dict[str, object]]:
    metadata: list[dict[str, object]] = []
    for row in universe:
        fundamentals = None
        if not skip_fundamentals and fundamentals_raw_dir is not None:
            fundamentals = read_fundamental_payload(fundamental_payload_path(fundamentals_raw_dir, str(row["eodhd_symbol"])))
        if not skip_fundamentals and fundamentals is None and fundamentals_raw_dir is None:
            try:
                fundamentals = client.get_fundamentals_general(str(row["eodhd_symbol"]))
            except EODHDError:
                fundamentals = None
        metadata.append(build_metadata_row(row, fundamentals=fundamentals))
    return metadata


def _completed_sentiment_symbols(status_path: Path) -> set[str]:
    return _completed_symbols(status_path=status_path, role="sentiment")


def _fetch_sentiment_to_checkpoint(
    *,
    client: EODHDRESTClient,
    symbols: list[dict[str, object]],
    start_date: str,
    end_date: str,
    final_path: Path,
    checkpoint_path: Path,
    status_path: Path,
    batch_size: int,
    force_refetch: bool,
    return_rows: bool = True,
) -> list[dict[str, object]]:
    existing_final = load_sentiment_rows(final_path) if return_rows and final_path.exists() else []
    checkpoint_rows = [] if force_refetch or not return_rows else load_sentiment_rows(checkpoint_path)
    completed = set() if force_refetch else _completed_sentiment_symbols(status_path)
    pending = [row for row in symbols if str(row["eodhd_symbol"]).upper() not in completed]
    batch_size = max(1, batch_size)

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        requested = [str(row["eodhd_symbol"]).upper() for row in batch]
        try:
            payload = client.get_sentiments(requested, from_date=start_date, to_date=end_date)
            rows = normalize_sentiment_response(payload)
            if rows:
                append_csv(checkpoint_path, rows, headers=SENTIMENT_HEADERS)
                if return_rows:
                    checkpoint_rows = merge_sentiment_rows(checkpoint_rows, rows)
            row_counts = {}
            for item in rows:
                symbol = str(item.get("eodhd_symbol") or "").upper()
                row_counts[symbol] = row_counts.get(symbol, 0) + 1
            for row in batch:
                symbol = str(row["eodhd_symbol"]).upper()
                ticker = str(row.get("ticker") or row.get("symbol") or symbol).upper()
                count = int(row_counts.get(symbol, 0))
                _append_status(
                    status_path,
                    {
                        "role": "sentiment",
                        "eodhd_symbol": symbol,
                        "ticker": ticker,
                        "status": "ok" if count else "empty",
                        "row_count": count,
                        "start_date": start_date,
                        "end_date": end_date,
                        "error": "",
                        "finished_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    },
                )
            print(json.dumps({"step": "fetched_sentiment_batch", "symbols": requested, "rows": len(rows)}), flush=True)
        except EODHDRateLimitError:
            raise
        except EODHDAPIError as exc:
            for row in batch:
                symbol = str(row["eodhd_symbol"]).upper()
                ticker = str(row.get("ticker") or row.get("symbol") or symbol).upper()
                _append_status(
                    status_path,
                    {
                        "role": "sentiment",
                        "eodhd_symbol": symbol,
                        "ticker": ticker,
                        "status": "error",
                        "row_count": 0,
                        "start_date": start_date,
                        "end_date": end_date,
                        "error": str(exc)[:500],
                        "finished_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    },
                )
            print(json.dumps({"step": "sentiment_batch_error", "symbols": requested, "error": str(exc)[:200]}), flush=True)

    if not return_rows:
        return []
    return merge_sentiment_rows(existing_final, checkpoint_rows)


def _fetch_bars_to_checkpoint(
    *,
    client: EODHDRESTClient,
    symbols: list[dict[str, object]],
    role: str,
    start_date: str,
    end_date: str,
    final_path: Path,
    checkpoint_path: Path,
    status_path: Path,
    force_refetch: bool,
    fetch_workers: int,
    return_rows: bool = True,
) -> list[dict[str, object]]:
    existing_final = load_daily_bars_csv(final_path) if return_rows and final_path.exists() else []
    completed = set() if force_refetch else _completed_symbols(
        final_path,
        checkpoint_path,
        status_path=status_path,
        role=role,
    )
    pending = [item for item in symbols if str(item["eodhd_symbol"]).upper() not in completed]

    def fetch_one(item: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
        symbol = str(item["eodhd_symbol"]).upper()
        ticker = str(item.get("ticker") or item.get("symbol") or symbol).upper()
        status = {
            "role": role,
            "eodhd_symbol": symbol,
            "ticker": ticker,
            "status": "empty",
            "row_count": 0,
            "start_date": start_date,
            "end_date": end_date,
            "error": "",
            "finished_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        try:
            raw_rows = client.get_eod(symbol, from_date=start_date, to_date=end_date)
            rows = normalize_eodhd_eod_rows(
                raw_rows,
                symbol=symbol,
                ticker=ticker,
                exchange=str(item.get("exchange") or "US"),
                adjusted=True,
            )
            status.update({"status": "ok" if rows else "empty", "row_count": len(rows)})
            return rows, status, {"step": "fetched_symbol", "role": role, "symbol": symbol, "rows": len(rows)}
        except EODHDRateLimitError:
            raise
        except EODHDAPIError as exc:
            status.update({"status": "error", "error": str(exc)[:500]})
            return [], status, {"step": "fetch_error", "role": role, "symbol": symbol, "error": str(exc)[:200]}

    def record_result(result: tuple[list[dict[str, object]], dict[str, object], dict[str, object]]) -> None:
        rows, status, message = result
        if rows:
            append_csv(checkpoint_path, rows, headers=EODHD_DAILY_BAR_HEADERS)
        _append_status(status_path, status)
        print(json.dumps(message), flush=True)

    _run_fetch_jobs(pending, fetch_workers=fetch_workers, fetch_one=fetch_one, record_result=record_result)

    if not return_rows:
        return []
    checkpoint_rows = load_daily_bars_csv(checkpoint_path) if checkpoint_path.exists() else []
    return merge_daily_bar_rows(existing_final, checkpoint_rows)


def _context_symbols(benchmark_ticker: str) -> list[dict[str, object]]:
    symbols = []
    for ticker in sorted({benchmark_ticker.upper(), *MARKET_CONTEXT_TICKERS}):
        symbols.append(
            {
                "symbol": ticker,
                "ticker": ticker,
                "eodhd_symbol": eodhd_symbol_for_code(ticker),
                "exchange": "US",
            }
        )
    return symbols


def main() -> None:
    args = parse_args()
    if args.fetch_only and args.skip_fetch:
        raise SystemExit("--fetch-only cannot be combined with --skip-fetch.")
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if end_date < start_date:
        raise SystemExit("--end-date must be on or after --start-date")

    dataset_root = Path(args.dataset_root)
    raw_dir = dataset_root / "raw"
    processed_dir = dataset_root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    stock_bars_path = raw_dir / "eodhd_stock_bars.csv"
    legacy_stock_bars_path = raw_dir / "daily_market_bars.csv"
    context_bars_path = raw_dir / "market_context_bars.csv"
    stock_checkpoint_path = raw_dir / "eodhd_stock_bars_checkpoint.csv"
    context_checkpoint_path = raw_dir / "eodhd_context_bars_checkpoint.csv"
    status_path = raw_dir / "eodhd_fetch_status.csv"
    fundamentals_raw_dir = raw_dir / "eodhd_fundamentals_raw"
    sentiment_path = raw_dir / "eodhd_sentiment_daily.csv"
    sentiment_checkpoint_path = raw_dir / "eodhd_sentiment_daily_checkpoint.csv"
    universe_path = raw_dir / "eodhd_common_stock_universe.csv"
    metadata_path = raw_dir / "eodhd_equity_metadata.csv"

    credentials = load_eodhd_credentials(args.credentials_path)
    client = EODHDRESTClient(
        credentials=credentials,
        rate_limiter=RateLimiter(
            max_calls=args.rate_limit_calls,
            period_seconds=args.rate_limit_period_seconds,
        ),
    )

    if (args.skip_fetch or args.fetch_only) and universe_path.exists():
        universe = _load_csv_rows(universe_path)
    else:
        universe = _collect_universe(client, args)
        _write_csv_with_headers(universe_path, universe, EODHD_UNIVERSE_HEADERS)

    if not args.skip_fetch and not args.skip_fundamentals:
        try:
            _fetch_fundamentals_to_raw_dir(
                client=client,
                universe=universe,
                raw_dir=fundamentals_raw_dir,
                status_path=status_path,
                filters=args.fundamentals_filter,
                force_refetch=args.force_refetch,
                fetch_workers=args.fetch_workers,
            )
        except EODHDRateLimitError as exc:
            manifest = _write_raw_fetch_manifest(
                dataset_root=dataset_root,
                raw_dir=raw_dir,
                universe_path=universe_path,
                metadata_path=metadata_path,
                fundamentals_raw_dir=fundamentals_raw_dir,
                stock_bars_path=stock_bars_path,
                context_bars_path=context_bars_path,
                sentiment_path=sentiment_path,
                status_path=status_path,
                start_date=args.start_date,
                end_date=args.end_date,
                universe_rows=len(universe),
                fetch_workers=args.fetch_workers,
                mode="fetch_interrupted_rate_limit",
                error=str(exc)[:500],
            )
            print(json.dumps({"step": "fetch_interrupted_rate_limit", **manifest}, indent=2), flush=True)
            return

    if (args.skip_fetch or args.fetch_only) and metadata_path.exists():
        metadata_rows = _load_csv_rows(metadata_path)
    else:
        metadata_rows = _collect_metadata(
            client,
            universe,
            skip_fundamentals=args.skip_fundamentals,
            fundamentals_raw_dir=fundamentals_raw_dir,
        )
        _write_csv_with_headers(metadata_path, metadata_rows, EODHD_METADATA_HEADERS)
    universe_for_fetch = _merge_universe_with_metadata(universe, metadata_rows)

    if args.fetch_only:
        try:
            _fetch_bars_to_checkpoint(
                client=client,
                symbols=universe_for_fetch,
                role="stock",
                start_date=args.start_date,
                end_date=args.end_date,
                final_path=stock_bars_path,
                checkpoint_path=stock_bars_path,
                status_path=status_path,
                force_refetch=args.force_refetch,
                fetch_workers=args.fetch_workers,
                return_rows=False,
            )
            _fetch_bars_to_checkpoint(
                client=client,
                symbols=_context_symbols(args.benchmark_ticker),
                role="context",
                start_date=args.start_date,
                end_date=args.end_date,
                final_path=context_bars_path,
                checkpoint_path=context_bars_path,
                status_path=status_path,
                force_refetch=args.force_refetch,
                fetch_workers=args.fetch_workers,
                return_rows=False,
            )
            if not args.skip_sentiment:
                _fetch_sentiment_to_checkpoint(
                    client=client,
                    symbols=universe_for_fetch,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    final_path=sentiment_path,
                    checkpoint_path=sentiment_path,
                    status_path=status_path,
                    batch_size=args.sentiment_batch_size,
                    force_refetch=args.force_refetch,
                    return_rows=False,
                )
            mode = "fetch_only"
            error = None
        except EODHDRateLimitError as exc:
            mode = "fetch_interrupted_rate_limit"
            error = str(exc)[:500]
        raw_manifest = _write_raw_fetch_manifest(
            dataset_root=dataset_root,
            raw_dir=raw_dir,
            universe_path=universe_path,
            metadata_path=metadata_path,
            fundamentals_raw_dir=fundamentals_raw_dir,
            stock_bars_path=stock_bars_path,
            context_bars_path=context_bars_path,
            sentiment_path=sentiment_path,
            status_path=status_path,
            start_date=args.start_date,
            end_date=args.end_date,
            universe_rows=len(universe),
            fetch_workers=args.fetch_workers,
            mode=mode,
            error=error,
        )
        step = "fetch_only_complete" if error is None else "fetch_interrupted_rate_limit"
        print(json.dumps({"step": step, **raw_manifest}, indent=2), flush=True)
        return

    if args.skip_fetch:
        stock_source_path = stock_bars_path if stock_bars_path.exists() else legacy_stock_bars_path
        stock_bars = load_daily_bars_csv(stock_source_path) if stock_source_path.exists() else []
        context_bars = load_daily_bars_csv(context_bars_path) if context_bars_path.exists() else []
        sentiment_rows = load_sentiment_rows(sentiment_path)
    else:
        stock_bars = _fetch_bars_to_checkpoint(
            client=client,
            symbols=universe_for_fetch,
            role="stock",
            start_date=args.start_date,
            end_date=args.end_date,
            final_path=stock_bars_path,
            checkpoint_path=stock_checkpoint_path,
            status_path=status_path,
            force_refetch=args.force_refetch,
            fetch_workers=args.fetch_workers,
        )
        context_bars = _fetch_bars_to_checkpoint(
            client=client,
            symbols=_context_symbols(args.benchmark_ticker),
            role="context",
            start_date=args.start_date,
            end_date=args.end_date,
            final_path=context_bars_path,
            checkpoint_path=context_checkpoint_path,
            status_path=status_path,
            force_refetch=args.force_refetch,
            fetch_workers=args.fetch_workers,
        )
        if args.skip_sentiment:
            sentiment_rows = []
        else:
            sentiment_rows = _fetch_sentiment_to_checkpoint(
                client=client,
                symbols=universe_for_fetch,
                start_date=args.start_date,
                end_date=args.end_date,
                final_path=sentiment_path,
                checkpoint_path=sentiment_checkpoint_path,
                status_path=status_path,
                batch_size=args.sentiment_batch_size,
                force_refetch=args.force_refetch,
            )

    _write_csv_with_headers(stock_bars_path, stock_bars, EODHD_DAILY_BAR_HEADERS)
    _write_csv_with_headers(legacy_stock_bars_path, stock_bars, EODHD_DAILY_BAR_HEADERS)
    _write_csv_with_headers(context_bars_path, context_bars, EODHD_DAILY_BAR_HEADERS)
    _write_csv_with_headers(sentiment_path, sentiment_rows, SENTIMENT_HEADERS)

    stock_features = compute_daily_features(stock_bars)
    universe_symbols = [str(row.get("eodhd_symbol") or "").upper() for row in universe if row.get("eodhd_symbol")]
    fundamental_rows = (
        []
        if args.skip_fundamentals
        else load_fundamental_feature_rows(fundamentals_raw_dir, symbols=universe_symbols)
    )
    if fundamental_rows:
        _write_csv_with_headers(processed_dir / "fundamental_features_asof.csv", fundamental_rows, [])
    if not args.skip_fundamentals:
        stock_features = add_fundamental_features(stock_features, fundamental_rows)
    if not args.skip_sentiment:
        stock_features = add_sentiment_features(stock_features, sentiment_rows)
    context_features = compute_daily_features(context_bars)
    features_path = processed_dir / "daily_features.csv"
    context_features_path = processed_dir / "market_context_features.csv"
    _write_csv_with_headers(features_path, stock_features, [])
    _write_csv_with_headers(context_features_path, context_features, [])

    normalized_rows: list[dict[str, object]] | None = None
    if not args.skip_normalization:
        sector_metadata = load_equity_metadata(metadata_path) if metadata_path.exists() else _metadata_by_ticker_from_rows(metadata_rows)
        normalized_rows = compute_normalized_feature_rows(stock_features, sector_metadata)
        normalized_path = processed_dir / "daily_features_normalized.csv"
        _write_csv_with_headers(normalized_path, normalized_rows, [])
        trade_dates = [str(row["date"]) for row in normalized_rows if row.get("date")]
        write_json(
            processed_dir / "daily_features_normalized_manifest.json",
            build_normalized_manifest(
                input_file=features_path,
                sector_source_file=metadata_path,
                row_count=len(normalized_rows),
                universe_count=len({str(row["ticker"]) for row in normalized_rows if row.get("ticker")}),
                min_date=min(trade_dates) if trade_dates else None,
                max_date=max(trade_dates) if trade_dates else None,
            ),
        )

    rows_for_windows = normalized_rows if normalized_rows is not None else stock_features
    eligibility_config = _episode_eligibility_config(args)
    eligibility_frame = pd.DataFrame(rows_for_windows)
    eligible_episode_keys: set[tuple[str, str]] | None = None
    eligibility_counts: dict[str, object] | None = None
    if eligibility_config is not None:
        eligibility_frame = add_episode_eligibility_columns(
            eligibility_frame,
            eligibility_config,
            benchmark_ticker=args.benchmark_ticker.upper(),
        )
        if eligibility_frame.empty:
            eligible_episode_keys = set()
        else:
            eligible_rows = eligibility_frame[eligibility_frame["episode_eligible"]]
            eligible_episode_keys = {
                (str(row["ticker"]).upper(), str(row["date"]))
                for row in eligible_rows[["ticker", "date"]].to_dict("records")
            }
        eligibility_counts = episode_eligibility_summary(
            pd.DataFrame(rows_for_windows),
            eligibility_config,
            benchmark_ticker=args.benchmark_ticker.upper(),
        )
    benchmark_features = [
        row for row in context_features if str(row.get("ticker")).upper() == args.benchmark_ticker.upper()
    ]
    episodes = compute_episode_index(
        feature_rows=[*stock_features, *benchmark_features],
        window_length=args.window_length,
        horizon_days=args.horizon_days,
        benchmark_ticker=args.benchmark_ticker.upper(),
    )
    if eligible_episode_keys is not None:
        episodes = [
            row
            for row in episodes
            if (str(row["ticker"]).upper(), str(row["anchor_date"])) in eligible_episode_keys
        ]
    prediction_windows = compute_latest_prediction_windows(
        rows_for_windows,
        window_length=args.window_length,
        target_horizon_days=args.horizon_days,
        benchmark_ticker=args.benchmark_ticker.upper(),
        anchor_date=args.prediction_anchor_date or None,
    )
    if eligible_episode_keys is not None:
        prediction_windows = [
            row
            for row in prediction_windows
            if (str(row["ticker"]).upper(), str(row["anchor_date"])) in eligible_episode_keys
        ]
    _write_csv_with_headers(processed_dir / "episode_index.csv", episodes, EPISODE_INDEX_HEADERS)
    _write_csv_with_headers(processed_dir / "prediction_windows.csv", prediction_windows, PREDICTION_WINDOW_HEADERS)

    min_date = min_bar_date(stock_bars)
    max_date = max_bar_date(stock_bars)
    write_json(
        processed_dir / "eodhd_update_manifest.json",
        {
            "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "vendor": "EODHD",
            "dataset_root": str(dataset_root.resolve()),
            "date_range": {
                "min_date": min_date.isoformat() if min_date else None,
                "max_date": max_date.isoformat() if max_date else None,
            },
            "parameters": {
                "start_date": args.start_date,
                "end_date": args.end_date,
                "window_length": args.window_length,
                "horizon_days": args.horizon_days,
                "benchmark_ticker": args.benchmark_ticker.upper(),
                "exchanges": [item.strip().upper() for item in args.exchanges.split(",") if item.strip()],
                "include_otc": bool(args.include_otc),
                "skip_fundamentals": bool(args.skip_fundamentals),
                "fundamentals_filter": args.fundamentals_filter if not args.skip_fundamentals else None,
                "skip_sentiment": bool(args.skip_sentiment),
                "sentiment_batch_size": args.sentiment_batch_size,
            },
            "episode_eligibility": (
                eligibility_config.to_dict()
                if eligibility_config is not None
                else {"enabled": False}
            ),
            "episode_eligibility_counts": eligibility_counts,
            "counts": {
                "universe_rows": len(universe),
                "metadata_rows": len(metadata_rows),
                "stock_bar_rows": len(stock_bars),
                "context_bar_rows": len(context_bars),
                "stock_feature_rows": len(stock_features),
                "context_feature_rows": len(context_features),
                "normalized_rows": len(normalized_rows) if normalized_rows is not None else None,
                "fundamental_feature_rows": len(fundamental_rows),
                "sentiment_rows": len(sentiment_rows),
                "episode_rows": len(episodes),
                "prediction_window_rows": len(prediction_windows),
            },
            "feature_policy": {
                "dropped_fields": ["vwap", "transactions", "close_to_vwap_pct"],
                "derived_fields": ["dollar_volume"],
                "price_adjustment": "Internal OHLC close is adjusted using EODHD adjusted_close when available; raw close is retained as raw_close.",
                "fundamentals": "Raw EODHD fundamentals are stored, but only records with explicit availability dates are joined as model features.",
                "sentiment": "Daily EODHD sentiment aggregates are lagged by one trading row before joining to model features.",
            },
            "known_risks": [
                "EODHD sector/industry metadata is not treated as point-in-time fundamentals.",
                "Ticker identity and symbol reuse are not fully resolved by this daily bar-only pipeline.",
                "Raw volume is not split-adjusted by this adapter; volume-ratio features may need additional corporate-action validation.",
                "Fundamental fields without explicit availability dates are stored but not used as historical model features.",
                "Sentiment dates are daily aggregates, so V1 uses a one-trading-day lag to avoid same-day cutoff leakage.",
            ],
        },
    )

    print(
        json.dumps(
            {
                "step": "complete",
                "dataset_root": str(dataset_root.resolve()),
                "universe_rows": len(universe),
                "stock_bar_rows": len(stock_bars),
                "context_bar_rows": len(context_bars),
                "episode_rows": len(episodes),
                "fundamental_feature_rows": len(fundamental_rows),
                "sentiment_rows": len(sentiment_rows),
                "latest_raw_date": max_date.isoformat() if max_date else None,
                "eligible_ticker_count": (
                    eligibility_counts.get("eligible_ticker_count")
                    if eligibility_counts is not None
                    else None
                ),
                "latest_eligible_ticker_count": (
                    eligibility_counts.get("latest_eligible_ticker_count")
                    if eligibility_counts is not None
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except EODHDError as exc:
        print(f"EODHD update failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
