from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.eodhd_stage1 import (  # noqa: E402
    DEFAULT_BENCHMARK_TICKER,
    DEFAULT_EXCHANGES,
    EODHDAPIError,
    EODHDError,
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
        "--skip-fundamentals",
        action="store_true",
        help="Use symbol-list metadata only. Missing sector/industry becomes Unknown.",
    )
    parser.add_argument("--skip-normalization", action="store_true")
    parser.add_argument("--force-refetch", action="store_true", help="Ignore checkpoint status and refetch symbols.")
    parser.add_argument("--rate-limit-calls", type=int, default=200)
    parser.add_argument("--rate-limit-period-seconds", type=float, default=60.0)
    return parser.parse_args()


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


def _completed_symbols(*paths: Path, status_path: Path | None = None, role: str | None = None) -> set[str]:
    completed: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in load_daily_bars_csv(path):
            symbol = row.get("eodhd_symbol")
            if symbol:
                completed.add(str(symbol).upper())
    if status_path and status_path.exists():
        for row in _load_csv_rows(status_path):
            if role and row.get("role") != role:
                continue
            if row.get("status") in {"ok", "empty"} and row.get("eodhd_symbol"):
                completed.add(str(row["eodhd_symbol"]).upper())
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


def _collect_metadata(
    client: EODHDRESTClient,
    universe: list[dict[str, object]],
    *,
    skip_fundamentals: bool,
) -> list[dict[str, object]]:
    metadata: list[dict[str, object]] = []
    for row in universe:
        fundamentals = None
        if not skip_fundamentals:
            try:
                fundamentals = client.get_fundamentals_general(str(row["eodhd_symbol"]))
            except EODHDError:
                fundamentals = None
        metadata.append(build_metadata_row(row, fundamentals=fundamentals))
    return metadata


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
) -> list[dict[str, object]]:
    existing_final = load_daily_bars_csv(final_path) if final_path.exists() else []
    completed = set() if force_refetch else _completed_symbols(
        final_path,
        checkpoint_path,
        status_path=status_path,
        role=role,
    )

    for item in symbols:
        symbol = str(item["eodhd_symbol"]).upper()
        ticker = str(item.get("ticker") or item.get("symbol") or symbol).upper()
        if symbol in completed:
            continue
        try:
            raw_rows = client.get_eod(symbol, from_date=start_date, to_date=end_date)
            rows = normalize_eodhd_eod_rows(
                raw_rows,
                symbol=symbol,
                ticker=ticker,
                exchange=str(item.get("exchange") or "US"),
                adjusted=True,
            )
            if rows:
                append_csv(checkpoint_path, rows, headers=EODHD_DAILY_BAR_HEADERS)
            _append_status(
                status_path,
                {
                    "role": role,
                    "eodhd_symbol": symbol,
                    "ticker": ticker,
                    "status": "ok" if rows else "empty",
                    "row_count": len(rows),
                    "start_date": start_date,
                    "end_date": end_date,
                    "error": "",
                    "finished_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                },
            )
            print(json.dumps({"step": "fetched_symbol", "role": role, "symbol": symbol, "rows": len(rows)}), flush=True)
        except EODHDAPIError as exc:
            _append_status(
                status_path,
                {
                    "role": role,
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
            print(json.dumps({"step": "fetch_error", "role": role, "symbol": symbol, "error": str(exc)[:200]}), flush=True)

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
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if end_date < start_date:
        raise SystemExit("--end-date must be on or after --start-date")

    dataset_root = Path(args.dataset_root)
    raw_dir = dataset_root / "raw"
    processed_dir = dataset_root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    stock_bars_path = raw_dir / "daily_market_bars.csv"
    context_bars_path = raw_dir / "market_context_bars.csv"
    stock_checkpoint_path = raw_dir / "eodhd_stock_bars_checkpoint.csv"
    context_checkpoint_path = raw_dir / "eodhd_context_bars_checkpoint.csv"
    status_path = raw_dir / "eodhd_fetch_status.csv"
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

    if args.skip_fetch and universe_path.exists():
        universe = _load_csv_rows(universe_path)
    else:
        universe = _collect_universe(client, args)
        _write_csv_with_headers(universe_path, universe, EODHD_UNIVERSE_HEADERS)

    if args.skip_fetch and metadata_path.exists():
        metadata_rows = _load_csv_rows(metadata_path)
    else:
        metadata_rows = _collect_metadata(client, universe, skip_fundamentals=args.skip_fundamentals)
        _write_csv_with_headers(metadata_path, metadata_rows, EODHD_METADATA_HEADERS)

    if args.skip_fetch:
        stock_bars = load_daily_bars_csv(stock_bars_path) if stock_bars_path.exists() else []
        context_bars = load_daily_bars_csv(context_bars_path) if context_bars_path.exists() else []
    else:
        stock_bars = _fetch_bars_to_checkpoint(
            client=client,
            symbols=universe,
            role="stock",
            start_date=args.start_date,
            end_date=args.end_date,
            final_path=stock_bars_path,
            checkpoint_path=stock_checkpoint_path,
            status_path=status_path,
            force_refetch=args.force_refetch,
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
        )

    _write_csv_with_headers(stock_bars_path, stock_bars, EODHD_DAILY_BAR_HEADERS)
    _write_csv_with_headers(context_bars_path, context_bars, EODHD_DAILY_BAR_HEADERS)

    stock_features = compute_daily_features(stock_bars)
    context_features = compute_daily_features(context_bars)
    features_path = processed_dir / "daily_features.csv"
    context_features_path = processed_dir / "market_context_features.csv"
    _write_csv_with_headers(features_path, stock_features, EODHD_DAILY_BAR_HEADERS)
    _write_csv_with_headers(context_features_path, context_features, EODHD_DAILY_BAR_HEADERS)

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
    benchmark_features = [
        row for row in context_features if str(row.get("ticker")).upper() == args.benchmark_ticker.upper()
    ]
    episodes = compute_episode_index(
        feature_rows=[*stock_features, *benchmark_features],
        window_length=args.window_length,
        horizon_days=args.horizon_days,
        benchmark_ticker=args.benchmark_ticker.upper(),
    )
    prediction_windows = compute_latest_prediction_windows(
        rows_for_windows,
        window_length=args.window_length,
        target_horizon_days=args.horizon_days,
        benchmark_ticker=args.benchmark_ticker.upper(),
        anchor_date=args.prediction_anchor_date or None,
    )
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
            },
            "counts": {
                "universe_rows": len(universe),
                "metadata_rows": len(metadata_rows),
                "stock_bar_rows": len(stock_bars),
                "context_bar_rows": len(context_bars),
                "stock_feature_rows": len(stock_features),
                "context_feature_rows": len(context_features),
                "normalized_rows": len(normalized_rows) if normalized_rows is not None else None,
                "episode_rows": len(episodes),
                "prediction_window_rows": len(prediction_windows),
            },
            "feature_policy": {
                "dropped_fields": ["vwap", "transactions", "close_to_vwap_pct"],
                "derived_fields": ["dollar_volume"],
                "price_adjustment": "Internal OHLC close is adjusted using EODHD adjusted_close when available; raw close is retained as raw_close.",
            },
            "known_risks": [
                "EODHD sector/industry metadata is not treated as point-in-time fundamentals.",
                "Ticker identity and symbol reuse are not fully resolved by this daily bar-only pipeline.",
                "Raw volume is not split-adjusted by this adapter; volume-ratio features may need additional corporate-action validation.",
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
                "latest_raw_date": max_date.isoformat() if max_date else None,
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
