from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.incremental_update import (  # noqa: E402
    build_incremental_update_manifest,
    compute_latest_prediction_windows,
    determine_incremental_fetch_start,
    load_tickers_from_constituents,
    max_bar_date,
    merge_daily_bar_rows,
    min_bar_date,
    unique_tickers_from_bars,
    write_json as write_incremental_json,
)
from src.data.massive_stage1 import (  # noqa: E402
    DEFAULT_BENCHMARK_TICKER,
    MassiveError,
    MassiveRESTClient,
    compute_daily_features,
    compute_episode_index,
    fetch_bars_for_tickers,
    load_daily_bars_csv,
    load_massive_credentials,
    write_csv,
    write_json,
)
from src.data.normalization import (  # noqa: E402
    build_normalized_manifest,
    compute_normalized_feature_rows,
    load_sp500_constituent_metadata,
)


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


def parse_args() -> argparse.Namespace:
    today = date.today()
    default_end = today - timedelta(days=1)
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally update a Massive daily-bar dataset and rebuild the "
            "training/inference data artifacts from the full merged panel."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="data/massive_sp500_current_constituents_history",
        help="Dataset folder containing raw/ and processed/ subfolders.",
    )
    parser.add_argument(
        "--credentials-path",
        default="MassiveApiKey",
        help="Path to the env-style Massive credential file.",
    )
    parser.add_argument(
        "--start-date",
        default="1995-01-01",
        help=(
            "Fallback inclusive YYYY-MM-DD used when no existing raw bars are present. "
            "This is premium-history ready; free-plan accounts may still receive a shorter range."
        ),
    )
    parser.add_argument("--end-date", default=default_end.isoformat(), help="Inclusive YYYY-MM-DD.")
    parser.add_argument(
        "--tickers",
        default="",
        help="Optional comma-separated ticker list. Defaults to constituent metadata, then existing raw bars.",
    )
    parser.add_argument(
        "--benchmark-ticker",
        default=DEFAULT_BENCHMARK_TICKER,
        help="Benchmark ticker used for market-adjusted targets.",
    )
    parser.add_argument("--window-length", type=int, default=60, help="Sliding window length in trading days.")
    parser.add_argument("--horizon-days", type=int, default=5, help="Future return horizon in trading days.")
    parser.add_argument(
        "--recent-overlap-days",
        type=int,
        default=7,
        help="Number of latest calendar days to refetch and replace during incremental updates.",
    )
    parser.add_argument(
        "--prediction-anchor-date",
        default="",
        help="Optional cutoff date for latest prediction windows. Defaults to each ticker's latest available row.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Rebuild processed artifacts from existing raw bars without calling Massive.",
    )
    parser.add_argument(
        "--unadjusted",
        action="store_true",
        help="Request unadjusted daily bars. By default the updater requests adjusted bars.",
    )
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Skip same-date cross-sectional and sector-relative normalized feature generation.",
    )
    return parser.parse_args()


def _write_csv_with_headers(path: Path, rows: list[dict[str, object]], headers: list[str]) -> None:
    if rows:
        write_csv(path, rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()


def _choose_tickers(
    *,
    explicit_tickers: str,
    metadata_path: Path,
    existing_bars: list[dict[str, object]],
    benchmark_ticker: str,
) -> list[str]:
    if explicit_tickers.strip():
        tickers = [ticker.strip().upper() for ticker in explicit_tickers.split(",") if ticker.strip()]
    elif metadata_path.exists():
        tickers = load_tickers_from_constituents(metadata_path)
    else:
        tickers = unique_tickers_from_bars(existing_bars)

    if benchmark_ticker.upper() not in tickers:
        tickers.append(benchmark_ticker.upper())
    return list(dict.fromkeys(tickers))


def main() -> None:
    args = parse_args()
    fallback_start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if end_date < fallback_start_date:
        raise SystemExit("--end-date must be on or after --start-date")

    dataset_root = Path(args.dataset_root)
    raw_dir = dataset_root / "raw"
    processed_dir = dataset_root / "processed"
    bars_path = raw_dir / "daily_market_bars.csv"
    metadata_path = raw_dir / "sp500_constituents_current.csv"
    features_path = processed_dir / "daily_features.csv"
    normalized_path = processed_dir / "daily_features_normalized.csv"
    normalized_manifest_path = processed_dir / "daily_features_normalized_manifest.json"
    episode_path = processed_dir / "episode_index.csv"
    prediction_windows_path = processed_dir / "prediction_windows.csv"
    update_manifest_path = processed_dir / "incremental_update_manifest.json"

    existing_bars = load_daily_bars_csv(bars_path) if bars_path.exists() else []
    raw_rows_before = len(existing_bars)
    benchmark_ticker = args.benchmark_ticker.upper()
    tickers = _choose_tickers(
        explicit_tickers=args.tickers,
        metadata_path=metadata_path,
        existing_bars=existing_bars,
        benchmark_ticker=benchmark_ticker,
    )

    fetch_start_date: date | None = None
    incoming_bars: list[dict[str, object]] = []
    if not args.skip_fetch:
        fetch_start_date = determine_incremental_fetch_start(
            existing_bars,
            fallback_start_date=fallback_start_date,
            overlap_days=args.recent_overlap_days,
        )
        if fetch_start_date <= end_date:
            credentials = load_massive_credentials(args.credentials_path)
            client = MassiveRESTClient(credentials=credentials)
            print(
                json.dumps(
                    {
                        "step": "fetch_recent_bars",
                        "start_date": fetch_start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "ticker_count": len(tickers),
                        "overlap_days": args.recent_overlap_days,
                    },
                    indent=2,
                )
            )
            incoming_bars = fetch_bars_for_tickers(
                client=client,
                tickers=tickers,
                start_date=fetch_start_date,
                end_date=end_date,
                adjusted=not args.unadjusted,
            )
        else:
            latest_existing = max_bar_date(existing_bars)
            print(
                json.dumps(
                    {
                        "step": "fetch_skipped_no_new_dates",
                        "latest_existing_date": latest_existing.isoformat() if latest_existing else None,
                    }
                )
            )

    merged_bars = merge_daily_bar_rows(existing_bars, incoming_bars)
    if not merged_bars:
        raise SystemExit("No raw bars are available after update; cannot build processed artifacts.")

    write_csv(bars_path, merged_bars)
    feature_rows = compute_daily_features(merged_bars)
    write_csv(features_path, feature_rows)

    normalized_rows: list[dict[str, object]] | None = None
    if not args.skip_normalization and metadata_path.exists():
        sector_metadata = load_sp500_constituent_metadata(metadata_path)
        normalized_rows = compute_normalized_feature_rows(feature_rows, sector_metadata)
        write_csv(normalized_path, normalized_rows)
        trade_dates = [str(row["date"]) for row in normalized_rows if row.get("date")]
        write_json(
            normalized_manifest_path,
            build_normalized_manifest(
                input_file=features_path,
                sector_source_file=metadata_path,
                row_count=len(normalized_rows),
                universe_count=len({str(row["ticker"]) for row in normalized_rows if row.get("ticker")}),
                min_date=min(trade_dates) if trade_dates else None,
                max_date=max(trade_dates) if trade_dates else None,
            ),
        )

    rows_for_windows = normalized_rows if normalized_rows is not None else feature_rows
    episodes = compute_episode_index(
        feature_rows=feature_rows,
        window_length=args.window_length,
        horizon_days=args.horizon_days,
        benchmark_ticker=benchmark_ticker,
    )
    prediction_windows = compute_latest_prediction_windows(
        rows_for_windows,
        window_length=args.window_length,
        target_horizon_days=args.horizon_days,
        benchmark_ticker=benchmark_ticker,
        anchor_date=args.prediction_anchor_date or None,
    )

    _write_csv_with_headers(episode_path, episodes, EPISODE_INDEX_HEADERS)
    _write_csv_with_headers(prediction_windows_path, prediction_windows, PREDICTION_WINDOW_HEADERS)

    min_date = min_bar_date(merged_bars)
    max_date = max_bar_date(merged_bars)
    write_incremental_json(
        update_manifest_path,
        build_incremental_update_manifest(
            dataset_root=dataset_root,
            raw_rows=raw_rows_before,
            incoming_rows=len(incoming_bars),
            merged_rows=len(merged_bars),
            feature_rows=len(feature_rows),
            normalized_rows=len(normalized_rows) if normalized_rows is not None else None,
            episode_rows=len(episodes),
            prediction_window_rows=len(prediction_windows),
            min_date=min_date.isoformat() if min_date else None,
            max_date=max_date.isoformat() if max_date else None,
            fetch_start_date=fetch_start_date.isoformat() if fetch_start_date else None,
            fetch_end_date=args.end_date if not args.skip_fetch else None,
            tickers=tickers,
            window_length=args.window_length,
            horizon_days=args.horizon_days,
            recent_overlap_days=args.recent_overlap_days,
            benchmark_ticker=benchmark_ticker,
            skipped_fetch=args.skip_fetch,
        ),
    )

    print(
        json.dumps(
            {
                "step": "complete",
                "dataset_root": str(dataset_root.resolve()),
                "raw_rows_before": raw_rows_before,
                "incoming_rows": len(incoming_bars),
                "raw_rows_after": len(merged_bars),
                "daily_feature_rows": len(feature_rows),
                "normalized_feature_rows": len(normalized_rows) if normalized_rows is not None else None,
                "episode_rows": len(episodes),
                "prediction_window_rows": len(prediction_windows),
                "latest_raw_date": max_date.isoformat() if max_date else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except MassiveError as exc:
        print(f"Massive update failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
