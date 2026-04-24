from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.massive_stage1 import (
    DEFAULT_BENCHMARK_TICKER,
    MassiveRESTClient,
    MassiveError,
    choose_sample_universe,
    compute_daily_features,
    compute_episode_index,
    default_collection_manifest,
    fetch_bars_for_dates,
    fetch_bars_for_tickers,
    fetch_ticker_details_snapshots,
    load_massive_credentials,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    today = date.today()
    default_end = today - timedelta(days=1)
    default_start = default_end - timedelta(days=140)

    parser = argparse.ArgumentParser(
        description=(
            "Collect a small backtest-minded Stage 1 / Version 1 dataset from Massive "
            "using daily data only."
        )
    )
    parser.add_argument("--start-date", default=default_start.isoformat(), help="Inclusive YYYY-MM-DD.")
    parser.add_argument("--end-date", default=default_end.isoformat(), help="Inclusive YYYY-MM-DD.")
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=8,
        help="Number of non-benchmark tickers to keep when tickers are not explicitly provided.",
    )
    parser.add_argument(
        "--tickers",
        default="",
        help="Optional comma-separated ticker list. If omitted, the script picks liquid names from the sample.",
    )
    parser.add_argument(
        "--benchmark-ticker",
        default=DEFAULT_BENCHMARK_TICKER,
        help="Benchmark ticker used for market-adjusted targets.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=60,
        help="Lookback window length in trading days for episode creation.",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=5,
        help="Forward return horizon in trading days.",
    )
    parser.add_argument(
        "--credentials-path",
        default="MassiveApiKey",
        help="Path to the env-style Massive credential file.",
    )
    parser.add_argument(
        "--output-root",
        default="data/massive_stage1_sample",
        help="Folder where raw and processed CSVs should be written.",
    )
    parser.add_argument(
        "--include-otc",
        action="store_true",
        help="Include OTC securities in grouped daily aggregates.",
    )
    parser.add_argument(
        "--unadjusted",
        action="store_true",
        help="Use unadjusted grouped bars. By default the script requests split-adjusted bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if end_date < start_date:
        raise SystemExit("--end-date must be on or after --start-date")

    requested_tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    credentials = load_massive_credentials(args.credentials_path)
    client = MassiveRESTClient(credentials=credentials)

    benchmark_ticker = args.benchmark_ticker.upper()
    if requested_tickers:
        selected_tickers = list(dict.fromkeys(requested_tickers + [benchmark_ticker]))
        print(
            json.dumps(
                {
                    "step": "fetch_ticker_range_aggs",
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                    "selected_tickers": selected_tickers,
                    "benchmark_ticker": benchmark_ticker,
                },
                indent=2,
            )
        )
        filtered_bars = fetch_bars_for_tickers(
            client=client,
            tickers=selected_tickers,
            start_date=start_date,
            end_date=end_date,
            adjusted=not args.unadjusted,
        )
    else:
        print(
            json.dumps(
                {
                    "step": "fetch_grouped_daily_aggs",
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                    "max_tickers": args.max_tickers,
                    "requested_tickers": requested_tickers,
                    "benchmark_ticker": benchmark_ticker,
                },
                indent=2,
            )
        )
        all_bars = fetch_bars_for_dates(
            client=client,
            start_date=start_date,
            end_date=end_date,
            adjusted=not args.unadjusted,
            include_otc=args.include_otc,
        )
        if not all_bars:
            raise SystemExit("No grouped daily bars were returned for the requested date range.")

        selected_tickers = choose_sample_universe(
            bars=all_bars,
            max_tickers=args.max_tickers,
            forced_tickers=requested_tickers,
            benchmark_ticker=benchmark_ticker,
        )
        filtered_bars = [row for row in all_bars if str(row["ticker"]).upper() in set(selected_tickers)]

    if not filtered_bars:
        raise SystemExit("No daily bars were returned for the requested date range and tickers.")

    print(
        json.dumps(
            {
                "step": "fetch_ticker_details",
                "selected_tickers": selected_tickers,
                "bar_rows": len(filtered_bars),
            },
            indent=2,
        )
    )

    ticker_details = fetch_ticker_details_snapshots(
        client=client,
        tickers=selected_tickers,
        as_of_date=args.end_date,
    )
    feature_rows = compute_daily_features(filtered_bars)
    episodes = compute_episode_index(
        feature_rows=feature_rows,
        window_length=args.window_length,
        horizon_days=args.horizon_days,
        benchmark_ticker=benchmark_ticker,
    )

    output_root = Path(args.output_root)
    write_csv(output_root / "raw" / "daily_market_bars.csv", filtered_bars)
    write_csv(output_root / "raw" / "ticker_reference.csv", ticker_details)
    write_csv(output_root / "processed" / "daily_features.csv", feature_rows)
    write_csv(output_root / "processed" / "episode_index.csv", episodes)
    write_json(
        output_root / "manifest.json",
        default_collection_manifest(
            start_date=args.start_date,
            end_date=args.end_date,
            max_tickers=args.max_tickers,
            selected_tickers=selected_tickers,
            window_length=args.window_length,
            horizon_days=args.horizon_days,
            benchmark_ticker=benchmark_ticker,
        ),
    )
    write_json(
        output_root / "summary.json",
        {
            "selected_tickers": selected_tickers,
            "daily_bar_rows": len(filtered_bars),
            "ticker_reference_rows": len(ticker_details),
            "daily_feature_rows": len(feature_rows),
            "episode_rows": len(episodes),
            "credentials_source": credentials.source_path,
            "notes": [
                "This bootstrap dataset is intentionally small and daily-only.",
                "True filing-dated fundamentals are not included yet.",
            ],
        },
    )

    print(
        json.dumps(
            {
                "step": "complete",
                "output_root": str(output_root.resolve()),
                "selected_tickers": selected_tickers,
                "daily_feature_rows": len(feature_rows),
                "episode_rows": len(episodes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except MassiveError as exc:
        print(f"Massive collection failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
