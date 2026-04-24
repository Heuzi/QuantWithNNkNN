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

from src.data.massive_stage1 import (  # noqa: E402
    MassiveError,
    MassiveRESTClient,
    append_csv,
    fetch_sp500_constituents_from_wikipedia,
    load_massive_credentials,
    normalize_ticker_range_bars,
    write_csv,
    write_json,
)


BARS_HEADERS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "transactions",
    "timestamp_ms",
    "adjusted",
    "dollar_volume",
]

FAILURE_HEADERS = ["ticker", "error"]


def parse_args() -> argparse.Namespace:
    today = date.today()
    default_end = today - timedelta(days=1)
    parser = argparse.ArgumentParser(
        description=(
            "Collect current S&P 500 constituent daily history from Massive with "
            "minimal REST calls by using one long range request per ticker."
        )
    )
    parser.add_argument("--start-date", default="1995-01-01", help="Inclusive YYYY-MM-DD.")
    parser.add_argument("--end-date", default=default_end.isoformat(), help="Inclusive YYYY-MM-DD.")
    parser.add_argument(
        "--output-root",
        default="data/massive_sp500_current_constituents_history",
        help="Output folder for raw bars, metadata, and progress files.",
    )
    parser.add_argument(
        "--credentials-path",
        default="MassiveApiKey",
        help="Path to the env-style Massive credential file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the progress file if a previous run exists.",
    )
    parser.add_argument(
        "--adjusted",
        action="store_true",
        default=True,
        help="Request adjusted daily bars. This is the default.",
    )
    return parser.parse_args()


def load_completed_tickers(progress_path: Path) -> set[str]:
    if not progress_path.exists():
        return set()
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    return set(payload.get("completed_tickers", []))


def main() -> None:
    args = parse_args()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if end_date < start_date:
        raise SystemExit("--end-date must be on or after --start-date")

    output_root = Path(args.output_root)
    raw_dir = output_root / "raw"
    progress_path = output_root / "progress.json"
    summary_path = output_root / "summary.json"
    metadata_path = raw_dir / "sp500_constituents_current.csv"
    bars_path = raw_dir / "daily_market_bars.csv"
    failures_path = raw_dir / "failures.csv"

    credentials = load_massive_credentials(args.credentials_path)
    client = MassiveRESTClient(credentials=credentials)

    constituents = fetch_sp500_constituents_from_wikipedia()
    tickers = [row["symbol"].upper() for row in constituents]
    write_csv(metadata_path, constituents)

    if not args.resume:
        if bars_path.exists():
            bars_path.unlink()
        if failures_path.exists():
            failures_path.unlink()
        completed_tickers: set[str] = set()
    else:
        completed_tickers = load_completed_tickers(progress_path)

    total_tickers = len(tickers)
    print(
        json.dumps(
            {
                "step": "start",
                "start_date": args.start_date,
                "end_date": args.end_date,
                "constituent_count": total_tickers,
                "resume": args.resume,
                "already_completed": len(completed_tickers),
                "output_root": str(output_root.resolve()),
            },
            indent=2,
        )
    )

    rows_written = 0
    failures = 0
    for index, ticker in enumerate(tickers, start=1):
        if ticker in completed_tickers:
            continue

        print(
            json.dumps(
                {
                    "step": "fetch_ticker",
                    "ticker": ticker,
                    "index": index,
                    "total_tickers": total_tickers,
                }
            ),
            flush=True,
        )

        try:
            payload = client.get_ticker_range_aggs(
                ticker=ticker,
                from_date=args.start_date,
                to_date=args.end_date,
                adjusted=args.adjusted,
            )
            rows = normalize_ticker_range_bars(payload)
            append_csv(bars_path, rows, headers=BARS_HEADERS)
            rows_written += len(rows)
            completed_tickers.add(ticker)
        except MassiveError as exc:
            failures += 1
            append_csv(failures_path, [{"ticker": ticker, "error": str(exc)}], headers=FAILURE_HEADERS)

        write_json(
            progress_path,
            {
                "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "start_date": args.start_date,
                "end_date": args.end_date,
                "constituent_count": total_tickers,
                "completed_tickers": sorted(completed_tickers),
                "completed_count": len(completed_tickers),
                "failures_logged": failures,
                "rows_written_this_run": rows_written,
            },
        )

    summary = {
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "constituent_count": total_tickers,
        "completed_count": len(completed_tickers),
        "failures_logged": failures,
        "bars_path": str(bars_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "notes": [
            "Constituents are the current S&P 500 list fetched at collection time, not a historical membership panel.",
            "Using the current constituent list for a 1995-present backfill introduces survivorship bias.",
            "This collector minimizes Massive REST usage by using one range request per ticker.",
        ],
    }
    write_json(summary_path, summary)
    print(json.dumps({"step": "complete", **summary}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except MassiveError as exc:
        print(f"Massive collection failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
