from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from datetime import date, datetime, timedelta
from io import BytesIO, TextIOWrapper
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.massive_stage1 import (  # noqa: E402
    DEFAULT_RATE_LIMIT_CALLS,
    DEFAULT_RATE_LIMIT_PERIOD_SECONDS,
    MassiveError,
    MassiveRESTClient,
    RateLimiter,
    compute_daily_features,
    fetch_bars_for_tickers,
    load_daily_bars_csv,
    load_massive_credentials,
    write_csv,
    write_json,
)
from src.data.v1_dataset import MARKET_CONTEXT_TICKERS, SECTOR_ETF_BY_GICS  # noqa: E402


def parse_args() -> argparse.Namespace:
    today = date.today()
    default_end = today - timedelta(days=1)
    parser = argparse.ArgumentParser(
        description="Collect SPY and sector ETF daily context bars from Massive for V1 supervised baselines."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/massive_sp500_current_constituents_history",
        help="Dataset folder where raw/market_context_bars.csv and processed/market_context_features.csv are written.",
    )
    parser.add_argument("--start-date", default="", help="Inclusive YYYY-MM-DD. Defaults to stock raw min date.")
    parser.add_argument("--end-date", default=default_end.isoformat(), help="Inclusive YYYY-MM-DD.")
    parser.add_argument("--credentials-path", default="MassiveApiKey", help="Path to Massive credential file.")
    parser.add_argument(
        "--source",
        choices=["rest", "flatfiles"],
        default="flatfiles",
        help="Use REST ticker range calls or S3 flat files. Flat files use MASSIVE_ACCESS_KEY_ID credentials.",
    )
    parser.add_argument(
        "--rate-limit-calls",
        type=int,
        default=DEFAULT_RATE_LIMIT_CALLS,
        help="Maximum Massive REST calls per rate-limit window.",
    )
    parser.add_argument(
        "--rate-limit-period-seconds",
        type=float,
        default=DEFAULT_RATE_LIMIT_PERIOD_SECONDS,
        help="Rate-limit rolling window in seconds.",
    )
    parser.add_argument(
        "--tickers",
        default="",
        help="Optional comma-separated context ticker list. Defaults to SPY and sector ETFs.",
    )
    parser.add_argument("--unadjusted", action="store_true", help="Request unadjusted bars.")
    return parser.parse_args()


def _infer_start_date(dataset_root: Path, explicit_start: str) -> str:
    if explicit_start:
        return explicit_start
    stock_bars_path = dataset_root / "raw" / "daily_market_bars.csv"
    if not stock_bars_path.exists():
        raise SystemExit("--start-date is required when raw/daily_market_bars.csv is missing.")
    bars = load_daily_bars_csv(stock_bars_path)
    dates = [str(row["date"]) for row in bars if row.get("date")]
    if not dates:
        raise SystemExit("Could not infer start date from empty raw/daily_market_bars.csv.")
    return min(dates)


def _iter_weekdays(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def _flatfile_key(trade_date: date) -> str:
    value = trade_date.isoformat()
    return f"us_stocks_sip/day_aggs_v1/{trade_date:%Y}/{trade_date:%m}/{value}.csv.gz"


def _build_s3_client(credentials):
    if not credentials.access_key_id or not credentials.secret_access_key:
        raise SystemExit("Flat-file collection needs MASSIVE_ACCESS_KEY_ID and MASSIVE_SECRET_ACCESS_KEY.")
    if not credentials.s3_endpoint or not credentials.s3_bucket:
        raise SystemExit("Flat-file collection needs MASSIVE_S3_ENDPOINT and MASSIVE_S3_BUCKET.")

    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=credentials.s3_endpoint,
        aws_access_key_id=credentials.access_key_id,
        aws_secret_access_key=credentials.secret_access_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def _parse_flatfile_row(row: dict[str, str], trade_date: str) -> dict[str, object]:
    ticker = row.get("ticker") or row.get("T") or row.get("symbol")
    close = row.get("close") or row.get("c")
    volume = row.get("volume") or row.get("v")
    open_ = row.get("open") or row.get("o")
    high = row.get("high") or row.get("h")
    low = row.get("low") or row.get("l")
    vwap = row.get("vwap") or row.get("vw")
    transactions = row.get("transactions") or row.get("n")
    window_start = row.get("window_start") or row.get("timestamp") or row.get("t")

    def as_float(value: str | None) -> float | None:
        if value in (None, ""):
            return None
        return float(value)

    close_value = as_float(close)
    volume_value = as_float(volume)
    return {
        "date": trade_date,
        "ticker": str(ticker).upper() if ticker else None,
        "open": as_float(open_),
        "high": as_float(high),
        "low": as_float(low),
        "close": close_value,
        "volume": volume_value,
        "vwap": as_float(vwap),
        "transactions": as_float(transactions),
        "timestamp_ms": as_float(window_start),
        "adjusted": True,
        "dollar_volume": close_value * volume_value if close_value is not None and volume_value is not None else None,
    }


def fetch_context_bars_from_flatfiles(
    *,
    credentials,
    tickers: list[str],
    start_date: date,
    end_date: date,
    rate_limit_calls: int,
    rate_limit_period_seconds: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    from botocore.exceptions import ClientError

    client = _build_s3_client(credentials)
    wanted = {ticker.upper() for ticker in tickers}
    rows: list[dict[str, object]] = []
    file_status: list[dict[str, object]] = []
    bucket = credentials.s3_bucket
    sleep_after_file = rate_limit_period_seconds / max(rate_limit_calls, 1)

    for trade_date in _iter_weekdays(start_date, end_date):
        key = _flatfile_key(trade_date)
        trade_date_str = trade_date.isoformat()
        try:
            response = client.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"NoSuchKey", "404", "NotFound"}:
                file_status.append({"date": trade_date_str, "key": key, "status": "missing"})
                time.sleep(sleep_after_file)
                continue
            raise

        body = response["Body"].read()
        matched = 0
        with gzip.GzipFile(fileobj=BytesIO(body)) as gz_handle:
            text_handle = TextIOWrapper(gz_handle, encoding="utf-8", newline="")
            reader = csv.DictReader(text_handle)
            for raw_row in reader:
                ticker = (raw_row.get("ticker") or raw_row.get("T") or raw_row.get("symbol") or "").upper()
                if ticker not in wanted:
                    continue
                rows.append(_parse_flatfile_row(raw_row, trade_date_str))
                matched += 1
        file_status.append({"date": trade_date_str, "key": key, "status": "ok", "matched_rows": matched})
        print(json.dumps({"step": "flatfile_date", "date": trade_date_str, "matched_rows": matched}), flush=True)
        time.sleep(sleep_after_file)

    return rows, file_status


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    start_date = datetime.strptime(_infer_start_date(dataset_root, args.start_date), "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if end_date < start_date:
        raise SystemExit("--end-date must be on or after start date")

    tickers = (
        [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
        if args.tickers.strip()
        else list(MARKET_CONTEXT_TICKERS)
    )
    credentials = load_massive_credentials(args.credentials_path)
    print(
        json.dumps(
            {
                "step": "fetch_market_context",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "tickers": tickers,
                "source": args.source,
                "rate_limit_calls": args.rate_limit_calls,
                "rate_limit_period_seconds": args.rate_limit_period_seconds,
            },
            indent=2,
        )
    )
    flatfile_status: list[dict[str, object]] = []
    if args.source == "rest":
        client = MassiveRESTClient(
            credentials=credentials,
            rate_limiter=RateLimiter(
                max_calls=args.rate_limit_calls,
                period_seconds=args.rate_limit_period_seconds,
            ),
        )
        bars = fetch_bars_for_tickers(
            client=client,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            adjusted=not args.unadjusted,
        )
    else:
        bars, flatfile_status = fetch_context_bars_from_flatfiles(
            credentials=credentials,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            rate_limit_calls=args.rate_limit_calls,
            rate_limit_period_seconds=args.rate_limit_period_seconds,
        )
    features = compute_daily_features(bars)

    raw_path = dataset_root / "raw" / "market_context_bars.csv"
    processed_path = dataset_root / "processed" / "market_context_features.csv"
    manifest_path = dataset_root / "processed" / "market_context_manifest.json"
    write_csv(raw_path, bars)
    write_csv(processed_path, features)
    write_json(
        manifest_path,
        {
            "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "source": "Massive ticker range daily aggregates" if args.source == "rest" else "Massive S3 flat-file day aggregates",
            "tickers": tickers,
            "sector_etf_by_gics": SECTOR_ETF_BY_GICS,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "flatfile_dataset": "us_stocks_sip/day_aggs_v1" if args.source == "flatfiles" else None,
            "flatfile_status": flatfile_status,
            "rate_limit": {
                "calls": args.rate_limit_calls,
                "period_seconds": args.rate_limit_period_seconds,
            },
            "raw_rows": len(bars),
            "feature_rows": len(features),
            "files": {
                "raw_bars": str(raw_path.resolve()),
                "features": str(processed_path.resolve()),
            },
            "timing": "End-of-day context. Downstream joins must use context dates <= anchor_date.",
        },
    )
    print(
        json.dumps(
            {
                "step": "complete",
                "raw_rows": len(bars),
                "feature_rows": len(features),
                "processed_path": str(processed_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except MassiveError as exc:
        print(f"Massive context collection failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
