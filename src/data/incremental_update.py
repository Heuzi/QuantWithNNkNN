from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence


DAILY_BAR_KEY_FIELDS = ("ticker", "date", "adjusted")


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_tickers_from_constituents(path: str | Path) -> list[str]:
    path = Path(path)
    tickers: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ticker = (row.get("symbol") or row.get("ticker") or "").strip().upper()
            if ticker:
                tickers.append(ticker)
    return list(dict.fromkeys(tickers))


def unique_tickers_from_bars(rows: Sequence[dict[str, object]]) -> list[str]:
    tickers = [str(row["ticker"]).upper() for row in rows if row.get("ticker")]
    return sorted(set(tickers))


def max_bar_date(rows: Sequence[dict[str, object]]) -> date | None:
    dates = [parse_iso_date(str(row["date"])) for row in rows if row.get("date")]
    return max(dates) if dates else None


def min_bar_date(rows: Sequence[dict[str, object]]) -> date | None:
    dates = [parse_iso_date(str(row["date"])) for row in rows if row.get("date")]
    return min(dates) if dates else None


def determine_incremental_fetch_start(
    existing_rows: Sequence[dict[str, object]],
    *,
    fallback_start_date: date,
    overlap_days: int,
) -> date:
    latest = max_bar_date(existing_rows)
    if latest is None:
        return fallback_start_date
    if overlap_days <= 0:
        return latest + timedelta(days=1)
    return latest - timedelta(days=overlap_days - 1)


def _daily_bar_key(row: dict[str, object]) -> tuple[str, str, bool]:
    return (
        str(row["ticker"]).upper(),
        str(row["date"]),
        bool(row.get("adjusted", True)),
    )


def merge_daily_bar_rows(
    existing_rows: Sequence[dict[str, object]],
    incoming_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """
    Merge daily bars by ticker/date/adjusted flag, with incoming rows replacing
    existing rows for the same key.
    """
    merged: dict[tuple[str, str, bool], dict[str, object]] = {}
    for row in existing_rows:
        normalized = dict(row)
        if normalized.get("ticker"):
            normalized["ticker"] = str(normalized["ticker"]).upper()
        merged[_daily_bar_key(normalized)] = normalized
    for row in incoming_rows:
        normalized = dict(row)
        if normalized.get("ticker"):
            normalized["ticker"] = str(normalized["ticker"]).upper()
        merged[_daily_bar_key(normalized)] = normalized

    return sorted(merged.values(), key=lambda item: (str(item["ticker"]), str(item["date"])))


def compute_latest_prediction_windows(
    feature_rows: Sequence[dict[str, object]],
    *,
    window_length: int,
    target_horizon_days: int,
    benchmark_ticker: str | None = None,
    anchor_date: str | None = None,
    tickers: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """
    Build one latest inference window per ticker.

    These rows intentionally do not include realized targets. They are suitable
    for production-style prediction after the latest market data update.
    """
    requested_tickers = {ticker.upper() for ticker in tickers or []}
    cutoff = parse_iso_date(anchor_date) if anchor_date else None
    benchmark = benchmark_ticker.upper() if benchmark_ticker else None

    by_ticker: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in feature_rows:
        ticker = str(row["ticker"]).upper()
        if benchmark and ticker == benchmark:
            continue
        if requested_tickers and ticker not in requested_tickers:
            continue
        row_date = parse_iso_date(str(row["date"]))
        if cutoff and row_date > cutoff:
            continue
        by_ticker[ticker].append(dict(row))

    windows: list[dict[str, object]] = []
    for ticker, rows in by_ticker.items():
        rows.sort(key=lambda item: str(item["date"]))
        if len(rows) < window_length:
            continue
        selected = rows[-window_length:]
        anchor = selected[-1]
        windows.append(
            {
                "ticker": ticker,
                "anchor_date": anchor["date"],
                "window_start_date": selected[0]["date"],
                "window_end_date": anchor["date"],
                "window_length": window_length,
                "available_window_rows": len(selected),
                "target_horizon_days": target_horizon_days,
                "target_status": "pending_future_return",
                "target_return": None,
                "market_adjusted_target_return": None,
                "benchmark_ticker": benchmark,
                "anchor_selection": "latest_on_or_before_anchor_date" if cutoff else "latest_available_per_ticker",
                "inference_ready": len(selected) == window_length,
            }
        )

    windows.sort(key=lambda item: (str(item["anchor_date"]), str(item["ticker"])))
    return windows


def build_incremental_update_manifest(
    *,
    dataset_root: str | Path,
    raw_rows: int,
    incoming_rows: int,
    merged_rows: int,
    feature_rows: int,
    normalized_rows: int | None,
    episode_rows: int,
    prediction_window_rows: int,
    min_date: str | None,
    max_date: str | None,
    fetch_start_date: str | None,
    fetch_end_date: str | None,
    tickers: Sequence[str],
    window_length: int,
    horizon_days: int,
    recent_overlap_days: int,
    benchmark_ticker: str,
    skipped_fetch: bool,
) -> dict[str, object]:
    return {
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset_root": str(Path(dataset_root).resolve()),
        "raw_rows_before_update": raw_rows,
        "incoming_rows": incoming_rows,
        "raw_rows_after_merge": merged_rows,
        "daily_feature_rows": feature_rows,
        "normalized_feature_rows": normalized_rows,
        "completed_episode_rows": episode_rows,
        "latest_prediction_window_rows": prediction_window_rows,
        "date_range": {
            "min_date": min_date,
            "max_date": max_date,
        },
        "fetch": {
            "skipped": skipped_fetch,
            "start_date": fetch_start_date,
            "end_date": fetch_end_date,
            "recent_overlap_days": recent_overlap_days,
            "ticker_count": len(tickers),
            "tickers": list(tickers),
        },
        "parameters": {
            "window_length": window_length,
            "horizon_days": horizon_days,
            "benchmark_ticker": benchmark_ticker,
        },
        "output_semantics": {
            "episode_index.csv": "Completed supervised rows only; every row has a realized future target.",
            "prediction_windows.csv": "Latest target-pending windows for production-style inference after the most recent data update.",
        },
        "known_risks": [
            "Recent updates refresh only the configured overlap window, so older vendor restatements or corporate-action adjustment changes may require a full rebuild.",
            "The current S&P 500 constituent file is not a historically point-in-time membership panel.",
            "Ticker identity and symbol reuse are not fully resolved by this daily bar-only updater.",
        ],
    }


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
