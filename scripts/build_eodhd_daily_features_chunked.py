from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.eodhd_enrichment import (  # noqa: E402
    add_fundamental_features,
    add_sentiment_features,
    load_fundamental_feature_rows,
    load_sentiment_rows,
)
from src.data.massive_stage1 import (  # noqa: E402
    append_csv,
    compute_daily_features,
    load_daily_bars_csv,
    write_csv,
    write_json,
)


NUMERIC_BAR_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "raw_close",
    "adjusted_close",
    "volume",
    "adjustment_factor",
    "dollar_volume",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build EODHD daily feature CSVs from raw bars without loading the full panel into memory."
    )
    parser.add_argument("--dataset-root", default="data/eodhd_us_equities_30y")
    parser.add_argument("--max-tickers", type=int, default=0, help="Optional smoke cap; 0 means all raw tickers.")
    parser.add_argument("--skip-fundamentals", action="store_true")
    parser.add_argument("--skip-sentiment", action="store_true")
    return parser.parse_args()


def _parse_bar_row(row: dict[str, str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in row.items():
        if value == "":
            parsed[key] = None
        elif key in NUMERIC_BAR_FIELDS:
            parsed[key] = float(value)
        elif key == "adjusted":
            parsed[key] = value.lower() == "true"
        elif key in {"ticker", "eodhd_symbol", "exchange"}:
            parsed[key] = value.upper() if value else value
        else:
            parsed[key] = value
    return parsed


def iter_contiguous_ticker_groups(path: Path) -> Iterator[tuple[str, str, list[dict[str, object]]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        current_ticker = ""
        current_symbol = ""
        rows: list[dict[str, object]] = []
        for raw_row in reader:
            row = _parse_bar_row(raw_row)
            ticker = str(row.get("ticker") or "").upper()
            symbol = str(row.get("eodhd_symbol") or "").upper()
            if rows and ticker != current_ticker:
                yield current_ticker, current_symbol, rows
                rows = []
            current_ticker = ticker
            current_symbol = symbol
            rows.append(row)
        if rows:
            yield current_ticker, current_symbol, rows


def _sentiment_by_ticker(path: Path) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in load_sentiment_rows(path):
        ticker = str(row.get("ticker") or "").upper()
        if ticker:
            grouped.setdefault(ticker, []).append(row)
    return grouped


def _write_chunk(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if rows:
        append_csv(path, rows)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    raw_dir = dataset_root / "raw"
    processed_dir = dataset_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    stock_bars_path = raw_dir / "eodhd_stock_bars.csv"
    context_bars_path = raw_dir / "market_context_bars.csv"
    fundamentals_raw_dir = raw_dir / "eodhd_fundamentals_raw"
    sentiment_path = raw_dir / "eodhd_sentiment_daily.csv"
    features_path = processed_dir / "daily_features.csv"
    context_features_path = processed_dir / "market_context_features.csv"
    manifest_path = processed_dir / "daily_features_chunked_manifest.json"

    if not stock_bars_path.exists():
        raise SystemExit(f"Missing raw stock bars: {stock_bars_path}")

    if features_path.exists():
        features_path.unlink()
    if context_features_path.exists():
        context_features_path.unlink()

    sentiment_by_ticker = {} if args.skip_sentiment else _sentiment_by_ticker(sentiment_path)

    ticker_count = 0
    feature_row_count = 0
    for ticker, symbol, raw_rows in iter_contiguous_ticker_groups(stock_bars_path):
        if args.max_tickers and ticker_count >= args.max_tickers:
            break
        feature_rows = compute_daily_features(raw_rows)
        if not args.skip_fundamentals:
            fundamental_rows = load_fundamental_feature_rows(fundamentals_raw_dir, symbols=[symbol])
            feature_rows = add_fundamental_features(feature_rows, fundamental_rows)
        if not args.skip_sentiment:
            feature_rows = add_sentiment_features(feature_rows, sentiment_by_ticker.get(ticker, []))
        _write_chunk(features_path, feature_rows)
        ticker_count += 1
        feature_row_count += len(feature_rows)

    context_rows = load_daily_bars_csv(context_bars_path) if context_bars_path.exists() else []
    context_features = compute_daily_features(context_rows)
    if context_features:
        write_csv(context_features_path, context_features)

    write_json(
        manifest_path,
        {
            "dataset_root": str(dataset_root.resolve()),
            "mode": "chunked_daily_features",
            "stock_tickers_processed": ticker_count,
            "stock_feature_rows": feature_row_count,
            "context_feature_rows": len(context_features),
            "fundamentals_enabled": not args.skip_fundamentals,
            "sentiment_enabled": not args.skip_sentiment,
            "normalized_features_written": False,
            "notes": [
                "Rows are processed one contiguous raw ticker block at a time.",
                "Full-panel cross-sectional normalization is intentionally not computed here.",
            ],
        },
    )
    print(
        json.dumps(
            {
            "step": "chunked_features_complete",
            "dataset_root": str(dataset_root.resolve()),
            "stock_tickers_processed": ticker_count,
            "stock_feature_rows": feature_row_count,
            "context_feature_rows": len(context_features),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
