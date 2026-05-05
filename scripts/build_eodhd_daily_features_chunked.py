from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
METADATA_COLUMNS = (
    "name",
    "country",
    "currency",
    "type",
    "isin",
    "sector",
    "industry",
    "gics_sector",
    "gics_sub_industry",
    "is_delisted",
    "delisted_date",
    "metadata_source",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build EODHD daily feature CSVs from raw bars without loading the full panel into memory."
    )
    parser.add_argument("--dataset-root", default="data/eodhd_us_equities_30y")
    parser.add_argument("--max-tickers", type=int, default=0, help="Optional smoke cap; 0 means all raw tickers.")
    parser.add_argument("--skip-fundamentals", action="store_true")
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing processed feature outputs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted stock feature build by appending after the last completed ticker.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress after this many processed tickers. Use 0 to disable.",
    )
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


def _load_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row.get("ticker") or "").upper(): row
            for row in reader
            if str(row.get("ticker") or "").strip()
        }


def _add_metadata(feature_rows: Sequence[dict[str, object]], metadata: dict[str, str]) -> list[dict[str, object]]:
    # The trainer treats ticker/symbol as metadata only, but sequence/static models
    # need sector/industry categorical fields. Join those here so future feature
    # rebuilds are directly trainable without a separate materialization fix-up.
    out: list[dict[str, object]] = []
    for row in feature_rows:
        enriched = dict(row)
        for column in METADATA_COLUMNS:
            if enriched.get(column) in (None, ""):
                enriched[column] = metadata.get(column) or ("Unknown" if column in {"gics_sector", "gics_sub_industry"} else "")
        out.append(enriched)
    return out


def _last_csv_data_line(path: Path, *, block_size: int = 65536) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        buffer = b""
        while position > 0:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            buffer = handle.read(read_size) + buffer
            lines = buffer.splitlines()
            if len(lines) > 1:
                for line in reversed(lines):
                    text = line.decode("utf-8", errors="ignore").strip()
                    if text:
                        return text
        text = buffer.decode("utf-8", errors="ignore").strip()
    return text


def _last_completed_ticker_from_csv(path: Path) -> str:
    line = _last_csv_data_line(path)
    if not line or line.lower().startswith("date,"):
        return ""
    try:
        row = next(csv.DictReader([line], fieldnames=["date", "ticker"]))
    except Exception:
        return ""
    return str(row.get("ticker") or "").upper()


def _load_resume_ticker(checkpoint_path: Path, features_path: Path) -> str:
    if checkpoint_path.exists():
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            ticker = str(payload.get("last_completed_ticker") or "").upper()
            if ticker:
                return ticker
        except Exception:
            pass
    return _last_completed_ticker_from_csv(features_path)


def _write_checkpoint(
    checkpoint_path: Path,
    *,
    ticker: str,
    symbol: str,
    ticker_count: int,
    feature_row_count: int,
    start_time: float,
) -> None:
    write_json(
        checkpoint_path,
        {
            "last_completed_ticker": ticker,
            "last_completed_symbol": symbol,
            "stock_tickers_processed_this_run": ticker_count,
            "stock_feature_rows_this_run": feature_row_count,
            "elapsed_seconds_this_run": round(time.monotonic() - start_time, 1),
        },
    )


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    raw_dir = dataset_root / "raw"
    processed_dir = dataset_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    stock_bars_path = raw_dir / "eodhd_stock_bars.csv"
    context_bars_path = raw_dir / "market_context_bars.csv"
    metadata_path = raw_dir / "eodhd_equity_metadata.csv"
    fundamentals_raw_dir = raw_dir / "eodhd_fundamentals_raw"
    sentiment_path = raw_dir / "eodhd_sentiment_daily.csv"
    features_path = processed_dir / "daily_features.csv"
    context_features_path = processed_dir / "market_context_features.csv"
    manifest_path = processed_dir / "daily_features_chunked_manifest.json"
    checkpoint_path = processed_dir / "daily_features_chunked_checkpoint.json"

    if not stock_bars_path.exists():
        raise SystemExit(f"Missing raw stock bars: {stock_bars_path}")

    if args.force and args.resume:
        raise SystemExit("--force and --resume cannot be used together.")
    if (features_path.exists() or context_features_path.exists()) and not (args.force or args.resume):
        raise SystemExit(
            "Processed feature output already exists. Use --force to rebuild: "
            f"{features_path} / {context_features_path}; or use --resume after an interrupted build."
        )
    if features_path.exists():
        if args.force:
            features_path.unlink()
    if context_features_path.exists():
        if args.force:
            context_features_path.unlink()
    if checkpoint_path.exists() and args.force:
        checkpoint_path.unlink()

    sentiment_by_ticker = {} if args.skip_sentiment else _sentiment_by_ticker(sentiment_path)
    metadata_by_ticker = _load_metadata(metadata_path)

    start_time = time.monotonic()
    resume_after_ticker = _load_resume_ticker(checkpoint_path, features_path) if args.resume else ""
    skipping_completed = bool(resume_after_ticker)
    ticker_count = 0
    feature_row_count = 0
    for ticker, symbol, raw_rows in iter_contiguous_ticker_groups(stock_bars_path):
        if skipping_completed:
            if ticker == resume_after_ticker:
                skipping_completed = False
            continue
        if args.max_tickers and ticker_count >= args.max_tickers:
            break
        feature_rows = compute_daily_features(raw_rows)
        feature_rows = _add_metadata(feature_rows, metadata_by_ticker.get(ticker, {}))
        if not args.skip_fundamentals:
            fundamental_rows = load_fundamental_feature_rows(fundamentals_raw_dir, symbols=[symbol])
            feature_rows = add_fundamental_features(feature_rows, fundamental_rows)
        if not args.skip_sentiment:
            feature_rows = add_sentiment_features(feature_rows, sentiment_by_ticker.get(ticker, []))
        _write_chunk(features_path, feature_rows)
        ticker_count += 1
        feature_row_count += len(feature_rows)
        _write_checkpoint(
            checkpoint_path,
            ticker=ticker,
            symbol=symbol,
            ticker_count=ticker_count,
            feature_row_count=feature_row_count,
            start_time=start_time,
        )
        if args.progress_every and ticker_count % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "step": "chunked_features_progress",
                        "tickers_processed": ticker_count,
                        "stock_feature_rows": feature_row_count,
                        "elapsed_seconds": round(time.monotonic() - start_time, 1),
                    }
                ),
                flush=True,
            )

    context_rows = load_daily_bars_csv(context_bars_path) if context_bars_path.exists() else []
    if resume_after_ticker and skipping_completed:
        raise SystemExit(f"Resume ticker was not found in raw stock bars: {resume_after_ticker}")
    context_features = compute_daily_features(context_rows)
    if context_features:
        write_csv(context_features_path, context_features)

    write_json(
        manifest_path,
        {
            "dataset_root": str(dataset_root.resolve()),
            "mode": "chunked_daily_features",
            "stock_tickers_processed_this_run": ticker_count,
            "stock_feature_rows_this_run": feature_row_count,
            "context_feature_rows": len(context_features),
            "fundamentals_enabled": not args.skip_fundamentals,
            "sentiment_enabled": not args.skip_sentiment,
            "resumed_after_ticker": resume_after_ticker or None,
            "normalized_features_written": False,
            "elapsed_seconds": round(time.monotonic() - start_time, 1),
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
            "stock_tickers_processed_this_run": ticker_count,
            "stock_feature_rows_this_run": feature_row_count,
            "context_feature_rows": len(context_features),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
