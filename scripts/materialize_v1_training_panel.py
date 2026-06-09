from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Iterable, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.episode_eligibility import parse_allowed_exchanges  # noqa: E402
from src.data.research_universe import (  # noqa: E402
    ConservativeResearchUniverseConfig,
    add_conservative_research_universe_columns,
)
from src.data.v1_dataset import preferred_stock_feature_path  # noqa: E402


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

RESEARCH_NUMERIC_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dollar_volume",
    "return_1d",
    "true_range_pct",
    "rolling_avg_dollar_volume_20d",
    "rolling_avg_dollar_volume_60d",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a trainable V1 dataset root from the full EODHD processed panel "
            "without loading the 30-year all-stock CSV into memory."
        )
    )
    parser.add_argument("--source-dataset-root", default="data/eodhd_us_equities_30y")
    parser.add_argument("--output-dataset-root", required=True)
    parser.add_argument("--start-date", default="", help="Inclusive YYYY-MM-DD stock/context date filter.")
    parser.add_argument("--end-date", default="", help="Inclusive YYYY-MM-DD stock/context date filter.")
    parser.add_argument("--tickers", default="", help="Comma-separated ticker allowlist.")
    parser.add_argument("--tickers-file", default="", help="Optional one-ticker-per-line allowlist.")
    parser.add_argument("--max-tickers", type=int, default=0, help="0 keeps every ticker passing other filters.")
    parser.add_argument(
        "--ticker-selection",
        choices=("first", "latest_dollar_volume"),
        default="first",
        help="How to choose max-tickers when no explicit allowlist is supplied.",
    )
    parser.add_argument(
        "--min-latest-dollar-volume",
        type=float,
        default=0.0,
        help="Optional latest-liquidity floor used by ticker-selection=latest_dollar_volume.",
    )
    parser.add_argument(
        "--min-latest-price",
        type=float,
        default=0.0,
        help="Optional latest-price floor used by ticker-selection=latest_dollar_volume.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing materialized processed outputs.")
    parser.add_argument("--progress-every", type=int, default=1_000_000, help="Print progress by written rows.")
    parser.add_argument(
        "--disable-conservative-research-universe",
        action="store_true",
        help="Disable the shared strategy-universe filter while materializing the bounded training panel.",
    )
    parser.add_argument("--research-universe-name", default="conservative")
    parser.add_argument("--research-common-stocks-only", action="store_true", default=True)
    parser.add_argument("--research-allowed-exchanges", default="NYSE,NASDAQ,AMEX")
    parser.add_argument("--research-min-price", type=float, default=10.0)
    parser.add_argument("--research-min-history-days", type=int, default=252)
    parser.add_argument("--research-min-median-dollar-volume-20d", type=float, default=10_000_000.0)
    parser.add_argument("--research-min-median-dollar-volume-60d", type=float, default=10_000_000.0)
    parser.add_argument("--research-max-zero-volume-day-ratio-60d", type=float, default=0.02)
    parser.add_argument("--research-min-current-dollar-volume-vs-median-20d", type=float, default=0.20)
    parser.add_argument("--research-liquidity-short-lookback-days", type=int, default=20)
    parser.add_argument("--research-liquidity-long-lookback-days", type=int, default=60)
    parser.add_argument("--research-trend-lookback-days", type=int, default=252)
    parser.add_argument("--research-return-6m-lookback-days", type=int, default=126)
    parser.add_argument("--research-sma-short-lookback-days", type=int, default=50)
    parser.add_argument("--research-sma-long-lookback-days", type=int, default=200)
    parser.add_argument("--research-min-return-6m", type=float, default=-0.15)
    parser.add_argument("--research-max-drawdown-from-252d-high-pct", type=float, default=35.0)
    parser.add_argument("--research-disable-close-above-sma200", action="store_true")
    parser.add_argument("--research-disable-sma50-above-sma200", action="store_true")
    parser.add_argument("--research-disable-spike-filter", action="store_true")
    parser.add_argument("--research-spike-lookback-days", type=int, default=60)
    parser.add_argument("--research-max-abs-return-1d-60d-pct", type=float, default=25.0)
    parser.add_argument("--research-max-true-range-60d-pct", type=float, default=25.0)
    return parser.parse_args()


def _date_ok(value: str, start_date: str, end_date: str) -> bool:
    return (not start_date or value >= start_date) and (not end_date or value <= end_date)


def _float_value(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_ticker_allowlist(raw: str, file_path: str) -> set[str]:
    tickers = {item.strip().upper() for item in raw.split(",") if item.strip()}
    if file_path:
        with Path(file_path).open("r", encoding="utf-8") as handle:
            tickers.update(line.strip().upper() for line in handle if line.strip())
    return tickers


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


def _research_universe_config(args: argparse.Namespace) -> ConservativeResearchUniverseConfig | None:
    if args.disable_conservative_research_universe:
        return None
    return ConservativeResearchUniverseConfig(
        name=args.research_universe_name,
        common_stocks_only=bool(args.research_common_stocks_only),
        allowed_exchanges=parse_allowed_exchanges(args.research_allowed_exchanges),
        min_price=args.research_min_price,
        min_history_days=args.research_min_history_days,
        liquidity_short_lookback=args.research_liquidity_short_lookback_days,
        liquidity_long_lookback=args.research_liquidity_long_lookback_days,
        min_median_dollar_volume_20d=args.research_min_median_dollar_volume_20d,
        min_median_dollar_volume_60d=args.research_min_median_dollar_volume_60d,
        max_zero_volume_day_ratio_60d=args.research_max_zero_volume_day_ratio_60d,
        min_current_dollar_volume_vs_median_20d=args.research_min_current_dollar_volume_vs_median_20d,
        trend_lookback_days=args.research_trend_lookback_days,
        return_6m_lookback_days=args.research_return_6m_lookback_days,
        sma_short_lookback_days=args.research_sma_short_lookback_days,
        sma_long_lookback_days=args.research_sma_long_lookback_days,
        min_return_6m=args.research_min_return_6m,
        max_drawdown_from_252d_high=args.research_max_drawdown_from_252d_high_pct / 100.0,
        require_close_above_sma200=not bool(args.research_disable_close_above_sma200),
        require_sma50_above_sma200=not bool(args.research_disable_sma50_above_sma200),
        spike_filter_enabled=not bool(args.research_disable_spike_filter),
        spike_lookback_days=args.research_spike_lookback_days,
        max_abs_return_1d_60d=args.research_max_abs_return_1d_60d_pct / 100.0,
        max_true_range_pct_60d=args.research_max_true_range_60d_pct / 100.0,
    )


def _stock_feature_path(dataset_root: Path) -> Path:
    return preferred_stock_feature_path(dataset_root)


def _select_first_tickers(
    path: Path,
    *,
    max_tickers: int,
    start_date: str,
    end_date: str,
) -> set[str]:
    selected: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ticker = str(row.get("ticker") or "").upper()
            if not ticker or ticker in seen:
                continue
            if not _date_ok(str(row.get("date") or ""), start_date, end_date):
                continue
            seen.add(ticker)
            selected.append(ticker)
            if max_tickers and len(selected) >= max_tickers:
                break
    return set(selected)


def _select_latest_liquid_tickers(
    path: Path,
    *,
    max_tickers: int,
    start_date: str,
    end_date: str,
    min_latest_dollar_volume: float,
    min_latest_price: float,
) -> set[str]:
    latest: dict[str, tuple[str, float, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            date_value = str(row.get("date") or "")
            if not _date_ok(date_value, start_date, end_date):
                continue
            ticker = str(row.get("ticker") or "").upper()
            if not ticker:
                continue
            dollar_volume = _float_value(row.get("rolling_avg_dollar_volume_60d") or row.get("dollar_volume"))
            close = _float_value(row.get("close"))
            if min_latest_dollar_volume and not (dollar_volume >= min_latest_dollar_volume):
                continue
            if min_latest_price and not (close >= min_latest_price):
                continue
            if ticker not in latest or date_value > latest[ticker][0]:
                latest[ticker] = (date_value, dollar_volume, close)
    ranked = sorted(latest.items(), key=lambda item: (item[1][1], item[1][0]), reverse=True)
    if max_tickers:
        ranked = ranked[:max_tickers]
    return {ticker for ticker, _ in ranked}


def _selected_tickers(args: argparse.Namespace, source_path: Path) -> set[str]:
    explicit = _load_ticker_allowlist(args.tickers, args.tickers_file)
    if explicit:
        return explicit
    if args.max_tickers <= 0:
        return set()
    if args.ticker_selection == "latest_dollar_volume":
        return _select_latest_liquid_tickers(
            source_path,
            max_tickers=args.max_tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            min_latest_dollar_volume=args.min_latest_dollar_volume,
            min_latest_price=args.min_latest_price,
        )
    return _select_first_tickers(
        source_path,
        max_tickers=args.max_tickers,
        start_date=args.start_date,
        end_date=args.end_date,
    )


def _fieldnames(source_fields: Sequence[str] | None) -> list[str]:
    fields = list(source_fields or [])
    for column in METADATA_COLUMNS:
        if column not in fields:
            fields.append(column)
    return fields


def _enriched_row(row: dict[str, str], metadata_by_ticker: dict[str, dict[str, str]]) -> dict[str, str]:
    out = dict(row)
    metadata = metadata_by_ticker.get(str(row.get("ticker") or "").upper(), {})
    for column in METADATA_COLUMNS:
        if out.get(column) in (None, ""):
            out[column] = str(metadata.get(column) or ("Unknown" if column in {"gics_sector", "gics_sub_industry"} else ""))
    return out


def _coerce_research_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in RESEARCH_NUMERIC_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _iter_selected_ticker_groups(
    path: Path,
    *,
    selected_tickers: set[str],
    start_date: str,
    end_date: str,
) -> Iterable[pd.DataFrame]:
    selected_max = max(selected_tickers) if selected_tickers else ""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        current_ticker = ""
        for row in reader:
            ticker = str(row.get("ticker") or "").upper()
            if not ticker:
                continue
            if selected_tickers:
                if selected_max and ticker > selected_max:
                    break
                if ticker not in selected_tickers:
                    continue
            date_value = str(row.get("date") or "")
            if not _date_ok(date_value, start_date, end_date):
                continue
            row["ticker"] = ticker
            if rows and ticker != current_ticker:
                yield pd.DataFrame(rows)
                rows = []
            current_ticker = ticker
            rows.append(row)
        if rows:
            yield pd.DataFrame(rows)


def _write_filtered_stock_features(
    *,
    source_path: Path,
    output_path: Path,
    metadata_by_ticker: dict[str, dict[str, str]],
    selected_tickers: set[str],
    start_date: str,
    end_date: str,
    progress_every: int,
    research_config: ConservativeResearchUniverseConfig | None,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_rows = 0
    tickers_written: set[str] = set()
    min_date = ""
    max_date = ""
    research_input_rows = 0
    research_passed_rows = 0
    start_time = time.monotonic()
    with source_path.open("r", encoding="utf-8", newline="") as source, output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(output, fieldnames=_fieldnames(reader.fieldnames))
        writer.writeheader()
        for group in _iter_selected_ticker_groups(
            source_path,
            selected_tickers=selected_tickers,
            start_date=start_date,
            end_date=end_date,
        ):
            if group.empty:
                continue
            rows = [
                _enriched_row({key: ("" if value is None else str(value)) for key, value in row.items()}, metadata_by_ticker)
                for row in group.to_dict("records")
            ]
            group_frame = pd.DataFrame(rows)
            if research_config is not None:
                diagnostics = add_conservative_research_universe_columns(
                    _coerce_research_numeric_columns(group_frame),
                    research_config,
                )
                research_input_rows += int(len(diagnostics))
                mask = diagnostics["research_universe_ok"].fillna(False).astype(bool)
                research_passed_rows += int(mask.sum())
                group_frame = diagnostics.loc[mask, writer.fieldnames].reset_index(drop=True)
            if group_frame.empty:
                continue
            for row in group_frame.to_dict("records"):
                date_value = str(row.get("date") or "")
                writer.writerow(row)
                written_rows += 1
                tickers_written.add(str(row.get("ticker") or "").upper())
                min_date = date_value if not min_date or date_value < min_date else min_date
                max_date = date_value if not max_date or date_value > max_date else max_date
            if progress_every and written_rows and written_rows % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "step": "materialize_stock_progress",
                            "rows_written": written_rows,
                            "tickers_written": len(tickers_written),
                            "elapsed_seconds": round(time.monotonic() - start_time, 1),
                        }
                    ),
                    flush=True,
                )
    return {
        "stock_rows": written_rows,
        "stock_tickers": len(tickers_written),
        "stock_min_date": min_date or None,
        "stock_max_date": max_date or None,
        "research_universe_input_rows": research_input_rows if research_config is not None else None,
        "research_universe_passed_rows": research_passed_rows if research_config is not None else None,
        "research_universe_removed_rows": (
            research_input_rows - research_passed_rows if research_config is not None else None
        ),
    }


def _copy_filtered_context(source_path: Path, output_path: Path, *, start_date: str, end_date: str) -> int:
    if not source_path.exists():
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with source_path.open("r", encoding="utf-8", newline="") as source, output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(output, fieldnames=list(reader.fieldnames or []))
        writer.writeheader()
        for row in reader:
            if not _date_ok(str(row.get("date") or ""), start_date, end_date):
                continue
            writer.writerow(row)
            row_count += 1
    return row_count


def _copy_optional_files(source_root: Path, output_root: Path, paths: Iterable[tuple[str, str]]) -> None:
    for source_rel, output_rel in paths:
        source = source_root / source_rel
        output = output_root / output_rel
        if not source.exists():
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_dataset_root)
    output_root = Path(args.output_dataset_root)
    if source_root.resolve() == output_root.resolve():
        raise SystemExit("Output dataset root must be different from source dataset root.")

    source_stock_path = _stock_feature_path(source_root)
    source_context_path = source_root / "processed" / "market_context_features.csv"
    source_metadata_path = source_root / "raw" / "eodhd_equity_metadata.csv"
    output_stock_path = output_root / "processed" / "daily_features.csv"
    output_context_path = output_root / "processed" / "market_context_features.csv"
    manifest_path = output_root / "processed" / "materialized_panel_manifest.json"

    if not source_stock_path.exists():
        raise SystemExit(f"Missing source stock features: {source_stock_path}")
    if output_stock_path.exists() and not args.force:
        raise SystemExit(f"Materialized panel already exists. Use --force to rebuild: {output_stock_path}")
    if args.force:
        for path in (output_stock_path, output_context_path, manifest_path):
            if path.exists():
                path.unlink()

    metadata_by_ticker = _load_metadata(source_metadata_path)
    selected_tickers = _selected_tickers(args, source_stock_path)
    research_config = _research_universe_config(args)
    start_time = time.monotonic()
    stats = _write_filtered_stock_features(
        source_path=source_stock_path,
        output_path=output_stock_path,
        metadata_by_ticker=metadata_by_ticker,
        selected_tickers=selected_tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        progress_every=args.progress_every,
        research_config=research_config,
    )
    context_rows = _copy_filtered_context(
        source_context_path,
        output_context_path,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    _copy_optional_files(
        source_root,
        output_root,
        (
            ("raw/eodhd_equity_metadata.csv", "raw/eodhd_equity_metadata.csv"),
            ("raw/eodhd_common_stock_universe.csv", "raw/eodhd_common_stock_universe.csv"),
            ("raw/eodhd_fetch_manifest.json", "raw/eodhd_fetch_manifest.json"),
        ),
    )

    manifest = {
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "mode": "materialized_v1_training_panel",
        "source_dataset_root": str(source_root.resolve()),
        "output_dataset_root": str(output_root.resolve()),
        "source_stock_features": str(source_stock_path.resolve()),
        "start_date": args.start_date or None,
        "end_date": args.end_date or None,
        "explicit_ticker_count": len(_load_ticker_allowlist(args.tickers, args.tickers_file)),
        "selected_ticker_count": len(selected_tickers) if selected_tickers else None,
        "ticker_selection": args.ticker_selection,
        "max_tickers": args.max_tickers or None,
        "context_rows": context_rows,
        "elapsed_seconds": round(time.monotonic() - start_time, 1),
        "research_universe": research_config.to_dict() if research_config is not None else {"enabled": False},
        "notes": [
            "This output is a normal dataset root for scripts/train_v1_supervised_baselines.py.",
            "Metadata columns are joined from raw/eodhd_equity_metadata.csv so static encoders do not depend on ticker identifiers.",
            "Use this stage to keep training commands stable while avoiding direct pandas loads of the full 30-year CSV.",
            "When enabled, the conservative research universe is applied while writing the materialized stock panel.",
        ],
        **stats,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"step": "materialized_panel_complete", **manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
