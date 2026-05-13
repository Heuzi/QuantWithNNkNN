from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from urllib.parse import quote
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.massive_stage1 import write_csv, write_json  # noqa: E402
from src.data.normalization import (  # noqa: E402
    _BOOLEAN_FIELDS,
    _CATEGORICAL_FIELDS,
    build_normalized_manifest,
    compute_normalized_feature_rows,
    load_equity_metadata,
    load_processed_feature_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PIT-safe same-date normalized daily features from a processed daily feature panel."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/eodhd_us_equities_30y",
        help="Dataset folder containing processed/daily_features.csv and raw/eodhd_equity_metadata.csv.",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Use the original in-memory implementation. Intended only for small smoke datasets.",
    )
    parser.add_argument(
        "--bucket-dir",
        default=None,
        help=(
            "Temporary month-bucket folder. Defaults to "
            "<dataset-root>/processed/.daily_features_normalized_month_buckets."
        ),
    )
    parser.add_argument(
        "--bucket-max-open-files",
        type=int,
        default=64,
        help="Maximum month bucket files to keep open while streaming the processed input.",
    )
    parser.add_argument(
        "--progress-rows",
        type=int,
        default=1_000_000,
        help="Print progress after this many streamed input rows. Use 0 to disable row progress.",
    )
    parser.add_argument(
        "--keep-buckets",
        action="store_true",
        help="Keep temporary month buckets after a successful run for debugging.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted streaming normalization from existing month/ticker buckets. "
            "Partial ticker rows at or after the first unfinished month are removed before continuing."
        ),
    )
    return parser.parse_args()


def _parse_processed_row(row: dict[str, str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in row.items():
        if value == "":
            parsed[key] = None
        elif key in _BOOLEAN_FIELDS:
            parsed[key] = value.lower() == "true"
        elif key in _CATEGORICAL_FIELDS:
            parsed[key] = value
        else:
            try:
                parsed[key] = float(value)
            except ValueError:
                parsed[key] = value
    return parsed


def _month_key(row: dict[str, str]) -> str:
    trade_date = row.get("date") or "unknown"
    if len(trade_date) >= 7 and trade_date[4:5] == "-" and trade_date[7:8] == "-":
        return trade_date[:7]
    return "unknown"


def _bucket_month_start(bucket_path: Path) -> str | None:
    stem = bucket_path.stem
    if len(stem) == 7 and stem[4:5] == "-":
        return f"{stem}-01"
    return None


def _bucket_file_key(value: object) -> str:
    text = str(value or "unknown").upper()
    return quote(text, safe="._-") or "unknown"


def _remove_tree_inside(*, base: Path, target: Path) -> None:
    base_resolved = base.resolve()
    target_resolved = target.resolve()
    if target_resolved == base_resolved or base_resolved not in target_resolved.parents:
        raise RuntimeError(f"Refusing to remove temporary directory outside {base_resolved}: {target_resolved}")
    if target_resolved.exists():
        shutil.rmtree(target_resolved)


def _write_checkpoint(
    path: Path,
    *,
    input_path: Path,
    bucket_dir: Path,
    ticker_bucket_dir: Path,
    completed_bucket: str | None,
    in_progress_bucket: str | None,
    row_count_this_process: int,
) -> None:
    checkpoint = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "daily_features_normalized_streaming_checkpoint",
        "input_file": str(input_path.resolve()),
        "bucket_dir": str(bucket_dir.resolve()),
        "ticker_bucket_dir": str(ticker_bucket_dir.resolve()),
        "completed_bucket": completed_bucket,
        "in_progress_bucket": in_progress_bucket,
        "row_count_this_process": int(row_count_this_process),
        "resume_rule": (
            "On resume, keep ticker-bucket rows before the first remaining month bucket, "
            "drop rows at or after that month, and continue from the remaining month buckets."
        ),
    }
    path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_ticker_buckets_from_date(
    *,
    ticker_bucket_dir: Path,
    cutoff_date: str,
    progress_every: int = 1_000,
) -> dict[str, int]:
    files = sorted(ticker_bucket_dir.glob("*.csv"))
    removed_rows = 0
    kept_rows = 0
    rewritten_files = 0
    started = time.monotonic()
    for index, path in enumerate(files, start=1):
        if path.stat().st_size == 0:
            path.unlink()
            rewritten_files += 1
            continue
        temp_path = path.with_suffix(path.suffix + ".resume_tmp")
        file_removed = 0
        file_kept = 0
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            fieldnames = list(reader.fieldnames or [])
            with temp_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
                if fieldnames:
                    writer.writeheader()
                for row in reader:
                    trade_date = str(row.get("date") or "")[:10]
                    if trade_date and trade_date >= cutoff_date:
                        file_removed += 1
                        continue
                    writer.writerow(row)
                    file_kept += 1
        temp_path.replace(path)
        if file_removed:
            rewritten_files += 1
        removed_rows += file_removed
        kept_rows += file_kept
        if index % progress_every == 0 or index == len(files):
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"Resume cleanup {index:,}/{len(files):,} ticker buckets "
                f"({removed_rows:,} rows removed; {kept_rows:,} rows kept; {elapsed:,.0f}s)",
                flush=True,
            )
    return {
        "files_seen": len(files),
        "files_rewritten_or_removed": rewritten_files,
        "rows_removed": removed_rows,
        "rows_kept": kept_rows,
    }


def _close_bucket(handles: dict[str, object], writers: dict[str, csv.DictWriter], key: str) -> None:
    handle = handles.pop(key, None)
    writers.pop(key, None)
    if handle is not None:
        handle.close()


def _get_bucket_writer(
    *,
    key: str,
    bucket_dir: Path,
    fieldnames: list[str],
    handles: dict[str, object],
    writers: dict[str, csv.DictWriter],
    max_open_files: int,
) -> csv.DictWriter:
    writer = writers.get(key)
    if writer is not None:
        return writer

    if max_open_files > 0 and len(handles) >= max_open_files:
        oldest_key = next(iter(handles))
        _close_bucket(handles, writers, oldest_key)

    path = bucket_dir / f"{key}.csv"
    write_header = not path.exists() or path.stat().st_size == 0
    handle = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    if write_header:
        writer.writeheader()
    handles[key] = handle
    writers[key] = writer
    return writer


def _bucket_processed_rows(
    *,
    input_path: Path,
    bucket_dir: Path,
    max_open_files: int,
    progress_rows: int,
) -> tuple[list[Path], int]:
    handles: dict[str, object] = {}
    writers: dict[str, csv.DictWriter] = {}
    row_count = 0
    started = time.monotonic()

    try:
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise RuntimeError(f"Input CSV has no header: {input_path}")
            fieldnames = list(reader.fieldnames)
            for row in reader:
                key = _month_key(row)
                writer = _get_bucket_writer(
                    key=key,
                    bucket_dir=bucket_dir,
                    fieldnames=fieldnames,
                    handles=handles,
                    writers=writers,
                    max_open_files=max_open_files,
                )
                writer.writerow(row)
                row_count += 1
                if progress_rows > 0 and row_count % progress_rows == 0:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    rate = row_count / elapsed
                    print(f"Bucketed {row_count:,} rows ({rate:,.0f} rows/sec)", flush=True)
    finally:
        for key in list(handles):
            _close_bucket(handles, writers, key)

    bucket_paths = sorted(bucket_dir.glob("*.csv"))
    return bucket_paths, row_count


def _load_bucket_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(_parse_processed_row(row))
    return rows


def _write_normalized_streaming(
    *,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    sector_path: Path,
    bucket_dir: Path,
    max_open_files: int,
    progress_rows: int,
    keep_buckets: bool,
    resume: bool,
) -> None:
    processed_dir = output_path.parent
    ticker_bucket_dir = processed_dir / ".daily_features_normalized_ticker_buckets"
    checkpoint_path = processed_dir / ".daily_features_normalized_checkpoint.json"
    resume_used = False
    resume_cleanup: dict[str, int] | None = None

    if resume and ticker_bucket_dir.exists() and any(ticker_bucket_dir.glob("*.csv")):
        resume_used = True
        print(f"Resuming normalization from existing buckets in {bucket_dir}", flush=True)
    else:
        _remove_tree_inside(base=processed_dir, target=bucket_dir)
        _remove_tree_inside(base=processed_dir, target=ticker_bucket_dir)
    bucket_dir.mkdir(parents=True, exist_ok=True)
    ticker_bucket_dir.mkdir(parents=True, exist_ok=True)

    tmp_output_path = output_path.with_name(f"{output_path.name}.tmp")
    tmp_manifest_path = manifest_path.with_name(f"{manifest_path.name}.tmp")
    for stale_path in (tmp_output_path, tmp_manifest_path, manifest_path):
        if stale_path.exists():
            stale_path.unlink()
    if output_path.exists() and not resume_used:
        output_path.unlink()

    if resume_used:
        bucket_paths = sorted(bucket_dir.glob("*.csv"))
        input_row_count: int | None = None
        first_unfinished_date = _bucket_month_start(bucket_paths[0]) if bucket_paths else None
        if first_unfinished_date:
            print(
                f"Cleaning partial ticker-bucket rows at or after {first_unfinished_date} before resume",
                flush=True,
            )
            resume_cleanup = _clean_ticker_buckets_from_date(
                ticker_bucket_dir=ticker_bucket_dir,
                cutoff_date=first_unfinished_date,
            )
            print(f"Resume cleanup summary: {resume_cleanup}", flush=True)
    else:
        print(f"Month-bucketing processed rows from {input_path}", flush=True)
        bucket_paths, input_row_count = _bucket_processed_rows(
            input_path=input_path,
            bucket_dir=bucket_dir,
            max_open_files=max_open_files,
            progress_rows=progress_rows,
        )
        print(f"Created {len(bucket_paths):,} month buckets for {input_row_count:,} rows", flush=True)

    sector_metadata = load_equity_metadata(sector_path)
    row_count = 0
    normalized_fieldnames: list[str] | None = None
    ticker_handles: dict[str, object] = {}
    ticker_writers: dict[str, csv.DictWriter] = {}

    try:
        for index, bucket_path in enumerate(bucket_paths, start=1):
            _write_checkpoint(
                checkpoint_path,
                input_path=input_path,
                bucket_dir=bucket_dir,
                ticker_bucket_dir=ticker_bucket_dir,
                completed_bucket=None,
                in_progress_bucket=bucket_path.stem,
                row_count_this_process=row_count,
            )
            month_rows = _load_bucket_rows(bucket_path)
            normalized_rows = compute_normalized_feature_rows(month_rows, sector_metadata)
            if normalized_rows and normalized_fieldnames is None:
                normalized_fieldnames = list(normalized_rows[0].keys())

            for row in normalized_rows:
                ticker = str(row.get("ticker") or "").upper()
                ticker_for_bucket = ticker or "UNKNOWN"
                writer = _get_bucket_writer(
                    key=_bucket_file_key(ticker_for_bucket),
                    bucket_dir=ticker_bucket_dir,
                    fieldnames=normalized_fieldnames or list(row.keys()),
                    handles=ticker_handles,
                    writers=ticker_writers,
                    max_open_files=max_open_files,
                )
                writer.writerow(row)

            row_count += len(normalized_rows)
            print(
                f"Normalized bucket {index:,}/{len(bucket_paths):,}: {bucket_path.stem} "
                f"({len(normalized_rows):,} rows; total {row_count:,})",
                flush=True,
            )
            if not keep_buckets:
                bucket_path.unlink()
            _write_checkpoint(
                checkpoint_path,
                input_path=input_path,
                bucket_dir=bucket_dir,
                ticker_bucket_dir=ticker_bucket_dir,
                completed_bucket=bucket_path.stem,
                in_progress_bucket=None,
                row_count_this_process=row_count,
            )
    finally:
        for key in list(ticker_handles):
            _close_bucket(ticker_handles, ticker_writers, key)

    if input_row_count is not None and row_count != input_row_count:
        raise RuntimeError(f"Normalized row count mismatch: input={input_row_count:,} output={row_count:,}")

    output_rows = 0
    tickers: set[str] = set()
    min_date: str | None = None
    max_date: str | None = None
    ticker_bucket_paths = sorted(ticker_bucket_dir.glob("*.csv"))
    if normalized_fieldnames is None:
        for ticker_bucket_path in ticker_bucket_paths:
            if ticker_bucket_path.stat().st_size == 0:
                continue
            with ticker_bucket_path.open("r", encoding="utf-8", newline="") as ticker_handle:
                reader = csv.DictReader(ticker_handle)
                normalized_fieldnames = list(reader.fieldnames or [])
                break
    if normalized_fieldnames is None:
        normalized_fieldnames = []
    with tmp_output_path.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=normalized_fieldnames)
        if normalized_fieldnames:
            writer.writeheader()
        for index, ticker_bucket_path in enumerate(ticker_bucket_paths, start=1):
            ticker = ticker_bucket_path.stem
            with ticker_bucket_path.open("r", encoding="utf-8", newline="") as ticker_handle:
                reader = csv.DictReader(ticker_handle)
                for row in reader:
                    trade_date = str(row.get("date") or "")[:10]
                    ticker_value = str(row.get("ticker") or ticker).upper()
                    if trade_date:
                        min_date = trade_date if min_date is None else min(min_date, trade_date)
                        max_date = trade_date if max_date is None else max(max_date, trade_date)
                    if ticker_value:
                        tickers.add(ticker_value)
                    writer.writerow(row)
                    output_rows += 1
            if not keep_buckets:
                ticker_bucket_path.unlink()
            if index % 1_000 == 0 or index == len(ticker_bucket_paths):
                print(
                    f"Wrote ticker bucket {index:,}/{len(ticker_bucket_paths):,} "
                    f"({output_rows:,} output rows)",
                    flush=True,
                )
    if not resume_used and output_rows != row_count:
        raise RuntimeError(f"Final normalized row count mismatch: normalized={row_count:,} output={output_rows:,}")

    manifest = build_normalized_manifest(
        input_file=input_path,
        sector_source_file=sector_path,
        row_count=output_rows,
        universe_count=len(tickers),
        min_date=min_date,
        max_date=max_date,
    )
    manifest["normalization_execution"] = {
        "mode": "month_bucket_streaming_resume" if resume_used else "month_bucket_streaming",
        "bucket_count": len(bucket_paths),
        "max_open_bucket_files": max_open_files,
        "resume_used": resume_used,
        "resume_cleanup": resume_cleanup,
        "reason": "Keeps same-date cross-sectional normalization complete while avoiding full-panel RAM materialization.",
    }
    write_json(tmp_manifest_path, manifest)

    tmp_output_path.replace(output_path)
    tmp_manifest_path.replace(manifest_path)
    if not keep_buckets:
        _remove_tree_inside(base=processed_dir, target=bucket_dir)
        _remove_tree_inside(base=processed_dir, target=ticker_bucket_dir)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    print(f"Wrote {output_rows:,} normalized rows to {output_path}", flush=True)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    input_path = dataset_root / "processed" / "daily_features.csv"
    sector_path = dataset_root / "raw" / "eodhd_equity_metadata.csv"
    output_path = dataset_root / "processed" / "daily_features_normalized.csv"
    manifest_path = dataset_root / "processed" / "daily_features_normalized_manifest.json"
    bucket_dir = Path(args.bucket_dir) if args.bucket_dir else dataset_root / "processed" / ".daily_features_normalized_month_buckets"

    if not input_path.exists():
        raise SystemExit(f"Processed feature file not found: {input_path}")
    if not sector_path.exists():
        raise SystemExit(f"Sector metadata file not found: {sector_path}")

    if not args.in_memory:
        _write_normalized_streaming(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
            sector_path=sector_path,
            bucket_dir=bucket_dir,
            max_open_files=max(args.bucket_max_open_files, 1),
            progress_rows=args.progress_rows,
            keep_buckets=args.keep_buckets,
            resume=args.resume,
        )
        return

    feature_rows = load_processed_feature_rows(input_path)
    sector_metadata = load_equity_metadata(sector_path)
    normalized_rows = compute_normalized_feature_rows(feature_rows, sector_metadata)

    trade_dates = [str(row["date"]) for row in normalized_rows if row.get("date")]
    tickers = {str(row["ticker"]) for row in normalized_rows if row.get("ticker")}

    write_csv(output_path, normalized_rows)
    write_json(
        manifest_path,
        build_normalized_manifest(
            input_file=input_path,
            sector_source_file=sector_path,
            row_count=len(normalized_rows),
            universe_count=len(tickers),
            min_date=min(trade_dates) if trade_dates else None,
            max_date=max(trade_dates) if trade_dates else None,
        ),
    )
    print(f"Wrote {len(normalized_rows)} normalized rows to {output_path}")


if __name__ == "__main__":
    main()
