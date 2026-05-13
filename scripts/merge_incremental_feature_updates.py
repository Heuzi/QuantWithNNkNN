from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import shutil
from pathlib import Path
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.normalization import (  # noqa: E402
    _BOOLEAN_FIELDS,
    _CATEGORICAL_FIELDS,
    compute_normalized_feature_rows,
    load_equity_metadata,
)


STOCK_UPDATE_FILENAME = "daily_features_incremental_updates.csv"
CONTEXT_UPDATE_FILENAME = "market_context_features_incremental_updates.csv"
MERGE_MANIFEST_FILENAME = "incremental_feature_merge_manifest.json"
DAILY_FEATURES_FILENAME = "daily_features.csv"
NORMALIZED_FEATURES_FILENAME = "daily_features_normalized.csv"
NORMALIZED_MANIFEST_FILENAME = "daily_features_normalized_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge latest-inference processed feature sidecars into the main processed "
            "feature CSVs before a retrain."
        )
    )
    parser.add_argument("--dataset-root", default="data/eodhd_us_equities_30y")
    parser.add_argument(
        "--archive-updates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move consumed sidecar files into processed/incremental_feature_updates_archive.",
    )
    parser.add_argument(
        "--update-normalized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When an existing full-panel normalized artifact is present, renormalize only "
            "the dates touched by daily-feature updates and merge them into that artifact."
        ),
    )
    return parser.parse_args()


def _key(row: dict[str, str]) -> tuple[str, str]:
    return (str(row.get("ticker") or "").upper(), str(row.get("date") or "")[:10])


def _ordered_fieldnames(base_path: Path, update_path: Path) -> list[str]:
    fieldnames: list[str] = []
    for path in (base_path, update_path):
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for column in reader.fieldnames or []:
                if column not in fieldnames:
                    fieldnames.append(column)
    return fieldnames


def _load_updates_by_ticker(path: Path) -> tuple[dict[str, dict[tuple[str, str], dict[str, str]]], int, set[str]]:
    updates: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    row_count = 0
    affected_dates: set[str] = set()
    if not path.exists() or path.stat().st_size == 0:
        return updates, row_count, affected_dates
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ticker, date_value = _key(row)
            if not ticker or not date_value:
                continue
            normalized = dict(row)
            normalized["ticker"] = ticker
            normalized["date"] = date_value
            updates.setdefault(ticker, {})[(ticker, date_value)] = normalized
            row_count += 1
            affected_dates.add(date_value)
    return updates, row_count, affected_dates


def _merged_group_rows(
    base_rows: Iterable[dict[str, str]],
    update_rows_by_key: dict[tuple[str, str], dict[str, str]] | None,
) -> list[dict[str, str]]:
    merged: dict[tuple[str, str], dict[str, str]] = {}
    for row in base_rows:
        key = _key(row)
        if not key[0] or not key[1]:
            continue
        normalized = dict(row)
        normalized["ticker"] = key[0]
        normalized["date"] = key[1]
        merged[key] = normalized
    for key, row in (update_rows_by_key or {}).items():
        merged[key] = dict(row)
    return [merged[key] for key in sorted(merged)]


def _archive_update_file(path: Path, archive_dir: Path) -> str | None:
    if not path.exists():
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    target = archive_dir / f"{path.stem}_{timestamp}{path.suffix}"
    shutil.move(str(path), str(target))
    return str(target.resolve())


def _merge_one(
    *,
    base_path: Path,
    update_path: Path,
    archive_dir: Path,
    archive_updates: bool,
) -> dict[str, object]:
    if not update_path.exists() or update_path.stat().st_size == 0:
        return {
            "base_file": str(base_path.resolve()),
            "update_file": str(update_path.resolve()),
            "updated": False,
            "reason": "no update sidecar",
        }
    fieldnames = _ordered_fieldnames(base_path, update_path)
    if not fieldnames:
        return {
            "base_file": str(base_path.resolve()),
            "update_file": str(update_path.resolve()),
            "updated": False,
            "reason": "no CSV header",
        }

    updates_by_ticker, update_rows_seen, affected_dates = _load_updates_by_ticker(update_path)
    if not updates_by_ticker:
        archived = _archive_update_file(update_path, archive_dir) if archive_updates else None
        return {
            "base_file": str(base_path.resolve()),
            "update_file": str(update_path.resolve()),
            "updated": False,
            "reason": "empty update rows",
            "archived_update_file": archived,
        }

    temp_path = base_path.with_suffix(base_path.suffix + ".tmp")
    written_rows = 0
    replaced_or_added_rows = 0
    base_tickers_seen: set[str] = set()
    base_path.parent.mkdir(parents=True, exist_ok=True)

    with temp_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        if base_path.exists() and base_path.stat().st_size > 0:
            with base_path.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                current_ticker = ""
                group: list[dict[str, str]] = []

                def flush_group() -> None:
                    nonlocal group, current_ticker, written_rows, replaced_or_added_rows
                    if not group:
                        return
                    update_rows = updates_by_ticker.pop(current_ticker, None)
                    merged_rows = _merged_group_rows(group, update_rows)
                    writer.writerows(merged_rows)
                    written_rows += len(merged_rows)
                    if update_rows:
                        replaced_or_added_rows += len(update_rows)
                    base_tickers_seen.add(current_ticker)
                    group = []

                for row in reader:
                    ticker = str(row.get("ticker") or "").upper()
                    if group and ticker != current_ticker:
                        flush_group()
                    current_ticker = ticker
                    group.append(row)
                flush_group()

        for ticker in sorted(updates_by_ticker):
            merged_rows = _merged_group_rows([], updates_by_ticker[ticker])
            writer.writerows(merged_rows)
            written_rows += len(merged_rows)
            replaced_or_added_rows += len(merged_rows)

    temp_path.replace(base_path)
    archived = _archive_update_file(update_path, archive_dir) if archive_updates else None
    return {
        "base_file": str(base_path.resolve()),
        "update_file": str(update_path.resolve()),
        "updated": True,
        "update_rows_seen": update_rows_seen,
        "update_rows_merged": replaced_or_added_rows,
        "output_rows": written_rows,
        "base_tickers_seen": len(base_tickers_seen),
        "affected_dates": sorted(affected_dates),
        "archived_update_file": archived,
    }


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


def _collect_processed_rows_for_dates(path: Path, dates: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not dates or not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            trade_date = str(row.get("date") or "")[:10]
            if trade_date in dates:
                normalized = dict(row)
                normalized["date"] = trade_date
                normalized["ticker"] = str(normalized.get("ticker") or "").upper()
                rows.append(_parse_processed_row(normalized))
    return rows


def _write_update_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for column in row:
            if column not in fieldnames:
                fieldnames.append(column)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _utc_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.utcfromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") + "Z"


def _update_normalized_for_dates(
    *,
    dataset_root: Path,
    affected_dates: set[str],
    archive_dir: Path,
) -> dict[str, object]:
    processed_dir = dataset_root / "processed"
    source_path = processed_dir / DAILY_FEATURES_FILENAME
    normalized_path = processed_dir / NORMALIZED_FEATURES_FILENAME
    manifest_path = processed_dir / NORMALIZED_MANIFEST_FILENAME
    sector_path = dataset_root / "raw" / "eodhd_equity_metadata.csv"

    if not affected_dates:
        return {"updated": False, "reason": "no affected stock-feature dates"}
    if not normalized_path.exists() or not manifest_path.exists():
        return {
            "updated": False,
            "reason": "normalized artifact missing; full normalization stage must build it",
            "affected_date_count": len(affected_dates),
        }
    if not source_path.exists():
        return {"updated": False, "reason": f"missing source file: {source_path}"}
    if not sector_path.exists():
        return {"updated": False, "reason": f"missing sector metadata file: {sector_path}"}

    processed_rows = _collect_processed_rows_for_dates(source_path, affected_dates)
    if not processed_rows:
        return {
            "updated": False,
            "reason": "no source rows found for affected dates",
            "affected_date_count": len(affected_dates),
        }

    sector_metadata = load_equity_metadata(sector_path)
    normalized_rows = compute_normalized_feature_rows(processed_rows, sector_metadata)
    temp_update_path = processed_dir / ".daily_features_normalized_incremental_update.csv"
    _write_update_rows(temp_update_path, normalized_rows)
    try:
        merge_result = _merge_one(
            base_path=normalized_path,
            update_path=temp_update_path,
            archive_dir=archive_dir,
            archive_updates=False,
        )
    finally:
        if temp_update_path.exists():
            temp_update_path.unlink()

    manifest: dict[str, object] = {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}

    update_record = {
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "mode": "incremental_same_date_update",
        "affected_date_count": len(affected_dates),
        "affected_date_range": {
            "min_date": min(affected_dates),
            "max_date": max(affected_dates),
        },
        "source_file": str(source_path.resolve()),
        "source_file_mtime_utc": _utc_mtime(source_path),
        "rows_renormalized": len(normalized_rows),
        "merge_result": merge_result,
    }
    prior_updates = list(manifest.get("incremental_normalization_updates") or [])
    prior_updates.append(update_record)
    manifest["incremental_normalization_updates"] = prior_updates[-20:]
    manifest["row_count"] = merge_result.get("output_rows", manifest.get("row_count"))
    manifest["last_incremental_update_utc"] = update_record["generated_utc"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "updated": bool(merge_result.get("updated")),
        "affected_date_count": len(affected_dates),
        "rows_renormalized": len(normalized_rows),
        "normalized_file": str(normalized_path.resolve()),
        "manifest_file": str(manifest_path.resolve()),
        "merge_result": merge_result,
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    processed_dir = dataset_root / "processed"
    archive_dir = processed_dir / "incremental_feature_updates_archive"
    stock_result = _merge_one(
        base_path=processed_dir / DAILY_FEATURES_FILENAME,
        update_path=processed_dir / STOCK_UPDATE_FILENAME,
        archive_dir=archive_dir,
        archive_updates=bool(args.archive_updates),
    )
    affected_dates = set(str(value) for value in stock_result.get("affected_dates", []) if value)
    normalized_result = (
        _update_normalized_for_dates(
            dataset_root=dataset_root,
            affected_dates=affected_dates,
            archive_dir=archive_dir,
        )
        if bool(args.update_normalized)
        else {"updated": False, "reason": "disabled by --no-update-normalized"}
    )
    results = {
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset_root": str(dataset_root.resolve()),
        "stock_features": stock_result,
        "normalized_features": normalized_result,
        "context_features": _merge_one(
            base_path=processed_dir / "market_context_features.csv",
            update_path=processed_dir / CONTEXT_UPDATE_FILENAME,
            archive_dir=archive_dir,
            archive_updates=bool(args.archive_updates),
        ),
    }
    manifest_path = processed_dir / MERGE_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
