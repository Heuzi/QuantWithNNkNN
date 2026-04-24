from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.massive_stage1 import write_csv, write_json  # noqa: E402
from src.data.normalization import (  # noqa: E402
    build_normalized_manifest,
    compute_normalized_feature_rows,
    load_processed_feature_rows,
    load_sp500_constituent_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PIT-safe same-date normalized daily features from a processed daily feature panel."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset folder containing processed/daily_features.csv and raw/sp500_constituents_current.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    input_path = dataset_root / "processed" / "daily_features.csv"
    sector_path = dataset_root / "raw" / "sp500_constituents_current.csv"
    output_path = dataset_root / "processed" / "daily_features_normalized.csv"
    manifest_path = dataset_root / "processed" / "daily_features_normalized_manifest.json"

    if not input_path.exists():
        raise SystemExit(f"Processed feature file not found: {input_path}")
    if not sector_path.exists():
        raise SystemExit(f"Sector metadata file not found: {sector_path}")

    feature_rows = load_processed_feature_rows(input_path)
    sector_metadata = load_sp500_constituent_metadata(sector_path)
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
