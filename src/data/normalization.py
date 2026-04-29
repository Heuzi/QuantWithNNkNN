from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Sequence


LOG1P_BASE_FEATURES = [
    "volume",
    "dollar_volume",
    "rolling_avg_volume_20d",
    "rolling_avg_volume_60d",
    "rolling_avg_dollar_volume_20d",
    "rolling_avg_dollar_volume_60d",
]

FULL_PANEL_NORMALIZATION_FEATURES = [
    "log1p_volume",
    "log1p_dollar_volume",
    "log1p_rolling_avg_volume_20d",
    "log1p_rolling_avg_volume_60d",
    "log1p_rolling_avg_dollar_volume_20d",
    "log1p_rolling_avg_dollar_volume_60d",
    "return_1d",
    "gap_pct",
    "intraday_return",
    "hl_range_pct",
    "rolling_return_5d",
    "rolling_return_20d",
    "rolling_return_60d",
    "rolling_vol_20d",
    "rolling_vol_60d",
    "price_vs_sma_20d",
    "price_vs_sma_60d",
    "momentum_20d",
    "momentum_60d",
    "volume_ratio_20d",
]

SECTOR_NORMALIZATION_FEATURES = [
    "log1p_volume",
    "log1p_dollar_volume",
    "log1p_rolling_avg_volume_20d",
    "log1p_rolling_avg_volume_60d",
    "rolling_vol_20d",
    "rolling_vol_60d",
    "rolling_return_20d",
    "rolling_return_60d",
    "momentum_20d",
    "momentum_60d",
    "volume_ratio_20d",
]

FULL_PANEL_MIN_GROUP_SIZE = 2
SECTOR_MIN_GROUP_SIZE = 5

_CATEGORICAL_FIELDS = {"date", "ticker", "gics_sector", "gics_sub_industry"}
_BOOLEAN_FIELDS = {"adjusted", "has_prev_close", "has_20d_history", "has_60d_history"}


def load_processed_feature_rows(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
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
            rows.append(parsed)
    return rows


def load_equity_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ticker = (row.get("symbol") or row.get("ticker") or row.get("code") or "").upper()
            if not ticker:
                continue
            metadata[ticker] = {
                "gics_sector": row.get("gics_sector") or row.get("sector") or "Unknown",
                "gics_sub_industry": row.get("gics_sub_industry") or row.get("industry") or "Unknown",
            }
    return metadata


def load_sp500_constituent_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    return load_equity_metadata(path)


def _safe_log1p(value: object) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if numeric < 0:
        return None
    return math.log1p(numeric)


def _percentile_ranks(values: Sequence[float]) -> list[float]:
    n = len(values)
    if n < 2:
        return [0.0 for _ in values]

    ranked = sorted(enumerate(values), key=lambda item: item[1])
    percentiles = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ranked[j + 1][1] == ranked[i][1]:
            j += 1
        avg_rank = ((i + 1) + (j + 1)) / 2.0
        percentile = (avg_rank - 1.0) / (n - 1.0)
        for k in range(i, j + 1):
            original_index = ranked[k][0]
            percentiles[original_index] = percentile
        i = j + 1
    return percentiles


def _initialize_normalized_row(row: dict[str, object]) -> dict[str, object]:
    normalized_row = dict(row)
    normalized_row["gics_sector"] = None
    normalized_row["gics_sub_industry"] = None
    normalized_row["cs_universe_count"] = None
    normalized_row["sector_universe_count"] = None

    for base_feature in LOG1P_BASE_FEATURES:
        normalized_row[f"log1p_{base_feature}"] = None

    for feature in FULL_PANEL_NORMALIZATION_FEATURES:
        normalized_row[f"{feature}__cs_z"] = None
        normalized_row[f"{feature}__cs_pct"] = None

    for feature in SECTOR_NORMALIZATION_FEATURES:
        normalized_row[f"{feature}__sector_cs_z"] = None
        normalized_row[f"{feature}__sector_cs_pct"] = None

    return normalized_row


def _update_min_count(row: dict[str, object], key: str, count: int) -> None:
    existing = row.get(key)
    if existing is None:
        row[key] = count
    else:
        row[key] = min(int(existing), count)


def _apply_group_transforms(
    rows: list[dict[str, object]],
    indices: Sequence[int],
    feature: str,
    z_column: str,
    pct_column: str,
    count_key: str,
    min_group_size: int,
) -> None:
    valid_items: list[tuple[int, float]] = []
    for idx in indices:
        value = rows[idx].get(feature)
        if value is None:
            continue
        valid_items.append((idx, float(value)))

    group_count = len(valid_items)
    if group_count == 0:
        return

    for idx, _ in valid_items:
        _update_min_count(rows[idx], count_key, group_count)

    if group_count < min_group_size:
        return

    values = [value for _, value in valid_items]
    mean_value = statistics.mean(values)
    std_value = statistics.pstdev(values) if group_count >= 2 else None
    percentiles = _percentile_ranks(values)

    for (idx, value), pct in zip(valid_items, percentiles):
        rows[idx][pct_column] = pct
        if std_value and std_value > 0:
            rows[idx][z_column] = (value - mean_value) / std_value


def compute_normalized_feature_rows(
    feature_rows: Sequence[dict[str, object]],
    sector_metadata_by_ticker: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    rows = [_initialize_normalized_row(row) for row in feature_rows]

    for row in rows:
        ticker = str(row["ticker"]).upper()
        metadata = sector_metadata_by_ticker.get(ticker, {})
        row["gics_sector"] = metadata.get("gics_sector")
        row["gics_sub_industry"] = metadata.get("gics_sub_industry")
        for base_feature in LOG1P_BASE_FEATURES:
            row[f"log1p_{base_feature}"] = _safe_log1p(row.get(base_feature))

    indices_by_date: dict[str, list[int]] = defaultdict(list)
    indices_by_date_sector: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        trade_date = str(row["date"])
        indices_by_date[trade_date].append(idx)
        sector = row.get("gics_sector")
        if sector:
            indices_by_date_sector[(trade_date, str(sector))].append(idx)

    for indices in indices_by_date.values():
        for feature in FULL_PANEL_NORMALIZATION_FEATURES:
            _apply_group_transforms(
                rows=rows,
                indices=indices,
                feature=feature,
                z_column=f"{feature}__cs_z",
                pct_column=f"{feature}__cs_pct",
                count_key="cs_universe_count",
                min_group_size=FULL_PANEL_MIN_GROUP_SIZE,
            )

    for indices in indices_by_date_sector.values():
        for feature in SECTOR_NORMALIZATION_FEATURES:
            _apply_group_transforms(
                rows=rows,
                indices=indices,
                feature=feature,
                z_column=f"{feature}__sector_cs_z",
                pct_column=f"{feature}__sector_cs_pct",
                count_key="sector_universe_count",
                min_group_size=SECTOR_MIN_GROUP_SIZE,
            )

    for row in rows:
        if row["cs_universe_count"] is not None:
            row["cs_universe_count"] = int(row["cs_universe_count"])
        if row["sector_universe_count"] is not None:
            row["sector_universe_count"] = int(row["sector_universe_count"])

    return rows


def build_normalized_manifest(
    *,
    input_file: str | Path,
    sector_source_file: str | Path,
    row_count: int,
    universe_count: int,
    min_date: str | None,
    max_date: str | None,
) -> dict[str, object]:
    return {
        "input_file": str(Path(input_file).resolve()),
        "sector_source_file": str(Path(sector_source_file).resolve()),
        "row_count": row_count,
        "universe_definition": "Equity panel present in the processed daily feature input on each date.",
        "universe_count": universe_count,
        "date_range": {
            "min_date": min_date,
            "max_date": max_date,
        },
        "timing_assumption": "End-of-day only. Same-date close/volume-based normalization is allowed because values are assumed known by the date close.",
        "same_date_only": True,
        "same_sector_only": True,
        "log1p_base_features": LOG1P_BASE_FEATURES,
        "full_panel_features": FULL_PANEL_NORMALIZATION_FEATURES,
        "sector_features": SECTOR_NORMALIZATION_FEATURES,
        "minimum_group_sizes": {
            "full_panel": FULL_PANEL_MIN_GROUP_SIZE,
            "sector": SECTOR_MIN_GROUP_SIZE,
        },
        "formulas": {
            "log1p_feature": "log1p(x)",
            "cross_sectional_zscore": "(x - same_date_mean) / same_date_std",
            "cross_sectional_percentile_rank": "(average_tie_rank - 1) / (n - 1)",
            "sector_cross_sectional_zscore": "(x - same_date_sector_mean) / same_date_sector_std",
            "sector_cross_sectional_percentile_rank": "(average_tie_rank - 1) / (n - 1)",
        },
        "coverage_columns": {
            "cs_universe_count": "Conservative row-level minimum of the non-null same-date full-panel group sizes across applicable transformed features.",
            "sector_universe_count": "Conservative row-level minimum of the non-null same-date same-sector group sizes across applicable sector-relative transformed features.",
        },
        "known_caveats": [
            "Normalization is point-in-time safe with respect to time because only same-date values are used.",
            "Sector metadata may be current vendor metadata, not a verified historical sector panel.",
            "Universe membership and ticker identity must be interpreted using the dataset manifest for the selected vendor.",
        ],
    }
