from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import heapq
import json
import os
from pathlib import Path
import time
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from src.data.episode_eligibility import (
    ELIGIBILITY_DIAGNOSTIC_COLUMNS,
    EpisodeEligibilityConfig,
    add_episode_eligibility_columns,
    eligibility_metadata_columns,
)
from src.data.eodhd_enrichment import FUNDAMENTAL_FEATURE_COLUMNS, SENTIMENT_FEATURE_COLUMNS
from src.data.research_universe import (
    ConservativeResearchUniverseConfig,
    RESEARCH_UNIVERSE_DIAGNOSTIC_COLUMNS,
    add_conservative_research_universe_columns,
    research_universe_metadata_columns,
)
from src.data.v1_dataset import (
    BASE_STOCK_FEATURES,
    CONTEXT_FEATURES,
    DEFAULT_BENCHMARK_TICKER,
    DEFAULT_CLASSIFICATION_EVENT_TYPE,
    DEFAULT_CLASSIFICATION_HORIZON,
    DEFAULT_CLASSIFICATION_THRESHOLD,
    DEFAULT_HORIZONS,
    DEFAULT_WINDOW_LENGTH,
    MARKET_CONTEXT_TICKERS,
    PATH_5PCT_20D_EVENT_TYPE,
    PATH_5PCT_20D_NEGATIVE_THRESHOLD,
    SECTOR_ETF_BY_GICS,
    SEQUENCE_FEATURE_SET_NAMES,
    STATIC_CATEGORICAL_COLUMNS,
    V1Dataset,
    add_priority_a_ohlcv_features,
    add_window_summaries,
    classification_target_column,
    load_market_context_features,
    parse_horizons,
    pathwise_classification_labels,
    preferred_stock_feature_path,
    sequence_feature_config,
    target_column,
    validate_classification_event_config,
    validate_model_feature_columns,
)


_MEMMAP_CACHE: dict[tuple[int, str], np.ndarray] = {}


def _open_memmap(path: Path) -> np.ndarray:
    key = (os.getpid(), str(Path(path).resolve()))
    array = _MEMMAP_CACHE.get(key)
    if array is None:
        array = np.load(path, mmap_mode="r")
        _MEMMAP_CACHE[key] = array
    return array


TABULAR_CONTEXT_MISSING_COLUMNS = ("market_context_context_missing", "sector_context_context_missing")
SEQUENCE_CONTEXT_MISSING_COLUMNS = ("market_context_missing", "sector_context_missing")


def _date_to_int(value: object) -> int:
    text = str(value or "")
    return int(text.replace("-", "")[:8]) if text else 0


def _safe_float_frame(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _available(header: Sequence[str], requested: Sequence[str]) -> list[str]:
    available = set(header)
    generated = {
        "close_location",
        "true_range_pct",
        "dollar_volume_ratio_5d",
        "volume_zscore_20d",
        "stock_vs_market_return_1d",
        "stock_vs_market_return_5d",
        "stock_vs_sector_return_1d",
        "stock_vs_sector_return_5d",
    }
    return [column for column in requested if column in available or column in generated]


def _relative_columns(header: Sequence[str]) -> list[str]:
    return [
        column
        for column in header
        if column.endswith("__cs_z")
        or column.endswith("__cs_pct")
        or column.endswith("__sector_cs_z")
        or column.endswith("__sector_cs_pct")
    ]


def _stock_columns_from_header(
    header: Sequence[str],
    *,
    include_relative: bool,
    include_sentiment: bool,
    include_fundamentals: bool,
) -> list[str]:
    columns = _available(header, BASE_STOCK_FEATURES)
    if include_relative:
        columns.extend(_relative_columns(header))
    if include_sentiment:
        columns.extend(_available(header, SENTIMENT_FEATURE_COLUMNS))
    if include_fundamentals:
        columns.extend(_available(header, FUNDAMENTAL_FEATURE_COLUMNS))
    return list(dict.fromkeys(columns))


def _summary_columns(prefix: str, feature_columns: Sequence[str]) -> list[str]:
    columns: list[str] = []
    for column in feature_columns:
        safe = column.replace(".", "_")
        columns.extend([f"{prefix}{safe}__last", f"{prefix}{safe}__mean60", f"{prefix}{safe}__std60"])
    return columns


def _context_summary_columns(context_columns: Sequence[str], *, context_prefix: str) -> list[str]:
    return [f"{context_prefix}{name}" for name in _summary_columns("context_", context_columns)]


def _sequence_columns(
    *,
    header: Sequence[str],
    context_columns: Sequence[str],
    feature_set: str,
) -> list[str]:
    config = sequence_feature_config(feature_set)
    columns = _stock_columns_from_header(
        header,
        include_relative=config.include_relative,
        include_sentiment=config.include_sentiment,
        include_fundamentals=False,
    )
    if config.include_market_context:
        columns.extend([f"market_context_{column}" for column in context_columns])
        columns.append("market_context_missing")
    if config.include_sector_context:
        columns.extend([f"sector_context_{column}" for column in context_columns])
        columns.append("sector_context_missing")
    return list(dict.fromkeys(columns))


def _tabular_columns(
    *,
    header: Sequence[str],
    context_columns: Sequence[str],
    feature_set: str,
) -> list[str]:
    include_sentiment = "sentiment" in feature_set
    include_fundamentals = "fundamentals" in feature_set
    stock_columns = _stock_columns_from_header(
        header,
        include_relative="relative" in feature_set,
        include_sentiment=include_sentiment,
        include_fundamentals=include_fundamentals,
    )
    columns = _summary_columns("stock_", stock_columns)
    if "market" in feature_set:
        columns.extend(_context_summary_columns(context_columns, context_prefix="market_"))
        columns.append("market_context_context_missing")
    if "sector" in feature_set:
        columns.extend(_context_summary_columns(context_columns, context_prefix="sector_"))
        columns.append("sector_context_context_missing")
    return list(dict.fromkeys(columns))


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def _iter_ticker_groups(path: Path) -> Iterator[pd.DataFrame]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        current_ticker = ""
        rows: list[dict[str, str]] = []
        closed_tickers: set[str] = set()
        for row in reader:
            ticker = str(row.get("ticker") or "").upper()
            if not ticker:
                continue
            if current_ticker and ticker != current_ticker:
                closed_tickers.add(current_ticker)
                yield pd.DataFrame(rows)
                rows = []
            if ticker in closed_tickers:
                raise ValueError(
                    f"Input feature CSV is not ticker-contiguous. Ticker {ticker} appeared after its group closed."
                )
            current_ticker = ticker
            row["ticker"] = ticker
            rows.append(row)
        if rows:
            yield pd.DataFrame(rows)


def _stock_feature_path(dataset_root: Path) -> Path:
    return preferred_stock_feature_path(dataset_root)


def _context_tables(
    dataset_root: Path,
    *,
    context_columns: Sequence[str],
    benchmark_ticker: str,
    window_length: int,
) -> dict[str, object]:
    context = load_market_context_features(dataset_root)
    if context.empty:
        raise ValueError(f"Missing processed market context features under {dataset_root}.")
    context = add_priority_a_ohlcv_features(context)
    context["ticker"] = context["ticker"].astype(str).str.upper()
    context["date"] = context["date"].astype(str)
    context = _safe_float_frame(context, [*context_columns, "close", "return_1d", "rolling_return_5d"])
    summary = add_window_summaries(context, feature_cols=context_columns, prefix="context_", window_length=window_length)
    summary_columns = [column for column in summary.columns if column not in {"ticker", "date"}]

    benchmark = context[context["ticker"] == benchmark_ticker.upper()].copy()
    if benchmark.empty:
        raise ValueError(f"Benchmark ticker {benchmark_ticker} is missing from context features.")
    sector_context = context[context["ticker"].isin(set(SECTOR_ETF_BY_GICS.values()))].copy()
    summary_by_ticker = {
        ticker: group.set_index("date")[summary_columns].astype(float)
        for ticker, group in summary.groupby("ticker", sort=False)
    }
    daily_by_ticker = {
        ticker: group.set_index("date")[list(context_columns)].astype(float)
        for ticker, group in context.groupby("ticker", sort=False)
    }
    return {
        "context": context,
        "context_summary_columns": summary_columns,
        "benchmark_close": benchmark.set_index("date")["close"].astype(float).to_dict(),
        "benchmark_return_1d": benchmark.set_index("date").get("return_1d", pd.Series(dtype=float)).astype(float).to_dict(),
        "benchmark_return_5d": benchmark.set_index("date").get("rolling_return_5d", pd.Series(dtype=float)).astype(float).to_dict(),
        "sector_return_1d": sector_context.set_index(["ticker", "date"]).get("return_1d", pd.Series(dtype=float)).astype(float).to_dict(),
        "sector_return_5d": sector_context.set_index(["ticker", "date"]).get("rolling_return_5d", pd.Series(dtype=float)).astype(float).to_dict(),
        "summary_by_ticker": summary_by_ticker,
        "daily_by_ticker": daily_by_ticker,
    }


def _prepare_stock_group(
    group: pd.DataFrame,
    *,
    numeric_columns: Sequence[str],
    context_tables: dict[str, object],
    eligibility_config: EpisodeEligibilityConfig | None,
    research_config: ConservativeResearchUniverseConfig | None,
    benchmark_ticker: str,
) -> pd.DataFrame:
    group = group.copy()
    group["ticker"] = group["ticker"].astype(str).str.upper()
    group["date"] = group["date"].astype(str)
    group = group.sort_values("date").reset_index(drop=True)
    group = _safe_float_frame(group, numeric_columns)
    group = add_priority_a_ohlcv_features(group)
    group["sector_etf"] = group.get("gics_sector", pd.Series("", index=group.index)).map(SECTOR_ETF_BY_GICS)

    benchmark_return_1d = context_tables["benchmark_return_1d"]
    benchmark_return_5d = context_tables["benchmark_return_5d"]
    if "return_1d" in group.columns:
        group["stock_vs_market_return_1d"] = group["return_1d"].astype(float) - group["date"].map(benchmark_return_1d).astype(float)
    if "rolling_return_5d" in group.columns:
        group["stock_vs_market_return_5d"] = group["rolling_return_5d"].astype(float) - group["date"].map(benchmark_return_5d).astype(float)
    sector_return_1d = context_tables["sector_return_1d"]
    sector_return_5d = context_tables["sector_return_5d"]
    if "return_1d" in group.columns:
        group["stock_vs_sector_return_1d"] = [
            float(row["return_1d"]) - float(sector_return_1d.get((row["sector_etf"], row["date"]), np.nan))
            for _, row in group.iterrows()
        ]
    if "rolling_return_5d" in group.columns:
        group["stock_vs_sector_return_5d"] = [
            float(row["rolling_return_5d"]) - float(sector_return_5d.get((row["sector_etf"], row["date"]), np.nan))
            for _, row in group.iterrows()
        ]
    if eligibility_config is not None:
        group = add_episode_eligibility_columns(group, eligibility_config, benchmark_ticker=benchmark_ticker)
    else:
        group["window_row_count"] = np.arange(len(group), dtype=np.int64) + 1
    if research_config is not None:
        group = add_conservative_research_universe_columns(
            group,
            research_config,
            benchmark_ticker=benchmark_ticker,
        )
    return group


def _add_targets(
    group: pd.DataFrame,
    *,
    horizons: Sequence[int],
    classification_horizon: int,
    classification_threshold: float,
    classification_event_type: str,
    benchmark_close_by_date: dict[str, float],
) -> pd.DataFrame:
    out = group.copy()
    max_horizon = max(max(horizons), int(classification_horizon))
    anchor_close = out["close"].astype(float)
    benchmark_anchor = out["date"].map(benchmark_close_by_date).astype(float)
    for horizon in horizons:
        future_close = out["close"].shift(-horizon).astype(float)
        future_date = out["date"].shift(-horizon)
        stock_return = future_close / anchor_close - 1.0
        benchmark_future = future_date.map(benchmark_close_by_date).astype(float)
        benchmark_return = benchmark_future / benchmark_anchor - 1.0
        out[target_column(horizon)] = stock_return - benchmark_return
    path_columns: list[str] = []
    for step in range(1, classification_horizon + 1):
        future_close = out["close"].shift(-step).astype(float)
        future_date = out["date"].shift(-step)
        stock_return = future_close / anchor_close - 1.0
        if classification_event_type == PATH_5PCT_20D_EVENT_TYPE:
            path_col = f"_forward_return_path_{step}d"
            out[path_col] = stock_return
        else:
            benchmark_future = future_date.map(benchmark_close_by_date).astype(float)
            benchmark_return = benchmark_future / benchmark_anchor - 1.0
            path_col = f"_market_adjusted_return_path_{step}d"
            out[path_col] = stock_return - benchmark_return
        path_columns.append(path_col)
    class_col = classification_target_column(
        horizon=classification_horizon,
        threshold=classification_threshold,
        event_type=classification_event_type,
    )
    out[class_col] = pathwise_classification_labels(
        out[path_columns],
        threshold=classification_threshold,
        event_type=classification_event_type,
    )
    out["_has_all_future_horizons"] = np.arange(len(out)) <= len(out) - max_horizon - 1
    return out.drop(columns=path_columns)


def _eligible_episode_rows(
    group: pd.DataFrame,
    *,
    horizons: Sequence[int],
    window_length: int,
    classification_horizon: int,
    classification_threshold: float,
    classification_event_type: str,
    eligibility_enabled: bool,
    research_enabled: bool,
) -> pd.DataFrame:
    target_cols = [target_column(horizon) for horizon in horizons]
    class_col = classification_target_column(
        horizon=classification_horizon,
        threshold=classification_threshold,
        event_type=classification_event_type,
    )
    mask = group["window_row_count"].astype(int) >= window_length
    if eligibility_enabled and "episode_eligible" in group.columns:
        mask &= group["episode_eligible"].astype(bool)
    if research_enabled and "research_universe_ok" in group.columns:
        mask &= group["research_universe_ok"].astype(bool)
    mask &= group["_has_all_future_horizons"].astype(bool)
    rows = group.loc[mask].dropna(subset=target_cols).dropna(subset=[class_col])
    return rows.reset_index(drop=True)


def _labeling_summary_seed(
    *,
    classification_event_type: str,
    classification_horizon: int,
    classification_threshold: float,
) -> dict[str, object]:
    return {
        "mode": classification_event_type,
        "horizon_days": int(classification_horizon),
        "positive_threshold": float(classification_threshold),
        "negative_threshold": (
            PATH_5PCT_20D_NEGATIVE_THRESHOLD
            if classification_event_type == PATH_5PCT_20D_EVENT_TYPE
            else None
        ),
        "use_close_only": classification_event_type == PATH_5PCT_20D_EVENT_TYPE,
        "require_full_forward_window": True,
        "negative_overrides_positive": classification_event_type == PATH_5PCT_20D_EVENT_TYPE,
        "research_universe_enabled": False,
        "research_universe_excluded_rows": 0,
        "candidate_row_count": 0,
        "unlabeled_missing_forward_window_rows": 0,
        "unlabeled_invalid_price_rows": 0,
        "unlabeled_missing_regression_target_rows": 0,
        "labeled_row_count": 0,
        "class_counts": {},
    }


def _accumulate_labeling_summary(
    summary: dict[str, object],
    group: pd.DataFrame,
    *,
    horizons: Sequence[int],
    window_length: int,
    classification_horizon: int,
    classification_threshold: float,
    classification_event_type: str,
    eligibility_enabled: bool,
    research_enabled: bool,
) -> None:
    target_cols = [target_column(horizon) for horizon in horizons]
    class_col = classification_target_column(
        horizon=classification_horizon,
        threshold=classification_threshold,
        event_type=classification_event_type,
    )
    candidate_mask = group["window_row_count"].astype(int) >= window_length
    if eligibility_enabled and "episode_eligible" in group.columns:
        candidate_mask &= group["episode_eligible"].fillna(False).astype(bool)
    pre_research_mask = candidate_mask.copy()
    if research_enabled and "research_universe_ok" in group.columns:
        candidate_mask &= group["research_universe_ok"].fillna(False).astype(bool)
    full_forward_mask = group["_has_all_future_horizons"].fillna(False).astype(bool)
    target_complete_mask = group[target_cols].notna().all(axis=1)
    label_complete_mask = group[class_col].notna()
    final_mask = candidate_mask & full_forward_mask & target_complete_mask & label_complete_mask

    summary["candidate_row_count"] = int(summary["candidate_row_count"]) + int(candidate_mask.sum())
    summary["research_universe_excluded_rows"] = int(summary.get("research_universe_excluded_rows", 0)) + int(
        (pre_research_mask & ~candidate_mask).sum()
    )
    summary["unlabeled_missing_forward_window_rows"] = int(
        summary["unlabeled_missing_forward_window_rows"]
    ) + int((candidate_mask & ~full_forward_mask).sum())
    summary["unlabeled_invalid_price_rows"] = int(summary["unlabeled_invalid_price_rows"]) + int(
        (candidate_mask & full_forward_mask & ~label_complete_mask).sum()
    )
    summary["unlabeled_missing_regression_target_rows"] = int(
        summary["unlabeled_missing_regression_target_rows"]
    ) + int((candidate_mask & full_forward_mask & label_complete_mask & ~target_complete_mask).sum())
    summary["labeled_row_count"] = int(summary["labeled_row_count"]) + int(final_mask.sum())
    class_counts = dict(summary.get("class_counts") or {})
    for key, value in group.loc[final_mask, class_col].astype(int).value_counts().sort_index().items():
        str_key = str(int(key))
        class_counts[str_key] = int(class_counts.get(str_key, 0)) + int(value)
    summary["class_counts"] = class_counts


def _maintain_episode_heap(
    heap: list[tuple[str, str, int]],
    rows: pd.DataFrame,
    *,
    max_episodes: int,
) -> None:
    for row in rows[["date", "ticker", "window_row_count"]].itertuples(index=False):
        item = (str(row.date), str(row.ticker), int(row.window_row_count))
        if len(heap) < max_episodes:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)


def _selected_keys_from_heap(heap: list[tuple[str, str, int]]) -> set[tuple[str, int]]:
    return {(ticker, int(row_count)) for _, ticker, row_count in heap}


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out


def _row_from_frame(frame: pd.DataFrame, key: str, columns: Sequence[str]) -> np.ndarray:
    if key not in frame.index:
        return np.full(len(columns), np.nan, dtype=np.float32)
    return frame.loc[key, columns].to_numpy(dtype=np.float32, copy=True)


def _fill_finite(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)


def _build_sequence_rows(
    group: pd.DataFrame,
    *,
    stock_columns: Sequence[str],
    context_columns: Sequence[str],
    feature_columns: Sequence[str],
    context_tables: dict[str, object],
    benchmark_ticker: str,
) -> np.ndarray:
    frame = _ensure_columns(group, stock_columns)
    values: dict[str, np.ndarray] = {
        column: pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
        for column in stock_columns
    }
    daily_by_ticker: dict[str, pd.DataFrame] = context_tables["daily_by_ticker"]
    market = daily_by_ticker.get(benchmark_ticker.upper(), pd.DataFrame())
    sector_ticker = str(frame["sector_etf"].iloc[0] or "")
    sector = daily_by_ticker.get(sector_ticker, pd.DataFrame())
    dates = frame["date"].astype(str).tolist()
    for prefix, context_frame, missing_name in (
        ("market_context_", market, "market_context_missing"),
        ("sector_context_", sector, "sector_context_missing"),
    ):
        rows = np.vstack([_row_from_frame(context_frame, date, context_columns) for date in dates]) if len(dates) else np.empty((0, len(context_columns)))
        missing = np.isnan(rows).all(axis=1).astype(np.float32) if len(rows) else np.empty(0, dtype=np.float32)
        for idx, column in enumerate(context_columns):
            values[f"{prefix}{column}"] = rows[:, idx].astype(np.float32)
        values[missing_name] = missing
    matrix = np.column_stack([values.get(column, np.zeros(len(frame), dtype=np.float32)) for column in feature_columns])
    return _fill_finite(matrix)


@dataclass(frozen=True)
class CachedTabularFeatureStore:
    feature_set: str
    path: Path
    feature_columns: list[str]
    shape: tuple[int, int]

    def view(self, rows: Sequence[bool] | np.ndarray | pd.Series) -> "CachedTabularView":
        if isinstance(rows, pd.Series):
            row_indices = np.flatnonzero(rows.to_numpy(dtype=bool))
        else:
            arr = np.asarray(rows)
            row_indices = np.flatnonzero(arr.astype(bool)) if arr.dtype == bool else arr.astype(np.int64)
        return CachedTabularView(self, row_indices.astype(np.int64))

    def open(self) -> np.ndarray:
        return _open_memmap(self.path)


@dataclass(frozen=True)
class CachedTabularView:
    store: CachedTabularFeatureStore
    row_indices: np.ndarray

    @property
    def columns(self) -> list[str]:
        return self.store.feature_columns

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.row_indices), len(self.store.feature_columns))

    def __len__(self) -> int:
        return len(self.row_indices)

    def to_numpy(self, dtype=np.float64) -> np.ndarray:
        data = self.store.open()
        return np.asarray(data[self.row_indices], dtype=dtype).copy()

    def iter_numpy_batches(
        self,
        *,
        batch_size: int,
        shuffle: bool = False,
        random_state: int = 0,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        batch_size = max(int(batch_size), 1)
        row_count = len(self.row_indices)
        data = self.store.open()
        if not shuffle:
            for start in range(0, row_count, batch_size):
                local = np.arange(start, min(start + batch_size, row_count), dtype=np.int64)
                source_rows = self.row_indices[local]
                yield local, np.asarray(data[source_rows], dtype=np.float32).copy()
            return
        raw_block_size = os.environ.get("V1_TABULAR_SHUFFLE_BLOCK_ROWS", "1048576").strip()
        try:
            block_size = max(int(raw_block_size), batch_size)
        except ValueError:
            block_size = max(1_048_576, batch_size)
        blocks = np.arange(0, row_count, block_size, dtype=np.int64)
        if len(blocks):
            rng = np.random.default_rng(random_state)
            rng.shuffle(blocks)
        for block_start in blocks:
            block_end = min(int(block_start) + block_size, row_count)
            for start in range(int(block_start), block_end, batch_size):
                local = np.arange(start, min(start + batch_size, block_end), dtype=np.int64)
                source_rows = self.row_indices[local]
                yield local, np.asarray(data[source_rows], dtype=np.float32).copy()


@dataclass(frozen=True)
class MemmapSequenceFeatureStore:
    feature_set: str
    feature_columns: list[str]
    path: Path
    date_path: Path
    shape: tuple[int, int]
    ticker_offsets: dict[str, int]
    ticker_lengths: dict[str, int]

    def open(self) -> np.ndarray:
        return _open_memmap(self.path)

    def open_dates(self) -> np.ndarray:
        return _open_memmap(self.date_path)

    def get_window(self, ticker: str, end_index: int, window_length: int) -> np.ndarray:
        symbol = str(ticker).upper()
        if symbol not in self.ticker_offsets:
            raise IndexError(f"Ticker {symbol} is not available in sequence cache.")
        start = int(end_index) - int(window_length) + 1
        if start < 0 or int(end_index) >= self.ticker_lengths[symbol]:
            raise IndexError(f"Invalid window for ticker={symbol!r} end_index={end_index} window_length={window_length}.")
        offset = self.ticker_offsets[symbol]
        data = self.open()
        return np.asarray(data[offset + start : offset + int(end_index) + 1], dtype=np.float32).copy()

    def iter_rows_through(self, cutoff_date: str, *, batch_size: int = 524_288) -> Iterator[np.ndarray]:
        cutoff = _date_to_int(cutoff_date)
        data = self.open()
        dates = self.open_dates()
        for start in range(0, self.shape[0], max(int(batch_size), 1)):
            end = min(start + int(batch_size), self.shape[0])
            date_block = dates[start:end]
            mask = date_block <= cutoff
            if mask.any():
                if bool(mask.all()):
                    yield data[start:end]
                else:
                    yield data[start:end][mask]


@dataclass(frozen=True)
class CachedV1Dataset(V1Dataset):
    cache_dir: Path | None = None


def load_cached_v1_dataset(cache_dir: str | Path) -> CachedV1Dataset:
    root = Path(cache_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    metadata = pd.read_csv(
        root / "episode_metadata.csv",
        dtype={
            "ticker": "string",
            "anchor_date": "string",
            "gics_sector": "string",
            "gics_sub_industry": "string",
            "exchange": "string",
        },
        keep_default_na=False,
    )
    targets = pd.read_csv(
        root / "targets.csv",
        dtype={"ticker": "string", "anchor_date": "string"},
        keep_default_na=False,
    )
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
    targets["ticker"] = targets["ticker"].astype(str).str.upper()
    for column in [*manifest["target_columns"], *manifest["classification_target_columns"]]:
        targets[column] = pd.to_numeric(targets[column], errors="coerce")
    feature_sets: dict[str, CachedTabularFeatureStore] = {}
    feature_columns: dict[str, list[str]] = {}
    for feature_set, payload in manifest.get("tabular_feature_sets", {}).items():
        columns = list(payload["feature_columns"])
        shape = tuple(int(value) for value in payload["shape"])
        feature_sets[feature_set] = CachedTabularFeatureStore(
            feature_set=feature_set,
            path=root / payload["path"],
            feature_columns=columns,
            shape=(shape[0], shape[1]),
        )
        feature_columns[feature_set] = columns
    return CachedV1Dataset(
        metadata=metadata,
        targets=targets,
        feature_sets=feature_sets,
        target_columns=list(manifest["target_columns"]),
        classification_target_columns=list(manifest["classification_target_columns"]),
        feature_columns=feature_columns,
        split_by_date={},
        labeling_summary=manifest.get("labeling"),
        cache_dir=root,
    )


def load_cached_sequence_stores(cache_dir: str | Path) -> dict[str, MemmapSequenceFeatureStore]:
    root = Path(cache_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    stores: dict[str, MemmapSequenceFeatureStore] = {}
    for feature_set, payload in manifest.get("sequence_feature_sets", {}).items():
        shape = tuple(int(value) for value in payload["shape"])
        stores[feature_set] = MemmapSequenceFeatureStore(
            feature_set=feature_set,
            feature_columns=list(payload["feature_columns"]),
            path=root / payload["path"],
            date_path=root / payload["date_path"],
            shape=(shape[0], shape[1]),
            ticker_offsets={str(k).upper(): int(v) for k, v in payload["ticker_offsets"].items()},
            ticker_lengths={str(k).upper(): int(v) for k, v in payload["ticker_lengths"].items()},
        )
    return stores


def build_episode_cache(
    *,
    dataset_root: str | Path,
    cache_dir: str | Path,
    feature_sets: Sequence[str],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    window_length: int = DEFAULT_WINDOW_LENGTH,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    max_episodes: int | None = None,
    classification_horizon: int = DEFAULT_CLASSIFICATION_HORIZON,
    classification_threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    classification_event_type: str = DEFAULT_CLASSIFICATION_EVENT_TYPE,
    eligibility_config: EpisodeEligibilityConfig | None = None,
    research_config: ConservativeResearchUniverseConfig | None = None,
    force: bool = False,
    progress_every: int = 250,
) -> dict[str, object]:
    validate_classification_event_config(
        event_type=classification_event_type,
        horizon=classification_horizon,
        threshold=classification_threshold,
    )
    dataset_root = Path(dataset_root)
    cache_dir = Path(cache_dir)
    stock_path = _stock_feature_path(dataset_root)
    if not stock_path.exists():
        raise FileNotFoundError(f"Missing stock features: {stock_path}")
    if cache_dir.exists() and any(cache_dir.iterdir()) and not force:
        raise FileExistsError(f"Episode cache already exists. Use force=True to rebuild: {cache_dir}")
    if force and cache_dir.exists():
        for child in cache_dir.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    cache_dir.mkdir(parents=True, exist_ok=True)

    header = _read_header(stock_path)
    context = load_market_context_features(dataset_root)
    context = add_priority_a_ohlcv_features(context)
    context_columns = [column for column in CONTEXT_FEATURES if column in context.columns]
    context_tables = _context_tables(
        dataset_root,
        context_columns=context_columns,
        benchmark_ticker=benchmark_ticker,
        window_length=window_length,
    )
    horizons = tuple(int(horizon) for horizon in horizons)
    target_columns = [target_column(horizon) for horizon in horizons]
    classification_columns = [
        classification_target_column(
            horizon=classification_horizon,
            threshold=classification_threshold,
            event_type=classification_event_type,
        )
    ]

    tabular_sets = [feature_set for feature_set in feature_sets if feature_set not in SEQUENCE_FEATURE_SET_NAMES]
    sequence_sets = [feature_set for feature_set in feature_sets if feature_set in SEQUENCE_FEATURE_SET_NAMES]
    tabular_feature_columns = {
        feature_set: _tabular_columns(header=header, context_columns=context_columns, feature_set=feature_set)
        for feature_set in tabular_sets
    }
    sequence_feature_columns = {
        feature_set: _sequence_columns(header=header, context_columns=context_columns, feature_set=feature_set)
        for feature_set in sequence_sets
    }
    for feature_set, columns in {**tabular_feature_columns, **sequence_feature_columns}.items():
        validate_model_feature_columns(columns, feature_set=feature_set)

    numeric_columns = set()
    for columns in [*tabular_feature_columns.values(), *sequence_feature_columns.values()]:
        for column in columns:
            clean = column
            for prefix in ("stock_", "market_context_", "sector_context_", "market_", "sector_"):
                if clean.startswith(prefix):
                    clean = clean[len(prefix) :]
            for suffix in ("__last", "__mean60", "__std60"):
                if clean.endswith(suffix):
                    clean = clean[: -len(suffix)]
            numeric_columns.add(clean)
    numeric_columns.update(
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "dollar_volume",
            "prev_close",
            "return_1d",
            "rolling_return_5d",
        ]
    )

    start_time = time.monotonic()
    heap: list[tuple[str, str, int]] = []
    stock_row_count = 0
    eligible_count = 0
    ticker_count = 0
    labeling_summary = _labeling_summary_seed(
        classification_event_type=classification_event_type,
        classification_horizon=classification_horizon,
        classification_threshold=classification_threshold,
    )
    labeling_summary["research_universe_enabled"] = bool(research_config is not None and research_config.enabled)
    for group in _iter_ticker_groups(stock_path):
        ticker = str(group["ticker"].iloc[0]).upper()
        if ticker in set(MARKET_CONTEXT_TICKERS) or ticker == benchmark_ticker.upper():
            continue
        ticker_count += 1
        prepared = _prepare_stock_group(
            group,
            numeric_columns=numeric_columns,
            context_tables=context_tables,
            eligibility_config=eligibility_config,
            research_config=research_config,
            benchmark_ticker=benchmark_ticker,
        )
        with_targets = _add_targets(
            prepared,
            horizons=horizons,
            classification_horizon=classification_horizon,
            classification_threshold=classification_threshold,
            classification_event_type=classification_event_type,
            benchmark_close_by_date=context_tables["benchmark_close"],
        )
        eligible = _eligible_episode_rows(
            with_targets,
            horizons=horizons,
            window_length=window_length,
            classification_horizon=classification_horizon,
            classification_threshold=classification_threshold,
            classification_event_type=classification_event_type,
            eligibility_enabled=eligibility_config is not None,
            research_enabled=research_config is not None,
        )
        stock_row_count += len(prepared)
        eligible_count += len(eligible)
        _accumulate_labeling_summary(
            labeling_summary,
            with_targets,
            horizons=horizons,
            window_length=window_length,
            classification_horizon=classification_horizon,
            classification_threshold=classification_threshold,
            classification_event_type=classification_event_type,
            eligibility_enabled=eligibility_config is not None,
            research_enabled=research_config is not None,
        )
        if max_episodes and max_episodes > 0:
            _maintain_episode_heap(heap, eligible, max_episodes=max_episodes)
        if progress_every and ticker_count % progress_every == 0:
            print(
                json.dumps(
                    {
                        "step": "episode_cache_count_progress",
                        "tickers": ticker_count,
                        "stock_rows": stock_row_count,
                        "eligible_episodes": eligible_count,
                        "unlabeled_missing_forward_window_rows": labeling_summary[
                            "unlabeled_missing_forward_window_rows"
                        ],
                        "selected_heap_size": len(heap) if max_episodes else None,
                        "elapsed_seconds": round(time.monotonic() - start_time, 1),
                    }
                ),
                flush=True,
            )
    selected_keys = _selected_keys_from_heap(heap) if max_episodes and max_episodes > 0 else None
    episode_count = len(selected_keys) if selected_keys is not None else eligible_count

    tabular_arrays = {}
    for feature_set, columns in tabular_feature_columns.items():
        path = cache_dir / "tabular" / f"{feature_set}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        tabular_arrays[feature_set] = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=(episode_count, len(columns)),
        )
    sequence_arrays = {}
    sequence_dates = {}
    for feature_set, columns in sequence_feature_columns.items():
        path = cache_dir / "sequence" / f"{feature_set}.npy"
        date_path = cache_dir / "sequence" / f"{feature_set}_dates.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        sequence_arrays[feature_set] = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=(stock_row_count, len(columns)),
        )
        sequence_dates[feature_set] = np.lib.format.open_memmap(
            date_path,
            mode="w+",
            dtype=np.int32,
            shape=(stock_row_count,),
        )

    metadata_path = cache_dir / "episode_metadata.csv"
    targets_path = cache_dir / "targets.csv"
    metadata_columns = [
        "ticker",
        "anchor_date",
        "gics_sector",
        "gics_sub_industry",
        "sector_etf",
        "window_row_count",
        *ELIGIBILITY_DIAGNOSTIC_COLUMNS,
        *RESEARCH_UNIVERSE_DIAGNOSTIC_COLUMNS,
    ]
    target_csv_columns = ["ticker", "anchor_date", *target_columns, *classification_columns]
    episode_idx = 0
    sequence_idx = 0
    ticker_offsets: dict[str, int] = {}
    ticker_lengths: dict[str, int] = {}
    with metadata_path.open("w", encoding="utf-8", newline="") as meta_handle, targets_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as target_handle:
        metadata_writer = csv.DictWriter(meta_handle, fieldnames=metadata_columns)
        target_writer = csv.DictWriter(target_handle, fieldnames=target_csv_columns)
        metadata_writer.writeheader()
        target_writer.writeheader()
        for group in _iter_ticker_groups(stock_path):
            ticker = str(group["ticker"].iloc[0]).upper()
            if ticker in set(MARKET_CONTEXT_TICKERS) or ticker == benchmark_ticker.upper():
                continue
            prepared = _prepare_stock_group(
                group,
                numeric_columns=numeric_columns,
                context_tables=context_tables,
                eligibility_config=eligibility_config,
                research_config=research_config,
                benchmark_ticker=benchmark_ticker,
            )
            with_targets = _add_targets(
                prepared,
                horizons=horizons,
                classification_horizon=classification_horizon,
                classification_threshold=classification_threshold,
                classification_event_type=classification_event_type,
                benchmark_close_by_date=context_tables["benchmark_close"],
            )
            ticker_offsets[ticker] = sequence_idx
            ticker_lengths[ticker] = len(with_targets)
            for feature_set, columns in sequence_feature_columns.items():
                stock_columns = [
                    column
                    for column in columns
                    if not column.startswith("market_context_")
                    and not column.startswith("sector_context_")
                    and column not in SEQUENCE_CONTEXT_MISSING_COLUMNS
                ]
                rows = _build_sequence_rows(
                    with_targets,
                    stock_columns=stock_columns,
                    context_columns=context_columns,
                    feature_columns=columns,
                    context_tables=context_tables,
                    benchmark_ticker=benchmark_ticker,
                )
                sequence_arrays[feature_set][sequence_idx : sequence_idx + len(rows), :] = rows
                sequence_dates[feature_set][sequence_idx : sequence_idx + len(rows)] = [
                    _date_to_int(value) for value in with_targets["date"].astype(str)
                ]
            eligible = _eligible_episode_rows(
                with_targets,
                horizons=horizons,
                window_length=window_length,
                classification_horizon=classification_horizon,
                classification_threshold=classification_threshold,
                classification_event_type=classification_event_type,
                eligibility_enabled=eligibility_config is not None,
                research_enabled=research_config is not None,
            )
            if selected_keys is not None:
                eligible = eligible[
                    [
                        (str(row.ticker).upper(), int(row.window_row_count)) in selected_keys
                        for row in eligible[["ticker", "window_row_count"]].itertuples(index=False)
                    ]
                ].reset_index(drop=True)
            if len(eligible):
                tabular_stock_kinds = {
                    feature_set: _stock_columns_from_header(
                        header,
                        include_relative="relative" in feature_set,
                        include_sentiment="sentiment" in feature_set,
                        include_fundamentals="fundamentals" in feature_set,
                    )
                    for feature_set in tabular_sets
                }
                stock_summary_by_kind: dict[tuple[str, ...], pd.DataFrame] = {}
                for columns in {tuple(value) for value in tabular_stock_kinds.values()}:
                    stock_summary_by_kind[columns] = add_window_summaries(
                        _ensure_columns(with_targets, columns),
                        feature_cols=columns,
                        prefix="stock_",
                        window_length=window_length,
                    ).set_index("date")
                summary_by_ticker: dict[str, pd.DataFrame] = context_tables["summary_by_ticker"]
                market_summary = summary_by_ticker.get(benchmark_ticker.upper(), pd.DataFrame())
                sector_summary = summary_by_ticker.get(str(with_targets["sector_etf"].iloc[0] or ""), pd.DataFrame())
                for _, row in eligible.iterrows():
                    date_value = str(row["date"])
                    for feature_set, columns in tabular_feature_columns.items():
                        stock_columns = tuple(tabular_stock_kinds[feature_set])
                        pieces = []
                        stock_summary = stock_summary_by_kind[stock_columns]
                        stock_summary_columns = _summary_columns("stock_", stock_columns)
                        pieces.append(_row_from_frame(stock_summary, date_value, stock_summary_columns))
                        if "market" in feature_set:
                            raw = _row_from_frame(market_summary, date_value, context_tables["context_summary_columns"])
                            pieces.append(raw)
                            pieces.append(np.array([float(np.isnan(raw).all())], dtype=np.float32))
                        if "sector" in feature_set:
                            raw = _row_from_frame(sector_summary, date_value, context_tables["context_summary_columns"])
                            pieces.append(raw)
                            pieces.append(np.array([float(np.isnan(raw).all())], dtype=np.float32))
                        tabular_arrays[feature_set][episode_idx, :] = _fill_finite(np.concatenate(pieces))
                    metadata_writer.writerow(
                        {
                            "ticker": row["ticker"],
                            "anchor_date": date_value,
                            "gics_sector": row.get("gics_sector", "Unknown") or "Unknown",
                            "gics_sub_industry": row.get("gics_sub_industry", "Unknown") or "Unknown",
                            "sector_etf": row.get("sector_etf", ""),
                            "window_row_count": int(row["window_row_count"]),
                            **{
                                column: row.get(column, "")
                                for column in [*eligibility_metadata_columns(eligible), *research_universe_metadata_columns(eligible)]
                            },
                        }
                    )
                    target_writer.writerow(
                        {
                            "ticker": row["ticker"],
                            "anchor_date": date_value,
                            **{column: row[column] for column in target_columns},
                            **{column: row[column] for column in classification_columns},
                        }
                    )
                    episode_idx += 1
            sequence_idx += len(with_targets)
            if progress_every and len(ticker_offsets) % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "step": "episode_cache_write_progress",
                            "tickers": len(ticker_offsets),
                            "stock_rows_written": sequence_idx,
                            "episodes_written": episode_idx,
                            "elapsed_seconds": round(time.monotonic() - start_time, 1),
                        }
                    ),
                    flush=True,
                )
    for array in [*tabular_arrays.values(), *sequence_arrays.values(), *sequence_dates.values()]:
        array.flush()

    sequence_payload = {}
    for feature_set, columns in sequence_feature_columns.items():
        sequence_payload[feature_set] = {
            "path": f"sequence/{feature_set}.npy",
            "date_path": f"sequence/{feature_set}_dates.npy",
            "shape": [stock_row_count, len(columns)],
            "feature_columns": columns,
            "ticker_offsets": ticker_offsets,
            "ticker_lengths": ticker_lengths,
        }
    tabular_payload = {
        feature_set: {
            "path": f"tabular/{feature_set}.npy",
            "shape": [episode_count, len(columns)],
            "feature_columns": columns,
        }
        for feature_set, columns in tabular_feature_columns.items()
    }
    manifest = {
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "mode": "v1_episode_cache",
        "dataset_root": str(dataset_root.resolve()),
        "stock_feature_path": str(stock_path.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "feature_sets": list(feature_sets),
        "tabular_feature_sets": tabular_payload,
        "sequence_feature_sets": sequence_payload,
        "target_columns": target_columns,
        "classification_target_columns": classification_columns,
        "horizons": list(horizons),
        "classification_horizon": classification_horizon,
        "classification_threshold": classification_threshold,
        "classification_event_type": classification_event_type,
        "labeling": labeling_summary,
        "window_length": window_length,
        "benchmark_ticker": benchmark_ticker.upper(),
        "max_episodes": max_episodes,
        "eligible_episode_count_before_cap": eligible_count,
        "episode_count": episode_idx,
        "stock_row_count": stock_row_count,
        "ticker_count": len(ticker_offsets),
        "elapsed_seconds": round(time.monotonic() - start_time, 1),
        "episode_eligibility": eligibility_config.to_dict() if eligibility_config is not None else {"enabled": False},
        "research_universe": research_config.to_dict() if research_config is not None else {"enabled": False},
        "notes": [
            "Feature engineering is materialized once into float32 arrays.",
            "Torch tabular and sequence models can train from these arrays without rebuilding pandas feature frames per fold.",
            "Raw identifiers remain in metadata only and are not included in feature arrays.",
        ],
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
