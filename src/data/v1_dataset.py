from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_HORIZONS = (1, 5, 10, 20)
DEFAULT_WINDOW_LENGTH = 60
DEFAULT_BENCHMARK_TICKER = "SPY"
DEFAULT_CLASSIFICATION_HORIZON = 20
DEFAULT_CLASSIFICATION_THRESHOLD = 0.05
DEFAULT_CLASSIFICATION_EVENT_TYPE = "anytime_pathwise_outperform"

SECTOR_ETF_BY_GICS = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

MARKET_CONTEXT_TICKERS = tuple(sorted({DEFAULT_BENCHMARK_TICKER, *SECTOR_ETF_BY_GICS.values()}))

IDENTIFIER_COLUMNS = {
    "date",
    "ticker",
    "gics_sector",
    "gics_sub_industry",
    "adjusted",
    "timestamp_ms",
}

RAW_LEVEL_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "transactions",
    "dollar_volume",
    "prev_close",
    "sma_close_20d",
    "sma_close_60d",
    "rolling_avg_volume_20d",
    "rolling_avg_volume_60d",
    "rolling_avg_dollar_volume_20d",
    "rolling_avg_dollar_volume_60d",
}

BASE_STOCK_FEATURES = [
    "return_1d",
    "log_return_1d",
    "gap_pct",
    "intraday_return",
    "hl_range_pct",
    "close_to_vwap_pct",
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
    "log1p_volume",
    "log1p_dollar_volume",
    "log1p_rolling_avg_volume_20d",
    "log1p_rolling_avg_volume_60d",
    "log1p_rolling_avg_dollar_volume_20d",
    "log1p_rolling_avg_dollar_volume_60d",
]

CONTEXT_FEATURES = [
    "return_1d",
    "log_return_1d",
    "gap_pct",
    "intraday_return",
    "hl_range_pct",
    "close_to_vwap_pct",
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

FEATURE_SET_NAMES = (
    "stock_only",
    "stock_relative",
    "stock_relative_market",
    "stock_relative_market_sector",
)
SEQUENCE_FEATURE_SET_NAMES = ("stock_only", "stock_relative")
STATIC_CATEGORICAL_COLUMNS = ("gics_sector", "gics_sub_industry")


@dataclass(frozen=True)
class V1Dataset:
    metadata: pd.DataFrame
    targets: pd.DataFrame
    feature_sets: dict[str, pd.DataFrame]
    target_columns: list[str]
    classification_target_columns: list[str]
    feature_columns: dict[str, list[str]]
    split_by_date: dict[str, tuple[str, str]]


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_dates: tuple[str, ...]
    val_dates: tuple[str, ...]
    oos_dates: tuple[str, ...]
    purge_gap: int

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "train_dates": list(self.train_dates),
            "val_dates": list(self.val_dates),
            "oos_dates": list(self.oos_dates),
            "purge_gap": self.purge_gap,
            "train_start_date": self.train_dates[0] if self.train_dates else None,
            "train_end_date": self.train_dates[-1] if self.train_dates else None,
            "val_start_date": self.val_dates[0] if self.val_dates else None,
            "val_end_date": self.val_dates[-1] if self.val_dates else None,
            "oos_start_date": self.oos_dates[0] if self.oos_dates else None,
            "oos_end_date": self.oos_dates[-1] if self.oos_dates else None,
        }


@dataclass(frozen=True)
class SequenceFeatureStore:
    feature_set: str
    feature_columns: list[str]
    ticker_arrays: dict[str, np.ndarray]
    ticker_dates: dict[str, np.ndarray]

    def get_window(self, ticker: str, end_index: int, window_length: int) -> np.ndarray:
        rows = self.ticker_arrays[str(ticker).upper()]
        start = end_index - window_length + 1
        if start < 0 or end_index >= len(rows):
            raise IndexError(f"Invalid window for {ticker=} {end_index=} {window_length=}")
        return rows[start : end_index + 1]

    def fit_rows(self, train_dates: Sequence[str]) -> np.ndarray:
        train_dates_set = {str(value) for value in train_dates}
        pieces: list[np.ndarray] = []
        for ticker, rows in self.ticker_arrays.items():
            dates = self.ticker_dates[ticker]
            mask = np.isin(dates, list(train_dates_set))
            if mask.any():
                pieces.append(rows[mask])
        if not pieces:
            return np.empty((0, len(self.feature_columns)), dtype=np.float32)
        return np.concatenate(pieces, axis=0)

    def fit_rows_through(self, cutoff_date: str) -> np.ndarray:
        pieces: list[np.ndarray] = []
        cutoff = str(cutoff_date)
        for ticker, rows in self.ticker_arrays.items():
            dates = self.ticker_dates[ticker]
            mask = dates <= cutoff
            if mask.any():
                pieces.append(rows[mask])
        if not pieces:
            return np.empty((0, len(self.feature_columns)), dtype=np.float32)
        return np.concatenate(pieces, axis=0)


def parse_horizons(value: str | Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return DEFAULT_HORIZONS
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(int(item) for item in value)


def target_column(horizon: int) -> str:
    return f"market_adjusted_return_{horizon}d"


def classification_target_column(*, horizon: int = DEFAULT_CLASSIFICATION_HORIZON, threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD) -> str:
    threshold_pct = int(round(float(threshold) * 100))
    return f"market_outperform_any_{int(horizon)}d_gt_{threshold_pct}pct"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def load_daily_features(dataset_root: str | Path) -> pd.DataFrame:
    root = Path(dataset_root)
    normalized = root / "processed" / "daily_features_normalized.csv"
    processed = root / "processed" / "daily_features.csv"
    path = normalized if normalized.exists() else processed
    df = _read_csv(path)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["date"] = df["date"].astype(str)
    return df


def load_market_context_features(dataset_root: str | Path, stock_features: pd.DataFrame | None = None) -> pd.DataFrame:
    root = Path(dataset_root)
    context_path = root / "processed" / "market_context_features.csv"
    if context_path.exists():
        df = pd.read_csv(context_path)
    elif stock_features is not None:
        df = stock_features[stock_features["ticker"].isin(MARKET_CONTEXT_TICKERS)].copy()
    else:
        df = pd.DataFrame()
    if not df.empty:
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["date"] = df["date"].astype(str)
    return df


def _available_numeric_columns(df: pd.DataFrame, requested: Sequence[str]) -> list[str]:
    return [col for col in requested if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]


def select_stock_feature_columns(df: pd.DataFrame, include_relative: bool) -> list[str]:
    cols = _available_numeric_columns(df, BASE_STOCK_FEATURES)
    if include_relative:
        relative = [
            col
            for col in df.columns
            if (
                col.endswith("__cs_z")
                or col.endswith("__cs_pct")
                or col.endswith("__sector_cs_z")
                or col.endswith("__sector_cs_pct")
            )
            and pd.api.types.is_numeric_dtype(df[col])
        ]
        cols.extend(relative)
    return list(dict.fromkeys(cols))


def select_context_feature_columns(df: pd.DataFrame) -> list[str]:
    return _available_numeric_columns(df, CONTEXT_FEATURES)


def select_sequence_feature_columns(stock_features: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set not in SEQUENCE_FEATURE_SET_NAMES:
        raise ValueError(f"Sequence inputs are only supported for {SEQUENCE_FEATURE_SET_NAMES}, got {feature_set}.")
    return select_stock_feature_columns(stock_features, include_relative=feature_set != "stock_only")


def _filtered_stock_universe(
    stock_features: pd.DataFrame,
    *,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
) -> pd.DataFrame:
    excluded = set(MARKET_CONTEXT_TICKERS)
    if benchmark_ticker.upper():
        excluded.add(benchmark_ticker.upper())
    stocks = stock_features[~stock_features["ticker"].isin(excluded)].copy()
    stocks = stocks.sort_values(["ticker", "date"]).reset_index(drop=True)
    stocks["window_row_count"] = stocks.groupby("ticker").cumcount() + 1
    return stocks


def build_sequence_feature_store(
    stock_features: pd.DataFrame,
    feature_set: str,
    *,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    feature_columns: Sequence[str] | None = None,
) -> SequenceFeatureStore:
    stocks = _filtered_stock_universe(stock_features, benchmark_ticker=benchmark_ticker)
    resolved_feature_columns = list(feature_columns) if feature_columns is not None else select_sequence_feature_columns(stocks, feature_set)
    if not resolved_feature_columns:
        raise ValueError(f"No sequence feature columns available for {feature_set}.")
    ticker_arrays: dict[str, np.ndarray] = {}
    ticker_dates: dict[str, np.ndarray] = {}
    for ticker, group in stocks.groupby("ticker", sort=False):
        group_values = group.reindex(columns=resolved_feature_columns, fill_value=0.0)
        ticker_arrays[str(ticker).upper()] = group_values.astype(float).to_numpy(dtype=np.float32)
        ticker_dates[str(ticker).upper()] = group["date"].astype(str).to_numpy(dtype=object)
    return SequenceFeatureStore(
        feature_set=feature_set,
        feature_columns=resolved_feature_columns,
        ticker_arrays=ticker_arrays,
        ticker_dates=ticker_dates,
    )


def build_category_vocabularies(
    metadata: pd.DataFrame,
    *,
    columns: Sequence[str] = STATIC_CATEGORICAL_COLUMNS,
) -> dict[str, dict[str, int]]:
    vocabularies: dict[str, dict[str, int]] = {}
    for column in columns:
        values = sorted(
            {
                str(value)
                for value in metadata[column].dropna().astype(str).tolist()
                if str(value).strip()
            }
        )
        vocabularies[column] = {value: idx + 1 for idx, value in enumerate(values)}
    return vocabularies


def encode_static_categories(
    metadata: pd.DataFrame,
    vocabularies: dict[str, dict[str, int]],
    *,
    columns: Sequence[str] = STATIC_CATEGORICAL_COLUMNS,
) -> dict[str, np.ndarray]:
    encoded: dict[str, np.ndarray] = {}
    for column in columns:
        vocab = vocabularies.get(column, {})
        values = metadata[column].fillna("").astype(str)
        encoded[column] = values.map(lambda value: int(vocab.get(value, 0))).to_numpy(dtype=np.int64)
    return encoded


def build_walk_forward_folds(
    metadata: pd.DataFrame,
    *,
    min_train_dates: int = 252,
    val_block_size: int = 21,
    oos_block_size: int = 21,
    purge_gap: int = 20,
) -> list[WalkForwardFold]:
    dates = sorted(metadata["anchor_date"].dropna().astype(str).unique().tolist())
    folds: list[WalkForwardFold] = []
    oos_start_idx = min_train_dates + val_block_size + purge_gap
    fold_id = 0
    while oos_start_idx + oos_block_size <= len(dates):
        eligible_dates = dates[: max(0, oos_start_idx - purge_gap)]
        if len(eligible_dates) < min_train_dates + val_block_size:
            break
        train_dates = tuple(eligible_dates[:-val_block_size])
        val_dates = tuple(eligible_dates[-val_block_size:])
        oos_dates = tuple(dates[oos_start_idx : oos_start_idx + oos_block_size])
        if len(train_dates) < min_train_dates or not val_dates or not oos_dates:
            break
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_dates=train_dates,
                val_dates=val_dates,
                oos_dates=oos_dates,
                purge_gap=purge_gap,
            )
        )
        fold_id += 1
        oos_start_idx += oos_block_size
    return folds


def rows_for_dates(metadata: pd.DataFrame, dates: Sequence[str]) -> pd.Series:
    return metadata["anchor_date"].astype(str).isin({str(value) for value in dates})


def add_window_summaries(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    prefix: str,
    window_length: int,
) -> pd.DataFrame:
    if not feature_cols:
        return df[["ticker", "date"]].copy()

    source = df[["ticker", "date", *feature_cols]].copy()
    source = source.sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = source.groupby("ticker", sort=False)
    columns: dict[str, pd.Series] = {
        "ticker": source["ticker"],
        "date": source["date"],
    }

    for col in feature_cols:
        series = grouped[col]
        safe_col = col.replace(".", "_")
        columns[f"{prefix}{safe_col}__last"] = source[col]
        columns[f"{prefix}{safe_col}__mean60"] = series.transform(
            lambda values: values.rolling(window_length, min_periods=window_length).mean()
        )
        columns[f"{prefix}{safe_col}__std60"] = series.transform(
            lambda values: values.rolling(window_length, min_periods=window_length).std(ddof=0)
        )
    return pd.DataFrame(columns)


def build_multi_horizon_targets(
    stock_features: pd.DataFrame,
    context_features: pd.DataFrame,
    *,
    horizons: Sequence[int],
    window_length: int,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    classification_horizon: int = DEFAULT_CLASSIFICATION_HORIZON,
    classification_threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> pd.DataFrame:
    benchmark = context_features[context_features["ticker"] == benchmark_ticker.upper()].copy()
    if benchmark.empty:
        raise ValueError(
            f"Benchmark ticker {benchmark_ticker} is missing. Build "
            "processed/market_context_features.csv before training V1 baselines."
        )
    benchmark_close_by_date = benchmark.set_index("date")["close"].astype(float)

    stocks = _filtered_stock_universe(stock_features, benchmark_ticker=benchmark_ticker)

    target_frames: list[pd.DataFrame] = []
    max_horizon = max(horizons)
    classification_col = classification_target_column(
        horizon=classification_horizon,
        threshold=classification_threshold,
    )
    for _, group in stocks.groupby("ticker", sort=False):
        group = group.copy()
        for horizon in horizons:
            future_close = group["close"].shift(-horizon).astype(float)
            future_date = group["date"].shift(-horizon)
            stock_return = future_close / group["close"].astype(float) - 1.0
            benchmark_anchor = group["date"].map(benchmark_close_by_date)
            benchmark_future = future_date.map(benchmark_close_by_date)
            benchmark_return = benchmark_future / benchmark_anchor - 1.0
            group[target_column(horizon)] = stock_return - benchmark_return
            group[f"future_date_{horizon}d"] = future_date
        path_excess_cols: list[str] = []
        benchmark_anchor = group["date"].map(benchmark_close_by_date)
        anchor_close = group["close"].astype(float)
        for path_step in range(1, classification_horizon + 1):
            future_close = group["close"].shift(-path_step).astype(float)
            future_date = group["date"].shift(-path_step)
            stock_return = future_close / anchor_close - 1.0
            benchmark_future = future_date.map(benchmark_close_by_date)
            benchmark_return = benchmark_future / benchmark_anchor - 1.0
            path_col = f"market_adjusted_return_path_{path_step}d"
            group[path_col] = stock_return - benchmark_return
            path_excess_cols.append(path_col)
        group[classification_col] = (group[path_excess_cols].max(axis=1) > float(classification_threshold)).astype(float)
        group["has_all_future_horizons"] = group.groupby("ticker").cumcount() <= len(group) - max_horizon - 1
        target_frames.append(group)

    targets = pd.concat(target_frames, ignore_index=True) if target_frames else pd.DataFrame()
    target_cols = [target_column(h) for h in horizons]
    keep_cols = [
        "ticker",
        "date",
        "gics_sector",
        "gics_sub_industry",
        "window_row_count",
        *target_cols,
        classification_col,
    ]
    targets = targets[keep_cols]
    targets = targets[targets["window_row_count"] >= window_length]
    targets = targets.dropna(subset=target_cols)
    targets = targets.dropna(subset=[classification_col])
    targets = targets.rename(columns={"date": "anchor_date"})
    targets["sector_etf"] = targets["gics_sector"].map(SECTOR_ETF_BY_GICS)
    return targets.reset_index(drop=True)


def _merge_context(
    base: pd.DataFrame,
    context_summary: pd.DataFrame,
    *,
    ticker: str | None,
    ticker_column: str | None,
    prefix: str,
) -> pd.DataFrame:
    if context_summary.empty:
        base[f"{prefix}context_missing"] = 1.0
        return base
    context = context_summary.copy()
    if ticker is not None:
        context = context[context["ticker"] == ticker]
    left = base.copy()
    if ticker_column is None:
        left = left.merge(context.drop(columns=["ticker"]), left_on="anchor_date", right_on="date", how="left")
        left = left.drop(columns=["date"], errors="ignore")
    else:
        context = context.rename(columns={"ticker": "_context_ticker"})
        left = left.merge(
            context,
            left_on=["anchor_date", ticker_column],
            right_on=["date", "_context_ticker"],
            how="left",
            suffixes=("", f"_{prefix}context"),
        )
        left = left.drop(columns=["date", "_context_ticker"], errors="ignore")
    feature_cols = [col for col in left.columns if col.startswith(prefix)]
    left[f"{prefix}context_missing"] = left[feature_cols].isna().all(axis=1).astype(float) if feature_cols else 1.0
    return left


def build_v1_dataset(
    stock_features: pd.DataFrame,
    context_features: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    window_length: int = DEFAULT_WINDOW_LENGTH,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    max_episodes: int | None = None,
    classification_horizon: int = DEFAULT_CLASSIFICATION_HORIZON,
    classification_threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> V1Dataset:
    horizons = tuple(horizons)
    target_cols = [target_column(horizon) for horizon in horizons]
    classification_cols = [
        classification_target_column(horizon=classification_horizon, threshold=classification_threshold)
    ]
    targets = build_multi_horizon_targets(
        stock_features,
        context_features,
        horizons=horizons,
        window_length=window_length,
        benchmark_ticker=benchmark_ticker,
        classification_horizon=classification_horizon,
        classification_threshold=classification_threshold,
    )
    if max_episodes is not None and len(targets) > max_episodes:
        targets = targets.sort_values(["anchor_date", "ticker"]).tail(max_episodes).reset_index(drop=True)

    stock_only_cols = select_stock_feature_columns(stock_features, include_relative=False)
    stock_relative_cols = select_stock_feature_columns(stock_features, include_relative=True)
    context_cols = select_context_feature_columns(context_features)

    stock_only_summary = add_window_summaries(
        stock_features,
        feature_cols=stock_only_cols,
        prefix="stock_",
        window_length=window_length,
    )
    stock_relative_summary = add_window_summaries(
        stock_features,
        feature_cols=stock_relative_cols,
        prefix="stock_",
        window_length=window_length,
    )
    context_summary = add_window_summaries(
        context_features,
        feature_cols=context_cols,
        prefix="context_",
        window_length=window_length,
    )

    metadata = targets[
        ["ticker", "anchor_date", "gics_sector", "gics_sub_industry", "sector_etf", "window_row_count"]
    ].copy()
    stock_only = targets.merge(
        stock_only_summary,
        left_on=["ticker", "anchor_date"],
        right_on=["ticker", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")
    stock_relative = targets.merge(
        stock_relative_summary,
        left_on=["ticker", "anchor_date"],
        right_on=["ticker", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")

    def feature_frame(base: pd.DataFrame, *, include_market: bool, include_sector: bool) -> pd.DataFrame:
        frame = base.copy()
        if include_market:
            market_context = context_summary.add_prefix("market_")
            market_context = market_context.rename(columns={"market_ticker": "ticker", "market_date": "date"})
            frame = _merge_context(frame, market_context, ticker=benchmark_ticker.upper(), ticker_column=None, prefix="market_context_")
        if include_sector:
            sector_context = context_summary.add_prefix("sector_")
            sector_context = sector_context.rename(columns={"sector_ticker": "ticker", "sector_date": "date"})
            frame = _merge_context(frame, sector_context, ticker=None, ticker_column="sector_etf", prefix="sector_context_")
        return frame

    frames = {
        "stock_only": stock_only,
        "stock_relative": stock_relative,
        "stock_relative_market": feature_frame(stock_relative, include_market=True, include_sector=False),
        "stock_relative_market_sector": feature_frame(stock_relative, include_market=True, include_sector=True),
    }

    feature_sets: dict[str, pd.DataFrame] = {}
    feature_columns: dict[str, list[str]] = {}
    non_features = {
        "ticker",
        "anchor_date",
        "gics_sector",
        "gics_sub_industry",
        "sector_etf",
        "window_row_count",
        *target_cols,
    }
    for name, frame in frames.items():
        numeric = [
            col
            for col in frame.columns
            if col not in non_features and pd.api.types.is_numeric_dtype(frame[col])
        ]
        feature_sets[name] = frame[["ticker", "anchor_date", *numeric]].copy()
        feature_columns[name] = numeric

    return V1Dataset(
        metadata=metadata,
        targets=targets[["ticker", "anchor_date", *target_cols, *classification_cols]].copy(),
        feature_sets=feature_sets,
        target_columns=target_cols,
        classification_target_columns=classification_cols,
        feature_columns=feature_columns,
        split_by_date={},
    )


def build_latest_v1_feature_sets(
    stock_features: pd.DataFrame,
    context_features: pd.DataFrame,
    *,
    window_length: int = DEFAULT_WINDOW_LENGTH,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    anchor_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, list[str]]]:
    cutoff = anchor_date
    stocks = stock_features.copy()
    context = context_features.copy()
    if cutoff:
        stocks = stocks[stocks["date"] <= cutoff]
        context = context[context["date"] <= cutoff]

    stocks = _filtered_stock_universe(stocks, benchmark_ticker=benchmark_ticker)
    latest_idx = stocks.groupby("ticker")["date"].idxmax()
    latest = stocks.loc[latest_idx].copy()
    latest = latest[latest["window_row_count"] >= window_length]
    latest = latest.rename(columns={"date": "anchor_date"})
    latest["sector_etf"] = latest["gics_sector"].map(SECTOR_ETF_BY_GICS)
    metadata = latest[
        ["ticker", "anchor_date", "gics_sector", "gics_sub_industry", "sector_etf", "window_row_count"]
    ].reset_index(drop=True)

    stock_only_cols = select_stock_feature_columns(stock_features, include_relative=False)
    stock_relative_cols = select_stock_feature_columns(stock_features, include_relative=True)
    context_cols = select_context_feature_columns(context_features)
    stock_only_summary = add_window_summaries(
        stock_features,
        feature_cols=stock_only_cols,
        prefix="stock_",
        window_length=window_length,
    )
    stock_relative_summary = add_window_summaries(
        stock_features,
        feature_cols=stock_relative_cols,
        prefix="stock_",
        window_length=window_length,
    )
    context_summary = add_window_summaries(
        context_features,
        feature_cols=context_cols,
        prefix="context_",
        window_length=window_length,
    )

    base = metadata.copy()
    stock_only = base.merge(
        stock_only_summary,
        left_on=["ticker", "anchor_date"],
        right_on=["ticker", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")
    stock_relative = base.merge(
        stock_relative_summary,
        left_on=["ticker", "anchor_date"],
        right_on=["ticker", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")

    def feature_frame(base_frame: pd.DataFrame, *, include_market: bool, include_sector: bool) -> pd.DataFrame:
        frame = base_frame.copy()
        if include_market:
            market_context = context_summary.add_prefix("market_")
            market_context = market_context.rename(columns={"market_ticker": "ticker", "market_date": "date"})
            frame = _merge_context(frame, market_context, ticker=benchmark_ticker.upper(), ticker_column=None, prefix="market_context_")
        if include_sector:
            sector_context = context_summary.add_prefix("sector_")
            sector_context = sector_context.rename(columns={"sector_ticker": "ticker", "sector_date": "date"})
            frame = _merge_context(frame, sector_context, ticker=None, ticker_column="sector_etf", prefix="sector_context_")
        return frame

    frames = {
        "stock_only": stock_only,
        "stock_relative": stock_relative,
        "stock_relative_market": feature_frame(stock_relative, include_market=True, include_sector=False),
        "stock_relative_market_sector": feature_frame(stock_relative, include_market=True, include_sector=True),
    }
    non_features = {"ticker", "anchor_date", "gics_sector", "gics_sub_industry", "sector_etf", "window_row_count"}
    feature_sets: dict[str, pd.DataFrame] = {}
    feature_columns: dict[str, list[str]] = {}
    for name, frame in frames.items():
        numeric = [
            col
            for col in frame.columns
            if col not in non_features and pd.api.types.is_numeric_dtype(frame[col])
        ]
        feature_sets[name] = frame[["ticker", "anchor_date", *numeric]].copy()
        feature_columns[name] = numeric
    return metadata, feature_sets, feature_columns


def chronological_split(metadata: pd.DataFrame, train_fraction: float = 0.7, val_fraction: float = 0.15) -> pd.Series:
    dates = sorted(metadata["anchor_date"].dropna().unique().tolist())
    if len(dates) < 3:
        raise ValueError("Need at least three unique anchor dates for chronological train/val/test split.")
    train_end = max(1, int(len(dates) * train_fraction))
    val_end = max(train_end + 1, int(len(dates) * (train_fraction + val_fraction)))
    val_end = min(val_end, len(dates) - 1)
    train_dates = set(dates[:train_end])
    val_dates = set(dates[train_end:val_end])
    split = pd.Series("test", index=metadata.index, dtype="object")
    split[metadata["anchor_date"].isin(train_dates)] = "train"
    split[metadata["anchor_date"].isin(val_dates)] = "val"
    return split


def split_ranges(metadata: pd.DataFrame, split: pd.Series) -> dict[str, dict[str, str | int | None]]:
    ranges: dict[str, dict[str, str | int | None]] = {}
    for split_name in ("train", "val", "test"):
        dates = metadata.loc[split == split_name, "anchor_date"]
        ranges[split_name] = {
            "row_count": int(len(dates)),
            "start_date": str(dates.min()) if len(dates) else None,
            "end_date": str(dates.max()) if len(dates) else None,
        }
    return ranges


def prepare_xy(
    dataset: V1Dataset,
    feature_set: str,
    split: pd.Series,
    split_name: str,
    *,
    task_type: str = "regression",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = dataset.feature_sets[feature_set]
    rows = split == split_name
    meta = dataset.metadata.loc[rows].reset_index(drop=True)
    x = frame.loc[rows, dataset.feature_columns[feature_set]].reset_index(drop=True)
    if task_type == "classification":
        target_columns = dataset.classification_target_columns
    else:
        target_columns = dataset.target_columns
    y = dataset.targets.loc[rows, target_columns].reset_index(drop=True)
    return meta, x, y


def save_dataset_manifest(path: str | Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
