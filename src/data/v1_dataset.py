from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.data.episode_eligibility import (
    EpisodeEligibilityConfig,
    add_episode_eligibility_columns,
    eligibility_metadata_columns,
)
from src.data.eodhd_enrichment import FUNDAMENTAL_FEATURE_COLUMNS, SENTIMENT_FEATURE_COLUMNS


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
MODEL_INPUT_RAW_LEVEL_COLUMNS = set(RAW_LEVEL_COLUMNS)
MODEL_INPUT_IDENTIFIER_COLUMNS = {
    "ticker",
    "symbol",
    "eodhd_symbol",
    "isin",
    "cusip",
    "cik",
    "figi",
    "primary_ticker",
    "name",
}
MODEL_INPUT_SUMMARY_SUFFIXES = ("__last", "__mean60", "__std60")
MODEL_INPUT_PREFIXES = ("stock_", "market_context_", "sector_context_", "context_")

BASE_STOCK_FEATURES = [
    "return_1d",
    "log_return_1d",
    "gap_pct",
    "intraday_return",
    "hl_range_pct",
    "close_location",
    "true_range_pct",
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
    "dollar_volume_ratio_5d",
    "volume_zscore_20d",
    "stock_vs_market_return_1d",
    "stock_vs_sector_return_1d",
    "stock_vs_market_return_5d",
    "stock_vs_sector_return_5d",
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
    "close_location",
    "true_range_pct",
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
    "dollar_volume_ratio_5d",
    "volume_zscore_20d",
]

COMPACT_STOCK_FEATURES = [
    "log_return_1d",
    "gap_pct",
    "intraday_return",
    "hl_range_pct",
    "close_location",
    "true_range_pct",
    "rolling_return_5d",
    "rolling_return_20d",
    "rolling_return_60d",
    "rolling_vol_20d",
    "rolling_vol_60d",
    "price_vs_sma_20d",
    "price_vs_sma_60d",
    "volume_ratio_20d",
    "dollar_volume_ratio_5d",
    "volume_zscore_20d",
    "stock_vs_market_return_1d",
    "stock_vs_sector_return_1d",
    "stock_vs_market_return_5d",
    "stock_vs_sector_return_5d",
    "log1p_dollar_volume",
]

COMPACT_FULL_PANEL_RELATIVE_FEATURES = [
    "log1p_dollar_volume__cs_z",
    "rolling_return_5d__cs_z",
    "rolling_return_20d__cs_z",
    "rolling_return_60d__cs_z",
    "rolling_vol_20d__cs_z",
    "rolling_vol_60d__cs_z",
    "price_vs_sma_20d__cs_z",
    "price_vs_sma_60d__cs_z",
    "volume_ratio_20d__cs_z",
]

COMPACT_SECTOR_RELATIVE_FEATURES = [
    "log1p_dollar_volume__sector_cs_z",
    "rolling_return_20d__sector_cs_z",
    "rolling_return_60d__sector_cs_z",
    "rolling_vol_20d__sector_cs_z",
    "rolling_vol_60d__sector_cs_z",
    "volume_ratio_20d__sector_cs_z",
]

COMPACT_CONTEXT_FEATURES = [
    "log_return_1d",
    "rolling_return_5d",
    "rolling_return_20d",
    "rolling_return_60d",
    "rolling_vol_20d",
    "rolling_vol_60d",
    "close_location",
    "true_range_pct",
    "price_vs_sma_20d",
    "price_vs_sma_60d",
    "volume_ratio_20d",
    "dollar_volume_ratio_5d",
    "volume_zscore_20d",
]

FEATURE_SET_NAMES = (
    "stock_only",
    "stock_relative",
    "stock_relative_market",
    "stock_relative_market_sector",
    "stock_compact",
    "stock_relative_compact",
    "stock_relative_market_compact",
    "stock_relative_market_sector_compact",
    "stock_only_sentiment",
    "stock_relative_market_sector_sentiment",
    "stock_only_fundamentals",
    "stock_relative_market_sector_fundamentals",
    "stock_only_fundamentals_sentiment",
    "stock_relative_market_sector_fundamentals_sentiment",
)
FULL_SEQUENCE_BASE_FEATURE_SET_NAMES = (
    "stock_only",
    "stock_relative",
    "stock_market",
    "stock_sector",
    "stock_market_sector",
    "stock_relative_market",
    "stock_relative_sector",
    "stock_relative_market_sector",
    "stock_sentiment",
    "stock_relative_market_sector_sentiment",
)
COMPACT_SEQUENCE_BASE_FEATURE_SET_NAMES = (
    "stock_compact",
    "stock_relative_compact",
    "stock_market_compact",
    "stock_sector_compact",
    "stock_market_sector_compact",
    "stock_relative_market_compact",
    "stock_relative_sector_compact",
    "stock_relative_market_sector_compact",
)
SEQUENCE_BASE_FEATURE_SET_NAMES = (
    *FULL_SEQUENCE_BASE_FEATURE_SET_NAMES,
    *COMPACT_SEQUENCE_BASE_FEATURE_SET_NAMES,
)
SEQUENCE_FEATURE_SET_NAMES = (
    *SEQUENCE_BASE_FEATURE_SET_NAMES,
    *(f"{name}_sequence" for name in SEQUENCE_BASE_FEATURE_SET_NAMES),
)
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


@dataclass(frozen=True)
class SequenceFeatureConfig:
    include_relative: bool
    include_market_context: bool
    include_sector_context: bool
    include_sentiment: bool
    compact: bool

    def to_dict(self) -> dict[str, bool]:
        return {
            "stock_features": True,
            "relative_stock_features": self.include_relative,
            "market_context_features": self.include_market_context,
            "sector_context_features": self.include_sector_context,
            "sentiment_features": self.include_sentiment,
            "compact_feature_profile": self.compact,
        }


def sequence_feature_config(feature_set: str) -> SequenceFeatureConfig:
    if feature_set not in SEQUENCE_FEATURE_SET_NAMES:
        raise ValueError(f"Sequence inputs are only supported for {SEQUENCE_FEATURE_SET_NAMES}, got {feature_set}.")
    canonical = feature_set.removesuffix("_sequence")
    parts = set(canonical.split("_"))
    return SequenceFeatureConfig(
        include_relative="relative" in parts,
        include_market_context="market" in parts,
        include_sector_context="sector" in parts,
        include_sentiment="sentiment" in parts,
        compact="compact" in parts,
    )


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


def _canonical_model_feature_name(column: str) -> str:
    name = str(column)
    for suffix in MODEL_INPUT_SUMMARY_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    for prefix in MODEL_INPUT_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name


def raw_level_model_input_columns(columns: Sequence[str]) -> list[str]:
    return [
        str(column)
        for column in columns
        if _canonical_model_feature_name(str(column)) in MODEL_INPUT_RAW_LEVEL_COLUMNS
    ]


def identifier_model_input_columns(columns: Sequence[str]) -> list[str]:
    return [
        str(column)
        for column in columns
        if _canonical_model_feature_name(str(column)) in MODEL_INPUT_IDENTIFIER_COLUMNS
    ]


def validate_model_feature_columns(columns: Sequence[str], *, feature_set: str) -> None:
    raw_columns = raw_level_model_input_columns(columns)
    if raw_columns:
        preview = ", ".join(raw_columns[:10])
        extra = "" if len(raw_columns) <= 10 else f", ... ({len(raw_columns)} total)"
        raise ValueError(
            f"Raw level columns are not allowed in model inputs for {feature_set}: {preview}{extra}. "
            "Use log-scaled, ratio, return, z-score, or percentile features instead."
        )
    identifier_columns = identifier_model_input_columns(columns)
    if identifier_columns:
        preview = ", ".join(identifier_columns[:10])
        extra = "" if len(identifier_columns) <= 10 else f", ... ({len(identifier_columns)} total)"
        raise ValueError(
            f"Identifier columns are not allowed in model inputs for {feature_set}: {preview}{extra}. "
            "Keep identifiers in metadata only so models can score unseen future tickers."
        )


def add_priority_a_ohlcv_features(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    if df.empty:
        return df
    df["_row_order"] = np.arange(len(df))
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if "close_location" not in df.columns and {"high", "low", "close"}.issubset(df.columns):
        spread = df["high"].astype(float) - df["low"].astype(float)
        df["close_location"] = np.where(
            spread != 0.0,
            (df["close"].astype(float) - df["low"].astype(float)) / spread,
            np.nan,
        )

    if "true_range_pct" not in df.columns and {"high", "low", "close"}.issubset(df.columns):
        prev_close = (
            df["prev_close"].astype(float)
            if "prev_close" in df.columns
            else df.groupby("ticker", sort=False)["close"].shift(1).astype(float)
        )
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["true_range_pct"] = np.where(prev_close != 0.0, true_range / prev_close, np.nan)

    if "dollar_volume_ratio_5d" not in df.columns and "dollar_volume" in df.columns:
        dollar_volume = df["dollar_volume"].astype(float)
        avg_5d = dollar_volume.groupby(df["ticker"], sort=False).transform(
            lambda values: values.rolling(5, min_periods=5).mean()
        )
        df["dollar_volume_ratio_5d"] = np.where(avg_5d != 0.0, dollar_volume / avg_5d, np.nan)

    if "volume_zscore_20d" not in df.columns and "volume" in df.columns:
        volume = df["volume"].astype(float)
        rolling_mean = volume.groupby(df["ticker"], sort=False).transform(
            lambda values: values.rolling(20, min_periods=20).mean()
        )
        rolling_std = volume.groupby(df["ticker"], sort=False).transform(
            lambda values: values.rolling(20, min_periods=20).std(ddof=0)
        )
        df["volume_zscore_20d"] = np.where(rolling_std != 0.0, (volume - rolling_mean) / rolling_std, np.nan)

    df = df.sort_values("_row_order").drop(columns=["_row_order"]).reset_index(drop=True)
    return df


def add_context_relative_return_features(
    stock_features: pd.DataFrame,
    context_features: pd.DataFrame,
    *,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
) -> pd.DataFrame:
    stocks = add_priority_a_ohlcv_features(stock_features)
    context = add_priority_a_ohlcv_features(context_features)
    if stocks.empty or context.empty:
        return stocks

    context = context.copy()
    context["ticker"] = context["ticker"].astype(str).str.upper()
    context["date"] = context["date"].astype(str)
    stocks["date"] = stocks["date"].astype(str)

    benchmark = context[context["ticker"] == benchmark_ticker.upper()].set_index("date")
    if "return_1d" in stocks.columns and "return_1d" in benchmark.columns:
        stocks["stock_vs_market_return_1d"] = stocks["return_1d"].astype(float) - stocks["date"].map(
            benchmark["return_1d"].astype(float)
        )
    if "rolling_return_5d" in stocks.columns and "rolling_return_5d" in benchmark.columns:
        stocks["stock_vs_market_return_5d"] = stocks["rolling_return_5d"].astype(float) - stocks["date"].map(
            benchmark["rolling_return_5d"].astype(float)
        )

    if "gics_sector" not in stocks.columns:
        return stocks
    stocks["sector_etf"] = stocks["gics_sector"].map(SECTOR_ETF_BY_GICS)
    sector_cols = ["ticker", "date"]
    for col in ("return_1d", "rolling_return_5d"):
        if col in context.columns:
            sector_cols.append(col)
    if len(sector_cols) <= 2:
        return stocks
    sector_context = context[context["ticker"].isin(set(SECTOR_ETF_BY_GICS.values()))][sector_cols].copy()
    sector_context = sector_context.rename(
        columns={
            "ticker": "_sector_etf",
            "return_1d": "_sector_return_1d",
            "rolling_return_5d": "_sector_return_5d",
        }
    )
    stocks = stocks.merge(
        sector_context,
        left_on=["date", "sector_etf"],
        right_on=["date", "_sector_etf"],
        how="left",
    )
    if "return_1d" in stocks.columns and "_sector_return_1d" in stocks.columns:
        stocks["stock_vs_sector_return_1d"] = stocks["return_1d"].astype(float) - stocks["_sector_return_1d"].astype(float)
    if "rolling_return_5d" in stocks.columns and "_sector_return_5d" in stocks.columns:
        stocks["stock_vs_sector_return_5d"] = stocks["rolling_return_5d"].astype(float) - stocks["_sector_return_5d"].astype(float)
    return stocks.drop(columns=["_sector_etf", "_sector_return_1d", "_sector_return_5d"], errors="ignore")


def select_stock_feature_columns(df: pd.DataFrame, include_relative: bool, *, compact: bool = False) -> list[str]:
    cols = _available_numeric_columns(df, COMPACT_STOCK_FEATURES if compact else BASE_STOCK_FEATURES)
    if include_relative:
        if compact:
            relative = _available_numeric_columns(
                df,
                [*COMPACT_FULL_PANEL_RELATIVE_FEATURES, *COMPACT_SECTOR_RELATIVE_FEATURES],
            )
        else:
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


def select_augmented_stock_feature_columns(
    df: pd.DataFrame,
    include_relative: bool,
    *,
    compact: bool = False,
    include_sentiment: bool = False,
    include_fundamentals: bool = False,
) -> list[str]:
    cols = select_stock_feature_columns(df, include_relative=include_relative, compact=compact)
    if include_sentiment:
        cols.extend(_available_numeric_columns(df, SENTIMENT_FEATURE_COLUMNS))
    if include_fundamentals:
        cols.extend(_available_numeric_columns(df, FUNDAMENTAL_FEATURE_COLUMNS))
    return list(dict.fromkeys(cols))


def select_context_feature_columns(df: pd.DataFrame, *, compact: bool = False) -> list[str]:
    return _available_numeric_columns(df, COMPACT_CONTEXT_FEATURES if compact else CONTEXT_FEATURES)


def select_sequence_feature_columns(
    stock_features: pd.DataFrame,
    feature_set: str,
    context_features: pd.DataFrame | None = None,
) -> list[str]:
    config = sequence_feature_config(feature_set)
    cols = select_augmented_stock_feature_columns(
        stock_features,
        include_relative=config.include_relative,
        compact=config.compact,
        include_sentiment=config.include_sentiment,
    )
    if config.include_market_context or config.include_sector_context:
        if context_features is None:
            raise ValueError(f"{feature_set} sequence inputs require context_features.")
        context_cols = select_context_feature_columns(context_features, compact=config.compact)
        if config.include_market_context:
            cols.extend(f"market_context_{col}" for col in context_cols)
            cols.append("market_context_missing")
        if config.include_sector_context:
            cols.extend(f"sector_context_{col}" for col in context_cols)
            cols.append("sector_context_missing")
    return list(dict.fromkeys(cols))


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
    context_features: pd.DataFrame | None = None,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    feature_columns: Sequence[str] | None = None,
) -> SequenceFeatureStore:
    config = sequence_feature_config(feature_set)
    if context_features is not None:
        stock_features = add_context_relative_return_features(
            stock_features,
            context_features,
            benchmark_ticker=benchmark_ticker,
        )
        context_features = add_priority_a_ohlcv_features(context_features)
    else:
        stock_features = add_priority_a_ohlcv_features(stock_features)
    stocks = _filtered_stock_universe(stock_features, benchmark_ticker=benchmark_ticker)
    stock_cols = select_augmented_stock_feature_columns(
        stocks,
        include_relative=config.include_relative,
        compact=config.compact,
        include_sentiment=config.include_sentiment,
    )
    frame = stocks[["ticker", "date", "gics_sector", *stock_cols]].copy()
    frame["sector_etf"] = frame["gics_sector"].map(SECTOR_ETF_BY_GICS)
    generated_feature_columns = list(stock_cols)

    if config.include_market_context or config.include_sector_context:
        if context_features is None:
            raise ValueError(f"{feature_set} sequence inputs require context_features.")
        context = context_features.copy()
        context["ticker"] = context["ticker"].astype(str).str.upper()
        context["date"] = context["date"].astype(str)
        context_cols = select_context_feature_columns(context, compact=config.compact)
        if not context_cols:
            raise ValueError(f"No context feature columns available for {feature_set}.")
        if config.include_market_context:
            market = context[context["ticker"] == benchmark_ticker.upper()][["date", *context_cols]].copy()
            market = market.rename(columns={col: f"market_context_{col}" for col in context_cols})
            frame = frame.merge(market, on="date", how="left")
            market_cols = [f"market_context_{col}" for col in context_cols]
            frame["market_context_missing"] = frame[market_cols].isna().all(axis=1).astype(float)
            generated_feature_columns.extend([*market_cols, "market_context_missing"])
        if config.include_sector_context:
            sector = context[context["ticker"].isin(set(SECTOR_ETF_BY_GICS.values()))][
                ["ticker", "date", *context_cols]
            ].copy()
            sector = sector.rename(
                columns={
                    "ticker": "_sector_context_ticker",
                    **{col: f"sector_context_{col}" for col in context_cols},
                }
            )
            frame = frame.merge(
                sector,
                left_on=["date", "sector_etf"],
                right_on=["date", "_sector_context_ticker"],
                how="left",
            ).drop(columns=["_sector_context_ticker"], errors="ignore")
            sector_cols = [f"sector_context_{col}" for col in context_cols]
            frame["sector_context_missing"] = frame[sector_cols].isna().all(axis=1).astype(float)
            generated_feature_columns.extend([*sector_cols, "sector_context_missing"])

    resolved_feature_columns = list(feature_columns) if feature_columns is not None else generated_feature_columns
    if not resolved_feature_columns:
        raise ValueError(f"No sequence feature columns available for {feature_set}.")
    validate_model_feature_columns(resolved_feature_columns, feature_set=feature_set)
    ticker_arrays: dict[str, np.ndarray] = {}
    ticker_dates: dict[str, np.ndarray] = {}
    for ticker, group in frame.groupby("ticker", sort=False):
        group_values = group.reindex(columns=resolved_feature_columns, fill_value=0.0).fillna(0.0)
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
    eligibility_config: EpisodeEligibilityConfig | None = None,
) -> pd.DataFrame:
    benchmark = context_features[context_features["ticker"] == benchmark_ticker.upper()].copy()
    if benchmark.empty:
        raise ValueError(
            f"Benchmark ticker {benchmark_ticker} is missing. Build "
            "processed/market_context_features.csv before training V1 baselines."
        )
    benchmark_close_by_date = benchmark.set_index("date")["close"].astype(float)

    stocks = _filtered_stock_universe(stock_features, benchmark_ticker=benchmark_ticker)
    if eligibility_config is not None:
        stocks = add_episode_eligibility_columns(
            stocks,
            eligibility_config,
            benchmark_ticker=benchmark_ticker,
        )

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
        *eligibility_metadata_columns(targets),
        *target_cols,
        classification_col,
    ]
    targets = targets[keep_cols]
    targets = targets[targets["window_row_count"] >= window_length]
    if eligibility_config is not None and "episode_eligible" in targets.columns:
        targets = targets[targets["episode_eligible"]]
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
    eligibility_config: EpisodeEligibilityConfig | None = None,
    feature_set_names: Sequence[str] | None = None,
) -> V1Dataset:
    stock_features = add_context_relative_return_features(
        stock_features,
        context_features,
        benchmark_ticker=benchmark_ticker,
    )
    context_features = add_priority_a_ohlcv_features(context_features)
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
        eligibility_config=eligibility_config,
    )
    if max_episodes is not None and len(targets) > max_episodes:
        targets = targets.sort_values(["anchor_date", "ticker"]).tail(max_episodes).reset_index(drop=True)

    metadata_columns = [
        "ticker",
        "anchor_date",
        "gics_sector",
        "gics_sub_industry",
        "sector_etf",
        "window_row_count",
        *eligibility_metadata_columns(targets),
    ]
    metadata = targets[metadata_columns].copy()

    requested_tabular = set(feature_set_names or FEATURE_SET_NAMES).intersection(FEATURE_SET_NAMES)
    stock_summary_cache: dict[str, pd.DataFrame] = {}
    context_summary_cache: dict[str, pd.DataFrame] = {}

    def stock_summary(kind: str) -> pd.DataFrame:
        # Large EODHD panels cannot afford every summary family at once.
        # Build only the base summary needed by the active profile feature set.
        if kind not in stock_summary_cache:
            if kind == "stock_only":
                cols = select_stock_feature_columns(stock_features, include_relative=False)
            elif kind == "stock_relative":
                cols = select_stock_feature_columns(stock_features, include_relative=True)
            elif kind == "stock_compact":
                cols = select_stock_feature_columns(stock_features, include_relative=False, compact=True)
            elif kind == "stock_relative_compact":
                cols = select_stock_feature_columns(stock_features, include_relative=True, compact=True)
            elif kind == "stock_sentiment":
                cols = select_augmented_stock_feature_columns(stock_features, include_relative=False, include_sentiment=True)
            elif kind == "stock_relative_sentiment":
                cols = select_augmented_stock_feature_columns(stock_features, include_relative=True, include_sentiment=True)
            elif kind == "stock_fundamental":
                cols = select_augmented_stock_feature_columns(stock_features, include_relative=False, include_fundamentals=True)
            elif kind == "stock_relative_fundamental":
                cols = select_augmented_stock_feature_columns(stock_features, include_relative=True, include_fundamentals=True)
            elif kind == "stock_fundamental_sentiment":
                cols = select_augmented_stock_feature_columns(
                    stock_features,
                    include_relative=False,
                    include_sentiment=True,
                    include_fundamentals=True,
                )
            elif kind == "stock_relative_fundamental_sentiment":
                cols = select_augmented_stock_feature_columns(
                    stock_features,
                    include_relative=True,
                    include_sentiment=True,
                    include_fundamentals=True,
                )
            else:
                raise ValueError(f"Unknown stock summary kind: {kind}")
            stock_summary_cache[kind] = add_window_summaries(
                stock_features,
                feature_cols=cols,
                prefix="stock_",
                window_length=window_length,
            )
        return stock_summary_cache[kind]

    def context_summary(compact: bool) -> pd.DataFrame:
        key = "compact" if compact else "full"
        if key not in context_summary_cache:
            cols = select_context_feature_columns(context_features, compact=compact)
            context_summary_cache[key] = add_window_summaries(
                context_features,
                feature_cols=cols,
                prefix="context_",
                window_length=window_length,
            )
        return context_summary_cache[key]

    def stock_frame(kind: str) -> pd.DataFrame:
        return targets.merge(
            stock_summary(kind),
            left_on=["ticker", "anchor_date"],
            right_on=["ticker", "date"],
            how="left",
        ).drop(columns=["date"], errors="ignore")

    def feature_frame(
        base: pd.DataFrame,
        *,
        context_summary_frame: pd.DataFrame,
        include_market: bool,
        include_sector: bool,
    ) -> pd.DataFrame:
        frame = base.copy()
        if include_market:
            market_context = context_summary_frame.add_prefix("market_")
            market_context = market_context.rename(columns={"market_ticker": "ticker", "market_date": "date"})
            frame = _merge_context(frame, market_context, ticker=benchmark_ticker.upper(), ticker_column=None, prefix="market_context_")
        if include_sector:
            sector_context = context_summary_frame.add_prefix("sector_")
            sector_context = sector_context.rename(columns={"sector_ticker": "ticker", "sector_date": "date"})
            frame = _merge_context(frame, sector_context, ticker=None, ticker_column="sector_etf", prefix="sector_context_")
        return frame

    frame_specs = {
        "stock_only": ("stock_only", False, False, False),
        "stock_relative": ("stock_relative", False, False, False),
        "stock_relative_market": ("stock_relative", False, True, False),
        "stock_relative_market_sector": ("stock_relative", False, True, True),
        "stock_compact": ("stock_compact", False, False, False),
        "stock_relative_compact": ("stock_relative_compact", False, False, False),
        "stock_relative_market_compact": ("stock_relative_compact", True, True, False),
        "stock_relative_market_sector_compact": ("stock_relative_compact", True, True, True),
        "stock_only_sentiment": ("stock_sentiment", False, False, False),
        "stock_relative_market_sector_sentiment": ("stock_relative_sentiment", False, True, True),
        "stock_only_fundamentals": ("stock_fundamental", False, False, False),
        "stock_relative_market_sector_fundamentals": ("stock_relative_fundamental", False, True, True),
        "stock_only_fundamentals_sentiment": ("stock_fundamental_sentiment", False, False, False),
        "stock_relative_market_sector_fundamentals_sentiment": (
            "stock_relative_fundamental_sentiment",
            False,
            True,
            True,
        ),
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
        *eligibility_metadata_columns(targets),
        *target_cols,
        *classification_cols,
    }
    for name in FEATURE_SET_NAMES:
        if name not in requested_tabular:
            continue
        stock_kind, compact_context, include_market, include_sector = frame_specs[name]
        base_frame = stock_frame(stock_kind)
        frame = (
            feature_frame(
                base_frame,
                context_summary_frame=context_summary(compact_context),
                include_market=include_market,
                include_sector=include_sector,
            )
            if include_market or include_sector
            else base_frame
        )
        numeric = [
            col
            for col in frame.columns
            if col not in non_features and pd.api.types.is_numeric_dtype(frame[col])
        ]
        validate_model_feature_columns(numeric, feature_set=name)
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
    eligibility_config: EpisodeEligibilityConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, list[str]]]:
    cutoff = anchor_date
    stocks = stock_features.copy()
    context = context_features.copy()
    if cutoff:
        stocks = stocks[stocks["date"] <= cutoff]
        context = context[context["date"] <= cutoff]

    stocks = add_context_relative_return_features(
        stocks,
        context,
        benchmark_ticker=benchmark_ticker,
    )
    context = add_priority_a_ohlcv_features(context)

    stocks = _filtered_stock_universe(stocks, benchmark_ticker=benchmark_ticker)
    if eligibility_config is not None:
        stocks = add_episode_eligibility_columns(
            stocks,
            eligibility_config,
            benchmark_ticker=benchmark_ticker,
        )
    latest_idx = stocks.groupby("ticker")["date"].idxmax()
    latest = stocks.loc[latest_idx].copy()
    latest = latest[latest["window_row_count"] >= window_length]
    if eligibility_config is not None and "episode_eligible" in latest.columns:
        latest = latest[latest["episode_eligible"]]
    latest = latest.rename(columns={"date": "anchor_date"})
    latest["sector_etf"] = latest["gics_sector"].map(SECTOR_ETF_BY_GICS)
    metadata_columns = [
        "ticker",
        "anchor_date",
        "gics_sector",
        "gics_sub_industry",
        "sector_etf",
        "window_row_count",
        *eligibility_metadata_columns(latest),
    ]
    metadata = latest[metadata_columns].reset_index(drop=True)

    stock_only_cols = select_stock_feature_columns(stocks, include_relative=False)
    stock_relative_cols = select_stock_feature_columns(stocks, include_relative=True)
    stock_compact_cols = select_stock_feature_columns(stocks, include_relative=False, compact=True)
    stock_relative_compact_cols = select_stock_feature_columns(stocks, include_relative=True, compact=True)
    stock_sentiment_cols = select_augmented_stock_feature_columns(stocks, include_relative=False, include_sentiment=True)
    stock_relative_sentiment_cols = select_augmented_stock_feature_columns(stocks, include_relative=True, include_sentiment=True)
    stock_fundamental_cols = select_augmented_stock_feature_columns(stocks, include_relative=False, include_fundamentals=True)
    stock_relative_fundamental_cols = select_augmented_stock_feature_columns(stocks, include_relative=True, include_fundamentals=True)
    stock_fundamental_sentiment_cols = select_augmented_stock_feature_columns(
        stocks,
        include_relative=False,
        include_sentiment=True,
        include_fundamentals=True,
    )
    stock_relative_fundamental_sentiment_cols = select_augmented_stock_feature_columns(
        stocks,
        include_relative=True,
        include_sentiment=True,
        include_fundamentals=True,
    )
    context_cols = select_context_feature_columns(context)
    compact_context_cols = select_context_feature_columns(context, compact=True)
    stock_only_summary = add_window_summaries(
        stocks,
        feature_cols=stock_only_cols,
        prefix="stock_",
        window_length=window_length,
    )
    stock_relative_summary = add_window_summaries(
        stocks,
        feature_cols=stock_relative_cols,
        prefix="stock_",
        window_length=window_length,
    )
    stock_compact_summary = add_window_summaries(
        stocks,
        feature_cols=stock_compact_cols,
        prefix="stock_",
        window_length=window_length,
    )
    stock_relative_compact_summary = add_window_summaries(
        stocks,
        feature_cols=stock_relative_compact_cols,
        prefix="stock_",
        window_length=window_length,
    )
    stock_sentiment_summary = add_window_summaries(
        stocks,
        feature_cols=stock_sentiment_cols,
        prefix="stock_",
        window_length=window_length,
    )
    stock_relative_sentiment_summary = add_window_summaries(
        stocks,
        feature_cols=stock_relative_sentiment_cols,
        prefix="stock_",
        window_length=window_length,
    )
    stock_fundamental_summary = add_window_summaries(
        stocks,
        feature_cols=stock_fundamental_cols,
        prefix="stock_",
        window_length=window_length,
    )
    stock_relative_fundamental_summary = add_window_summaries(
        stocks,
        feature_cols=stock_relative_fundamental_cols,
        prefix="stock_",
        window_length=window_length,
    )
    stock_fundamental_sentiment_summary = add_window_summaries(
        stocks,
        feature_cols=stock_fundamental_sentiment_cols,
        prefix="stock_",
        window_length=window_length,
    )
    stock_relative_fundamental_sentiment_summary = add_window_summaries(
        stocks,
        feature_cols=stock_relative_fundamental_sentiment_cols,
        prefix="stock_",
        window_length=window_length,
    )
    context_summary = add_window_summaries(
        context,
        feature_cols=context_cols,
        prefix="context_",
        window_length=window_length,
    )
    compact_context_summary = add_window_summaries(
        context,
        feature_cols=compact_context_cols,
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
    stock_compact = base.merge(
        stock_compact_summary,
        left_on=["ticker", "anchor_date"],
        right_on=["ticker", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")
    stock_relative_compact = base.merge(
        stock_relative_compact_summary,
        left_on=["ticker", "anchor_date"],
        right_on=["ticker", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")
    stock_sentiment = base.merge(
        stock_sentiment_summary,
        left_on=["ticker", "anchor_date"],
        right_on=["ticker", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")
    stock_relative_sentiment = base.merge(
        stock_relative_sentiment_summary,
        left_on=["ticker", "anchor_date"],
        right_on=["ticker", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")
    stock_fundamental = base.merge(
        stock_fundamental_summary,
        left_on=["ticker", "anchor_date"],
        right_on=["ticker", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")
    stock_relative_fundamental = base.merge(
        stock_relative_fundamental_summary,
        left_on=["ticker", "anchor_date"],
        right_on=["ticker", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")
    stock_fundamental_sentiment = base.merge(
        stock_fundamental_sentiment_summary,
        left_on=["ticker", "anchor_date"],
        right_on=["ticker", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")
    stock_relative_fundamental_sentiment = base.merge(
        stock_relative_fundamental_sentiment_summary,
        left_on=["ticker", "anchor_date"],
        right_on=["ticker", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")

    def feature_frame(
        base_frame: pd.DataFrame,
        *,
        context_summary_frame: pd.DataFrame,
        include_market: bool,
        include_sector: bool,
    ) -> pd.DataFrame:
        frame = base_frame.copy()
        if include_market:
            market_context = context_summary_frame.add_prefix("market_")
            market_context = market_context.rename(columns={"market_ticker": "ticker", "market_date": "date"})
            frame = _merge_context(frame, market_context, ticker=benchmark_ticker.upper(), ticker_column=None, prefix="market_context_")
        if include_sector:
            sector_context = context_summary_frame.add_prefix("sector_")
            sector_context = sector_context.rename(columns={"sector_ticker": "ticker", "sector_date": "date"})
            frame = _merge_context(frame, sector_context, ticker=None, ticker_column="sector_etf", prefix="sector_context_")
        return frame

    frames = {
        "stock_only": stock_only,
        "stock_relative": stock_relative,
        "stock_relative_market": feature_frame(
            stock_relative,
            context_summary_frame=context_summary,
            include_market=True,
            include_sector=False,
        ),
        "stock_relative_market_sector": feature_frame(
            stock_relative,
            context_summary_frame=context_summary,
            include_market=True,
            include_sector=True,
        ),
        "stock_compact": stock_compact,
        "stock_relative_compact": stock_relative_compact,
        "stock_relative_market_compact": feature_frame(
            stock_relative_compact,
            context_summary_frame=compact_context_summary,
            include_market=True,
            include_sector=False,
        ),
        "stock_relative_market_sector_compact": feature_frame(
            stock_relative_compact,
            context_summary_frame=compact_context_summary,
            include_market=True,
            include_sector=True,
        ),
        "stock_only_sentiment": stock_sentiment,
        "stock_relative_market_sector_sentiment": feature_frame(
            stock_relative_sentiment,
            context_summary_frame=context_summary,
            include_market=True,
            include_sector=True,
        ),
        "stock_only_fundamentals": stock_fundamental,
        "stock_relative_market_sector_fundamentals": feature_frame(
            stock_relative_fundamental,
            context_summary_frame=context_summary,
            include_market=True,
            include_sector=True,
        ),
        "stock_only_fundamentals_sentiment": stock_fundamental_sentiment,
        "stock_relative_market_sector_fundamentals_sentiment": feature_frame(
            stock_relative_fundamental_sentiment,
            context_summary_frame=context_summary,
            include_market=True,
            include_sector=True,
        ),
    }
    non_features = {
        "ticker",
        "anchor_date",
        "gics_sector",
        "gics_sub_industry",
        "sector_etf",
        "window_row_count",
        *eligibility_metadata_columns(metadata),
    }
    feature_sets: dict[str, pd.DataFrame] = {}
    feature_columns: dict[str, list[str]] = {}
    for name, frame in frames.items():
        numeric = [
            col
            for col in frame.columns
            if col not in non_features and pd.api.types.is_numeric_dtype(frame[col])
        ]
        validate_model_feature_columns(numeric, feature_set=name)
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
