from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_ALLOWED_EXCHANGES = ("NYSE", "NASDAQ", "AMEX", "BATS")
ELIGIBILITY_DIAGNOSTIC_COLUMNS = (
    "eligibility_common_equity_ok",
    "eligibility_history_ok",
    "eligibility_valid_ohlcv_ok",
    "eligibility_liquidity_ok",
    "eligibility_price_ok",
    "eligibility_exchange_ok",
    "eligibility_valid_ohlcv_rows",
    "eligibility_avg_dollar_volume",
    "episode_eligible",
)

_EXCHANGE_ALIASES = {
    "NYSE AMERICAN": "AMEX",
    "NYSE MKT": "AMEX",
    "NYSEMKT": "AMEX",
}

_COMMON_EQUITY_VALUES = {
    "COMMON",
    "COMMON STOCK",
    "COMMON SHARE",
    "COMMON SHARES",
    "COMMON EQUITY",
    "CS",
    "EQUITY",
}

_NON_COMMON_MARKERS = (
    "ETF",
    "FUND",
    "TRUST",
    "ETN",
    "WARRANT",
    "WARRANTS",
    "UNIT",
    "UNITS",
    "RIGHT",
    "RIGHTS",
    "PREFERRED",
    "PREFERENCE",
)


def _canonical_exchange(value: object) -> str:
    exchange = str(value or "").strip().upper()
    return _EXCHANGE_ALIASES.get(exchange, exchange)


def parse_allowed_exchanges(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_ALLOWED_EXCHANGES
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",")]
    else:
        parts = [str(item).strip() for item in value]
    return tuple(dict.fromkeys(_canonical_exchange(item) for item in parts if item.strip()))


@dataclass(frozen=True)
class EpisodeEligibilityConfig:
    enabled: bool = True
    min_history_days: int = 60
    valid_ohlcv_lookback: int = 60
    min_valid_ohlcv_days: int = 55
    dollar_volume_lookback: int = 60
    min_avg_dollar_volume: float = 100_000.0
    min_price: float = 1.0
    allowed_exchanges: tuple[str, ...] = DEFAULT_ALLOWED_EXCHANGES

    def __post_init__(self) -> None:
        if self.min_history_days < 1:
            raise ValueError("min_history_days must be >= 1.")
        if self.valid_ohlcv_lookback < 1:
            raise ValueError("valid_ohlcv_lookback must be >= 1.")
        if self.min_valid_ohlcv_days < 1:
            raise ValueError("min_valid_ohlcv_days must be >= 1.")
        if self.min_valid_ohlcv_days > self.valid_ohlcv_lookback:
            raise ValueError("min_valid_ohlcv_days cannot exceed valid_ohlcv_lookback.")
        if self.dollar_volume_lookback < 1:
            raise ValueError("dollar_volume_lookback must be >= 1.")
        object.__setattr__(self, "allowed_exchanges", parse_allowed_exchanges(self.allowed_exchanges))

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "min_history_days": self.min_history_days,
            "valid_ohlcv_lookback": self.valid_ohlcv_lookback,
            "min_valid_ohlcv_days": self.min_valid_ohlcv_days,
            "dollar_volume_lookback": self.dollar_volume_lookback,
            "min_avg_dollar_volume": self.min_avg_dollar_volume,
            "min_price": self.min_price,
            "allowed_exchanges": list(self.allowed_exchanges),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "EpisodeEligibilityConfig | None":
        if not payload or payload.get("enabled") is False:
            return None
        return cls(
            enabled=bool(payload.get("enabled", True)),
            min_history_days=int(payload.get("min_history_days", cls.min_history_days)),
            valid_ohlcv_lookback=int(payload.get("valid_ohlcv_lookback", cls.valid_ohlcv_lookback)),
            min_valid_ohlcv_days=int(payload.get("min_valid_ohlcv_days", cls.min_valid_ohlcv_days)),
            dollar_volume_lookback=int(payload.get("dollar_volume_lookback", cls.dollar_volume_lookback)),
            min_avg_dollar_volume=float(payload.get("min_avg_dollar_volume", cls.min_avg_dollar_volume)),
            min_price=float(payload.get("min_price", cls.min_price)),
            allowed_exchanges=parse_allowed_exchanges(payload.get("allowed_exchanges", cls.allowed_exchanges)),
        )


def eligibility_metadata_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in ELIGIBILITY_DIAGNOSTIC_COLUMNS if col in frame.columns]


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _common_equity_mask(frame: pd.DataFrame) -> pd.Series:
    type_columns = [
        col
        for col in ("type", "security_type", "asset_type", "eodhd_type")
        if col in frame.columns
    ]
    if not type_columns:
        return pd.Series(True, index=frame.index, dtype=bool)

    mask = pd.Series(True, index=frame.index, dtype=bool)
    for col in type_columns:
        values = frame[col].fillna("").astype(str).str.strip().str.upper()
        known = values != ""
        common = values.isin(_COMMON_EQUITY_VALUES)
        non_common = values.apply(lambda item: any(marker in item for marker in _NON_COMMON_MARKERS))
        mask &= (~known) | (common & ~non_common)
    return mask


def add_episode_eligibility_columns(
    feature_rows: pd.DataFrame,
    config: EpisodeEligibilityConfig,
    *,
    benchmark_ticker: str = "SPY",
) -> pd.DataFrame:
    out = feature_rows.copy()
    if out.empty:
        for col in ELIGIBILITY_DIAGNOSTIC_COLUMNS:
            out[col] = []
        return out
    if "ticker" not in out.columns or "date" not in out.columns:
        raise ValueError("feature_rows must include ticker and date columns.")

    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["date"] = out["date"].astype(str)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = out.groupby("ticker", sort=False)
    out["window_row_count"] = grouped.cumcount() + 1

    if not config.enabled:
        out["eligibility_common_equity_ok"] = True
        out["eligibility_history_ok"] = True
        out["eligibility_valid_ohlcv_ok"] = True
        out["eligibility_liquidity_ok"] = True
        out["eligibility_price_ok"] = True
        out["eligibility_exchange_ok"] = True
        out["eligibility_valid_ohlcv_rows"] = out["window_row_count"]
        out["eligibility_avg_dollar_volume"] = np.nan
        out["episode_eligible"] = True
        return out

    open_ = _numeric_column(out, "open")
    high = _numeric_column(out, "high")
    low = _numeric_column(out, "low")
    close = _numeric_column(out, "close")
    volume = _numeric_column(out, "volume")
    dollar_volume = _numeric_column(out, "dollar_volume")

    valid_ohlcv_row = (
        np.isfinite(open_)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
        & np.isfinite(volume)
        & np.isfinite(dollar_volume)
        & (open_ > 0)
        & (high > 0)
        & (low > 0)
        & (close > 0)
        & (high >= low)
        & (volume >= 0)
        & (dollar_volume >= 0)
    )

    out["_eligibility_valid_ohlcv_row"] = valid_ohlcv_row.astype(float)
    out["_eligibility_dollar_volume"] = dollar_volume
    grouped = out.groupby("ticker", sort=False)
    out["eligibility_valid_ohlcv_rows"] = grouped["_eligibility_valid_ohlcv_row"].transform(
        lambda values: values.rolling(
            config.valid_ohlcv_lookback,
            min_periods=config.valid_ohlcv_lookback,
        ).sum()
    )
    out["eligibility_avg_dollar_volume"] = grouped["_eligibility_dollar_volume"].transform(
        lambda values: values.rolling(
            config.dollar_volume_lookback,
            min_periods=config.dollar_volume_lookback,
        ).mean()
    )

    if "exchange" in out.columns:
        exchanges = out["exchange"].apply(_canonical_exchange)
        out["eligibility_exchange_ok"] = exchanges.isin(config.allowed_exchanges)
    else:
        out["eligibility_exchange_ok"] = not bool(config.allowed_exchanges)

    out["eligibility_common_equity_ok"] = _common_equity_mask(out)
    out["eligibility_history_ok"] = out["window_row_count"] >= config.min_history_days
    out["eligibility_valid_ohlcv_ok"] = out["eligibility_valid_ohlcv_rows"].fillna(0) >= config.min_valid_ohlcv_days
    out["eligibility_liquidity_ok"] = (
        out["eligibility_avg_dollar_volume"].fillna(-np.inf) >= config.min_avg_dollar_volume
    )
    out["eligibility_price_ok"] = close >= config.min_price
    out["episode_eligible"] = (
        out["eligibility_common_equity_ok"]
        & out["eligibility_history_ok"]
        & out["eligibility_valid_ohlcv_ok"]
        & out["eligibility_liquidity_ok"]
        & out["eligibility_price_ok"]
        & out["eligibility_exchange_ok"]
        & (out["ticker"] != benchmark_ticker.upper())
    )

    out = out.drop(columns=["_eligibility_valid_ohlcv_row", "_eligibility_dollar_volume"])
    return out


def episode_eligibility_summary(
    feature_rows: pd.DataFrame,
    config: EpisodeEligibilityConfig,
    *,
    benchmark_ticker: str = "SPY",
) -> dict[str, object]:
    if feature_rows.empty:
        return {
            "config": config.to_dict(),
            "total_ticker_count": 0,
            "eligible_ticker_count": 0,
            "eligible_episode_rows": 0,
            "latest_date": None,
            "latest_eligible_ticker_count": 0,
        }
    eligible = add_episode_eligibility_columns(
        feature_rows,
        config,
        benchmark_ticker=benchmark_ticker,
    )
    if eligible.empty:
        return {
            "config": config.to_dict(),
            "total_ticker_count": 0,
            "eligible_ticker_count": 0,
            "eligible_episode_rows": 0,
            "latest_date": None,
            "latest_eligible_ticker_count": 0,
        }

    stock_rows = eligible[eligible["ticker"] != benchmark_ticker.upper()].copy()
    eligible_rows = stock_rows[stock_rows["episode_eligible"]]
    latest_date = stock_rows["date"].max() if not stock_rows.empty else None
    latest_rows = stock_rows[stock_rows["date"] == latest_date] if latest_date else stock_rows.iloc[0:0]
    return {
        "config": config.to_dict(),
        "total_ticker_count": int(stock_rows["ticker"].nunique()),
        "eligible_ticker_count": int(eligible_rows["ticker"].nunique()),
        "eligible_episode_rows": int(len(eligible_rows)),
        "latest_date": str(latest_date) if latest_date else None,
        "latest_eligible_ticker_count": int(latest_rows.loc[latest_rows["episode_eligible"], "ticker"].nunique()),
    }
