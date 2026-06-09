from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.data.episode_eligibility import canonical_exchange, common_equity_mask, parse_allowed_exchanges


RESEARCH_UNIVERSE_DIAGNOSTIC_COLUMNS = (
    "research_common_equity_ok",
    "research_exchange_ok",
    "research_history_ok",
    "research_price_ok",
    "research_liquidity_ok",
    "research_trend_ok",
    "research_path_quality_ok",
    "research_history_days",
    "research_close",
    "research_current_dollar_volume",
    "research_median_dollar_volume_20d",
    "research_median_dollar_volume_60d",
    "research_zero_volume_day_ratio_60d",
    "research_current_dollar_volume_vs_median20",
    "research_return_6m",
    "research_drawdown_from_252d_high",
    "research_close_to_sma200",
    "research_sma50_to_sma200",
    "research_max_abs_return_60d",
    "research_max_true_range_pct_60d",
    "research_primary_rejection_reason",
    "research_rejection_reasons",
    "research_universe_ok",
)


@dataclass(frozen=True)
class ConservativeResearchUniverseConfig:
    """Shared strategy-universe guardrail for train/test/live scoring."""

    name: str = "conservative"
    enabled: bool = True
    common_stocks_only: bool = True
    allowed_exchanges: tuple[str, ...] = ("NYSE", "NASDAQ", "AMEX")
    min_price: float = 10.0
    min_history_days: int = 252
    liquidity_short_lookback: int = 20
    liquidity_long_lookback: int = 60
    min_median_dollar_volume_20d: float = 10_000_000.0
    min_median_dollar_volume_60d: float = 10_000_000.0
    max_zero_volume_day_ratio_60d: float = 0.02
    min_current_dollar_volume_vs_median_20d: float = 0.20
    trend_lookback_days: int = 252
    return_6m_lookback_days: int = 126
    sma_short_lookback_days: int = 50
    sma_long_lookback_days: int = 200
    min_return_6m: float = -0.15
    max_drawdown_from_252d_high: float = 0.35
    require_close_above_sma200: bool = True
    require_sma50_above_sma200: bool = True
    spike_filter_enabled: bool = True
    spike_lookback_days: int = 60
    max_abs_return_1d_60d: float = 0.25
    max_true_range_pct_60d: float = 0.25

    def __post_init__(self) -> None:
        positive_ints = {
            "min_history_days": self.min_history_days,
            "liquidity_short_lookback": self.liquidity_short_lookback,
            "liquidity_long_lookback": self.liquidity_long_lookback,
            "trend_lookback_days": self.trend_lookback_days,
            "return_6m_lookback_days": self.return_6m_lookback_days,
            "sma_short_lookback_days": self.sma_short_lookback_days,
            "sma_long_lookback_days": self.sma_long_lookback_days,
            "spike_lookback_days": self.spike_lookback_days,
        }
        for name, value in positive_ints.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be >= 1.")
        if not str(self.name).strip():
            raise ValueError("Research-universe name must be non-empty.")
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "allowed_exchanges", parse_allowed_exchanges(self.allowed_exchanges))
        if min(
            self.min_price,
            self.min_median_dollar_volume_20d,
            self.min_median_dollar_volume_60d,
            self.max_zero_volume_day_ratio_60d,
            self.min_current_dollar_volume_vs_median_20d,
            self.max_drawdown_from_252d_high,
            self.max_abs_return_1d_60d,
            self.max_true_range_pct_60d,
        ) < 0:
            raise ValueError("Research-universe thresholds must be non-negative where applicable.")

    @property
    def required_recent_rows_per_ticker(self) -> int:
        return max(
            self.min_history_days,
            self.liquidity_long_lookback,
            self.trend_lookback_days,
            self.return_6m_lookback_days,
            self.sma_short_lookback_days,
            self.sma_long_lookback_days,
            self.spike_lookback_days,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "common_stocks_only": self.common_stocks_only,
            "allowed_exchanges": list(self.allowed_exchanges),
            "min_price": self.min_price,
            "min_history_days": self.min_history_days,
            "liquidity_short_lookback": self.liquidity_short_lookback,
            "liquidity_long_lookback": self.liquidity_long_lookback,
            "min_median_dollar_volume_20d": self.min_median_dollar_volume_20d,
            "min_median_dollar_volume_60d": self.min_median_dollar_volume_60d,
            "max_zero_volume_day_ratio_60d": self.max_zero_volume_day_ratio_60d,
            "min_current_dollar_volume_vs_median_20d": self.min_current_dollar_volume_vs_median_20d,
            "trend_lookback_days": self.trend_lookback_days,
            "return_6m_lookback_days": self.return_6m_lookback_days,
            "sma_short_lookback_days": self.sma_short_lookback_days,
            "sma_long_lookback_days": self.sma_long_lookback_days,
            "min_return_6m": self.min_return_6m,
            "max_drawdown_from_252d_high": self.max_drawdown_from_252d_high,
            "require_close_above_sma200": self.require_close_above_sma200,
            "require_sma50_above_sma200": self.require_sma50_above_sma200,
            "spike_filter_enabled": self.spike_filter_enabled,
            "spike_lookback_days": self.spike_lookback_days,
            "max_abs_return_1d_60d": self.max_abs_return_1d_60d,
            "max_true_range_pct_60d": self.max_true_range_pct_60d,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ConservativeResearchUniverseConfig":
        return cls(
            name=str(payload.get("name", cls.name)),
            enabled=bool(payload.get("enabled", True)),
            common_stocks_only=bool(payload.get("common_stocks_only", True)),
            allowed_exchanges=parse_allowed_exchanges(payload.get("allowed_exchanges", cls.allowed_exchanges)),
            min_price=float(payload.get("min_price", cls.min_price)),
            min_history_days=int(payload.get("min_history_days", cls.min_history_days)),
            liquidity_short_lookback=int(payload.get("liquidity_short_lookback", cls.liquidity_short_lookback)),
            liquidity_long_lookback=int(payload.get("liquidity_long_lookback", cls.liquidity_long_lookback)),
            min_median_dollar_volume_20d=float(
                payload.get("min_median_dollar_volume_20d", cls.min_median_dollar_volume_20d)
            ),
            min_median_dollar_volume_60d=float(
                payload.get("min_median_dollar_volume_60d", cls.min_median_dollar_volume_60d)
            ),
            max_zero_volume_day_ratio_60d=float(
                payload.get("max_zero_volume_day_ratio_60d", cls.max_zero_volume_day_ratio_60d)
            ),
            min_current_dollar_volume_vs_median_20d=float(
                payload.get(
                    "min_current_dollar_volume_vs_median_20d",
                    cls.min_current_dollar_volume_vs_median_20d,
                )
            ),
            trend_lookback_days=int(payload.get("trend_lookback_days", cls.trend_lookback_days)),
            return_6m_lookback_days=int(payload.get("return_6m_lookback_days", cls.return_6m_lookback_days)),
            sma_short_lookback_days=int(payload.get("sma_short_lookback_days", cls.sma_short_lookback_days)),
            sma_long_lookback_days=int(payload.get("sma_long_lookback_days", cls.sma_long_lookback_days)),
            min_return_6m=float(payload.get("min_return_6m", cls.min_return_6m)),
            max_drawdown_from_252d_high=float(
                payload.get("max_drawdown_from_252d_high", cls.max_drawdown_from_252d_high)
            ),
            require_close_above_sma200=bool(
                payload.get("require_close_above_sma200", cls.require_close_above_sma200)
            ),
            require_sma50_above_sma200=bool(
                payload.get("require_sma50_above_sma200", cls.require_sma50_above_sma200)
            ),
            spike_filter_enabled=bool(payload.get("spike_filter_enabled", cls.spike_filter_enabled)),
            spike_lookback_days=int(payload.get("spike_lookback_days", cls.spike_lookback_days)),
            max_abs_return_1d_60d=float(payload.get("max_abs_return_1d_60d", cls.max_abs_return_1d_60d)),
            max_true_range_pct_60d=float(
                payload.get("max_true_range_pct_60d", cls.max_true_range_pct_60d)
            ),
        )


def research_universe_metadata_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in RESEARCH_UNIVERSE_DIAGNOSTIC_COLUMNS if column in frame.columns]


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _rolling_by_ticker(
    frame: pd.DataFrame,
    values: pd.Series,
    *,
    window: int,
    min_periods: int | None = None,
    method: str,
) -> pd.Series:
    source = pd.DataFrame({"ticker": frame["ticker"], "_value": values})
    grouped = source.groupby("ticker", sort=False)["_value"]
    periods = int(min_periods if min_periods is not None else window)
    if method == "mean":
        rolled = grouped.rolling(window, min_periods=periods).mean()
    elif method == "median":
        rolled = grouped.rolling(window, min_periods=periods).median()
    elif method == "max":
        rolled = grouped.rolling(window, min_periods=periods).max()
    else:
        raise ValueError(f"Unsupported rolling method: {method}")
    return rolled.reset_index(level=0, drop=True).reindex(frame.index)


def _append_reason(reasons: pd.Series, mask: pd.Series, code: str) -> pd.Series:
    values = reasons.astype(object).to_numpy(copy=True)
    mask_values = mask.fillna(False).to_numpy(dtype=bool)
    if mask_values.any():
        selected = values[mask_values]
        values[mask_values] = np.where(selected == "", code, selected + "|" + code)
    return pd.Series(values, index=reasons.index, dtype="object")


def add_conservative_research_universe_columns(
    feature_rows: pd.DataFrame,
    config: ConservativeResearchUniverseConfig,
    *,
    benchmark_ticker: str = "SPY",
) -> pd.DataFrame:
    out = feature_rows.copy()
    if out.empty:
        for column in RESEARCH_UNIVERSE_DIAGNOSTIC_COLUMNS:
            out[column] = []
        return out
    if "ticker" not in out.columns or "date" not in out.columns:
        raise ValueError("feature_rows must include ticker and date columns.")

    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["date"] = out["date"].astype(str)
    out = out.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    grouped = out.groupby("ticker", sort=False)
    out["research_history_days"] = grouped.cumcount() + 1

    close = _numeric_column(out, "close")
    high = _numeric_column(out, "high")
    low = _numeric_column(out, "low")
    volume = _numeric_column(out, "volume")
    dollar_volume = _numeric_column(out, "dollar_volume")
    out["research_close"] = close
    out["research_current_dollar_volume"] = dollar_volume
    out["research_median_dollar_volume_20d"] = _rolling_by_ticker(
        out,
        dollar_volume,
        window=config.liquidity_short_lookback,
        method="median",
    )
    out["research_median_dollar_volume_60d"] = _rolling_by_ticker(
        out,
        dollar_volume,
        window=config.liquidity_long_lookback,
        method="median",
    )

    zero_volume_flag = (
        ~np.isfinite(volume)
        | ~np.isfinite(dollar_volume)
        | (volume <= 0.0)
        | (dollar_volume <= 0.0)
    ).astype(float)
    out["research_zero_volume_day_ratio_60d"] = _rolling_by_ticker(
        out,
        pd.Series(zero_volume_flag, index=out.index),
        window=config.liquidity_long_lookback,
        method="mean",
    )
    out["research_current_dollar_volume_vs_median20"] = (
        dollar_volume / out["research_median_dollar_volume_20d"].replace(0.0, np.nan)
    )

    close_grouped = close.groupby(out["ticker"], sort=False)
    previous_close_6m = close_grouped.shift(config.return_6m_lookback_days).astype(float)
    out["research_return_6m"] = close / previous_close_6m - 1.0

    high_for_drawdown = high.where(np.isfinite(high), close)
    rolling_high = _rolling_by_ticker(
        out,
        high_for_drawdown,
        window=config.trend_lookback_days,
        min_periods=config.trend_lookback_days,
        method="max",
    )
    out["research_drawdown_from_252d_high"] = 1.0 - (close / rolling_high)
    sma50 = _rolling_by_ticker(
        out,
        close,
        window=config.sma_short_lookback_days,
        min_periods=config.sma_short_lookback_days,
        method="mean",
    )
    sma200 = _rolling_by_ticker(
        out,
        close,
        window=config.sma_long_lookback_days,
        min_periods=config.sma_long_lookback_days,
        method="mean",
    )
    out["research_close_to_sma200"] = close / sma200
    out["research_sma50_to_sma200"] = sma50 / sma200

    return_1d = _numeric_column(out, "return_1d")
    if return_1d.isna().all():
        return_1d = close_grouped.pct_change()
    out["research_max_abs_return_60d"] = _rolling_by_ticker(
        out,
        return_1d.abs(),
        window=config.spike_lookback_days,
        method="max",
    )

    true_range_pct = _numeric_column(out, "true_range_pct")
    if true_range_pct.isna().all():
        true_range_pct = (high - low).abs() / close.replace(0.0, np.nan)
    out["research_max_true_range_pct_60d"] = _rolling_by_ticker(
        out,
        true_range_pct,
        window=config.spike_lookback_days,
        method="max",
    )

    if config.common_stocks_only:
        out["research_common_equity_ok"] = common_equity_mask(out)
    else:
        out["research_common_equity_ok"] = True
    if config.allowed_exchanges:
        if "exchange" in out.columns:
            exchanges = out["exchange"].apply(canonical_exchange)
            out["research_exchange_ok"] = exchanges.isin(config.allowed_exchanges)
        else:
            out["research_exchange_ok"] = False
    else:
        out["research_exchange_ok"] = True

    if not config.enabled:
        out["research_history_ok"] = True
        out["research_price_ok"] = True
        out["research_liquidity_ok"] = True
        out["research_trend_ok"] = True
        out["research_path_quality_ok"] = True
        out["research_primary_rejection_reason"] = ""
        out["research_rejection_reasons"] = ""
        out["research_universe_ok"] = True
        return out

    out["research_history_ok"] = out["research_history_days"] >= config.min_history_days
    out["research_price_ok"] = close >= config.min_price
    out["research_liquidity_ok"] = (
        out["research_median_dollar_volume_20d"].fillna(-np.inf) >= config.min_median_dollar_volume_20d
    ) & (
        out["research_median_dollar_volume_60d"].fillna(-np.inf) >= config.min_median_dollar_volume_60d
    ) & (
        out["research_zero_volume_day_ratio_60d"].fillna(np.inf) <= config.max_zero_volume_day_ratio_60d
    ) & (
        out["research_current_dollar_volume_vs_median20"].fillna(-np.inf)
        >= config.min_current_dollar_volume_vs_median_20d
    )
    trend_ok = (
        out["research_return_6m"].fillna(-np.inf) >= config.min_return_6m
    ) & (
        out["research_drawdown_from_252d_high"].fillna(np.inf) <= config.max_drawdown_from_252d_high
    )
    if config.require_close_above_sma200:
        trend_ok &= out["research_close_to_sma200"].fillna(-np.inf) >= 1.0
    if config.require_sma50_above_sma200:
        trend_ok &= out["research_sma50_to_sma200"].fillna(-np.inf) >= 1.0
    out["research_trend_ok"] = trend_ok
    if config.spike_filter_enabled:
        out["research_path_quality_ok"] = (
            out["research_max_abs_return_60d"].fillna(np.inf) <= config.max_abs_return_1d_60d
        ) & (
            out["research_max_true_range_pct_60d"].fillna(np.inf) <= config.max_true_range_pct_60d
        )
    else:
        out["research_path_quality_ok"] = True

    out["research_universe_ok"] = (
        out["research_common_equity_ok"]
        & out["research_exchange_ok"]
        & out["research_history_ok"]
        & out["research_price_ok"]
        & out["research_liquidity_ok"]
        & out["research_trend_ok"]
        & out["research_path_quality_ok"]
        & (out["ticker"] != benchmark_ticker.upper())
    )

    reasons = pd.Series("", index=out.index, dtype="object")
    reasons = _append_reason(reasons, ~out["research_common_equity_ok"], "REJECTED_INVALID_SECURITY_TYPE")
    reasons = _append_reason(reasons, ~out["research_exchange_ok"], "REJECTED_UNSUPPORTED_EXCHANGE")
    reasons = _append_reason(reasons, ~out["research_history_ok"], "REJECTED_INSUFFICIENT_HISTORY")
    reasons = _append_reason(reasons, ~out["research_price_ok"], "REJECTED_LOW_PRICE")
    reasons = _append_reason(
        reasons,
        out["research_median_dollar_volume_20d"].fillna(-np.inf) < config.min_median_dollar_volume_20d,
        "REJECTED_LOW_MEDIAN_DOLLAR_VOLUME_20D",
    )
    reasons = _append_reason(
        reasons,
        out["research_median_dollar_volume_60d"].fillna(-np.inf) < config.min_median_dollar_volume_60d,
        "REJECTED_LOW_MEDIAN_DOLLAR_VOLUME_60D",
    )
    reasons = _append_reason(
        reasons,
        out["research_zero_volume_day_ratio_60d"].fillna(np.inf) > config.max_zero_volume_day_ratio_60d,
        "REJECTED_ZERO_VOLUME_DAYS",
    )
    reasons = _append_reason(
        reasons,
        out["research_current_dollar_volume_vs_median20"].fillna(-np.inf)
        < config.min_current_dollar_volume_vs_median_20d,
        "REJECTED_CURRENT_DOLLAR_VOLUME_TOO_LOW",
    )
    reasons = _append_reason(
        reasons,
        out["research_return_6m"].fillna(-np.inf) < config.min_return_6m,
        "REJECTED_POOR_6M_TREND",
    )
    reasons = _append_reason(
        reasons,
        out["research_drawdown_from_252d_high"].fillna(np.inf) > config.max_drawdown_from_252d_high,
        "REJECTED_EXCESSIVE_52W_DRAWDOWN",
    )
    if config.require_close_above_sma200:
        reasons = _append_reason(
            reasons,
            out["research_close_to_sma200"].fillna(-np.inf) < 1.0,
            "REJECTED_BELOW_200DMA",
        )
    if config.require_sma50_above_sma200:
        reasons = _append_reason(
            reasons,
            out["research_sma50_to_sma200"].fillna(-np.inf) < 1.0,
            "REJECTED_50DMA_BELOW_200DMA",
        )
    if config.spike_filter_enabled:
        reasons = _append_reason(
            reasons,
            out["research_max_abs_return_60d"].fillna(np.inf) > config.max_abs_return_1d_60d,
            "REJECTED_SPIKE_ABS_RETURN",
        )
        reasons = _append_reason(
            reasons,
            out["research_max_true_range_pct_60d"].fillna(np.inf) > config.max_true_range_pct_60d,
            "REJECTED_SPIKE_TRUE_RANGE",
        )
    out["research_rejection_reasons"] = reasons
    out["research_primary_rejection_reason"] = reasons.str.split("|").str[0].fillna("")
    return out


def latest_research_universe_diagnostics(
    feature_rows: pd.DataFrame,
    metadata: pd.DataFrame,
    config: ConservativeResearchUniverseConfig,
    *,
    benchmark_ticker: str = "SPY",
) -> pd.DataFrame:
    diagnostics = add_conservative_research_universe_columns(
        feature_rows,
        config,
        benchmark_ticker=benchmark_ticker,
    )
    lookup_columns = ["ticker", "date", *research_universe_metadata_columns(diagnostics)]
    lookup = diagnostics[lookup_columns].rename(columns={"date": "anchor_date"})
    out = metadata.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["anchor_date"] = out["anchor_date"].astype(str)
    out = out.merge(lookup, on=["ticker", "anchor_date"], how="left")
    bool_columns = [
        "research_common_equity_ok",
        "research_exchange_ok",
        "research_history_ok",
        "research_price_ok",
        "research_liquidity_ok",
        "research_trend_ok",
        "research_path_quality_ok",
        "research_universe_ok",
    ]
    for column in bool_columns:
        if column in out.columns:
            out[column] = out[column].fillna(False).astype(bool)
    for column in ("research_primary_rejection_reason", "research_rejection_reasons"):
        if column in out.columns:
            out[column] = out[column].fillna("").astype(str)
    return out


def conservative_research_universe_summary(
    diagnostics: pd.DataFrame,
    config: ConservativeResearchUniverseConfig,
) -> dict[str, object]:
    if diagnostics.empty:
        return {
            "research_universe_enabled": bool(config.enabled),
            "research_universe_config": config.to_dict(),
            "research_universe_input_rows": 0,
            "research_universe_passed_rows": 0,
            "research_universe_removed_rows": 0,
            "research_rejection_reason_counts": {},
        }
    ok = diagnostics["research_universe_ok"].fillna(False).astype(bool)
    summary: dict[str, object] = {
        "research_universe_enabled": bool(config.enabled),
        "research_universe_config": config.to_dict(),
        "research_universe_input_rows": int(len(diagnostics)),
        "research_universe_passed_rows": int(ok.sum()),
        "research_universe_removed_rows": int((~ok).sum()),
    }
    for column in (
        "research_common_equity_ok",
        "research_exchange_ok",
        "research_history_ok",
        "research_price_ok",
        "research_liquidity_ok",
        "research_trend_ok",
        "research_path_quality_ok",
    ):
        if column in diagnostics.columns:
            column_ok = diagnostics[column].fillna(False).astype(bool)
            summary[f"{column}_failed_rows"] = int((~column_ok).sum())
    reason_counts: dict[str, int] = {}
    if "research_rejection_reasons" in diagnostics.columns:
        for text in diagnostics.loc[~ok, "research_rejection_reasons"].fillna("").astype(str):
            for reason in [item for item in text.split("|") if item]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary["research_rejection_reason_counts"] = dict(sorted(reason_counts.items()))
    return summary
