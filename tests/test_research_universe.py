from __future__ import annotations

from datetime import date, timedelta
import unittest

import pandas as pd

from src.data.research_universe import (
    ConservativeResearchUniverseConfig,
    add_conservative_research_universe_columns,
    latest_research_universe_diagnostics,
)


def _rows(
    ticker: str,
    *,
    start_price: float,
    days: int = 260,
    daily_drift: float = 0.05,
    dollar_volume: float = 20_000_000.0,
    zero_volume_days: set[int] | None = None,
    spike_day: int | None = None,
    exchange: str = "NASDAQ",
    security_type: str = "Common Stock",
) -> list[dict[str, object]]:
    start = date(2025, 1, 2)
    rows: list[dict[str, object]] = []
    price = start_price
    zero_volume_days = zero_volume_days or set()
    for idx in range(days):
        if idx > 0:
            price = max(0.50, price + daily_drift)
        close = price
        high = close * 1.01
        low = close * 0.99
        if spike_day is not None and idx == spike_day:
            high = close * 1.35
            close = close * 1.22
        current_dollar_volume = 0.0 if idx in zero_volume_days else dollar_volume
        current_volume = 0.0 if current_dollar_volume <= 0 else current_dollar_volume / max(close, 0.01)
        prev_close = rows[-1]["close"] if rows else None
        return_1d = close / float(prev_close) - 1.0 if prev_close else None
        rows.append(
            {
                "ticker": ticker,
                "date": (start + timedelta(days=idx)).isoformat(),
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": current_volume,
                "dollar_volume": current_dollar_volume,
                "return_1d": return_1d,
                "true_range_pct": (high - low) / close,
                "exchange": exchange,
                "type": security_type,
            }
        )
    return rows


class ConservativeResearchUniverseTests(unittest.TestCase):
    def test_latest_diagnostics_filter_price_liquidity_trend_zero_volume_and_spikes(self) -> None:
        frame = pd.DataFrame(
            [
                *_rows("GOOD", start_price=40.0, daily_drift=0.06, dollar_volume=25_000_000.0),
                *_rows("PENNY", start_price=4.0, daily_drift=0.00, dollar_volume=25_000_000.0),
                *_rows("ILLIQ", start_price=40.0, daily_drift=0.06, dollar_volume=2_000_000.0),
                *_rows("FALL", start_price=90.0, daily_drift=-0.20, dollar_volume=25_000_000.0),
                *_rows("ZEROV", start_price=40.0, daily_drift=0.04, dollar_volume=25_000_000.0, zero_volume_days=set(range(210, 221))),
                *_rows("SPIKE", start_price=40.0, daily_drift=0.03, dollar_volume=25_000_000.0, spike_day=230),
                *_rows("ETF1", start_price=40.0, daily_drift=0.06, dollar_volume=25_000_000.0, security_type="ETF"),
                *_rows("OTCX", start_price=40.0, daily_drift=0.06, dollar_volume=25_000_000.0, exchange="OTC"),
            ]
        )
        latest = frame.sort_values(["ticker", "date"]).groupby("ticker").tail(1)
        metadata = latest[["ticker", "date"]].rename(columns={"date": "anchor_date"}).reset_index(drop=True)
        config = ConservativeResearchUniverseConfig()

        diagnostics = latest_research_universe_diagnostics(frame, metadata, config)
        passed = set(diagnostics.loc[diagnostics["research_universe_ok"], "ticker"])

        self.assertEqual(passed, {"GOOD"})
        by_ticker = diagnostics.set_index("ticker")
        self.assertFalse(bool(by_ticker.loc["PENNY", "research_price_ok"]))
        self.assertFalse(bool(by_ticker.loc["ILLIQ", "research_liquidity_ok"]))
        self.assertFalse(bool(by_ticker.loc["FALL", "research_trend_ok"]))
        self.assertFalse(bool(by_ticker.loc["ZEROV", "research_liquidity_ok"]))
        self.assertFalse(bool(by_ticker.loc["SPIKE", "research_path_quality_ok"]))
        self.assertFalse(bool(by_ticker.loc["ETF1", "research_common_equity_ok"]))
        self.assertFalse(bool(by_ticker.loc["OTCX", "research_exchange_ok"]))
        self.assertIn("REJECTED_LOW_PRICE", by_ticker.loc["PENNY", "research_rejection_reasons"])
        self.assertIn("REJECTED_ZERO_VOLUME_DAYS", by_ticker.loc["ZEROV", "research_rejection_reasons"])
        self.assertIn(
            by_ticker.loc["SPIKE", "research_primary_rejection_reason"],
            {"REJECTED_SPIKE_ABS_RETURN", "REJECTED_SPIKE_TRUE_RANGE"},
        )

    def test_numeric_string_close_columns_support_pct_change_fallback(self) -> None:
        frame = pd.DataFrame(_rows("GOOD", start_price=40.0, daily_drift=0.06, dollar_volume=25_000_000.0))
        for column in ("open", "high", "low", "close", "volume", "dollar_volume", "true_range_pct"):
            frame[column] = frame[column].map(str)
        frame["return_1d"] = ""

        diagnostics = add_conservative_research_universe_columns(frame, ConservativeResearchUniverseConfig())

        self.assertIn("research_max_abs_return_60d", diagnostics.columns)
        self.assertGreater(diagnostics["research_max_abs_return_60d"].notna().sum(), 0)
        self.assertTrue(pd.api.types.is_float_dtype(diagnostics["research_close"]))

    def test_config_name_round_trips_for_independent_sleeves(self) -> None:
        config = ConservativeResearchUniverseConfig(
            name="momentum_breakout",
            min_return_6m=0.05,
            max_drawdown_from_252d_high=0.60,
            require_close_above_sma200=False,
            require_sma50_above_sma200=False,
            spike_filter_enabled=False,
        )

        payload = config.to_dict()
        restored = ConservativeResearchUniverseConfig.from_mapping(payload)

        self.assertEqual(payload["name"], "momentum_breakout")
        self.assertEqual(restored.name, "momentum_breakout")
        self.assertFalse(restored.require_close_above_sma200)
        self.assertFalse(restored.require_sma50_above_sma200)
        self.assertFalse(restored.spike_filter_enabled)
        self.assertEqual(restored.min_return_6m, 0.05)


if __name__ == "__main__":
    unittest.main()
