from __future__ import annotations

from datetime import date, timedelta
import unittest

import pandas as pd

from src.data.episode_eligibility import (
    EpisodeEligibilityConfig,
    add_episode_eligibility_columns,
    episode_eligibility_summary,
)


def _rows(ticker: str, *, exchange: str, close: float, dollar_volume: float, days: int = 8) -> list[dict[str, object]]:
    start = date(2024, 1, 2)
    rows: list[dict[str, object]] = []
    for idx in range(days):
        rows.append(
            {
                "ticker": ticker,
                "date": (start + timedelta(days=idx)).isoformat(),
                "exchange": exchange,
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": dollar_volume / close,
                "dollar_volume": dollar_volume,
                "type": "Common Stock",
            }
        )
    return rows


class EpisodeEligibilityTests(unittest.TestCase):
    def test_default_filter_is_broad_sixty_day_standard(self) -> None:
        config = EpisodeEligibilityConfig()

        self.assertEqual(config.min_history_days, 60)
        self.assertEqual(config.valid_ohlcv_lookback, 60)
        self.assertEqual(config.min_valid_ohlcv_days, 55)
        self.assertEqual(config.min_avg_dollar_volume, 100_000.0)
        self.assertEqual(config.min_price, 1.0)

    def test_filter_enforces_history_liquidity_price_exchange_and_type(self) -> None:
        frame = pd.DataFrame(
            [
                *_rows("GOOD", exchange="NASDAQ", close=10.0, dollar_volume=10_000.0),
                *_rows("LOWVOL", exchange="NYSE", close=10.0, dollar_volume=100.0),
                *_rows("LOWPX", exchange="NYSE", close=2.0, dollar_volume=10_000.0),
                *_rows("OTCROW", exchange="OTC", close=10.0, dollar_volume=10_000.0),
                *_rows("MKTROW", exchange="NYSE MKT", close=10.0, dollar_volume=10_000.0),
                *_rows("ETFROW", exchange="NASDAQ", close=10.0, dollar_volume=10_000.0),
            ]
        )
        frame.loc[frame["ticker"] == "ETFROW", "type"] = "ETF"
        config = EpisodeEligibilityConfig(
            min_history_days=5,
            valid_ohlcv_lookback=5,
            min_valid_ohlcv_days=5,
            dollar_volume_lookback=3,
            min_avg_dollar_volume=5_000.0,
            min_price=5.0,
            allowed_exchanges=("NYSE", "NASDAQ", "AMEX", "BATS"),
        )

        eligible = add_episode_eligibility_columns(frame, config)
        latest = eligible[eligible["date"] == eligible["date"].max()]
        latest_eligible = set(latest.loc[latest["episode_eligible"], "ticker"])

        self.assertEqual(latest_eligible, {"GOOD", "MKTROW"})
        self.assertFalse(latest.loc[latest["ticker"] == "LOWVOL", "eligibility_liquidity_ok"].iloc[0])
        self.assertFalse(latest.loc[latest["ticker"] == "LOWPX", "eligibility_price_ok"].iloc[0])
        self.assertFalse(latest.loc[latest["ticker"] == "OTCROW", "eligibility_exchange_ok"].iloc[0])
        self.assertFalse(latest.loc[latest["ticker"] == "ETFROW", "eligibility_common_equity_ok"].iloc[0])

    def test_summary_counts_eligible_tickers_and_episode_rows(self) -> None:
        frame = pd.DataFrame(
            [
                *_rows("A", exchange="NASDAQ", close=10.0, dollar_volume=10_000.0),
                *_rows("B", exchange="NYSE", close=10.0, dollar_volume=100.0),
            ]
        )
        config = EpisodeEligibilityConfig(
            min_history_days=5,
            valid_ohlcv_lookback=5,
            min_valid_ohlcv_days=5,
            dollar_volume_lookback=3,
            min_avg_dollar_volume=5_000.0,
            min_price=5.0,
            allowed_exchanges=("NYSE", "NASDAQ"),
        )

        summary = episode_eligibility_summary(frame, config)

        self.assertEqual(summary["total_ticker_count"], 2)
        self.assertEqual(summary["eligible_ticker_count"], 1)
        self.assertEqual(summary["latest_eligible_ticker_count"], 1)
        self.assertEqual(summary["eligible_episode_rows"], 4)


if __name__ == "__main__":
    unittest.main()
