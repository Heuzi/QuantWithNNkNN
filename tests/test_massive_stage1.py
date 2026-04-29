from __future__ import annotations

import unittest

from src.data.massive_stage1 import (
    _clean_wiki_markup,
    choose_sample_universe,
    compute_daily_features,
    compute_episode_index,
    normalize_ticker_range_bars,
)


class MassiveStage1Tests(unittest.TestCase):
    def test_choose_sample_universe_keeps_requested_names_and_benchmark(self) -> None:
        bars = [
            {"ticker": "AAA", "dollar_volume": 10.0},
            {"ticker": "BBB", "dollar_volume": 20.0},
            {"ticker": "CCC", "dollar_volume": 30.0},
            {"ticker": "SPY", "dollar_volume": 100.0},
        ]

        chosen = choose_sample_universe(
            bars=bars,
            max_tickers=2,
            forced_tickers=["AAA"],
            benchmark_ticker="SPY",
        )

        self.assertIn("AAA", chosen)
        self.assertIn("SPY", chosen)
        self.assertGreaterEqual(len(chosen), 2)

    def test_compute_daily_features_adds_basic_returns(self) -> None:
        bars = [
            {"ticker": "AAA", "date": "2024-01-02", "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.0, "volume": 100.0, "dollar_volume": 1000.0},
            {"ticker": "AAA", "date": "2024-01-03", "open": 10.5, "high": 11.0, "low": 10.0, "close": 11.0, "volume": 110.0, "vwap": 10.7, "dollar_volume": 1210.0},
        ]

        features = compute_daily_features(bars)
        second = features[-1]

        self.assertAlmostEqual(second["return_1d"], 0.10)
        self.assertAlmostEqual(second["gap_pct"], 0.05)
        self.assertAlmostEqual(second["intraday_return"], (11.0 / 10.5) - 1.0)
        self.assertNotIn("close_to_vwap_pct", second)
        self.assertAlmostEqual(second["close_location"], 1.0)
        self.assertAlmostEqual(second["true_range_pct"], (11.0 - 10.0) / 10.0)

    def test_compute_daily_features_adds_rolling_volume_and_momentum_features(self) -> None:
        bars = []
        for idx in range(61):
            close = 100.0 + idx
            bars.append(
                {
                    "ticker": "AAA",
                    "date": f"2024-03-{idx + 1:02d}",
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0 + idx,
                    "vwap": close - 0.25,
                    "dollar_volume": close * (1000.0 + idx),
                }
            )

        last = compute_daily_features(bars)[-1]
        self.assertIsNotNone(last["rolling_avg_volume_20d"])
        self.assertIsNotNone(last["rolling_avg_volume_60d"])
        self.assertIsNotNone(last["rolling_return_60d"])
        self.assertIsNotNone(last["rolling_vol_60d"])
        self.assertIsNotNone(last["momentum_20d"])
        self.assertIsNotNone(last["momentum_60d"])
        self.assertIsNotNone(last["price_vs_sma_20d"])
        self.assertIsNotNone(last["price_vs_sma_60d"])
        self.assertIsNotNone(last["volume_ratio_20d"])
        self.assertIsNotNone(last["dollar_volume_ratio_5d"])
        self.assertIsNotNone(last["volume_zscore_20d"])
        self.assertTrue(last["has_60d_history"])

    def test_compute_episode_index_respects_window_and_benchmark_adjustment(self) -> None:
        rows = []
        aaa_prices = [10.0, 11.0, 12.0, 13.0, 14.0]
        spy_prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        for idx, close in enumerate(aaa_prices):
            rows.append(
                {
                    "ticker": "AAA",
                    "date": f"2024-01-0{idx + 1}",
                    "close": close,
                }
            )
        for idx, close in enumerate(spy_prices):
            rows.append(
                {
                    "ticker": "SPY",
                    "date": f"2024-01-0{idx + 1}",
                    "close": close,
                }
            )

        episodes = compute_episode_index(rows, window_length=3, horizon_days=1, benchmark_ticker="SPY")

        self.assertEqual(len(episodes), 2)
        first = episodes[0]
        self.assertEqual(first["anchor_date"], "2024-01-03")
        self.assertAlmostEqual(first["target_return"], (13.0 / 12.0) - 1.0)
        self.assertAlmostEqual(
            first["market_adjusted_target_return"],
            ((13.0 / 12.0) - 1.0) - ((103.0 / 102.0) - 1.0),
        )

    def test_clean_wiki_markup_handles_templates_and_links(self) -> None:
        raw = "{{NyseSymbol|BRK.B}} <!-- note --> [[Berkshire Hathaway|Berkshire]]"
        self.assertEqual(_clean_wiki_markup(raw), "BRK.B Berkshire")

    def test_normalize_ticker_range_bars_converts_timestamp_to_date(self) -> None:
        payload = {
            "ticker": "AAA",
            "adjusted": True,
            "results": [
                {"t": 1704153600000, "o": 10.0, "h": 11.0, "l": 9.0, "c": 10.5, "v": 100.0, "vw": 10.2, "n": 25}
            ],
        }
        rows = normalize_ticker_range_bars(payload)
        self.assertEqual(rows[0]["ticker"], "AAA")
        self.assertEqual(rows[0]["date"], "2024-01-02")
        self.assertAlmostEqual(rows[0]["dollar_volume"], 1050.0)


if __name__ == "__main__":
    unittest.main()
