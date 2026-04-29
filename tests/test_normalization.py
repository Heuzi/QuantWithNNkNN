from __future__ import annotations

import unittest

from src.data.normalization import compute_normalized_feature_rows


def _base_row(ticker: str, trade_date: str, sector: str, value: float) -> dict[str, object]:
    return {
        "date": trade_date,
        "ticker": ticker,
        "volume": value,
        "dollar_volume": value * 10.0,
        "rolling_avg_volume_20d": value,
        "rolling_avg_volume_60d": value,
        "rolling_avg_dollar_volume_20d": value * 10.0,
        "rolling_avg_dollar_volume_60d": value * 10.0,
        "return_1d": value / 100.0,
        "gap_pct": value / 200.0,
        "intraday_return": value / 300.0,
        "hl_range_pct": value / 400.0,
        "rolling_return_5d": value / 100.0,
        "rolling_return_20d": value / 100.0,
        "rolling_return_60d": value / 100.0,
        "rolling_vol_20d": value / 1000.0,
        "rolling_vol_60d": value / 1000.0,
        "price_vs_sma_20d": value / 100.0,
        "price_vs_sma_60d": value / 100.0,
        "momentum_20d": value / 100.0,
        "momentum_60d": value / 100.0,
        "volume_ratio_20d": value / 10.0,
        "gics_sector": sector,
        "gics_sub_industry": "Sub",
    }


class NormalizationTests(unittest.TestCase):
    def test_same_date_zscore_and_percentile_rank(self) -> None:
        rows = [
            _base_row("AAA", "2024-01-02", "Tech", 1.0),
            _base_row("BBB", "2024-01-02", "Tech", 2.0),
            _base_row("CCC", "2024-01-02", "Tech", 3.0),
        ]
        metadata = {
            "AAA": {"gics_sector": "Tech", "gics_sub_industry": "Sub"},
            "BBB": {"gics_sector": "Tech", "gics_sub_industry": "Sub"},
            "CCC": {"gics_sector": "Tech", "gics_sub_industry": "Sub"},
        }

        normalized = compute_normalized_feature_rows(rows, metadata)
        self.assertAlmostEqual(normalized[0]["return_1d__cs_pct"], 0.0)
        self.assertAlmostEqual(normalized[1]["return_1d__cs_pct"], 0.5)
        self.assertAlmostEqual(normalized[2]["return_1d__cs_pct"], 1.0)
        self.assertAlmostEqual(normalized[1]["return_1d__cs_z"], 0.0)

    def test_percentile_rank_handles_ties(self) -> None:
        rows = [
            _base_row("AAA", "2024-01-02", "Tech", 1.0),
            _base_row("BBB", "2024-01-02", "Tech", 1.0),
            _base_row("CCC", "2024-01-02", "Tech", 3.0),
        ]
        metadata = {
            "AAA": {"gics_sector": "Tech", "gics_sub_industry": "Sub"},
            "BBB": {"gics_sector": "Tech", "gics_sub_industry": "Sub"},
            "CCC": {"gics_sector": "Tech", "gics_sub_industry": "Sub"},
        }

        normalized = compute_normalized_feature_rows(rows, metadata)
        self.assertAlmostEqual(normalized[0]["return_1d__cs_pct"], 0.25)
        self.assertAlmostEqual(normalized[1]["return_1d__cs_pct"], 0.25)
        self.assertAlmostEqual(normalized[2]["return_1d__cs_pct"], 1.0)

    def test_sector_relative_requires_minimum_group_size(self) -> None:
        rows = [
            _base_row("AAA", "2024-01-02", "Tech", 1.0),
            _base_row("BBB", "2024-01-02", "Tech", 2.0),
            _base_row("CCC", "2024-01-02", "Tech", 3.0),
            _base_row("DDD", "2024-01-02", "Tech", 4.0),
            _base_row("EEE", "2024-01-02", "Tech", 5.0),
            _base_row("FFF", "2024-01-02", "Utilities", 10.0),
            _base_row("GGG", "2024-01-02", "Utilities", 20.0),
            _base_row("HHH", "2024-01-02", "Utilities", 30.0),
            _base_row("III", "2024-01-02", "Utilities", 40.0),
        ]
        metadata = {
            row["ticker"]: {"gics_sector": row["gics_sector"], "gics_sub_industry": "Sub"} for row in rows
        }

        normalized = compute_normalized_feature_rows(rows, metadata)
        self.assertIsNotNone(normalized[0]["rolling_return_20d__sector_cs_pct"])
        self.assertIsNone(normalized[5]["rolling_return_20d__sector_cs_pct"])

    def test_no_future_date_leakage_in_cross_section(self) -> None:
        rows = [
            _base_row("AAA", "2024-01-02", "Tech", 1.0),
            _base_row("BBB", "2024-01-02", "Tech", 2.0),
            _base_row("AAA", "2024-01-03", "Tech", 1000.0),
            _base_row("BBB", "2024-01-03", "Tech", 2000.0),
        ]
        metadata = {
            "AAA": {"gics_sector": "Tech", "gics_sub_industry": "Sub"},
            "BBB": {"gics_sector": "Tech", "gics_sub_industry": "Sub"},
        }

        normalized = compute_normalized_feature_rows(rows, metadata)
        self.assertAlmostEqual(normalized[0]["return_1d__cs_pct"], 0.0)
        self.assertAlmostEqual(normalized[1]["return_1d__cs_pct"], 1.0)

    def test_row_count_and_sector_join_preserved(self) -> None:
        rows = [
            _base_row("AAA", "2024-01-02", "Tech", 1.0),
            _base_row("BBB", "2024-01-02", "Utilities", 2.0),
        ]
        metadata = {
            "AAA": {"gics_sector": "Tech", "gics_sub_industry": "Software"},
            "BBB": {"gics_sector": "Utilities", "gics_sub_industry": "Electric"},
        }

        normalized = compute_normalized_feature_rows(rows, metadata)
        self.assertEqual(len(normalized), len(rows))
        self.assertEqual(normalized[0]["gics_sub_industry"], "Software")
        self.assertEqual(normalized[1]["gics_sector"], "Utilities")
        self.assertIsNotNone(normalized[0]["log1p_volume"])


if __name__ == "__main__":
    unittest.main()
