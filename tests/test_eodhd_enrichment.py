from __future__ import annotations

import unittest

from src.data.eodhd_enrichment import (
    add_fundamental_features,
    add_sentiment_features,
    normalize_sentiment_response,
    parse_fundamental_feature_rows,
)


class EODHDEnrichmentTests(unittest.TestCase):
    def test_fundamental_parser_only_emits_rows_with_availability_date(self) -> None:
        payload = {
            "Financials": {
                "Income_Statement": {
                    "quarterly": {
                        "2024-03-31": {
                            "date": "2024-03-31",
                            "filing_date": "2024-04-25",
                            "totalRevenue": "1000",
                            "netIncome": "100",
                            "grossProfit": "400",
                        },
                        "2024-06-30": {
                            "date": "2024-06-30",
                            "totalRevenue": "1100",
                        },
                    }
                },
                "Balance_Sheet": {
                    "quarterly": {
                        "2024-03-31": {
                            "date": "2024-03-31",
                            "filing_date": "2024-04-25",
                            "totalAssets": "2000",
                            "totalLiab": "800",
                        }
                    }
                },
                "Cash_Flow": {
                    "quarterly": {
                        "2024-03-31": {
                            "date": "2024-03-31",
                            "filing_date": "2024-04-25",
                            "totalCashFromOperatingActivities": "150",
                        }
                    }
                },
            }
        }

        rows = parse_fundamental_feature_rows("AAPL.US", payload)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["availability_date"], "2024-04-25")
        self.assertAlmostEqual(rows[0]["fundamental_net_margin"], 0.1)
        self.assertAlmostEqual(rows[0]["fundamental_debt_to_assets"], 0.4)

    def test_fundamental_join_is_asof_and_missing_does_not_drop_rows(self) -> None:
        features = [
            {"ticker": "AAPL", "date": "2024-04-24", "close": 10.0},
            {"ticker": "AAPL", "date": "2024-04-25", "close": 11.0},
            {"ticker": "MSFT", "date": "2024-04-25", "close": 20.0},
        ]
        fundamentals = [
            {
                "ticker": "AAPL",
                "eodhd_symbol": "AAPL.US",
                "availability_date": "2024-04-25",
                "source_period_end": "2024-03-31",
                "source_frequency": "quarterly",
                "fundamental_revenue": 1000.0,
            }
        ]

        enriched = add_fundamental_features(features, fundamentals)

        self.assertEqual(len(enriched), 3)
        by_key = {(row["ticker"], row["date"]): row for row in enriched}
        self.assertEqual(by_key[("AAPL", "2024-04-24")]["fundamental_missing"], 1.0)
        self.assertEqual(by_key[("AAPL", "2024-04-25")]["fundamental_missing"], 0.0)
        self.assertEqual(by_key[("MSFT", "2024-04-25")]["fundamental_missing"], 1.0)

    def test_sentiment_parser_and_one_trading_day_lag(self) -> None:
        payload = {
            "AAPL.US": [
                {"date": "2024-01-02", "count": 3, "normalized": 0.5},
                {"date": "2024-01-03", "count": 1, "normalized": -0.2},
            ]
        }
        sentiment = normalize_sentiment_response(payload)
        features = [
            {"ticker": "AAPL", "date": "2024-01-02", "close": 10.0},
            {"ticker": "AAPL", "date": "2024-01-03", "close": 11.0},
            {"ticker": "AAPL", "date": "2024-01-04", "close": 12.0},
        ]

        enriched = add_sentiment_features(features, sentiment)

        self.assertEqual(enriched[0]["sentiment_count"], 0.0)
        self.assertEqual(enriched[0]["sentiment_missing"], 1.0)
        self.assertEqual(enriched[1]["sentiment_count"], 3.0)
        self.assertAlmostEqual(enriched[1]["sentiment_normalized"], 0.5)
        self.assertEqual(enriched[2]["sentiment_count"], 1.0)
        self.assertAlmostEqual(enriched[2]["sentiment_normalized"], -0.2)


if __name__ == "__main__":
    unittest.main()
