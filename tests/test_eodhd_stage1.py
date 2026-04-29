from __future__ import annotations

import unittest

from src.data.eodhd_stage1 import (
    build_metadata_row,
    eodhd_symbol_for_code,
    normalize_eodhd_eod_rows,
    normalize_exchange_symbol_rows,
    parse_fundamentals_general,
)


class EODHDStage1Tests(unittest.TestCase):
    def test_symbol_list_filter_keeps_listed_common_stock_and_excludes_otc(self) -> None:
        rows = [
            {
                "Code": "AAA",
                "Name": "AAA Corp",
                "Country": "USA",
                "Exchange": "NASDAQ",
                "Currency": "USD",
                "Type": "Common Stock",
            },
            {
                "Code": "BBB",
                "Name": "BBB ETF",
                "Country": "USA",
                "Exchange": "NYSE",
                "Currency": "USD",
                "Type": "ETF",
            },
            {
                "Code": "CCC",
                "Name": "CCC OTC",
                "Country": "USA",
                "Exchange": "PINK",
                "Currency": "USD",
                "Type": "Common Stock",
            },
        ]

        normalized = normalize_exchange_symbol_rows(rows, exchange="NASDAQ", is_delisted=False)

        self.assertEqual([row["ticker"] for row in normalized], ["AAA"])
        self.assertEqual(normalized[0]["eodhd_symbol"], "AAA.US")
        self.assertFalse(normalized[0]["is_delisted"])

    def test_symbol_list_filter_can_mark_delisted_endpoint_rows(self) -> None:
        rows = [
            {
                "Code": "OLD",
                "Name": "Old Co",
                "Country": "USA",
                "Exchange": "NASDAQ",
                "Currency": "USD",
                "Type": "Common Stock",
            }
        ]

        normalized = normalize_exchange_symbol_rows(rows, exchange="NASDAQ", is_delisted=True)

        self.assertEqual(normalized[0]["ticker"], "OLD")
        self.assertTrue(normalized[0]["is_delisted"])

    def test_symbol_list_filter_excludes_units_warrants_rights_and_preferreds(self) -> None:
        rows = [
            {"Code": "A", "Name": "Agilent Technologies Inc", "Exchange": "NYSE", "Currency": "USD", "Type": "Common Stock"},
            {"Code": "ACAAU", "Name": "Athena Consumer Acquisition Corp Units", "Exchange": "NYSE", "Currency": "USD", "Type": "Common Stock"},
            {"Code": "ACHR-WS", "Name": "Archer Aviation Inc Warrants", "Exchange": "NYSE", "Currency": "USD", "Type": "Common Stock"},
            {"Code": "AACBR", "Name": "Artius II Acquisition Inc Rights", "Exchange": "NASDAQ", "Currency": "USD", "Type": "Common Stock"},
            {"Code": "ACONW", "Name": "Aclarion Inc", "Exchange": "NASDAQ", "Currency": "USD", "Type": "Common Stock"},
            {"Code": "AAUGD", "Name": "Ault Disruptive Technologies Corp", "Exchange": "NYSE", "Currency": "USD", "Type": "Common Stock"},
            {"Code": "ABC-PR", "Name": "ABC Preferred Shares", "Exchange": "NYSE", "Currency": "USD", "Type": "Common Stock"},
            {"Code": "SNOW", "Name": "Snowflake Inc", "Exchange": "NYSE", "Currency": "USD", "Type": "Common Stock"},
        ]

        normalized = normalize_exchange_symbol_rows(rows, exchange="NYSE", is_delisted=False)

        self.assertEqual([row["ticker"] for row in normalized], ["A", "SNOW"])

    def test_eod_rows_use_adjusted_close_and_derive_dollar_volume(self) -> None:
        rows = [
            {
                "date": "1995-01-03",
                "open": 10.0,
                "high": 12.0,
                "low": 9.0,
                "close": 10.0,
                "adjusted_close": 5.0,
                "volume": 100.0,
            }
        ]

        normalized = normalize_eodhd_eod_rows(rows, symbol="AAA.US")

        self.assertEqual(normalized[0]["ticker"], "AAA")
        self.assertAlmostEqual(normalized[0]["close"], 5.0)
        self.assertAlmostEqual(normalized[0]["open"], 5.0)
        self.assertAlmostEqual(normalized[0]["adjustment_factor"], 0.5)
        self.assertAlmostEqual(normalized[0]["dollar_volume"], 500.0)
        self.assertNotIn("vwap", normalized[0])
        self.assertNotIn("transactions", normalized[0])

    def test_fundamentals_general_maps_sector_fields_with_unknown_fallback(self) -> None:
        payload = {
            "General::Code": "AAPL",
            "General::Sector": "Technology",
            "General::Industry": "Consumer Electronics",
            "General::GicSector": "Information Technology",
            "General::IsDelisted": False,
        }

        parsed = parse_fundamentals_general(payload)

        self.assertEqual(parsed["gics_sector"], "Information Technology")
        self.assertEqual(parsed["gics_sub_industry"], "Consumer Electronics")
        self.assertFalse(parsed["is_delisted"])

    def test_metadata_row_falls_back_to_symbol_list_when_fundamentals_missing(self) -> None:
        row = {
            "symbol": "AAA",
            "ticker": "AAA",
            "eodhd_symbol": eodhd_symbol_for_code("AAA"),
            "name": "AAA Corp",
            "exchange": "NYSE",
            "currency": "USD",
            "type": "Common Stock",
            "isin": None,
            "country": "USA",
            "is_delisted": None,
        }

        metadata = build_metadata_row(row)

        self.assertEqual(metadata["gics_sector"], "Unknown")
        self.assertEqual(metadata["metadata_source"], "eodhd_symbol_list")


if __name__ == "__main__":
    unittest.main()
