from __future__ import annotations

from datetime import date
import csv
import tempfile
import unittest
from pathlib import Path

from src.data.incremental_update import (
    compute_latest_prediction_windows,
    determine_incremental_fetch_start,
    max_bar_date,
    merge_daily_bar_rows,
)
from scripts.merge_incremental_feature_updates import _merge_one, _update_normalized_for_dates


class IncrementalUpdateTests(unittest.TestCase):
    def test_determine_incremental_fetch_start_uses_overlap_tail(self) -> None:
        rows = [
            {"ticker": "AAA", "date": "2024-01-02", "adjusted": True},
            {"ticker": "AAA", "date": "2024-01-10", "adjusted": True},
        ]

        fetch_start = determine_incremental_fetch_start(
            rows,
            fallback_start_date=date(2023, 1, 1),
            overlap_days=3,
        )

        self.assertEqual(fetch_start, date(2024, 1, 8))

    def test_determine_incremental_fetch_start_uses_fallback_for_empty_dataset(self) -> None:
        fetch_start = determine_incremental_fetch_start(
            [],
            fallback_start_date=date(2024, 1, 1),
            overlap_days=7,
        )

        self.assertEqual(fetch_start, date(2024, 1, 1))

    def test_merge_daily_bar_rows_replaces_duplicate_with_incoming_row(self) -> None:
        existing = [
            {"ticker": "aaa", "date": "2024-01-02", "adjusted": True, "close": 10.0},
            {"ticker": "AAA", "date": "2024-01-03", "adjusted": True, "close": 11.0},
        ]
        incoming = [
            {"ticker": "AAA", "date": "2024-01-03", "adjusted": True, "close": 12.0},
            {"ticker": "BBB", "date": "2024-01-02", "adjusted": True, "close": 20.0},
        ]

        merged = merge_daily_bar_rows(existing, incoming)
        close_by_key = {(row["ticker"], row["date"]): row["close"] for row in merged}

        self.assertEqual(len(merged), 3)
        self.assertEqual(close_by_key[("AAA", "2024-01-03")], 12.0)
        self.assertEqual(max_bar_date(merged), date(2024, 1, 3))

    def test_compute_latest_prediction_windows_omits_benchmark_and_targets(self) -> None:
        rows = []
        for idx in range(4):
            rows.append({"ticker": "AAA", "date": f"2024-01-0{idx + 1}", "close": 10.0 + idx})
            rows.append({"ticker": "SPY", "date": f"2024-01-0{idx + 1}", "close": 100.0 + idx})

        windows = compute_latest_prediction_windows(
            rows,
            window_length=3,
            target_horizon_days=5,
            benchmark_ticker="SPY",
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["ticker"], "AAA")
        self.assertEqual(windows[0]["anchor_date"], "2024-01-04")
        self.assertEqual(windows[0]["window_start_date"], "2024-01-02")
        self.assertEqual(windows[0]["target_status"], "pending_future_return")
        self.assertIsNone(windows[0]["target_return"])
        self.assertTrue(windows[0]["inference_ready"])

    def test_compute_latest_prediction_windows_can_cut_off_anchor_date(self) -> None:
        rows = [{"ticker": "AAA", "date": f"2024-01-0{idx + 1}", "close": 10.0 + idx} for idx in range(5)]

        windows = compute_latest_prediction_windows(
            rows,
            window_length=3,
            target_horizon_days=5,
            anchor_date="2024-01-04",
        )

        self.assertEqual(windows[0]["anchor_date"], "2024-01-04")
        self.assertEqual(windows[0]["window_start_date"], "2024-01-02")
        self.assertEqual(windows[0]["anchor_selection"], "latest_on_or_before_anchor_date")

    def test_merge_incremental_feature_updates_preserves_ticker_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "daily_features.csv"
            update = root / "daily_features_incremental_updates.csv"
            archive = root / "archive"
            fieldnames = ["ticker", "date", "close", "feature"]
            with base.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(
                    [
                        {"ticker": "AAA", "date": "2024-01-01", "close": "10", "feature": "1"},
                        {"ticker": "AAA", "date": "2024-01-02", "close": "11", "feature": "2"},
                        {"ticker": "BBB", "date": "2024-01-01", "close": "20", "feature": "3"},
                    ]
                )
            with update.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(
                    [
                        {"ticker": "AAA", "date": "2024-01-02", "close": "12", "feature": "22"},
                        {"ticker": "AAA", "date": "2024-01-03", "close": "13", "feature": "23"},
                        {"ticker": "CCC", "date": "2024-01-01", "close": "30", "feature": "4"},
                    ]
                )

            result = _merge_one(
                base_path=base,
                update_path=update,
                archive_dir=archive,
                archive_updates=True,
            )

            self.assertTrue(result["updated"])
            self.assertFalse(update.exists())
            with base.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [(row["ticker"], row["date"], row["close"]) for row in rows],
                [
                    ("AAA", "2024-01-01", "10"),
                    ("AAA", "2024-01-02", "12"),
                    ("AAA", "2024-01-03", "13"),
                    ("BBB", "2024-01-01", "20"),
                    ("CCC", "2024-01-01", "30"),
                ],
            )

    def test_incremental_feature_merge_updates_existing_normalized_dates_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed"
            raw = root / "raw"
            processed.mkdir()
            raw.mkdir()

            with (raw / "eodhd_equity_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["symbol", "gics_sector", "gics_sub_industry"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"symbol": "AAA", "gics_sector": "Tech", "gics_sub_industry": "Software"},
                        {"symbol": "BBB", "gics_sector": "Tech", "gics_sub_industry": "Hardware"},
                    ]
                )

            feature_fields = ["ticker", "date", "volume", "dollar_volume", "return_1d"]
            with (processed / "daily_features.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=feature_fields)
                writer.writeheader()
                writer.writerows(
                    [
                        {"ticker": "AAA", "date": "2024-01-01", "volume": "10", "dollar_volume": "100", "return_1d": "0.01"},
                        {"ticker": "AAA", "date": "2024-01-02", "volume": "100", "dollar_volume": "1000", "return_1d": "0.02"},
                        {"ticker": "BBB", "date": "2024-01-02", "volume": "20", "dollar_volume": "200", "return_1d": "-0.01"},
                    ]
                )

            normalized_fields = [
                "ticker",
                "date",
                "volume",
                "gics_sector",
                "log1p_volume",
                "log1p_volume__cs_pct",
                "cs_universe_count",
            ]
            with (processed / "daily_features_normalized.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=normalized_fields)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "ticker": "AAA",
                            "date": "2024-01-01",
                            "volume": "10",
                            "gics_sector": "OLD",
                            "log1p_volume": "unchanged",
                            "log1p_volume__cs_pct": "unchanged",
                            "cs_universe_count": "unchanged",
                        },
                        {"ticker": "AAA", "date": "2024-01-02", "volume": "1"},
                        {"ticker": "BBB", "date": "2024-01-02", "volume": "2"},
                    ]
                )
            (processed / "daily_features_normalized_manifest.json").write_text(
                '{"row_count": 3}\n',
                encoding="utf-8",
            )

            result = _update_normalized_for_dates(
                dataset_root=root,
                affected_dates={"2024-01-02"},
                archive_dir=processed / "archive",
            )

            self.assertTrue(result["updated"])
            with (processed / "daily_features_normalized.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            by_key = {(row["ticker"], row["date"]): row for row in rows}
            self.assertEqual(by_key[("AAA", "2024-01-01")]["log1p_volume"], "unchanged")
            self.assertEqual(by_key[("AAA", "2024-01-02")]["volume"], "100.0")
            self.assertEqual(by_key[("AAA", "2024-01-02")]["gics_sector"], "Tech")
            self.assertEqual(by_key[("BBB", "2024-01-02")]["volume"], "20.0")
            manifest_text = (processed / "daily_features_normalized_manifest.json").read_text(encoding="utf-8")
            self.assertIn("incremental_same_date_update", manifest_text)


if __name__ == "__main__":
    unittest.main()
