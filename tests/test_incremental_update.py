from __future__ import annotations

from datetime import date
import unittest

from src.data.incremental_update import (
    compute_latest_prediction_windows,
    determine_incremental_fetch_start,
    max_bar_date,
    merge_daily_bar_rows,
)


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


if __name__ == "__main__":
    unittest.main()
