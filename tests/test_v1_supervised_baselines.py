from __future__ import annotations

from datetime import date, timedelta
import unittest
from pathlib import Path

import pandas as pd

from src.data.massive_stage1 import compute_daily_features
from src.data.normalization import compute_normalized_feature_rows
from src.data.v1_dataset import build_latest_v1_feature_sets, build_v1_dataset, chronological_split, prepare_xy
from src.models.v1_baselines import (
    build_leaderboard,
    evaluate_predictions,
    load_model_bundle,
    make_model,
    save_model_bundle,
)


def _bars(ticker: str, start_price: float, days: int = 90) -> list[dict[str, object]]:
    rows = []
    start = date(2024, 1, 2)
    for idx in range(days):
        close = start_price + idx * 0.5
        rows.append(
            {
                "ticker": ticker,
                "date": (start + timedelta(days=idx)).isoformat(),
                "open": close - 0.1,
                "high": close + 0.3,
                "low": close - 0.3,
                "close": close,
                "volume": 1000.0 + idx,
                "vwap": close,
                "transactions": 10.0 + idx,
                "adjusted": True,
                "dollar_volume": close * (1000.0 + idx),
            }
        )
    return rows


def _stock_and_context_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_bars = _bars("AAA", 20.0) + _bars("BBB", 30.0) + _bars("CCC", 40.0)
    context_bars = _bars("SPY", 100.0) + _bars("XLK", 50.0)
    stock_features = compute_daily_features(stock_bars)
    metadata = {
        "AAA": {"gics_sector": "Information Technology", "gics_sub_industry": "Software"},
        "BBB": {"gics_sector": "Information Technology", "gics_sub_industry": "Hardware"},
        "CCC": {"gics_sector": "Information Technology", "gics_sub_industry": "Semis"},
    }
    stock_normalized = compute_normalized_feature_rows(stock_features, metadata)
    context_features = compute_daily_features(context_bars)
    return pd.DataFrame(stock_normalized), pd.DataFrame(context_features)


class V1SupervisedBaselineTests(unittest.TestCase):
    def test_build_v1_dataset_creates_multi_horizon_targets_and_context_features(self) -> None:
        stock_features, context_features = _stock_and_context_frames()

        dataset = build_v1_dataset(
            stock_features,
            context_features,
            horizons=(1, 5, 10, 20),
            window_length=10,
            benchmark_ticker="SPY",
        )

        self.assertIn("market_adjusted_return_20d", dataset.target_columns)
        self.assertIn("stock_relative_market_sector", dataset.feature_sets)
        sector_cols = dataset.feature_columns["stock_relative_market_sector"]
        self.assertTrue(any(col.startswith("market_context_") for col in sector_cols))
        self.assertTrue(any(col.startswith("sector_context_") for col in sector_cols))
        self.assertGreater(len(dataset.targets), 0)

    def test_train_evaluate_save_and_reload_model(self) -> None:
        stock_features, context_features = _stock_and_context_frames()
        dataset = build_v1_dataset(
            stock_features,
            context_features,
            horizons=(1, 5),
            window_length=10,
            benchmark_ticker="SPY",
        )
        split = chronological_split(dataset.metadata, train_fraction=0.6, val_fraction=0.2)
        train_meta, x_train, y_train = prepare_xy(dataset, "stock_only", split, "train")
        val_meta, x_val, y_val = prepare_xy(dataset, "stock_only", split, "val")

        model = make_model("ridge").fit(x_train, y_train)
        pred = model.predict(x_val)
        metrics = pd.DataFrame(
            evaluate_predictions(
                val_meta,
                y_val,
                pred,
                target_columns=dataset.target_columns,
                model_name="ridge",
                feature_set="stock_only",
                split_name="val",
            )
        )
        leaderboard = build_leaderboard(metrics)

        self.assertEqual(pred.shape[1], 2)
        self.assertEqual(int(leaderboard.iloc[0]["leaderboard_rank"]), 1)
        path = Path("artifacts") / "test_v1_supervised_model.pkl"
        save_model_bundle(path, model=model, metadata={"model_name": "ridge"})
        loaded = load_model_bundle(path)
        loaded_pred = loaded["model"].predict(x_val)
        self.assertEqual(loaded_pred.shape, pred.shape)
        self.assertGreaterEqual(len(train_meta), 1)

    def test_latest_feature_sets_use_target_pending_windows(self) -> None:
        stock_features, context_features = _stock_and_context_frames()

        metadata, feature_sets, feature_columns = build_latest_v1_feature_sets(
            stock_features,
            context_features,
            window_length=10,
            benchmark_ticker="SPY",
        )

        self.assertIn("stock_relative_market_sector", feature_sets)
        self.assertEqual(set(metadata["ticker"]), {"AAA", "BBB", "CCC"})
        self.assertTrue(feature_columns["stock_relative_market_sector"])


if __name__ == "__main__":
    unittest.main()
