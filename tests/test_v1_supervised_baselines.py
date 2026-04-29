from __future__ import annotations

from datetime import date, timedelta
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.episode_eligibility import EpisodeEligibilityConfig
from src.data.massive_stage1 import compute_daily_features
from src.data.normalization import compute_normalized_feature_rows
from src.data.v1_dataset import (
    STATIC_CATEGORICAL_COLUMNS,
    build_category_vocabularies,
    build_latest_v1_feature_sets,
    build_sequence_feature_store,
    build_v1_dataset,
    build_walk_forward_folds,
    classification_target_column,
    chronological_split,
    encode_static_categories,
    identifier_model_input_columns,
    prepare_xy,
    raw_level_model_input_columns,
    rows_for_dates,
)
from src.models.v1_baselines import (
    build_leaderboard,
    evaluate_predictions,
    load_model_bundle,
    make_model,
    prediction_frame,
    save_model_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _bars(ticker: str, start_price: float, days: int = 90) -> list[dict[str, object]]:
    rows = []
    start = date(2024, 1, 2)
    for idx in range(days):
        drift = 0.35 + (hash(ticker) % 7) * 0.01
        close = start_price + idx * drift
        rows.append(
            {
                "ticker": ticker,
                "date": (start + timedelta(days=idx)).isoformat(),
                "open": close - 0.1,
                "high": close + 0.3,
                "low": close - 0.3,
                "close": close,
                "volume": 1000.0 + idx + (hash(ticker) % 17),
                "adjusted": True,
                "dollar_volume": close * (1000.0 + idx),
            }
        )
    return rows


def _ticker_name(idx: int) -> str:
    return f"T{idx:02d}"


def _stock_and_context_frames(*, ticker_count: int = 12, days: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_bars: list[dict[str, object]] = []
    metadata: dict[str, dict[str, str]] = {}
    sub_industries = ["Software", "Hardware", "Semis", "IT Services"]
    for idx in range(ticker_count):
        ticker = _ticker_name(idx)
        stock_bars.extend(_bars(ticker, 20.0 + idx * 3.0, days=days))
        metadata[ticker] = {
            "gics_sector": "Information Technology",
            "gics_sub_industry": sub_industries[idx % len(sub_industries)],
        }
    context_bars = _bars("SPY", 100.0, days=days) + _bars("XLK", 50.0, days=days)
    stock_features = compute_daily_features(stock_bars)
    stock_normalized = compute_normalized_feature_rows(stock_features, metadata)
    context_features = compute_daily_features(context_bars)
    return pd.DataFrame(stock_normalized), pd.DataFrame(context_features)


def _write_dataset_root(root: Path, stock_features: pd.DataFrame, context_features: pd.DataFrame) -> None:
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    stock_features.to_csv(processed / "daily_features_normalized.csv", index=False)
    context_features.to_csv(processed / "market_context_features.csv", index=False)


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
        self.assertIn(classification_target_column(), dataset.classification_target_columns)
        self.assertIn("stock_relative_market_sector", dataset.feature_sets)
        self.assertIn("stock_relative_market_sector_compact", dataset.feature_sets)
        sector_cols = dataset.feature_columns["stock_relative_market_sector"]
        compact_sector_cols = dataset.feature_columns["stock_relative_market_sector_compact"]
        self.assertFalse(any("close_to_vwap_pct" in col for col in sector_cols))
        self.assertFalse(any("vwap" in col for col in sector_cols))
        self.assertFalse(any("transactions" in col for col in sector_cols))
        self.assertTrue(any(col.startswith("market_context_") for col in sector_cols))
        self.assertTrue(any(col.startswith("sector_context_") for col in sector_cols))
        self.assertTrue(any("stock_vs_market_return_1d" in col for col in sector_cols))
        self.assertTrue(any("stock_vs_sector_return_5d" in col for col in sector_cols))
        self.assertTrue(any("close_location" in col for col in sector_cols))
        self.assertTrue(any("true_range_pct" in col for col in sector_cols))
        self.assertTrue(any(col.startswith("market_context_") for col in compact_sector_cols))
        self.assertTrue(any(col.startswith("sector_context_") for col in compact_sector_cols))
        self.assertLess(len(compact_sector_cols), len(sector_cols))
        self.assertGreater(len(dataset.targets), 0)
        self.assertIn("window_row_count", dataset.metadata.columns)

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
        _, _, y_val_class = prepare_xy(dataset, "stock_only", split, "val", task_type="classification")

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
        self.assertEqual(y_val_class.shape[1], 1)
        self.assertEqual(int(leaderboard.iloc[0]["leaderboard_rank"]), 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_v1_supervised_model.pkl"
            save_model_bundle(path, model=model, metadata={"model_name": "ridge"})
            loaded = load_model_bundle(path)
            loaded_pred = loaded["model"].predict(x_val)
        self.assertEqual(loaded_pred.shape, pred.shape)
        self.assertGreaterEqual(len(train_meta), 1)

    def test_single_horizon_1d_prediction_arrays_are_accepted(self) -> None:
        metadata = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "anchor_date": ["2024-01-02", "2024-01-02", "2024-01-02"],
                "anchor_close": [10.0, 20.0, 30.0],
            }
        )
        y_true = pd.DataFrame({"market_adjusted_return_5d": [0.01, -0.02, 0.03]})
        y_pred = np.array([0.02, -0.01, 0.01], dtype=float)

        metrics = evaluate_predictions(
            metadata,
            y_true,
            y_pred,
            target_columns=["market_adjusted_return_5d"],
            model_name="ridge",
            feature_set="stock_only",
            split_name="val",
        )
        frame = prediction_frame(
            metadata,
            y_pred,
            target_columns=["market_adjusted_return_5d"],
            model_name="ridge",
            feature_set="stock_only",
            y_true=y_true,
        )

        self.assertEqual(len(metrics), 1)
        self.assertIn("pred_market_adjusted_return_5d", frame.columns)

    def test_walk_forward_folds_and_static_vocabularies_are_pit_safe(self) -> None:
        stock_features, context_features = _stock_and_context_frames(days=70)
        dataset = build_v1_dataset(
            stock_features,
            context_features,
            horizons=(1, 5),
            window_length=10,
            benchmark_ticker="SPY",
        )

        folds = build_walk_forward_folds(
            dataset.metadata,
            min_train_dates=12,
            val_block_size=5,
            oos_block_size=5,
            purge_gap=5,
        )

        self.assertGreaterEqual(len(folds), 1)
        first = folds[0]
        self.assertLess(max(first.train_dates), min(first.val_dates))
        self.assertLess(max(first.val_dates), min(first.oos_dates))
        self.assertGreaterEqual(len(set(first.oos_dates)), len(first.oos_dates))

        train_rows = rows_for_dates(dataset.metadata, first.train_dates)
        oos_rows = rows_for_dates(dataset.metadata, first.oos_dates)
        train_meta = dataset.metadata.loc[train_rows].reset_index(drop=True)
        oos_meta = dataset.metadata.loc[oos_rows].reset_index(drop=True)
        vocabularies = build_category_vocabularies(train_meta, columns=STATIC_CATEGORICAL_COLUMNS)
        encoded_train = encode_static_categories(train_meta, vocabularies, columns=STATIC_CATEGORICAL_COLUMNS)
        encoded_oos = encode_static_categories(oos_meta, vocabularies, columns=STATIC_CATEGORICAL_COLUMNS)

        self.assertEqual(set(encoded_train.keys()), set(STATIC_CATEGORICAL_COLUMNS))
        self.assertEqual(len(encoded_train["gics_sector"]), len(train_meta))
        self.assertEqual(len(encoded_oos["gics_sub_industry"]), len(oos_meta))

    def test_sequence_static_model_smoke(self) -> None:
        stock_features, context_features = _stock_and_context_frames(days=75)
        dataset = build_v1_dataset(
            stock_features,
            context_features,
            horizons=(1, 5),
            window_length=10,
            benchmark_ticker="SPY",
        )
        folds = build_walk_forward_folds(
            dataset.metadata,
            min_train_dates=12,
            val_block_size=5,
            oos_block_size=5,
            purge_gap=5,
        )
        fold = folds[0]
        train_rows = rows_for_dates(dataset.metadata, fold.train_dates)
        val_rows = rows_for_dates(dataset.metadata, fold.val_dates)
        train_meta = dataset.metadata.loc[train_rows].reset_index(drop=True)
        val_meta = dataset.metadata.loc[val_rows].reset_index(drop=True)
        y_train = dataset.targets.loc[train_rows, dataset.target_columns].reset_index(drop=True)
        y_val = dataset.targets.loc[val_rows, dataset.target_columns].reset_index(drop=True)
        vocabularies = build_category_vocabularies(train_meta, columns=STATIC_CATEGORICAL_COLUMNS)
        store = build_sequence_feature_store(
            stock_features,
            "stock_relative_market_sector_sequence",
            context_features=context_features,
            benchmark_ticker="SPY",
        )
        self.assertTrue(any(col.startswith("market_context_") for col in store.feature_columns))
        self.assertTrue(any(col.startswith("sector_context_") for col in store.feature_columns))
        self.assertIn("stock_vs_market_return_1d", store.feature_columns)
        self.assertIn("stock_vs_sector_return_5d", store.feature_columns)
        self.assertIn("market_context_missing", store.feature_columns)
        self.assertIn("sector_context_missing", store.feature_columns)
        self.assertEqual(store.get_window(train_meta.iloc[0]["ticker"], 9, 10).shape[1], len(store.feature_columns))
        x_train = {
            "store": store,
            "metadata": train_meta,
            "static_categorical": encode_static_categories(train_meta, vocabularies, columns=STATIC_CATEGORICAL_COLUMNS),
        }
        x_val = {
            "store": store,
            "metadata": val_meta,
            "static_categorical": encode_static_categories(val_meta, vocabularies, columns=STATIC_CATEGORICAL_COLUMNS),
        }

        model = make_model("torch_seq_static", window_length=10)
        model.fit(x_train, y_train, val_x=x_val, val_y=y_val)
        pred = model.predict(x_val)

        self.assertEqual(pred.shape, (len(val_meta), len(dataset.target_columns)))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "torch_seq_static.pkl"
            save_model_bundle(path, model=model, metadata={"model_name": "torch_seq_static"})
            loaded = load_model_bundle(path)
            loaded_pred = loaded["model"].predict(x_val)
        self.assertTrue(np.allclose(pred, loaded_pred))

    def test_classification_target_and_model_smoke(self) -> None:
        stock_features, context_features = _stock_and_context_frames(days=75)
        dataset = build_v1_dataset(
            stock_features,
            context_features,
            horizons=(1, 5, 20),
            window_length=10,
            benchmark_ticker="SPY",
        )
        self.assertIn(classification_target_column(), dataset.targets.columns)
        self.assertTrue(set(dataset.targets[classification_target_column()].unique()).issubset({0.0, 1.0}))
        split = chronological_split(dataset.metadata, train_fraction=0.6, val_fraction=0.2)
        _, x_train, y_train = prepare_xy(dataset, "stock_only", split, "train", task_type="classification")
        val_meta, x_val, y_val = prepare_xy(dataset, "stock_only", split, "val", task_type="classification")
        model = make_model("logistic_regression", task_type="classification").fit(x_train, y_train)
        pred = model.predict(x_val)
        self.assertEqual(pred.shape, (len(val_meta), 1))
        self.assertTrue(np.all((pred >= 0.0) & (pred <= 1.0)))

    def test_latest_feature_sets_use_target_pending_windows(self) -> None:
        stock_features, context_features = _stock_and_context_frames()

        metadata, feature_sets, feature_columns = build_latest_v1_feature_sets(
            stock_features,
            context_features,
            window_length=10,
            benchmark_ticker="SPY",
        )

        self.assertIn("stock_relative_market_sector", feature_sets)
        self.assertIn("stock_relative_market_sector_compact", feature_sets)
        self.assertEqual(len(metadata["ticker"].unique()), 12)
        self.assertTrue(feature_columns["stock_relative_market_sector"])
        self.assertLess(
            len(feature_columns["stock_relative_market_sector_compact"]),
            len(feature_columns["stock_relative_market_sector"]),
        )

    def test_episode_eligibility_filter_removes_untradable_windows(self) -> None:
        stock_features, context_features = _stock_and_context_frames(ticker_count=3, days=45)
        stock_features["exchange"] = "NASDAQ"
        stock_features.loc[stock_features["ticker"] == "T01", "dollar_volume"] = 100.0
        stock_features.loc[stock_features["ticker"] == "T02", "exchange"] = "OTC"
        config = EpisodeEligibilityConfig(
            min_history_days=20,
            valid_ohlcv_lookback=20,
            min_valid_ohlcv_days=20,
            dollar_volume_lookback=10,
            min_avg_dollar_volume=5_000.0,
            min_price=5.0,
            allowed_exchanges=("NASDAQ",),
        )

        dataset = build_v1_dataset(
            stock_features,
            context_features,
            horizons=(1, 5),
            window_length=10,
            benchmark_ticker="SPY",
            eligibility_config=config,
        )

        self.assertEqual(set(dataset.metadata["ticker"].unique()), {"T00"})
        self.assertIn("episode_eligible", dataset.metadata.columns)
        self.assertTrue(dataset.metadata["episode_eligible"].all())

    def test_sequence_feature_store_can_include_market_and_sector_context(self) -> None:
        stock_features, context_features = _stock_and_context_frames(days=75)

        store = build_sequence_feature_store(
            stock_features,
            "stock_relative_market_sector_sequence",
            context_features=context_features,
            benchmark_ticker="SPY",
        )

        self.assertTrue(any(col.endswith("__cs_z") for col in store.feature_columns))
        self.assertTrue(any(col.endswith("__sector_cs_z") for col in store.feature_columns))
        self.assertTrue(any(col.startswith("market_context_") for col in store.feature_columns))
        self.assertTrue(any(col.startswith("sector_context_") for col in store.feature_columns))
        self.assertIn("market_context_missing", store.feature_columns)
        self.assertIn("sector_context_missing", store.feature_columns)
        self.assertEqual(store.get_window("T00", 9, 10).shape, (10, len(store.feature_columns)))

        compact_store = build_sequence_feature_store(
            stock_features,
            "stock_relative_market_sector_compact_sequence",
            context_features=context_features,
            benchmark_ticker="SPY",
        )
        self.assertTrue(any(col.startswith("market_context_") for col in compact_store.feature_columns))
        self.assertTrue(any(col.startswith("sector_context_") for col in compact_store.feature_columns))
        self.assertLess(len(compact_store.feature_columns), len(store.feature_columns))

    def test_sequence_feature_store_can_include_sentiment_without_ticker_input(self) -> None:
        stock_features, context_features = _stock_and_context_frames(days=75)
        stock_features["sentiment_count"] = 1.0
        stock_features["sentiment_normalized"] = 0.2
        stock_features["sentiment_missing"] = 0.0

        store = build_sequence_feature_store(
            stock_features,
            "stock_relative_market_sector_sentiment_sequence",
            context_features=context_features,
            benchmark_ticker="SPY",
        )

        self.assertIn("sentiment_count", store.feature_columns)
        self.assertEqual(identifier_model_input_columns(store.feature_columns), [])

    def test_model_feature_columns_reject_raw_level_inputs(self) -> None:
        stock_features, context_features = _stock_and_context_frames(days=75)

        with self.assertRaises(ValueError):
            build_sequence_feature_store(
                stock_features,
                "stock_only_sequence",
                benchmark_ticker="SPY",
                feature_columns=["open", "log_return_1d"],
            )

        dataset = build_v1_dataset(
            stock_features,
            context_features,
            horizons=(1, 5),
            window_length=10,
            benchmark_ticker="SPY",
        )
        for feature_set, columns in dataset.feature_columns.items():
            self.assertEqual(raw_level_model_input_columns(columns), [], feature_set)
            self.assertEqual(identifier_model_input_columns(columns), [], feature_set)

    def test_identifier_columns_are_metadata_only_and_augmented_feature_sets_exist(self) -> None:
        stock_features, context_features = _stock_and_context_frames(days=75)
        stock_features["eodhd_symbol"] = stock_features["ticker"] + ".US"
        stock_features["sentiment_count"] = 1.0
        stock_features["sentiment_normalized"] = 0.1
        stock_features["sentiment_missing"] = 0.0
        stock_features["fundamental_revenue"] = 1000.0
        stock_features["fundamental_missing"] = 0.0
        stock_features["fundamental_staleness_days"] = 5.0

        dataset = build_v1_dataset(
            stock_features,
            context_features,
            horizons=(1, 5),
            window_length=10,
            benchmark_ticker="SPY",
        )

        self.assertIn("ticker", dataset.metadata.columns)
        self.assertIn("stock_only_sentiment", dataset.feature_sets)
        self.assertIn("stock_only_fundamentals", dataset.feature_sets)
        self.assertIn("stock_only_fundamentals_sentiment", dataset.feature_sets)
        for feature_set, columns in dataset.feature_columns.items():
            self.assertNotIn("ticker", columns)
            self.assertNotIn("eodhd_symbol", columns)
            self.assertEqual(identifier_model_input_columns(columns), [], feature_set)
        self.assertTrue(any("sentiment_count" in col for col in dataset.feature_columns["stock_only_sentiment"]))
        self.assertTrue(any("fundamental_revenue" in col for col in dataset.feature_columns["stock_only_fundamentals"]))

    def test_end_to_end_walk_forward_and_prediction_smoke(self) -> None:
        stock_features, context_features = _stock_and_context_frames(days=55)
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            dataset_root = temp_root / "dataset"
            output_root = temp_root / "artifacts"
            _write_dataset_root(dataset_root, stock_features, context_features)

            legacy_run = output_root / "legacy_holdout"
            train_script = REPO_ROOT / "scripts" / "train_v1_supervised_baselines.py"
            predict_script = REPO_ROOT / "scripts" / "predict_v1_supervised_baselines.py"

            subprocess.run(
                [
                    sys.executable,
                    str(train_script),
                    "--dataset-root",
                    str(dataset_root),
                    "--output-root",
                    str(output_root),
                    "--run-name",
                    legacy_run.name,
                    "--eval-mode",
                    "holdout",
                    "--models",
                    "ridge",
                    "--feature-sets",
                    "stock_only",
                    "--horizons",
                    "1,5",
                    "--window-length",
                    "10",
                    "--train-fraction",
                    "0.6",
                    "--val-fraction",
                    "0.2",
                    "--disable-episode-eligibility-filter",
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            walk_run_name = "walk_forward_smoke"
            subprocess.run(
                [
                    sys.executable,
                    str(train_script),
                    "--dataset-root",
                    str(dataset_root),
                    "--output-root",
                    str(output_root),
                    "--run-name",
                    walk_run_name,
                    "--eval-mode",
                    "walk_forward",
                    "--models",
                    "ridge,torch_seq_static",
                    "--feature-sets",
                    "stock_only,stock_relative",
                    "--horizons",
                    "1,5",
                    "--window-length",
                    "10",
                    "--walk-forward-min-train-dates",
                    "8",
                    "--walk-forward-val-block-size",
                    "4",
                    "--walk-forward-oos-block-size",
                    "4",
                    "--walk-forward-max-folds",
                    "2",
                    "--walk-forward-purge-gap",
                    "5",
                    "--final-stop-block-size",
                    "4",
                    "--compare-against-run",
                    str(legacy_run),
                    "--disable-episode-eligibility-filter",
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            walk_run = output_root / walk_run_name
            self.assertTrue((walk_run / "folds.json").exists())
            self.assertTrue((walk_run / "fold_metrics.csv").exists())
            self.assertTrue((walk_run / "oos_predictions.csv").exists())
            self.assertTrue((walk_run / "oos_leaderboard.csv").exists())
            self.assertTrue((walk_run / "classification_oos_predictions.csv").exists())
            self.assertTrue((walk_run / "classification_oos_leaderboard.csv").exists())
            self.assertTrue((walk_run / "final_models.json").exists())
            self.assertTrue((walk_run / "final_classification_models.json").exists())
            self.assertTrue((walk_run / "comparison.csv").exists())
            self.assertTrue((walk_run / "comparison_summary.json").exists())

            comparison_summary = json.loads((walk_run / "comparison_summary.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(int(comparison_summary["matched_combo_count"]), 1)
            folds = json.loads((walk_run / "folds.json").read_text(encoding="utf-8"))["folds"]
            self.assertLessEqual(len(folds), 2)

            subprocess.run(
                [
                    sys.executable,
                    str(predict_script),
                    "--dataset-root",
                    str(dataset_root),
                    "--run-dir",
                    str(walk_run),
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            latest_predictions = pd.read_csv(walk_run / "latest_predictions.csv")
            self.assertIn("torch_seq_static", set(latest_predictions["model_name"]))
            self.assertIn("ridge", set(latest_predictions["model_name"]))
            self.assertIn("classification", set(latest_predictions["task_type"]))

    @unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None, "lightgbm is not installed")
    def test_lightgbm_fit_predict_smoke(self) -> None:
        stock_features, context_features = _stock_and_context_frames(days=60)
        dataset = build_v1_dataset(
            stock_features,
            context_features,
            horizons=(1, 5),
            window_length=10,
            benchmark_ticker="SPY",
        )
        split = chronological_split(dataset.metadata, train_fraction=0.6, val_fraction=0.2)
        _, x_train, y_train = prepare_xy(dataset, "stock_only", split, "train")
        _, x_val, y_val = prepare_xy(dataset, "stock_only", split, "val")
        model = make_model("lightgbm").fit(x_train, y_train, val_x=x_val, val_y=y_val)
        pred = model.predict(x_val)
        self.assertEqual(pred.shape, (len(x_val), len(dataset.target_columns)))

    @unittest.skipUnless(importlib.util.find_spec("xgboost") is not None, "xgboost is not installed")
    def test_xgboost_fit_predict_smoke(self) -> None:
        stock_features, context_features = _stock_and_context_frames(days=60)
        dataset = build_v1_dataset(
            stock_features,
            context_features,
            horizons=(1, 5),
            window_length=10,
            benchmark_ticker="SPY",
        )
        split = chronological_split(dataset.metadata, train_fraction=0.6, val_fraction=0.2)
        _, x_train, y_train = prepare_xy(dataset, "stock_only", split, "train")
        _, x_val, y_val = prepare_xy(dataset, "stock_only", split, "val")
        model = make_model("xgboost").fit(x_train, y_train, val_x=x_val, val_y=y_val)
        pred = model.predict(x_val)
        self.assertEqual(pred.shape, (len(x_val), len(dataset.target_columns)))


if __name__ == "__main__":
    unittest.main()
