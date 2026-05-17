from __future__ import annotations

import argparse
from datetime import date, timedelta
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.episode_eligibility import EpisodeEligibilityConfig
from src.data.research_universe import ConservativeResearchUniverseConfig
from src.data.massive_stage1 import compute_daily_features
from src.data.normalization import compute_normalized_feature_rows
from src.data.v1_dataset import (
    PATH_5PCT_20D_EVENT_TYPE,
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
    pathwise_classification_labels,
    prepare_xy,
    preferred_stock_feature_path,
    raw_level_model_input_columns,
    rows_for_dates,
)
from src.data.v1_episode_cache import (
    build_episode_cache,
    load_cached_sequence_stores,
    load_cached_v1_dataset,
)
from src.models.v1_baselines import (
    evaluate_classification_predictions,
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

    def test_sustained_pathwise_label_is_explicit_opt_in(self) -> None:
        path_returns = pd.DataFrame(
            {
                "path_1": [0.07, 0.07, 0.02],
                "path_2": [-0.01, 0.04, 0.03],
                "path_3": [-0.04, 0.02, 0.04],
                "path_4": [-0.03, 0.03, 0.04],
                "path_5": [-0.02, 0.04, 0.04],
            }
        )

        anytime = pathwise_classification_labels(
            path_returns,
            threshold=0.05,
            event_type="anytime_pathwise_outperform",
        )
        sustained = pathwise_classification_labels(
            path_returns,
            threshold=0.05,
            event_type="sustained_pathwise_outperform",
        )

        self.assertEqual(anytime.tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(sustained.tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(
            classification_target_column(event_type="sustained_pathwise_outperform"),
            "market_outperform_sustained_20d_gt_5pct",
        )

    def test_path_5pct_20d_label_boundaries_and_missing_path(self) -> None:
        path_returns = pd.DataFrame(
            [
                [0.050, 0.010, 0.000],
                [0.120, -0.050, 0.060],
                [0.049, -0.049, 0.010],
                [0.100, -0.051, 0.020],
                [0.060, np.nan, 0.020],
            ]
        )

        labels = pathwise_classification_labels(
            path_returns,
            threshold=0.05,
            event_type=PATH_5PCT_20D_EVENT_TYPE,
        )

        self.assertEqual(labels.iloc[:4].tolist(), [2.0, 0.0, 1.0, 0.0])
        self.assertTrue(np.isnan(labels.iloc[4]))
        self.assertEqual(
            classification_target_column(event_type=PATH_5PCT_20D_EVENT_TYPE),
            PATH_5PCT_20D_EVENT_TYPE,
        )

    def test_path_5pct_20d_requires_full_forward_window(self) -> None:
        stock_features, context_features = _stock_and_context_frames(ticker_count=2, days=35)
        dataset = build_v1_dataset(
            stock_features,
            context_features,
            horizons=(5,),
            window_length=5,
            benchmark_ticker="SPY",
            classification_horizon=20,
            classification_threshold=0.05,
            classification_event_type=PATH_5PCT_20D_EVENT_TYPE,
        )

        self.assertEqual(dataset.classification_target_columns, [PATH_5PCT_20D_EVENT_TYPE])
        self.assertTrue(set(dataset.targets[PATH_5PCT_20D_EVENT_TYPE].unique()).issubset({0.0, 1.0, 2.0}))
        self.assertIsNotNone(dataset.labeling_summary)
        self.assertGreater(int(dataset.labeling_summary["unlabeled_missing_forward_window_rows"]), 0)
        self.assertEqual(dataset.labeling_summary["mode"], PATH_5PCT_20D_EVENT_TYPE)

    def test_multiclass_prediction_frame_and_metrics_use_class_2_alias(self) -> None:
        from scripts.run_trading_strategy import _prediction_column

        metadata = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                "anchor_date": ["2024-01-02"] * 10,
                "anchor_close": np.arange(10.0, 20.0),
            }
        )
        y_true = pd.DataFrame({PATH_5PCT_20D_EVENT_TYPE: [2, 1, 0, 2, 1, 0, 2, 1, 0, 2]})
        y_pred = np.array(
            [
                [0.05, 0.10, 0.85],
                [0.10, 0.80, 0.10],
                [0.80, 0.10, 0.10],
                [0.10, 0.20, 0.70],
                [0.30, 0.60, 0.10],
                [0.70, 0.20, 0.10],
                [0.20, 0.20, 0.60],
                [0.20, 0.70, 0.10],
                [0.60, 0.30, 0.10],
                [0.20, 0.30, 0.50],
            ]
        )

        frame = prediction_frame(
            metadata,
            y_pred,
            target_columns=[PATH_5PCT_20D_EVENT_TYPE],
            model_name="torch_mlp_classifier",
            feature_set="stock_only",
            y_true=y_true,
            task_type="classification",
        )
        metrics = evaluate_classification_predictions(
            metadata,
            y_true,
            y_pred,
            target_columns=[PATH_5PCT_20D_EVENT_TYPE],
            model_name="torch_mlp_classifier",
            feature_set="stock_only",
            split_name="val",
        )

        self.assertIn("pred_prob_path_5pct_20d_class_0", frame.columns)
        self.assertIn("pred_prob_path_5pct_20d_class_1", frame.columns)
        self.assertIn("pred_prob_path_5pct_20d_class_2", frame.columns)
        self.assertIn("pred_class_path_5pct_20d", frame.columns)
        self.assertIn("pred_score_path_5pct_20d", frame.columns)
        self.assertTrue(np.allclose(frame["pred_prob_path_5pct_20d"], y_pred[:, 2]))
        self.assertTrue(np.allclose(frame["pred_score_path_5pct_20d"], y_pred[:, 2] - y_pred[:, 0]))
        self.assertEqual(_prediction_column(frame), "pred_score_path_5pct_20d")
        self.assertEqual(frame["pred_class_path_5pct_20d"].tolist()[:3], [2, 1, 0])
        self.assertEqual(metrics[0]["positive_class"], 2)
        self.assertEqual(metrics[0]["row_count"], 10)

    def test_path_5pct_20d_rejects_unsupported_classifier_request(self) -> None:
        from scripts.train_v1_supervised_baselines import _resolve_model_names

        args = argparse.Namespace(
            classification_models="logistic_regression",
            models="",
            eval_mode="walk_forward",
            classification_event_type=PATH_5PCT_20D_EVENT_TYPE,
        )

        with self.assertRaises(SystemExit) as raised:
            _resolve_model_names(args, "classification")
        self.assertIn("path_5pct_20d supports only", str(raised.exception))

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

    def test_torch_multiclass_classifiers_smoke(self) -> None:
        x = pd.DataFrame(
            {
                "a": np.linspace(0.0, 1.0, 12),
                "b": np.tile([0.0, 1.0, 2.0], 4),
                "c": np.tile([2.0, 1.0, 0.0], 4),
            }
        )
        y = pd.DataFrame({PATH_5PCT_20D_EVENT_TYPE: np.tile([0, 1, 2], 4)})
        mlp = make_model(
            "torch_mlp_classifier",
            task_type="classification",
            model_kwargs={
                "num_classes": 3,
                "max_epochs": 2,
                "patience": 2,
                "batch_size": 6,
                "hidden_units": 8,
            },
        )
        mlp.fit(x, y)
        mlp_pred = mlp.predict(x)
        self.assertEqual(mlp_pred.shape, (len(x), 3))
        self.assertTrue(np.allclose(mlp_pred.sum(axis=1), 1.0, atol=1e-5))

        stock_features, context_features = _stock_and_context_frames(ticker_count=3, days=28)
        store = build_sequence_feature_store(
            stock_features,
            "stock_only_sequence",
            context_features=context_features,
            benchmark_ticker="SPY",
        )
        rows = []
        for ticker in ["T00", "T01", "T02"]:
            ticker_dates = stock_features[stock_features["ticker"] == ticker]["date"].astype(str).tolist()
            for window_row_count in range(10, 14):
                rows.append(
                    {
                        "ticker": ticker,
                        "anchor_date": ticker_dates[window_row_count - 1],
                        "window_row_count": window_row_count,
                        "gics_sector": "Information Technology",
                        "gics_sub_industry": "Software",
                    }
                )
        metadata = pd.DataFrame(rows)
        seq_y = pd.DataFrame({PATH_5PCT_20D_EVENT_TYPE: np.tile([0, 1, 2], 4)})
        vocabularies = build_category_vocabularies(metadata, columns=STATIC_CATEGORICAL_COLUMNS)
        seq_x = {
            "store": store,
            "metadata": metadata,
            "static_categorical": encode_static_categories(metadata, vocabularies, columns=STATIC_CATEGORICAL_COLUMNS),
        }
        seq = make_model(
            "torch_seq_static_classifier",
            task_type="classification",
            window_length=10,
            model_kwargs={
                "num_classes": 3,
                "max_epochs": 1,
                "patience": 1,
                "batch_size": 6,
                "hidden_dim": 16,
            },
        )
        seq.fit(seq_x, seq_y, val_x=seq_x, val_y=seq_y)
        seq_pred = seq.predict(seq_x)
        self.assertEqual(seq_pred.shape, (len(metadata), 3))
        self.assertTrue(np.allclose(seq_pred.sum(axis=1), 1.0, atol=1e-5))

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

    def test_research_universe_filter_removes_small_illiquid_or_falling_windows(self) -> None:
        stock_features, context_features = _stock_and_context_frames(ticker_count=3, days=300)
        stock_features["exchange"] = "NASDAQ"
        stock_features["type"] = "Common Stock"
        stock_features.loc[stock_features["ticker"] == "T00", "dollar_volume"] = 20_000_000.0
        stock_features.loc[stock_features["ticker"] == "T01", "dollar_volume"] = 2_000_000.0
        stock_features.loc[stock_features["ticker"] == "T02", "close"] = np.linspace(90.0, 30.0, len(stock_features.loc[stock_features["ticker"] == "T02"]))
        stock_features.loc[stock_features["ticker"] == "T02", "open"] = stock_features.loc[stock_features["ticker"] == "T02", "close"]
        stock_features.loc[stock_features["ticker"] == "T02", "high"] = stock_features.loc[stock_features["ticker"] == "T02", "close"] * 1.01
        stock_features.loc[stock_features["ticker"] == "T02", "low"] = stock_features.loc[stock_features["ticker"] == "T02", "close"] * 0.99
        stock_features.loc[stock_features["ticker"] == "T02", "dollar_volume"] = 20_000_000.0
        stock_features = stock_features.sort_values(["ticker", "date"]).reset_index(drop=True)

        dataset = build_v1_dataset(
            stock_features,
            context_features,
            horizons=(1, 5),
            window_length=20,
            benchmark_ticker="SPY",
            research_config=ConservativeResearchUniverseConfig(),
        )

        self.assertEqual(set(dataset.metadata["ticker"].unique()), {"T00"})
        self.assertIn("research_universe_ok", dataset.metadata.columns)
        self.assertTrue(dataset.metadata["research_universe_ok"].all())

    def test_materialized_panel_applies_research_universe_before_cache_build(self) -> None:
        stock_features, context_features = _stock_and_context_frames(ticker_count=3, days=300)
        stock_features["exchange"] = "NASDAQ"
        stock_features["type"] = "Common Stock"
        stock_features.loc[stock_features["ticker"] == "T00", "dollar_volume"] = 20_000_000.0
        stock_features.loc[stock_features["ticker"] == "T01", "dollar_volume"] = 2_000_000.0
        falling_mask = stock_features["ticker"] == "T02"
        stock_features.loc[falling_mask, "close"] = np.linspace(90.0, 30.0, int(falling_mask.sum()))
        stock_features.loc[falling_mask, "open"] = stock_features.loc[falling_mask, "close"]
        stock_features.loc[falling_mask, "high"] = stock_features.loc[falling_mask, "close"] * 1.01
        stock_features.loc[falling_mask, "low"] = stock_features.loc[falling_mask, "close"] * 0.99
        stock_features.loc[falling_mask, "dollar_volume"] = 20_000_000.0
        stock_features = stock_features.sort_values(["ticker", "date"]).reset_index(drop=True)
        context_features = context_features.sort_values(["ticker", "date"]).reset_index(drop=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            dataset_root = temp_root / "dataset"
            output_root = temp_root / "panel"
            _write_dataset_root(dataset_root, stock_features, context_features)
            script = REPO_ROOT / "scripts" / "materialize_v1_training_panel.py"
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--source-dataset-root",
                    str(dataset_root),
                    "--output-dataset-root",
                    str(output_root),
                    "--force",
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            filtered = pd.read_csv(output_root / "processed" / "daily_features.csv")
            manifest = json.loads((output_root / "processed" / "materialized_panel_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(set(filtered["ticker"].unique()), {"T00"})
        self.assertEqual(manifest["research_universe"]["min_price"], 10.0)
        self.assertGreater(manifest["research_universe_removed_rows"], 0)

    def test_preferred_stock_feature_path_uses_newer_processed_file_when_normalized_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            processed_dir = root / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            normalized_path = processed_dir / "daily_features_normalized.csv"
            processed_path = processed_dir / "daily_features.csv"
            normalized_path.write_text("ticker,date,close\nAAA,2024-01-01,10\n", encoding="utf-8")
            processed_path.write_text("ticker,date,close\nAAA,2024-01-02,11\n", encoding="utf-8")
            old_time = normalized_path.stat().st_mtime - 10
            new_time = processed_path.stat().st_mtime + 10
            os.utime(normalized_path, (old_time, old_time))
            os.utime(processed_path, (new_time, new_time))

            chosen = preferred_stock_feature_path(root)

        self.assertEqual(chosen.name, "daily_features.csv")

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

    def test_episode_cache_materializes_lazy_tabular_and_sequence_inputs(self) -> None:
        stock_features, context_features = _stock_and_context_frames(ticker_count=5, days=45)
        stock_features = stock_features.sort_values(["ticker", "date"]).reset_index(drop=True)
        context_features = context_features.sort_values(["ticker", "date"]).reset_index(drop=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            cache_dir = Path(tmpdir) / "episode_cache"
            _write_dataset_root(root, stock_features, context_features)
            manifest = build_episode_cache(
                dataset_root=root,
                cache_dir=cache_dir,
                feature_sets=[
                    "stock_relative_market_sector_fundamentals_sentiment",
                    "stock_relative_market_sector_sentiment_sequence",
                ],
                horizons=(5,),
                window_length=10,
                benchmark_ticker="SPY",
                max_episodes=100,
                classification_horizon=5,
                classification_threshold=0.01,
                eligibility_config=None,
                force=True,
                progress_every=0,
            )
            cached = load_cached_v1_dataset(cache_dir)
            stores = load_cached_sequence_stores(cache_dir)

            self.assertEqual(manifest["episode_count"], len(cached.metadata))
            self.assertIn("stock_relative_market_sector_fundamentals_sentiment", cached.feature_sets)
            self.assertIn("stock_relative_market_sector_sentiment_sequence", stores)
            rows = cached.metadata["anchor_date"].notna()
            view = cached.feature_sets["stock_relative_market_sector_fundamentals_sentiment"].view(rows)
            self.assertEqual(view.shape[0], len(cached.metadata))
            self.assertEqual(identifier_model_input_columns(view.columns), [])
            store = stores["stock_relative_market_sector_sentiment_sequence"]
            first = cached.metadata.iloc[0]
            window = store.get_window(first["ticker"], int(first["window_row_count"]) - 1, 10)
            self.assertEqual(window.shape, (10, len(store.feature_columns)))

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
                    "--disable-conservative-research-universe",
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
                    "--disable-conservative-research-universe",
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

    @unittest.skipUnless(importlib.util.find_spec("xgboost") is not None, "xgboost is not installed")
    def test_xgboost_multiclass_classifier_smoke(self) -> None:
        x = pd.DataFrame(
            {
                "a": np.linspace(0.0, 1.0, 12),
                "b": np.tile([0.0, 1.0, 2.0], 4),
                "c": np.tile([2.0, 1.0, 0.0], 4),
            }
        )
        y = pd.DataFrame({PATH_5PCT_20D_EVENT_TYPE: np.tile([0, 1, 2], 4)})
        model = make_model(
            "xgboost_classifier",
            task_type="classification",
            model_kwargs={
                "num_classes": 3,
                "n_estimators": 5,
                "patience": 2,
                "prefer_gpu": False,
            },
        )
        model.fit(x, y)
        pred = model.predict(x)

        self.assertEqual(pred.shape, (len(x), 3))
        self.assertTrue(np.allclose(pred.sum(axis=1), 1.0, atol=1e-5))

    @unittest.skipUnless(importlib.util.find_spec("xgboost") is not None, "xgboost is not installed")
    def test_xgboost_multiclass_classifier_lazy_smoke(self) -> None:
        class LazyFrame:
            def __init__(self, values: np.ndarray) -> None:
                self.values = values.astype(np.float32)
                self.shape = self.values.shape

            def __len__(self) -> int:
                return self.values.shape[0]

            def iter_numpy_batches(self, *, batch_size: int, shuffle: bool = False, random_state: int = 0):
                rows = np.arange(len(self), dtype=np.int64)
                if shuffle:
                    rng = np.random.default_rng(random_state)
                    rng.shuffle(rows)
                for start in range(0, len(rows), batch_size):
                    local = rows[start : start + batch_size]
                    yield local, self.values[local].copy()

        values = np.column_stack(
            [
                np.linspace(0.0, 1.0, 18),
                np.tile([0.0, 1.0, 2.0], 6),
                np.tile([2.0, 1.0, 0.0], 6),
            ]
        )
        x = LazyFrame(values)
        y = pd.DataFrame({PATH_5PCT_20D_EVENT_TYPE: np.tile([0, 1, 2], 6)})
        model = make_model(
            "xgboost_classifier",
            task_type="classification",
            model_kwargs={
                "num_classes": 3,
                "n_estimators": 3,
                "patience": 2,
                "prefer_gpu": False,
            },
        )
        model.fit(x, y)
        pred = model.predict(x)

        self.assertEqual(pred.shape, (len(x), 3))
        self.assertTrue(np.allclose(pred.sum(axis=1), 1.0, atol=1e-5))

    @unittest.skipUnless(importlib.util.find_spec("xgboost") is not None, "xgboost is not installed")
    def test_xgboost_lazy_classifier_predict_uses_best_iteration(self) -> None:
        import xgboost as xgb

        class LazyFrame:
            def __init__(self, values: np.ndarray) -> None:
                self.values = values.astype(np.float32)
                self.shape = self.values.shape

            def __len__(self) -> int:
                return self.values.shape[0]

            def iter_numpy_batches(self, *, batch_size: int, shuffle: bool = False, random_state: int = 0):
                rows = np.arange(len(self), dtype=np.int64)
                if shuffle:
                    rng = np.random.default_rng(random_state)
                    rng.shuffle(rows)
                for start in range(0, len(rows), batch_size):
                    local = rows[start : start + batch_size]
                    yield local, self.values[local].copy()

        values = np.column_stack(
            [
                np.linspace(-2.0, 2.0, 90),
                np.sin(np.linspace(0.0, 6.0, 90)),
                np.cos(np.linspace(0.0, 6.0, 90)),
            ]
        )
        x = LazyFrame(values)
        y = pd.DataFrame({PATH_5PCT_20D_EVENT_TYPE: np.tile([0, 1, 2], 30)})
        model = make_model(
            "xgboost_classifier",
            task_type="classification",
            model_kwargs={
                "num_classes": 3,
                "n_estimators": 8,
                "patience": 2,
                "prefer_gpu": False,
            },
        )
        model.fit(x, y)
        model.best_iteration_ = 1

        pred = model.predict(x)
        dmatrix = xgb.DMatrix(values.astype(np.float32))
        expected = model._format_prediction(model.model_.predict(dmatrix, iteration_range=(0, 1)))
        default = model._format_prediction(model.model_.predict(dmatrix))

        self.assertTrue(np.allclose(pred, expected, atol=1e-7))
        self.assertFalse(np.allclose(default, expected, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
