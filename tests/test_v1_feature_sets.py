import unittest

import pandas as pd

from src.data.eodhd_enrichment import FUNDAMENTAL_FEATURE_COLUMNS, SENTIMENT_FEATURE_COLUMNS
from src.data.v1_dataset import (
    CONTEXT_FEATURES,
    FEATURE_SET_NAMES,
    LEAN_ABSOLUTE_STOCK_FEATURES,
    LEAN_FULL_PANEL_RELATIVE_FEATURES,
    LEAN_SECTOR_RELATIVE_FEATURES,
    NORMALIZED_LEAN_TABULAR_FEATURE_SET_NAMES,
    SEQUENCE_FEATURE_SET_NAMES,
    select_augmented_stock_feature_columns,
    sequence_feature_config,
    validate_model_feature_columns,
)
from src.data.v1_episode_cache import _sequence_columns, _tabular_columns


class LeanNormalizedFeatureSetTests(unittest.TestCase):
    def _stock_frame(self) -> pd.DataFrame:
        columns = [
            *LEAN_ABSOLUTE_STOCK_FEATURES,
            *LEAN_FULL_PANEL_RELATIVE_FEATURES,
            *LEAN_SECTOR_RELATIVE_FEATURES,
            *SENTIMENT_FEATURE_COLUMNS[:2],
            *FUNDAMENTAL_FEATURE_COLUMNS[:2],
            "log1p_volume",
            "log1p_rolling_avg_volume_20d",
            "log1p_rolling_avg_dollar_volume_60d",
        ]
        return pd.DataFrame({column: [1.0] for column in columns})

    def test_lean_normalized_stock_columns_prefer_relative_and_scale_free_inputs(self) -> None:
        cols = select_augmented_stock_feature_columns(
            self._stock_frame(),
            include_relative=True,
            normalized_lean=True,
            include_sentiment=True,
            include_fundamentals=True,
        )

        self.assertIn("log_return_1d", cols)
        self.assertIn("log1p_dollar_volume__cs_z", cols)
        self.assertIn("log1p_dollar_volume__sector_cs_pct", cols)
        self.assertIn(SENTIMENT_FEATURE_COLUMNS[0], cols)
        self.assertIn(FUNDAMENTAL_FEATURE_COLUMNS[0], cols)
        self.assertNotIn("log1p_volume", cols)
        self.assertNotIn("log1p_rolling_avg_volume_20d", cols)
        self.assertNotIn("log1p_rolling_avg_dollar_volume_60d", cols)

    def test_lean_family_exposes_component_combinations(self) -> None:
        expected_tabular = {
            "stock_normalized_lean",
            "stock_normalized_lean_market",
            "stock_normalized_lean_sector",
            "stock_normalized_lean_market_sector",
            "stock_normalized_lean_sentiment",
            "stock_normalized_lean_market_sentiment",
            "stock_normalized_lean_sector_sentiment",
            "stock_normalized_lean_market_sector_sentiment",
            "stock_normalized_lean_fundamentals",
            "stock_normalized_lean_market_fundamentals",
            "stock_normalized_lean_sector_fundamentals",
            "stock_normalized_lean_market_sector_fundamentals",
            "stock_normalized_lean_fundamentals_sentiment",
            "stock_normalized_lean_market_fundamentals_sentiment",
            "stock_normalized_lean_sector_fundamentals_sentiment",
            "stock_normalized_lean_market_sector_fundamentals_sentiment",
        }
        self.assertTrue(expected_tabular.issubset(set(NORMALIZED_LEAN_TABULAR_FEATURE_SET_NAMES)))
        self.assertTrue(expected_tabular.issubset(set(FEATURE_SET_NAMES)))
        self.assertIn("stock_normalized_lean_market_sequence", SEQUENCE_FEATURE_SET_NAMES)
        self.assertIn("stock_normalized_lean_sentiment_sequence", SEQUENCE_FEATURE_SET_NAMES)

    def test_cache_tabular_columns_support_market_only_lean_feature_set(self) -> None:
        header = list(self._stock_frame().columns)
        cols = _tabular_columns(
            header=header,
            context_columns=CONTEXT_FEATURES,
            feature_set="stock_normalized_lean_market",
        )

        self.assertIn("stock_log_return_1d__last", cols)
        self.assertIn("market_context_log_return_1d__last", cols)
        self.assertNotIn("sector_context_log_return_1d__last", cols)
        self.assertNotIn("stock_log1p_volume__last", cols)

    def test_cache_tabular_columns_use_compact_context_for_lean_feature_set(self) -> None:
        header = list(self._stock_frame().columns)
        cols = _tabular_columns(
            header=header,
            context_columns=CONTEXT_FEATURES,
            feature_set="stock_normalized_lean_market_sector_fundamentals_sentiment",
        )

        self.assertIn("stock_log_return_1d__last", cols)
        self.assertIn("stock_log1p_dollar_volume__cs_z__last", cols)
        self.assertIn("market_context_log_return_1d__last", cols)
        self.assertIn("sector_context_log_return_1d__last", cols)
        self.assertNotIn("stock_log1p_volume__last", cols)
        self.assertNotIn("market_context_momentum_20d__last", cols)
        validate_model_feature_columns(cols, feature_set="stock_normalized_lean_market_sector_fundamentals_sentiment")

    def test_cache_sequence_columns_use_lean_profile_without_fundamentals(self) -> None:
        config = sequence_feature_config("stock_normalized_lean_market_sector_sentiment_sequence")
        self.assertTrue(config.normalized_lean)
        self.assertTrue(config.include_relative)
        self.assertTrue(config.include_market_context)
        self.assertTrue(config.include_sector_context)

        header = list(self._stock_frame().columns)
        cols = _sequence_columns(
            header=header,
            context_columns=CONTEXT_FEATURES,
            feature_set="stock_normalized_lean_market_sector_sentiment_sequence",
        )

        self.assertIn("log_return_1d", cols)
        self.assertIn("log1p_dollar_volume__cs_z", cols)
        self.assertIn("market_context_log_return_1d", cols)
        self.assertIn("sector_context_log_return_1d", cols)
        self.assertNotIn("log1p_volume", cols)
        self.assertNotIn("market_context_momentum_20d", cols)
        self.assertNotIn(FUNDAMENTAL_FEATURE_COLUMNS[0], cols)
        validate_model_feature_columns(cols, feature_set="stock_normalized_lean_market_sector_sentiment_sequence")


if __name__ == "__main__":
    unittest.main()
