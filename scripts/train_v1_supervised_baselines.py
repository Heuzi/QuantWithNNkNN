from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import time
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.v1_dataset import (  # noqa: E402
    DEFAULT_BENCHMARK_TICKER,
    DEFAULT_CLASSIFICATION_EVENT_TYPE,
    DEFAULT_CLASSIFICATION_HORIZON,
    DEFAULT_CLASSIFICATION_THRESHOLD,
    DEFAULT_WINDOW_LENGTH,
    FEATURE_SET_NAMES,
    PATH_5PCT_20D_EVENT_TYPE,
    SEQUENCE_FEATURE_SET_NAMES,
    STATIC_CATEGORICAL_COLUMNS,
    V1Dataset,
    build_category_vocabularies,
    build_sequence_feature_store,
    build_v1_dataset,
    build_walk_forward_folds,
    classification_class_labels,
    classification_label_kind,
    classification_positive_class,
    chronological_split,
    encode_static_categories,
    load_daily_features,
    load_market_context_features,
    parse_horizons,
    rows_for_dates,
    save_dataset_manifest,
    sequence_feature_config,
    split_ranges,
    target_column,
)
from src.data.v1_episode_cache import (  # noqa: E402
    CachedV1Dataset,
    load_cached_sequence_stores,
    load_cached_v1_dataset,
)
from src.data.episode_eligibility import (  # noqa: E402
    EpisodeEligibilityConfig,
    parse_allowed_exchanges,
)
from src.data.research_universe import ConservativeResearchUniverseConfig  # noqa: E402
from src.models.v1_baselines import (  # noqa: E402
    SEQUENCE_STATIC_CLASSIFICATION_MODEL_NAME,
    build_classification_leaderboard,
    build_leaderboard,
    build_metric_summary,
    default_model_names,
    evaluate_classification_predictions,
    evaluate_predictions,
    is_sequence_static_model,
    load_model_bundle,
    make_model,
    prediction_frame,
    save_model_bundle,
    write_json,
)


PATH_5PCT_20D_SUPPORTED_CLASSIFIERS = (
    "xgboost_classifier",
    "torch_mlp_classifier",
    SEQUENCE_STATIC_CLASSIFICATION_MODEL_NAME,
)


def _progress_bar(completed: int, total: int, *, width: int = 24) -> str:
    safe_total = max(int(total), 1)
    safe_completed = min(max(int(completed), 0), safe_total)
    filled = int(round((safe_completed / safe_total) * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {safe_completed}/{safe_total}"


def _emit_combo_progress(prefix: str, completed: int, total: int, *, detail: str = "") -> None:
    message = f"{prefix} {_progress_bar(completed, total)}"
    if detail:
        message += f" {detail}"
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train V1 supervised baselines for regression and/or event classification."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/eodhd_us_equities_30y",
        help="Dataset folder containing processed daily features and market context features.",
    )
    parser.add_argument("--output-root", default="artifacts/v1_baselines", help="Artifact output root.")
    parser.add_argument("--run-name", default="", help="Optional run folder name.")
    parser.add_argument("--horizons", default="1,5,10,20", help="Comma-separated regression target horizons.")
    parser.add_argument("--window-length", type=int, default=DEFAULT_WINDOW_LENGTH)
    parser.add_argument("--benchmark-ticker", default=DEFAULT_BENCHMARK_TICKER)
    parser.add_argument("--feature-sets", default=",".join(FEATURE_SET_NAMES))
    parser.add_argument("--models", default="", help="Optional comma-separated regression model list.")
    parser.add_argument(
        "--classification-models",
        default="",
        help="Optional comma-separated classification model list.",
    )
    parser.add_argument("--task-type", choices=("regression", "classification", "both"), default="both")
    parser.add_argument("--eval-mode", choices=("walk_forward", "holdout"), default="walk_forward")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--walk-forward-min-train-dates", type=int, default=252)
    parser.add_argument("--walk-forward-val-block-size", type=int, default=21)
    parser.add_argument("--walk-forward-oos-block-size", type=int, default=21)
    parser.add_argument(
        "--walk-forward-max-folds",
        type=int,
        default=0,
        help="Optional cap on walk-forward folds. Keeps the most recent folds.",
    )
    parser.add_argument(
        "--walk-forward-purge-gap",
        type=int,
        default=0,
        help="Trading-date gap between validation and OOS blocks. Defaults to max(horizons).",
    )
    parser.add_argument(
        "--final-stop-block-size",
        type=int,
        default=21,
        help="Most recent resolved trading dates used as the early-stop tail for final deploy fits.",
    )
    parser.add_argument("--compare-against-run", default="", help="Optional legacy regression run directory.")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Optional cap for smoke runs. Keeps the most recent N eligible episodes.",
    )
    parser.add_argument("--classification-horizon", type=int, default=DEFAULT_CLASSIFICATION_HORIZON)
    parser.add_argument("--classification-threshold", type=float, default=DEFAULT_CLASSIFICATION_THRESHOLD)
    parser.add_argument(
        "--classification-event-type",
        default=DEFAULT_CLASSIFICATION_EVENT_TYPE,
        help=(
            "Classification label semantics. Default preserves the current anytime "
            "pathwise target; sustained_pathwise_outperform and path_5pct_20d are opt-in."
        ),
    )
    parser.add_argument("--xgboost-n-estimators", type=int, default=0, help="Override XGBoost max boosting rounds.")
    parser.add_argument("--xgboost-patience", type=int, default=0, help="Override XGBoost early-stop patience.")
    parser.add_argument("--torch-max-epochs", type=int, default=0, help="Override max epochs for torch models.")
    parser.add_argument("--torch-patience", type=int, default=0, help="Override early-stop patience for torch models.")
    parser.add_argument("--torch-batch-size", type=int, default=0, help="Override batch size for torch models.")
    parser.add_argument("--torch-hidden-units", type=int, default=0, help="Override hidden units for torch MLP models.")
    parser.add_argument("--torch-hidden-dim", type=int, default=0, help="Override hidden dimension for sequence/static torch models.")
    parser.add_argument(
        "--episode-cache-dir",
        default="",
        help=(
            "Optional materialized V1 episode cache. When set, training loads cached "
            "metadata/targets plus memmapped feature arrays instead of rebuilding "
            "feature frames from daily CSVs."
        ),
    )
    parser.add_argument(
        "--disable-episode-eligibility-filter",
        action="store_true",
        help="Disable as-of common-stock/history/liquidity/price/exchange episode filtering.",
    )
    parser.add_argument("--eligibility-min-history-days", type=int, default=0)
    parser.add_argument("--eligibility-valid-ohlcv-lookback", type=int, default=60)
    parser.add_argument("--eligibility-min-valid-ohlcv-days", type=int, default=55)
    parser.add_argument("--eligibility-dollar-volume-lookback", type=int, default=60)
    parser.add_argument("--eligibility-min-avg-dollar-volume", type=float, default=100_000.0)
    parser.add_argument("--eligibility-min-price", type=float, default=1.0)
    parser.add_argument(
        "--eligibility-allowed-exchanges",
        default="NYSE,NASDAQ,AMEX,BATS",
        help="Comma-separated exchange allowlist. AMEX also matches EODHD NYSE MKT / NYSE American.",
    )
    parser.add_argument(
        "--disable-conservative-research-universe",
        action="store_true",
        help="Disable the shared strategy-universe filter for train/test/latest prediction windows.",
    )
    parser.add_argument("--research-universe-name", default="conservative")
    parser.add_argument("--research-common-stocks-only", action="store_true", default=True)
    parser.add_argument("--research-allowed-exchanges", default="NYSE,NASDAQ,AMEX")
    parser.add_argument("--research-min-price", type=float, default=10.0)
    parser.add_argument("--research-min-history-days", type=int, default=252)
    parser.add_argument("--research-min-median-dollar-volume-20d", type=float, default=10_000_000.0)
    parser.add_argument("--research-min-median-dollar-volume-60d", type=float, default=10_000_000.0)
    parser.add_argument("--research-max-zero-volume-day-ratio-60d", type=float, default=0.02)
    parser.add_argument("--research-min-current-dollar-volume-vs-median-20d", type=float, default=0.20)
    parser.add_argument("--research-liquidity-short-lookback-days", type=int, default=20)
    parser.add_argument("--research-liquidity-long-lookback-days", type=int, default=60)
    parser.add_argument("--research-trend-lookback-days", type=int, default=252)
    parser.add_argument("--research-return-6m-lookback-days", type=int, default=126)
    parser.add_argument("--research-sma-short-lookback-days", type=int, default=50)
    parser.add_argument("--research-sma-long-lookback-days", type=int, default=200)
    parser.add_argument("--research-min-return-6m", type=float, default=-0.15)
    parser.add_argument("--research-max-drawdown-from-252d-high-pct", type=float, default=35.0)
    parser.add_argument("--research-disable-close-above-sma200", action="store_true")
    parser.add_argument("--research-disable-sma50-above-sma200", action="store_true")
    parser.add_argument("--research-disable-spike-filter", action="store_true")
    parser.add_argument("--research-spike-lookback-days", type=int, default=60)
    parser.add_argument("--research-max-abs-return-1d-60d-pct", type=float, default=25.0)
    parser.add_argument("--research-max-true-range-60d-pct", type=float, default=25.0)
    return parser.parse_args()


def _task_types(args: argparse.Namespace) -> list[str]:
    if args.task_type == "both":
        return ["regression", "classification"]
    return [args.task_type]


def _resolve_model_names(args: argparse.Namespace, task_type: str) -> list[str]:
    raw = args.classification_models if task_type == "classification" else args.models
    if raw.strip():
        names = [item.strip() for item in raw.split(",") if item.strip()]
    elif task_type == "classification" and args.classification_event_type == PATH_5PCT_20D_EVENT_TYPE:
        names = list(PATH_5PCT_20D_SUPPORTED_CLASSIFIERS)
    else:
        names = default_model_names(args.eval_mode, task_type=task_type)
    if task_type == "classification" and args.classification_event_type == PATH_5PCT_20D_EVENT_TYPE:
        unsupported = [name for name in names if name not in PATH_5PCT_20D_SUPPORTED_CLASSIFIERS]
        if unsupported:
            supported = ", ".join(PATH_5PCT_20D_SUPPORTED_CLASSIFIERS)
            raise SystemExit(
                "path_5pct_20d supports only these classification models: "
                f"{supported}. Unsupported requested model(s): {unsupported}"
            )
    return names


def _supported_feature_set(model_name: str, feature_set: str) -> bool:
    if is_sequence_static_model(model_name):
        return feature_set in SEQUENCE_FEATURE_SET_NAMES
    return feature_set in FEATURE_SET_NAMES


def _build_combo_iterable(feature_sets: list[str], model_names: list[str]) -> list[tuple[str, str]]:
    combos: list[tuple[str, str]] = []
    for feature_set in feature_sets:
        for model_name in model_names:
            if _supported_feature_set(model_name, feature_set):
                combos.append((feature_set, model_name))
    if not combos:
        raise SystemExit("No valid (feature_set, model_name) combinations to train.")
    return combos


def _episode_eligibility_config(args: argparse.Namespace) -> EpisodeEligibilityConfig | None:
    if args.disable_episode_eligibility_filter:
        return None
    return EpisodeEligibilityConfig(
        min_history_days=args.eligibility_min_history_days or args.window_length,
        valid_ohlcv_lookback=args.eligibility_valid_ohlcv_lookback,
        min_valid_ohlcv_days=args.eligibility_min_valid_ohlcv_days,
        dollar_volume_lookback=args.eligibility_dollar_volume_lookback,
        min_avg_dollar_volume=args.eligibility_min_avg_dollar_volume,
        min_price=args.eligibility_min_price,
        allowed_exchanges=parse_allowed_exchanges(args.eligibility_allowed_exchanges),
    )


def _research_universe_config(args: argparse.Namespace) -> ConservativeResearchUniverseConfig | None:
    if args.disable_conservative_research_universe:
        return None
    return ConservativeResearchUniverseConfig(
        name=args.research_universe_name,
        common_stocks_only=bool(args.research_common_stocks_only),
        allowed_exchanges=parse_allowed_exchanges(args.research_allowed_exchanges),
        min_price=args.research_min_price,
        min_history_days=args.research_min_history_days,
        liquidity_short_lookback=args.research_liquidity_short_lookback_days,
        liquidity_long_lookback=args.research_liquidity_long_lookback_days,
        min_median_dollar_volume_20d=args.research_min_median_dollar_volume_20d,
        min_median_dollar_volume_60d=args.research_min_median_dollar_volume_60d,
        max_zero_volume_day_ratio_60d=args.research_max_zero_volume_day_ratio_60d,
        min_current_dollar_volume_vs_median_20d=args.research_min_current_dollar_volume_vs_median_20d,
        trend_lookback_days=args.research_trend_lookback_days,
        return_6m_lookback_days=args.research_return_6m_lookback_days,
        sma_short_lookback_days=args.research_sma_short_lookback_days,
        sma_long_lookback_days=args.research_sma_long_lookback_days,
        min_return_6m=args.research_min_return_6m,
        max_drawdown_from_252d_high=args.research_max_drawdown_from_252d_high_pct / 100.0,
        require_close_above_sma200=not bool(args.research_disable_close_above_sma200),
        require_sma50_above_sma200=not bool(args.research_disable_sma50_above_sma200),
        spike_filter_enabled=not bool(args.research_disable_spike_filter),
        spike_lookback_days=args.research_spike_lookback_days,
        max_abs_return_1d_60d=args.research_max_abs_return_1d_60d_pct / 100.0,
        max_true_range_pct_60d=args.research_max_true_range_60d_pct / 100.0,
    )


def _torch_model_kwargs(args: argparse.Namespace, model_name: str) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if not model_name.startswith("torch_"):
        return kwargs
    if args.torch_max_epochs and args.torch_max_epochs > 0:
        kwargs["max_epochs"] = args.torch_max_epochs
    if args.torch_patience and args.torch_patience > 0:
        kwargs["patience"] = args.torch_patience
    if args.torch_batch_size and args.torch_batch_size > 0:
        kwargs["batch_size"] = args.torch_batch_size
    if is_sequence_static_model(model_name):
        if args.torch_hidden_dim and args.torch_hidden_dim > 0:
            kwargs["hidden_dim"] = args.torch_hidden_dim
    elif args.torch_hidden_units and args.torch_hidden_units > 0:
        kwargs["hidden_units"] = args.torch_hidden_units
    return kwargs


def _model_kwargs(args: argparse.Namespace, model_name: str, task_type: str) -> dict[str, object]:
    kwargs = _torch_model_kwargs(args, model_name)
    if model_name in {"xgboost", "xgboost_classifier"}:
        if args.xgboost_n_estimators and args.xgboost_n_estimators > 0:
            kwargs["n_estimators"] = args.xgboost_n_estimators
        if args.xgboost_patience and args.xgboost_patience > 0:
            kwargs["patience"] = args.xgboost_patience
    if task_type == "classification" and args.classification_event_type == PATH_5PCT_20D_EVENT_TYPE:
        kwargs["num_classes"] = len(classification_class_labels(args.classification_event_type))
    return kwargs


def _task_target_columns(dataset: V1Dataset, task_type: str) -> list[str]:
    if task_type == "classification":
        return dataset.classification_target_columns
    return dataset.target_columns


def _prepare_flat_inputs(
    dataset: V1Dataset,
    feature_set: str,
    rows: pd.Series,
    *,
    task_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = dataset.metadata.loc[rows].reset_index(drop=True)
    feature_frame = dataset.feature_sets[feature_set]
    if hasattr(feature_frame, "view"):
        x = feature_frame.view(rows)
    else:
        x = feature_frame.loc[rows, dataset.feature_columns[feature_set]].reset_index(drop=True)
    y = dataset.targets.loc[rows, _task_target_columns(dataset, task_type)].reset_index(drop=True)
    return meta, x, y


def _prepare_sequence_inputs(
    store,
    metadata: pd.DataFrame,
    vocabularies: dict[str, dict[str, int]],
) -> dict[str, object]:
    metadata = metadata.reset_index(drop=True)
    return {
        "store": store,
        "metadata": metadata,
        "static_categorical": encode_static_categories(
            metadata,
            vocabularies,
            columns=STATIC_CATEGORICAL_COLUMNS,
        ),
    }


def _prediction_targets_from_frame(frame: pd.DataFrame, target_columns: list[str], *, task_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if task_type == "classification":
        actual = pd.DataFrame({target: frame[f"actual_{target}"] for target in target_columns})
        if len(target_columns) == 1:
            target = target_columns[0]
            class_cols = [
                col
                for col in frame.columns
                if col.startswith(f"pred_prob_{target}_class_")
            ]
            class_cols = sorted(class_cols, key=lambda name: int(name.rsplit("_class_", 1)[1]))
            if class_cols:
                return actual, frame[class_cols].copy()
        pred = pd.DataFrame({target: frame[f"pred_prob_{target}"] for target in target_columns})
        return actual, pred
    actual = pd.DataFrame(
        {
            target: frame[target.replace("market_adjusted_return", "actual_market_adjusted_return")]
            for target in target_columns
        }
    )
    pred = pd.DataFrame(
        {
            target: frame[target.replace("market_adjusted_return", "pred_market_adjusted_return")]
            for target in target_columns
        }
    )
    return actual, pred


def _fold_summary(fold, metadata: pd.DataFrame) -> dict[str, object]:
    out = fold.to_dict()
    for split_name, dates in (("train", fold.train_dates), ("val", fold.val_dates), ("oos", fold.oos_dates)):
        rows = rows_for_dates(metadata, dates)
        out[f"{split_name}_row_count"] = int(rows.sum())
        out[f"{split_name}_date_count"] = len(dates)
    return out


def _build_model_metadata(
    *,
    task_type: str,
    model_name: str,
    feature_set: str,
    target_columns: list[str],
    horizons: tuple[int, ...],
    window_length: int,
    benchmark_ticker: str,
    eval_mode: str,
    split_summary: dict[str, dict[str, str | int | None]] | None = None,
    flat_feature_columns: list[str] | None = None,
    sequence_feature_columns: list[str] | None = None,
    sequence_components: dict[str, bool] | None = None,
    static_vocabularies: dict[str, dict[str, int]] | None = None,
    runtime: dict[str, object] | None = None,
    classification_threshold: float | None = None,
    classification_horizon: int | None = None,
    classification_event_type: str = DEFAULT_CLASSIFICATION_EVENT_TYPE,
) -> dict[str, object]:
    input_layout = "sequence_static" if is_sequence_static_model(model_name) else "tabular"
    metadata = {
        "task_type": task_type,
        "model_name": model_name,
        "feature_set": feature_set,
        "target_columns": target_columns,
        "horizons": list(horizons),
        "window_length": window_length,
        "benchmark_ticker": benchmark_ticker.upper(),
        "evaluation_mode": eval_mode,
        "input_layout": input_layout,
    }
    if task_type == "classification":
        label_kind = classification_label_kind(classification_event_type)
        class_labels = classification_class_labels(classification_event_type)
        positive_class = classification_positive_class(classification_event_type)
        metadata["decision_threshold"] = 0.5 if label_kind == "binary" else None
        if label_kind == "multiclass" and len(target_columns) == 1:
            target = target_columns[0]
            metadata["probability_column_names"] = [
                *(f"pred_prob_{target}_class_{class_label}" for class_label in class_labels),
                f"pred_prob_{target}",
            ]
            metadata["predicted_class_column_name"] = f"pred_class_{target}"
        else:
            metadata["probability_column_names"] = [f"pred_prob_{target}" for target in target_columns]
        metadata["classification_horizon_days"] = classification_horizon
        metadata["classification_threshold"] = classification_threshold
        metadata["classification_event_type"] = classification_event_type
        metadata["classification_label_kind"] = label_kind
        metadata["classification_class_labels"] = class_labels
        metadata["classification_positive_class"] = positive_class
    if split_summary is not None:
        metadata["split_summary"] = split_summary
    if flat_feature_columns is not None:
        metadata["feature_columns"] = flat_feature_columns
    if sequence_feature_columns is not None:
        metadata["sequence_feature_columns"] = sequence_feature_columns
        metadata["sequence_components"] = sequence_components or sequence_feature_config(feature_set).to_dict()
        metadata["static_categorical_columns"] = list(STATIC_CATEGORICAL_COLUMNS)
        metadata["static_vocabularies"] = static_vocabularies or {}
    if runtime:
        metadata["runtime"] = runtime
    return metadata


def _model_runtime_metadata(model: object) -> dict[str, object]:
    runtime: dict[str, object] = {}
    if hasattr(model, "device"):
        runtime["requested_device"] = str(getattr(model, "device"))
    if hasattr(model, "device_"):
        runtime["resolved_device"] = str(getattr(model, "device_"))
    if hasattr(model, "device_type_"):
        runtime["lightgbm_device_type"] = str(getattr(model, "device_type_"))
    if type(model).__name__ in {"XGBoostRegressor", "XGBoostClassifier"} and hasattr(model, "device_"):
        runtime["xgboost_device"] = str(getattr(model, "device_"))
    if hasattr(model, "prefer_gpu"):
        runtime["prefer_gpu"] = bool(getattr(model, "prefer_gpu"))
    if getattr(model, "gpu_fallback_error_", ""):
        runtime["gpu_fallback_error"] = str(getattr(model, "gpu_fallback_error_"))
    return runtime


def _runtime_environment() -> dict[str, object]:
    info: dict[str, object] = {"gpu_available": False}
    gpu_visible_from_env = False
    for env_name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        if env_name in os.environ:
            visible_devices = os.environ[env_name]
            info[env_name.lower()] = visible_devices
            if visible_devices.strip().lower() not in {"", "-1", "none", "void"}:
                gpu_visible_from_env = True
    info["gpu_visible_from_env"] = gpu_visible_from_env
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["gpu_available"] = bool(torch.cuda.is_available()) or gpu_visible_from_env
        if torch.cuda.is_available():
            info["cuda_device_count"] = int(torch.cuda.device_count())
            info["cuda_device_name"] = str(torch.cuda.get_device_name(0))
    except Exception:
        info["gpu_available"] = gpu_visible_from_env
    return info


def _realized_return_column(dataset: V1Dataset, classification_horizon: int) -> str | None:
    col = target_column(classification_horizon)
    return col if col in dataset.targets.columns else None


def _fit_final_deploy_model(
    *,
    dataset: V1Dataset,
    feature_set: str,
    model_name: str,
    task_type: str,
    all_dates: list[str],
    final_stop_block_size: int,
    sequence_store=None,
    window_length: int,
    model_kwargs: dict[str, object] | None = None,
) -> tuple[object, dict[str, dict[str, int]] | None]:
    final_val_dates = all_dates[-min(final_stop_block_size, max(1, len(all_dates) - 1)) :]
    final_train_dates = all_dates[: len(all_dates) - len(final_val_dates)]
    if not final_train_dates:
        final_train_dates = all_dates[:-1]
        final_val_dates = all_dates[-1:]
    train_rows = rows_for_dates(dataset.metadata, final_train_dates)
    val_rows = rows_for_dates(dataset.metadata, final_val_dates)
    full_rows = rows_for_dates(dataset.metadata, all_dates)
    train_meta = dataset.metadata.loc[train_rows].reset_index(drop=True)
    val_meta = dataset.metadata.loc[val_rows].reset_index(drop=True)
    full_meta = dataset.metadata.loc[full_rows].reset_index(drop=True)
    target_columns = _task_target_columns(dataset, task_type)
    y_train = dataset.targets.loc[train_rows, target_columns].reset_index(drop=True)
    y_val = dataset.targets.loc[val_rows, target_columns].reset_index(drop=True)
    y_full = dataset.targets.loc[full_rows, target_columns].reset_index(drop=True)
    final_model = make_model(
        model_name,
        window_length=window_length,
        task_type=task_type,
        model_kwargs=model_kwargs,
    )
    final_vocabularies: dict[str, dict[str, int]] | None = None
    if is_sequence_static_model(model_name):
        final_vocabularies = build_category_vocabularies(full_meta, columns=STATIC_CATEGORICAL_COLUMNS)
        x_train = _prepare_sequence_inputs(sequence_store, train_meta, final_vocabularies)
        x_val = _prepare_sequence_inputs(sequence_store, val_meta, final_vocabularies)
        x_full = _prepare_sequence_inputs(sequence_store, full_meta, final_vocabularies)
    else:
        _, x_train, _ = _prepare_flat_inputs(dataset, feature_set, train_rows, task_type=task_type)
        _, x_val, _ = _prepare_flat_inputs(dataset, feature_set, val_rows, task_type=task_type)
        _, x_full, _ = _prepare_flat_inputs(dataset, feature_set, full_rows, task_type=task_type)
    final_model.fit(x_train, y_train, val_x=x_val, val_y=y_val)
    if hasattr(final_model, "refit_full"):
        final_model.refit_full(x_full, y_full)
    else:
        final_model = make_model(
            model_name,
            window_length=window_length,
            task_type=task_type,
            model_kwargs=model_kwargs,
        )
        final_model.fit(x_full, y_full)
    return final_model, final_vocabularies


def _compare_against_legacy(
    *,
    legacy_run_dir: Path,
    new_leaderboard: pd.DataFrame,
    output_dir: Path,
) -> None:
    legacy_metrics_path = legacy_run_dir / "metrics.csv"
    if not legacy_metrics_path.exists():
        raise SystemExit(f"Missing legacy metrics file: {legacy_metrics_path}")
    legacy_metrics = pd.read_csv(legacy_metrics_path)
    legacy_summary = build_metric_summary(legacy_metrics, split_name="test")
    comparison = legacy_summary.merge(
        new_leaderboard,
        on=["model_name", "feature_set"],
        how="inner",
        suffixes=("_legacy", "_new"),
    )
    for metric in ("mean_rank_ic", "mean_top_bottom_spread", "mean_rmse", "mean_mae", "selection_score"):
        comparison[f"{metric}_delta"] = comparison[f"{metric}_new"] - comparison[f"{metric}_legacy"]
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    score_delta = comparison["selection_score_delta"]
    improved = int((score_delta > 0).sum())
    worsened = int((score_delta < 0).sum())
    ties = int((score_delta == 0).sum())
    summary = {
        "matched_combo_count": int(len(comparison)),
        "improved_count": improved,
        "worsened_count": worsened,
        "tie_count": ties,
        "median_selection_score_delta": float(score_delta.median()) if len(comparison) else None,
        "new_scheme_better": bool(len(comparison) and float(score_delta.median()) > 0 and improved > worsened),
        "legacy_run_dir": str(legacy_run_dir.resolve()),
    }
    write_json(output_dir / "comparison_summary.json", summary)


def _write_model_index(path: Path, *, generated_utc: str, eval_mode: str, records: list[dict[str, object]]) -> None:
    trained_at_utc = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    write_json(
        path,
        {
            "generated_utc": generated_utc,
            "trained_at_utc": trained_at_utc,
            "evaluation_mode": eval_mode,
            "models": records,
        },
    )


def _task_file_prefix(task_type: str) -> str:
    return "" if task_type == "regression" else "classification_"


def _write_task_outputs(
    *,
    task_type: str,
    eval_mode: str,
    metrics: pd.DataFrame,
    leaderboard: pd.DataFrame,
    predictions: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    prefix = _task_file_prefix(task_type)
    if task_type == "regression":
        metrics.to_csv(output_dir / "metrics.csv", index=False)
        leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
        if eval_mode == "walk_forward":
            leaderboard.to_csv(output_dir / "oos_leaderboard.csv", index=False)
        if predictions is not None:
            target_name = "oos_predictions.csv" if eval_mode == "walk_forward" else "val_test_predictions.csv"
            predictions.to_csv(output_dir / target_name, index=False)
    else:
        metrics.to_csv(output_dir / f"{prefix}metrics.csv", index=False)
        leaderboard.to_csv(output_dir / f"{prefix}leaderboard.csv", index=False)
        if eval_mode == "walk_forward":
            leaderboard.to_csv(output_dir / f"{prefix}oos_leaderboard.csv", index=False)
        if predictions is not None:
            target_name = f"{prefix}oos_predictions.csv" if eval_mode == "walk_forward" else f"{prefix}val_test_predictions.csv"
            predictions.to_csv(output_dir / target_name, index=False)


def _safe_artifact_component(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return text or "item"


def _walk_forward_checkpoint_paths(
    *,
    output_dir: Path,
    task_type: str,
    model_name: str,
    feature_set: str,
    fold_id: int,
) -> tuple[Path, Path]:
    checkpoint_dir = output_dir / "checkpoints" / "walk_forward" / task_type
    stem = "__".join(
        _safe_artifact_component(part)
        for part in (feature_set, model_name, f"fold_{fold_id}")
    )
    return checkpoint_dir / f"{stem}.pkl", checkpoint_dir / f"{stem}.json"


def _walk_forward_resume_key(
    *,
    args: argparse.Namespace,
    task_type: str,
    model_name: str,
    feature_set: str,
    fold_id: int,
    target_columns: list[str],
    model_kwargs: dict[str, object],
) -> dict[str, object]:
    return {
        "task_type": task_type,
        "model_name": model_name,
        "feature_set": feature_set,
        "fold_id": int(fold_id),
        "target_columns": list(target_columns),
        "window_length": int(args.window_length),
        "classification_event_type": args.classification_event_type if task_type == "classification" else None,
        "classification_horizon": int(args.classification_horizon) if task_type == "classification" else None,
        "classification_threshold": float(args.classification_threshold) if task_type == "classification" else None,
        "model_kwargs": model_kwargs,
    }


def _append_training_progress(output_dir: Path, payload: dict[str, object]) -> None:
    path = output_dir / "training_progress.jsonl"
    event = {
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _load_fold_checkpoint(path: Path, *, resume_key: dict[str, object]) -> object | None:
    if not path.exists():
        return None
    try:
        bundle = load_model_bundle(path)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "step": "walk_forward_fold_checkpoint_ignored",
                    "artifact_path": str(path.resolve()),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            ),
            flush=True,
        )
        return None
    metadata = bundle.get("metadata") if isinstance(bundle, dict) else None
    if not isinstance(metadata, dict) or metadata.get("training_resume_key") != resume_key:
        print(
            json.dumps(
                {
                    "step": "walk_forward_fold_checkpoint_ignored",
                    "artifact_path": str(path.resolve()),
                    "reason": "resume key mismatch",
                }
            ),
            flush=True,
        )
        return None
    return bundle.get("model")


def _classification_metrics(
    *,
    metadata: pd.DataFrame,
    y_true: pd.DataFrame,
    y_pred,
    dataset: V1Dataset,
    model_name: str,
    feature_set: str,
    split_name: str,
    realized_returns: pd.DataFrame,
    classification_horizon: int,
) -> list[dict[str, object]]:
    realized_col = _realized_return_column(dataset, classification_horizon)
    return evaluate_classification_predictions(
        metadata,
        y_true,
        y_pred.to_numpy(dtype=float) if isinstance(y_pred, pd.DataFrame) else y_pred,
        target_columns=_task_target_columns(dataset, "classification"),
        model_name=model_name,
        feature_set=feature_set,
        split_name=split_name,
        realized_returns=realized_returns,
        realized_return_column=realized_col,
    )


def _train_task_holdout(
    *,
    args: argparse.Namespace,
    dataset: V1Dataset,
    stock_features: pd.DataFrame,
    context_features: pd.DataFrame,
    feature_sets: list[str],
    model_names: list[str],
    horizons: tuple[int, ...],
    output_dir: Path,
    models_dir: Path,
    generated_utc: str,
    task_type: str,
    sequence_stores: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    split = chronological_split(dataset.metadata, train_fraction=args.train_fraction, val_fraction=args.val_fraction)
    split_summary = split_ranges(dataset.metadata, split)
    dataset.metadata.assign(split=split).to_csv(output_dir / "episode_metadata.csv", index=False)
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    trained_records: list[dict[str, object]] = []
    if sequence_stores is None:
        sequence_stores = {
            feature_set: build_sequence_feature_store(
                stock_features,
                feature_set,
                context_features=context_features,
                benchmark_ticker=args.benchmark_ticker,
            )
            for feature_set in feature_sets
            if any(is_sequence_static_model(model_name) for model_name in model_names) and feature_set in SEQUENCE_FEATURE_SET_NAMES
        }
    target_columns = _task_target_columns(dataset, task_type)
    realized_col = _realized_return_column(dataset, args.classification_horizon)
    combos = _build_combo_iterable(feature_sets, model_names)
    combo_total = len(combos)
    combo_start_time = time.monotonic()
    for combo_index, (feature_set, model_name) in enumerate(combos, start=1):
        print(f"Training {task_type}::{model_name} on {feature_set} ({args.eval_mode})...", flush=True)
        _emit_combo_progress(
            f"{task_type}:{model_name}:{feature_set}",
            combo_index - 1,
            combo_total,
            detail="starting holdout fit",
        )
        train_rows = split == "train"
        val_rows = split == "val"
        test_rows = split == "test"
        train_meta = dataset.metadata.loc[train_rows].reset_index(drop=True)
        val_meta = dataset.metadata.loc[val_rows].reset_index(drop=True)
        test_meta = dataset.metadata.loc[test_rows].reset_index(drop=True)
        y_train = dataset.targets.loc[train_rows, target_columns].reset_index(drop=True)
        y_val = dataset.targets.loc[val_rows, target_columns].reset_index(drop=True)
        y_test = dataset.targets.loc[test_rows, target_columns].reset_index(drop=True)
        final_vocabularies = None
        if is_sequence_static_model(model_name):
            store = sequence_stores[feature_set]
            vocabularies = build_category_vocabularies(train_meta, columns=STATIC_CATEGORICAL_COLUMNS)
            x_train = _prepare_sequence_inputs(store, train_meta, vocabularies)
            x_val = _prepare_sequence_inputs(store, val_meta, vocabularies)
            x_test = _prepare_sequence_inputs(store, test_meta, vocabularies)
            final_vocabularies = vocabularies
        else:
            _, x_train, _ = _prepare_flat_inputs(dataset, feature_set, train_rows, task_type=task_type)
            _, x_val, _ = _prepare_flat_inputs(dataset, feature_set, val_rows, task_type=task_type)
            _, x_test, _ = _prepare_flat_inputs(dataset, feature_set, test_rows, task_type=task_type)
        model_kwargs = _model_kwargs(args, model_name, task_type)
        model = make_model(
            model_name,
            window_length=args.window_length,
            task_type=task_type,
            model_kwargs=model_kwargs,
        )
        model.fit(x_train, y_train, val_x=x_val, val_y=y_val)
        for split_name, meta, x_eval, y_eval, rows in (
            ("train", train_meta, x_train, y_train, train_rows),
            ("val", val_meta, x_val, y_val, val_rows),
            ("test", test_meta, x_test, y_test, test_rows),
        ):
            pred = model.predict(x_eval)
            if task_type == "classification":
                realized_returns = dataset.targets.loc[rows, [realized_col]].reset_index(drop=True) if realized_col else pd.DataFrame()
                metrics_rows.extend(
                    _classification_metrics(
                        metadata=meta,
                        y_true=y_eval,
                        y_pred=pred,
                        dataset=dataset,
                        model_name=model_name,
                        feature_set=feature_set,
                        split_name=split_name,
                        realized_returns=realized_returns,
                        classification_horizon=args.classification_horizon,
                    )
                )
            else:
                metrics_rows.extend(
                    evaluate_predictions(
                        meta,
                        y_eval,
                        pred,
                        target_columns=target_columns,
                        model_name=model_name,
                        feature_set=feature_set,
                        split_name=split_name,
                    )
                )
            if split_name in {"val", "test"}:
                prediction_frames.append(
                    prediction_frame(
                        meta,
                        pred,
                        target_columns=target_columns,
                        model_name=model_name,
                        feature_set=feature_set,
                        y_true=y_eval,
                        task_type=task_type,
                    ).assign(split=split_name)
                )
                if task_type == "classification" and realized_col:
                    prediction_frames[-1][f"actual_{realized_col}"] = (
                        dataset.targets.loc[rows, realized_col].reset_index(drop=True)
                    )
        model_path = models_dir / f"{feature_set}__{model_name}.pkl"
        model_metadata = _build_model_metadata(
            task_type=task_type,
            model_name=model_name,
            feature_set=feature_set,
            target_columns=target_columns,
            horizons=horizons,
            window_length=args.window_length,
            benchmark_ticker=args.benchmark_ticker,
            eval_mode=args.eval_mode,
            split_summary=split_summary,
            flat_feature_columns=dataset.feature_columns.get(feature_set),
            sequence_feature_columns=sequence_stores[feature_set].feature_columns if is_sequence_static_model(model_name) else None,
            sequence_components=sequence_feature_config(feature_set).to_dict() if is_sequence_static_model(model_name) else None,
            static_vocabularies=final_vocabularies,
            runtime=_model_runtime_metadata(model),
            classification_threshold=args.classification_threshold,
            classification_horizon=args.classification_horizon,
            classification_event_type=args.classification_event_type,
        )
        save_model_bundle(model_path, model=model, metadata=model_metadata)
        trained_records.append({**model_metadata, "artifact_path": model_path.relative_to(output_dir).as_posix()})
        _emit_combo_progress(
            f"{task_type}:{model_name}:{feature_set}",
            combo_index,
            combo_total,
            detail=f"saved holdout artifact elapsed={round(time.monotonic() - combo_start_time, 1)}s",
        )

    metrics = pd.DataFrame(metrics_rows)
    leaderboard = (
        build_classification_leaderboard(metrics, split_name="val")
        if task_type == "classification"
        else build_leaderboard(metrics, split_name="val")
    )
    predictions = None
    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions = predictions.merge(
            leaderboard[["model_name", "feature_set", "leaderboard_rank", "recommended"]],
            on=["model_name", "feature_set"],
            how="left",
            suffixes=("", "_from_leaderboard"),
        )
        predictions["leaderboard_rank"] = predictions["leaderboard_rank_from_leaderboard"]
        predictions["recommended"] = predictions["recommended_from_leaderboard"]
        predictions = predictions.drop(columns=["leaderboard_rank_from_leaderboard", "recommended_from_leaderboard"])
    _write_task_outputs(
        task_type=task_type,
        eval_mode=args.eval_mode,
        metrics=metrics,
        leaderboard=leaderboard,
        predictions=predictions,
        output_dir=output_dir,
    )
    return trained_records


def _train_task_walk_forward(
    *,
    args: argparse.Namespace,
    dataset: V1Dataset,
    stock_features: pd.DataFrame,
    context_features: pd.DataFrame,
    feature_sets: list[str],
    model_names: list[str],
    horizons: tuple[int, ...],
    output_dir: Path,
    models_dir: Path,
    generated_utc: str,
    task_type: str,
    sequence_stores: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    purge_gap = args.walk_forward_purge_gap or max(horizons)
    folds = build_walk_forward_folds(
        dataset.metadata,
        min_train_dates=args.walk_forward_min_train_dates,
        val_block_size=args.walk_forward_val_block_size,
        oos_block_size=args.walk_forward_oos_block_size,
        purge_gap=purge_gap,
    )
    if not folds:
        raise SystemExit("No walk-forward folds available. Reduce min-train/val/oos settings or add more history.")
    if args.walk_forward_max_folds and args.walk_forward_max_folds > 0:
        folds = folds[-args.walk_forward_max_folds :]
    dataset.metadata.to_csv(output_dir / "episode_metadata.csv", index=False)
    write_json(output_dir / "folds.json", {"folds": [_fold_summary(fold, dataset.metadata) for fold in folds]})
    if sequence_stores is None:
        sequence_stores = {
            feature_set: build_sequence_feature_store(
                stock_features,
                feature_set,
                context_features=context_features,
                benchmark_ticker=args.benchmark_ticker,
            )
            for feature_set in feature_sets
            if any(is_sequence_static_model(model_name) for model_name in model_names) and feature_set in SEQUENCE_FEATURE_SET_NAMES
        }
    fold_metric_rows: list[dict[str, object]] = []
    oos_prediction_frames: list[pd.DataFrame] = []
    oos_metric_rows: list[dict[str, object]] = []
    trained_records: list[dict[str, object]] = []
    all_dates = sorted(dataset.metadata["anchor_date"].dropna().astype(str).unique().tolist())
    target_columns = _task_target_columns(dataset, task_type)
    realized_col = _realized_return_column(dataset, args.classification_horizon)
    combos = _build_combo_iterable(feature_sets, model_names)
    combo_total = len(combos)
    run_start_time = time.monotonic()
    for combo_index, (feature_set, model_name) in enumerate(combos, start=1):
        print(f"Training {task_type}::{model_name} on {feature_set} ({args.eval_mode})...", flush=True)
        _emit_combo_progress(
            f"{task_type}:{model_name}:{feature_set}",
            combo_index - 1,
            combo_total,
            detail="starting walk-forward",
        )
        combo_oos_frames: list[pd.DataFrame] = []
        for fold_index, fold in enumerate(folds, start=1):
            print(
                json.dumps(
                    {
                        "step": "walk_forward_fold_start",
                        "task_type": task_type,
                        "model_name": model_name,
                        "feature_set": feature_set,
                        "fold_id": fold.fold_id,
                    }
                ),
                flush=True,
            )
            _emit_combo_progress(
                f"{task_type}:{model_name}:{feature_set}:folds",
                fold_index - 1,
                len(folds) + 1,
                detail=f"starting fold {fold_index}/{len(folds)} id={fold.fold_id}",
            )
            train_rows = rows_for_dates(dataset.metadata, fold.train_dates)
            val_rows = rows_for_dates(dataset.metadata, fold.val_dates)
            oos_rows = rows_for_dates(dataset.metadata, fold.oos_dates)
            train_meta = dataset.metadata.loc[train_rows].reset_index(drop=True)
            val_meta = dataset.metadata.loc[val_rows].reset_index(drop=True)
            oos_meta = dataset.metadata.loc[oos_rows].reset_index(drop=True)
            y_train = dataset.targets.loc[train_rows, target_columns].reset_index(drop=True)
            y_val = dataset.targets.loc[val_rows, target_columns].reset_index(drop=True)
            y_oos = dataset.targets.loc[oos_rows, target_columns].reset_index(drop=True)
            vocabularies = None
            if is_sequence_static_model(model_name):
                vocabularies = build_category_vocabularies(train_meta, columns=STATIC_CATEGORICAL_COLUMNS)
                store = sequence_stores[feature_set]
                x_train = _prepare_sequence_inputs(store, train_meta, vocabularies)
                x_val = _prepare_sequence_inputs(store, val_meta, vocabularies)
                x_oos = _prepare_sequence_inputs(store, oos_meta, vocabularies)
            else:
                _, x_train, _ = _prepare_flat_inputs(dataset, feature_set, train_rows, task_type=task_type)
                _, x_val, _ = _prepare_flat_inputs(dataset, feature_set, val_rows, task_type=task_type)
                _, x_oos, _ = _prepare_flat_inputs(dataset, feature_set, oos_rows, task_type=task_type)
            model_kwargs = _model_kwargs(args, model_name, task_type)
            checkpoint_path, checkpoint_manifest_path = _walk_forward_checkpoint_paths(
                output_dir=output_dir,
                task_type=task_type,
                model_name=model_name,
                feature_set=feature_set,
                fold_id=fold.fold_id,
            )
            resume_key = _walk_forward_resume_key(
                args=args,
                task_type=task_type,
                model_name=model_name,
                feature_set=feature_set,
                fold_id=fold.fold_id,
                target_columns=target_columns,
                model_kwargs=model_kwargs,
            )
            model = _load_fold_checkpoint(checkpoint_path, resume_key=resume_key)
            if model is not None:
                event = {
                    "step": "walk_forward_fold_checkpoint_loaded",
                    "task_type": task_type,
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "fold_id": fold.fold_id,
                    "artifact_path": str(checkpoint_path.resolve()),
                }
                print(json.dumps(event), flush=True)
                _append_training_progress(output_dir, event)
            else:
                model = make_model(
                    model_name,
                    window_length=args.window_length,
                    task_type=task_type,
                    model_kwargs=model_kwargs,
                )
                model.fit(x_train, y_train, val_x=x_val, val_y=y_val)
                checkpoint_metadata = _build_model_metadata(
                    task_type=task_type,
                    model_name=model_name,
                    feature_set=feature_set,
                    target_columns=target_columns,
                    horizons=horizons,
                    window_length=args.window_length,
                    benchmark_ticker=args.benchmark_ticker,
                    eval_mode=args.eval_mode,
                    split_summary={"fold": _fold_summary(fold, dataset.metadata)},
                    flat_feature_columns=dataset.feature_columns.get(feature_set),
                    sequence_feature_columns=sequence_stores[feature_set].feature_columns if is_sequence_static_model(model_name) else None,
                    sequence_components=sequence_feature_config(feature_set).to_dict() if is_sequence_static_model(model_name) else None,
                    static_vocabularies=vocabularies,
                    runtime=_model_runtime_metadata(model),
                    classification_threshold=args.classification_threshold,
                    classification_horizon=args.classification_horizon,
                    classification_event_type=args.classification_event_type,
                )
                checkpoint_metadata["checkpoint_type"] = "walk_forward_fold"
                checkpoint_metadata["training_resume_key"] = resume_key
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                save_model_bundle(checkpoint_path, model=model, metadata=checkpoint_metadata)
                write_json(
                    checkpoint_manifest_path,
                    {
                        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "checkpoint_type": "walk_forward_fold",
                        "artifact_path": str(checkpoint_path.resolve()),
                        "training_resume_key": resume_key,
                        "runtime": checkpoint_metadata.get("runtime"),
                    },
                )
                event = {
                    "step": "walk_forward_fold_checkpoint_saved",
                    "task_type": task_type,
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "fold_id": fold.fold_id,
                    "artifact_path": str(checkpoint_path.resolve()),
                }
                print(json.dumps(event), flush=True)
                _append_training_progress(output_dir, event)
            print(
                json.dumps(
                    {
                        "step": "walk_forward_fold_fit_complete",
                        "task_type": task_type,
                        "model_name": model_name,
                        "feature_set": feature_set,
                        "fold_id": fold.fold_id,
                    }
                ),
                flush=True,
            )
            _emit_combo_progress(
                f"{task_type}:{model_name}:{feature_set}:folds",
                fold_index,
                len(folds) + 1,
                detail=f"fit complete fold {fold_index}/{len(folds)} id={fold.fold_id}",
            )
            for split_name, meta, x_eval, y_eval, rows in (("val", val_meta, x_val, y_val, val_rows), ("oos", oos_meta, x_oos, y_oos, oos_rows)):
                pred = model.predict(x_eval)
                if task_type == "classification":
                    realized_returns = dataset.targets.loc[rows, [realized_col]].reset_index(drop=True) if realized_col else pd.DataFrame()
                    rows_out = _classification_metrics(
                        metadata=meta,
                        y_true=y_eval,
                        y_pred=pred,
                        dataset=dataset,
                        model_name=model_name,
                        feature_set=feature_set,
                        split_name=split_name,
                        realized_returns=realized_returns,
                        classification_horizon=args.classification_horizon,
                    )
                else:
                    rows_out = evaluate_predictions(
                        meta,
                        y_eval,
                        pred,
                        target_columns=target_columns,
                        model_name=model_name,
                        feature_set=feature_set,
                        split_name=split_name,
                    )
                for row in rows_out:
                    row["fold_id"] = fold.fold_id
                fold_metric_rows.extend(rows_out)
                if split_name == "oos":
                    combo_oos_frames.append(
                        prediction_frame(
                            meta,
                            pred,
                            target_columns=target_columns,
                            model_name=model_name,
                            feature_set=feature_set,
                            y_true=y_eval,
                            task_type=task_type,
                        ).assign(fold_id=fold.fold_id, split="oos")
                    )
                    if task_type == "classification" and realized_col:
                        combo_oos_frames[-1][f"actual_{realized_col}"] = (
                            dataset.targets.loc[oos_rows, realized_col].reset_index(drop=True)
                        )
        combo_oos = pd.concat(combo_oos_frames, ignore_index=True)
        combo_actual, combo_pred = _prediction_targets_from_frame(combo_oos, target_columns, task_type=task_type)
        if task_type == "classification":
            realized_frame = pd.DataFrame()
            if realized_col and f"actual_{realized_col}" in combo_oos.columns:
                realized_frame = pd.DataFrame({realized_col: combo_oos[f"actual_{realized_col}"]})
            oos_metric_rows.extend(
                evaluate_classification_predictions(
                    combo_oos[["ticker", "anchor_date"]],
                    combo_actual,
                    combo_pred.to_numpy(dtype=float),
                    target_columns=target_columns,
                    model_name=model_name,
                    feature_set=feature_set,
                    split_name="oos",
                    realized_returns=realized_frame,
                    realized_return_column=realized_col,
                )
            )
        else:
            oos_metric_rows.extend(
                evaluate_predictions(
                    combo_oos[["ticker", "anchor_date"]],
                    combo_actual,
                    combo_pred.to_numpy(dtype=float),
                    target_columns=target_columns,
                    model_name=model_name,
                    feature_set=feature_set,
                    split_name="oos",
                )
            )
        oos_prediction_frames.append(combo_oos)

        print(
            json.dumps(
                {
                    "step": "final_deploy_fit_start",
                    "task_type": task_type,
                    "model_name": model_name,
                    "feature_set": feature_set,
                }
            ),
            flush=True,
        )
        _emit_combo_progress(
            f"{task_type}:{model_name}:{feature_set}:folds",
            len(folds),
            len(folds) + 1,
            detail="starting final deploy fit",
        )
        final_model, final_vocabularies = _fit_final_deploy_model(
            dataset=dataset,
            feature_set=feature_set,
            model_name=model_name,
            task_type=task_type,
            all_dates=all_dates,
            final_stop_block_size=args.final_stop_block_size,
            sequence_store=sequence_stores.get(feature_set),
            window_length=args.window_length,
            model_kwargs=_model_kwargs(args, model_name, task_type),
        )
        model_path = models_dir / f"{feature_set}__{model_name}.pkl"
        model_metadata = _build_model_metadata(
            task_type=task_type,
            model_name=model_name,
            feature_set=feature_set,
            target_columns=target_columns,
            horizons=horizons,
            window_length=args.window_length,
            benchmark_ticker=args.benchmark_ticker,
            eval_mode=args.eval_mode,
            flat_feature_columns=dataset.feature_columns.get(feature_set),
            sequence_feature_columns=sequence_stores[feature_set].feature_columns if is_sequence_static_model(model_name) else None,
            sequence_components=sequence_feature_config(feature_set).to_dict() if is_sequence_static_model(model_name) else None,
            static_vocabularies=final_vocabularies,
            runtime=_model_runtime_metadata(final_model),
            classification_threshold=args.classification_threshold,
            classification_horizon=args.classification_horizon,
            classification_event_type=args.classification_event_type,
        )
        save_model_bundle(model_path, model=final_model, metadata=model_metadata)
        print(
            json.dumps(
                {
                    "step": "final_deploy_model_saved",
                    "task_type": task_type,
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "artifact_path": str(model_path.resolve()),
                }
            ),
                flush=True,
            )
        trained_records.append({**model_metadata, "artifact_path": model_path.relative_to(output_dir).as_posix()})
        _emit_combo_progress(
            f"{task_type}:{model_name}:{feature_set}",
            combo_index,
            combo_total,
            detail=f"saved final artifact total_elapsed={round(time.monotonic() - run_start_time, 1)}s",
        )

    fold_metrics = pd.DataFrame(fold_metric_rows)
    oos_metrics = pd.DataFrame(oos_metric_rows)
    leaderboard = (
        build_classification_leaderboard(oos_metrics, split_name="oos")
        if task_type == "classification"
        else build_leaderboard(oos_metrics, split_name="oos")
    )
    predictions = None
    if oos_prediction_frames:
        predictions = pd.concat(oos_prediction_frames, ignore_index=True)
        predictions = predictions.merge(
            leaderboard[["model_name", "feature_set", "leaderboard_rank", "recommended"]],
            on=["model_name", "feature_set"],
            how="left",
            suffixes=("", "_from_leaderboard"),
        )
        predictions["leaderboard_rank"] = predictions["leaderboard_rank_from_leaderboard"]
        predictions["recommended"] = predictions["recommended_from_leaderboard"]
        predictions = predictions.drop(columns=["leaderboard_rank_from_leaderboard", "recommended_from_leaderboard"])
    prefix = _task_file_prefix(task_type)
    fold_metrics.to_csv(output_dir / f"{prefix}fold_metrics.csv", index=False)
    _write_task_outputs(
        task_type=task_type,
        eval_mode=args.eval_mode,
        metrics=oos_metrics,
        leaderboard=leaderboard,
        predictions=predictions,
        output_dir=output_dir,
    )
    return trained_records, leaderboard


def main() -> None:
    args = parse_args()
    horizons = parse_horizons(args.horizons)
    feature_sets = [item.strip() for item in args.feature_sets.split(",") if item.strip()]
    valid_feature_sets = tuple(dict.fromkeys((*FEATURE_SET_NAMES, *SEQUENCE_FEATURE_SET_NAMES)))
    invalid_sets = [name for name in feature_sets if name not in valid_feature_sets]
    if invalid_sets:
        raise SystemExit(f"Unknown feature set(s): {invalid_sets}. Valid: {valid_feature_sets}")

    run_name = args.run_name or datetime.utcnow().strftime(f"{args.eval_mode}_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_name
    models_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = output_dir / "training_progress.jsonl"
    os.environ["V1_TRAINING_PROGRESS_LOG"] = str(progress_log_path.resolve())
    _append_training_progress(
        output_dir,
        {
            "step": "training_run_start",
            "run_name": run_name,
            "eval_mode": args.eval_mode,
            "task_type": args.task_type,
        },
    )

    eligibility_config = _episode_eligibility_config(args)
    research_config = _research_universe_config(args)
    cached_sequence_stores: dict[str, object] | None = None
    if args.episode_cache_dir:
        print("Loading materialized V1 episode cache...", flush=True)
        dataset = load_cached_v1_dataset(args.episode_cache_dir)
        cached_sequence_stores = load_cached_sequence_stores(args.episode_cache_dir)
        cache_manifest_path = Path(args.episode_cache_dir) / "manifest.json"
        if cache_manifest_path.exists():
            cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
            payload = cache_manifest.get("research_universe")
            if isinstance(payload, dict) and payload.get("enabled") is not False:
                research_config = ConservativeResearchUniverseConfig.from_mapping(payload)
            elif isinstance(payload, dict):
                research_config = None
        stock_features = pd.DataFrame()
        context_features = pd.DataFrame()
    else:
        print("Loading daily stock features...", flush=True)
        stock_features = load_daily_features(args.dataset_root)
        context_features = load_market_context_features(args.dataset_root, stock_features=stock_features)
        if context_features.empty:
            raise SystemExit("Market context features are missing. Run scripts/update_eodhd_daily_dataset.py first.")

        print("Building V1 supervised dataset...", flush=True)
        dataset = build_v1_dataset(
            stock_features,
            context_features,
            horizons=horizons,
            window_length=args.window_length,
            benchmark_ticker=args.benchmark_ticker,
            max_episodes=args.max_episodes or None,
            classification_horizon=args.classification_horizon,
            classification_threshold=args.classification_threshold,
            classification_event_type=args.classification_event_type,
            eligibility_config=eligibility_config,
            research_config=research_config,
            feature_set_names=feature_sets,
        )
    if dataset.classification_target_columns == [PATH_5PCT_20D_EVENT_TYPE]:
        args.classification_event_type = PATH_5PCT_20D_EVENT_TYPE
    elif args.classification_event_type == PATH_5PCT_20D_EVENT_TYPE:
        raise SystemExit(
            "Requested classification_event_type=path_5pct_20d, but the loaded dataset/cache "
            f"has classification_target_columns={dataset.classification_target_columns}."
        )
    dataset.targets.to_csv(output_dir / "episode_targets.csv", index=False)
    generated_utc = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    task_types = _task_types(args)
    task_models = {task_type: _resolve_model_names(args, task_type) for task_type in task_types}
    dataset_manifest = {
        "generated_utc": generated_utc,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "episode_cache_dir": str(Path(args.episode_cache_dir).resolve()) if args.episode_cache_dir else None,
        "horizons": list(horizons),
        "target_columns": dataset.target_columns,
        "classification_target_columns": dataset.classification_target_columns,
        "classification_threshold": args.classification_threshold,
        "classification_horizon_days": args.classification_horizon,
        "classification_event_type": args.classification_event_type,
        "classification_label_kind": classification_label_kind(args.classification_event_type),
        "classification_class_labels": classification_class_labels(args.classification_event_type),
        "classification_positive_class": classification_positive_class(args.classification_event_type),
        "labeling": dataset.labeling_summary,
        "window_length": args.window_length,
        "benchmark_ticker": args.benchmark_ticker.upper(),
        "feature_sets": {
            name: (
                len(dataset.feature_columns[name])
                if name in dataset.feature_columns
                else {"sequence_components": sequence_feature_config(name).to_dict()}
            )
            for name in feature_sets
        },
        "models": task_models,
        "row_count": int(len(dataset.metadata)),
        "eval_mode": args.eval_mode,
        "task_types": task_types,
        "runtime_environment": _runtime_environment(),
        "episode_eligibility": (
            eligibility_config.to_dict()
            if eligibility_config is not None
            else {"enabled": False}
        ),
        "research_universe": (
            research_config.to_dict()
            if research_config is not None
            else {"enabled": False}
        ),
        "notes": [
            "Current default data source is EODHD daily EOD OHLCV.",
            "Regression targets are market-adjusted using the benchmark context table.",
            "Default binary classification is positive when pathwise market-adjusted excess return exceeds the threshold within the next horizon window.",
            "path_5pct_20d is an opt-in 3-class close-return path label with class 2 as the ranking class.",
            "Feature summaries are rolling-window last/mean/std values computed from dates <= anchor_date.",
            "Episode eligibility is evaluated as of anchor_date using available history, valid adjusted OHLCV rows, liquidity, price, and exchange filters.",
            "The conservative research universe is a shared strategy-universe filter used for train/test/live by default.",
            "Walk-forward mode evaluates on aggregated out-of-sample folds and excludes 1-day regression targets from leaderboard ranking.",
            "EODHD sector/industry metadata is not treated as point-in-time fundamentals.",
        ],
    }
    if args.eval_mode == "holdout":
        split = chronological_split(dataset.metadata, train_fraction=args.train_fraction, val_fraction=args.val_fraction)
        dataset_manifest["split_summary"] = split_ranges(dataset.metadata, split)
    else:
        dataset_manifest["walk_forward"] = {
            "min_train_dates": args.walk_forward_min_train_dates,
            "val_block_size": args.walk_forward_val_block_size,
            "oos_block_size": args.walk_forward_oos_block_size,
            "max_folds": args.walk_forward_max_folds or None,
            "purge_gap": args.walk_forward_purge_gap or max(horizons),
            "final_stop_block_size": args.final_stop_block_size,
        }
    torch_training_overrides = {
        "max_epochs": args.torch_max_epochs or None,
        "patience": args.torch_patience or None,
        "batch_size": args.torch_batch_size or None,
        "hidden_units": args.torch_hidden_units or None,
        "hidden_dim": args.torch_hidden_dim or None,
    }
    xgboost_training_overrides = {
        "n_estimators": args.xgboost_n_estimators or None,
        "patience": args.xgboost_patience or None,
    }
    if any(value is not None for value in torch_training_overrides.values()):
        dataset_manifest["torch_training_overrides"] = torch_training_overrides
    if any(value is not None for value in xgboost_training_overrides.values()):
        dataset_manifest["xgboost_training_overrides"] = xgboost_training_overrides
    dataset_manifest["training_progress_log"] = progress_log_path.relative_to(output_dir).as_posix()
    dataset_manifest["checkpoint_dir"] = (output_dir / "checkpoints").relative_to(output_dir).as_posix()
    save_dataset_manifest(output_dir / "dataset_manifest.json", dataset_manifest)

    combined_records: list[dict[str, object]] = []
    regression_leaderboard = None
    for task_type in task_types:
        model_names = task_models[task_type]
        if args.eval_mode == "holdout":
            task_records = _train_task_holdout(
                args=args,
                dataset=dataset,
                stock_features=stock_features,
                context_features=context_features,
                feature_sets=feature_sets,
                model_names=model_names,
                horizons=horizons,
                output_dir=output_dir,
                models_dir=models_dir,
                generated_utc=generated_utc,
                task_type=task_type,
                sequence_stores=cached_sequence_stores,
            )
            leaderboard = pd.read_csv(output_dir / ("leaderboard.csv" if task_type == "regression" else "classification_leaderboard.csv"))
        else:
            task_records, leaderboard = _train_task_walk_forward(
                args=args,
                dataset=dataset,
                stock_features=stock_features,
                context_features=context_features,
                feature_sets=feature_sets,
                model_names=model_names,
                horizons=horizons,
                output_dir=output_dir,
                models_dir=models_dir,
                generated_utc=generated_utc,
                task_type=task_type,
                sequence_stores=cached_sequence_stores,
            )
        combined_records.extend(task_records)
        suffix = "regression" if task_type == "regression" else "classification"
        _write_model_index(
            output_dir / f"final_{suffix}_models.json",
            generated_utc=generated_utc,
            eval_mode=args.eval_mode,
            records=task_records,
        )
        _write_model_index(
            output_dir / f"trained_{suffix}_models.json",
            generated_utc=generated_utc,
            eval_mode=args.eval_mode,
            records=task_records,
        )
        if task_type == "regression":
            regression_leaderboard = leaderboard
    _write_model_index(output_dir / "final_models.json", generated_utc=generated_utc, eval_mode=args.eval_mode, records=combined_records)
    _write_model_index(output_dir / "trained_models.json", generated_utc=generated_utc, eval_mode=args.eval_mode, records=combined_records)

    legacy_run_dir = Path(args.compare_against_run) if args.compare_against_run else None
    if legacy_run_dir and legacy_run_dir.exists() and regression_leaderboard is not None:
        _compare_against_legacy(legacy_run_dir=legacy_run_dir, new_leaderboard=regression_leaderboard, output_dir=output_dir)
    print(f"Training complete. Artifacts written to {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
