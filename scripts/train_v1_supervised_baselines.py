from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.v1_dataset import (  # noqa: E402
    DEFAULT_BENCHMARK_TICKER,
    DEFAULT_WINDOW_LENGTH,
    FEATURE_SET_NAMES,
    SEQUENCE_FEATURE_SET_NAMES,
    STATIC_CATEGORICAL_COLUMNS,
    V1Dataset,
    build_category_vocabularies,
    build_sequence_feature_store,
    build_v1_dataset,
    build_walk_forward_folds,
    chronological_split,
    encode_static_categories,
    load_daily_features,
    load_market_context_features,
    parse_horizons,
    rows_for_dates,
    save_dataset_manifest,
    split_ranges,
)
from src.models.v1_baselines import (  # noqa: E402
    build_leaderboard,
    build_metric_summary,
    default_model_names,
    evaluate_predictions,
    is_sequence_static_model,
    make_model,
    prediction_frame,
    save_model_bundle,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train V1 supervised multi-output return baselines and write model artifacts."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/massive_sp500_current_constituents_history",
        help="Dataset folder containing processed daily features and market context features.",
    )
    parser.add_argument("--output-root", default="artifacts/v1_baselines", help="Artifact output root.")
    parser.add_argument("--run-name", default="", help="Optional run folder name.")
    parser.add_argument("--horizons", default="1,5,10,20", help="Comma-separated target horizons.")
    parser.add_argument("--window-length", type=int, default=DEFAULT_WINDOW_LENGTH)
    parser.add_argument("--benchmark-ticker", default=DEFAULT_BENCHMARK_TICKER)
    parser.add_argument("--feature-sets", default=",".join(FEATURE_SET_NAMES))
    parser.add_argument("--models", default="", help="Comma-separated model list. Defaults depend on eval mode.")
    parser.add_argument("--eval-mode", choices=("walk_forward", "holdout"), default="walk_forward")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--walk-forward-min-train-dates", type=int, default=252)
    parser.add_argument("--walk-forward-val-block-size", type=int, default=21)
    parser.add_argument("--walk-forward-oos-block-size", type=int, default=21)
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
    parser.add_argument("--compare-against-run", default="", help="Optional legacy run directory for comparison.")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Optional cap for smoke runs. Keeps the most recent N eligible episodes.",
    )
    return parser.parse_args()


def _resolve_model_names(args: argparse.Namespace) -> list[str]:
    if args.models.strip():
        return [item.strip() for item in args.models.split(",") if item.strip()]
    return default_model_names(args.eval_mode)


def _supported_feature_set(model_name: str, feature_set: str) -> bool:
    if is_sequence_static_model(model_name):
        return feature_set in SEQUENCE_FEATURE_SET_NAMES
    return feature_set in FEATURE_SET_NAMES


def _prepare_flat_inputs(
    dataset: V1Dataset,
    feature_set: str,
    rows: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = dataset.metadata.loc[rows].reset_index(drop=True)
    x = dataset.feature_sets[feature_set].loc[rows, dataset.feature_columns[feature_set]].reset_index(drop=True)
    y = dataset.targets.loc[rows, dataset.target_columns].reset_index(drop=True)
    return meta, x, y


def _prepare_sequence_inputs(
    store,
    metadata: pd.DataFrame,
    vocabularies: dict[str, dict[str, int]],
) -> dict[str, object]:
    return {
        "store": store,
        "metadata": metadata.reset_index(drop=True),
        "static_categorical": encode_static_categories(
            metadata.reset_index(drop=True),
            vocabularies,
            columns=STATIC_CATEGORICAL_COLUMNS,
        ),
    }


def _prediction_targets_from_frame(frame: pd.DataFrame, target_columns: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    actual = pd.DataFrame(
        {
            target: frame[target.replace("market_adjusted_return", "actual_market_adjusted_return")]
            for target in target_columns
        }
    )
    pred = np.column_stack(
        [frame[target.replace("market_adjusted_return", "pred_market_adjusted_return")].to_numpy() for target in target_columns]
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
    static_vocabularies: dict[str, dict[str, int]] | None = None,
) -> dict[str, object]:
    input_layout = "sequence_static" if is_sequence_static_model(model_name) else "tabular"
    metadata = {
        "model_name": model_name,
        "feature_set": feature_set,
        "target_columns": target_columns,
        "horizons": list(horizons),
        "window_length": window_length,
        "benchmark_ticker": benchmark_ticker.upper(),
        "evaluation_mode": eval_mode,
        "input_layout": input_layout,
    }
    if split_summary is not None:
        metadata["split_summary"] = split_summary
    if flat_feature_columns is not None:
        metadata["feature_columns"] = flat_feature_columns
    if sequence_feature_columns is not None:
        metadata["sequence_feature_columns"] = sequence_feature_columns
        metadata["static_categorical_columns"] = list(STATIC_CATEGORICAL_COLUMNS)
        metadata["static_vocabularies"] = static_vocabularies or {}
    return metadata


def _build_combo_iterable(feature_sets: list[str], model_names: list[str]) -> list[tuple[str, str]]:
    combos: list[tuple[str, str]] = []
    for feature_set in feature_sets:
        for model_name in model_names:
            if _supported_feature_set(model_name, feature_set):
                combos.append((feature_set, model_name))
    if not combos:
        raise SystemExit("No valid (feature_set, model_name) combinations to train.")
    return combos


def _fit_final_deploy_model(
    *,
    dataset: V1Dataset,
    feature_set: str,
    model_name: str,
    all_dates: list[str],
    final_stop_block_size: int,
    sequence_store=None,
    window_length: int,
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
    y_train = dataset.targets.loc[train_rows, dataset.target_columns].reset_index(drop=True)
    y_val = dataset.targets.loc[val_rows, dataset.target_columns].reset_index(drop=True)
    y_full = dataset.targets.loc[full_rows, dataset.target_columns].reset_index(drop=True)
    final_model = make_model(model_name, window_length=window_length)
    final_vocabularies: dict[str, dict[str, int]] | None = None
    if is_sequence_static_model(model_name):
        final_vocabularies = build_category_vocabularies(full_meta, columns=STATIC_CATEGORICAL_COLUMNS)
        x_train = _prepare_sequence_inputs(sequence_store, train_meta, final_vocabularies)
        x_val = _prepare_sequence_inputs(sequence_store, val_meta, final_vocabularies)
        x_full = _prepare_sequence_inputs(sequence_store, full_meta, final_vocabularies)
    else:
        _, x_train, _ = _prepare_flat_inputs(dataset, feature_set, train_rows)
        _, x_val, _ = _prepare_flat_inputs(dataset, feature_set, val_rows)
        _, x_full, _ = _prepare_flat_inputs(dataset, feature_set, full_rows)
    final_model.fit(x_train, y_train, val_x=x_val, val_y=y_val)
    if hasattr(final_model, "refit_full"):
        final_model.refit_full(x_full, y_full)
    else:
        final_model = make_model(model_name, window_length=window_length)
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
    write_json(
        path,
        {
            "generated_utc": generated_utc,
            "evaluation_mode": eval_mode,
            "models": records,
        },
    )


def _train_holdout(
    *,
    args: argparse.Namespace,
    dataset: V1Dataset,
    stock_features: pd.DataFrame,
    feature_sets: list[str],
    model_names: list[str],
    horizons: tuple[int, ...],
    output_dir: Path,
    models_dir: Path,
    generated_utc: str,
) -> None:
    split = chronological_split(dataset.metadata, train_fraction=args.train_fraction, val_fraction=args.val_fraction)
    split_summary = split_ranges(dataset.metadata, split)
    dataset.metadata.assign(split=split).to_csv(output_dir / "episode_metadata.csv", index=False)

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    trained_records: list[dict[str, object]] = []
    sequence_stores = {
        feature_set: build_sequence_feature_store(
            stock_features,
            feature_set,
            benchmark_ticker=args.benchmark_ticker,
        )
        for feature_set in feature_sets
        if any(is_sequence_static_model(model_name) for model_name in model_names) and feature_set in SEQUENCE_FEATURE_SET_NAMES
    }

    for feature_set, model_name in _build_combo_iterable(feature_sets, model_names):
        print(f"Training {model_name} on {feature_set} ({args.eval_mode})...")
        train_rows = split == "train"
        val_rows = split == "val"
        test_rows = split == "test"
        train_meta = dataset.metadata.loc[train_rows].reset_index(drop=True)
        val_meta = dataset.metadata.loc[val_rows].reset_index(drop=True)
        test_meta = dataset.metadata.loc[test_rows].reset_index(drop=True)
        y_train = dataset.targets.loc[train_rows, dataset.target_columns].reset_index(drop=True)
        y_val = dataset.targets.loc[val_rows, dataset.target_columns].reset_index(drop=True)
        y_test = dataset.targets.loc[test_rows, dataset.target_columns].reset_index(drop=True)
        final_vocabularies = None
        if is_sequence_static_model(model_name):
            store = sequence_stores[feature_set]
            vocabularies = build_category_vocabularies(train_meta, columns=STATIC_CATEGORICAL_COLUMNS)
            x_train = _prepare_sequence_inputs(store, train_meta, vocabularies)
            x_val = _prepare_sequence_inputs(store, val_meta, vocabularies)
            x_test = _prepare_sequence_inputs(store, test_meta, vocabularies)
            final_vocabularies = vocabularies
        else:
            _, x_train, _ = _prepare_flat_inputs(dataset, feature_set, train_rows)
            _, x_val, _ = _prepare_flat_inputs(dataset, feature_set, val_rows)
            _, x_test, _ = _prepare_flat_inputs(dataset, feature_set, test_rows)
        model = make_model(model_name, window_length=args.window_length)
        model.fit(x_train, y_train, val_x=x_val, val_y=y_val)
        for split_name, meta, x_eval, y_eval in (
            ("train", train_meta, x_train, y_train),
            ("val", val_meta, x_val, y_val),
            ("test", test_meta, x_test, y_test),
        ):
            pred = model.predict(x_eval)
            metrics_rows.extend(
                evaluate_predictions(
                    meta,
                    y_eval,
                    pred,
                    target_columns=dataset.target_columns,
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
                        target_columns=dataset.target_columns,
                        model_name=model_name,
                        feature_set=feature_set,
                        y_true=y_eval,
                    ).assign(split=split_name)
                )
        model_path = models_dir / f"{feature_set}__{model_name}.pkl"
        model_metadata = _build_model_metadata(
            model_name=model_name,
            feature_set=feature_set,
            target_columns=dataset.target_columns,
            horizons=horizons,
            window_length=args.window_length,
            benchmark_ticker=args.benchmark_ticker,
            eval_mode=args.eval_mode,
            split_summary=split_summary,
            flat_feature_columns=dataset.feature_columns.get(feature_set),
            sequence_feature_columns=sequence_stores[feature_set].feature_columns if is_sequence_static_model(model_name) else None,
            static_vocabularies=final_vocabularies,
        )
        save_model_bundle(model_path, model=model, metadata=model_metadata)
        trained_records.append({**model_metadata, "artifact_path": str(model_path.resolve())})

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    leaderboard = build_leaderboard(metrics, split_name="val")
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
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
        predictions.to_csv(output_dir / "val_test_predictions.csv", index=False)
    _write_model_index(output_dir / "trained_models.json", generated_utc=generated_utc, eval_mode=args.eval_mode, records=trained_records)
    _write_model_index(output_dir / "final_models.json", generated_utc=generated_utc, eval_mode=args.eval_mode, records=trained_records)


def _train_walk_forward(
    *,
    args: argparse.Namespace,
    dataset: V1Dataset,
    stock_features: pd.DataFrame,
    feature_sets: list[str],
    model_names: list[str],
    horizons: tuple[int, ...],
    output_dir: Path,
    models_dir: Path,
    generated_utc: str,
) -> None:
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

    dataset.metadata.to_csv(output_dir / "episode_metadata.csv", index=False)
    write_json(output_dir / "folds.json", {"folds": [_fold_summary(fold, dataset.metadata) for fold in folds]})

    sequence_stores = {
        feature_set: build_sequence_feature_store(
            stock_features,
            feature_set,
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

    for feature_set, model_name in _build_combo_iterable(feature_sets, model_names):
        print(f"Training {model_name} on {feature_set} ({args.eval_mode})...")
        combo_oos_frames: list[pd.DataFrame] = []
        for fold in folds:
            train_rows = rows_for_dates(dataset.metadata, fold.train_dates)
            val_rows = rows_for_dates(dataset.metadata, fold.val_dates)
            oos_rows = rows_for_dates(dataset.metadata, fold.oos_dates)
            train_meta = dataset.metadata.loc[train_rows].reset_index(drop=True)
            val_meta = dataset.metadata.loc[val_rows].reset_index(drop=True)
            oos_meta = dataset.metadata.loc[oos_rows].reset_index(drop=True)
            y_train = dataset.targets.loc[train_rows, dataset.target_columns].reset_index(drop=True)
            y_val = dataset.targets.loc[val_rows, dataset.target_columns].reset_index(drop=True)
            y_oos = dataset.targets.loc[oos_rows, dataset.target_columns].reset_index(drop=True)
            if is_sequence_static_model(model_name):
                vocabularies = build_category_vocabularies(train_meta, columns=STATIC_CATEGORICAL_COLUMNS)
                store = sequence_stores[feature_set]
                x_train = _prepare_sequence_inputs(store, train_meta, vocabularies)
                x_val = _prepare_sequence_inputs(store, val_meta, vocabularies)
                x_oos = _prepare_sequence_inputs(store, oos_meta, vocabularies)
            else:
                _, x_train, _ = _prepare_flat_inputs(dataset, feature_set, train_rows)
                _, x_val, _ = _prepare_flat_inputs(dataset, feature_set, val_rows)
                _, x_oos, _ = _prepare_flat_inputs(dataset, feature_set, oos_rows)
            model = make_model(model_name, window_length=args.window_length)
            model.fit(x_train, y_train, val_x=x_val, val_y=y_val)
            for split_name, meta, x_eval, y_eval in (("val", val_meta, x_val, y_val), ("oos", oos_meta, x_oos, y_oos)):
                pred = model.predict(x_eval)
                rows = evaluate_predictions(
                    meta,
                    y_eval,
                    pred,
                    target_columns=dataset.target_columns,
                    model_name=model_name,
                    feature_set=feature_set,
                    split_name=split_name,
                )
                for row in rows:
                    row["fold_id"] = fold.fold_id
                fold_metric_rows.extend(rows)
                if split_name == "oos":
                    combo_oos_frames.append(
                        prediction_frame(
                            meta,
                            pred,
                            target_columns=dataset.target_columns,
                            model_name=model_name,
                            feature_set=feature_set,
                            y_true=y_eval,
                        ).assign(fold_id=fold.fold_id, split="oos")
                    )
        combo_oos = pd.concat(combo_oos_frames, ignore_index=True)
        combo_actual, combo_pred = _prediction_targets_from_frame(combo_oos, dataset.target_columns)
        oos_metric_rows.extend(
            evaluate_predictions(
                combo_oos[["ticker", "anchor_date"]],
                combo_actual,
                combo_pred,
                target_columns=dataset.target_columns,
                model_name=model_name,
                feature_set=feature_set,
                split_name="oos",
            )
        )
        oos_prediction_frames.append(combo_oos)

        final_model, final_vocabularies = _fit_final_deploy_model(
            dataset=dataset,
            feature_set=feature_set,
            model_name=model_name,
            all_dates=all_dates,
            final_stop_block_size=args.final_stop_block_size,
            sequence_store=sequence_stores.get(feature_set),
            window_length=args.window_length,
        )
        model_path = models_dir / f"{feature_set}__{model_name}.pkl"
        model_metadata = _build_model_metadata(
            model_name=model_name,
            feature_set=feature_set,
            target_columns=dataset.target_columns,
            horizons=horizons,
            window_length=args.window_length,
            benchmark_ticker=args.benchmark_ticker,
            eval_mode=args.eval_mode,
            flat_feature_columns=dataset.feature_columns.get(feature_set),
            sequence_feature_columns=sequence_stores[feature_set].feature_columns if is_sequence_static_model(model_name) else None,
            static_vocabularies=final_vocabularies,
        )
        save_model_bundle(model_path, model=final_model, metadata=model_metadata)
        trained_records.append({**model_metadata, "artifact_path": str(model_path.resolve())})

    fold_metrics = pd.DataFrame(fold_metric_rows)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    oos_metrics = pd.DataFrame(oos_metric_rows)
    oos_metrics.to_csv(output_dir / "metrics.csv", index=False)
    oos_leaderboard = build_leaderboard(oos_metrics, split_name="oos")
    oos_leaderboard.to_csv(output_dir / "oos_leaderboard.csv", index=False)
    oos_leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    if oos_prediction_frames:
        oos_predictions = pd.concat(oos_prediction_frames, ignore_index=True)
        oos_predictions = oos_predictions.merge(
            oos_leaderboard[["model_name", "feature_set", "leaderboard_rank", "recommended"]],
            on=["model_name", "feature_set"],
            how="left",
            suffixes=("", "_from_leaderboard"),
        )
        oos_predictions["leaderboard_rank"] = oos_predictions["leaderboard_rank_from_leaderboard"]
        oos_predictions["recommended"] = oos_predictions["recommended_from_leaderboard"]
        oos_predictions = oos_predictions.drop(columns=["leaderboard_rank_from_leaderboard", "recommended_from_leaderboard"])
        oos_predictions.to_csv(output_dir / "oos_predictions.csv", index=False)

    _write_model_index(output_dir / "final_models.json", generated_utc=generated_utc, eval_mode=args.eval_mode, records=trained_records)
    _write_model_index(output_dir / "trained_models.json", generated_utc=generated_utc, eval_mode=args.eval_mode, records=trained_records)

    legacy_run_dir = Path(args.compare_against_run) if args.compare_against_run else None
    if legacy_run_dir and legacy_run_dir.exists():
        _compare_against_legacy(legacy_run_dir=legacy_run_dir, new_leaderboard=oos_leaderboard, output_dir=output_dir)


def main() -> None:
    args = parse_args()
    horizons = parse_horizons(args.horizons)
    feature_sets = [item.strip() for item in args.feature_sets.split(",") if item.strip()]
    model_names = _resolve_model_names(args)
    invalid_sets = [name for name in feature_sets if name not in FEATURE_SET_NAMES]
    if invalid_sets:
        raise SystemExit(f"Unknown feature set(s): {invalid_sets}. Valid: {FEATURE_SET_NAMES}")

    run_name = args.run_name or datetime.utcnow().strftime(f"{args.eval_mode}_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_name
    models_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading daily stock features...")
    stock_features = load_daily_features(args.dataset_root)
    context_features = load_market_context_features(args.dataset_root, stock_features=stock_features)
    if context_features.empty:
        raise SystemExit(
            "Market context features are missing. Run scripts/collect_massive_market_context.py first."
        )

    print("Building V1 supervised dataset...")
    dataset = build_v1_dataset(
        stock_features,
        context_features,
        horizons=horizons,
        window_length=args.window_length,
        benchmark_ticker=args.benchmark_ticker,
        max_episodes=args.max_episodes or None,
    )
    dataset.targets.to_csv(output_dir / "episode_targets.csv", index=False)
    generated_utc = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    dataset_manifest = {
        "generated_utc": generated_utc,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "horizons": list(horizons),
        "target_columns": dataset.target_columns,
        "window_length": args.window_length,
        "benchmark_ticker": args.benchmark_ticker.upper(),
        "feature_sets": {name: len(dataset.feature_columns[name]) for name in feature_sets},
        "models": model_names,
        "row_count": int(len(dataset.metadata)),
        "eval_mode": args.eval_mode,
        "notes": [
            "Targets are market-adjusted using the benchmark context table.",
            "Feature summaries are rolling-window last/mean/std values computed from dates <= anchor_date.",
            "Walk-forward mode evaluates on aggregated out-of-sample folds and excludes 1-day targets from leaderboard ranking.",
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
            "purge_gap": args.walk_forward_purge_gap or max(horizons),
            "final_stop_block_size": args.final_stop_block_size,
        }
    save_dataset_manifest(output_dir / "dataset_manifest.json", dataset_manifest)

    if args.eval_mode == "holdout":
        _train_holdout(
            args=args,
            dataset=dataset,
            stock_features=stock_features,
            feature_sets=feature_sets,
            model_names=model_names,
            horizons=horizons,
            output_dir=output_dir,
            models_dir=models_dir,
            generated_utc=generated_utc,
        )
    else:
        _train_walk_forward(
            args=args,
            dataset=dataset,
            stock_features=stock_features,
            feature_sets=feature_sets,
            model_names=model_names,
            horizons=horizons,
            output_dir=output_dir,
            models_dir=models_dir,
            generated_utc=generated_utc,
        )
    print(f"Training complete. Artifacts written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
