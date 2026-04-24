from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.v1_dataset import (  # noqa: E402
    DEFAULT_BENCHMARK_TICKER,
    DEFAULT_WINDOW_LENGTH,
    FEATURE_SET_NAMES,
    build_v1_dataset,
    chronological_split,
    load_daily_features,
    load_market_context_features,
    parse_horizons,
    prepare_xy,
    save_dataset_manifest,
    split_ranges,
)
from src.models.v1_baselines import (  # noqa: E402
    build_leaderboard,
    default_model_names,
    evaluate_predictions,
    make_model,
    prediction_frame,
    save_model_bundle,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train V1 supervised multi-output return baselines and write all model artifacts."
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
    parser.add_argument("--models", default=",".join(default_model_names()))
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Optional cap for smoke runs. Keeps the most recent N eligible episodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = parse_horizons(args.horizons)
    feature_sets = [item.strip() for item in args.feature_sets.split(",") if item.strip()]
    model_names = [item.strip() for item in args.models.split(",") if item.strip()]
    invalid_sets = [name for name in feature_sets if name not in FEATURE_SET_NAMES]
    if invalid_sets:
        raise SystemExit(f"Unknown feature set(s): {invalid_sets}. Valid: {FEATURE_SET_NAMES}")

    run_name = args.run_name or datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
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
    split = chronological_split(dataset.metadata, train_fraction=args.train_fraction, val_fraction=args.val_fraction)
    split_summary = split_ranges(dataset.metadata, split)

    dataset.metadata.assign(split=split).to_csv(output_dir / "episode_metadata.csv", index=False)
    dataset.targets.to_csv(output_dir / "episode_targets.csv", index=False)
    save_dataset_manifest(
        output_dir / "dataset_manifest.json",
        {
            "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "horizons": list(horizons),
            "target_columns": dataset.target_columns,
            "window_length": args.window_length,
            "benchmark_ticker": args.benchmark_ticker.upper(),
            "feature_sets": {name: len(dataset.feature_columns[name]) for name in feature_sets},
            "models": model_names,
            "split_summary": split_summary,
            "row_count": int(len(dataset.metadata)),
            "notes": [
                "Targets are market-adjusted using the benchmark context table.",
                "Feature summaries are rolling-window last/mean/std values computed from dates <= anchor_date.",
                "1-day target is reported but not primary for leaderboard ranking.",
            ],
        },
    )

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    trained_records: list[dict[str, object]] = []

    for feature_set in feature_sets:
        train_meta, x_train, y_train = prepare_xy(dataset, feature_set, split, "train")
        _, x_val, _ = prepare_xy(dataset, feature_set, split, "val")
        _, x_test, _ = prepare_xy(dataset, feature_set, split, "test")
        for model_name in model_names:
            print(f"Training {model_name} on {feature_set}...")
            model = make_model(model_name)
            model.fit(x_train, y_train)
            model_path = models_dir / f"{feature_set}__{model_name}.pkl"
            model_metadata = {
                "model_name": model_name,
                "feature_set": feature_set,
                "feature_columns": dataset.feature_columns[feature_set],
                "target_columns": dataset.target_columns,
                "horizons": list(horizons),
                "window_length": args.window_length,
                "benchmark_ticker": args.benchmark_ticker.upper(),
                "split_summary": split_summary,
            }
            save_model_bundle(model_path, model=model, metadata=model_metadata)
            trained_records.append({**model_metadata, "artifact_path": str(model_path.resolve())})

            for split_name, x_eval in (("train", x_train), ("val", x_val), ("test", x_test)):
                eval_meta, _, y_eval = prepare_xy(dataset, feature_set, split, split_name)
                pred = model.predict(x_eval)
                metrics_rows.extend(
                    evaluate_predictions(
                        eval_meta,
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
                            eval_meta,
                            pred,
                            target_columns=dataset.target_columns,
                            model_name=model_name,
                            feature_set=feature_set,
                        ).assign(split=split_name)
                    )

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    leaderboard = build_leaderboard(metrics)
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

    write_json(
        output_dir / "trained_models.json",
        {
            "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "models": trained_records,
        },
    )
    print(f"Training complete. Artifacts written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
