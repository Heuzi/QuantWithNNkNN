from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.v1_dataset import (  # noqa: E402
    STATIC_CATEGORICAL_COLUMNS,
    build_latest_v1_feature_sets,
    build_sequence_feature_store,
    encode_static_categories,
    load_daily_features,
    load_market_context_features,
)
from src.data.episode_eligibility import EpisodeEligibilityConfig  # noqa: E402
from src.models.v1_baselines import load_model_bundle, prediction_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all trained V1 baseline models from a run directory on latest prediction windows."
    )
    parser.add_argument("--run-dir", required=True, help="Training run directory containing final_models.json.")
    parser.add_argument(
        "--dataset-root",
        default="data/eodhd_us_equities_30y",
        help="Dataset folder containing latest stock and context features.",
    )
    parser.add_argument("--anchor-date", default="", help="Optional prediction cutoff date.")
    parser.add_argument("--output-file", default="", help="Optional output CSV path.")
    parser.add_argument(
        "--benchmark-return-assumption",
        default="0",
        help=(
            "Assumed benchmark return used to translate regression market-adjusted "
            "returns into implied future closes. Use one value for all horizons, "
            "or comma-separated values matching a regression model's target horizons."
        ),
    )
    return parser.parse_args()


def _aligned_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    columns = {
        col: frame[col] if col in frame.columns else pd.Series(0.0, index=frame.index)
        for col in feature_columns
    }
    return pd.DataFrame(columns, index=frame.index)


def _resolve_model_index_path(run_dir: Path) -> Path:
    preferred = run_dir / "final_models.json"
    fallback = run_dir / "trained_models.json"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise SystemExit(f"Missing model index. Checked: {preferred} and {fallback}")


def _resolve_artifact_path(run_dir: Path, artifact_path: str) -> Path:
    recorded = Path(artifact_path)
    candidates: list[Path]
    if recorded.is_absolute():
        candidates = [recorded, run_dir / "models" / recorded.name]
    else:
        candidates = [run_dir / recorded, recorded]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_text = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Model artifact not found. Tried: {candidate_text}")


def _parse_benchmark_return_assumption(value: str, target_columns: list[str]) -> dict[str, float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        parts = ["0"]
    returns = [float(part) for part in parts]
    if len(returns) == 1:
        returns = returns * len(target_columns)
    if len(returns) != len(target_columns):
        raise SystemExit(
            "--benchmark-return-assumption must be one value or match the number of regression target horizons."
        )
    return dict(zip(target_columns, returns))


def _add_anchor_close(metadata: pd.DataFrame, stock_features: pd.DataFrame) -> pd.DataFrame:
    close_lookup = stock_features[["ticker", "date", "close"]].copy()
    close_lookup["ticker"] = close_lookup["ticker"].astype(str).str.upper()
    close_lookup = close_lookup.rename(columns={"date": "anchor_date", "close": "anchor_close"})
    out = metadata.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out = out.merge(close_lookup, on=["ticker", "anchor_date"], how="left")
    return out


def _add_implied_close_columns(
    predictions: pd.DataFrame,
    *,
    target_columns: list[str],
    benchmark_return_by_target: dict[str, float],
) -> pd.DataFrame:
    out = predictions.copy()
    if "anchor_close" not in out.columns:
        return out
    for target in target_columns:
        pred_col = target.replace("market_adjusted_return", "pred_market_adjusted_return")
        close_col = target.replace("market_adjusted_return", "pred_close_if_benchmark_assumption")
        if pred_col not in out.columns:
            continue
        benchmark_return = benchmark_return_by_target[target]
        out[close_col] = out["anchor_close"] * (1.0 + out[pred_col] + benchmark_return)
    return out


def _leaderboard_for_task(run_dir: Path, task_type: str) -> pd.DataFrame:
    path = run_dir / ("leaderboard.csv" if task_type == "regression" else "classification_leaderboard.csv")
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _episode_eligibility_config_from_run(run_dir: Path) -> EpisodeEligibilityConfig | None:
    manifest_path = run_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = manifest.get("episode_eligibility")
    if not isinstance(payload, dict):
        return None
    return EpisodeEligibilityConfig.from_mapping(payload)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    model_index_path = _resolve_model_index_path(run_dir)
    model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
    if not model_index.get("models"):
        raise SystemExit(f"No models found in {model_index_path}")

    stock_features = load_daily_features(args.dataset_root)
    context_features = load_market_context_features(args.dataset_root, stock_features=stock_features)
    if context_features.empty:
        raise SystemExit("Market context features are missing. Run scripts/update_eodhd_daily_dataset.py first.")
    eligibility_config = _episode_eligibility_config_from_run(run_dir)

    regression_records = [record for record in model_index["models"] if record.get("task_type", "regression") == "regression"]
    first_model = model_index["models"][0]
    metadata, feature_sets, _ = build_latest_v1_feature_sets(
        stock_features,
        context_features,
        window_length=int(first_model["window_length"]),
        benchmark_ticker=str(first_model["benchmark_ticker"]),
        anchor_date=args.anchor_date or None,
        eligibility_config=eligibility_config,
    )
    metadata = _add_anchor_close(metadata, stock_features)
    sequence_stores: dict[tuple[str, tuple[str, ...]], object] = {}
    leaderboards = {
        "regression": _leaderboard_for_task(run_dir, "regression"),
        "classification": _leaderboard_for_task(run_dir, "classification"),
    }
    prediction_frames: list[pd.DataFrame] = []

    for record in model_index["models"]:
        task_type = str(record.get("task_type") or "regression")
        feature_set = record["feature_set"]
        model_name = record["model_name"]
        bundle = load_model_bundle(_resolve_artifact_path(run_dir, str(record["artifact_path"])))
        model = bundle["model"]
        bundle_metadata = bundle.get("metadata", {})
        input_layout = str(record.get("input_layout") or bundle_metadata.get("input_layout") or "tabular")
        if input_layout == "sequence_static":
            sequence_feature_columns = list(
                record.get("sequence_feature_columns")
                or bundle_metadata.get("sequence_feature_columns")
                or []
            )
            static_columns = list(
                record.get("static_categorical_columns")
                or bundle_metadata.get("static_categorical_columns")
                or STATIC_CATEGORICAL_COLUMNS
            )
            static_vocabularies = dict(
                record.get("static_vocabularies")
                or bundle_metadata.get("static_vocabularies")
                or {}
            )
            benchmark_ticker = str(
                record.get("benchmark_ticker") or bundle_metadata.get("benchmark_ticker") or first_model["benchmark_ticker"]
            )
            cache_key = (feature_set, tuple(sequence_feature_columns))
            if cache_key not in sequence_stores:
                sequence_stores[cache_key] = build_sequence_feature_store(
                    stock_features,
                    feature_set,
                    context_features=context_features,
                    benchmark_ticker=benchmark_ticker,
                    feature_columns=sequence_feature_columns,
                )
            x = {
                "store": sequence_stores[cache_key],
                "metadata": metadata.reset_index(drop=True),
                "static_categorical": encode_static_categories(
                    metadata.reset_index(drop=True),
                    static_vocabularies,
                    columns=static_columns,
                ),
            }
        else:
            feature_columns = list(record.get("feature_columns") or bundle_metadata.get("feature_columns") or [])
            x = _aligned_features(feature_sets[feature_set], feature_columns)
        pred = model.predict(x)
        leaderboard = leaderboards.get(task_type, pd.DataFrame())
        rank = None
        recommended = False
        if not leaderboard.empty:
            match = leaderboard[
                (leaderboard["model_name"] == model_name) & (leaderboard["feature_set"] == feature_set)
            ]
            if not match.empty:
                rank = int(match.iloc[0]["leaderboard_rank"])
                recommended = bool(match.iloc[0]["recommended"])
        prediction_frames.append(
            prediction_frame(
                metadata,
                pred,
                target_columns=list(record["target_columns"]),
                model_name=model_name,
                feature_set=feature_set,
                leaderboard_rank=rank,
                recommended=recommended,
                task_type=task_type,
            )
        )

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    if not predictions.empty and regression_records:
        regression_targets = list(regression_records[0]["target_columns"])
        benchmark_return_by_target = _parse_benchmark_return_assumption(
            args.benchmark_return_assumption,
            regression_targets,
        )
        regression_mask = predictions["task_type"] == "regression"
        regression_predictions = _add_implied_close_columns(
            predictions.loc[regression_mask].copy(),
            target_columns=regression_targets,
            benchmark_return_by_target=benchmark_return_by_target,
        )
        non_regression_predictions = predictions.loc[~regression_mask].copy()
        predictions = pd.concat([regression_predictions, non_regression_predictions], ignore_index=True)
    output_file = Path(args.output_file) if args.output_file else run_dir / "latest_predictions.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_file, index=False)
    print(f"Wrote {len(predictions)} prediction rows to {output_file.resolve()}")


if __name__ == "__main__":
    main()
