"""Build a combined leaderboard for the balanced true-full ablation study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


NEW_RUNS = [
    "eodhd_true_full_ablation_xgboost",
    "eodhd_true_full_ablation_torch_mlp",
    "eodhd_true_full_ablation_torch_seq_static",
]
REFERENCE_RUNS = [
    "eodhd_true_full_xgboost",
    "eodhd_true_full_torch_mlp",
    "eodhd_true_full_torch_seq_static",
]
SORT_COLUMNS = [
    "selection_score",
    "mean_pr_auc",
    "mean_roc_auc",
    "mean_top_decile_precision",
    "mean_top_bottom_spread",
]
SUMMARY_COLUMNS = [
    "overall_rank",
    "model_family_rank",
    "experiment_group",
    "model_name",
    "feature_set",
    "selection_score",
    "mean_pr_auc",
    "mean_roc_auc",
    "mean_top_decile_precision",
    "mean_accuracy",
    "mean_macro_f1",
    "mean_multiclass_log_loss",
    "positive_base_rate",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", default="artifacts/v1_baselines")
    parser.add_argument(
        "--output-dir",
        default="artifacts/v1_baselines/eodhd_true_full_ablation_balanced_summary",
    )
    parser.add_argument("--new-run", action="append", default=None)
    parser.add_argument("--reference-run", action="append", default=None)
    return parser.parse_args()


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required ablation summary input is missing: {path}")
    return pd.read_csv(path)


def _load_run(artifacts_root: Path, run_name: str, experiment_group: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = artifacts_root / run_name
    leaderboard = _read_required_csv(run_dir / "classification_oos_leaderboard.csv")
    metrics = _read_required_csv(run_dir / "classification_metrics.csv")

    leaderboard = leaderboard.copy()
    metrics = metrics.copy()
    for frame in (leaderboard, metrics):
        frame.insert(0, "source_run_name", run_name)
        frame.insert(0, "source_run_dir", str(run_dir))
        frame.insert(0, "experiment_group", experiment_group)

    oos_metrics = metrics[metrics["split"].astype(str).str.lower().eq("oos")].copy()
    metric_columns = [
        column
        for column in [
            "model_name",
            "feature_set",
            "macro_f1",
            "multiclass_log_loss",
            "positive_rate",
            "row_count",
            "date_count",
            "class_0_rate",
            "class_1_rate",
            "class_2_rate",
        ]
        if column in oos_metrics.columns
    ]
    metric_aliases = {
        "macro_f1": "mean_macro_f1",
        "multiclass_log_loss": "mean_multiclass_log_loss",
        "positive_rate": "positive_base_rate",
        "row_count": "oos_row_count",
        "date_count": "oos_date_count",
    }
    if {"model_name", "feature_set"}.issubset(metric_columns):
        oos_for_join = oos_metrics[metric_columns].rename(columns=metric_aliases)
        leaderboard = leaderboard.merge(oos_for_join, on=["model_name", "feature_set"], how="left")

    if "positive_base_rate" not in leaderboard.columns and "mean_positive_rate" in leaderboard.columns:
        leaderboard["positive_base_rate"] = leaderboard["mean_positive_rate"]
    return leaderboard, metrics


def _rank_leaderboard(leaderboard: pd.DataFrame) -> pd.DataFrame:
    ranked = leaderboard.copy()
    for column in SORT_COLUMNS:
        if column in ranked.columns:
            ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    ranked = ranked.sort_values(
        [column for column in SORT_COLUMNS if column in ranked.columns],
        ascending=[False for column in SORT_COLUMNS if column in ranked.columns],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "overall_rank", range(1, len(ranked) + 1))
    ranked["model_family_rank"] = ranked.groupby("model_name", sort=False).cumcount() + 1
    return ranked


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    present = [column for column in columns if column in frame.columns]
    lines = [
        "| " + " | ".join(present) + " |",
        "| " + " | ".join("---" for _ in present) + " |",
    ]
    for _, row in frame[present].iterrows():
        lines.append("| " + " | ".join(_format_value(row[column]) for column in present) + " |")
    return "\n".join(lines)


def _write_summary(output_dir: Path, leaderboard: pd.DataFrame, metrics: pd.DataFrame) -> None:
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    new_rows = int((leaderboard["experiment_group"] == "new_ablation").sum())
    ref_rows = int((leaderboard["experiment_group"] == "existing_reference").sum())
    overall_winner = leaderboard.iloc[[0]]
    family_winners = leaderboard[leaderboard["model_family_rank"].eq(1)].sort_values("model_name")
    top_rows = leaderboard.head(10)

    lines = [
        "# Balanced True-Full Ablation Summary",
        "",
        f"Generated UTC: {generated}",
        "",
        "Selection score: mean_pr_auc + mean_top_decile_precision + mean_top_bottom_spread.",
        "",
        f"New ablation rows: {new_rows}",
        f"Existing reference rows: {ref_rows}",
        f"Combined metrics rows: {len(metrics)}",
        "",
        "## Overall Winner",
        "",
        _markdown_table(overall_winner, SUMMARY_COLUMNS),
        "",
        "## Best By Model Family",
        "",
        _markdown_table(family_winners, SUMMARY_COLUMNS),
        "",
        "## Top 10 Overall",
        "",
        _markdown_table(top_rows, SUMMARY_COLUMNS),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    artifacts_root = Path(args.artifacts_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    new_runs = args.new_run or NEW_RUNS
    reference_runs = args.reference_run or REFERENCE_RUNS
    leaderboard_parts: list[pd.DataFrame] = []
    metric_parts: list[pd.DataFrame] = []
    for run_name in new_runs:
        leaderboard, metrics = _load_run(artifacts_root, run_name, "new_ablation")
        leaderboard_parts.append(leaderboard)
        metric_parts.append(metrics)
    for run_name in reference_runs:
        leaderboard, metrics = _load_run(artifacts_root, run_name, "existing_reference")
        leaderboard_parts.append(leaderboard)
        metric_parts.append(metrics)

    combined_leaderboard = _rank_leaderboard(pd.concat(leaderboard_parts, ignore_index=True))
    combined_metrics = pd.concat(metric_parts, ignore_index=True)
    combined_leaderboard.to_csv(output_dir / "combined_classification_oos_leaderboard.csv", index=False)
    combined_metrics.to_csv(output_dir / "combined_classification_oos_metrics.csv", index=False)
    _write_summary(output_dir, combined_leaderboard, combined_metrics)
    print(f"Wrote ablation summary to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
