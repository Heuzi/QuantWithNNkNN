"""Run the balanced true-full ablation experiment sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


FEATURE_SETS = {
    "stock_only",
    "stock_relative",
    "stock_normalized_lean",
    "stock_normalized_lean_market_sector",
    "stock_only_fundamentals",
    "stock_only_sentiment",
    "stock_relative_market_sector_fundamentals",
    "stock_relative_market_sector_sentiment",
    "stock_normalized_lean_market_sector_fundamentals",
    "stock_normalized_lean_market_sector_sentiment",
    "stock_only_sequence",
    "stock_relative_sequence",
    "stock_sentiment_sequence",
    "stock_relative_market_sector_sequence",
    "stock_normalized_lean_sequence",
    "stock_normalized_lean_market_sector_sequence",
    "stock_normalized_lean_sentiment_sequence",
}
TRAIN_PROFILES = [
    "eodhd_true_full_ablation_xgboost",
    "eodhd_true_full_ablation_torch_mlp",
    "eodhd_true_full_ablation_torch_seq_static",
]
REQUIRED_ARTIFACTS = [
    "classification_oos_leaderboard.csv",
    "classification_metrics.csv",
    "classification_fold_metrics.csv",
    "trained_classification_models.json",
    "training_progress.jsonl",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-pid", type=int, default=None)
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--log-dir", default="logs/ablation_balanced")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _process_exists(pid: int) -> bool:
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}",
    ]
    return subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _wait_for_pid(pid: int) -> None:
    print(json.dumps({"step": "wait_for_pid_start", "pid": pid, "generated_utc": _utc_now()}), flush=True)
    while _process_exists(pid):
        time.sleep(30)
        print(json.dumps({"step": "wait_for_pid_progress", "pid": pid, "generated_utc": _utc_now()}), flush=True)
    print(json.dumps({"step": "wait_for_pid_done", "pid": pid, "generated_utc": _utc_now()}), flush=True)


def _run_stage(name: str, command: list[str], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.out.log"
    stderr_path = log_dir / f"{name}.err.log"
    print(
        json.dumps(
            {
                "step": "stage_start",
                "stage": name,
                "command": command,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "generated_utc": _utc_now(),
            }
        ),
        flush=True,
    )
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr)
    print(
        json.dumps(
            {
                "step": "stage_done",
                "stage": name,
                "returncode": result.returncode,
                "generated_utc": _utc_now(),
            }
        ),
        flush=True,
    )
    if result.returncode:
        raise SystemExit(f"Stage {name} failed with return code {result.returncode}; see {stderr_path}")


def _validate_cache() -> None:
    manifest_path = Path(
        "data/eodhd_training_panels/eodhd_true_full_walk_forward/episode_cache_ablation_balanced/manifest.json"
    )
    if not manifest_path.exists():
        raise SystemExit(f"Ablation cache manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = FEATURE_SETS.difference(manifest.get("feature_sets", []))
    if missing:
        raise SystemExit(f"Ablation cache manifest is missing feature sets: {sorted(missing)}")
    checks = {
        "classification_event_type": "path_5pct_20d",
        "window_length": 60,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise SystemExit(f"Unexpected cache manifest {key}: {manifest.get(key)!r}; expected {expected!r}")
    print(
        json.dumps(
            {
                "step": "cache_validation_ok",
                "episode_count": manifest.get("episode_count"),
                "ticker_count": manifest.get("ticker_count"),
                "feature_set_count": len(manifest.get("feature_sets", [])),
                "generated_utc": _utc_now(),
            }
        ),
        flush=True,
    )


def _validate_training_artifacts(profile: str) -> None:
    run_dir = Path("artifacts/v1_baselines") / profile
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).exists()]
    if missing:
        raise SystemExit(f"Training artifacts missing for {profile}: {missing}")
    print(json.dumps({"step": "training_artifacts_ok", "profile": profile, "generated_utc": _utc_now()}), flush=True)


def main() -> None:
    args = _parse_args()
    log_dir = Path(args.log_dir)
    if args.wait_for_pid is not None:
        _wait_for_pid(args.wait_for_pid)
    if not args.skip_cache:
        _run_stage(
            "materialize_cache",
            [
                sys.executable,
                "scripts/run_v1_pipeline.py",
                "--profile",
                "eodhd_true_full_ablation_balanced_cache",
                "--stage",
                "materialize_cache",
            ],
            log_dir,
        )
    _validate_cache()
    for profile in TRAIN_PROFILES:
        _run_stage(
            f"train_{profile.removeprefix('eodhd_true_full_ablation_')}",
            [sys.executable, "scripts/run_v1_pipeline.py", "--profile", profile, "--stage", "train"],
            log_dir,
        )
        _validate_training_artifacts(profile)
    _run_stage(
        "summarize",
        [sys.executable, "scripts/summarize_v1_ablation_balanced.py"],
        log_dir,
    )
    print(json.dumps({"step": "ablation_run_complete", "generated_utc": _utc_now()}), flush=True)


if __name__ == "__main__":
    main()
