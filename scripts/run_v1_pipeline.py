from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_DIR = REPO_ROOT / "configs" / "v1_runs"
STAGE_ORDER = ("build_features", "materialize_panel", "materialize_cache", "train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standardized V1 data-prep and train/test profiles."
    )
    parser.add_argument(
        "--profile",
        default="eodhd_full_walk_forward",
        help=(
            "Profile name under configs/v1_runs without .json, or a path to a JSON profile."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("all", *STAGE_ORDER),
        default="all",
        help="Run every stage in the profile or one stage only.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def _profile_path(profile: str) -> Path:
    candidate = Path(profile)
    if candidate.exists():
        return candidate
    if candidate.suffix != ".json":
        candidate = DEFAULT_PROFILE_DIR / f"{profile}.json"
    return candidate


def _load_profile(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Profile not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    if not isinstance(profile, dict):
        raise SystemExit(f"Profile must be a JSON object: {path}")
    return profile


def _arg_name(name: str) -> str:
    return "--" + name.replace("_", "-")


def _append_option(command: list[str], key: str, value: Any) -> None:
    # Profiles use JSON-friendly snake_case keys. CLIs use kebab-case flags.
    # Keep this conversion centralized so future agents add parameters in JSON,
    # not by constructing long one-off shell commands in conversation.
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(_arg_name(key))
        return
    if isinstance(value, (list, tuple)):
        value = ",".join(str(item) for item in value)
    command.extend([_arg_name(key), str(value)])


def _build_features_command(profile: dict[str, Any]) -> list[str]:
    config = dict(profile.get("build_features") or {})
    dataset_root = profile.get("dataset_root")
    if dataset_root and "dataset_root" not in config:
        config["dataset_root"] = dataset_root
    command = [sys.executable, "scripts/build_eodhd_daily_features_chunked.py"]
    for key, value in config.items():
        _append_option(command, key, value)
    return command


def _materialize_panel_command(profile: dict[str, Any]) -> list[str]:
    config = dict(profile.get("materialize_panel") or {})
    dataset_root = profile.get("dataset_root")
    if dataset_root and "source_dataset_root" not in config:
        config["source_dataset_root"] = dataset_root
    if "output_dataset_root" not in config:
        raise SystemExit("Profiles with materialize_panel must set materialize_panel.output_dataset_root.")
    command = [sys.executable, "scripts/materialize_v1_training_panel.py"]
    for key, value in config.items():
        _append_option(command, key, value)
    return command


def _episode_cache_dir(profile: dict[str, Any]) -> str | None:
    cache = profile.get("materialize_cache") or {}
    if cache.get("cache_dir"):
        return str(cache["cache_dir"])
    return profile.get("episode_cache_dir")


def _materialize_cache_command(profile: dict[str, Any]) -> list[str]:
    config = dict(profile.get("materialize_cache") or {})
    dataset_root = _training_dataset_root(profile)
    if dataset_root and "dataset_root" not in config:
        config["dataset_root"] = dataset_root
    if "cache_dir" not in config:
        raise SystemExit("Profiles with materialize_cache must set materialize_cache.cache_dir.")
    train_config = profile.get("train") or {}
    # The cache must match the train command exactly for feature set, target, and
    # eligibility semantics. Pull shared values from train unless the cache stage
    # explicitly overrides them.
    for key in (
        "feature_sets",
        "horizons",
        "window_length",
        "benchmark_ticker",
        "max_episodes",
        "classification_horizon",
        "classification_threshold",
        "disable_episode_eligibility_filter",
        "eligibility_min_history_days",
        "eligibility_valid_ohlcv_lookback",
        "eligibility_min_valid_ohlcv_days",
        "eligibility_dollar_volume_lookback",
        "eligibility_min_avg_dollar_volume",
        "eligibility_min_price",
        "eligibility_allowed_exchanges",
    ):
        if key in train_config and key not in config:
            config[key] = train_config[key]
    command = [sys.executable, "scripts/materialize_v1_episode_cache.py"]
    for key, value in config.items():
        _append_option(command, key, value)
    return command


def _training_dataset_root(profile: dict[str, Any]) -> str | None:
    # If a profile materializes a bounded panel, training should read that output
    # root. This is the memory-control mechanism for full EODHD experiments:
    # the raw/full processed root stays intact, while train/test consumes the
    # smaller panel described by the profile.
    if profile.get("training_dataset_root"):
        return str(profile["training_dataset_root"])
    materialize = profile.get("materialize_panel") or {}
    if materialize.get("output_dataset_root"):
        return str(materialize["output_dataset_root"])
    if profile.get("dataset_root"):
        return str(profile["dataset_root"])
    return None


def _train_command(profile: dict[str, Any]) -> list[str]:
    config = dict(profile.get("train") or {})
    dataset_root = _training_dataset_root(profile)
    if dataset_root and "dataset_root" not in config:
        config["dataset_root"] = dataset_root
    cache_dir = _episode_cache_dir(profile)
    if cache_dir and "episode_cache_dir" not in config:
        config["episode_cache_dir"] = cache_dir
    command = [sys.executable, "scripts/train_v1_supervised_baselines.py"]
    for key, value in config.items():
        _append_option(command, key, value)
    return command


def _stage_names(profile: dict[str, Any], requested: str) -> list[str]:
    if requested != "all":
        return [requested]
    stages = profile.get("stages") or STAGE_ORDER
    return [str(stage) for stage in stages]


def _write_run_manifest(
    *,
    profile_path: Path,
    profile: dict[str, Any],
    stage_commands: list[dict[str, Any]],
) -> Path:
    log_dir = REPO_ROOT / "logs" / "v1_pipeline"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    profile_name = str(profile.get("name") or profile_path.stem)
    manifest_path = log_dir / f"{profile_name}_{timestamp}.json"
    manifest = {
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "profile_path": str(profile_path.resolve()),
        "profile_name": profile_name,
        "dataset_root": profile.get("dataset_root"),
        "stage_commands": stage_commands,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def main() -> None:
    args = parse_args()
    profile_path = _profile_path(args.profile)
    profile = _load_profile(profile_path)
    command_builders = {
        "build_features": _build_features_command,
        "materialize_panel": _materialize_panel_command,
        "materialize_cache": _materialize_cache_command,
        "train": _train_command,
    }
    stage_commands: list[dict[str, Any]] = []
    for stage in _stage_names(profile, args.stage):
        if stage not in command_builders:
            raise SystemExit(f"Unknown stage in profile: {stage}")
        command = command_builders[stage](profile)
        stage_commands.append({"stage": stage, "command": command})
        print(json.dumps({"stage": stage, "command": command}, indent=2), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    if args.dry_run:
        return
    manifest_path = _write_run_manifest(
        profile_path=profile_path,
        profile=profile,
        stage_commands=stage_commands,
    )
    print(f"Wrote pipeline run manifest to {manifest_path}")


if __name__ == "__main__":
    main()
