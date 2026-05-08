from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor a staged V1 training run.")
    parser.add_argument("--log", required=True, help="Stdout log from scripts/run_v1_pipeline.py.")
    parser.add_argument("--err", default="", help="Optional stderr log.")
    parser.add_argument("--run-dir", default="", help="Optional artifact run directory.")
    parser.add_argument("--pid", type=int, default=0, help="Optional active worker PID.")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--watch", action="store_true")
    return parser.parse_args()


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if not data:
        return []
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff") or data[:200].count(b"\x00") > 20:
        return data.decode("utf-16", errors="replace").splitlines()
    return data.decode("utf-8", errors="replace").splitlines()


def _json_events(lines: list[str]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "step" in value:
            events.append(value)
    return events


def _fold_ids(events: list[dict[str, object]]) -> list[int]:
    ids: list[int] = []
    for event in events:
        if event.get("step") == "walk_forward_fold_start":
            try:
                ids.append(int(event["fold_id"]))
            except (KeyError, TypeError, ValueError):
                pass
    return ids


def _latest_epoch(events: list[dict[str, object]]) -> dict[str, object] | None:
    epoch_events = [
        event
        for event in events
        if str(event.get("step") or "").endswith("_epoch")
    ]
    return epoch_events[-1] if epoch_events else None


def _latest_batch(events: list[dict[str, object]]) -> dict[str, object] | None:
    batch_events = [
        event
        for event in events
        if str(event.get("step") or "").endswith("_batch")
    ]
    return batch_events[-1] if batch_events else None


def _latest_chunk_train(events: list[dict[str, object]]) -> dict[str, object] | None:
    chunks = [
        event
        for event in events
        if str(event.get("step") or "").endswith("_chunk_train")
        or str(event.get("step") or "") == "xgboost_classifier_chunk_train"
    ]
    return chunks[-1] if chunks else None


def _latest_standardizer_scan(events: list[dict[str, object]]) -> dict[str, object] | None:
    scans = [
        event
        for event in events
        if event.get("step") == "sequence_standardizer_scan"
        or str(event.get("step") or "").endswith("_scaler_scan")
        or str(event.get("step") or "").endswith("_data_scan")
    ]
    return scans[-1] if scans else None


def _phase_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
    start = 0
    for index, event in enumerate(events):
        if event.get("step") in {"walk_forward_fold_start", "final_deploy_fit_start"}:
            start = index
    return events[start:]


def _epoch_progress(events: list[dict[str, object]], max_epochs: int) -> tuple[float, str]:
    latest_epoch = _latest_epoch(events)
    completed_epochs = int(latest_epoch.get("epoch", 0)) if latest_epoch else 0
    latest_batch = _latest_batch(events)
    if not latest_batch:
        chunk = _latest_chunk_train(events)
        if chunk:
            try:
                chunk_index = int(chunk.get("chunk", 0))
                chunks_total = max(int(chunk.get("chunks_total", 0)), 1)
                rounds_done = int(chunk.get("rounds_done", 0))
                rounds_total = max(int(chunk.get("rounds_total", 0)), 1)
            except (TypeError, ValueError):
                return float(completed_epochs), f"epoch {completed_epochs}/{max_epochs}"
            fraction = min(rounds_done / rounds_total, 1.0)
            progress = max(float(completed_epochs), fraction * max_epochs)
            return (
                progress,
                f"chunk training, chunk {chunk_index}/{chunks_total}, rounds {rounds_done}/{rounds_total}",
            )
        scan = _latest_standardizer_scan(events)
        if scan:
            try:
                rows_seen = int(scan.get("rows_seen", 0))
                rows_total = int(scan.get("source_rows_total", scan.get("rows_total", 0)))
                batch = int(scan.get("batch", 0))
            except (TypeError, ValueError):
                return float(completed_epochs), f"epoch {completed_epochs}/{max_epochs}"
            label = "standardizing/scanning data"
            if str(scan.get("step") or "").endswith("_scaler_scan"):
                label = "fitting scaler"
            elif str(scan.get("step") or "").endswith("_data_scan"):
                label = "streaming data"
            return (
                float(completed_epochs),
                f"{label}, scan batch {batch}, rows {rows_seen}/{rows_total}",
            )
        return float(completed_epochs), f"epoch {completed_epochs}/{max_epochs}"
    try:
        epoch = max(int(latest_batch.get("epoch", 1)), 1)
        batch = max(int(latest_batch.get("batch", 0)), 0)
        total_batches = max(int(latest_batch.get("total_batches", 0)), 1)
    except (TypeError, ValueError):
        return float(completed_epochs), f"epoch {completed_epochs}/{max_epochs}"
    fraction = min(batch / total_batches, 1.0)
    progress = max(float(completed_epochs), (epoch - 1) + fraction)
    return progress, f"epoch {epoch}/{max_epochs}, batch {batch}/{total_batches}"


def _progress(events: list[dict[str, object]], max_epochs: int) -> tuple[float, str]:
    fold_ids = _fold_ids(events)
    current_fold_index = max(len(fold_ids) - 1, 0)
    saved = any(event.get("step") == "final_deploy_model_saved" for event in events)
    if saved:
        return 1.0, "complete"
    if not fold_ids:
        return 0.0, "loading cache"
    current_events = _phase_events(events)
    epoch_units, epoch_phase = _epoch_progress(current_events, max_epochs)
    # Six walk-forward folds plus one final deployment fit. Treat the final fit
    # as one more epoch-budgeted unit for a stable, conservative progress bar.
    total_units = 7 * max_epochs
    completed_units = current_fold_index * max_epochs + epoch_units
    fold_id = fold_ids[-1]
    phase = f"fold {current_fold_index + 1}/6 id={fold_id}, {epoch_phase}"
    if any(event.get("step") == "final_deploy_fit_start" for event in events):
        completed_units = 6 * max_epochs + epoch_units
        phase = f"final deploy fit, {epoch_phase}"
    return min(completed_units / total_units, 0.999), phase


def _bar(progress: float, width: int = 30) -> str:
    filled = int(round(progress * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {progress * 100:6.2f}%"


def _process_snapshot(pid: int) -> str:
    if not pid:
        return ""
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | "
            "Select-Object Id,CPU,WorkingSet64,StartTime | ConvertTo-Json -Compress"
        ),
    ]
    try:
        out = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "process: unavailable"
    if not out:
        return "process: not running"
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        return f"process: {out}"
    gb = float(payload.get("WorkingSet64", 0)) / (1024**3)
    return f"pid={payload.get('Id')} cpu={payload.get('CPU')} mem_gb={gb:.2f}"


def _status(args: argparse.Namespace) -> str:
    log_path = Path(args.log)
    err_path = Path(args.err) if args.err else None
    lines = _read_lines(log_path)
    events = _json_events(lines)
    progress, phase = _progress(events, args.max_epochs)
    parts = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        _bar(progress),
        phase,
    ]
    process = _process_snapshot(args.pid)
    if process:
        parts.append(process)
    if err_path and err_path.exists() and err_path.stat().st_size:
        parts.append(f"stderr_bytes={err_path.stat().st_size}")
    return " | ".join(parts)


def main() -> None:
    args = parse_args()
    if args.watch:
        while True:
            print(_status(args), flush=True)
            time.sleep(max(args.poll_seconds, 1))
    print(_status(args))


if __name__ == "__main__":
    main()
