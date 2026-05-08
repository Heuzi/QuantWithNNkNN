from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


DEFAULT_LATEST_INFERENCE_DIR = "data/eodhd_us_equities_30y/processed/latest_inference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor a trading strategy refresh/report run.")
    parser.add_argument("--pid", type=int, default=0, help="Optional process id for the running strategy process.")
    parser.add_argument("--latest-inference-dir", default=DEFAULT_LATEST_INFERENCE_DIR)
    parser.add_argument("--report-dir", default="", help="Optional artifacts/production_reports/<run> directory.")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--bar-width", type=int, default=30)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _progress_bar(current: int, total: int | None, width: int) -> str:
    if not total or total <= 0:
        return f"[{'?' * width}]"
    current = max(min(int(current), int(total)), 0)
    filled = int(round(width * current / total))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def _process_info(pid: int) -> dict[str, object]:
    if not pid:
        return {"alive": None}
    command = (
        f"$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; "
        "if ($p) { $p | Select-Object Id,CPU,WorkingSet64,StartTime | ConvertTo-Json -Compress }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        timeout=10,
    )
    text = result.stdout.strip()
    if not text:
        return {"alive": False, "pid": pid}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"alive": True, "pid": pid}
    return {
        "alive": True,
        "pid": int(payload.get("Id", pid)),
        "cpu": payload.get("CPU"),
        "mem_gb": round(float(payload.get("WorkingSet64") or 0) / 1024 / 1024 / 1024, 2),
        "start_time": payload.get("StartTime"),
    }


def _latest_file_summary(path: Path) -> str:
    if not path.exists():
        return "latest_inference=missing"
    names = [
        "recent_stock_bars.csv",
        "latest_daily_features.csv",
        "latest_market_context_features.csv",
        "prediction_windows.csv",
        "run_manifest.json",
    ]
    parts: list[str] = []
    for name in names:
        file_path = path / name
        if file_path.exists():
            size_mb = file_path.stat().st_size / 1024 / 1024
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%H:%M:%S")
            parts.append(f"{name}={size_mb:.1f}MB@{mtime}")
    return "; ".join(parts) if parts else "latest_inference=empty"


def _select_progress(latest_dir: Path, report_dir: Path | None) -> tuple[str, dict[str, object]]:
    candidates: list[tuple[str, dict[str, object]]] = []
    if report_dir is not None:
        candidates.append(("report", _read_json(report_dir / "progress.json")))
    candidates.append(("latest", _read_json(latest_dir / "progress.json")))
    for label, payload in candidates:
        if payload:
            return label, payload
    manifest = _read_json(latest_dir / "run_manifest.json")
    if manifest:
        return "latest", {
            "phase": "latest_inference_ready",
            "current": 1,
            "total": 1,
            "detail": f"local_data_end_date={manifest.get('local_data_end_date')}",
            "percent": 100.0,
        }
    return "inferred", {"phase": "unknown", "current": 0, "total": None, "detail": _latest_file_summary(latest_dir)}


def render_once(args: argparse.Namespace) -> bool:
    latest_dir = Path(args.latest_inference_dir)
    report_dir = Path(args.report_dir) if args.report_dir else None
    source, progress = _select_progress(latest_dir, report_dir)
    process = _process_info(args.pid)
    phase = str(progress.get("phase") or "unknown")
    current = int(progress.get("current") or 0)
    total = progress.get("total")
    total_int = int(total) if total is not None else None
    percent = progress.get("percent")
    percent_text = f"{float(percent):5.1f}%" if percent is not None else "     "
    proc_bits = []
    if process.get("alive") is True:
        proc_bits.append(f"pid={process.get('pid')}")
        proc_bits.append(f"cpu={process.get('cpu')}")
        proc_bits.append(f"mem_gb={process.get('mem_gb')}")
    elif process.get("alive") is False:
        proc_bits.append(f"pid={process.get('pid')} stopped")
    else:
        proc_bits.append("pid=not supplied")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{timestamp} | {_progress_bar(current, total_int, args.bar_width)} {percent_text} | "
        f"{phase} | source={source} | {' '.join(proc_bits)} | {progress.get('detail') or ''}",
        flush=True,
    )
    if source == "inferred":
        print(f"  {_latest_file_summary(latest_dir)}", flush=True)
    return bool(process.get("alive"))


def main() -> None:
    args = parse_args()
    while True:
        alive = render_once(args)
        if not args.watch:
            break
        if args.pid and not alive:
            break
        time.sleep(max(int(args.poll_seconds), 1))


if __name__ == "__main__":
    main()
