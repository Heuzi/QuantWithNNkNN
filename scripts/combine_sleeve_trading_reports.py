from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


COMBINED_CSVS = (
    "ranked_signals.csv",
    "entry_candidates.csv",
    "watchlist.csv",
    "model_agreement_summary.csv",
    "all_model_predictions.csv",
    "research_universe_diagnostics.csv",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine per-sleeve trading-strategy reports.")
    parser.add_argument(
        "--sleeve-report",
        action="append",
        required=True,
        help="Sleeve report mapping as sleeve_name=artifacts/production_reports/report_dir.",
    )
    parser.add_argument("--output-dir", required=True, help="Combined report directory.")
    return parser.parse_args()


def _parse_sleeve_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"--sleeve-report must be sleeve=path, got: {value}")
    sleeve, path = value.split("=", 1)
    sleeve = sleeve.strip()
    if not sleeve:
        raise SystemExit(f"Sleeve name is empty in --sleeve-report {value!r}")
    report_dir = Path(path.strip())
    if not report_dir.exists():
        raise SystemExit(f"Sleeve report directory does not exist: {report_dir}")
    return sleeve, report_dir


def _read_summary(sleeve: str, report_dir: Path) -> dict[str, object]:
    path = report_dir / "run_manifest.json"
    if not path.exists():
        path = report_dir / "summary.json"
    if not path.exists():
        raise SystemExit(f"Sleeve report is missing run_manifest.json/summary.json: {report_dir}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sleeve"] = sleeve
    payload["source_report_dir"] = str(report_dir)
    return payload


def _combine_csv(name: str, sleeve_reports: list[tuple[str, Path]], output_dir: Path) -> int:
    frames: list[pd.DataFrame] = []
    for sleeve, report_dir in sleeve_reports:
        path = report_dir / name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "sleeve", sleeve)
        frame.insert(1, "source_report_dir", str(report_dir))
        frames.append(frame)
    if not frames:
        return 0
    combined = pd.concat(frames, ignore_index=True)
    sort_cols = [col for col in ("suggested_action", "ensemble_percentile", "sleeve", "ticker") if col in combined.columns]
    if sort_cols:
        ascending = [True for _ in sort_cols]
        if "suggested_action" in sort_cols:
            order = {"ENTRY CANDIDATE": 0, "WATCHLIST": 1, "IGNORE": 2}
            combined["_action_order"] = combined["suggested_action"].map(order).fillna(99)
            sort_cols = ["_action_order", *[col for col in sort_cols if col != "suggested_action"]]
            ascending = [True for _ in sort_cols]
        combined = combined.sort_values(sort_cols, ascending=ascending, kind="mergesort").drop(
            columns=["_action_order"],
            errors="ignore",
        )
    combined.to_csv(output_dir / f"combined_{name}", index=False)
    return len(combined)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sleeve_reports = [_parse_sleeve_report(value) for value in args.sleeve_report]
    summaries = [_read_summary(sleeve, report_dir) for sleeve, report_dir in sleeve_reports]

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_dir / "sleeve_summary.csv", index=False)

    combined_counts = {
        name: _combine_csv(name, sleeve_reports, output_dir)
        for name in COMBINED_CSVS
    }
    generated_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    summary = {
        "generated_utc": generated_utc,
        "sleeve_count": len(sleeve_reports),
        "sleeves": [sleeve for sleeve, _ in sleeve_reports],
        "source_report_dirs": {sleeve: str(report_dir) for sleeve, report_dir in sleeve_reports},
        "combined_counts": combined_counts,
        "entry_candidate_count": int(summary_frame.get("entry_candidate_count", pd.Series(dtype=int)).sum()),
        "watchlist_count": int(summary_frame.get("watchlist_count", pd.Series(dtype=int)).sum()),
        "ranked_signal_rows": int(summary_frame.get("ranked_signal_rows", pd.Series(dtype=int)).sum()),
        "model_count": int(summary_frame.get("model_count", pd.Series(dtype=int)).sum()),
    }
    summary_text = json.dumps(summary, indent=2, sort_keys=True)
    (output_dir / "summary.json").write_text(summary_text, encoding="utf-8")
    (output_dir / "run_manifest.json").write_text(summary_text, encoding="utf-8")
    lines = [
        "# Combined Sleeve Trading Strategy Summary",
        "",
        f"- Generated UTC: `{generated_utc}`",
        f"- Sleeves: `{', '.join(summary['sleeves'])}`",
        f"- Total models scored: `{summary['model_count']}`",
        f"- Ranked signals: `{summary['ranked_signal_rows']}`",
        f"- Entry candidates: `{summary['entry_candidate_count']}`",
        f"- Watchlist rows: `{summary['watchlist_count']}`",
        "",
        "Sleeve rankings are computed independently, then combined for review. This file does not blend conservative and momentum/breakout model scores into one cross-sleeve ensemble.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote combined sleeve report to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
