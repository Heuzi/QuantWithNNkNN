from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.eodhd_enrichment import fundamental_payload_path  # noqa: E402
from src.data.eodhd_stage1 import (  # noqa: E402
    EODHDAPIError,
    EODHDAuthError,
    EODHDRateLimitError,
    EODHDRESTClient,
    RateLimiter,
    load_eodhd_credentials,
)
from src.data.massive_stage1 import append_csv, write_json  # noqa: E402


STATUS_HEADERS = [
    "role",
    "eodhd_symbol",
    "ticker",
    "status",
    "row_count",
    "start_date",
    "end_date",
    "error",
    "finished_utc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch EODHD fundamentals using the cheaper bulk-fundamentals endpoint."
    )
    parser.add_argument("--dataset-root", default="data/eodhd_us_equities_30y")
    parser.add_argument("--credentials-path", default="EODHD_api_key")
    parser.add_argument("--exchange", default="NASDAQ", help="Required by endpoint; ignored when symbols= is used.")
    parser.add_argument("--symbols", default="", help="Optional comma-separated EODHD symbols, e.g. AAPL.US,MSFT.US.")
    parser.add_argument("--max-symbols", type=int, default=0, help="Optional cap for testing. 0 means no cap.")
    parser.add_argument("--batch-size", type=int, default=500, help="Bulk symbols per request. EODHD limit is 500.")
    parser.add_argument("--version", default="1.2")
    parser.add_argument("--force-refetch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rate-limit-calls", type=int, default=200)
    parser.add_argument("--rate-limit-period-seconds", type=float, default=60.0)
    return parser.parse_args()


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _ticker_from_symbol(symbol: str) -> str:
    return symbol.upper().split(".", 1)[0]


def _completed_symbols(status_path: Path, raw_dir: Path) -> set[str]:
    completed: set[str] = set()
    for row in _load_csv_rows(status_path):
        if row.get("role") == "fundamentals" and row.get("status") == "ok" and row.get("eodhd_symbol"):
            completed.add(str(row["eodhd_symbol"]).upper())
    for path in raw_dir.glob("*.json"):
        completed.add(path.stem.replace("__", ".").upper())
    return completed


def _candidate_symbols(args: argparse.Namespace, universe_path: Path, raw_dir: Path, status_path: Path) -> list[str]:
    if args.symbols.strip():
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    else:
        rows = _load_csv_rows(universe_path)
        symbols = [str(row.get("eodhd_symbol") or "").upper() for row in rows if row.get("eodhd_symbol")]
    if not args.force_refetch:
        completed = _completed_symbols(status_path, raw_dir)
        symbols = [symbol for symbol in symbols if symbol not in completed]
    symbols = list(dict.fromkeys(symbols))
    if args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]
    return symbols


def _iter_batches(items: Sequence[str], size: int) -> list[list[str]]:
    size = max(1, min(int(size), 500))
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


def _payload_items(payload: object) -> list[tuple[str, Mapping[str, object]]]:
    items: list[tuple[str, Mapping[str, object]]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if isinstance(value, Mapping):
                symbol = str(key).upper()
                general = value.get("General")
                if isinstance(general, Mapping):
                    code = general.get("Code")
                    exchange = general.get("Exchange")
                    if code and exchange:
                        symbol = f"{str(code).upper()}.{str(exchange).upper()}"
                items.append((symbol, value))
    elif isinstance(payload, list):
        for value in payload:
            if not isinstance(value, Mapping):
                continue
            general = value.get("General")
            symbol = ""
            if isinstance(general, Mapping):
                code = general.get("Code")
                exchange = general.get("Exchange")
                if code and exchange:
                    symbol = f"{str(code).upper()}.{str(exchange).upper()}"
            if not symbol:
                symbol = str(value.get("Code") or value.get("code") or "").upper()
            if symbol:
                items.append((symbol, value))
    return items


def _append_status(path: Path, symbol: str, status: str, row_count: int, error: str = "") -> None:
    append_csv(
        path,
        [
            {
                "role": "fundamentals",
                "eodhd_symbol": symbol,
                "ticker": _ticker_from_symbol(symbol),
                "status": status,
                "row_count": row_count,
                "start_date": "",
                "end_date": "",
                "error": error[:500],
                "finished_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
        ],
        headers=STATUS_HEADERS,
    )


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    raw_dir = dataset_root / "raw"
    fundamentals_raw_dir = raw_dir / "eodhd_fundamentals_raw"
    status_path = raw_dir / "eodhd_fetch_status.csv"
    universe_path = raw_dir / "eodhd_common_stock_universe.csv"
    fundamentals_raw_dir.mkdir(parents=True, exist_ok=True)

    symbols = _candidate_symbols(args, universe_path, fundamentals_raw_dir, status_path)
    batches = _iter_batches(symbols, args.batch_size)
    estimated_calls = sum(100 + len(batch) for batch in batches)
    plan = {
        "dataset_root": str(dataset_root.resolve()),
        "candidate_symbols": len(symbols),
        "batch_size": max(1, min(args.batch_size, 500)),
        "batch_count": len(batches),
        "estimated_api_calls": estimated_calls,
        "endpoint": f"/bulk-fundamentals/{args.exchange}",
        "version": args.version,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps({"step": "bulk_fundamentals_plan", **plan}, indent=2), flush=True)
    if args.dry_run or not symbols:
        write_json(raw_dir / "eodhd_bulk_fundamentals_plan.json", plan)
        return

    client = EODHDRESTClient(
        credentials=load_eodhd_credentials(args.credentials_path),
        rate_limiter=RateLimiter(args.rate_limit_calls, args.rate_limit_period_seconds),
    )

    fetched_symbols: set[str] = set()
    try:
        for batch in batches:
            payload = client.get_bulk_fundamentals(
                args.exchange,
                symbols=batch,
                version=args.version or None,
            )
            items = _payload_items(payload)
            returned_symbols = set()
            for symbol, item in items:
                symbol = symbol.upper()
                if "." not in symbol:
                    symbol = f"{symbol}.US"
                fundamental_payload_path(fundamentals_raw_dir, symbol).write_text(
                    json.dumps(item, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                _append_status(status_path, symbol, "ok", 1)
                returned_symbols.add(symbol)
                fetched_symbols.add(symbol)
            missing = [symbol for symbol in batch if symbol not in returned_symbols]
            for symbol in missing:
                _append_status(status_path, symbol, "empty", 0, "bulk response did not include symbol")
            print(
                json.dumps(
                    {
                        "step": "fetched_bulk_fundamentals_batch",
                        "requested": len(batch),
                        "returned": len(items),
                        "missing": len(missing),
                    }
                ),
                flush=True,
            )
    except EODHDAuthError as exc:
        print(
            json.dumps(
                {
                    "step": "bulk_fundamentals_not_available",
                    "error": str(exc)[:500],
                    "action": "Ask EODHD support to enable Bulk Fundamentals / Extended Fundamentals access.",
                },
                indent=2,
            ),
            flush=True,
        )
    except EODHDRateLimitError:
        raise
    except EODHDAPIError as exc:
        print(json.dumps({"step": "bulk_fundamentals_error", "error": str(exc)[:500]}), flush=True)
        raise
    finally:
        summary = {**plan, "fetched_symbols": len(fetched_symbols)}
        write_json(raw_dir / "eodhd_bulk_fundamentals_plan.json", summary)


if __name__ == "__main__":
    main()
