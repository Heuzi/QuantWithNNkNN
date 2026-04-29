# HANDOFF

Use this file as the shared handoff note when switching between computers or Codex sessions.

## Read First

- `AGENTS.md`
- `ARCHITECTURE.md`
- `DATA_SCHEMA.md`
- `HANDOFF.md`

## Current Task

- Summary: The project is migrating daily market data from Massive to EODHD.
- Goal: Build a 30-year EODHD U.S. listed common-stock dataset, include delisted names, drop VWAP/transaction-derived features, and treat all Massive-era artifacts as legacy.
- Current default dataset root: `data/eodhd_us_equities_30y`.
- Current target universe policy: listed U.S. common stocks plus delisted names; exclude ETFs, funds, and OTC/PINK from target assets.
- Universe filtering also excludes units, warrants, rights, and preferred/preference shares when EODHD labels those rows as common stock.
- Context instruments remain `SPY` plus sector ETFs and are stored in the separate market-context table.

## Current Branch And State

- Branch: `codex/full-classification-benchmark`.
- Local secrets:
  - `EODHD_api_key` is local-only and must stay ignored.
  - Do not print API keys in logs or docs.
- Existing Massive/S&P500 artifacts are legacy and should be archived under `artifacts/archive/massive_legacy_<date>/`.
- `dev notes.txt` may contain unrelated local edits; do not stage or revert it unless explicitly requested.

## Main Code Paths

- EODHD adapter: `src/data/eodhd_stage1.py`
- EODHD rebuild script: `scripts/update_eodhd_daily_dataset.py`
- Dataset and feature construction: `src/data/v1_dataset.py`
- Normalization helpers: `src/data/normalization.py`
- Baseline trainer: `scripts/train_v1_supervised_baselines.py`
- Prediction script: `scripts/predict_v1_supervised_baselines.py`

Legacy Massive scripts remain in the repo for reproducibility but are no longer the default path.

## Current Data Policy

- EODHD EOD rows are normalized into adjusted internal OHLC using `adjusted_close` when available.
- `dollar_volume` is derived locally as adjusted internal `close * volume`.
- The model feature schema intentionally drops:
  - `vwap`
  - `transactions`
  - `close_to_vwap_pct`
- Raw price, raw volume, raw dollar volume, legacy VWAP, previous-close, and moving-average level columns remain forbidden as model inputs.
- EODHD fundamentals `General` metadata may provide sector/industry labels, but it is not treated as point-in-time fundamentals.
- Missing sector/industry metadata falls back to `Unknown`.
- EODHD symbol lists are collected from both current and delisted views, because the `delisted=1` view behaves as delisted-only in live checks.

## Recent Decisions

- Decision: Use EODHD as the current primary daily market vendor.
  Reason: The upgraded plan exposes long EOD history and filtered fundamentals metadata.
- Decision: Exclude ETFs/funds/OTC from the first target universe.
  Reason: They represent different prediction problems and add liquidity/microstructure noise.
- Decision: Keep `SPY` and sector ETFs as context instruments.
  Reason: They are still needed for benchmark-relative targets and regime/context features.
- Decision: Keep old Massive code as legacy instead of deleting it.
  Reason: Existing experiments remain reproducible, but their artifacts are not comparable with the new full-universe EODHD dataset.

## Validation Checklist

Before trusting a new EODHD run:

- `py -3.11 -m unittest discover -s tests`
- Live smoke with a tiny ticker set, for example:
  - `py -3.11 scripts/update_eodhd_daily_dataset.py --tickers AAPL,MSFT --max-tickers 2 --start-date 2025-01-01 --end-date 2025-06-30 --dataset-root data/eodhd_smoke --window-length 10 --horizon-days 5`
- Universe smoke through EODHD symbol lists:
  - `py -3.11 scripts/update_eodhd_daily_dataset.py --max-tickers 3 --start-date 2025-01-01 --end-date 2025-03-31 --dataset-root data/eodhd_universe_smoke --window-length 10 --horizon-days 5 --skip-fundamentals`
- Tiny training smoke:
  - `py -3.11 scripts/train_v1_supervised_baselines.py --dataset-root data/eodhd_smoke --output-root artifacts/v1_baselines --run-name eodhd_smoke --eval-mode holdout --task-type both --models ridge,torch_seq_static --classification-models lightgbm_classifier --feature-sets stock_only --horizons 5 --classification-horizon 5 --window-length 10`
- Verify feature columns contain no `close_to_vwap_pct`, `vwap`, or `transactions`.

## Known Risks

- Ticker identity and symbol reuse are not fully resolved by the daily bar adapter.
- Delisted coverage should be audited before final backtests.
- Raw volume is not currently split-adjusted by the adapter.
- EODHD sector/industry metadata is current vendor metadata, not a PIT sector history.
- Full all-stock rebuild can consume thousands of paid API calls and should be run intentionally after smoke/pilot validation.

## Resume Prompt

```text
Read AGENTS.md, ARCHITECTURE.md, DATA_SCHEMA.md, and HANDOFF.md. The repo is migrating V1 daily market data from Massive to EODHD. Current default dataset root is data/eodhd_us_equities_30y. Use scripts/update_eodhd_daily_dataset.py for EODHD collection/rebuild, then train with scripts/train_v1_supervised_baselines.py. Keep EODHD_api_key local and ignored. Treat Massive-era artifacts as archived legacy and do not compare them directly with EODHD full-universe results.
```
