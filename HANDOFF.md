# HANDOFF

Use this file as the shared handoff note when switching between computers or Codex sessions.

## Read First

- `AGENTS.md`
- `ARCHITECTURE.md`
- `DATA_SCHEMA.md`
- `HANDOFF.md`

## Current Task

- Summary: V1 supervised baseline architecture is implemented as a full ML-stack train/predict pipeline with multi-horizon market-adjusted targets, SPY/sector ETF context support, all-model artifact retention, and leaderboard metrics.
- Goal: Train multiple baseline model families on one shared 60-day episode input and generate future predictions from every trained model.
- Status: Code and tests are implemented. Live V1 training needs `processed/market_context_features.csv`, which should be collected from Massive first.
- Why it matters: The project can now move from feature tables to a reproducible multi-model supervised baseline workflow.

## Current Branch And State

- Branch: Not recorded here. `git` was not available in the current shell, so check branch status manually in the next session if needed.
- Last good commit: Not recorded in this handoff for the same reason as above.
- Uncommitted changes: Assume there are local code, doc, and data updates from the Massive ingestion, feature engineering, normalization, and doc refresh work.
- Environment notes: Main modeling dataset is `data/massive_sp500_current_constituents_history`. Actual processed coverage is `2024-04-22` through `2026-04-21` with `249,497` rows across `503` current S&P 500 names. Normalization assumes end-of-day availability and uses same-date only statistics.

## Files In Play

- Main files being edited:
  - `HANDOFF.md`
  - `AGENTS.md`
  - `ARCHITECTURE.md`
  - `DATA_SCHEMA.md`
  - `src/data/massive_stage1.py`
  - `src/data/incremental_update.py`
  - `src/data/normalization.py`
  - `src/data/v1_dataset.py`
  - `src/models/v1_baselines.py`
  - `scripts/build_daily_features_from_raw.py`
  - `scripts/build_normalized_features_from_processed.py`
  - `scripts/collect_massive_market_context.py`
  - `scripts/train_v1_supervised_baselines.py`
  - `scripts/predict_v1_supervised_baselines.py`
  - `scripts/update_massive_daily_dataset.py`
  - `tests/test_incremental_update.py`
  - `tests/test_v1_supervised_baselines.py`
- Related config/data paths:
  - `data/massive_sp500_current_constituents_history/raw/daily_market_bars.csv`
  - `data/massive_sp500_current_constituents_history/raw/sp500_constituents_current.csv`
  - `data/massive_sp500_current_constituents_history/processed/daily_features.csv`
  - `data/massive_sp500_current_constituents_history/processed/daily_features_normalized.csv`
  - `data/massive_sp500_current_constituents_history/processed/daily_features_normalized_manifest.json`
  - `data/massive_sp500_current_constituents_history/summary.json`
  - `data/massive_sp500_current_constituents_history/README.md`
- Files to avoid touching:
  - Avoid recollecting or overwriting the raw Massive dump unless there is a clear need; modeling should proceed off the existing processed tables first.

## Recent Decisions

- Decision: Use the current S&P 500 development panel as the working cross-sectional universe for feature engineering.
  Reason: It provides enough breadth for a realistic first baseline while keeping the free-plan data collection path manageable.
- Decision: Add normalized features in a separate processed file rather than overwriting the original engineered feature table.
  Reason: This preserves auditability, keeps ablations easy, and lets baseline modeling compare normalized vs. unnormalized inputs later.
- Decision: Cross-sectional and sector-relative normalization must be same-date only.
  Reason: This keeps the normalization step point-in-time safe with respect to time for an end-of-day model.
- Decision: Keep completed supervised episodes separate from latest prediction windows.
  Reason: Latest windows should include the most recently updated data but must not invent unavailable future returns.
- Decision: Incremental updates refetch a short recent overlap and replace duplicate `(ticker, date, adjusted)` rows.
  Reason: This gives a simple append/update path while allowing recent vendor corrections to overwrite stale rows.
- Decision: Keep SPY and sector ETF context in a separate market context table rather than adding ETFs to the stock cross-sectional panel.
  Reason: This avoids contaminating stock cross-sectional normalization while still supporting market-adjusted targets and trend context.
- Decision: Do not keep lightweight model fallbacks for V1 supervised models.
  Reason: The full ML libraries are installed and should be required for the real V1 baseline suite.
- Decision: Use Python 3.11 for full V1 model training.
  Reason: scikit-learn, LightGBM, XGBoost, torch, and pandas were installed into Python 3.11; the default Python is 3.14.

## Assumptions

- Assumption: The first baseline will be an end-of-day prediction setup, so same-day close, volume, VWAP, and same-date normalized features are allowed.
- Assumption: For this development phase, using the current S&P 500 constituent and sector snapshot is acceptable even though it is not a historically correct membership panel.

## Leakage And Data Risks

- Known leakage risks:
  - The universe is the current S&P 500 constituent snapshot projected across the available history, so it carries survivorship bias.
  - Sector labels come from the current constituent snapshot rather than a verified historical sector panel.
  - Final evaluation must not use global scaling statistics fit on future periods; keep using same-date or train-only scaling rules.
  - Latest prediction windows are target-pending by design; they should not be mixed into supervised training until their future horizon has elapsed and targets are regenerated.
- Data quality risks:
  - Massive free-plan history did not deliver the requested `1995-01-01` start; the usable panel currently begins on `2024-04-22`.
  - This development dataset is daily-only and currently does not include verified filing-dated fundamentals or broader context features.
  - Incremental updates only refresh a recent overlap window; full rebuild/refetch is still needed if Massive revises older adjusted bars after corporate actions.
  - Current processed stock features do not include SPY or sector ETFs; V1 training requires running the market context collector first.
- Fields or joins that still need verification:
  - Point-in-time fundamentals and valuation fields
  - Historical sector or constituent membership if a stricter backtest universe is needed later
  - Benchmark or market-adjusted target construction for the large panel if used in the first baseline

## What Was Tested

- Commands run:
  - `python -m unittest tests.test_massive_stage1 tests.test_normalization tests.test_incremental_update`
  - `python scripts\update_massive_daily_dataset.py --help`
  - `python -m unittest tests.test_massive_stage1 tests.test_normalization tests.test_incremental_update tests.test_v1_supervised_baselines`
  - `python scripts\collect_massive_market_context.py --help`
  - `python scripts\train_v1_supervised_baselines.py --help`
  - `python scripts\predict_v1_supervised_baselines.py --help`
  - `py -3.11 -m pip install --user scikit-learn lightgbm xgboost torch`
  - `py -3.11 -m pip install --user pandas`
  - `py -3.11 -m unittest tests.test_v1_supervised_baselines`
  - `py -3.11 scripts\collect_massive_market_context.py --dataset-root data\massive_sp500_current_constituents_history --rate-limit-calls 5 --rate-limit-period-seconds 60`
  - `py -3.11 -m pip install --user boto3`
  - `py -3.11 scripts\collect_massive_market_context.py --dataset-root data\massive_sp500_current_constituents_history --source flatfiles --start-date 2024-04-22 --end-date 2024-04-22 --rate-limit-calls 5 --rate-limit-period-seconds 60`
- Results:
  - Data feature engineering, normalization, incremental update, and V1 supervised baseline tests passed.
  - The market context collector, V1 trainer, and V1 predictor CLIs import and render help successfully.
  - Python 3.11 full-stack model factory sees `lightgbm`, `xgboost`, `sklearn_hist_gb`, `sklearn_mlp`, and `torch_mlp`; lightweight ridge/elastic-net/tree/MLP fallbacks were removed.
  - Live Massive context collection was attempted with the 5-calls/minute limiter but stopped before any request because `MASSIVE_API_KEY`/`MassiveApiKey` is missing in this shell.
  - After correcting the copied secret/API key, REST context collection succeeded from the free-plan entitlement start date `2024-04-24` through `2026-04-23`.
  - S3 flat-file credentials can list the daily aggregate catalog but cannot read objects; `GetObject` returns 403 Forbidden.
  - Smoke V1 training completed at `artifacts/v1_baselines/smoke_context_check`.
  - Smoke latest prediction generation wrote `3018` all-model rows.
- What is still untested:
  - No full-size V1 baseline training run has been executed yet.
  - No walk-forward train/validation/test split has been finalized yet.
  - No modeling ablation has been run on the real dataset yet.

## Blockers

- Blocker: No hard blocker right now.
  Needed to unblock: Run the market context collector to create `processed/market_context_features.csv`, then run a smoke V1 training job.

## Next Steps

1. Run the full V1 baseline training now that context exists, for example `py -3.11 scripts/train_v1_supervised_baselines.py --dataset-root data\massive_sp500_current_constituents_history`.
2. Use `scripts/predict_v1_supervised_baselines.py` against the chosen full run directory to generate all-model latest predictions.
3. Review whether the free-plan context start date `2024-04-24` should force trimming or regenerating any stock feature rows before final evaluation.

## Resume Prompt

Use this to restart cleanly in a new Codex session:

```text
Read AGENTS.md, ARCHITECTURE.md, DATA_SCHEMA.md, and HANDOFF.md. V1 supervised baseline code now lives in `src/data/v1_dataset.py`, `src/models/v1_baselines.py`, `scripts/collect_massive_market_context.py`, `scripts/train_v1_supervised_baselines.py`, and `scripts/predict_v1_supervised_baselines.py`. Use Python 3.11 for full V1 training because the ML stack is installed there. First collect SPY/sector ETF context, then run a smoke training job, then generate all-model latest predictions from the saved run directory.
```

## Update Checklist

Before stopping work, update:

- `Current Task`
- `Current Branch And State`
- `Files In Play`
- `Recent Decisions`
- `What Was Tested`
- `Next Steps`
