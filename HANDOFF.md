# HANDOFF

Use this file as the shared handoff note when switching between computers or Codex sessions.

## Read First

- `AGENTS.md`
- `ARCHITECTURE.md`
- `DATA_SCHEMA.md`
- `HANDOFF.md`

## Current Task

- Summary: Massive daily market data has been collected for the current S&P 500 development panel, Layer 2 daily features are built, and a PIT-safe normalized feature table is ready for modeling.
- Goal: Start the first supervised baseline model on the extracted normalized daily features, using a leakage-safe time split.
- Status: Data ingestion and feature engineering are complete for the current development panel. Modeling has not started yet.
- Why it matters: This is the first real benchmark step for the project and will validate whether the extracted feature stack supports predictive signal before adding more complexity.

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
  - `src/data/normalization.py`
  - `scripts/build_daily_features_from_raw.py`
  - `scripts/build_normalized_features_from_processed.py`
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

## Assumptions

- Assumption: The first baseline will be an end-of-day prediction setup, so same-day close, volume, VWAP, and same-date normalized features are allowed.
- Assumption: For this development phase, using the current S&P 500 constituent and sector snapshot is acceptable even though it is not a historically correct membership panel.

## Leakage And Data Risks

- Known leakage risks:
  - The universe is the current S&P 500 constituent snapshot projected across the available history, so it carries survivorship bias.
  - Sector labels come from the current constituent snapshot rather than a verified historical sector panel.
  - Final evaluation must not use global scaling statistics fit on future periods; keep using same-date or train-only scaling rules.
- Data quality risks:
  - Massive free-plan history did not deliver the requested `1995-01-01` start; the usable panel currently begins on `2024-04-22`.
  - This development dataset is daily-only and currently does not include verified filing-dated fundamentals or broader context features.
- Fields or joins that still need verification:
  - Point-in-time fundamentals and valuation fields
  - Historical sector or constituent membership if a stricter backtest universe is needed later
  - Benchmark or market-adjusted target construction for the large panel if used in the first baseline

## What Was Tested

- Commands run:
  - `python -m unittest tests.test_massive_stage1 tests.test_normalization`
  - `python scripts/build_normalized_features_from_processed.py --dataset-root data\massive_sp500_current_constituents_history`
- Results:
  - Data feature engineering and normalization tests passed.
  - The normalized feature table was built successfully with `249,497` rows and the expected same-date cross-sectional and sector-relative columns.
- What is still untested:
  - No baseline model training pipeline has been run yet.
  - No walk-forward train/validation/test split has been finalized yet.
  - No modeling ablation has been run between normalized and unnormalized features yet.

## Blockers

- Blocker: No hard blocker right now.
  Needed to unblock: Tomorrow's session should define the exact first baseline setup, select the target table and time split, and begin model training on `daily_features_normalized.csv`.

## Next Steps

1. Build the first leakage-safe baseline model on `data/massive_sp500_current_constituents_history/processed/daily_features_normalized.csv`.
2. Define a walk-forward or strict chronological train/validation/test split and make the exact target construction explicit.
3. Run a simple baseline comparison such as linear or ridge regression, MLP, or gradient-boosted trees and compare normalized inputs against the original engineered feature table as an ablation.

## Resume Prompt

Use this to restart cleanly in a new Codex session:

```text
Read AGENTS.md, ARCHITECTURE.md, DATA_SCHEMA.md, and HANDOFF.md. Use `data/massive_sp500_current_constituents_history/processed/daily_features_normalized.csv` as the main modeling table. Continue by building the first leakage-safe baseline model on the extracted features, define a strict time-based split, keep assumptions explicit, and prioritize correctness over complexity.
```

## Update Checklist

Before stopping work, update:

- `Current Task`
- `Current Branch And State`
- `Files In Play`
- `Recent Decisions`
- `What Was Tested`
- `Next Steps`
