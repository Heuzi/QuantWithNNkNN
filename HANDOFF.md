# HANDOFF

Use this file as the shared handoff note when switching between computers or Codex sessions.

## Read First

- `AGENTS.md`
- `ARCHITECTURE.md`
- `DATA_SCHEMA.md`
- `HANDOFF.md`

## Current Task

- Summary: The main Massive dataset has now been refreshed to the latest available free-plan data, and the V1 supervised stack still supports both regression and 20-day outperformance event classification with walk-forward as the default evaluation scheme.
- Goal: Use the refreshed dataset to run the next real-data walk-forward benchmark, especially to compare whether the event-classification target is more learnable than the current regression targets.
- Status: On April 26, 2026, the stock raw plus processed artifacts were incrementally refreshed through `2026-04-24`, and the separate market-context raw plus processed tables were refreshed through the same latest date. The next meaningful experiment is a real-data walk-forward classification benchmark on this refreshed dataset.
- Why it matters: The repo can now ask both "how much excess return?" and "will this become a strong benchmark-relative winner?" without maintaining separate pipelines.

## Current Branch And State

- Branch: Not recorded here. `git` was still not available in the shell session used for the last major run.
- Last good commit: Not recorded here for the same reason.
- Uncommitted changes: Assume local code and doc edits in the V1 data/model/training stack and markdown files.
- Environment notes:
  - Main modeling dataset is `data/massive_sp500_current_constituents_history`.
  - Stock raw plus processed coverage is approximately `2024-04-22` through `2026-04-24` with `251,014` rows across `504` tickers.
  - The `504` stock-panel tickers are the `503` current S&P 500 names plus benchmark ticker `SPY`, which the incremental updater appends so it can rebuild market-adjusted episode targets.
  - Market-context coverage is approximately `2024-04-26` through `2026-04-24` with `6,000` rows across `12` tickers.
  - CUDA is available on this machine and the code now uses it when possible for `torch`, `xgboost`, and `lightgbm`.
  - Python 3.11 is still the intended runtime for the full ML stack.

## Files In Play

- Main code paths:
  - `src/data/v1_dataset.py`
  - `src/models/v1_baselines.py`
  - `scripts/train_v1_supervised_baselines.py`
  - `scripts/predict_v1_supervised_baselines.py`
  - `scripts/update_massive_daily_dataset.py`
  - `tests/test_v1_supervised_baselines.py`
- Main docs touched recently:
  - `ARCHITECTURE.md`
  - `HANDOFF.md`
  - `data/massive_sp500_current_constituents_history/README.md`
- Key artifact directories:
  - `artifacts/v1_baselines/current_v1_20260424`
  - `artifacts/v1_baselines/walk_forward_full_gpu_20260425`
  - `artifacts/v1_baselines/gpu_smoke_20260425`
  - `artifacts/v1_baselines/walk_forward_realdata_ridge_smoke`

## Recent Decisions

- Decision: Make expanding-window walk-forward the default evaluation mode for V1 training.
  Reason: It is closer to the intended final evaluation standard than a single chronological holdout split.
- Decision: After an OOS block is scored, allow that block to join the training pool for later folds.
  Reason: That matches how the model would actually be refreshed in production.
- Decision: Keep legacy holdout mode behind a flag.
  Reason: It preserves reproducibility and gives a direct baseline for comparison.
- Decision: Add `torch_seq_static` as the first true sequence-plus-static model.
  Reason: It matches the architecture roadmap better than the older flattened-only baselines.
- Decision: Prefer GPU execution automatically for `torch`, `xgboost`, and `lightgbm`, with CPU fallback and persisted fallback metadata.
  Reason: Full walk-forward benchmarking is expensive enough that automatic acceleration is worth using wherever the stack supports it, and benchmark artifacts need to show whether a model really used an accelerator.
- Decision: Serialize torch-based model bundles in a CPU-safe way, then move them back onto the best available device after load.
  Reason: GPU-trained artifacts should stay portable across sessions and machines.
- Decision: Add a binary classification task alongside regression instead of replacing regression.
  Reason: The event target may be easier to learn and more useful for screening, but we still want direct regression baselines for comparison.
- Decision: Define the new classification label as `market_outperform_any_20d_gt_5pct`.
  Reason: It stays PIT-safe, is easy to interpret, and captures "strong winner" behavior better than an exact continuous endpoint target.
- Decision: Make the incremental Massive updater premium-history ready by defaulting its cold-start fetch to `1995-01-01`.
  Reason: A premium account should not require a code edit to pull a longer range, while free-plan responses still degrade gracefully to the shorter vendor-limited history.
- Decision: Treat the stock refresh and market-context refresh as a paired workflow.
  Reason: `scripts/update_massive_daily_dataset.py` rebuilds the stock-side raw and processed artifacts, but it does not touch the separate context files, so `scripts/collect_massive_market_context.py --source rest` should be rerun immediately afterward to keep dates aligned.

## Current Results

- Existing regression benchmark:
  - `artifacts/v1_baselines/walk_forward_full_gpu_20260425/comparison.csv`
  - `artifacts/v1_baselines/walk_forward_full_gpu_20260425/comparison_summary.json`
- Latest dataset refresh:
  - Stock refresh on `2026-04-26` fetched overlap dates `2026-04-15` through `2026-04-25`, appended `4,032` incoming rows, and rebuilt `raw/daily_market_bars.csv`, `processed/daily_features.csv`, `processed/daily_features_normalized.csv`, `processed/episode_index.csv`, `processed/prediction_windows.csv`, and `processed/incremental_update_manifest.json`.
  - Stock raw plus processed tables now end on `2026-04-24`.
  - Market-context refresh on `2026-04-26` rebuilt `raw/market_context_bars.csv`, `processed/market_context_features.csv`, and `processed/market_context_manifest.json` through `2026-04-24` using the REST path.
- New implementation status:
  - regression and classification now train side by side
  - task-specific artifacts are written separately
  - latest prediction now handles mixed-task model indexes
  - `torch_seq_static` now supports sequence component ablations with stock-only, relative-stock, market-context, and sector-context daily tokens
  - sequence-static torch baselines now use a CPU-feasible default epoch budget: `max_epochs=20`, `patience=4`, batch size `512`
  - the fullest sequence layout is `stock_relative_market_sector_sequence`
  - Priority A daily features from the current Massive OHLCV/context tables are available: `close_location`, `true_range_pct`, `dollar_volume_ratio_5d`, `volume_zscore_20d`, and stock-vs-market/sector 1d/5d return features
  - compact tabular and sequence feature profiles are available, with `stock_relative_market_sector_compact` around 191 tabular columns and `stock_relative_market_sector_compact_sequence` around 65 daily-token columns after Priority A fields
- Classification task details:
  - label column: `market_outperform_any_20d_gt_5pct`
  - positive when pathwise stock excess return over `SPY` exceeds `5%` anywhere in the next 20 trading days
  - classification models currently include logistic regression, elastic-net logistic regression, LightGBM, XGBoost, sklearn MLP, torch MLP, and optional `torch_seq_static_classifier`
- Input layout details:
  - tabular baselines consume one flattened episode row with rolling `__last`, `__mean60`, and `__std60` features
  - sequence models consume `[batch_size, window_length, features_per_day]` daily-token tensors
  - default `window_length` is 60 trading days
- Evidence from tests:
  - the end-to-end smoke run now writes both regression and classification OOS artifacts
  - `latest_predictions.csv` now includes both `task_type = regression` and `task_type = classification`

## Leakage And Data Risks

- Known leakage risks:
  - The universe is still the current S&P 500 constituent snapshot projected backward across history, so it carries survivorship bias.
  - Sector labels still come from the current constituent snapshot rather than a verified historical sector panel.
  - Latest prediction windows are target-pending by design and must stay out of supervised training until their horizons have elapsed and targets are rebuilt.
- Data quality risks:
  - Massive free-plan stock history still begins around `2024-04-22`, not `1995-01-01`.
  - The stock and context tables now share the same latest date, `2026-04-24`, but they still have different earliest available dates.
  - The stock-side raw and processed panels now include `SPY` in addition to the `503` current S&P 500 names because the incremental updater uses it to rebuild benchmark-relative artifacts.
  - Filing-dated fundamentals are still absent.
- Classification-specific risks:
  - some early or tiny walk-forward folds can be extremely imbalanced or even single-class; the code now falls back to constant-probability classifiers in those cases, but the metric quality is still limited on such folds.
  - classification top-bottom realized spread currently uses the realized 20-day market-adjusted endpoint return as its realized spread proxy.
- Modeling risks:
  - `torch_seq_static` is implemented, but its current architecture/hyperparameters may still underperform the better tabular baselines and should be rebenchmarked after the new context-aware sequence layouts are used.
  - `elastic_net` can emit convergence warnings on the widest feature sets.

## What Was Tested

- Commands run:
  - `py -3.11 scripts/update_massive_daily_dataset.py --dataset-root data\massive_sp500_current_constituents_history`
  - `py -3.11 scripts/collect_massive_market_context.py --dataset-root data\massive_sp500_current_constituents_history --source rest --start-date 2024-04-22`
  - `py -3.11 -m py_compile src/data/v1_dataset.py src/models/v1_baselines.py scripts/train_v1_supervised_baselines.py scripts/predict_v1_supervised_baselines.py tests/test_v1_supervised_baselines.py`
  - `py -3.11 -m unittest tests.test_v1_supervised_baselines`
  - `py -3.11 -m pip install "lightgbm>=4.6" "xgboost>=3.2"`
  - `py -3.11 -m unittest tests.test_v1_supervised_baselines`
  - `py -3.11 scripts/train_v1_supervised_baselines.py --dataset-root data\massive_sp500_current_constituents_history --output-root artifacts\v1_baselines --run-name walk_forward_realdata_ridge_smoke --eval-mode walk_forward --models ridge --feature-sets stock_only --compare-against-run artifacts\v1_baselines\current_v1_20260424`
  - `py -3.11 scripts/train_v1_supervised_baselines.py --dataset-root data\massive_sp500_current_constituents_history --output-root artifacts\v1_baselines --run-name walk_forward_full_gpu_20260425 --eval-mode walk_forward --compare-against-run artifacts\v1_baselines\current_v1_20260424`
  - `py -3.11 scripts/train_v1_supervised_baselines.py --dataset-root data\massive_sp500_current_constituents_history --output-root artifacts\v1_baselines --run-name gpu_smoke_20260425 --eval-mode walk_forward --models lightgbm,xgboost,torch_mlp,torch_seq_static --feature-sets stock_only,stock_relative --max-episodes 25000 --walk-forward-min-train-dates 20 --walk-forward-val-block-size 5 --walk-forward-oos-block-size 5 --walk-forward-purge-gap 10 --final-stop-block-size 5`
  - `py -3.11 scripts/predict_v1_supervised_baselines.py --run-dir artifacts\v1_baselines\walk_forward_full_gpu_20260425 --dataset-root data\massive_sp500_current_constituents_history`
- Results:
  - The main Massive stock dataset now ends on `2026-04-24` and the separate market-context dataset was refreshed to the same latest date.
  - `processed/episode_index.csv`, `processed/prediction_windows.csv`, and `processed/incremental_update_manifest.json` now exist and were regenerated by the stock refresh workflow.
  - The updated V1 test suite passed with the new regression-plus-classification pipeline.
  - Full walk-forward training completed with all final artifacts written.
  - Comparison output and latest predictions were generated successfully.
  - GPU acceleration is now active for the supported model families on this machine.
  - The post-change GPU smoke run completed successfully on real data and wrote runtime metadata into `final_models.json`.
  - The end-to-end smoke test now verifies `classification_oos_predictions.csv`, `classification_oos_leaderboard.csv`, and `final_classification_models.json`.

## Blockers

- Blocker: No hard blocker right now.
  Needed to unblock: If runtime becomes a problem on other machines, verify CUDA availability first because the full benchmark is much slower on CPU-only setups.

## Next Steps

1. Tune or simplify `torch_seq_static`; right now it is architecturally useful but empirically weak.
2. Run a real-data classification benchmark on the refreshed `2026-04-24` dataset and compare the classification leaderboard against the current regression leaderboard.
3. If a premium Massive upgrade becomes available later, rerun the same paired refresh workflow and check whether longer history materially stabilizes the new classification task.
4. Decide whether to raise `ElasticNet` regularization / iteration settings to reduce convergence warnings on the widest feature sets.
5. If stricter realism is needed next, prioritize historical membership / sector PIT fixes before adding more feature families.

## Resume Prompt

Use this to restart cleanly in a new Codex session:

```text
Read AGENTS.md, ARCHITECTURE.md, DATA_SCHEMA.md, and HANDOFF.md. The main Massive dataset was refreshed on April 26, 2026 through `2026-04-24` for both stock and market-context tables. The default V1 trainer supports both regression and classification in `scripts/train_v1_supervised_baselines.py`, the classification label is `market_outperform_any_20d_gt_5pct`, GPU acceleration is enabled where supported in `src/models/v1_baselines.py`, and latest prediction handles mixed-task final deployment bundles. Start by running a real-data walk-forward benchmark on the refreshed dataset unless a new premium-history refresh or model-tuning task takes priority.
```

## Update Checklist

Before stopping work, update:

- `Current Task`
- `Current Branch And State`
- `Recent Decisions`
- `Current Results`
- `What Was Tested`
- `Next Steps`
