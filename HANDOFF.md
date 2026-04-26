# HANDOFF

Use this file as the shared handoff note when switching between computers or Codex sessions.

## Read First

- `AGENTS.md`
- `ARCHITECTURE.md`
- `DATA_SCHEMA.md`
- `HANDOFF.md`

## Current Task

- Summary: The V1 supervised stack now supports both regression and 20-day outperformance event classification. Walk-forward remains the default evaluation scheme, GPU paths are enabled where supported, and latest prediction works off mixed-task final deployment bundles.
- Goal: Compare whether the new event-classification target is more learnable than the current regression targets once longer history is available, while keeping the pipeline PIT-safe.
- Status: Code implementation and test coverage for the new classification path are complete. The next meaningful experiment is a real-data walk-forward classification benchmark, ideally after a premium Massive refresh extends the stock history.
- Why it matters: The repo can now ask both "how much excess return?" and "will this become a strong benchmark-relative winner?" without maintaining separate pipelines.

## Current Branch And State

- Branch: Not recorded here. `git` was still not available in the shell session used for the last major run.
- Last good commit: Not recorded here for the same reason.
- Uncommitted changes: Assume local code and doc edits in the V1 data/model/training stack and markdown files.
- Environment notes:
  - Main modeling dataset is `data/massive_sp500_current_constituents_history`.
  - Stock coverage is approximately `2024-04-22` through `2026-04-21` with `249,497` rows across `503` current S&P 500 names.
  - Market-context coverage is approximately `2024-04-24` through `2026-04-23`.
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
- Decision: Prefer GPU execution automatically for `torch`, `xgboost`, and `lightgbm`, with CPU fallback.
  Reason: Full walk-forward benchmarking is expensive enough that automatic acceleration is worth using wherever the stack supports it.
- Decision: Serialize torch-based model bundles in a CPU-safe way, then move them back onto the best available device after load.
  Reason: GPU-trained artifacts should stay portable across sessions and machines.
- Decision: Add a binary classification task alongside regression instead of replacing regression.
  Reason: The event target may be easier to learn and more useful for screening, but we still want direct regression baselines for comparison.
- Decision: Define the new classification label as `market_outperform_any_20d_gt_5pct`.
  Reason: It stays PIT-safe, is easy to interpret, and captures "strong winner" behavior better than an exact continuous endpoint target.
- Decision: Make the incremental Massive updater premium-history ready by defaulting its cold-start fetch to `1995-01-01`.
  Reason: A premium account should not require a code edit to pull a longer range, while free-plan responses still degrade gracefully to the shorter vendor-limited history.

## Current Results

- Existing regression benchmark:
  - `artifacts/v1_baselines/walk_forward_full_gpu_20260425/comparison.csv`
  - `artifacts/v1_baselines/walk_forward_full_gpu_20260425/comparison_summary.json`
- New implementation status:
  - regression and classification now train side by side
  - task-specific artifacts are written separately
  - latest prediction now handles mixed-task model indexes
- Classification task details:
  - label column: `market_outperform_any_20d_gt_5pct`
  - positive when pathwise stock excess return over `SPY` exceeds `5%` anywhere in the next 20 trading days
  - classification models currently include logistic regression, LightGBM, XGBoost, sklearn MLP, torch MLP, and optional `torch_seq_static_classifier`
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
  - The stock table and context table end on different dates, so latest prediction anchors follow the stock table, not the context table.
  - Filing-dated fundamentals are still absent.
- Classification-specific risks:
  - some early or tiny walk-forward folds can be extremely imbalanced or even single-class; the code now falls back to constant-probability classifiers in those cases, but the metric quality is still limited on such folds.
  - classification top-bottom realized spread currently uses the realized 20-day market-adjusted endpoint return as its realized spread proxy.
- Modeling risks:
  - `torch_seq_static` is implemented, but its current architecture/hyperparameters underperform the better tabular baselines.
  - `elastic_net` can emit convergence warnings on the widest feature sets.

## What Was Tested

- Commands run:
  - `py -3.11 -m py_compile src/data/v1_dataset.py src/models/v1_baselines.py scripts/train_v1_supervised_baselines.py scripts/predict_v1_supervised_baselines.py tests/test_v1_supervised_baselines.py`
  - `py -3.11 -m unittest tests.test_v1_supervised_baselines`
  - `py -3.11 -m pip install "lightgbm>=4.6" "xgboost>=3.2"`
  - `py -3.11 -m unittest tests.test_v1_supervised_baselines`
  - `py -3.11 scripts/train_v1_supervised_baselines.py --dataset-root data\massive_sp500_current_constituents_history --output-root artifacts\v1_baselines --run-name walk_forward_realdata_ridge_smoke --eval-mode walk_forward --models ridge --feature-sets stock_only --compare-against-run artifacts\v1_baselines\current_v1_20260424`
  - `py -3.11 scripts/train_v1_supervised_baselines.py --dataset-root data\massive_sp500_current_constituents_history --output-root artifacts\v1_baselines --run-name walk_forward_full_gpu_20260425 --eval-mode walk_forward --compare-against-run artifacts\v1_baselines\current_v1_20260424`
  - `py -3.11 scripts/train_v1_supervised_baselines.py --dataset-root data\massive_sp500_current_constituents_history --output-root artifacts\v1_baselines --run-name gpu_smoke_20260425 --eval-mode walk_forward --models lightgbm,xgboost,torch_mlp,torch_seq_static --feature-sets stock_only,stock_relative --max-episodes 25000 --walk-forward-min-train-dates 20 --walk-forward-val-block-size 5 --walk-forward-oos-block-size 5 --walk-forward-purge-gap 10 --final-stop-block-size 5`
  - `py -3.11 scripts/predict_v1_supervised_baselines.py --run-dir artifacts\v1_baselines\walk_forward_full_gpu_20260425 --dataset-root data\massive_sp500_current_constituents_history`
- Results:
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
2. Run a real-data classification benchmark and compare the classification leaderboard against the current regression leaderboard.
3. Refresh the dataset after a premium Massive upgrade to see whether longer history materially stabilizes the new classification task.
4. Decide whether to raise `ElasticNet` regularization / iteration settings to reduce convergence warnings on the widest feature sets.
5. If stricter realism is needed next, prioritize historical membership / sector PIT fixes before adding more feature families.

## Resume Prompt

Use this to restart cleanly in a new Codex session:

```text
Read AGENTS.md, ARCHITECTURE.md, DATA_SCHEMA.md, and HANDOFF.md. The default V1 trainer now supports both regression and classification in `scripts/train_v1_supervised_baselines.py`, the new classification label is `market_outperform_any_20d_gt_5pct`, GPU acceleration is enabled where supported in `src/models/v1_baselines.py`, and latest prediction now handles mixed-task final model indexes. Start by deciding whether to run a real-data classification benchmark, refresh the dataset with premium Massive history, or tune the static sequence branch.
```

## Update Checklist

Before stopping work, update:

- `Current Task`
- `Current Branch And State`
- `Recent Decisions`
- `Current Results`
- `What Was Tested`
- `Next Steps`
