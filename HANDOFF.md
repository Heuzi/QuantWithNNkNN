# HANDOFF

Use this file as the shared handoff note when switching between computers or Codex sessions.

## Read First

- `AGENTS.md`
- `ARCHITECTURE.md`
- `DATA_SCHEMA.md`
- `HANDOFF.md`

## Current Task

- Summary: The V1 supervised baseline stack is implemented and benchmarked. Walk-forward is now the default training scheme, GPU paths are enabled where supported, and latest prediction works off final deployment bundles.
- Goal: Improve the strongest walk-forward models, keep the pipeline PIT-safe, and decide what to do next about the underperforming static branch.
- Status: Full benchmark run completed at `artifacts/v1_baselines/walk_forward_full_gpu_20260425`.
- Why it matters: The repo now has a reproducible full-dataset baseline suite with a real old-vs-new comparison instead of only smoke runs.

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

## Current Results

- Full walk-forward comparison output:
  - `artifacts/v1_baselines/walk_forward_full_gpu_20260425/comparison.csv`
  - `artifacts/v1_baselines/walk_forward_full_gpu_20260425/comparison_summary.json`
- High-level verdict:
  - `matched_combo_count = 40`
  - `improved_count = 22`
  - `worsened_count = 18`
  - `median_selection_score_delta = +0.005876159864072738`
  - `new_scheme_better = true`
- Post-change GPU smoke:
  - `artifacts/v1_baselines/gpu_smoke_20260425`
  - This run exercised `lightgbm`, `xgboost`, `torch_mlp`, and `torch_seq_static` on real data after the latest GPU-wrapper updates.
- Best new OOS leaderboard rows:
  - `torch_mlp / stock_only`
  - `elastic_net / stock_only`
  - `momentum_heuristic / stock_relative_market_sector`
- Notable pattern:
  - `elastic_net`, `ridge`, `torch_mlp`, and `sklearn_mlp` improved the most on average.
  - `torch_seq_static` finished and saved correctly, but it ranked poorly and is not yet competitive.

## Leakage And Data Risks

- Known leakage risks:
  - The universe is still the current S&P 500 constituent snapshot projected backward across history, so it carries survivorship bias.
  - Sector labels still come from the current constituent snapshot rather than a verified historical sector panel.
  - Latest prediction windows are target-pending by design and must stay out of supervised training until their horizons have elapsed and targets are rebuilt.
- Data quality risks:
  - Massive free-plan stock history still begins around `2024-04-22`, not `1995-01-01`.
  - The stock table and context table end on different dates, so latest prediction anchors follow the stock table, not the context table.
  - Filing-dated fundamentals are still absent.
- Modeling risks:
  - `torch_seq_static` is implemented, but its current architecture/hyperparameters underperform the better tabular baselines.
  - `elastic_net` can emit convergence warnings on the widest feature sets.

## What Was Tested

- Commands run:
  - `py -3.11 -m pip install "lightgbm>=4.6" "xgboost>=3.2"`
  - `py -3.11 -m unittest tests.test_v1_supervised_baselines`
  - `py -3.11 scripts/train_v1_supervised_baselines.py --dataset-root data\massive_sp500_current_constituents_history --output-root artifacts\v1_baselines --run-name walk_forward_realdata_ridge_smoke --eval-mode walk_forward --models ridge --feature-sets stock_only --compare-against-run artifacts\v1_baselines\current_v1_20260424`
  - `py -3.11 scripts/train_v1_supervised_baselines.py --dataset-root data\massive_sp500_current_constituents_history --output-root artifacts\v1_baselines --run-name walk_forward_full_gpu_20260425 --eval-mode walk_forward --compare-against-run artifacts\v1_baselines\current_v1_20260424`
  - `py -3.11 scripts/train_v1_supervised_baselines.py --dataset-root data\massive_sp500_current_constituents_history --output-root artifacts\v1_baselines --run-name gpu_smoke_20260425 --eval-mode walk_forward --models lightgbm,xgboost,torch_mlp,torch_seq_static --feature-sets stock_only,stock_relative --max-episodes 25000 --walk-forward-min-train-dates 20 --walk-forward-val-block-size 5 --walk-forward-oos-block-size 5 --walk-forward-purge-gap 10 --final-stop-block-size 5`
  - `py -3.11 scripts/predict_v1_supervised_baselines.py --run-dir artifacts\v1_baselines\walk_forward_full_gpu_20260425 --dataset-root data\massive_sp500_current_constituents_history`
- Results:
  - The updated V1 test suite passed.
  - Full walk-forward training completed with all final artifacts written.
  - Comparison output and latest predictions were generated successfully.
  - GPU acceleration is now active for the supported model families on this machine.
  - The post-change GPU smoke run completed successfully on real data and wrote runtime metadata into `final_models.json`.

## Blockers

- Blocker: No hard blocker right now.
  Needed to unblock: If runtime becomes a problem on other machines, verify CUDA availability first because the full benchmark is much slower on CPU-only setups.

## Next Steps

1. Tune or simplify `torch_seq_static`; right now it is architecturally useful but empirically weak.
2. Decide whether to raise `ElasticNet` regularization / iteration settings to reduce convergence warnings on the widest feature sets.
3. Consider adding explicit runtime summaries to artifact review notebooks or reports so GPU-vs-CPU paths are visible without inspecting model metadata.
4. If stricter realism is needed next, prioritize historical membership / sector PIT fixes before adding more feature families.

## Resume Prompt

Use this to restart cleanly in a new Codex session:

```text
Read AGENTS.md, ARCHITECTURE.md, DATA_SCHEMA.md, and HANDOFF.md. The default V1 trainer now uses walk-forward evaluation in `scripts/train_v1_supervised_baselines.py`, GPU acceleration is enabled where supported in `src/models/v1_baselines.py`, and the latest full benchmark is `artifacts/v1_baselines/walk_forward_full_gpu_20260425`. Start by reviewing `comparison_summary.json`, `comparison.csv`, and `oos_leaderboard.csv`, then decide whether to tune `torch_seq_static`, improve the strongest tabular models, or tighten PIT realism around the universe and sector metadata.
```

## Update Checklist

Before stopping work, update:

- `Current Task`
- `Current Branch And State`
- `Recent Decisions`
- `Current Results`
- `What Was Tested`
- `Next Steps`
