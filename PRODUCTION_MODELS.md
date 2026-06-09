# Production Models

This file tracks the intended V1 production classifier policy and the promotion state for deployable bundles.

## Production Status

The current active `path_5pct_20d` production-candidate set is the best model from each supported classifier family after the balanced true-full ablation study.

The previous binary production artifacts are deprecated and must not be treated as the active production recommendation set. They were trained on the old binary target policy and have been removed from the promoted production state.

Intended production target:

- `path_5pct_20d`
- Inputs use the prior `60` trading days and as-of-safe metadata/features.
- The label is defined from close-to-close forward returns over the next `20` trading days:
  - class `0`: any forward path drawdown is `<= -5%`
  - class `2`: the path reaches `>= +5%` without ever breaching `-5%`
  - class `1`: otherwise neutral
- The full 20-trading-day forward window is required. Unresolved rows are unlabeled and excluded from supervised train/eval.

Current promoted candidate model families:

| Model | Run directory | Selected feature set | Selection score | Top-decile precision | Accuracy | Log loss | Output behavior |
|---|---|---|---:|---:|---:|---:|---|
| `xgboost_classifier` | `artifacts/v1_baselines/eodhd_true_full_xgboost` | `stock_normalized_lean_market_sector_fundamentals_sentiment` | `1.143022` | `0.591422` | `0.478090` | `0.956338` | `multi:softprob`, 3 class probabilities |
| `torch_mlp_classifier` | `artifacts/v1_baselines/eodhd_true_full_ablation_torch_mlp` | `stock_only_fundamentals` | `1.114128` | `0.566842` | `0.487801` | `0.941229` | 3-logit softmax output |
| `torch_seq_static_classifier` | `artifacts/v1_baselines/eodhd_true_full_ablation_torch_seq_static` | `stock_only_sequence` | `1.020999` | `0.530210` | `0.458752` | `0.968252` | 3-logit softmax output |

Only those three selected classifier bundles are supported for current `path_5pct_20d` production-style reports. Other multiclass classification artifacts remain research-only unless separately reviewed and promoted.

Selection rationale is recorded in `artifacts/v1_baselines/eodhd_true_full_ablation_balanced_summary/combined_classification_oos_leaderboard.csv` and `artifacts/v1_baselines/eodhd_true_full_ablation_balanced_summary/summary.md`. The promotion rule is `model_family_rank = 1` by the trading-oriented OOS `selection_score = mean_pr_auc + mean_top_decile_precision + mean_top_bottom_spread`.

Prediction columns for the future promoted set:

- `pred_prob_path_5pct_20d_class_0`
- `pred_prob_path_5pct_20d_class_1`
- `pred_prob_path_5pct_20d_class_2`
- `pred_class_path_5pct_20d`
- `pred_prob_path_5pct_20d`
- `pred_score_path_5pct_20d`

`pred_prob_path_5pct_20d` remains the class-2 probability alias for transparency and reporting. The default ranking score for `path_5pct_20d` is `pred_score_path_5pct_20d = P(class 2) - P(class 0)`.

## Artifact Timestamp Rules

For each run, `final_models.json` is the deployable model index. The individual model path is stored in each record's `artifact_path`.
Committed production manifests should use portable paths relative to the run directory, such as `models/MODEL_FILE.pkl`.

Future retrains write `trained_at_utc` into:

- `final_models.json`
- `trained_models.json`
- `final_classification_models.json`
- `trained_classification_models.json`

After promoting a newly retrained model set, update this file with:

- run directory
- artifact path or run directory
- last trained/saved timestamp
- OOS metrics
- any caveats, such as chunked XGBoost training

## Prediction Refresh Policy

Refreshing market data does not require retraining. Production-style scoring uses the promoted `path_5pct_20d` candidate set above.

Normal production cycle:

1. Refresh latest EODHD data after market close when fresh predictions are needed.
2. Rebuild prediction-ready features/windows using the existing PIT-safe schema in the bounded latest-inference cache.
3. Run the promoted final `path_5pct_20d` classifier bundles.
4. Apply the same conservative research universe policy used in train/test to the latest production prediction universe.
5. Rank surviving target-pending stock-window episodes by `pred_score_path_5pct_20d = P(class 2) - P(class 0)`.
6. Produce trading reports for human review.

Daily prediction refreshes should not rebuild the full 30-year normalized panel. They should reuse saved model bundles and build only the compact latest-window feature set under `data/eodhd_us_equities_30y/processed/latest_inference/`.

When a daily prediction refresh fetches new EOD bars, it also persists newly computed stock/context feature rows into retrain sidecars:

- `processed/daily_features_incremental_updates.csv`
- `processed/market_context_features_incremental_updates.csv`

Full-panel normalization is a retrain/data-snapshot artifact, saved as `processed/daily_features_normalized.csv` with `processed/daily_features_normalized_manifest.json`. The full retrain wrapper consolidates incremental feature sidecars into the main processed feature files first; when the normalized artifact already exists, the consolidator refreshes only the touched same-date cross sections and merges them into that artifact.

Use saved models for daily or every-few-days prediction refreshes as long as:

- target definition is unchanged
- feature schema is unchanged
- EODHD vendor semantics are unchanged
- ticker/window eligibility policy is unchanged
- new rows can be built with the same point-in-time rules

The conservative research universe is part of the shared strategy-universe policy. Standard training, walk-forward evaluation, latest prediction, and trading reports should all use the same filter configuration unless a run explicitly documents a different universe profile.

Run latest predictions through the trading-strategy wrapper, which loads the selected model-family run directories and filters each to `--leaderboard-rank 1`:

```powershell
py -3.11 scripts\run_trading_strategy.py `
  --dataset-root data\eodhd_us_equities_30y `
  --credentials-path EODHD_api_key `
  --force-rebuild-latest-inference `
  --leaderboard-rank 1
```

Use a bounded prediction dataset root instead of the full EODHD root if the full processed feature table is too large for the prediction machine.

## Retraining Policy

Default retrain cadence:

- Retrain every `2` to `4` weeks.
- Monthly is the default operating cadence.
- Use the shorter two-week cadence during volatile regimes, after material data additions, or while the strategy is still being monitored closely.

Do not retrain just because a new daily bar arrived. New data should usually be scored by the saved final bundles after promotion.

Retrain immediately when any of these change:

- target definition or horizon
- feature schema
- episode eligibility filter
- vendor data semantics
- fundamentals or sentiment join logic
- material leakage fix
- live/OOS monitoring shows meaningful drift
- a new model family is promoted into the production candidate set

Recommended full retrain command pattern:

```powershell
.\scripts\run_full_universe_retrain.ps1 -Resume
```

The wrapper refreshes raw EODHD bars, resumes chunked daily-feature construction, consolidates incremental processed-feature sidecars, incrementally refreshes or builds the saved normalized feature artifact, materializes the filtered true-full strategy-universe panel, materializes the shared episode cache, and trains `xgboost_classifier`, `torch_mlp_classifier`, and `torch_seq_static_classifier` separately. `-Resume` skips completed stages, including normalization once both normalized output files exist and are at least as fresh as `processed/daily_features.csv`.

Two-sleeve retraining is now the preferred research path before changing production promotion state:

```powershell
.\scripts\run_two_sleeve_retrain.ps1 -Resume
```

This trains `conservative` and `momentum_breakout` models independently. For the current refresh, run only `-Sleeve momentum_breakout` because the conservative production-candidate models already exist. The current momentum/breakout profile trains bounded tabular model/feature-set candidates and the trading wrapper scores the top three OOS leaderboard rows with `--leaderboard-top-k 3`. The sleeve profiles currently use `walk_forward_max_folds=1`, which evaluates only the latest chronological train/validation/OOS fold to reduce runtime. That is acceptable for fast sleeve iteration, but a promotion decision should treat it as a recent-regime check rather than a full multi-regime walk-forward study. The sleeves must not share promoted model artifacts. Compare each sleeve's OOS leaderboard and final-model manifests independently before deciding whether either sleeve should be promoted into production reports.

Only promote the new models after confirming:

- all final model manifests exist
- all model `.pkl` artifacts exist
- OOS metrics are acceptable versus the previously accepted benchmark or review threshold
- the run manifests and dataset manifests record `classification_event_type=path_5pct_20d`
- prediction outputs include the class probability columns, the `pred_prob_path_5pct_20d` class-2 probability alias, and the `pred_score_path_5pct_20d` ranking score
- `latest_predictions.csv` generation succeeds on the refreshed prediction dataset

## XGBoost Caveat

The promoted `xgboost_classifier` multiclass artifact may still use the chunked fallback path if one-shot full-fold XGBoost stalls at full fold size.

All rows are still eligible for use, but not every tree necessarily sees every row in a chunked run. Keep this caveat attached to any promoted XGBoost candidate until a cleaner full-batch or staged/subsampled multiclass XGBoost run is validated.
