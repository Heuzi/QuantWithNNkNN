# Production Models

This file tracks the current deployable V1 classifier bundles and the policy for refreshing predictions versus retraining models.

## Current Production Set

Current target:

- `market_outperform_any_20d_gt_5pct`
- Positive means the stock's pathwise excess return over `SPY` exceeds `5%` at any point in the next `20` trading days.
- Inputs use the prior `60` trading days and as-of-safe metadata/features.
- OOS positive baseline from the full-run test folds: `50.25%`.

Current deployable models:

| Model | Run directory | Feature set | Last trained/saved local time | PR AUC | ROC AUC | Top-decile precision | Accuracy |
|---|---|---|---|---:|---:|---:|---:|
| `xgboost_classifier` | `artifacts/v1_baselines/eodhd_true_full_xgboost` | `stock_relative_market_sector_fundamentals_sentiment` | `2026-05-08 11:39:31 America/New_York` | `0.6048` | `0.6371` | `0.6411` | `0.5974` |
| `torch_mlp_classifier` | `artifacts/v1_baselines/eodhd_true_full_torch_mlp` | `stock_relative_market_sector_fundamentals_sentiment` | `2026-05-07 16:27:54 America/New_York` | `0.5934` | `0.6172` | `0.6221` | `0.5700` |
| `torch_seq_static_classifier` | `artifacts/v1_baselines/eodhd_true_full_torch_seq_static` | `stock_relative_market_sector_sentiment_sequence` | `2026-05-06 23:42:46 America/New_York` | `0.5711` | `0.6095` | `0.6066` | `0.5666` |

Production model set last completed:

- `2026-05-08 11:39:31 America/New_York`
- The XGBoost model is the latest completed member of the current three-model set.

## Artifact Timestamp Rules

For each run, `final_models.json` is the deployable model index. The individual model path is stored in each record's `artifact_path`.
Committed production manifests use portable paths relative to the run directory, such as `models/MODEL_FILE.pkl`. `scripts/predict_v1_supervised_baselines.py` also falls back to `RUN_DIR/models/MODEL_FILE.pkl` for older local manifests that still contain absolute paths.

Existing May 2026 artifacts contain `generated_utc`, but that value was created near the beginning of the training run. For these existing artifacts, use the model `.pkl` file write time above as the completion timestamp.

Future retrains write `trained_at_utc` into:

- `final_models.json`
- `trained_models.json`
- `final_classification_models.json`
- `trained_classification_models.json`

After promoting a newly retrained model set, update this file's table with:

- run directory
- artifact path or run directory
- last trained/saved timestamp
- OOS metrics
- any caveats, such as chunked XGBoost training

## Prediction Refresh Policy

Refreshing market data does not require retraining.

Normal production cycle:

1. Refresh latest EODHD data after market close when fresh predictions are needed.
2. Rebuild prediction-ready features/windows using the existing PIT-safe schema.
3. Run the saved final classifier bundles.
4. Rank eligible target-pending stock-window episodes.
5. Produce trading reports for human review.

Use saved models for daily or every-few-days prediction refreshes as long as:

- target definition is unchanged
- feature schema is unchanged
- EODHD vendor semantics are unchanged
- ticker/window eligibility policy is unchanged
- new rows can be built with the same point-in-time rules

Run latest predictions separately for each current model directory, for example:

```powershell
py -3.11 scripts\predict_v1_supervised_baselines.py `
  --run-dir artifacts\v1_baselines\eodhd_true_full_xgboost `
  --dataset-root data\eodhd_us_equities_30y `
  --output-file artifacts\production_predictions\latest_xgboost.csv

py -3.11 scripts\predict_v1_supervised_baselines.py `
  --run-dir artifacts\v1_baselines\eodhd_true_full_torch_mlp `
  --dataset-root data\eodhd_us_equities_30y `
  --output-file artifacts\production_predictions\latest_torch_mlp.csv

py -3.11 scripts\predict_v1_supervised_baselines.py `
  --run-dir artifacts\v1_baselines\eodhd_true_full_torch_seq_static `
  --dataset-root data\eodhd_us_equities_30y `
  --output-file artifacts\production_predictions\latest_torch_seq_static.csv
```

Use a bounded prediction dataset root instead of the full EODHD root if the full processed feature table is too large for the prediction machine.

## Retraining Policy

Default retrain cadence:

- Retrain every `2` to `4` weeks.
- Monthly is the default operating cadence.
- Use the shorter two-week cadence during volatile regimes, after material data additions, or while the strategy is still being monitored closely.

Do not retrain just because a new daily bar arrived. New data should usually be scored by the saved final bundles.

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
py -3.11 scripts\run_v1_pipeline.py --profile eodhd_true_full_torch_seq_static
py -3.11 scripts\run_v1_pipeline.py --profile eodhd_true_full_torch_mlp

$env:V1_XGBOOST_TRAINING_MODE = 'chunked'
$env:V1_XGBOOST_DEVICE = 'cpu'
$env:V1_XGBOOST_NTHREAD = '8'
$env:V1_XGBOOST_CHUNK_ROWS = '1048576'
py -3.11 scripts\run_v1_pipeline.py --profile eodhd_true_full_xgboost
```

Only promote the new models after confirming:

- all final model manifests exist
- all model `.pkl` artifacts exist
- OOS metrics are acceptable versus the previous production set
- `latest_predictions.csv` generation succeeds on the refreshed prediction dataset

## XGBoost Caveat

The current `xgboost_classifier` production artifact was trained with the chunked fallback path because one-shot full-fold XGBoost stalled at the 31M-row fold size.

All rows were used, but not every tree saw every row. The OOS leaderboard is the acceptance evidence for this candidate. Keep this caveat attached to the model until a cleaner full-batch or staged/subsampled XGBoost candidate is validated.
