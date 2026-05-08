# Daily Trading Signal and Position Review Script

## Purpose

Build an automated daily or every-few-days script that refreshes market data, preprocesses features, runs saved prediction models, ranks current stock-window episodes, suggests new entries, and reviews existing open positions.

The script should not place trades automatically. It should produce an entry/review report for human decision-making.

## High-Level Strategy

The model predicts whether a stock-window episode is likely to achieve a strong positive move within the forward prediction window. The strategy uses model output primarily as a ranking signal.

Use the model to identify high-confidence candidates, but keep actual trading rules simple and conservative.

Core principle:

- Top-ranked model agreement is mainly an entry filter.
- Price target, stop-loss, and holding-period rules are the hard exit rules.
- Signal deterioration after entry is a warning/review trigger, not always an automatic sell trigger.

## Daily Workflow

Each run should do the following:

1. Refresh most recent data using the existing project data-refresh pipeline.
2. Preprocess/build the latest prediction-ready dataset using the existing project schema and point-in-time rules.
3. Load the saved final deployment model bundles.
4. Generate latest predictions for all eligible target-pending stock-window episodes.
5. Rank all eligible stocks by model confidence.
6. Identify top-ranked candidates.
7. Check agreement across available production models.
8. Produce an entry-candidate report.
9. Load the current open-position ledger.
10. Review previously entered positions against updated model rank and trade-management rules.
11. Produce a position-review report.

## Data and Model Assumptions

Follow all existing project MD instructions and repo conventions.

Use the current production classification setup.

The current production model registry is `PRODUCTION_MODELS.md`. It lists the deployed classifier bundles, their OOS metrics, and their last trained/saved timestamps.

Refreshing data is separate from retraining models:

- Prediction refresh: update latest data and score target-pending windows with saved final model bundles.
- Model retrain: rebuild training panels/caches and fit new deployable bundles.

Do not retrain models every time new market data arrives. Use saved models for normal daily or every-few-days prediction refreshes as long as the target, feature schema, eligibility filter, and vendor semantics are unchanged.

Default retrain cadence:

- Monthly by default.
- Every 2 weeks during volatile regimes, after material data additions, or during closer live-monitoring periods.
- Immediately after target, feature schema, data semantics, leakage, or material live-drift changes.

Do not expose proprietary model internals in output reports. Reports should only show practical trading information such as ticker, date, current price, rank bucket, entry suggestion, review status, and action reason. You may print the model prediction score (the final output or probability percentage)

Do not print or save sensitive algorithm details, full feature vectors, model architecture internals, or raw proprietary scoring logic in user-facing trading reports.

## Current Production Models

Use these full-universe classifier bundles until a new retrained set is promoted:

| Model | Run directory | Last trained/saved local time | OOS PR AUC | OOS ROC AUC | Top-decile precision | OOS accuracy |
|---|---|---|---:|---:|---:|---:|
| `xgboost_classifier` | `artifacts/v1_baselines/eodhd_true_full_xgboost` | `2026-05-08 11:39:31 America/New_York` | `0.6048` | `0.6371` | `0.6411` | `0.5974` |
| `torch_mlp_classifier` | `artifacts/v1_baselines/eodhd_true_full_torch_mlp` | `2026-05-07 16:27:54 America/New_York` | `0.5934` | `0.6172` | `0.6221` | `0.5700` |
| `torch_seq_static_classifier` | `artifacts/v1_baselines/eodhd_true_full_torch_seq_static` | `2026-05-06 23:42:46 America/New_York` | `0.5711` | `0.6095` | `0.6066` | `0.5666` |

The OOS positive baseline for the current full-run test folds is `50.25%`. Treat model output primarily as a ranking signal, with extra emphasis on top-decile precision and cross-model agreement.

### Prediction Refresh

After a data refresh, run predictions from the saved bundles. This does not retrain the models.

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

If the full EODHD root is too large for the prediction machine, use a bounded prediction dataset root that preserves the same feature schema and point-in-time rules.

### Retrain And Promote

When retraining is due, rerun the full classifier profiles, compare the new OOS metrics with `PRODUCTION_MODELS.md`, and promote only if the new artifacts are complete and metrics are acceptable.

Future training manifests write `trained_at_utc`. Existing May 2026 artifacts did not have that field, so `PRODUCTION_MODELS.md` records their model file save timestamps.

## Entry Candidate Logic

For each latest prediction date:

1. Score all eligible stocks.
2. Rank stocks by model confidence.
3. Determine rank percentile or rank bucket.

Preferred entry universe:

- Strong entry candidates: top 3% to 5%.
- Watchlist candidates: top 5% to 10%.
- Ignore: below top 10%.

If multiple production models are available, calculate model agreement.

Suggested simple agreement logic:

- Strong agreement: at least two production models rank the stock in the top decile.
- Very strong agreement: at least two production models rank the stock in the top 5%, or the ensemble rank is in the top 3% to 5%.
- Weak/no agreement: only one model likes the stock or ensemble rank is below top decile.

Entry recommendation:

- Suggest "ENTRY CANDIDATE" only if the stock is high-ranked and has sufficient model agreement.
- Suggest "WATCHLIST" if the stock is top decile but not high-conviction enough.
- Do not suggest entry below top decile.

Keep output simple:

| ticker | prediction_date | rank_bucket | agreement_status | suggested_action | reason |

Example actions:

- ENTRY CANDIDATE
- WATCHLIST
- IGNORE

Example reasons:

- Top 5% with model agreement
- Top decile but weak agreement
- Below top decile

## Position Ledger Input

The script should read a simple open-position ledger.

Required fields:

- ticker
- entry_date
- entry_price
- shares
- current_status
- optional notes

Optional but useful fields:

- target_profit_pct
- stop_loss_pct
- max_hold_days

Default trade-management settings:

- target profit: +5%
- stop loss: -5% by default
- max holding period: 20 trading days
- review threshold: falls below top 20%
- stronger warning threshold: falls below top 30% or loses model agreement

## Open Position Review Logic

For each open position:

1. Update current price.
2. Calculate current raw return.
3. Calculate days held.
4. Check whether the stock still appears in the latest prediction ranking.
5. Determine latest rank bucket.
6. Determine whether model agreement still exists.
7. Generate an action/review signal.

Hard exit rules:

- SELL - TARGET if current return >= +5%.
- SELL - STOP if current return <= -5%.
- SELL - TIME if days held >= 20 trading days.

Soft review rules:

- REVIEW if the position falls out of the top 20%.
- REVIEW STRONGLY if the position falls out of the top 30%.
- REVIEW if model agreement disappears.
- REVIEW if signal weakens and current return is flat or negative.
- REVIEW if signal weakens and days held > 10.

Do not automatically mark SELL only because a stock falls out of the top decile.

Preferred logic:

- Top decile after entry: HOLD if no hard exit is triggered.
- Top 10% to 20% after entry: HOLD / REVIEW.
- Below top 20% after entry: REVIEW.
- Below top 30% after entry: STRONG REVIEW.
- Weak signal + negative return: CONSIDER EXIT.
- Weak signal + days held > 10: CONSIDER EXIT.
- Hit target, stop, or time limit: SELL.

Suggested action categories:

- HOLD
- REVIEW
- STRONG REVIEW
- CONSIDER EXIT
- SELL - TARGET
- SELL - STOP
- SELL - TIME

## Recommended Decision Hierarchy

Use this hierarchy when generating position-review output:

1. If current return >= target profit:
   - action = SELL - TARGET

2. Else if current return <= stop loss:
   - action = SELL - STOP

3. Else if days held >= max hold days:
   - action = SELL - TIME

4. Else if latest rank is below top 30%:
   - action = STRONG REVIEW

5. Else if latest rank is below top 20%:
   - action = REVIEW

6. Else if model agreement disappeared and current return <= 0:
   - action = CONSIDER EXIT

7. Else if model agreement disappeared and days held > 10:
   - action = CONSIDER EXIT

8. Else:
   - action = HOLD

## Important Interpretation

Top decile should be treated mostly as an entry condition.

Falling out of top decile should not automatically close the trade. The original prediction has a forward window, so the trade should be given time to work unless price action, time, or major signal deterioration suggests otherwise.

Use updated model ranking as a review tool, not as a high-turnover automatic exit engine.

## Output Files

The script should produce at least two CSV or Excel outputs.

### 1. Entry Candidates Report

Suggested columns:

- run_date
- prediction_date
- ticker
- rank_bucket
- agreement_status
- suggested_action
- reason
- current_price

### 2. Open Position Review Report

Suggested columns:

- run_date
- ticker
- entry_date
- entry_price
- current_price
- current_return_pct
- days_held
- latest_rank_bucket
- agreement_status
- review_action
- reason

### 3. Optional Combined Summary

Suggested fields:

- number of entry candidates
- number of watchlist names
- number of open positions
- number of HOLD positions
- number of REVIEW positions
- number of CONSIDER EXIT positions
- number of SELL signals

## Privacy and IP Protection

Do not expose algorithm details in trading reports.

Avoid outputting:

- raw model internals
- full feature vectors
- proprietary model architecture details
- detailed scoring formulas
- embeddings
- neighbor/retrieval internals unless explicitly requested for private debugging

Use simple output labels:

- rank_bucket
- agreement_status
- suggested_action
- reason

## Safety Guardrails

The script should only suggest actions. It should not place brokerage orders.

The human trader will decide whether to buy, hold, or sell.

The strategy is experimental and should be monitored through realized trade logs before increasing position size.

## Default Parameters

Use these defaults unless overridden by config:

```yaml
entry_top_percent: 5
watchlist_top_percent: 10
review_rank_threshold_percent: 20
strong_review_rank_threshold_percent: 30
target_profit_pct: 5
stop_loss_pct: -5
max_hold_days: 20
min_model_agreement_count: 2
suggest_trades_only: true
place_orders: false
```
