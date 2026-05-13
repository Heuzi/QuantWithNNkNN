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

1. Determine the latest local EODHD market-bar date from the latest-inference manifest, falling back to the raw fetch manifest.
2. Fetch only missing recent EOD OHLCV bars after that date.
3. Prefer EODHD's bulk EOD endpoint for whole-exchange daily bars. Fall back to per-symbol EOD only for missing context tickers or endpoint failures.
4. Do not refresh fundamentals or sentiment during the normal daily run. Existing saved fundamentals and sentiment are joined from local raw files.
5. Build/update the compact latest-inference cache under `data/eodhd_us_equities_30y/processed/latest_inference/`.
6. Load the promoted final deployment `path_5pct_20d` model bundles.
7. Generate latest predictions for all eligible target-pending stock-window episodes.
8. Apply the conservative research universe filter to the latest prediction universe.
9. Rank only stocks that pass the conservative research universe by the default multiclass score `pred_score_path_5pct_20d = P(class 2) - P(class 0)`.
10. Identify top-ranked candidates.
11. Check agreement across available production models.
12. Produce entry-candidate, watchlist, ranked-prediction, model-agreement, and research-universe diagnostics reports.
13. Load the current open-position ledger, if present.
14. Review previously entered positions against updated model rank and trade-management rules.
15. Produce a position-review report.

## Data and Model Assumptions

Follow all existing project MD instructions and repo conventions.

Use the intended production classification setup based on `path_5pct_20d`.

`PRODUCTION_MODELS.md` is the source of truth for promotion status. Under the current repo state, there is no active promoted multiclass production set until the three supported classifiers are retrained and promoted under `path_5pct_20d`.

Refreshing data is separate from retraining models:

- Prediction refresh: update latest data and score target-pending windows with promoted final `path_5pct_20d` model bundles.
- Model retrain: rebuild training panels/caches and fit new deployable bundles.

Do not retrain models every time new market data arrives. After the multiclass retrain/promotion step is complete, use the saved promoted models for normal daily or every-few-days prediction refreshes as long as the target, feature schema, eligibility filter, and vendor semantics are unchanged.

The same feature semantics are used for both retraining and latest prediction, but the operating scale differs:

- Full retrain uses the full EODHD root, rebuilds `processed/daily_features.csv`, rebuilds or reuses `processed/daily_features_normalized.csv`, materializes the filtered strategy-universe panel, materializes the shared episode cache, and then trains the three supported classifiers.
- Routine latest prediction uses the bounded cache under `processed/latest_inference/` and should not rescan the full historical feature CSV or retrain models.
- When routine prediction fetches new EOD bars and computes latest features, it also persists the newly computed stock/context feature rows into processed incremental sidecars: `processed/daily_features_incremental_updates.csv` and `processed/market_context_features_incremental_updates.csv`.
- The true-full retrain wrapper consolidates those sidecars into `processed/daily_features.csv` and `processed/market_context_features.csv` before normalization. This lets future retrains reuse daily prediction-time feature work instead of recomputing every already-processed recent row from raw bars.
- A completed full-panel normalized artifact is reusable. `processed/daily_features_normalized.csv` plus `processed/daily_features_normalized_manifest.json` are the completion markers.
- If sidecar consolidation changes only recent dates and a normalized artifact already exists, `scripts/merge_incremental_feature_updates.py` recomputes same-date normalization only for those touched dates and merges them into `processed/daily_features_normalized.csv`. If the normalized artifact is missing, the full out-of-core normalizer builds it once before train/test panel materialization.

Conservative research universe filtering is part of the standard strategy-universe policy. By default it is applied to train/test/live so the classifiers are fit and evaluated on the same larger-cap, more tradable, more stable name set that the trading reports rank.

Default retrain cadence:

- Monthly by default.
- Every 2 weeks during volatile regimes, after material data additions, or during closer live-monitoring periods.
- Immediately after target, feature schema, data semantics, leakage, or material live-drift changes.

Do not expose proprietary model internals in output reports. Reports should only show practical trading information such as ticker, date, current price, rank bucket, entry suggestion, review status, and action reason. You may print the model prediction score (the final output or probability percentage)

Do not print or save sensitive algorithm details, full feature vectors, model architecture internals, or raw proprietary scoring logic in user-facing trading reports.

## Promotion State

No active promoted multiclass production set exists yet.

The previous binary promoted set is deprecated and must not be used as the production recommendation set. The intended promoted set after retraining remains:

- `xgboost_classifier`
- `torch_mlp_classifier`
- `torch_seq_static_classifier`

Those three models are the only supported `path_5pct_20d` production classifiers. They must be retrained and promoted before this strategy is used as the active production workflow.

The daily strategy commands remain the same after retraining and promotion. The default production ranking score is `pred_score_path_5pct_20d = P(class 2) - P(class 0)`, while `pred_prob_path_5pct_20d` remains the raw class-2 probability column.

### Prediction Refresh

After a data refresh, run predictions from the promoted multiclass bundles. This does not retrain the models and should not scan the full 34GB `processed/daily_features.csv`.

Default daily command:

```powershell
py -3.11 scripts\run_trading_strategy.py `
  --dataset-root data\eodhd_us_equities_30y `
  --credentials-path EODHD_api_key
```

The script writes the bounded latest-window feature cache to:

- `data/eodhd_us_equities_30y/processed/latest_inference/recent_stock_bars.csv`
- `data/eodhd_us_equities_30y/processed/latest_inference/recent_context_bars.csv`
- `data/eodhd_us_equities_30y/processed/latest_inference/latest_daily_features.csv`
- `data/eodhd_us_equities_30y/processed/latest_inference/latest_market_context_features.csv`
- `data/eodhd_us_equities_30y/processed/latest_inference/prediction_windows.csv`
- `data/eodhd_us_equities_30y/processed/latest_inference/run_manifest.json`

The report folder is timestamped under `artifacts/production_reports/` and includes:

- `entry_candidates.csv`
- `watchlist.csv`
- `all_ranked_predictions.csv`
- `research_universe_diagnostics.csv`
- `model_agreement_summary.csv`
- `position_review.csv`
- `all_model_predictions.csv`
- `run_manifest.json`
- `summary.md`

For a local-only run without new API calls:

```powershell
py -3.11 scripts\run_trading_strategy.py `
  --dataset-root data\eodhd_us_equities_30y `
  --skip-fetch
```

Monitor a running strategy process:

```powershell
py -3.11 scripts\monitor_trading_strategy_progress.py `
  --pid <PID> `
  --latest-inference-dir data\eodhd_us_equities_30y\processed\latest_inference `
  --report-dir artifacts\production_reports\<RUN_FOLDER> `
  --watch `
  --poll-seconds 30
```

The runner also writes machine-readable progress files:

- `data/eodhd_us_equities_30y/processed/latest_inference/progress.json`
- `artifacts/production_reports/<RUN_FOLDER>/progress.json`

For a smoke test on a small ticker subset:

```powershell
py -3.11 scripts\run_trading_strategy.py `
  --dataset-root data\eodhd_us_equities_30y `
  --skip-fetch `
  --max-tickers 100 `
  --report-name smoke_latest_inference
```

Daily API-efficiency rule:

- Use whole-exchange EODHD bulk EOD calls first. EODHD documents whole-exchange bulk EOD as 100 API-call units per date, while symbol-filtered bulk adds 1 unit per ticker.
- Fetch only missing dates after the local latest-inference date.
- Rank only current windows by default. The runner keeps tickers whose latest anchor is within 3 calendar days of the newest available anchor date, which avoids stale/delisted symbols appearing as current trading candidates.
- The conservative research universe defaults to common stocks on `NYSE`, `NASDAQ`, or `AMEX`, at least `$10` adjusted close, at least `252` recent trading rows, at least `$10M` 20-day and 60-day median dollar volume, at most `2%` zero-volume days over the last `60` sessions, current dollar volume at least `20%` of the 20-day median, close above the 200-day SMA, 50-day SMA above the 200-day SMA, 6-month return no worse than `-15%`, no worse than `35%` below the trailing 252-day high, and no recent 60-day spike beyond configured limits.
- Exclude exchange test symbols such as `ZVZZT`, `ZWZZT`, and `NTEST` by default.
- Reuse the bounded latest-inference feature cache when no new EOD bars are fetched. Use `--force-rebuild-latest-inference` only when the cache needs to be rebuilt.
- Do not refetch the full universe, full history, fundamentals, or sentiment during normal prediction refresh.

### Retrain And Promote

When retraining is due, rerun the full classifier profiles under `path_5pct_20d`, compare the new OOS metrics with the accepted benchmark or review threshold, and promote only if the new artifacts are complete and metrics are acceptable.

Future training manifests should write `trained_at_utc` and record `classification_event_type=path_5pct_20d` in the run metadata before promotion.

Recommended true-full retrain wrapper:

```powershell
.\scripts\run_full_universe_retrain.ps1 -Resume
```

That wrapper runs the intended large-data sequence: raw EODHD refresh, chunked daily-feature rebuild, incremental feature sidecar consolidation, normalized-feature refresh, filtered panel materialization, episode-cache materialization, and the three model-specific training profiles. Use `-Resume` after an interruption so completed stages are skipped. The normalized stage is skipped only after both `data\eodhd_us_equities_30y\processed\daily_features_normalized.csv` and `data\eodhd_us_equities_30y\processed\daily_features_normalized_manifest.json` exist and are at least as fresh as `processed\daily_features.csv`; sidecar consolidation updates those normalized files incrementally for touched dates when possible.

## Entry Candidate Logic

For each latest prediction date:

1. Score all eligible stocks.
2. Keep only stocks that pass the conservative research universe.
3. Rank stocks by `pred_score_path_5pct_20d = P(class 2) - P(class 0)`.
4. Determine rank percentile or rank bucket.

For `path_5pct_20d`, rank by `pred_score_path_5pct_20d = P(class 2) - P(class 0)`. Keep `pred_prob_path_5pct_20d` as the raw class-2 probability for inspection. The multiclass prediction surface may also include:

- `pred_prob_path_5pct_20d_class_0`
- `pred_prob_path_5pct_20d_class_1`
- `pred_prob_path_5pct_20d_class_2`
- `pred_class_path_5pct_20d`
- `pred_score_path_5pct_20d`

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
