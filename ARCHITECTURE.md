# Architecture

## Purpose
This document defines the preferred model roadmap for the quant return-prediction project.

The project predicts the future return of a **stock-window episode** and optionally supplements the prediction with **NN-kNN retrieval-based case support**.

## Core task
Given:
- a ticker
- an anchor date
- the prior 60 trading days of aligned features
- as-of static features known by the anchor date

Predict:
- next 5-day or 10-day return
- preferably market-adjusted or sector-adjusted return

## Modeling principle
Start with the simplest architecture that:
1. respects time order
2. handles multimodal inputs cleanly
3. produces a fused representation suitable for NN-kNN retrieval
4. can be trained and debugged reliably on real historical data

Do not begin with full RL or aggressive architecture complexity.

## Recommended progression

### Version 1: supervised baseline architecture
Use this first.

Components:
1. **Market/time-series encoder**
   - input: rolling sequence of daily aligned numeric features
   - default window: 60 trading days
   - tokenization: **one token per day**
   - each day token contains that day's aligned numeric inputs

2. **Static/tabular encoder**
   - input: slower-moving or non-sequential features
   - examples: sector, industry, company type, selected fundamentals snapshot, metadata

3. **Fusion head**
   - concatenate or gated-fuse the stock encoder, explicit context encoder, and static encoder representations

4. **Prediction head**
   - output a vector of market-adjusted return targets
   - default horizons: 1, 5, 10, and 20 trading days
   - optional support for uncertainty/confidence later

5. **Optional NN-kNN retrieval head**
   - operate on the fused representation
   - retrieve similar **stock-window episodes**
   - expose scores/weights and top neighbors

Why this is the default:
- easy to debug
- easy to ablate
- compatible with baselines
- compatible with NN-kNN
- keeps multimodal design without overcomplicating early experiments

V1 implementation note:
- train multiple baseline model families, not only one winner
- save all trained model artifacts and their metrics
- rank models on a leaderboard, mainly by 5/10/20-day rank IC and portfolio spread
- run all saved models for latest prediction windows so users can compare agreement and disagreement

Current implementation snapshot:
- Default daily market data source is now EODHD, targeting `data/eodhd_us_equities_30y`.
- Massive-era datasets and model artifacts are legacy and should not be compared directly against EODHD full-universe results.
- Full-universe EODHD raw collection uses `scripts/update_eodhd_daily_dataset.py --fetch-only --max-tickers 0`; per-ticker daily features can be generated through the standard profile runner or the true-full wrapper. The existing in-memory processed rebuild path is reserved for smoke/pilot panels.
- Full-panel cross-sectional normalization is handled out of core by `scripts/build_normalized_features_from_processed.py`. It streams `processed/daily_features.csv` into complete calendar-month buckets, computes same-date full-panel and same-sector transforms for each month, then regroups the final `processed/daily_features_normalized.csv` back into ticker/date order for downstream panel materialization. The completed normalized CSV and `daily_features_normalized_manifest.json` are saved and reused by resume-aware full retrain runs.
- Standard walk-forward experiments should use JSON profiles under `configs/v1_runs/` through `scripts/run_v1_pipeline.py`, rather than ad hoc long command strings. Profiles can include a `materialize_panel` stage so the trainer consumes a bounded dataset root under `data/eodhd_training_panels/` instead of loading the full 34GB feature CSV.
- `scripts/run_v1_pipeline.py` now writes stage state under `logs/v1_pipeline_state/`, supports `--resume` to skip stages already marked completed, and requests Windows keep-awake by default during non-dry-run execution so long jobs are less likely to die from sleep/power-saving behavior.
- Full and near-full EODHD experiments should also use the `materialize_cache` stage. It creates an episode-level cache with float32 tabular matrices and memmapped sequence arrays so feature engineering runs once and model/fold training reads cached arrays.
- `scripts/train_v1_supervised_baselines.py` now defaults to expanding-window walk-forward evaluation.
- The code still supports regression and event classification, but the current V1 production direction is classification-only.
- Regression targets are retained for diagnostics and research ablations, not for the standard full EODHD walk-forward run.
- The intended production classification label is `path_5pct_20d`.
- `path_5pct_20d` is a 3-class path label over the next 20 trading days using close-to-close forward returns from `anchor_date`:
  - class `0`: any forward path drawdown is `<= -5%`
  - class `2`: the path reaches `>= +5%` without ever breaching `-5%`
  - class `1`: otherwise neutral
- The full 20-trading-day forward window is required. Rows without it are unlabeled and excluded from supervised training and evaluation.
- The standard full profile `configs/v1_runs/eodhd_full_walk_forward.json` trains only the selected classifier candidates from model selection:
  - `torch_seq_static_classifier` with `stock_relative_market_sector_sentiment_sequence`
  - `torch_mlp_classifier` with `stock_relative_market_sector_fundamentals_sentiment`
  - `xgboost_classifier` with `stock_relative_market_sector_fundamentals_sentiment`
- The trainer can still run `regression`, `classification`, or `both` for experiments, but production runs should use `task_type=classification`.
- The current baseline suite includes tabular baselines plus `torch_seq_static`.
- Regression tabular baselines include `zero`, `mean`, `momentum_heuristic`, `ridge`, `elastic_net`, `lightgbm`, `xgboost`, `sklearn_hist_gb`, `sklearn_mlp`, and `torch_mlp`.
- Classification tabular baselines include `logistic_regression`, `elastic_net_classifier`, `lightgbm_classifier`, `xgboost_classifier`, `sklearn_mlp_classifier`, and `torch_mlp_classifier`.
- `torch_seq_static` uses a real 60-day sequence branch plus static categorical embeddings for `gics_sector` and `gics_sub_industry`.
- Sequence-static torch baselines use a bounded CPU-feasible default training budget (`max_epochs=20`, `patience=4`, batch size `512`) so full walk-forward ablations can complete without a GPU.
- The sequence-static model supports component ablations over stock-only, relative stock, market context, and sector ETF context sequence tokens, including `stock_relative_market_sector_sequence`.
- Compact sequence feature profiles are available for lower-variance ablations, including `stock_relative_market_sector_compact_sequence`.
- Sequence inputs have shape `[batch_size, window_length, features_per_day]`. The default `window_length` is 60 trading days, but training can override it with `--window-length`.
- Tabular inputs have shape `[episode_count, flattened_feature_count]` and use rolling-window summary columns such as `__last`, `__mean60`, and `__std60`.
- `torch`, `xgboost`, and `lightgbm` now prefer GPU execution when available and fall back to CPU otherwise; torch models require the local PyTorch build to report `torch.cuda.is_available()`, while XGBoost/LightGBM may attempt vendor GPU paths and record any fallback error.
- Full-run training now also uses more CPU by default when no explicit overrides are set:
  - XGBoost defaults `V1_XGBOOST_NTHREAD` to nearly the host CPU count instead of `1`
  - torch dataloaders default `V1_TORCH_NUM_WORKERS` to a small positive worker count capped for host-memory stability
- Training logs now include human-readable progress bars in addition to JSON events:
  - trainer-level bars across model combos, folds, and final deploy fits
  - batch/epoch bars for `torch_mlp_classifier` and `torch_seq_static_classifier`
  - round/chunk bars for cache-backed `xgboost_classifier`
- V1 training, walk-forward testing, latest inference, and trading reports apply the shared broad episode eligibility filter first: listed common-stock universe upstream, then as-of 60-day default history, at least 55 valid adjusted OHLCV rows, 60-day average dollar volume, adjusted close price, and exchange allowlist checks at `anchor_date`.
- A stricter conservative research universe then defines the actual strategy universe used for train/test/live by default. This shared gate keeps the recommendation universe focused on larger, more stable, more tradable names:
  - major exchanges only: `NYSE`, `NASDAQ`, `AMEX`
  - common stocks only
  - at least `252` trading rows of history
  - adjusted close at least `$10`
  - 20-day and 60-day median dollar volume at least `$10M`
  - at most `2%` zero-volume days over the last `60` trading days
  - current dollar volume at least `20%` of the 20-day median dollar volume
  - close above the 200-day moving average
  - 50-day moving average above the 200-day moving average
  - 6-month return no worse than `-15%`
  - no worse than `35%` below the trailing 252-day high
  - optional recent spike filter on 60-day max 1-day return and true-range behavior
- Training and testing mechanics otherwise stay the same under `path_5pct_20d`: the same stock-window episode construction, the same cache-backed materialization path, the same expanding-window walk-forward protocol with purge gap, the same final refit flow, and the same latest-inference construction. The universe policy now applies at panel materialization, cache construction, train/test fold membership, and latest inference; it changes which rows enter the dataset but does not change fold construction or label-generation mechanics beyond removing out-of-universe rows.
- Only `xgboost_classifier`, `torch_mlp_classifier`, and `torch_seq_static_classifier` support `path_5pct_20d`. Other classification baselines remain binary-only and are unsupported for this label mode.
- For `path_5pct_20d`, only the output layer and prediction surface change:
  - `xgboost_classifier` returns 3-class probabilities via `multi:softprob`
  - `torch_mlp_classifier` and `torch_seq_static_classifier` use 3-logit output heads with cross-entropy training
  - prediction outputs include `pred_prob_path_5pct_20d_class_0`, `pred_prob_path_5pct_20d_class_1`, `pred_prob_path_5pct_20d_class_2`, `pred_class_path_5pct_20d`, `pred_prob_path_5pct_20d` for class-2 probability, and `pred_score_path_5pct_20d = P(class 2) - P(class 0)` for default ranking
- EODHD Fundamentals v1.1 and daily sentiment are supported as optional enrichment sources. For each `anchor_date`, fundamentals use the latest filing/public record with `availability_date <= anchor_date`; fiscal period end alone is not enough. Sentiment is lagged one trading row by default.
- Raw identifiers such as `ticker`, `eodhd_symbol`, ISIN, CIK/CUSIP/FIGI, and company name are metadata only and are rejected from model feature columns.
- Latest inference uses final deployment bundles saved after the walk-forward run completes. Scoring a new target-pending stock-window episode does not require retraining as long as the required features can be built.
- Raw identifiers remain metadata at inference time. A novel ticker can be scored if it exists in the prediction dataset root, passes the same episode eligibility filter, and has enough prior window data. Unseen sector or industry categories map to unknown id `0` in static categorical encoders.
- `PRODUCTION_MODELS.md` tracks production promotion state. The previous binary classifier set is deprecated, and no promoted multiclass production set exists until the three supported classifiers are retrained under `path_5pct_20d`.
- Routine data refreshes should score latest target-pending windows with saved final bundles. Do not retrain just because a new daily bar arrived.
- Production retraining cadence is every 2 to 4 weeks, with monthly as the default. Retrain immediately only after target, feature schema, universe policy, vendor semantics, leakage, or material live/OOS monitoring changes.
- Future training indexes write `trained_at_utc`; existing May 2026 production artifacts should use the model file save timestamp recorded in `PRODUCTION_MODELS.md`.
- True-full retraining should use `scripts/run_full_universe_retrain.ps1` rather than hand-running the individual large stages. The wrapper refreshes raw EODHD bars, resumes chunked feature construction, consolidates latest-prediction feature sidecars, incrementally refreshes or builds the saved full-panel normalized feature artifact, materializes the filtered strategy-universe panel, materializes the shared episode cache, and then trains the three supported `path_5pct_20d` classifiers. With `-Resume`, completed stage outputs are skipped, including `processed/daily_features_normalized.csv` plus `processed/daily_features_normalized_manifest.json`.
- Routine prediction refreshes persist newly computed latest feature rows into incremental processed-feature sidecars. The true-full retrain wrapper consolidates those sidecars before normalization so retraining can reuse daily prediction-time feature processing instead of recomputing the same recent rows from raw bars.
- Cache-backed training path:
  - `scripts/materialize_v1_episode_cache.py` streams a ticker-contiguous daily feature CSV and writes `episode_metadata.csv`, `targets.csv`, tabular `.npy` matrices, sequence `.npy` row stores, date arrays, and a manifest.
  - `scripts/train_v1_supervised_baselines.py --episode-cache-dir ...` loads metadata/targets plus memory-mapped arrays instead of loading/rebuilding tabular feature frames from daily CSV.
  - `torch_seq_static_classifier` gets a store whose `get_window(...)` slices from memmap at batch time.
  - `torch_mlp_classifier` iterates cached tabular batches directly.
  - `xgboost_classifier` uses the cached float32 tabular matrix; external-memory `QuantileDMatrix` remains a later optimization if the cached split matrices become too large.
- Standard run-size taxonomy:
  - Smoke runs validate code and data plumbing only. They use tiny ticker/date/episode limits, short torch budgets, and are not performance evidence.
  - `eodhd_full_walk_forward` is the current serious benchmark profile. It uses the full EODHD source root but materializes a bounded high-liquidity panel, currently capped at 1,500 tickers, 2014-01-01 through 2026-04-24, 500,000 most recent eligible episodes, and 6 walk-forward folds. This is the required engineering gate before any multi-day run.
  - A true full-universe run is the eventual large experiment. It should use `max_tickers=0`, the intended long history, the same cache-backed training path, and no small benchmark episode cap unless a cap is explicitly chosen for compute control. It is not the current default profile.
  - `eodhd_model_selection_walk_forward` remains an intermediate research profile for comparing more model families before changing the selected top-three classifiers.

## Input organization

### Daily aligned sequence features
Each day in the rolling window may include:
- OHLCV-derived features
- returns
- realized volatility
- volume changes
- technical indicators
- price-vs-moving-average features
- log-scaled liquidity features
- same-date cross-sectional normalized features
- same-date same-sector relative features
- daily sentiment/news aggregates
- fundamentals carried forward only if already public by that date

Implemented V1 sequence feature sets:
- `stock_only_sequence`: stock daily features only.
- `stock_relative_sequence`: stock daily features plus same-date full-panel and same-sector relative stock features.
- `stock_market_sequence`: stock daily features plus `SPY` context features.
- `stock_sector_sequence`: stock daily features plus mapped sector ETF context features.
- `stock_market_sector_sequence`: stock daily features plus `SPY` and mapped sector ETF context features.
- `stock_relative_market_sequence`: stock daily features plus relative stock features and `SPY` context.
- `stock_relative_sector_sequence`: stock daily features plus relative stock features and mapped sector ETF context.
- `stock_relative_market_sector_sequence`: stock daily features plus relative stock features, `SPY` context, and mapped sector ETF context.
- `stock_sentiment_sequence`: stock daily features plus lagged daily sentiment features.
- `stock_relative_market_sector_sentiment_sequence`: relative stock features plus `SPY`, sector ETF, and lagged sentiment context.

Each sequence set also has a compact counterpart, for example `stock_relative_market_sector_compact_sequence`. Compact sequence profiles keep the same component structure but use a smaller feature list, dropping obvious duplicate or highly correlated fields such as exact momentum aliases, percentile-rank counterparts, and some overlapping liquidity columns.

For compatibility, the non-`_sequence` names can also be used by sequence models when the name is listed in `SEQUENCE_FEATURE_SET_NAMES`. Tabular models use the flattened feature sets described below.

Typical daily sequence token contents:
- stock features such as `return_1d`, `log_return_1d`, `rolling_return_20d`, `rolling_vol_20d`, and `log1p_volume`
- Priority A daily shape/liquidity/regime features such as `close_location`, `true_range_pct`, `dollar_volume_ratio_5d`, `volume_zscore_20d`, `stock_vs_market_return_1d`, and `stock_vs_sector_return_5d`
- relative features such as `return_1d__cs_z`, `log1p_volume__cs_pct`, and `rolling_vol_20d__sector_cs_z`
- market context features such as `market_context_return_1d`, `market_context_rolling_vol_20d`, and `market_context_missing`
- sector context features such as `sector_context_return_1d`, `sector_context_rolling_vol_20d`, and `sector_context_missing`

The sequence model receives raw daily tokens and must learn useful temporal summaries itself. This is intentionally different from tabular baselines, which receive precomputed rolling summaries.

### Flattened tabular episode features
Tabular baselines receive one row per `(ticker, anchor_date)` episode.

Implemented V1 tabular feature sets:
- `stock_only`
- `stock_relative`
- `stock_relative_market`
- `stock_relative_market_sector`
- `stock_compact`
- `stock_relative_compact`
- `stock_relative_market_compact`
- `stock_relative_market_sector_compact`
- `stock_only_sentiment`
- `stock_relative_market_sector_sentiment`
- `stock_only_fundamentals`
- `stock_relative_market_sector_fundamentals`
- `stock_only_fundamentals_sentiment`
- `stock_relative_market_sector_fundamentals_sentiment`

Approximate full vs compact feature counts when all expected columns are present:
- `stock_only`: about 90
- `stock_relative`: about 279
- `stock_relative_market`: about 340
- `stock_relative_market_sector`: about 401
- `stock_compact`: about 66
- `stock_relative_compact`: about 111
- `stock_relative_market_compact`: about 151
- `stock_relative_market_sector_compact`: about 191

Each selected daily feature is summarized over the prior `window_length` trading days ending on `anchor_date`:
- `<feature>__last`
- `<feature>__mean60`
- `<feature>__std60`

For example:
- `stock_return_1d__last`
- `stock_return_1d__mean60`
- `stock_return_1d__std60`

These tabular features are compressed and denoised compared with the raw daily token stream.

Raw level guardrail:
- model inputs must not include raw price, raw volume, raw dollar-volume, legacy raw VWAP, raw previous-close, or raw moving-average level columns
- use returns, ratios, same-date normalizations, percentile ranks, and `log1p_*` liquidity fields instead
- `src.data.v1_dataset.validate_model_feature_columns` enforces this for tabular and sequence feature columns

### Explicit market and sector context
V1 includes a separate daily context table, not mixed into the stock cross-sectional universe.

Default context instruments:
- `SPY` for full-market context and market-adjusted targets
- sector ETFs by GICS sector:
  - Communication Services: `XLC`
  - Consumer Discretionary: `XLY`
  - Consumer Staples: `XLP`
  - Energy: `XLE`
  - Financials: `XLF`
  - Health Care: `XLV`
  - Industrials: `XLI`
  - Information Technology: `XLK`
  - Materials: `XLB`
  - Real Estate: `XLRE`
  - Utilities: `XLU`

Context features are computed with the same daily feature logic as stocks.
Join SPY context by date and sector ETF context by `(date, gics_sector)`.
Leave missing context missing with flags; do not unsafe-fill unavailable dates.

### Static or slow features
These may include:
- sector / industry / exchange
- company profile fields
- slower valuation or quality features
- durable metadata that does not need dense daily sequencing

## Transformer guidance

### Default temporal design
Use a transformer on daily tokens when sequence modeling is desired.

Recommended default:
- one token = one day
- one day token = all aligned numeric features for that date
- prefer scale-free and normalized features rather than relying heavily on raw price and raw volume levels
- positional encoding over the 60-day window
- pooled or final-step representation used for downstream fusion

Important:
Using one day token does **not** eliminate all within-day feature interactions.
Those interactions can still be learned through:
- input projections
- MLP/feed-forward blocks
- learned fusion layers

So do not move to cross-attention just because features are heterogeneous.

## Cross-attention guidance

### When cross-attention is justified
Cross-attention is worth trying only when there are genuinely distinct streams, such as:
- market time series
- fundamentals
- news/sentiment
- macro/regime context

This is most justified when:
- streams have different frequencies
- streams have different noise properties
- one stream should query another
- ablations suggest simple fusion is leaving performance on the table

### When cross-attention is not the default
Do not use cross-attention merely to separate:
- price
- volume
- PE
- sentiment
into independent scalar streams by default.

That often adds complexity faster than it adds value.

### Preferred progression for multimodal fusion
1. **Single-branch day-token transformer**
2. **Multi-branch encoders + concatenation/gated fusion**
3. **Multi-branch encoders + cross-attention fusion**
4. **Hierarchical modality-token architecture** only if earlier versions justify it

## Suggested encoder designs

### Market/time-series encoder
Possible implementations:
- transformer encoder
- TCN
- LSTM
- lightweight temporal MLP mixer

Start with:
- a compact transformer or TCN baseline
- avoid very deep architectures until data and pipeline are stable

### Static/tabular encoder
Possible implementations:
- MLP
- small residual MLP
- embedding layers for categorical metadata

### News/sentiment branch
Add later if Stage 1 works.
Possible implementations:
- daily aggregated numeric sentiment features
- event count features
- precomputed text embeddings aggregated by day

### Context branch
Use in V1 for:
- SPY market trend context
- sector ETF trend context

Add later for:
- VIX
- rates
- peer-performance summaries

## NN-kNN integration

## Retrieval object
A retrieved case is a **stock-window episode**, not just a stock ticker.

Recommended case contents:
- ticker
- anchor date
- fused representation
- target
- optional metadata for explanation

## Retrieval placement
The preferred retrieval input is the fused representation after the main supervised encoders.

Why:
- it includes both temporal and static information
- it stays modular
- it allows the retrieval head to be added without redesigning the whole network

## Retrieval usage modes
Support two modes:
1. **Pure predictive mode**
   - standard supervised prediction only

2. **Predictive + retrieval mode**
   - prediction + top-k similar stock-window episodes
   - case scores/weights logged for inspection
   - later: optional retrieval-aware aggregation or hybrid head

## Baseline experiments
For broad model-search phases, include:
- linear/logistic baselines where sensible
- MLP on flattened engineered features
- ablations comparing engineered raw features vs normalized cross-sectional feature sets
- XGBoost / LightGBM
- sklearn histogram gradient boosting
- torch MLP
- temporal baseline without retrieval
- same model with retrieval branch removed
- same model with static branch removed
- same model with context branch removed

For the current full EODHD V1 benchmark, run only the three selected classification candidates listed in the implementation snapshot. Regression is no longer a standard full-run task.

V1 artifact policy:
- keep every trained model
- write per-model metrics by horizon
- write a leaderboard with recommended models marked
- produce one inference row per `(ticker, anchor_date, model_name)`

## Evaluation design
Use:
- rolling or walk-forward splits
- strictly later validation and test windows
- regime-aware reporting if possible

Current V1 walk-forward protocol:
- Each fold uses an expanding training window.
- The validation block is strictly later than the training block and is used for model selection or early stopping when the model supports it.
- A purge gap separates validation from OOS testing. The default purge gap is the maximum forward target horizon, currently `20` trading days.
- The OOS test block is strictly later than the purge gap and is used only for scoring, not for training or model selection.
- After an OOS block is scored, it can join the expanding training history for later folds.

The purge gap exists because targets are forward-looking windows. For example, a 20-day validation target uses future prices after the validation anchor date. Starting OOS immediately after validation would make validation label windows overlap with OOS periods, so the gap keeps OOS scoring more independent.

Current final deploy training:
- After walk-forward scoring, the trainer fits final deployment bundles for selected model/feature combinations.
- It reserves the latest resolved `final_stop_block_size` trading dates, default `21`, as a final validation tail.
- It trains on the earlier resolved dates and uses that final validation tail for early stopping or best-epoch/best-iteration selection.
- If the model implements `refit_full`, it is then reinitialized and trained again on all resolved dates for the selected number of epochs/iterations.
- Final deploy performance is not reported from this refit. Reported performance comes from the earlier OOS walk-forward blocks.
- Sequence-static final deploy models follow the same rule: pick `best_epoch_` from the final train/validation split, then retrain on all resolved sequence episodes for `best_epoch_` epochs.

Report:
- PR AUC for the positive event label
- ROC AUC
- top-decile precision
- realized top-bottom spread
- directional or hit-rate diagnostics when relevant
- regression metrics such as RMSE/MAE only for explicit research ablations

## Continual-learning stance
For domain shift and production maintenance:
- run data refreshes for prediction more often than model retraining
- score latest windows with saved deploy bundles after each data refresh
- retrain every 2 to 4 weeks, with monthly as the default operating cadence
- retrain immediately after feature schema, target, vendor semantics, or universe-policy changes
- retrain sooner if OOS monitoring shows material performance drift
- prefer rolling retraining, periodic supervised refresh, replay windows, and recency weighting before adding online learning

Do not move to RL unless the task is explicitly redefined as sequential action optimization.

## Initial implementation order
1. Build the dataset pipeline first.
2. Implement flattened-feature baselines.
3. Implement Version 1 supervised architecture.
4. Add fused-embedding logging.
5. Add NN-kNN retrieval on fused embeddings.
6. Add daily news/context features.
7. Only then test multi-branch cross-attention.

## Guardrails
- Keep every branch optional through configuration.
- Keep retrieval swappable.
- Avoid premature architecture inflation.
- Favor interpretable and auditable representations over maximal novelty early on.
- For end-of-day panels, same-date cross-sectional and same-date sector-relative normalization are preferred over full-dataset scaling.
