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
- `scripts/train_v1_supervised_baselines.py` now defaults to expanding-window walk-forward evaluation.
- The current baseline suite now supports both regression and event classification.
- Regression still uses the multi-horizon market-adjusted return targets.
- Classification now adds `market_outperform_any_20d_gt_5pct`, a PIT-safe label that is positive when pathwise benchmark-relative excess return exceeds `5%` at any point in the next 20 trading days.
- The trainer can run `regression`, `classification`, or `both` in one pass.
- Regression and classification leaderboards and deploy bundles are written separately so downstream analysis can choose one or both tasks.
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
- Latest inference uses final deployment bundles saved after the walk-forward run completes.

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
Always include:
- ridge or linear regression where sensible
- MLP on flattened engineered features
- ablations comparing engineered raw features vs normalized cross-sectional feature sets
- XGBoost / LightGBM
- sklearn histogram gradient boosting
- torch MLP
- temporal baseline without retrieval
- same model with retrieval branch removed
- same model with static branch removed
- same model with context branch removed

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
- RMSE
- MAE
- rank correlation / IC where useful
- directional accuracy
- simple threshold-strategy metrics when relevant

## Continual-learning stance
For domain shift, prefer:
- rolling retraining
- periodic supervised refresh
- replay windows
- recency weighting

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
