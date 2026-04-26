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
- `scripts/train_v1_supervised_baselines.py` now defaults to expanding-window walk-forward evaluation.
- The current baseline suite now supports both regression and event classification.
- Regression still uses the multi-horizon market-adjusted return targets.
- Classification now adds `market_outperform_any_20d_gt_5pct`, a PIT-safe label that is positive when pathwise benchmark-relative excess return exceeds `5%` at any point in the next 20 trading days.
- The trainer can run `regression`, `classification`, or `both` in one pass.
- Regression and classification leaderboards and deploy bundles are written separately so downstream analysis can choose one or both tasks.
- The current baseline suite includes tabular baselines plus `torch_seq_static`.
- `torch_seq_static` uses a real 60-day sequence branch plus static categorical embeddings for `gics_sector` and `gics_sub_industry`.
- The sequence-static model is currently wired only for `stock_only` and `stock_relative`.
- `torch`, `xgboost`, and `lightgbm` now prefer GPU execution when available and fall back to CPU otherwise.
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
- close-vs-VWAP features
- log-scaled liquidity features
- same-date cross-sectional normalized features
- same-date same-sector relative features
- daily sentiment/news aggregates
- fundamentals carried forward only if already public by that date

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
