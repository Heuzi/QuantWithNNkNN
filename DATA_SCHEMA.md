# Data Schema

## Purpose
This document defines the backtest-safe dataset schema for the quant return-prediction project.

Current primary vendor: **Massive**

This schema is intentionally conservative. If a field's historical availability is unclear, treat it as unsafe until verified.

## Core entity
The core training example is a **stock-window episode**.

Definition:
- `ticker`
- `anchor_date`
- `window_length`
- `past_window_features`
- `as_of_static_features`
- `target`

Default:
- `window_length = 60` trading days
- target horizon = 5 or 10 trading days

## Fundamental design rule
Every feature attached to an anchor date must have been knowable on or before that anchor date.

That means:
- use historical timestamps
- preserve time order
- avoid future leakage through joins or fills
- record the timestamp logic for every feature source

## Dataset layers

### Layer 1: entity and calendar keys
Required keys:
- `ticker`
- `anchor_date`
- `entity_id` if available from vendor
- optional stable IDs such as CIK / FIGI when available
- trading calendar info if needed

Notes:
- Do not rely on ticker symbol alone for identity across long periods.
- Ticker reuse and ticker changes must be handled explicitly.

### Layer 2: rolling daily market features
Examples:
- open
- high
- low
- close
- adjusted price fields if available and appropriate
- volume
- VWAP if available
- daily returns
- rolling returns
- rolling volatility
- rolling average volume
- gap features
- momentum features
- price-vs-moving-average features
- close-vs-VWAP features
- log-scaled liquidity features
- same-date cross-sectional z-scores
- same-date cross-sectional percentile ranks
- same-date same-sector relative z-scores
- same-date same-sector relative percentile ranks

Rules:
- all market features must come only from dates <= anchor_date
- corporate-action handling must be consistent
- if using adjusted fields, document exactly how adjustments are defined
- for end-of-day models, same-date cross-sectional normalization is allowed only when it uses values from that same date and no future dates
- sector-relative normalization must also be computed on the same date only, never using a sector's future history
- do not use global full-dataset scaling statistics across future periods
- if raw level features are retained, also provide scale-free or normalized alternatives where possible

### Layer 3: as-of fundamentals and valuation features
Examples:
- earnings-related fields
- revenue
- net income
- assets
- liabilities
- cash flow features
- margins
- valuation ratios
- profitability ratios
- leverage ratios

Rules:
- use filing/public availability date, not only fiscal period end
- a value can only appear after it becomes public
- forward-fill is allowed only after public availability
- never backfill into the past
- record both:
  - `source_period_end`
  - `source_filing_date` or equivalent availability timestamp
- if a vendor ratio is used, verify that it is point-in-time safe

### Layer 4: company metadata
Examples:
- sector
- industry
- exchange
- company type
- locale
- market
- share class metadata when useful

Rules:
- prefer stable metadata sources
- if metadata can change through time, attach it using as-of logic
- do not assume today's metadata applies historically

### Layer 5: news and sentiment features
Examples:
- article count in trailing windows
- source-weighted article count
- daily sentiment mean
- daily sentiment dispersion
- daily event flags
- embedding aggregates

Rules:
- only use articles with `published_utc <= anchor_date` cutoff
- for intraday work, use exact timestamps, not only dates
- do not blindly trust vendor ticker labels for renamed/reused symbols
- log the article-to-entity mapping logic
- if using vendor sentiment, verify field meaning and historical consistency before relying on it

### Layer 6: broader market context
Examples:
- SPY returns and volatility
- sector ETF returns
- VIX
- rates
- peer group summaries
- market breadth measures if available later

Rules:
- align all context to the same anchor date
- ensure every context feature is available by that date
- document whether context is raw, lagged, or windowed

## Recommended table structure

### Episode index table
One row per `(ticker, anchor_date)`.

Suggested columns:
- `ticker`
- `anchor_date`
- `window_start_date`
- `window_end_date`
- `target_horizon_days`
- `target_return`
- `market_adjusted_target_return`
- `sector_adjusted_target_return`
- `is_train`
- `is_val`
- `is_test`

V1 supervised targets:
- Regression targets are continuous market-adjusted future returns, named `market_adjusted_return_{horizon}d`.
- Default regression horizons are `1`, `5`, `10`, and `20` trading days.
- The current event-classification target is `market_outperform_any_20d_gt_5pct`.
- `market_outperform_any_20d_gt_5pct` is positive when the pathwise stock excess return over `SPY` exceeds `5%` at any point in the next 20 trading days.
- Targets are labels only and must not be included in model input feature columns.

### Prediction window index table
One row per latest target-pending `(ticker, anchor_date)` used for production-style inference.

Suggested columns:
- `ticker`
- `anchor_date`
- `window_start_date`
- `window_end_date`
- `window_length`
- `target_horizon_days`
- `target_status`
- `inference_ready`

Rules:
- rows must not include realized future returns
- use only feature rows with `date <= anchor_date`
- keep this table separate from completed supervised episode rows

### Daily sequence table
One row per `(ticker, date)` for aligned daily features.

Suggested columns:
- `ticker`
- `date`
- market features
- technical features
- normalized market features
- cross-sectional normalized features
- sector-relative normalized features
- context features
- news aggregate features
- as-of fundamentals that are safe to carry on that date
- sector / industry metadata used for same-date relative transforms
- flags for missingness and source availability

V1 sequence-model layout:
- A sequence sample is keyed by `(ticker, anchor_date)`.
- It contains the prior `window_length` rows from this daily sequence table, ending on `anchor_date`.
- Default `window_length` is 60 trading days.
- Tensor shape is `[batch_size, window_length, features_per_day]`.
- Each day is one token. The model sees daily values directly rather than precomputed window summaries.

Implemented sequence feature-set components:
- stock daily features are always included
- Priority A daily features add OHLCV-derived shape/liquidity fields and same-date context-relative return fields: `close_location`, `true_range_pct`, `dollar_volume_ratio_5d`, `volume_zscore_20d`, `stock_vs_market_return_1d`, `stock_vs_sector_return_1d`, `stock_vs_market_return_5d`, and `stock_vs_sector_return_5d`
- relative stock features can add same-date full-panel and same-sector fields such as `return_1d__cs_z`, `log1p_volume__cs_pct`, and `rolling_vol_20d__sector_cs_z`
- market context can add `SPY` fields prefixed with `market_context_`
- sector context can add mapped sector ETF fields prefixed with `sector_context_`
- missing context is represented with `market_context_missing` and `sector_context_missing`

Supported sequence feature-set names:
- `stock_only_sequence`
- `stock_relative_sequence`
- `stock_market_sequence`
- `stock_sector_sequence`
- `stock_market_sector_sequence`
- `stock_relative_market_sequence`
- `stock_relative_sector_sequence`
- `stock_relative_market_sector_sequence`
- `stock_compact_sequence`
- `stock_relative_compact_sequence`
- `stock_market_compact_sequence`
- `stock_sector_compact_sequence`
- `stock_market_sector_compact_sequence`
- `stock_relative_market_compact_sequence`
- `stock_relative_sector_compact_sequence`
- `stock_relative_market_sector_compact_sequence`

The same component combinations are also accepted without the `_sequence` suffix when used by sequence-capable models.

Compact sequence names use the same component joins as their full counterparts, but select a smaller daily feature profile to reduce redundant and highly correlated inputs.

### Flattened episode feature table
One row per `(ticker, anchor_date)` for tabular baselines.

Implemented tabular feature sets:
- `stock_only`
- `stock_relative`
- `stock_relative_market`
- `stock_relative_market_sector`
- `stock_compact`
- `stock_relative_compact`
- `stock_relative_market_compact`
- `stock_relative_market_sector_compact`

Approximate feature counts when all expected columns are present:
- `stock_only`: about 90
- `stock_relative`: about 279
- `stock_relative_market`: about 340
- `stock_relative_market_sector`: about 401
- `stock_compact`: about 66
- `stock_relative_compact`: about 111
- `stock_relative_market_compact`: about 151
- `stock_relative_market_sector_compact`: about 191

Compact profile policy:
- prefer `log_return_1d` over both `return_1d` and `log_return_1d`
- use `rolling_return_*` and drop current exact `momentum_*` aliases
- prefer `log1p_dollar_volume` over carrying every overlapping volume and dollar-volume level
- keep Priority A daily shape/liquidity/regime fields because they are low-cost and available from the current Massive OHLCV/context tables
- keep z-score relative features and drop percentile-rank counterparts for the first compact ablation
- keep a smaller context set focused on returns, volatility, trend, and liquidity ratio

Rules:
- use only daily rows with `date <= anchor_date`
- summarize the prior `window_length` trading days ending on `anchor_date`
- summarize each selected daily feature as `__last`, `__mean60`, and `__std60`
- do not include target columns or future-derived values
- do not include raw level fields such as `open`, `high`, `low`, `close`, `volume`, `dollar_volume`, `vwap`, raw moving averages, or raw previous close in model inputs
- use transformed alternatives such as returns, ratios, z-scores, percentile ranks, and `log1p_*` liquidity fields

Example columns:
- `stock_return_1d__last`
- `stock_return_1d__mean60`
- `stock_return_1d__std60`
- `stock_log1p_volume__cs_z__last`
- `market_context_rolling_vol_20d__mean60`
- `sector_context_momentum_60d__std60`

This table is intentionally different from the sequence-model input. Tabular baselines receive compressed rolling summaries; sequence models receive the raw daily token sequence.

### Market context table
One row per `(context_ticker, date)` for benchmark and sector ETF context.

Default context tickers:
- `SPY`
- `XLC`, `XLY`, `XLP`, `XLE`, `XLF`, `XLV`, `XLI`, `XLK`, `XLB`, `XLRE`, `XLU`

Rules:
- compute context features with the same daily feature logic as stock bars
- keep this table separate from the stock cross-sectional normalization universe
- join SPY features by anchor date
- join sector ETF features by anchor date and the stock's GICS sector ETF mapping
- sequence models join context by daily token date; tabular models join context after rolling-window summarization
- do not forward-fill missing context unless a later implementation explicitly audits that policy

### As-of fundamentals table
One row per `(ticker, source_filing_date, source_period_end)` or vendor-equivalent record.

Suggested columns:
- `ticker`
- `source_period_start`
- `source_period_end`
- `source_filing_date`
- raw financial fields
- derived ratio fields
- vendor record ID if available

### Metadata table
One row per entity or one row per `(ticker, as_of_date)` depending on whether metadata varies over time.

### News table
One row per article.

Suggested columns:
- `article_id`
- `published_utc`
- raw ticker tags from vendor
- normalized entity mapping
- title
- description / summary
- publisher
- sentiment fields if available
- confidence or quality flags if you create them

## Point-in-time join rules

### Market features join
For anchor date `t`:
- only use market rows with `date <= t`
- rolling window features must be computed from dates within the allowed historical window

### Market and sector context join
For anchor date `t`:
- join SPY context using rows with `date <= t`
- join sector ETF context using the ETF mapped from the stock's GICS sector and rows with `date <= t`
- context rows are inputs only; sector ETFs are not the V1 target adjustment

### Fundamentals join
For anchor date `t`:
- only join the most recent fundamental record with availability timestamp `<= t`
- never join on reporting period end alone
- preserve the underlying filing/public timestamp for auditability

### News join
For anchor date `t`:
- only use articles with `published_utc <= cutoff`
- define the cutoff clearly:
  - end-of-day models may use end-of-day cutoff
  - pre-open models may only use articles available before market open
- aggregate news into daily or windowed features after applying the cutoff

### Metadata join
For anchor date `t`:
- use as-of metadata if the field can change historically
- otherwise use stable IDs with caution documented

## Missing-data policy
There will be missing data. Design for it explicitly.

Allowed responses:
- missingness indicator flags
- forward-fill only when economically and temporally valid
- model-side masking
- minimum-history inclusion rules

Do not:
- silently fill with future values
- use global dataset statistics computed across future periods in a leakage-prone way
- drop rows in ways that create unintended survivorship bias without documenting it

## Massive-specific cautions
Use Massive as the primary vendor for now, but assume the following:
- ticker-history behavior looked promising in early manual checks
- news timestamps looked usable
- historical news ticker labels may not always be trustworthy for renamed/reused symbols
- financial endpoints must be verified for entitlement and time semantics before being treated as fully backtest-safe
- derived ratios should be audited before use in final experiments
- current-constituent cross-sectional panels are acceptable for development, but are not the same as a historically point-in-time index membership panel

## Suggested feature families

### Stage 1 minimum viable features
- daily returns
- rolling returns over multiple windows
- rolling volatility
- rolling average volume
- log-scaled volume and dollar-volume features
- price-vs-SMA and close-vs-VWAP features
- same-date cross-sectional z-scores or percentile ranks for selected continuous features
- same-date same-sector relative versions for selected liquidity / momentum / volatility features
- simple technical indicators
- SPY and sector ETF trend context
- selected as-of valuation/profitability features
- sector / industry metadata

### Stage 2 added features
- peer-relative performance
- news count features
- sentiment aggregates
- event indicators
- optional embedding-based news summaries

## Target specification
Preferred targets:
- next 1-day market-adjusted return
- next 5-day market-adjusted return
- next 10-day market-adjusted return
- next 20-day market-adjusted return
- optional raw return variants
- optional sector-adjusted variants

Recommended default:
- V1 uses a multi-output target vector for 1, 5, 10, and 20 trading days
- rank models mainly on 5/10/20-day rank IC and portfolio spread
- report 1-day as a noisy diagnostic and prediction output

## Split policy
Do not use random splitting for final results.

Use:
- walk-forward train/validation/test
- rolling windows
- regime-aware analysis when possible

## Auditability requirements
Every feature family should record:
- source endpoint or source table
- primary entity key
- timestamp used for availability
- join rule
- missing-data rule
- known risks

This information should live in code or config, not only in prose.

## Suggested config fields
Every feature source should declare something like:
- `name`
- `vendor`
- `endpoint`
- `entity_key`
- `event_timestamp_field`
- `effective_date_field`
- `join_type`
- `fill_policy`
- `lag_policy`
- `notes`

## First implementation tasks
1. Create an episode index table.
2. Create a daily aligned feature table.
3. Implement point-in-time fundamentals join logic.
4. Implement news cutoff and aggregation logic.
5. Add missingness flags.
6. Generate the first walk-forward dataset snapshot.
7. Add a normalized daily feature table for model-ready scale-safe inputs.
8. Freeze that snapshot for baseline experiments.
