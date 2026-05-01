# AGENTS.md

## Project
Build a multimodal ML system for U.S. equities that predicts a stock-window episode's future return and can optionally provide NN-kNN retrieval-based support.

A **case** is:
- `(ticker, anchor_date, prior_window_features, as-of static features)`
- Default window: prior **60 trading days**
- Default target: future **5-day or 10-day return**, preferably **market-adjusted or sector-adjusted return**

## North-star goals
1. Strong out-of-sample predictive performance on realistic time splits.
2. Backtest-safe data handling. No leakage.
3. Modular architecture that supports:
   - pure predictive mode
   - predictive + NN-kNN retrieval mode
4. Clear, reproducible experiments with strong baselines.

## Current source of truth
- Use `ARCHITECTURE.md` for model and experiment structure.
- Use `DATA_SCHEMA.md` for dataset design and point-in-time join rules.
- Treat those docs as deeper guidance; keep this file short and operational.

## Non-negotiables
- Never use information that would not have been available on the anchor date.
- Do not use random train/test shuffles across time for final evaluation.
- Prefer simpler working baselines before adding architectural complexity.
- Keep the system modular so encoders, fusion, and retrieval can be ablated independently.
- When uncertain about a data field's historical availability, treat it as unsafe until verified.

## Backtest-safety rules
- Treat **point-in-time correctness** as mandatory.
- For fundamentals, use **filing/public availability date**, not only fiscal period end.
- For each stock-window episode, join the latest safe fundamental record with `availability_date <= anchor_date`; never use a filing that became public after the anchor date.
- Include delisted names when the dataset supports them.
- Handle ticker changes, symbol reuse, splits, and dividends carefully.
- News features must be filtered by article timestamp.
- Do **not** blindly trust vendor ticker tags in historical news for renamed/reused symbols.
- Forward-fill only after a value becomes public.
- Never backfill future information into earlier dates.
- Any derived feature must be computable from information known at that time.
- Same-date cross-sectional or sector-relative normalization is acceptable only for end-of-day models and only when computed from that same date's available values.
- Never use global scaling statistics computed over future periods for final evaluation.

## Vendor assumptions
- **EODHD is the primary vendor for current V1 daily market data.**
- Massive code and artifacts are legacy and retained only for reproducibility.
- EODHD raw OHLCV, fundamentals, and sentiment must still be treated as vendor data requiring explicit time-semantics checks before model use.
- Do not assume an EODHD endpoint is backtest-safe unless its time semantics are verified.
- Build vendor adapters so the project can switch data providers later if needed.

## Default modeling roadmap
### Stage 1
Use supervised learning first.
Inputs:
- price/return/volume history
- technical indicators
- normalized market and liquidity features
- same-date cross-sectional and selected sector-relative features for continuous inputs
- as-of fundamentals / valuation / profitability
- company / sector / industry metadata

Recommended base architecture:
1. **time-series encoder** for rolling market features
2. **tabular/static encoder** for slower-moving company features
3. **fusion head**
4. **prediction head**
5. optional **NN-kNN retrieval head** on fused representation

### Stage 2
Add broader context:
- SPY / sector ETF / VIX / rates
- peer performance
- news / sentiment / event features
- market regime indicators

## Architecture guidance
### Start simple
Preferred first model:
- one token per day
- each token contains that day's aligned numeric features
- prefer scale-free, log-scaled, and same-date normalized features over raw level features when possible
- transformer over the 60-day sequence
- separate static encoder for non-sequential features
- fused representation for prediction and retrieval

### Cross-attention
Do **not** default to cross-attention only because features are heterogeneous.
Use cross-attention only if it is justified by a real separation of modalities, such as:
- market time series
- fundamentals
- news / sentiment
- macro / regime context

Good progression:
1. single-branch day-token transformer
2. multi-branch encoders + simple fusion
3. multi-branch + cross-attention fusion if ablations justify it

## NN-kNN guidance
- Retrieval should operate on a learned fused representation.
- Retrieved neighbors are **stock-window episodes**, not abstract stocks.
- Optimize retrieval for both prediction utility and local plausibility.
- Treat retrieved cases as **case-based support**, not guaranteed causal explanations.
- Log top retrieved neighbors, distances/scores, and contribution weights for analysis.
- Keep the retrieval module swappable so pure supervised baselines remain easy to run.

## Baselines
Always compare against:
- linear or ridge regression baseline where appropriate
- MLP
- XGBoost / LightGBM
- temporal baseline (LSTM/TCN or simpler transformer)
- ablations without retrieval
- ablations without static features
- ablations without news/context

## Evaluation protocol
- Use **walk-forward** or rolling time splits.
- Keep validation and test strictly later than training.
- Report multiple metrics:
  - RMSE / MAE for regression
  - rank IC / Spearman if relevant
  - hit rate / directional accuracy
  - portfolio-style metrics for thresholded strategies
- Include turnover, transaction-cost sensitivity, and drawdown analysis for any trading-style evaluation.
- Report results across multiple market regimes if possible.

## Continual learning / domain shift
- Prefer rolling retraining or periodic supervised refresh before RL.
- Track regime drift explicitly.
- Support replay windows and recency-weighted training.
- Do not convert the project to RL unless the task becomes sequential decision-making rather than return prediction.

## Coding rules
- Write modular, readable code with clear interfaces.
- Separate:
  - data ingestion
  - point-in-time joins
  - feature engineering
  - model definition
  - training
  - evaluation
  - backtest / simulation
- Every new feature source should declare:
  - source endpoint/table
  - entity key
  - timestamp field
  - as-of join logic
  - missing-data handling
- Prefer configuration-driven experiments.
- Avoid hidden magic constants.
- Add tests for data joins, leakage-sensitive transformations, and target generation.

## Output expectations
When asked to implement something, prioritize:
1. correctness
2. leakage safety
3. reproducibility
4. simplicity
5. speed

When finishing a task, provide:
- what changed
- assumptions made
- leakage or data-quality risks
- what was tested
- recommended next experiment

## When uncertain
- Choose the simpler design.
- Preserve modularity.
- Surface assumptions explicitly.
- Do not silently invent data semantics.
- If a field or join is ambiguous, mark it as a risk in code comments and outputs.

## Initial directory suggestions
- `data/`
- `src/data/`
- `src/features/`
- `src/models/`
- `src/retrieval/`
- `src/train/`
- `src/eval/`
- `configs/`
- `notebooks/`
- `docs/`

## Immediate priorities
1. define the exact target
2. build a backtest-safe dataset schema
3. implement robust baselines on the engineered and normalized daily feature tables
4. build the first supervised model
5. add NN-kNN retrieval on top of the fused embedding
6. run walk-forward evaluation
7. only then consider cross-attention and richer multimodal fusion
