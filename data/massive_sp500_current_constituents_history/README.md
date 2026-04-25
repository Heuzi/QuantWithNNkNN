# Massive S&P 500 Current-Constituent History

This folder contains a large daily bar dump for the current S&P 500 constituent list, collected from Massive with one long-range request per ticker to minimize REST calls.

## Purpose

Use this dataset for:

- broad daily feature engineering
- larger-scale baseline training
- cross-sectional experiments across many large-cap names
- validating that the ingestion path works at higher volume

## Scope

- Constituent source: current S&P 500 list at collection time
- Constituent count: `503` symbols
- Requested date range: `1995-01-01` to `2026-04-21`
- Actual returned stock coverage from Massive free-plan data: approximately `2024-04-22` to `2026-04-21`
- Market-context coverage: approximately `2024-04-24` to `2026-04-23`
- Frequency: daily

## Files

- [summary.json](c:/Users/yexia/Documents/GitHub/QuantWithNNkNN/data/massive_sp500_current_constituents_history/summary.json): collection summary and high-level caveats
- [progress.json](c:/Users/yexia/Documents/GitHub/QuantWithNNkNN/data/massive_sp500_current_constituents_history/progress.json): restart/checkpoint metadata from the bulk collection job
- [raw/daily_market_bars.csv](c:/Users/yexia/Documents/GitHub/QuantWithNNkNN/data/massive_sp500_current_constituents_history/raw/daily_market_bars.csv): adjusted daily OHLCV, VWAP, transaction count, and derived dollar volume for all collected symbols
- [raw/sp500_constituents_current.csv](c:/Users/yexia/Documents/GitHub/QuantWithNNkNN/data/massive_sp500_current_constituents_history/raw/sp500_constituents_current.csv): current constituent snapshot with symbol, security name, GICS sector, sub-industry, headquarters, date added, CIK, and founded year
- [processed/daily_features.csv](c:/Users/yexia/Documents/GitHub/QuantWithNNkNN/data/massive_sp500_current_constituents_history/processed/daily_features.csv): engineered Layer 2 daily features including returns, rolling returns, rolling volatility, rolling average volume, gap features, VWAP-relative features, and momentum-style features. This large generated CSV is versioned in Git LFS.
- [processed/daily_features_normalized.csv](c:/Users/yexia/Documents/GitHub/QuantWithNNkNN/data/massive_sp500_current_constituents_history/processed/daily_features_normalized.csv): same-date full-panel and same-date sector-relative normalized features for quant-style modeling. This large generated CSV is versioned in Git LFS.
- [processed/daily_features_normalized_manifest.json](c:/Users/yexia/Documents/GitHub/QuantWithNNkNN/data/massive_sp500_current_constituents_history/processed/daily_features_normalized_manifest.json): formulas, feature lists, universe definition, and PIT-safe timing assumptions for normalization
- `processed/episode_index.csv`: completed supervised stock-window episodes with realized future targets, produced by `scripts/update_massive_daily_dataset.py`
- `processed/prediction_windows.csv`: latest target-pending sliding windows for production-style prediction after an update, produced by `scripts/update_massive_daily_dataset.py`
- `processed/incremental_update_manifest.json`: audit record for the latest incremental refresh and rebuild
- `raw/market_context_bars.csv`: SPY and sector ETF raw context bars for V1 supervised models, produced by `scripts/collect_massive_market_context.py`
- `processed/market_context_features.csv`: SPY and sector ETF engineered context features used for market-adjusted targets and context inputs

## Current Scale

- Symbols collected: `503`
- Daily stock bar rows: about `249,497`
- Failures remaining: `0`

## Important Caveats

- This is not a historical S&P 500 membership panel.
- Using today's constituent list for older periods introduces survivorship bias.
- Despite requesting data back to `1995-01-01`, the free-plan response only returned about two years of stock history.
- The stock table and the market-context table do not end on the same date. Latest stock anchors currently top out at `2026-04-21`, while the context table extends to `2026-04-23`.
- This folder does not yet include filing-dated fundamentals or a historically correct constituent-membership panel.

## Suggested Agent Usage

- Prefer this folder when you need broad large-cap daily market data rather than a tiny smoke-test sample.
- Before training, build or refresh the processed feature tables from `raw/daily_market_bars.csv`.
- Before V1 supervised training, collect market context with `py -3.11 scripts/collect_massive_market_context.py --dataset-root data\massive_sp500_current_constituents_history --rate-limit-calls 5 --rate-limit-period-seconds 60`.
- Train V1 supervised baselines with `py -3.11 scripts/train_v1_supervised_baselines.py --dataset-root data\massive_sp500_current_constituents_history`.
  The trainer now defaults to walk-forward evaluation and will use GPU-accelerated `torch`, `xgboost`, and `lightgbm` paths when available.
- Score latest windows with all saved models using `py -3.11 scripts/predict_v1_supervised_baselines.py --run-dir artifacts\v1_baselines\<run_name> --dataset-root data\massive_sp500_current_constituents_history`.
- To refresh the dataset with recent Massive bars and rebuild training/inference artifacts, run `python scripts/update_massive_daily_dataset.py --dataset-root data\massive_sp500_current_constituents_history`.
- Do not present results from this folder as historically unbiased S&P 500 backtests unless historical membership is added later.
