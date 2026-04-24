# Massive S&P 500 Current-Constituent History

This folder contains a large daily bar dump for the current S&P 500 constituent list, collected from Massive with one long range request per ticker to minimize REST calls.

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
- Actual returned coverage from Massive free-plan data: approximately `2024-04-22` to `2026-04-21`
- Frequency: daily

## Files

- [summary.json](c:/Users/yexia/Documents/GitHub/QuantWithNNkNN/data/massive_sp500_current_constituents_history/summary.json): collection summary and high-level caveats
- [progress.json](c:/Users/yexia/Documents/GitHub/QuantWithNNkNN/data/massive_sp500_current_constituents_history/progress.json): restart/checkpoint metadata from the bulk collection job
- [raw/daily_market_bars.csv](c:/Users/yexia/Documents/GitHub/QuantWithNNkNN/data/massive_sp500_current_constituents_history/raw/daily_market_bars.csv): adjusted daily OHLCV, VWAP, transaction count, and derived dollar volume for all collected symbols
- [raw/sp500_constituents_current.csv](c:/Users/yexia/Documents/GitHub/QuantWithNNkNN/data/massive_sp500_current_constituents_history/raw/sp500_constituents_current.csv): current constituent snapshot with symbol, security name, GICS sector, sub-industry, headquarters, date added, CIK, and founded year
- [processed/daily_features.csv](c:/Users/yexia/Documents/GitHub/QuantWithNNkNN/data/massive_sp500_current_constituents_history/processed/daily_features.csv): engineered Layer 2 daily features including returns, rolling returns, rolling volatility, rolling average volume, gap features, VWAP-relative features, and momentum-style features. This large generated CSV is versioned in Git LFS.
- [processed/daily_features_normalized.csv](c:/Users/yexia/Documents/GitHub/QuantWithNNkNN/data/massive_sp500_current_constituents_history/processed/daily_features_normalized.csv): same-date full-panel and same-date sector-relative normalized features for quant-style modeling. This large generated CSV is versioned in Git LFS.
- [processed/daily_features_normalized_manifest.json](c:/Users/yexia/Documents/GitHub/QuantWithNNkNN/data/massive_sp500_current_constituents_history/processed/daily_features_normalized_manifest.json): formulas, feature lists, universe definition, and PIT-safe timing assumptions for normalization

## Current Scale

- Symbols collected: `503`
- Daily bar rows: about `249,497`
- Failures remaining: `0`

## Important Caveats

- This is not a historical S&P 500 membership panel.
- Using today’s constituent list for older periods introduces survivorship bias.
- Despite requesting data back to `1995-01-01`, the free-plan response only returned about two years of history.
- This folder now includes engineered daily features and same-date normalized features, with the large processed CSV outputs stored in Git LFS.
- It still does not include anchor episodes or filing-dated fundamentals.

## Suggested Agent Usage

- Prefer this folder when you need broad large-cap daily market data rather than a tiny smoke-test sample.
- Before training, build a processed feature table and episode index from `raw/daily_market_bars.csv`.
- Do not present results from this folder as historically unbiased S&P 500 backtests unless historical membership is added later.
