# Synthetic Equities Fixture

This directory contains tiny hand-written synthetic rows for documentation and lightweight examples. These files are not vendor data, are not real prices, and are not suitable for model-quality evaluation.

Use this fixture only to inspect expected folder shapes:

- `raw/daily_market_bars.csv`
- `raw/eodhd_equity_metadata.csv`
- `processed/daily_features.csv`
- `processed/market_context_features.csv`

For real training or testing, recreate a local dataset with your own vendor license and API key using `scripts/recreate_dataset.ps1`.
