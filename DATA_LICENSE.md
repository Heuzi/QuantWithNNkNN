# Data License

This public repository does not redistribute vendor market data.

The code, schemas, configs, tests, and documentation are intended to be public. Real market-data files generated from EODHD, Massive, Polygon, exchanges, or other vendors are not included. This includes raw OHLCV files, fundamentals, sentiment, processed feature tables, derived training panels, episode caches, model artifacts, prediction reports, and backtest outputs.

Users who want to train or test models on real market data must obtain their own vendor license/API key and recreate the dataset locally with `scripts/recreate_dataset.ps1` or the underlying ingestion scripts. The resulting data is governed by that user's agreement with the vendor. Do not commit regenerated vendor data or derived artifacts back to this repository unless explicit redistribution permission has been obtained from the relevant vendor and upstream data providers.

The files under `data/fixtures/` are synthetic toy fixtures. They are not vendor data and are provided only for examples, documentation, and lightweight tests.

When adding a new data source, document:

- source endpoint or table
- entity key
- timestamp field
- as-of join logic
- missing-data handling
- redistribution and publication status

If the redistribution status of a field is unclear, treat it as non-public.
