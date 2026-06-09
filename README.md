# QuantWithNNkNN

Multimodal ML research code for U.S. equities return prediction with optional NN-kNN retrieval support.

The public repo contains code, schemas, configs, docs, tests, and synthetic fixtures. It does not redistribute vendor market data. See [DATA_LICENSE.md](DATA_LICENSE.md).

## What Is Public

- `src/` data, feature, model, training, evaluation, and retrieval-oriented modules
- `scripts/` dataset recreation and experiment runners
- `configs/` reproducible V1 experiment profiles
- `tests/` leakage-sensitive and model-pipeline tests
- `data/fixtures/` synthetic toy data only
- `ARCHITECTURE.md` and `DATA_SCHEMA.md` for design and point-in-time rules

## What Is Not Shipped

Vendor-derived data and generated artifacts are intentionally ignored:

- raw EODHD/Massive/Polygon market data
- fundamentals, sentiment, and metadata dumps
- processed feature CSVs
- materialized training panels and episode caches
- model bundles and production reports

Generate those locally with your own vendor account and keep them out of Git.

## Recreate A Local Dataset

Install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set your own EODHD key in the environment:

```powershell
$env:EODHD_API_KEY = "your_eodhd_api_key"
```

Create a small local smoke dataset:

```powershell
.\scripts\recreate_dataset.ps1 `
  -Vendor eodhd `
  -DatasetRoot data\eodhd_smoke `
  -Tickers AAPL,MSFT,NVDA,SPY `
  -StartDate 2022-01-01 `
  -EndDate 2024-12-31 `
  -SkipFundamentals `
  -SkipSentiment
```

Train a small baseline on that local dataset:

```powershell
py -3.11 scripts\train_v1_supervised_baselines.py `
  --dataset-root data\eodhd_smoke `
  --output-root artifacts\v1_baselines `
  --run-name eodhd_smoke_public `
  --eval-mode walk_forward `
  --task-type classification `
  --classification-models lightgbm_classifier `
  --feature-sets stock_only,stock_relative_market_sector `
  --classification-horizon 5 `
  --window-length 20 `
  --max-episodes 2000
```

For a full licensed local run, use the same wrapper with `-FullUniverse`, then run the JSON profiles in `configs/v1_runs/` with `scripts/run_v1_pipeline.py`.

For the current two-sleeve research workflow, train independent conservative and momentum/breakout model sets with:

```powershell
.\scripts\run_two_sleeve_retrain.ps1 -Sleeve momentum_breakout -Resume
```

The momentum/breakout sleeve uses a bounded model-selection profile across broad tabular feature sets, then the trading runner scores the top three OOS leaderboard rows for that sleeve. The sleeve profiles write separate training panels, episode caches, and model artifacts so their models are not reused across sleeves. They intentionally run only the most recent walk-forward fold for faster iteration; use the older multi-fold profiles when broad historical regime coverage is required.

Run the current two-sleeve trading report with:

```powershell
.\scripts\run_two_sleeve_trading_strategy.ps1 -ForceRebuildLatestInference
```

## Synthetic Fixture

`data/fixtures/synthetic_equities/` contains hand-written toy rows for docs and smoke examples. It is not suitable for model quality evaluation.

## Safety Rules

Use `DATA_SCHEMA.md` for point-in-time data rules. In short: never use fields unavailable on the anchor date, never random-shuffle final time-series evaluation, and never commit real vendor data to the public repository.
