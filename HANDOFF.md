# HANDOFF

Use this file as the shared handoff note when switching between computers or Codex sessions.

## Read First

- `AGENTS.md`
- `ARCHITECTURE.md`
- `DATA_SCHEMA.md`
- `HANDOFF.md`

## Current Task

- Summary: The project is migrating daily market data from Massive to EODHD.
- Goal: Build a 30-year EODHD U.S. listed common-stock dataset, include delisted names, drop VWAP/transaction-derived features, and treat all Massive-era artifacts as legacy.
- Current default dataset root: `data/eodhd_us_equities_30y`.
- Current target universe policy: listed U.S. common stocks plus delisted names; exclude ETFs, funds, and OTC/PINK from target assets.
- Universe filtering also excludes units, warrants, rights, and preferred/preference shares when EODHD labels those rows as common stock.
- Full-universe rebuilds use `--max-tickers 0`; smaller `--max-tickers` values are smoke/pilot limits only.
- Full-universe raw collection should use `--fetch-only --fetch-workers <N>` first. The old pilot-sized in-memory rebuild path is still valid for smoke and pilot datasets, but it should not be used directly on the full 30-year all-stock panel.
- Standard episode eligibility is intentionally broad: 60-day default history, at least 55 valid adjusted OHLCV rows, 60-day average dollar volume >= 100000, adjusted close >= 1, and NYSE/NASDAQ/AMEX/BATS only.
- Context instruments remain `SPY` plus sector ETFs and are stored in the separate market-context table.
- Current full-run plan: classification-only. The standard full profile trains `torch_seq_static_classifier`, `torch_mlp_classifier`, and `xgboost_classifier`; regression is kept only for explicit research ablations.
- Prediction plan: refresh data and score latest windows with saved final classifier bundles between retrains. New ticker windows do not require retraining if their features can be built and they pass eligibility.

## Current Branch And State

- Branch: `codex/full-classification-benchmark`.
- Local secrets:
  - `EODHD_api_key` is local-only and must stay ignored.
  - Do not print API keys in logs or docs.
- Existing Massive/S&P500 artifacts are legacy and should be archived under `artifacts/archive/massive_legacy_<date>/`.
- `dev notes.txt` may contain unrelated local edits; do not stage or revert it unless explicitly requested.

## Main Code Paths

- EODHD adapter: `src/data/eodhd_stage1.py`
- EODHD fundamentals/sentiment feature helpers: `src/data/eodhd_enrichment.py`
- EODHD rebuild script: `scripts/update_eodhd_daily_dataset.py`
- Chunked EODHD raw-to-feature builder: `scripts/build_eodhd_daily_features_chunked.py`
- Standard V1 pipeline runner: `scripts/run_v1_pipeline.py`
- Materialized V1 training-panel builder: `scripts/materialize_v1_training_panel.py`
- Episode-level V1 cache builder: `scripts/materialize_v1_episode_cache.py`
- Standard V1 run profiles: `configs/v1_runs/`
- Bulk EODHD fundamentals helper: `scripts/fetch_eodhd_fundamentals_bulk.py`
- Dataset and feature construction: `src/data/v1_dataset.py`
- Normalization helpers: `src/data/normalization.py`
- Baseline trainer: `scripts/train_v1_supervised_baselines.py`
- Prediction script: `scripts/predict_v1_supervised_baselines.py`

Legacy Massive scripts remain in the repo for reproducibility but are no longer the default path.

## Current Data Policy

- EODHD EOD rows are normalized into adjusted internal OHLC using `adjusted_close` when available.
- `dollar_volume` is derived locally as adjusted internal `close * volume`.
- The model feature schema intentionally drops:
  - `vwap`
  - `transactions`
  - `close_to_vwap_pct`
- Raw price, raw volume, raw dollar volume, legacy VWAP, previous-close, and moving-average level columns remain forbidden as model inputs.
- EODHD fundamentals `General` metadata may provide sector/industry labels, but it is not treated as point-in-time fundamentals.
- Full EODHD Fundamentals v1.1 raw JSON is stored under `raw/eodhd_fundamentals_raw/`; only records with explicit availability dates become historical model features.
- Fundamental model features are joined as of each episode's `anchor_date`: use the latest filing/public row with `availability_date <= anchor_date`, never a row whose fiscal period ended before `anchor_date` but was filed later.
- EODHD daily sentiment is stored in `raw/eodhd_sentiment_daily.csv` and lagged by one trading row before model use.
- Full raw fetch outputs are resumable and local-only: `raw/eodhd_stock_bars.csv`, `raw/market_context_bars.csv`, `raw/eodhd_fundamentals_raw/`, `raw/eodhd_sentiment_daily.csv`, `raw/eodhd_fetch_status.csv`, and `raw/eodhd_fetch_manifest.json`.
- Missing sector/industry metadata falls back to `Unknown`.
- Missing fundamentals or sentiment must not remove stock-window episodes; use missing indicators and neutral defaults.
- EODHD symbol lists are collected from both current and delisted views, because the `delisted=1` view behaves as delisted-only in live checks.
- Raw identifiers including ticker, EODHD symbol, ISIN, CIK/CUSIP/FIGI, and company name are metadata only and must not enter model feature columns.

## Recent Decisions

- Decision: Use EODHD as the current primary daily market vendor.
  Reason: The upgraded plan exposes long EOD history and filtered fundamentals metadata.
- Decision: Exclude ETFs/funds/OTC from the first target universe.
  Reason: They represent different prediction problems and add liquidity/microstructure noise.
- Decision: Keep `SPY` and sector ETFs as context instruments.
  Reason: They are still needed for benchmark-relative targets and regime/context features.
- Decision: Keep old Massive code as legacy instead of deleting it.
  Reason: Existing experiments remain reproducible, but their artifacts are not comparable with the new full-universe EODHD dataset.
- Decision: Keep partial-history stocks once they pass the broad 60-day eligibility filter.
  Reason: The project needs more stock-window episodes, and missing fundamentals/sentiment should be modeled rather than used as exclusion criteria.
- Decision: Use classification-only for the standard full V1 EODHD run.
  Reason: Model selection favored classifiers, and the production question is now event probability for outperformance rather than direct return regression.
- Decision: Retrain periodically, not for every prediction refresh.
  Reason: Saved classifier bundles can score new target-pending stock-window episodes without retraining as long as the feature schema is unchanged.

## Run Size Profiles

- Smoke run:
  - Purpose: validate code, schema, cache materialization, and train/predict plumbing.
  - Shape: tiny ticker/date/episode limits and short model budgets.
  - Interpretation: execution evidence only; do not use smoke metrics for model choice.
- `eodhd_full_walk_forward`:
  - Purpose: current serious EODHD benchmark and the required gate before a multi-day run.
  - Shape: reads from `data/eodhd_us_equities_30y`, materializes a bounded high-liquidity panel, builds the episode cache, and trains `torch_seq_static_classifier`, `torch_mlp_classifier`, and `xgboost_classifier`.
  - Current scale: 1,500 tickers, 2014-01-01 through 2026-04-24, 500,000 most recent eligible episodes, and 6 walk-forward folds.
- True full-universe run:
  - Purpose: eventual large-scale experiment after the benchmark profile is stable.
  - Shape: `max_tickers=0`, intended long EODHD history, same materialized-panel plus episode-cache path, and no small benchmark cap unless a compute cap is deliberately documented.
  - Status: not the current default profile.
- `eodhd_model_selection_walk_forward` is separate from the three sizes above. Use it for research when changing the model candidate list.

## Validation Checklist

Before trusting a new EODHD run:

- `py -3.11 -m unittest discover -s tests`
- Live smoke with a tiny ticker set, for example:
  - `py -3.11 scripts/update_eodhd_daily_dataset.py --tickers AAPL,MSFT,NVDA --start-date 2025-01-01 --end-date 2025-06-30 --dataset-root data/eodhd_smoke --window-length 10 --horizon-days 5 --eligibility-min-history-days 10 --eligibility-valid-ohlcv-lookback 10 --eligibility-min-valid-ohlcv-days 8`
- Full raw fetch, after smoke validation:
  - `py -3.11 scripts/update_eodhd_daily_dataset.py --dataset-root data/eodhd_us_equities_30y --start-date 1995-01-01 --end-date 2026-04-24 --max-tickers 0 --fetch-only --fetch-workers 8`
- Finish or estimate missing fundamentals through Bulk Fundamentals, if EODHD enables that entitlement:
  - dry run: `py -3.11 scripts/fetch_eodhd_fundamentals_bulk.py --dataset-root data/eodhd_us_equities_30y --dry-run --batch-size 500`
  - fetch: `py -3.11 scripts/fetch_eodhd_fundamentals_bulk.py --dataset-root data/eodhd_us_equities_30y --batch-size 500`
- Chunked raw-to-feature build after raw fetch:
  - `py -3.11 scripts/run_v1_pipeline.py --profile eodhd_full_walk_forward --stage build_features`
- Full standardized classification walk-forward train/test after processed features exist. This first materializes a trainable high-liquidity panel, then materializes an episode cache, then trains the selected top-three classifier candidates from the cached root:
  - `py -3.11 scripts/run_v1_pipeline.py --profile eodhd_full_walk_forward`
- Score latest target-pending windows after a data refresh, using saved deploy bundles. Point `--dataset-root` at a bounded prediction/materialized dataset root, not directly at the 34GB full processed feature CSV:
  - `py -3.11 scripts/predict_v1_supervised_baselines.py --run-dir artifacts/v1_baselines/eodhd_full_walk_forward --dataset-root data/eodhd_training_panels/eodhd_full_walk_forward`
- Rebuild only the materialized trainable panel:
  - `py -3.11 scripts/run_v1_pipeline.py --profile eodhd_full_walk_forward --stage materialize_panel`
- Rebuild only the episode cache after the trainable panel is ready:
  - `py -3.11 scripts/run_v1_pipeline.py --profile eodhd_full_walk_forward --stage materialize_cache`
- Short standardized smoke train/test after processed features exist:
  - `py -3.11 scripts/run_v1_pipeline.py --profile eodhd_smoke_walk_forward --stage train`
- Model-selection run with sequence models and broader tabular families:
  - `py -3.11 scripts/run_v1_pipeline.py --profile eodhd_model_selection_walk_forward`
- Inspect the exact commands without running them:
  - `py -3.11 scripts/run_v1_pipeline.py --profile eodhd_full_walk_forward --dry-run`
- Universe smoke through EODHD symbol lists:
  - `py -3.11 scripts/update_eodhd_daily_dataset.py --max-tickers 3 --start-date 2025-01-01 --end-date 2025-03-31 --dataset-root data/eodhd_universe_smoke --window-length 10 --horizon-days 5 --skip-fundamentals`
- Tiny training smoke:
  - `py -3.11 scripts/train_v1_supervised_baselines.py --dataset-root data/eodhd_smoke --output-root artifacts/v1_baselines --run-name eodhd_smoke --eval-mode holdout --task-type classification --classification-models lightgbm_classifier --feature-sets stock_relative_market_sector_sentiment --classification-horizon 5 --window-length 10 --eligibility-min-history-days 10 --eligibility-valid-ohlcv-lookback 10 --eligibility-min-valid-ohlcv-days 8`
- Verify feature columns contain no `close_to_vwap_pct`, `vwap`, `transactions`, `ticker`, `eodhd_symbol`, or raw symbol identifiers.

## Recommended Operating Cadence

- Daily or weekly after market close: refresh OHLCV, market context, and sentiment when fresh predictions are needed.
- Monthly by default: refresh fundamentals, with extra refreshes around earnings/filing seasons if quota allows.
- After each data refresh: rebuild the bounded prediction/materialized panel and run latest prediction from saved final classifier bundles.
- Monthly or quarterly: run the full classification retrain and walk-forward benchmark.
- Immediately: retrain after target, feature schema, universe policy, vendor semantics, or material OOS monitoring changes.

## Known Risks

- Ticker identity and symbol reuse are not fully resolved by the daily bar adapter.
- Delisted coverage should be audited before final backtests.
- Raw volume is not currently split-adjusted by the adapter.
- EODHD sector/industry metadata is current vendor metadata, not a PIT sector history.
- EODHD fundamental field coverage is incomplete for some companies; missingness is expected.
- Regular per-symbol EODHD fundamentals are expensive at full-universe scale. Prefer Bulk Fundamentals if the account has Extended Fundamentals/Bulk access enabled; current probe returned HTTP 403, so EODHD support may need to enable it.
- EODHD sentiment ticker mapping may be unsafe for renamed or reused symbols without additional identity validation.
- Full all-stock rebuild can consume many paid API calls and should be run intentionally after smoke/pilot validation.
- The full 30-year all-stock panel is too large for the pilot in-memory feature builder. Use `--fetch-only` for raw collection and `scripts/build_eodhd_daily_features_chunked.py` for per-ticker daily features. Full-panel cross-sectional normalization still needs a chunked or out-of-core implementation.
- For train/test memory, use the episode cache path. Do not run the full profile by pointing the trainer directly at the full daily feature CSV; the trainer should receive `--episode-cache-dir` from `configs/v1_runs/eodhd_full_walk_forward.json`.

## Resume Prompt

```text
Read AGENTS.md, ARCHITECTURE.md, DATA_SCHEMA.md, and HANDOFF.md. The repo is migrating V1 daily market data from Massive to EODHD. Current default dataset root is data/eodhd_us_equities_30y. Use scripts/update_eodhd_daily_dataset.py for EODHD collection/rebuild, scripts/run_v1_pipeline.py for standardized profiles, and the classification-only eodhd_full_walk_forward profile for the standard full run. Keep EODHD_api_key local and ignored. Treat Massive-era artifacts as archived legacy and do not compare them directly with EODHD full-universe results.
```
