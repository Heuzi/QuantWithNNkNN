# Massive Legacy Artifact Archive

Archived on: 2026-04-28
Source folder: artifacts/v1_baselines
Archive folder: artifacts/archive/massive_legacy_20260428/v1_baselines

These outputs were produced before the EODHD migration, using the Massive/S&P500-era dataset design and older feature surface. They are retained for historical reference only and should not be compared directly with EODHD full-universe runs.

Key incompatibilities:
- Vendor/source changed from Massive to EODHD.
- Universe changed from the prior S&P500-centered setup to listed US common stocks including delisted names where available.
- Feature schema dropped VWAP/transaction-dependent fields.
- Future EODHD runs target a much larger 30-year training panel.
