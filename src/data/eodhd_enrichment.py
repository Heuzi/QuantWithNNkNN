from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.eodhd_stage1 import internal_ticker_from_eodhd_symbol


FUNDAMENTAL_FEATURE_COLUMNS = [
    "fundamental_revenue",
    "fundamental_net_income",
    "fundamental_gross_profit",
    "fundamental_ebitda",
    "fundamental_total_assets",
    "fundamental_total_liabilities",
    "fundamental_total_equity",
    "fundamental_operating_cash_flow",
    "fundamental_free_cash_flow",
    "fundamental_shares_outstanding",
    "fundamental_net_margin",
    "fundamental_gross_margin",
    "fundamental_roa",
    "fundamental_debt_to_assets",
    "fundamental_ocf_margin",
    "fundamental_missing",
    "fundamental_staleness_days",
]

SENTIMENT_FEATURE_COLUMNS = [
    "sentiment_count",
    "sentiment_normalized",
    "sentiment_missing",
    "sentiment_count_5d",
    "sentiment_count_20d",
    "sentiment_count_60d",
    "sentiment_normalized_mean_5d",
    "sentiment_normalized_mean_20d",
    "sentiment_normalized_mean_60d",
    "sentiment_weighted_5d",
    "sentiment_weighted_20d",
    "sentiment_weighted_60d",
]

FUNDAMENTAL_RAW_FILTER = "General,Highlights,Valuation,SharesStats,Earnings,Financials"
FUNDAMENTAL_AVAILABILITY_FIELDS = (
    "filing_date",
    "filingDate",
    "filingDateTime",
    "acceptedDate",
    "accepted_date",
)


def _safe_float(value: object) -> float | None:
    if value in (None, "", "NA", "N/A"):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _first_float(record: Mapping[str, object], names: Sequence[str]) -> float | None:
    for name in names:
        value = _safe_float(record.get(name))
        if value is not None:
            return value
    return None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _date_value(record: Mapping[str, object], names: Sequence[str]) -> str | None:
    for name in names:
        value = record.get(name)
        if value not in (None, ""):
            text = str(value)
            return text[:10]
    return None


def _records_from_statement(statement: object) -> dict[tuple[str, str], dict[str, object]]:
    if not isinstance(statement, Mapping):
        return {}
    out: dict[tuple[str, str], dict[str, object]] = {}
    for frequency in ("quarterly", "yearly"):
        section = statement.get(frequency)
        if not isinstance(section, Mapping):
            continue
        for key, raw_record in section.items():
            if not isinstance(raw_record, Mapping):
                continue
            period_end = _date_value(raw_record, ("date", "period_end", "periodEndDate")) or str(key)[:10]
            availability = _date_value(raw_record, FUNDAMENTAL_AVAILABILITY_FIELDS)
            if not period_end or not availability:
                continue
            merged = dict(raw_record)
            merged["_source_period_end"] = period_end
            merged["_availability_date"] = availability
            merged["_source_frequency"] = frequency
            out[(frequency, period_end)] = merged
    return out


def parse_fundamental_feature_rows(symbol: str, payload: Mapping[str, object]) -> list[dict[str, object]]:
    financials = payload.get("Financials")
    if not isinstance(financials, Mapping):
        return []

    income = _records_from_statement(financials.get("Income_Statement") or financials.get("IncomeStatement"))
    balance = _records_from_statement(financials.get("Balance_Sheet") or financials.get("BalanceSheet"))
    cash = _records_from_statement(financials.get("Cash_Flow") or financials.get("CashFlow"))
    keys = sorted(set(income) | set(balance) | set(cash), key=lambda item: (item[1], item[0]))
    rows: list[dict[str, object]] = []
    ticker = internal_ticker_from_eodhd_symbol(symbol)

    for key in keys:
        income_row = income.get(key, {})
        balance_row = balance.get(key, {})
        cash_row = cash.get(key, {})
        availability_date = (
            income_row.get("_availability_date")
            or balance_row.get("_availability_date")
            or cash_row.get("_availability_date")
        )
        source_period_end = (
            income_row.get("_source_period_end")
            or balance_row.get("_source_period_end")
            or cash_row.get("_source_period_end")
        )
        if not availability_date or not source_period_end:
            continue

        revenue = _first_float(income_row, ("totalRevenue", "revenue", "Revenue"))
        net_income = _first_float(income_row, ("netIncome", "netIncomeApplicableToCommonShares"))
        gross_profit = _first_float(income_row, ("grossProfit",))
        ebitda = _first_float(income_row, ("ebitda", "EBITDA"))
        assets = _first_float(balance_row, ("totalAssets",))
        liabilities = _first_float(balance_row, ("totalLiab", "totalLiabilities", "totalLiabilitiesNetMinorityInterest"))
        equity = _first_float(balance_row, ("totalStockholderEquity", "totalEquityGrossMinorityInterest"))
        operating_cf = _first_float(cash_row, ("totalCashFromOperatingActivities", "operatingCashflow", "operatingCashFlow"))
        free_cf = _first_float(cash_row, ("freeCashFlow", "FreeCashFlow"))
        shares = _first_float(balance_row, ("commonStockSharesOutstanding", "commonStockShares"))

        rows.append(
            {
                "ticker": ticker,
                "eodhd_symbol": str(symbol).upper(),
                "availability_date": str(availability_date)[:10],
                "source_period_end": str(source_period_end)[:10],
                "source_frequency": key[0],
                "fundamental_revenue": revenue,
                "fundamental_net_income": net_income,
                "fundamental_gross_profit": gross_profit,
                "fundamental_ebitda": ebitda,
                "fundamental_total_assets": assets,
                "fundamental_total_liabilities": liabilities,
                "fundamental_total_equity": equity,
                "fundamental_operating_cash_flow": operating_cf,
                "fundamental_free_cash_flow": free_cf,
                "fundamental_shares_outstanding": shares,
                "fundamental_net_margin": _safe_ratio(net_income, revenue),
                "fundamental_gross_margin": _safe_ratio(gross_profit, revenue),
                "fundamental_roa": _safe_ratio(net_income, assets),
                "fundamental_debt_to_assets": _safe_ratio(liabilities, assets),
                "fundamental_ocf_margin": _safe_ratio(operating_cf, revenue),
            }
        )
    return rows


def load_fundamental_feature_rows(raw_dir: str | Path, symbols: Sequence[str] | None = None) -> list[dict[str, object]]:
    root = Path(raw_dir)
    wanted = {str(symbol).upper() for symbol in symbols or []}
    rows: list[dict[str, object]] = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("*.json")):
        symbol = path.stem.replace("__", ".").upper()
        if wanted and symbol not in wanted:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.extend(parse_fundamental_feature_rows(symbol, payload))
    rows.sort(key=lambda item: (str(item["ticker"]), str(item["availability_date"]), str(item["source_period_end"])))
    return rows


def read_fundamental_payload(path: str | Path) -> dict[str, object] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def fundamental_payload_path(raw_dir: str | Path, symbol: str) -> Path:
    return Path(raw_dir) / f"{str(symbol).upper().replace('.', '__')}.json"


def normalize_sentiment_response(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, Mapping):
        return []
    rows: list[dict[str, object]] = []
    for symbol, items in payload.items():
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            continue
        ticker = internal_ticker_from_eodhd_symbol(str(symbol))
        for item in items:
            if not isinstance(item, Mapping):
                continue
            date_value = item.get("date")
            if date_value in (None, ""):
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "eodhd_symbol": str(symbol).upper(),
                    "date": str(date_value)[:10],
                    "sentiment_count_raw": int(_safe_float(item.get("count")) or 0),
                    "sentiment_normalized_raw": _safe_float(item.get("normalized")),
                }
            )
    rows.sort(key=lambda item: (str(item["ticker"]), str(item["date"])))
    return rows


def load_sentiment_rows(path: str | Path) -> list[dict[str, object]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    rows: list[dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "ticker": str(row.get("ticker") or "").upper(),
                    "eodhd_symbol": str(row.get("eodhd_symbol") or "").upper(),
                    "date": str(row.get("date") or "")[:10],
                    "sentiment_count_raw": int(_safe_float(row.get("sentiment_count_raw")) or 0),
                    "sentiment_normalized_raw": _safe_float(row.get("sentiment_normalized_raw")),
                }
            )
    return [row for row in rows if row["ticker"] and row["date"]]


def merge_sentiment_rows(existing_rows: Sequence[dict[str, object]], incoming_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[str, str], dict[str, object]] = {}
    for row in [*existing_rows, *incoming_rows]:
        ticker = str(row.get("ticker") or "").upper()
        date_value = str(row.get("date") or "")[:10]
        if ticker and date_value:
            normalized = dict(row)
            normalized["ticker"] = ticker
            normalized["date"] = date_value
            merged[(ticker, date_value)] = normalized
    return sorted(merged.values(), key=lambda item: (str(item["ticker"]), str(item["date"])))


def add_sentiment_features(feature_rows: Sequence[dict[str, object]], sentiment_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    if not feature_rows:
        return []
    features = pd.DataFrame(feature_rows).copy()
    features["ticker"] = features["ticker"].astype(str).str.upper()
    features["date"] = features["date"].astype(str)
    if sentiment_rows:
        sentiment = pd.DataFrame(sentiment_rows).copy()
        sentiment["ticker"] = sentiment["ticker"].astype(str).str.upper()
        sentiment["date"] = sentiment["date"].astype(str)
        features = features.merge(
            sentiment[["ticker", "date", "sentiment_count_raw", "sentiment_normalized_raw"]],
            on=["ticker", "date"],
            how="left",
        )
    else:
        features["sentiment_count_raw"] = np.nan
        features["sentiment_normalized_raw"] = np.nan

    features = features.sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = features.groupby("ticker", sort=False)
    features["sentiment_count"] = grouped["sentiment_count_raw"].shift(1).fillna(0.0)
    features["sentiment_normalized"] = grouped["sentiment_normalized_raw"].shift(1).fillna(0.0)
    features["sentiment_missing"] = grouped["sentiment_normalized_raw"].shift(1).isna().astype(float)

    weighted = features["sentiment_count"].astype(float) * features["sentiment_normalized"].astype(float)
    features["_sentiment_weighted"] = weighted
    for window in (5, 20, 60):
        count_col = f"sentiment_count_{window}d"
        mean_col = f"sentiment_normalized_mean_{window}d"
        weighted_col = f"sentiment_weighted_{window}d"
        features[count_col] = grouped["sentiment_count"].transform(
            lambda values: values.rolling(window, min_periods=1).sum()
        )
        features[mean_col] = grouped["sentiment_normalized"].transform(
            lambda values: values.rolling(window, min_periods=1).mean()
        )
        features[weighted_col] = features.groupby("ticker", sort=False)["_sentiment_weighted"].transform(
            lambda values: values.rolling(window, min_periods=1).sum()
        )
    features = features.drop(columns=["sentiment_count_raw", "sentiment_normalized_raw", "_sentiment_weighted"])
    return features.to_dict("records")


def add_fundamental_features(feature_rows: Sequence[dict[str, object]], fundamental_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    if not feature_rows:
        return []
    features = pd.DataFrame(feature_rows).copy()
    features["ticker"] = features["ticker"].astype(str).str.upper()
    features["date"] = pd.to_datetime(features["date"].astype(str), errors="coerce")

    if not fundamental_rows:
        for column in FUNDAMENTAL_FEATURE_COLUMNS:
            features[column] = 1.0 if column == "fundamental_missing" else np.nan
        features["fundamental_staleness_days"] = np.nan
        features["date"] = features["date"].dt.strftime("%Y-%m-%d")
        return features.to_dict("records")

    fundamentals = pd.DataFrame(fundamental_rows).copy()
    fundamentals["ticker"] = fundamentals["ticker"].astype(str).str.upper()
    fundamentals["availability_date"] = pd.to_datetime(fundamentals["availability_date"].astype(str), errors="coerce")
    fundamentals = fundamentals.dropna(subset=["ticker", "availability_date"])
    fundamentals = fundamentals.drop(columns=["eodhd_symbol"], errors="ignore")
    if fundamentals.empty:
        out = features.copy()
        for column in FUNDAMENTAL_FEATURE_COLUMNS:
            out[column] = 1.0 if column == "fundamental_missing" else np.nan
        out["fundamental_staleness_days"] = np.nan
    else:
        left = features.sort_values(["date", "ticker"]).reset_index(drop=True)
        right = fundamentals.sort_values(["availability_date", "ticker"]).reset_index(drop=True)
        out = pd.merge_asof(
            left,
            right,
            left_on="date",
            right_on="availability_date",
            by="ticker",
            direction="backward",
        )
        out["fundamental_missing"] = out["availability_date"].isna().astype(float)
        out["fundamental_staleness_days"] = (out["date"] - out["availability_date"]).dt.days
        out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    for column in FUNDAMENTAL_FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = 1.0 if column == "fundamental_missing" else np.nan
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.drop(columns=["availability_date", "source_period_end", "source_frequency"], errors="ignore").to_dict("records")
