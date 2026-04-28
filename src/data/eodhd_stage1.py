from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_API_BASE_URL = "https://eodhd.com/api"
DEFAULT_BENCHMARK_TICKER = "SPY"
DEFAULT_RATE_LIMIT_CALLS = 200
DEFAULT_RATE_LIMIT_PERIOD_SECONDS = 60.0
DEFAULT_EXCHANGES = ("NYSE", "NASDAQ", "AMEX", "BATS")
DEFAULT_SYMBOL_TYPE = "common_stock"
UNKNOWN_SECTOR = "Unknown"

OTC_EXCHANGES = {
    "OTC",
    "OTCBB",
    "OTCGREY",
    "OTCMKTS",
    "OTCQB",
    "OTCQX",
    "OTCM",
    "PINK",
}

EODHD_DAILY_BAR_HEADERS = [
    "date",
    "ticker",
    "eodhd_symbol",
    "exchange",
    "open",
    "high",
    "low",
    "close",
    "raw_close",
    "adjusted_close",
    "volume",
    "adjustment_factor",
    "adjusted",
    "dollar_volume",
]

EODHD_UNIVERSE_HEADERS = [
    "symbol",
    "ticker",
    "eodhd_symbol",
    "name",
    "country",
    "exchange",
    "currency",
    "type",
    "isin",
    "is_delisted",
]

EODHD_METADATA_HEADERS = [
    "symbol",
    "ticker",
    "eodhd_symbol",
    "name",
    "country",
    "exchange",
    "currency",
    "type",
    "isin",
    "sector",
    "industry",
    "gics_sector",
    "gics_sub_industry",
    "is_delisted",
    "delisted_date",
    "metadata_source",
]


class EODHDError(RuntimeError):
    """Base error for EODHD ingestion failures."""


class EODHDAuthError(EODHDError):
    """Raised when EODHD credentials are missing or rejected."""


class EODHDAPIError(EODHDError):
    """Raised when an EODHD API call fails."""


@dataclass(frozen=True)
class EODHDCredentials:
    api_key: str | None
    source_path: str | None = None


def load_eodhd_credentials(credentials_path: str | Path = "EODHD_api_key") -> EODHDCredentials:
    path = Path(credentials_path)
    values: dict[str, str] = {}
    raw_token: str | None = None

    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        raw_token = text or None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

    api_key = os.environ.get("EODHD_API_KEY") or values.get("EODHD_API_KEY") or raw_token
    return EODHDCredentials(
        api_key=api_key,
        source_path=str(path.resolve()) if path.exists() else None,
    )


@dataclass
class RateLimiter:
    max_calls: int = DEFAULT_RATE_LIMIT_CALLS
    period_seconds: float = DEFAULT_RATE_LIMIT_PERIOD_SECONDS
    _call_times: deque[float] = field(default_factory=deque, init=False, repr=False)

    def wait(self) -> None:
        if self.max_calls <= 0:
            return
        now = time.monotonic()
        while self._call_times and now - self._call_times[0] >= self.period_seconds:
            self._call_times.popleft()
        if len(self._call_times) >= self.max_calls:
            sleep_for = self.period_seconds - (now - self._call_times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            while self._call_times and now - self._call_times[0] >= self.period_seconds:
                self._call_times.popleft()
        self._call_times.append(time.monotonic())


@dataclass
class EODHDRESTClient:
    credentials: EODHDCredentials
    base_url: str = DEFAULT_API_BASE_URL
    rate_limiter: RateLimiter = field(default_factory=RateLimiter)
    user_agent: str = "QuantWithNNkNN/0.1"
    max_retries: int = 3

    def _build_url(self, path: str, params: dict[str, object] | None = None) -> str:
        if not self.credentials.api_key:
            raise EODHDAuthError("EODHD_API_KEY is missing. Set it in the environment or EODHD_api_key.")

        encoded: list[tuple[str, str]] = []
        for key, value in (params or {}).items():
            if value is None:
                continue
            if isinstance(value, bool):
                encoded.append((key, "1" if value else "0"))
            else:
                encoded.append((key, str(value)))
        encoded.append(("api_token", self.credentials.api_key))
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}?{urllib.parse.urlencode(encoded)}"

    def request_json(self, path: str, params: dict[str, object] | None = None) -> object:
        url = self._build_url(path, params=params)
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self.rate_limiter.wait()
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = response.read().decode("utf-8")
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise EODHDAPIError("EODHD returned invalid JSON.") from exc
                if isinstance(data, dict):
                    message = str(data.get("message") or data.get("error") or "")
                    if "invalid api" in message.lower() or "api token" in message.lower() and "invalid" in message.lower():
                        raise EODHDAuthError("EODHD rejected the configured API token.")
                    if data.get("errors"):
                        raise EODHDAPIError(f"EODHD returned errors: {data.get('errors')}")
                return data
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code in {401, 403}:
                    raise EODHDAuthError(f"EODHD authorization failed with HTTP {exc.code}: {body}") from exc
                if exc.code == 429 and attempt < self.max_retries:
                    time.sleep(max(self.rate_limiter.period_seconds, 60.0))
                    last_error = exc
                    continue
                raise EODHDAPIError(f"EODHD request failed with HTTP {exc.code}: {body}") from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    time.sleep(min(10.0 * (attempt + 1), 30.0))
                    last_error = exc
                    continue
                raise EODHDAPIError(f"EODHD request failed: {exc}") from exc

        if last_error is not None:
            raise EODHDAPIError(f"EODHD request failed after retries: {last_error}")
        raise EODHDAPIError("EODHD request failed for an unknown reason.")

    def get_eod(
        self,
        symbol: str,
        *,
        from_date: str,
        to_date: str,
        period: str = "d",
    ) -> list[dict[str, object]]:
        data = self.request_json(
            f"/eod/{symbol}",
            params={"from": from_date, "to": to_date, "period": period, "fmt": "json"},
        )
        if isinstance(data, list):
            return [dict(row) for row in data if isinstance(row, dict)]
        raise EODHDAPIError(f"EODHD EOD response for {symbol} was not a list.")

    def get_exchange_symbol_list(
        self,
        exchange: str,
        *,
        symbol_type: str = DEFAULT_SYMBOL_TYPE,
        include_delisted: bool = True,
    ) -> list[dict[str, object]]:
        """Fetch one EODHD symbol-list view.

        For U.S. lists, EODHD's delisted=1 parameter behaves as a delisted-only
        view, so callers that need current plus delisted names should merge two
        calls: include_delisted=False and include_delisted=True.
        """
        data = self.request_json(
            f"/exchange-symbol-list/{exchange}",
            params={
                "fmt": "json",
                "type": symbol_type,
                "delisted": 1 if include_delisted else None,
            },
        )
        if isinstance(data, list):
            return [dict(row) for row in data if isinstance(row, dict)]
        raise EODHDAPIError(f"EODHD symbol-list response for {exchange} was not a list.")

    def get_fundamentals_general(self, symbol: str) -> dict[str, object]:
        filters = ",".join(
            [
                "General::Code",
                "General::Type",
                "General::Name",
                "General::Exchange",
                "General::CurrencyCode",
                "General::CountryName",
                "General::ISIN",
                "General::Sector",
                "General::Industry",
                "General::GicSector",
                "General::IsDelisted",
                "General::DelistedDate",
                "General::PrimaryTicker",
            ]
        )
        data = self.request_json(
            f"/v1.1/fundamentals/{symbol}",
            params={"filter": filters, "fmt": "json"},
        )
        if isinstance(data, dict):
            return dict(data)
        raise EODHDAPIError(f"EODHD fundamentals response for {symbol} was not an object.")


def eodhd_symbol_for_code(code: str, default_exchange: str = "US") -> str:
    cleaned = str(code).strip().upper()
    if not cleaned:
        return cleaned
    suffix = f".{default_exchange.upper()}"
    return cleaned if cleaned.endswith(suffix) else f"{cleaned}{suffix}"


def internal_ticker_from_eodhd_symbol(symbol: str) -> str:
    cleaned = str(symbol).strip().upper()
    return cleaned[:-3] if cleaned.endswith(".US") else cleaned


def _as_float(value: object) -> float | None:
    if value in (None, "", "NA", "N/A"):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _as_bool(value: object) -> bool | None:
    if value in (None, "", "NA", "N/A"):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def normalize_eodhd_eod_rows(
    rows: Sequence[dict[str, object]],
    *,
    symbol: str,
    ticker: str | None = None,
    exchange: str | None = None,
    adjusted: bool = True,
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    internal_ticker = (ticker or internal_ticker_from_eodhd_symbol(symbol)).upper()
    for row in rows:
        raw_close = _as_float(row.get("close"))
        adjusted_close = _as_float(row.get("adjusted_close"))
        close = adjusted_close if adjusted and adjusted_close is not None else raw_close
        adjustment_factor = 1.0
        if adjusted and raw_close not in (None, 0) and close is not None:
            adjustment_factor = close / raw_close

        open_value = _as_float(row.get("open"))
        high_value = _as_float(row.get("high"))
        low_value = _as_float(row.get("low"))
        volume = _as_float(row.get("volume"))

        normalized.append(
            {
                "date": str(row.get("date")),
                "ticker": internal_ticker,
                "eodhd_symbol": symbol.upper(),
                "exchange": exchange,
                "open": open_value * adjustment_factor if open_value is not None else None,
                "high": high_value * adjustment_factor if high_value is not None else None,
                "low": low_value * adjustment_factor if low_value is not None else None,
                "close": close,
                "raw_close": raw_close,
                "adjusted_close": adjusted_close,
                "volume": volume,
                "adjustment_factor": adjustment_factor,
                "adjusted": adjusted,
                "dollar_volume": close * volume if close is not None and volume is not None else None,
            }
        )
    return [row for row in normalized if row.get("date") and row.get("close") is not None]


def normalize_exchange_symbol_rows(
    rows: Sequence[dict[str, object]],
    *,
    exchange: str,
    include_otc: bool = False,
    is_delisted: bool | None = None,
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        code = str(row.get("Code") or row.get("code") or "").strip().upper()
        if not code:
            continue
        row_exchange = str(row.get("Exchange") or exchange or "").strip().upper()
        if not include_otc and row_exchange in OTC_EXCHANGES:
            continue
        currency = str(row.get("Currency") or row.get("currency") or "").strip().upper()
        if currency and currency != "USD":
            continue
        row_type = str(row.get("Type") or row.get("type") or "").strip()
        if row_type and "common" not in row_type.lower():
            continue
        eodhd_symbol = eodhd_symbol_for_code(code)
        if eodhd_symbol in seen:
            continue
        seen.add(eodhd_symbol)
        row_is_delisted = _as_bool(row.get("IsDelisted") or row.get("isDelisted") or row.get("is_delisted"))
        normalized.append(
            {
                "symbol": code,
                "ticker": internal_ticker_from_eodhd_symbol(eodhd_symbol),
                "eodhd_symbol": eodhd_symbol,
                "name": row.get("Name") or row.get("name"),
                "country": row.get("Country") or row.get("country"),
                "exchange": row_exchange,
                "currency": currency or None,
                "type": row_type or None,
                "isin": row.get("Isin") or row.get("ISIN") or row.get("isin"),
                "is_delisted": row_is_delisted if row_is_delisted is not None else is_delisted,
            }
        )
    return sorted(normalized, key=lambda item: (str(item["ticker"]), str(item.get("exchange") or "")))


def merge_universe_rows(rows_by_exchange: Iterable[Sequence[dict[str, object]]]) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for rows in rows_by_exchange:
        for row in rows:
            symbol = str(row["eodhd_symbol"]).upper()
            if symbol not in merged:
                merged[symbol] = dict(row)
    return sorted(merged.values(), key=lambda item: str(item["eodhd_symbol"]))


def parse_fundamentals_general(payload: dict[str, object]) -> dict[str, object]:
    general = payload.get("General")
    if isinstance(general, dict):
        source = general
    else:
        source = {
            key.split("::", 1)[1] if "::" in key else key: value
            for key, value in payload.items()
        }

    gics_sector = source.get("GicSector") or source.get("GICS Sector") or source.get("Sector") or UNKNOWN_SECTOR
    industry = source.get("Industry") or UNKNOWN_SECTOR
    return {
        "name": source.get("Name"),
        "sector": source.get("Sector") or UNKNOWN_SECTOR,
        "industry": industry,
        "gics_sector": gics_sector,
        "gics_sub_industry": industry,
        "is_delisted": _as_bool(source.get("IsDelisted")),
        "delisted_date": source.get("DelistedDate"),
        "primary_ticker": source.get("PrimaryTicker"),
        "currency": source.get("CurrencyCode"),
        "country": source.get("CountryName"),
        "isin": source.get("ISIN"),
        "type": source.get("Type"),
        "exchange": source.get("Exchange"),
    }


def build_metadata_row(
    universe_row: dict[str, object],
    fundamentals: dict[str, object] | None = None,
) -> dict[str, object]:
    parsed = parse_fundamentals_general(fundamentals or {}) if fundamentals else {}
    return {
        "symbol": universe_row.get("symbol"),
        "ticker": universe_row.get("ticker"),
        "eodhd_symbol": universe_row.get("eodhd_symbol"),
        "name": parsed.get("name") or universe_row.get("name"),
        "country": parsed.get("country") or universe_row.get("country"),
        "exchange": parsed.get("exchange") or universe_row.get("exchange"),
        "currency": parsed.get("currency") or universe_row.get("currency"),
        "type": parsed.get("type") or universe_row.get("type"),
        "isin": parsed.get("isin") or universe_row.get("isin"),
        "sector": parsed.get("sector") or UNKNOWN_SECTOR,
        "industry": parsed.get("industry") or UNKNOWN_SECTOR,
        "gics_sector": parsed.get("gics_sector") or UNKNOWN_SECTOR,
        "gics_sub_industry": parsed.get("gics_sub_industry") or UNKNOWN_SECTOR,
        "is_delisted": parsed.get("is_delisted")
        if parsed.get("is_delisted") is not None
        else universe_row.get("is_delisted"),
        "delisted_date": parsed.get("delisted_date"),
        "metadata_source": "eodhd_fundamentals_general" if fundamentals else "eodhd_symbol_list",
    }
