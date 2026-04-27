from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_API_BASE_URL = "https://api.massive.com"
DEFAULT_RATE_LIMIT_CALLS = 5
DEFAULT_RATE_LIMIT_PERIOD_SECONDS = 60.0
DEFAULT_BENCHMARK_TICKER = "SPY"


class MassiveError(RuntimeError):
    """Base error for Massive ingestion failures."""


class MassiveAuthError(MassiveError):
    """Raised when Massive rejects the supplied credentials."""


class MassiveAPIError(MassiveError):
    """Raised when a Massive REST call fails."""


@dataclass(frozen=True)
class MassiveCredentials:
    api_key: str | None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    s3_endpoint: str | None = None
    s3_bucket: str | None = None
    source_path: str | None = None


def load_massive_credentials(credentials_path: str | Path = "MassiveApiKey") -> MassiveCredentials:
    """Load Massive credentials from env vars and an env-style file."""
    path = Path(credentials_path)
    values: dict[str, str] = {}

    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

    def pick(name: str) -> str | None:
        return os.environ.get(name) or values.get(name)

    return MassiveCredentials(
        api_key=pick("MASSIVE_API_KEY"),
        access_key_id=pick("MASSIVE_ACCESS_KEY_ID"),
        secret_access_key=pick("MASSIVE_SECRET_ACCESS_KEY"),
        s3_endpoint=pick("MASSIVE_S3_ENDPOINT"),
        s3_bucket=pick("MASSIVE_S3_BUCKET"),
        source_path=str(path.resolve()) if path.exists() else None,
    )


@dataclass
class RateLimiter:
    """Simple rolling-window limiter for free-plan REST usage."""

    max_calls: int = DEFAULT_RATE_LIMIT_CALLS
    period_seconds: float = DEFAULT_RATE_LIMIT_PERIOD_SECONDS
    _call_times: deque[float] = field(default_factory=deque, init=False, repr=False)

    def wait(self) -> None:
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
class MassiveRESTClient:
    credentials: MassiveCredentials
    base_url: str = DEFAULT_API_BASE_URL
    rate_limiter: RateLimiter = field(default_factory=RateLimiter)
    user_agent: str = "QuantWithNNkNN/0.1"
    max_retries: int = 3

    def _build_url(self, path: str, params: dict[str, object] | None = None) -> str:
        if not self.credentials.api_key:
            raise MassiveAuthError(
                "MASSIVE_API_KEY is missing. The REST collector needs a valid Massive REST API key."
            )

        encoded: list[tuple[str, str]] = []
        for key, value in (params or {}).items():
            if value is None:
                continue
            if isinstance(value, bool):
                encoded.append((key, "true" if value else "false"))
            else:
                encoded.append((key, str(value)))
        encoded.append(("apiKey", self.credentials.api_key))
        return f"{self.base_url.rstrip('/')}{path}?{urllib.parse.urlencode(encoded)}"

    def request_json(self, path: str, params: dict[str, object] | None = None) -> dict:
        url = self._build_url(path, params=params)
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self.rate_limiter.wait()
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read().decode("utf-8")
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise MassiveAPIError("Massive returned invalid JSON.") from exc

                status = data.get("status")
                if status not in (None, "OK"):
                    raise MassiveAPIError(f"Massive returned non-OK status: {status}")
                return data
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 401 and "Unknown API Key" in body:
                    raise MassiveAuthError(
                        "Massive rejected MASSIVE_API_KEY as an unknown REST key. "
                        "The current file appears to contain S3 credentials, but the REST API "
                        "still needs a valid dashboard API key."
                    ) from exc
                if exc.code == 429 and attempt < self.max_retries:
                    time.sleep(max(self.rate_limiter.period_seconds, 65.0))
                    last_error = exc
                    continue
                raise MassiveAPIError(f"Massive request failed with HTTP {exc.code}: {body}") from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    time.sleep(min(10.0 * (attempt + 1), 30.0))
                    last_error = exc
                    continue
                raise MassiveAPIError(f"Massive request failed: {exc}") from exc

        if last_error is not None:
            raise MassiveAPIError(f"Massive request failed after retries: {last_error}")
        raise MassiveAPIError("Massive request failed for an unknown reason.")

    def get_grouped_daily_aggs(self, trade_date: str, adjusted: bool = True, include_otc: bool = False) -> dict:
        return self.request_json(
            f"/v2/aggs/grouped/locale/us/market/stocks/{trade_date}",
            params={"adjusted": adjusted, "include_otc": include_otc},
        )

    def get_ticker_details(self, ticker: str, as_of_date: str | None = None) -> dict:
        params: dict[str, object] = {}
        if as_of_date:
            params["date"] = as_of_date
        return self.request_json(f"/v3/reference/tickers/{ticker}", params=params)

    def get_ticker_range_aggs(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 50000,
    ) -> dict:
        return self.request_json(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}",
            params={"adjusted": adjusted, "sort": sort, "limit": limit},
        )


def iter_weekdays(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def normalize_grouped_bar(trade_date: str, payload: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    adjusted = bool(payload.get("adjusted", True))
    for item in payload.get("results", []) or []:
        ticker = item.get("T")
        if not ticker:
            continue
        close = item.get("c")
        volume = item.get("v")
        rows.append(
            {
                "date": trade_date,
                "ticker": ticker,
                "open": item.get("o"),
                "high": item.get("h"),
                "low": item.get("l"),
                "close": close,
                "volume": volume,
                "vwap": item.get("vw"),
                "transactions": item.get("n"),
                "timestamp_ms": item.get("t"),
                "adjusted": adjusted,
                "dollar_volume": (close * volume) if close is not None and volume is not None else None,
            }
        )
    return rows


def normalize_ticker_range_bars(payload: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ticker = payload.get("ticker")
    adjusted = bool(payload.get("adjusted", True))
    for item in payload.get("results", []) or []:
        timestamp_ms = item.get("t")
        trade_date = None
        if timestamp_ms is not None:
            trade_date = datetime.utcfromtimestamp(int(timestamp_ms) / 1000).date().isoformat()
        rows.append(
            {
                "date": trade_date,
                "ticker": ticker,
                "open": item.get("o"),
                "high": item.get("h"),
                "low": item.get("l"),
                "close": item.get("c"),
                "volume": item.get("v"),
                "vwap": item.get("vw"),
                "transactions": item.get("n"),
                "timestamp_ms": timestamp_ms,
                "adjusted": adjusted,
                "dollar_volume": (
                    item.get("c") * item.get("v")
                    if item.get("c") is not None and item.get("v") is not None
                    else None
                ),
            }
        )
    return rows


def choose_sample_universe(
    bars: Sequence[dict[str, object]],
    max_tickers: int,
    forced_tickers: Sequence[str] | None = None,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
) -> list[str]:
    forced = [ticker.upper() for ticker in (forced_tickers or []) if ticker]
    dollar_volume_by_ticker: dict[str, list[float]] = defaultdict(list)

    for row in bars:
        ticker = str(row["ticker"]).upper()
        if row.get("dollar_volume") is None:
            continue
        dollar_volume_by_ticker[ticker].append(float(row["dollar_volume"]))

    ranked = sorted(
        (
            (statistics.median(values), ticker)
            for ticker, values in dollar_volume_by_ticker.items()
            if values and ticker != benchmark_ticker
        ),
        reverse=True,
    )

    chosen = list(dict.fromkeys(forced))
    for _, ticker in ranked:
        if ticker not in chosen:
            chosen.append(ticker)
        if len(chosen) >= max_tickers:
            break

    if benchmark_ticker not in chosen:
        chosen.append(benchmark_ticker)
    return chosen


def fetch_bars_for_dates(
    client: MassiveRESTClient,
    start_date: date,
    end_date: date,
    adjusted: bool = True,
    include_otc: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trade_date in iter_weekdays(start_date, end_date):
        trade_date_str = trade_date.isoformat()
        payload = client.get_grouped_daily_aggs(
            trade_date=trade_date_str,
            adjusted=adjusted,
            include_otc=include_otc,
        )
        rows.extend(normalize_grouped_bar(trade_date_str, payload))
    return rows


def fetch_bars_for_tickers(
    client: MassiveRESTClient,
    tickers: Sequence[str],
    start_date: date,
    end_date: date,
    adjusted: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    from_date = start_date.isoformat()
    to_date = end_date.isoformat()
    for ticker in tickers:
        payload = client.get_ticker_range_aggs(
            ticker=ticker,
            from_date=from_date,
            to_date=to_date,
            adjusted=adjusted,
        )
        rows.extend(normalize_ticker_range_bars(payload))
    return rows


def fetch_ticker_details_snapshots(
    client: MassiveRESTClient,
    tickers: Sequence[str],
    as_of_date: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        payload = client.get_ticker_details(ticker, as_of_date=as_of_date)
        result = payload.get("results") or {}
        rows.append(
            {
                "ticker": result.get("ticker", ticker),
                "as_of_date": as_of_date,
                "active": result.get("active"),
                "name": result.get("name"),
                "market": result.get("market"),
                "locale": result.get("locale"),
                "type": result.get("type"),
                "primary_exchange": result.get("primary_exchange"),
                "currency_name": result.get("currency_name"),
                "sic_code": result.get("sic_code"),
                "sic_description": result.get("sic_description"),
                "market_cap": result.get("market_cap"),
                "weighted_shares_outstanding": result.get("weighted_shares_outstanding"),
                "share_class_shares_outstanding": result.get("share_class_shares_outstanding"),
                "total_employees": result.get("total_employees"),
                "list_date": result.get("list_date"),
                "homepage_url": result.get("homepage_url"),
                "cik": result.get("cik"),
                "composite_figi": result.get("composite_figi"),
                "share_class_figi": result.get("share_class_figi"),
            }
        )
    return rows


def _mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def compute_daily_features(bars: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    bars_by_ticker: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in bars:
        bars_by_ticker[str(row["ticker"])].append(dict(row))

    feature_rows: list[dict[str, object]] = []
    for rows in bars_by_ticker.values():
        rows.sort(key=lambda item: item["date"])
        prior_closes: deque[float] = deque()
        prior_returns: deque[float] = deque()
        prior_dollar_volume: deque[float] = deque()
        prior_volume: deque[float] = deque()

        for row in rows:
            close = row.get("close")
            open_ = row.get("open")
            high = row.get("high")
            low = row.get("low")
            vwap = row.get("vwap")
            volume = row.get("volume")
            dollar_volume = row.get("dollar_volume")

            prev_close = prior_closes[-1] if prior_closes else None
            return_1d = None
            log_return_1d = None
            gap_pct = None
            intraday_return = None
            hl_range_pct = None
            close_to_vwap_pct = None
            close_location = None
            true_range_pct = None

            if prev_close and close is not None and prev_close != 0:
                return_1d = (float(close) / prev_close) - 1.0
                if 1.0 + return_1d > 0:
                    log_return_1d = math.log1p(return_1d)
            if prev_close and open_ is not None and prev_close != 0:
                gap_pct = (float(open_) / prev_close) - 1.0
            if open_ not in (None, 0) and close is not None:
                intraday_return = (float(close) / float(open_)) - 1.0
            if close not in (None, 0) and high is not None and low is not None:
                hl_range_pct = (float(high) - float(low)) / float(close)
                if float(high) != float(low):
                    close_location = (float(close) - float(low)) / (float(high) - float(low))
            if prev_close and prev_close != 0 and high is not None and low is not None:
                high_f = float(high)
                low_f = float(low)
                true_range = max(high_f - low_f, abs(high_f - prev_close), abs(low_f - prev_close))
                true_range_pct = true_range / prev_close
            if close not in (None, 0) and vwap not in (None, 0):
                close_to_vwap_pct = (float(close) / float(vwap)) - 1.0

            rolling_return_5d = None
            rolling_return_20d = None
            rolling_return_60d = None
            rolling_vol_20d = None
            rolling_vol_60d = None
            rolling_avg_volume_20d = None
            rolling_avg_volume_60d = None
            rolling_avg_dollar_volume_20d = None
            rolling_avg_dollar_volume_60d = None
            sma_close_20d = None
            sma_close_60d = None
            price_vs_sma_20d = None
            price_vs_sma_60d = None
            momentum_20d = None
            momentum_60d = None
            volume_ratio_20d = None
            dollar_volume_ratio_5d = None
            volume_zscore_20d = None

            if len(prior_closes) >= 4 and close is not None and prior_closes[-4] != 0:
                rolling_return_5d = (float(close) / prior_closes[-4]) - 1.0
            if len(prior_closes) >= 19 and close is not None and prior_closes[-19] != 0:
                rolling_return_20d = (float(close) / prior_closes[-19]) - 1.0
            if len(prior_closes) >= 59 and close is not None and prior_closes[-59] != 0:
                rolling_return_60d = (float(close) / prior_closes[-59]) - 1.0
            if len(prior_returns) >= 19 and return_1d is not None:
                rolling_vol_20d = statistics.pstdev(list(prior_returns)[-19:] + [return_1d])
            if len(prior_returns) >= 59 and return_1d is not None:
                rolling_vol_60d = statistics.pstdev(list(prior_returns)[-59:] + [return_1d])
            if len(prior_volume) >= 19 and volume is not None:
                rolling_avg_volume_20d = statistics.mean(list(prior_volume)[-19:] + [float(volume)])
            if len(prior_volume) >= 59 and volume is not None:
                rolling_avg_volume_60d = statistics.mean(list(prior_volume)[-59:] + [float(volume)])
            if len(prior_dollar_volume) >= 19 and dollar_volume is not None:
                rolling_avg_dollar_volume_20d = statistics.mean(
                    list(prior_dollar_volume)[-19:] + [float(dollar_volume)]
                )
            if len(prior_dollar_volume) >= 59 and dollar_volume is not None:
                rolling_avg_dollar_volume_60d = statistics.mean(
                    list(prior_dollar_volume)[-59:] + [float(dollar_volume)]
                )
            if len(prior_dollar_volume) >= 4 and dollar_volume not in (None, 0):
                avg_dollar_volume_5d = statistics.mean(list(prior_dollar_volume)[-4:] + [float(dollar_volume)])
                if avg_dollar_volume_5d != 0:
                    dollar_volume_ratio_5d = float(dollar_volume) / avg_dollar_volume_5d
            if len(prior_volume) >= 19 and volume is not None:
                volume_window_20d = list(prior_volume)[-19:] + [float(volume)]
                volume_std_20d = statistics.pstdev(volume_window_20d)
                if volume_std_20d != 0:
                    volume_zscore_20d = (float(volume) - statistics.mean(volume_window_20d)) / volume_std_20d
            if len(prior_closes) >= 19:
                sma_close_20d = _mean_or_none(list(prior_closes)[-19:] + ([float(close)] if close is not None else []))
            if len(prior_closes) >= 59:
                sma_close_60d = _mean_or_none(list(prior_closes)[-59:] + ([float(close)] if close is not None else []))
            if close not in (None, 0) and sma_close_20d not in (None, 0):
                price_vs_sma_20d = (float(close) / float(sma_close_20d)) - 1.0
            if close not in (None, 0) and sma_close_60d not in (None, 0):
                price_vs_sma_60d = (float(close) / float(sma_close_60d)) - 1.0
            if rolling_return_20d is not None:
                momentum_20d = rolling_return_20d
            if rolling_return_60d is not None:
                momentum_60d = rolling_return_60d
            if volume not in (None, 0) and rolling_avg_volume_20d not in (None, 0):
                volume_ratio_20d = float(volume) / float(rolling_avg_volume_20d)

            feature_row = dict(row)
            feature_row.update(
                {
                    "prev_close": prev_close,
                    "return_1d": return_1d,
                    "log_return_1d": log_return_1d,
                    "gap_pct": gap_pct,
                    "intraday_return": intraday_return,
                    "hl_range_pct": hl_range_pct,
                    "close_to_vwap_pct": close_to_vwap_pct,
                    "close_location": close_location,
                    "true_range_pct": true_range_pct,
                    "rolling_return_5d": rolling_return_5d,
                    "rolling_return_20d": rolling_return_20d,
                    "rolling_return_60d": rolling_return_60d,
                    "rolling_vol_20d": rolling_vol_20d,
                    "rolling_vol_60d": rolling_vol_60d,
                    "rolling_avg_volume_20d": rolling_avg_volume_20d,
                    "rolling_avg_volume_60d": rolling_avg_volume_60d,
                    "rolling_avg_dollar_volume_20d": rolling_avg_dollar_volume_20d,
                    "rolling_avg_dollar_volume_60d": rolling_avg_dollar_volume_60d,
                    "sma_close_20d": sma_close_20d,
                    "sma_close_60d": sma_close_60d,
                    "price_vs_sma_20d": price_vs_sma_20d,
                    "price_vs_sma_60d": price_vs_sma_60d,
                    "momentum_20d": momentum_20d,
                    "momentum_60d": momentum_60d,
                    "volume_ratio_20d": volume_ratio_20d,
                    "dollar_volume_ratio_5d": dollar_volume_ratio_5d,
                    "volume_zscore_20d": volume_zscore_20d,
                    "has_prev_close": prev_close is not None,
                    "has_20d_history": len(prior_closes) >= 19,
                    "has_60d_history": len(prior_closes) >= 59,
                }
            )
            feature_rows.append(feature_row)

            if close is not None:
                prior_closes.append(float(close))
                if len(prior_closes) > 60:
                    prior_closes.popleft()
            if return_1d is not None:
                prior_returns.append(return_1d)
                if len(prior_returns) > 60:
                    prior_returns.popleft()
            if volume is not None:
                prior_volume.append(float(volume))
                if len(prior_volume) > 60:
                    prior_volume.popleft()
            if dollar_volume is not None:
                prior_dollar_volume.append(float(dollar_volume))
                if len(prior_dollar_volume) > 60:
                    prior_dollar_volume.popleft()

    feature_rows.sort(key=lambda item: (item["ticker"], item["date"]))
    return feature_rows


def load_daily_bars_csv(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    numeric_fields = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "transactions",
        "timestamp_ms",
        "dollar_volume",
    }
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if value == "":
                    parsed[key] = None
                elif key in numeric_fields:
                    parsed[key] = float(value)
                elif key == "adjusted":
                    parsed[key] = value.lower() == "true"
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def compute_episode_index(
    feature_rows: Sequence[dict[str, object]],
    window_length: int,
    horizon_days: int,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
) -> list[dict[str, object]]:
    by_ticker: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in feature_rows:
        by_ticker[str(row["ticker"])].append(dict(row))

    benchmark_rows = by_ticker.get(benchmark_ticker, [])
    benchmark_by_date = {str(row["date"]): row for row in benchmark_rows}
    benchmark_dates = [str(row["date"]) for row in benchmark_rows]

    episodes: list[dict[str, object]] = []
    for ticker, rows in by_ticker.items():
        if ticker == benchmark_ticker:
            continue
        rows.sort(key=lambda item: item["date"])
        if len(rows) < window_length + horizon_days:
            continue

        dates = [str(row["date"]) for row in rows]
        for anchor_index in range(window_length - 1, len(rows) - horizon_days):
            anchor_row = rows[anchor_index]
            future_row = rows[anchor_index + horizon_days]
            anchor_close = anchor_row.get("close")
            future_close = future_row.get("close")
            if anchor_close in (None, 0) or future_close is None:
                continue

            anchor_date = dates[anchor_index]
            future_date = dates[anchor_index + horizon_days]
            if anchor_date not in benchmark_by_date or future_date not in benchmark_by_date:
                continue

            benchmark_anchor = benchmark_by_date[anchor_date].get("close")
            benchmark_future = benchmark_by_date[future_date].get("close")
            if benchmark_anchor in (None, 0) or benchmark_future is None:
                continue

            target_return = (float(future_close) / float(anchor_close)) - 1.0
            benchmark_return = (float(benchmark_future) / float(benchmark_anchor)) - 1.0

            episodes.append(
                {
                    "ticker": ticker,
                    "anchor_date": anchor_date,
                    "window_start_date": dates[anchor_index - window_length + 1],
                    "window_end_date": anchor_date,
                    "target_horizon_days": horizon_days,
                    "target_return": target_return,
                    "market_adjusted_target_return": target_return - benchmark_return,
                    "benchmark_ticker": benchmark_ticker,
                    "benchmark_anchor_date": anchor_date,
                    "benchmark_future_date": future_date,
                    "available_window_rows": window_length,
                }
            )

    episodes.sort(key=lambda item: (item["anchor_date"], item["ticker"]))
    return episodes


def _csv_headers(rows: Sequence[dict[str, object]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = _csv_headers(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_csv(path: Path, rows: Sequence[dict[str, object]], headers: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not headers:
        return

    file_exists = path.exists()
    fieldnames = list(headers) if headers else _csv_headers(rows)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _clean_wiki_markup(value: str) -> str:
    text = value.strip()
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*?/?>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"<ref[^/]*/>", "", text)
    text = re.sub(r"\{\{[^{}|]+\|([^{}|]+)\}\}", r"\1", text)
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = text.replace("'''", "").replace("''", "")
    text = text.replace("&amp;", "&")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_sp500_constituents_from_wikipedia() -> list[dict[str, str]]:
    """
    Fetch the current S&P 500 constituent table from Wikipedia raw wikitext.

    This is intentionally lightweight so we do not add parsing dependencies for a
    one-shot current constituent bootstrap. The output represents today's
    constituent list and therefore carries survivorship-bias risk if used for
    long backfills.
    """
    url = "https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&action=raw"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8", errors="replace")

    table_marker = '{| class="wikitable sortable mw-collapsible sticky-header" id="constituents"'
    start = text.index(table_marker)
    table_text = text[start:]
    end = table_text.index("\n|}")
    table_text = table_text[:end]

    parts = table_text.split("\n|-")
    rows: list[dict[str, str]] = []
    for chunk in parts[1:]:
        row_text = chunk.strip()
        if not row_text:
            continue
        row_text = row_text.replace("\n|", "||")
        raw_cells = [cell for cell in row_text.split("||") if cell.strip()]
        cells = [_clean_wiki_markup(cell.lstrip("|")) for cell in raw_cells]
        if len(cells) < 8:
            continue
        rows.append(
            {
                "symbol": cells[0],
                "security": cells[1],
                "gics_sector": cells[2],
                "gics_sub_industry": cells[3],
                "headquarters_location": cells[4],
                "date_added": cells[5],
                "cik": cells[6],
                "founded": cells[7],
                "constituents_source": url,
            }
        )
    return rows


def default_collection_manifest(
    *,
    start_date: str,
    end_date: str,
    max_tickers: int,
    selected_tickers: Sequence[str],
    window_length: int,
    horizon_days: int,
    benchmark_ticker: str,
) -> dict[str, object]:
    return {
        "dataset_name": "massive_stage1_v1_small",
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "parameters": {
            "start_date": start_date,
            "end_date": end_date,
            "max_tickers": max_tickers,
            "selected_tickers": list(selected_tickers),
            "window_length": window_length,
            "horizon_days": horizon_days,
            "benchmark_ticker": benchmark_ticker,
            "granularity": "daily",
        },
        "feature_sources": [
            {
                "name": "daily_market_bars",
                "vendor": "Massive",
                "endpoint": "/v2/aggs/grouped/locale/us/market/stocks/{date}",
                "entity_key": "ticker",
                "event_timestamp_field": "date",
                "effective_date_field": "date",
                "join_type": "exact date join",
                "fill_policy": "no fill at raw layer",
                "lag_policy": "features use dates <= anchor_date only",
                "notes": "Adjusted daily OHLCV and VWAP bars for the full US stock market by trading date.",
            },
            {
                "name": "ticker_reference_snapshot",
                "vendor": "Massive",
                "endpoint": "/v3/reference/tickers/{ticker}",
                "entity_key": "ticker",
                "event_timestamp_field": "date query parameter",
                "effective_date_field": "date query parameter",
                "join_type": "as-of static snapshot joined by ticker",
                "fill_policy": "carry forward in downstream modeling only if economically valid",
                "lag_policy": "queried as-of collection end date for this small v1 sample",
                "notes": "Provides slower-moving metadata such as exchange, SIC, market cap, shares outstanding, and identifiers.",
            },
            {
                "name": "fundamentals_snapshot",
                "vendor": "Massive",
                "endpoint": "not collected in this script",
                "entity_key": "ticker",
                "event_timestamp_field": "n/a",
                "effective_date_field": "n/a",
                "join_type": "not joined",
                "fill_policy": "n/a",
                "lag_policy": "n/a",
                "notes": "Skipped for the small free-plan bootstrap because entitlement and point-in-time semantics need explicit verification first.",
            },
        ],
        "known_risks": [
            "The current script only collects static reference fields from ticker details; true filing-dated fundamentals are intentionally excluded until entitlement and time semantics are confirmed.",
            "Ticker details are queried as-of the collection end date, which is acceptable for a tiny bootstrap sample but should move to anchor-date snapshots before larger historical studies.",
            "The Massive REST API key in MassiveApiKey must be a valid dashboard REST key; S3 credentials alone are insufficient for this script.",
        ],
    }
