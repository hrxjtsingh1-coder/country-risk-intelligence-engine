"""
Public-data collectors and indicator-panel construction.

Primary source:
    World Bank Indicators API

Optional enrichments:
    FRED public graph CSV for the US federal-funds rate
    ECB/BIS endpoints can be added through config without changing the panel
    contract.

The collector is deliberately fault-tolerant: a source failure for one
indicator does not destroy all other country-year observations.

All API responses are cached in a local SQLite database (src/cache/api_cache.py).
On live fetch failure, the last-known-good cached response is returned with a
visible staleness flag so the dashboard can show "using cached data from [date]".
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.cache.api_cache import (
    get_cached,
    get_last_known_good,
    store_response,
)
from src.indicators.imf_collector import fetch_imf_indicator

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "indicators.yaml"

LOG = logging.getLogger("country-risk.collectors")

WB_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

DEFAULT_TTL_SECONDS = 6 * 60 * 60  # 6 hours


@dataclass
class FetchMetadata:
    """Tracks which sources are fresh vs cached for provenance reporting."""

    fresh_sources: list[str] = field(default_factory=list)
    cached_sources: list[str] = field(default_factory=list)
    cache_dates: dict[str, str] = field(default_factory=dict)  # source -> "DD Mon YYYY, HH:MM UTC"


def _config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def indicator_records() -> list[dict]:
    cfg = _config()
    records = cfg.get("indicators", []) if isinstance(cfg, dict) else []
    return [r for r in records if isinstance(r, dict)]


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "country-risk-intelligence-engine/1.0",
            "Accept": "application/json,text/csv;q=0.9,*/*;q=0.8",
        }
    )
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _world_bank_indicator(
    session: requests.Session,
    iso3: str,
    wb_code: str,
    start: int,
    end: int,
) -> pd.DataFrame:
    url = WB_URL.format(country=iso3, indicator=wb_code)
    params: dict[str, str | int] = {
        "format": "json",
        "per_page": 1000,
        "date": f"{start}:{end}",
    }

    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, list) or len(payload) < 2:
        return pd.DataFrame()

    records = payload[1] or []
    rows = []

    for item in records:
        year = item.get("date")
        value = item.get("value")

        try:
            year_int = int(year)
        except (TypeError, ValueError):
            continue

        rows.append(
            {
                "country_iso3": iso3.upper(),
                "year": year_int,
                "value": value,
                "source": "World Bank",
                "flag": "ok",
            }
        )

    return pd.DataFrame(rows)


def _fred_annual_mean(
    session: requests.Session,
    series_id: str,
    start: int,
    end: int,
) -> pd.DataFrame:
    url = FRED_CSV_URL.format(series=series_id)
    response = session.get(url, timeout=30)
    response.raise_for_status()

    data = pd.read_csv(io.StringIO(response.text))
    if data.empty:
        return pd.DataFrame()

    date_col = "observation_date"
    value_col = series_id

    if date_col not in data.columns or value_col not in data.columns:
        return pd.DataFrame()

    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[date_col, value_col])

    data["year"] = data[date_col].dt.year.astype(int)
    data = data[data["year"].between(start, end)]

    annual = data.groupby("year", as_index=False)[[value_col]].mean()

    return annual.rename(columns={value_col: "value"})[["year", "value"]]


def _build_fx_depreciation(
    session: requests.Session,
    iso3: str,
    start: int,
    end: int,
) -> pd.DataFrame:
    """Use the World Bank official exchange-rate indicator and calculate YoY."""
    raw = _world_bank_indicator(
        session,
        iso3,
        "PA.NUS.FCRF",
        start - 1,
        end,
    )

    if raw.empty:
        return pd.DataFrame()

    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["value"]).sort_values("year")
    raw["fx_yoy"] = raw["value"].pct_change() * 100.0

    out = raw[raw["year"].between(start, end)].copy()
    out["value"] = out["fx_yoy"]
    out["source"] = "World Bank; derived from PA.NUS.FCRF"
    out["indicator_code"] = "FX_YOY_DEPRECIATION_PCT"

    return out[["country_iso3", "indicator_code", "year", "value", "source"]]


def _policy_rate_for_country(
    session: requests.Session,
    iso3: str,
    start: int,
    end: int,
) -> pd.DataFrame:
    """Currently supplies the US policy proxy from FRED DFF.

    Other countries remain intentionally missing rather than receiving a
    fabricated policy rate. The scoring layer renormalizes available weights.
    """
    if iso3.upper() != "USA":
        return pd.DataFrame()

    annual = _fred_annual_mean(session, "DFF", start - 1, end)
    if annual.empty:
        return pd.DataFrame()

    annual["change_bps"] = annual["value"].diff() * 100.0
    annual = annual[annual["year"].between(start, end)].copy()
    annual["country_iso3"] = iso3.upper()
    annual["indicator_code"] = "POLICY_RATE_YOY_CHANGE_BPS"
    annual["source"] = "FRED DFF; annual mean change in basis points"

    return annual[["country_iso3", "indicator_code", "year", "change_bps", "source"]].rename(
        columns={"change_bps": "value"}
    )


def _world_bank_indicator_batch(
    session: requests.Session,
    iso3_codes: list[str],
    wb_code: str,
    start: int,
    end: int,
    chunk_size: int = 60,
) -> pd.DataFrame:
    """
    Same as _world_bank_indicator, but fetches ALL given countries for one
    indicator in a single request (World Bank's API accepts semicolon-joined
    ISO3 codes in the URL path). This is what fetch_live_panel() actually
    calls: for a 20-country panel this turns "20 countries x 9 indicators
    = 180 requests" into "9 requests", one per indicator — see Phase 5.
    Chunked defensively at `chunk_size` countries per request in case the
    configured country list ever grows past what a single URL should carry.
    """
    frames = []
    codes = [c.strip().upper() for c in iso3_codes if c and c.strip()]
    for i in range(0, len(codes), chunk_size):
        chunk = codes[i : i + chunk_size]
        url = WB_URL.format(country=";".join(chunk), indicator=wb_code)
        params: dict[str, str | int] = {"format": "json", "per_page": 20000, "date": f"{start}:{end}"}

        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            # HTTP 200 with no data rows — bad indicator code or genuinely no
            # data for this slice. Not an error; just nothing to add.
            continue

        for item in payload[1]:
            value = item.get("value")
            if value is None:
                continue
            try:
                year_int = int(item.get("date"))
            except (TypeError, ValueError):
                continue
            iso3 = item.get("countryiso3code") or (item.get("country") or {}).get("id")
            if not iso3:
                continue
            frames.append(
                {
                    "country_iso3": str(iso3).upper(),
                    "year": year_int,
                    "value": value,
                    "source": "World Bank",
                    "flag": "ok",
                }
            )
    return pd.DataFrame(frames)


def _cached_world_bank_batch(
    session: requests.Session,
    iso3_codes: list[str],
    wb_code: str,
    start: int,
    end: int,
    fetch_meta: FetchMetadata,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    chunk_size: int = 60,
) -> pd.DataFrame:
    """World Bank batch fetch with persistent cache and last-known-good fallback."""
    source = "World Bank"
    codes_key = ";".join(sorted(iso3_codes))
    url = WB_URL.format(country=codes_key, indicator=wb_code)
    params = {"format": "json", "per_page": 20000, "date": f"{start}:{end}"}

    # 1. Check cache
    cached = get_cached(source, url, params, ttl_seconds)
    if cached is not None:
        fetch_meta.fresh_sources.append(source)
        LOG.info("Cache HIT for World Bank indicator %s", wb_code)
        # Parse the cached JSON response
        payload = cached.data
        return _parse_wb_batch_response(payload, iso3_codes, wb_code)

    # 2. Fetch live
    try:
        raw_df = _world_bank_indicator_batch(session, iso3_codes, wb_code, start, end, chunk_size)
        if not raw_df.empty:
            # Store in cache — reconstruct the JSON payload shape for consistency
            # We store the raw DataFrame as JSON records for simplicity
            cache_payload = raw_df.to_dict(orient="records")
            store_response(
                source=source,
                url=url,
                params=params,
                response_body=json.dumps(cache_payload),
                status_code=200,
                ttl_seconds=ttl_seconds,
            )
            fetch_meta.fresh_sources.append(source)
            return raw_df
    except requests.RequestException as exc:
        LOG.warning("Live fetch failed for WB indicator %s: %s — trying cache", wb_code, exc)

    # 3. Fallback to last-known-good
    lkg = get_last_known_good(source, url, params)
    if lkg is not None:
        fetch_meta.cached_sources.append(source)
        fetch_meta.cache_dates[source] = lkg.cached_at or "unknown"
        LOG.warning("Using cached World Bank data from %s for indicator %s", lkg.cached_at, wb_code)
        if isinstance(lkg.data, list):
            # Reconstruct DataFrame from cached records
            return pd.DataFrame(lkg.data)
        return _parse_wb_batch_response(lkg.data, iso3_codes, wb_code)

    return pd.DataFrame()


def _parse_wb_batch_response(
    payload: Any,
    iso3_codes: list[str],
    wb_code: str,
) -> pd.DataFrame:
    """Parse a World Bank API batch response (JSON list format) into a DataFrame."""
    frames = []

    # Handle both raw API format and cached records format
    if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "country_iso3" in payload[0]:
        # Already parsed records (from cache)
        for item in payload:
            frames.append(
                {
                    "country_iso3": str(item.get("country_iso3", "")).upper(),
                    "year": int(item["year"]),
                    "value": item["value"],
                    "source": "World Bank",
                    "flag": "ok",
                }
            )
    elif isinstance(payload, list) and len(payload) >= 2 and payload[1] is not None:
        # Raw World Bank API format
        for item in payload[1]:
            value = item.get("value")
            if value is None:
                continue
            try:
                year_int = int(item.get("date"))
            except (TypeError, ValueError):
                continue
            iso3 = item.get("countryiso3code") or (item.get("country") or {}).get("id")
            if not iso3:
                continue
            frames.append(
                {
                    "country_iso3": str(iso3).upper(),
                    "year": year_int,
                    "value": value,
                    "source": "World Bank",
                    "flag": "ok",
                }
            )

    return pd.DataFrame(frames) if frames else pd.DataFrame()


def _cached_fred_fetch(
    session: requests.Session,
    series_id: str,
    start: int,
    end: int,
    fetch_meta: FetchMetadata,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> pd.DataFrame:
    """FRED CSV fetch with persistent cache and last-known-good fallback."""
    source = "FRED"
    url = FRED_CSV_URL.format(series=series_id)
    params = {"start": start, "end": end}

    # 1. Check cache
    cached = get_cached(source, url, params, ttl_seconds)
    if cached is not None:
        fetch_meta.fresh_sources.append(source)
        LOG.info("Cache HIT for FRED series %s", series_id)
        return _parse_fred_csv(cached.data, series_id, start, end)

    # 2. Fetch live
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        csv_text = response.text
        # Store raw CSV text in cache
        store_response(
            source=source,
            url=url,
            params=params,
            response_body=json.dumps({"csv": csv_text}),
            status_code=200,
            ttl_seconds=ttl_seconds,
        )
        fetch_meta.fresh_sources.append(source)
        return _parse_fred_csv({"csv": csv_text}, series_id, start, end)
    except requests.RequestException as exc:
        LOG.warning("Live fetch failed for FRED series %s: %s — trying cache", series_id, exc)

    # 3. Fallback to last-known-good
    lkg = get_last_known_good(source, url, params)
    if lkg is not None:
        fetch_meta.cached_sources.append(source)
        fetch_meta.cache_dates[source] = lkg.cached_at or "unknown"
        LOG.warning("Using cached FRED data from %s for series %s", lkg.cached_at, series_id)
        return _parse_fred_csv(lkg.data, series_id, start, end)

    return pd.DataFrame()


def _parse_fred_csv(data: Any, series_id: str, start: int, end: int) -> pd.DataFrame:
    """Parse a FRED CSV response (possibly from cache) into a DataFrame."""
    if isinstance(data, dict) and "csv" in data:
        csv_text = data["csv"]
    elif isinstance(data, str):
        csv_text = data
    else:
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(csv_text))
    if df.empty:
        return pd.DataFrame()

    date_col = "observation_date"
    value_col = series_id
    if date_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col])
    df["year"] = df[date_col].dt.year.astype(int)
    df = df[df["year"].between(start, end)]
    annual = df.groupby("year", as_index=False)[[value_col]].mean()
    return annual.rename(columns={value_col: "value"})[["year", "value"]]


LONG_COLUMNS = ["country_iso3", "indicator_code", "year", "value", "source", "flag"]


def _series_slice(panel: pd.DataFrame, code: str) -> pd.DataFrame:
    """Country/year/value rows for one indicator code, sorted for derivation."""
    subset = panel[panel["indicator_code"].eq(code)][["country_iso3", "year", "value"]].copy()
    return subset.sort_values(["country_iso3", "year"])


def _derive_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    """Append derived indicators that reference other panel series.

    OUTPUT_GAP_PROXY_PCT       : real GDP growth minus trailing 3-year own-growth
                                 mean (a documented proxy for cyclical slack).
    PUBLIC_DEBT_TRAJECTORY_PCT : 3-year change in the public-debt-to-GDP ratio.
    """
    frames = []

    if "NY.GDP.MKTP.KD.ZG" in set(panel["indicator_code"]):
        growth = _series_slice(panel, "NY.GDP.MKTP.KD.ZG")
        growth["_mav3"] = growth.groupby("country_iso3")["value"].transform(
            lambda s: s.rolling(3, min_periods=2).mean()
        )
        gap = growth.assign(value=(growth["value"] - growth["_mav3"]).round(3)).drop(columns=["_mav3"])
        gap["indicator_code"] = "OUTPUT_GAP_PROXY_PCT"
        gap["source"] = "Derived from World Bank NY.GDP.MKTP.KD.ZG"
        gap["flag"] = "ok"
        frames.append(gap[LONG_COLUMNS])

    if "GC.DOD.TOTL.GD.ZS" in set(panel["indicator_code"]):
        debt = _series_slice(panel, "GC.DOD.TOTL.GD.ZS")
        debt["_traj"] = debt.groupby("country_iso3")["value"].diff(3)
        traj = debt.assign(value=debt["_traj"].round(3)).drop(columns=["_traj"])
        traj["indicator_code"] = "PUBLIC_DEBT_TRAJECTORY_PCT"
        traj["source"] = "Derived from World Bank GC.DOD.TOTL.GD.ZS"
        traj["flag"] = "ok"
        frames.append(traj[LONG_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=LONG_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _finalize_long_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, append derived indicators, then deduplicate and sort."""
    if panel is None or panel.empty or "year" not in panel.columns:
        return pd.DataFrame(columns=LONG_COLUMNS)

    merged = pd.concat([panel, _derive_indicators(panel)], ignore_index=True)
    merged["value"] = pd.to_numeric(merged["value"], errors="coerce")
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce")
    merged = merged.dropna(subset=["year", "value"])
    merged["year"] = merged["year"].astype(int)
    if "flag" in merged.columns:
        merged["flag"] = merged["flag"].fillna("ok")
    else:
        merged["flag"] = "ok"

    return (
        merged[LONG_COLUMNS]
        .drop_duplicates(subset=["country_iso3", "indicator_code", "year"], keep="last")
        .sort_values(["country_iso3", "indicator_code", "year"])
        .reset_index(drop=True)
    )


def build_long_panel_batched(
    iso3_codes: Iterable[str],
    start: int,
    end: int,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> tuple[pd.DataFrame, FetchMetadata]:
    """
    Batched counterpart to build_long_panel(): one HTTP request per
    World-Bank-sourced indicator (covering every configured country at
    once) instead of one request per country per indicator.

    Now with persistent caching: API responses are stored in SQLite and
    reused on subsequent runs. If a live fetch fails, the last-known-good
    cached response is returned with a staleness flag.

    Returns:
        (panel, fetch_metadata) where fetch_metadata tracks which sources
        are fresh vs cached so the dashboard can show provenance info.
    """
    session = _session()
    records = indicator_records()
    iso3_codes = [str(c).strip().upper() for c in iso3_codes if str(c).strip()]
    output = []
    fetch_meta = FetchMetadata()

    for indicator in records:
        code = str(indicator.get("code", "")).strip()
        wb_code = indicator.get("world_bank")
        source_type = indicator.get("source", "world_bank")
        if not code:
            continue

        try:
            if code == "FX_YOY_DEPRECIATION_PCT":
                for iso3 in iso3_codes:
                    frame = _build_fx_depreciation(session, iso3, start, end)
                    if not frame.empty:
                        output.append(frame)
            elif code == "POLICY_RATE_YOY_CHANGE_BPS":
                for iso3 in iso3_codes:
                    frame = _policy_rate_for_country(session, iso3, start, end)
                    if not frame.empty:
                        output.append(frame)
            elif source_type in ("imf_weo", "imf_ifs"):
                imf_flow = indicator.get("imf_flow", "WEO")
                imf_indicator = indicator.get("imf_indicator", "")
                if imf_indicator:
                    raw = fetch_imf_indicator(imf_flow, imf_indicator, iso3_codes, start, end)
                    if not raw.empty:
                        raw["indicator_code"] = code
                        output.append(raw[["country_iso3", "indicator_code", "year", "value", "source", "flag"]])
                        fetch_meta.fresh_sources.append(f"IMF {imf_flow}")
            elif wb_code and source_type == "world_bank":
                raw = _cached_world_bank_batch(session, iso3_codes, str(wb_code), start, end, fetch_meta, ttl_seconds)
                if not raw.empty:
                    raw["indicator_code"] = code
                    output.append(raw[["country_iso3", "indicator_code", "year", "value", "source", "flag"]])
        except requests.RequestException as exc:
            LOG.warning("Batched source request failed for indicator %s: %s", code, exc)
        except Exception as exc:  # noqa: BLE001 - one bad indicator must not sink the whole panel
            LOG.warning("Indicator %s failed in batched fetch: %s", code, exc)

    if not output:
        return pd.DataFrame(columns=LONG_COLUMNS), fetch_meta

    panel = pd.concat(output, ignore_index=True)
    return _finalize_long_panel(panel), fetch_meta


def build_long_panel(
    iso3_codes: Iterable[str],
    start: int,
    end: int,
) -> pd.DataFrame:
    """Collect configured indicators into the canonical long panel."""
    session = _session()
    records = indicator_records()

    output = []

    for iso3 in iso3_codes:
        iso3 = str(iso3).strip().upper()
        LOG.info("Collecting %s", iso3)

        for indicator in records:
            code = str(indicator.get("code", "")).strip()
            wb_code = indicator.get("world_bank")
            source_type = indicator.get("source", "world_bank")

            if not code:
                continue

            try:
                if code == "FX_YOY_DEPRECIATION_PCT":
                    frame = _build_fx_depreciation(session, iso3, start, end)
                elif code == "POLICY_RATE_YOY_CHANGE_BPS":
                    frame = _policy_rate_for_country(session, iso3, start, end)
                elif source_type in ("imf_weo", "imf_ifs"):
                    imf_flow = indicator.get("imf_flow", "WEO")
                    imf_indicator_code = indicator.get("imf_indicator", "")
                    if imf_indicator_code:
                        frame = fetch_imf_indicator(imf_flow, imf_indicator_code, [iso3], start, end)
                        if not frame.empty:
                            frame["indicator_code"] = code
                            frame = frame[["country_iso3", "indicator_code", "year", "value", "source", "flag"]]
                    else:
                        frame = pd.DataFrame()
                elif wb_code and source_type == "world_bank":
                    raw = _world_bank_indicator(
                        session,
                        iso3,
                        str(wb_code),
                        start,
                        end,
                    )
                    if raw.empty:
                        frame = raw
                    else:
                        raw["indicator_code"] = code
                        frame = raw[
                            [
                                "country_iso3",
                                "indicator_code",
                                "year",
                                "value",
                                "source",
                            ]
                        ]
                else:
                    # Optional external sources can be implemented here later.
                    frame = pd.DataFrame()

                if frame is not None and not frame.empty:
                    output.append(frame)

            except requests.RequestException as exc:
                LOG.warning(
                    "Source request failed for %s / %s: %s",
                    iso3,
                    code,
                    exc,
                )
            except Exception as exc:
                LOG.warning(
                    "Indicator %s failed for %s: %s",
                    code,
                    iso3,
                    exc,
                )

    if not output:
        return pd.DataFrame(columns=LONG_COLUMNS)

    panel = pd.concat(output, ignore_index=True)
    return _finalize_long_panel(panel)
