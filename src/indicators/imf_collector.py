"""
IMF SDMX data collector.

Fetches data from the IMF's public SDMX-JSON REST API, primarily:
  - WEO (World Economic Outlook): GDP growth, inflation, debt, current account
  - IFS (International Financial Statistics): reserves, exchange rates

API docs: https://datahelp.imf.org/knowledgebase/articles/667681-using-json-restful-web-service

The IMF uses ISO3 country codes in its API, same as our internal format.
The collector is fault-tolerant: if the IMF endpoint is unreachable, the
caller gets an empty DataFrame and falls back to other sources.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

LOG = logging.getLogger("country-risk.collectors.imf")

IMF_BASE = "https://dataservices.imf.org/REST/SDMX_JSON.svc/"

# IMF country codes — most are ISO3, but some differ
# https://datahelp.imf.org/knowledgebase/articles/667681-using-json-restful-web-service
IMF_COUNTRY_MAP: dict[str, str] = {
    "USA": "US",
    "CAN": "CA",
    "GBR": "GB",
    "DEU": "DE",
    "FRA": "FR",
    "ITA": "IT",
    "ESP": "ES",
    "JPN": "JP",
    "KOR": "KR",
    "AUS": "AU",
    "NLD": "NL",
    "SGP": "SG",
    "IND": "IN",
    "CHN": "CN",
    "BRA": "BR",
    "MEX": "MX",
    "ZAF": "ZA",
    "IDN": "ID",
    "TUR": "TR",
    "POL": "PL",
}


def _to_imf_code(iso3: str) -> str:
    """Convert ISO3 to IMF country code, defaulting to ISO3 if unknown."""
    return IMF_COUNTRY_MAP.get(iso3.upper(), iso3.upper())


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "country-risk-intelligence-engine/1.0",
            "Accept": "application/json",
        }
    )
    return session


def fetch_weo_indicator(
    session: requests.Session,
    indicator_code: str,
    iso3_codes: list[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Fetch a single indicator from the IMF WEO dataset.

    WEO flow code: WEO
    Indicator codes: NGDP_RPCH (GDP growth), PCPIPCH (inflation),
                     GGXWDG_NGDP (govt debt), BCA_NGDPD (current account),
                     LUR (unemployment), etc.

    API pattern:
        /SDMX_JSON.svc/data/WEO/{country_codes}/{indicator}?startPeriod=Y&endPeriod=Y
    """
    imf_codes = ";".join(sorted({_to_imf_code(c) for c in iso3_codes}))
    url = f"{IMF_BASE}data/WEO/{imf_codes}/{indicator_code}"
    params = {"startPeriod": str(start_year), "endPeriod": str(end_year)}

    try:
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        LOG.warning("IMF WEO fetch failed for %s: %s", indicator_code, exc)
        return pd.DataFrame()

    return _parse_sdmx_json(payload, iso3_codes, indicator_code, "IMF WEO")


def fetch_ifs_indicator(
    session: requests.Session,
    indicator_code: str,
    iso3_codes: list[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Fetch a single indicator from the IMF IFS dataset.

    IFS flow code: IFS
    Indicator codes: FASMB (total reserves minus gold, USD),
                     ENDA_XDC_USD_RATE (exchange rate, LCU per USD),
                     etc.
    """
    imf_codes = ";".join(sorted({_to_imf_code(c) for c in iso3_codes}))
    url = f"{IMF_BASE}data/IFS/{imf_codes}/{indicator_code}"
    params = {"startPeriod": str(start_year), "endPeriod": str(end_year)}

    try:
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        LOG.warning("IMF IFS fetch failed for %s: %s", indicator_code, exc)
        return pd.DataFrame()

    return _parse_sdmx_json(payload, iso3_codes, indicator_code, "IMF IFS")


def _parse_sdmx_json(
    payload: dict[str, Any],
    iso3_codes: list[str],
    indicator_code: str,
    source_label: str,
) -> pd.DataFrame:
    """
    Parse an IMF SDMX-JSON response into the canonical long-panel format.

    SDMX-JSON structure:
        payload["dataSets"][0]["series"][series_key]["observations"][period] = [value, ...]
        payload["structure"]["dimensions"]["series"] contains the dimension metadata
        payload["structure"]["dimensions"]["observation"] contains time periods

    The series key is a semicolon-separated index matching the dimensions order.
    """
    rows = []

    try:
        datasets = payload.get("dataSets", [])
        if not datasets:
            return pd.DataFrame()

        series_dim = payload.get("structure", {}).get("dimensions", {}).get("series", [])
        obs_dim = payload.get("structure", {}).get("dimensions", {}).get("observation", [])

        # Find the country dimension index (usually 0)
        country_dim_idx = None
        for i, dim in enumerate(series_dim):
            if dim.get("id") == "REF_AREA":
                country_dim_idx = i
                break
        if country_dim_idx is None and series_dim:
            country_dim_idx = 0

        # Build country code lookup from dimension values
        country_values = []
        if country_dim_idx is not None and country_dim_idx < len(series_dim):
            country_values = [v.get("id", "") for v in series_dim[country_dim_idx].get("values", [])]

        # Build year lookup from observation dimension
        year_values: list[int | None] = []
        for dim in obs_dim:
            for val in dim.get("values", []):
                # IMF periods can be "2023", "2023-Q1", "2023-Q2", etc.
                # For annual data, we take the year part
                period = val.get("id", "")
                try:
                    year = int(period[:4])
                    year_values.append(year)
                except (ValueError, IndexError):
                    year_values.append(None)

        # Extract observations
        series_data = datasets[0].get("series", {})
        for series_key, series_val in series_data.items():
            # Parse series key to get country index
            key_parts = series_key.split(":")
            try:
                country_idx = int(key_parts[country_dim_idx]) if country_dim_idx is not None else 0
            except (ValueError, IndexError):
                continue

            country_code = country_values[country_idx] if country_idx < len(country_values) else ""

            # Map IMF country code back to ISO3
            iso3 = _imf_to_iso3(country_code)
            if iso3 not in [c.upper() for c in iso3_codes]:
                continue

            observations = series_val.get("observations", {})
            for period_idx_str, obs_value in observations.items():
                try:
                    period_idx = int(period_idx_str)
                except ValueError:
                    continue

                if period_idx >= len(year_values):
                    continue
                year_val: int | None = year_values[period_idx]
                if year_val is None:
                    continue

                # obs_value is [value, status_code, ...]
                value = obs_value[0] if obs_value else None
                if value is None:
                    continue

                rows.append(
                    {
                        "country_iso3": iso3,
                        "year": year_val,
                        "value": float(value),
                        "source": source_label,
                        "flag": "ok",
                    }
                )

    except (KeyError, IndexError, TypeError) as exc:
        LOG.warning("Failed to parse IMF SDMX response for %s: %s", indicator_code, exc)
        return pd.DataFrame()

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _imf_to_iso3(imf_code: str) -> str:
    """Convert IMF country code back to ISO3."""
    imf_code = imf_code.upper()
    for iso3, imf in IMF_COUNTRY_MAP.items():
        if imf == imf_code:
            return iso3
    return imf_code  # Assume ISO3 if no mapping found


def fetch_imf_indicator(
    flow: str,
    indicator_code: str,
    iso3_codes: list[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Unified entry point: fetch any IMF indicator by flow and code.

    Args:
        flow: "WEO" or "IFS"
        indicator_code: IMF indicator code (e.g. "NGDP_RPCH", "FASMB")
        iso3_codes: Country codes to fetch
        start_year: First year of data
        end_year: Last year of data
    """
    session = _session()
    flow = flow.upper()

    if flow == "WEO":
        return fetch_weo_indicator(session, indicator_code, iso3_codes, start_year, end_year)
    elif flow == "IFS":
        return fetch_ifs_indicator(session, indicator_code, iso3_codes, start_year, end_year)
    else:
        LOG.warning("Unknown IMF flow: %s", flow)
        return pd.DataFrame()
