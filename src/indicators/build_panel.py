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
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "indicators.yaml"

LOG = logging.getLogger("country-risk.collectors")

WB_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


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
    params = {
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

    annual = data.groupby("year", as_index=False)[value_col].mean()

    return annual.rename(columns={value_col: "value"})[
        ["year", "value"]
    ]


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

    return out[
        ["country_iso3", "indicator_code", "year", "value", "source"]
    ]


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

    return annual[
        ["country_iso3", "indicator_code", "year", "change_bps", "source"]
    ].rename(columns={"change_bps": "value"})


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
                    frame = _build_fx_depreciation(
                        session, iso3, start, end
                    )
                elif code == "POLICY_RATE_YOY_CHANGE_BPS":
                    frame = _policy_rate_for_country(
                        session, iso3, start, end
                    )
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
        return pd.DataFrame(
            columns=[
                "country_iso3",
                "indicator_code",
                "year",
                "value",
                "source",
                "flag",
            ]
        )

    panel = pd.concat(output, ignore_index=True)

    panel["value"] = pd.to_numeric(panel["value"], errors="coerce")
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")

    panel = panel.dropna(subset=["year"])
    panel["year"] = panel["year"].astype(int)
    panel["flag"] = "ok"

    return panel[
        [
            "country_iso3",
            "indicator_code",
            "year",
            "value",
            "source",
            "flag",
        ]
    ].drop_duplicates(
        subset=["country_iso3", "indicator_code", "year"],
        keep="last",
    )
