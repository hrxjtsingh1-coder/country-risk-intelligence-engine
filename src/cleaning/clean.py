"""
Cleaning and panel-shaping utilities for the Country Risk Intelligence Engine.

The cleaning layer is deliberately conservative:
- identifiers are normalized;
- numeric values are coerced safely;
- configured bounds are applied as flags rather than silently rewriting data;
- duplicate country/indicator/year observations keep the latest source row;
- long data are converted to a country-year wide panel for scoring/dashboard use.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "indicators.yaml"


def _indicator_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _indicator_records() -> list[dict]:
    cfg = _indicator_config()
    records = cfg.get("indicators", []) if isinstance(cfg, dict) else []
    return [r for r in records if isinstance(r, dict)]


def _bounds() -> dict[str, tuple[float | None, float | None]]:
    result = {}
    for record in _indicator_records():
        code = record.get("code")
        if not code:
            continue
        low = record.get("min")
        high = record.get("max")
        result[str(code)] = (
            float(low) if low is not None else None,
            float(high) if high is not None else None,
        )
    return result


def clean_long_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a long country/indicator/year panel."""
    if df is None or df.empty:
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

    out = df.copy()

    required = ["country_iso3", "indicator_code", "year", "value"]
    for column in required:
        if column not in out.columns:
            out[column] = pd.NA

    out["country_iso3"] = out["country_iso3"].astype(str).str.upper().str.strip()
    out["indicator_code"] = out["indicator_code"].astype(str).str.strip()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    if "source" not in out.columns:
        out["source"] = "unknown"
    if "flag" not in out.columns:
        out["flag"] = "ok"

    bounds = _bounds()

    flags = []
    for _, row in out.iterrows():
        code = row["indicator_code"]
        value = row["value"]

        if pd.isna(value):
            flags.append("missing")
            continue

        low, high = bounds.get(code, (None, None))

        if low is not None and value < low:
            flags.append("out_of_range")
        elif high is not None and value > high:
            flags.append("out_of_range")
        else:
            flags.append("ok")

    out["flag"] = flags

    out = out.dropna(subset=["country_iso3", "indicator_code", "year"])
    out["year"] = out["year"].astype(int)

    # Keep the last observation for duplicate keys. Source collectors append
    # their records in deterministic order.
    out = (
        out.sort_values(["country_iso3", "indicator_code", "year"])
        .drop_duplicates(
            subset=["country_iso3", "indicator_code", "year"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return out[
        ["country_iso3", "indicator_code", "year", "value", "source", "flag"]
    ]


def to_wide_panel(df_long: pd.DataFrame) -> pd.DataFrame:
    """Convert long observations into one row per country/year."""
    clean = clean_long_panel(df_long)

    if clean.empty:
        return pd.DataFrame(columns=["country_iso3", "year"])

    wide = (
        clean.pivot_table(
            index=["country_iso3", "year"],
            columns="indicator_code",
            values="value",
            aggfunc="last",
        )
        .reset_index()
    )

    wide.columns.name = None
    return wide.sort_values(["country_iso3", "year"]).reset_index(drop=True)


def coverage_report(
    df_long: pd.DataFrame,
    countries: list[str],
    indicators: list[str],
    years: list[int],
) -> pd.DataFrame:
    """Return simple country/indicator/year coverage diagnostics."""
    clean = clean_long_panel(df_long)

    expected = max(1, len(indicators) * len(years))

    rows = []
    for country in countries:
        subset = clean[clean["country_iso3"].eq(str(country).upper())]
        observed = int(
            subset[
                subset["indicator_code"].isin(indicators)
                & subset["year"].isin(years)
            ][["indicator_code", "year"]].drop_duplicates().shape[0]
        )

        rows.append(
            {
                "country_iso3": str(country).upper(),
                "expected_observations": expected,
                "observed_observations": observed,
                "coverage_pct": 100.0 * observed / expected,
            }
        )

    return pd.DataFrame(rows)
