"""
Runtime live-data provider for the deployed dashboard.

The public app fetches official data directly into memory. Production data is
restricted to the current calendar year; older observations are not exposed
through the runtime panel, even when an upstream response or cache contains
historical records.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

from src.cleaning.clean import clean_long_panel, to_wide_panel
from src.indicators.build_panel import FetchMetadata, build_long_panel_batched
from src.runtime.provenance import Provenance, make_provenance
from src.runtime.year_policy import CURRENT_YEAR, validate_current_year_range

WORLD_BANK_SOURCE = {"name": "World Bank Indicators API", "url": "https://api.worldbank.org/v2"}
FRED_SOURCE = {"name": "FRED (Federal Reserve Economic Data)", "url": "https://fred.stlouisfed.org"}
BIS_SOURCE = {"name": "BIS Statistics (credit-to-GDP gaps)", "url": "https://stats.bis.org/"}


class LiveDataUnavailable(Exception):
    """Raised when a live fetch could not produce a usable current-year panel."""

    def __init__(self, message: str, technical_detail: str = ""):
        super().__init__(message)
        self.technical_detail = technical_detail


@dataclass
class LiveResult:
    wide_panel: pd.DataFrame
    long_panel: pd.DataFrame
    provenance: Provenance
    latest_common_year: int
    world_bank_ok: bool
    fred_ok: bool
    bis_ok: bool = False
    fetch_metadata: FetchMetadata = field(default_factory=FetchMetadata)


def select_latest_common_year(
    long_panel: pd.DataFrame,
    weighted_indicators: list[tuple[str, float]],
    min_coverage: float = 0.8,
    min_countries: int = 5,
) -> int | None:
    """Return the current year when a current-year panel is available."""
    del weighted_indicators, min_coverage, min_countries
    if long_panel is None or long_panel.empty or "year" not in long_panel.columns:
        return None
    years = pd.to_numeric(long_panel["year"], errors="coerce").dropna().astype(int).unique()
    if CURRENT_YEAR not in years:
        return None
    return CURRENT_YEAR


def fetch_live_panel(
    iso3_codes: list[str],
    indicators_cfg: dict,
    start_year: int,
    end_year: int,
    config_version: str = "unversioned",
) -> LiveResult:
    """Fetch and validate a canonical current-year panel from live sources."""
    if os.environ.get("COUNTRY_RISK_OFFLINE", "").strip().lower() in {"1", "true", "yes"}:
        raise LiveDataUnavailable(
            "Live data is disabled in this environment (COUNTRY_RISK_OFFLINE).",
            technical_detail="COUNTRY_RISK_OFFLINE is set; no network request was attempted.",
        )

    try:
        current_start, current_end = validate_current_year_range(start_year, end_year)
    except ValueError as exc:
        raise LiveDataUnavailable(
            f"Only current-year live data ({CURRENT_YEAR}) is supported.",
            technical_detail=str(exc),
        ) from exc

    records = indicators_cfg.get("indicators", []) if isinstance(indicators_cfg, dict) else []
    if not records:
        raise LiveDataUnavailable("No indicators are configured (config/indicators.yaml is empty).")

    iso3_codes = [c for c in iso3_codes if c]
    if not iso3_codes:
        raise LiveDataUnavailable("No countries are configured (config/countries.yaml is empty).")

    try:
        long_panel, fetch_meta = build_long_panel_batched(
            iso3_codes,
            current_start,
            current_end,
        )
    except Exception as exc:  # noqa: BLE001 - this boundary must never raise past it
        raise LiveDataUnavailable(
            "Official public data could not be retrieved right now.",
            technical_detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    if long_panel.empty:
        raise LiveDataUnavailable(
            f"Official sources returned no usable observations for {CURRENT_YEAR}. "
            "This can happen when an annual indicator has not yet been published for the current year."
        )

    long_panel = clean_long_panel(long_panel)
    if long_panel.empty:
        raise LiveDataUnavailable(
            f"Official sources returned no usable observations for current year {CURRENT_YEAR}."
        )

    wide_panel = to_wide_panel(long_panel)
    weighted_indicators = [
        (r["code"], float(r.get("weight", 0) or 0))
        for r in records
        if r.get("code")
    ]
    latest_year = select_latest_common_year(long_panel, weighted_indicators)
    if latest_year is None:
        raise LiveDataUnavailable(f"No current-year ({CURRENT_YEAR}) observations are available.")

    world_bank_ok = bool(long_panel["source"].astype(str).str.contains("World Bank", case=False).any())
    fred_ok = bool(long_panel["source"].astype(str).str.contains("FRED", case=False).any())
    bis_ok = bool(long_panel["source"].astype(str).str.contains("BIS", case=False).any())

    expected = len(iso3_codes) * len(weighted_indicators)
    received = (
        long_panel.drop_duplicates(subset=["country_iso3", "indicator_code", "year"])
        .pipe(lambda df: df[df["year"] == CURRENT_YEAR])
        .shape[0]
    )

    validation_failures = []
    if "flag" in long_panel.columns and long_panel["flag"].eq("out_of_range").any():
        n_bad = int(long_panel["flag"].eq("out_of_range").sum())
        validation_failures.append(
            f"{n_bad} observation(s) fell outside configured plausibility bounds "
            "(kept and flagged, not discarded — see Data Quality)."
        )
    if not fred_ok:
        validation_failures.append(
            "FRED enrichment is unavailable this run; affected indicators remain missing "
            "rather than being replaced with stale historical values."
        )
    if not bis_ok:
        validation_failures.append(
            "BIS credit-to-GDP gap is unavailable this run; the current-year panel remains "
            "explicitly missing for that driver rather than using older data."
        )

    provenance = make_provenance(
        sources=[WORLD_BANK_SOURCE]
        + ([FRED_SOURCE] if fred_ok else [])
        + ([BIS_SOURCE] if bis_ok else []),
        requested_period=f"{current_start}-{current_end}",
        latest_observation_year=CURRENT_YEAR,
        country_count=len(iso3_codes),
        indicator_count=len(weighted_indicators),
        expected_observations=expected,
        received_observations=received,
        validation_failures=validation_failures,
        config_version=config_version,
    )

    return LiveResult(
        wide_panel=wide_panel,
        long_panel=long_panel,
        provenance=provenance,
        latest_common_year=CURRENT_YEAR,
        world_bank_ok=world_bank_ok,
        fred_ok=fred_ok,
        bis_ok=bis_ok,
        fetch_metadata=fetch_meta,
    )
