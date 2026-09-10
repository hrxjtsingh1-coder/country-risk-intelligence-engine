"""
Runtime live-data provider for the deployed dashboard.

This is what the PUBLIC app calls automatically — no one should ever need
to run `python -m src.pipeline.run_all` to see live data (PHASE 3).
src/pipeline/run_all.py remains a separate, optional local/reproducible
path that writes data/processed/panel_wide.csv; this module never reads or
writes that file (PHASE 7) — it builds the panel in memory for the current
session and hands it back to the caller.

Contract: either return a populated, validated LiveResult, or raise
LiveDataUnavailable with a message written for an end user, not a
developer. This module never imports Streamlit — dashboard/app.py decides
what to render; this module only decides what's true about the data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

from src.cleaning.clean import clean_long_panel, to_wide_panel
from src.indicators.build_panel import FetchMetadata, build_long_panel_batched
from src.runtime.provenance import Provenance, make_provenance

WORLD_BANK_SOURCE = {"name": "World Bank Indicators API", "url": "https://api.worldbank.org/v2"}
FRED_SOURCE = {"name": "FRED (Federal Reserve Economic Data)", "url": "https://fred.stlouisfed.org"}


class LiveDataUnavailable(Exception):
    """
    Raised when a live fetch could not produce a usable panel.
    Message text is shown directly to end users (PHASE 9) — keep it plain,
    not a stack trace. Technical detail, if any, goes in .technical_detail.
    """

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
    fetch_metadata: FetchMetadata = field(default_factory=FetchMetadata)


def select_latest_common_year(
    long_panel: pd.DataFrame,
    weighted_indicators: list[tuple[str, float]],
    min_coverage: float = 0.8,
    min_countries: int = 5,
) -> int | None:
    """
    Walk years newest-to-oldest; return the first one where enough of the
    SCORE-RELEVANT indicator weight is populated across enough countries to
    compute a meaningful cross-section (PHASE 11). Deliberately not "the
    current calendar year" (annual macro data lags by 1-2 years almost
    everywhere) and not "the newest year with ANY data" (one stray
    observation for one country would otherwise win).

    Falls back to the single newest year with any data at all if nothing
    clears the bar — callers should treat that case as low-confidence and
    say so (see dashboard's coverage warning).
    """
    if long_panel.empty:
        return None
    total_weight = sum(w for _, w in weighted_indicators) or 1.0

    for year in sorted(long_panel["year"].unique(), reverse=True):
        year_slice = long_panel[long_panel["year"] == year]
        countries_present = year_slice["country_iso3"].nunique()
        if countries_present < min_countries:
            continue
        covered_weight = 0.0
        for code, weight in weighted_indicators:
            n_with_indicator = year_slice.loc[year_slice["indicator_code"] == code, "country_iso3"].nunique()
            covered_weight += weight * (n_with_indicator / max(countries_present, 1))
        if covered_weight / total_weight >= min_coverage:
            return int(year)

    return int(long_panel["year"].max())


def fetch_live_panel(
    iso3_codes: list[str],
    indicators_cfg: dict,
    start_year: int,
    end_year: int,
    config_version: str = "unversioned",
) -> LiveResult:
    """
    Fetch, validate, and build the canonical wide panel from live public
    data. A PARTIAL result (some indicators or countries missing) is still
    a successful return — "usable" means "enough to compute at least one
    country's score", not "complete"; coverage gaps show up in provenance,
    not as a hard failure. Only genuinely empty/unusable output raises.
    """
    if os.environ.get("COUNTRY_RISK_OFFLINE", "").strip().lower() in {"1", "true", "yes"}:
        raise LiveDataUnavailable(
            "Live data is disabled in this environment (COUNTRY_RISK_OFFLINE).",
            technical_detail="COUNTRY_RISK_OFFLINE is set; no network request was attempted.",
        )

    records = indicators_cfg.get("indicators", []) if isinstance(indicators_cfg, dict) else []
    if not records:
        raise LiveDataUnavailable("No indicators are configured (config/indicators.yaml is empty).")

    iso3_codes = [c for c in iso3_codes if c]
    if not iso3_codes:
        raise LiveDataUnavailable("No countries are configured (config/countries.yaml is empty).")

    try:
        long_panel, fetch_meta = build_long_panel_batched(iso3_codes, start_year, end_year)
    except Exception as exc:  # noqa: BLE001 - this boundary must never raise past it
        raise LiveDataUnavailable(
            "Official public data could not be retrieved right now.",
            technical_detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    if long_panel.empty:
        raise LiveDataUnavailable(
            "The World Bank Indicators API returned no usable observations for the "
            "configured countries and period. This is usually transient."
        )

    long_panel = clean_long_panel(long_panel)
    if long_panel.empty:
        raise LiveDataUnavailable("Every observation retrieved failed validation (out-of-range or malformed).")
    wide_panel = to_wide_panel(long_panel)

    weighted_indicators = [(r["code"], float(r.get("weight", 0) or 0)) for r in records if r.get("code")]
    latest_year = select_latest_common_year(long_panel, weighted_indicators)
    if latest_year is None:
        raise LiveDataUnavailable("Could not determine a usable analysis year from live data.")

    world_bank_ok = bool(long_panel["source"].astype(str).str.contains("World Bank", case=False).any())
    fred_ok = bool(long_panel["source"].astype(str).str.contains("FRED", case=False).any())

    expected = len(iso3_codes) * len(weighted_indicators)
    received = (
        long_panel.drop_duplicates(subset=["country_iso3", "indicator_code", "year"])
        .pipe(lambda df: df[df["year"] == latest_year])
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
            "FRED enrichment (US policy-rate YoY change) unavailable this run — "
            "World Bank data is unaffected; this indicator carries zero weight "
            "in the composite score regardless (see config/indicators.yaml)."
        )

    provenance = make_provenance(
        sources=[WORLD_BANK_SOURCE] + ([FRED_SOURCE] if fred_ok else []),
        requested_period=f"{start_year}-{end_year}",
        latest_observation_year=latest_year,
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
        latest_common_year=latest_year,
        world_bank_ok=world_bank_ok,
        fred_ok=fred_ok,
        fetch_metadata=fetch_meta,
    )
