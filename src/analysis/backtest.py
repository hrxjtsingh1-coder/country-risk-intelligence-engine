"""
Historical backtest: does the scoring engine actually flag known real
macro-stress episodes, using the country's OWN scored history — not just
"does the UI look right".

This is deliberately narrow and honest about what it proves. It does NOT
re-fetch data — it evaluates whatever the caller's `scores` DataFrame
already contains (the live-fetched or demo panel already in memory), so it
costs nothing extra in production and is transparent about running on
demo/synthetic data when that's what's loaded (see `data_is_synthetic` in
the result).

Episodes are chosen from countries already in config/countries.yaml and
years within the standard live-fetch window (2012+), so this runs against
real data automatically once the app is deployed with live World Bank
data — no separate step required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

EPISODES: list[dict[str, Any]] = [
    {
        "iso3": "TUR",
        "label": "Turkiye — 2018 currency crisis",
        "baseline_year": 2016,
        "event_year": 2018,
        "note": (
            "The lira lost roughly 30% of its value against the US dollar in "
            "2018, driven by a large current-account deficit, heavy external "
            "corporate debt, and rapid monetary tightening."
        ),
    },
    {
        "iso3": "BRA",
        "label": "Brazil — 2015-16 recession",
        "baseline_year": 2013,
        "event_year": 2016,
        "note": (
            "GDP contracted roughly 7% cumulatively over 2015-16, alongside "
            "double-digit inflation and a widening fiscal deficit during a "
            "deep political and fiscal crisis."
        ),
    },
    {
        "iso3": "ZAF",
        "label": "South Africa — fiscal deterioration, 2017-2020",
        "baseline_year": 2017,
        "event_year": 2020,
        "note": (
            "Persistent fiscal slippage and state-utility stress widened the "
            "budget deficit and public debt trajectory well before the "
            "COVID-19 shock added a sharp additional contraction in 2020."
        ),
    },
    {
        "iso3": "GBR",
        "label": "United Kingdom — 2022 gilt market shock",
        "baseline_year": 2019,
        "event_year": 2022,
        "note": (
            "September 2022's unfunded fiscal package triggered a sharp gilt "
            "sell-off and sterling depreciation, on top of the UK's post-2020 "
            "inflation and current-account pressures."
        ),
    },
]

PASS_THRESHOLD_POINTS = 3.0  # a rise smaller than this is noise, not a signal


@dataclass
class EpisodeResult:
    iso3: str
    label: str
    baseline_year: int
    event_year: int
    baseline_score: float | None
    event_score: float | None
    delta: float | None
    verdict: str  # "flagged" | "missed" | "inconclusive"
    note: str


def run_backtest(scores: pd.DataFrame, data_is_synthetic: bool) -> list[EpisodeResult]:
    """
    Evaluate every episode in EPISODES against the given scores DataFrame
    (country_iso3, year, risk_score). Returns one EpisodeResult per episode,
    in EPISODES order — callers should still check data_is_synthetic before
    treating "flagged"/"missed" as meaningful (a demo/synthetic panel can't
    actually validate anything; see the dashboard's rendering of this).
    """
    results = []
    if not isinstance(scores, pd.DataFrame) or scores.empty:
        return [
            EpisodeResult(
                e["iso3"],
                e["label"],
                e["baseline_year"],
                e["event_year"],
                None,
                None,
                None,
                "inconclusive",
                e["note"],
            )
            for e in EPISODES
        ]

    s = scores.copy()
    s["year"] = pd.to_numeric(s["year"], errors="coerce")
    s["risk_score"] = pd.to_numeric(s["risk_score"], errors="coerce")

    for ep in EPISODES:
        row_base = s[(s["country_iso3"] == ep["iso3"]) & (s["year"] == ep["baseline_year"])]
        row_event = s[(s["country_iso3"] == ep["iso3"]) & (s["year"] == ep["event_year"])]

        base_score = (
            float(row_base["risk_score"].iloc[0])
            if not row_base.empty and pd.notna(row_base["risk_score"].iloc[0])
            else None
        )
        event_score = (
            float(row_event["risk_score"].iloc[0])
            if not row_event.empty and pd.notna(row_event["risk_score"].iloc[0])
            else None
        )

        if base_score is None or event_score is None:
            verdict = "inconclusive"
            delta = None
        else:
            delta = round(event_score - base_score, 1)
            verdict = "flagged" if delta >= PASS_THRESHOLD_POINTS else "missed"

        results.append(
            EpisodeResult(
                ep["iso3"],
                ep["label"],
                ep["baseline_year"],
                ep["event_year"],
                base_score,
                event_score,
                delta,
                verdict,
                ep["note"],
            )
        )
    return results
