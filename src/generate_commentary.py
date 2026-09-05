"""
Turns numbers into the analyst-style write-up: Risk Score / Main drivers /
Scenario / Impact / Analyst view / Limitations.

This is rule-based text generation, not an LLM call — deterministic, free to
run, and every sentence traces back to a specific number computed upstream.
(An optional LLM-polish pass exists in commentary/llm_enhance.py for anyone
who wants smoother prose; it's opt-in and never the only way to get output.)

The goal isn't eloquence, it's traceability: a reader should be able to look
at any sentence here and find the exact number it came from in scores/drivers/
scenario_result.
"""
from __future__ import annotations

import pandas as pd

DIRECTION_PHRASE = {
    "FP.CPI.TOTL.ZG": ("Elevated inflation", "Contained inflation"),
    "NY.GDP.MKTP.KD.ZG": ("Weakening growth", "Resilient growth"),
    "SL.UEM.TOTL.ZS": ("High unemployment", "Tight labor market"),
    "POLICY_RATE_YOY_CHANGE_BPS": ("Rapid policy tightening", "Easing monetary stance"),
    "GC.DOD.TOTL.GD.ZS": ("High public debt", "Contained public debt"),
    "BN.CAB.XOKA.GD.ZS": ("Current account deficit pressure", "Current account surplus cushion"),
    "FX_YOY_DEPRECIATION_PCT": ("Currency depreciation", "Currency strength"),
    "FB.AST.NPER.ZS": ("Rising bad loans in the banking sector", "Healthy loan books"),
    "BIS_CREDIT_GAP": ("Credit-boom warning signal", "Subdued credit growth"),
    "FI.RES.TOTL.MO": ("Thin FX-reserve buffer", "Ample FX reserves"),
    "DT.DOD.DECT.GN.ZS": ("High external debt burden", "Low external debt burden"),
}


def _driver_phrase(row: pd.Series) -> str:
    worse_phrase, better_phrase = DIRECTION_PHRASE.get(row["indicator_code"], (row["label"], row["label"]))
    return worse_phrase if row["weighted_contribution"] > 0 else better_phrase


def _trend_phrase(scores: pd.DataFrame, country_iso3: str, year: int) -> str | None:
    """Compare this year's score to the prior year's, if available, for an 'improving/worsening' line."""
    hist = scores[(scores["country_iso3"] == country_iso3) & (scores["year"].isin([year, year - 1]))]
    hist = hist.dropna(subset=["risk_score"]).sort_values("year")
    if len(hist) < 2:
        return None
    prev_score = hist.iloc[0]["risk_score"]
    curr_score = hist.iloc[1]["risk_score"]
    delta = curr_score - prev_score
    if abs(delta) < 1.5:
        return f"broadly unchanged from {prev_score:.0f} a year earlier"
    direction = "up" if delta > 0 else "down"
    return f"{direction} from {prev_score:.0f} a year earlier ({delta:+.1f} pts)"


def _peer_phrase(scores: pd.DataFrame, country_iso3: str, year: int, peer_group: list[str] | None) -> str | None:
    if not peer_group:
        return None
    peers = scores[(scores["year"] == year) & (scores["country_iso3"].isin(peer_group))]
    peers = peers.dropna(subset=["risk_score"])
    if peers.empty:
        return None
    median = peers["risk_score"].median()
    country_score = scores[(scores["country_iso3"] == country_iso3) & (scores["year"] == year)]["risk_score"]
    if country_score.empty or pd.isna(country_score.iloc[0]):
        return None
    diff = country_score.iloc[0] - median
    if abs(diff) < 2:
        return "roughly in line with its peer group"
    return f"{'above' if diff > 0 else 'below'} its peer-group median ({median:.0f})"


def generate_report(
    country_name: str,
    country_iso3: str,
    year: int,
    scores: pd.DataFrame,
    drivers: pd.DataFrame,
    scenario_result: dict | None = None,
    peer_group: list[str] | None = None,
    n_drivers: int = 3,
) -> str:
    """
    Build the full analyst-style text block for one country-year.
    Returns a plain-text string; the Streamlit dashboard renders it as
    markdown (the section headers below are already markdown-ready).
    """
    row = scores[(scores["country_iso3"] == country_iso3) & (scores["year"] == year)]
    if row.empty or pd.isna(row.iloc[0]["risk_score"]):
        return f"No sufficient data to score {country_name} for {year}."
    score = row.iloc[0]["risk_score"]
    band = row.iloc[0]["risk_band"]
    completeness = row.iloc[0]["data_completeness"]

    top = drivers[(drivers["country_iso3"] == country_iso3) & (drivers["year"] == year)]
    top = top.reindex(top["weighted_contribution"].abs().sort_values(ascending=False).index).head(n_drivers)
    driver_bullets = "\n".join(f"- {_driver_phrase(r)}" for _, r in top.iterrows())

    trend = _trend_phrase(scores, country_iso3, year)
    peer = _peer_phrase(scores, country_iso3, year, peer_group)

    lines = [
        f"**{country_name} — Risk Score: {score:.0f}/100 ({band})**",
        "",
        "**Main drivers:**",
        driver_bullets if driver_bullets else "- Not enough indicator coverage to identify drivers.",
        "",
    ]

    if scenario_result:
        shock_desc = _format_shock(scenario_result)
        lines += [
            "**Scenario:**",
            f'"{shock_desc}"',
            "",
            "**Impact:**",
            _format_impact(scenario_result),
            "",
        ]

    lines += [
        "**Analyst view:**",
        _analyst_view(country_name, score, band, top, trend, peer, scenario_result),
        "",
        "**Limitations:**",
        _limitations(completeness),
    ]
    return "\n".join(lines)


def _format_shock(sc: dict) -> str:
    driver_label = {"POLICY_RATE_YOY_CHANGE_BPS": "policy rates increase by"}.get(sc["driver_code"], f"{sc['driver_code']} moves by")
    unit = "bps" if sc["driver_code"] == "POLICY_RATE_YOY_CHANGE_BPS" else ""
    return f"If {driver_label} {sc['shock_amount']:+.0f}{unit} from current levels..."


def _format_impact(sc: dict) -> str:
    if pd.isna(sc["delta"]):
        return "Insufficient historical data in this panel to estimate a reliable effect — see Limitations."
    direction = "deterioration" if sc["delta"] > 0 else "improvement"
    parts = [
        f"Estimated {direction} in the composite risk score of {abs(sc['delta']):.1f} points "
        f"({sc['baseline_score']:.0f} -> {sc['scenario_score']:.0f}), "
        f"{'moving the country into the ' + sc['scenario_band'] + ' band' if sc['scenario_band'] != sc['baseline_band'] else 'staying within the ' + sc['scenario_band'] + ' band'}."
    ]
    for d in sc["indicator_deltas"]:
        confidence = "low-confidence" if d["r_squared"] < 0.1 else "moderate-confidence"
        parts.append(
            f"Channel: {d['indicator_code']} estimated to move by {d['estimated_delta']:+.2f} "
            f"({confidence}, R²={d['r_squared']:.2f}, n={d['n_obs']})."
        )
    return " ".join(parts)


def _analyst_view(
    country_name: str,
    score: float,
    band: str,
    top: pd.DataFrame,
    trend: str | None,
    peer: str | None,
    scenario_result: dict | None,
) -> str:
    sentences = [f"{country_name}'s composite score of {score:.0f} places it in the {band.lower()} risk band."]
    if trend:
        sentences.append(f"That score is {trend}.")
    if peer:
        sentences.append(f"It sits {peer}.")
    if not top.empty:
        lead_driver = _driver_phrase(top.iloc[0]).lower()
        sentences.append(f"The single largest swing factor this period is {lead_driver}.")
    if scenario_result:
        driver_name = scenario_result["driver_code"].replace("_", " ").lower()
        sentences.append(f"The Scenario section above stress-tests sensitivity to {driver_name} specifically — it is one lever among several, not necessarily the top driver above.")
    sentences.append(
        "This score is a relative-positioning signal within the panel, not a standalone forecast or a credit rating — "
        "treat it as a prioritization tool for where to look closer, not a substitute for that closer look."
    )
    return " ".join(sentences)


def _limitations(completeness: float) -> str:
    base = [
        "Cross-sectional z-scores compare a country to the OTHER countries in this panel in the SAME year, "
        "not to a fixed global benchmark or to its own history beyond the single year-over-year comparison shown above.",
        "Scenario elasticities are pooled-panel OLS correlations, not a causal or structural model — "
        "they ignore lags, expectations effects, and country-specific transmission channels.",
        "Several indicators (e.g. external debt, credit-to-GDP gap) are only reported by their source for a subset "
        "of countries; a missing indicator is excluded and remaining weights renormalized, which can understate risk "
        "for a country with a real but unmeasured vulnerability in that category.",
    ]
    if completeness < 0.7:
        base.insert(
            0,
            f"Only {completeness * 100:.0f}% of the indicator weight had data for this country-year — "
            "treat this score with extra caution relative to a fully-populated one.",
        )
    return "\n".join(f"- {b}" for b in base)


if __name__ == "__main__":
    print("See tests/test_commentary.py for a runnable example against synthetic data.")
