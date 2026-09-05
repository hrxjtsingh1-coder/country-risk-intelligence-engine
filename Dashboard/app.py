"""
Streamlit dashboard for the Global Country Risk Intelligence Engine.

Run with:
    streamlit run dashboard/app.py

Reads data/processed/panel_wide.csv, which src/pipeline/run_all.py produces
from the live World Bank / FRED / ECB / BIS collectors. Scores and drivers
are recomputed on the fly from that panel (cheap, and guarantees the
dashboard always matches the current config/indicators.yaml weights even if
you tweak them without re-running the full pipeline).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.commentary.generate_commentary import generate_report
from src.scenario.scenario_engine import run_shock_scenario
from src.scoring.risk_score import score_panel, top_drivers

st.set_page_config(page_title="Country Risk Intelligence Engine", layout="wide")

PANEL_PATH = ROOT / "data" / "processed" / "panel_wide.csv"

BAND_COLORS = {
    "Low": "#2e7d32",
    "Moderate": "#9e9d24",
    "Elevated": "#f9a825",
    "High": "#ef6c00",
    "Severe": "#c62828",
}


@st.cache_data
def load_config():
    with open(ROOT / "config" / "countries.yaml") as f:
        countries_cfg = yaml.safe_load(f)
    with open(ROOT / "config" / "indicators.yaml") as f:
        indicators_cfg = yaml.safe_load(f)
    return countries_cfg, indicators_cfg


@st.cache_data
def load_panel():
    if not PANEL_PATH.exists():
        return None
    return pd.read_csv(PANEL_PATH)


def main():
    countries_cfg, indicators_cfg = load_config()
    name_lookup = {c["iso3"]: c["name"] for c in countries_cfg["countries"]}
    peer_groups = countries_cfg.get("peer_groups", {})

    st.title("🌍 Global Country Risk Intelligence Engine")
    st.caption(
        "Country → Data collection → Cleaning → Indicators → Risk scoring → Scenario analysis → Dashboard → Analyst commentary"
    )

    panel = load_panel()
    if panel is None:
        st.warning(
            "No panel found at `data/processed/panel_wide.csv`. Run the pipeline first:\n\n"
            "`python -m src.pipeline.run_all`\n\n"
            "(This needs normal internet access to reach the World Bank / FRED / ECB / BIS APIs.)"
        )
        st.stop()

    scores, drivers = score_panel(panel)

    with st.sidebar:
        st.header("Controls")
        available = sorted(panel["country_iso3"].unique())
        country = st.selectbox("Country", available, format_func=lambda c: name_lookup.get(c, c))
        years_for_country = sorted(panel[panel["country_iso3"] == country]["year"].unique())
        year = st.select_slider("Year", years_for_country, value=years_for_country[-1])

        st.divider()
        st.subheader("Scenario shock")
        driver_code = st.selectbox("Shock variable", ["POLICY_RATE_YOY_CHANGE_BPS"])
        shock_amount = st.slider("Change (bps)", -300, 300, 100, step=25)
        run_scenario_btn = st.button("Run scenario", type="primary")

    row = scores[(scores["country_iso3"] == country) & (scores["year"] == year)]
    if row.empty or pd.isna(row.iloc[0]["risk_score"]):
        st.error(f"No sufficient indicator data to score {name_lookup.get(country, country)} in {year}.")
        st.stop()
    score_val = row.iloc[0]["risk_score"]
    band = row.iloc[0]["risk_band"]
    completeness = row.iloc[0]["data_completeness"]

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric(f"{name_lookup.get(country, country)} — Risk Score", f"{score_val:.0f} / 100", band)
        st.markdown(
            f'<div style="background:{BAND_COLORS.get(band, "#999")};color:white;padding:6px 12px;'
            f'border-radius:6px;display:inline-block;font-weight:600;">{band} risk</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.metric("Data completeness", f"{completeness * 100:.0f}%", help="Share of indicator weight populated for this country-year.")
    with col3:
        trend = scores[scores["country_iso3"] == country].dropna(subset=["risk_score"]).sort_values("year")
        fig = px.line(trend, x="year", y="risk_score", markers=True, title="Risk score over time")
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Main drivers")
    country_drivers = top_drivers(drivers, country, year, n=6)
    if not country_drivers.empty:
        bar = go.Figure(
            go.Bar(
                x=country_drivers["weighted_contribution"],
                y=country_drivers["label"],
                orientation="h",
                marker_color=["#c62828" if v > 0 else "#2e7d32" for v in country_drivers["weighted_contribution"]],
            )
        )
        bar.update_layout(
            title="Contribution to composite risk score (red = adds risk, green = reduces it)",
            xaxis_title="Weighted z-contribution",
            height=350,
        )
        st.plotly_chart(bar, use_container_width=True)

    st.subheader("Peer comparison")
    peer_group = next((members for members in peer_groups.values() if country in members), None)
    year_scores = scores[(scores["year"] == year)].dropna(subset=["risk_score"]).sort_values("risk_score", ascending=False)
    year_scores = year_scores.copy()
    year_scores["name"] = year_scores["country_iso3"].map(name_lookup)
    fig2 = px.bar(year_scores, x="name", y="risk_score", color="risk_band", color_discrete_map=BAND_COLORS, title=f"All tracked countries, {year}")
    fig2.update_yaxes(range=[0, 100])
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Analyst commentary")
    scenario_result = None
    if run_scenario_btn:
        try:
            scenario_result = run_shock_scenario(
                panel, country, year, driver_code, shock_amount,
                ["FX_YOY_DEPRECIATION_PCT", "NY.GDP.MKTP.KD.ZG", "GC.DOD.TOTL.GD.ZS"],
            )
        except Exception as exc:
            st.warning(f"Couldn't run that scenario: {exc}")

    report = generate_report(
        country_name=name_lookup.get(country, country),
        country_iso3=country,
        year=year,
        scores=scores,
        drivers=drivers,
        scenario_result=scenario_result,
        peer_group=[c for c in (peer_group or []) if c != country],
    )
    st.markdown(report)


if __name__ == "__main__":
    main()
