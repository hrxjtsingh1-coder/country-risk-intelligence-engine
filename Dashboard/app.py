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

st.markdown('\n<style>\n/* ==========================================================================\n   COUNTRY RISK INTELLIGENCE ENGINE — PREMIUM UI SYSTEM\n   Presentation-only layer. No analytical logic lives here.\n   ========================================================================== */\n\n@import url(\'https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap\');\n\n:root {\n  --cri-bg: #070a0f;\n  --cri-surface: #0d121a;\n  --cri-surface-2: #101720;\n  --cri-surface-3: #151d27;\n  --cri-text: #f5f7fb;\n  --cri-muted: #8b98a8;\n  --cri-dim: #5f6c7d;\n  --cri-line: rgba(255,255,255,.075);\n  --cri-cyan: #61dafb;\n  --cri-purple: #a78bfa;\n  --cri-green: #42d392;\n  --cri-red: #ff5470;\n  --cri-amber: #f5c451;\n}\n\n/* Base application */\nhtml, body, [class*="css"] {\n  font-family: "DM Sans", sans-serif;\n}\n\n.stApp {\n  background:\n    radial-gradient(circle at 5% -5%, rgba(97,218,251,.085), transparent 26%),\n    radial-gradient(circle at 100% 5%, rgba(167,139,250,.075), transparent 25%),\n    radial-gradient(circle at 55% 105%, rgba(66,211,146,.04), transparent 27%),\n    var(--cri-bg);\n  color: var(--cri-text);\n}\n\n.stApp::before {\n  content: "";\n  position: fixed;\n  inset: 0;\n  z-index: 0;\n  pointer-events: none;\n  opacity: .11;\n  background-image:\n    linear-gradient(rgba(255,255,255,.026) 1px, transparent 1px),\n    linear-gradient(90deg, rgba(255,255,255,.026) 1px, transparent 1px);\n  background-size: 42px 42px;\n  mask-image: linear-gradient(to bottom, black, transparent 86%);\n  animation: criGrid 16s linear infinite;\n}\n\n.stApp::after {\n  content: "";\n  position: fixed;\n  width: 520px;\n  height: 520px;\n  left: -280px;\n  top: 18%;\n  border-radius: 50%;\n  border: 1px solid rgba(97,218,251,.07);\n  box-shadow:\n    0 0 0 55px rgba(97,218,251,.018),\n    0 0 0 110px rgba(97,218,251,.012);\n  pointer-events: none;\n  animation: criOrb 12s ease-in-out infinite alternate;\n}\n\n[data-testid="stHeader"] {\n  background: rgba(7,10,15,.72);\n  backdrop-filter: blur(20px);\n}\n\n[data-testid="stToolbar"] {\n  opacity: .55;\n}\n\n.block-container {\n  max-width: 1500px;\n  padding-top: 2.1rem;\n  padding-bottom: 4rem;\n}\n\n[data-testid="stSidebar"] {\n  background:\n    linear-gradient(180deg, rgba(13,18,26,.98), rgba(7,10,15,.99));\n  border-right: 1px solid var(--cri-line);\n}\n\n[data-testid="stSidebar"] > div:first-child {\n  padding-top: 1.2rem;\n}\n\n/* Typography */\nh1, h2, h3, h4, h5 {\n  font-family: "Space Grotesk", sans-serif !important;\n  letter-spacing: -.035em;\n}\n\nh1 {\n  font-size: clamp(2rem, 4.5vw, 4.4rem) !important;\n  line-height: .98 !important;\n  background: linear-gradient(105deg, #ffffff 0%, #9edff1 40%, #a78bfa 78%, #ffffff 100%);\n  background-size: 180% auto;\n  -webkit-background-clip: text;\n  background-clip: text;\n  color: transparent !important;\n  animation: criShimmer 8s ease-in-out infinite;\n}\n\nh2 {\n  font-size: 1.55rem !important;\n}\n\nh3 {\n  font-size: 1.05rem !important;\n}\n\np, li, label {\n  color: #c4cdd8;\n}\n\nsmall {\n  color: var(--cri-muted);\n}\n\ncode, pre, [data-testid="stCode"] {\n  font-family: "DM Mono", monospace !important;\n}\n\n/* Main title becomes a product hero */\n[data-testid="stAppViewContainer"] h1 {\n  position: relative;\n  margin-bottom: .45rem;\n  animation:\n    criFadeUp .7s cubic-bezier(.2,.8,.2,1) both,\n    criShimmer 8s ease-in-out 1s infinite;\n}\n\n[data-testid="stAppViewContainer"] h1::before {\n  content: "GLOBAL MACRO · COUNTRY RISK INTELLIGENCE";\n  display: block;\n  margin-bottom: .8rem;\n  font-family: "DM Mono", monospace;\n  font-size: .67rem;\n  font-weight: 500;\n  letter-spacing: .17em;\n  color: #718095;\n  background: none;\n  -webkit-text-fill-color: #718095;\n  animation: none;\n}\n\n[data-testid="stAppViewContainer"] h1::after {\n  content: "";\n  display: block;\n  width: 82px;\n  height: 2px;\n  margin-top: 1.1rem;\n  border-radius: 99px;\n  background: linear-gradient(90deg, var(--cri-cyan), var(--cri-purple));\n  box-shadow: 0 0 22px rgba(97,218,251,.32);\n}\n\n/* Caption / product strapline */\n[data-testid="stAppViewContainer"] h1 + div {\n  color: #8290a1 !important;\n  font-family: "DM Mono", monospace;\n  font-size: .68rem !important;\n  letter-spacing: .04em;\n  line-height: 1.7;\n}\n\n/* Sidebar */\n[data-testid="stSidebar"] h2,\n[data-testid="stSidebar"] h3 {\n  color: #f2f6fa !important;\n}\n\n[data-testid="stSidebar"] .stMarkdown {\n  animation: criFadeUp .45s ease both;\n}\n\n[data-testid="stSidebar"] hr {\n  border-color: rgba(255,255,255,.07);\n}\n\n[data-testid="stSidebar"] [data-baseweb="select"] > div,\n[data-testid="stSidebar"] [data-baseweb="input"] > div {\n  background: rgba(255,255,255,.035);\n  border: 1px solid rgba(255,255,255,.08);\n  border-radius: 11px;\n  transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;\n}\n\n[data-testid="stSidebar"] [data-baseweb="select"] > div:hover,\n[data-testid="stSidebar"] [data-baseweb="input"] > div:hover {\n  border-color: rgba(97,218,251,.28);\n  box-shadow: 0 0 0 3px rgba(97,218,251,.035);\n  transform: translateY(-1px);\n}\n\n/* Buttons */\n.stButton > button,\n.stDownloadButton > button {\n  min-height: 2.55rem;\n  border-radius: 12px;\n  border: 1px solid rgba(97,218,251,.18);\n  background:\n    linear-gradient(135deg, rgba(97,218,251,.12), rgba(167,139,250,.08)),\n    rgba(255,255,255,.02);\n  color: #f2f6fb;\n  font-family: "DM Sans", sans-serif;\n  font-weight: 700;\n  letter-spacing: -.01em;\n  transition:\n    transform .2s ease,\n    border-color .2s ease,\n    box-shadow .2s ease,\n    background .2s ease;\n}\n\n.stButton > button:hover,\n.stDownloadButton > button:hover {\n  transform: translateY(-2px);\n  border-color: rgba(97,218,251,.48);\n  background:\n    linear-gradient(135deg, rgba(97,218,251,.17), rgba(167,139,250,.11)),\n    rgba(255,255,255,.025);\n  box-shadow:\n    0 10px 35px rgba(0,0,0,.28),\n    0 0 24px rgba(97,218,251,.08);\n}\n\n.stButton > button:active,\n.stDownloadButton > button:active {\n  transform: translateY(0);\n}\n\n/* Metrics become glass intelligence cards */\ndiv[data-testid="stMetric"] {\n  position: relative;\n  overflow: hidden;\n  min-height: 112px;\n  padding: 1rem 1.05rem;\n  border: 1px solid var(--cri-line);\n  border-radius: 18px;\n  background:\n    radial-gradient(circle at 90% 0%, rgba(97,218,251,.065), transparent 34%),\n    linear-gradient(145deg, rgba(255,255,255,.045), rgba(255,255,255,.012));\n  box-shadow:\n    0 18px 55px rgba(0,0,0,.2),\n    inset 0 1px 0 rgba(255,255,255,.025);\n  animation: criFadeUp .55s ease both;\n  transition:\n    transform .22s ease,\n    border-color .22s ease,\n    box-shadow .22s ease;\n}\n\ndiv[data-testid="stMetric"]::after {\n  content: "";\n  position: absolute;\n  left: 0;\n  right: 0;\n  bottom: 0;\n  height: 1px;\n  background: linear-gradient(90deg, transparent, rgba(97,218,251,.25), transparent);\n  transform: translateX(-100%);\n  transition: transform .55s ease;\n}\n\ndiv[data-testid="stMetric"]:hover {\n  transform: translateY(-3px);\n  border-color: rgba(97,218,251,.22);\n  box-shadow:\n    0 22px 70px rgba(0,0,0,.28),\n    0 0 30px rgba(97,218,251,.045);\n}\n\ndiv[data-testid="stMetric"]:hover::after {\n  transform: translateX(100%);\n}\n\ndiv[data-testid="stMetricLabel"] {\n  color: #7e8da0 !important;\n  font-family: "DM Mono", monospace;\n  font-size: .66rem;\n  letter-spacing: .09em;\n  text-transform: uppercase;\n}\n\ndiv[data-testid="stMetricValue"] {\n  color: #f5f8fc !important;\n  font-family: "Space Grotesk", sans-serif;\n  font-weight: 700;\n  letter-spacing: -.04em;\n}\n\ndiv[data-testid="stMetricDelta"] {\n  font-family: "DM Mono", monospace;\n  font-size: .68rem;\n}\n\n/* Charts */\n[data-testid="stPlotlyChart"] {\n  border: 1px solid var(--cri-line);\n  border-radius: 20px;\n  overflow: hidden;\n  background:\n    linear-gradient(145deg, rgba(255,255,255,.028), rgba(255,255,255,.008));\n  box-shadow: 0 18px 60px rgba(0,0,0,.22);\n  animation: criFadeUp .6s ease both;\n  transition: border-color .25s ease, transform .25s ease, box-shadow .25s ease;\n}\n\n[data-testid="stPlotlyChart"]:hover {\n  border-color: rgba(255,255,255,.11);\n  transform: translateY(-2px);\n  box-shadow: 0 24px 75px rgba(0,0,0,.28);\n}\n\n/* Dividers */\nhr {\n  border-color: rgba(255,255,255,.07) !important;\n}\n\n/* Alerts */\n[data-testid="stAlert"] {\n  border-radius: 16px;\n  border: 1px solid rgba(97,218,251,.12);\n  background: rgba(97,218,251,.035);\n}\n\n/* Expanders */\n[data-testid="stExpander"] {\n  border: 1px solid var(--cri-line);\n  border-radius: 17px;\n  background: rgba(255,255,255,.018);\n  transition: border-color .2s ease, background .2s ease;\n}\n\n[data-testid="stExpander"]:hover {\n  border-color: rgba(255,255,255,.12);\n  background: rgba(255,255,255,.024);\n}\n\n/* Dataframe */\n[data-testid="stDataFrame"] {\n  border: 1px solid var(--cri-line);\n  border-radius: 17px;\n  overflow: hidden;\n}\n\n/* Select slider */\n[data-testid="stSlider"] [role="slider"] {\n  box-shadow: 0 0 0 5px rgba(97,218,251,.08);\n}\n\n/* Generic Streamlit columns get subtle entrance motion */\n[data-testid="column"] {\n  animation: criFadeUp .58s ease both;\n}\n\n/* ==========================================================================\n   CUSTOM SECTION MARKERS\n   ========================================================================== */\n\n[data-testid="stAppViewContainer"] h2 {\n  position: relative;\n  margin-top: 2.1rem !important;\n  padding-top: 1rem;\n  color: #eef3f8 !important;\n}\n\n[data-testid="stAppViewContainer"] h2::before {\n  content: "";\n  position: absolute;\n  left: 0;\n  top: 0;\n  width: 34px;\n  height: 2px;\n  border-radius: 99px;\n  background: linear-gradient(90deg, var(--cri-cyan), var(--cri-purple));\n  box-shadow: 0 0 18px rgba(97,218,251,.22);\n}\n\n[data-testid="stAppViewContainer"] h2::after {\n  content: "INTELLIGENCE LAYER";\n  display: block;\n  margin-top: .3rem;\n  color: #566477;\n  font-family: "DM Mono", monospace;\n  font-size: .57rem;\n  font-weight: 400;\n  letter-spacing: .13em;\n}\n\n/* ==========================================================================\n   SCENARIO LAB VISUAL EMPHASIS\n   ========================================================================== */\n\n[data-testid="stSidebar"] h3 {\n  color: #ded8ff !important;\n}\n\n[data-testid="stSidebar"] h3::before {\n  content: "◈ ";\n  color: var(--cri-purple);\n}\n\n[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {\n  border-color: rgba(167,139,250,.3);\n  background:\n    linear-gradient(135deg, rgba(167,139,250,.16), rgba(97,218,251,.08));\n  box-shadow: 0 0 24px rgba(167,139,250,.06);\n}\n\n[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]:hover {\n  border-color: rgba(167,139,250,.6);\n  box-shadow:\n    0 12px 35px rgba(0,0,0,.3),\n    0 0 28px rgba(167,139,250,.11);\n}\n\n/* ==========================================================================\n   MICRO-INTERACTIONS\n   ========================================================================== */\n\na {\n  color: var(--cri-cyan) !important;\n  transition: opacity .2s ease;\n}\n\na:hover {\n  opacity: .8;\n}\n\n.stCaption {\n  color: #6f7d8e !important;\n}\n\n[data-testid="stMarkdownContainer"] strong {\n  color: #e7edf4;\n}\n\n[data-testid="stMarkdownContainer"] blockquote {\n  margin: 1rem 0;\n  padding: .8rem 1rem;\n  border-left: 2px solid var(--cri-purple);\n  border-radius: 0 12px 12px 0;\n  background: rgba(167,139,250,.035);\n  color: #b4becb;\n}\n\n/* ==========================================================================\n   RESPONSIVE LAYOUT\n   ========================================================================== */\n\n@media (max-width: 1200px) {\n  .block-container {\n    padding-left: 2rem;\n    padding-right: 2rem;\n  }\n}\n\n@media (max-width: 800px) {\n  .block-container {\n    padding-top: 1rem;\n    padding-left: 1rem;\n    padding-right: 1rem;\n  }\n\n  h1 {\n    font-size: 2.35rem !important;\n  }\n\n  [data-testid="stAppViewContainer"] h1::before {\n    font-size: .56rem;\n  }\n\n  div[data-testid="stMetric"] {\n    min-height: 98px;\n  }\n}\n\n/* ==========================================================================\n   MOTION\n   ========================================================================== */\n\n@keyframes criFadeUp {\n  from {\n    opacity: 0;\n    transform: translateY(9px);\n  }\n  to {\n    opacity: 1;\n    transform: translateY(0);\n  }\n}\n\n@keyframes criShimmer {\n  0%, 100% {\n    background-position: 0% center;\n  }\n  50% {\n    background-position: 100% center;\n  }\n}\n\n@keyframes criGrid {\n  from {\n    background-position: 0 0;\n  }\n  to {\n    background-position: 42px 42px;\n  }\n}\n\n@keyframes criOrb {\n  from {\n    transform: translate(0, 0) rotate(0deg);\n  }\n  to {\n    transform: translate(35px, 20px) rotate(12deg);\n  }\n}\n\n/* ==========================================================================\n   DESIGN TOKEN REFERENCE\n   ==========================================================================\n\n   Surface hierarchy:\n     --cri-bg        application background\n     --cri-surface   primary cards\n     --cri-surface-2 secondary cards\n     --cri-surface-3 interactive surfaces\n\n   Semantic accents:\n     --cri-cyan      data / active / primary\n     --cri-purple    intelligence / scenario\n     --cri-green     positive / risk reduction\n     --cri-red       deterioration / risk increase\n     --cri-amber     stress / scenario\n\n   Motion:\n     short hover transitions: 200–250ms\n     entrance animations: ~550–700ms\n     ambient animations: 8–16s\n\n   Accessibility:\n     text remains high contrast against the dark base\n     color is used as reinforcement rather than the sole label\n     controls remain native Streamlit controls\n     charts retain textual hover values\n   ========================================================================== */\n\n/* Additional component states */\n[data-testid="stFileUploader"] {\n  border-radius: 15px;\n}\n\n[data-testid="stProgressBar"] > div > div {\n  border-radius: 999px;\n}\n\n[data-testid="stToast"] {\n  border-radius: 14px;\n}\n\n[data-testid="stStatusWidget"] {\n  opacity: .75;\n}\n\n/* Loading polish */\n.stSpinner > div {\n  border-top-color: var(--cri-cyan) !important;\n}\n\n/* Tooltip polish */\n[data-baseweb="tooltip"] {\n  font-family: "DM Sans", sans-serif !important;\n}\n\n/* Scrollbar */\n::-webkit-scrollbar {\n  width: 9px;\n  height: 9px;\n}\n\n::-webkit-scrollbar-track {\n  background: #080b10;\n}\n\n::-webkit-scrollbar-thumb {\n  background: #27313d;\n  border-radius: 999px;\n}\n\n::-webkit-scrollbar-thumb:hover {\n  background: #364454;\n}\n\n/* More subtle card depth for every bordered Streamlit container */\ndiv[data-testid="stVerticalBlockBorderWrapper"] {\n  border-color: var(--cri-line) !important;\n  border-radius: 18px !important;\n  background: rgba(255,255,255,.015);\n}\n\n/* Form fields */\ninput, textarea {\n  color-scheme: dark;\n}\n\n/* Focus states */\nbutton:focus-visible,\ninput:focus-visible,\ntextarea:focus-visible,\nselect:focus-visible {\n  outline: 2px solid rgba(97,218,251,.55) !important;\n  outline-offset: 2px;\n}\n\n/* Print fallback */\n@media print {\n  .stApp::before,\n  .stApp::after {\n    display: none;\n  }\n\n  [data-testid="stSidebar"] {\n    display: none;\n  }\n\n  .block-container {\n    max-width: none;\n  }\n}\n\n/* ==========================================================================\n   PORTFOLIO POLISH\n   ========================================================================== */\n\n[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] {\n  animation: criFadeUp .45s ease both;\n}\n\n[data-testid="stAppViewContainer"] .stCaption {\n  letter-spacing: .01em;\n}\n\n[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"] {\n  gap: 1rem;\n}\n\n[data-testid="stAppViewContainer"] [data-testid="stVerticalBlock"] {\n  gap: .65rem;\n}\n\n/* Keep native Streamlit spacing readable while making the interface dense */\n[data-testid="stAppViewContainer"] .element-container {\n  margin-bottom: .15rem;\n}\n\n/* End of presentation-only design system */\n</style>\n', unsafe_allow_html=True)

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


# ============================================================================
# UI IMPLEMENTATION NOTES
# ============================================================================
# The CSS above is intentionally self-contained so this portfolio dashboard
# requires no Node.js build, React bundle, or separate front-end server.
#
# Original analytical flow remains:
#   load_config()
#   load_panel()
#   score_panel(panel)
#   top_drivers(...)
#   run_shock_scenario(...)
#   generate_report(...)
#
# The UI layer only changes typography, surfaces, spacing, motion, chart
# framing, control styling, responsive behavior, and visual hierarchy.
#
# This makes the file safe to drop into the existing repository while keeping
# the original model behavior and data pipeline untouched.
#
# Design direction:
#   - dark institutional / terminal aesthetic
#   - restrained cyan / violet intelligence accents
#   - glass surfaces rather than flat boxes
#   - low-amplitude motion for perceived responsiveness
#   - strong hierarchy around score, drivers, peers, scenarios, commentary
#   - native Streamlit controls retained for reliability
#
# No scoring weights, indicators, scenario targets, API calls, database logic,
# or commentary-generation rules are changed in this presentation edition.
# ============================================================================ 
