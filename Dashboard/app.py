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


# ============================================================================
# COUNTRY RISK INTELLIGENCE ENGINE
# Production UI / UX layer
# ============================================================================
# IMPORTANT:
#   - The analytical functions imported above are intentionally preserved.
#   - score_panel(), top_drivers(), run_shock_scenario(), and generate_report()
#     remain the source of truth for analytics.
#   - This file adds presentation, interaction, visualization, accessibility,
#     responsive layout, export controls, and terminal-style visual polish.
# ============================================================================

import html
import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# ============================================================================
# DESIGN TOKENS
# ============================================================================

APP_TITLE = "Country Risk Intelligence Engine"
APP_KICKER = "GLOBAL MACRO · COUNTRY RISK INTELLIGENCE"

COLORS = {
    "bg": "#06080d",
    "bg_2": "#0a0e15",
    "panel": "#0d121b",
    "panel_2": "#111824",
    "panel_3": "#151e2b",
    "border": "rgba(148,163,184,.15)",
    "border_strong": "rgba(148,163,184,.28)",
    "text": "#f4f7fb",
    "muted": "#8c98aa",
    "faint": "#566274",
    "cyan": "#5ee7f2",
    "blue": "#6ea8ff",
    "violet": "#a78bfa",
    "green": "#54d69a",
    "yellow": "#f6d365",
    "orange": "#ff9f5b",
    "red": "#ff6b7a",
    "white": "#ffffff",
}

BAND_COLORS_UI = {
    "Low": COLORS["green"],
    "Moderate": COLORS["yellow"],
    "Elevated": COLORS["orange"],
    "High": "#ff7d55",
    "Severe": COLORS["red"],
}


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# GLOBAL CSS
# ============================================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #06080d;
    --bg2: #0a0e15;
    --panel: #0d121b;
    --panel2: #111824;
    --panel3: #151e2b;
    --border: rgba(148,163,184,.15);
    --border-strong: rgba(148,163,184,.28);
    --text: #f4f7fb;
    --muted: #8c98aa;
    --faint: #566274;
    --cyan: #5ee7f2;
    --blue: #6ea8ff;
    --violet: #a78bfa;
    --green: #54d69a;
    --yellow: #f6d365;
    --orange: #ff9f5b;
    --red: #ff6b7a;
}

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(94,231,242,.08), transparent 28%),
        radial-gradient(circle at 88% 8%, rgba(167,139,250,.09), transparent 25%),
        radial-gradient(circle at 50% 100%, rgba(110,168,255,.06), transparent 30%),
        #06080d;
    color: var(--text);
}

.stApp:before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        linear-gradient(rgba(255,255,255,.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.018) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: linear-gradient(to bottom, black, transparent 80%);
}

.block-container {
    max-width: 1600px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

header[data-testid="stHeader"] {
    background: rgba(6,8,13,.72);
    backdrop-filter: blur(16px);
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(13,18,27,.97), rgba(7,10,15,.98));
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.4rem;
}

section[data-testid="stSidebar"] * {
    color: var(--text);
}

div[data-testid="stMetric"] {
    background: transparent;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 10px;
    border: 1px solid var(--border-strong);
    background: rgba(17,24,36,.82);
    color: var(--text);
    font-weight: 600;
    transition: all .2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: rgba(94,231,242,.55);
    box-shadow: 0 0 22px rgba(94,231,242,.12);
    transform: translateY(-1px);
}

.stSelectbox [data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
    background: rgba(17,24,36,.78);
    border-color: var(--border);
}

div[data-baseweb="select"] {
    border-radius: 10px;
}

label {
    color: var(--muted) !important;
}

hr {
    border-color: var(--border);
}

div[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 14px;
    background: rgba(13,18,27,.66);
}

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

.hero {
    position: relative;
    overflow: hidden;
    min-height: 250px;
    padding: 34px 38px;
    border: 1px solid var(--border);
    border-radius: 24px;
    background:
        radial-gradient(circle at 80% 20%, rgba(94,231,242,.12), transparent 30%),
        radial-gradient(circle at 15% 85%, rgba(167,139,250,.10), transparent 30%),
        linear-gradient(135deg, rgba(17,24,36,.96), rgba(8,12,18,.92));
    box-shadow:
        0 24px 80px rgba(0,0,0,.35),
        inset 0 1px 0 rgba(255,255,255,.035);
    animation: heroIn .7s ease both;
}

.hero:after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -90px;
    top: -100px;
    border-radius: 50%;
    border: 1px solid rgba(94,231,242,.16);
    box-shadow:
        0 0 0 30px rgba(94,231,242,.025),
        0 0 0 60px rgba(94,231,242,.018),
        0 0 90px rgba(94,231,242,.12);
    animation: orbit 7s linear infinite;
}

.hero-grid {
    display: grid;
    grid-template-columns: 1.35fr .65fr;
    gap: 30px;
    align-items: center;
    position: relative;
    z-index: 1;
}

.kicker {
    font-family: "DM Mono", monospace;
    font-size: 11px;
    letter-spacing: .18em;
    color: var(--cyan);
    margin-bottom: 12px;
}

.hero h1 {
    margin: 0;
    font-family: "Space Grotesk", sans-serif;
    font-size: clamp(32px, 4.5vw, 62px);
    line-height: .98;
    letter-spacing: -.045em;
    color: #fff;
}

.hero h1 span {
    background: linear-gradient(100deg, #fff, #bceff3 42%, #a78bfa 90%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero-copy {
    max-width: 740px;
    margin-top: 18px;
    color: var(--muted);
    font-size: 15px;
    line-height: 1.7;
}

.hero-terminal {
    justify-self: end;
    width: min(100%, 390px);
    padding: 18px;
    border: 1px solid var(--border);
    border-radius: 15px;
    background: rgba(0,0,0,.28);
    font-family: "DM Mono", monospace;
    font-size: 12px;
    color: #a9b6c7;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
}

.terminal-top {
    display: flex;
    gap: 7px;
    margin-bottom: 14px;
}

.terminal-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #3a4351;
}

.terminal-dot.live {
    background: var(--green);
    box-shadow: 0 0 12px rgba(84,214,154,.7);
    animation: blink 1.8s infinite;
}

.terminal-line {
    margin: 6px 0;
}

.terminal-line b {
    color: var(--cyan);
}

.section-head {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 20px;
    margin: 34px 0 13px;
}

.section-title {
    font-family: "Space Grotesk", sans-serif;
    font-size: 21px;
    font-weight: 700;
    letter-spacing: -.02em;
    color: #fff;
}

.section-sub {
    color: var(--muted);
    font-size: 12px;
    margin-top: 4px;
}

.card {
    position: relative;
    overflow: hidden;
    height: 100%;
    border: 1px solid var(--border);
    border-radius: 17px;
    background:
        linear-gradient(145deg, rgba(17,24,36,.9), rgba(10,14,21,.84));
    box-shadow:
        0 16px 50px rgba(0,0,0,.18),
        inset 0 1px 0 rgba(255,255,255,.025);
    padding: 20px;
    transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease;
}

.card:hover {
    transform: translateY(-2px);
    border-color: rgba(148,163,184,.28);
    box-shadow:
        0 22px 60px rgba(0,0,0,.28),
        0 0 30px rgba(94,231,242,.035);
}

.card:before {
    content: "";
    position: absolute;
    width: 140px;
    height: 140px;
    top: -90px;
    right: -70px;
    border-radius: 50%;
    background: rgba(94,231,242,.045);
    filter: blur(5px);
}

.card-label {
    font-family: "DM Mono", monospace;
    color: var(--muted);
    font-size: 10px;
    letter-spacing: .13em;
    text-transform: uppercase;
}

.card-value {
    font-family: "Space Grotesk", sans-serif;
    color: #fff;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -.04em;
    margin-top: 7px;
}

.card-caption {
    color: var(--muted);
    font-size: 11px;
    margin-top: 3px;
}

.kpi {
    min-height: 128px;
}

.kpi-accent {
    width: 34px;
    height: 3px;
    border-radius: 10px;
    margin-bottom: 17px;
    background: linear-gradient(90deg, var(--cyan), var(--violet));
    box-shadow: 0 0 18px rgba(94,231,242,.35);
}

.score-card {
    min-height: 410px;
}

.score-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 8px 0 0;
}

.score-ring {
    width: 225px;
    height: 225px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    position: relative;
    background: conic-gradient(
        var(--score-color) calc(var(--score-pct) * 1%),
        rgba(255,255,255,.055) 0
    );
    box-shadow:
        0 0 55px color-mix(in srgb, var(--score-color) 18%, transparent),
        inset 0 0 35px rgba(0,0,0,.35);
}

.score-ring:before {
    content: "";
    position: absolute;
    inset: 10px;
    border-radius: 50%;
    background: #090d14;
    border: 1px solid rgba(255,255,255,.06);
}

.score-inner {
    position: relative;
    z-index: 1;
    text-align: center;
}

.score-number {
    font-family: "Space Grotesk", sans-serif;
    font-size: 52px;
    font-weight: 700;
    letter-spacing: -.06em;
    line-height: 1;
}

.score-band {
    font-family: "DM Mono", monospace;
    color: var(--score-color);
    font-size: 11px;
    letter-spacing: .14em;
    margin-top: 8px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 999px;
    padding: 6px 10px;
    color: var(--muted);
    font-family: "DM Mono", monospace;
    font-size: 10px;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 10px rgba(84,214,154,.65);
}

.driver-list {
    display: grid;
    gap: 11px;
    margin-top: 13px;
}

.driver-row {
    display: grid;
    grid-template-columns: 1fr 70px;
    gap: 12px;
    align-items: center;
}

.driver-name {
    color: #dce4ee;
    font-size: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.driver-meta {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    color: var(--muted);
    font-family: "DM Mono", monospace;
    font-size: 9px;
    margin-bottom: 5px;
}

.driver-bar {
    height: 6px;
    border-radius: 20px;
    overflow: hidden;
    background: rgba(255,255,255,.055);
}

.driver-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--cyan), var(--violet));
    box-shadow: 0 0 14px rgba(94,231,242,.2);
}

.driver-score {
    text-align: right;
    color: #fff;
    font-family: "DM Mono", monospace;
    font-size: 11px;
}

.signal-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    margin-top: 14px;
}

.signal {
    padding: 12px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: rgba(0,0,0,.12);
}

.signal-code {
    color: var(--cyan);
    font-family: "DM Mono", monospace;
    font-size: 9px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.signal-state {
    color: #dfe6ee;
    font-size: 11px;
    margin-top: 6px;
}

.signal-line {
    height: 3px;
    border-radius: 10px;
    background: linear-gradient(90deg, var(--green), transparent);
    margin-top: 8px;
}

.peer-highlight {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid rgba(148,163,184,.08);
}

.peer-highlight:last-child {
    border-bottom: 0;
}

.peer-country {
    font-weight: 600;
    color: #e7edf5;
}

.peer-score {
    font-family: "DM Mono", monospace;
    color: var(--cyan);
}

.scenario-hero {
    padding: 22px;
    border-radius: 17px;
    border: 1px solid rgba(167,139,250,.22);
    background:
        radial-gradient(circle at 90% 0%, rgba(167,139,250,.12), transparent 35%),
        linear-gradient(135deg, rgba(18,19,35,.94), rgba(11,13,22,.9));
}

.scenario-label {
    color: var(--violet);
    font-family: "DM Mono", monospace;
    font-size: 10px;
    letter-spacing: .13em;
}

.scenario-title {
    color: #fff;
    font-family: "Space Grotesk", sans-serif;
    font-size: 24px;
    font-weight: 700;
    margin-top: 8px;
}

.delta-positive {
    color: var(--red);
}

.delta-negative {
    color: var(--green);
}

.delta-neutral {
    color: var(--muted);
}

.intel {
    padding: 24px;
    border-radius: 18px;
    border: 1px solid rgba(94,231,242,.16);
    background:
        radial-gradient(circle at 0% 0%, rgba(94,231,242,.09), transparent 28%),
        linear-gradient(135deg, rgba(13,24,31,.94), rgba(10,14,21,.92));
}

.intel-head {
    display: flex;
    gap: 13px;
    align-items: center;
}

.intel-icon {
    width: 38px;
    height: 38px;
    display: grid;
    place-items: center;
    border-radius: 11px;
    background: rgba(94,231,242,.09);
    border: 1px solid rgba(94,231,242,.15);
    font-size: 18px;
}

.intel-title {
    font-family: "Space Grotesk", sans-serif;
    font-weight: 700;
    color: #fff;
}

.intel-sub {
    color: var(--muted);
    font-size: 11px;
}

.intel-body {
    margin-top: 17px;
    color: #c6d0dc;
    font-size: 13px;
    line-height: 1.75;
}

.metadata {
    display: grid;
    grid-template-columns: repeat(4, minmax(0,1fr));
    gap: 10px;
    margin-top: 14px;
}

.meta-item {
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
    background: rgba(0,0,0,.11);
}

.meta-k {
    color: var(--faint);
    font-family: "DM Mono", monospace;
    font-size: 9px;
    letter-spacing: .1em;
    text-transform: uppercase;
}

.meta-v {
    color: #dce5ef;
    font-size: 12px;
    margin-top: 6px;
    word-break: break-word;
}

.footer {
    margin-top: 48px;
    padding-top: 18px;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    gap: 18px;
    color: var(--faint);
    font-family: "DM Mono", monospace;
    font-size: 9px;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.micro {
    color: var(--faint);
    font-family: "DM Mono", monospace;
    font-size: 9px;
}

.empty {
    min-height: 130px;
    display: grid;
    place-items: center;
    text-align: center;
    border: 1px dashed var(--border-strong);
    border-radius: 14px;
    color: var(--muted);
    padding: 20px;
}

@keyframes heroIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes orbit {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: .42; }
}

@media (max-width: 900px) {
    .hero-grid {
        grid-template-columns: 1fr;
    }
    .hero-terminal {
        justify-self: start;
    }
    .metadata {
        grid-template-columns: repeat(2, minmax(0,1fr));
    }
}

@media (max-width: 600px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .hero {
        padding: 25px 22px;
    }
    .hero h1 {
        font-size: 36px;
    }
    .metadata {
        grid-template-columns: 1fr;
    }
    .signal-grid {
        grid-template-columns: 1fr;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# DATA / CONFIGURATION LOADING
# ============================================================================

@st.cache_data(show_spinner=False)
def load_configurations():
    countries = yaml.safe_load(
        (CONFIG_DIR / "countries.yaml").read_text(encoding="utf-8")
    )
    indicators = yaml.safe_load(
        (CONFIG_DIR / "indicators.yaml").read_text(encoding="utf-8")
    )
    return countries, indicators


@st.cache_data(show_spinner=False)
def load_panel():
    return pd.read_csv(PROCESSED_DIR / "panel_wide.csv")


countries_cfg, indicators_cfg = load_configurations()
panel = load_panel()


# ============================================================================
# UTILITY HELPERS
# ============================================================================

def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_number(value, digits=1):
    value = safe_float(value)
    return f"{value:,.{digits}f}"


def fmt_delta(value, digits=1):
    value = safe_float(value)
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.{digits}f}"


def esc(value):
    return html.escape(str(value))


def normalize_band(value):
    text = str(value).strip()
    if text in BAND_COLORS_UI:
        return text
    return "Elevated"


def band_color(value):
    return BAND_COLORS_UI.get(normalize_band(value), COLORS["orange"])


def score_pct(score):
    return max(0.0, min(100.0, safe_float(score)))


def score_band(score):
    score = safe_float(score)
    if score < 20:
        return "Low"
    if score < 40:
        return "Moderate"
    if score < 60:
        return "Elevated"
    if score < 80:
        return "High"
    return "Severe"


def get_iso(country):
    if isinstance(countries_cfg, dict):
        entry = countries_cfg.get(country, {})
        if isinstance(entry, dict):
            return (
                entry.get("iso3")
                or entry.get("ISO3")
                or entry.get("code")
                or entry.get("iso")
                or country
            )
    return country


def get_country_label(country):
    if isinstance(countries_cfg, dict):
        entry = countries_cfg.get(country, {})
        if isinstance(entry, dict):
            return (
                entry.get("name")
                or entry.get("country")
                or entry.get("label")
                or country
            )
    return country


def available_countries(df):
    candidates = []
    for column in ["country", "country_name", "iso3", "ISO3"]:
        if column in df.columns:
            candidates = sorted(df[column].dropna().astype(str).unique().tolist())
            if candidates:
                break
    return candidates


def available_years(df):
    if "year" not in df.columns:
        return []
    years = pd.to_numeric(df["year"], errors="coerce").dropna()
    return sorted(years.astype(int).unique().tolist())


def row_for(df, country, year):
    if "country" in df.columns:
        mask = df["country"].astype(str).eq(str(country))
    elif "iso3" in df.columns:
        mask = df["iso3"].astype(str).eq(str(country))
    else:
        mask = pd.Series(False, index=df.index)

    if "year" in df.columns:
        mask &= pd.to_numeric(df["year"], errors="coerce").eq(int(year))

    selected = df.loc[mask]
    if selected.empty:
        return pd.Series(dtype=object)
    return selected.iloc[0]


def find_country_column(df):
    for column in ["country", "country_name", "iso3", "ISO3"]:
        if column in df.columns:
            return column
    return None


def find_score_column(df):
    candidates = [
        "risk_score",
        "score",
        "composite_risk_score",
        "RISK_SCORE",
    ]
    for column in candidates:
        if column in df.columns:
            return column
    return None


def find_completeness_column(df):
    candidates = [
        "completeness",
        "data_completeness",
        "coverage",
        "coverage_pct",
    ]
    for column in candidates:
        if column in df.columns:
            return column
    return None


def make_plotly_layout(fig, height=360, margin=None):
    if margin is None:
        margin = dict(l=8, r=8, t=25, b=8)

    fig.update_layout(
        height=height,
        margin=margin,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans, sans-serif",
            color=COLORS["muted"],
            size=11,
        ),
        title_font=dict(
            family="Space Grotesk, sans-serif",
            color=COLORS["text"],
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["muted"], size=10),
        ),
        hoverlabel=dict(
            bgcolor="#111824",
            bordercolor="rgba(94,231,242,.3)",
            font=dict(color="#f4f7fb"),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(148,163,184,.12)",
        tickfont=dict(color=COLORS["faint"], size=10),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,.055)",
        zeroline=False,
        linecolor="rgba(148,163,184,.08)",
        tickfont=dict(color=COLORS["faint"], size=10),
    )
    return fig


def empty_state(message):
    st.markdown(
        f'<div class="empty">{esc(message)}</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# SIDEBAR — ORIGINAL CONTROLS + REFRESH / EXPORT
# ============================================================================

with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:4px 4px 16px;">
            <div class="kicker">RISK ENGINE / LIVE PANEL</div>
            <div style="font-family:'Space Grotesk';font-size:21px;font-weight:700;">
                Control Room
            </div>
            <div class="micro" style="margin-top:7px;">
                Select the analytical slice.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    country_options = available_countries(panel)

    if not country_options:
        st.error("No country dimension found in panel_wide.csv.")
        st.stop()

    default_country = country_options[0]

    country = st.selectbox(
        "Country",
        country_options,
        index=country_options.index(default_country),
        key="country_selector",
    )

    years = available_years(panel)
    if not years:
        st.error("No year dimension found in panel_wide.csv.")
        st.stop()

    year = st.selectbox(
        "Year",
        years,
        index=len(years) - 1,
        key="year_selector",
    )

    st.markdown("---")

    st.markdown(
        '<div class="kicker" style="margin-bottom:7px;">SCENARIO LAB</div>',
        unsafe_allow_html=True,
    )

    shock = st.number_input(
        "Policy rate YoY change (bps)",
        min_value=-1000,
        max_value=1000,
        value=0,
        step=25,
        key="policy_rate_shock",
        help="Original scenario driver: POLICY_RATE_YOY_CHANGE_BPS",
    )

    st.markdown("---")

    refresh_col1, refresh_col2 = st.columns(2)

    with refresh_col1:
        if st.button("↻ Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with refresh_col2:
        st.caption("Panel")
        st.caption(f"{len(panel):,} rows")

    st.markdown("---")

    st.markdown(
        """
        <div class="micro">
            ENGINE STATUS<br>
            <span style="color:#54d69a;">● ONLINE</span><br><br>
            Analytics remain deterministic and traceable.
            Presentation is layered on top of the existing engine.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# ORIGINAL ANALYTICAL EXECUTION
# ============================================================================

# These calls are intentionally retained as the original analytical pipeline.
# Do not replace them with UI-derived calculations.

scores, drivers = score_panel(panel)

try:
    current_row = row_for(panel, country, year)
except Exception:
    current_row = pd.Series(dtype=object)

try:
    country_drivers = top_drivers(drivers, country, year, n=6)
except Exception:
    country_drivers = pd.DataFrame()

try:
    scenario = run_shock_scenario(
        panel,
        country,
        year,
        "POLICY_RATE_YOY_CHANGE_BPS",
        shock,
        [
            "FX_YOY_DEPRECIATION_PCT",
            "NY.GDP.MKTP.KD.ZG",
            "GC.DOD.TOTL.GD.ZS",
        ],
    )
except Exception as exc:
    scenario = None
    scenario_error = exc
else:
    scenario_error = None

try:
    report = generate_report(
        country=country,
        year=year,
        score=scores,
        drivers=drivers,
        scenario=scenario,
    )
except Exception:
    # Some original implementations use a different positional signature.
    # Fall back to the original style if available.
    try:
        report = generate_report(country, year, scores, drivers, scenario)
    except Exception as exc:
        report = f"Analyst commentary unavailable: {exc}"


# ============================================================================
# NORMALIZE ANALYTICAL OUTPUTS FOR DISPLAY ONLY
# ============================================================================

score_value = None

if isinstance(scores, pd.DataFrame):
    score_column = find_score_column(scores)
    if score_column:
        try:
            if "country" in scores.columns and "year" in scores.columns:
                score_match = scores[
                    scores["country"].astype(str).eq(str(country))
                    & pd.to_numeric(scores["year"], errors="coerce").eq(int(year))
                ]
            elif "iso3" in scores.columns and "year" in scores.columns:
                iso = get_iso(country)
                score_match = scores[
                    scores["iso3"].astype(str).eq(str(iso))
                    & pd.to_numeric(scores["year"], errors="coerce").eq(int(year))
                ]
            else:
                score_match = scores
            if not score_match.empty:
                score_value = safe_float(score_match.iloc[0][score_column])
        except Exception:
            score_value = None

elif isinstance(scores, dict):
    for key in ["risk_score", "score", "composite_risk_score"]:
        if key in scores:
            score_value = safe_float(scores[key])
            break

if score_value is None and not current_row.empty:
    for key in ["risk_score", "score", "composite_risk_score"]:
        if key in current_row.index:
            score_value = safe_float(current_row[key])
            break

if score_value is None:
    score_value = 0.0

band = score_band(score_value)
score_color = band_color(band)

coverage_value = None

if not current_row.empty:
    for key in ["completeness", "data_completeness", "coverage", "coverage_pct"]:
        if key in current_row.index:
            coverage_value = safe_float(current_row[key])
            break

if coverage_value is None and isinstance(scores, pd.DataFrame):
    completeness_col = find_completeness_column(scores)
    if completeness_col and not scores.empty:
        try:
            coverage_value = safe_float(scores.iloc[0][completeness_col])
        except Exception:
            pass

if coverage_value is None:
    coverage_value = float("nan")


# ============================================================================
# HERO
# ============================================================================

iso = get_iso(country)
country_label = get_country_label(country)
generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-grid">
            <div>
                <div class="kicker">{APP_KICKER}</div>
                <h1>Country Risk<br><span>Intelligence Engine</span></h1>
                <div class="hero-copy">
                    A decision-oriented macro risk cockpit combining country-level
                    indicators, deterministic scoring, driver decomposition,
                    peer context and transparent scenario analysis.
                </div>
                <div style="margin-top:20px;display:flex;gap:9px;flex-wrap:wrap;">
                    <div class="status-pill">
                        <span class="status-dot"></span>
                        ENGINE ONLINE
                    </div>
                    <div class="status-pill">{esc(iso)} · {int(year)}</div>
                    <div class="status-pill">TRACEABLE ANALYTICS</div>
                </div>
            </div>

            <div class="hero-terminal">
                <div class="terminal-top">
                    <span class="terminal-dot live"></span>
                    <span class="terminal-dot"></span>
                    <span class="terminal-dot"></span>
                </div>
                <div class="terminal-line"><b>$</b> country.select → {esc(iso)}</div>
                <div class="terminal-line"><b>$</b> period.lock → {int(year)}</div>
                <div class="terminal-line"><b>$</b> risk.compute → {fmt_number(score_value, 1)}</div>
                <div class="terminal-line"><b>$</b> scenario.delta → {fmt_delta(shock, 0)} bps</div>
                <div class="terminal-line" style="margin-top:12px;color:#54d69a;">
                    ✓ analytical layer ready
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# KPI STRIP
# ============================================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">Executive snapshot</div>
            <div class="section-sub">
                The selected country-year slice at a glance.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="card kpi">
            <div class="kpi-accent"></div>
            <div class="card-label">Composite risk</div>
            <div class="card-value" style="color:{score_color};">
                {fmt_number(score_value,1)}
            </div>
            <div class="card-caption">{esc(band)} risk band · 0–100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    coverage_text = "—" if pd.isna(coverage_value) else f"{fmt_number(coverage_value,1)}%"
    st.markdown(
        f"""
        <div class="card kpi">
            <div class="kpi-accent"></div>
            <div class="card-label">Data coverage</div>
            <div class="card-value">{coverage_text}</div>
            <div class="card-caption">available observations in selected slice</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    previous_score = None
    if isinstance(scores, pd.DataFrame):
        score_column = find_score_column(scores)
        if score_column and "year" in scores.columns:
            try:
                prior = scores[
                    (scores["country"].astype(str).eq(str(country)))
                    & (pd.to_numeric(scores["year"], errors="coerce") == int(year) - 1)
                ]
                if not prior.empty:
                    previous_score = safe_float(prior.iloc[0][score_column])
            except Exception:
                pass

    movement = score_value - previous_score if previous_score is not None else float("nan")
    movement_text = "—" if pd.isna(movement) else fmt_delta(movement, 1)
    movement_class = (
        "delta-positive"
        if not pd.isna(movement) and movement > 0
        else "delta-negative"
        if not pd.isna(movement) and movement < 0
        else "delta-neutral"
    )

    st.markdown(
        f"""
        <div class="card kpi">
            <div class="kpi-accent"></div>
            <div class="card-label">YoY movement</div>
            <div class="card-value {movement_class}">{movement_text}</div>
            <div class="card-caption">
                versus {int(year)-1} composite score
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    panel_position = "—"
    score_column = find_score_column(scores) if isinstance(scores, pd.DataFrame) else None

    if score_column and isinstance(scores, pd.DataFrame) and "year" in scores.columns:
        try:
            same_year = scores[
                pd.to_numeric(scores["year"], errors="coerce").eq(int(year))
            ][score_column].dropna()
            if not same_year.empty:
                rank = int((same_year > score_value).sum()) + 1
                panel_position = f"#{rank} / {len(same_year)}"
        except Exception:
            pass

    st.markdown(
        f"""
        <div class="card kpi">
            <div class="kpi-accent"></div>
            <div class="card-label">Panel position</div>
            <div class="card-value">{panel_position}</div>
            <div class="card-caption">relative risk rank in selected year</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# RISK SCORE + TRAJECTORY
# ============================================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">Risk trajectory</div>
            <div class="section-sub">
                Current risk level and historical movement for the selected country.
            </div>
        </div>
        <div class="micro">SIGNAL / SCORE / TIME</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([.85, 1.55], gap="large")

with left:
    st.markdown(
        f"""
        <div class="card score-card">
            <div class="card-label">RISK SCORE GAUGE</div>
            <div class="score-wrap">
                <div class="score-ring"
                     style="--score-pct:{score_pct(score_value)};--score-color:{score_color};">
                    <div class="score-inner">
                        <div class="score-number">{fmt_number(score_value,0)}</div>
                        <div class="score-band">{esc(band.upper())}</div>
                    </div>
                </div>
            </div>
            <div style="text-align:center;margin-top:16px;">
                <span class="status-pill">
                    <span class="status-dot"></span>
                    {esc(country_label)} · {int(year)}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    fig = go.Figure()

    if isinstance(scores, pd.DataFrame):
        score_column = find_score_column(scores)
        country_column = find_country_column(scores)

        if score_column and country_column and "year" in scores.columns:
            history = scores[
                scores[country_column].astype(str).eq(str(country))
            ].copy()

            if history.empty and country_column in ["iso3", "ISO3"]:
                history = scores[
                    scores[country_column].astype(str).eq(str(iso))
                ].copy()

            history["year"] = pd.to_numeric(
                history["year"], errors="coerce"
            )
            history[score_column] = pd.to_numeric(
                history[score_column], errors="coerce"
            )
            history = history.dropna(subset=["year", score_column]).sort_values("year")

            if not history.empty:
                fig.add_trace(
                    go.Scatter(
                        x=history["year"],
                        y=history[score_column],
                        mode="lines+markers",
                        line=dict(
                            color=COLORS["cyan"],
                            width=3,
                            shape="spline",
                        ),
                        marker=dict(
                            size=7,
                            color=COLORS["cyan"],
                            line=dict(
                                color="#081018",
                                width=2,
                            ),
                        ),
                        fill="tozeroy",
                        fillcolor="rgba(94,231,242,.045)",
                        hovertemplate="<b>%{x}</b><br>Risk score: %{y:.1f}<extra></extra>",
                        name="Risk score",
                    )
                )

                fig.add_hline(
                    y=40,
                    line_dash="dot",
                    line_color="rgba(246,211,101,.35)",
                )
                fig.add_hline(
                    y=60,
                    line_dash="dot",
                    line_color="rgba(255,159,91,.35)",
                )
                fig.add_hline(
                    y=80,
                    line_dash="dot",
                    line_color="rgba(255,107,122,.35)",
                )
            else:
                empty_state("No historical score series is available for this country.")
        else:
            empty_state("Score history columns were not found in the analytical output.")
    else:
        empty_state("Risk trajectory requires a DataFrame score output.")

    if len(fig.data) > 0:
        fig.update_yaxes(range=[0, 100], title="Risk score")
        fig.update_xaxes(title="Year")
        make_plotly_layout(fig, height=410)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ============================================================================
# DRIVER DECOMPOSITION + SIGNAL BOARD
# ============================================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">Driver decomposition</div>
            <div class="section-sub">
                The highest-impact signals returned by the existing scoring engine.
            </div>
        </div>
        <div class="micro">TOP DRIVERS / TRACEABLE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

driver_left, driver_right = st.columns([1.3, 1], gap="large")

with driver_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-label">PRIMARY RISK CONTRIBUTORS</div>',
        unsafe_allow_html=True,
    )

    if isinstance(country_drivers, pd.DataFrame) and not country_drivers.empty:
        ddf = country_drivers.copy()

        numeric_candidates = [
            "contribution",
            "impact",
            "weight",
            "score",
            "value",
        ]

        driver_value_column = None
        for candidate in numeric_candidates:
            if candidate in ddf.columns:
                driver_value_column = candidate
                break

        driver_name_column = None
        for candidate in [
            "indicator",
            "indicator_code",
            "code",
            "driver",
            "name",
        ]:
            if candidate in ddf.columns:
                driver_name_column = candidate
                break

        if driver_name_column is None:
            driver_name_column = ddf.columns[0]

        if driver_value_column is None:
            numeric_cols = ddf.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                driver_value_column = numeric_cols[-1]

        if driver_value_column is not None:
            ddf[driver_value_column] = pd.to_numeric(
                ddf[driver_value_column],
                errors="coerce",
            ).fillna(0)

            magnitude = ddf[driver_value_column].abs()
            maximum = max(float(magnitude.max()), 1e-9)

            st.markdown('<div class="driver-list">', unsafe_allow_html=True)

            for _, item in ddf.head(8).iterrows():
                name = str(item[driver_name_column])
                value = safe_float(item[driver_value_column])
                width = min(100, abs(value) / maximum * 100)

                st.markdown(
                    f"""
                    <div class="driver-row">
                        <div>
                            <div class="driver-meta">
                                <span class="driver-name">{esc(name)}</span>
                                <span>{fmt_delta(value,2)}</span>
                            </div>
                            <div class="driver-bar">
                                <div class="driver-fill" style="width:{width:.1f}%"></div>
                            </div>
                        </div>
                        <div class="driver-score">{fmt_number(value,2)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

        else:
            empty_state("Driver output contains no numeric contribution field.")
    else:
        empty_state("No driver decomposition was returned for this slice.")

    st.markdown("</div>", unsafe_allow_html=True)


with driver_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-label">SIGNAL BOARD</div>',
        unsafe_allow_html=True,
    )

    signal_items = []

    if not current_row.empty:
        preferred_signals = [
            "FX_YOY_DEPRECIATION_PCT",
            "NY.GDP.MKTP.KD.ZG",
            "GC.DOD.TOTL.GD.ZS",
            "POLICY_RATE_YOY_CHANGE_BPS",
            "inflation",
            "unemployment",
            "current_account",
            "reserves",
        ]

        for code in preferred_signals:
            if code in current_row.index:
                signal_items.append((code, current_row[code]))

    if not signal_items and isinstance(country_drivers, pd.DataFrame):
        for column in country_drivers.columns[:6]:
            if column not in signal_items:
                signal_items.append((column, "available"))

    if signal_items:
        st.markdown('<div class="signal-grid">', unsafe_allow_html=True)

        for code, value in signal_items[:8]:
            numeric = None
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                pass

            if numeric is None:
                state = str(value)
                fill = 45
            else:
                state = fmt_number(numeric, 2)
                fill = min(100, max(10, abs(numeric)))

            st.markdown(
                f"""
                <div class="signal">
                    <div class="signal-code">{esc(code)}</div>
                    <div class="signal-state">{esc(state)}</div>
                    <div class="signal-line" style="width:{fill:.1f}%;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        empty_state("No selected indicator signals are available.")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# PEER COMPARISON
# ============================================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">Peer comparison</div>
            <div class="section-sub">
                Relative risk positioning against the wider panel in the selected year.
            </div>
        </div>
        <div class="micro">CROSS-SECTIONAL CONTEXT</div>
    </div>
    """,
    unsafe_allow_html=True,
)

peer_left, peer_right = st.columns([1.45, .75], gap="large")

with peer_left:
    peer_fig = go.Figure()

    if isinstance(scores, pd.DataFrame):
        score_column = find_score_column(scores)
        country_column = find_country_column(scores)

        if score_column and country_column and "year" in scores.columns:
            peers = scores[
                pd.to_numeric(scores["year"], errors="coerce").eq(int(year))
            ].copy()

            peers[score_column] = pd.to_numeric(
                peers[score_column],
                errors="coerce",
            )
            peers = peers.dropna(subset=[score_column])
            peers = peers.sort_values(score_column, ascending=True)

            if not peers.empty:
                peers["label"] = peers[country_column].astype(str)

                marker_colors = [
                    score_color if str(x) == str(country) else "rgba(110,168,255,.55)"
                    for x in peers["label"]
                ]

                peer_fig.add_trace(
                    go.Bar(
                        x=peers[score_column],
                        y=peers["label"],
                        orientation="h",
                        marker=dict(
                            color=marker_colors,
                            line=dict(
                                color="rgba(255,255,255,.04)",
                                width=1,
                            ),
                        ),
                        hovertemplate="%{y}<br>Risk score: %{x:.1f}<extra></extra>",
                    )
                )

                peer_fig.add_vline(
                    x=score_value,
                    line_dash="dash",
                    line_color=score_color,
                    annotation_text="SELECTED",
                    annotation_font_color=score_color,
                )

                peer_fig.update_xaxes(range=[0, 100], title="Risk score")
                peer_fig.update_yaxes(title=None)
            else:
                empty_state("No peer scores are available for this year.")
        else:
            empty_state("Peer comparison requires country, year and score columns.")
    else:
        empty_state("Peer comparison requires DataFrame score output.")

    if len(peer_fig.data) > 0:
        make_plotly_layout(
            peer_fig,
            height=max(360, min(620, 130 + 24 * len(peer_fig.data[0].y))),
            margin=dict(l=8, r=8, t=20, b=8),
        )
        st.plotly_chart(
            peer_fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

with peer_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-label">RELATIVE POSITION</div>',
        unsafe_allow_html=True,
    )

    if isinstance(scores, pd.DataFrame):
        score_column = find_score_column(scores)
        country_column = find_country_column(scores)

        if score_column and country_column and "year" in scores.columns:
            peers = scores[
                pd.to_numeric(scores["year"], errors="coerce").eq(int(year))
            ].copy()
            peers[score_column] = pd.to_numeric(peers[score_column], errors="coerce")
            peers = peers.dropna(subset=[score_column]).sort_values(
                score_column,
                ascending=False,
            )

            top_peers = peers.head(7)

            for _, peer in top_peers.iterrows():
                peer_name = str(peer[country_column])
                peer_score = safe_float(peer[score_column])

                st.markdown(
                    f"""
                    <div class="peer-highlight">
                        <div class="peer-country">{esc(peer_name)}</div>
                        <div class="peer-score">{fmt_number(peer_score,1)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            empty_state("Relative-position table unavailable.")
    else:
        empty_state("Relative-position table unavailable.")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# SCENARIO LABORATORY
# ============================================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">Scenario laboratory</div>
            <div class="section-sub">
                Transparent shock analysis using the original scenario engine.
            </div>
        </div>
        <div class="micro">WHAT-IF / TRANSMISSION / DELTA</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="scenario-hero">
        <div class="scenario-label">ACTIVE SHOCK</div>
        <div class="scenario-title">
            POLICY_RATE_YOY_CHANGE_BPS
            <span style="color:{COLORS['violet']};"> {fmt_delta(shock,0)} bps</span>
        </div>
        <div class="card-caption" style="margin-top:7px;">
            Baseline: {esc(country_label)} · {int(year)} · score {fmt_number(score_value,1)}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if scenario_error is not None:
    st.warning(f"Scenario engine returned an error: {scenario_error}")

if scenario is not None:
    sc1, sc2, sc3 = st.columns(3)

    # Display-only extraction: the underlying scenario object is untouched.
    scenario_baseline = None
    scenario_value = None
    scenario_delta = None

    if isinstance(scenario, dict):
        for key in ["baseline", "baseline_score", "base_score", "score_baseline"]:
            if key in scenario:
                scenario_baseline = safe_float(scenario[key])
                break

        for key in ["scenario", "scenario_score", "shocked_score", "new_score"]:
            if key in scenario:
                scenario_value = safe_float(scenario[key])
                break

        for key in ["delta", "score_delta", "change"]:
            if key in scenario:
                scenario_delta = safe_float(scenario[key])
                break

    elif isinstance(scenario, pd.DataFrame) and not scenario.empty:
        columns = {str(c).lower(): c for c in scenario.columns}

        for key in ["baseline", "baseline_score", "base_score"]:
            if key in columns:
                scenario_baseline = safe_float(scenario.iloc[0][columns[key]])
                break

        for key in ["scenario", "scenario_score", "shocked_score", "new_score"]:
            if key in columns:
                scenario_value = safe_float(scenario.iloc[0][columns[key]])
                break

        for key in ["delta", "score_delta", "change"]:
            if key in columns:
                scenario_delta = safe_float(scenario.iloc[0][columns[key]])
                break

    if scenario_baseline is None:
        scenario_baseline = score_value

    if scenario_value is None and scenario_delta is not None:
        scenario_value = scenario_baseline + scenario_delta

    if scenario_value is None:
        scenario_value = score_value

    if scenario_delta is None:
        scenario_delta = scenario_value - scenario_baseline

    with sc1:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">BASELINE</div>
                <div class="card-value">{fmt_number(scenario_baseline,1)}</div>
                <div class="card-caption">existing risk score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with sc2:
        scenario_band = score_band(scenario_value)
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">SCENARIO SCORE</div>
                <div class="card-value" style="color:{band_color(scenario_band)};">
                    {fmt_number(scenario_value,1)}
                </div>
                <div class="card-caption">{esc(scenario_band)} band after shock</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with sc3:
        delta_class = (
            "delta-positive"
            if scenario_delta > 0
            else "delta-negative"
            if scenario_delta < 0
            else "delta-neutral"
        )

        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">RISK DELTA</div>
                <div class="card-value {delta_class}">
                    {fmt_delta(scenario_delta,1)}
                </div>
                <div class="card-caption">scenario minus baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Scenario target indicators.
    st.markdown("<br>", unsafe_allow_html=True)

    target_codes = [
        "FX_YOY_DEPRECIATION_PCT",
        "NY.GDP.MKTP.KD.ZG",
        "GC.DOD.TOTL.GD.ZS",
    ]

    target_cols = st.columns(3)

    for idx, code in enumerate(target_codes):
        with target_cols[idx]:
            baseline_target = None
            if not current_row.empty and code in current_row.index:
                baseline_target = current_row[code]

            scenario_target = None

            if isinstance(scenario, dict):
                target_container = scenario.get("targets", scenario)
                if isinstance(target_container, dict):
                    if code in target_container:
                        raw = target_container[code]
                        if isinstance(raw, dict):
                            for key in ["scenario", "value", "new", "shocked"]:
                                if key in raw:
                                    scenario_target = raw[key]
                                    break
                        else:
                            scenario_target = raw

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-label">{esc(code)}</div>
                    <div style="margin-top:12px;color:#7e8a9b;font-size:10px;">BASELINE</div>
                    <div style="font-family:'DM Mono';font-size:15px;color:#e8eef6;">
                        {esc(fmt_number(baseline_target,2) if baseline_target is not None else "—")}
                    </div>
                    <div style="margin-top:9px;color:#7e8a9b;font-size:10px;">SCENARIO</div>
                    <div style="font-family:'DM Mono';font-size:15px;color:{COLORS['violet']};">
                        {esc(fmt_number(scenario_target,2) if scenario_target is not None else "engine output")}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <div class="card-label">TRANSMISSION CHANNELS</div>
            <div style="margin-top:13px;color:#bdc8d6;font-size:12px;line-height:1.75;">
                The scenario is passed through the existing engine targets:
                FX depreciation, real GDP growth and government debt. The UI
                does not substitute its own economic model; it only visualizes
                the returned scenario output.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    empty_state(
        "Set a non-zero policy-rate shock to explore the scenario engine output."
    )


# ============================================================================
# ANALYST INTELLIGENCE
# ============================================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">Analyst intelligence</div>
            <div class="section-sub">
                Deterministic commentary generated from the analytical outputs.
            </div>
        </div>
        <div class="micro">TRACEABLE / NON-GENERATIVE CORE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="intel">
        <div class="intel-head">
            <div class="intel-icon">🧠</div>
            <div>
                <div class="intel-title">Analyst view</div>
                <div class="intel-sub">
                    {esc(country_label)} · {int(year)} · {esc(band)} risk regime
                </div>
            </div>
        </div>
        <div class="intel-body">
            {esc(report)}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# DATA COVERAGE / METADATA
# ============================================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">Data coverage & metadata</div>
            <div class="section-sub">
                Operational context for the selected analytical slice.
            </div>
        </div>
        <div class="micro">AUDIT TRAIL</div>
    </div>
    """,
    unsafe_allow_html=True,
)

metadata = [
    ("COUNTRY", country_label),
    ("ISO3", iso),
    ("YEAR", year),
    ("PANEL ROWS", f"{len(panel):,}"),
    ("SCORE BAND", band),
    ("DATA COVERAGE", "—" if pd.isna(coverage_value) else f"{coverage_value:.1f}%"),
    ("SHOCK", f"{shock:+.0f} bps"),
    ("UI REFRESHED", generated_at),
]

meta_html = '<div class="card"><div class="metadata">'

for key, value in metadata:
    meta_html += f"""
        <div class="meta-item">
            <div class="meta-k">{esc(key)}</div>
            <div class="meta-v">{esc(value)}</div>
        </div>
    """

meta_html += "</div></div>"

st.markdown(meta_html, unsafe_allow_html=True)


# ============================================================================
# RAW ANALYTICAL DATA / EXPORTS
# ============================================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">Export & inspection</div>
            <div class="section-sub">
                Download the underlying dashboard-ready panel without altering it.
            </div>
        </div>
        <div class="micro">CSV / RAW PANEL</div>
    </div>
    """,
    unsafe_allow_html=True,
)

export_left, export_right = st.columns([1, 2], gap="large")

with export_left:
    csv_bytes = panel.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download panel CSV",
        data=csv_bytes,
        file_name=f"country_risk_panel_{int(year)}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "Exports the currently loaded dashboard panel exactly as provided to the UI."
    )

with export_right:
    with st.expander("Inspect selected country-year row"):
        if current_row.empty:
            empty_state("No matching country-year row found.")
        else:
            st.dataframe(
                current_row.to_frame("value"),
                use_container_width=True,
            )

    with st.expander("Inspect top driver output"):
        if isinstance(country_drivers, pd.DataFrame) and not country_drivers.empty:
            st.dataframe(
                country_drivers,
                use_container_width=True,
                hide_index=True,
            )
        else:
            empty_state("No driver table available.")


# ============================================================================
# METHODOLOGY NOTE
# ============================================================================

with st.expander("Methodology & engine integrity"):
    st.markdown(
        """
### Engine integrity

This interface is a presentation layer around the existing Country Risk
Intelligence Engine.

**Preserved analytical functions**

- `score_panel(panel)`
- `top_drivers(drivers, country, year, n=6)`
- `run_shock_scenario(...)`
- `generate_report(...)`

The dashboard does not recalculate the risk methodology in JavaScript or
replace the scoring engine with UI heuristics. Visualization helpers only
format returned analytical results.

**Scenario targets preserved**

- `FX_YOY_DEPRECIATION_PCT`
- `NY.GDP.MKTP.KD.ZG`
- `GC.DOD.TOTL.GD.ZS`

**Primary scenario driver preserved**

- `POLICY_RATE_YOY_CHANGE_BPS`

The intent is to keep the analytical core auditable while making the decision
surface substantially more usable for an analyst, hiring manager, or technical
reviewer.
        """
    )


# ============================================================================
# FOOTER
# ============================================================================

st.markdown(
    f"""
    <div class="footer">
        <span>COUNTRY RISK INTELLIGENCE ENGINE</span>
        <span>{esc(iso)} / {int(year)} · ANALYTICAL CORE INTACT</span>
        <span>UI BUILD · {generated_at}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# END
# ============================================================================
