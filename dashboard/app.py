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

CONFIG_DIR = ROOT / "config"
PROCESSED_DIR = ROOT / "data" / "processed"
PANEL_PATH = PROCESSED_DIR / "panel_wide.csv"
DEMO_PANEL_PATH = ROOT / "data" / "demo" / "panel_wide.csv"

from src.commentary.generate_commentary import generate_report
from src.scenario.scenario_engine import run_shock_scenario
from src.scoring.risk_score import score_panel, top_drivers
from src.analysis.backtest import run_backtest

# Runtime live-data provider (fetches World Bank/FRED at runtime — see
# src/runtime/live_data.py). Kept separate from the analytical core above:
# this only decides WHERE the panel comes from, never how it's scored.
from src.runtime import data_state
from src.runtime.live_data import LiveDataUnavailable, fetch_live_panel


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
import re
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# Streamlit's Markdown parser treats indented HTML as a code block. The
# dashboard intentionally uses readable, indented HTML templates. Keep this
# compatibility wrapper idempotent: Streamlit reruns the script in the same
# process, so wrapping st.markdown repeatedly can create recursive renderers
# and unreliable refreshes.
if not getattr(st.markdown, "_cri_html_normalized", False):
    _streamlit_markdown = st.markdown

    def _dedented_markdown(body, *args, **kwargs):
        if isinstance(body, str) and kwargs.get("unsafe_allow_html"):
            body = textwrap.dedent(body)
            body = "\n".join(line.lstrip() for line in body.splitlines())
        return _streamlit_markdown(body, *args, **kwargs)

    _dedented_markdown._cri_html_normalized = True
    st.markdown = _dedented_markdown


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
    overflow-x: hidden;
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
    animation: gridDrift 28s linear infinite;
}

.block-container {
    max-width: 1600px;
    padding-top: 2rem;
    padding-bottom: 4rem;
    overflow-x: hidden;
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
}

.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible,
[data-baseweb="select"]:focus-within,
input:focus-visible {
    outline: 2px solid rgba(94,231,242,.85);
    outline-offset: 2px;
}

.stAlert {
    border-radius: 14px;
    border: 1px solid rgba(94,231,242,.18);
    background: linear-gradient(135deg, rgba(21,46,72,.9), rgba(12,25,43,.86));
    animation: alertIn .55s ease both;
}

.stAlert p {
    color: #b7d9ee;
}

.stCaption {
    color: var(--muted);
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
    box-sizing: border-box;
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
    grid-template-columns: minmax(0, 1.35fr) minmax(0, .65fr);
    gap: 30px;
    align-items: center;
    position: relative;
    z-index: 1;
}

.hero-grid > * {
    min-width: 0;
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
    max-width: 100%;
    overflow-wrap: normal;
}

.hero h1 span {
    background: linear-gradient(100deg, #fff, #bceff3 42%, #a78bfa 90%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero-copy {
    max-width: 740px;
    width: 100%;
    margin-top: 18px;
    color: var(--muted);
    font-size: 15px;
    line-height: 1.7;
    overflow-wrap: anywhere;
}

.hero-terminal {
    position: relative;
    isolation: isolate;
    justify-self: end;
    width: 100%;
    max-width: 390px;
    min-width: 0;
    overflow: hidden;
    padding: 18px;
    border: 1px solid var(--border);
    border-radius: 15px;
    background: rgba(0,0,0,.28);
    font-family: "DM Mono", monospace;
    font-size: 12px;
    color: #a9b6c7;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
    animation: terminalIn .8s .18s ease both;
}

.hero-terminal:after {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    z-index: -1;
    background: linear-gradient(
        to bottom,
        transparent 0%,
        rgba(94,231,242,.08) 48%,
        transparent 52%,
        transparent 100%
    );
    transform: translateY(-100%);
    animation: scan 5.5s 1.2s ease-in-out infinite;
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
    overflow-wrap: anywhere;
    animation: terminalLineIn .5s ease both;
}

.terminal-line:nth-child(2) { animation-delay: .16s; }
.terminal-line:nth-child(3) { animation-delay: .28s; }
.terminal-line:nth-child(4) { animation-delay: .40s; }
.terminal-line:nth-child(5) { animation-delay: .52s; }
.terminal-line:nth-child(6) { animation-delay: .64s; }
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
    animation: sectionIn .55s ease both;
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
    animation: cardIn .65s ease both;
}

.card:hover {
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
    animation: scorePulse 4.5s ease-in-out infinite;
}

.js-plotly-plot,
.plotly-graph-div {
    max-width: 100%;
    overflow: hidden;
    transform: none !important;
    animation: none !important;
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

.peer-position-row {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 14px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
}

.peer-position-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 700;
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
    transform: scaleX(0);
    transform-origin: left center;
    animation: barGrow 1.05s .18s cubic-bezier(.2,.75,.2,1) forwards;
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
    animation: signalIn .55s ease both;
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
    transform: scaleX(0);
    transform-origin: left center;
    animation: barGrow .9s .2s cubic-bezier(.2,.75,.2,1) forwards;
}

.peer-highlight {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid rgba(148,163,184,.08);
    animation: signalIn .55s ease both;
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
    animation: scenarioPulse 5s ease-in-out infinite;
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

@keyframes gridDrift {
    from { background-position: 0 0, 0 0; }
    to { background-position: 42px 42px, -42px 42px; }
}

@keyframes sectionIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes cardIn {
    from { opacity: 0; transform: translateY(14px) scale(.985); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

@keyframes terminalIn {
    from { opacity: 0; transform: translateX(18px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes terminalLineIn {
    from { opacity: 0; transform: translateX(8px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes scan {
    0%, 18% { transform: translateY(-100%); opacity: 0; }
    28% { opacity: 1; }
    68% { opacity: .7; }
    82%, 100% { transform: translateY(100%); opacity: 0; }
}

@keyframes barGrow {
    from { transform: scaleX(0); }
    to { transform: scaleX(1); }
}

@keyframes signalIn {
    from { opacity: 0; transform: translateY(7px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes scorePulse {
    0%, 100% { filter: drop-shadow(0 0 0 rgba(94,231,242,0)); }
    50% { filter: drop-shadow(0 0 12px color-mix(in srgb, var(--score-color) 24%, transparent)); }
}

@keyframes scenarioPulse {
    0%, 100% { box-shadow: 0 0 0 rgba(167,139,250,0); }
    50% { box-shadow: 0 0 28px rgba(167,139,250,.08); }
}

@keyframes alertIn {
    from { opacity: 0; transform: translateY(-8px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 900px) {
    .hero-grid {
        grid-template-columns: 1fr;
    }
    .hero-terminal {
        justify-self: start;
        max-width: none;
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
    .section-head {
        display: block;
        margin-top: 27px;
    }
    .section-head .micro {
        margin-top: 8px;
    }
    .hero {
        padding: 25px 22px;
        border-radius: 18px;
    }
    .hero h1 {
        font-size: clamp(30px, 8.8vw, 38px);
        line-height: 1;
        word-break: normal;
        overflow-wrap: normal;
    }
    .hero-copy {
        font-size: 14px;
    }
    .score-ring {
        width: 190px;
        height: 190px;
    }
    .score-number {
        font-size: 44px;
    }
    .stButton > button,
    .stDownloadButton > button {
        min-height: 44px;
    }
    .metadata {
        grid-template-columns: 1fr;
    }
    .signal-grid {
        grid-template-columns: 1fr;
    }
    .peer-position-row {
        flex-wrap: wrap;
        gap: 14px !important;
    }
    .peer-position-row > div {
        min-width: 30%;
    }
    .peer-position-value {
        font-size: 18px !important;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: .01ms !important;
    }
}
/* ============================================================================
   NEW — ambient background, self-drawing score ring, staggered reveals.
   Purely additive: nothing above is changed. Everything here degrades
   gracefully — a browser without @property support just shows the
   final/static state immediately instead of animating into it, never blank.
   ============================================================================ */

.aurora {
    position: fixed;
    inset: 0;
    z-index: -1;
    overflow: hidden;
    pointer-events: none;
}
.aurora span {
    position: absolute;
    width: 46vw;
    height: 46vw;
    border-radius: 50%;
    filter: blur(90px);
    opacity: .16;
    animation: auroraDrift 22s ease-in-out infinite alternate;
}
.aurora span:nth-child(1) { background: var(--cyan);   top: -14%; left: -10%; }
.aurora span:nth-child(2) { background: var(--violet); bottom: -18%; right: -8%; animation-duration: 26s; animation-delay: -6s; }
.aurora span:nth-child(3) { background: var(--blue);   top: 30%; left: 60%; width: 34vw; height: 34vw; animation-duration: 30s; animation-delay: -12s; }

@keyframes auroraDrift {
    from { transform: translate(0, 0) scale(1); }
    to   { transform: translate(4%, 6%) scale(1.12); }
}

@property --score-pct {
    syntax: '<number>';
    inherits: true;
    initial-value: 0;
}
.score-ring {
    --score-pct: var(--score-pct-target, 0);
    animation: scorePulse 4.5s ease-in-out infinite, scoreFillIn 1.3s .1s cubic-bezier(.16,1,.3,1) both;
}
@keyframes scoreFillIn {
    from { --score-pct: 0; }
    to   { --score-pct: var(--score-pct-target, 0); }
}

.card:hover { transform: translateY(-3px); }

.driver-row { animation: signalIn .5s ease both; }
.driver-row:nth-child(1) { animation-delay: .04s; }
.driver-row:nth-child(2) { animation-delay: .10s; }
.driver-row:nth-child(3) { animation-delay: .16s; }
.driver-row:nth-child(4) { animation-delay: .22s; }
.driver-row:nth-child(5) { animation-delay: .28s; }
.driver-row:nth-child(6) { animation-delay: .34s; }
.driver-row:nth-child(7) { animation-delay: .40s; }
.driver-row:nth-child(8) { animation-delay: .46s; }

.signal:nth-child(1) { animation-delay: .04s; }
.signal:nth-child(2) { animation-delay: .09s; }
.signal:nth-child(3) { animation-delay: .14s; }
.signal:nth-child(4) { animation-delay: .19s; }
.signal:nth-child(5) { animation-delay: .24s; }
.signal:nth-child(6) { animation-delay: .29s; }
.signal:nth-child(7) { animation-delay: .34s; }
.signal:nth-child(8) { animation-delay: .39s; }

.peer-highlight:nth-child(1) { animation-delay: .03s; }
.peer-highlight:nth-child(2) { animation-delay: .07s; }
.peer-highlight:nth-child(3) { animation-delay: .11s; }
.peer-highlight:nth-child(4) { animation-delay: .15s; }
.peer-highlight:nth-child(5) { animation-delay: .19s; }
.peer-highlight:nth-child(6) { animation-delay: .23s; }
.peer-highlight:nth-child(7) { animation-delay: .27s; }

.meta-item { animation: cardIn .5s ease both; }
.meta-item:nth-child(1) { animation-delay: .03s; }
.meta-item:nth-child(2) { animation-delay: .07s; }
.meta-item:nth-child(3) { animation-delay: .11s; }
.meta-item:nth-child(4) { animation-delay: .15s; }
.meta-item:nth-child(5) { animation-delay: .19s; }
.meta-item:nth-child(6) { animation-delay: .23s; }
.meta-item:nth-child(7) { animation-delay: .27s; }
.meta-item:nth-child(8) { animation-delay: .31s; }

/* Formatted analyst commentary (see markdown_to_html() in Python) */
.intel-body p { margin: 0 0 12px; }
.intel-body p:last-child { margin-bottom: 0; }
.intel-body strong { color: #f4f7fb; font-weight: 700; }
.intel-body ul { margin: 4px 0 14px; padding-left: 18px; }
.intel-body li { margin-bottom: 6px; animation: signalIn .45s ease both; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="aurora"><span></span><span></span><span></span></div>',
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
def load_panel(panel_path: Path):
    return pd.read_csv(panel_path)


countries_cfg, indicators_cfg = load_configurations()
peer_groups = (
    countries_cfg.get("peer_groups", {})
    if isinstance(countries_cfg, dict)
    else {}
)

# ============================================================================
# LIVE / DEMO DATA STATE MACHINE
#
# The public app fetches live World Bank/FRED data itself — nobody should
# need to run a Python command before anything appears. A live failure
# shows a clean status screen with one explicit way forward (Open Demo
# Dataset); it never silently substitutes synthetic data for real data.
# ============================================================================

COUNTRY_ISO3_LIST = tuple(
    c["iso3"] for c in countries_cfg.get("countries", []) if c.get("iso3")
) if isinstance(countries_cfg, dict) else tuple()

LIVE_START_YEAR = 2012
LIVE_END_YEAR = datetime.now().year


def _config_version() -> str:
    """Short hash of the config files so a cached live fetch invalidates
    itself if you edit indicators.yaml/countries.yaml, without needing a
    manual cache clear."""
    import hashlib

    payload = (CONFIG_DIR / "indicators.yaml").read_bytes() + (CONFIG_DIR / "countries.yaml").read_bytes()
    return hashlib.sha256(payload).hexdigest()[:10]


@st.cache_data(ttl=data_state.LIVE_CACHE_TTL_SECONDS, show_spinner=False)
def _cached_fetch_live(iso3_codes: tuple, start_year: int, end_year: int, config_version: str):
    return fetch_live_panel(list(iso3_codes), indicators_cfg, start_year, end_year, config_version)


if "data_mode" not in st.session_state:
    st.session_state.data_mode = None  # unset -> attempt live below

live_result = None
live_error = None

if st.session_state.data_mode != data_state.DEMO:
    try:
        with st.spinner("Connecting to World Bank..."):
            live_result = _cached_fetch_live(
                COUNTRY_ISO3_LIST, LIVE_START_YEAR, LIVE_END_YEAR, _config_version()
            )
        st.session_state.data_mode = data_state.LIVE
    except LiveDataUnavailable as exc:
        live_error = exc
        st.session_state.data_mode = data_state.UNAVAILABLE

if st.session_state.data_mode == data_state.UNAVAILABLE:
    st.markdown(
        f"""
        <div class="card" style="margin-top:24px;">
            <div class="card-label">LIVE DATA UNAVAILABLE</div>
            <div class="card-value" style="font-size:22px;">
                Official public data could not be verified right now.
            </div>
            <div class="card-caption" style="margin-top:10px;">
                Source: World Bank Indicators API &middot; Status: Unavailable
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if live_error is not None:
        st.caption(str(live_error))
        if live_error.technical_detail:
            with st.expander("Technical details"):
                st.code(live_error.technical_detail)
    if st.button("Open Demo Dataset"):
        st.session_state.data_mode = data_state.DEMO
        st.rerun()
    st.caption("Reloading this page will automatically retry live data.")
    st.stop()

USING_DEMO_DATA = st.session_state.data_mode == data_state.DEMO
live_provenance = None

if USING_DEMO_DATA:
    if not DEMO_PANEL_PATH.exists():
        st.error("Demo dataset file is missing (data/demo/panel_wide.csv).")
        st.stop()
    panel = load_panel(DEMO_PANEL_PATH)
    st.error(
        "DEMO DATA — SYNTHETIC DATASET. For interface / methodology "
        "demonstration only — these are not real economic observations."
    )
else:
    panel = live_result.wide_panel
    live_provenance = live_result.provenance


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


def markdown_to_html(text: str) -> str:
    """
    BUG FIX: generate_report() returns Markdown (**bold** headers, "- " bullet
    lists, blank-line-separated paragraphs). The old code did `esc(report)`
    straight into a raw <div>, which escapes HTML but does nothing to
    Markdown syntax and collapses all the newlines — so the Analyst
    Intelligence card showed one flattened line with literal ** and - marks
    instead of headings and a bullet list. This is a tiny, deliberately
    narrow converter for exactly generate_report()'s output shape — not a
    general Markdown engine.
    """
    lines = str(text).split("\n")
    html_parts = []
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            html_parts.append("</ul>")
            in_list = False

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            close_list()
            continue

        if line.startswith("- "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{esc(line[2:])}</li>")
            continue

        close_list()

        # "**Bold header:**" (with or without trailing text) -> its own line.
        # `line` is escaped FIRST, then bolded — group(1) is already-escaped
        # text at this point, so it must NOT be escaped again here.
        line_html = re.sub(r"\*\*(.+?)\*\*", lambda m: f"<strong>{m.group(1)}</strong>", esc(line))
        html_parts.append(f"<p>{line_html}</p>")

    close_list()
    return "\n".join(html_parts)


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


def country_records():
    if isinstance(countries_cfg, dict):
        records = countries_cfg.get("countries", [])
        if isinstance(records, list):
            return [r for r in records if isinstance(r, dict)]
    return []


def country_lookup():
    return {
        str(record.get("iso3")): record
        for record in country_records()
        if record.get("iso3")
    }


def get_iso(country):
    return str(country)


def get_country_label(country):
    record = country_lookup().get(str(country))
    if record:
        return str(record.get("name") or country)
    return str(country)


def available_countries(df):
    if "country_iso3" in df.columns:
        return sorted(df["country_iso3"].dropna().astype(str).unique().tolist())

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
    if "country_iso3" in df.columns:
        mask = df["country_iso3"].astype(str).eq(str(country))
    elif "country" in df.columns:
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
    for column in ["country_iso3", "country", "country_name", "iso3", "ISO3"]:
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
        title=dict(text=""),  # BUG FIX: an unset title rendered as literal "undefined" text
        uirevision="country-risk-intelligence",
        dragmode=False,  # PHASE 32: no box/drag zoom — hover stays on, dragging the chart doesn't
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans, sans-serif",
            color=COLORS["muted"],
            size=11,
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
        fixedrange=True,  # PHASE 32: axis can't be zoomed/panned by dragging
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,.055)",
        zeroline=False,
        linecolor="rgba(148,163,184,.08)",
        tickfont=dict(color=COLORS["faint"], size=10),
        fixedrange=True,  # PHASE 32: axis can't be zoomed/panned by dragging
    )
    return fig


def empty_state(message):
    st.markdown(
        f'<div class="empty">{esc(message)}</div>',
        unsafe_allow_html=True,
    )


def user_error(message: str, exc: Exception | None = None):
    """
    PHASE 35: a clean, human-readable warning — never a raw Python
    traceback as the primary message. The actual exception, if any, goes
    behind a collapsed "Technical details" expander for debugging.
    """
    st.warning(message)
    if exc is not None:
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")


# ============================================================================
# SIDEBAR — ORIGINAL CONTROLS + REFRESH / EXPORT
# ============================================================================

with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:4px 4px 16px;">
            <div class="kicker">RISK ENGINE / {('DEMO PANEL' if USING_DEMO_DATA else 'LIVE PANEL')}</div>
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
        format_func=get_country_label,
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
        if USING_DEMO_DATA:
            if st.button("Return to Live Data", width="stretch"):
                st.session_state.data_mode = None
                st.rerun()
        else:
            if st.button("↻ Refresh Live Data", width="stretch"):
                _cached_fetch_live.clear()
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
#
# The calls below intentionally mirror the original dashboard contract:
#
#   scores, drivers = score_panel(panel)
#   top_drivers(drivers, country, year, n=6)
#   run_shock_scenario(panel, ...)
#   generate_report(...)
#
# The UI never substitutes a second scoring methodology.
# ============================================================================

scores, drivers = score_panel(panel)

if "country_iso3" not in scores.columns or "year" not in scores.columns:
    st.error("Scoring output is missing country_iso3/year columns.")
    st.stop()

row = scores[
    scores["country_iso3"].astype(str).eq(str(country))
    & pd.to_numeric(scores["year"], errors="coerce").eq(int(year))
]

if row.empty or pd.isna(row.iloc[0]["risk_score"]):
    st.error(
        f"No sufficient indicator data to score "
        f"{get_country_label(country)} in {year}."
    )
    st.stop()

score_value = safe_float(row.iloc[0]["risk_score"])
band = str(row.iloc[0]["risk_band"])
score_color = band_color(band)
coverage_value = safe_float(row.iloc[0]["data_completeness"], default=float("nan"))

try:
    current_row = row_for(panel, country, year)
except Exception:
    current_row = pd.Series(dtype=object)

try:
    country_drivers = top_drivers(drivers, country, year, n=6)
except Exception as exc:
    country_drivers = pd.DataFrame()
    user_error("Driver decomposition is temporarily unavailable for this slice.", exc)

driver_code = "POLICY_RATE_YOY_CHANGE_BPS"
shock_amount = float(shock)
run_scenario_btn = not np.isclose(shock_amount, 0.0)
scenario_result = None
scenario = None
scenario_error = None

if run_scenario_btn:
    try:
        scenario_result = run_shock_scenario(
            panel,
            country,
            year,
            driver_code,
            shock_amount,
            [
                "FX_YOY_DEPRECIATION_PCT",
                "NY.GDP.MKTP.KD.ZG",
                "GC.DOD.TOTL.GD.ZS",
            ],
        )
        scenario = scenario_result
    except Exception as exc:
        scenario_error = exc

peer_group = next(
    (members for members in peer_groups.values() if country in members),
    None,
)

report = generate_report(
    country_name=get_country_label(country),
    country_iso3=country,
    year=int(year),
    scores=scores,
    drivers=drivers,
    scenario_result=scenario_result,
    peer_group=[c for c in (peer_group or []) if c != country],
)

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
# DATA STATUS / PROVENANCE
#
# "Where did this number come from?" answered without opening source code
# — source, retrieval time, latest underlying observation, coverage.
# ============================================================================

if USING_DEMO_DATA:
    st.markdown(
        """
        <div class="card" style="margin-top:18px;">
            <div class="card-label">DEMO DATA</div>
            <div class="card-value" style="font-size:18px;">Synthetic dataset</div>
            <div class="card-caption" style="margin-top:6px;">
                For interface / methodology demonstration only.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    prov = live_provenance
    source_names = " · ".join(s["name"] for s in prov.sources) if prov else "—"
    st.markdown(
        f"""
        <div class="card" style="margin-top:18px;">
            <div class="card-label">LIVE PUBLIC DATA</div>
            <div class="card-value" style="font-size:18px;">{esc(source_names)}</div>
            <div class="card-caption" style="margin-top:6px;">
                Retrieved: {esc(prov.retrieved_at)} &nbsp;·&nbsp;
                Latest common analysis year: <strong>{prov.latest_observation_year}</strong>
                &nbsp;·&nbsp; Coverage: {prov.coverage_pct:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if prov and prov.validation_failures:
        with st.expander("Data quality notes"):
            for note in prov.validation_failures:
                st.caption(f"• {note}")
    st.caption(
        f"Retrieval timestamp above reflects when this data was fetched — not when the "
        f"underlying {prov.latest_observation_year if prov else ''} figures were measured. "
        "Annual macro data is typically published with a lag."
    )

with st.expander("How to use this", expanded=False):
    st.markdown(
        """
1. Pick a **country** and **year** in the sidebar.
2. Read the **risk score** and **"What does this mean?"** panel — that's the whole story in one screen.
3. Check **drivers** to see which indicators are pushing risk up vs pulling it down.
4. Scroll to **Deterioration watch** to see what's moving across the whole panel, not just your selection.
5. Open **Model validation** to see whether this scoring approach would actually have caught real past crises.
6. Try the **Scenario Lab** to stress-test a policy-rate shock.
7. Open **Methodology** if you want the actual math, or **Export & inspection** for the raw data.
        """
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
                    (scores["country_iso3"].astype(str).eq(str(country)))
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
# "WHAT DOES THIS MEAN?" — dynamic interpretation, generated only from
# computed outputs (never fabricated) — the reader shouldn't have to parse
# raw indicator codes or contribution signs to understand the score.
# ============================================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">What does this mean?</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

_slice = drivers[
    (drivers["country_iso3"].astype(str) == str(country))
    & (pd.to_numeric(drivers["year"], errors="coerce") == int(year))
].copy() if isinstance(drivers, pd.DataFrame) and not drivers.empty else pd.DataFrame()

if not _slice.empty and "weighted_contribution" in _slice.columns:
    _slice = _slice.dropna(subset=["weighted_contribution"])
    _higher = _slice[_slice["weighted_contribution"] > 0].sort_values(
        "weighted_contribution", ascending=False
    )
    _lower = _slice[_slice["weighted_contribution"] < 0].sort_values(
        "weighted_contribution", ascending=True
    )
    _label_col = "label" if "label" in _slice.columns else (
        "indicator_code" if "indicator_code" in _slice.columns else _slice.columns[0]
    )

    _up_names = [str(r[_label_col]) for _, r in _higher.head(3).iterrows()]
    _down_names = [str(r[_label_col]) for _, r in _lower.head(2).iterrows()]

    _country_label = get_country_label(country)

    _sentence = (
        f"**{esc(_country_label)}** is currently positioned in the **{esc(band.lower())}** "
        f"relative-risk band within the selected comparison panel."
    )
    st.markdown(f'<div class="card"><div class="card-value" style="font-size:16px;font-weight:500;line-height:1.5;">{_sentence}</div>', unsafe_allow_html=True)

    if _up_names:
        st.markdown(
            '<div class="card-caption" style="margin-top:12px;">The strongest upward risk signals are:</div>'
            + "".join(f'<div class="card-caption">&nbsp;&nbsp;↑ {esc(n)}</div>' for n in _up_names),
            unsafe_allow_html=True,
        )
    if _down_names:
        st.markdown(
            '<div class="card-caption" style="margin-top:10px;">Mitigating signals include:</div>'
            + "".join(f'<div class="card-caption">&nbsp;&nbsp;↓ {esc(n)}</div>' for n in _down_names),
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="card-caption" style="margin-top:12px;font-style:italic;">'
        "This score is relative to the available comparison panel — it is not a credit rating "
        "or a probability of default.</div></div>",
        unsafe_allow_html=True,
    )
else:
    empty_state("Not enough driver data to generate an interpretation for this slice.")


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
                     style="--score-pct-target:{score_pct(score_value)};--score-color:{score_color};">
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
        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": False,
                "responsive": True,
                "scrollZoom": False,
                "doubleClick": False,
                "showAxisDragHandles": False,
                "modeBarButtonsToRemove": [
                    "zoom2d", "pan2d", "select2d", "lasso2d",
                    "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
                ],
            },
        )


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

driver_left, driver_right = st.columns([1, 1], gap="large")

_full_slice = drivers[
    (drivers["country_iso3"].astype(str) == str(country))
    & (pd.to_numeric(drivers["year"], errors="coerce") == int(year))
].copy() if isinstance(drivers, pd.DataFrame) and not drivers.empty else pd.DataFrame()

if not _full_slice.empty and "weighted_contribution" in _full_slice.columns:
    _full_slice = _full_slice.dropna(subset=["weighted_contribution"])
    _label_col = "label" if "label" in _full_slice.columns else "indicator_code"
    _has_raw = "raw_value" in _full_slice.columns
    _has_unit = "unit" in _full_slice.columns

    def _render_signal_group(df_group: pd.DataFrame, arrow: str, pts_prefix: str):
        for _, r in df_group.iterrows():
            name = str(r[_label_col])
            pts = safe_float(r["weighted_contribution"]) * 100  # display as "points" on the 0-100 score scale
            if _has_raw and pd.notna(r.get("raw_value")):
                unit = str(r.get("unit", "")).strip() if _has_unit else ""
                value_text = f"{fmt_number(safe_float(r['raw_value']), 1)}{(' ' + unit) if unit else ''}"
            else:
                value_text = "—"
            width = min(100, abs(pts) / max(1e-9, _magnitude_ref) * 100)
            st.markdown(
                f"""
                <div class="driver-row">
                    <div>
                        <div class="driver-meta">
                            <span class="driver-name">{arrow} {esc(name)}</span>
                            <span>{esc(value_text)}</span>
                        </div>
                        <div class="driver-bar">
                            <div class="driver-fill" style="width:{width:.1f}%"></div>
                        </div>
                    </div>
                    <div class="driver-score">{pts_prefix}{abs(pts):.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    _higher_full = _full_slice[_full_slice["weighted_contribution"] > 0].sort_values(
        "weighted_contribution", ascending=False
    )
    _lower_full = _full_slice[_full_slice["weighted_contribution"] < 0].sort_values(
        "weighted_contribution", ascending=True
    )
    _magnitude_ref = max(
        _full_slice["weighted_contribution"].abs().max() * 100, 1e-9
    )
else:
    _higher_full = pd.DataFrame()
    _lower_full = pd.DataFrame()

with driver_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label" style="color:var(--red,#ff6b7a);">↑ HIGHER-RISK SIGNALS</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption" style="margin-bottom:10px;">Pushing the score up, largest first.</div>', unsafe_allow_html=True)
    if not _higher_full.empty:
        st.markdown('<div class="driver-list">', unsafe_allow_html=True)
        _render_signal_group(_higher_full.head(6), "↑", "+")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        empty_state("No indicators are currently adding to relative risk.")
    st.markdown("</div>", unsafe_allow_html=True)

with driver_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label" style="color:var(--green,#54d69a);">↓ MITIGATING SIGNALS</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption" style="margin-bottom:10px;">Pulling the score down, largest first.</div>', unsafe_allow_html=True)
    if not _lower_full.empty:
        st.markdown('<div class="driver-list">', unsafe_allow_html=True)
        _render_signal_group(_lower_full.head(6), "↓", "-")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        empty_state("No indicators are currently reducing relative risk.")
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Technical details — indicator codes, weights, sources"):
    if not _full_slice.empty:
        _tech_cols = [c for c in ["indicator_code", _label_col, "category", "raw_value", "unit", "weight", "risk_direction", "z_risk", "weighted_contribution"] if c in _full_slice.columns]
        st.dataframe(_full_slice[_tech_cols].reset_index(drop=True), width="stretch", hide_index=True)
    else:
        st.caption("No driver data available for this technical view.")


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
            width="stretch",
            config={
                "displayModeBar": False,
                "responsive": True,
                "scrollZoom": False,
                "doubleClick": False,
                "showAxisDragHandles": False,
                "modeBarButtonsToRemove": [
                    "zoom2d", "pan2d", "select2d", "lasso2d",
                    "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
                ],
            },
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

            if not peers.empty:
                _median = float(peers[score_column].median())
                _rank = int((peers[score_column] > score_value).sum()) + 1
                _total = len(peers)
                st.markdown(
                    f"""
                    <div class="peer-position-row">
                        <div>
                            <div class="micro">SELECTED</div>
                            <div class="peer-position-value">{fmt_number(score_value,1)}</div>
                        </div>
                        <div>
                            <div class="micro">PEER MEDIAN</div>
                            <div class="peer-position-value">{fmt_number(_median,1)}</div>
                        </div>
                        <div>
                            <div class="micro">POSITION</div>
                            <div class="peer-position-value">{_rank} / {_total}</div>
                        </div>
                    </div>
                    <div class="card-caption" style="margin-bottom:10px;">
                        Relative position within the selected comparison set — not an absolute universal ranking.
                    </div>
                    """,
                    unsafe_allow_html=True,
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
# DETERIORATION WATCH
#
# Not in the original brief — added deliberately. Every other section
# answers "how risky is this ONE country". This is the cross-country
# question a risk desk actually asks day to day: "what's moving, right
# now, across the whole panel". Ranks every tracked country by year-over-
# year score change and flags anyone who stepped into a worse risk band —
# the same instinct as an early-warning/OSINT watchlist, applied to macro
# data instead of geopolitical signals.
# ============================================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">Deterioration watch</div>
            <div class="section-sub">
                Which countries are moving fastest, across the whole panel — not just the one selected.
            </div>
        </div>
        <div class="micro">CROSS-COUNTRY / YEAR-OVER-YEAR</div>
    </div>
    """,
    unsafe_allow_html=True,
)

_BAND_ORDER = ["Low", "Moderate", "Elevated", "High", "Severe"]


def _band_rank(b):
    try:
        return _BAND_ORDER.index(str(b))
    except ValueError:
        return -1


_watch_scores = scores.dropna(subset=["risk_score"]).copy() if isinstance(scores, pd.DataFrame) else pd.DataFrame()
_watch_scores["year"] = pd.to_numeric(_watch_scores.get("year"), errors="coerce")
_watch_years = sorted(_watch_scores["year"].dropna().unique()) if not _watch_scores.empty else []

if len(_watch_years) >= 2:
    _wy_latest, _wy_prior = int(_watch_years[-1]), int(_watch_years[-2])
    _latest_s = _watch_scores[_watch_scores.year == _wy_latest].set_index("country_iso3")
    _prior_s = _watch_scores[_watch_scores.year == _wy_prior].set_index("country_iso3")
    _common = _latest_s.index.intersection(_prior_s.index)

    _delta = (_latest_s.loc[_common, "risk_score"] - _prior_s.loc[_common, "risk_score"]).sort_values(ascending=False)

    watch_left, watch_right = st.columns(2, gap="large")

    with watch_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-label" style="color:var(--red,#ff6b7a);">↑ FASTEST DETERIORATING</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card-caption" style="margin-bottom:10px;">{_wy_prior} &rarr; {_wy_latest}, largest increase first.</div>',
            unsafe_allow_html=True,
        )
        _worsening = _delta[_delta > 0].head(5)
        if not _worsening.empty:
            for iso3, d in _worsening.items():
                st.markdown(
                    f"""
                    <div class="peer-highlight">
                        <div class="peer-country">{esc(get_country_label(iso3))}</div>
                        <div class="peer-score" style="color:var(--red,#ff6b7a);">+{d:.1f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            empty_state("No country worsened between these two years.")
        st.markdown("</div>", unsafe_allow_html=True)

    with watch_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-label" style="color:var(--green,#54d69a);">↓ FASTEST IMPROVING</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card-caption" style="margin-bottom:10px;">{_wy_prior} &rarr; {_wy_latest}, largest decrease first.</div>',
            unsafe_allow_html=True,
        )
        _improving = _delta[_delta < 0].sort_values().head(5)
        if not _improving.empty:
            for iso3, d in _improving.items():
                st.markdown(
                    f"""
                    <div class="peer-highlight">
                        <div class="peer-country">{esc(get_country_label(iso3))}</div>
                        <div class="peer-score" style="color:var(--green,#54d69a);">{d:.1f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            empty_state("No country improved between these two years.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Band-transition alerts: a country whose risk BAND itself stepped down,
    # not just a score wobble within the same band — the closest thing
    # this project has to a "rating action" alert.
    _band_alerts = []
    for iso3 in _common:
        b_latest = _latest_s.loc[iso3, "risk_band"]
        b_prior = _prior_s.loc[iso3, "risk_band"]
        if _band_rank(b_latest) > _band_rank(b_prior):
            _band_alerts.append((iso3, b_prior, b_latest))

    if _band_alerts:
        st.markdown('<div class="card" style="margin-top:18px;">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">BAND TRANSITIONS</div>', unsafe_allow_html=True)
        for iso3, b_prior, b_latest in _band_alerts:
            st.markdown(
                f'<div class="card-caption">• {esc(get_country_label(iso3))}: '
                f'{esc(str(b_prior))} &rarr; <strong>{esc(str(b_latest))}</strong></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
else:
    empty_state("Deterioration watch needs at least two years of scored data across the panel.")


# ============================================================================
# MODEL VALIDATION — DOES THIS ACTUALLY WORK?
#
# Not decoration — a real check. Runs the SAME scoring engine's already-
# computed history against known, real macro-stress episodes for countries
# already in this panel, and reports honestly whether the score actually
# rose. No extra network calls: it evaluates the `scores` DataFrame already
# in memory, so this is a genuine check against live data once deployed —
# and it says so plainly when it's only looking at demo data instead.
# ============================================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">Model validation</div>
            <div class="section-sub">
                Would this score have actually flagged known real macro-stress episodes?
            </div>
        </div>
        <div class="micro">BACKTEST / HONESTY CHECK</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if USING_DEMO_DATA:
    st.warning(
        "Running on synthetic demo data — the results below are NOT a real "
        "validation of anything. Switch to live data to see the actual backtest."
    )

_bt_results = run_backtest(scores, data_is_synthetic=USING_DEMO_DATA)
_bt_flagged = sum(1 for r in _bt_results if r.verdict == "flagged")
_bt_missed = sum(1 for r in _bt_results if r.verdict == "missed")
_bt_inconclusive = sum(1 for r in _bt_results if r.verdict == "inconclusive")

st.markdown(
    f"""
    <div class="card" style="margin-bottom:16px;">
        <div class="card-caption">
            <strong>{_bt_flagged}</strong> flagged &middot;
            <strong>{_bt_missed}</strong> missed &middot;
            <strong>{_bt_inconclusive}</strong> inconclusive (data gap) &middot;
            {len(_bt_results)} known episodes checked
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

_verdict_style = {
    "flagged": ("var(--green,#54d69a)", "✓ FLAGGED"),
    "missed": ("var(--red,#ff6b7a)", "✗ MISSED"),
    "inconclusive": ("var(--muted,#8c98aa)", "— NO DATA"),
}

for r in _bt_results:
    color, tag = _verdict_style[r.verdict]
    detail = (
        f"Score {r.baseline_year}: {r.baseline_score:.1f} &rarr; {r.event_year}: {r.event_score:.1f} "
        f"({r.delta:+.1f} pts)"
        if r.delta is not None
        else f"No scored data for {r.baseline_year} and/or {r.event_year} in the current panel."
    )
    st.markdown(
        f"""
        <div class="card" style="margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;">
                <div class="card-label">{esc(r.label)}</div>
                <div style="color:{color};font-weight:700;font-size:12px;white-space:nowrap;">{tag}</div>
            </div>
            <div class="card-caption" style="margin-top:6px;">{esc(r.note)}</div>
            <div class="card-caption" style="margin-top:8px;font-family:'DM Mono',monospace;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("Why this matters / how to read it"):
    st.markdown(
        """
A dashboard that looks convincing and a model that actually works are two
different things — this section checks the second one honestly, using real
historical episodes for countries already tracked here.

**Flagged** means the composite score rose by at least 3 points from the
baseline year to the event year — a real signal, not noise.
**Missed** means it didn't, which is a genuine limitation worth naming, not
hiding: this is an annual, backward-looking, cross-sectional model — it
will structurally lag fast-moving currency or market-confidence shocks that
unfold within a single year, and it has no early-warning mechanism beyond
what's already in the YoY indicator changes it's built from.
**No data** means the panel doesn't currently have both years scored for
that country — expand the fetch window to see it.
        """
    )


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
        <div class="scenario-label">
            {"ACTIVE SHOCK" if run_scenario_btn else "SCENARIO READY"}
        </div>
        <div class="scenario-title">
            Policy-rate sensitivity
            <span style="color:{COLORS['violet']};"> {fmt_delta(shock,0)} bps</span>
        </div>
        <div class="card-caption" style="margin-top:7px;">
            Driver: POLICY_RATE_YOY_CHANGE_BPS · Baseline: {esc(country_label)}
            · {int(year)} · score {fmt_number(score_value,1)}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if scenario_error is not None:
    user_error("The scenario engine could not complete this run.", scenario_error)

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

            # BUG FIX: the real run_shock_scenario() return shape has no
            # "targets" dict — the per-indicator estimates live in a LIST
            # under "indicator_deltas", each item shaped like
            # {"indicator_code": ..., "baseline_value": ..., "estimated_delta": ...}.
            # The old lookup here checked for a "targets" key that never
            # existed, so scenario_target was always None and every card
            # silently fell back to the "engine output" placeholder text.
            if isinstance(scenario, dict):
                for item in scenario.get("indicator_deltas", []):
                    if isinstance(item, dict) and item.get("indicator_code") == code:
                        delta = item.get("estimated_delta")
                        base = item.get("baseline_value")
                        if delta is not None and not pd.isna(delta) and base is not None and not pd.isna(base):
                            scenario_target = base + delta
                        break

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
            {markdown_to_html(report)}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# METHODOLOGY / MODEL CARD
#
# Written like a desk note, not generic docs — explicit about how this
# compares to real institutional practice, and honest about exactly where
# it's a deliberate simplification of that, not an accident.
# ============================================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">Methodology &amp; model card</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Purpose, inputs, scoring, limitations, and intended use", expanded=False):
    st.markdown(
        """
**Purpose.** Relative macro-risk positioning across a country panel, built for
transparency — every number here should be traceable back to a public data
point and a documented weight, not a black box.

**Inputs.** World Bank Indicators API (cross-country annual panel) plus a
US-only FRED enrichment for policy-rate change. See `config/indicators.yaml`
for the exact series codes, weights, and direction — that file *is* the
methodology; nothing here is hardcoded in Python.

**Normalization.** Each indicator is z-scored against the *same-year* panel
(not a fixed historical baseline), then combined into a weighted composite
and mapped to 0–100.

**Missing data.** A missing indicator for a country is excluded and the
remaining weights renormalized — it is never filled with a zero, a mean, or
a guess. `data_completeness` on every score is a warning label, not
decoration.

**Scenario model.** Shock sensitivities are estimated by pooled OLS
regression across the panel itself, not asserted from a paper. That makes
them correlational and honestly weak with a small country/year sample —
the R² and observation count shown with every scenario result are there so
you can judge that for yourself, not so you can ignore them.

---

**How this compares to real practice.** This borrows structurally from how
institutional country-risk frameworks are built — the EIU's Country Risk
Service scores political, economic, and financial risk separately; Moody's
sovereign methodology weights economic strength, institutional strength,
fiscal strength, and susceptibility to event risk; the IMF's Debt
Sustainability Analysis stress-tests debt trajectories under shocks the way
the Scenario Lab here does, at a much smaller scale. This project follows
that *shape* — transparent weighted indicators, peer-relative scoring,
shock sensitivity — using only public annual data and a single analyst's
weighting, not a rated agency's committee process, proprietary data, or
qualitative overlay. It should be read as a worked illustration of the
approach, not a substitute for one of those ratings.

**Not intended for:** investment decisions, credit ratings, sovereign debt
pricing, or any use where being wrong has real financial consequences. It
is intended to demonstrate a transparent, reproducible approach to
macro-risk scoring — nothing here is investment advice.
        """
    )


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
    meta_html += (
        f'<div class="meta-item">'
        f'<div class="meta-k">{esc(key)}</div>'
        f'<div class="meta-v">{esc(value)}</div>'
        f"</div>"
    )

meta_html += "</div></div>"

st.markdown(meta_html, unsafe_allow_html=True)

with st.expander("Source traceability — every indicator, its source, and its weight"):
    _src_rows = []
    for _ind in indicators_cfg.get("indicators", []):
        _src_rows.append(
            {
                "Indicator": _ind.get("label", _ind.get("code", "")),
                "Code": _ind.get("code", ""),
                "Source": {
                    "world_bank": "World Bank",
                    "fred": "FRED",
                    "fred_us_only": "FRED (US only)",
                    "derived": "Derived (WB/FRED)",
                }.get(_ind.get("source", ""), _ind.get("source", "")),
                "Unit": _ind.get("unit", ""),
                "Weight": _ind.get("weight", ""),
                "Direction": "Higher = worse" if _ind.get("risk_direction", 1) in (1, "1", "higher_is_worse") else "Higher = better",
            }
        )
    if _src_rows:
        st.dataframe(pd.DataFrame(_src_rows), width="stretch", hide_index=True)
        st.caption(
            "World Bank: https://api.worldbank.org/v2 · FRED: https://fred.stlouisfed.org · "
            "full definitions in config/indicators.yaml."
        )
    else:
        empty_state("No indicator metadata available.")


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
        width="stretch",
    )

    st.caption(
        "Exports the currently loaded dashboard panel exactly as provided to the UI."
    )

with export_right:
    with st.expander("Inspect selected country-year row"):
        if current_row.empty:
            empty_state("No matching country-year row found.")
        else:
            selected_row = current_row.to_frame("value").reset_index()
            selected_row.columns = ["field", "value"]
            selected_row["field"] = selected_row["field"].astype(str)
            selected_row["value"] = selected_row["value"].astype(str)
            st.dataframe(
                selected_row,
                width="stretch",
            )

    with st.expander("Inspect top driver output"):
        if isinstance(country_drivers, pd.DataFrame) and not country_drivers.empty:
            st.dataframe(
                country_drivers,
                width="stretch",
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
