"""Shared design tokens and presentation helpers for the dashboard.

These mirror the helpers that previously lived at the top of
`dashboard/app.py`, extracted so page modules can reuse them without
coupling to the app shell. Nothing here performs score computation or
data loading — only formatting and light chart/HTML scaffolding.
"""

from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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

# Shared plotly chart config so every chart is read-only (hover only —
# no zoom / pan / drag selection) with a consistent stripped toolbar.
CHART_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "scrollZoom": False,
    "doubleClick": False,
    "showAxisDragHandles": False,
    "modeBarButtonsToRemove": [
        "zoom2d",
        "pan2d",
        "select2d",
        "lasso2d",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "resetScale2d",
    ],
}

# Live country metadata (iso3 -> record) for human-readable labels. Set once
# per run from the app shell. Streamlit reruns top-to-bottom per interaction,
# so this is always refreshed before any page renders.
COUNTRIES_CFG: dict | None = None


def set_countries_cfg(cfg: dict | None) -> None:
    global COUNTRIES_CFG
    COUNTRIES_CFG = cfg


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
    Narrow Markdown-to-HTML converter for exactly `generate_report()`'s output
    shape: **bold** inline markers, "- " bullet lists and blank-line-separated
    paragraphs. Not a general Markdown engine.
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
    if isinstance(COUNTRIES_CFG, dict):
        records = COUNTRIES_CFG.get("countries", [])
        if isinstance(records, list):
            return [r for r in records if isinstance(r, dict)]
    return []


def country_lookup():
    return {str(record.get("iso3")): record for record in country_records() if record.get("iso3")}


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
        title=dict(text=""),
        uirevision="country-risk-intelligence",
        dragmode=False,
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
        fixedrange=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,.055)",
        zeroline=False,
        linecolor="rgba(148,163,184,.08)",
        tickfont=dict(color=COLORS["faint"], size=10),
        fixedrange=True,
    )
    return fig


def empty_state(message):
    st.markdown(
        f'<div class="empty">{esc(message)}</div>',
        unsafe_allow_html=True,
    )


def user_error(message: str, exc: Exception | None = None):
    st.warning(message)
    if exc is not None:
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")


def plotly_chart(fig: go.Figure, height=360, margin: dict[str, Any] | None = None):
    """Render a read-only plotly chart with the shared stripping config."""
    make_plotly_layout(fig, height=height, margin=margin)
    st.plotly_chart(
        fig,
        width="stretch",
        config=CHART_CONFIG,
    )


def provenance_line(ctx: Any) -> str:
    """The small 'Verified · Source · as of · manifest' footer shown under scores/charts.

    Demo mode is labeled honestly as synthetic — it is never presented as
    verified public data.
    """
    from dashboard.context import Context  # local import to avoid a cycle

    if not isinstance(ctx, Context):
        return ""

    if ctx.using_demo_data:
        bits = ["DEMO DATA — SYNTHETIC", "not verified"]
        if ctx.manifest_hash:
            bits.append(f"manifest {ctx.manifest_hash}")
        return '<div class="provenance-line" style="color:var(--orange,#ff9f5b);">' + " · ".join(bits) + "</div>"

    if getattr(ctx, "using_cached_data", False):
        bits = ["CACHED DATA", "not freshly verified"]
        if ctx.provenance_sources:
            bits.append(ctx.provenance_sources)
        if ctx.manifest_hash:
            bits.append(f"manifest {ctx.manifest_hash}")
        return '<div class="provenance-line" style="color:var(--orange,#ff9f5b);">' + " · ".join(bits) + "</div>"

    status = "VERIFIED ✓" if ctx.data_verified else "VERIFIED"
    parts = [status]
    if ctx.provenance_sources:
        parts.append(f"SOURCE: {ctx.provenance_sources}")
    if ctx.provenance_asof:
        parts.append(f"AS OF: {ctx.provenance_asof}")
    if ctx.manifest_hash:
        parts.append(f"MANIFEST: {ctx.manifest_hash}")
    return '<div class="provenance-line">' + " · ".join(parts) + "</div>"


def render_provenance_line(ctx: Any) -> None:
    """Emit the provenance footer line (used under every score and chart)."""
    body = provenance_line(ctx)
    if body:
        st.markdown(body, unsafe_allow_html=True)
