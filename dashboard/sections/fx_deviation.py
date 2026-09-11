"""FX deviation from long-run trend.

Uses the persisted nominal FX level (World Bank PA.NUS.FCRF, LCU per USD) the
pipeline already collects and the derived FX_TREND_DEVIATION_PCT column the
live pipeline produces (nominal level vs its own centered 5-year trend). This
is a nominal, REER-style proxy — explicitly not a true trade-weighted REER —
and is labeled as such in the UI.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.ui import COLORS, empty_state, esc, fmt_number, plotly_chart  # noqa: E402


def _country_fx(ctx) -> tuple[list[int], list[float], list[float | None]]:
    """:returns (years, level, deviation) for the selected country."""
    if not hasattr(ctx, "panel") or ctx.panel is None or "FX_LEVEL_USD_LCU" not in ctx.panel.columns:
        return [], [], []
    frame = ctx.panel[ctx.panel["country_iso3"].astype(str).eq(str(ctx.country))].copy()
    if frame.empty:
        return [], [], []
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame = frame.sort_values("year")
    years = [int(y) for y in frame["year"].tolist()]
    level = [float(v) for v in frame["FX_LEVEL_USD_LCU"].tolist()]
    if "FX_TREND_DEVIATION_PCT" in ctx.panel.columns:
        deviation = [float(v) if pd.notna(v) else None for v in frame["FX_TREND_DEVIATION_PCT"].tolist()]
    else:
        deviation = [None] * len(years)
    return years, level, deviation


def _latest_deviation(deviation: list[float | None]) -> float | None:
    for value in reversed(deviation):
        if value is not None:
            return value
    return None


def render_fx_deviation(ctx) -> None:
    years, level, deviation = _country_fx(ctx)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">FX DEVIATION FROM TREND</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-caption" style="margin-bottom:10px;">Nominal exchange rate vs its own long-run trend '
        "(centered 5-year mean). A nominal REER-style proxy — not a true trade-weighted REER.</div>",
        unsafe_allow_html=True,
    )

    if not level:
        empty_state("FX level series not available for this slice.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    latest = _latest_deviation(deviation)
    if latest is None:
        cols = st.columns(2)
        cols[0].markdown(
            f'<div class="card"><div class="card-label" style="font-size:11px;">LATEST FX LEVEL (LCU / USD)</div>'
            f'<div class="card-value">{fmt_number(level[-1], 2)}</div></div>',
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            '<div class="card"><div class="card-label" style="font-size:11px;">TREND DEVIATION</div>'
            '<div class="card-caption">Not scorable from this slice.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        if latest > 0.0:
            direction = "WEAKER than its 5-year trend"
            color = COLORS["red"]
        elif latest < 0.0:
            direction = "STRONGER than its 5-year trend"
            color = COLORS["green"]
        else:
            direction = "AT its 5-year trend"
            color = COLORS["muted"]
        cols = st.columns(2)
        cols[0].markdown(
            f'<div class="card"><div class="card-label" style="font-size:11px;">LATEST FX LEVEL (LCU / USD)</div>'
            f'<div class="card-value">{fmt_number(level[-1], 2)}</div></div>',
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            f'<div class="card"><div class="card-label" style="font-size:11px;">DEVIATION vs TREND</div>'
            f'<div class="card-value" style="color:{color};">{fmt_number(latest, 1)}%</div>'
            f'<div class="card-caption">{esc(direction)} — higher LCU per USD means a weaker currency.</div></div>',
            unsafe_allow_html=True,
        )

    trend = _rolling_trend(level, window=5, min_periods=3)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=years,
            y=[round(t, 3) if t is not None else None for t in trend],
            mode="lines",
            name="5-yr trend",
            line=dict(color=COLORS["muted"], width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=level,
            mode="lines+markers",
            name="FX level (LCU/USD)",
            line=dict(color=COLORS["cyan"], width=2.5),
            marker=dict(size=5),
        )
    )
    bars = go.Bar(
        x=years,
        y=deviation,
        name="Deviation vs trend (%)",
        marker_color=[COLORS["red"] if (v or 0.0) >= 0 else COLORS["green"] for v in deviation],
        yaxis="y2",
        opacity=0.55,
    )
    fig.add_trace(bars)
    fig.update_layout(
        **{
            k: v
            for k, v in {
                "title": "Exchange rate level and deviation from its long-run trend",
                "xaxis": dict(title="Year", gridcolor="rgba(148,163,184,.08)"),
                "yaxis": dict(title="LCU per USD (level)", gridcolor="rgba(148,163,184,.08)"),
                "yaxis2": dict(title="Deviation (%)", overlaying="y", side="right"),
                "legend": dict(orientation="h", yanchor="bottom", y=1.02),
                "height": 340,
            }.items()
        },
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        margin=dict(l=0, r=0, t=44, b=0),
    )
    plotly_chart(fig)

    if ctx.using_demo_data:
        st.markdown(
            '<div class="card-caption" style="margin-top:8px;"><b>DEMO PANEL</b> — FX level here is synthetic; '
            "the deviation is computed from the demo level with the same derivation the live pipeline uses.</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="card-caption" style="margin-top:6px;">Derived from the already-collected World Bank '
        "PA.NUS.FCRF series; deviation = (level / 5-yr centered trend − 1). Not included in the composite score.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _rolling_trend(values: list[float], window: int, min_periods: int) -> list[float | None]:
    """Centered rolling mean matching src.indicators.build_panel derivation."""
    series = pd.Series(values)
    trend = series.rolling(window, center=True, min_periods=min_periods).mean()
    return [None if v is None else float(v) for v in trend.tolist()]
