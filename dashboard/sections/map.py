"""Global risk map page.

Animated, cursor-reactive world choropleth of the panel's `risk_score` by
ISO3, broken into the same five risk bands used across the dashboard:

* five-tier band colors (Low / Moderate / Elevated / High / Severe),
* a hover tooltip that follows the cursor with score + band + country,
* a play-able year animation (frames per panel year) with a halo that pulses
  on High / Severe countries while animating,
* click-to-drill: clicking a country drops the sidebar Country selector to
  that ISO3 and reruns, landing on the Country Deep-Dive below.

Presentation only — scores come straight from `ctx.scores` (computed by the
app shell), the map never re-runs the methodology.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.context import Context
from dashboard.ui import (
    BAND_COLORS_UI,
    CHART_CONFIG,
    esc,
    get_country_label,
    score_band,
)

_BAND_ORDER = ["Low", "Moderate", "Elevated", "High", "Severe"]

# Rough country centroids for the glow overlay (scattergeo needs lat/lon).
# Covers the config country set; countries without an entry simply get no glow.
_CENTROIDS: dict[str, tuple[float, float]] = {
    "USA": (39.8, -98.6),
    "CAN": (56.1, -106.3),
    "MEX": (23.6, -102.5),
    "BRA": (-14.2, -51.9),
    "GBR": (54.0, -2.6),
    "DEU": (51.1, 10.4),
    "FRA": (46.6, 2.2),
    "ITA": (41.9, 12.6),
    "ESP": (40.4, -3.7),
    "NLD": (52.2, 5.3),
    "POL": (52.1, 19.4),
    "TUR": (39.0, 35.3),
    "CHN": (35.9, 104.2),
    "JPN": (36.2, 138.3),
    "KOR": (36.5, 127.8),
    "IND": (20.6, 79.0),
    "IDN": (-2.5, 117.9),
    "SGP": (1.35, 103.8),
    "AUS": (-25.3, 134.0),
    "ZAF": (-29.0, 25.1),
}

# Discrete five-band colorscale over the 0..100 score axis.
_BAND_STOPS = [0, 20, 40, 60, 80, 100]
_COLORSCALE = []
for _i in range(5):
    _color = BAND_COLORS_UI[_BAND_ORDER[_i]]
    _COLORSCALE.append((_BAND_STOPS[_i] / 100.0, _color))
    _COLORSCALE.append((_BAND_STOPS[_i + 1] / 100.0, _color))

_HOVER_TEMPLATE = "<b>%{customdata[0]}</b><br>Score: %{customdata[1]}<br>Band: %{customdata[2]}<extra></extra>"


def _score_matrix(ctx: Context) -> pd.DataFrame:
    scores = ctx.scores.dropna(subset=["risk_score"])
    matrix = scores.pivot_table(
        index="country_iso3",
        columns="year",
        values="risk_score",
        aggfunc="first",
    )
    matrix = matrix.sort_index()
    return matrix[matrix.columns.astype(int).sort_values()]


def _customdata_for(matrix: pd.DataFrame, year: int) -> list[list[str]]:
    custom = []
    for iso3, row in matrix.iterrows():
        value = row.get(year, math.nan)
        if pd.isna(value):
            custom.append([get_country_label(iso3), "—", "No data", iso3])
        else:
            custom.append([get_country_label(iso3), f"{float(value):.1f}", score_band(value), iso3])
    return custom


def _z_for(matrix: pd.DataFrame, year: int) -> list[float]:
    return [float(v) if not pd.isna(v := row.get(year, math.nan)) else math.nan for _, row in matrix.iterrows()]


def _glow_for(matrix: pd.DataFrame, year: int) -> list[str]:
    return [
        str(iso3)
        for iso3, row in matrix.iterrows()
        if not pd.isna(row.get(year)) and score_band(row.get(year)) in {"High", "Severe"} and iso3 in _CENTROIDS
    ]


def _choropleth_trace(matrix: pd.DataFrame, year: int) -> go.Choropleth:
    return go.Choropleth(
        locations=[str(i) for i in matrix.index],
        locationmode="ISO-3",
        z=_z_for(matrix, year),
        zmin=0,
        zmax=100,
        colorscale=_COLORSCALE,
        customdata=_customdata_for(matrix, year),
        hovertemplate=_HOVER_TEMPLATE,
        colorbar=dict(
            title=dict(text="Risk score", side="top"),
            thickness=12,
            len=0.45,
            tickvals=[10, 30, 50, 70, 90],
            ticktext=_BAND_ORDER,
            tickfont=dict(color="#7e8a9b", size=10),
            outlinewidth=0,
        ),
    )


def _glow_traces(matrix: pd.DataFrame, year: int, year_index: int) -> list[go.Scattergeo]:
    glow_iso3 = _glow_for(matrix, year)
    if not glow_iso3:
        return []
    lat = [_CENTROIDS[c][0] for c in glow_iso3]
    lon = [_CENTROIDS[c][1] for c in glow_iso3]
    # Halo size oscillates across frames so the glow visibly pulses when the
    # year animation plays; static view keeps a steady halo.
    halo_size = 30.0 if year_index % 2 == 0 else 20.0
    return [
        go.Scattergeo(
            lat=lat,
            lon=lon,
            mode="markers",
            marker=dict(symbol="circle", size=halo_size, color="rgba(255,107,122,0.18)", line=dict(width=0)),
            hoverinfo="skip",
            showlegend=False,
        ),
        go.Scattergeo(
            lat=lat,
            lon=lon,
            mode="markers",
            marker=dict(symbol="circle", size=8, color=BAND_COLORS_UI["Severe"], line=dict(width=0)),
            hoverinfo="skip",
            showlegend=False,
        ),
    ]


def _build_figure(ctx: Context) -> go.Figure:
    matrix = _score_matrix(ctx)
    years = list(matrix.columns)
    if not years:
        return go.Figure()

    year_index = max(0, years.index(ctx.year)) if ctx.year in years else len(years) - 1
    base_year = years[year_index]

    fig = go.Figure()
    fig.add_trace(_choropleth_trace(matrix, base_year))
    for trace in _glow_traces(matrix, base_year, year_index):
        fig.add_trace(trace)

    frames = []
    for idx, year in enumerate(years):
        frame_data = [_choropleth_trace(matrix, year)]
        frame_data.extend(_glow_traces(matrix, year, idx))
        frames.append(go.Frame(name=str(year), data=frame_data, traces=list(range(len(frame_data)))))

    fig.frames = frames

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type="natural earth",
            bgcolor="rgba(0,0,0,0)",
            landcolor="rgba(30,38,54,0.75)",
            coastlinecolor="rgba(255,255,255,0.12)",
            countrycolor="rgba(255,255,255,0.18)",
            showcountries=True,
            showlakes=False,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        height=560,
        uirevision="risk-map",
        hoverlabel=dict(bgcolor="#101624", bordercolor="rgba(255,255,255,0.2)", font=dict(color="#e8eef6", size=13)),
        sliders=[
            dict(
                active=year_index,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(prefix="Year: ", font=dict(color="#9aa7b8", size=13)),
                pad=dict(b=10, t=60),
                len=0.8,
                font=dict(color="#7e8a9b", size=11),
                steps=[
                    dict(
                        label=str(year),
                        method="animate",
                        args=[
                            [str(year)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 450, "redraw": True},
                                "transition": {"duration": 250},
                            },
                        ],
                    )
                    for year in years
                ],
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 650, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 350},
                            },
                        ],
                    ),
                    dict(
                        label="❚❚ Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
                direction="left",
                pad=dict(r=10, t=55),
                bgcolor="rgba(30,38,54,0.9)",
                bordercolor="rgba(255,255,255,0.25)",
                font=dict(color="#e8eef6", size=12),
            )
        ],
    )
    return fig


def render_map(ctx: Context) -> None:
    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">Global risk map</div>
                <div class="section-sub">
                    Animated world risk surface — five-tier bands across the panel.
                </div>
            </div>
            <div class="micro">ANIMATED / 5-TIER / CLICK-TO-DRILL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    legend_dots = "".join(
        f'<span style="display:inline-block;margin:0 14px 0 0;"><span style="display:inline-block;'
        f'width:11px;height:11px;border-radius:50%;background:{color};margin-right:6px;"></span>'
        f'<span style="vertical-align:middle;">{band}</span></span>'
        for band, color in BAND_COLORS_UI.items()
    )
    st.markdown(
        f'<div style="margin:0 0 6px;font-size:12px;color:#9aa7b8;">{legend_dots}</div>',
        unsafe_allow_html=True,
    )

    fig = _build_figure(ctx)
    if not fig.data:
        st.info("No scored country-years to display on the global map.")
        return

    selection = st.plotly_chart(
        fig,
        key="global_risk_map",
        on_select="rerun",
        selection_mode="points",
        config=CHART_CONFIG,
    )

    if selection is not None:
        sel = getattr(selection, "selection", selection)
        if isinstance(sel, dict) and isinstance(sel.get("points"), list):
            for point in sel["points"]:
                custom = point.get("customdata") if isinstance(point, dict) else None
                if isinstance(custom, list) and len(custom) >= 4:
                    st.session_state["country_selector"] = str(custom[3])
                    st.rerun()

    st.markdown(
        f"""
        <div class="card-caption" style="margin-top:10px;color:#7e8a9b;">
            {esc(get_country_label(ctx.country))} {ctx.iso} / {int(ctx.year)} ·
            hover a country for score + band · click to open the Country Deep-Dive.
        </div>
        """,
        unsafe_allow_html=True,
    )
