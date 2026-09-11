"""Contagion & Correlations page.

Builds a network from the engine's OWN scored history: each country's YoY
risk-score delta is correlated pairwise with every other country's, edges are
drawn where two countries move together (|r| >= threshold with enough shared
years), and clusters are the connected components of that graph — the set of
countries that visibly co-move.

Presentation only: the correlations are computed by src/analysis/contagion.py
from `ctx.scores` (produced by the app shell), never re-derived here.
"""

from __future__ import annotations

import math

import plotly.graph_objects as go
import streamlit as st

from dashboard.context import Context
from dashboard.ui import (
    esc,
    fmt_number,
    get_country_label,
    plotly_chart,
    render_provenance_line,
)
from src.analysis.contagion import (
    CORRELATION_THRESHOLD_DEFAULT,
    MAX_EDGES_DEFAULT,
    MIN_PAIRS_DEFAULT,
    ContagionNetwork,
    correlation_network,
)

_CLUSTER_PALETTE = [
    "#5ee7f2",
    "#54d69a",
    "#ff9f5b",
    "#ffd166",
    "#7e6fd8",
    "#ff6b7a",
    "#8c98aa",
    "#4f9cf7",
]

_POSITIVE_COLOR = "rgba(94,231,242,0.85)"
_NEGATIVE_COLOR = "rgba(255,159,91,0.8)"


def _build_network_figure(network: ContagionNetwork) -> go.Figure:
    nodes = network.nodes
    if not nodes:
        return go.Figure()

    n = len(nodes)
    positions: dict[str, tuple[float, float]] = {}
    for i, node in enumerate(nodes):
        angle = 2.0 * math.pi * i / n
        positions[str(node["iso3"])] = (math.cos(angle), math.sin(angle))

    fig = go.Figure()

    for edge in network.edges:
        xa, ya = positions[str(edge["a"])]
        xb, yb = positions[str(edge["b"])]
        width = 1.5 + 4.5 * abs(float(edge["correlation"]))
        color = _POSITIVE_COLOR if float(edge["correlation"]) >= 0 else _NEGATIVE_COLOR
        fig.add_trace(
            go.Scatter(
                x=[xa, xb, None],
                y=[ya, yb, None],
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    xs = []
    ys = []
    custom = []
    sizes = []
    colors = []
    text = []
    for node in nodes:
        x, y = positions[str(node["iso3"])]
        xs.append(x)
        ys.append(y)
        iso3 = str(node["iso3"])
        integration = node.get("integration")
        sizes.append(10.0 if integration is None else 12.0 + 16.0 * float(integration))
        colors.append(_CLUSTER_PALETTE[int(node.get("cluster", 0)) % len(_CLUSTER_PALETTE)])
        text.append(iso3)
        custom.append(
            [
                get_country_label(iso3),
                iso3,
                int(node.get("cluster", 0)) + 1,
                "—" if integration is None else f"{float(integration):.2f}",
                int(node.get("degree", 0)),
            ]
        )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=text,
            textposition="top center",
            textfont=dict(color="#c8d3e0", size=11),
            marker=dict(size=sizes, color=colors, line=dict(color="#0a0f16", width=1.5)),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Cluster: %{customdata[2]}<br>"
                "Integration (mean |r|): %{customdata[3]}<br>"
                "Significant edges: %{customdata[4]}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        width=None,
        height=620,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="#101624", bordercolor="rgba(255,255,255,0.2)", font=dict(color="#e8eef6", size=13)),
        showlegend=False,
    )
    return fig


def _build_heatmap(network: ContagionNetwork) -> go.Figure:
    corr = network.correlation_matrix
    if corr.empty:
        return go.Figure()

    iso3s = [str(c) for c in corr.index]
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=iso3s,
            y=iso3s,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0.0, "#ff6b7a"],
                [0.5, "#1b2333"],
                [1.0, "#5ee7f2"],
            ],
            colorbar=dict(title=dict(text="Correlation"), thickness=10, len=0.6),
            hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(tickangle=-90)
    return fig


def _render_kpis(ctx: Context, network: ContagionNetwork) -> None:
    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">Contagion &amp; correlation network</div>
                <div class="section-sub">
                    Do countries' risk scores move together? Year-over-year score
                    deltas are correlated pairwise across the scored history; the
                    graph below draws the edges that clear the relevance bar.
                </div>
            </div>
            <div class="micro">DESCRIPTIVE CO-MOVEMENT / NOT CAUSAL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    n_edges = len(network.edges)
    mean_abs = network.mean_abs_correlation
    pos_share = network.positive_edge_share

    k1, k2, k3, k4 = st.columns(4)
    cells = [
        ("COUNTRIES", f"{network.n_countries}", "in the scored panel"),
        ("EDGES", f"{n_edges}", f"pairs with |r| ≥ {network.threshold:g}"),
        (
            "MEAN |R|",
            "—" if math.isnan(mean_abs) else fmt_number(mean_abs, 2),
            "avg absolute pairwise correlation",
        ),
        (
            "EDGES POSITIVE",
            "—" if math.isnan(pos_share) else f"{pos_share:.0%}",
            "of co-movement is same-direction",
        ),
    ]
    for col, (label, value, caption) in zip([k1, k2, k3, k4], cells, strict=False):
        with col:
            st.markdown(
                f"""
                <div class="card kpi">
                    <div class="kpi-accent"></div>
                    <div class="card-label">{esc(label)}</div>
                    <div class="card-value" style="font-size:26px;">{esc(value)}</div>
                    <div class="card-caption">{esc(caption)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if network.clusters and len(network.clusters) > 1:
        cluster_lines = "".join(
            f'<div class="card-caption">&nbsp;&nbsp;<span style="color:{color};">●</span> '
            f"Cluster {i + 1}: {esc(', '.join(get_country_label(c) for c in members))}</div>"
            for i, (color, members) in enumerate(
                zip(
                    (_CLUSTER_PALETTE[i] for i in range(len(network.clusters))),
                    network.clusters,
                    strict=False,
                )
            )
        )
        st.markdown(
            f'<div class="card" style="margin:16px 0;"><div class="card-caption">'
            f"The thresholded network splits the panel into {len(network.clusters)} cluster(s) "
            f"— the countries whose risk deltas visibly move together. Node size = how tightly "
            f"that country is coupled to the panel (mean |correlation|).</div>{cluster_lines}</div>",
            unsafe_allow_html=True,
        )

    if n_edges == 0:
        st.markdown(
            '<div class="card-caption" style="margin-top:12px;">No country pair clears the '
            f"|r| ≥ {network.threshold:g} bar with at least {int(network.min_pairs)} shared years in "
            "the current scored history. Loosen the threshold or extend the panel window.</div>",
            unsafe_allow_html=True,
        )


def _render_edge_highlights(network: ContagionNetwork) -> None:
    if not network.edges:
        return

    strongest = sorted(network.edges, key=lambda e: abs(e["correlation"]), reverse=True)[:8]
    rows = []
    for e in strongest:
        sign = "+" if e["correlation"] >= 0 else "−"
        rows.append(
            f'<div class="card-caption">&nbsp;&nbsp;{esc(get_country_label(str(e["a"])))} ↔ '
            f"{esc(get_country_label(str(e['b'])))} &nbsp; "
            f"<strong>{sign}{abs(e['correlation']):.2f}</strong> &nbsp; ({int(e['pairs'])} yrs)</div>"
        )
    st.markdown(
        '<div class="card"><div class="card-label">STRONGEST CO-MOVEMENTS</div>' + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )


def render_contagion(ctx: Context) -> None:
    network = correlation_network(
        ctx.scores,
        min_pairs=MIN_PAIRS_DEFAULT,
        threshold=CORRELATION_THRESHOLD_DEFAULT,
        max_edges=MAX_EDGES_DEFAULT,
    )

    if network.n_countries == 0:
        st.info("Not enough scored history to build a correlation network (need ≥ 2 countries and years).")
        return

    _render_kpis(ctx, network)
    render_provenance_line(ctx)

    st.plotly_chart(
        _build_network_figure(network),
        width="stretch",
        key="contagion_network",
    )

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        _render_edge_highlights(network)

    with right:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">HOW TO READ THIS</div>
                <div class="card-caption">
                    Edges are drawn between countries whose <strong>year-over-year risk-score
                    movements</strong> are correlated at |r| ≥ {network.threshold:g} across at least
                    {int(network.min_pairs)} shared years. Cyan edges = same-direction co-movement;
                    orange = opposite-direction. Clusters are discovered from the data (connected
                    components), not assumed from region or income level.
                </div>
                <div class="card-caption" style="margin-top:10px;font-style:italic;">
                    Correlation is descriptive, not causal: two countries moving together can
                    share a common shock without transmitting anything to each other. Demo data
                    is synthetic — these edges are interface demonstrations, not evidence.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Correlation matrix (country × country, all pairs)"):
        heatmap = _build_heatmap(network)
        if heatmap.data:
            plotly_chart(heatmap, height=620)

    with st.expander("Method & thresholds"):
        st.markdown(
            f"""
- **Delta series**: each country's `risk_score` is differenced by one year, so the
  network measures *co-movement of risk changes*, not shared levels.
- **Correlation**: Pearson r between two countries' delta series, using only years where
  both countries were scored (`min_pairs = {int(network.min_pairs)}` shared years required).
- **Edge rule**: |r| ≥ `{network.threshold:g}`, capped at {MAX_EDGES_DEFAULT} strongest pairs so
  the graph stays legible.
- **Integration**: a node's mean absolute correlation against every other panel country —
  a rough "how coupled is this country" score, not a directional measure.
            """
        )

    render_provenance_line(ctx)
