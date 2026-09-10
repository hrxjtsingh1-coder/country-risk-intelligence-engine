"""Comparison page.

Cross-sectional context across the whole panel: peer-relative risk
positioning for the selected country-year and the panel-wide
year-over-year deterioration / improvement watch (the closest thing the
dashboard has to a daily early-warning list).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.context import Context
from dashboard.ui import (
    empty_state,
    esc,
    find_country_column,
    find_score_column,
    fmt_number,
    get_country_label,
    plotly_chart,
    safe_float,
)


def render_peer_comparison(ctx: Context) -> None:
    scores = ctx.scores
    year = ctx.year
    country = ctx.country
    score_value = ctx.score_value
    score_color = ctx.score_color

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

    peer_left, peer_right = st.columns([1.45, 0.75], gap="large")

    with peer_left:
        peer_fig = go.Figure()

        if isinstance(scores, pd.DataFrame):
            score_column = find_score_column(scores)
            country_column = find_country_column(scores)

            if score_column and country_column and "year" in scores.columns:
                peers = scores[pd.to_numeric(scores["year"], errors="coerce").eq(int(year))].copy()

                peers[score_column] = pd.to_numeric(
                    peers[score_column],
                    errors="coerce",
                )
                peers = peers.dropna(subset=[score_column])
                peers = peers.sort_values(score_column, ascending=True)

                if not peers.empty:
                    peers["label"] = peers[country_column].astype(str)

                    marker_colors = [
                        score_color if str(x) == str(country) else "rgba(110,168,255,.55)" for x in peers["label"]
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
            plotly_chart(
                peer_fig,
                height=max(360, min(620, 130 + 24 * len(peer_fig.data[0].y))),
                margin=dict(l=8, r=8, t=20, b=8),
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
                peers = scores[pd.to_numeric(scores["year"], errors="coerce").eq(int(year))].copy()
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
                                <div class="peer-position-value">{fmt_number(score_value, 1)}</div>
                            </div>
                            <div>
                                <div class="micro">PEER MEDIAN</div>
                                <div class="peer-position-value">{fmt_number(_median, 1)}</div>
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
                            <div class="peer-score">{fmt_number(peer_score, 1)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                empty_state("Relative-position table unavailable.")
        else:
            empty_state("Relative-position table unavailable.")

        st.markdown("</div>", unsafe_allow_html=True)


_BAND_ORDER = ["Low", "Moderate", "Elevated", "High", "Severe"]


def _band_rank(b):
    try:
        return _BAND_ORDER.index(str(b))
    except ValueError:
        return -1


def render_deterioration_watch(ctx: Context) -> None:
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

    _watch_scores = (
        ctx.scores.dropna(subset=["risk_score"]).copy() if isinstance(ctx.scores, pd.DataFrame) else pd.DataFrame()
    )
    _watch_scores["year"] = pd.to_numeric(_watch_scores.get("year"), errors="coerce")
    _watch_years = sorted(_watch_scores["year"].dropna().unique()) if not _watch_scores.empty else []

    if len(_watch_years) >= 2:
        _wy_latest, _wy_prior = int(_watch_years[-1]), int(_watch_years[-2])
        _latest_s = _watch_scores[_watch_scores.year == _wy_latest].set_index("country_iso3")
        _prior_s = _watch_scores[_watch_scores.year == _wy_prior].set_index("country_iso3")
        _common = _latest_s.index.intersection(_prior_s.index)

        _delta = (_latest_s.loc[_common, "risk_score"] - _prior_s.loc[_common, "risk_score"]).sort_values(
            ascending=False
        )

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
                    f"{esc(str(b_prior))} &rarr; <strong>{esc(str(b_latest))}</strong></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        empty_state("Deterioration watch needs at least two years of scored data across the panel.")
