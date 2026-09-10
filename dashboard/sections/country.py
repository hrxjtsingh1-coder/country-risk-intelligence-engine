"""Country deep-dive page.

Everything scoped to the currently selected country-year slice: hero,
data provenance, executive KPI snapshot, plain-language interpretation,
risk trajectory, driver decomposition, analyst intelligence and the
data-coverage audit block.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.context import Context
from dashboard.ui import (
    APP_KICKER,
    COLORS,
    empty_state,
    esc,
    find_country_column,
    find_score_column,
    fmt_delta,
    fmt_number,
    get_country_label,
    markdown_to_html,
    plotly_chart,
    safe_float,
    score_pct,
)


def render_hero(ctx: Context) -> None:
    iso = ctx.iso
    score_value = ctx.score_value
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
                        <div class="status-pill">{esc(iso)} · {int(ctx.year)}</div>
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
                    <div class="terminal-line"><b>$</b> period.lock → {int(ctx.year)}</div>
                    <div class="terminal-line"><b>$</b> risk.compute → {fmt_number(score_value, 1)}</div>
                    <div class="terminal-line"><b>$</b> scenario.delta → {fmt_delta(ctx.shock, 0)} bps</div>
                    <div class="terminal-line" style="margin-top:12px;color:#54d69a;">
                        ✓ analytical layer ready
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_provenance(ctx: Context) -> None:
    if ctx.using_demo_data:
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
        prov = ctx.live_provenance
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


def render_kpi(ctx: Context) -> None:
    scores = ctx.scores
    country = ctx.country
    year = ctx.year
    score_value = ctx.score_value
    score_color = ctx.score_color
    coverage_value = ctx.coverage_value
    band = ctx.band

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
                    {fmt_number(score_value, 1)}
                </div>
                <div class="card-caption">{esc(band)} risk band · 0–100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        coverage_text = "—" if pd.isna(coverage_value) else f"{fmt_number(coverage_value, 1)}%"
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
                    versus {int(year) - 1} composite score
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
                same_year = scores[pd.to_numeric(scores["year"], errors="coerce").eq(int(year))][score_column].dropna()
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


def render_interpretation(ctx: Context) -> None:
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

    _slice = (
        ctx.drivers[
            (ctx.drivers["country_iso3"].astype(str) == str(ctx.country))
            & (pd.to_numeric(ctx.drivers["year"], errors="coerce") == int(ctx.year))
        ].copy()
        if isinstance(ctx.drivers, pd.DataFrame) and not ctx.drivers.empty
        else pd.DataFrame()
    )

    if not _slice.empty and "weighted_contribution" in _slice.columns:
        _slice = _slice.dropna(subset=["weighted_contribution"])
        _higher = _slice[_slice["weighted_contribution"] > 0].sort_values("weighted_contribution", ascending=False)
        _lower = _slice[_slice["weighted_contribution"] < 0].sort_values("weighted_contribution", ascending=True)
        _label_col = (
            "label"
            if "label" in _slice.columns
            else ("indicator_code" if "indicator_code" in _slice.columns else _slice.columns[0])
        )

        _up_names = [str(r[_label_col]) for _, r in _higher.head(3).iterrows()]
        _down_names = [str(r[_label_col]) for _, r in _lower.head(2).iterrows()]

        _country_label = get_country_label(ctx.country)

        _sentence = (
            f"**{esc(_country_label)}** is currently positioned in the **{esc(ctx.band.lower())}** "
            f"relative-risk band within the selected comparison panel."
        )
        st.markdown(
            f'<div class="card"><div class="card-value" style="font-size:16px;font-weight:500;line-height:1.5;">{_sentence}</div>',
            unsafe_allow_html=True,
        )

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


def render_trajectory(ctx: Context) -> None:
    scores = ctx.scores
    country = ctx.country
    iso = ctx.iso

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

    left, right = st.columns([0.85, 1.55], gap="large")

    with left:
        st.markdown(
            f"""
            <div class="card score-card">
                <div class="card-label">RISK SCORE GAUGE</div>
                <div class="score-wrap">
                    <div class="score-ring"
                         style="--score-pct-target:{score_pct(ctx.score_value)};--score-color:{ctx.score_color};">
                        <div class="score-inner">
                            <div class="score-number">{fmt_number(ctx.score_value, 0)}</div>
                            <div class="score-band">{esc(ctx.band.upper())}</div>
                        </div>
                    </div>
                </div>
                <div style="text-align:center;margin-top:16px;">
                    <span class="status-pill">
                        <span class="status-dot"></span>
                        {esc(ctx.country_label)} · {int(ctx.year)}
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
                history = scores[scores[country_column].astype(str).eq(str(country))].copy()

                if history.empty and country_column in ["iso3", "ISO3"]:
                    history = scores[scores[country_column].astype(str).eq(str(iso))].copy()

                history["year"] = pd.to_numeric(history["year"], errors="coerce")
                history[score_column] = pd.to_numeric(history[score_column], errors="coerce")
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
            plotly_chart(fig, height=410)


def render_drivers(ctx: Context) -> None:
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

    _full_slice = (
        ctx.drivers[
            (ctx.drivers["country_iso3"].astype(str) == str(ctx.country))
            & (pd.to_numeric(ctx.drivers["year"], errors="coerce") == int(ctx.year))
        ].copy()
        if isinstance(ctx.drivers, pd.DataFrame) and not ctx.drivers.empty
        else pd.DataFrame()
    )

    if not _full_slice.empty and "weighted_contribution" in _full_slice.columns:
        _full_slice = _full_slice.dropna(subset=["weighted_contribution"])
        _label_col = "label" if "label" in _full_slice.columns else "indicator_code"
        _has_raw = "raw_value" in _full_slice.columns
        _has_unit = "unit" in _full_slice.columns

        def _render_signal_group(df_group: pd.DataFrame, arrow: str, pts_prefix: str):
            for _, r in df_group.iterrows():
                name = str(r[_label_col])
                pts = safe_float(r["weighted_contribution"]) * 100
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
        _magnitude_ref = max(_full_slice["weighted_contribution"].abs().max() * 100, 1e-9)
    else:
        _higher_full = pd.DataFrame()
        _lower_full = pd.DataFrame()

    with driver_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-label" style="color:var(--red,#ff6b7a);">↑ HIGHER-RISK SIGNALS</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="card-caption" style="margin-bottom:10px;">Pushing the score up, largest first.</div>',
            unsafe_allow_html=True,
        )
        if not _higher_full.empty:
            st.markdown('<div class="driver-list">', unsafe_allow_html=True)
            _render_signal_group(_higher_full.head(6), "↑", "+")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            empty_state("No indicators are currently adding to relative risk.")
        st.markdown("</div>", unsafe_allow_html=True)

    with driver_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-label" style="color:var(--green,#54d69a);">↓ MITIGATING SIGNALS</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="card-caption" style="margin-bottom:10px;">Pulling the score down, largest first.</div>',
            unsafe_allow_html=True,
        )
        if not _lower_full.empty:
            st.markdown('<div class="driver-list">', unsafe_allow_html=True)
            _render_signal_group(_lower_full.head(6), "↓", "-")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            empty_state("No indicators are currently reducing relative risk.")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Technical details — indicator codes, weights, sources"):
        if not _full_slice.empty:
            _tech_cols = [
                c
                for c in [
                    "indicator_code",
                    _label_col,
                    "category",
                    "raw_value",
                    "unit",
                    "weight",
                    "risk_direction",
                    "z_risk",
                    "weighted_contribution",
                ]
                if c in _full_slice.columns
            ]
            st.dataframe(_full_slice[_tech_cols].reset_index(drop=True), width="stretch", hide_index=True)
        else:
            st.caption("No driver data available for this technical view.")


def render_analyst_intelligence(ctx: Context) -> None:
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
                        {esc(ctx.country_label)} · {int(ctx.year)} · {esc(ctx.band)} risk regime
                    </div>
                </div>
            </div>
            <div class="intel-body">
                {markdown_to_html(ctx.report)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_coverage(ctx: Context) -> None:
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
        ("COUNTRY", ctx.country_label),
        ("ISO3", ctx.iso),
        ("YEAR", ctx.year),
        ("PANEL ROWS", f"{len(ctx.panel):,}"),
        ("SCORE BAND", ctx.band),
        ("DATA COVERAGE", "—" if pd.isna(ctx.coverage_value) else f"{ctx.coverage_value:.1f}%"),
        ("SHOCK", f"{ctx.shock:+.0f} bps"),
        ("UI REFRESHED", ctx.generated_at),
    ]

    meta_html = '<div class="card"><div class="metadata">'

    for key, value in metadata:
        meta_html += (
            f'<div class="meta-item"><div class="meta-k">{esc(key)}</div><div class="meta-v">{esc(value)}</div></div>'
        )

    meta_html += "</div></div>"

    st.markdown(meta_html, unsafe_allow_html=True)

    with st.expander("Source traceability — every indicator, its source, and its weight"):
        _src_rows = []
        for _ind in ctx.indicators_cfg.get("indicators", []):
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
                    "Direction": "Higher = worse"
                    if _ind.get("risk_direction", 1) in (1, "1", "higher_is_worse")
                    else "Higher = better",
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
