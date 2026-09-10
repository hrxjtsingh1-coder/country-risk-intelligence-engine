"""Scenario simulator page.

Renders the scenario engine's output for the current sidebar shock:
baseline / scenario / risk-delta cards, the per-target indicator cards
and the transmission-channels note. Only presentation — the scenario is
computed by the app shell (its result also feeds `generate_report`), so
this module never re-runs the engine.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.context import Context
from dashboard.ui import (
    COLORS,
    band_color,
    empty_state,
    esc,
    fmt_delta,
    fmt_number,
    safe_float,
    score_band,
    user_error,
)


def render_scenario_laboratory(ctx: Context) -> None:
    run_scenario_btn = ctx.run_scenario_btn
    scenario_error = ctx.scenario_error
    scenario = ctx.scenario

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
                <span style="color:{COLORS["violet"]};"> {fmt_delta(ctx.shock, 0)} bps</span>
            </div>
            <div class="card-caption" style="margin-top:7px;">
                Driver: POLICY_RATE_YOY_CHANGE_BPS · Baseline: {esc(ctx.country_label)}
                · {int(ctx.year)} · score {fmt_number(ctx.score_value, 1)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if scenario_error is not None:
        user_error("The scenario engine could not complete this run.", scenario_error)

    if scenario is not None:
        sc1, sc2, sc3 = st.columns(3)

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
            scenario_baseline = ctx.score_value

        if scenario_value is None and scenario_delta is not None:
            scenario_value = scenario_baseline + scenario_delta

        if scenario_value is None:
            scenario_value = ctx.score_value

        if scenario_delta is None:
            scenario_delta = scenario_value - scenario_baseline

        with sc1:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-label">BASELINE</div>
                    <div class="card-value">{fmt_number(scenario_baseline, 1)}</div>
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
                        {fmt_number(scenario_value, 1)}
                    </div>
                    <div class="card-caption">{esc(scenario_band)} band after shock</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with sc3:
            delta_class = (
                "delta-positive" if scenario_delta > 0 else "delta-negative" if scenario_delta < 0 else "delta-neutral"
            )

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-label">RISK DELTA</div>
                    <div class="card-value {delta_class}">
                        {fmt_delta(scenario_delta, 1)}
                    </div>
                    <div class="card-caption">scenario minus baseline</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
                if not ctx.current_row.empty and code in ctx.current_row.index:
                    baseline_target = ctx.current_row[code]

                scenario_target = None

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
                            {esc(fmt_number(baseline_target, 2) if baseline_target is not None else "—")}
                        </div>
                        <div style="margin-top:9px;color:#7e8a9b;font-size:10px;">SCENARIO</div>
                        <div style="font-family:'DM Mono';font-size:15px;color:{COLORS["violet"]};">
                            {esc(fmt_number(scenario_target, 2) if scenario_target is not None else "engine output")}
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
        empty_state("Set a non-zero policy-rate shock to explore the scenario engine output.")
