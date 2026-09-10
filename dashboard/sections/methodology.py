"""Methodology page.

The honesty-and-interoperability layer: the backtest / model-validation
section, the methodology & model card, the CSV export row inspector and
the engine-integrity note describing exactly what the presentation layer
does and does not recompute.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.context import Context
from dashboard.ui import empty_state, esc
from src.analysis.backtest import backtest_summary, run_backtest


def render_model_validation(ctx: Context) -> None:
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

    if ctx.using_demo_data:
        st.warning(
            "Running on synthetic demo data — the results below are NOT a real "
            "validation of anything. Switch to live data to see the actual backtest."
        )

    _bt_results = run_backtest(ctx.scores, data_is_synthetic=ctx.using_demo_data)
    _bt_summary = backtest_summary(_bt_results)
    _bt_flagged = _bt_summary.flagged
    _bt_missed = _bt_summary.missed
    _bt_inconclusive = _bt_summary.inconclusive

    _bt_metrics = [
        f"<strong>{_bt_flagged}</strong> flagged",
        f"<strong>{_bt_missed}</strong> missed",
        f"<strong>{_bt_inconclusive}</strong> inconclusive (data gap)",
    ]
    if _bt_summary.detection_rate is not None:
        _bt_metrics.append(f"detection rate <strong>{_bt_summary.detection_rate:.0%}</strong>")
    if _bt_summary.median_peak_delta is not None:
        _bt_metrics.append(f"median peak delta <strong>{_bt_summary.median_peak_delta:+.1f} pts</strong>")

    st.markdown(
        f"""
        <div class="card" style="margin-bottom:16px;">
            <div class="card-caption">
                {" &middot; ".join(_bt_metrics)} &middot; {len(_bt_results)} known episodes checked
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
        if r.delta is not None:
            detail = (
                f"Score {r.baseline_year}: {r.baseline_score:.1f} &rarr; {r.event_year}: {r.event_score:.1f} "
                f"({r.delta:+.1f} pts at event)"
            )
            if r.peak_year != r.event_year and r.peak_score is not None:
                detail += f" &middot; peak {r.peak_year}: {r.peak_score:.1f} ({r.peak_delta:+.1f} pts)"
        else:
            detail = f"No scored data for {r.baseline_year} and/or {r.event_year}+ window in the current panel."

        diagnosics = []
        if r.verdict == "flagged" and r.peak_delta is not None:
            diagnosics.append(f"rise {r.peak_delta:+.1f} pts &middot; threshold &ge; {r.threshold_points:g}")
        if r.peer_drift is not None:
            diagnosics.append(f"{r.peer_drift:.0%} of other panel countries rose >= threshold")
        if r.verdict == "flagged":
            if r.dominant_pillar:
                diagnosics.append(
                    f"dominant pillar: {esc(r.dominant_pillar.replace('pillar_', '').replace('_score', ''))}"
                )
            if r.dominant_sector:
                diagnosics.append(
                    f"dominant sector: {esc(r.dominant_sector.replace('sector_', '').replace('_score', ''))}"
                )
        extra = (
            f"<div class='card-caption' style='margin-top:6px;'>{' &middot; '.join(diagnosics)}</div>"
            if diagnosics
            else ""
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
                {extra}
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

**Flagged** means the composite score rose by at least the threshold (default
3 points) from the baseline year to the peak inside the detection window —
a real signal, not noise. The window is the event year plus one year, so a
fast-moving shock that registers just after the event still counts as caught,
and the dominant pillar/sector that actually moved is named.
**Missed** means it didn't rise enough, which is a genuine limitation worth
naming, not hiding, and **peer drift** says whether the move was country-specific
or just the rest of the panel moving with it (high drift = low information).
**No data** means the panel doesn't currently have both years scored for
that country — expand the fetch window to see it.
            """
        )


def render_model_card(ctx: Context) -> None:
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


def render_export_inspection(ctx: Context) -> None:
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
        csv_bytes = ctx.panel.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download panel CSV",
            data=csv_bytes,
            file_name=f"country_risk_panel_{int(ctx.year)}.csv",
            mime="text/csv",
            width="stretch",
        )

        st.caption("Exports the currently loaded dashboard panel exactly as provided to the UI.")

    with export_right:
        with st.expander("Inspect selected country-year row"):
            if ctx.current_row.empty:
                empty_state("No matching country-year row found.")
            else:
                selected_row = ctx.current_row.to_frame("value").reset_index()
                selected_row.columns = ["field", "value"]
                selected_row["field"] = selected_row["field"].astype(str)
                selected_row["value"] = selected_row["value"].astype(str)
                st.dataframe(
                    selected_row,
                    width="stretch",
                )

        with st.expander("Inspect top driver output"):
            if isinstance(ctx.country_drivers, pd.DataFrame) and not ctx.country_drivers.empty:
                st.dataframe(
                    ctx.country_drivers,
                    width="stretch",
                    hide_index=True,
                )
            else:
                empty_state("No driver table available.")


def render_engine_integrity(ctx: Context) -> None:
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
