"""Track Record page — "would this model have caught it?"

The proof layer: every configured historical crisis episode is replayed
against the engine's own composite scores, with a verdict (flagged / missed /
inconclusive), the score path baseline -> event -> peak, lead time, the
dominant pillar and sector that actually moved, and a peer-drift diagnostic.
Alongside the hit rate we show a confusion-matrix row (precision, recall,
false-positive/negative rates, warning frequency) built from the same rise
rule applied to ordinary country-years — so the hit rate is never presented
in isolation.
"""

from __future__ import annotations

import streamlit as st

from dashboard.context import Context
from dashboard.ui import esc
from src.analysis.backtest import (
    backtest_metrics,
    backtest_summary,
    false_alarm_rate,
    run_backtest,
)


def _render_episode_cards(results) -> None:
    verdict_style = {
        "flagged": ("var(--green,#54d69a)", "✓ FLAGGED"),
        "missed": ("var(--red,#ff6b7a)", "✗ MISSED"),
        "inconclusive": ("var(--muted,#8c98aa)", "— NO DATA"),
    }

    for r in results:
        color, tag = verdict_style[r.verdict]
        if r.delta is not None:
            detail = (
                f"Score {r.baseline_year}: {r.baseline_score:.1f} &rarr; {r.event_year}: {r.event_score:.1f} "
                f"({r.delta:+.1f} pts at event)"
            )
            if r.peak_year != r.event_year and r.peak_score is not None:
                detail += f" &middot; peak {r.peak_year}: {r.peak_score:.1f} ({r.peak_delta:+.1f} pts)"
        else:
            detail = f"No scored data for {r.baseline_year} and/or {r.event_year}+ window in the current panel."

        diagnostics = []
        if r.verdict == "flagged" and r.peak_delta is not None:
            diagnostics.append(f"rise {r.peak_delta:+.1f} pts &middot; threshold &ge; {r.threshold_points:g}")
        if r.verdict == "flagged" and r.lead_time is not None:
            lead = f"{r.lead_time}" if r.lead_time > 0 else "same year"
            diagnostics.append(f"lead time ~{lead} yr before {r.event_year}")
        if r.peer_drift is not None:
            diagnostics.append(f"{r.peer_drift:.0%} of other panel countries rose >= threshold")
        if r.verdict == "flagged":
            if r.dominant_pillar:
                diagnostics.append(
                    f"dominant pillar: {esc(r.dominant_pillar.replace('pillar_', '').replace('_score', ''))}"
                )
            if r.dominant_sector:
                diagnostics.append(
                    f"dominant sector: {esc(r.dominant_sector.replace('sector_', '').replace('_score', ''))}"
                )
        extra = (
            f"<div class='card-caption' style='margin-top:6px;'>{' &middot; '.join(diagnostics)}</div>"
            if diagnostics
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


def _render_kpi_row(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items), gap="small")
    for col, (label, value, caption) in zip(cols, items, strict=False):
        with col:
            st.markdown(
                f"""
                <div class="card kpi">
                    <div class="kpi-accent"></div>
                    <div class="card-label">{esc(label)}</div>
                    <div class="card-value" style="font-size:22px;">{esc(value)}</div>
                    <div class="card-caption">{esc(caption)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_track_record(ctx: Context) -> None:
    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">Track record</div>
                <div class="section-sub">
                    Would this model have caught it? Every known historical crisis, replayed against
                    the engine's own scores — an honest hit-rate, not a claim.
                </div>
            </div>
            <div class="micro">INSTANT REPLAY / PROOF LAYER</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if ctx.using_demo_data:
        st.warning(
            "Running on synthetic demo data — the replay below is NOT a real validation of "
            "anything. Switch to live data for the actual track record."
        )

    results = run_backtest(ctx.scores, data_is_synthetic=ctx.using_demo_data)
    summary = backtest_summary(results)
    bt = backtest_metrics(results, ctx.scores)
    fpr = false_alarm_rate(ctx.scores)

    k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")

    cols = [k1, k2, k3, k4, k5, k6]
    labels = [
        ("EPISODES", f"{summary.episodes_total}", "configured crises"),
        ("FLAGGED", f"{summary.flagged}", "caught before/during"),
        ("MISSED", f"{summary.missed}", "not caught"),
        ("NO DATA", f"{summary.inconclusive}", "outside panel window"),
        (
            "DETECTION RATE",
            f"{summary.detection_rate:.0%}" if summary.detection_rate is not None else "—",
            "flagged of evaluable",
        ),
        (
            "FALSE-ALARM RATE",
            f"{fpr:.0%}" if fpr is not None else "—",
            "rule trips on non-crisis years",
        ),
    ]

    for col, (label, value, caption) in zip(cols, labels, strict=False):
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

    _render_kpi_row(
        [
            (
                "MEDIAN LEAD TIME",
                f"{bt.median_lead_time:g} yrs" if bt.median_lead_time is not None else "—",
                "before event_year warning fired",
            ),
            ("PRECISION", f"{bt.precision:.0%}" if bt.precision is not None else "—", "flagged that were real"),
            ("RECALL", f"{bt.recall:.0%}" if bt.recall is not None else "—", "crises the rule caught"),
            (
                "FALSE-POSITIVE RATE",
                f"{bt.false_positive_rate:.0%}" if bt.false_positive_rate is not None else "—",
                "ordinary-year alarms",
            ),
            (
                "FALSE-NEGATIVE RATE",
                f"{bt.false_negative_rate:.0%}" if bt.false_negative_rate is not None else "—",
                "crises the rule missed",
            ),
            (
                "WARNING FREQUENCY",
                f"{bt.warning_frequency:.0%}" if bt.warning_frequency is not None else "—",
                "all windows that tripped",
            ),
        ]
    )

    st.markdown(
        f"""
        <div class="card" style="margin:16px 0;">
            <div class="card-caption">
                The detection rate counts only episodes the panel can actually score. The
                confusion-matrix row sits below: episodes themselves are the positives
                (flagged = true positive, missed = false negative), and every other
                country-year window in the panel is a negative (tripped = false positive,
                quiet = true negative). Read them together — a high hit rate on a rule that
                cries wolf is worth less than one that stays quiet.
                {" · median peak rise " + f"{summary.median_peak_delta:+.1f} pts" if summary.median_peak_delta is not None else ""}
                {" · median lead time " + f"{bt.median_lead_time:g} yrs" if bt.median_lead_time is not None else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_episode_cards(results)

    with st.expander("How to read this honestly"):
        st.markdown(
            """
**Flagged** means the composite score rose by at least the threshold (default 3 points)
from the baseline year to the peak inside the detection window (event year plus one) —
a real signal, not noise. The dominant pillar/sector that actually moved is named.
**Lead time** is how many years before the event year that same rule first fired when you
stand at every year from the baseline forward — "the model was already warning N years early."
**Missed** means it did not rise enough — a genuine limitation being named, not hidden.
**No data** means the panel does not currently span both years for that country —
expand the fetch window and this resolves itself.
**Peer drift** says whether the move was country-specific or just the whole panel moving
(low drift = high information in the signal).

**About the confusion-matrix row:** positives are the episodes, negatives are every other
country-year window in the panel. **Precision** = flagged episodes / all windows the rule
flagged; **recall** (== detection rate) = flagged episodes / evaluable episodes;
**false-positive rate** = ordinary windows tripped / ordinary windows; **false-negative rate**
= missed episodes / evaluable episodes; **warning frequency** = share of ALL windows
(episodes + ordinary years) where the rule fired. The negative side is the model's own rule
applied to every country-year that is not part of a known episode — a base rate, not a
perfect false-positive rate (there is no authoritative list of "non-crises"), and episode
windows are excluded so known crises never inflate their own benchmark. On a small panel
with only a handful of episodes, treat every number here as indicative, not statistical
evidence.
            """
        )
