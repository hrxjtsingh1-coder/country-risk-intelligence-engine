"""Benchmark vs real agency ratings (S&P / Moody's / Fitch).

Reference-only: the static snapshot in config/agency_ratings.yaml is used to
see whether the engine's country *ordering* agrees with agency views, via rank
correlation and band-match share. Ratings are not scores and not outcomes — the
caveat is stated in the UI and the reference file is not part of the scoring
pipeline.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.ui import esc  # noqa: E402
from src.benchmark.agency_reference import band_for_reference_risk, benchmark_against_scores, load_agency_reference


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def render_agency_benchmark(ctx) -> None:
    reference = load_agency_reference()
    result = benchmark_against_scores(ctx.scores, reference)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">BENCHMARK VS AGENCY RATINGS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-caption" style="margin-bottom:12px;">How the engine\'s country ordering compares with '
        "S&amp;P / Moody's / Fitch long-term issuer ratings. Reference-only — not advice, not an outcome.</div>",
        unsafe_allow_html=True,
    )

    if result["n"] == 0:
        st.markdown(
            '<div class="card-caption">No scored countries match the reference set.</div>', unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    cols = st.columns(3)
    cols[0].markdown(
        f'<div class="card"><div class="card-label" style="font-size:11px;">COUNTRIES MATCHED</div>'
        f'<div class="card-value">{result["n"]}</div></div>',
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        f'<div class="card"><div class="card-label" style="font-size:11px;">RANK AGREEMENT (SPEARMAN)</div>'
        f'<div class="card-value">{esc(result["spearman"] if result["spearman"] is not None else "—")}</div></div>',
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        f'<div class="card"><div class="card-label" style="font-size:11px;">BAND MATCH SHARE</div>'
        f'<div class="card-value">{esc(_percent(result["band_match_share"]))}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="card-caption" style="margin:10px 0 8px;">Reference snapshot: {esc(result["asof"])}. '
        "Ratings rescaled to the same 0-100 banding; matching bands is a bonus, ordering agreement is the point.</div>",
        unsafe_allow_html=True,
    )

    rows = []
    for r in result["rows"]:
        match = r.get("bands_match")
        rows.append(
            {
                "Country": r["country"],
                "Engine score": r["engine_score"],
                "Engine band": r["engine_band"],
                "Ref. risk (0-100)": r["reference_risk"],
                "Ref. band": band_for_reference_risk(r["reference_risk"]),
                "Band match": "Yes" if match else ("No" if match is False else "—"),
                "S&P": r["S&P"] or "—",
                "Moody's": r["Moody's"] or "—",
                "Fitch": r["Fitch"] or "—",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if ctx.using_demo_data:
        st.markdown(
            '<div class="card-caption" style="margin-top:10px;"><b>DEMO PANEL</b> — engine scores here are from the '
            "synthetic fixture; the comparison is a methodology demonstration only. Agency letters are a static, "
            "dated reference snapshot.</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
