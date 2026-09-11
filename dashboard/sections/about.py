"""About page.

What this engine is, what it is deliberately not, how the numbers are
produced, and where the data comes from. Rendered as the closing page
before the footer; every claim here reflects the repo as it actually runs
(deterministic, offline-runnable, config-driven, CI-governed).
"""

from __future__ import annotations

import streamlit as st

from dashboard.context import Context
from dashboard.ui import esc


def render_about(ctx: Context) -> None:
    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">About this engine</div>
                <div class="section-sub">
                    What it is, how the numbers are made, and where the data comes from.
                </div>
            </div>
            <div class="micro">ABOUT / METHOD / SOURCES</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _card(
        "What it is",
        (
            "A transparent, deterministic country-risk prior. Each country-year is scored "
            "0–100 and placed in one of five risk bands (Low / Moderate / Elevated / High / "
            "Severe), with drivers decomposed to the indicator level so any score can be "
            "traced back to the exact numbers and weights that produced it."
        ),
    )

    _card(
        "What it is not",
        (
            "Not a credit rating, not a forecast, and not an LLM opinion. The score is a "
            "relative-positioning signal within the panel — a prioritization tool for where "
            "to look closer, not a substitute for that closer look. Commentary is rule-based "
            "text generated from computed numbers; there is no generative model anywhere in "
            "the pipeline."
        ),
    )

    _card(
        "How scores are built",
        (
            "Raw indicators from the config-driven collectors enter a wide panel, and each "
            "indicator is z-scored against both the country's own history (minimum five "
            "years) and its peer group in the same year, then blended — a side with too few "
            "observations is dropped rather than forcing garbage into the blend. Pillar "
            "scores feed a configurable sector aggregate and a composite, renormalized "
            "around missing data. Scoring, sensitivity and text generation all run in "
            "src/ and are exercised by the test suite."
        ),
    )

    _card(
        "Scenario & sensitivity",
        (
            "The scenario laboratory reshocks a named preset or a custom driver and estimates "
            "pooled-panel bivariate OLS sensitivities from the panel itself (no hardcoded "
            "elasticities). Every run reports the estimation window, per-channel R² and n, an "
            "information assessment, an extrapolation flag when the shock leaves the observed "
            "driver range, and an engine-level narrative sentence. The commentary adds a "
            "data-quality confidence flag and a source + timestamp trail."
        ),
    )

    _card(
        "Data sources",
        (
            "Primary public sources are declared per indicator in config/indicators.yaml — "
            "World Bank indicators, FRED policy rates, BIS banking statistics, IMF IFS "
            "exchange rates, plus clearly-labeled derived series. Live collection can run "
            "absent; the repository ships with demo and committed fixtures so everything is "
            "reproducible offline."
        ),
    )

    _card(
        "Governance",
        (
            "Two blocking CI jobs police drift: a model-coverage check that every configured "
            "indicator is resolved by the committed demo fixture, and a build→verify "
            "reproducibility manifest roundtrip. Pipelines, scoring and commentary are "
            "byte-deterministic for fixed inputs, and this page renders from the same "
            "computed state as every other panel."
        ),
    )

    st.markdown(
        f"""
        <div class="card" style="margin-top:16px;">
            <div class="micro" style="margin-bottom:8px;">
                CURRENT RUN · {esc(ctx.iso)} / {int(ctx.year)}
                · {"DEMO PANEL" if ctx.using_demo_data else ("CACHED PANEL" if getattr(ctx, "using_cached_data", False) else "LIVE PANEL")}
            </div>
            <div style="color:#bdc8d6;font-size:12px;line-height:1.75;">
                Showing {esc(ctx.country_label)} · composite {ctx.score_value:.1f}/100
                · report generated at {esc(ctx.generated_at)}.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="card" style="margin-bottom:16px;">
            <div class="card-label">{esc(title)}</div>
            <div style="margin-top:12px;color:#bdc8d6;font-size:13px;line-height:1.8;">
                {esc(body)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
