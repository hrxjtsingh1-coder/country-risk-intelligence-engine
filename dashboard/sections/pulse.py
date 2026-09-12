"""Global Pulse landing page.

This module is intentionally presentation-only. It turns the already-scored
panel into a small orientation layer: what the global panel looks like now,
where risk is rising, and where it is easing. The scoring engine remains the
single source of truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.context import Context
from dashboard.ui import (
    band_color,
    esc,
    fmt_delta,
    fmt_number,
    get_country_label,
    normalize_band,
    safe_float,
)


_BAND_ORDER = {"Low": 0, "Moderate": 1, "Elevated": 2, "High": 3, "Severe": 4}


def _slice(scores: pd.DataFrame, year: int) -> pd.DataFrame:
    if scores.empty or "year" not in scores.columns:
        return pd.DataFrame()
    result = scores[pd.to_numeric(scores["year"], errors="coerce").eq(int(year))].copy()
    if "country_iso3" not in result.columns:
        return pd.DataFrame()
    result["country_iso3"] = result["country_iso3"].astype(str)
    if "risk_score" not in result.columns:
        result["risk_score"] = np.nan
    result["risk_score"] = pd.to_numeric(result["risk_score"], errors="coerce")
    result["risk_band"] = result.get("risk_band", result["risk_score"].map(lambda x: "Elevated" if pd.isna(x) else ""))
    result["risk_band"] = result.apply(
        lambda row: normalize_band(row["risk_band"])
        if str(row.get("risk_band", "")).strip()
        else ("Severe" if row["risk_score"] >= 80 else "High" if row["risk_score"] >= 60 else "Elevated"),
        axis=1,
    )
    result = result.dropna(subset=["risk_score"]).drop_duplicates("country_iso3")
    result["country_name"] = result["country_iso3"].map(get_country_label)
    return result


def _with_change(ctx: Context) -> pd.DataFrame:
    current = _slice(ctx.scores, ctx.year)
    previous = _slice(ctx.scores, ctx.year - 1)
    if current.empty:
        return current
    if previous.empty:
        current["change"] = np.nan
    else:
        previous = previous[["country_iso3", "risk_score"]].rename(columns={"risk_score": "previous_score"})
        current = current.merge(previous, on="country_iso3", how="left")
        current["change"] = current["risk_score"] - current["previous_score"]
    return current


def _metric_card(label: str, value: str, caption: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="pulse-metric">
            <div class="pulse-accent" style="background:{accent};"></div>
            <div class="card-label">{esc(label)}</div>
            <div class="pulse-value">{esc(value)}</div>
            <div class="card-caption">{esc(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _ranking(title: str, rows: pd.DataFrame, ctx: Context, direction: str) -> None:
    st.markdown(f'<div class="pulse-ranking-title">{esc(title)}</div>', unsafe_allow_html=True)
    if rows.empty:
        st.caption("Not enough year-on-year observations.")
        return

    for idx, item in enumerate(rows.head(5).itertuples(index=False)):
        iso = str(item.country_iso3)
        name = str(item.country_name)
        score = safe_float(item.risk_score)
        change = safe_float(getattr(item, "change", np.nan), default=float("nan"))
        band = normalize_band(getattr(item, "risk_band", "Elevated"))
        row_cols = st.columns([2.2, 0.65, 0.8])
        with row_cols[0]:
            if st.button(name, key=f"pulse_{direction}_{iso}_{idx}", width="stretch"):
                st.session_state["country_selector"] = iso
                st.session_state["page_nav"] = "Country Intelligence"
                st.rerun()
        with row_cols[1]:
            st.markdown(f'<div class="pulse-score">{score:.1f}</div>', unsafe_allow_html=True)
        with row_cols[2]:
            change_text = "—" if np.isnan(change) else fmt_delta(change)
            st.markdown(
                f'<div class="pulse-change" style="color:{band_color(band)};">{esc(change_text)}</div>',
                unsafe_allow_html=True,
            )


def render_global_pulse(ctx: Context) -> None:
    """Render the concise, orientation-first landing page."""
    data = _with_change(ctx)
    covered = len(data)
    elevated_plus = int(data["risk_band"].map(lambda band: _BAND_ORDER.get(normalize_band(band), 0) >= 2).sum()) if covered else 0
    median = safe_float(data["risk_score"].median(), default=float("nan")) if covered else float("nan")
    coverage_col = "data_completeness" if "data_completeness" in data.columns else None
    coverage = safe_float(pd.to_numeric(data[coverage_col], errors="coerce").median(), default=float("nan")) if coverage_col else float("nan")

    deteriorating = data.dropna(subset=["change"]).sort_values("change", ascending=False)
    improving = data.dropna(subset=["change"]).sort_values("change", ascending=True)
    fastest = deteriorating.iloc[0]["country_name"] if not deteriorating.empty else "—"
    improving_name = improving.iloc[0]["country_name"] if not improving.empty else "—"

    st.markdown(
        """
        <div class="pulse-intro">
            <div>
                <div class="kicker">GLOBAL PULSE / ORIENTATION</div>
                <div class="pulse-title">What is happening across countries?</div>
                <div class="pulse-copy">
                    Start with the panel, then open a country when a signal deserves a closer look.
                    Scores are positioning signals—not forecasts or credit ratings.
                </div>
            </div>
            <div class="pulse-question">
                <span>READING THIS VIEW</span>
                <strong>Risk · change · coverage</strong>
                <small>Use the map and rankings to choose the next question.</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(6)
    metrics = [
        ("COUNTRIES COVERED", f"{covered}", f"scored in {ctx.year}", "var(--cyan)"),
        ("ELEVATED+", f"{elevated_plus}", "countries needing attention", "var(--orange)"),
        ("MEDIAN RISK", "—" if np.isnan(median) else f"{median:.1f}", "panel score / 100", "var(--violet)"),
        ("FASTEST RISE", esc(fastest), "largest 1Y score increase", "var(--red)"),
        ("MOST IMPROVING", esc(improving_name), "largest 1Y score decrease", "var(--green)"),
        ("MEDIAN COVERAGE", "—" if np.isnan(coverage) else f"{coverage:.0%}", "indicator completeness", "var(--blue)"),
    ]
    for col, (label, value, caption, accent) in zip(metric_cols, metrics):
        with col:
            _metric_card(label, value, caption, accent)

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">Global watch</div>
                <div class="section-sub">A short list of countries worth opening next.</div>
            </div>
            <div class="micro">DERIVED FROM SCORED PANEL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    rank_cols = st.columns(3)
    with rank_cols[0]:
        _ranking("HIGHEST RISK", data.sort_values("risk_score", ascending=False), ctx, "risk")
    with rank_cols[1]:
        _ranking("FASTEST DETERIORATION", deteriorating, ctx, "rise")
    with rank_cols[2]:
        _ranking("MOST IMPROVING", improving, ctx, "improve")
