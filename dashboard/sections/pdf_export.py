"""One-click per-country PDF export.

The export formats what the app already computed — the same scores, drivers,
pillar breakdown, scenario output and analyst commentary shown on the Overview
page. `assemble_report_data` copies those values off `Context` into a plain
`CountryRiskReportData`; `render_pdf_export` renders a download button that
hands `build_risk_report_pdf` those already-computed values. Nothing is
re-scored or re-shaped here.
"""

from __future__ import annotations

import math
import re

import pandas as pd
import streamlit as st

from dashboard.ui import esc, fmt_number  # noqa: E402
from src.reporting.pdf_report import (
    CountryRiskReportData,
    DriverRow,
    PillarRow,
    ScenarioChannelRow,
    ScenarioRow,
    build_risk_report_pdf,
)


def _threshold(x: object) -> float | None:
    try:
        if x is None:
            return None
        value = float(x)
        if math.isnan(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def _plain(markdown_text: str) -> str:
    """Loose markdown -> plain-text conversion for the PDF body."""
    text = str(markdown_text or "")
    text = text.replace("`", "")
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "•  ", text, flags=re.MULTILINE)
    return text.strip()


def _trend_values(ctx) -> tuple[str, float | None]:
    if not isinstance(ctx.scores, pd.DataFrame) or ctx.scores.empty:
        return "", None
    row = ctx.scores[
        ctx.scores["country_iso3"].astype(str).eq(str(ctx.country))
        & pd.to_numeric(ctx.scores["year"], errors="coerce").eq(int(ctx.year))
    ]
    if row.empty:
        return "", None
    direction = ""
    if "trend_direction" in row.columns:
        value = row.iloc[0].get("trend_direction")
        if pd.notna(value):
            direction = str(value)
    delta = _threshold(row.iloc[0].get("trend_1y_delta")) if "trend_1y_delta" in row.columns else None
    return direction, delta


def _assemble_pillars(ctx) -> list[PillarRow]:
    if not isinstance(ctx.pillar_scores, pd.DataFrame) or ctx.pillar_scores.empty:
        return []
    frame = ctx.pillar_scores[
        ctx.pillar_scores["country_iso3"].astype(str).eq(str(ctx.country))
        & pd.to_numeric(ctx.pillar_scores["year"], errors="coerce").eq(int(ctx.year))
    ]
    rows: list[PillarRow] = []
    for _, r in frame.iterrows():
        rows.append(
            PillarRow(
                name=str(r.get("pillar", "")),
                score=_threshold(r.get("pillar_score")),
                band=str(r.get("pillar_band", "") or ""),
            )
        )
    rows.sort(key=lambda p: (p.score is None, -(p.score or float("-inf"))))
    return rows


def _assemble_drivers(ctx) -> list[DriverRow]:
    if not isinstance(ctx.country_drivers, pd.DataFrame) or ctx.country_drivers.empty:
        return []
    rows: list[DriverRow] = []
    for _, r in ctx.country_drivers.iterrows():
        raw = _threshold(r.get("raw_value"))
        unit = str(r.get("unit", "") or "").strip()
        if raw is not None:
            value_text = f"{fmt_number(raw, 1)}{(' ' + unit) if unit else ''}"
        else:
            value_text = "—"
        label = str(r.get("label", "") or r.get("indicator_code", "") or "")
        pts = _threshold(r.get("weighted_contribution"))
        rows.append(DriverRow(label=label, value_text=value_text, contribution_points=(pts or 0.0) * 100))
    return rows


def _assemble_scenario(ctx) -> ScenarioRow | None:
    sc = getattr(ctx, "scenario_result", None)
    if not isinstance(sc, dict):
        return None

    baseline = _threshold(sc.get("baseline_score"))
    scenario_score = _threshold(sc.get("scenario_score"))
    if baseline is None or scenario_score is None:
        return None

    channels: list[ScenarioChannelRow] = []
    for ch in sc.get("indicator_deltas", []) or []:
        if not isinstance(ch, dict):
            continue
        estimated = _threshold(ch.get("estimated_delta"))
        if estimated is None:
            continue
        channels.append(
            ScenarioChannelRow(
                indicator=str(ch.get("indicator_code", "") or ""),
                estimated_delta=estimated,
                r_squared=_threshold(ch.get("r_squared")) or 0.0,
                n_obs=int(ch.get("n_obs") or 0),
            )
        )

    return ScenarioRow(
        name=str(sc.get("preset") or "Custom"),
        driver_code=str(sc.get("driver_code") or ""),
        baseline_score=baseline,
        scenario_score=scenario_score,
        delta=_threshold(sc.get("delta")) or (scenario_score - baseline),
        baseline_band=str(sc.get("baseline_band") or ""),
        scenario_band=str(sc.get("scenario_band") or ""),
        information=str(sc.get("information_assessment") or ""),
        narrative=str(sc.get("narrative") or ""),
        out_of_sample=bool(sc.get("out_of_sample_shock")),
        channels=channels,
    )


def _assemble_trajectory(ctx) -> tuple[list[int], list[float]]:
    if not isinstance(ctx.scores, pd.DataFrame) or ctx.scores.empty:
        return [], []
    frame = ctx.scores[ctx.scores["country_iso3"].astype(str).eq(str(ctx.country))].copy()
    if frame.empty:
        return [], []
    frame["_year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["_score"] = pd.to_numeric(frame["risk_score"], errors="coerce")
    frame = frame.dropna(subset=["_year", "_score"]).sort_values("_year")
    if frame.empty:
        return [], []
    return [int(y) for y in frame["_year"].tolist()], [float(s) for s in frame["_score"].tolist()]


def _sources(ctx) -> list[str]:
    sources: list[str] = []
    for part in str(getattr(ctx, "provenance_sources", "") or "").split(","):
        clean = part.strip()
        if clean:
            sources.append(clean)
    if not sources:
        sources.append("Indicator-level sources per config/indicators.yaml")
    return sources


def assemble_report_data(ctx) -> CountryRiskReportData:
    """Copy the current app slice into the PDF builder's plain data object."""
    trend_direction, trend_1y_delta = _trend_values(ctx)
    return CountryRiskReportData(
        country_label=str(ctx.country_label or ctx.country),
        iso3=str(ctx.iso or ctx.country),
        year=int(ctx.year or 0),
        risk_score=round(float(ctx.score_value or 0.0), 1),
        band=str(ctx.band or ""),
        coverage_value=_threshold(ctx.coverage_value),
        trend_direction=trend_direction,
        trend_1y_delta=trend_1y_delta,
        report_text=_plain(ctx.report),
        sources=_sources(ctx),
        generated_at=str(ctx.generated_at or ""),
        using_demo_data=bool(ctx.using_demo_data),
        data_verified=bool(ctx.data_verified),
        provenance_sources=str(getattr(ctx, "provenance_sources", "") or ""),
        provenance_asof=str(getattr(ctx, "provenance_asof", "") or ""),
        manifest_hash=str(getattr(ctx, "manifest_hash", "") or ""),
        pillars=_assemble_pillars(ctx),
        drivers=_assemble_drivers(ctx),
        scenario=_assemble_scenario(ctx),
        trajectory_years=[],  # filled below to keep the dataclass construction linear
        trajectory_scores=[],
    )


def _with_trajectory(data: CountryRiskReportData, ctx) -> CountryRiskReportData:
    years, scores = _assemble_trajectory(ctx)
    data.trajectory_years = years
    data.trajectory_scores = scores
    return data


def render_pdf_export(ctx) -> None:
    """Download button that emits the current country-year risk note as PDF."""
    data = _with_trajectory(assemble_report_data(ctx), ctx)
    pdf_bytes = build_risk_report_pdf(data)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-label" style="margin-bottom:6px;">PDF EXPORT</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="card-caption" style="margin-bottom:8px;">Download this country-year note as a two-page '
        "institutional-style PDF — score, trend, pillars, drivers, scenario and provenance included.</div>",
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name=f"{esc(data.iso3)}-{data.year}-country-risk-report.pdf",
        mime="application/pdf",
    )
    st.markdown("</div>", unsafe_allow_html=True)
