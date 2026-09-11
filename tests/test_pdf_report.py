"""PDF export: bytes are valid PDF and assemble_report_data copies the slice."""

from __future__ import annotations

import pandas as pd
import pytest

from dashboard.context import Context
from dashboard.sections.pdf_export import assemble_report_data
from src.reporting.pdf_report import (
    CountryRiskReportData,
    build_risk_report_pdf,
)
from src.scoring.risk_score import score_panel, top_drivers


def _ctx_from_panel(panel: pd.DataFrame) -> Context:
    scores, drivers, pillar_scores = score_panel(panel)
    iso = "USA"
    year = 2020
    row = scores[(scores["country_iso3"] == iso) & (scores["year"] == year)].iloc[0]
    return Context(
        panel=panel,
        scores=scores,
        drivers=drivers,
        pillar_scores=pillar_scores,
        country=iso,
        year=year,
        iso=iso,
        country_label="United States",
        band=str(row["risk_band"]),
        score_value=float(row["risk_score"]),
        coverage_value=float(row["data_completeness"]),
        country_drivers=top_drivers(drivers, iso, year, n=6),
        report="**United States — Risk Score: 30/100 (Moderate)**\n\n**Main drivers:**\n- first\n- second",
        scenario_result={
            "preset": "Policy-rate hike",
            "driver_code": "POLICY_RATE_YOY_CHANGE_BPS",
            "baseline_score": 30.0,
            "scenario_score": 33.2,
            "delta": 3.2,
            "baseline_band": "Moderate",
            "scenario_band": "Elevated",
            "information_assessment": "MODERATE INFORMATION",
            "narrative": "text",
            "out_of_sample_shock": False,
            "indicator_deltas": [
                {"indicator_code": "FX_YOY_DEPRECIATION_PCT", "estimated_delta": 1.25, "r_squared": 0.62, "n_obs": 60}
            ],
        },
        generated_at="2026-09-11 10:00:00",
        using_demo_data=True,
        data_verified=False,
        provenance_sources="World Bank, IMF",
        provenance_asof="2026Q3",
        manifest_hash="abc123",
    )


def test_build_risk_report_pdf_returns_valid_pdf_bytes(sample_wide_panel: pd.DataFrame) -> None:
    ctx = _ctx_from_panel(sample_wide_panel)
    data = assemble_report_data(ctx)
    data.trajectory_years = [2015, 2016, 2017, 2018, 2019, 2020]
    data.trajectory_scores = [30.0, 32.0, 31.0, 33.0, 34.0, 33.0]
    pdf_bytes = build_risk_report_pdf(data)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes[:5] == b"%PDF-"
    assert len(pdf_bytes) > 2000


def test_assemble_report_data_copies_slice_values(sample_wide_panel: pd.DataFrame) -> None:
    ctx = _ctx_from_panel(sample_wide_panel)
    data = assemble_report_data(ctx)
    assert data.iso3 == "USA"
    assert data.year == 2020
    assert data.band == ctx.band
    assert data.risk_score == pytest.approx(ctx.score_value, abs=0.1)
    assert data.using_demo_data is True
    assert data.provenance_sources == "World Bank, IMF"
    assert data.manifest_hash == "abc123"
    assert "Risk Score" in data.report_text
    assert "**" not in data.report_text
    assert any(p.name for p in data.pillars)
    assert any(d.label for d in data.drivers)
    assert data.scenario is not None
    assert data.scenario.delta == pytest.approx(3.2)
    assert data.scenario.channels[0].indicator == "FX_YOY_DEPRECIATION_PCT"
    assert data.trajectory_years == data.trajectory_scores == []  # filled by the render helper


def test_pdf_builds_without_scenario_or_pillars() -> None:
    data = CountryRiskReportData(
        country_label="Testland",
        iso3="TST",
        year=2024,
        risk_score=55.0,
        band="Elevated",
        coverage_value=None,
        trend_direction="Deteriorating",
        trend_1y_delta=4.2,
        report_text="A short note with no extra tables.",
        sources=[],
        generated_at="now",
        using_demo_data=False,
        data_verified=True,
        provenance_sources="",
        provenance_asof="",
        manifest_hash="",
        pillars=[],
        drivers=[],
        scenario=None,
    )
    pdf_bytes = build_risk_report_pdf(data)
    assert pdf_bytes[:5] == b"%PDF-"
    assert len(pdf_bytes) > 1000


def test_snapshot_coverage_nan_renders_as_dash(sample_wide_panel: pd.DataFrame) -> None:
    ctx = _ctx_from_panel(sample_wide_panel)
    ctx.coverage_value = float("nan")
    data = assemble_report_data(ctx)
    assert data.coverage_value is None
    pdf_bytes = build_risk_report_pdf(data)
    assert pdf_bytes[:5] == b"%PDF-"
