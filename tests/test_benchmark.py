"""Agency benchmark: rescale, join against the engine, and agreement stats."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.benchmark.agency_reference import (
    band_for_reference_risk,
    benchmark_against_scores,
    load_agency_reference,
    reference_rows,
)
from src.scoring.risk_score import score_panel


def _demo_scores():
    path = Path(__file__).resolve().parents[1] / "data" / "demo" / "panel_wide.csv"
    scores, _, _ = score_panel(pd.read_csv(path))
    return scores


def test_load_agency_reference_has_entries():
    ref = load_agency_reference()
    assert ref["asof"]
    assert ref["entries"]


def test_reference_rows_rescale_to_0_100():
    rows = reference_rows()
    assert rows
    assert all(0.0 <= r["reference_risk"] <= 100.0 for r in rows)
    by_iso = {r["iso3"]: r for r in rows}
    assert by_iso["DEU"]["reference_risk"] < by_iso["TUR"]["reference_risk"]  # AAA safer than B
    assert by_iso["USA"]["S&P"] == "AA+"
    assert by_iso["USA"]["Moody's"] == "Aaa"


def test_reference_rows_ignore_unknown_ratings():
    rows = reference_rows({"asof": "x", "entries": [{"iso3": "ZZZ", "country": "Z", "S&P": "ZZ+"}]})
    assert rows == []


def test_band_for_reference_risk_thresholds():
    assert band_for_reference_risk(10) == "Low"
    assert band_for_reference_risk(30) == "Moderate"
    assert band_for_reference_risk(50) == "Elevated"
    assert band_for_reference_risk(70) == "High"
    assert band_for_reference_risk(90) == "Severe"


def test_benchmark_against_demo_scores():
    result = benchmark_against_scores(_demo_scores())
    assert result["n"] > 0
    assert set(result) >= {"asof", "n", "spearman", "band_match_share", "rows"}
    row = result["rows"][0]
    assert {"iso3", "engine_score", "engine_band", "reference_risk", "S&P", "Moody's", "Fitch"} <= set(row)
    assert 0.0 <= result["band_match_share"] <= 1.0


def test_benchmark_spearman_is_bounded_and_sane():
    result = benchmark_against_scores(_demo_scores())
    assert result["spearman"] is None or -1.0 <= result["spearman"] <= 1.0
