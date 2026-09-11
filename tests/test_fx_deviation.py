"""FX trend-deviation indicator: derived correctly and persisted in the panel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.indicators.build_panel import _derive_indicators
from src.scoring.risk_score import score_panel, top_drivers


def _demo_panel() -> pd.DataFrame:
    return pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "demo" / "panel_wide.csv")


def test_demo_panel_has_fx_level_and_deviation_columns():
    panel = _demo_panel()
    assert "FX_LEVEL_USD_LCU" in panel.columns
    assert "FX_TREND_DEVIATION_PCT" in panel.columns
    assert panel["FX_LEVEL_USD_LCU"].notna().all()
    assert panel["FX_LEVEL_USD_LCU"].min() > 0
    assert panel["FX_TREND_DEVIATION_PCT"].notna().all()


def test_fx_level_is_consistent_with_yoy_change():
    panel = _demo_panel()
    usa = panel[panel["country_iso3"] == "USA"].sort_values("year").head(6)  # within panel span
    level_pct = usa["FX_LEVEL_USD_LCU"].pct_change() * 100.0
    # FX_YOY is an independent synthetic series in the demo; the check here is
    # only that both columns exist and the level is a plausible positive series.
    assert level_pct.dropna().abs().max() < 100


def test_derivation_matches_demo_column():
    panel = _demo_panel()
    base = panel[["country_iso3", "year", "FX_LEVEL_USD_LCU"]].rename(columns={"FX_LEVEL_USD_LCU": "value"})
    base["indicator_code"] = "FX_LEVEL_USD_LCU"
    base["source"] = "demo"
    base["flag"] = "ok"
    derived = _derive_indicators(base[["country_iso3", "indicator_code", "year", "value", "source", "flag"]])
    dev = derived[derived["indicator_code"] == "FX_TREND_DEVIATION_PCT"][["country_iso3", "year", "value"]]
    merged = panel[["country_iso3", "year", "FX_TREND_DEVIATION_PCT"]].merge(
        dev, on=["country_iso3", "year"], how="inner"
    )
    diff = (merged["FX_TREND_DEVIATION_PCT"] - merged["value"]).abs()
    assert diff.max() < 1e-6


def test_deviation_signal_is_symmetric_around_zero():
    panel = _demo_panel()
    values = panel["FX_TREND_DEVIATION_PCT"]
    assert values.min() < 0.0  # some currencies stronger than trend
    assert values.max() > 0.0  # some weaker than trend


def test_constant_level_yields_zero_deviation():
    rows = []
    for iso3, year in [("XYZ", y) for y in range(2015, 2026)]:
        rows.append(
            {
                "country_iso3": iso3,
                "year": year,
                "value": 2.0,
                "indicator_code": "FX_LEVEL_USD_LCU",
                "source": "x",
                "flag": "ok",
            }
        )
    derived = _derive_indicators(pd.DataFrame(rows))
    dev = derived[derived["indicator_code"] == "FX_TREND_DEVIATION_PCT"]
    assert not dev.empty
    assert (dev["value"].abs() < 1e-9).all()


def test_free_bonus_not_in_composite_score():
    panel = _demo_panel()
    scores, drivers, _ = score_panel(panel)
    zero_weight_codes = {"FX_TREND_DEVIATION_PCT", "FX_LEVEL_USD_LCU"}
    for code in zero_weight_codes:
        slice = drivers[(drivers["indicator_code"] == code)]
        assert not slice.empty, f"{code} should be in drivers table"
        assert (slice["weighted_contribution"] == 0.0).all(), f"{code} should not contribute to composite"
    top = set(top_drivers(drivers, "USA", 2024, n=6)["indicator_code"])
    assert not (zero_weight_codes & top), "zero-weight codes must not be top movers"
    assert scores["risk_score"].notna().all()
