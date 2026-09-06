import pandas as pd
import pytest
from pathlib import Path

from src.scoring.risk_score import score_panel
from src.scenario.scenario_engine import run_shock_scenario
from src.cleaning.clean import clean_long_panel


def panel():
    return pd.DataFrame([
        {"country_iso3": c, "year": y, "FP.CPI.TOTL.ZG": i + y - 2020,
         "NY.GDP.MKTP.KD.ZG": 10 - i, "SL.UEM.TOTL.ZS": i + 1,
         "GC.DOD.TOTL.GD.ZS": 40 + i, "BN.CAB.XOKA.GD.ZS": -i,
         "FI.RES.TOTL.MO": 8 - i / 5, "DT.DOD.DECT.GN.ZS": 20 + i,
         "FB.AST.NPER.ZS": 1 + i / 10, "FX_YOY_DEPRECIATION_PCT": i,
         "POLICY_RATE_YOY_CHANGE_BPS": i * 10}
        for y in range(2020, 2026) for i, c in enumerate(["USA", "CAN", "DEU", "IND", "BRA"], 1)
    ])


def test_score_completeness_is_fraction_and_risk_direction():
    scores, drivers = score_panel(panel())
    assert scores["data_completeness"].eq(1.0).all()
    debt = drivers[drivers.indicator_code.eq("GC.DOD.TOTL.GD.ZS")]
    assert debt.sort_values("raw_value").weighted_contribution.is_monotonic_increasing


def test_score_missing_data_reduces_completeness():
    data = panel(); data.loc[0, "FP.CPI.TOTL.ZG"] = None
    scores, _ = score_panel(data)
    assert scores.loc[(scores.country_iso3 == "USA") & (scores.year == 2020), "data_completeness"].iloc[0] < 1


def test_duplicate_cleaning_and_range_flag():
    clean = clean_long_panel(pd.DataFrame([
        {"country_iso3":"usa", "indicator_code":"FP.CPI.TOTL.ZG", "year":2024, "value": 2},
        {"country_iso3":"USA", "indicator_code":"FP.CPI.TOTL.ZG", "year":2024, "value": 200},
    ]))
    assert len(clean) == 1 and clean.iloc[0].flag == "out_of_range"


def test_scenario_reports_information_and_extrapolation():
    result = run_shock_scenario(panel(), "USA", 2025, "POLICY_RATE_YOY_CHANGE_BPS", 10000, ["NY.GDP.MKTP.KD.ZG"])
    assert result["out_of_sample_shock"] is True
    assert result["information_assessment"] in {"LOW INFORMATION", "MODERATE INFORMATION", "HIGH INFORMATION"}
    assert "Pooled-panel" in result["model_specification"]


def test_scenario_rejects_zero_variance_driver():
    data = panel(); data["POLICY_RATE_YOY_CHANGE_BPS"] = 1
    result = run_shock_scenario(data, "USA", 2025, "POLICY_RATE_YOY_CHANGE_BPS", 25, ["NY.GDP.MKTP.KD.ZG"])
    assert result["information_assessment"] == "INSUFFICIENT DATA"


def test_dashboard_requires_explicit_demo_selection():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard" / "app.py", default_timeout=30)
    app.run(timeout=30)
    assert not app.exception
    assert [button.label for button in app.button] == ["Open Demo Dataset"]
    app.button[0].click()
    app.run(timeout=30)
    assert not app.exception
    assert any("DEMO DATA — SYNTHETIC DATASET" in str(alert.value) for alert in app.error)
