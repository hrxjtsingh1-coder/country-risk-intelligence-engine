import pandas as pd

from src.commentary.generate_commentary import generate_report
from src.scenario.scenario_engine import run_shock_scenario
from src.scoring.risk_score import score_panel


def panel():
    return pd.DataFrame(
        [
            {
                "country_iso3": c,
                "year": y,
                "FP.CPI.TOTL.ZG": i + y - 2020,
                "NY.GDP.MKTP.KD.ZG": 10 - i,
                "SL.UEM.TOTL.ZS": i + 1,
                "OUTPUT_GAP_PROXY_PCT": i - 3,
                "GC.DOD.TOTL.GD.ZS": 40 + i,
                "GC.NLD.TOTL.GD.ZS": -2 - i,
                "PUBLIC_DEBT_TRAJECTORY_PCT": i * 0.5,
                "BN.CAB.XOKA.GD.ZS": -i,
                "FI.RES.TOTL.MO": 8 - i / 5,
                "DT.DOD.DECT.GN.ZS": 20 + i,
                "NE.RSB.GNFS.ZS": 2 - i,
                "FB.AST.NPER.ZS": 1 + i / 10,
                "FX_YOY_DEPRECIATION_PCT": i,
                "POLICY_RATE_YOY_CHANGE_BPS": i * 10,
            }
            for y in range(2020, 2026)
            for i, c in enumerate(["USA", "CAN", "DEU", "IND", "BRA"], 1)
        ]
    )


def _report(**overrides):
    data = panel()
    scores, drivers, _ = score_panel(data)
    kwargs = {
        "country_name": "United States",
        "country_iso3": "USA",
        "year": 2025,
        "scores": scores,
        "drivers": drivers,
        "scenario_result": None,
        "peer_group": ["CAN", "DEU"],
        "sources": ["world_bank", "fred_us_only"],
        "generated_at": "2026-09-10 12:00:00",
    }
    kwargs.update(overrides)
    return generate_report(**kwargs)


def test_report_structure_has_confidence_and_sources_with_timestamp():
    report = _report()
    assert "**Confidence:**" in report
    assert "data-quality flag" in report
    assert "% of indicator weight populated" in report
    assert "**Sources:**" in report
    assert "- world_bank" in report
    assert "- fred_us_only" in report
    assert "Generated: 2026-09-10 12:00:00" in report


def test_report_is_deterministic_for_fixed_timestamp():
    assert _report() == _report()


def test_report_reflects_scenario_information_and_extrapolation():
    data = panel()
    scenario = run_shock_scenario(data, "USA", 2025, preset="rate_hike")
    report = _report(scenario_result=scenario)
    assert "Scenario estimate" in report
    assert "extrapolation" in report


def test_report_confidences_are_flagged_on_missing_data():
    data = panel()
    mask = (data["country_iso3"] == "USA") & (data["year"] == 2025)
    data.loc[mask, "FP.CPI.TOTL.ZG"] = None
    scores, drivers, _ = score_panel(data)
    report = generate_report(
        "United States",
        "USA",
        2025,
        scores,
        drivers,
        generated_at="2026-09-10 12:00:00",
    )
    assert "100% of indicator weight populated" not in report
    assert "indicator weight populated" in report
