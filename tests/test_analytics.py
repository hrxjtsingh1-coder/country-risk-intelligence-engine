from pathlib import Path

import pandas as pd
import yaml

from src.cleaning.clean import clean_long_panel
from src.scenario.scenario_engine import available_shock_presets, run_shock_scenario, shock_preset
from src.scoring import risk_score as rs
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


def test_score_completeness_is_fraction_and_risk_direction():
    scores, drivers, _ = score_panel(panel())
    assert scores["data_completeness"].eq(1.0).all()
    # Within a single peer group, higher raw_value should yield higher risk contribution
    advanced = drivers[
        drivers.indicator_code.eq("GC.DOD.TOTL.GD.ZS") & drivers.country_iso3.isin(["USA", "CAN", "DEU"])
    ]
    assert advanced.sort_values("raw_value").weighted_contribution.is_monotonic_increasing


def test_score_missing_data_reduces_completeness():
    data = panel()
    data.loc[0, "FP.CPI.TOTL.ZG"] = None
    scores, _, _ = score_panel(data)
    assert scores.loc[(scores.country_iso3 == "USA") & (scores.year == 2020), "data_completeness"].iloc[0] < 1


def _config_with_sector_weight(tmp_path, monkeypatch, sector_composite_weight):
    with open(rs.CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("scoring", {})["sector_composite_weight"] = sector_composite_weight
    config_path = tmp_path / "indicators.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(rs, "CONFIG_PATH", config_path)


def test_sector_score_columns_present_and_in_range():
    scores, _, _ = score_panel(panel())
    assert "sector_score" in scores.columns
    assert "sector_macro_fiscal_sector_score" in scores.columns
    assert "sector_financial_external_sector_score" in scores.columns
    assert scores["sector_score"].between(0, 100).all()
    assert scores["sector_score"].notna().all()


def test_composite_blends_sector_score_with_configurable_weight(tmp_path, monkeypatch):
    _config_with_sector_weight(tmp_path, monkeypatch, 0.0)
    scores_pure_pillar, _, _ = score_panel(panel())

    _config_with_sector_weight(tmp_path, monkeypatch, 1.0)
    scores_sector_only, _, _ = score_panel(panel())
    assert scores_sector_only["risk_score"].round(6).eq(scores_sector_only["sector_score"].round(6)).all()

    _config_with_sector_weight(tmp_path, monkeypatch, 0.5)
    scores_blended, _, _ = score_panel(panel())
    blended = scores_blended.set_index(["country_iso3", "year"])["risk_score"]
    lo = scores_pure_pillar.set_index(["country_iso3", "year"])["risk_score"]
    hi = scores_sector_only.set_index(["country_iso3", "year"])["risk_score"]
    lower = pd.concat([lo, hi], axis=1).min(axis=1)
    upper = pd.concat([lo, hi], axis=1).max(axis=1)
    assert (blended >= lower).all()
    assert (blended <= upper).all()
    assert not blended.round(6).eq(lo.round(6)).all()
    assert not blended.round(6).eq(hi.round(6)).all()


def test_duplicate_cleaning_and_range_flag():
    clean = clean_long_panel(
        pd.DataFrame(
            [
                {"country_iso3": "usa", "indicator_code": "FP.CPI.TOTL.ZG", "year": 2024, "value": 2},
                {"country_iso3": "USA", "indicator_code": "FP.CPI.TOTL.ZG", "year": 2024, "value": 200},
            ]
        )
    )
    assert len(clean) == 1 and clean.iloc[0].flag == "out_of_range"


def test_scenario_reports_information_and_extrapolation():
    result = run_shock_scenario(panel(), "USA", 2025, "POLICY_RATE_YOY_CHANGE_BPS", 10000, ["NY.GDP.MKTP.KD.ZG"])
    assert result["out_of_sample_shock"] is True
    assert result["information_assessment"] in {"LOW INFORMATION", "MODERATE INFORMATION", "HIGH INFORMATION"}
    assert "Pooled-panel" in result["model_specification"]


def test_scenario_rejects_zero_variance_driver():
    data = panel()
    data["POLICY_RATE_YOY_CHANGE_BPS"] = 1
    result = run_shock_scenario(data, "USA", 2025, "POLICY_RATE_YOY_CHANGE_BPS", 25, ["NY.GDP.MKTP.KD.ZG"])
    assert result["information_assessment"] == "INSUFFICIENT DATA"


def test_shock_library_exposes_five_named_presets():
    presets = available_shock_presets()
    assert presets == ["rate_hike", "growth_slowdown", "fx_devaluation", "commodity_collapse", "banking_stress"]
    for name in presets:
        preset = shock_preset(name)
        assert preset["name"]
        assert preset["driver_code"]
        assert isinstance(preset["default_shock_amount"], (int, float))
        assert preset["scenario_targets"]
        assert preset["description"]
        assert preset["display_unit"]


def test_shock_library_unknown_preset_raises():
    import pytest

    with pytest.raises(ValueError, match="Unknown scenario preset"):
        shock_preset("not_a_real_preset")


def test_scenario_runs_predefined_rate_preset():
    result = run_shock_scenario(panel(), "USA", 2025, preset="rate_hike")
    assert result["preset"] == "Rate shock (+100bps)"
    assert result["driver_code"] == "POLICY_RATE_YOY_CHANGE_BPS"
    assert result["shock_amount"] == 100.0
    assert {d["indicator_code"] for d in result["indicator_deltas"]} == {
        "FX_YOY_DEPRECIATION_PCT",
        "NY.GDP.MKTP.KD.ZG",
        "GC.DOD.TOTL.GD.ZS",
    }


def test_scenario_returns_engine_level_narrative():
    result = run_shock_scenario(panel(), "USA", 2025, preset="rate_hike")
    narrative = result["narrative"]
    assert isinstance(narrative, str) and "\n" not in narrative
    assert "Rate shock (+100bps)" in narrative
    assert "100 bps" in narrative
    assert "USA's risk score" in narrative

    in_sample = run_shock_scenario(panel(), "USA", 2025, "POLICY_RATE_YOY_CHANGE_BPS", 10, ["NY.GDP.MKTP.KD.ZG"])
    assert "extrapolation" not in in_sample["narrative"]

    big = run_shock_scenario(panel(), "USA", 2025, "POLICY_RATE_YOY_CHANGE_BPS", 10000, ["NY.GDP.MKTP.KD.ZG"])
    assert "extrapolation" in big["narrative"]


def test_scenario_preset_gaps_driver_not_in_panel():
    data = panel()
    data["COMMODITY_PRICE_INDEX_PCT"] = 100.0
    result = run_shock_scenario(data, "USA", 2025, preset="commodity_collapse")
    assert result["driver_code"] == "COMMODITY_PRICE_INDEX_PCT"
    assert result["shock_amount"] == -20.0

    import pytest

    with pytest.raises(ValueError, match="not present in the panel"):
        run_shock_scenario(panel(), "USA", 2025, preset="commodity_collapse")


def test_dashboard_requires_explicit_demo_selection(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("COUNTRY_RISK_OFFLINE", "1")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard" / "app.py", default_timeout=60)
    app.run(timeout=60)
    assert not app.exception
    assert [button.label for button in app.button] == ["Open Demo Dataset"]
    app.button[0].click()
    app.run(timeout=60)
    assert not app.exception
    assert any("DEMO DATA — SYNTHETIC DATASET" in str(alert.value) for alert in app.error)
