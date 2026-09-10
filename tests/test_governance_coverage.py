import textwrap

import pandas as pd
import yaml

from src.governance.coverage import verify_model_coverage

DEMO_PANEL = pd.read_csv("data/demo/panel_wide.csv")


def test_coverage_passes_on_demo_panel():
    report = verify_model_coverage(DEMO_PANEL, panel_source="demo")
    assert report.status == "ok"
    assert not [i for i in report.issues if i.severity == "error"]
    assert report.indicator_total == report.indicator_covered
    # Structural checks all pass; the BRA episode warning (baseline 2013 outside
    # the demo window) is expected and reported as a warning, not an error.
    expected_fail = {"episode_windows_in_range"}
    assert all(v for k, v in report.checks.items() if k not in expected_fail)


def test_coverage_detects_missing_positive_weight_column():
    panel = DEMO_PANEL.drop(columns=["GC.DOD.TOTL.GD.ZS"])
    report = verify_model_coverage(panel, panel_source="test")
    assert report.status == "issues"
    refs = {i.ref for i in report.issues if i.severity == "error"}
    assert "panel/GC.DOD.TOTL.GD.ZS" in refs


def test_coverage_detects_wrong_derived_shape():
    panel = DEMO_PANEL.copy()
    panel.loc[panel["year"] == 2020, "PUBLIC_DEBT_TRAJECTORY_PCT"] += 1.0
    report = verify_model_coverage(panel, panel_source="test")
    assert any(i.ref == "indicator/PUBLIC_DEBT_TRAJECTORY_PCT" for i in report.issues)


def test_coverage_detects_phantom_pillar_with_no_indicators(tmp_path, monkeypatch):
    import src.scoring.risk_score as rs

    cfg = yaml.safe_load(rs.CONFIG_PATH.read_text()) or {}
    weights = dict(cfg["pillar_weights"])
    weights["Imaginary Pillar"] = 0.02
    cfg["pillar_weights"] = weights
    cfg_path = tmp_path / "indicators.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(rs, "CONFIG_PATH", cfg_path)

    report = verify_model_coverage(DEMO_PANEL, panel_source="test")
    assert any(i.ref == "config/pillars" for i in report.issues)


def test_coverage_detects_bad_pillar_weights(tmp_path, monkeypatch):
    import src.scoring.risk_score as rs

    cfg = yaml.safe_load(rs.CONFIG_PATH.read_text()) or {}
    cfg["pillar_weights"] = {"Macro Growth & Inflation": 0.4, "Fiscal Sustainability": 0.4}
    cfg_path = tmp_path / "indicators.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(rs, "CONFIG_PATH", cfg_path)

    report = verify_model_coverage(DEMO_PANEL, panel_source="test")
    assert any(i.ref == "config/pillar_weights" for i in report.issues)


def test_coverage_warns_when_episode_years_outside_panel(tmp_path, monkeypatch):
    import src.governance.coverage as gov

    episodes = tmp_path / "episodes.yaml"
    episodes.write_text(
        textwrap.dedent(
            """
            episodes:
              - iso3: TUR
                label: Test episode
                baseline_year: 1990
                event_year: 2030
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(gov, "EPISODES_PATH", episodes)
    report = verify_model_coverage(DEMO_PANEL, panel_source="test")
    assert any(i.ref.startswith("episodes/") and i.severity == "warning" for i in report.issues)
    assert report.checks["episode_windows_in_range"] is False
