import json

import pandas as pd

from src.analysis import backtest
from src.analysis.backtest import (
    EPISODES,
    PASS_THRESHOLD_POINTS,
    backtest_summary,
    run_backtest,
)


def test_backtest_all_inconclusive_on_empty_scores():
    results = run_backtest(pd.DataFrame(), data_is_synthetic=True)
    assert len(results) == len(EPISODES)
    assert all(r.verdict == "inconclusive" for r in results)


def test_backtest_flags_a_real_rise():
    scores = pd.DataFrame(
        [
            {"country_iso3": "TUR", "year": 2016, "risk_score": 40.0},
            {"country_iso3": "TUR", "year": 2018, "risk_score": 62.0},
        ]
    )
    results = {r.iso3: r for r in run_backtest(scores, data_is_synthetic=False)}
    assert results["TUR"].verdict == "flagged"
    assert results["TUR"].delta == 22.0
    assert results["TUR"].peak_delta == 22.0


def test_backtest_misses_a_flat_score():
    scores = pd.DataFrame(
        [
            {"country_iso3": "TUR", "year": 2016, "risk_score": 40.0},
            {"country_iso3": "TUR", "year": 2018, "risk_score": 41.0},  # below PASS_THRESHOLD_POINTS
        ]
    )
    results = {r.iso3: r for r in run_backtest(scores, data_is_synthetic=False)}
    assert results["TUR"].verdict == "missed"
    assert results["TUR"].delta < PASS_THRESHOLD_POINTS


def test_backtest_missing_year_is_inconclusive_not_a_false_pass():
    scores = pd.DataFrame([{"country_iso3": "TUR", "year": 2016, "risk_score": 40.0}])
    results = {r.iso3: r for r in run_backtest(scores, data_is_synthetic=False)}
    assert results["TUR"].verdict == "inconclusive"


def test_backtest_peak_in_window_flags_a_late_overshoot():
    # 2018 is below threshold on its own, but the peak in the 1-year detection
    # window (2018-2019) overshoots — this should count as flagged, because a
    # fast-moving shock often registers the year after the event.
    scores = pd.DataFrame(
        [
            {"country_iso3": "BRA", "year": 2013, "risk_score": 40.0},
            {"country_iso3": "BRA", "year": 2016, "risk_score": 42.0},
            {"country_iso3": "BRA", "year": 2017, "risk_score": 45.0},
        ]
    )
    results = {r.iso3: r for r in run_backtest(scores, data_is_synthetic=False)}
    r = results["BRA"]
    assert r.verdict == "flagged"
    assert r.peak_delta == 5.0
    assert r.peak_year == 2017


def test_backtest_peer_drift_is_reported():
    scores = pd.DataFrame(
        [
            # Episode country: TUR rises 10 pts -> flagged
            {"country_iso3": "TUR", "year": 2016, "risk_score": 40.0},
            {"country_iso3": "TUR", "year": 2018, "risk_score": 50.0},
            # Peer A also rose -> contributes to drift
            {"country_iso3": "USA", "year": 2016, "risk_score": 40.0},
            {"country_iso3": "USA", "year": 2018, "risk_score": 50.0},
            # Peer B barely moved -> does not
            {"country_iso3": "CAN", "year": 2016, "risk_score": 40.0},
            {"country_iso3": "CAN", "year": 2018, "risk_score": 41.0},
        ]
    )
    results = {r.iso3: r for r in run_backtest(scores, data_is_synthetic=False)}
    assert results["TUR"].peer_drift == 0.5


def test_backtest_dominant_component_is_reported():
    scores = pd.DataFrame(
        [
            {
                "country_iso3": "TUR",
                "year": 2016,
                "risk_score": 40.0,
                "pillar_fiscal_sustainability_score": 40.0,
                "pillar_macro_growth_inflation_score": 40.0,
                "sector_macro_fiscal_sector_score": 40.0,
                "sector_financial_external_sector_score": 40.0,
            },
            {
                "country_iso3": "TUR",
                "year": 2018,
                "risk_score": 62.0,
                "pillar_fiscal_sustainability_score": 60.0,
                "pillar_macro_growth_inflation_score": 45.0,
                "sector_macro_fiscal_sector_score": 58.0,
                "sector_financial_external_sector_score": 44.0,
            },
        ]
    )
    r = {x.iso3: x for x in run_backtest(scores, data_is_synthetic=False)}["TUR"]
    assert r.dominant_pillar == "pillar_fiscal_sustainability_score"
    assert r.dominant_sector == "sector_macro_fiscal_sector_score"


def test_backtest_summary_totals_and_detection_rate():
    scores = pd.DataFrame(
        [
            {"country_iso3": "TUR", "year": 2016, "risk_score": 40.0},
            {"country_iso3": "TUR", "year": 2018, "risk_score": 62.0},
            {"country_iso3": "ZAF", "year": 2017, "risk_score": 40.0},
            {"country_iso3": "ZAF", "year": 2020, "risk_score": 41.0},  # +1 -> missed
        ]
    )
    results = run_backtest(scores, data_is_synthetic=False)
    summary = backtest_summary(results)
    assert summary.episodes_total == len(EPISODES)
    assert summary.flagged == 1
    assert summary.missed == 1
    assert summary.inconclusive == len(EPISODES) - 2
    assert summary.detection_rate == 0.5


def test_backtest_cli_writes_json(tmp_path, capsys):
    panel = pd.DataFrame(
        [{"country_iso3": "USA", "year": y, "FP.CPI.TOTL.ZG": 3, "GC.DOD.TOTL.GD.ZS": 60} for y in range(2016, 2023)]
    )
    panel_path = tmp_path / "panel.csv"
    panel.to_csv(panel_path, index=False)
    json_path = tmp_path / "report.json"

    exit_code = backtest.main(["--panel", str(panel_path), "--mode", "demo", "--json", str(json_path)])
    assert exit_code == 0
    payload = json.loads(json_path.read_text())
    assert payload["synthetic"] is True
    assert len(payload["episodes"]) == len(EPISODES)
    assert set(payload["summary"]) == {
        "episodes_total",
        "flagged",
        "missed",
        "inconclusive",
        "detection_rate",
        "median_peak_delta",
        "systemic_episode",
        "highest_peer_drift",
    }
    # USA-only panel scores no episode years, so every episode is inconclusive
    # and the CLI says so explicitly rather than printing a bogus detection rate.
    assert payload["summary"]["inconclusive"] == len(EPISODES)
    assert "All 4 episodes inconclusive" in capsys.readouterr().out
