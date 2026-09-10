import pandas as pd

from src.analysis.backtest import EPISODES, PASS_THRESHOLD_POINTS, run_backtest


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
