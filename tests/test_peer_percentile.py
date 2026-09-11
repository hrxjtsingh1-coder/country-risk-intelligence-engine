import pandas as pd

from src.scoring.risk_score import peer_percentile


def _panel_scores(rows):
    return pd.DataFrame(rows)


def test_peer_percentile_ranks_riskiest_country_first():
    scores = _panel_scores(
        [
            {"country_iso3": "AUS", "year": 2020, "risk_score": 30.0},
            {"country_iso3": "BRA", "year": 2020, "risk_score": 60.0},
            {"country_iso3": "CHN", "year": 2020, "risk_score": 50.0},
            {"country_iso3": "DEU", "year": 2020, "risk_score": 40.0},
        ]
    )
    info = peer_percentile(scores, "BRA", 2020, peer_groups={"em": ["BRA", "CHN"]})
    assert info["rank"] == 1
    assert info["n"] == 1
    assert info["percentile"] == 100
    assert info["riskier_share"] == 0
    assert info["group_name"] == "em"


def test_peer_percentile_middle_rank_and_shares():
    scores = _panel_scores(
        [
            {"country_iso3": "AUS", "year": 2020, "risk_score": 30.0},
            {"country_iso3": "BRA", "year": 2020, "risk_score": 50.0},
            {"country_iso3": "CHN", "year": 2020, "risk_score": 55.0},
            {"country_iso3": "DEU", "year": 2020, "risk_score": 60.0},
        ]
    )
    # BRA is in none of the provided groups, so the full panel cross-section is used.
    info = peer_percentile(scores, "BRA", 2020, peer_groups={"other": ["CHN", "DEU", "AUS"]})
    assert info["rank"] == 3  # two peers are riskier (55, 60)
    assert info["n"] == 3
    assert info["percentile"] == 33  # round(1/3 * 100)
    assert info["riskier_share"] == 67  # round(2/3 * 100)
    assert info["group_name"] == "panel"


def test_peer_percentile_defaults_to_panel_without_a_group():
    scores = _panel_scores(
        [
            {"country_iso3": "AUS", "year": 2020, "risk_score": 30.0},
            {"country_iso3": "BRA", "year": 2020, "risk_score": 70.0},
            {"country_iso3": "CHN", "year": 2020, "risk_score": 50.0},
            {"country_iso3": "DEU", "year": 2020, "risk_score": 40.0},
        ]
    )
    info = peer_percentile(scores, "BRA", 2020, peer_groups={"em": ["CHN"]})
    assert info["n"] == 3
    assert info["rank"] == 1
    assert info["group_name"] == "panel"


def test_peer_percentile_empty_or_invalid_returns_none_fields():
    empty = peer_percentile(pd.DataFrame(), "BRA", 2020, peer_groups={"em": ["BRA", "CHN"]})
    assert empty["percentile"] is None
    assert empty["rank"] is None
    assert empty["n"] == 0
    assert empty["group_name"] is None

    no_score_col = peer_percentile(pd.DataFrame([{"country_iso3": "BRA", "year": 2020}]), "BRA", 2020)
    assert no_score_col["percentile"] is None


def test_peer_percentile_no_comparable_peers_keeps_group_name():
    scores = _panel_scores([{"country_iso3": "BRA", "year": 2020, "risk_score": 60.0}])
    info = peer_percentile(scores, "BRA", 2020, peer_groups={"em": ["BRA", "CHN"]})
    assert info["percentile"] is None
    assert info["n"] == 0
    assert info["group_name"] == "em"


def test_peer_percentile_is_case_insensitive():
    scores = _panel_scores(
        [
            {"country_iso3": "bra", "year": 2020, "risk_score": 60.0},
            {"country_iso3": "chn", "year": 2020, "risk_score": 50.0},
        ]
    )
    info = peer_percentile(scores, "BRA", 2020, peer_groups={"em": ["Bra", "CHN"]})
    assert info["rank"] == 1
    assert info["n"] == 1
    assert info["group_name"] == "em"
