"""Unit tests for the contagion / correlation analysis module."""

from __future__ import annotations

import pandas as pd

from src.analysis.contagion import correlation_network, score_delta_matrix


def _scores(series: dict[str, dict[int, float]]) -> pd.DataFrame:
    rows = []
    for iso3, hist in series.items():
        for year, score in hist.items():
            rows.append({"country_iso3": iso3, "year": year, "risk_score": score})
    return pd.DataFrame(rows)


def test_score_delta_matrix_returns_yoy_deltas_per_country():
    scores = _scores(
        {
            "A": {2020: 10.0, 2021: 15.0, 2022: 13.0},
            "B": {2020: 30.0, 2021: 25.0, 2022: 40.0},
        }
    )
    deltas = score_delta_matrix(scores)

    assert list(deltas.index) == ["A", "B"]
    assert list(deltas.columns) == [2020, 2021, 2022]
    # First delta year is NaN (no prior year), then year-over-year differences.
    assert deltas.loc["A", 2021] == 5.0
    assert deltas.loc["A", 2022] == -2.0
    assert deltas.loc["B", 2021] == -5.0
    assert deltas.loc["B", 2022] == 15.0


def test_score_delta_matrix_empty_with_insufficient_years():
    scores = _scores({"A": {2020: 10.0}})
    assert score_delta_matrix(scores).empty


def test_correlation_network_links_countries_that_move_together():
    scores = _scores(
        {
            # A and B share the exact same delta pattern -> perfect correlation.
            "A": {y: 10.0 + (5.0 if y % 2 == 0 else 0.0) for y in range(2015, 2021)},
            "B": {y: 30.0 + (5.0 if y % 2 == 0 else 0.0) for y in range(2015, 2021)},
            # C drifts steadily -> every delta is the same -> no variance, no edge.
            "C": {y: float(y - 2010) for y in range(2015, 2021)},
        }
    )
    network = correlation_network(scores)

    assert network.n_countries == 3
    assert {frozenset((e["a"], e["b"])) for e in network.edges} == {frozenset(("A", "B"))}
    link = next(e for e in network.edges if set((e["a"], e["b"])) == {"A", "B"})
    assert link["correlation"] == 1.0
    assert link["pairs"] >= 4

    # A and B cluster together; C is its own singleton cluster.
    cluster_sets = [frozenset(cl) for cl in network.clusters]
    assert frozenset(("A", "B")) in cluster_sets
    assert frozenset(("C",)) in cluster_sets


def test_correlation_network_respects_minimum_overlap():
    scores = _scores(
        {
            "X": {2017: 10.0, 2018: 15.0, 2019: 12.0},
            "Y": {2017: 30.0, 2018: 35.0, 2019: 32.0},
        }
    )
    # Only two shared delta years exist — below the default min_pairs of 4.
    network = correlation_network(scores, min_pairs=4)

    assert network.edges == []
    assert network.n_countries == 2
    assert all(int(n["degree"]) == 0 for n in network.nodes)


def test_correlation_network_tolerates_empty_input():
    empty = pd.DataFrame(columns=["country_iso3", "year", "risk_score"])
    network = correlation_network(empty)

    assert network.n_countries == 0
    assert network.nodes == []
    assert network.edges == []
    assert network.clusters == []


def test_integration_and_degree_are_reported_per_node():
    scores = _scores(
        {
            "A": {y: 10.0 + (5.0 if y % 2 == 0 else 0.0) for y in range(2015, 2021)},
            "B": {y: 30.0 + (5.0 if y % 2 == 0 else 0.0) for y in range(2015, 2021)},
        }
    )
    network = correlation_network(scores)
    by_iso = {n["iso3"]: n for n in network.nodes}

    assert by_iso["A"]["degree"] == 1
    assert by_iso["B"]["degree"] == 1
    assert by_iso["A"]["integration"] == 1.0
    assert by_iso["B"]["integration"] == 1.0
