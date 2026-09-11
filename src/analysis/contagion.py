"""Contagion / correlation analysis over the engine's own scored history.

The dashboard already stores a full score history (country x year of
`risk_score` produced by src/scoring/risk_score.py). This module turns that
history into a network:

* each country's score is differenced year-over-year, so the network reflects
  *co-movement* of risk changes, not levels (a country at a permanently high
  level is not "correlated" with another that happens to sit at the same level);
* a pairwise Pearson correlation is estimated over the overlapping delta years,
  requiring a minimum overlap (`min_pairs`) so a single shared year can never
  fabricate a link;
* edges are the pairs whose |r| clears `threshold`; clusters are the connected
  components of that thresholded graph — i.e. the set of countries that move
  together, discovered from the data rather than assumed;
* every node carries an "integration" score (mean |r| against the other panel
  countries) so the most contagion-prone countries are identifiable at a glance.

This is descriptive co-movement analysis, not a causal contagion model —
callers must say so (the dashboard page does).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_PAIRS_DEFAULT = 4
CORRELATION_THRESHOLD_DEFAULT = 0.5
MAX_EDGES_DEFAULT = 60


@dataclass
class ContagionNetwork:
    """The thresholded correlation network computed from scored history.

    `nodes` and `edges` are the network proper; `correlation_matrix` and
    `delta_matrix` are the raw material so views can render a heatmap or
    inspect the underlying deltas without recomputing.
    """

    nodes: list[dict]
    edges: list[dict]
    clusters: list[list[str]]
    correlation_matrix: pd.DataFrame
    delta_matrix: pd.DataFrame
    min_pairs: int = MIN_PAIRS_DEFAULT
    threshold: float = CORRELATION_THRESHOLD_DEFAULT

    n_countries: int = 0
    n_pairs_considered: int = 0
    mean_abs_correlation: float = float("nan")
    positive_edge_share: float = float("nan")


def _pivot_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """country x year matrix of risk scores, columns sorted by year."""
    if not isinstance(scores, pd.DataFrame) or scores.empty:
        return pd.DataFrame()
    required = {"country_iso3", "year", "risk_score"}
    if not required.issubset(scores.columns):
        return pd.DataFrame()

    sub = scores[["country_iso3", "year", "risk_score"]].copy()
    sub["country_iso3"] = sub["country_iso3"].astype(str).str.upper()
    sub["year"] = pd.to_numeric(sub["year"], errors="coerce")
    sub = sub.dropna(subset=["year", "risk_score"])
    sub["year"] = sub["year"].astype(int)

    matrix = sub.pivot_table(index="country_iso3", columns="year", values="risk_score", aggfunc="first")
    matrix = matrix.reindex(sorted(map(int, matrix.columns)), axis=1)
    return matrix.sort_index()


def score_delta_matrix(scores: pd.DataFrame) -> pd.DataFrame:
    """country x year matrix of year-over-year risk-score deltas.

    Every entry is `score(year) - score(year - 1)` for that country, so a
    row can be read as "how this country's risk moved each year".
    Returns an empty frame when there are fewer than two scored years.
    """
    matrix = _pivot_scores(scores)
    if matrix.shape[1] < 2:
        return pd.DataFrame()
    return matrix.diff(axis=1)


def _iso3_list(countries) -> list[str]:
    return [str(c).upper() for c in countries]


def _pairwise_overlap(delta_matrix: pd.DataFrame) -> pd.DataFrame:
    """country x country count of jointly-observed delta years."""
    wide = delta_matrix.T  # columns = countries, index = years
    present = wide.notna().astype(int)
    valid_idx = present[present.sum(axis=1) >= 2].index
    present = present.loc[valid_idx]
    pairs = present.T @ present
    return pairs.astype(int)


def _connected_components(countries: list[str], edges: list[dict]) -> list[list[str]]:
    """Union-find connected components over the thresholded edge set."""
    index = {iso3: i for i, iso3 in enumerate(countries)}
    parent = list(range(len(countries)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for edge in edges:
        a = index.get(str(edge["a"]).upper())
        b = index.get(str(edge["b"]).upper())
        if a is not None and b is not None:
            union(a, b)

    groups: dict[int, list[str]] = {}
    for iso3 in countries:
        groups.setdefault(find(index[iso3]), []).append(iso3)
    clusters = sorted((sorted(members) for members in groups.values()), key=lambda m: m[0])
    return clusters


def correlation_network(
    scores: pd.DataFrame,
    min_pairs: int = MIN_PAIRS_DEFAULT,
    threshold: float = CORRELATION_THRESHOLD_DEFAULT,
    max_edges: int = MAX_EDGES_DEFAULT,
) -> ContagionNetwork:
    """Build the thresholded correlation network from scored history.

    Nodes are panel countries; an edge exists where two countries' YoY score
    deltas share at least `min_pairs` years AND |Pearson r| >= `threshold`.
    Clusters are the graph's connected components. Countries with no
    qualifying edge are returned as singleton nodes so the full panel stays
    visible in the network view.
    """
    deltas = score_delta_matrix(scores)
    if deltas.empty:
        return ContagionNetwork(nodes=[], edges=[], clusters=[], correlation_matrix=pd.DataFrame(), delta_matrix=deltas)

    # Correlate countries (columns of the transposed delta matrix = countries).
    corr = deltas.T.corr(min_periods=max(2, int(min_pairs)))

    overlap = _pairwise_overlap(deltas)
    countries = _iso3_list(deltas.index)

    corr_arr = corr.to_numpy(dtype=float)
    overlap_arr = overlap.to_numpy(dtype=float)

    edges: list[dict] = []
    considered = 0
    all_abs = []

    for i, a in enumerate(countries):
        for j in range(i + 1, len(countries)):
            b = countries[j]
            r = corr_arr[i, j] if i < corr_arr.shape[0] and j < corr_arr.shape[1] else float("nan")
            pairs = int(overlap_arr[i, j]) if i < overlap_arr.shape[0] and j < overlap_arr.shape[1] else 0
            considered += 1
            if not np.isnan(r):
                all_abs.append(abs(float(r)))
            if int(pairs) >= int(min_pairs) and not np.isnan(r) and abs(float(r)) >= float(threshold):
                edges.append({"a": a, "b": b, "correlation": round(float(r), 4), "pairs": int(pairs)})

    edges = sorted(edges, key=lambda e: abs(e["correlation"]), reverse=True)
    if len(edges) > int(max_edges):
        edges = edges[: int(max_edges)]
    edges = sorted(edges, key=lambda e: (e["a"], e["b"]))

    clusters = _connected_components(countries, edges)

    cluster_of: dict[str, int] = {}
    for cluster_index, members in enumerate(clusters):
        for iso3 in members:
            cluster_of[iso3] = cluster_index

    nodes: list[dict] = []
    for iso3 in countries:
        row = corr.loc[iso3] if iso3 in corr.index else pd.Series(dtype=float)
        others = row.drop(index=iso3).dropna()
        integration = float(others.abs().mean()) if not others.empty else float("nan")
        degree = sum(1 for e in edges if e["a"] == iso3 or e["b"] == iso3)
        nodes.append(
            {
                "iso3": iso3,
                "cluster": cluster_of.get(iso3, 0),
                "integration": (round(integration, 4) if not np.isnan(integration) else None),
                "degree": degree,
            }
        )

    nodes = sorted(nodes, key=lambda n: n["iso3"])

    return ContagionNetwork(
        nodes=nodes,
        edges=edges,
        clusters=clusters,
        correlation_matrix=corr,
        delta_matrix=deltas,
        min_pairs=int(min_pairs),
        threshold=float(threshold),
        n_countries=len(countries),
        n_pairs_considered=considered,
        mean_abs_correlation=(round(float(np.mean(all_abs)), 4) if all_abs else float("nan")),
        positive_edge_share=(
            round(float(np.mean([1.0 if e["correlation"] >= 0 else 0.0 for e in edges])), 4) if edges else float("nan")
        ),
    )
