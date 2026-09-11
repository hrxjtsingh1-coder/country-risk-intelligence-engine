"""Shared analytical slice passed from the app shell to every page section.

A `Context` bundles everything a section module needs to render the
current country / year / shock selection plus the already-computed
engine outputs. The app shell builds exactly one of these per run and
passes it to each section's render functions in render order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Context:
    panel: pd.DataFrame
    scores: pd.DataFrame
    drivers: pd.DataFrame
    pillar_scores: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    indicators_cfg: dict[str, Any] = field(default_factory=dict)

    country: str = ""
    year: int = 0
    iso: str = ""
    country_label: str = ""
    band: str = ""
    score_value: float = 0.0
    score_color: str = ""
    coverage_value: float = float("nan")
    current_row: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    country_drivers: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    report: str = ""
    scenario: Any = None
    scenario_result: Any = None
    scenario_error: Exception | None = None
    shock: int = 0
    run_scenario_btn: bool = False

    using_demo_data: bool = False
    live_provenance: Any = None
    generated_at: str = ""

    # Peer-group relative standing for the selected country-year.
    peer_percentile: float | None = None
    peer_riskier_share: int | None = None
    peer_rank: int | None = None
    peer_n: int = 0
    peer_group_name: str = ""

    # Data provenance surfaced next to every score/chart.
    data_verified: bool = False
    provenance_sources: str = ""
    provenance_asof: str = ""
    manifest_hash: str = ""
