"""
Provenance metadata for a live data pull.

The point of this module: a user looking at a risk score should be able to
answer "where did this number come from?" from the app itself — source,
retrieval time, latest underlying observation, and coverage — without
opening the source code (PHASE 13).
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime


@dataclass
class Provenance:
    run_id: str
    retrieved_at: str  # human-readable, UTC
    sources: list[dict] = field(default_factory=list)  # [{"name": ..., "url": ...}]
    requested_period: str = ""
    latest_observation_year: int | None = None
    country_count: int = 0
    indicator_count: int = 0
    expected_observations: int = 0
    received_observations: int = 0
    coverage_pct: float = 0.0
    validation_failures: list[str] = field(default_factory=list)
    config_version: str = "unversioned"

    def as_dict(self) -> dict:
        return asdict(self)


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def now_iso() -> str:
    return datetime.now(UTC).strftime("%d %b %Y, %H:%M UTC")


def make_provenance(
    *,
    sources: list[dict],
    requested_period: str,
    latest_observation_year: int | None,
    country_count: int,
    indicator_count: int,
    expected_observations: int,
    received_observations: int,
    validation_failures: list[str] | None = None,
    config_version: str = "unversioned",
) -> Provenance:
    coverage = round(100 * received_observations / expected_observations, 1) if expected_observations else 0.0
    return Provenance(
        run_id=new_run_id(),
        retrieved_at=now_iso(),
        sources=sources,
        requested_period=requested_period,
        latest_observation_year=latest_observation_year,
        country_count=country_count,
        indicator_count=indicator_count,
        expected_observations=expected_observations,
        received_observations=received_observations,
        coverage_pct=coverage,
        validation_failures=validation_failures or [],
        config_version=config_version,
    )
