"""Shared HTTP helpers for official data-provider adapters."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import requests


USER_AGENT = "country-risk-intelligence-engine/1.0"
DEFAULT_TIMEOUT = 90


def session() -> requests.Session:
    """Create a configured HTTP session for public provider APIs."""
    client = requests.Session()
    client.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return client


def fetched_at() -> str:
    """Return a UTC timestamp for provenance metadata."""
    return datetime.now(UTC).isoformat()


def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Any:
    """GET a JSON resource and fail loudly on HTTP errors."""
    response = session().get(url, params=params or {}, timeout=timeout)
    response.raise_for_status()
    return response.json()
