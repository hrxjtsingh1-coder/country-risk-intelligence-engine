"""Shared HTTP helpers for official data-provider adapters."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "country-risk-intelligence-engine/1.0"})
    retry = Retry(total=4, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def fetched_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_json(url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, timeout: int = 60) -> Any:
    r = session().get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()
