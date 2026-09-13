"""OECD Data Explorer SDMX REST API adapter. Public."""
from __future__ import annotations

from typing import Any

from .common import fetched_at, session

BASE_URL = "https://sdmx.oecd.org/public/rest"


def fetch_data(data_path: str, *, params: dict[str, Any] | None = None, accept: str = "application/vnd.sdmx.data+csv;version=2.0.0") -> dict[str, Any]:
    url = f"{BASE_URL}/data/{data_path.lstrip('/')}"
    r = session().get(url, params=params or {}, headers={"Accept": accept}, timeout=90)
    r.raise_for_status()
    return {"fetched_at": fetched_at(), "url": r.url, "content_type": r.headers.get("content-type", ""), "body": r.text}


def fetch_dataflow(agency: str, dataflow: str, version: str = "latest") -> dict[str, Any]:
    url = f"{BASE_URL}/dataflow/{agency}/{dataflow}/{version}"
    r = session().get(url, headers={"Accept": "application/vnd.sdmx.structure+csv;version=2.0"}, timeout=60)
    r.raise_for_status()
    return {"fetched_at": fetched_at(), "url": r.url, "content_type": r.headers.get("content-type", ""), "body": r.text}


def smoke_test() -> dict[str, Any]:
    result = fetch_dataflow("OECD.SDD.NAD", "DSD_NASU@DF_INDICATOR", "latest")
    return {"source": "OECD", "ok": bool(result.get("body")), "endpoint": result["url"]}
