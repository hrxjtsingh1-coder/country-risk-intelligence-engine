"""
Persistent SQLite cache for public API responses.

Every World Bank / FRED / IMF fetch goes through this layer. On success,
the response is stored with a timestamp. On failure, the last-known-good
response is returned with a visible staleness flag so the dashboard can
show "using cached data from [date]" instead of silently serving stale data.

Design:
    - Cache key = SHA-256(source + url + sorted params)
    - TTL is configurable per fetch (default 6 hours)
    - The cache DB lives at data/processed/api_cache.db
    - Historical pulls are kept (not cleared on each run) so we can chart
      data freshness over time
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger("country-risk.cache")

ROOT = Path(__file__).resolve().parents[2]
CACHE_DB_PATH = ROOT / "data" / "processed" / "api_cache.db"
SCHEMA_PATH = ROOT / "src" / "cache" / "schema.sql"

DEFAULT_TTL_SECONDS = 6 * 60 * 60  # 6 hours


@dataclass
class CacheResult:
    """Result of a cache lookup or fetch."""

    data: Any  # The parsed response body (JSON-decoded or raw text)
    from_cache: bool  # True if served from cache (not a fresh fetch)
    cached_at: str | None  # ISO timestamp of when the cached data was stored
    source: str  # e.g. "World Bank", "FRED"
    url: str  # The API endpoint URL
    freshness_seconds: int | None = None  # How old the cached data is


def _get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (or create) the cache database and ensure the schema exists."""
    path = db_path or CACHE_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema)
    return conn


def _make_cache_key(source: str, url: str, params: dict | None = None) -> str:
    """Deterministic cache key from source + URL + params."""
    raw = source + "|" + url
    if params:
        raw += "|" + json.dumps(params, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def store_response(
    source: str,
    url: str,
    params: dict | None,
    response_body: str,
    status_code: int,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    db_path: Path | None = None,
) -> None:
    """Store an API response in the cache."""
    key = _make_cache_key(source, url, params)
    conn = _get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO api_cache
                (cache_key, source, url, response_body, status_code, retrieved_at, TTL_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (key, source, url, response_body, status_code, _now_iso(), ttl_seconds),
        )
        conn.commit()
    finally:
        conn.close()


def get_cached(
    source: str,
    url: str,
    params: dict | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    db_path: Path | None = None,
) -> CacheResult | None:
    """
    Look up a cached response. Returns None if:
    - No cache entry exists
    - The entry has expired (older than TTL_seconds)

    Returns a CacheResult with from_cache=True if a valid cached entry exists.
    """
    key = _make_cache_key(source, url, params)
    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT response_body, status_code, retrieved_at FROM api_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    body, status, retrieved_at = row
    try:
        cached_time = datetime.fromisoformat(retrieved_at)
    except ValueError:
        return None

    age = (datetime.now(timezone.utc) - cached_time).total_seconds()
    if age > ttl_seconds:
        LOG.info("Cache entry for %s expired (%.0fs old, TTL=%ds)", source, age, ttl_seconds)
        return None

    return CacheResult(
        data=json.loads(body),
        from_cache=True,
        cached_at=retrieved_at,
        source=source,
        url=url,
        freshness_seconds=int(age),
    )


def get_last_known_good(
    source: str,
    url: str,
    params: dict | None = None,
    db_path: Path | None = None,
) -> CacheResult | None:
    """
    Retrieve the most recent cached response regardless of TTL.
    Used as a fallback when live fetch fails — the dashboard shows
    "using cached data from [date]" so the user knows the data is stale.
    """
    key = _make_cache_key(source, url, params)
    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT response_body, status_code, retrieved_at FROM api_cache WHERE cache_key = ? "
            "ORDER BY retrieved_at DESC LIMIT 1",
            (key,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    body, status, retrieved_at = row
    try:
        cached_time = datetime.fromisoformat(retrieved_at)
    except ValueError:
        return None

    age = (datetime.now(timezone.utc) - cached_time).total_seconds()

    return CacheResult(
        data=json.loads(body),
        from_cache=True,
        cached_at=retrieved_at,
        source=source,
        url=url,
        freshness_seconds=int(age),
    )


def get_cache_stats(db_path: Path | None = None) -> dict:
    """Return summary statistics about the cache contents."""
    conn = _get_connection(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM api_cache").fetchone()[0]
        sources = conn.execute(
            "SELECT source, COUNT(*), MIN(retrieved_at), MAX(retrieved_at) FROM api_cache GROUP BY source"
        ).fetchall()
        return {
            "total_entries": total,
            "sources": {
                s[0]: {"count": s[1], "oldest": s[2], "newest": s[3]} for s in sources
            },
        }
    finally:
        conn.close()
