-- Persistent cache for API responses.
-- Stores every fetch with source metadata so we can serve last-known-good
-- data when live APIs are unreachable.

CREATE TABLE IF NOT EXISTS api_cache (
    cache_key       TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    url             TEXT NOT NULL,
    response_body   TEXT NOT NULL,
    status_code     INTEGER NOT NULL,
    retrieved_at    TEXT NOT NULL,  -- ISO 8601 UTC
    TTL_seconds     INTEGER NOT NULL DEFAULT 21600
);

CREATE INDEX IF NOT EXISTS idx_api_cache_source ON api_cache(source);
CREATE INDEX IF NOT EXISTS idx_api_cache_retrieved ON api_cache(retrieved_at);
