"""
Explicit data-state constants for the dashboard's live/demo state machine
(see PHASE 8 of the productization brief).

Kept as a tiny, dependency-free module — no Streamlit, no requests — so
anything can check "is this LIVE or DEMO" without importing the runtime
fetch machinery.

Mode sequence: an unset mode attempts live; on success the app lands in
LIVE, on a slow/failed fetch it falls back to CACHED (the last successful
pipeline panel, when present), then DEMO, and only then UNAVAILABLE.
"""

from __future__ import annotations

LOADING = "LOADING"
LIVE = "LIVE"
STALE = "STALE"
CACHED = "CACHED"
UNAVAILABLE = "UNAVAILABLE"
DEMO = "DEMO"

VALID_STATES = {LOADING, LIVE, STALE, CACHED, UNAVAILABLE, DEMO}

# How long a successful live fetch is trusted before it's considered stale
# enough to need a background refresh flag (still shown, just labeled).
# Matches PHASE 6's recommended cache TTL.
LIVE_CACHE_TTL_SECONDS = 6 * 60 * 60
