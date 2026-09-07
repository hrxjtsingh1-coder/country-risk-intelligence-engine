"""
Explicit data-state constants for the dashboard's live/demo state machine
(see PHASE 8 of the productization brief).

Kept as a tiny, dependency-free module — no Streamlit, no requests — so
anything can check "is this LIVE or DEMO" without importing the runtime
fetch machinery.
"""
from __future__ import annotations

LOADING = "LOADING"
LIVE = "LIVE"
STALE = "STALE"
UNAVAILABLE = "UNAVAILABLE"
DEMO = "DEMO"

VALID_STATES = {LOADING, LIVE, STALE, UNAVAILABLE, DEMO}

# How long a successful live fetch is trusted before it's considered stale
# enough to need a background refresh flag (still shown, just labeled).
# Matches PHASE 6's recommended cache TTL.
LIVE_CACHE_TTL_SECONDS = 6 * 60 * 60
