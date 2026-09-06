"""Compatibility wrapper for the public runtime collector."""
from __future__ import annotations

from src.runtime.live_data import create_wide_panel_from_long, fetch_live_data


def build_long_panel(iso3_codes=None, start=2012, end=2025):
    long, _ = fetch_live_data(start, end)
    if iso3_codes is not None:
        wanted = {str(x).upper() for x in iso3_codes}
        long = long[long["country_iso3"].isin(wanted)].copy()
    return long


def build_panel(*args, **kwargs):
    long = build_long_panel(*args, **kwargs)
    return create_wide_panel_from_long(long)
