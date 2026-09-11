"""Dashboard page sections.

Each module owns one page's worth of components. The app shell composes
them in render order — mechanical split only, no navigation/routing yet.
"""

from dashboard.sections import (
    benchmark,
    comparison,
    contagion,
    country,
    fx_deviation,
    map,
    methodology,
    pdf_export,
    scenario,
    track_record,
)

__all__ = [
    "benchmark",
    "comparison",
    "contagion",
    "country",
    "fx_deviation",
    "map",
    "methodology",
    "pdf_export",
    "scenario",
    "track_record",
]
