"""Dashboard page sections.

Each module owns one page's worth of components. The app shell composes
them in render order — mechanical split only, no navigation/routing yet.
"""

from dashboard.sections import comparison, contagion, country, map, methodology, pdf_export, scenario, track_record

__all__ = ["comparison", "contagion", "country", "map", "methodology", "pdf_export", "scenario", "track_record"]
