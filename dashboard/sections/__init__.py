"""Dashboard page sections.

Each module owns one page's worth of components. The app shell composes
them in render order — mechanical split only, no navigation/routing yet.
"""

from dashboard.sections import comparison, country, map, methodology, scenario

__all__ = ["comparison", "country", "map", "methodology", "scenario"]
