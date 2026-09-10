"""Global risk map page.

Foundation only — the animated geographic map is intentionally not built
in this pass. This module defines the render contract the app shell will
call once the map lands, so the page structure exists without the
implementation.

```python
# once wired, in the app shell:
from dashboard.sections import map as map_page
map_page.render(ctx)
```

The renderer is expected to draw a choropleth of the panel's current-year
`risk_score` by ISO3, with band colors, hover tooltips and the same
read-only plotly config as the rest of the dashboard.
"""

from __future__ import annotations

from dashboard.context import Context


def render(ctx: Context) -> None:
    raise NotImplementedError(
        "The animated geographic map is scheduled for the next build. "
        "render(ctx) is the contract the app shell will call once it lands."
    )
