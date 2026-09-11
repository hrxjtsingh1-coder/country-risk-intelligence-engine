"""Cached-data auto-fallback for the live dashboard.

When the live World Bank fetch fails or exceeds its wall-clock budget, the
app must not hang on "Connecting to World Bank..." — it falls back to the
last successful pipeline panel (CACHED state) with an explicit notice, and
renders the full cockpit without a dead-end demo selection.
"""

from pathlib import Path


def test_dashboard_falls_back_to_cached_panel_when_live_unavailable(monkeypatch):
    from streamlit.testing.v1 import AppTest

    def _raise_unavailable(*args, **kwargs):
        from src.runtime.live_data import LiveDataUnavailable

        raise LiveDataUnavailable("test: live fetch unavailable")

    monkeypatch.setattr("dashboard.app.fetch_live_panel", _raise_unavailable)
    monkeypatch.delenv("COUNTRY_RISK_OFFLINE", raising=False)

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard" / "app.py", default_timeout=90)
    app.run(timeout=90)
    assert not app.exception
    assert [b.label for b in app.button] == ["↻ Retry Live Data"]

    warnings = "\n".join(str(w.value) for w in app.warning if w.value)
    assert "USING CACHED DATA" in warnings

    kickers = " ".join(str(m.value) for m in app.markdown if "RISK ENGINE" in str(m.value))
    assert "CACHED PANEL" in kickers

    body = "\n".join(str(m.value) for m in app.markdown if m.value)
    assert "CACHED DATA" in body
