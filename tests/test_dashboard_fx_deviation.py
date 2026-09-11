"""Smoke tests for the FX deviation card on the Overview page."""

from pathlib import Path


def _overview_app(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("COUNTRY_RISK_OFFLINE", "1")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard" / "app.py", default_timeout=120)
    app.run(timeout=120)
    assert not app.exception

    if app.button:
        app.button[0].click()
        app.run(timeout=120)
    assert not app.exception
    return app


def test_fx_deviation_section_renders(monkeypatch):
    app = _overview_app(monkeypatch)
    body = "\n".join(str(m.value) for m in app.markdown if m.value)
    assert "FX DEVIATION FROM TREND" in body
    assert "nominal REER-style proxy" in body


def test_fx_deviation_section_has_no_exception(monkeypatch):
    app = _overview_app(monkeypatch)
    assert not app.exception
