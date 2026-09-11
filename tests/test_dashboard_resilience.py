"""Smoke tests for the External & fiscal resilience section.

Twin-deficits chart (fiscal + current-account balances together) plus the
Greenspan-Guidotti reserves-to-short-term-debt ratio with its 1.0 threshold —
both built from series the engine already collects.
"""

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


def test_resilience_section_renders_both_views(monkeypatch):
    app = _overview_app(monkeypatch)
    body = "\n".join(str(m.value) for m in app.markdown if m.value)

    assert "External &amp; fiscal resilience" in body
    assert "GREENSPAN–GUIDOTTI RULE" in body
    assert "TWIN DEFICITS" in body


def test_twin_deficits_named_and_explained(monkeypatch):
    app = _overview_app(monkeypatch)
    body = "\n".join(str(m.value) for m in app.markdown if m.value)

    assert "Fiscal balance" in body
    assert "Current account" in body
    # Both are explicitly zero-weight transparency signals.
    assert "zero weight" in body or "carry zero" in body


def test_resilience_section_respects_demo_provenance(monkeypatch):
    app = _overview_app(monkeypatch)
    body = "\n".join(str(m.value) for m in app.markdown if m.value)

    assert "DEMO DATA — SYNTHETIC" in body
