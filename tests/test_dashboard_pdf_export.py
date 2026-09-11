"""Smoke tests for the per-country PDF export button on the Overview page."""

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


def test_pdf_export_section_renders_on_overview(monkeypatch):
    app = _overview_app(monkeypatch)
    body = "\n".join(str(m.value) for m in app.markdown if m.value)
    assert "PDF EXPORT" in body


def test_pdf_export_button_renders_on_overview(monkeypatch):
    app = _overview_app(monkeypatch)
    labels = [str(b.label) for b in app.download_button]
    assert any("Download PDF report" in label for label in labels)
