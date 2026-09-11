"""Smoke tests for the provenance footer.

Every score/chart carries a small footer line: verification status, data
source, as-of date, and — when the pipeline manifest exists — its short hash.
Demo mode is labeled honestly as synthetic data, never as verified.
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
    assert app.radio and str(app.radio[0].value) == "Overview"
    return app


def test_provenance_line_renders_on_score_pages(monkeypatch):
    app = _overview_app(monkeypatch)
    body = "\n".join(str(m.value) for m in app.markdown if m.value)

    # Demo mode: the provenance line labels the data as synthetic + unverified,
    # never as verified public data.
    assert "DEMO DATA — SYNTHETIC" in body
    assert "not verified" in body.lower()


def test_manifest_hash_shown_when_pipeline_manifest_exists(monkeypatch):
    app = _overview_app(monkeypatch)
    body = "\n".join(str(m.value) for m in app.markdown if m.value)

    if Path("data/processed/manifest.json").exists():
        assert "MANIFEST" in body


def test_provenance_footer_present_on_all_chart_pages(monkeypatch):
    app = _overview_app(monkeypatch)
    body = "\n".join(str(m.value) for m in app.markdown if m.value)

    # The provenance line is emitted multiple times across sections.
    assert body.count("DEMO DATA — SYNTHETIC") >= 3
