"""Smoke test for the About page section.

The full-app AppTest already exercises rendering (it raises on any
exception); these assertions pin the About page to actually be present and
carry its core claims.
"""

from pathlib import Path


def test_about_page_renders_claims(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("COUNTRY_RISK_OFFLINE", "1")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard" / "app.py", default_timeout=90)
    app.run(timeout=90)
    assert not app.exception
    app.button[0].click()
    app.run(timeout=90)
    assert not app.exception

    body = "\n".join(str(m.value) for m in app.markdown if m.value)
    assert "About this engine" in body
    assert "What it is not" in body
    assert "credit rating" in body
    assert "Data sources" in body
    assert "LLM" in body
