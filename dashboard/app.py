"""
Streamlit dashboard for the Global Country Risk Intelligence Engine.

Run with:
    streamlit run dashboard/app.py

The dashboard is a thin shell: it owns data loading, the live/demo state
machine, the sidebar controls and the analytical execution, builds a
single shared `Context`, then composes the page section modules in render
order. Section modules (dashboard/sections/) render presentation only —
they never recompute the risk methodology.

Reads data/processed/panel_wide.csv, which src/pipeline/run_all.py produces
from the live World Bank / FRED / ECB / BIS collectors. Scores and drivers
are recomputed on the fly from that panel (cheap, and guarantees the
dashboard always matches the current config/indicators.yaml weights even if
you tweak them without re-running the full pipeline).
"""

from __future__ import annotations

import concurrent.futures
import os
import sys
import textwrap
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.commentary.generate_commentary import generate_report
from src.runtime import data_state
from src.runtime.live_data import LiveDataUnavailable, fetch_live_panel
from src.scenario.scenario_engine import available_shock_presets, run_shock_scenario, shock_preset
from src.scoring.risk_score import peer_percentile, score_panel, top_drivers

# Run from any working directory: make both the package root `src` and the
# dashboard package resolvable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import ui  # noqa: E402
from dashboard.context import Context  # noqa: E402
from dashboard.sections import about as about_section  # noqa: E402
from dashboard.sections import benchmark as benchmark_section  # noqa: E402
from dashboard.sections import comparison as comparison_section  # noqa: E402
from dashboard.sections import contagion as contagion_section  # noqa: E402
from dashboard.sections import country as country_section  # noqa: E402
from dashboard.sections import fx_deviation as fx_deviation_section  # noqa: E402
from dashboard.sections import map as map_section  # noqa: E402
from dashboard.sections import methodology as methodology_section  # noqa: E402
from dashboard.sections import pdf_export as pdf_export_section  # noqa: E402
from dashboard.sections import pulse as pulse_section  # noqa: E402
from dashboard.sections import scenario as scenario_section  # noqa: E402
from dashboard.sections import track_record as track_record_section  # noqa: E402
from dashboard.ui import (  # noqa: E402
    available_countries,
    available_years,
    band_color,
    esc,
    get_country_label,
    get_iso,
    row_for,
    safe_float,
    set_countries_cfg,
    user_error,
)

CONFIG_DIR = ROOT / "config"
PROCESSED_DIR = ROOT / "data" / "processed"
PANEL_PATH = PROCESSED_DIR / "panel_wide.csv"
DEMO_PANEL_PATH = ROOT / "data" / "demo" / "panel_wide.csv"


# Streamlit's Markdown parser treats indented HTML as a code block. The
# dashboard intentionally uses readable, indented HTML templates. Keep this
# compatibility wrapper idempotent: Streamlit reruns the script in the same
# process, so wrapping st.markdown repeatedly can create recursive renderers
# and unreliable refreshes.
st_markdown_original = getattr(st.markdown, "_cri_html_normalized", False)
if not st_markdown_original:
    _streamlit_markdown = st.markdown

    def _dedented_markdown(body, *args, **kwargs):
        if isinstance(body, str) and kwargs.get("unsafe_allow_html"):
            body = textwrap.dedent(body)
            body = "\n".join(line.lstrip() for line in body.splitlines())
        return _streamlit_markdown(body, *args, **kwargs)

    _dedented_markdown._cri_html_normalized = True
    st.markdown = _dedented_markdown


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=ui.APP_TITLE,
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# GLOBAL CSS (extracted to dashboard/styles.css)
# ============================================================================

st.markdown(
    f"<style>\n{(ROOT / 'dashboard' / 'styles.css').read_text(encoding='utf-8')}\n</style>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="aurora"><span></span><span></span><span></span></div>',
    unsafe_allow_html=True,
)


# ============================================================================
# DATA / CONFIGURATION LOADING
# ============================================================================


@st.cache_data(show_spinner=False)
def load_configurations():
    countries = yaml.safe_load((CONFIG_DIR / "countries.yaml").read_text(encoding="utf-8"))
    indicators = yaml.safe_load((CONFIG_DIR / "indicators.yaml").read_text(encoding="utf-8"))
    return countries, indicators


@st.cache_data(show_spinner=False)
def load_panel(panel_path: Path):
    return pd.read_csv(panel_path)


countries_cfg, indicators_cfg = load_configurations()
peer_groups = countries_cfg.get("peer_groups", {}) if isinstance(countries_cfg, dict) else {}
set_countries_cfg(countries_cfg)

# ============================================================================
# LIVE / DEMO DATA STATE MACHINE
#
# The public app fetches live World Bank/FRED data itself — nobody should
# need to run a Python command before anything appears. A live failure
# shows a clean status screen with one explicit way forward (Open Demo
# Dataset); it never silently substitutes synthetic data for real data.
# ============================================================================

COUNTRY_ISO3_LIST = (
    tuple(c["iso3"] for c in countries_cfg.get("countries", []) if c.get("iso3"))
    if isinstance(countries_cfg, dict)
    else tuple()
)

LIVE_START_YEAR = 2012
LIVE_END_YEAR = datetime.now().year


def _config_version() -> str:
    """Short hash of the config files so a cached live fetch invalidates
    itself if you edit indicators.yaml/countries.yaml, without needing a
    manual cache clear."""
    import hashlib

    payload = (CONFIG_DIR / "indicators.yaml").read_bytes() + (CONFIG_DIR / "countries.yaml").read_bytes()
    return hashlib.sha256(payload).hexdigest()[:10]


def _manifest_short_hash() -> str:
    """Best-effort short sha256 of the pipeline manifest, so the footer can
    prove the displayed numbers come from the exact pipeline run at a hash.
    The manifest lives under the gitignored data/processed dir and may not
    exist on demo-only installs — fall back to the config version."""
    import hashlib

    manifest_path = Path("data/processed/manifest.json")
    if manifest_path.exists():
        return hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:8]
    return ""


# Wall-clock budget for a live World Bank attempt. The dashboard must load
# within a few seconds even when WB is slow or down, so instead of a
# dead-end screen the app falls back to the last successful pipeline panel
# (CACHED) or the demo dataset (DEMO). Overridable for slow/spotty networks.
LIVE_FETCH_TIMEOUT_SECONDS = float(os.environ.get("COUNTRY_RISK_LIVE_TIMEOUT_SECONDS", "6"))


def _is_offline() -> bool:
    return os.environ.get("COUNTRY_RISK_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def _fetch_live_bounded(iso3_codes: tuple, start_year: int, end_year: int, config_version: str):
    """Run a live fetch with a strict wall-clock budget.

    The requests layer already gives each call its own timeout + retry, but a
    slow World Bank endpoint can still stretch that to minutes across many
    calls. Bounding the whole attempt here is what guarantees the app loads
    quickly; a timed-out attempt is reported as LiveDataUnavailable so the
    caller can fall back to cached/demo data instead of hanging the UI. The
    worker thread is daemonic so an abandoned slow fetch never blocks app or
    process shutdown.
    """
    future: concurrent.futures.Future = concurrent.futures.Future()

    def _run_fetch() -> None:
        try:
            result = fetch_live_panel(list(iso3_codes), indicators_cfg, start_year, end_year, config_version)
            future.set_result(result)
        except BaseException as exc:  # noqa: BLE001 - propagate whatever the fetch raises
            future.set_exception(exc)

    threading.Thread(target=_run_fetch, daemon=True, name="cri-live").start()

    try:
        return future.result(timeout=LIVE_FETCH_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError as exc:
        raise LiveDataUnavailable(
            "The live World Bank fetch did not finish within the dashboard's load budget.",
            technical_detail=(
                f"live fetch exceeded the {LIVE_FETCH_TIMEOUT_SECONDS:.0f}s wall-clock budget "
                "(set COUNTRY_RISK_LIVE_TIMEOUT_SECONDS to raise it); "
                "showing the last successful pipeline panel instead."
            ),
        ) from exc


@st.cache_data(ttl=data_state.LIVE_CACHE_TTL_SECONDS, show_spinner=False)
def _cached_fetch_live(iso3_codes: tuple, start_year: int, end_year: int, config_version: str):
    return _fetch_live_bounded(iso3_codes, start_year, end_year, config_version)


if "data_mode" not in st.session_state:
    st.session_state.data_mode = None  # unset -> attempt live below

if st.session_state.data_mode is None:
    try:
        with st.spinner("Connecting to World Bank..."):
            _live_result = _cached_fetch_live(COUNTRY_ISO3_LIST, LIVE_START_YEAR, LIVE_END_YEAR, _config_version())
        st.session_state.live_result = _live_result
        st.session_state.live_error = None
        st.session_state.data_mode = data_state.LIVE
    except LiveDataUnavailable as exc:
        st.session_state.live_result = None
        st.session_state.live_error = exc
        if _is_offline():
            # Explicit offline/dev signal: keep the explicit demo-selection
            # dead-end rather than silently substituting data.
            st.session_state.data_mode = data_state.UNAVAILABLE
        elif PANEL_PATH.exists():
            st.session_state.data_mode = data_state.CACHED
        elif DEMO_PANEL_PATH.exists():
            st.session_state.data_mode = data_state.DEMO
        else:
            st.session_state.data_mode = data_state.UNAVAILABLE

live_result = st.session_state.get("live_result")
live_error = st.session_state.get("live_error")

if st.session_state.data_mode == data_state.UNAVAILABLE:
    st.markdown(
        """
        <div class="card" style="margin-top:24px;">
            <div class="card-label">LIVE DATA UNAVAILABLE</div>
            <div class="card-value" style="font-size:22px;">
                Official public data could not be verified right now.
            </div>
            <div class="card-caption" style="margin-top:10px;">
                Source: World Bank Indicators API &middot; Status: Unavailable
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if live_error is not None:
        st.caption(str(live_error))
        if live_error.technical_detail:
            with st.expander("Technical details"):
                st.code(live_error.technical_detail)
    if st.button("Open Demo Dataset"):
        st.session_state.data_mode = data_state.DEMO
        st.rerun()
    st.caption("Reloading this page will automatically retry live data.")
    st.stop()

USING_DEMO_DATA = st.session_state.data_mode == data_state.DEMO
USING_CACHED_DATA = st.session_state.data_mode == data_state.CACHED
live_provenance = None

if USING_DEMO_DATA:
    if not DEMO_PANEL_PATH.exists():
        st.error("Demo dataset file is missing (data/demo/panel_wide.csv).")
        st.stop()
    panel = load_panel(DEMO_PANEL_PATH)
    st.error(
        "DEMO DATA — SYNTHETIC DATASET. For interface / methodology "
        "demonstration only — these are not real economic observations."
    )
elif USING_CACHED_DATA:
    if not PANEL_PATH.exists():
        st.session_state.data_mode = data_state.UNAVAILABLE
        st.rerun()
    panel = load_panel(PANEL_PATH)
    if live_error is not None:
        st.warning(
            f"USING CACHED DATA — the live World Bank fetch was unavailable or "
            f"too slow, so this view shows the last successful pipeline panel "
            f"instead of freshly fetched data. {str(live_error)}"
        )
    else:
        st.warning(
            "USING CACHED DATA — showing the last successful pipeline panel; "
            "the live World Bank fetch was unavailable or too slow."
        )
else:
    if live_result is None:
        st.error("No data source is available.")
        st.stop()
    panel = live_result.wide_panel
    live_provenance = live_result.provenance


# ============================================================================
# SIDEBAR — WORKSPACE CONTEXT + PRIMARY NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:4px 4px 16px;">
            <div class="kicker">GLOBAL COUNTRY RISK</div>
            <div style="font-family:'Space Grotesk';font-size:21px;font-weight:700;">Intelligence Engine</div>
            <div class="micro" style="margin-top:7px;">
                Evidence first. Context always visible.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    country_options = available_countries(panel)

    if not country_options:
        st.error("No country dimension found in panel_wide.csv.")
        st.stop()

    default_country = country_options[0]

    country = st.selectbox(
        "Country",
        country_options,
        index=country_options.index(default_country),
        format_func=get_country_label,
        key="country_selector",
    )

    years = available_years(panel)
    if not years:
        st.error("No year dimension found in panel_wide.csv.")
        st.stop()

    year = st.selectbox(
        "Analysis year",
        years,
        index=len(years) - 1,
        key="year_selector",
        help="Annual analysis year. This may lag the date the source was retrieved.",
    )

    st.markdown("---")

    st.markdown(
        '<div class="kicker" style="margin-bottom:7px;">PRIMARY NAVIGATION</div>',
        unsafe_allow_html=True,
    )

    # Keep the historic internal values so existing bookmarked/tested routes
    # remain valid, while format_func gives users the new product language.
    nav_options = [
        "Overview",
        "Country Intelligence",
        "Compare",
        "Scenario Lab",
        "Early Warning",
        "Contagion & Correlations",
        "Track Record & Model Validation",
        "Data & Methodology",
    ]
    nav_labels = {
        "Overview": "Global Pulse",
        "Track Record & Model Validation": "Validation",
        "Contagion & Correlations": "Global Linkages",
    }
    page = st.radio(
        "Primary navigation",
        nav_options,
        index=0,
        format_func=lambda value: nav_labels.get(value, value),
        key="page_nav",
        help="Move from global orientation to country detail, scenarios, monitoring, and evidence.",
    )

    st.markdown("---")

    refresh_col1, refresh_col2 = st.columns(2)

    with refresh_col1:
        if USING_DEMO_DATA:
            if st.button("Return to Live Data", width="stretch"):
                st.session_state.data_mode = None
                st.session_state.live_result = None
                st.session_state.live_error = None
                _cached_fetch_live.clear()
                st.rerun()
        elif USING_CACHED_DATA:
            if st.button("↻ Retry Live Data", width="stretch"):
                st.session_state.data_mode = None
                st.session_state.live_result = None
                st.session_state.live_error = None
                _cached_fetch_live.clear()
                st.rerun()
        else:
            if st.button("↻ Refresh Live Data", width="stretch"):
                st.session_state.data_mode = None
                st.session_state.live_result = None
                st.session_state.live_error = None
                _cached_fetch_live.clear()
                st.rerun()

    with refresh_col2:
        if st.button("Reset View", width="stretch"):
            for key in ("country_selector", "year_selector", "page_nav"):
                st.session_state.pop(key, None)
            st.rerun()

    st.caption(
        f"Data mode: {'DEMO' if USING_DEMO_DATA else 'CACHED' if USING_CACHED_DATA else 'LIVE'}"
    )


# ============================================================================
# EXECUTION CONTEXT
# ============================================================================

scores = score_panel(panel, indicators_cfg)
selected_country = get_iso(country)
selected_row = row_for(scores, selected_country, year)

ctx = Context(
    panel=panel,
    scores=scores,
    country=selected_country,
    year=year,
    countries_cfg=countries_cfg,
    indicators_cfg=indicators_cfg,
    peer_groups=peer_groups,
    using_demo=USING_DEMO_DATA,
    using_cached=USING_CACHED_DATA,
    live_provenance=live_provenance,
    live_error=live_error,
)


# ============================================================================
# PAGE RENDERING
# ============================================================================

if page == "Overview":
    pulse_section.render(ctx)
elif page == "Country Intelligence":
    country_section.render(ctx)
elif page == "Compare":
    comparison_section.render(ctx)
elif page == "Scenario Lab":
    scenario_section.render(ctx)
elif page == "Early Warning":
    fx_deviation_section.render(ctx)
elif page == "Contagion & Correlations":
    contagion_section.render(ctx)
elif page == "Track Record & Model Validation":
    track_record_section.render(ctx)
elif page == "Data & Methodology":
    methodology_section.render(ctx)


# ============================================================================
# OPTIONAL SECONDARY PANELS / PDF EXPORT
# ============================================================================

with st.expander("More analysis & tools"):
    tool_col1, tool_col2 = st.columns(2)
    with tool_col1:
        st.markdown("### Benchmark")
        benchmark_section.render(ctx)
    with tool_col2:
        st.markdown("### PDF export")
        pdf_export_section.render(ctx)

with st.expander("About the engine"):
    about_section.render(ctx)
