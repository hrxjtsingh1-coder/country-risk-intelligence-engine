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
        f"""
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
        st.caption("Panel")
        st.caption(f"{len(panel):,} rows")

    st.markdown("---")

    st.markdown(
        """
        <div class="micro">
            ENGINE STATUS<br>
            <span style="color:#54d69a;">● ONLINE</span><br><br>
            Analytics remain deterministic and traceable.
            The interface is layered on top of the existing engine.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# USER-FACING CONTEXT / SCENARIO WORKSPACE
#
# Scenario controls live in the main workspace rather than the sidebar. This
# keeps the sidebar focused on navigation and the selected analytical context.
# ============================================================================

page_title, page_subtitle = {
    "Overview": ("GLOBAL PULSE", "A compact read of the panel: what is changing, where, and how much coverage exists."),
    "Country Intelligence": ("COUNTRY INTELLIGENCE", "Move from the score to the drivers, evidence, resilience, and what to watch next."),
    "Compare": ("COMPARE", "Understand why countries rank differently instead of comparing two tables in isolation."),
    "Scenario Lab": ("SCENARIO LAB", "Stress-test supported drivers and see the transmission path from shock to composite risk."),
    "Early Warning": ("EARLY WARNING", "Monitor deterioration signals and data-quality warnings without calling them predictions."),
    "Global Linkages": ("GLOBAL LINKAGES", "Explore co-movement and regional structure. Correlation is not causation."),
    "Validation": ("VALIDATION", "See what the engine flagged, missed, or could not judge in historical episodes."),
    "Data & Methodology": ("DATA & METHODOLOGY", "Inspect how raw observations become scores, commentary, exports, and limitations."),
}.get(page, ("GLOBAL PULSE", "A compact read of the panel."))

status_label = "DEMO" if USING_DEMO_DATA else "CACHED" if USING_CACHED_DATA else "LIVE"
status_color = "#ff9f5b" if USING_DEMO_DATA or USING_CACHED_DATA else "#54d69a"
st.markdown(
    f"""
    <div class="context-bar">
        <div>
            <div class="kicker">{esc(page_title)}</div>
            <div class="context-title">{esc(page_subtitle)}</div>
        </div>
        <div class="context-state">
            <span class="context-dot" style="background:{status_color};"></span>
            <strong>{esc(get_country_label(country))}</strong>
            <span>{esc(str(country))}</span>
            <span class="context-divider">/</span>
            <strong>{int(year)}</strong>
            <span class="context-divider">/</span>
            <span>{status_label}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

selected_preset_key = None
shock = 0
if page == "Scenario Lab":
    preset_choices = ["+ None (custom shock)"] + [shock_preset(name)["name"] for name in available_shock_presets()]
    st.markdown(
        """
        <div class="workspace-callout">
            <div class="card-label">SCENARIO WORKSPACE</div>
            <div class="card-caption">Choose a supported preset or define a transparent policy-rate shock. The output below is an estimate, not a forecast.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    scenario_cols = st.columns([1.35, 1, 1])
    with scenario_cols[0]:
        preset_label = st.selectbox("Scenario preset", preset_choices, index=0, key="scenario_lab_preset")
    if preset_label != preset_choices[0]:
        for _name in available_shock_presets():
            if shock_preset(_name)["name"] == preset_label:
                selected_preset_key = _name
                break
    with scenario_cols[1]:
        if selected_preset_key is None:
            shock = st.slider(
                "Policy rate shock (bps)",
                min_value=-1000,
                max_value=1000,
                value=0,
                step=25,
                key="scenario_lab_shock",
            )
        else:
            _preset_config = shock_preset(selected_preset_key)
            shock = float(_preset_config["default_shock_amount"])
            st.metric("Preset shock", f"{shock:g} {_preset_config['display_unit']}")
    with scenario_cols[2]:
        if selected_preset_key is None:
            st.caption("Driver")
            st.code("POLICY_RATE_YOY_CHANGE_BPS", language=None)
        else:
            _preset_config = shock_preset(selected_preset_key)
            st.caption("Supported driver")
            st.code(str(_preset_config["driver_code"]), language=None)


# ============================================================================
# ORIGINAL ANALYTICAL EXECUTION
#
# The calls below intentionally mirror the original dashboard contract:
#
#   scores, drivers = score_panel(panel)
#   top_drivers(drivers, country, year, n=6)
#   run_shock_scenario(panel, ...)
#   generate_report(...)
#
# The UI never substitutes a second scoring methodology.
# ============================================================================

scores, drivers, pillar_scores = score_panel(panel)

if "country_iso3" not in scores.columns or "year" not in scores.columns:
    st.error("Scoring output is missing country_iso3/year columns.")
    st.stop()

row = scores[
    scores["country_iso3"].astype(str).eq(str(country)) & pd.to_numeric(scores["year"], errors="coerce").eq(int(year))
]

if row.empty or pd.isna(row.iloc[0]["risk_score"]):
    st.error(f"No sufficient indicator data to score {get_country_label(country)} in {year}.")
    st.stop()

score_value = safe_float(row.iloc[0]["risk_score"])
band = str(row.iloc[0]["risk_band"])
score_color = band_color(band)
coverage_value = safe_float(row.iloc[0]["data_completeness"], default=float("nan"))

try:
    current_row = row_for(panel, country, year)
except Exception:
    current_row = pd.Series(dtype=object)

try:
    country_drivers = top_drivers(drivers, country, year, n=6)
except Exception as exc:
    country_drivers = pd.DataFrame()
    user_error("Driver decomposition is temporarily unavailable for this slice.", exc)

driver_code = "POLICY_RATE_YOY_CHANGE_BPS"
shock_amount = float(shock)
run_scenario_btn = False
scenario_result = None
scenario = None
scenario_error = None

if selected_preset_key is not None:
    _preset_config = shock_preset(selected_preset_key)
    run_scenario_btn = True
    preset_driver = str(_preset_config["driver_code"])
    if preset_driver not in panel.columns:
        scenario_error = ValueError(
            f"The '{_preset_config['name']}' preset needs the driver series "
            f"{preset_driver}, which is not present in this panel. The collector "
            "now supplies it (see config/indicators.yaml); a freshly built or "
            "refreshed panel will enable this preset as-is."
        )
    else:
        try:
            scenario_result = run_shock_scenario(panel, country, year, preset=selected_preset_key)
            scenario = scenario_result
        except Exception as exc:
            scenario_error = exc
elif not np.isclose(shock_amount, 0.0):
    run_scenario_btn = True
    try:
        scenario_result = run_shock_scenario(
            panel,
            country,
            year,
            driver_code,
            shock_amount,
            [
                "FX_YOY_DEPRECIATION_PCT",
                "NY.GDP.MKTP.KD.ZG",
                "GC.DOD.TOTL.GD.ZS",
            ],
        )
        scenario = scenario_result
    except Exception as exc:
        scenario_error = exc

peer_group = next(
    (members for members in peer_groups.values() if country in members),
    None,
)

source_by_code = {}
for indicator in indicators_cfg.get("indicators", []) if isinstance(indicators_cfg, dict) else []:
    if isinstance(indicator, dict) and indicator.get("code"):
        source_by_code[str(indicator["code"])] = str(indicator.get("source", ""))

_report_sources = None
if (
    isinstance(country_drivers, pd.DataFrame)
    and not country_drivers.empty
    and "indicator_code" in country_drivers.columns
):
    _report_sources = sorted(
        {s for c in country_drivers["indicator_code"] for s in [source_by_code.get(str(c), str(c))]}
    )
    if not _report_sources:
        _report_sources = None

generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

report = generate_report(
    country_name=get_country_label(country),
    country_iso3=country,
    year=int(year),
    scores=scores,
    drivers=drivers,
    scenario_result=scenario_result,
    peer_group=[c for c in (peer_group or []) if c != country],
    sources=_report_sources,
    generated_at=generated_at,
)


# Provenance surfaced under every score and chart: where the numbers came
# from and which exact pipeline manifest hash produced them. Demo data is
# labeled honestly as synthetic and never presented as verified.
if USING_DEMO_DATA:
    data_verified = False
    provenance_sources = ""
    provenance_asof = ""
elif USING_CACHED_DATA:
    data_verified = False
    provenance_sources = "Cached pipeline panel"
    try:
        provenance_asof = datetime.fromtimestamp(PANEL_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        provenance_asof = ""
elif live_provenance is not None:
    data_verified = True
    _prov_sources = getattr(live_provenance, "sources", None) or []
    provenance_sources = ", ".join(str(s.get("name")) for s in _prov_sources if isinstance(s, dict) and s.get("name"))
    provenance_asof = str(getattr(live_provenance, "retrieved_at", "") or "")
else:
    data_verified = False
    provenance_sources = ""
    provenance_asof = ""

manifest_hash = _manifest_short_hash()


# Peer-relative standing for the selected slice. `peer_groups` is loaded from
# config/countries.yaml; a country in no group is compared against the whole
# panel cross-section (consistent with the scoring's own fallback).
peer_info = peer_percentile(scores, country, year, peer_groups=peer_groups) or {}


# ============================================================================
# SHARED CONTEXT + PAGE COMPOSITION
#
# Each product destination composes existing section modules from the same
# Context. Navigation changes the reading path, never the scoring method.
# ============================================================================

ctx = Context(
    panel=panel,
    scores=scores,
    drivers=drivers,
    pillar_scores=pillar_scores,
    indicators_cfg=indicators_cfg,
    country=country,
    year=int(year),
    iso=get_iso(country),
    country_label=get_country_label(country),
    band=band,
    score_value=score_value,
    score_color=score_color,
    coverage_value=coverage_value,
    current_row=current_row,
    country_drivers=country_drivers,
    report=report,
    scenario=scenario,
    scenario_result=scenario_result,
    scenario_error=scenario_error,
    shock=shock,
    run_scenario_btn=run_scenario_btn,
    using_demo_data=USING_DEMO_DATA,
    using_cached_data=USING_CACHED_DATA,
    live_provenance=live_provenance,
    generated_at=generated_at,
    peer_percentile=peer_info.get("percentile"),
    peer_riskier_share=peer_info.get("riskier_share"),
    peer_rank=peer_info.get("rank"),
    peer_n=int(peer_info.get("n") or 0),
    peer_group_name=peer_info.get("group_name") or "",
    data_verified=data_verified,
    provenance_sources=provenance_sources,
    provenance_asof=provenance_asof,
    manifest_hash=manifest_hash,
)

def _render_nav_footer(ctx: Context) -> None:
    """Footer shared by the dedicated nav pages (Track Record, Contagion, ...)."""
    st.markdown(
        f"""
        <div class="footer">
            <span>COUNTRY RISK INTELLIGENCE ENGINE</span>
            <span>{esc(ctx.iso)} / {int(ctx.year)} · ANALYTICAL CORE INTACT</span>
            <span>
                UI BUILD · {ctx.generated_at}
                {f" · MANIFEST {esc(ctx.manifest_hash)}" if ctx.manifest_hash else ""}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if page == "Overview":
    pulse_section.render_global_pulse(ctx)
    map_section.render_map(ctx)
    _render_nav_footer(ctx)
    st.stop()

if page == "Country Intelligence":
    country_section.render_hero(ctx)
    country_section.render_provenance(ctx)
    country_section.render_kpi(ctx)
    country_section.render_interpretation(ctx)
    country_section.render_trajectory(ctx)
    country_section.render_drivers(ctx)
    country_section.render_resilience(ctx)
    fx_deviation_section.render_fx_deviation(ctx)
    country_section.render_analyst_intelligence(ctx)
    country_section.render_data_coverage(ctx)
    _render_nav_footer(ctx)
    st.stop()

if page == "Compare":
    comparison_section.render_peer_comparison(ctx)
    _render_nav_footer(ctx)
    st.stop()

if page == "Scenario Lab":
    scenario_section.render_scenario_laboratory(ctx)
    _render_nav_footer(ctx)
    st.stop()

if page == "Early Warning":
    comparison_section.render_deterioration_watch(ctx)
    _render_nav_footer(ctx)
    st.stop()

if page in {"Global Linkages", "Contagion & Correlations"}:
    contagion_section.render_contagion(ctx)
    _render_nav_footer(ctx)
    st.stop()

if page in {"Validation", "Track Record & Model Validation"}:
    track_record_section.render_track_record(ctx)
    methodology_section.render_model_validation(ctx)
    _render_nav_footer(ctx)
    st.stop()

if page == "Data & Methodology":
    methodology_section.render_model_card(ctx)
    country_section.render_provenance(ctx)
    methodology_section.render_export_inspection(ctx)
    methodology_section.render_engine_integrity(ctx)
    benchmark_section.render_agency_benchmark(ctx)
    pdf_export_section.render_pdf_export(ctx)
    about_section.render_about(ctx)
    _render_nav_footer(ctx)
    st.stop()


# ============================================================================
# FOOTER
# ============================================================================

st.markdown(
    f"""
    <div class="footer">
        <span>COUNTRY RISK INTELLIGENCE ENGINE</span>
        <span>{esc(ctx.iso)} / {int(ctx.year)} · ANALYTICAL CORE INTACT</span>
        <span>
            UI BUILD · {ctx.generated_at}
            {f" · MANIFEST {esc(ctx.manifest_hash)}" if ctx.manifest_hash else ""}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# END
# ============================================================================
