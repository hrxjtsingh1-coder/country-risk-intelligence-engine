"""
Streamlit dashboard for the Global Country Risk Intelligence Engine.
Live data integration with automatic World Bank/FRED fetching and state management.

Run with:
    streamlit run dashboard/app.py

This dashboard now fetches live data automatically from official sources and
falls back to demo data when needed, with full state management and user controls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Core imports for analytics
from src.commentary.generate_commentary import generate_report
from src.scenario.scenario_engine import run_shock_scenario
from src.scoring.risk_score import score_panel, top_drivers

# Live data runtime imports
from src.runtime.live_data import (
    fetch_live_data, 
    validate_panel_data, 
    get_latest_common_year, 
    create_wide_panel_from_long
)
from src.runtime.data_state import (
    DataState, 
    DataStateMachine, 
    get_data_freshness_info,
    validate_state_transition,
    create_default_session_state,
    initialize_streamlit_session_state
)
from src.runtime.provenance import ProvenanceManager, ProvenanceRecord

# UI components
from src.ui.components import (
    render_data_status_banner,
    render_controls,
    render_demo_banner,
    render_return_to_live_button
)

from src.ui.charts import (
    create_risk_score_chart,
    create_peer_comparison_chart,
    create_risk_heatmap
)

# Configuration paths
CONFIG_DIR = ROOT / "config"
PROCESSED_DIR = ROOT / "data" / "processed"
PANEL_PATH = PROCESSED_DIR / "panel_wide.csv"
DEMO_PANEL_PATH = ROOT / "data" / "demo" / "panel_wide.csv"

# ============================================================================
# COUNTRY RISK INTELLIGENCE ENGINE
# Production UI / UX layer with live data integration
# ============================================================================

st.set_page_config(
    page_title="Country Risk Intelligence Engine",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state machine
initialize_streamlit_session_state(st)

# Access session state objects
state_machine = st.session_state.data_state_machine
session_state = st.session_state

# Load configurations
@st.cache_data(ttl=3600)  # Cache configs for 1 hour
def load_configurations():
    countries = yaml.safe_load(
        (CONFIG_DIR / "countries.yaml").read_text(encoding="utf-8")
    )
    indicators = yaml.safe_load(
        (CONFIG_DIR / "indicators.yaml").read_text(encoding="utf-8")
    )
    return countries, indicators

countries_cfg, indicators_cfg = load_configurations()
peer_groups = (
    countries_cfg.get("peer_groups", {})
    if isinstance(countries_cfg, dict)
    else {}
)
indicator_catalog = {
    str(item.get("code")): item
    for item in (
        indicators_cfg.get("indicators", [])
        if isinstance(indicators_cfg, dict)
        else []
    )
    if isinstance(item, dict) and item.get("code")
}

# Helper functions for panel processing
def get_available_years(panel: pd.DataFrame) -> list[int]:
    """Get sorted list of available years in panel."""
    if panel.empty or "year" not in panel.columns:
        return []
    return sorted([int(y) for y in panel["year"].dropna().unique()])

def get_available_countries(panel: pd.DataFrame) -> list[str]:
    """Get sorted list of available countries in panel.""" 
    if panel.empty or "country_iso3" not in panel.columns:
        return []
    return sorted([str(c).upper() for c in panel["country_iso3"].dropna().unique()])

def safe_float(value, default=0.0):
    """Safely convert value to float."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

# Chart helper functions with zoom disabled
def create_plotly_figure(func, *args, **kwargs):
    """Create a plotly figure with zoom disabled."""
    fig = func(*args, **kwargs)
    # Disable zoom interactions as required
    fig.update_layout(
        dragmode=False,
        modebar_remove=[
            "zoom", "pan", "select", "lasso2d", "preview", "reset", "save",
            "edit", "spatial", "lasso", "contour", "histogram", "colorscale"
        ]
    )
    return fig

# State management functions
def attempt_live_data_fetch():
    """Attempt to fetch live data and update state machine."""
    # Reset state for attempt
    state_machine.transition_to(DataState.LOADING)
    
    # Show loading indicator
    with st.spinner("Fetching live data from official sources..."):
        try:
            # Attempt to fetch live data
            long_panel, metadata = fetch_live_data()
            
            # Validate the data
            is_valid, validation_issues = validate_panel_data(long_panel)
            
            if not is_valid:
                raise ValueError(f"Data validation failed: {'; '.join(validation_issues)}")
            
            # Determine latest valid analysis year
            latest_year = get_latest_common_year(long_panel)
            if latest_year is None:
                raise ValueError("No valid analysis year with sufficient coverage found")
            
            # Update metadata with latest year
            metadata["latest_available_observation"] = latest_year
            
            # Convert to wide format for analytics
            wide_panel = create_wide_panel_from_long(long_panel)
            
            if wide_panel.empty:
                raise ValueError("Failed to create wide panel from live data")
            
            # Check if we have enough data for meaningful analysis
            if len(wide_panel) == 0:
                raise ValueError("No country-year data available after processing")
            
            # Update state machine with live data
            provenance_info = {
                "run_id": metadata.get("run_id"),
                "source_name": "World Bank/FRED",
                "retrieved_at": metadata.get("retrieved_at"),
                "country_count": metadata.get("country_count", 0),
                "indicator_count": metadata.get("indicator_count", 0),
                "latest_available_observation": latest_year,
                "fetch_statistics": metadata.get("fetch_statistics", {}),
                "retrieved_at_formatted": metadata.get("retrieved_at", "")
            }
            
            state_machine.set_live_data_available(
                provenance=provenance_info,
                is_fresh=True  # Just fetched, so it's fresh
            )
            
            # Store data in session state
            session_state.live_panel = wide_panel
            session_state.long_panel = long_panel
            session_state.current_provenance = metadata
            
            # Update provenance manager
            provenance_manager = ProvenanceManager()
            provenance_record = ProvenanceRecord(
                run_id=metadata.get("run_id", "unknown"),
                retrieved_at=datetime.fromisoformat(metadata.get("retrieved_at", datetime.now(timezone.utc).isoformat()).replace("Z", "+00:00")),
                source_name="World Bank/FRED",
                source_endpoint="https://api.worldbank.org/v2",
                source_series="Multiple indicators",
                requested_period=f"{metadata.get('requested_period', 'unknown')}",
                latest_observation=latest_year,
                country_count=metadata.get("country_count", 0),
                indicator_count=metadata.get("indicator_count", 0),
                expected_observations=metadata.get("fetch_statistics", {}).get("total_requests", 0),
                received_observations=metadata.get("fetch_statistics", {}).get("successful_requests", 0),
                coverage=metadata.get("fetch_statistics", {}).get("successful_requests", 0) / max(metadata.get("fetch_statistics", {}).get("total_requests", 1), 1),
                validation_failures=len(validation_issues) if not is_valid else 0
            )
            provenance_manager.add_provenance_record(provenance_record)
            
            return True, metadata
            
        except Exception as e:
            # Handle failure
            error_details = {
                "source_name": "World Bank/FRED",
                "operation": "Data fetching and validation",
                "details": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            state_machine.set_unavailable(error_details)
            return False, error_details

def load_demo_data():
    """Load demo data and switch to demo mode."""
    try:
        if DEMO_PANEL_PATH.exists():
            demo_panel = pd.read_csv(DEMO_PANEL_PATH)
            session_state.demo_panel = demo_panel
            state_machine.set_demo_mode()
            return True
        else:
            st.error(f"Demo data not found at {DEMO_PANEL_PATH}")
            return False
    except Exception as e:
        st.error(f"Failed to load demo data: {e}")
        return False

def get_current_panel() -> tuple[pd.DataFrame, bool]:
    """
    Get the currently active panel based on state.
    
    Returns:
        Tuple of (panel_dataframe, is_demo_mode)
    """
    if state_machine.is_demo_mode() and hasattr(session_state, 'demo_panel'):
        return session_state.demo_panel, True
    elif state_machine.is_live_state() and hasattr(session_state, 'live_panel'):
        return session_state.live_panel, False
    else:
        # Fallback: try to load existing processed data or demo
        if PANEL_PATH.exists():
            try:
                panel = pd.read_csv(PANEL_PATH)
                return panel, False
            except:
                pass
        
        if DEMO_PANEL_PATH.exists():
            demo_panel = pd.read_csv(DEMO_PANEL_PATH)
            return demo_panel, True
            
        return pd.DataFrame(), False

def handle_refresh_live_data():
    """Handle refresh live data button click."""
    state_machine.set_loading()
    success, result = attempt_live_data_fetch()
    if success:
        st.rerun()
    else:
        st.rerun()

# ============================================================================
# MAIN APPLICATION LOGIC
# ============================================================================

# Main application header
st.markdown(
    """
    <div class="hero">
        <h1>Country Risk Intelligence Engine</h1>
        <p class="kicker">GLOBAL MACRO · COUNTRY RISK INTELLIGENCE</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Auto-attempt live data fetch on first load if no data available
if not hasattr(session_state, 'live_panel') and not hasattr(session_state, 'demo_panel'):
    if state_machine.current_state == DataState.UNAVAILABLE:
        # Try to fetch live data on first load
        success, result = attempt_live_data_fetch()
        if not success:
            # If live fetch fails, offer demo data
            pass  # Will show unavailable state with demo button

# Handle button interactions from sidebar/state changes
if hasattr(session_state, '_refresh_triggered') and session_state._refresh_triggered:
    session_state._refresh_triggered = False
    handle_refresh_live_data()

# Get current data state info
current_state = state_machine.current_state
provenance = getattr(session_state, 'current_provenance', None)

# Render data status banner
render_data_status_banner(
    state=current_state.value,
    provenance=provenance
)

# Get current panel data
panel_data, is_demo_mode = get_current_panel()

# If we have data, render the dashboard
if not panel_data.empty:
    # Get available countries and years
    available_countries = get_available_countries(panel_data)
    available_years = get_available_years(panel_data)
    
    # Set defaults if not already set
    if session_state.selected_country not in available_countries:
        # Default to India if available, else first country
        default_country = "IND" if "IND" in available_countries else (available_countries[0] if available_countries else "")
        session_state.selected_country = default_country
    
    if session_state.selected_year not in available_years:
        # Default to latest available year
        session_state.selected_year = max(available_years) if available_years else 2020
    
    if session_state.selected_peer_group not in ["Global", "Advanced", "Emerging", "Regional"]:
        session_state.selected_peer_group = "Global"
    
    # Render controls in sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Country/Year/Peer controls
        selected_country, selected_year, selected_peer_group = render_controls(
            countries=available_countries,
            country_names={c: c for c in available_countries},  # Simplified - would use country names from config
            selected_country=session_state.selected_country,
            selected_year=session_state.selected_year,
            selected_peer_group=session_state.selected_peer_group,
            available_years=available_years,
            on_country_change=lambda c: setattr(session_state, 'selected_country', c),
            on_year_change=lambda y: setattr(session_state, 'selected_year', y),
            on_peer_change=lambda p: setattr(session_state, 'selected_peer_group', p)
        )
        
        # Update session state
        session_state.selected_country = selected_country
        session_state.selected_year = selected_year
        session_state.selected_peer_group = selected_peer_group
        
        st.divider()
        
        # Live data controls
        if state_machine.is_live_state():
            if st.button("🔄 Refresh Live Data", type="secondary", use_container_width=True):
                session_state._refresh_triggered = True
                st.rerun()
        elif state_machine.is_unavailable():
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Retry", type="primary", use_container_width=True):
                    session_state._refresh_triggered = True
                    st.rerun()
            with col2:
                if st.button("📂 Open Demo Dataset", type="secondary", use_container_width=True):
                    if load_demo_data():
                        st.rerun()
        
        # Demo mode controls
        if state_machine.is_demo_mode():
            if st.button("← Return to Live Data", type="primary", use_container_width=True):
                # Clear demo data from session state if needed
                if hasattr(session_state, 'demo_panel'):
                    delattr(session_state, 'demo_panel')
                state_machine.transition_to(DataState.UNAVAILABLE)  # Will trigger retry
                st.rerun()
        
        st.divider()
        
        # Show data freshness info if available
        if provenance:
            freshness_info = get_data_freshness_info(provenance)
            if freshness_info.get("retrieved_at_formatted"):
                st.caption(f"Data retrieved: {freshness_info['retrieved_at_formatted']}")
            if freshness_info.get("latest_observation_year"):
                st.caption(f"Latest observation: {freshness_info['latest_observation_year']}")
        
        # Mode indicator
        if state_machine.is_demo_mode():
            st.error("📊 DEMO MODE - Synthetic Data", icon="📊")
        elif state_machine.is_live_state():
            st.success("📡 LIVE DATA - Public Sources", icon="📡")
        elif state_machine.is_unavailable():
            st.error("📡 LIVE DATA - Unavailable", icon="📡")
        
        # Show data freshness info if available
        if provenance:
            freshness_info = get_data_freshness_info(provenance)
            if freshness_info.get("retrieved_at_formatted"):
                st.caption(f"Data retrieved: {freshness_info['retrieved_at_formatted']}")
            if freshness_info.get("latest_observation_year"):
                st.caption(f"Latest observation: {freshness_info['latest_observation_year']}")
        
        # Show demo mode banner if in demo
        if state_machine.is_demo_mode():
            render_demo_banner()
        
        # Return to live button
        if state_machine.is_demo_mode():
            render_return_to_live_button()
        
        st.divider()
        
        # Show data freshness info if available
        if provenance:
            freshness_info = get_data_freshness_info(provenance)
            if freshness_info.get("retrieved_at_formatted"):
                st.caption(f"Data retrieved: {freshness_info['retrieved_at_formatted']}")
            if freshness_info.get("latest_observation_year"):
                st.caption(f"Latest observation: {freshness_info['latest_observation_year']}")

    # Main dashboard content
    st.header("Country Risk Intelligence")
    
    # Compute scores on the fly from panel data if not precomputed
    if "risk_score" not in panel_data.columns:
        try:
            # Compute scores using the scoring engine
            scores_df, drivers_df = score_panel(panel_data)
            # Merge scores back into panel
            panel_data = panel_data.merge(
                scores_df[["country_iso3", "year", "risk_score", "risk_band", "data_completeness"]],
                on=["country_iso3", "year"],
                how="left",
                suffixes=("", "_scored")
            )
        except Exception as e:
            st.error(f"Failed to compute scores: {e}")
            st.stop()
    
    # Get data for selected country/year
    country_data = panel_data[
        (panel_data["country_iso3"] == session_state.selected_country) & 
        (panel_data["year"] == session_state.selected_year)
    ]
    
    if country_data.empty:
        st.warning(f"No data available for {session_state.selected_country} in {session_state.selected_year}")
        st.stop()
    
    # Score metrics
    col1, col2, col3, col4 = st.columns(4)
    
    risk_score = safe_float(country_data.iloc[0].get("risk_score", 0.0))
    risk_band = country_data.iloc[0].get("risk_band", "Moderate")
    yoy_change = safe_float(country_data.iloc[0].get("risk_score_yoy_change", 0.0))
    peer_position = "—"
    
    # Calculate peer position if we have scores
    if "risk_score" in panel_data.columns:
        peer_scores = panel_data[
            panel_data["country_iso3"] != session_state.selected_country
        ]["risk_score"].dropna()
        if not peer_scores.empty:
            rank = (peer_scores > risk_score).sum() + 1
            peer_position = f"{int(rank)} / {int(len(peer_scores)) + 1}"
    
    with col1:
        st.metric(
            label="Risk Score",
            value=f"{risk_score:.1f}",
            delta=f"{yoy_change:+.1f}" if yoy_change != 0 else None,
            help="Relative risk score (0-100 scale)"
        )
    
    with col2:
        band_colors = {
            "Low": "#54d69a",
            "Moderate": "#f6d365", 
            "Elevated": "#ff9f5b",
            "High": "#ff7d55",
            "Severe": "#ff6b7a"
        }
        color = band_colors.get(risk_band, "#8c98aa")
        st.markdown(
            f"""
            <div style="
                background-color: {color}20;
                border: 1px solid {color}40;
                border-radius: 8px;
                padding: 12px;
                text-align: center;
            ">
                <div style="font-size: 14px; font-weight: 600; color: {color};">
                    Risk Band
                </div>
                <div style="font-size: 20px; font-weight: 700; color: {color};">
                    {risk_band}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.metric(
            label="YoY Change",
            value=f"{yoy_change:+.1f}",
            help="Year-over-year change in risk score"
        )
    
    with col4:
        st.metric(
            label="Peer Position",
            value=peer_position,
            help="Ranking among peer group (lower is better)"
        )
    
    # What does this mean?
    st.subheader("What does this mean?")
    interpretation = f"The country risk score for {session_state.selected_country} in {session_state.selected_year} is {risk_score:.1f}, which places it in the '{risk_band}' risk category. "
    interpretation += f"This score is derived from a weighted combination of {len(indicator_catalog)} macroeconomic indicators, "
    interpretation += "with higher scores indicating greater relative risk compared to other countries in the sample."
    
    st.info(interpretation)
    
    # Drivers section
    st.subheader("Key Risk Drivers")
    
    # Get drivers data - either compute on the fly or from precomputed
    if "indicator_code" in panel_data.columns:
        # If we have long format data, compute drivers
        drivers_data = panel_data.melt(
            id_vars=["country_iso3", "year"], 
            value_vars=list(indicator_catalog.keys()),
            var_name="indicator_code",
            value_name="raw_value"
        ).dropna()
        
        # Add metadata
        drivers_data["label"] = drivers_data["indicator_code"].map(
            lambda x: indicator_catalog.get(x, {}).get("label", x)
        )
        drivers_data["category"] = drivers_data["indicator_code"].map(
            lambda x: indicator_catalog.get(x, {}).get("category", "Macro")
        )
        drivers_data["unit"] = drivers_data["indicator_code"].map(
            lambda x: indicator_catalog.get(x, {}).get("unit", "")
        )
    else:
        # Try to compute from wide data using scoring functions
        try:
            # Compute scores and drivers
            scores_df, drivers_df = score_panel(panel_data)
            
            # Get drivers for selected country/year
            drivers_data = drivers_df[
                (drivers_df["country_iso3"] == session_state.selected_country) & 
                (drivers_df["year"] == session_state.selected_year)
            ].copy()
            
        except Exception as e:
            st.error(f"Failed to compute drivers: {e}")
            drivers_data = pd.DataFrame()
    
    if not drivers_data.empty:
        # Get top drivers
        top_drivers_df = top_drivers(drivers_data, session_state.selected_country, session_state.selected_year, n=6)
        
        if not top_drivers_df.empty:
            for _, driver in top_drivers_df.iterrows():
                contribution = safe_float(driver.get("weighted_contribution", 0.0))
                label = driver.get("label", driver.get("indicator_code", "Unknown"))
                unit = driver.get("unit", "")
                raw_value = safe_float(driver.get("raw_value", 0.0))
                
                # Determine risk direction and color
                risk_direction = "increases" if contribution > 0 else "decreases" if contribution < 0 else "no effect"
                color = "#ff6b7a" if contribution > 0 else "#54d69a" if contribution < 0 else "#8c98aa"
                
                with st.container():
                    st.markdown(
                        f"""
                        <div style="
                            border: 1px solid rgba(255,255,255,0.1);
                            border-radius: 8px;
                            padding: 12px;
                            margin: 8px 0;
                        ">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <div style="font-weight: 600; font-size: 16px; margin-bottom: 4px;">
                                        {label}
                                    </div>
                                    <div style="font-size: 12px; color: #8c98aa; margin-bottom: 4px;">
                                        {raw_value:,.1f}{unit if unit else ''}
                                    </div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="
                                        background-color: {color}20;
                                        border: 1px solid {color}40;
                                        border-radius: 6px;
                                        padding: 4px 8px;
                                        font-weight: 600;
                                        font-size: 14px;
                                    ">
                                        {contribution:+.1f} pts
                                    </div>
                                    <div style="font-size: 11px; color: #8c98aa; margin-top: 4px;">
                                        Risk contribution ({risk_direction})
                                    </div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("Driver analysis not available for this data configuration.")
    
    # Risk trajectory chart
    st.subheader("Risk Trajectory")
    
    # Prepare data for trajectory chart
    trajectory_data = panel_data[
        panel_data["country_iso3"] == session_state.selected_country
    ].copy()
    # Only select columns that actually exist
    available_cols = ["year"]
    if "risk_score" in trajectory_data.columns:
        available_cols.append("risk_score")
    trajectory_data = trajectory_data[available_cols].dropna()
    
    if not trajectory_data.empty and len(trajectory_data) > 1:
        fig = create_plotly_figure(
            create_risk_score_chart,
            trajectory_data,
            session_state.selected_country,
            session_state.selected_year
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Insufficient historical data for trajectory chart.")
    
    # Peer comparison
    st.subheader("Peer Position")
    
    # Add peer group info to data for charting
    chart_data = panel_data.copy()
    # In a full implementation, we'd add peer_group column based on config
    # For now, use simplified approach
    
    peer_fig = create_plotly_figure(
        create_peer_comparison_chart,
        chart_data,
        session_state.selected_country,
        session_state.selected_year,
        session_state.selected_peer_group
    )
    st.plotly_chart(peer_fig, use_container_width=True, config={"displayModeBar": False})
    
    # Scenario Lab
    with st.expander("Scenario Lab", expanded=False):
        st.caption("Historical sensitivity analysis - not a causal forecast")
        
        # Scenario controls
        scen_col1, scen_col2 = st.columns(2)
        with scen_col1:
            scenario_driver = st.selectbox(
                "Shock Driver",
                options=["POLICY_RATE_YOY_CHANGE_BPS", "GC.DOD.TOTL.GD.ZS", "NY.GDP.MKTP.KD.ZG"],
                index=0,
                help="Select indicator to shock"
            )
        with scen_col2:
            scenario_amount = st.slider(
                "Shock Amount",
                min_value=-100,
                max_value=100,
                value=25,
                step=1,
                help="Magnitude of shock to apply"
            )
        
        if st.button("Run Scenario", type="primary"):
            try:
                with st.spinner("Running scenario analysis..."):
                    # This would use the actual scenario engine
                    scenario_result = run_shock_scenario(
                        panel_data,
                        session_state.selected_country,
                        session_state.selected_year,
                        scenario_driver,
                        scenario_amount,
                        ["FX_YOY_DEPRECIATION_PCT", "NY.GDP.MKTP.KD.ZG", "GC.DOD.TOTL.GD.ZS"]
                    )
                    
                    # Display results
                    st.success("Scenario analysis complete")
                    
                    scen_col1, scen_col2 = st.columns(2)
                    with scen_col1:
                        st.metric(
                            label="Baseline Score",
                            value=f"{scenario_result.get('baseline_score', 0):.1f}",
                            delta=None
                        )
                        st.metric(
                            label="Baseline Band", 
                            value=scenario_result.get('baseline_band', 'Unknown'),
                            delta=None
                        )
                    with scen_col2:
                        st.metric(
                            label="Scenario Score", 
                            value=f"{scenario_result.get('scenario_score', 0):.1f}",
                            delta=None
                        )
                        st.metric(
                            label="Scenario Band",
                            value=scenario_result.get('scenario_band', 'Unknown'),
                            delta=None
                        )
                    
                    st.metric(
                        label="Score Delta",
                        value=f"{scenario_result.get('delta', 0):.1f}",
                        delta=None
                    )
                    
                    st.caption(f"Information quality: {scenario_result.get('information_assessment', 'Unknown')}")
                    
            except Exception as e:
                st.error(f"Scenario analysis failed: {e}")
    
    # Data Quality & Methodology
    st.subheader("Data Quality & Methodology")
    
    qual_col1, qual_col2 = st.columns(2)
    
    with qual_col1:
        st.markdown("**Data Coverage**")
        if provenance:
            stats = provenance.get("fetch_statistics", {})
            if stats:
                total_req = stats.get("total_requests", 0)
                success_req = stats.get("successful_requests", 0)
                coverage_pct = (success_req / total_req * 100) if total_req > 0 else 0
                
                st.metric("Indicator Coverage", f"{coverage_pct:.1f}%")
                st.metric("Countries Covered", stats.get("country_count", 0))
                st.metric("Indicators Covered", stats.get("indicator_count", 0))
                st.metric("Observations Received", f"{stats.get('observations_received', 0):,}")
                
                if stats.get("errors"):
                    with st.expander(f"⚠️ {len(stats['errors'])} Warnings", expanded=False):
                        for error in stats["errors"][:5]:  # Show first 5 errors
                            st.caption(f"• {error}")
            else:
                st.info("Fetch statistics not available")
        else:
            st.info("Live data provenance not available")
    
    with qual_col2:
        st.markdown("**Methodology**")
        st.caption("""
        **Scoring Method**: Cross-sectional z-score weighting
        **Data Sources**: World Bank Indicators API v2, FRED (US-only)
        **Update Frequency**: Automatic 6-hour cache refresh
        **Peer Groups**: Global, Advanced, Emerging, Regional
        """)
        
        # Show technical details toggle
        if st.checkbox("Show technical details"):
            if provenance:
                with st.expander("Live Data Provenance", expanded=False):
                    st.json({
                        "run_id": provenance.get("run_id"),
                        "retrieved_at": provenance.get("retrieved_at"),
                        "sources": provenance.get("sources", []),
                        "requested_period": provenance.get("requested_period"),
                        "fetch_statistics": provenance.get("fetch_statistics", {})
                    })
            else:
                st.info("No provenance data available")
    
    # Sources and disclaimer
    st.divider()
    
    sources_col, disclaimer_col = st.columns([1, 2])
    
    with sources_col:
        st.markdown("**Sources**")
        st.caption("• World Bank Indicators API")
        st.caption("• FRED DFF (US-only enrichment)")
        if state_machine.is_live_state():
            st.caption("• Live data fetched at runtime")
        elif state_machine.is_demo_mode():
            st.caption("• Synthetic demonstration dataset")
    
    with disclaimer_col:
        st.markdown("""
        <div style="font-size: 12px; color: #8c98aa; line-height: 1.4;">
        For research and educational use. This relative-risk model is not a credit rating, 
        probability of default, investment recommendation, or professional financial advice. 
        The engine provides comparative risk positioning based on publicly available macroeconomic 
        indicators. Past patterns do not guarantee future results.
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    # No data available state - show appropriate UI based on state machine
    if state_machine.current_state == DataState.LOADING:
        st.info("🔄 **Fetching official public data…**\n\nValidating observations from World Bank Indicators API v2…", icon="⏳")
    elif state_machine.current_state == DataState.UNAVAILABLE:
        # Show unavailable state with retry and demo options
        error_msg = "Official public data could not be retrieved right now."
        if hasattr(session_state, 'last_error'):
            error_msg += f"\\n\\nLast error: {session_state.last_error}"
        
        st.error(f"**LIVE DATA UNAVAILABLE**\\n\\n{error_msg}", icon="🚫")
        
        retry_col, demo_col = st.columns(2)
        with retry_col:
            if st.button("🔄 Retry Live Data", type="primary", use_container_width=True):
                session_state._refresh_triggered = True
                st.rerun()
        with demo_col:
            if st.button("📂 Open Demo Dataset", type="secondary", use_container_width=True):
                if load_demo_data():
                    st.rerun()
    elif state_machine.current_state == DataState.DEMO:
        # Should have demo data loaded, but fallback
        st.info("**DEMO DATA — SYNTHETIC DATASET**\\n\\nFor methodology/interface demonstration only.", icon="📊")
        if st.button("Load Demo Data", type="primary"):
            if load_demo_data():
                st.rerun()
    else:
        # Default fallback
        st.info("Initializing data fetch... Please wait.", icon="⏳")
        # Trigger initial load
        if st.button("Start Live Data Fetch", type="primary"):
            session_state._refresh_triggered = True
            st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown(
    f"""
    <div class="footer">
        <span>COUNTRY RISK INTELLIGENCE ENGINE</span>
        <span>{session_state.selected_country if hasattr(session_state, 'selected_country') else '—'} / {session_state.selected_year if hasattr(session_state, 'selected_year') else '—'} · ANALYTICAL CORE INTACT</span>
        <span>UI BUILD · {datetime.now(timezone.utc).strftime('%d %b %Y %H:%M UTC')}</span>
    </div>
    """,
    unsafe_allow_html=True
)