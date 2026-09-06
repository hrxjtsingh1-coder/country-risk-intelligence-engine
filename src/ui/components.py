"""
UI components for the Country Risk Intelligence Engine.
Includes data status banners, control components, and helper elements.
"""

from __future__ import annotations

import streamlit as st
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pathlib import Path
import pandas as pd


def render_data_status_banner(
    state: str,  # "LIVE", "STALE", "UNAVAILABLE", "DEMO", "LOADING"
    provenance: Optional[Dict[str, Any]] = None,
    error_details: Optional[Dict[str, Any]] = None,
    live_source_info: Optional[str] = None
) -> None:
    """
    Render a compact, professional data status component.
    
    Args:
        state: Current data state
        provenance: Optional provenance data
        error_details: Optional error details for unavailable state
        live_source_info: Source information
    """
    if state == "LOADING":
        st.info(
            "🔄 **Fetching official public data…**\n\n"
            "Validating observations from World Bank Indicators API v2…\n"
            "Building country-risk panel from verified observations.",
            icon="⏳"
        )
        return
    
    if state == "LIVE":
        # Extract info from provenance if available
        source_name = provenance.get("source_name", "World Bank") if provenance else "World Bank"
        retrieved_at = provenance.get("retrieved_at") if provenance else None
        latest_year = provenance.get("latest_available_observation") if provenance else None
        country_count = provenance.get("country_count") if provenance else None
        
        # Format retrieval time
        retrieved_text = ""
        if retrieved_at:
            try:
                dt = datetime.fromisoformat(str(retrieved_at).replace("Z", "+00:00"))
                retrieved_text = dt.strftime("%d %b %Y · %H:%M UTC")
            except (ValueError, TypeError):
                retrieved_text = str(retrieved_at)
        
        # Build coverage text
        coverage_text = ""
        if provenance and "fetch_statistics" in provenance:
            stats = provenance["fetch_statistics"]
            if "successful_requests" in stats and "total_requests" in stats:
                coverage = stats["successful_requests"] / stats["total_requests"] if stats["total_requests"] > 0 else 0
                coverage_text = f"Coverage: {coverage:.1%}"
        
        # Build header content
        header_content = f"""**LIVE PUBLIC DATA**
{source_name}
"""
        if retrieved_text:
            header_content += f"Retrieved {retrieved_text}\n"
        if latest_year:
            header_content += f"Latest valid analysis year: {latest_year}\n"
        if coverage_text:
            header_content += f"{coverage_text}"
        
        # Render as a clean status banner
        st.success(
            header_content,
            icon="✅"
        )
        return
    
    if state == "STALE":
        retrieved_text = ""
        if provenance and provenance.get("retrieved_at"):
            retrieved_at = provenance.get("retrieved_at")
            try:
                dt = datetime.fromisoformat(str(retrieved_at).replace("Z", "+00:00"))
                age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                retrieved_text = f"{dt.strftime('%d %b %Y %H:%M')} UTC ({age_hours:.0f}h old)"
            except (ValueError, TypeError):
                retrieved_text = str(retrieved_at)
        
        latest_year = provenance.get("latest_available_observation") if provenance else None
        
        content = f"**STALE DATA — USING CACHED LIVE DATA**\nCache age: {retrieved_text or 'unknown'}. Latest valid year: {latest_year or 'unknown'}. Data remains valid under documented policy but should be refreshed."
        
        st.warning(content, icon="⚠️")
        return
    
    if state == "UNAVAILABLE":
        source_name = error_details.get("source_name", "World Bank") if error_details else "World Bank"
        failed_operation = error_details.get("operation", "Data retrieval") if error_details else "Data retrieval"
        
        content = f"""**LIVE DATA UNAVAILABLE**
{source_name} data could not be verified.
Failed operation: {failed_operation}.

Official public data could not be retrieved right now."""
        
        # Add retry and demo buttons
        col1, col2 = st.columns(2)
        with col1:
            st.button("🔄 Retry", type="primary", key="retry_live_button", use_container_width=True)
        with col2:
            st.button("📂 Open Demo Dataset", key="open_demo_button_unavailable", use_container_width=True)
        
        # Show error details under expander
        with st.expander("Technical details"):
            if error_details:
                st.code(str(error_details.get("details", error_details)), language="json")
        
        return
    
    if state == "DEMO":
        content = "**DEMO DATA — SYNTHETIC DATASET**\n\nFor methodology/interface demonstration only. Not live public data."
        
        # Show banner prominently but compact
        st.info(content, icon="📊")
        return
    
    # Default / unknown state
    st.info("Data status: Unknown — please refresh or try again.", icon="❓")


def render_controls(
    countries: List[str],
    country_names: Dict[str, str],
    selected_country: str,
    selected_year: int,
    selected_peer_group: str,
    available_years: List[int],
    on_country_change: Optional[callable] = None,
    on_year_change: Optional[callable] = None,
    on_peer_change: Optional[callable] = None
) -> Tuple[str, int, str]:
    """
    Render country/year/peer controls.
    
    Args:
        countries: List of available country codes
        country_names: Mapping from code to display name
        selected_country: Currently selected country
        selected_year: Currently selected year
        selected_peer_group: Currently selected peer group
        available_years: List of available years for selection
        on_*_change: Callbacks for change events
        
    Returns:
        Tuple of (selected_country, selected_year, selected_peer_group)
    """
    # Create three-column layout for controls
    col1, col2, col3 = st.columns([4, 2, 2], gap="small")
    
    with col1:
        country_options = [(code, country_names.get(code, code)) for code in countries]
        country_display = {code: name for code, name in country_options}
        
        selected = st.selectbox(
            "**COUNTRY**",
            options=[code for code, _ in country_options],
            format_func=lambda x: country_display.get(x, x),
            index=[code for code, _ in country_options].index(selected_country) if selected_country in [c for c, _ in country_options] else 0,
            key="country_select"
        )
    
    with col2:
        # Filter available years to those with data
        year_options = sorted([y for y in available_years if y is not None])
        if not year_options:
            year_options = list(range(2012, 2026))
        
        selected_yr = st.selectbox(
            "**YEAR**",
            options=year_options,
            index=min(len(year_options)-1, max(0, year_options.index(selected_year) if selected_year in year_options else len(year_options)-1)),
            key="year_select"
        )
    
    with col3:
        peer_options = ["Global", "Advanced", "Emerging", "Regional"]
        selected_peer = st.selectbox(
            "**PEER GROUP**",
            options=peer_options,
            index=peer_options.index(selected_peer_group) if selected_peer_group in peer_options else 0,
            key="peer_select"
        )
    
    return selected, selected_yr, selected_peer


def render_demo_banner() -> None:
    """Render a small persistent demo mode banner."""
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, rgba(246,211,101,.15), rgba(255,159,91,.15));
            border: 1px solid rgba(246,211,101,.4);
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 1rem;
            font-family: 'DM Mono', monospace;
            font-size: 11px;
            color: #f6d365;
            text-align: center;
        ">
            <strong>DEMO MODE</strong> · Synthetic dataset · For methodology/interface demonstration only · 
            <span style="opacity: 0.7;">Not live public data</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_return_to_live_button() -> None:
    """Render button to return from demo to live mode."""
    if st.button("← Return to Live Data", type="primary", key="return_to_live"):
        # Clear demo from session state if needed
        pass  # The button click will trigger a rerun with state change