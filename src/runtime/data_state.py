"""
Live data state management for the Country Risk Intelligence Engine.
Manages the application state machine: LIVE, STALE, UNAVAILABLE, DEMO, LOADING
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone
import pandas as pd


class DataState(str, Enum):
    """Explicit application states for data availability."""
    LOADING = "LOADING"
    LIVE = "LIVE"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"
    DEMO = "DEMO"


class DataStateMachine:
    """
    Manages the live/demo state machine and validates transitions.
    """
    
    def __init__(self):
        self.current_state = DataState.UNAVAILABLE
        self.state_history: list[str] = []
        self.last_state_change: Optional[datetime] = None
        self.state_metadata: Dict[str, Any] = {}
        
    def transition_to(self, new_state: DataState, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Transition to a new state with optional metadata."""
        if new_state == self.current_state:
            # Update metadata even for same-state updates
            if metadata is not None:
                self.state_metadata.update(metadata)
            return
        
        # Log transition
        self.state_history.append({
            "from": self.current_state.value,
            "to": new_state.value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        self.current_state = new_state
        self.last_state_change = datetime.now(timezone.utc)
        
        if metadata is not None:
            self.state_metadata.update(metadata)
    
    def is_live_state(self) -> bool:
        """Check if current state is a live data state."""
        return self.current_state in {DataState.LIVE, DataState.STALE}
    
    def is_demo_mode(self) -> bool:
        """Check if currently in demo mode."""
        return self.current_state == DataState.DEMO
    
    def is_unavailable(self) -> bool:
        """Check if live data is unavailable."""
        return self.current_state == DataState.UNAVAILABLE
    
    def can_display_dashboard(self) -> bool:
        """Check if dashboard can be displayed (live or demo, not loading/unavailable)."""
        return self.current_state in {DataState.LIVE, DataState.STALE, DataState.DEMO}
    
    def get_state_description(self) -> str:
        """Get human-readable description of current state."""
        descriptions = {
            DataState.LOADING: "Fetching official public data...",
            DataState.LIVE: "Live public data available",
            DataState.STALE: "Using cached live data (may be stale)",
            DataState.UNAVAILABLE: "Live data unavailable - official API could not be reached",
            DataState.DEMO: "Using synthetic demonstration dataset",
        }
        return descriptions.get(self.current_state, "Unknown state")
    
    def get_display_badge(self) -> Tuple[str, str]:
        """Get badge text and CSS class for current state."""
        badges = {
            DataState.LOADING: ("FETCHING DATA", "badge-loading"),
            DataState.LIVE: ("LIVE PUBLIC DATA", "badge-live"),
            DataState.STALE: ("STALE DATA", "badge-stale"),
            DataState.UNAVAILABLE: ("LIVE DATA UNAVAILABLE", "badge-unavailable"),
            DataState.DEMO: ("DEMO DATA — SYNTHETIC DATASET", "badge-demo"),
        }
        return badges.get(self.current_state, ("UNKNOWN", "badge-unknown"))
    
    def reset_for_retry(self) -> None:
        """Reset state for retry attempt."""
        self.transition_to(DataState.LOADING, {"action": "retry"})
    
    def set_live_data_available(
        self,
        provenance: Optional[Dict[str, Any]] = None,
        is_fresh: bool = True
    ) -> None:
        """Set state to LIVE (or STALE) based on data freshness."""
        new_state = DataState.LIVE if is_fresh else DataState.STALE
        self.transition_to(new_state, provenance or {})
    
    def set_unavailable(self, error_details: Optional[Dict[str, Any]] = None) -> None:
        """Set state to UNAVAILABLE with error details."""
        metadata = {"error_details": error_details or {}, "failed_at": datetime.now(timezone.utc).isoformat()}
        self.transition_to(DataState.UNAVAILABLE, metadata)
    
    def set_demo_mode(self) -> None:
        """Switch to demo mode with synthetic data."""
        self.transition_to(DataState.DEMO, {"mode": "demo", "data_source": "synthetic"})
    
    def set_loading(self) -> None:
        """Set state to LOADING."""
        self.transition_to(DataState.LOADING)


def get_data_freshness_info(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract data freshness information from provenance metadata.
    
    Args:
        metadata: Provenance metadata dict
        
    Returns:
        Dict with freshness information
    """
    if metadata is None:
        return {
            "retrieved_at": None,
            "latest_observation_year": None,
            "is_fresh": False,
            "age_seconds": None
        }
    
    # Extract retrieval time
    retrieved_at_str = metadata.get("retrieved_at")
    retrieved_at = None
    if retrieved_at_str:
        try:
            retrieved_at = datetime.fromisoformat(retrieved_at_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            retrieved_at = None
    
    # Extract latest observation year
    latest_year = metadata.get("latest_available_observation")
    
    # Calculate staleness
    is_fresh = True
    age_seconds = 0
    
    if retrieved_at:
        now = datetime.now(timezone.utc)
        if retrieved_at.tzinfo is None:
            retrieved_at = retrieved_at.replace(tzinfo=timezone.utc)
        
        age = (now - retrieved_at).total_seconds()
        age_seconds = int(age)
        
        # Consider data stale if retrieved more than 6 hours ago (matching cache TTL)
        is_fresh = age < 21600  # 6 hours = 6 * 60 * 60
    
    return {
        "retrieved_at": retrieved_at_str,
        "retrieved_at_formatted": retrieved_at.strftime("%d %b %Y · %H:%M UTC") if retrieved_at else None,
        "latest_observation_year": latest_year,
        "is_fresh": is_fresh,
        "age_seconds": age_seconds,
        "cache_ttl_hours": 6,
        "sources": metadata.get("sources", [])
    }


def validate_state_transition(
    current_state: DataState,
    proposed_state: DataState,
    has_data: bool = False
) -> Tuple[bool, str]:
    """
    Check if a state transition is valid.
    
    Args:
        current_state: Current state
        proposed_state: State being proposed
        has_data: Whether data is available for the proposed state
        
    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    # Always allow loading state
    if proposed_state == DataState.LOADING:
        return True, ""
    
    # From unavailable, can go to loading or demo
    if current_state == DataState.UNAVAILABLE:
        if proposed_state in {DataState.LOADING, DataState.DEMO}:
            return True, ""
        return False, f"From UNAVAILABLE, must go through LOADING first (got {proposed_state.value})"
    
    # From demo, can go back to loading (for retry) or stay in demo
    if current_state == DataState.DEMO:
        if proposed_state in {DataState.LOADING, DataState.DEMO}:
            return True, ""
        # Going directly from DEMO to LIVE requires data
        if proposed_state == DataState.LIVE:
            if has_data:
                return True, ""
            return False, "Cannot transition from DEMO to LIVE without valid data"
    
    # From live/stale, can go to unavailable (data failure) or loading (refresh)
    if current_state in {DataState.LIVE, DataState.STALE}:
        if proposed_state in {DataState.LOADING, DataState.UNAVAILABLE}:
            return True, ""
    
    # From loading, can go to any final state
    if current_state == DataState.LOADING:
        if proposed_state in {DataState.LIVE, DataState.STALE, DataState.UNAVAILABLE, DataState.DEMO}:
            return True, ""
    
    # Default: allow any transition that isn't nonsensical
    if proposed_state == current_state:
        return True, "No state change requested"
    
    return True, f"Transition allowed from {current_state.value} to {proposed_state.value}"


def create_default_session_state() -> Dict[str, Any]:
    """
    Create default Streamlit session state structure.
    
    Returns:
        Dict with default session state values
    """
    return {
        "data_state": DataState.UNAVAILABLE,
        "live_panel": None,
        "demo_panel_path": None,
        "selected_country": None,
        "selected_year": None,
        "selected_peer_group": "Global",
        "refresh_triggered": False,
        "last_refresh_time": None,
        "current_provenance": None,
        "show_technical_details": False
    }


def initialize_streamlit_session_state(st) -> None:
    """
    Initialize Streamlit session state with default values.
    
    Args:
        st: Streamlit module
    """
    defaults = create_default_session_state()
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize data state machine
    if "data_state_machine" not in st.session_state:
        st.session_state.data_state_machine = DataStateMachine()
    
    # Set default country to India if configured, else first country
    from src.runtime.live_data import _load_countries_config
    countries_config = _load_countries_config()
    
    country_codes = [str(c.get("iso3", "")).upper() for c in countries_config]
    default_country = "IND" if "IND" in country_codes else (country_codes[0] if country_codes else "USA")
    
    if st.session_state.selected_country is None:
        st.session_state.selected_country = default_country
    
    if st.session_state.selected_year is None:
        # Default to latest common year if available, else current year
        current_year = pd.Timestamp.now().year
        st.session_state.selected_year = current_year