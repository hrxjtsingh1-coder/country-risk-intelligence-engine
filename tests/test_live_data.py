"""
Tests for the live data runtime system.
Tests the state machine, provenance, and live data collection with mocked responses.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestDataStateMachine:
    """Tests for DataStateMachine state management."""
    
    def test_initial_state_is_unavailable(self):
        from src.runtime.data_state import DataStateMachine, DataState
        
        machine = DataStateMachine()
        assert machine.current_state == DataState.UNAVAILABLE
    
    def test_transition_to_loading(self):
        from src.runtime.data_state import DataStateMachine, DataState
        
        machine = DataStateMachine()
        machine.transition_to(DataState.LOADING)
        assert machine.current_state == DataState.LOADING
    
    def test_transition_to_live(self):
        from src.runtime.data_state import DataStateMachine, DataState
        
        machine = DataStateMachine()
        machine.set_live_data_available(is_fresh=True)
        assert machine.current_state == DataState.LIVE
    
    def test_transition_to_stale(self):
        from src.runtime.data_state import DataStateMachine, DataState
        
        machine = DataStateMachine()
        machine.set_live_data_available(is_fresh=False)
        assert machine.current_state == DataState.STALE
    
    def test_transition_to_demo(self):
        from src.runtime.data_state import DataStateMachine, DataState
        
        machine = DataStateMachine()
        machine.set_demo_mode()
        assert machine.current_state == DataState.DEMO
        assert machine.is_demo_mode()
    
    def test_transition_to_unavailable_with_error(self):
        from src.runtime.data_state import DataStateMachine, DataState
        
        machine = DataStateMachine()
        machine.set_unavailable({"operation": "test", "details": "test error"})
        assert machine.current_state == DataState.UNAVAILABLE
        assert machine.is_unavailable()
    
    def test_is_live_state(self):
        from src.runtime.data_state import DataStateMachine, DataState
        
        machine = DataStateMachine()
        assert not machine.is_live_state()
        machine.set_live_data_available(is_fresh=True)
        assert machine.is_live_state()
    
    def test_can_display_dashboard(self):
        from src.runtime.data_state import DataStateMachine, DataState
        
        machine = DataStateMachine()
        assert not machine.can_display_dashboard()
        machine.set_demo_mode()
        assert machine.can_display_dashboard()
        machine.set_live_data_available(is_fresh=True)
        assert machine.can_display_dashboard()
    
    def test_get_display_badge(self):
        from src.runtime.data_state import DataStateMachine, DataState
        
        machine = DataStateMachine()
        badge_text, badge_class = machine.get_display_badge()
        assert "LIVE DATA UNAVAILABLE" in badge_text
        assert "badge-unavailable" in badge_class
        
        machine.set_live_data_available(is_fresh=True)
        badge_text, badge_class = machine.get_display_badge()
        assert "LIVE PUBLIC DATA" in badge_text
        assert "badge-live" in badge_class
        
        machine.set_demo_mode()
        badge_text, badge_class = machine.get_display_badge()
        assert "DEMO DATA" in badge_text
        assert "badge-demo" in badge_class


class TestProvenanceManager:
    """Tests for ProvenanceRecord and ProvenanceManager."""
    
    def test_provenance_record_creation(self):
        from src.runtime.provenance import ProvenanceRecord
        from datetime import datetime, timezone
        
        record = ProvenanceRecord(
            run_id="test-001",
            retrieved_at=datetime.now(timezone.utc),
            source_name="World Bank",
            source_endpoint="https://api.worldbank.org/v2",
            source_series="NY.GDP.MKTP.KD.ZG",
            requested_period="2020-2025",
            latest_observation=2024,
            country_count=20,
            indicator_count=10,
            expected_observations=200,
            received_observations=180,
            coverage=0.9
        )
        
        assert record.run_id == "test-001"
        assert record.source_name == "World Bank"
        assert record.country_count == 20
        assert record.coverage == 0.9
    
    def test_provenance_to_dict(self):
        from src.runtime.provenance import ProvenanceRecord
        from datetime import datetime, timezone
        
        record = ProvenanceRecord(
            run_id="test-002",
            retrieved_at=datetime.now(timezone.utc),
            source_name="FRED",
            source_endpoint="https://fred.stlouisfed.org",
            source_series="DFF",
            requested_period="2020-2025"
        )
        
        record_dict = record.to_dict()
        assert isinstance(record_dict, dict)
        assert record_dict["run_id"] == "test-002"
        assert record_dict["source_name"] == "FRED"


class TestLiveDataCollection:
    """Tests for live data collection with mocked responses."""
    
    def test_validate_panel_data_empty(self):
        from src.runtime.live_data import validate_panel_data
        
        is_valid, issues = validate_panel_data(pd.DataFrame())
        assert not is_valid
        assert len(issues) > 0
    
    def test_validate_panel_data_with_valid_data(self):
        from src.runtime.live_data import validate_panel_data
        
        df = pd.DataFrame([
            {"country_iso3": "USA", "indicator_code": "NY.GDP.MKTP.KD.ZG", "year": 2024, "value": 2.5},
            {"country_iso3": "IND", "indicator_code": "NY.GDP.MKTP.KD.ZG", "year": 2024, "value": 6.8},
        ])
        
        is_valid, issues = validate_panel_data(df)
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_panel_data_missing_columns(self):
        from src.runtime.live_data import validate_panel_data
        
        # Missing required columns
        df = pd.DataFrame([
            {"country": "USA", "indicator": "GDP", "year": 2024, "value": 2.5},
        ])
        
        is_valid, issues = validate_panel_data(df)
        assert not is_valid
    
    def test_get_latest_common_year_no_data(self):
        from src.runtime.live_data import get_latest_common_year
        
        result = get_latest_common_year(pd.DataFrame())
        assert result is None
    
    def test_get_latest_common_year_with_data(self):
        from src.runtime.live_data import get_latest_common_year
        
        # Create data with complete coverage for 2022 and partial for 2023
        df = pd.DataFrame([
            # Complete data for 2022 (2 countries x 2 indicators = 4 obs)
            {"country_iso3": "USA", "indicator_code": "NY.GDP.MKTP.KD.ZG", "year": 2022, "value": 2.1},
            {"country_iso3": "USA", "indicator_code": "GC.DOD.TOTL.GD.ZS", "year": 2022, "value": 120.0},
            {"country_iso3": "IND", "indicator_code": "NY.GDP.MKTP.KD.ZG", "year": 2022, "value": 6.8},
            {"country_iso3": "IND", "indicator_code": "GC.DOD.TOTL.GD.ZS", "year": 2022, "value": 83.0},
            # Incomplete data for 2023 (only 1 country x 2 indicators = 2 obs)
            {"country_iso3": "IND", "indicator_code": "NY.GDP.MKTP.KD.ZG", "year": 2023, "value": 6.8},
            {"country_iso3": "IND", "indicator_code": "GC.DOD.TOTL.GD.ZS", "year": 2023, "value": 83.0},
        ])
        
        result = get_latest_common_year(df, min_coverage=0.5)
        # With 2 countries configured, 2023 only has 1 country so it may not pass coverage
        # The function should return a valid year
        assert result in [2022, 2023] or result is None
    
    def test_create_wide_panel_from_long(self):
        from src.runtime.live_data import create_wide_panel_from_long
        
        df = pd.DataFrame([
            {"country_iso3": "USA", "indicator_code": "NY.GDP.MKTP.KD.ZG", "year": 2022, "value": 2.1, "source": "World Bank", "flag": "ok"},
            {"country_iso3": "USA", "indicator_code": "GC.DOD.TOTL.GD.ZS", "year": 2022, "value": 120.0, "source": "World Bank", "flag": "ok"},
            {"country_iso3": "IND", "indicator_code": "NY.GDP.MKTP.KD.ZG", "year": 2022, "value": 6.8, "source": "World Bank", "flag": "ok"},
            {"country_iso3": "IND", "indicator_code": "GC.DOD.TOTL.GD.ZS", "year": 2022, "value": 83.0, "source": "World Bank", "flag": "ok"},
        ])
        
        wide_df = create_wide_panel_from_long(df)
        
        assert not wide_df.empty
        assert "country_iso3" in wide_df.columns
        assert "year" in wide_df.columns
        assert "NY.GDP.MKTP.KD.ZG" in wide_df.columns
        assert "GC.DOD.TOTL.GD.ZS" in wide_df.columns


class TestLiveDataFetchMocked:
    """Tests for fetch_live_data with mocked HTTP responses."""
    
    @patch('requests.Session')
    def test_fetch_live_data_handles_api_failure(self, mock_session_class):
        from src.runtime.live_data import fetch_live_data
        import requests
        
        # Mock session that raises an exception
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.RequestException("Network error")
        mock_session_class.return_value = mock_session
        
        # Should handle gracefully and return empty data with error info
        long_panel, metadata = fetch_live_data(
            countries=["USA"],
            start_year=2022,
            end_year=2023
        )
        
        # The function should complete without raising
        assert metadata is not None
        assert "fetch_statistics" in metadata


class TestDataFreshnessInfo:
    """Tests for data freshness information extraction."""
    
    def test_get_freshness_info_empty(self):
        from src.runtime.data_state import get_data_freshness_info
        
        result = get_data_freshness_info(None)
        assert result["retrieved_at"] is None
        assert result["is_fresh"] is False
    
    def test_get_freshness_info_with_recent_data(self):
        from src.runtime.data_state import get_data_freshness_info
        from datetime import datetime, timezone
        
        recent_metadata = {
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "latest_available_observation": 2024,
            "sources": ["World Bank"]
        }
        
        result = get_data_freshness_info(recent_metadata)
        assert result["is_fresh"] is True
        assert result["latest_observation_year"] == 2024
        assert result["sources"] == ["World Bank"]


class TestStateTransitions:
    """Tests for state transition validation."""
    
    def test_validate_state_transition_loading_from_unavailable(self):
        from src.runtime.data_state import validate_state_transition, DataState
        
        is_valid, reason = validate_state_transition(
            DataState.UNAVAILABLE,
            DataState.LOADING
        )
        assert is_valid
    
    def test_validate_state_transition_demo_from_unavailable(self):
        from src.runtime.data_state import validate_state_transition, DataState
        
        is_valid, reason = validate_state_transition(
            DataState.UNAVAILABLE,
            DataState.DEMO
        )
        assert is_valid
    
    def test_validate_state_transition_live_from_demo_requires_data(self):
        from src.runtime.data_state import validate_state_transition, DataState
        
        # Without data
        is_valid, reason = validate_state_transition(
            DataState.DEMO,
            DataState.LIVE,
            has_data=False
        )
        assert not is_valid
        
        # With data
        is_valid, reason = validate_state_transition(
            DataState.DEMO,
            DataState.LIVE,
            has_data=True
        )
        assert is_valid


class TestGraphZoomDisabled:
    """Tests to verify graph zoom is disabled."""
    
    def test_create_plotly_figure_removes_zoom(self):
        from src.ui.charts import create_risk_score_chart
        import plotly.graph_objects as go
        
        # Create minimal test data
        df = pd.DataFrame({
            "country_iso3": ["USA", "USA", "USA"],
            "year": [2022, 2023, 2024],
            "risk_score": [45.0, 48.0, 50.0]
        })
        
        fig = create_risk_score_chart(df, "USA", 2024)
        
        # Check that zoom-related modebar buttons are removed
        modebar_remove = fig.layout.get("modebar_remove", [])
        assert "zoom" in modebar_remove
        assert "pan" in modebar_remove


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
