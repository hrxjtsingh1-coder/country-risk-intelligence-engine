"""
Provenance tracking for Country Risk Intelligence Engine.
Tracks data source, collection time, and verification details for all live data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import pandas as pd


@dataclass
class ProvenanceRecord:
    """Record of data provenance for a live data fetch."""
    run_id: str
    retrieved_at: datetime
    source_name: str
    source_endpoint: str
    source_series: str
    requested_period: str
    latest_observation: Optional[int] = None
    country_count: int = 0
    indicator_count: int = 0
    expected_observations: int = 0
    received_observations: int = 0
    missing_observations: int = 0
    coverage: float = 0.0
    validation_failures: int = 0
    config_model_version: str = "1.1.0"
    model_version_hash: str = "unknown"
    retrieval_duration_s: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class ProvenanceManager:
    """Manages provenance records for the application."""
    
    def __init__(self):
        self.records: List[ProvenanceRecord] = []
        self.current_run_id: Optional[str] = None
    
    def start_new_run(self, run_id: str, source_name: str, source_endpoint: str, 
                      source_series: str, requested_period: str) -> None:
        """Start a new data collection run."""
        self.current_run_id = run_id
        self.records.append(ProvenanceRecord(
            run_id=run_id,
            retrieved_at=datetime.now(timezone.utc),
            source_name=source_name,
            source_endpoint=source_endpoint,
            source_series=source_series,
            requested_period=requested_period,
            country_count=0,
            indicator_count=0,
            expected_observations=0,
            received_observations=0,
            missing_observations=0,
            coverage=0.0,
            validation_failures=0,
            config_model_version="1.1.0",
            model_version_hash="unknown",
            retrieval_duration_s=0.0
        ))
    
    def update_current_run(self, **kwargs) -> None:
        """Update the current run with additional details."""
        if self.current_run_id is None:
            return
            
        # Find and update the current record
        for record in self.records:
            if record.run_id == self.current_run_id:
                for key, value in kwargs.items():
                    if hasattr(record, key):
                        setattr(record, key, value)
                break
    
    def get_current_provenance(self) -> Optional[ProvenanceRecord]:
        """Get the provenance record for the current run."""
        if self.current_run_id is None:
            return None
        for record in self.records:
            if record.run_id == self.current_run_id:
                return record
        return None
    
    def get_latest_provenance(self) -> Optional[ProvenanceRecord]:
        """Get the most recent provenance record."""
        if not self.records:
            return None
        return self.records[-1]
    
    def add_provenance_record(self, record: ProvenanceRecord) -> None:
        """Add a new provenance record."""
        self.records.append(record)
    
    def get_provenance_summary(self) -> Dict[str, Any]:
        """Get summary information about current provenance."""
        current = self.get_current_provenance()
        if current is None:
            return {}
            
        return {
            "run_id": current.run_id,
            "retrieved_at": current.retrieved_at.isoformat(),
            "source_name": current.source_name,
            "country_count": current.country_count,
            "indicator_count": current.indicator_count,
            "coverage": f"{current.coverage:.1%}",
            "status": "LIVE" if current.retrieved_at else "UNAVAILABLE",
            "latest_observation_year": current.latest_observation,
        }