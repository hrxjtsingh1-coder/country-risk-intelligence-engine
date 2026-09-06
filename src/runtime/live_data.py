"""
Live data collector for Country Risk Intelligence Engine.
Fetches data directly from World Bank API and FRED at runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import yaml
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "indicators.yaml"
COUNTRIES_PATH = ROOT / "config" / "countries.yaml"

LOG = logging.getLogger("country-risk.live")

# World Bank API configuration
WB_BASE_URL = "https://api.worldbank.org/v2"
WB_INDICATOR_ENDPOINT = f"{WB_BASE_URL}/country/{{country}}/indicator/{{indicator}}"

# FRED configuration  
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={{series}}"

# Cache configuration
DEFAULT_CACHE_TTL = 21600  # 6 hours in seconds
MAX_RETRIES = 3
BACKOFF_FACTOR = 0.5
REQUEST_TIMEOUT = 30


def _create_session() -> requests.Session:
    """Create a requests session with retry strategy."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "country-risk-intelligence-engine/1.0 (+https://github.com/hrxjtsingh1-coder/country-risk-intelligence-engine)",
        "Accept": "application/json,text/csv;q=0.9,*/*;q=0.8",
    })
    
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",)
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    return session


def _load_indicators_config() -> Dict[str, Any]:
    """Load indicator configuration from YAML."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    
    indicators = config.get("indicators", [])
    return {
        str(ind["code"]): ind
        for ind in indicators
        if isinstance(ind, dict) and ind.get("code")
    }


def _load_countries_config() -> List[Dict[str, str]]:
    """Load countries configuration from YAML."""
    with open(COUNTRIES_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    
    return config.get("countries", [])


def _world_bank_indicator(
    session: requests.Session,
    iso3: str,
    wb_code: str,
    start_year: int,
    end_year: int
) -> pd.DataFrame:
    """
    Fetch a single indicator from World Bank API for a specific country.
    
    Args:
        session: Requests session
        iso3: Country ISO3 code
        wb_code: World Bank indicator code
        start_year: Start year for data
        end_year: End year for data
        
    Returns:
        DataFrame with columns: country_iso3, year, value, source
    """
    url = WB_INDICATOR_ENDPOINT.format(country=iso3, indicator=wb_code)
    params = {
        "format": "json",
        "per_page": 10000,  # Large enough to get all years
        "date": f"{start_year}:{end_year}",
    }
    
    try:
        response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        
        if not isinstance(payload, list) or len(payload) < 2:
            LOG.warning("Unexpected World Bank response format for %s/%s", iso3, wb_code)
            return pd.DataFrame()
        
        metadata, records = payload[0], payload[1]
        if not records:
            LOG.debug("No records returned for %s/%s", iso3, wb_code)
            return pd.DataFrame()
        
        rows = []
        for record in records:
            try:
                year = int(record.get("date")) if record.get("date") and record.get("date") != "" else None
                value = record.get("value")
                
                if year is None:
                    continue
                    
                # Convert None values to NaN for consistency
                if value is None:
                    value = float("nan")
                else:
                    value = float(value)
                
                rows.append({
                    "country_iso3": iso3.upper(),
                    "year": year,
                    "value": value,
                    "source": "World Bank"
                })
            except (ValueError, TypeError) as e:
                LOG.debug("Skipping invalid record for %s/%s: %s", iso3, wb_code, e)
                continue
        
        return pd.DataFrame(rows)
        
    except requests.RequestException as e:
        LOG.error("Failed to fetch World Bank data for %s/%s: %s", iso3, wb_code, str(e))
        return pd.DataFrame()
    except Exception as e:
        LOG.error("Unexpected error fetching World Bank data for %s/%s: %s", iso3, wb_code, str(e))
        return pd.DataFrame()


def _fred_annual_mean(
    session: requests.Session,
    series_id: str,
    start_year: int,
    end_year: int
) -> pd.DataFrame:
    """
    Fetch annual mean data from FRED for US-only indicators.
    
    Args:
        session: Requests session
        series_id: FRED series ID
        start_year: Start year for data
        end_year: End year for data
        
    Returns:
        DataFrame with columns: year, value, source
    """
    url = FRED_CSV_URL.format(series=series_id)
    
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        # Parse CSV data
        df = pd.read_csv(pd.io.common.StringIO(response.text))
        
        if df.empty:
            LOG.warning("Empty FRED response for series %s", series_id)
            return pd.DataFrame()
        
        # Find date and value columns
        date_col = None
        value_col = None
        
        for col in df.columns:
            if "date" in col.lower() or "observation" in col.lower():
                date_col = col
            elif col.upper() == series_id.upper():
                value_col = col
        
        if date_col is None or value_col is None:
            # Try to infer from first two columns
            if len(df.columns) >= 2:
                date_col, value_col = df.columns[0], df.columns[1]
            else:
                LOG.error("Could not identify date/value columns in FRED response for %s", series_id)
                return pd.DataFrame()
        
        # Clean and convert data
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[date_col, value_col])
        
        if df.empty:
            LOG.warning("No valid data after cleaning for FRED series %s", series_id)
            return pd.DataFrame()
        
        # Extract year and aggregate to annual mean
        df["year"] = df[date_col].dt.year
        df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
        
        if df.empty:
            LOG.warning("No data in requested year range for FRED series %s", series_id)
            return pd.DataFrame()
        
        annual_df = df.groupby("year", as_index=False)[value_col].mean()
        annual_df = annual_df.rename(columns={value_col: "value"})
        annual_df["source"] = "FRED"
        
        return annual_df[["year", "value", "source"]]
        
    except requests.RequestException as e:
        LOG.error("Failed to fetch FRED data for series %s: %s", series_id, str(e))
        return pd.DataFrame()
    except Exception as e:
        LOG.error("Unexpected error fetching FRED data for series %s: %s", series_id, str(e))
        return pd.DataFrame()


def _build_fx_depreciation(
    session: requests.Session,
    iso3: str,
    start_year: int,
    end_year: int
) -> pd.DataFrame:
    """
    Calculate FX depreciation from World Bank official exchange rate.
    
    Args:
        session: Requests session
        iso3: Country ISO3 code
        start_year: Start year for data
        end_year: End year for data
        
    Returns:
        DataFrame with FX YoY depreciation data
    """
    # Get exchange rate data (need extra year for YoY calculation)
    raw = _world_bank_indicator(session, iso3, "PA.NUS.FCRF", start_year - 1, end_year)
    
    if raw.empty:
        return pd.DataFrame()
    
    # Calculate year-over-year percent change
    raw = raw.sort_values("year")
    raw["fx_yoy"] = raw["value"].pct_change() * 100.0
    
    # Filter to requested years and format output
    result = raw[raw["year"].between(start_year, end_year)].copy()
    result["value"] = result["fx_yoy"]
    result["indicator_code"] = "FX_YOY_DEPRECIATION_PCT"
    result["source"] = "World Bank; derived from PA.NUS.FCRF"
    
    return result[["country_iso3", "indicator_code", "year", "value", "source"]]


def _policy_rate_for_country(
    session: requests.Session,
    iso3: str,
    start_year: int,
    end_year: int
) -> pd.DataFrame:
    """
    Get US policy rate from FRED (only for USA).
    
    Args:
        session: Requests session
        iso3: Country ISO3 code
        start_year: Start year for data
        end_year: End year for data
        
    Returns:
        DataFrame with policy rate change data (US-only)
    """
    if iso3.upper() != "USA":
        return pd.DataFrame()
    
    # Get FRED DFF data (need extra year for YoY change calculation)
    annual = _fred_annual_mean(session, "DFF", start_year - 1, end_year)
    
    if annual.empty:
        return pd.DataFrame()
    
    # Calculate year-over-year change in basis points
    annual = annual.sort_values("year")
    annual["change_bps"] = annual["value"].diff() * 100.0
    
    # Filter to requested years and format output
    result = annual[annual["year"].between(start_year, end_year)].copy()
    result["country_iso3"] = iso3.upper()
    result["indicator_code"] = "POLICY_RATE_YOY_CHANGE_BPS"
    result["source"] = "FRED DFF; annual mean change in basis points"
    result["value"] = result["change_bps"]
    
    return result[["country_iso3", "indicator_code", "year", "value", "source"]]


def fetch_live_data(
    countries: Optional[List[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    use_cache: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fetch live data from World Bank and FRED APIs.
    
    Args:
        countries: List of ISO3 country codes (None = all configured)
        start_year: Start year for data collection (None = use default)
        end_year: End year for data collection (None = current year)
        use_cache: Whether to use cached data if available
        
    Returns:
        Tuple of (long_panel_dataframe, metadata_dict)
    """
    session = _create_session()
    
    # Load configurations
    indicators_config = _load_indicators_config()
    countries_config = _load_countries_config()
    
    # Determine countries to process
    if countries is None:
        countries = [str(c["iso3"]) for c in countries_config]
    else:
        countries = [str(c).upper().strip() for c in countries]
    
    # Validate countries against configuration
    valid_countries = {str(c["iso3"]).upper() for c in countries_config}
    invalid_countries = set(countries) - valid_countries
    if invalid_countries:
        LOG.warning("Invalid country codes ignored: %s", ", ".join(sorted(invalid_countries)))
        countries = [c for c in countries if c in valid_countries]
    
    if not countries:
        raise ValueError("No valid countries specified for data collection")
    
    # Determine year range
    current_year = pd.Timestamp.now().year
    if end_year is None:
        end_year = current_year
    if start_year is None:
        start_year = 2012  # Default start year matching existing pipeline
    
    if start_year > end_year:
        raise ValueError(f"Start year ({start_year}) cannot be greater than end year ({end_year})")
    
    LOG.info(
        "Fetching live data for %d countries (%d-%d) from %d indicators",
        len(countries), start_year, end_year, len(indicators_config)
    )
    
    # Fetch data for each country-indicator combination
    all_records = []
    fetch_stats = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "start_time": time.time(),
        "countries_processed": [],
        "indicators_processed": [],
        "errors": []
    }
    
    for iso3 in countries:
        fetch_stats["countries_processed"].append(iso3)
        country_records = []
        
        for indicator_code, indicator_meta in indicators_config.items():
            fetch_stats["indicators_processed"].append(indicator_code)
            fetch_stats["total_requests"] += 1
            
            try:
                source_type = indicator_meta.get("source", "world_bank")
                wb_code = indicator_meta.get("world_bank")
                
                frame = None
                
                if source_type == "world_bank" and wb_code:
                    frame = _world_bank_indicator(
                        session, iso3, str(wb_code), start_year, end_year
                    )
                    if not frame.empty:
                        frame["indicator_code"] = indicator_code
                
                elif indicator_code == "FX_YOY_DEPRECIATION_PCT":
                    frame = _build_fx_depreciation(session, iso3, start_year, end_year)
                
                elif indicator_code == "POLICY_RATE_YOY_CHANGE_BPS":
                    frame = _policy_rate_for_country(session, iso3, start_year, end_year)
                
                # Handle FRED-sourced indicators (US-only)
                elif source_type == "fred_us_only":
                    fred_series = indicator_meta.get("fred")
                    if fred_series and iso3.upper() == "USA":
                        fred_frame = _fred_annual_mean(session, str(fred_series), start_year, end_year)
                        if not fred_frame.empty:
                            # Calculate the specific transformation needed
                            if indicator_code == "POLICY_RATE_YOY_CHANGE_BPS":
                                fred_frame = fred_frame.copy()
                                fred_frame = fred_frame.sort_values("year")
                                fred_frame["value"] = fred_frame["value"].diff() * 100.0
                                fred_frame = fred_frame[
                                    fred_frame["year"].between(start_year, end_year)
                                ]
                            fred_frame["indicator_code"] = indicator_code
                            fred_frame["source"] = "FRED"
                            frame = fred_frame[["country_iso3", "indicator_code", "year", "value", "source"]]
                
                if frame is not None and not frame.empty:
                    country_records.append(frame)
                    fetch_stats["successful_requests"] += 1
                else:
                    fetch_stats["failed_requests"] += 1
                    
            except Exception as e:
                fetch_stats["failed_requests"] += 1
                error_msg = f"Error fetching {indicator_code} for {iso3}: {str(e)}"
                fetch_stats["errors"].append(error_msg)
                LOG.warning(error_msg)
        
        if country_records:
            all_records.extend(country_records)
    
    fetch_stats["end_time"] = time.time()
    fetch_stats["duration_seconds"] = fetch_stats["end_time"] - fetch_stats["start_time"]
    
    LOG.info(
        "Live data fetch completed: %d successful, %d failed requests in %.2f seconds",
        fetch_stats["successful_requests"],
        fetch_stats["failed_requests"],
        fetch_stats["duration_seconds"]
    )
    
    # Combine all records
    if not all_records:
        long_panel = pd.DataFrame(
            columns=["country_iso3", "indicator_code", "year", "value", "source"]
        )
    else:
        long_panel = pd.concat(all_records, ignore_index=True)
        
        # Add flag column for compatibility with existing cleaning pipeline
        long_panel["flag"] = "ok"
        
        # Ensure proper data types
        long_panel["country_iso3"] = long_panel["country_iso3"].astype(str).str.upper().str.strip()
        long_panel["indicator_code"] = long_panel["indicator_code"].astype(str).str.strip()
        long_panel["year"] = pd.to_numeric(long_panel["year"], errors="coerce").astype("Int64")
        long_panel["value"] = pd.to_numeric(long_panel["value"], errors="coerce")
    
    # Create metadata
    metadata = {
        "run_id": pd.Timestamp.now().strftime("live-%Y%m%dT%H%M%SZ"),
        "mode": "LIVE",
        "retrieved_at": pd.Timestamp.now().isoformat(),
        "requested_period": f"{start_year}–{end_year}",
        "latest_available_observation": (
            int(long_panel["year"].max()) 
            if not long_panel.empty and not long_panel["year"].isna().all() 
            else None
        ),
        "country_count": (
            int(long_panel["country_iso3"].nunique()) 
            if not long_panel.empty 
            else 0
        ),
        "indicator_count": (
            int(long_panel["indicator_code"].nunique()) 
            if not long_panel.empty 
            else 0
        ),
        "observations_received": int(long_panel.shape[0]) if not long_panel.empty else 0,
        "sources": list(long_panel["source"].unique()) if not long_panel.empty else [],
        "fetch_statistics": fetch_stats
    }
    
    return long_panel, metadata


def validate_panel_data(long_panel: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that the panel data is suitable for analysis.
    
    Args:
        long_panel: Long-format panel data
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if long_panel.empty:
        issues.append("No data received from live sources")
        return False, issues
    
    # Check required columns
    required_columns = ["country_iso3", "indicator_code", "year", "value"]
    missing_columns = [col for col in required_columns if col not in long_panel.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
        return False, issues
    
    # Check for excessive missing values
    total_possible = len(long_panel)
    missing_values = long_panel["value"].isna().sum()
    missing_ratio = missing_values / total_possible if total_possible > 0 else 1.0
    
    if missing_ratio > 0.9:  # More than 90% missing
        issues.append(f"Too many missing values: {missing_ratio:.1%} of observations are missing")
    
    # Check year validity
    if not long_panel["year"].isna().all():
        min_year = long_panel["year"].min()
        max_year = long_panel["year"].max()
        current_year = pd.Timestamp.now().year
        
        if min_year < 1960:
            issues.append(f"Unusually early year detected: {min_year}")
        if max_year > current_year + 2:
            issues.append(f"Unusually future year detected: {max_year}")
    
    # Check country coverage
    if not long_panel.empty:
        unique_countries = long_panel["country_iso3"].nunique()
        if unique_countries == 0:
            issues.append("No valid countries in data")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def get_latest_common_year(long_panel: pd.DataFrame, min_coverage: float = 0.8) -> Optional[int]:
    """
    Determine the latest year with sufficient data coverage across countries and indicators.
    
    Args:
        long_panel: Long-format panel data
        min_coverage: Minimum coverage ratio required (0.0 to 1.0)
        
    Returns:
        Latest year meeting coverage requirements, or None if none found
    """
    if long_panel.empty:
        return None
    
    # Load expected indicators and countries from config
    indicators_config = _load_indicators_config()
    countries_config = _load_countries_config()
    
    expected_indicators = set(indicators_config.keys())
    expected_countries = {str(c["iso3"]).upper() for c in countries_config}
    
    # Get actual data
    actual_countries = set(long_panel["country_iso3"].dropna().unique())
    actual_indicators = set(long_panel["indicator_code"].dropna().unique())
    actual_years = set(long_panel["year"].dropna().unique())
    
    if not actual_years:
        return None
    
    # Sort years descending to find latest valid year first
    sorted_years = sorted(actual_years, reverse=True)
    
    for year in sorted_years:
        year_data = long_panel[long_panel["year"] == year]
        
        if year_data.empty:
            continue
        
        # Calculate coverage for this year
        country_coverage = len(
            actual_countries & set(year_data["country_iso3"].unique())
        ) / len(expected_countries) if expected_countries else 0
        
        indicator_coverage = len(
            actual_indicators & set(year_data["indicator_code"].unique())
        ) / len(expected_indicators) if expected_indicators else 0
        
        # Overall coverage (could be weighted differently)
        overall_coverage = min(country_coverage, indicator_coverage)
        
        if overall_coverage >= min_coverage:
            return int(year)
    
    return None


def create_wide_panel_from_long(long_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long panel data to wide format compatible with existing analytics.
    
    Args:
        long_panel: Long-format panel data with columns: country_iso3, indicator_code, year, value
        
    Returns:
        Wide-format DataFrame with one row per country/year and indicators as columns
    """
    if long_panel.empty:
        return pd.DataFrame(columns=["country_iso3", "year"])
    
    # Clean the data first (remove duplicates, handle missing values)
    clean_panel = long_panel.dropna(subset=["country_iso3", "indicator_code", "year", "value"]).copy()
    
    if clean_panel.empty:
        return pd.DataFrame(columns=["country_iso3", "year"])
    
    # Convert year to int
    clean_panel["year"] = clean_panel["year"].astype(int)
    
    # Pivot to wide format
    wide_panel = (
        clean_panel
        .pivot_table(
            index=["country_iso3", "year"],
            columns="indicator_code",
            values="value",
            aggfunc="last"  # Take last value if duplicates exist
        )
        .reset_index()
    )
    
    wide_panel.columns.name = None  # Remove column name from pivot
    return wide_panel.sort_values(["country_iso3", "year"]).reset_index(drop=True)


if __name__ == "__main__":
    # Simple test when run directly
    logging.basicConfig(level=logging.INFO)
    
    try:
        print("Testing live data fetch...")
        panel, metadata = fetch_live_data(
            countries=["USA", "IND", "DEU"],
            start_year=2020,
            end_year=2025
        )
        
        print(f"Fetched {len(panel)} records")
        print(f"Countries: {metadata['country_count']}")
        print(f"Indicators: {metadata['indicator_count']}")
        print(f"Latest year: {metadata['latest_available_observation']}")
        
        if not panel.empty:
            print("\nSample data:")
            print(panel.head(10))
            
            # Test wide panel conversion
            wide_panel = create_wide_panel_from_long(panel)
            print(f"\nWide panel shape: {wide_panel.shape}")
            print(f"Wide panel columns: {list(wide_panel.columns)}")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()