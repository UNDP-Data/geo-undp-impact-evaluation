#!/usr/bin/env python3
"""
config/schema.py
================
Configuration schema and validation for impact evaluation pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import yaml


@dataclass
class BoundaryConfig:
    """Administrative boundary configuration."""
    file_path: str
    admin_level: int  # 1=State, 2=District, 3=Township, etc.
    id_column: str
    name_column: str
    parent_name_column: Optional[str] = None  # For state/region name


@dataclass
class EventConfig:
    """Disaster/intervention event configuration."""
    disaster_date: str  # ISO format YYYY-MM-DD
    disaster_type: str  # earthquake, flood, conflict, etc.
    intervention_start: str  # When UNDP/intervention began
    epicenter: Optional[Tuple[float, float]] = None  # (lat, lon) for spatial analysis
    exclude_months: List[str] = field(default_factory=list)  # Months to exclude (disaster month)


@dataclass
class DataSourceConfig:
    """Configuration for each data source."""
    enabled: bool = True
    custom_params: Dict = field(default_factory=dict)


@dataclass
class DataConfig:
    """Data sources configuration."""
    # Satellite Data
    viirs_nightlights: DataSourceConfig = field(default_factory=lambda: DataSourceConfig())
    modis_vegetation: DataSourceConfig = field(default_factory=lambda: DataSourceConfig())
    sentinel1_sar: DataSourceConfig = field(default_factory=lambda: DataSourceConfig())
    chirps_rainfall: DataSourceConfig = field(default_factory=lambda: DataSourceConfig())
    
    # External Data
    acled_conflict: DataSourceConfig = field(default_factory=lambda: DataSourceConfig())
    wfp_prices: DataSourceConfig = field(default_factory=lambda: DataSourceConfig())
    google_trends: DataSourceConfig = field(default_factory=lambda: DataSourceConfig())
    
    # Local rasters (if GEE not available)
    local_viirs_dir: Optional[str] = None


@dataclass
class AnalysisConfig:
    """Analysis parameters."""
    spatial_radii: List[float] = field(default_factory=lambda: [0.25, 0.50])  # degrees
    donut_range: Tuple[float, float] = (0.15, 0.50)
    outcome_variables: List[str] = field(default_factory=lambda: [
        'nightlights', 'log_nightlights', 'ndvi', 'rvi', 'ndbi'
    ])
    control_variables: List[str] = field(default_factory=lambda: ['rainfall'])


@dataclass
class CountryConfig:
    """Complete country configuration."""
    country_code: str  # ISO 3166-1 alpha-3 (MMR, PSE, etc.)
    country_name: str
    study_period_start: str
    study_period_end: str
    boundaries: BoundaryConfig
    events: EventConfig
    data_sources: DataConfig = field(default_factory=DataConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    # Treatment configuration
    treatment_file: Optional[str] = None  # Path to UNDP project data
    treatment_id_column: str = 'township'
    
    # ACLED specific
    acled_country_name: Optional[str] = None  # Name as it appears in ACLED
    
    # WFP specific  
    wfp_hdx_dataset: Optional[str] = None  # HDX dataset identifier
    wfp_commodity: str = 'Rice'  # Primary commodity to track
    
    # Google Trends
    gt_geo_code: Optional[str] = None  # e.g., 'MM', 'PS'
    gt_keywords: Dict[str, str] = field(default_factory=dict)
    
    # --- Visualization & display options (used by visualizations.py) ---
    disaster_label: Optional[str] = None  # e.g., "Earthquake", "Conflict"
    
    # Epicenter for map markers (earthquakes only)
    epicenter_config: Optional[Dict[str, Any]] = None  # {lat, lon, magnitude}
    
    # Raster map date overrides
    raster_maps: Optional[Dict[str, str]] = None  # {pre_month, post_month, recent_month}


# ─── Known extra keys that come from YAML but aren't dataclass fields ─────────
# These are consumed by other parts of the pipeline (visualizations, undp_projects)
# and should be stripped before constructing CountryConfig
_EXTRA_KEYS = {'epicenter', 'raster_maps', 'disaster_label'}


def load_config(config_path: str) -> CountryConfig:
    """Load and validate configuration from YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Parse nested configs
    boundaries = BoundaryConfig(**data.pop('boundaries'))
    events = EventConfig(**data.pop('events'))
    
    data_sources_raw = data.pop('data_sources', {})
    data_sources = DataConfig(
        viirs_nightlights=DataSourceConfig(**data_sources_raw.get('viirs_nightlights', {})),
        modis_vegetation=DataSourceConfig(**data_sources_raw.get('modis_vegetation', {})),
        sentinel1_sar=DataSourceConfig(**data_sources_raw.get('sentinel1_sar', {})),
        chirps_rainfall=DataSourceConfig(**data_sources_raw.get('chirps_rainfall', {})),
        acled_conflict=DataSourceConfig(**data_sources_raw.get('acled_conflict', {})),
        wfp_prices=DataSourceConfig(**data_sources_raw.get('wfp_prices', {})),
        google_trends=DataSourceConfig(**data_sources_raw.get('google_trends', {})),
        local_viirs_dir=data_sources_raw.get('local_viirs_dir')
    )
    
    analysis_raw = data.pop('analysis', {})
    analysis = AnalysisConfig(**analysis_raw) if analysis_raw else AnalysisConfig()
    
    # Extract visualization-specific fields before passing to CountryConfig
    # These are stored as attributes on the config object for visualizations.py
    epicenter_data = data.pop('epicenter', None)
    raster_maps_data = data.pop('raster_maps', None)
    disaster_label = data.pop('disaster_label', None)
    
    data.pop("google_trends_keywords", None)
    config = CountryConfig(
        boundaries=boundaries,
        events=events,
        data_sources=data_sources,
        analysis=analysis,
        disaster_label=disaster_label,
        epicenter_config=epicenter_data,
        raster_maps=raster_maps_data,
        **data
    )
    
    return config


def load_raw_config(config_path: str) -> dict:
    """Load raw YAML config without validation (for scripts that need all fields)."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_config(config: CountryConfig) -> List[str]:
    """Validate configuration and return list of warnings/errors."""
    issues = []
    
    # Check boundary file exists
    if not Path(config.boundaries.file_path).exists():
        issues.append(f"WARNING: Boundary file not found: {config.boundaries.file_path}")
    
    # Check treatment file if specified
    if config.treatment_file and not Path(config.treatment_file).exists():
        issues.append(f"WARNING: Treatment file not found: {config.treatment_file}")
    
    # Check ACLED config
    if config.data_sources.acled_conflict.enabled and not config.acled_country_name:
        issues.append("WARNING: ACLED enabled but acled_country_name not set")
    
    # Check WFP config
    if config.data_sources.wfp_prices.enabled and not config.wfp_hdx_dataset:
        issues.append("WARNING: WFP prices enabled but wfp_hdx_dataset not set")
    
    return issues
