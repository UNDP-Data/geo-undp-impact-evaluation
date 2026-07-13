#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNDP Myanmar Disaster Impact Assessment - Nightlights Analysis
================================================================

This script analyzes the impact of disasters (earthquake/cyclone) in Myanmar 
using nighttime lights data and UNDP assessment locations.

Features:
- Downloads VIIRS nightlight data (with or without Google Earth Engine)
- Creates before/after comparison maps
- Builds treatment/control framework for quasi-experimental analysis
- Generates difference-in-differences (DiD) ready panel data

Requirements:
    pip install pandas numpy geopandas matplotlib openpyxl shapely rasterio rioxarray

Optional (for GEE download):
    pip install earthengine-api geemap

Usage WITHOUT Google Earth Engine (recommended for first run):
    python undp_myanmar_nightlights_analysis.py \\
        --undp-data Myanmar_Early_Recovery_Needs_Assessment_-_Dataset_cleaned_anonymous_Final.xlsx \\
        --event-date 2023-05-14 \\
        --event-name "Cyclone Mocha" \\
        --output-dir outputs

Author: Modified for UNDP Myanmar Analysis
Date: 2025
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Try importing rasterio components
try:
    import rioxarray as rxr
    import rasterio
    from rasterio.warp import transform_bounds
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False
    print("Warning: rioxarray/rasterio not installed. Install with:")
    print("  pip install rioxarray rasterio")

# Try importing Earth Engine (optional)
try:
    import ee
    import geemap
    EE_OK = True
except ImportError:
    EE_OK = False

# Configuration
MYANMAR_BBOX = (92.0, 9.0, 101.5, 28.5)  # [lon_min, lat_min, lon_max, lat_max]
CRS_WGS84 = "EPSG:4326"
CRS_UTM47N = "EPSG:32647"  # UTM zone 47N for Myanmar


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="UNDP Myanmar Nightlights Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--undp-data",
        type=str,
        required=True,
        help="Path to UNDP assessment Excel file"
    )
    
    # Event configuration
    parser.add_argument(
        "--event-date",
        type=str,
        default="2023-05-14",
        help="Event date (YYYY-MM-DD). Default: 2023-05-14 (Cyclone Mocha)"
    )
    
    parser.add_argument(
        "--event-name",
        type=str,
        default="Cyclone Mocha",
        help="Name of the disaster event for labeling"
    )
    
    # Temporal parameters
    parser.add_argument(
        "--months-before",
        type=int,
        default=6,
        help="Number of months before event to analyze (default: 6)"
    )
    
    parser.add_argument(
        "--months-after",
        type=int,
        default=12,
        help="Number of months after event to analyze (default: 12)"
    )
    
    # Spatial parameters
    parser.add_argument(
        "--grid-km",
        type=float,
        default=5.0,
        help="Grid cell size in kilometers (default: 5.0)"
    )
    
    parser.add_argument(
        "--treatment-radius-km",
        type=float,
        default=10.0,
        help="Radius around UNDP sites to define treatment (default: 10.0 km)"
    )
    
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        help="Custom bounding box [lon_min lat_min lon_max lat_max]"
    )
    
    # Data source options
    parser.add_argument(
        "--viirs-dir",
        type=str,
        default=None,
        help="Directory containing pre-downloaded VIIRS GeoTIFF files"
    )
    
    parser.add_argument(
        "--use-gee",
        action="store_true",
        help="Use Google Earth Engine to download VIIRS data"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for output maps (default: 200)"
    )
    
    return parser.parse_args()


def setup_directories(base_dir):
    """Create necessary output directories."""
    base_path = Path(base_dir)
    dirs = {
        'base': base_path,
        'viirs': base_path / "viirs_tifs",
        'maps': base_path / "maps",
        'data': base_path / "data",
        'figures': base_path / "figures"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def load_undp_data(excel_path):
    """
    Load and clean UNDP assessment data.
    
    Parameters:
        excel_path: Path to Excel file
        
    Returns:
        GeoDataFrame with UNDP assessment locations
    """
    print(f"\n[1/8] Loading UNDP assessment data from {excel_path}...")
    
    df = pd.read_excel(excel_path, sheet_name=0)
    
    lat_col = '_L6. GPS_latitude'
    lon_col = '_L6. GPS_longitude'
    
    # Filter to valid coordinates
    df_valid = df[(df[lat_col].notna()) & (df[lon_col].notna())].copy()
    
    print(f"  • Loaded {len(df)} total records")
    print(f"  • Found {len(df_valid)} records with valid GPS coordinates")
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df_valid,
        geometry=gpd.points_from_xy(df_valid[lon_col], df_valid[lat_col]),
        crs=CRS_WGS84
    )
    
    # Add useful columns
    gdf['site_id'] = range(len(gdf))
    gdf['region'] = gdf['L1. State/ Region']
    gdf['township'] = gdf['L3. Township']
    
    # Print summary
    print(f"\n  Geographic coverage:")
    region_counts = gdf['region'].value_counts()
    for region, count in region_counts.head(10).items():
        print(f"    - {region}: {count} sites")
    
    return gdf


def create_study_bbox(gdf, buffer_km=50):
    """
    Create bounding box around UNDP sites with buffer.
    
    Parameters:
        gdf: GeoDataFrame with UNDP sites
        buffer_km: Buffer in kilometers
        
    Returns:
        Bounding box [lon_min, lat_min, lon_max, lat_max]
    """
    # Reproject to UTM for buffering
    gdf_utm = gdf.to_crs(CRS_UTM47N)
    
    # Get bounds and add buffer
    bounds = gdf_utm.total_bounds
    buffer_m = buffer_km * 1000
    
    bbox_utm = [
        bounds[0] - buffer_m,  # xmin
        bounds[1] - buffer_m,  # ymin
        bounds[2] + buffer_m,  # xmax
        bounds[3] + buffer_m   # ymax
    ]
    
    # Convert back to WGS84
    bbox_wgs84 = transform_bounds(
        CRS_UTM47N, CRS_WGS84,
        bbox_utm[0], bbox_utm[1], bbox_utm[2], bbox_utm[3]
    )
    
    return bbox_wgs84


def generate_month_list(event_date, months_before, months_after):
    """
    Generate list of months around event date.
    
    Parameters:
        event_date: datetime object
        months_before: number of months before event
        months_after: number of months after event
        
    Returns:
        List of datetime objects (first day of each month)
    """
    months = []
    
    # Months before
    for i in range(months_before, 0, -1):
        month = event_date - pd.DateOffset(months=i)
        months.append(month.replace(day=1))
    
    # Event month
    months.append(event_date.replace(day=1))
    
    # Months after
    for i in range(1, months_after + 1):
        month = event_date + pd.DateOffset(months=i)
        months.append(month.replace(day=1))
    
    # Remove duplicates and sort
    months = sorted(list(set(months)))
    
    return months


def download_viirs_gee(month, bbox, output_path):
    """
    Download VIIRS nightlight data using Google Earth Engine.
    
    Parameters:
        month: datetime object
        bbox: bounding box [lon_min, lat_min, lon_max, lat_max]
        output_path: Path object for output GeoTIFF
    """
    if not EE_OK:
        raise RuntimeError("Earth Engine not available. Install with: pip install earthengine-api geemap")
    
    try:
        ee.Initialize()
    except Exception:
        print("  Authenticating with Google Earth Engine...")
        ee.Authenticate()
        ee.Initialize()
    
    # Get month end date
    month_end = month + pd.offsets.MonthEnd(0)
    
    # Query VIIRS collection
    ic = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG') \
            .filterDate(month.strftime('%Y-%m-01'), month_end.strftime('%Y-%m-%d'))
    
    img = ic.first().select('avg_rad')
    region = ee.Geometry.Rectangle(list(bbox))
    
    # Export
    geemap.ee_export_image(
        image=img,
        filename=str(output_path),
        scale=500,  # 500m resolution
        region=region,
        file_per_band=False,
        crs='EPSG:4326'
    )
    
    print(f"  ✓ Downloaded: {output_path.name}")


def find_local_viirs(month, viirs_dir):
    """
    Find locally stored VIIRS file for a given month.
    
    Parameters:
        month: datetime object
        viirs_dir: Path to directory with VIIRS files
        
    Returns:
        Path to VIIRS file or None
    """
    yyyymm = month.strftime("%Y%m")
    viirs_path = Path(viirs_dir)
    
    # Look for files with YYYYMM pattern
    for tif_file in viirs_path.glob(f"*{yyyymm}*.tif"):
        return tif_file
    
    return None


def get_viirs_data(months, bbox, viirs_dir, use_gee, output_dir):
    """
    Get VIIRS nightlight data for all months.
    
    Parameters:
        months: list of datetime objects
        bbox: bounding box
        viirs_dir: directory with existing VIIRS files (or None)
        use_gee: whether to use Google Earth Engine
        output_dir: directory to save downloaded files
        
    Returns:
        List of (month, file_path) tuples
    """
    print(f"\n[2/8] Acquiring VIIRS nightlight data...")
    print(f"  • Need data for {len(months)} months")
    print(f"  • Time span: {months[0].strftime('%Y-%m')} to {months[-1].strftime('%Y-%m')}")
    
    viirs_files = []
    
    if use_gee:
        print("  • Using Google Earth Engine to download data...")
        for month in months:
            output_file = output_dir / f"viirs_vcmslcfg_{month.strftime('%Y%m')}.tif"
            
            if output_file.exists():
                print(f"  ✓ Already exists: {output_file.name}")
                viirs_files.append((month, output_file))
            else:
                try:
                    download_viirs_gee(month, bbox, output_file)
                    viirs_files.append((month, output_file))
                except Exception as e:
                    print(f"  ✗ Failed to download {month.strftime('%Y-%m')}: {e}")
    
    elif viirs_dir:
        print(f"  • Looking for VIIRS files in: {viirs_dir}")
        for month in months:
            tif_file = find_local_viirs(month, viirs_dir)
            if tif_file:
                print(f"  ✓ Found: {tif_file.name}")
                viirs_files.append((month, tif_file))
            else:
                print(f"  ✗ Missing: {month.strftime('%Y%m')}")
    
    else:
        print("\n  ERROR: No VIIRS data source specified!")
        print("  Please either:")
        print("    1. Use --use-gee to download from Google Earth Engine")
        print("    2. Provide --viirs-dir with pre-downloaded VIIRS GeoTIFF files")
        print("\n  To download VIIRS manually:")
        print("    - Visit: https://eogdata.mines.edu/products/vnl/")
        print("    - Download VIIRS monthly composites (VNL V2)")
        print("    - File naming: VIIRS_YYYYMM*.tif")
        sys.exit(1)
    
    print(f"\n  ✓ Successfully acquired {len(viirs_files)}/{len(months)} files")
    
    return viirs_files


def create_analysis_grid(bbox, resolution_km=5.0):
    """
    Create regular grid for analysis.
    
    Parameters:
        bbox: bounding box [lon_min, lat_min, lon_max, lat_max]
        resolution_km: grid cell size in kilometers
        
    Returns:
        GeoDataFrame with grid cells
    """
    print(f"\n[3/8] Creating analysis grid...")
    print(f"  • Grid resolution: {resolution_km} km")
    
    lon_min, lat_min, lon_max, lat_max = bbox
    
    # Approximate degrees per km at Myanmar's latitude (~16-23°N)
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * np.cos(np.radians(20))  # Use mid-latitude
    
    deg_lat = resolution_km / km_per_deg_lat
    deg_lon = resolution_km / km_per_deg_lon
    
    # Create grid
    lons = np.arange(lon_min, lon_max, deg_lon)
    lats = np.arange(lat_min, lat_max, deg_lat)
    
    cells = []
    for lon in lons:
        for lat in lats:
            cell = box(lon, lat, lon + deg_lon, lat + deg_lat)
            cells.append(cell)
    
    grid = gpd.GeoDataFrame(geometry=cells, crs=CRS_WGS84)
    grid['cell_id'] = range(len(grid))
    
    # Add cell centroids for later use
    grid['centroid_lon'] = grid.geometry.centroid.x
    grid['centroid_lat'] = grid.geometry.centroid.y
    
    print(f"  ✓ Created grid with {len(grid)} cells")
    print(f"  • Coverage: {len(lons)} × {len(lats)} = {len(cells)} cells")
    
    return grid


def assign_treatment_status(undp_gdf, grid, radius_km=10.0):
    """
    Assign treatment status to grid cells based on proximity to UNDP sites.
    
    Parameters:
        undp_gdf: GeoDataFrame with UNDP sites
        grid: GeoDataFrame with analysis grid
        radius_km: radius for treatment definition
        
    Returns:
        DataFrame with cell_id and treatment status
    """
    print(f"\n[4/8] Assigning treatment status...")
    print(f"  • Treatment radius: {radius_km} km around UNDP sites")
    
    # Reproject to UTM for accurate distance calculation
    undp_utm = undp_gdf.to_crs(CRS_UTM47N)
    grid_utm = grid.to_crs(CRS_UTM47N)
    
    # Create buffer around UNDP sites
    buffer_m = radius_km * 1000
    undp_buffers = undp_utm.geometry.buffer(buffer_m)
    treatment_area = undp_buffers.unary_union
    
    # Check which grid cells intersect treatment area
    grid_utm['treated'] = grid_utm.geometry.intersects(treatment_area).astype(int)
    
    # Create treatment assignment table
    treatment_df = pd.DataFrame({
        'cell_id': grid['cell_id'],
        'treated': grid_utm['treated']
    })
    
    n_treated = treatment_df['treated'].sum()
    n_control = len(treatment_df) - n_treated
    pct_treated = 100 * n_treated / len(treatment_df)
    
    print(f"  ✓ Treatment assignment complete")
    print(f"  • Treated cells: {n_treated} ({pct_treated:.1f}%)")
    print(f"  • Control cells: {n_control} ({100-pct_treated:.1f}%)")
    
    return treatment_df


def extract_nightlight_values(viirs_files, grid, bbox):
    """
    Extract nightlight values for each grid cell and month.
    
    Parameters:
        viirs_files: list of (month, file_path) tuples
        grid: GeoDataFrame with analysis grid
        bbox: bounding box
        
    Returns:
        DataFrame with panel data (cell_id, month, nightlight_value)
    """
    print(f"\n[5/8] Extracting nightlight values...")
    
    if not RASTERIO_OK:
        print("  ERROR: rioxarray/rasterio not installed")
        print("  Install with: pip install rioxarray rasterio")
        sys.exit(1)
    
    all_data = []
    
    for i, (month, tif_path) in enumerate(viirs_files, 1):
        print(f"  [{i}/{len(viirs_files)}] Processing {month.strftime('%Y-%m')}...", end=' ')
        
        try:
            # Read raster
            raster = rxr.open_rasterio(tif_path, masked=True).squeeze()
            
            # Clip to bbox
            raster = raster.rio.clip_box(
                minx=bbox[0], miny=bbox[1],
                maxx=bbox[2], maxy=bbox[3]
            )
            
            # Extract values at grid centroids
            values = []
            for _, cell in grid.iterrows():
                lon, lat = cell['centroid_lon'], cell['centroid_lat']
                
                try:
                    # Convert coordinates to raster indices
                    col = int((lon - raster.x.values.min()) / (raster.x.values[1] - raster.x.values[0]))
                    row = int((raster.y.values.max() - lat) / (raster.y.values[0] - raster.y.values[1]))
                    
                    if 0 <= row < len(raster.y) and 0 <= col < len(raster.x):
                        val = float(raster.values[row, col])
                    else:
                        val = np.nan
                except:
                    val = np.nan
                
                values.append(val)
            
            # Create dataframe for this month
            month_data = pd.DataFrame({
                'cell_id': grid['cell_id'],
                'month': month,
                'nightlight': values
            })
            
            all_data.append(month_data)
            
            n_valid = (~np.isnan(values)).sum()
            print(f"✓ ({n_valid}/{len(values)} valid)")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Combine all months
    panel_data = pd.concat(all_data, ignore_index=True)
    
    print(f"\n  ✓ Extracted nightlight data")
    print(f"  • Total observations: {len(panel_data):,}")
    print(f"  • Valid nightlight values: {panel_data['nightlight'].notna().sum():,}")
    
    return panel_data


def create_did_dataset(panel_data, treatment_df, event_date):
    """
    Create difference-in-differences ready dataset.
    
    Parameters:
        panel_data: DataFrame with panel data
        treatment_df: DataFrame with treatment assignments
        event_date: datetime of event
        
    Returns:
        DataFrame ready for DiD analysis
    """
    print(f"\n[6/8] Creating DiD dataset...")
    
    # Merge with treatment status
    did_data = panel_data.merge(treatment_df, on='cell_id', how='left')
    
    # Create post-treatment indicator
    did_data['post'] = (did_data['month'] >= event_date).astype(int)
    
    # Create relative time variable (months from event)
    did_data['months_from_event'] = (
        (did_data['month'].dt.year - event_date.year) * 12 +
        (did_data['month'].dt.month - event_date.month)
    )
    
    # Create interaction term for DiD
    did_data['treated_x_post'] = did_data['treated'] * did_data['post']
    
    # Add log transformation (add small constant to handle zeros)
    did_data['log_nightlight'] = np.log(did_data['nightlight'] + 0.01)
    
    print(f"  ✓ DiD dataset created")
    print(f"  • Pre-period observations: {(did_data['post']==0).sum():,}")
    print(f"  • Post-period observations: {(did_data['post']==1).sum():,}")
    print(f"  • Treated × Post observations: {(did_data['treated_x_post']==1).sum():,}")
    
    return did_data


def create_comparison_maps(did_data, grid, undp_gdf, event_date, event_name, output_dir, dpi=200):
    """
    Create before/after comparison maps.
    
    Parameters:
        did_data: DiD dataset
        grid: analysis grid
        undp_gdf: UNDP sites
        event_date: event date
        event_name: name of event
        output_dir: output directory
        dpi: DPI for saved figures
    """
    print(f"\n[7/8] Creating comparison maps...")
    
    # Select before/after months
    before_month = event_date - pd.DateOffset(months=1)
    before_month = before_month.replace(day=1)
    
    after_month = event_date + pd.DateOffset(months=6)
    after_month = after_month.replace(day=1)
    
    # Create maps for both periods
    for period_label, period_month in [('before', before_month), ('after', after_month)]:
        
        print(f"  • Creating {period_label} map ({period_month.strftime('%Y-%m')})...")
        
        # Get data for this month
        month_data = did_data[did_data['month'] == period_month].copy()
        
        if len(month_data) == 0:
            print(f"    ✗ No data available for {period_month.strftime('%Y-%m')}")
            continue
        
        # Merge with grid geometry
        map_data = grid.merge(month_data[['cell_id', 'nightlight', 'treated']], on='cell_id', how='left')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot nightlights
        map_data.plot(
            column='nightlight',
            ax=ax,
            cmap='YlOrRd',
            legend=True,
            legend_kwds={
                'label': 'Nightlight Radiance (nW/cm²/sr)',
                'orientation': 'horizontal',
                'pad': 0.05,
                'shrink': 0.8
            },
            missing_kwds={'color': 'lightgray', 'alpha': 0.3}
        )
        
        # Overlay UNDP sites
        undp_gdf.plot(
            ax=ax,
            color='blue',
            markersize=30,
            alpha=0.6,
            edgecolor='white',
            linewidth=0.5,
            label='UNDP Sites'
        )
        
        # Styling
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        
        title = f"{event_name} - Nightlights {period_label.upper()}\n"
        title += f"{period_month.strftime('%B %Y')}"
        if period_label == 'before':
            title += f" ({abs((period_month - event_date).days // 30)} months before event)"
        else:
            title += f" ({(period_month - event_date).days // 30} months after event)"
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add legend for UNDP sites
        legend_elements = [
            mpatches.Patch(facecolor='blue', edgecolor='white', alpha=0.6, label=f'UNDP Sites (n={len(undp_gdf)})')
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
        
        # Add grid
        ax.grid(True, alpha=0.2, linestyle='--')
        
        plt.tight_layout()
        
        # Save
        output_file = output_dir / f"map_{period_label}_{period_month.strftime('%Y_%m')}.png"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved: {output_file.name}")


def save_outputs(did_data, grid, treatment_df, undp_gdf, output_dir):
    """
    Save all analysis outputs.
    
    Parameters:
        did_data: DiD dataset
        grid: analysis grid
        treatment_df: treatment assignments
        undp_gdf: UNDP sites
        output_dir: output directory
    """
    print(f"\n[8/8] Saving outputs...")
    
    data_dir = output_dir / 'data'
    
    # Save DiD panel data (main analysis file)
    did_file = data_dir / 'panel_did_analysis.csv'
    did_data.to_csv(did_file, index=False)
    print(f"  ✓ Saved DiD panel data: {did_file.name}")
    
    # Save grid with treatment status
    grid_with_treatment = grid.merge(treatment_df, on='cell_id', how='left')
    grid_file = data_dir / 'analysis_grid.geojson'
    grid_with_treatment.to_file(grid_file, driver='GeoJSON')
    print(f"  ✓ Saved analysis grid: {grid_file.name}")
    
    # Save UNDP sites
    undp_file = data_dir / 'undp_sites.geojson'
    undp_gdf[['site_id', 'region', 'township', 'geometry']].to_file(undp_file, driver='GeoJSON')
    print(f"  ✓ Saved UNDP sites: {undp_file.name}")
    
    # Save summary statistics
    summary_stats = {
        'Total grid cells': len(grid),
        'Treated cells': int(treatment_df['treated'].sum()),
        'Control cells': int((treatment_df['treated']==0).sum()),
        'UNDP sites': len(undp_gdf),
        'Months in panel': did_data['month'].nunique(),
        'Total observations': len(did_data),
        'Valid nightlight obs': int(did_data['nightlight'].notna().sum())
    }
    
    summary_file = data_dir / 'summary_statistics.txt'
    with open(summary_file, 'w') as f:
        f.write("UNDP Myanmar Nightlights Analysis - Summary Statistics\n")
        f.write("=" * 60 + "\n\n")
        for key, value in summary_stats.items():
            f.write(f"{key:.<40} {value:>15,}\n")
    
    print(f"  ✓ Saved summary statistics: {summary_file.name}")


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE!")
    print("="*70)
    print("\nNext steps for quasi-experimental analysis:")
    print("\n1. Load the DiD panel data:")
    print("   df = pd.read_csv('outputs/data/panel_did_analysis.csv')")
    print("\n2. Run basic DiD regression:")
    print("   import statsmodels.formula.api as smf")
    print("   model = smf.ols('log_nightlight ~ treated + post + treated_x_post', data=df).fit()")
    print("   print(model.summary())")
    print("\n3. Add fixed effects:")
    print("   model_fe = smf.ols('log_nightlight ~ treated_x_post + C(cell_id) + C(month)', data=df).fit()")
    print("\n4. Check parallel trends assumption:")
    print("   - Plot nightlight trends for treated vs control")
    print("   - Run event study with leads/lags")
    print("\n5. Explore maps:")
    print("   - Check outputs/maps/ for before/after visualizations")
    print("   - Load GeoJSON files in QGIS for interactive exploration")
    print("\n" + "="*70 + "\n")


def main():
    """Main analysis pipeline."""
    
    print("\n" + "="*70)
    print("  UNDP MYANMAR DISASTER IMPACT ASSESSMENT")
    print("  Nightlights-based Quasi-Experimental Analysis")
    print("="*70)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup
    dirs = setup_directories(args.output_dir)
    event_date = pd.Timestamp(args.event_date)
    
    print(f"\nConfiguration:")
    print(f"  • Event: {args.event_name}")
    print(f"  • Event date: {event_date.strftime('%Y-%m-%d')}")
    print(f"  • Analysis window: {args.months_before} months before, {args.months_after} months after")
    print(f"  • Grid resolution: {args.grid_km} km")
    print(f"  • Treatment radius: {args.treatment_radius_km} km")
    
    # Load UNDP data
    undp_gdf = load_undp_data(args.undp_data)
    
    # Define study area
    if args.bbox:
        bbox = args.bbox
        print(f"\nUsing custom bounding box: {bbox}")
    else:
        bbox = create_study_bbox(undp_gdf, buffer_km=50)
        print(f"\nDefined study area bbox: {bbox}")
    
    # Generate month list
    months = generate_month_list(event_date, args.months_before, args.months_after)
    
    # Get VIIRS data
    viirs_files = get_viirs_data(
        months, bbox, args.viirs_dir, args.use_gee, dirs['viirs']
    )
    
    # Create analysis grid
    grid = create_analysis_grid(bbox, args.grid_km)
    
    # Assign treatment
    treatment_df = assign_treatment_status(undp_gdf, grid, args.treatment_radius_km)
    
    # Extract nightlight values
    panel_data = extract_nightlight_values(viirs_files, grid, bbox)
    
    # Create DiD dataset
    did_data = create_did_dataset(panel_data, treatment_df, event_date)
    
    # Create maps
    create_comparison_maps(
        did_data, grid, undp_gdf, event_date, 
        args.event_name, dirs['maps'], args.dpi
    )
    
    # Save outputs
    save_outputs(did_data, grid, treatment_df, undp_gdf, dirs['base'])
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
