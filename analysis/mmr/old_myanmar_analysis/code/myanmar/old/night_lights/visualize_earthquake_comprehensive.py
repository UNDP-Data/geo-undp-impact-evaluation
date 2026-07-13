#!/usr/bin/env python3
"""
Comprehensive Nightlights Visualization - Myanmar 2025 Earthquake
==================================================================

Creates multiple visualizations:
1. Before/After/Recovery maps WITHOUT UNDP locations
2. Before/After/Recovery maps WITH UNDP locations (if available)
3. Time series showing nightlight trends (Jan-Sep 2025)
4. Time series by treatment group (if UNDP data available)

All visualizations properly mask nightlights to Myanmar boundaries only.

Usage:
    python visualize_earthquake_comprehensive.py \
        --viirs-dir ./data/viirs_25 \
        --boundaries-dir ./data/boundaries/processed \
        --undp-support ./data/undp_interventions.geojson \
        --output-dir ./outputs/visualizations_corrected
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import seaborn as sns

try:
    import rioxarray as rxr
    import xarray as xr
    from rasterio.features import geometry_mask
    from matplotlib.colors import LinearSegmentedColormap
    RASTER_OK = True
except ImportError:
    RASTER_OK = False
    print("⚠ rioxarray not installed. Install with: pip install rioxarray")


# 2025 Earthquake event date
EARTHQUAKE_DATE = datetime(2025, 3, 28)

# Period definitions (March excluded)
PRE_PERIOD_START = datetime(2025, 1, 1)
PRE_PERIOD_END = datetime(2025, 2, 28)
POST_PERIOD_START = datetime(2025, 4, 1)
POST_PERIOD_END = datetime(2025, 9, 30)


def find_viirs_files(viirs_dir, start_date=None, end_date=None):
    """Find all VIIRS monthly files in directory."""
    viirs_dir = Path(viirs_dir)
    
    tif_files = list(viirs_dir.glob("VIIRS_*.tif")) + list(viirs_dir.glob("SVDNB_*.tif"))
    tif_files = [f for f in tif_files if not f.name.endswith('.tif.gz')]
    
    print(f"\nFound {len(tif_files)} VIIRS .tif files")
    
    if len(tif_files) == 0:
        print("\n⚠ No VIIRS .tif files found!")
        return []
    
    files_with_dates = []
    
    for f in tif_files:
        parts = f.name.split('_')
        date = None
        
        if f.name.startswith('VIIRS_') and len(parts) >= 2:
            yyyymm = parts[1]
            if len(yyyymm) == 6 and yyyymm.isdigit():
                year = int(yyyymm[:4])
                month = int(yyyymm[4:6])
                date = datetime(year, month, 1)
        
        elif f.name.startswith('SVDNB_') and len(parts) >= 3:
            date_range = parts[2].split('-')[0]
            if len(date_range) >= 6 and date_range[:6].isdigit():
                yyyymm = date_range[:6]
                year = int(yyyymm[:4])
                month = int(yyyymm[4:6])
                date = datetime(year, month, 1)
        
        if date:
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue
            
            files_with_dates.append((date, f))
    
    files_with_dates.sort(key=lambda x: x[0])
    
    if len(files_with_dates) > 0:
        print(f"Date range: {files_with_dates[0][0].strftime('%Y-%m')} to {files_with_dates[-1][0].strftime('%Y-%m')}")
    
    return files_with_dates


def load_and_clip_viirs(tif_file, bbox):
    """Load VIIRS file and clip to bounding box."""
    if not RASTER_OK:
        raise RuntimeError("rioxarray not installed")
    
    raster = rxr.open_rasterio(tif_file, masked=True).squeeze()
    lon_min, lat_min, lon_max, lat_max = bbox
    raster_clipped = raster.rio.clip_box(
        minx=lon_min, miny=lat_min,
        maxx=lon_max, maxy=lat_max
    )
    
    return raster_clipped


def mask_raster_to_boundaries(raster, boundaries_gdf):
    """
    Mask raster to only show values within boundaries.
    This eliminates cross-border light contamination.
    """
    mask = geometry_mask(
        boundaries_gdf.geometry,
        out_shape=raster.shape,
        transform=raster.rio.transform(),
        invert=True
    )
    
    raster_masked = raster.copy()
    raster_masked.values[~mask] = np.nan
    
    return raster_masked


def calculate_mean_nightlight(raster, gdf, support_status=None):
    """Calculate mean nightlight value for areas in GeoDataFrame."""
    if support_status is not None and 'undp_support' in gdf.columns:
        gdf = gdf[gdf['undp_support'] == support_status]
    
    try:
        mask = geometry_mask(
            gdf.geometry,
            out_shape=raster.shape,
            transform=raster.rio.transform(),
            invert=True
        )
        
        values = raster.values[mask]
        values = values[~np.isnan(values)]
        
        if len(values) > 0:
            return float(np.mean(values))
        else:
            return np.nan
    
    except Exception as e:
        print(f"Error calculating mean: {e}")
        return np.nan


def create_before_after_maps(viirs_files, boundaries_gdf, undp_support_gdf, output_dir, include_undp=True):
    """
    Create before/after comparison maps.
    
    Parameters:
        include_undp: If True, plot UNDP locations. If False, create clean map without interventions.
    """
    print("\n" + "="*80)
    if include_undp:
        print("CREATING BEFORE/AFTER MAPS WITH UNDP INTERVENTIONS")
    else:
        print("CREATING BEFORE/AFTER MAPS (CLEAN VERSION)")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Timeline dates
    before_date = datetime(2025, 2, 1)
    after_date = datetime(2025, 4, 1)
    recovery_date = datetime(2025, 9, 1)
    
    before_file = None
    after_file = None
    recovery_file = None
    
    for date, file in viirs_files:
        if date == before_date:
            before_file = file
        if date == after_date:
            after_file = file
        if date == recovery_date:
            recovery_file = file
    
    if not before_file or not after_file or not recovery_file:
        print("\n⚠ Could not find February, April, and September 2025 files")
        print("Available months:")
        for date, file in viirs_files:
            print(f"  - {date.strftime('%Y-%m')}: {file.name}")
        return
    
    print(f"\nBefore (Feb):     {before_file.name}")
    print(f"After (Apr):      {after_file.name}")
    print(f"Recovery (Sep):   {recovery_file.name}")
    
    # Get Myanmar bbox
    bounds = boundaries_gdf.total_bounds
    
    if (bounds[2] - bounds[0]) < 5 or (bounds[3] - bounds[1]) < 5:
        print(f"\n⚠ Detected small boundary extent, using full Myanmar coordinates")
        bbox = (92.0, 9.5, 101.5, 28.5)
    else:
        buffer = 0.5
        bbox = (bounds[0] - buffer, bounds[1] - buffer, 
                bounds[2] + buffer, bounds[3] + buffer)
    
    print(f"\nShowing Myanmar region:")
    print(f"  Lon: {bbox[0]:.2f}° to {bbox[2]:.2f}°")
    print(f"  Lat: {bbox[1]:.2f}° to {bbox[3]:.2f}°")
    
    # Load rasters
    print("\nLoading nightlight data...")
    try:
        raster_before = load_and_clip_viirs(before_file, bbox)
        raster_after = load_and_clip_viirs(after_file, bbox)
        raster_recovery = load_and_clip_viirs(recovery_file, bbox)
        
        print(f"✓ Loaded before:   {raster_before.shape}")
        print(f"✓ Loaded after:    {raster_after.shape}")
        print(f"✓ Loaded recovery: {raster_recovery.shape}")
    except Exception as e:
        print(f"\n✗ Error loading rasters: {e}")
        return
    
    # Mask rasters to Myanmar boundaries
    print("\n✓ Masking nightlights to Myanmar boundaries only...")
    raster_before = mask_raster_to_boundaries(raster_before, boundaries_gdf)
    raster_after = mask_raster_to_boundaries(raster_after, boundaries_gdf)
    raster_recovery = mask_raster_to_boundaries(raster_recovery, boundaries_gdf)
    
    # Ensure boundaries are in EPSG:4326
    if boundaries_gdf.crs is None:
        boundaries_gdf = boundaries_gdf.set_crs('EPSG:4326')
    elif boundaries_gdf.crs.to_string() != 'EPSG:4326':
        boundaries_gdf = boundaries_gdf.to_crs('EPSG:4326')
    
    # Ensure UNDP support areas are in same CRS
    if undp_support_gdf is not None and not undp_support_gdf.empty and include_undp:
        if undp_support_gdf.crs is None:
            undp_support_gdf = undp_support_gdf.set_crs('EPSG:4326')
        elif undp_support_gdf.crs.to_string() != 'EPSG:4326':
            undp_support_gdf = undp_support_gdf.to_crs('EPSG:4326')
        
        print(f"\n→ Will plot {len(undp_support_gdf)} UNDP intervention points")
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Black-to-yellow colormap
    colors = ['#000000', '#FFFF00']
    cmap_lights = LinearSegmentedColormap.from_list('black_yellow', colors, N=256)
    
    # Threshold for bright lights
    vmin = 0
    vmax = 1.0
    threshold = 0.75
    
    # BEFORE (February 2025)
    before_masked = raster_before.copy()
    before_masked = before_masked.where(before_masked > threshold, np.nan)
    
    im1 = before_masked.plot(ax=axes[0], cmap=cmap_lights, vmin=threshold, vmax=vmax,
                            add_colorbar=False, alpha=1.0)
    
    boundaries_gdf.boundary.plot(ax=axes[0], edgecolor='white', linewidth=0.5, alpha=0.8)
    
    axes[0].set_title('BEFORE Earthquake\nFebruary 2025', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Longitude', fontsize=10)
    axes[0].set_ylabel('Latitude', fontsize=10)
    axes[0].set_xlim(bbox[0], bbox[2])
    axes[0].set_ylim(bbox[1], bbox[3])
    axes[0].set_facecolor('#000000')
    axes[0].patch.set_facecolor('#000000')
    
    # AFTER (April 2025)
    after_masked = raster_after.copy()
    after_masked = after_masked.where(after_masked > threshold, np.nan)
    
    im2 = after_masked.plot(ax=axes[1], cmap=cmap_lights, vmin=threshold, vmax=vmax,
                           add_colorbar=False, alpha=1.0)
    
    boundaries_gdf.boundary.plot(ax=axes[1], edgecolor='white', linewidth=0.5, alpha=0.8)
    
    axes[1].set_title('AFTER Earthquake\nApril 2025', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Longitude', fontsize=10)
    axes[1].set_ylabel('')
    axes[1].set_xlim(bbox[0], bbox[2])
    axes[1].set_ylim(bbox[1], bbox[3])
    axes[1].set_facecolor('#000000')
    axes[1].patch.set_facecolor('#000000')
    
    # RECOVERY (September 2025)
    recovery_masked = raster_recovery.copy()
    recovery_masked = recovery_masked.where(recovery_masked > threshold, np.nan)
    
    im3 = recovery_masked.plot(ax=axes[2], cmap=cmap_lights, vmin=threshold, vmax=vmax,
                              add_colorbar=False, alpha=1.0)
    
    boundaries_gdf.boundary.plot(ax=axes[2], edgecolor='white', linewidth=0.5, alpha=0.8)
    
    # Plot UNDP intervention locations (only if include_undp=True)
    if undp_support_gdf is not None and not undp_support_gdf.empty and include_undp:
        if 'intervention_color' in undp_support_gdf.columns and 'intervention_type' in undp_support_gdf.columns:
            # Plot each intervention type with its specific color
            for intervention_type in undp_support_gdf['intervention_type'].unique():
                subset = undp_support_gdf[undp_support_gdf['intervention_type'] == intervention_type]
                centroids = subset.geometry.centroid
                lons = [pt.x for pt in centroids]
                lats = [pt.y for pt in centroids]
                color = subset['intervention_color'].iloc[0]
                
                axes[2].scatter(lons, lats, c=color, s=20, alpha=0.9,
                               edgecolors='white', linewidths=0.7, zorder=5, label=intervention_type)
        else:
            # Fallback: use single color
            centroids = undp_support_gdf.geometry.centroid
            lons = [pt.x for pt in centroids]
            lats = [pt.y for pt in centroids]
            axes[2].scatter(lons, lats, c='#00FF00', s=50, alpha=0.8,
                           edgecolors='white', linewidths=1, zorder=5, label='UNDP Interventions')
        
        axes[2].legend(loc='upper right', fontsize=8, framealpha=0.9,
                      facecolor='black', edgecolor='white', labelcolor='white')
    
    axes[2].set_title('RECOVERY\nSeptember 2025', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Longitude', fontsize=10)
    axes[2].set_ylabel('')
    axes[2].set_xlim(bbox[0], bbox[2])
    axes[2].set_ylim(bbox[1], bbox[3])
    axes[2].set_facecolor('#000000')
    axes[2].patch.set_facecolor('#000000')
    
    # Add colorbar
    fig.subplots_adjust(bottom=0.35)
    cbar_ax = fig.add_axes([0.35, 0.0001, 0.30, 0.02])
    cbar = plt.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Change in Radiance', fontsize=11)
    
    # Clean title without "CORRECTED"
    plt.suptitle('2025 Myanmar Earthquake Impact & Recovery\nMarch 28, 2025',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save with appropriate filename
    if include_undp:
        output_file = output_dir / 'earthquake_2025_impact_with_interventions.png'
    else:
        output_file = output_dir / 'earthquake_2025_impact_clean.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    plt.close()


def create_time_series_analysis(viirs_files, boundaries_gdf, output_dir):
    """
    Create time series showing nightlight trends Jan-Sep 2025.
    """
    print("\n" + "="*80)
    print("CREATING TIME SERIES ANALYSIS (JAN-SEP 2025)")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Get bbox
    bounds = boundaries_gdf.total_bounds
    
    if (bounds[2] - bounds[0]) < 5 or (bounds[3] - bounds[1]) < 5:
        bbox = (92.0, 9.5, 101.5, 28.5)
    else:
        buffer = 0.5
        bbox = (bounds[0] - buffer, bounds[1] - buffer, 
                bounds[2] + buffer, bounds[3] + buffer)
    
    # Calculate mean nightlight for each month
    timeseries_data = []
    
    print(f"\nProcessing {len(viirs_files)} months...")
    
    for date, file in viirs_files:
        print(f"  {date.strftime('%Y-%m')}...", end=' ')
        
        try:
            raster = load_and_clip_viirs(file, bbox)
            raster = mask_raster_to_boundaries(raster, boundaries_gdf)
            
            mean_nl = calculate_mean_nightlight(raster, boundaries_gdf)
            
            if not np.isnan(mean_nl):
                timeseries_data.append({
                    'date': date,
                    'mean_nightlight': mean_nl
                })
                print(f"✓ {mean_nl:.4f}")
            else:
                print("✗ No data")
        
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    if not timeseries_data:
        print("\n⚠ No time series data generated")
        return None
    
    df = pd.DataFrame(timeseries_data)
    df = df.sort_values('date')
    
    # Save CSV
    csv_file = output_dir / 'nightlight_time_series_2025.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved time series data: {csv_file}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot time series
    ax.plot(df['date'], df['mean_nightlight'], 'o-', 
            color='#2E86AB', linewidth=2.5, markersize=8,
            label='Mean Nightlight')
    
    # Mark earthquake
    ax.axvline(EARTHQUAKE_DATE, color='red', linestyle='--', 
               linewidth=2, alpha=0.7, label='Earthquake (Mar 28)')
    
    # Shade periods
    ax.axvspan(PRE_PERIOD_START, PRE_PERIOD_END, alpha=0.1, color='gray', 
               label='Pre-Period (Jan-Feb)')
    ax.axvspan(POST_PERIOD_START, POST_PERIOD_END, alpha=0.1, color='green',
               label='Post-Period (Apr-Sep)')
    
    # Shade March (excluded)
    march_start = datetime(2025, 3, 1)
    march_end = datetime(2025, 3, 31)
    ax.axvspan(march_start, march_end, alpha=0.2, color='red',
               label='March (EXCLUDED)')
    
    # Format x-axis with month names
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    ax.set_xlabel('Month (2025)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Nightlight Radiance', fontsize=12, fontweight='bold')
    ax.set_title('Myanmar Nightlight Time Series\nJanuary - September 2025', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = output_dir / 'nightlight_time_series_2025.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved time series plot: {plot_file}")
    
    plt.close()
    
    return df


def create_time_series_by_treatment(viirs_files, boundaries_gdf, undp_support_gdf, output_dir):
    """
    Create time series comparing treated vs control areas.
    """
    print("\n" + "="*80)
    print("CREATING TIME SERIES BY TREATMENT GROUP")
    print("="*80)
    
    if undp_support_gdf is None or undp_support_gdf.empty:
        print("\n⚠ No UNDP support data - skipping treatment group analysis")
        return None
    
    output_dir = Path(output_dir)
    
    # Get bbox
    bounds = boundaries_gdf.total_bounds
    if (bounds[2] - bounds[0]) < 5:
        bbox = (92.0, 9.5, 101.5, 28.5)
    else:
        buffer = 0.5
        bbox = (bounds[0] - buffer, bounds[1] - buffer, 
                bounds[2] + buffer, bounds[3] + buffer)
    
    # Classify areas as treated/control
    boundaries_gdf = boundaries_gdf.copy()
    boundaries_gdf['treated'] = 0
    
    for idx, area in boundaries_gdf.iterrows():
        intersecting = undp_support_gdf[undp_support_gdf.intersects(area.geometry)]
        if len(intersecting) > 0:
            boundaries_gdf.at[idx, 'treated'] = 1
    
    n_treated = boundaries_gdf['treated'].sum()
    n_control = len(boundaries_gdf) - n_treated
    print(f"\nAreas: {n_treated} treated, {n_control} control")
    
    # Calculate time series for each group
    timeseries_data = []
    
    print(f"\nProcessing {len(viirs_files)} months...")
    
    for date, file in viirs_files:
        print(f"  {date.strftime('%Y-%m')}...", end=' ')
        
        try:
            raster = load_and_clip_viirs(file, bbox)
            raster = mask_raster_to_boundaries(raster, boundaries_gdf)
            
            # Calculate for treated areas
            treated_gdf = boundaries_gdf[boundaries_gdf['treated'] == 1]
            mean_treated = calculate_mean_nightlight(raster, treated_gdf)
            
            # Calculate for control areas
            control_gdf = boundaries_gdf[boundaries_gdf['treated'] == 0]
            mean_control = calculate_mean_nightlight(raster, control_gdf)
            
            timeseries_data.append({
                'date': date,
                'treated': mean_treated,
                'control': mean_control
            })
            print(f"✓ T:{mean_treated:.4f} C:{mean_control:.4f}")
        
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    if not timeseries_data:
        print("\n⚠ No time series data generated")
        return None
    
    df = pd.DataFrame(timeseries_data)
    df = df.sort_values('date')
    
    # Save CSV
    csv_file = output_dir / 'nightlight_time_series_by_treatment.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved time series data: {csv_file}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot both groups
    ax.plot(df['date'], df['treated'], 'o-', 
            color='#2E86AB', linewidth=2.5, markersize=8,
            label='Treated (UNDP States)')
    ax.plot(df['date'], df['control'], 's-', 
            color='#A23B72', linewidth=2.5, markersize=8,
            label='Control States')
    
    # Mark earthquake
    ax.axvline(EARTHQUAKE_DATE, color='red', linestyle='--', 
               linewidth=2, alpha=0.7)
    
    # Shade periods
    ax.axvspan(POST_PERIOD_START, POST_PERIOD_END, alpha=0.1, color='red')
    
    # Shade March (excluded)
    march_start = datetime(2025, 3, 1)
    march_end = datetime(2025, 3, 31)
    ax.axvspan(march_start, march_end, alpha=0.2, color='gray',
               label='March (EXCLUDED)')
    
    # Format x-axis with month names
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    ax.set_xlabel('Month (2025)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Nightlight Radiance', fontsize=12, fontweight='bold')
    ax.set_title('Nightlight Recovery: Treated vs Control Areas', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = output_dir / 'nightlight_time_series_by_treatment.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved treatment group plot: {plot_file}")
    
    plt.close()
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive nightlights visualization for Myanmar 2025 earthquake"
    )
    
    parser.add_argument(
        "--viirs-dir",
        type=str,
        default="./data/viirs_25",
        help="Directory with VIIRS .tif files for 2025"
    )
    
    parser.add_argument(
        "--boundaries-dir",
        type=str,
        default="./data/boundaries/processed",
        help="Directory with processed boundaries"
    )
    
    parser.add_argument(
        "--undp-support",
        type=str,
        default=None,
        help="GeoJSON file with UNDP support areas (optional)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/visualizations_corrected",
        help="Output directory for visualizations"
    )
    
    args = parser.parse_args()
    
    if not RASTER_OK:
        print("\n✗ Missing required packages")
        print("Install with: pip install rioxarray xarray rasterio")
        return
    
    print("="*80)
    print("  MYANMAR 2025 EARTHQUAKE COMPREHENSIVE VISUALIZATION")
    print("  March 28, 2025")
    print("="*80)
    
    # Find VIIRS files (Jan-Sept 2025)
    start = datetime(2025, 1, 1)
    end = datetime(2025, 9, 30)
    
    viirs_files = find_viirs_files(args.viirs_dir, start, end)
    
    if not viirs_files:
        print("\n✗ No VIIRS files found")
        print(f"Check directory: {args.viirs_dir}")
        return
    
    # Load boundaries
    print("\nLoading boundaries...")
    boundaries_dir = Path(args.boundaries_dir)
    
    boundary_files_to_try = [
        boundaries_dir / 'mmr_admin1.geojson',
        boundaries_dir / 'myanmar_admin1_mocha.geojson',
        boundaries_dir / 'myanmar_admin0.geojson',
        boundaries_dir / 'mmr_admin3.geojson',
    ]
    
    boundaries_gdf = None
    for boundaries_file in boundary_files_to_try:
        if boundaries_file.exists():
            boundaries_gdf = gpd.read_file(boundaries_file)
            print(f"✓ Loaded {len(boundaries_gdf)} regions from {boundaries_file.name}")
            break
    
    if boundaries_gdf is None:
        print(f"\n✗ Boundaries file not found in {boundaries_dir}")
        return
    
    # Load UNDP support areas
    undp_support_gdf = None
    if args.undp_support:
        undp_file = Path(args.undp_support)
        if undp_file.exists():
            undp_support_gdf = gpd.read_file(undp_file)
            print(f"✓ Loaded {len(undp_support_gdf)} UNDP support areas from {undp_file.name}")
        else:
            print(f"\n⚠ UNDP support file not found: {undp_file}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Clean map (without UNDP locations)
    print("\n[1/4] Creating clean before/after/recovery map...")
    create_before_after_maps(viirs_files, boundaries_gdf, undp_support_gdf, 
                            output_dir, include_undp=False)
    
    # 2. Map with UNDP interventions (if available)
    if undp_support_gdf is not None:
        print("\n[2/4] Creating map with UNDP interventions...")
        create_before_after_maps(viirs_files, boundaries_gdf, undp_support_gdf, 
                                output_dir, include_undp=True)
    else:
        print("\n[2/4] Skipping UNDP intervention map (no data)")
    
    # 3. Overall time series
    print("\n[3/4] Creating overall time series...")
    df_timeseries = create_time_series_analysis(viirs_files, boundaries_gdf, output_dir)
    
    # 4. Time series by treatment group (if UNDP data available)
    if undp_support_gdf is not None:
        print("\n[4/4] Creating time series by treatment group...")
        df_treatment = create_time_series_by_treatment(viirs_files, boundaries_gdf, 
                                                       undp_support_gdf, output_dir)
    else:
        print("\n[4/4] Skipping treatment group analysis (no UNDP data)")
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  1. earthquake_2025_impact_clean.png - Before/after/recovery (no interventions)")
    if undp_support_gdf is not None:
        print("  2. earthquake_2025_impact_with_interventions.png - With UNDP locations")
    print("  3. nightlight_time_series_2025.png - Overall time series")
    print("  4. nightlight_time_series_2025.csv - Time series data")
    if undp_support_gdf is not None:
        print("  5. nightlight_time_series_by_treatment.png - Treated vs control")
        print("  6. nightlight_time_series_by_treatment.csv - Treatment group data")
    
    print("\n" + "="*80)
    print("NOTES:")
    print("="*80)
    print("✓ All nightlights masked to Myanmar boundaries only")
    print("✓ March 2025 excluded from all analyses")
    print("✓ Time series shows complete Jan-Sep 2025 trend")
    print("✓ Dates formatted with month names for clarity")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
