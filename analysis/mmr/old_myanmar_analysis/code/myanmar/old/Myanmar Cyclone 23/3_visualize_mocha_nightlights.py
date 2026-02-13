#!/usr/bin/env python3
"""
Visualize Nightlights Before/After Cyclone Mocha
=================================================

Creates visualizations showing nightlight changes in Cyclone Mocha affected areas.

Cyclone Mocha: May 14, 2023 (Rakhine State, Myanmar)

Analysis:
- Compare nightlights before (pre-May 2023) vs after (post-May 2023)
- Focus on Rakhine State (primary impact)
- Track recovery over time
- Prepare for budget correlation analysis

Requirements:
    pip install rioxarray xarray rasterio geopandas matplotlib pandas numpy

Usage:
    python 3_visualize_mocha_nightlights.py \\
        --viirs-dir ./data/viirs \\
        --boundaries-dir ./data/boundaries/processed \\
        --output-dir ./outputs/mocha_analysis
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
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import seaborn as sns

try:
    import rioxarray as rxr
    import xarray as xr
    RASTER_OK = True
except ImportError:
    RASTER_OK = False
    print("⚠ rioxarray not installed. Install with: pip install rioxarray")


# Cyclone Mocha event date
MOCHA_DATE = datetime(2023, 5, 14)

# Color schemes
IMPACT_COLORS = {
    'primary': '#d62728',
    'moderate': '#ff7f0e', 
    'secondary': '#ffbb78',
    'none': '#d3d3d3'
}


def find_viirs_files(viirs_dir, start_date=None, end_date=None):
    """
    Find all VIIRS monthly files in directory.
    
    Parameters:
        viirs_dir: Directory with VIIRS files
        start_date: Optional datetime to filter
        end_date: Optional datetime to filter
    
    Returns:
        list of (datetime, Path) tuples
    """
    viirs_dir = Path(viirs_dir)
    
    # Find all VIIRS .tif files (not .gz)
    # Try both VIIRS_* and SVDNB_* patterns
    tif_files = list(viirs_dir.glob("VIIRS_*.tif")) + list(viirs_dir.glob("SVDNB_*.tif"))
    tif_files = [f for f in tif_files if not f.name.endswith('.tif.gz')]
    
    print(f"\nFound {len(tif_files)} VIIRS .tif files")
    
    if len(tif_files) == 0:
        print("\n⚠ No VIIRS .tif files found!")
        print("Make sure you've:")
        print("  1. Downloaded VIIRS files")
        print("  2. Extracted .gz files (use script 1 with --extract flag)")
        return []
    
    # Extract dates from filenames
    files_with_dates = []
    
    for f in tif_files:
        # Try different filename patterns
        parts = f.name.split('_')
        date = None
        
        # Pattern 1: VIIRS_YYYYMM_*.tif
        if f.name.startswith('VIIRS_') and len(parts) >= 2:
            yyyymm = parts[1]
            if len(yyyymm) == 6 and yyyymm.isdigit():
                year = int(yyyymm[:4])
                month = int(yyyymm[4:6])
                date = datetime(year, month, 1)
        
        # Pattern 2: SVDNB_npp_YYYYMMDD-YYYYMMDD_*.tif
        elif f.name.startswith('SVDNB_') and len(parts) >= 3:
            # Date is in parts[2], format: "20230301-20230331"
            date_range = parts[2].split('-')[0]  # Get "20230301"
            if len(date_range) >= 6 and date_range[:6].isdigit():
                yyyymm = date_range[:6]  # Get "202303"
                year = int(yyyymm[:4])
                month = int(yyyymm[4:6])
                date = datetime(year, month, 1)
        
        if date:
            # Filter by date range
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue
            
            files_with_dates.append((date, f))
    
    # Sort by date
    files_with_dates.sort(key=lambda x: x[0])
    
    if len(files_with_dates) > 0:
        print(f"\nDate range: {files_with_dates[0][0].strftime('%Y-%m')} to {files_with_dates[-1][0].strftime('%Y-%m')}")
    else:
        print("\n⚠ No files matched the date range or naming pattern")
    
    return files_with_dates


def load_and_clip_viirs(tif_file, bbox, affected_areas_gdf=None):
    """
    Load VIIRS file and clip to bounding box.
    
    Parameters:
        tif_file: Path to VIIRS .tif file
        bbox: (lon_min, lat_min, lon_max, lat_max)
        affected_areas_gdf: Optional GeoDataFrame to mask
    
    Returns:
        xarray DataArray
    """
    if not RASTER_OK:
        raise RuntimeError("rioxarray not installed")
    
    # Open raster
    raster = rxr.open_rasterio(tif_file, masked=True).squeeze()
    
    # Clip to bbox
    lon_min, lat_min, lon_max, lat_max = bbox
    raster_clipped = raster.rio.clip_box(
        minx=lon_min, miny=lat_min,
        maxx=lon_max, maxy=lat_max
    )
    
    return raster_clipped


def calculate_mean_nightlight(raster, gdf, impact_level=None):
    """
    Calculate mean nightlight value for areas in GeoDataFrame.
    
    Parameters:
        raster: xarray DataArray with nightlights
        gdf: GeoDataFrame with geometries
        impact_level: Optional filter by mocha_impact level
    
    Returns:
        float: mean nightlight value
    """
    if impact_level:
        gdf = gdf[gdf['mocha_impact'] == impact_level]
    
    # Get raster values within geometries
    try:
        from rasterio.features import geometry_mask
        
        # Create mask
        mask = geometry_mask(
            gdf.geometry,
            out_shape=raster.shape,
            transform=raster.rio.transform(),
            invert=True
        )
        
        # Calculate mean (excluding masked values)
        values = raster.values[mask]
        values = values[~np.isnan(values)]
        
        if len(values) > 0:
            return float(np.mean(values))
        else:
            return np.nan
    
    except Exception as e:
        print(f"Error calculating mean: {e}")
        return np.nan


def create_before_after_maps(viirs_files, boundaries_gdf, output_dir):
    """
    Create before/after comparison maps.
    
    Parameters:
        viirs_files: list of (datetime, Path) tuples
        boundaries_gdf: GeoDataFrame with affected areas
        output_dir: Where to save maps
    """
    print("\n" + "="*80)
    print("CREATING BEFORE/AFTER COMPARISON MAPS")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find files for timeline: Before → After → Recovery
    # Before: April 2023 (one month before cyclone)
    # After: June 2023 (one month after cyclone)
    # Recovery: September 2023 (4 months after cyclone - showing reconstruction)
    
    before_date = datetime(2023, 4, 1)
    after_date = datetime(2023, 6, 1)
    recovery_date = datetime(2023, 9, 1)
    
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
        print("\n⚠ Could not find April, June, and September 2023 files")
        print("Available months:")
        for date, _ in viirs_files[:10]:
            print(f"  - {date.strftime('%Y-%m')}")
        return
    
    print(f"\nBefore:   {before_file.name}")
    print(f"After:    {after_file.name}")
    print(f"Recovery: {recovery_file.name}")
    
    # Get Myanmar full bbox (to show whole country)
    bounds = boundaries_gdf.total_bounds  # [minx, miny, maxx, maxy]
    # Add small buffer
    buffer = 0.5
    bbox = (bounds[0] - buffer, bounds[1] - buffer, 
            bounds[2] + buffer, bounds[3] + buffer)
    
    # Get Rakhine for highlighting
    rakhine = boundaries_gdf[boundaries_gdf['mocha_impact'] == 'primary']
    if rakhine.empty:
        print("\n⚠ No primary impact areas found")
        return
    
    print(f"\nShowing full Myanmar:")
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
    
    # Create figure with 3 panels: before, after, change
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Simple black-to-yellow with HIGH threshold
    # Use 0-1 scale (actual radiance range)
    vmin = 0
    vmax = 1.0
    
    # Create black-to-yellow colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#000000', '#FFFF00']  # Black to bright yellow
    cmap_lights = LinearSegmentedColormap.from_list('black_yellow', colors, N=256)
    
    # HIGH threshold - only show very bright areas
    threshold = 0.75  # 75% threshold - only brightest lights show
    
    # Before
    before_masked = raster_before.copy()
    before_masked = before_masked.where(before_masked > threshold, np.nan)
    
    im1 = before_masked.plot(ax=axes[0], cmap=cmap_lights, vmin=threshold, vmax=vmax,
                            add_colorbar=False, alpha=1.0)
    
    # Plot boundaries
    boundaries_gdf.boundary.plot(ax=axes[0], edgecolor='white', linewidth=0.4, alpha=0.6)
    axes[0].set_title('BEFORE Cyclone Mocha\nApril 2023', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Longitude', fontsize=10)
    axes[0].set_ylabel('Latitude', fontsize=10)
    axes[0].set_facecolor('#000000')
    axes[0].patch.set_facecolor('#000000')
    
    # After
    after_masked = raster_after.copy()
    after_masked = after_masked.where(after_masked > threshold, np.nan)
    
    im2 = after_masked.plot(ax=axes[1], cmap=cmap_lights, vmin=threshold, vmax=vmax,
                           add_colorbar=False, alpha=1.0)
    
    # Plot boundaries
    boundaries_gdf.boundary.plot(ax=axes[1], edgecolor='white', linewidth=0.4, alpha=0.6)
    axes[1].set_title('AFTER Cyclone Mocha\nJune 2023', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Longitude', fontsize=10)
    axes[1].set_ylabel('')
    axes[1].set_facecolor('#000000')
    axes[1].patch.set_facecolor('#000000')
    
    # Recovery (September 2023 - 4 months after cyclone)
    recovery_masked = raster_recovery.copy()
    recovery_masked = recovery_masked.where(recovery_masked > threshold, np.nan)
    
    im3 = recovery_masked.plot(ax=axes[2], cmap=cmap_lights, vmin=threshold, vmax=vmax,
                              add_colorbar=False, alpha=1.0)
    
    # Plot boundaries
    boundaries_gdf.boundary.plot(ax=axes[2], edgecolor='white', linewidth=0.4, alpha=0.6)
    axes[2].set_title('RECOVERY\nSeptember 2023', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Longitude', fontsize=10)
    axes[2].set_ylabel('')
    axes[2].set_facecolor('#000000')
    axes[2].patch.set_facecolor('#000000')
    
    # Add single centered colorbar below all maps
    # Create a new axis for the colorbar below all subplots
    fig.subplots_adjust(bottom=0.35)  # Make room for colorbar
    cbar_ax = fig.add_axes([0.35, 0.0001, 0.30, 0.02])  # [left, bottom, width, height]
    cbar = plt.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Change in Radiance', fontsize=11, fontweight='normal')
    
    # Add simple legend explaining boundaries
    legend_elements = [
        Line2D([0], [0], color='white', linewidth=0.4, label='State/Region Boundaries')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=9, 
                   framealpha=0.9, facecolor='black', edgecolor='white', labelcolor='white')
    
    plt.suptitle('Cyclone Mocha Impact & Recovery - Myanmar\nMay 14, 2023',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / 'mocha_impact_recovery_timeline.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    plt.close()


def create_time_series_analysis(viirs_files, boundaries_gdf, output_dir):
    """
    Create time series showing nightlight trends before/after Mocha.
    
    Parameters:
        viirs_files: list of (datetime, Path) tuples
        boundaries_gdf: GeoDataFrame with affected areas
        output_dir: Where to save figures
    """
    print("\n" + "="*80)
    print("CREATING TIME SERIES ANALYSIS")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get Rakhine bbox
    rakhine = boundaries_gdf[boundaries_gdf['mocha_impact'].isin(['primary', 'moderate'])]
    if rakhine.empty:
        print("\n⚠ No Rakhine areas found")
        return
    
    bounds = rakhine.total_bounds
    bbox = tuple(bounds)
    
    print(f"\nProcessing {len(viirs_files)} months...")
    print(f"This may take several minutes...")
    
    # Calculate mean nightlight for each month and impact level
    results = []
    
    for i, (date, file) in enumerate(viirs_files, 1):
        print(f"\r  Processing {date.strftime('%Y-%m')} ({i}/{len(viirs_files)})...", end='')
        
        try:
            raster = load_and_clip_viirs(file, bbox)
            
            # Calculate mean for each impact level
            for impact in ['primary', 'moderate', 'secondary']:
                mean_val = calculate_mean_nightlight(raster, boundaries_gdf, impact)
                
                results.append({
                    'date': date,
                    'impact_level': impact,
                    'mean_nightlight': mean_val
                })
        
        except Exception as e:
            print(f"\n  ⚠ Error processing {date.strftime('%Y-%m')}: {e}")
            continue
    
    print("\n✓ Processing complete")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save data
    data_file = output_dir / 'nightlight_time_series.csv'
    df.to_csv(data_file, index=False)
    print(f"\n✓ Saved data: {data_file}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for impact in ['primary', 'moderate', 'secondary']:
        data = df[df['impact_level'] == impact]
        if not data.empty:
            ax.plot(data['date'], data['mean_nightlight'], 
                   marker='o', linewidth=2, markersize=6,
                   color=IMPACT_COLORS[impact],
                   label=f'{impact.upper()} impact areas')
    
    # Add vertical line at Cyclone Mocha
    ax.axvline(MOCHA_DATE, color='red', linestyle='--', linewidth=2, 
              alpha=0.7, label='Cyclone Mocha (May 14, 2023)')
    
    # Shade pre/post periods
    pre_dates = [d for d, _ in viirs_files if d < MOCHA_DATE]
    post_dates = [d for d, _ in viirs_files if d >= MOCHA_DATE]
    
    if pre_dates and post_dates:
        ax.axvspan(min(pre_dates), MOCHA_DATE, alpha=0.1, color='blue', label='Pre-Mocha')
        ax.axvspan(MOCHA_DATE, max(post_dates), alpha=0.1, color='green', label='Post-Mocha')
    
    # Styling
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Nightlight Radiance', fontsize=12, fontweight='bold')
    ax.set_title('Nightlight Trends: Cyclone Mocha Impact Areas\nRakhine State, Myanmar',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / 'mocha_nightlight_time_series.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()
    
    return df


def prepare_for_budget_analysis(df_nightlights, output_dir):
    """
    Prepare nightlight data for budget correlation analysis.
    
    Parameters:
        df_nightlights: DataFrame with time series data
        output_dir: Where to save prepared data
    """
    print("\n" + "="*80)
    print("PREPARING FOR BUDGET ANALYSIS")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Create quarterly aggregates
    df = df_nightlights.copy()
    df['quarter'] = df['date'].dt.to_period('Q')
    df['year'] = df['date'].dt.year
    
    # Aggregate by quarter and impact level
    quarterly = df.groupby(['quarter', 'impact_level'])['mean_nightlight'].agg(['mean', 'std', 'count']).reset_index()
    quarterly['quarter_str'] = quarterly['quarter'].astype(str)
    
    # Save
    output_file = output_dir / 'nightlights_quarterly_for_budget.csv'
    quarterly.to_csv(output_file, index=False)
    print(f"\n✓ Saved quarterly aggregates: {output_file}")
    
    print("\nQuarterly summary (Primary impact areas):")
    primary = quarterly[quarterly['impact_level'] == 'primary']
    print(primary[['quarter_str', 'mean', 'std']].to_string(index=False))
    
    # Create template for budget data
    print("\n" + "="*80)
    print("BUDGET DATA TEMPLATE")
    print("="*80)
    print("""
To correlate UNDP budget with nightlight recovery, create a CSV file with:

Columns:
  - quarter: Quarter (e.g., '2023Q2', '2023Q3', '2024Q1')
  - impact_area: Impact level ('primary', 'moderate', 'secondary')
  - budget_disbursed: Amount in USD
  - intervention_type: Type of assistance (e.g., 'infrastructure', 'livelihoods')
  
Example:
  quarter,impact_area,budget_disbursed,intervention_type
  2023Q2,primary,500000,debris_clearance
  2023Q3,primary,1200000,infrastructure
  2023Q4,primary,800000,livelihoods
  2024Q1,primary,1500000,infrastructure
  ...

Save as: budget_disbursement_quarterly.csv

Then run script 4 to analyze budget correlation with nightlight recovery.
    """)
    
    # Create template file
    template = pd.DataFrame({
        'quarter': primary['quarter_str'].unique()[:4],
        'impact_area': 'primary',
        'budget_disbursed': [0] * 4,  # Fill in actual amounts
        'intervention_type': 'TBD'
    })
    
    template_file = output_dir / 'budget_template.csv'
    template.to_csv(template_file, index=False)
    print(f"\n✓ Created template: {template_file}")
    print("  → Fill in actual budget amounts from UNDP")


def create_summary_statistics(df_nightlights, output_dir):
    """
    Create summary statistics table.
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Pre/post Mocha comparison
    df = df_nightlights.copy()
    df['period'] = df['date'].apply(lambda x: 'Pre-Mocha' if x < MOCHA_DATE else 'Post-Mocha')
    
    summary = df.groupby(['impact_level', 'period'])['mean_nightlight'].agg(['mean', 'std', 'count'])
    print("\nMean nightlight by period and impact level:")
    print(summary)
    
    # Calculate % change
    print("\n% Change (Post vs Pre):")
    for impact in df['impact_level'].unique():
        pre = df[(df['impact_level'] == impact) & (df['period'] == 'Pre-Mocha')]['mean_nightlight'].mean()
        post = df[(df['impact_level'] == impact) & (df['period'] == 'Post-Mocha')]['mean_nightlight'].mean()
        
        if not np.isnan(pre) and not np.isnan(post) and pre > 0:
            pct_change = ((post - pre) / pre) * 100
            print(f"  {impact.upper()}: {pct_change:+.1f}%")
    
    # Save summary
    summary_file = output_dir / 'summary_statistics.csv'
    summary.to_csv(summary_file)
    print(f"\n✓ Saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Cyclone Mocha nightlight changes"
    )
    
    parser.add_argument(
        "--viirs-dir",
        type=str,
        default="./data/viirs",
        help="Directory with VIIRS .tif files"
    )
    
    parser.add_argument(
        "--boundaries-dir",
        type=str,
        default="./data/boundaries/processed",
        help="Directory with processed boundaries"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/mocha_analysis",
        help="Output directory for visualizations"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default="2021-01",
        help="Start date (YYYY-MM)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-10",
        help="End date (YYYY-MM)"
    )
    
    args = parser.parse_args()
    
    if not RASTER_OK:
        print("\n✗ Missing required packages")
        print("Install with: pip install rioxarray xarray rasterio")
        return
    
    print("="*80)
    print("  CYCLONE MOCHA NIGHTLIGHT VISUALIZATION")
    print("  Myanmar - May 14, 2023")
    print("="*80)
    
    # Find VIIRS files
    start = datetime.strptime(args.start_date, "%Y-%m")
    end = datetime.strptime(args.end_date, "%Y-%m")
    
    viirs_files = find_viirs_files(args.viirs_dir, start, end)
    
    if not viirs_files:
        print("\n✗ No VIIRS files found")
        return
    
    # Load boundaries
    print("\nLoading boundaries...")
    boundaries_dir = Path(args.boundaries_dir)
    
    # Try different boundary files
    boundary_files_to_try = [
        boundaries_dir / 'mmr_admin1.geojson',  # Real Myanmar admin1 file
        boundaries_dir / 'myanmar_admin1_mocha.geojson',  # Generated file
        boundaries_dir / 'myanmar_admin0.geojson',  # Country outline
    ]
    
    boundaries_gdf = None
    for boundaries_file in boundary_files_to_try:
        if boundaries_file.exists():
            boundaries_gdf = gpd.read_file(boundaries_file)
            print(f"✓ Loaded {len(boundaries_gdf)} regions from {boundaries_file.name}")
            
            # Add mocha_impact column if not present
            if 'mocha_impact' not in boundaries_gdf.columns:
                boundaries_gdf['mocha_impact'] = 'none'
                # Mark Rakhine as primary impact
                if 'name' in boundaries_gdf.columns:
                    rakhine_mask = boundaries_gdf['name'].str.contains('Rakhine', case=False, na=False)
                elif 'ST' in boundaries_gdf.columns:
                    rakhine_mask = boundaries_gdf['ST'].str.contains('Rakhine', case=False, na=False)
                elif 'ST_RG' in boundaries_gdf.columns:
                    rakhine_mask = boundaries_gdf['ST_RG'].str.contains('Rakhine', case=False, na=False)
                else:
                    # Try any column with name-like content
                    name_col = [c for c in boundaries_gdf.columns if 'name' in c.lower() or 'st' in c.lower()]
                    if name_col:
                        rakhine_mask = boundaries_gdf[name_col[0]].str.contains('Rakhine', case=False, na=False)
                    else:
                        rakhine_mask = pd.Series([False] * len(boundaries_gdf))
                
                boundaries_gdf.loc[rakhine_mask, 'mocha_impact'] = 'primary'
                print(f"  Identified {rakhine_mask.sum()} region(s) as Rakhine/primary impact")
            break
    
    if boundaries_gdf is None:
        print(f"\n✗ Boundaries file not found in {boundaries_dir}")
        print("Tried: mmr_admin1.geojson, myanmar_admin1_mocha.geojson, myanmar_admin0.geojson")
        print("Run: python create_myanmar_boundary.py")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    create_before_after_maps(viirs_files, boundaries_gdf, output_dir)
    
    df_nightlights = create_time_series_analysis(viirs_files, boundaries_gdf, output_dir)
    
    if df_nightlights is not None:
        create_summary_statistics(df_nightlights, output_dir)
        prepare_for_budget_analysis(df_nightlights, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  1. mocha_before_after_comparison.png - Visual comparison")
    print("  2. mocha_nightlight_time_series.png - Time series plot")
    print("  3. nightlight_time_series.csv - Raw data")
    print("  4. nightlights_quarterly_for_budget.csv - Quarterly aggregates")
    print("  5. budget_template.csv - Template for UNDP budget data")
    print("\nNext steps:")
    print("  1. Fill in budget_template.csv with actual UNDP disbursement data")
    print("  2. Run script 4 to correlate budget with recovery")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
