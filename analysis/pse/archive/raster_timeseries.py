#!/usr/bin/env python3
"""
analysis/pse/raster_timeseries.py
=================================
Extract time series from raster files and create visualizations.

This extracts zonal statistics from GeoTIFF files for each time period
and creates time series plots showing pixel-level satellite data over time.

Usage:
    python analysis/pse/raster_timeseries.py
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    from rasterstats import zonal_stats
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio/rasterstats not installed. Run: pip install rasterio rasterstats")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data/pse")
RASTER_DIR = DATA_DIR / "rasters"
OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARIES_FILE = DATA_DIR / "boundaries" / "gaza_admin2.geojson"

# UNDP Projects file
UNDP_PROJECTS_FILE = DATA_DIR / "external" / "undp_projects.csv"

# Key dates
CONFLICT_DATE = pd.Timestamp("2023-10-07")

# Dataset configurations
RASTER_DATASETS = {
    'viirs': {
        'name': 'Nighttime Lights (VIIRS)',
        'pattern': 'viirs_pse_*.tif',
        'line_color': 'darkblue',
        'unit': 'nW/cm²/sr',
        'fill_color': 'blue'
    },
    'ndvi': {
        'name': 'Vegetation Index (NDVI)',
        'pattern': 'ndvi_pse_*.tif',
        'line_color': 'forestgreen',
        'unit': 'Index',
        'fill_color': 'green'
    },
    'ndbi': {
        'name': 'Built-up Index (NDBI)',
        'pattern': 'ndbi_pse_*.tif',
        'line_color': 'brown',
        'unit': 'Index',
        'fill_color': 'sienna'
    },
    'bsi': {
        'name': 'Bare Soil Index (BSI)',
        'pattern': 'bsi_pse_*.tif',
        'line_color': 'orange',
        'unit': 'Index',
        'fill_color': 'darkorange'
    },
    'sar': {
        'name': 'SAR Backscatter (VV+VH)',
        'pattern': 'sar_pse_*.tif',
        'line_color': 'purple',
        'unit': 'dB',
        'fill_color': 'mediumpurple'
    },
    'sar_vh': {
        'name': 'SAR VH Backscatter',
        'pattern': 'sar_vh_pse_*.tif',
        'line_color': 'indigo',
        'unit': 'dB',
        'fill_color': 'slateblue'
    },
}

# Category colors for UNDP projects
CATEGORY_COLORS = {
    'Education': '#3498DB',
    'Debris & Solid Waste': '#795548',
    'Water & Sanitation (WASH)': '#00BCD4',
    'Energy & Power': '#FFC107',
    'Health & Emergency': '#F44336',
    'Infrastructure & Housing': '#9C27B0',
    'Livelihoods & Economy': '#4CAF50',
    'Governance & Rule of Law': '#607D8B',
    'Social Inclusion': '#E91E63',
    'Environment': '#8BC34A',
    'Other / Unspecified': '#9E9E9E'
}


def parse_date_from_filename(filename):
    """Extract date from raster filename like 'viirs_pse_202309.tif'"""
    match = re.search(r'(\d{6})\.tif$', filename)
    if match:
        date_str = match.group(1)
        try:
            return datetime.strptime(date_str, '%Y%m')
        except:
            pass
    return None


def load_undp_projects():
    """Load UNDP projects data for timeline annotations."""
    if UNDP_PROJECTS_FILE.exists():
        df = pd.read_csv(UNDP_PROJECTS_FILE)
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        if 'is_likely_umbrella' in df.columns:
            df = df[df['is_likely_umbrella'] != True]
        return df
    return None


def get_undp_markers(undp_df, min_date, max_date):
    """Get UNDP project markers for a date range."""
    if undp_df is None:
        return []
    
    df = undp_df[
        (undp_df['start_date'] >= min_date) & 
        (undp_df['start_date'] <= max_date)
    ].copy()
    
    if df.empty:
        return []
    
    markers = []
    for _, row in df.iterrows():
        markers.append((
            row['start_date'],
            row['ml_category'],
            str(row['title'])[:30]
        ))
    
    return sorted(markers, key=lambda x: x[0])


def extract_raster_timeseries(gdf_gaza, gdf_westbank=None):
    """
    Extract time series from all raster files.
    
    Returns:
        DataFrame with columns: dataset, date, gaza_mean, gaza_std, westbank_mean, westbank_std
    """
    if not HAS_RASTERIO:
        print("ERROR: rasterio not installed")
        return None
    
    if not RASTER_DIR.exists():
        print(f"ERROR: Raster directory not found: {RASTER_DIR}")
        return None
    
    results = []
    
    for dataset, cfg in RASTER_DATASETS.items():
        pattern = RASTER_DIR / cfg['pattern']
        files = sorted(glob.glob(str(pattern)))
        
        if not files:
            print(f"  ⚠ No {dataset} rasters found: {pattern}")
            continue
        
        print(f"  Processing {dataset}: {len(files)} files")
        
        for raster_path in files:
            date = parse_date_from_filename(raster_path)
            if date is None:
                continue
            
            try:
                # Extract Gaza statistics
                gaza_stats = zonal_stats(
                    gdf_gaza, 
                    raster_path, 
                    stats=['mean', 'std', 'count'],
                    nodata=np.nan
                )
                
                gaza_means = [s['mean'] for s in gaza_stats if s['mean'] is not None]
                gaza_mean = np.mean(gaza_means) if gaza_means else np.nan
                gaza_std = np.std(gaza_means) if len(gaza_means) > 1 else 0
                
                # Extract West Bank statistics (if available)
                wb_mean = np.nan
                wb_std = 0
                if gdf_westbank is not None and len(gdf_westbank) > 0:
                    wb_stats = zonal_stats(
                        gdf_westbank,
                        raster_path,
                        stats=['mean', 'std', 'count'],
                        nodata=np.nan
                    )
                    wb_means = [s['mean'] for s in wb_stats if s['mean'] is not None]
                    wb_mean = np.mean(wb_means) if wb_means else np.nan
                    wb_std = np.std(wb_means) if len(wb_means) > 1 else 0
                
                results.append({
                    'dataset': dataset,
                    'date': date,
                    'gaza_mean': gaza_mean,
                    'gaza_std': gaza_std,
                    'westbank_mean': wb_mean,
                    'westbank_std': wb_std
                })
                
            except Exception as e:
                print(f"    Error processing {raster_path}: {e}")
    
    if not results:
        return None
    
    df = pd.DataFrame(results)
    df['date'] = pd.to_datetime(df['date'])
    return df


def create_raster_timeseries_plot(ts_df, dataset, cfg, undp_df):
    """Create a time series plot for a single raster dataset."""
    
    data = ts_df[ts_df['dataset'] == dataset].sort_values('date')
    
    if len(data) < 2:
        return False
    
    # Create figure with UNDP strip
    fig = plt.figure(figsize=(16, 9))
    ax_main = fig.add_axes([0.08, 0.22, 0.85, 0.68])
    ax_strip = fig.add_axes([0.08, 0.12, 0.85, 0.06])
    
    # Plot Gaza
    ax_main.fill_between(data['date'], data['gaza_mean'], 
                         alpha=0.3, color=cfg['fill_color'])
    ax_main.plot(data['date'], data['gaza_mean'], 
                 color=cfg['line_color'], linewidth=2.5, marker='o', markersize=5,
                 label='Gaza Strip')
    
    # Plot West Bank if available
    if data['westbank_mean'].notna().any():
        ax_main.plot(data['date'], data['westbank_mean'], 
                     color='gray', linewidth=2, linestyle='--', alpha=0.7,
                     label='West Bank (Control)')
    
    # Pre/post means
    pre_data = data[data['date'] < CONFLICT_DATE]['gaza_mean']
    post_data = data[data['date'] >= CONFLICT_DATE]['gaza_mean']
    
    pre_mean = pre_data.mean() if len(pre_data) > 0 else np.nan
    post_mean = post_data.mean() if len(post_data) > 0 else np.nan
    
    if not pd.isna(pre_mean):
        ax_main.axhline(y=pre_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    if not pd.isna(post_mean):
        ax_main.axhline(y=post_mean, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Conflict line
    ax_main.axvline(x=CONFLICT_DATE, color='red', linestyle='--', linewidth=2.5)
    
    # Shade periods
    xmin = data['date'].min()
    xmax = data['date'].max()
    
    if xmin < CONFLICT_DATE:
        ax_main.axvspan(xmin, min(CONFLICT_DATE, xmax), alpha=0.1, color='green')
    if xmax > CONFLICT_DATE:
        ax_main.axvspan(max(CONFLICT_DATE, xmin), xmax, alpha=0.08, color='red')
    
    # Conflict label
    y_top = ax_main.get_ylim()[1]
    ax_main.text(CONFLICT_DATE, y_top * 0.95, 'Conflict\nOct 7, 2023', 
                 ha='center', va='top', fontsize=9, fontweight='bold', color='red',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
    
    # Calculate % change
    if pre_mean != 0 and not pd.isna(pre_mean) and not pd.isna(post_mean):
        pct_change = ((post_mean - pre_mean) / abs(pre_mean)) * 100
        title_suffix = f" ({pct_change:+.1f}% After Conflict)"
    else:
        title_suffix = ""
    
    ax_main.set_title(f'Gaza: {cfg["name"]} (Raster){title_suffix}', fontsize=14, fontweight='bold', pad=20)
    ax_main.set_ylabel(f'{cfg["name"]} ({cfg["unit"]})', fontsize=11)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_main.set_xticklabels([])
    ax_main.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color=cfg['line_color'], linewidth=2.5, marker='o', label='Gaza Strip'),
    ]
    if data['westbank_mean'].notna().any():
        legend_elements.append(Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='West Bank'))
    if not pd.isna(pre_mean):
        legend_elements.append(Line2D([0], [0], color='green', linestyle=':', linewidth=1.5, label=f'Pre-Conflict Mean: {pre_mean:.2f}'))
    if not pd.isna(post_mean):
        legend_elements.append(Line2D([0], [0], color='red', linestyle=':', linewidth=1.5, label=f'Post-Conflict Mean: {post_mean:.2f}'))
    
    ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # === UNDP PROJECT STRIP ===
    ax_strip.set_xlim(ax_main.get_xlim())
    ax_strip.set_ylim(0, 1)
    
    if xmin < CONFLICT_DATE:
        ax_strip.axvspan(xmin, min(CONFLICT_DATE, xmax), alpha=0.15, color='green')
    if xmax > CONFLICT_DATE:
        ax_strip.axvspan(max(CONFLICT_DATE, xmin), xmax, alpha=0.1, color='red')
    
    ax_strip.axvline(x=CONFLICT_DATE, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Plot UNDP projects
    categories_shown = set()
    if undp_df is not None:
        markers = get_undp_markers(undp_df, xmin, xmax)
        
        if markers:
            from collections import defaultdict
            monthly_markers = defaultdict(list)
            for date, category, title in markers:
                month_key = date.to_period('M').to_timestamp()
                monthly_markers[month_key].append((category, title))
            
            for month, projects in sorted(monthly_markers.items()):
                for i, (category, title) in enumerate(projects[:3]):
                    color = CATEGORY_COLORS.get(category, '#9E9E9E')
                    y_pos = 0.5 + (i - len(projects)/2 + 0.5) * 0.25
                    y_pos = max(0.15, min(0.85, y_pos))
                    
                    ax_strip.scatter([month], [y_pos], color=color, s=80, 
                                    marker='s', edgecolor='black', linewidth=0.5, zorder=5)
                    categories_shown.add(category)
    
    ax_strip.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_strip.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_strip.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_strip.set_yticks([])
    ax_strip.set_ylabel('UNDP', fontsize=9, rotation=0, ha='right', va='center')
    ax_strip.spines['top'].set_visible(False)
    ax_strip.spines['right'].set_visible(False)
    ax_strip.spines['left'].set_visible(False)
    
    # Category legend at bottom
    if categories_shown:
        legend_patches = [mpatches.Patch(color=CATEGORY_COLORS.get(cat, '#9E9E9E'), label=cat[:20]) 
                          for cat in sorted(categories_shown)]
        fig.legend(handles=legend_patches, loc='lower center', 
                   bbox_to_anchor=(0.5, -0.02), ncol=min(5, len(legend_patches)), 
                   fontsize=7, frameon=False)
    
    plt.savefig(FIGURES_DIR / f'ts_raster_{dataset}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def main():
    print("=" * 60)
    print("RASTER TIME SERIES EXTRACTION")
    print("=" * 60)
    
    if not HAS_RASTERIO:
        print("\nERROR: Install rasterio first:")
        print("  pip install rasterio rasterstats")
        return
    
    # Load boundaries
    if not BOUNDARIES_FILE.exists():
        print(f"\nERROR: Boundaries file not found: {BOUNDARIES_FILE}")
        return
    
    print(f"\nLoading boundaries: {BOUNDARIES_FILE}")
    gdf = gpd.read_file(BOUNDARIES_FILE)
    
    # Identify columns
    id_col = 'adm2_pcode' if 'adm2_pcode' in gdf.columns else gdf.columns[gdf.columns.str.contains('pcode', case=False)][0]
    
    # Separate Gaza and West Bank
    gdf_gaza = gdf[gdf[id_col].str.startswith('PS02', na=False)].copy()
    gdf_westbank = gdf[gdf[id_col].str.startswith('PS01', na=False)].copy()
    
    print(f"  Gaza districts: {len(gdf_gaza)}")
    print(f"  West Bank districts: {len(gdf_westbank)}")
    
    # Check raster directory
    if not RASTER_DIR.exists():
        print(f"\nERROR: Raster directory not found: {RASTER_DIR}")
        print("  Download rasters first: python -m pipeline.raster_export --config config/gaza.yaml")
        return
    
    raster_files = list(RASTER_DIR.glob("*.tif"))
    print(f"\nRaster directory: {RASTER_DIR}")
    print(f"  Found {len(raster_files)} GeoTIFF files")
    
    if len(raster_files) == 0:
        print("\nNo raster files found. Download them first:")
        print("  python -m pipeline.raster_export --config config/gaza.yaml")
        return
    
    # Load UNDP projects
    undp_df = load_undp_projects()
    if undp_df is not None:
        print(f"\n  UNDP projects loaded: {len(undp_df)}")
    
    # Extract time series
    print("\n--- Extracting Zonal Statistics ---")
    ts_df = extract_raster_timeseries(gdf_gaza, gdf_westbank)
    
    if ts_df is None or len(ts_df) == 0:
        print("\nNo data extracted from rasters")
        return
    
    # Save raw data
    ts_df.to_csv(OUTPUT_DIR / 'raster_timeseries.csv', index=False)
    print(f"\n  Saved: {OUTPUT_DIR / 'raster_timeseries.csv'}")
    
    # Create plots
    print("\n--- Creating Time Series Plots ---")
    
    for dataset, cfg in RASTER_DATASETS.items():
        if dataset not in ts_df['dataset'].values:
            print(f"  ✗ ts_raster_{dataset}.png (no data)")
            continue
        
        success = create_raster_timeseries_plot(ts_df, dataset, cfg, undp_df)
        if success:
            print(f"  ✓ ts_raster_{dataset}.png")
        else:
            print(f"  ✗ ts_raster_{dataset}.png (insufficient data)")
    
    # Summary table
    print("\n--- Summary ---")
    summary = ts_df.groupby('dataset').agg({
        'date': ['min', 'max', 'count'],
        'gaza_mean': ['mean', 'std']
    }).round(3)
    print(summary)
    
    print(f"\n{'='*60}")
    print(f"✓ Raster time series complete")
    print(f"  Output: {FIGURES_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
