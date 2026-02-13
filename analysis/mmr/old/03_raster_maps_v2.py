#!/usr/bin/env python3
"""
analysis/mmr/03_raster_maps_v2.py
=================================
Myanmar three-panel raster maps with:
- Full country view with national boundary
- BLACK TO YELLOW colormap for nightlights
- UN BLUE color for UNDP sites (#009EDB)
- Project categories from undp_projects.csv
- Better spacing and centering

Usage:
    python analysis/mmr/03_raster_maps_v2.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import glob
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("⚠ rasterio not installed. Run: pip install rasterio --break-system-packages")

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data/mmr")
RASTER_DIR = DATA_DIR / "rasters"
OUTPUT_DIR = Path("outputs/mmr")
FIGURES_DIR = OUTPUT_DIR / "figures" / "raster_maps"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Key dates
EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")
UNDP_START_DATE = pd.Timestamp("2025-05-01")  # From old analysis script

# Files
BOUNDARIES_FILE = DATA_DIR / "boundaries" / "mmr_admin3.geojson"
UNDP_PROJECTS_FILE = DATA_DIR / "external" / "undp_projects.csv"
UNDP_SITES_FILE = DATA_DIR / "external" / "undp_sites.csv"

# UN BLUE color for UNDP sites
UN_BLUE = '#009EDB'  # Official UN blue

# BLACK TO YELLOW colormap for nightlights
NIGHTLIGHTS_CMAP = LinearSegmentedColormap.from_list('nightlights_black_yellow', [
    '#000000',  # Black
    '#1a0a00',  # Very dark
    '#3d1a00',  # Dark brown
    '#7b3d00',  # Brown
    '#c46600',  # Orange-brown
    '#f09000',  # Orange
    '#ffc000',  # Gold
    '#ffe066',  # Light gold
    '#ffff00',  # Yellow
])

# ML Category colors (from UNDP projects classification)
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

# Colormaps for other indicators
COLORMAPS = {
    'viirs': NIGHTLIGHTS_CMAP,  # Black to yellow
    'ndvi': 'RdYlGn',
    'ndbi': 'RdBu_r',
    'bsi': 'YlOrBr',
    'sar_vh': 'viridis',
    'sar_vv': 'viridis',
}


def load_boundaries():
    """Load Myanmar national boundary."""
    for path in [BOUNDARIES_FILE,
                 DATA_DIR / "myanmar_boundaries" / "mmr_admin3.geojson"]:
        if path.exists():
            gdf = gpd.read_file(path).to_crs("EPSG:4326")
            national = gdf.dissolve()
            return gdf, national
    return None, None


def load_undp_projects():
    """Load UNDP projects with ML categories and dates."""
    if not UNDP_PROJECTS_FILE.exists():
        return None
    
    df = pd.read_csv(UNDP_PROJECTS_FILE)
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    
    if 'is_likely_umbrella' in df.columns:
        df = df[df['is_likely_umbrella'] != True]
    
    return df


def load_undp_sites():
    """Load UNDP site coordinates."""
    if not UNDP_SITES_FILE.exists():
        return []
    
    df = pd.read_csv(UNDP_SITES_FILE)
    
    sites = []
    for _, row in df.iterrows():
        lat = row.get('latitude', row.get('lat'))
        lon = row.get('longitude', row.get('lon'))
        if pd.notna(lat) and pd.notna(lon):
            sites.append({
                'lat': lat,
                'lon': lon,
                'site_id': row.get('site_id', ''),
            })
    
    return sites


def get_project_categories(undp_projects):
    """Get project category distribution."""
    if undp_projects is None or len(undp_projects) == 0:
        return {}, 'Other / Unspecified'
    
    # Get post-earthquake projects
    post_eq = undp_projects[undp_projects['start_date'] >= EARTHQUAKE_DATE]
    
    if len(post_eq) == 0:
        post_eq = undp_projects
    
    if 'ml_category' in post_eq.columns:
        category_counts = post_eq['ml_category'].value_counts()
        return category_counts.to_dict(), category_counts.index[0] if len(category_counts) > 0 else 'Other / Unspecified'
    
    return {}, 'Other / Unspecified'


def find_raster_files():
    """Find all raster files."""
    rasters = {}
    
    if not RASTER_DIR.exists():
        return rasters
    
    for f in sorted(RASTER_DIR.glob("*.tif")):
        name = f.stem
        parts = name.split('_')
        
        if len(parts) >= 3:
            indicator = parts[0]
            if indicator == 'sar':
                indicator = f"{parts[0]}_{parts[1]}" if len(parts) >= 4 else parts[0]
                date_str = parts[-1]
            else:
                date_str = parts[-1]
            
            if len(date_str) == 6:
                try:
                    date = datetime(int(date_str[:4]), int(date_str[4:6]), 1)
                    if indicator not in rasters:
                        rasters[indicator] = {}
                    rasters[indicator][date] = f
                except:
                    pass
    
    return rasters


def load_raster(filepath):
    """Load a raster file."""
    with rasterio.open(filepath) as src:
        data = src.read(1)
        bounds = src.bounds
    
    return {
        'data': data,
        'bounds': bounds,
        'extent': [bounds.left, bounds.right, bounds.bottom, bounds.top],
    }


def select_comparison_dates(rasters_dict, indicator):
    """Select three dates for comparison."""
    if indicator not in rasters_dict:
        return None
    
    available = sorted(rasters_dict[indicator].keys())
    
    if len(available) < 2:
        return None
    
    post_dates = [d for d in available if pd.Timestamp(d) >= EARTHQUAKE_DATE]
    pre_dates = [d for d in available if pd.Timestamp(d) < EARTHQUAKE_DATE]
    
    if not post_dates:
        if len(available) >= 3:
            return [(available[0], rasters_dict[indicator][available[0]]),
                    (available[1], rasters_dict[indicator][available[1]]),
                    (available[-1], rasters_dict[indicator][available[-1]])]
        elif len(available) >= 2:
            return [(available[0], rasters_dict[indicator][available[0]]),
                    (available[0], rasters_dict[indicator][available[0]]),
                    (available[-1], rasters_dict[indicator][available[-1]])]
    
    post_undp = post_dates[-1] if post_dates else available[-1]
    post_disaster = post_dates[0] if post_dates else available[-1]
    
    target_month = post_undp.month
    pre_candidates = [d for d in pre_dates if d.month == target_month]
    pre_disaster = pre_candidates[-1] if pre_candidates else (pre_dates[-1] if pre_dates else available[0])
    
    return [
        (pre_disaster, rasters_dict[indicator][pre_disaster]),
        (post_disaster, rasters_dict[indicator][post_disaster]),
        (post_undp, rasters_dict[indicator][post_undp]),
    ]


def create_three_panel_map(indicator, rasters_dict, sites, undp_projects, gdf, national, 
                          output_path, show_markers=True):
    """
    Create three-panel map with:
    - Full country view
    - BLACK TO YELLOW colormap for nightlights
    - UN BLUE for UNDP sites
    """
    
    selected = select_comparison_dates(rasters_dict, indicator)
    
    if selected is None:
        print(f"  ⚠ Insufficient data for {indicator}")
        return False
    
    # Load rasters
    raster_data = [load_raster(path) for date, path in selected]
    
    # Get project categories
    category_counts, _ = get_project_categories(undp_projects)
    
    # Set up figure with better spacing
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 3, wspace=0.12, left=0.05, right=0.95, top=0.88, bottom=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    
    # Get colormap - BLACK TO YELLOW for nightlights
    if indicator == 'viirs':
        cmap = NIGHTLIGHTS_CMAP
    else:
        cmap = COLORMAPS.get(indicator, 'viridis')
    
    # Compute common color scale
    all_vals = []
    for r in raster_data:
        all_vals.extend(r['data'].flatten())
    all_vals = np.array(all_vals)
    all_vals = all_vals[~np.isnan(all_vals)]
    
    if indicator == 'viirs':
        vmin = 0
        vmax = np.percentile(all_vals, 95)
    else:
        vmin = np.percentile(all_vals, 2)
        vmax = np.percentile(all_vals, 98)
    
    # Get FULL country bounds
    if national is not None:
        bounds = national.total_bounds
        pad_x = (bounds[2] - bounds[0]) * 0.02
        pad_y = (bounds[3] - bounds[1]) * 0.02
        country_extent = [bounds[0] - pad_x, bounds[2] + pad_x, bounds[1] - pad_y, bounds[3] + pad_y]
    else:
        country_extent = raster_data[0]['extent']
    
    titles = [
        ('Pre-Earthquake', 'forestgreen'),
        ('Post-Earthquake', 'crimson'),
        ('Post-UNDP Recovery', 'royalblue'),
    ]
    
    indicator_display = {
        'viirs': 'Nightlights (VIIRS)',
        'ndvi': 'Vegetation (NDVI)',
        'ndbi': 'Built-up (NDBI)',
        'bsi': 'Bare Soil (BSI)',
        'sar_vh': 'SAR VH',
        'sar_vv': 'SAR VV',
    }.get(indicator, indicator.upper())
    
    for i, (ax, (date, _), data, (title, color)) in enumerate(zip(axes, selected, raster_data, titles)):
        # Plot raster
        im = ax.imshow(data['data'], extent=data['extent'],
                       cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
        
        # Set to FULL country extent
        ax.set_xlim(country_extent[0], country_extent[1])
        ax.set_ylim(country_extent[2], country_extent[3])
        
        # Add national boundary (white line)
        if national is not None:
            national.boundary.plot(ax=ax, color='white', linewidth=1.5, alpha=0.9)
        
        # Add admin boundaries (thin)
        if gdf is not None:
            gdf.boundary.plot(ax=ax, color='white', linewidth=0.2, alpha=0.3)
        
        # Add earthquake epicenter (yellow star)
        ax.scatter([95.925], [22.001], marker='*', s=300, c='yellow',
                  edgecolor='black', linewidth=1.5, zorder=15)
        
        # Only add UNDP sites to the LAST panel (in UN BLUE)
        if i == 2 and show_markers and sites:
            lons = [s['lon'] for s in sites if s.get('lon') is not None]
            lats = [s['lat'] for s in sites if s.get('lat') is not None]
            
            if lons and lats:
                ax.scatter(lons, lats, marker='o', c=UN_BLUE,
                          s=12, edgecolors='white', linewidths=0.3, 
                          zorder=10, alpha=0.8,
                          label=f'UNDP Sites ({len(lons)})')
        
        ax.set_title(f"{date.strftime('%Y-%m')}\n({title})",
                    fontsize=14, fontweight='bold', color=color)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)
    
    # Colorbar at bottom center
    cbar_ax = fig.add_axes([0.25, 0.10, 0.50, 0.025])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'{indicator_display}', fontsize=12)
    
    # Legend
    if show_markers and sites:
        legend_handles = []
        
        # UNDP Sites (UN BLUE)
        site_handle = mlines.Line2D([], [], marker='o', color=UN_BLUE,
                                   markersize=10, linestyle='None',
                                   markeredgecolor='white', markeredgewidth=0.5,
                                   label=f'UNDP Sites ({len(sites)})')
        legend_handles.append(site_handle)
        
        # Project categories
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:4]:
            color = CATEGORY_COLORS.get(cat, '#9E9E9E')
            handle = mlines.Line2D([], [], marker='s', color=color,
                                  markersize=8, linestyle='None',
                                  markeredgecolor='black', markeredgewidth=0.5,
                                  label=f'{cat} ({count})')
            legend_handles.append(handle)
        
        # Epicenter
        epicenter_handle = mlines.Line2D([], [], marker='*', color='yellow',
                                        markersize=15, linestyle='None',
                                        markeredgecolor='black', markeredgewidth=1,
                                        label='Epicenter (M7.7)')
        legend_handles.append(epicenter_handle)
        
        if legend_handles:
            fig.legend(handles=legend_handles, loc='lower right',
                      bbox_to_anchor=(0.95, 0.02), fontsize=9,
                      title='UNDP Response',
                      framealpha=0.95, ncol=1)
    
    # Main title
    n_sites = len(sites) if sites else 0
    title_text = f'Myanmar Earthquake (M7.7): {indicator_display}'
    if show_markers and n_sites > 0:
        title_text += f' with {n_sites} UNDP Sites'
    
    fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.96)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def main():
    print("=" * 70)
    print("MYANMAR EARTHQUAKE: RASTER MAPS (V2)")
    print("  - Black to Yellow colormap for nightlights")
    print("  - UN Blue for UNDP sites")
    print("  - Full country view")
    print("=" * 70)
    
    if not HAS_RASTERIO:
        print("\n⚠ rasterio required")
        return
    
    # Load data
    print("\n--- Loading Data ---")
    gdf, national = load_boundaries()
    if gdf is not None:
        print(f"  Admin units: {len(gdf)}")
    
    undp_projects = load_undp_projects()
    if undp_projects is not None:
        print(f"  UNDP projects: {len(undp_projects)}")
    
    sites = load_undp_sites()
    print(f"  UNDP sites: {len(sites)}")
    
    # Find rasters
    print("\n--- Finding Raster Files ---")
    rasters = find_raster_files()
    
    if not rasters:
        print("  ⚠ No raster files found!")
        return
    
    for indicator, dates in rasters.items():
        print(f"  {indicator}: {len(dates)} files ({min(dates).strftime('%Y-%m')} to {max(dates).strftime('%Y-%m')})")
    
    # Create maps
    print("\n--- Creating Maps ---")
    
    indicators_to_map = ['viirs', 'ndvi', 'ndbi', 'sar_vh', 'bsi']
    total_maps = 0
    
    for indicator in indicators_to_map:
        if indicator not in rasters:
            print(f"\n  ⚠ {indicator} not found, skipping...")
            continue
        
        print(f"\n  Processing {indicator.upper()}...")
        
        # With UNDP markers
        success = create_three_panel_map(
            indicator, rasters, sites, undp_projects, gdf, national,
            FIGURES_DIR / f'{indicator}_three_panel_with_undp.png',
            show_markers=True
        )
        if success:
            print(f"    ✓ {indicator}_three_panel_with_undp.png")
            total_maps += 1
        
        # Clean
        success = create_three_panel_map(
            indicator, rasters, sites, undp_projects, gdf, national,
            FIGURES_DIR / f'{indicator}_three_panel_clean.png',
            show_markers=False
        )
        if success:
            print(f"    ✓ {indicator}_three_panel_clean.png")
            total_maps += 1
    
    print(f"\n{'='*70}")
    print(f"✓ COMPLETE: {total_maps} maps created")
    print(f"  Output: {FIGURES_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
