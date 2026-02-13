#!/usr/bin/env python3
"""
analysis/mmr/03_raster_maps.py
==============================
Create three-panel raster maps showing UNDP intervention locations.

Creates side-by-side comparisons:
- Left: Pre-disaster period (same calendar month, previous year)
- Middle: Post-disaster period (before UNDP arrives)
- Right: Post-UNDP period (with UNDP markers showing where projects are)

Maps use the SAME calendar month across years for comparability.

Usage:
    python analysis/mmr/03_raster_maps.py
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
    from rasterio.plot import show
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("⚠ rasterio not installed. Run: pip install rasterio --break-system-packages")

try:
    import geopandas as gpd
    from shapely.geometry import Point, box
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

# Key dates for Myanmar earthquake
EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")
INTERVENTION_START = pd.Timestamp("2025-04-01")

# Affected region bounds (Sagaing/Mandalay area)
AFFECTED_BOUNDS = [94.5, 20.5, 97.5, 23.5]  # [minx, miny, maxx, maxy]

# Boundaries file
BOUNDARIES_FILE = DATA_DIR / "boundaries" / "mmr_admin3.geojson"

# UNDP Sites
UNDP_SITES_CSV = DATA_DIR / "external" / "undp_sites.csv"
UNDP_PROJECTS_CSV = DATA_DIR / "external" / "undp_projects.csv"

# Color maps for different indicators
COLORMAPS = {
    'viirs': 'magma',       # Dark to bright (nightlights)
    'ndvi': 'RdYlGn',       # Red to Green (vegetation)
    'ndbi': 'RdBu_r',       # Blue to Red (built-up)
    'bsi': 'YlOrBr',        # Yellow to Brown (bare soil)
    'sar_vh': 'viridis',    # SAR backscatter
    'sar_vv': 'viridis',
}

# Intervention type colors
TYPE_COLORS = {
    'community leader': ('#00FF00', 'o', 'Community Leader'),
    'community action board': ('#00BFFF', 's', 'Community Board'),
    'women community leader': ('#FF69B4', '^', 'Women Leader'),
    'youth': ('#9370DB', 'D', 'Youth'),
    'monastic': ('#FFA500', 'p', 'Monastic'),
    'other': ('#808080', 'o', 'Other'),
}


def load_undp_sites():
    """Load UNDP intervention sites from CSV."""
    sites = []
    
    # Try sites CSV first
    if UNDP_SITES_CSV.exists():
        df = pd.read_csv(UNDP_SITES_CSV)
        print(f"  Loaded {len(df)} UNDP sites from {UNDP_SITES_CSV.name}")
        
        for _, row in df.iterrows():
            sites.append({
                'lat': row.get('latitude', row.get('lat')),
                'lon': row.get('longitude', row.get('lon')),
                'type': str(row.get('intervention_type', 'other')).lower(),
                'site_id': row.get('site_id', ''),
                'location': row.get('location_name', ''),
            })
    
    return sites


def load_undp_projects():
    """Load UNDP projects with dates and categories."""
    if not UNDP_PROJECTS_CSV.exists():
        return None
    
    df = pd.read_csv(UNDP_PROJECTS_CSV)
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    
    # Filter to post-earthquake projects
    df = df[df['start_date'] >= EARTHQUAKE_DATE]
    
    print(f"  Loaded {len(df)} post-earthquake UNDP projects")
    return df


def load_boundaries():
    """Load country boundaries."""
    for path in [BOUNDARIES_FILE,
                 DATA_DIR / "myanmar_boundaries" / "mmr_admin3.geojson"]:
        if path.exists():
            return gpd.read_file(path).to_crs("EPSG:4326")
    return None


def find_raster_files():
    """Find all raster files grouped by indicator and date."""
    rasters = {}
    
    if not RASTER_DIR.exists():
        print(f"⚠ Raster directory not found: {RASTER_DIR}")
        return rasters
    
    for f in sorted(RASTER_DIR.glob("*.tif")):
        name = f.stem
        
        # Parse indicator and date from filename like "viirs_mmr_202411"
        parts = name.split('_')
        if len(parts) >= 3:
            indicator = parts[0]
            if indicator == 'sar':
                # Handle sar_vh_mmr_YYYYMM
                if len(parts) >= 4:
                    indicator = f"{parts[0]}_{parts[1]}"
                    date_str = parts[-1]
                else:
                    date_str = parts[-1]
            else:
                date_str = parts[-1]
            
            # Parse date
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
    """Load a raster file and return data + metadata."""
    with rasterio.open(filepath) as src:
        data = src.read(1)
        bounds = src.bounds
        transform = src.transform
        crs = src.crs
    
    return {
        'data': data,
        'bounds': bounds,
        'transform': transform,
        'crs': crs,
        'extent': [bounds.left, bounds.right, bounds.bottom, bounds.top],
    }


def select_comparison_dates(rasters_dict, indicator):
    """
    Select three dates for comparison:
    1. Pre-disaster: Same calendar month, 1 year before
    2. Post-disaster: First available month after earthquake
    3. Post-UNDP: Most recent available month
    
    Uses same calendar month for pre/post comparison when possible.
    """
    if indicator not in rasters_dict:
        return None
    
    available = sorted(rasters_dict[indicator].keys())
    
    if len(available) < 2:
        return None
    
    # Find post-earthquake dates
    post_dates = [d for d in available if pd.Timestamp(d) >= EARTHQUAKE_DATE]
    pre_dates = [d for d in available if pd.Timestamp(d) < EARTHQUAKE_DATE]
    
    if not post_dates or not pre_dates:
        return None
    
    # Get most recent post date
    post_undp = post_dates[-1]
    
    # Get first post-earthquake date (before much UNDP activity)
    post_disaster = post_dates[0]
    
    # Find pre-disaster date - same calendar month as post_undp, but before earthquake
    target_month = post_undp.month
    pre_candidates = [d for d in pre_dates if d.month == target_month]
    
    if pre_candidates:
        pre_disaster = pre_candidates[-1]  # Most recent pre-disaster with same month
    else:
        pre_disaster = pre_dates[-1]  # Just use most recent pre-disaster
    
    return [
        (pre_disaster, rasters_dict[indicator][pre_disaster]),
        (post_disaster, rasters_dict[indicator][post_disaster]),
        (post_undp, rasters_dict[indicator][post_undp]),
    ]


def plot_intervention_markers(ax, sites, show_legend=True):
    """Add intervention site markers to a plot."""
    
    legend_handles = []
    plotted_types = set()
    
    for site in sites:
        site_type = site.get('type', 'other').lower()
        
        # Match to known types
        matched_type = 'other'
        for known_type in TYPE_COLORS.keys():
            if known_type in site_type:
                matched_type = known_type
                break
        
        color, marker, label = TYPE_COLORS[matched_type]
        
        ax.scatter(site['lon'], site['lat'], marker=marker, c=color,
                  s=50, edgecolors='black', linewidths=0.5, zorder=10, alpha=0.8)
        
        plotted_types.add(matched_type)
    
    # Create legend handles
    for t in sorted(plotted_types):
        color, marker, label = TYPE_COLORS[t]
        handle = mlines.Line2D([], [], marker=marker, color=color,
                              markersize=8, linestyle='None',
                              markeredgecolor='black', markeredgewidth=0.5,
                              label=label)
        legend_handles.append(handle)
    
    return legend_handles


def create_three_panel_map(indicator, rasters_dict, sites, output_path, 
                          show_markers=True, zoom_bounds=None):
    """
    Create three-panel map: Pre-Disaster | Post-Disaster | Post-UNDP
    
    Parameters:
    - indicator: 'viirs', 'ndvi', etc.
    - rasters_dict: Dictionary of {indicator: {date: filepath}}
    - sites: List of UNDP site dictionaries
    - output_path: Where to save the figure
    - show_markers: Whether to show UNDP markers on third panel
    - zoom_bounds: [minx, miny, maxx, maxy] or None for auto
    """
    
    selected = select_comparison_dates(rasters_dict, indicator)
    
    if selected is None:
        print(f"  ⚠ Insufficient data for {indicator}")
        return
    
    # Load rasters
    raster_data = [load_raster(path) for date, path in selected]
    
    # Load boundaries for overlay
    gdf = load_boundaries()
    
    # Set up figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    fig.subplots_adjust(wspace=0.05, top=0.88, bottom=0.15)
    
    # Get colormap
    if indicator == 'viirs':
        cmap = LinearSegmentedColormap.from_list('nightlights', [
            '#000000', '#1a0a2e', '#3d1a5c', '#7b2d8e', '#c44e52', 
            '#f07c3e', '#ffc93c', '#fff9ae', '#ffffff'
        ])
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
    
    # Determine zoom bounds
    if zoom_bounds is None:
        zoom_bounds = AFFECTED_BOUNDS
    
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
        im = ax.imshow(data['data'], extent=data['extent'],
                       cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
        
        # Zoom to affected area
        ax.set_xlim(zoom_bounds[0], zoom_bounds[2])
        ax.set_ylim(zoom_bounds[1], zoom_bounds[3])
        
        # Add boundary overlay
        if gdf is not None:
            gdf.boundary.plot(ax=ax, color='white', linewidth=0.5, alpha=0.6)
        
        # Add earthquake epicenter
        ax.scatter([95.925], [22.001], marker='*', s=200, c='yellow',
                  edgecolor='black', linewidth=1, zorder=15, label='Epicenter')
        
        # Only add UNDP markers to the LAST panel if show_markers is True
        if i == 2 and show_markers and sites:
            legend_handles = plot_intervention_markers(ax, sites, show_legend=False)
        
        ax.set_title(f"{date.strftime('%Y-%m')}\n({title})",
                    fontsize=14, fontweight='bold', color=color)
        ax.set_xlabel('Longitude', fontsize=11)
        if i == 0:
            ax.set_ylabel('Latitude', fontsize=11)
        else:
            ax.set_yticklabels([])
    
    # Colorbar
    cbar_ax = fig.add_axes([0.15, 0.08, 0.55, 0.025])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'{indicator_display}', fontsize=12)
    
    # Legend for UNDP markers
    if show_markers and sites:
        # Count sites by type
        type_counts = {}
        for site in sites:
            t = site.get('type', 'other').lower()
            for known in TYPE_COLORS.keys():
                if known in t:
                    t = known
                    break
            type_counts[t] = type_counts.get(t, 0) + 1
        
        legend_handles = []
        for t, count in sorted(type_counts.items()):
            color, marker, label = TYPE_COLORS.get(t, TYPE_COLORS['other'])
            handle = mlines.Line2D([], [], marker=marker, color=color,
                                  markersize=8, linestyle='None',
                                  markeredgecolor='black', markeredgewidth=0.5,
                                  label=f'{label} ({count})')
            legend_handles.append(handle)
        
        if legend_handles:
            fig.legend(handles=legend_handles, loc='lower right',
                      bbox_to_anchor=(0.95, 0.08), fontsize=9,
                      title='UNDP Interventions',
                      framealpha=0.95, ncol=1)
    
    # Main title
    n_sites = len(sites) if sites else 0
    title_suffix = f" with {n_sites} UNDP Sites" if show_markers and n_sites > 0 else ""
    fig.suptitle(f'Myanmar Earthquake: {indicator_display} Change Analysis{title_suffix}',
                fontsize=16, fontweight='bold', y=0.96)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")


def create_summary_comparison(rasters_dict, sites, output_path):
    """Create a 2x3 grid showing multiple indicators."""
    
    indicators = ['viirs', 'ndvi', 'sar_vh']
    available = [i for i in indicators if i in rasters_dict]
    
    if len(available) < 2:
        return
    
    fig, axes = plt.subplots(len(available), 3, figsize=(16, 5*len(available)))
    
    for row, indicator in enumerate(available):
        selected = select_comparison_dates(rasters_dict, indicator)
        
        if selected is None:
            continue
        
        cmap = COLORMAPS.get(indicator, 'viridis')
        
        for col, (date, path) in enumerate(selected):
            ax = axes[row, col] if len(available) > 1 else axes[col]
            
            data = load_raster(path)
            
            vals = data['data'].flatten()
            vals = vals[~np.isnan(vals)]
            vmin, vmax = np.percentile(vals, [2, 98])
            
            ax.imshow(data['data'], extent=data['extent'],
                     cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
            
            ax.set_xlim(AFFECTED_BOUNDS[0], AFFECTED_BOUNDS[2])
            ax.set_ylim(AFFECTED_BOUNDS[1], AFFECTED_BOUNDS[3])
            
            if col == 2 and sites:
                plot_intervention_markers(ax, sites)
            
            if row == 0:
                period = ['Pre-Earthquake', 'Post-Earthquake', 'Post-UNDP'][col]
                ax.set_title(f'{date.strftime("%Y-%m")}\n({period})', fontsize=11)
            
            if col == 0:
                ax.set_ylabel(indicator.upper(), fontsize=11, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle('Myanmar Earthquake: Multi-Indicator Comparison',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")


def main():
    print("=" * 70)
    print("MYANMAR EARTHQUAKE: RASTER MAPS WITH UNDP LOCATIONS")
    print("=" * 70)
    
    if not HAS_RASTERIO:
        print("\n⚠ rasterio required. Install: pip install rasterio --break-system-packages")
        return
    
    # Load UNDP sites
    print("\n--- Loading UNDP Sites ---")
    sites = load_undp_sites()
    print(f"  Total sites: {len(sites)}")
    
    # Load UNDP projects (for dates/categories)
    undp_projects = load_undp_projects()
    
    # Find raster files
    print("\n--- Finding Raster Files ---")
    rasters = find_raster_files()
    
    if not rasters:
        print("  ⚠ No raster files found!")
        print(f"\n  To download rasters, run:")
        print(f"    python -m pipeline.raster_export --config config/myanmar.yaml")
        print(f"\n  Or manually place GeoTIFFs in: {RASTER_DIR}")
        return
    
    for indicator, dates in rasters.items():
        print(f"  {indicator}: {len(dates)} files ({min(dates).strftime('%Y-%m')} to {max(dates).strftime('%Y-%m')})")
    
    # Create three-panel maps for each indicator
    print("\n--- Creating Three-Panel Maps ---")
    print("  (Pre-Earthquake | Post-Earthquake | Post-UNDP)")
    
    indicators_to_map = ['viirs', 'ndvi', 'ndbi', 'sar_vh', 'bsi']
    total_maps = 0
    
    for indicator in indicators_to_map:
        if indicator not in rasters:
            print(f"\n  ⚠ {indicator} not found in rasters, skipping...")
            continue
        
        print(f"\n  Processing {indicator.upper()}...")
        
        # Version WITH markers
        create_three_panel_map(
            indicator, rasters, sites,
            FIGURES_DIR / f'{indicator}_three_panel_with_undp.png',
            show_markers=True
        )
        total_maps += 1
        
        # Version WITHOUT markers (clean)
        create_three_panel_map(
            indicator, rasters, sites,
            FIGURES_DIR / f'{indicator}_three_panel_clean.png',
            show_markers=False
        )
        total_maps += 1
    
    # Create summary comparison
    print("\n--- Creating Summary Comparison ---")
    create_summary_comparison(rasters, sites, FIGURES_DIR / 'summary_comparison.png')
    total_maps += 1
    
    print("\n" + "=" * 70)
    print("✓ RASTER MAPS COMPLETE")
    print(f"  Total maps created: {total_maps}")
    print(f"  Output directory: {FIGURES_DIR}")
    print("\n  Files created:")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"    - {f.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
