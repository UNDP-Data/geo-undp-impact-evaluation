#!/usr/bin/env python3
"""
analysis/mmr/03_raster_maps_v3.py
=================================
Myanmar three-panel raster maps with:
- Full country view with national boundary
- PURE BLACK TO YELLOW colormap (no brown tones!)
- UN BLUE color for UNDP sites (#009EDB)
- Project categories from undp_projects.csv
- Better spacing and centering

Usage:
    python analysis/mmr/03_raster_maps_v3.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("⚠ rasterio not installed")

try:
    import geopandas as gpd
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

# Files
BOUNDARIES_FILE = DATA_DIR / "boundaries" / "mmr_admin3.geojson"
UNDP_PROJECTS_FILE = DATA_DIR / "external" / "undp_projects.csv"
UNDP_SITES_FILE = DATA_DIR / "external" / "undp_sites.csv"

# UN BLUE for UNDP sites
UN_BLUE = '#009EDB'

# PURE BLACK TO YELLOW colormap (NO brown tones!)
NIGHTLIGHTS_CMAP = LinearSegmentedColormap.from_list('black_to_yellow', [
    (0.0, '#000000'),   # Black
    (0.15, '#1a1a00'),  # Very dark yellow-black
    (0.3, '#333300'),   # Dark olive
    (0.45, '#666600'),  # Olive
    (0.6, '#999900'),   # Yellow-olive
    (0.75, '#cccc00'),  # Dark yellow
    (0.9, '#ffff00'),   # Bright yellow
    (1.0, '#ffff66'),   # Light yellow
])

# Alternative: Even simpler black to yellow
NIGHTLIGHTS_SIMPLE = LinearSegmentedColormap.from_list('black_yellow_simple', [
    '#000000',  # Black
    '#404000',  # Dark
    '#808000',  # Olive (slight green-yellow)
    '#c0c000',  # Yellow-green
    '#ffff00',  # Yellow
    '#ffff80',  # Light yellow
])

# Category colors
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


def load_boundaries():
    """Load Myanmar national boundary."""
    for path in [BOUNDARIES_FILE, DATA_DIR / "myanmar_boundaries" / "mmr_admin3.geojson"]:
        if path.exists():
            gdf = gpd.read_file(path).to_crs("EPSG:4326")
            national = gdf.dissolve()
            return gdf, national
    return None, None


def load_undp_projects():
    """Load UNDP projects."""
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
            sites.append({'lat': lat, 'lon': lon})
    return sites


def get_project_categories(undp_projects):
    """Get project category distribution."""
    if undp_projects is None or len(undp_projects) == 0:
        return {}
    post_eq = undp_projects[undp_projects['start_date'] >= EARTHQUAKE_DATE]
    if len(post_eq) == 0:
        post_eq = undp_projects
    if 'ml_category' in post_eq.columns:
        return post_eq['ml_category'].value_counts().to_dict()
    return {}


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
    """Create three-panel map."""
    
    selected = select_comparison_dates(rasters_dict, indicator)
    if selected is None:
        return False
    
    raster_data = [load_raster(path) for date, path in selected]
    category_counts = get_project_categories(undp_projects)
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 3, wspace=0.12, left=0.05, right=0.95, top=0.88, bottom=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    
    # Use SIMPLE black-to-yellow (no brown!)
    if indicator == 'viirs':
        cmap = NIGHTLIGHTS_SIMPLE
    else:
        cmap = 'viridis'
    
    # Compute color scale
    all_vals = np.concatenate([r['data'].flatten() for r in raster_data])
    all_vals = all_vals[~np.isnan(all_vals)]
    
    if indicator == 'viirs':
        vmin = 0
        vmax = np.percentile(all_vals, 98)  # Slightly higher to make yellows more visible
    else:
        vmin, vmax = np.percentile(all_vals, [2, 98])
    
    # Country bounds
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
    }.get(indicator, indicator.upper())
    
    for i, (ax, (date, _), data, (title, color)) in enumerate(zip(axes, selected, raster_data, titles)):
        # Plot raster
        im = ax.imshow(data['data'], extent=data['extent'],
                       cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
        
        # Country extent
        ax.set_xlim(country_extent[0], country_extent[1])
        ax.set_ylim(country_extent[2], country_extent[3])
        
        # National boundary (white)
        if national is not None:
            national.boundary.plot(ax=ax, color='white', linewidth=1.5, alpha=0.9)
        
        # Admin boundaries (thin)
        if gdf is not None:
            gdf.boundary.plot(ax=ax, color='white', linewidth=0.2, alpha=0.3)
        
        # Epicenter
        ax.scatter([95.925], [22.001], marker='*', s=300, c='red',
                  edgecolor='white', linewidth=1.5, zorder=15)
        
        # UNDP sites (UN BLUE) - only on last panel
        if i == 2 and show_markers and sites:
            lons = [s['lon'] for s in sites]
            lats = [s['lat'] for s in sites]
            ax.scatter(lons, lats, marker='o', c=UN_BLUE,
                      s=12, edgecolors='white', linewidths=0.3, 
                      zorder=10, alpha=0.8)
        
        ax.set_title(f"{date.strftime('%Y-%m')}\n({title})",
                    fontsize=14, fontweight='bold', color=color)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Colorbar
    cbar_ax = fig.add_axes([0.25, 0.10, 0.50, 0.025])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'{indicator_display} (nW/cm²/sr)', fontsize=12)
    
    # Legend
    if show_markers and sites:
        legend_handles = [
            mlines.Line2D([], [], marker='o', color=UN_BLUE, markersize=10, linestyle='None',
                         markeredgecolor='white', markeredgewidth=0.5, label=f'UNDP Sites ({len(sites)})'),
            mlines.Line2D([], [], marker='*', color='red', markersize=15, linestyle='None',
                         markeredgecolor='white', markeredgewidth=1, label='Epicenter (M7.7)'),
        ]
        
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:4]:
            color = CATEGORY_COLORS.get(cat, '#9E9E9E')
            legend_handles.append(mlines.Line2D([], [], marker='s', color=color, markersize=8,
                                               linestyle='None', markeredgecolor='black',
                                               markeredgewidth=0.5, label=f'{cat} ({count})'))
        
        fig.legend(handles=legend_handles, loc='lower right', bbox_to_anchor=(0.95, 0.02),
                  fontsize=9, title='UNDP Response', framealpha=0.95)
    
    # Title
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
    print("MYANMAR EARTHQUAKE: RASTER MAPS (V3)")
    print("  - PURE black-to-yellow colormap (no brown)")
    print("  - UN Blue for UNDP sites")
    print("=" * 70)
    
    if not HAS_RASTERIO:
        return
    
    gdf, national = load_boundaries()
    undp_projects = load_undp_projects()
    sites = load_undp_sites()
    rasters = find_raster_files()
    
    print(f"\n  Admin units: {len(gdf) if gdf is not None else 0}")
    print(f"  UNDP projects: {len(undp_projects) if undp_projects is not None else 0}")
    print(f"  UNDP sites: {len(sites)}")
    print(f"  Rasters: {sum(len(v) for v in rasters.values())} files")
    
    if not rasters:
        print("  ⚠ No rasters found")
        return
    
    print("\n--- Creating Maps ---")
    
    for indicator in ['viirs', 'ndvi', 'sar_vh']:
        if indicator not in rasters:
            continue
        
        print(f"\n  {indicator.upper()}...")
        
        success = create_three_panel_map(
            indicator, rasters, sites, undp_projects, gdf, national,
            FIGURES_DIR / f'{indicator}_three_panel_with_undp.png', show_markers=True)
        if success:
            print(f"    ✓ {indicator}_three_panel_with_undp.png")
        
        success = create_three_panel_map(
            indicator, rasters, sites, undp_projects, gdf, national,
            FIGURES_DIR / f'{indicator}_three_panel_clean.png', show_markers=False)
        if success:
            print(f"    ✓ {indicator}_three_panel_clean.png")
    
    print(f"\n{'='*70}")
    print(f"✓ COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
