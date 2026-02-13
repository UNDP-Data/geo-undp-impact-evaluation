#!/usr/bin/env python3
"""
pipeline/raster_maps.py
=======================
Three-panel raster maps showing actual satellite imagery (not aggregated).
Pre-Disaster | Post-Disaster | Most Recent

Requires:
- Raster files in data/{country}/rasters/*.tif
- Run raster_export.py first to download rasters from GEE

Usage:
    python -m pipeline.raster_maps --config config/myanmar.yaml
    python -m pipeline.raster_maps --config config/afghanistan.yaml
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import yaml
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import geopandas as gpd
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap


# Pure black to yellow (no brown tones)
NIGHTLIGHTS_CMAP = LinearSegmentedColormap.from_list('black_yellow', [
    '#000000', '#404000', '#808000', '#c0c000', '#ffff00', '#ffff80'
])

UN_BLUE = '#009EDB'

CATEGORY_COLORS = {
    'Education': '#3498DB', 'Debris & Solid Waste': '#795548',
    'Water & Sanitation (WASH)': '#00BCD4', 'Energy & Power': '#FFC107',
    'Health & Emergency': '#F44336', 'Infrastructure & Housing': '#9C27B0',
    'Livelihoods & Economy': '#4CAF50', 'Governance & Rule of Law': '#607D8B',
    'Social Inclusion': '#E91E63', 'Environment': '#8BC34A',
    'Other / Unspecified': '#9E9E9E'
}


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_raster_files(raster_dir):
    """Find all raster files organized by indicator and date."""
    rasters = {}
    if not raster_dir.exists():
        return rasters
    
    for f in sorted(raster_dir.glob("*.tif")):
        name = f.stem
        parts = name.split('_')
        if len(parts) >= 2:
            # Handle formats: viirs_202503, viirs_mmr_202503, sar_vh_202503
            indicator = parts[0]
            if indicator == 'sar' and len(parts) >= 3:
                indicator = f"{parts[0]}_{parts[1]}"
            
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


def select_dates(rasters_dict, indicator, disaster_date):
    """Select pre-disaster, post-disaster, and most recent dates."""
    if indicator not in rasters_dict:
        return None
    
    available = sorted(rasters_dict[indicator].keys())
    if len(available) < 2:
        return None
    
    disaster_dt = pd.Timestamp(disaster_date)
    pre_dates = [d for d in available if pd.Timestamp(d) < disaster_dt]
    post_dates = [d for d in available if pd.Timestamp(d) >= disaster_dt]
    
    if not pre_dates:
        # All dates are post-disaster
        if len(available) >= 3:
            return [(available[0], 'First Available'),
                    (available[len(available)//2], 'Middle'),
                    (available[-1], 'Most Recent')]
        return None
    
    if not post_dates:
        # All dates are pre-disaster
        return [(available[0], 'Earliest'),
                (available[-1], 'Pre-Disaster'),
                (available[-1], 'Latest')]
    
    # Normal case: pre and post available
    pre = pre_dates[-1]  # Last pre-disaster
    post_first = post_dates[0]  # First post-disaster
    recent = post_dates[-1]  # Most recent
    
    return [
        (pre, 'Pre-Disaster'),
        (post_first, 'Post-Disaster'),
        (recent, 'Most Recent')
    ]


def create_raster_map(indicator, rasters_dict, config, gdf, national, sites, 
                      undp_projects, output_path, show_markers=True):
    """Create three-panel raster map."""
    
    disaster_date = config['events']['disaster_date']
    disaster_type = config['events'].get('disaster_type', 'Disaster').title()
    country_name = config['country_name']
    
    selected = select_dates(rasters_dict, indicator, disaster_date)
    if selected is None:
        return False
    
    # Load rasters
    raster_data = []
    dates_info = []
    for date, label in selected:
        raster_data.append(load_raster(rasters_dict[indicator][date]))
        dates_info.append((date, label))
    
    # Get project categories
    category_counts = {}
    if undp_projects is not None and len(undp_projects) > 0:
        disaster_dt = pd.Timestamp(disaster_date)
        post = undp_projects[undp_projects['start_date'] >= disaster_dt]
        if len(post) > 0 and 'ml_category' in post.columns:
            category_counts = post['ml_category'].value_counts().to_dict()
    
    # Create figure
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 3, wspace=0.12, left=0.05, right=0.95, top=0.88, bottom=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    
    # Colormap
    if indicator == 'viirs':
        cmap = NIGHTLIGHTS_CMAP
    elif indicator == 'ndvi':
        cmap = 'RdYlGn'
    else:
        cmap = 'viridis'
    
    # Color scale
    all_vals = np.concatenate([r['data'].flatten() for r in raster_data])
    all_vals = all_vals[~np.isnan(all_vals)]
    
    if indicator == 'viirs':
        vmin = 0
        vmax = np.percentile(all_vals, 98)
    else:
        vmin, vmax = np.percentile(all_vals, [2, 98])
    
    # Country extent
    if national is not None:
        bounds = national.total_bounds
        pad_x = (bounds[2] - bounds[0]) * 0.02
        pad_y = (bounds[3] - bounds[1]) * 0.02
        country_extent = [bounds[0] - pad_x, bounds[2] + pad_x, bounds[1] - pad_y, bounds[3] + pad_y]
    else:
        country_extent = raster_data[0]['extent']
    
    # Title colors
    title_colors = ['forestgreen', 'crimson', 'royalblue']
    
    # Indicator display name
    indicator_names = {
        'viirs': 'Nightlights (VIIRS)',
        'ndvi': 'Vegetation (NDVI)',
        'sar_vh': 'SAR VH (Radar)',
        'sar_vv': 'SAR VV (Radar)',
    }
    indicator_display = indicator_names.get(indicator, indicator.upper())
    
    # Plot panels
    for i, (ax, data, (date, label), color) in enumerate(zip(axes, raster_data, dates_info, title_colors)):
        im = ax.imshow(data['data'], extent=data['extent'],
                      cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
        
        ax.set_xlim(country_extent[0], country_extent[1])
        ax.set_ylim(country_extent[2], country_extent[3])
        
        # National boundary
        if national is not None:
            national.boundary.plot(ax=ax, color='white', linewidth=1.5, alpha=0.9)
        
        # Admin boundaries
        if gdf is not None:
            gdf.boundary.plot(ax=ax, color='white', linewidth=0.2, alpha=0.3)
        
        # UNDP sites (only on last panel)
        if i == 2 and show_markers and sites:
            lons = [s['lon'] for s in sites]
            lats = [s['lat'] for s in sites]
            ax.scatter(lons, lats, marker='o', c=UN_BLUE, s=12,
                      edgecolors='white', linewidths=0.3, zorder=10, alpha=0.8)
        
        ax.set_title(f"{date.strftime('%Y-%m')}\n({label})",
                    fontsize=14, fontweight='bold', color=color)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Colorbar
    cbar_ax = fig.add_axes([0.25, 0.10, 0.50, 0.025])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    
    unit = 'nW/cm²/sr' if indicator == 'viirs' else 'Index'
    cbar.set_label(f'{indicator_display} ({unit})', fontsize=12)
    
    # Legend
    if show_markers and sites:
        legend_handles = [
            mlines.Line2D([], [], marker='o', color=UN_BLUE, markersize=10, linestyle='None',
                         markeredgecolor='white', markeredgewidth=0.5, label=f'UNDP Sites ({len(sites)})'),
        ]
        
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:4]:
            c = CATEGORY_COLORS.get(cat, '#9E9E9E')
            legend_handles.append(mlines.Line2D([], [], marker='s', color=c, markersize=8,
                                               linestyle='None', markeredgecolor='black',
                                               markeredgewidth=0.5, label=f'{cat} ({count})'))
        
        fig.legend(handles=legend_handles, loc='lower right', bbox_to_anchor=(0.95, 0.02),
                  fontsize=9, title='UNDP Response', framealpha=0.95)
    
    # Title
    n_sites = len(sites) if sites else 0
    title_text = f'{country_name} {disaster_type}: {indicator_display}'
    if show_markers and n_sites > 0:
        title_text += f' with {n_sites} UNDP Sites'
    fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.96)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    args = parser.parse_args()
    
    if not HAS_RASTERIO:
        print("⚠ rasterio not installed. Run: pip install rasterio")
        return
    
    config = load_config(args.config)
    
    country_name = config['country_name']
    country_code = config['country_code'].lower()
    
    data_dir = Path(f"data/{country_code}")
    raster_dir = data_dir / "rasters"
    output_dir = Path(f"outputs/{country_code}")
    figures_dir = output_dir / "figures" / "raster_maps"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"RASTER MAPS: {country_name}")
    print(f"{'='*70}")
    
    # Load boundaries
    gdf, national = None, None
    boundary_path = Path(config['boundaries']['file_path'])
    if boundary_path.exists():
        gdf = gpd.read_file(boundary_path).to_crs("EPSG:4326")
        national = gdf.dissolve()
        print(f"Boundaries: {len(gdf)} units")
    
    # Load UNDP projects
    undp_path = data_dir / "external" / "undp_projects.csv"
    undp_projects = None
    if undp_path.exists():
        undp_projects = pd.read_csv(undp_path)
        undp_projects['start_date'] = pd.to_datetime(undp_projects['start_date'], errors='coerce')
        if 'is_likely_umbrella' in undp_projects.columns:
            undp_projects = undp_projects[undp_projects['is_likely_umbrella'] != True]
        print(f"UNDP Projects: {len(undp_projects)}")
    
    # Load UNDP sites
    sites = []
    sites_path = data_dir / "external" / "undp_sites.csv"
    if sites_path.exists():
        df = pd.read_csv(sites_path)
        for _, row in df.iterrows():
            lat = row.get('latitude', row.get('lat'))
            lon = row.get('longitude', row.get('lon'))
            if pd.notna(lat) and pd.notna(lon):
                sites.append({'lat': lat, 'lon': lon})
        print(f"UNDP Sites: {len(sites)}")
    
    # Find rasters
    rasters = find_raster_files(raster_dir)
    n_rasters = sum(len(v) for v in rasters.values())
    print(f"Rasters: {n_rasters} files")
    
    if not rasters:
        print(f"\n⚠ No rasters found in {raster_dir}")
        print(f"  Run: python -m pipeline.raster_export --config {args.config}")
        return
    
    print(f"\nIndicators: {list(rasters.keys())}")
    
    # Create maps
    print("\n--- Creating Maps ---")
    
    for indicator in rasters.keys():
        print(f"\n  {indicator.upper()}...")
        
        # With UNDP sites
        success = create_raster_map(
            indicator, rasters, config, gdf, national, sites, undp_projects,
            figures_dir / f'{indicator}_raster_with_undp.png', show_markers=True)
        if success:
            print(f"    ✓ {indicator}_raster_with_undp.png")
        
        # Clean (no markers)
        success = create_raster_map(
            indicator, rasters, config, gdf, national, [], None,
            figures_dir / f'{indicator}_raster_clean.png', show_markers=False)
        if success:
            print(f"    ✓ {indicator}_raster_clean.png")
    
    print(f"\n{'='*70}")
    print(f"✓ Complete: {figures_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
