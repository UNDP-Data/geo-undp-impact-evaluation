#!/usr/bin/env python3
"""
analysis/pse/09_conflict_maps.py
================================
Create conflict event density maps for Sep 2023, Sep 2024, Sep 2025.

UPDATES:
- Zoomed to Gaza bounds only
- Two versions: with and without UNDP sites
- Fixed colormap to show actual hotspots in red/dark colors
- Uses percentile-based normalization for better visualization

Usage:
    python analysis/pse/09_conflict_maps.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
SITES_CSV = DATA_DIR / "pse" / "external" / "undp_sites_digitized.csv"
BOUNDARIES_FILE = DATA_DIR / "pse" / "boundaries" / "gaza_admin2.geojson"

# ACLED paths
ACLED_PATHS = [
    DATA_DIR / "pse" / "external" / "acled_conflict.csv",  # Your actual file
    DATA_DIR / "pse" / "conflict" / "acled_pse.csv",
    DATA_DIR / "pse" / "acled_pse.csv",
    DATA_DIR / "pse" / "external" / "acled_pse.csv",
    DATA_DIR / "pse" / "conflict" / "acled_gaza.csv",
    DATA_DIR / "pse" / "acled.csv",
]

OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# GAZA BOUNDS (tight zoom)
GAZA_BOUNDS = {
    'minx': 34.20,
    'maxx': 34.60,
    'miny': 31.20,
    'maxy': 31.60,
}

# Time periods for maps
MAP_PERIODS = [
    ('Sep 2023', pd.Timestamp('2023-09-01'), pd.Timestamp('2023-09-30'), 'Pre-Conflict'),
    ('Sep 2024', pd.Timestamp('2024-09-01'), pd.Timestamp('2024-09-30'), 'During Conflict'),
    ('Sep 2025', pd.Timestamp('2025-09-01'), pd.Timestamp('2025-09-30'), 'Post-Intervention'),
]

# Alternative periods if Sep 2025 has no data
MAP_PERIODS_ALT = [
    ('Sep 2023', pd.Timestamp('2023-09-01'), pd.Timestamp('2023-09-30'), 'Pre-Conflict'),
    ('Sep 2024', pd.Timestamp('2024-09-01'), pd.Timestamp('2024-09-30'), 'During Conflict'),
    ('Jan 2025', pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-31'), 'Recent'),
]

# Grid resolution
GRID_RESOLUTION = 0.002  # ~200m cells for finer detail

# =============================================================================
# DATA LOADING
# =============================================================================

def load_acled():
    """Load ACLED conflict data."""
    for path in ACLED_PATHS:
        if path.exists():
            print(f"  Found ACLED: {path}")
            df = pd.read_csv(path)
            
            # Standardize columns
            col_map = {
                'event_date': 'date', 'EVENT_DATE': 'date', 'Date': 'date',
                'latitude': 'lat', 'LATITUDE': 'lat', 'Latitude': 'lat',
                'longitude': 'lon', 'LONGITUDE': 'lon', 'Longitude': 'lon',
                'fatalities': 'fatalities', 'FATALITIES': 'fatalities',
            }
            for old, new in col_map.items():
                if old in df.columns and new not in df.columns:
                    df[new] = df[old]
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Filter to Gaza bounds
            df = df[(df['lat'] >= GAZA_BOUNDS['miny']) & (df['lat'] <= GAZA_BOUNDS['maxy']) &
                   (df['lon'] >= GAZA_BOUNDS['minx']) & (df['lon'] <= GAZA_BOUNDS['maxx'])]
            
            return df
    
    print("  ⚠ No ACLED file found")
    return None


def load_treated_sites():
    """Load UNDP intervention sites."""
    return pd.read_csv(SITES_CSV)


def load_boundaries():
    """Load Gaza boundaries."""
    return gpd.read_file(BOUNDARIES_FILE)


# =============================================================================
# CONFLICT DENSITY CALCULATION
# =============================================================================

def create_conflict_density_grid(acled_df, start_date, end_date, resolution=GRID_RESOLUTION):
    """Create a density grid of conflict events."""
    # Filter to date range
    period_data = acled_df[(acled_df['date'] >= start_date) & 
                           (acled_df['date'] <= end_date)].copy()
    
    # Use Gaza bounds
    minx, maxx = GAZA_BOUNDS['minx'], GAZA_BOUNDS['maxx']
    miny, maxy = GAZA_BOUNDS['miny'], GAZA_BOUNDS['maxy']
    
    # Create grid
    x_edges = np.arange(minx, maxx + resolution, resolution)
    y_edges = np.arange(miny, maxy + resolution, resolution)
    
    xx, yy = np.meshgrid(x_edges[:-1] + resolution/2, y_edges[:-1] + resolution/2)
    
    # Calculate event density
    density = np.zeros((len(y_edges)-1, len(x_edges)-1))
    fatality_density = np.zeros_like(density)
    
    if not period_data.empty:
        for _, event in period_data.iterrows():
            x_idx = np.searchsorted(x_edges, event['lon']) - 1
            y_idx = np.searchsorted(y_edges, event['lat']) - 1
            
            if 0 <= x_idx < len(x_edges)-1 and 0 <= y_idx < len(y_edges)-1:
                density[y_idx, x_idx] += 1
                if 'fatalities' in event and pd.notna(event['fatalities']):
                    fatality_density[y_idx, x_idx] += event['fatalities']
    
    return {
        'xx': xx,
        'yy': yy,
        'density': density,
        'fatality_density': fatality_density,
        'n_events': len(period_data),
        'n_fatalities': period_data['fatalities'].sum() if 'fatalities' in period_data.columns else 0,
        'start_date': start_date,
        'end_date': end_date,
    }


# =============================================================================
# THREE-PANEL CONFLICT MAPS
# =============================================================================

def create_three_panel_conflict_map(acled_df, boundaries, treated_sites, output_path, show_undp=True):
    """
    Create three-panel conflict density map.
    Darker = more conflict events.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # Check data availability and choose periods
    latest_date = acled_df['date'].max()
    print(f"  Latest ACLED date: {latest_date}")
    
    if latest_date < pd.Timestamp('2025-09-01'):
        # Use alternative periods
        periods = MAP_PERIODS_ALT
        print(f"  Using alternative periods (data ends before Sep 2025)")
    else:
        periods = MAP_PERIODS
    
    # Calculate density for each period
    density_data = []
    for period_name, start_date, end_date, phase in periods:
        data = create_conflict_density_grid(acled_df, start_date, end_date)
        data['period_name'] = period_name
        data['phase'] = phase
        density_data.append(data)
        print(f"    {period_name}: {data['n_events']} events, {data['n_fatalities']:.0f} fatalities")
    
    # Get vmax based on 95th percentile of non-zero values (to avoid outlier-driven scaling)
    all_nonzero = []
    for d in density_data:
        nonzero = d['density'][d['density'] > 0]
        if len(nonzero) > 0:
            all_nonzero.extend(nonzero.flatten())
    
    if all_nonzero:
        vmax = np.percentile(all_nonzero, 95)
        vmax = max(vmax, 5)  # At least 5 for visibility
    else:
        vmax = 10
    
    print(f"  Colormap vmax: {vmax:.1f} (95th percentile)")
    
    # Custom colormap: light → dark (yellow → orange → red → dark red → black)
    colors = ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', 
              '#e31a1c', '#bd0026', '#800026', '#4d0014', '#1a0008']
    cmap = mcolors.LinearSegmentedColormap.from_list('conflict_hot', colors)
    
    for i, (ax, data) in enumerate(zip(axes, density_data)):
        # Plot boundaries (background)
        boundaries.plot(ax=ax, color='#f5f5f5', edgecolor='black', linewidth=1.5)
        
        # Mask zero values for transparency
        density_masked = np.ma.masked_where(data['density'] == 0, data['density'])
        
        # Plot density heatmap
        im = ax.pcolormesh(data['xx'], data['yy'], density_masked,
                          cmap=cmap, vmin=0, vmax=vmax, alpha=0.9,
                          shading='auto')
        
        # Re-plot boundary outline
        boundaries.boundary.plot(ax=ax, color='black', linewidth=1.5)
        
        # Plot UNDP sites if requested
        if show_undp:
            ax.scatter(treated_sites['longitude'], treated_sites['latitude'],
                      c='blue', s=9, alpha=0.9, marker='o', zorder=10,
                      edgecolors='none', label='UNDP Sites')
        
        # Title
        ax.set_title(f"{data['period_name']}\n({data['phase']})\n"
                    f"{data['n_events']:,} events",
                    fontsize=12, fontweight='bold')
        
        # Set Gaza bounds (ZOOM)
        ax.set_xlim(GAZA_BOUNDS['minx'], GAZA_BOUNDS['maxx'])
        ax.set_ylim(GAZA_BOUNDS['miny'], GAZA_BOUNDS['maxy'])
        
        ax.set_xlabel('Longitude', fontsize=10)
        if i == 0:
            ax.set_ylabel('Latitude', fontsize=10)
        ax.set_aspect('equal')
        
        # Legend on first panel
        if i == 0 and show_undp:
            ax.legend(loc='upper left', fontsize=9, markerscale=1.5)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='max')
    cbar.set_label('Conflict Events per Cell (~200m)\nDarker = More Conflict', fontsize=11)
    
    # Title
    undp_text = "with UNDP Sites" if show_undp else "without UNDP Sites"
    fig.suptitle(f'Conflict Event Density: Before vs During Conflict\n({undp_text})',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.10, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def create_conflict_hotspot_map(acled_df, boundaries, treated_sites, output_path, show_undp=True):
    """
    Create cumulative conflict hotspot map (Oct 2023 - latest data).
    """
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Cumulative data from conflict start
    start_date = pd.Timestamp('2023-10-07')
    end_date = acled_df['date'].max()
    
    data = create_conflict_density_grid(acled_df, start_date, end_date)
    
    print(f"  Cumulative period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"    Total events: {data['n_events']:,}")
    
    # Plot boundaries
    boundaries.plot(ax=ax, color='#f5f5f5', edgecolor='black', linewidth=1.5)
    
    # Use log scale for cumulative data (likely has more variation)
    density_log = np.log1p(data['density'])
    density_masked = np.ma.masked_where(data['density'] == 0, density_log)
    
    # Vmax based on 95th percentile
    nonzero = density_log[data['density'] > 0]
    if len(nonzero) > 0:
        vmax = np.percentile(nonzero, 95)
    else:
        vmax = 3
    
    colors = ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', 
              '#e31a1c', '#bd0026', '#800026', '#4d0014', '#1a0008']
    cmap = mcolors.LinearSegmentedColormap.from_list('conflict_hot', colors)
    
    im = ax.pcolormesh(data['xx'], data['yy'], density_masked,
                      cmap=cmap, vmin=0, vmax=vmax, alpha=0.9, shading='auto')
    
    boundaries.boundary.plot(ax=ax, color='black', linewidth=1.5)
    
    if show_undp:
        ax.scatter(treated_sites['longitude'], treated_sites['latitude'],
                  c='blue', s=50, alpha=0.9, marker='o', zorder=10,
                  edgecolors='white', linewidth=0.8, label='UNDP Sites')
        ax.legend(loc='upper left', fontsize=10, markerscale=1.5)
    
    # Zoom to Gaza
    ax.set_xlim(GAZA_BOUNDS['minx'], GAZA_BOUNDS['maxx'])
    ax.set_ylim(GAZA_BOUNDS['miny'], GAZA_BOUNDS['maxy'])
    
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_aspect('equal')
    
    undp_text = "with UNDP Sites" if show_undp else "Conflict Only"
    ax.set_title(f'Cumulative Conflict Hotspots: Oct 2023 - {end_date.strftime("%b %Y")}\n'
                 f'{data["n_events"]:,} total events ({undp_text})',
                 fontsize=13, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8, extend='max')
    cbar.set_label('Log(Events + 1) per Cell\nDarker = More Conflict', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def create_point_scatter_map(acled_df, boundaries, treated_sites, output_path, show_undp=True):
    """
    Create a simple point scatter map as alternative visualization.
    Each conflict event is a semi-transparent red dot.
    """
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Conflict period data
    conflict_data = acled_df[acled_df['date'] >= pd.Timestamp('2023-10-07')]
    
    # Plot boundaries
    boundaries.plot(ax=ax, color='#f5f5f5', edgecolor='black', linewidth=1.5)
    
    # Plot conflict events as points
    ax.scatter(conflict_data['lon'], conflict_data['lat'],
              c='red', s=35, alpha=0.25, marker='o', zorder=5)
    
    # Re-plot boundary
    boundaries.boundary.plot(ax=ax, color='black', linewidth=1.5)
    
    if show_undp:
        ax.scatter(treated_sites['longitude'], treated_sites['latitude'],
                  c='blue', s=35, alpha=0.9, marker='o', zorder=10,
                  edgecolors='none', label='UNDP Sites')
        ax.legend(loc='upper left', fontsize=10, markerscale=1.5)
    
    # Zoom to Gaza
    ax.set_xlim(GAZA_BOUNDS['minx'], GAZA_BOUNDS['maxx'])
    ax.set_ylim(GAZA_BOUNDS['miny'], GAZA_BOUNDS['maxy'])
    
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_aspect('equal')
    
    undp_text = "with UNDP Sites" if show_undp else ""
    ax.set_title(f'Conflict Events Since Oct 7, 2023 {undp_text}\n'
                 f'{len(conflict_data):,} events (each dot = 1 event, overlapping shows density)',
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" CONFLICT EVENT DENSITY MAPS (Gaza Zoomed)")
    print("=" * 70)
    
    # Load data
    print("\n--- Loading Data ---")
    
    acled_df = load_acled()
    if acled_df is None:
        print("  ⚠ Cannot proceed without ACLED data")
        return
    
    print(f"  Events in Gaza: {len(acled_df)}")
    print(f"  Date range: {acled_df['date'].min().strftime('%Y-%m-%d')} to {acled_df['date'].max().strftime('%Y-%m-%d')}")
    
    boundaries = load_boundaries()
    print(f"  Boundaries loaded")
    
    treated_sites = load_treated_sites()
    print(f"  UNDP sites: {len(treated_sites)}")
    
    # Create maps
    print("\n--- Creating Maps ---")
    
    # THREE-PANEL: WITH UNDP sites
    print("\n  Three-panel map WITH UNDP sites...")
    create_three_panel_conflict_map(acled_df, boundaries, treated_sites,
                                    FIGURES_DIR / "conflict_map_three_panel_with_undp.png",
                                    show_undp=True)
    
    # THREE-PANEL: WITHOUT UNDP sites
    print("\n  Three-panel map WITHOUT UNDP sites...")
    create_three_panel_conflict_map(acled_df, boundaries, treated_sites,
                                    FIGURES_DIR / "conflict_map_three_panel_no_undp.png",
                                    show_undp=False)
    
    # CUMULATIVE HOTSPOT: WITH UNDP
    print("\n  Cumulative hotspot map WITH UNDP sites...")
    create_conflict_hotspot_map(acled_df, boundaries, treated_sites,
                               FIGURES_DIR / "conflict_hotspot_with_undp.png",
                               show_undp=True)
    
    # CUMULATIVE HOTSPOT: WITHOUT UNDP
    print("\n  Cumulative hotspot map WITHOUT UNDP sites...")
    create_conflict_hotspot_map(acled_df, boundaries, treated_sites,
                               FIGURES_DIR / "conflict_hotspot_no_undp.png",
                               show_undp=False)
    
    # POINT SCATTER: WITH UNDP
    print("\n  Point scatter map WITH UNDP sites...")
    create_point_scatter_map(acled_df, boundaries, treated_sites,
                            FIGURES_DIR / "conflict_scatter_with_undp.png",
                            show_undp=True)
    
    # POINT SCATTER: WITHOUT UNDP
    print("\n  Point scatter map WITHOUT UNDP sites...")
    create_point_scatter_map(acled_df, boundaries, treated_sites,
                            FIGURES_DIR / "conflict_scatter_no_undp.png",
                            show_undp=False)
    
    print(f"""

  OUTPUT FILES:
  ─────────────
  Three-panel (Sep 2023 / Sep 2024 / Recent):
    • conflict_map_three_panel_with_undp.png
    • conflict_map_three_panel_no_undp.png
  
  Cumulative hotspots (Oct 2023 - present):
    • conflict_hotspot_with_undp.png
    • conflict_hotspot_no_undp.png
  
  Point scatter (each event as a dot):
    • conflict_scatter_with_undp.png
    • conflict_scatter_no_undp.png
    """)
    
    print("=" * 70)
    print(" MAPS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
