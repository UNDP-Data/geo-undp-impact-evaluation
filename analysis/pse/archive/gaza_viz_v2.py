#!/usr/bin/env python3
"""
analysis/pse/gaza_viz_v2.py
===========================
Improved Gaza visualizations:
- Line plots for time series (not bar charts)
- Specific month maps (Sep 2023, Sep 2024, Sep 2025)
- Black-to-yellow nightlight color scheme
- West Bank as potential control group comparison
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
PANEL_FILE = "outputs/pse/panel_data.csv"
BOUNDARIES_FILE = "data/pse/boundaries/gaza_admin2.geojson"
FIGURES_DIR = Path("outputs/pse/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Key dates
CONFLICT_DATE = pd.Timestamp("2023-10-07")
UNDP_DATE = pd.Timestamp("2025-06-01")

# Months to show in maps
MAP_MONTHS = ["2023-09", "2024-09", "2025-09"]

# Custom colormaps
NIGHTLIGHT_CMAP = LinearSegmentedColormap.from_list(
    'nightlights', ['#000000', '#1a1a00', '#4d4d00', '#808000', '#b3b300', '#ffff00', '#ffff66']
)

# Outcome configurations
OUTCOMES = {
    'nightlights': {
        'name': 'Nighttime Lights', 
        'cmap': NIGHTLIGHT_CMAP, 
        'line_color': '#FFD700',
        'fill_color': '#FFF8DC',
        'unit': 'nW/cm²/sr'
    },
    'ndvi': {
        'name': 'Vegetation (NDVI)', 
        'cmap': 'YlGn', 
        'line_color': 'darkgreen',
        'fill_color': 'lightgreen',
        'unit': 'Index'
    },
    'conflict_events': {
        'name': 'Conflict Events', 
        'cmap': 'YlOrRd', 
        'line_color': 'darkred',
        'fill_color': 'lightcoral',
        'unit': 'Events'
    },
    'rainfall': {
        'name': 'Rainfall', 
        'cmap': 'Blues', 
        'line_color': 'darkblue',
        'fill_color': 'lightblue',
        'unit': 'mm'
    },
    'price_bread': {
        'name': 'Bread Price', 
        'cmap': 'RdYlGn_r', 
        'line_color': 'purple',
        'fill_color': 'lavender',
        'unit': 'ILS'
    },
    'vv': {
        'name': 'SAR VV Backscatter', 
        'cmap': 'viridis', 
        'line_color': 'teal',
        'fill_color': 'lightcyan',
        'unit': 'dB'
    },
    'vh': {
        'name': 'SAR VH Backscatter', 
        'cmap': 'viridis', 
        'line_color': 'navy',
        'fill_color': 'lightsteelblue',
        'unit': 'dB'
    },
    'rvi': {
        'name': 'Radar Vegetation Index', 
        'cmap': 'YlGn', 
        'line_color': 'olive',
        'fill_color': 'khaki',
        'unit': 'Index'
    },
    'ndbi': {
        'name': 'Built-up Index (NDBI)', 
        'cmap': 'RdYlBu_r', 
        'line_color': 'brown',
        'fill_color': 'bisque',
        'unit': 'Index'
    },
    'bsi': {
        'name': 'Bare Soil Index', 
        'cmap': 'YlOrBr', 
        'line_color': 'sienna',
        'fill_color': 'wheat',
        'unit': 'Index'
    },
    'gt_cement': {
        'name': 'Google Trends: Cement', 
        'cmap': 'Purples', 
        'line_color': 'purple',
        'fill_color': 'lavender',
        'unit': 'Interest'
    },
    'gt_water': {
        'name': 'Google Trends: Water', 
        'cmap': 'Blues', 
        'line_color': 'blue',
        'fill_color': 'lightblue',
        'unit': 'Interest'
    },
    'gt_food': {
        'name': 'Google Trends: Food', 
        'cmap': 'Oranges', 
        'line_color': 'orange',
        'fill_color': 'moccasin',
        'unit': 'Interest'
    },
    'gt_passport': {
        'name': 'Google Trends: Passport', 
        'cmap': 'Reds', 
        'line_color': 'crimson',
        'fill_color': 'mistyrose',
        'unit': 'Interest'
    },
}

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading data...")
panel = pd.read_csv(PANEL_FILE)
panel['period_start'] = pd.to_datetime(panel['period_start'])
panel['year_month'] = panel['period_start'].dt.strftime('%Y-%m')

# Load boundaries
gdf = gpd.read_file(BOUNDARIES_FILE)

# Find the ID column
id_col = 'adm2_pcode' if 'adm2_pcode' in gdf.columns else gdf.columns[gdf.columns.str.contains('pcode', case=False)][0]
name_col = 'adm2_name' if 'adm2_name' in gdf.columns else gdf.columns[gdf.columns.str.contains('name', case=False)][0]

print(f"\nBoundary columns: {gdf.columns.tolist()}")
print(f"Using ID column: {id_col}")
print(f"Panel unit_id values: {panel['unit_id'].unique()[:5]}...")
print(f"Boundary {id_col} values: {gdf[id_col].unique()[:5]}...")

# Auto-detect Gaza codes based on ADM1 or geography
# Gaza codes typically start with PS02 (Gaza governorate) vs PS01 (West Bank)
if 'adm1_pcode' in gdf.columns:
    # Gaza = PS02, West Bank = PS01
    gaza_gdf = gdf[gdf['adm1_pcode'].str.startswith('PS02', na=False)]
    westbank_gdf = gdf[gdf['adm1_pcode'].str.startswith('PS01', na=False)]
    GAZA_CODES = gaza_gdf[id_col].tolist()
    WESTBANK_CODES = westbank_gdf[id_col].tolist()
elif 'adm1_name' in gdf.columns:
    # Try by name
    gaza_names = ['gaza', 'north gaza', 'deir', 'khan', 'rafah']
    gaza_gdf = gdf[gdf['adm1_name'].str.lower().str.contains('|'.join(gaza_names), na=False)]
    westbank_gdf = gdf[~gdf[id_col].isin(gaza_gdf[id_col])]
    GAZA_CODES = gaza_gdf[id_col].tolist()
    WESTBANK_CODES = westbank_gdf[id_col].tolist()
else:
    # Fallback: use centroid longitude (Gaza is west of West Bank)
    gdf['centroid_x'] = gdf.geometry.centroid.x
    gaza_gdf = gdf[gdf['centroid_x'] < 34.6]  # Gaza is roughly < 34.6°E
    westbank_gdf = gdf[gdf['centroid_x'] >= 34.6]
    GAZA_CODES = gaza_gdf[id_col].tolist()
    WESTBANK_CODES = westbank_gdf[id_col].tolist()

print(f"\nDetected Gaza codes ({len(GAZA_CODES)}): {GAZA_CODES}")
print(f"Detected West Bank codes ({len(WESTBANK_CODES)}): {WESTBANK_CODES[:5]}...")

# Separate Gaza and West Bank in panel data
panel['is_gaza'] = panel['unit_id'].isin(GAZA_CODES)
gdf['is_gaza'] = gdf[id_col].isin(GAZA_CODES)

gaza_panel = panel[panel['is_gaza']].copy()
westbank_panel = panel[~panel['is_gaza']].copy()

gdf_gaza = gdf[gdf['is_gaza']].copy()
gdf_westbank = gdf[~gdf['is_gaza']].copy()

# Available outcomes - check BOTH full panel and Gaza-specific
available_full = [o for o in OUTCOMES.keys() if o in panel.columns and panel[o].notna().any()]
available_gaza = [o for o in OUTCOMES.keys() if o in gaza_panel.columns and gaza_panel[o].notna().any()]
available_wb = [o for o in OUTCOMES.keys() if o in westbank_panel.columns and westbank_panel[o].notna().any()]

# Use full list for iteration, but we'll skip Gaza-specific plots if no data
available = available_full

print(f"\n  Gaza: {gaza_panel['unit_id'].nunique()} districts, {len(gaza_panel)} obs")
print(f"  West Bank: {westbank_panel['unit_id'].nunique()} districts, {len(westbank_panel)} obs")
print(f"  Available outcomes (total): {', '.join(available_full)}")
print(f"  Available for Gaza: {', '.join(available_gaza)}")
if set(available_full) - set(available_gaza):
    print(f"  ⚠ Missing in Gaza: {', '.join(set(available_full) - set(available_gaza))}")

# If Gaza is empty, use all data as "Gaza" (the boundary file might only contain Gaza)
if len(gaza_panel) == 0:
    print("\n⚠ No Gaza-specific codes found. Using all data as Gaza region.")
    gaza_panel = panel.copy()
    gdf_gaza = gdf.copy()
    westbank_panel = pd.DataFrame()  # Empty
    gdf_westbank = gdf.iloc[0:0]  # Empty GeoDataFrame
    available_gaza = available_full

# =============================================================================
# 1. TIME SERIES - LINE PLOTS
# =============================================================================
print("\n--- Time Series (Line Plots) ---")

for outcome in available:
    cfg = OUTCOMES[outcome]
    
    # Aggregate by month for Gaza
    gaza_monthly = gaza_panel.groupby('period_start')[outcome].mean().reset_index()
    gaza_monthly = gaza_monthly.sort_values('period_start').dropna(subset=[outcome])
    
    if len(gaza_monthly) == 0:
        print(f"  ✗ ts_{outcome}_line.png (no data)")
        continue
    
    # West Bank for comparison (if available)
    if len(westbank_panel) > 0:
        wb_monthly = westbank_panel.groupby('period_start')[outcome].mean().reset_index()
        wb_monthly = wb_monthly.sort_values('period_start').dropna(subset=[outcome])
    else:
        wb_monthly = pd.DataFrame()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot Gaza with shaded area
    ax.fill_between(gaza_monthly['period_start'], gaza_monthly[outcome], 
                    alpha=0.3, color=cfg['line_color'], label='_nolegend_')
    ax.plot(gaza_monthly['period_start'], gaza_monthly[outcome], 
            color=cfg['line_color'], linewidth=2.5, marker='o', markersize=4,
            label='Gaza Strip')
    
    # Plot West Bank for comparison (if has data)
    if len(wb_monthly) > 0 and not wb_monthly[outcome].isna().all():
        ax.plot(wb_monthly['period_start'], wb_monthly[outcome], 
                color='gray', linewidth=2, linestyle='--', alpha=0.7,
                label='West Bank (Control)')
    
    # Pre/post means for Gaza
    pre_data = gaza_monthly[gaza_monthly['period_start'] < CONFLICT_DATE][outcome]
    post_data = gaza_monthly[(gaza_monthly['period_start'] >= CONFLICT_DATE) & 
                             (gaza_monthly['period_start'] < UNDP_DATE)][outcome]
    
    pre_mean = pre_data.mean() if len(pre_data) > 0 else np.nan
    post_mean = post_data.mean() if len(post_data) > 0 else np.nan
    
    if not pd.isna(pre_mean):
        ax.axhline(y=pre_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    if not pd.isna(post_mean):
        ax.axhline(y=post_mean, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Vertical lines for events
    ax.axvline(x=CONFLICT_DATE, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=UNDP_DATE, color='blue', linestyle='--', linewidth=2)
    
    # Shade periods (only if we have data in those periods)
    xmin = gaza_monthly['period_start'].min()
    xmax = gaza_monthly['period_start'].max()
    
    if xmin < CONFLICT_DATE:
        ax.axvspan(xmin, min(CONFLICT_DATE, xmax), alpha=0.1, color='green')
    if xmin < UNDP_DATE and xmax > CONFLICT_DATE:
        ax.axvspan(max(CONFLICT_DATE, xmin), min(UNDP_DATE, xmax), alpha=0.1, color='red')
    if xmax > UNDP_DATE:
        ax.axvspan(max(UNDP_DATE, xmin), xmax, alpha=0.1, color='blue')
    
    # Labels - position at top of axis with offset (below title)
    ax.text(CONFLICT_DATE, 1.01, 'Oct 07, 2023', transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')
    ax.text(UNDP_DATE, 1.01, 'UNDP\nJun 2025', transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='blue')
    
    # Calculate % change
    if pre_mean != 0 and not pd.isna(pre_mean) and not pd.isna(post_mean):
        pct_change = ((post_mean - pre_mean) / abs(pre_mean)) * 100
        title_suffix = f" ({pct_change:+.1f}% After Conflict)"
    else:
        title_suffix = ""
    
    ax.set_title(f'Gaza: {cfg["name"]}{title_suffix}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel(f'{cfg["name"]} ({cfg["unit"]})', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color=cfg['line_color'], linewidth=2.5, marker='o', label='Gaza Strip'),
    ]
    if len(wb_monthly) > 0:
        legend_elements.append(Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='West Bank'))
    if not pd.isna(pre_mean):
        legend_elements.append(Line2D([0], [0], color='green', linestyle=':', linewidth=1.5, label=f'Pre-Conflict Mean: {pre_mean:.2f}'))
    if not pd.isna(post_mean):
        legend_elements.append(Line2D([0], [0], color='red', linestyle=':', linewidth=1.5, label=f'Post-Conflict Mean: {post_mean:.2f}'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.subplots_adjust(top=0.85)  # Leave room for title and labels above plot
    plt.savefig(FIGURES_DIR / f'ts_{outcome}_line.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ ts_{outcome}_line.png")

# =============================================================================
# 2. MAPS - SPECIFIC MONTHS (Sep 2023, Sep 2024, Sep 2025)
# =============================================================================
print("\n--- Maps (Specific Months) ---")

# Get bounds for each region (for independent scaling)
gaza_bounds = gdf_gaza.total_bounds
wb_bounds = gdf_westbank.total_bounds

for outcome in available:
    cfg = OUTCOMES[outcome]
    
    # Get data for ALL regions (Gaza + West Bank) for each month
    month_data = {}
    for month in MAP_MONTHS:
        # Use FULL panel (both regions)
        month_df = panel[panel['year_month'] == month]
        if not month_df.empty and not month_df[outcome].isna().all():
            agg = month_df.groupby('unit_id')[outcome].mean().reset_index()
            # Merge with FULL boundary file (both regions)
            merged = gdf.merge(agg, left_on=id_col, right_on='unit_id', how='left')
            month_data[month] = merged
    
    if len(month_data) < 2:
        print(f"  ✗ map_{outcome}_months.png (insufficient data)")
        continue
    
    # Create figure with 2 rows: West Bank (top), Gaza (bottom)
    n_months = len(month_data)
    fig, axes = plt.subplots(2, n_months, figsize=(4*n_months + 1, 10),
                             gridspec_kw={'height_ratios': [1.2, 1], 'hspace': 0.05, 'wspace': 0.05})
    
    # Determine color scale across all months (for comparability)
    all_vals = pd.concat([df[[outcome]] for df in month_data.values() if outcome in df.columns]).dropna()
    if len(all_vals) == 0:
        plt.close()
        continue
    vmin, vmax = all_vals[outcome].quantile([0.02, 0.98])
    
    # Plot each month
    for col_idx, (month, gdf_month) in enumerate(month_data.items()):
        if outcome not in gdf_month.columns:
            axes[0, col_idx].set_visible(False)
            axes[1, col_idx].set_visible(False)
            continue
        
        # Separate Gaza and West Bank data
        gdf_wb = gdf_month[gdf_month[id_col].str.startswith('PS01', na=False)]
        gdf_gz = gdf_month[gdf_month[id_col].str.startswith('PS02', na=False)]
        
        # --- TOP ROW: West Bank ---
        ax_wb = axes[0, col_idx]
        if len(gdf_wb) > 0:
            gdf_wb.plot(column=outcome, ax=ax_wb, cmap=cfg['cmap'], 
                       vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.5,
                       legend=False, missing_kwds={'color': 'lightgray'})
        ax_wb.set_xlim(wb_bounds[0] - 0.03, wb_bounds[2] + 0.03)
        ax_wb.set_ylim(wb_bounds[1] - 0.03, wb_bounds[3] + 0.03)
        ax_wb.axis('off')
        
        # Title only on top row
        if month < "2023-10":
            period_label = "(Pre-Conflict)"
            title_color = 'green'
        elif month < "2025-06":
            period_label = "(Post-Conflict)"
            title_color = 'red'
        else:
            period_label = "(Post-UNDP)"
            title_color = 'blue'
        ax_wb.set_title(f'{month}\n{period_label}', fontsize=11, fontweight='bold', color=title_color, pad=2)
        
        # --- BOTTOM ROW: Gaza ---
        ax_gz = axes[1, col_idx]
        if len(gdf_gz) > 0:
            gdf_gz.plot(column=outcome, ax=ax_gz, cmap=cfg['cmap'], 
                       vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.5,
                       legend=False, missing_kwds={'color': 'lightgray'})
            # Add value labels
            for idx, row in gdf_gz.iterrows():
                if outcome in row and pd.notna(row[outcome]):
                    c = row.geometry.centroid
                    val_str = f'{row[outcome]:.1f}' if abs(row[outcome]) < 100 else f'{row[outcome]:.0f}'
                    ax_gz.annotate(val_str, xy=(c.x, c.y), ha='center', fontsize=8, fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        ax_gz.set_xlim(gaza_bounds[0] - 0.02, gaza_bounds[2] + 0.02)
        ax_gz.set_ylim(gaza_bounds[1] - 0.02, gaza_bounds[3] + 0.02)
        ax_gz.axis('off')
    
    # Row labels - positioned closer to maps
    fig.text(0.08, 0.68, 'West Bank\n(Control)', ha='center', va='center', fontsize=10, 
             fontweight='bold', rotation=90)
    fig.text(0.08, 0.22, 'Gaza\n(Treatment)', ha='center', va='center', fontsize=10, 
             fontweight='bold', rotation=90, color='darkred')
    
    # Colorbar at bottom
    sm = plt.cm.ScalarMappable(cmap=cfg['cmap'], norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'{cfg["name"]} ({cfg["unit"]})', fontsize=11)
    
    fig.suptitle(f'Palestine: {cfg["name"]} - Monthly Comparison', 
                fontsize=13, fontweight='bold', y=0.97)
    
    plt.savefig(FIGURES_DIR / f'map_{outcome}_months.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ map_{outcome}_months.png")

# =============================================================================
# 3. GAZA VS WEST BANK COMPARISON (if West Bank data available)
# =============================================================================
if len(westbank_panel) > 0:
    print("\n--- Gaza vs West Bank Comparison ---")
    
    for outcome in ['nightlights', 'ndvi', 'price_bread']:
        if outcome not in available:
            continue
        
        cfg = OUTCOMES[outcome]
        
        # Monthly aggregates
        gaza_monthly = gaza_panel.groupby('period_start')[outcome].mean().reset_index()
        wb_monthly = westbank_panel.groupby('period_start')[outcome].mean().reset_index()
        
        gaza_monthly = gaza_monthly.sort_values('period_start').dropna(subset=[outcome])
        wb_monthly = wb_monthly.sort_values('period_start').dropna(subset=[outcome])
        
        if len(gaza_monthly) == 0 or len(wb_monthly) == 0:
            continue
        
        # Normalize to Jan 2023 = 100
        gaza_base = gaza_monthly[gaza_monthly['period_start'] == '2023-01-01'][outcome].values
        wb_base = wb_monthly[wb_monthly['period_start'] == '2023-01-01'][outcome].values
        
        if len(gaza_base) > 0 and gaza_base[0] != 0:
            gaza_monthly['normalized'] = (gaza_monthly[outcome] / gaza_base[0]) * 100
        else:
            gaza_monthly['normalized'] = gaza_monthly[outcome]
        
        if len(wb_base) > 0 and wb_base[0] != 0:
            wb_monthly['normalized'] = (wb_monthly[outcome] / wb_base[0]) * 100
        else:
            wb_monthly['normalized'] = wb_monthly[outcome]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot both regions
        ax.plot(gaza_monthly['period_start'], gaza_monthly['normalized'], 
                color='red', linewidth=2.5, marker='o', markersize=4, label='Gaza Strip')
        ax.plot(wb_monthly['period_start'], wb_monthly['normalized'], 
                color='blue', linewidth=2.5, marker='s', markersize=4, label='West Bank')
        
        # Event lines
        ax.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
        ax.axvline(x=UNDP_DATE, color='green', linestyle='--', linewidth=2)
        ax.axhline(y=100, color='gray', linestyle=':', linewidth=1)
        
        # Labels - position above plot area (below title)
        ax.text(CONFLICT_DATE, 1.01, 'Oct 07, 2023', transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(UNDP_DATE, 1.01, 'UNDP\nJun 2025', transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='green')
        
        ax.set_title(f'{cfg["name"]}: Gaza vs West Bank (Indexed to Jan 2023 = 100)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Index (Jan 2023 = 100)', fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        plt.subplots_adjust(top=0.85)  # Leave room for title and labels above plot
        plt.savefig(FIGURES_DIR / f'comparison_{outcome}_gaza_vs_wb.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ comparison_{outcome}_gaza_vs_wb.png")
else:
    print("\n--- Skipping Gaza vs West Bank (no West Bank data) ---")

# =============================================================================
# 4. EVENT STUDY (Improved)
# =============================================================================
print("\n--- Event Study ---")

# Calculate months since conflict
gaza_panel['months_since_conflict'] = (
    (gaza_panel['period_start'].dt.year - CONFLICT_DATE.year) * 12 + 
    (gaza_panel['period_start'].dt.month - CONFLICT_DATE.month)
)

es_outcomes = [o for o in ['nightlights', 'ndvi', 'vh'] if o in available]

if es_outcomes:
    fig, axes = plt.subplots(1, len(es_outcomes), figsize=(5*len(es_outcomes), 5))
    if len(es_outcomes) == 1:
        axes = [axes]
    
    for ax, outcome in zip(axes, es_outcomes):
        cfg = OUTCOMES[outcome]
        
        # Calculate monthly means
        monthly = gaza_panel.groupby('months_since_conflict')[outcome].agg(['mean', 'std', 'count']).reset_index()
        monthly['se'] = monthly['std'] / np.sqrt(monthly['count'])
        
        # Normalize to t=-1
        baseline = monthly[monthly['months_since_conflict'] == -1]['mean'].values
        if len(baseline) > 0 and baseline[0] != 0:
            monthly['effect'] = ((monthly['mean'] - baseline[0]) / abs(baseline[0])) * 100
            monthly['effect_se'] = (monthly['se'] / abs(baseline[0])) * 100
        else:
            monthly['effect'] = monthly['mean']
            monthly['effect_se'] = monthly['se']
        
        # Plot
        colors = ['green' if x < 0 else 'red' for x in monthly['months_since_conflict']]
        ax.bar(monthly['months_since_conflict'], monthly['effect'], color=colors, alpha=0.7, edgecolor='white')
        
        # Error bars
        ax.errorbar(monthly['months_since_conflict'], monthly['effect'], 
                    yerr=monthly['effect_se']*1.96, fmt='none', color='black', capsize=2, alpha=0.5)
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        
        ax.set_title(cfg['name'], fontsize=12, fontweight='bold')
        ax.set_xlabel('Months Since Conflict', fontsize=10)
        ax.set_ylabel('% Change from Baseline', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Gaza: Event Study (Baseline = Sep 2023)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'event_study_improved.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ event_study_improved.png")

# =============================================================================
# 5. SUMMARY STATISTICS TABLE
# =============================================================================
print("\n--- Summary Statistics ---")

summary_rows = []
for outcome in available:
    cfg = OUTCOMES[outcome]
    
    pre = gaza_panel[gaza_panel['period_start'] < CONFLICT_DATE][outcome]
    post = gaza_panel[(gaza_panel['period_start'] >= CONFLICT_DATE) & 
                      (gaza_panel['period_start'] < UNDP_DATE)][outcome]
    post_undp = gaza_panel[gaza_panel['period_start'] >= UNDP_DATE][outcome]
    
    pre_mean = pre.mean()
    post_mean = post.mean()
    
    summary_rows.append({
        'Outcome': cfg['name'],
        'Pre-Conflict Mean': f"{pre_mean:.2f}" if not pd.isna(pre_mean) else "N/A",
        'Pre-Conflict SD': f"{pre.std():.2f}" if not pd.isna(pre.std()) else "N/A",
        'Post-Conflict Mean': f"{post_mean:.2f}" if not pd.isna(post_mean) else "N/A",
        'Post-Conflict SD': f"{post.std():.2f}" if not pd.isna(post.std()) else "N/A",
        'Change (%)': f"{((post_mean - pre_mean) / abs(pre_mean) * 100):+.1f}%" if (pre_mean != 0 and not pd.isna(pre_mean) and not pd.isna(post_mean)) else "N/A",
        'Post-UNDP Mean': f"{post_undp.mean():.2f}" if len(post_undp) > 0 and not pd.isna(post_undp.mean()) else "N/A",
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(FIGURES_DIR / 'summary_statistics.csv', index=False)
print("  ✓ summary_statistics.csv")

# =============================================================================
# COMPLETE
# =============================================================================
n_figures = len(list(FIGURES_DIR.glob("*.png")))
print(f"\n{'='*60}")
print(f"✓ Visualizations complete: {FIGURES_DIR}")
print(f"  Generated {n_figures} figures")
print(f"{'='*60}")
