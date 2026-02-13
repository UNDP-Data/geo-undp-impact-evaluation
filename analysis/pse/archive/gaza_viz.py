#!/usr/bin/env python3
"""
gaza_viz.py - Standalone Gaza Visualization Script
==================================================
Run after preprocessing to generate all Gaza-specific visualizations.

Usage:
    python gaza_viz.py
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Gaza district codes (PS02xx = Gaza Strip, PS01xx = West Bank)
GAZA_CODES = ['PS0255', 'PS0260', 'PS0265', 'PS0270', 'PS0275']

# Key dates
ATTACK_DATE = pd.Timestamp('2023-10-07')
UNDP_DATE = pd.Timestamp('2025-06-01')
DISASTER_TYPE = 'Conflict'

# Paths
OUTPUT_DIR = Path('outputs/pse')
FIGURES_DIR = OUTPUT_DIR / 'figures'
PANEL_PATH = OUTPUT_DIR / 'panel_data.csv'
BOUNDARY_PATH = Path('data/gaza/boundaries/gaza_admin2.geojson')

# Outcome configurations
OUTCOMES = {
    'nightlights': {'name': 'Nighttime Lights', 'line': 'darkorange', 'fill': 'gold', 'cmap': 'YlOrBr', 'agg': 'mean'},
    'ndvi': {'name': 'NDVI (Vegetation)', 'line': 'darkgreen', 'fill': 'lightgreen', 'cmap': 'YlGn', 'agg': 'mean'},
    'conflict_events': {'name': 'Conflict Events', 'line': 'darkred', 'fill': 'lightcoral', 'cmap': 'YlOrRd', 'agg': 'sum'},
    'rainfall': {'name': 'Rainfall (mm)', 'line': 'darkblue', 'fill': 'lightblue', 'cmap': 'Blues', 'agg': 'mean'},
    'price_bread': {'name': 'Bread Price', 'line': 'purple', 'fill': 'lavender', 'cmap': 'RdYlGn_r', 'agg': 'mean'},
    'vv': {'name': 'SAR VV Backscatter (Sentinel-1)', 'line': 'teal', 'fill': 'lightcyan', 'cmap': 'viridis', 'agg': 'mean'},
    'vh': {'name': 'SAR VH Backscatter (Sentinel-1)', 'line': 'navy', 'fill': 'lightsteelblue', 'cmap': 'viridis', 'agg': 'mean'},
}

# Which outcomes to create maps for
MAP_OUTCOMES = ['nightlights', 'ndvi', 'conflict_events', 'rainfall', 'vv', 'vh']

# =============================================================================
# LOAD DATA
# =============================================================================

print("="*60)
print("GAZA IMPACT VISUALIZATIONS")
print("="*60)

# Load and filter data
df = pd.read_csv(PANEL_PATH)
df['period_start'] = pd.to_datetime(df['period_start'])
df = df[df['unit_id'].isin(GAZA_CODES)].copy()

# Load and filter boundaries
gdf = gpd.read_file(BOUNDARY_PATH)
gdf = gdf[gdf['adm2_pcode'].isin(GAZA_CODES)].copy()

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Attack Date: {ATTACK_DATE.strftime('%B %d, %Y')}")
print(f"UNDP Intervention: {UNDP_DATE.strftime('%B %Y')}")
print(f"Data: {len(df):,} obs, {df['unit_id'].nunique()} districts")
print(f"Period: {df['period_start'].min().strftime('%Y-%m-%d')} to {df['period_start'].max().strftime('%Y-%m-%d')}")

# Detect available outcomes
available = [o for o in OUTCOMES.keys() if o in df.columns and df[o].notna().sum() > 10]
print(f"Outcomes: {', '.join(available)}")

# =============================================================================
# AGGREGATE DATA
# =============================================================================

# Monthly aggregation
agg_dict = {o: OUTCOMES[o]['agg'] for o in available}
monthly = df.groupby('period_start').agg(agg_dict).reset_index()

# Period aggregations for maps
pre = df[df['period_start'] < ATTACK_DATE]
post = df[(df['period_start'] >= ATTACK_DATE) & (df['period_start'] < UNDP_DATE)]
undp = df[df['period_start'] >= UNDP_DATE]

pre_agg = pre.groupby('unit_id').mean(numeric_only=True).reset_index()
post_agg = post.groupby('unit_id').mean(numeric_only=True).reset_index()
undp_agg = undp.groupby('unit_id').mean(numeric_only=True).reset_index()

gdf_pre = gdf.merge(pre_agg, left_on='adm2_pcode', right_on='unit_id', how='left')
gdf_post = gdf.merge(post_agg, left_on='adm2_pcode', right_on='unit_id', how='left')
gdf_undp = gdf.merge(undp_agg, left_on='adm2_pcode', right_on='unit_id', how='left')

has_undp = len(undp) > 0

# =============================================================================
# 1. TIME SERIES PLOTS
# =============================================================================
print("\n--- Time Series ---")

for outcome in available:
    if 'log_' in outcome:
        continue
    
    cfg = OUTCOMES[outcome]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.fill_between(monthly['period_start'], monthly[outcome].fillna(0), 
                   alpha=0.3, color=cfg['fill'])
    ax.plot(monthly['period_start'], monthly[outcome], color=cfg['line'], 
           linewidth=2.5, marker='o', markersize=5, 
           markerfacecolor='white', markeredgecolor=cfg['line'])
    
    # Key dates
    ax.axvline(ATTACK_DATE, color='red', linestyle='--', linewidth=2.5, zorder=5)
    ax.axvline(UNDP_DATE, color='blue', linestyle='--', linewidth=2.5, zorder=5)
    
    # Calculate stats
    pre_vals = monthly[monthly['period_start'] < ATTACK_DATE][outcome].dropna()
    post_vals = monthly[(monthly['period_start'] >= ATTACK_DATE) & 
                        (monthly['period_start'] < UNDP_DATE)][outcome].dropna()
    
    title = f"Gaza: {cfg['name']}"
    legend_elements = []
    
    if len(pre_vals) > 0 and len(post_vals) > 0:
        pre_mean, post_mean = pre_vals.mean(), post_vals.mean()
        
        ax.axhline(pre_mean, color='green', linestyle=':', alpha=0.6, linewidth=1.5)
        ax.axhline(post_mean, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
        
        if pre_mean != 0:
            pct = ((post_mean - pre_mean) / abs(pre_mean)) * 100
            sign = '+' if pct > 0 else ''
            title = f"Gaza: {cfg['name']} ({sign}{pct:.1f}% After {DISASTER_TYPE})"
        
        legend_elements = [
            Line2D([0], [0], color='green', linestyle=':', linewidth=2, 
                   label=f'Pre-{DISASTER_TYPE} Mean: {pre_mean:.2f}'),
            Line2D([0], [0], color='red', linestyle=':', linewidth=2, 
                   label=f'Post-{DISASTER_TYPE} Mean: {post_mean:.2f}'),
        ]
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Annotations
    y_max = ax.get_ylim()[1]
    ax.annotate(f'{ATTACK_DATE.strftime("%b %d, %Y")}', 
               xy=(ATTACK_DATE, y_max * 0.95), ha='center', 
               color='red', fontweight='bold', fontsize=9)
    ax.annotate(f'UNDP\n{UNDP_DATE.strftime("%b %Y")}', 
               xy=(UNDP_DATE, y_max * 0.85), ha='center', 
               color='blue', fontweight='bold', fontsize=9)
    
    ax.set_ylabel(cfg['name'], fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'timeseries_{outcome}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {outcome}")

# =============================================================================
# 2. EVENT STUDY
# =============================================================================
print("\n--- Event Study ---")

df['event_time'] = ((df['period_start'].dt.year - ATTACK_DATE.year) * 12 + 
                    (df['period_start'].dt.month - ATTACK_DATE.month))

event_data = df.groupby('event_time').agg(agg_dict).reset_index()

key_outcomes = [o for o in ['nightlights', 'ndvi', 'conflict_events', 'rainfall'] if o in available]

if key_outcomes:
    n = len(key_outcomes)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]
    
    for ax, outcome in zip(axes, key_outcomes):
        cfg = OUTCOMES[outcome]
        data = event_data[['event_time', outcome]].copy()
        
        pre_data = data[data['event_time'] < 0][outcome]
        baseline = pre_data.mean() if len(pre_data) > 0 and pre_data.mean() != 0 else 1
        
        if cfg['agg'] == 'sum':
            data['plot_val'] = data[outcome]
            ylabel = f'{cfg["name"]} (Monthly)'
        else:
            data['plot_val'] = (data[outcome] / baseline - 1) * 100
            ylabel = '% Change from Baseline'
        
        colors = ['forestgreen' if t < 0 else 'crimson' for t in data['event_time']]
        ax.bar(data['event_time'], data['plot_val'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=2.5)
        
        if cfg['agg'] != 'sum':
            ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        
        ax.set_xlabel(f'Months Since {DISASTER_TYPE}', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(cfg['name'], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        legend_elements = [
            Patch(facecolor='forestgreen', edgecolor='black', alpha=0.7, label=f'Pre-{DISASTER_TYPE}'),
            Patch(facecolor='crimson', edgecolor='black', alpha=0.7, label=f'Post-{DISASTER_TYPE}'),
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=8)
    
    plt.suptitle(f'Gaza: Event Study - Months Relative to October 7, 2023 {DISASTER_TYPE}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'event_study.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ event_study.png")

# =============================================================================
# 3. DISTRIBUTIONS
# =============================================================================
print("\n--- Distributions ---")

dist_outcomes = [o for o in available if 'log_' not in o][:6]

if dist_outcomes:
    n_cols = min(3, len(dist_outcomes))
    n_rows = (len(dist_outcomes) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = np.array(axes).flatten()
    
    for idx, outcome in enumerate(dist_outcomes):
        ax = axes[idx]
        cfg = OUTCOMES[outcome]
        
        pre_data = df[df['period_start'] < ATTACK_DATE][outcome].dropna()
        post_data = df[df['period_start'] >= ATTACK_DATE][outcome].dropna()
        
        if len(pre_data) < 5 or len(post_data) < 5:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(cfg['name'], fontsize=11, fontweight='bold')
            continue
        
        all_data = pd.concat([pre_data, post_data])
        bins = np.linspace(all_data.quantile(0.01), all_data.quantile(0.99), 30)
        
        ax.hist(pre_data, bins=bins, alpha=0.6, color='forestgreen', label=f'Pre (n={len(pre_data)})', density=True)
        ax.hist(post_data, bins=bins, alpha=0.6, color='crimson', label=f'Post (n={len(post_data)})', density=True)
        
        pre_mean, post_mean = pre_data.mean(), post_data.mean()
        ax.axvline(pre_mean, color='darkgreen', linestyle='--', linewidth=2.5)
        ax.axvline(post_mean, color='darkred', linestyle='--', linewidth=2.5)
        
        if pre_mean != 0:
            pct = ((post_mean - pre_mean) / abs(pre_mean)) * 100
            sign = '+' if pct > 0 else ''
            title = f"{cfg['name']} ({sign}{pct:.1f}%)"
        else:
            title = cfg['name']
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(cfg['name'], fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        
        legend_elements = [
            Patch(facecolor='forestgreen', alpha=0.6, label=f'Pre (n={len(pre_data)})'),
            Patch(facecolor='crimson', alpha=0.6, label=f'Post (n={len(post_data)})'),
            Line2D([0], [0], color='darkgreen', linestyle='--', linewidth=2, label=f'Pre Mean: {pre_mean:.2f}'),
            Line2D([0], [0], color='darkred', linestyle='--', linewidth=2, label=f'Post Mean: {post_mean:.2f}'),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(dist_outcomes), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Gaza: Distributions Before vs After October 7, 2023 {DISASTER_TYPE}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'distributions.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ distributions.png")

# =============================================================================
# 4. MAPS (THREE PERIODS)
# =============================================================================
print("\n--- Maps ---")

# Create maps for all configured outcomes (not just nightlights, ndvi, conflict)
map_outcomes = [o for o in MAP_OUTCOMES if o in available]

for outcome in map_outcomes:
    cfg = OUTCOMES[outcome]
    
    if gdf_pre[outcome].isna().all() and gdf_post[outcome].isna().all():
        continue
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    fig.subplots_adjust(top=0.85, bottom=0.15)
    
    all_vals = pd.concat([gdf_pre[[outcome]], gdf_post[[outcome]]]).dropna()
    if len(all_vals) == 0:
        plt.close()
        continue
    
    vmin, vmax = all_vals[outcome].quantile([0.02, 0.98])
    
    # Pre-attack
    gdf_pre.plot(column=outcome, ax=axes[0], cmap=cfg['cmap'], vmin=vmin, vmax=vmax,
                edgecolor='black', linewidth=1, legend=False, missing_kwds={'color': 'lightgray'})
    axes[0].set_title(f'Pre-{DISASTER_TYPE}\n(Jan - Sep 2023)', fontsize=11, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    for idx, row in gdf_pre.iterrows():
        if pd.notna(row[outcome]):
            c = row.geometry.centroid
            val_str = f'{row[outcome]:.1f}' if abs(row[outcome]) < 100 else f'{row[outcome]:.0f}'
            axes[0].annotate(val_str, xy=(c.x, c.y), ha='center', fontsize=9, fontweight='bold')
    
    # Post-attack (before UNDP)
    gdf_post.plot(column=outcome, ax=axes[1], cmap=cfg['cmap'], vmin=vmin, vmax=vmax,
                 edgecolor='black', linewidth=1, legend=False, missing_kwds={'color': 'lightgray'})
    axes[1].set_title(f'Post-{DISASTER_TYPE}\n(Oct 2023 - May 2025)', fontsize=11, fontweight='bold', pad=10)
    axes[1].axis('off')
    
    for idx, row in gdf_post.iterrows():
        if pd.notna(row[outcome]):
            c = row.geometry.centroid
            val_str = f'{row[outcome]:.1f}' if abs(row[outcome]) < 100 else f'{row[outcome]:.0f}'
            axes[1].annotate(val_str, xy=(c.x, c.y), ha='center', fontsize=9, fontweight='bold')
    
    # Post-UNDP
    if has_undp and not gdf_undp[outcome].isna().all():
        gdf_undp.plot(column=outcome, ax=axes[2], cmap=cfg['cmap'], vmin=vmin, vmax=vmax,
                     edgecolor='black', linewidth=1, legend=False, missing_kwds={'color': 'lightgray'})
        axes[2].set_title(f'Post-UNDP\n(Jun 2025+)', fontsize=11, fontweight='bold', pad=10)
        
        for idx, row in gdf_undp.iterrows():
            if pd.notna(row[outcome]):
                c = row.geometry.centroid
                val_str = f'{row[outcome]:.1f}' if abs(row[outcome]) < 100 else f'{row[outcome]:.0f}'
                axes[2].annotate(val_str, xy=(c.x, c.y), ha='center', fontsize=9, fontweight='bold')
    else:
        gdf.plot(ax=axes[2], color='lightgray', edgecolor='black', linewidth=1)
        axes[2].set_title(f'Post-UNDP\n(Jun 2025+)', fontsize=11, fontweight='bold', pad=10)
        axes[2].text(0.5, 0.5, 'Data not yet\navailable', transform=axes[2].transAxes,
                    ha='center', va='center', fontsize=12, style='italic')
    axes[2].axis('off')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cfg['cmap'], norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(cfg['name'], fontsize=11)
    
    fig.suptitle(f"Gaza: {cfg['name']} - Impact Assessment", fontsize=14, fontweight='bold', y=0.94)
    
    plt.savefig(FIGURES_DIR / f'map_{outcome}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ map_{outcome}.png")

# =============================================================================
# 5. CONFLICT TIMELINE
# =============================================================================
if 'conflict_events' in available:
    print("\n--- Conflict Timeline ---")
    
    monthly['conflict_events'] = monthly['conflict_events'].fillna(0)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = []
    for d in monthly['period_start']:
        if d < ATTACK_DATE:
            colors.append('forestgreen')
        elif d < UNDP_DATE:
            colors.append('crimson')
        else:
            colors.append('steelblue')
    
    ax.bar(monthly['period_start'], monthly['conflict_events'], 
           width=20, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.axvline(ATTACK_DATE, color='black', linestyle='--', linewidth=2.5)
    ax.axvline(UNDP_DATE, color='blue', linestyle='--', linewidth=2.5)
    
    max_events = monthly['conflict_events'].max()
    ax.annotate(f'{ATTACK_DATE.strftime("%b %d, %Y")}', 
               xy=(ATTACK_DATE, max_events * 0.95),
               fontsize=10, ha='center', fontweight='bold')
    ax.annotate(f'UNDP\n{UNDP_DATE.strftime("%b %Y")}', 
               xy=(UNDP_DATE, max_events * 0.8),
               fontsize=10, ha='center', fontweight='bold', color='blue')
    
    legend_elements = [
        Patch(facecolor='forestgreen', edgecolor='black', alpha=0.8, label=f'Pre-{DISASTER_TYPE}'),
        Patch(facecolor='crimson', edgecolor='black', alpha=0.8, label=f'Post-{DISASTER_TYPE} (Pre-UNDP)'),
        Patch(facecolor='steelblue', edgecolor='black', alpha=0.8, label='Post-UNDP'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_title('Gaza: Conflict Events (ACLED)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Monthly Events', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'conflict_timeline.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ conflict_timeline.png")

# =============================================================================
# DONE
# =============================================================================
print(f"\n{'='*60}")
print(f"✓ All visualizations saved to: {FIGURES_DIR}")
print(f"  Generated {len(list(FIGURES_DIR.glob('*.png')))} figures")
print(f"{'='*60}")
