#!/usr/bin/env python3
"""
pipeline/visualizations.py
==========================
Impact evaluation visualizations - GAZA FORMAT.

Creates time series with:
- Continuous line plot with 95% CI ribbon
- Pre/post mean dotted horizontal lines  
- Vertical dashed lines at disaster + intervention
- UNDP project strip at bottom with ML categories
- Pre/post period background shading
- % change in title

Usage:
    python -m pipeline.visualizations --config config/myanmar.yaml
    python -m pipeline.visualizations --config config/gaza.yaml
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from collections import defaultdict
import yaml
import argparse
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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

# Outcome configurations
OUTCOME_CONFIG = {
    'nightlights': {'name': 'Nighttime Lights (VIIRS)', 'line': 'darkblue', 'fill': 'blue', 'unit': 'nW/cm²/sr', 'agg': 'mean'},
    'ndvi': {'name': 'Vegetation Index (NDVI)', 'line': 'forestgreen', 'fill': 'lightgreen', 'unit': 'Index', 'agg': 'mean'},
    'conflict_events': {'name': 'Conflict Events', 'line': 'crimson', 'fill': 'lightcoral', 'unit': 'Count', 'agg': 'sum'},
    'conflict_fatalities': {'name': 'Conflict Fatalities', 'line': 'darkred', 'fill': 'salmon', 'unit': 'Count', 'agg': 'sum'},
    'rainfall': {'name': 'Rainfall', 'line': 'steelblue', 'fill': 'lightblue', 'unit': 'mm', 'agg': 'mean'},
    'price_bread': {'name': 'Bread Price', 'line': 'purple', 'fill': 'lavender', 'unit': 'Local Currency', 'agg': 'mean'},
    'price_rice': {'name': 'Rice Price', 'line': 'olive', 'fill': 'khaki', 'unit': 'Local Currency', 'agg': 'mean'},
    'price_oil': {'name': 'Oil Price', 'line': 'darkorange', 'fill': 'peachpuff', 'unit': 'Local Currency', 'agg': 'mean'},
    'vv': {'name': 'SAR Backscatter (VV)', 'line': 'dimgray', 'fill': 'lightgray', 'unit': 'dB', 'agg': 'mean'},
    'vh': {'name': 'SAR Backscatter (VH)', 'line': 'slategray', 'fill': 'silver', 'unit': 'dB', 'agg': 'mean'},
    'rvi': {'name': 'Radar Vegetation Index', 'line': 'darkgreen', 'fill': 'palegreen', 'unit': 'Index', 'agg': 'mean'},
}


def main():
    parser = argparse.ArgumentParser(description='Generate impact evaluation visualizations')
    parser.add_argument('--config', '-c', required=True, help='Path to country config YAML')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    country_name = config.get('country_name', 'Unknown')
    country_code = config.get('country_code', 'XXX').lower()
    disaster_date = pd.Timestamp(config['events']['disaster_date'])
    intervention_date = pd.Timestamp(config['events']['intervention_start'])
    disaster_type = config['events'].get('disaster_type', 'Disaster').title()
    
    # Paths
    data_dir = Path(f"data/{country_code}")
    output_dir = Path(f"outputs/{country_code}")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"VISUALIZATIONS: {country_name}")
    print(f"{'='*60}")
    print(f"Disaster: {disaster_date.strftime('%B %d, %Y')} ({disaster_type})")
    print(f"Intervention: {intervention_date.strftime('%B %Y')}")
    
    # Load panel data
    panel_path = output_dir / "panel_data.csv"
    if not panel_path.exists():
        print(f"⚠ Panel data not found: {panel_path}")
        return
    
    panel = pd.read_csv(panel_path)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    
    print(f"Data: {len(panel)} obs, {panel['unit_id'].nunique()} units")
    print(f"Period: {panel['period_start'].min().date()} to {panel['period_start'].max().date()}")
    
    # Load UNDP projects (for ML categories)
    undp_path = data_dir / "external" / "undp_projects.csv"
    undp_df = None
    if undp_path.exists():
        undp_df = pd.read_csv(undp_path)
        undp_df['start_date'] = pd.to_datetime(undp_df['start_date'], errors='coerce')
        if 'is_likely_umbrella' in undp_df.columns:
            undp_df = undp_df[undp_df['is_likely_umbrella'] != True]
        print(f"UNDP Projects: {len(undp_df)}")
    
    # Find available outcomes
    outcomes = [o for o in OUTCOME_CONFIG.keys() if o in panel.columns and panel[o].notna().any()]
    print(f"Outcomes: {', '.join(outcomes)}")
    
    # =========================================================================
    # TIME SERIES (GAZA FORMAT)
    # =========================================================================
    print("\n--- Time Series ---")
    
    for outcome in outcomes:
        cfg = OUTCOME_CONFIG[outcome]
        
        # Aggregate monthly
        if cfg['agg'] == 'sum':
            monthly = panel.groupby('period_start')[outcome].sum().reset_index()
        else:
            monthly = panel.groupby('period_start').agg({
                outcome: ['mean', 'std', 'count']
            }).reset_index()
            monthly.columns = ['period_start', 'mean', 'std', 'count']
            monthly['se'] = monthly['std'] / np.sqrt(monthly['count'])
            monthly = monthly.rename(columns={'mean': outcome})
        
        monthly = monthly.sort_values('period_start')
        
        if monthly[outcome].isna().all():
            print(f"  ✗ ts_{outcome}.png (no data)")
            continue
        
        # Create figure with main panel + UNDP strip
        fig = plt.figure(figsize=(16, 9))
        ax_main = fig.add_axes([0.08, 0.22, 0.85, 0.68])
        ax_strip = fig.add_axes([0.08, 0.12, 0.85, 0.06])
        
        # === MAIN TIME SERIES ===
        
        # Fill ribbon (95% CI)
        if 'se' in monthly.columns:
            ax_main.fill_between(
                monthly['period_start'],
                monthly[outcome] - 1.96 * monthly['se'],
                monthly[outcome] + 1.96 * monthly['se'],
                alpha=0.3, color=cfg['fill']
            )
        else:
            ax_main.fill_between(monthly['period_start'], monthly[outcome],
                                alpha=0.3, color=cfg['fill'])
        
        # Main line with markers
        ax_main.plot(monthly['period_start'], monthly[outcome],
                    color=cfg['line'], linewidth=2.5, marker='o', markersize=5,
                    label=country_name)
        
        # Pre/post means
        pre_data = monthly[monthly['period_start'] < disaster_date][outcome]
        post_data = monthly[monthly['period_start'] >= disaster_date][outcome]
        
        pre_mean = pre_data.mean() if len(pre_data) > 0 else np.nan
        post_mean = post_data.mean() if len(post_data) > 0 else np.nan
        
        # Dotted horizontal lines for means
        if not pd.isna(pre_mean):
            ax_main.axhline(y=pre_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        if not pd.isna(post_mean):
            ax_main.axhline(y=post_mean, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # Vertical dashed lines
        ax_main.axvline(x=disaster_date, color='red', linestyle='--', linewidth=2.5)
        
        # Period shading
        xmin = monthly['period_start'].min()
        xmax = monthly['period_start'].max()
        
        if xmin < disaster_date:
            ax_main.axvspan(xmin, min(disaster_date, xmax), alpha=0.1, color='green')
        if xmax > disaster_date:
            ax_main.axvspan(max(disaster_date, xmin), xmax, alpha=0.08, color='red')
        
        # Event label
        y_top = ax_main.get_ylim()[1]
        ax_main.text(disaster_date, y_top * 0.95,
                    f'{disaster_type}\n{disaster_date.strftime("%b %d, %Y")}',
                    ha='center', va='top', fontsize=9, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
        
        # Calculate % change
        if pre_mean != 0 and not pd.isna(pre_mean) and not pd.isna(post_mean):
            pct_change = ((post_mean - pre_mean) / abs(pre_mean)) * 100
            title_suffix = f" ({pct_change:+.1f}% After {disaster_type})"
        else:
            title_suffix = ""
        
        ax_main.set_title(f'{country_name}: {cfg["name"]}{title_suffix}',
                         fontsize=14, fontweight='bold', pad=20)
        ax_main.set_ylabel(f'{cfg["name"]} ({cfg["unit"]})', fontsize=11)
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax_main.set_xticklabels([])
        ax_main.grid(True, alpha=0.3)
        
        # Legend
        legend_elements = [
            Line2D([0], [0], color=cfg['line'], linewidth=2.5, marker='o', label=country_name),
        ]
        if not pd.isna(pre_mean):
            legend_elements.append(Line2D([0], [0], color='green', linestyle=':',
                                         linewidth=1.5, label=f'Pre-{disaster_type} Mean: {pre_mean:.2f}'))
        if not pd.isna(post_mean):
            legend_elements.append(Line2D([0], [0], color='red', linestyle=':',
                                         linewidth=1.5, label=f'Post-{disaster_type} Mean: {post_mean:.2f}'))
        
        ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # === UNDP PROJECT STRIP ===
        ax_strip.set_xlim(ax_main.get_xlim())
        ax_strip.set_ylim(0, 1)
        
        # Period shading on strip
        if xmin < disaster_date:
            ax_strip.axvspan(xmin, min(disaster_date, xmax), alpha=0.15, color='green')
        if xmax > disaster_date:
            ax_strip.axvspan(max(disaster_date, xmin), xmax, alpha=0.1, color='red')
        
        ax_strip.axvline(x=disaster_date, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Plot UNDP project markers (using ML categories!)
        categories_shown = set()
        
        if undp_df is not None:
            markers_df = undp_df[
                (undp_df['start_date'] >= xmin) & 
                (undp_df['start_date'] <= xmax)
            ]
            
            if len(markers_df) > 0:
                monthly_markers = defaultdict(list)
                for _, row in markers_df.iterrows():
                    month_key = row['start_date'].to_period('M').to_timestamp()
                    category = row.get('ml_category', 'Other / Unspecified')
                    monthly_markers[month_key].append(category)
                
                for month, categories in sorted(monthly_markers.items()):
                    for i, category in enumerate(categories[:3]):
                        color = CATEGORY_COLORS.get(category, '#9E9E9E')
                        y_pos = 0.5 + (i - len(categories)/2 + 0.5) * 0.25
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
            legend_patches = [
                mpatches.Patch(color=CATEGORY_COLORS.get(cat, '#9E9E9E'), label=cat[:25])
                for cat in sorted(categories_shown)
            ]
            fig.legend(handles=legend_patches, loc='lower center',
                       bbox_to_anchor=(0.5, -0.02), ncol=min(5, len(legend_patches)),
                       fontsize=7, frameon=False)
        
        plt.savefig(figures_dir / f'ts_{outcome}.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ ts_{outcome}.png")
    
    # =========================================================================
    # CONFLICT TIMELINE
    # =========================================================================
    if 'conflict_events' in panel.columns and panel['conflict_events'].sum() > 0:
        print("\n--- Conflict Timeline ---")
        
        monthly = panel.groupby('period_start')['conflict_events'].sum().reset_index()
        monthly = monthly.sort_values('period_start')
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for _, row in monthly.iterrows():
            date = row['period_start']
            val = row['conflict_events']
            color = 'forestgreen' if date < disaster_date else 'crimson'
            ax.bar(date, val, width=25, color=color, alpha=0.7, edgecolor='white')
        
        ax.axvline(x=disaster_date, color='black', linestyle='--', linewidth=2)
        ax.text(disaster_date, ax.get_ylim()[1]*0.95,
               f'{disaster_type}\n{disaster_date.strftime("%b %d, %Y")}',
               ha='center', fontsize=10, fontweight='bold')
        
        ax.set_title(f'{country_name}: Conflict Events (ACLED)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Monthly Events', fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'conflict_timeline.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("  ✓ conflict_timeline.png")
    
    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    print("\n--- Summary ---")
    
    summary_rows = []
    for outcome in outcomes:
        cfg = OUTCOME_CONFIG[outcome]
        
        pre = panel[panel['period_start'] < disaster_date][outcome]
        post = panel[panel['period_start'] >= disaster_date][outcome]
        
        pre_mean = pre.mean()
        post_mean = post.mean()
        
        summary_rows.append({
            'Outcome': cfg['name'],
            f'Pre-{disaster_type} Mean': f"{pre_mean:.2f}" if not pd.isna(pre_mean) else "N/A",
            f'Post-{disaster_type} Mean': f"{post_mean:.2f}" if not pd.isna(post_mean) else "N/A",
            'Change (%)': f"{((post_mean - pre_mean) / abs(pre_mean) * 100):+.1f}%"
                         if (pre_mean != 0 and not pd.isna(pre_mean) and not pd.isna(post_mean)) else "N/A",
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(figures_dir / 'summary_statistics.csv', index=False)
    print("  ✓ summary_statistics.csv")
    
    # Count figures
    n_figures = len(list(figures_dir.glob("*.png")))
    
    print(f"\n{'='*60}")
    print(f"✓ All visualizations saved to: {figures_dir}")
    print(f"  Generated {n_figures} figures")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
