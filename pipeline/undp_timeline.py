#!/usr/bin/env python3
"""
pipeline/undp_timeline.py
=========================
UNDP Projects Timeline visualization.

Fixes:
- Disaster label ABOVE the plot (not blocking data)
- X-axis range based on config study_period
- Clean title (no "(1 year before...)")

Usage:
    python -m pipeline.undp_timeline --config config/myanmar.yaml
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import timedelta
import yaml
import argparse
import warnings
warnings.filterwarnings('ignore')


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    country_name = config['country_name']
    country_code = config['country_code'].lower()
    disaster_date = pd.Timestamp(config['events']['disaster_date'])
    disaster_type = config['events'].get('disaster_type', 'Disaster').title()
    
    # Get study period from config
    study_start = pd.Timestamp(config.get('study_period_start', disaster_date - pd.Timedelta(days=365)))
    study_end = pd.Timestamp(config.get('study_period_end', pd.Timestamp.now()))
    
    data_dir = Path(f"data/{country_code}")
    output_dir = Path(f"outputs/{country_code}")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load UNDP projects
    undp_path = data_dir / "external" / "undp_projects.csv"
    if not undp_path.exists():
        print(f"⚠ UNDP projects not found: {undp_path}")
        return
    
    df = pd.read_csv(undp_path)
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    
    # Filter umbrella projects
    if 'is_likely_umbrella' in df.columns:
        df = df[df['is_likely_umbrella'] != True]
    
    df = df.dropna(subset=['start_date'])
    
    # Filter to study period
    df = df[(df['start_date'] >= study_start) & (df['start_date'] <= study_end)]
    df = df.sort_values('start_date')
    
    if len(df) == 0:
        print(f"⚠ No projects in study period")
        return
    
    print(f"\n{'='*70}")
    print(f"UNDP TIMELINE: {country_name}")
    print(f"{'='*70}")
    print(f"Projects: {len(df)}")
    print(f"Period: {study_start.strftime('%Y-%m')} to {study_end.strftime('%Y-%m')}")
    
    # Create figure
    fig_height = max(6, len(df) * 0.4)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    
    # X-axis limits (from config study period)
    xmin = study_start - timedelta(days=30)
    xmax = study_end + timedelta(days=30)
    
    # Shading: pre/post disaster
    if xmin < disaster_date:
        ax.axvspan(xmin, disaster_date, alpha=0.1, color='green')
    if xmax > disaster_date:
        ax.axvspan(disaster_date, xmax, alpha=0.08, color='red')
    
    # Disaster line
    ax.axvline(disaster_date, color='red', linestyle='--', linewidth=2.5, zorder=5)
    
    # Plot each project
    for i, (_, row) in enumerate(df.iterrows()):
        cat = row.get('ml_category', 'Other / Unspecified')
        color = CATEGORY_COLORS.get(cat, '#9E9E9E')
        
        ax.barh(i, width=25, left=row['start_date'], height=0.7, 
               color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
        
        # Project title
        title = str(row.get('title', ''))[:50]
        budget = row.get('total_budget', 0)
        if pd.notna(budget) and budget > 0:
            bs = f" [${budget/1e6:.1f}M]" if budget >= 1e6 else f" [${budget/1e3:.0f}K]"
        else:
            bs = ""
        
        ax.text(row['start_date'] + timedelta(days=30), i, f"{title}{bs}", 
               va='center', fontsize=8)
        ax.text(row['start_date'] - timedelta(days=3), i, 
               row['start_date'].strftime('%Y-%m-%d'), 
               va='center', ha='right', fontsize=7, alpha=0.7)
    
    # Set limits — tight y-axis (no wasted space above top project)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.8, len(df) - 0.2)
    ax.set_yticks([])
    
    # X-axis formatting
    ax.set_xlabel('Date', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Disaster label — placed ABOVE plot using axes coordinates
    # so it never overlaps with project bars
    ax.annotate(f'{disaster_type}\n{disaster_date.strftime("%b %d, %Y")}',
               xy=(disaster_date, 1.0), xycoords=('data', 'axes fraction'),
               xytext=(0, 8), textcoords='offset points',
               ha='center', va='bottom',
               fontsize=11, fontweight='bold', color='red',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        alpha=0.9, edgecolor='red'))
    
    # Title — extra pad to clear the disaster label
    ax.set_title(f'{country_name}: UNDP Projects Timeline', 
                fontsize=14, fontweight='bold', pad=45)
    
    # Category legend
    cats = df['ml_category'].unique() if 'ml_category' in df.columns else []
    patches = [mpatches.Patch(color=CATEGORY_COLORS.get(c, '#9E9E9E'), label=c) 
               for c in sorted(cats) if c in CATEGORY_COLORS]
    if patches:
        ax.legend(handles=patches, loc='upper left', fontsize=8, ncol=2, title='Category')
    
    # Add summary stats
    n_pre = len(df[df['start_date'] < disaster_date])
    n_post = len(df[df['start_date'] >= disaster_date])
    ax.text(0.98, 0.02, f'Pre-{disaster_type}: {n_pre} | Post-{disaster_type}: {n_post}',
           transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'undp_timeline.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {figures_dir / 'undp_timeline.png'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
