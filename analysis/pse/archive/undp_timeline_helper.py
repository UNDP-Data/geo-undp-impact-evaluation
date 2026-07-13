#!/usr/bin/env python3
"""
analysis/pse/undp_timeline_helper.py
====================================
Helper functions to load UNDP project data and add to visualizations.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# Default UNDP projects file
DEFAULT_UNDP_FILE = "data/pse/external/undp_projects.csv"
ALTERNATE_UNDP_FILE = "data/pse/pal_undp_projects_ml_classified.csv"


def load_undp_projects(file_path=None):
    """
    Load UNDP projects data.
    
    Returns:
        DataFrame with columns: project_id, title, ml_category, start_date, end_date
    """
    paths_to_try = [
        file_path,
        DEFAULT_UNDP_FILE,
        ALTERNATE_UNDP_FILE,
        "data/pse/external/pal_undp_projects_ml_classified.csv",
    ]
    
    for path in paths_to_try:
        if path and Path(path).exists():
            df = pd.read_csv(path)
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
            if 'end_date' in df.columns:
                df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
            return df
    
    return None


def get_active_projects(undp_df, reference_date, exclude_umbrella=True):
    """
    Get projects that were active on a specific date.
    
    Args:
        undp_df: DataFrame with UNDP projects
        reference_date: Date to check
        exclude_umbrella: If True, exclude long-running umbrella projects
    
    Returns:
        DataFrame of active projects
    """
    if undp_df is None:
        return pd.DataFrame()
    
    df = undp_df.copy()
    ref = pd.to_datetime(reference_date)
    
    if exclude_umbrella:
        df = df[df['is_likely_umbrella'] != True]
    
    # Project is active if start_date <= ref and (end_date is null or end_date >= ref)
    mask = df['start_date'] <= ref
    if 'end_date' in df.columns:
        mask = mask & (df['end_date'].isna() | (df['end_date'] >= ref))
    
    return df[mask]


def get_projects_by_period(undp_df, start_date, end_date, exclude_umbrella=True):
    """
    Get projects that started within a period.
    
    Args:
        undp_df: DataFrame with UNDP projects
        start_date: Period start
        end_date: Period end
        exclude_umbrella: If True, exclude long-running umbrella projects
    
    Returns:
        DataFrame of projects starting in period
    """
    if undp_df is None:
        return pd.DataFrame()
    
    df = undp_df.copy()
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    if exclude_umbrella:
        df = df[df['is_likely_umbrella'] != True]
    
    mask = (df['start_date'] >= start) & (df['start_date'] <= end)
    return df[mask].sort_values('start_date')


def get_category_summary(undp_df, exclude_umbrella=True):
    """
    Get summary of projects by category.
    
    Returns:
        Series with category counts
    """
    if undp_df is None:
        return pd.Series()
    
    df = undp_df.copy()
    if exclude_umbrella:
        df = df[df['is_likely_umbrella'] != True]
    
    return df['ml_category'].value_counts()


def create_project_timeline_annotations(undp_df, ax, y_position=0.95, exclude_umbrella=True,
                                        min_date=None, max_date=None):
    """
    Add UNDP project annotations to a matplotlib axis.
    
    Args:
        undp_df: DataFrame with UNDP projects
        ax: Matplotlib axis
        y_position: Y position for annotations (in axis coordinates)
        exclude_umbrella: If True, exclude long-running umbrella projects
        min_date: Minimum date to show
        max_date: Maximum date to show
    """
    if undp_df is None:
        return
    
    df = undp_df.copy()
    if exclude_umbrella:
        df = df[df['is_likely_umbrella'] != True]
    
    if min_date:
        df = df[df['start_date'] >= pd.to_datetime(min_date)]
    if max_date:
        df = df[df['start_date'] <= pd.to_datetime(max_date)]
    
    if len(df) == 0:
        return
    
    # Group by month for cleaner display
    df['start_month'] = df['start_date'].dt.to_period('M').dt.to_timestamp()
    
    monthly_projects = df.groupby('start_month').agg({
        'title': 'first',  # Just show first project name
        'ml_category': lambda x: ', '.join(x.unique()[:2])  # Show up to 2 categories
    }).reset_index()
    
    # Add markers for each month with new projects
    for _, row in monthly_projects.iterrows():
        ax.axvline(x=row['start_month'], color='purple', linestyle=':', alpha=0.5, linewidth=1)
        
        # Add small annotation
        short_cat = row['ml_category'][:20] + '...' if len(row['ml_category']) > 20 else row['ml_category']
        ax.annotate(
            short_cat,
            xy=(row['start_month'], y_position),
            xycoords=('data', 'axes fraction'),
            fontsize=6,
            rotation=90,
            ha='center',
            va='bottom',
            alpha=0.7,
            color='purple'
        )


def get_projects_summary_text(undp_df, reference_date, period_name="Current"):
    """
    Get a text summary of active projects at a given date.
    
    Returns:
        String with project summary
    """
    active = get_active_projects(undp_df, reference_date)
    
    if len(active) == 0:
        return f"No active UNDP projects on {reference_date}"
    
    categories = active['ml_category'].value_counts()
    
    lines = [f"{period_name} Active UNDP Projects: {len(active)}"]
    for cat, count in categories.head(5).items():
        lines.append(f"  • {cat}: {count}")
    
    return "\n".join(lines)


def get_timeline_events(undp_df, conflict_date="2023-10-07", exclude_umbrella=True):
    """
    Get key project start dates for timeline visualization.
    
    Returns:
        List of tuples: (date, category, title)
    """
    if undp_df is None:
        return []
    
    df = undp_df.copy()
    if exclude_umbrella:
        df = df[df['is_likely_umbrella'] != True]
    
    # Only projects after conflict
    conflict = pd.to_datetime(conflict_date)
    df = df[df['start_date'] >= conflict]
    
    events = []
    for _, row in df.iterrows():
        events.append((
            row['start_date'],
            row['ml_category'],
            row['title'][:50] if pd.notna(row['title']) else 'Unknown'
        ))
    
    return sorted(events, key=lambda x: x[0])
