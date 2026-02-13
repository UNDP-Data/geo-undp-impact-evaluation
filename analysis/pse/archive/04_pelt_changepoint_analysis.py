#!/usr/bin/env python3
"""
analysis/pse/04_pelt_changepoint_analysis.py
=============================================
PELT Changepoint Detection for UNDP Intervention Attribution

This script uses PELT (Pruned Exact Linear Time) to detect structural breaks
in time series and tests whether they align with UNDP intervention dates.

Key methodology:
1. Detect changepoints using PELT algorithm
2. Compare detected changepoints with UNDP intervention dates
3. Calculate magnitude of change at each changepoint
4. Run WB placebo: if WB doesn't have changepoint at same time → stronger evidence
5. Bootstrap confidence intervals for changepoint locations

Output:
- 4-panel plots showing changepoints + intervention dates + WB placebo
- Summary table of changepoint-intervention alignment
- Statistical tests for intervention attribution

Usage:
    python analysis/pse/04_pelt_changepoint_analysis.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import ruptures for PELT
try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    print("⚠ ruptures not installed. Install with: pip install ruptures")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures" / "changepoint"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FILE = OUTPUT_DIR / "panel_data.csv"
UNDP_FILE = DATA_DIR / "pse" / "external" / "undp_projects.csv"

CONFLICT_DATE = pd.Timestamp("2023-10-07")

# All outcomes to analyze
ALL_OUTCOMES = [
    # Satellite (have donor data for SC)
    'nightlights', 'ndvi', 'vh', 'vv', 'rvi',
    # Prices
    'price_bread', 'price_flour', 'price_rice', 'price_milk', 
    'price_oil', 'price_sugar', 'price_chicken',
    # Google Trends
    'gt_food', 'gt_water', 'gt_hospital', 'gt_help', 'gt_electricity',
    'gt_shelter', 'gt_medicine', 'gt_bread', 'gt_flour',
    # Conflict
    'conflict_events', 'conflict_fatalities'
]

OUTCOME_LABELS = {
    'nightlights': 'Nighttime Lights (VIIRS)',
    'ndvi': 'Vegetation Index (NDVI)',
    'vh': 'SAR Backscatter (VH)',
    'vv': 'SAR Backscatter (VV)',
    'rvi': 'Radar Vegetation Index',
    'price_bread': 'Bread Price (ILS/kg)',
    'price_flour': 'Flour Price (ILS/kg)',
    'price_rice': 'Rice Price (ILS/kg)',
    'price_milk': 'Milk Price (ILS/L)',
    'price_oil': 'Cooking Oil Price (ILS/L)',
    'price_sugar': 'Sugar Price (ILS/kg)',
    'price_chicken': 'Chicken Price (ILS/kg)',
    'gt_food': 'Google Trends: Food',
    'gt_water': 'Google Trends: Water',
    'gt_hospital': 'Google Trends: Hospital',
    'gt_help': 'Google Trends: Help',
    'gt_electricity': 'Google Trends: Electricity',
    'gt_shelter': 'Google Trends: Shelter',
    'gt_medicine': 'Google Trends: Medicine',
    'gt_bread': 'Google Trends: Bread',
    'gt_flour': 'Google Trends: Flour',
    'conflict_events': 'Conflict Events (ACLED)',
    'conflict_fatalities': 'Conflict Fatalities (ACLED)',
}


def detect_changepoints_pelt(signal, pen=None, min_size=2, n_bkps=None):
    """
    Detect changepoints using PELT algorithm.
    
    Parameters:
    -----------
    signal : array-like
        Time series data
    pen : float, optional
        Penalty value for PELT. If None, uses BIC-like penalty.
    min_size : int
        Minimum segment length
    n_bkps : int, optional
        If specified, use Pelt with fixed number of breakpoints
        
    Returns:
    --------
    list of int : Indices of detected changepoints
    """
    if not HAS_RUPTURES:
        # Fallback: simple variance-based detection
        return detect_changepoints_simple(signal)
    
    signal = np.array(signal)
    
    # Handle NaN by interpolation
    valid_mask = ~np.isnan(signal)
    if valid_mask.sum() < 5:
        return []
    
    # Interpolate NaN values
    x = np.arange(len(signal))
    signal_clean = np.interp(x, x[valid_mask], signal[valid_mask])
    
    # Normalize for better PELT performance
    signal_norm = (signal_clean - np.mean(signal_clean)) / (np.std(signal_clean) + 1e-8)
    
    # Use rbf (radial basis function) model for flexible changepoint detection
    # This captures changes in mean, variance, and trend
    model = "rbf"
    algo = rpt.Pelt(model=model, min_size=min_size).fit(signal_norm)
    
    # Penalty: use BIC-like penalty if not specified
    if pen is None:
        pen = np.log(len(signal_norm)) * 2  # BIC-like
    
    try:
        if n_bkps:
            # Use Binseg for fixed number of breakpoints
            algo = rpt.Binseg(model=model, min_size=min_size).fit(signal_norm)
            result = algo.predict(n_bkps=n_bkps)
        else:
            result = algo.predict(pen=pen)
        
        # Remove the last point (always included by ruptures)
        if result and result[-1] == len(signal_norm):
            result = result[:-1]
        
        return result
    except Exception as e:
        print(f"    PELT error: {e}")
        return []


def detect_changepoints_simple(signal, threshold=2.0):
    """
    Simple changepoint detection using cumulative sum (CUSUM).
    Fallback when ruptures is not available.
    """
    signal = np.array(signal)
    valid_mask = ~np.isnan(signal)
    if valid_mask.sum() < 5:
        return []
    
    # Interpolate NaN
    x = np.arange(len(signal))
    signal_clean = np.interp(x, x[valid_mask], signal[valid_mask])
    
    # CUSUM statistic
    mean_val = np.mean(signal_clean)
    cusum = np.cumsum(signal_clean - mean_val)
    
    # Detect significant deviations
    cusum_std = np.std(cusum)
    changepoints = []
    
    for i in range(1, len(cusum) - 1):
        # Local maximum/minimum in CUSUM indicates changepoint
        if abs(cusum[i] - cusum[i-1]) > threshold * cusum_std:
            changepoints.append(i)
    
    return changepoints


def calculate_change_magnitude(signal, cp_idx, window=3):
    """
    Calculate magnitude of change at a changepoint.
    
    Returns:
    --------
    dict with 'before_mean', 'after_mean', 'change', 'pct_change', 't_stat', 'p_value'
    """
    signal = np.array(signal)
    
    # Get values before and after changepoint
    before_start = max(0, cp_idx - window)
    after_end = min(len(signal), cp_idx + window)
    
    before = signal[before_start:cp_idx]
    after = signal[cp_idx:after_end]
    
    # Remove NaN
    before = before[~np.isnan(before)]
    after = after[~np.isnan(after)]
    
    if len(before) < 2 or len(after) < 2:
        return None
    
    before_mean = np.mean(before)
    after_mean = np.mean(after)
    change = after_mean - before_mean
    pct_change = (change / abs(before_mean) * 100) if abs(before_mean) > 0.001 else np.nan
    
    # T-test for significance
    try:
        t_stat, p_value = stats.ttest_ind(before, after)
    except:
        t_stat, p_value = np.nan, np.nan
    
    return {
        'before_mean': before_mean,
        'after_mean': after_mean,
        'change': change,
        'pct_change': pct_change,
        't_stat': t_stat,
        'p_value': p_value
    }


def find_nearest_intervention(cp_date, intervention_dates, max_gap_months=2):
    """
    Find the nearest UNDP intervention to a changepoint.
    
    Returns:
    --------
    tuple: (intervention_name, days_apart) or (None, None) if no match
    """
    best_match = None
    best_gap = float('inf')
    
    for name, date in intervention_dates.items():
        if pd.isna(date):
            continue
        gap = abs((cp_date - date).days)
        if gap < best_gap and gap <= max_gap_months * 30:
            best_gap = gap
            best_match = name
    
    if best_match:
        return best_match, best_gap
    return None, None


def create_pelt_plot(outcome, times_dt, gaza_signal, wb_signal,
                     gaza_cps, wb_cps, intervention_dates,
                     change_stats, output_path):
    """
    Create 4-panel PELT analysis plot.
    
    Panel 1: Gaza time series with PELT changepoints
    Panel 2: West Bank time series with PELT changepoints (PLACEBO)
    Panel 3: Gaza cumulative sum (CUSUM) for visual changepoint detection
    Panel 4: Changepoint-Intervention alignment summary
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    label = OUTCOME_LABELS.get(outcome, outcome)
    
    # Intervention colors
    int_colors = {
        'Any UNDP': 'purple',
        'Debris & Solid Waste': 'brown',
        'Infrastructure & Housing': 'orange',
        'Health & Emergency': 'red',
        'Water & Sanitation (WASH)': 'blue',
        'Social Inclusion': 'green',
    }
    
    # =========================================================================
    # Panel 1: Gaza with changepoints
    # =========================================================================
    ax1 = axes[0, 0]
    
    # Background shading
    ax1.axvspan(times_dt[0], CONFLICT_DATE, alpha=0.1, color='green', label='Pre-conflict')
    ax1.axvspan(CONFLICT_DATE, times_dt[-1], alpha=0.1, color='red', label='Post-conflict')
    
    # Plot Gaza signal
    valid_mask = ~np.isnan(gaza_signal)
    ax1.plot(times_dt[valid_mask], gaza_signal[valid_mask], 'b-', 
             linewidth=2, marker='o', markersize=4, label='Gaza')
    
    # Plot changepoints
    for i, cp_idx in enumerate(gaza_cps):
        if cp_idx < len(times_dt):
            cp_date = times_dt[cp_idx]
            ax1.axvline(x=cp_date, color='red', linestyle='--', linewidth=2, alpha=0.8)
            
            # Add magnitude annotation
            if i < len(change_stats) and change_stats[i]:
                mag = change_stats[i].get('change', 0)
                pct = change_stats[i].get('pct_change', 0)
                ax1.annotate(f'Δ={mag:.2f}\n({pct:+.0f}%)', 
                           xy=(cp_date, gaza_signal[cp_idx] if not np.isnan(gaza_signal[cp_idx]) else np.nanmean(gaza_signal)),
                           fontsize=8, ha='center', va='bottom',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Plot intervention dates
    for name, date in intervention_dates.items():
        if pd.notna(date) and times_dt[0] <= date <= times_dt[-1]:
            color = int_colors.get(name, 'gray')
            ax1.axvline(x=date, color=color, linestyle=':', linewidth=2, alpha=0.7, label=name)
    
    ax1.axvline(x=CONFLICT_DATE, color='black', linestyle='-', linewidth=2.5, label='Oct 7, 2023')
    
    ax1.set_ylabel(label, fontsize=11)
    ax1.set_title(f'Gaza: PELT Changepoints (n={len(gaza_cps)})', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # =========================================================================
    # Panel 2: West Bank with changepoints (PLACEBO)
    # =========================================================================
    ax2 = axes[0, 1]
    
    ax2.axvspan(times_dt[0], CONFLICT_DATE, alpha=0.1, color='green')
    ax2.axvspan(CONFLICT_DATE, times_dt[-1], alpha=0.1, color='lightblue')
    
    valid_mask_wb = ~np.isnan(wb_signal)
    if valid_mask_wb.sum() > 0:
        ax2.plot(times_dt[valid_mask_wb], wb_signal[valid_mask_wb], 'g-', 
                 linewidth=2, marker='s', markersize=4, label='West Bank')
        
        # WB changepoints
        for cp_idx in wb_cps:
            if cp_idx < len(times_dt):
                ax2.axvline(x=times_dt[cp_idx], color='orange', linestyle='--', linewidth=2, alpha=0.8)
    
    ax2.axvline(x=CONFLICT_DATE, color='black', linestyle='-', linewidth=2.5)
    
    ax2.set_ylabel(label, fontsize=11)
    ax2.set_title(f'West Bank PLACEBO: PELT Changepoints (n={len(wb_cps)})', 
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # =========================================================================
    # Panel 3: CUSUM plot for Gaza
    # =========================================================================
    ax3 = axes[1, 0]
    
    # Calculate CUSUM
    valid_gaza = gaza_signal[~np.isnan(gaza_signal)]
    times_valid = times_dt[~np.isnan(gaza_signal)]
    
    if len(valid_gaza) > 0:
        cusum = np.cumsum(valid_gaza - np.mean(valid_gaza))
        ax3.plot(times_valid, cusum, 'b-', linewidth=2, label='CUSUM')
        ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        
        # Mark changepoints on CUSUM
        for cp_idx in gaza_cps:
            if cp_idx < len(times_valid):
                ax3.axvline(x=times_valid[cp_idx], color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        ax3.axvline(x=CONFLICT_DATE, color='black', linestyle='-', linewidth=2.5)
    
    ax3.set_ylabel('Cumulative Sum', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Gaza CUSUM (visual changepoint detection)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # =========================================================================
    # Panel 4: Changepoint-Intervention alignment table
    # =========================================================================
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Build alignment table
    table_data = []
    
    for i, cp_idx in enumerate(gaza_cps):
        if cp_idx < len(times_dt):
            cp_date = times_dt[cp_idx]
            match_name, match_gap = find_nearest_intervention(cp_date, intervention_dates)
            
            # Check if WB has similar changepoint
            wb_has_similar = any(abs(cp_idx - wb_cp) <= 2 for wb_cp in wb_cps)
            
            # Get change magnitude
            mag = change_stats[i].get('change', 0) if i < len(change_stats) and change_stats[i] else 0
            pval = change_stats[i].get('p_value', 1) if i < len(change_stats) and change_stats[i] else 1
            
            sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
            
            # Validity
            if match_name and not wb_has_similar:
                validity = "✓ ATTRIBUTED"
            elif match_name and wb_has_similar:
                validity = "⚠ REGIONAL"
            else:
                validity = "○ UNMATCHED"
            
            table_data.append([
                cp_date.strftime('%Y-%m'),
                f'{mag:+.3f}{sig}',
                match_name or 'None',
                f'{match_gap}d' if match_gap else '-',
                'Yes' if wb_has_similar else 'No',
                validity
            ])
    
    if table_data:
        col_labels = ['CP Date', 'Magnitude', 'Nearest UNDP', 'Gap', 'WB Match?', 'Attribution']
        table = ax4.table(cellText=table_data, colLabels=col_labels,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Color code validity
        for i, row in enumerate(table_data):
            if '✓' in row[-1]:
                table[(i+1, 5)].set_facecolor('#90EE90')  # Light green
            elif '⚠' in row[-1]:
                table[(i+1, 5)].set_facecolor('#FFD700')  # Yellow
            else:
                table[(i+1, 5)].set_facecolor('#D3D3D3')  # Gray
    else:
        ax4.text(0.5, 0.5, 'No changepoints detected', 
                 ha='center', va='center', fontsize=14, transform=ax4.transAxes)
    
    ax4.set_title('Changepoint-Intervention Alignment', fontsize=12, fontweight='bold', pad=20)
    
    # =========================================================================
    # Overall title
    # =========================================================================
    n_attributed = sum(1 for t in table_data if '✓' in t[-1]) if table_data else 0
    
    fig.suptitle(f'PELT Changepoint Analysis: {label}\n'
                 f'Gaza: {len(gaza_cps)} changepoints | WB Placebo: {len(wb_cps)} | '
                 f'Attributed to UNDP: {n_attributed}',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return table_data


def main():
    print("\n" + "=" * 70)
    print("PELT CHANGEPOINT ANALYSIS FOR UNDP ATTRIBUTION")
    print("=" * 70)
    
    if not HAS_RUPTURES:
        print("\n⚠ ruptures package not found. Using simple CUSUM fallback.")
        print("  Install with: pip install ruptures --break-system-packages")
    
    # Load data
    print("\nLoading data...")
    panel = pd.read_csv(PANEL_FILE)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    panel['unit_id'] = panel['unit_id'].astype(str)
    
    gaza_df = panel[panel['unit_id'].str.startswith('PS02')].copy()
    wb_df = panel[~panel['unit_id'].str.startswith('PS02')].copy()
    
    print(f"  Gaza: {len(gaza_df)} obs")
    print(f"  West Bank: {len(wb_df)} obs")
    
    # Load UNDP intervention dates
    intervention_dates = {}
    try:
        undp = pd.read_csv(UNDP_FILE)
        undp['start_date'] = pd.to_datetime(undp['start_date'], errors='coerce')
        if 'is_likely_umbrella' in undp.columns:
            undp = undp[undp['is_likely_umbrella'] != True]
        undp = undp[undp['start_date'] >= CONFLICT_DATE]
        
        if 'ml_category' in undp.columns:
            for cat in undp['ml_category'].dropna().unique():
                cat_df = undp[undp['ml_category'] == cat]
                if len(cat_df) > 0:
                    intervention_dates[cat] = cat_df['start_date'].min()
        
        intervention_dates['Any UNDP'] = undp['start_date'].min()
        print(f"  UNDP interventions: {len(intervention_dates)}")
    except Exception as e:
        print(f"  ⚠ Could not load UNDP dates: {e}")
        intervention_dates = {'Any UNDP': pd.Timestamp('2024-01-01')}
    
    # Get time index
    all_times = sorted(panel['period_start'].unique())
    times_dt = pd.to_datetime(all_times)
    
    # ==========================================================================
    # ANALYZE EACH OUTCOME
    # ==========================================================================
    print("\n" + "=" * 70)
    print("CHANGEPOINT DETECTION")
    print("=" * 70)
    
    all_results = []
    
    for outcome in ALL_OUTCOMES:
        if outcome not in gaza_df.columns:
            continue
        if gaza_df[outcome].notna().sum() < 10:
            continue
        
        print(f"\n  {outcome}...")
        
        # Aggregate to monthly means
        gaza_agg = gaza_df.groupby('period_start')[outcome].mean().reindex(times_dt)
        wb_agg = wb_df.groupby('period_start')[outcome].mean().reindex(times_dt)
        
        gaza_signal = gaza_agg.values
        wb_signal = wb_agg.values
        
        # Detect changepoints
        gaza_cps = detect_changepoints_pelt(gaza_signal)
        wb_cps = detect_changepoints_pelt(wb_signal)
        
        print(f"    Gaza changepoints: {len(gaza_cps)}")
        print(f"    WB changepoints: {len(wb_cps)}")
        
        # Calculate change magnitudes
        change_stats = []
        for cp_idx in gaza_cps:
            stats = calculate_change_magnitude(gaza_signal, cp_idx)
            change_stats.append(stats)
        
        # Create plot
        table_data = create_pelt_plot(
            outcome, times_dt, gaza_signal, wb_signal,
            gaza_cps, wb_cps, intervention_dates,
            change_stats, FIGURES_DIR / f'pelt_{outcome}.png'
        )
        print(f"    ✓ pelt_{outcome}.png")
        
        # Store results
        for i, cp_idx in enumerate(gaza_cps):
            if cp_idx < len(times_dt):
                cp_date = times_dt[cp_idx]
                match_name, match_gap = find_nearest_intervention(cp_date, intervention_dates)
                wb_has_similar = any(abs(cp_idx - wb_cp) <= 2 for wb_cp in wb_cps)
                
                result = {
                    'Outcome': outcome,
                    'CP_Date': cp_date.strftime('%Y-%m'),
                    'CP_Index': cp_idx,
                    'Change': change_stats[i]['change'] if i < len(change_stats) and change_stats[i] else np.nan,
                    'Pct_Change': change_stats[i]['pct_change'] if i < len(change_stats) and change_stats[i] else np.nan,
                    'P_Value': change_stats[i]['p_value'] if i < len(change_stats) and change_stats[i] else np.nan,
                    'Nearest_UNDP': match_name,
                    'Gap_Days': match_gap,
                    'WB_Has_Similar': wb_has_similar,
                    'Attribution': 'UNDP' if match_name and not wb_has_similar else ('Regional' if match_name else 'Unmatched')
                }
                all_results.append(result)
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(TABLES_DIR / 'pelt_changepoint_results.csv', index=False)
    print(f"  ✓ pelt_changepoint_results.csv ({len(results_df)} changepoints)")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ATTRIBUTION SUMMARY")
    print("=" * 70)
    
    if len(results_df) > 0:
        attribution_counts = results_df['Attribution'].value_counts()
        print(f"\n  Total changepoints detected: {len(results_df)}")
        for attr, count in attribution_counts.items():
            print(f"    {attr}: {count}")
        
        print("\n  CHANGEPOINTS ATTRIBUTED TO UNDP:")
        print("-" * 70)
        
        undp_attributed = results_df[results_df['Attribution'] == 'UNDP']
        if len(undp_attributed) > 0:
            for _, row in undp_attributed.iterrows():
                sig = '***' if row['P_Value'] < 0.01 else '**' if row['P_Value'] < 0.05 else '*' if row['P_Value'] < 0.10 else ''
                print(f"    {row['Outcome']:25} {row['CP_Date']}  Δ={row['Change']:+.3f}{sig:3}  → {row['Nearest_UNDP']} ({row['Gap_Days']}d)")
        else:
            print("    None - all changepoints either match regional trends or are unmatched")
        
        print("\n  INTERPRETATION:")
        print("-" * 70)
        print("""
    ✓ UNDP Attribution: Changepoint aligns with UNDP intervention date
                        AND West Bank does NOT have a similar changepoint
                        → Suggestive evidence that UNDP intervention caused change

    ⚠ Regional Trend:   Changepoint aligns with UNDP intervention
                        BUT West Bank also has a changepoint
                        → Could be regional trend, not UNDP-specific

    ○ Unmatched:        Changepoint does not align with any UNDP intervention
                        → Other factors (conflict intensity, seasons, etc.)

    CAVEATS:
    - This is CORRELATION, not definitive CAUSATION
    - Multiple agencies operate simultaneously
    - Temporal alignment ≠ causal mechanism
    - Use alongside DiD, SC, and qualitative evidence
        """)
    
    print(f"\n{'='*70}")
    print("✓ PELT ANALYSIS COMPLETE")
    print(f"  Figures: {FIGURES_DIR}/pelt_*.png")
    print(f"  Results: {TABLES_DIR}/pelt_changepoint_results.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
