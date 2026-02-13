#!/usr/bin/env python3
"""
Myanmar Monthly Panel Visualizations - Landsat, Nightlights, Rain
=================================================================

This script uses the MONTHLY township-level panel exported from GEE:

    data/myanmar/gee_panel_landsat_viirs_chirps_townships_monthly.csv

It produces:

1. Monthly trends (Oct 2024 – Oct 2025) for:
   - NDVI
   - Nightlights (NL_AVG_RAD)
   - Rainfall (RAIN_MEAN)
   with separate lines for treated vs control townships,
   and a vertical line at the earthquake date (2025-03-28).

2. Choropleth maps by township for:
   - NDVI (pre vs post)
   - NL_AVG_RAD (pre vs post)
   - RAIN_MEAN (pre vs post)
   each in two versions:
       (a) plain choropleth
       (b) with treated townships highlighted (UNDP areas).

Outputs are saved under:
    outputs/myanmar/plots/
"""

import os
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx

warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, "data", "myanmar")
BOUND_DIR = os.path.join(DATA_DIR, "myanmar_boundaries")

PANEL_CSV = os.path.join(
    DATA_DIR, "gee_panel_landsat_viirs_chirps_townships_monthly.csv"
)
ADMIN3_GEOJSON = os.path.join(BOUND_DIR, "mmr_admin3.geojson")

OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "myanmar")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

EARTHQUAKE_DATE = datetime(2025, 3, 28)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_panel():
    print(f"Loading panel from: {PANEL_CSV}")
    df = pd.read_csv(PANEL_CSV)

    # Expect columns: NDVI, NL_AVG_RAD, RAIN_MEAN, RAIN_TOTAL,
    # admin3_id, admin3_nam, undp_treat, period_start, period_end, period, .geo
    # Ensure period_start is datetime
    df["period_start"] = pd.to_datetime(df["period_start"])
    df["period_end"] = pd.to_datetime(df["period_end"])

    # Create monthly label
    df["ym"] = df["period_start"].dt.to_period("M").astype(str)

    # Treated flag
    df["treated"] = df["undp_treat"].astype(int)

    # Pre/post quake
    df["post_quake"] = df["period_start"] >= EARTHQUAKE_DATE.replace(day=1) + pd.offsets.MonthBegin(1)
    # Or if you prefer strictly April 1:
    # df["post_quake"] = df["period_start"] >= pd.Timestamp("2025-04-01")

    # Limit to Oct 2024 – Oct 2025
    df = df[(df["period_start"] >= "2024-10-01") & (df["period_start"] <= "2025-10-31")]

    print(f"Loaded {len(df)} rows.")
    print("Panel sample:")
    print(df.head(6))
    print()
    return df


def load_admin3_geojson(panel_df: pd.DataFrame) -> gpd.GeoDataFrame:
    print(f"Loading admin3 boundaries from: {ADMIN3_GEOJSON}")
    gdf = gpd.read_file(ADMIN3_GEOJSON)

    # Try to identify the township name column
    name_candidates = ["admin3_nam", "ADM3_EN", "adm3_name", "NAME_3", "adm3_nam"]
    gdf_name_col = None
    for col in name_candidates:
        if col in gdf.columns:
            gdf_name_col = col
            break

    if gdf_name_col is None:
        raise ValueError(
            f"Could not find a suitable township-name column in {ADMIN3_GEOJSON}. "
            f"Available columns: {list(gdf.columns)}"
        )

    # Panel township name column
    panel_name_col = "admin3_nam"
    if panel_name_col not in panel_df.columns:
        raise ValueError(f"Column '{panel_name_col}' not found in monthly panel CSV.")

    # Deduplicate panel to 1 row per township for the join
    panel_townships = (
        panel_df[[panel_name_col, "admin3_id", "treated"]]
        .drop_duplicates(subset=[panel_name_col])
        .rename(columns={panel_name_col: "township_name"})
    )

    gdf = gdf.rename(columns={gdf_name_col: "township_name"})

    # Inner join on township_name
    merged = gdf.merge(panel_townships, on="township_name", how="inner")

    overlap = len(merged) / float(panel_townships["township_name"].nunique()) * 100
    print(
        f"Using name-based join on 'township_name' with overlap ≈ {overlap:0.1f}%."
    )

    return merged


# ---------------------------------------------------------------------
# Plotting: monthly trends
# ---------------------------------------------------------------------
def plot_monthly_trend(df: pd.DataFrame, value_col: str, ylabel: str, outname: str):
    """
    Plot monthly trends (mean across townships) for treated vs control.
    """
    if value_col not in df.columns:
        print(f"Column {value_col} not in panel; skipping trend plot.")
        return

    # Average by month & treated
    df_trend = (
        df.groupby(["period_start", "treated"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "value"})
    )

    df_trend["group"] = np.where(df_trend["treated"] == 1, "Treated (UNDP)", "Control")

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_trend,
        x="period_start",
        y="value",
        hue="group",
        marker="o",
    )

    # Vertical line at earthquake date
    plt.axvline(
        EARTHQUAKE_DATE,
        linestyle="--",
        linewidth=1.5,
        label="Earthquake (28 Mar 2025)",
    )

    plt.title(ylabel + " – Monthly Average by Treatment Status")
    plt.xlabel("Month")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(PLOTS_DIR, outname)
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved trend plot: {outpath}")


# ---------------------------------------------------------------------
# Plotting: maps (pre vs post)
# ---------------------------------------------------------------------
def aggregate_pre_post(
    df: pd.DataFrame,
    value_col: str,
    pre_end: str = "2025-03-31",
    post_start: str = "2025-04-01",
):
    """
    Aggregate a value_col to pre- and post-quake means by township.
    """
    if value_col not in df.columns:
        return None

    pre_mask = df["period_start"] <= pd.to_datetime(pre_end)
    post_mask = df["period_start"] >= pd.to_datetime(post_start)

    pre = (
        df.loc[pre_mask]
        .groupby("admin3_id", as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: f"{value_col}_pre"})
    )
    post = (
        df.loc[post_mask]
        .groupby("admin3_id", as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: f"{value_col}_post"})
    )

    out = pre.merge(post, on="admin3_id", how="outer")
    return out


def plot_pre_post_maps(
    gdf_admin3: gpd.GeoDataFrame,
    agg_df: pd.DataFrame,
    value_col: str,
    title_base: str,
    outname_plain: str,
    outname_treated: str,
):
    """
    Plot pre and post choropleths (side by side):
        - Plain version
        - Version with treated townships highlighted.
    """
    if agg_df is None:
        print(f"Skipping maps for {value_col}: no aggregated data.")
        return

    pre_col = f"{value_col}_pre"
    post_col = f"{value_col}_post"

    gdf = gdf_admin3.merge(agg_df, on="admin3_id", how="left")

    # Plain maps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, col, subtitle in zip(
        axes,
        [pre_col, post_col],
        ["Pre-quake mean", "Post-quake mean"],
    ):
        gdf.plot(
            column=col,
            ax=ax,
            legend=True,
            cmap="viridis",
            edgecolor="black",
            linewidth=0.2,
            missing_kwds={"color": "lightgrey", "label": "No data"},
        )
        ax.set_axis_off()
        ax.set_title(f"{title_base} – {subtitle}")
        try:
            ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.Stamen.TerrainBackground)
        except Exception:
            pass

    plt.tight_layout()
    outpath_plain = os.path.join(PLOTS_DIR, outname_plain)
    plt.savefig(outpath_plain, dpi=300)
    plt.close()
    print(f"Saved map: {outpath_plain}")

    # Maps with treated highlighted
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    treated_mask = gdf["treated"] == 1

    for ax, col, subtitle in zip(
        axes,
        [pre_col, post_col],
        ["Pre-quake mean", "Post-quake mean"],
    ):
        gdf.plot(
            column=col,
            ax=ax,
            legend=True,
            cmap="viridis",
            edgecolor="black",
            linewidth=0.2,
            missing_kwds={"color": "lightgrey", "label": "No data"},
        )
        # Highlight treated townships with thicker boundary
        gdf.loc[treated_mask].boundary.plot(ax=ax, edgecolor="red", linewidth=1.0)

        ax.set_axis_off()
        ax.set_title(f"{title_base} – {subtitle} (UNDP highlighted)")
        try:
            ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.Stamen.TerrainBackground)
        except Exception:
            pass

    plt.tight_layout()
    outpath_treated = os.path.join(PLOTS_DIR, outname_treated)
    plt.savefig(outpath_treated, dpi=300)
    plt.close()
    print(f"Saved map: {outpath_treated}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    df = load_panel()
    gdf_admin3 = load_admin3_geojson(df)

    # --- Monthly trends (treated vs control) ---
    plot_monthly_trend(
        df, "NDVI", "NDVI (mean across townships)", "trend_ndvi_treated_control_monthly.png"
    )
    plot_monthly_trend(
        df, "NL_AVG_RAD", "Nightlights (avg_rad)", "trend_nl_treated_control_monthly.png"
    )
    plot_monthly_trend(
        df, "RAIN_MEAN", "Daily Rainfall (CHIRPS, mean mm/day)", "trend_rain_treated_control_monthly.png"
    )

    # --- Pre vs Post maps for NDVI, NL, Rain ---
    agg_ndvi = aggregate_pre_post(df, "NDVI")
    agg_nl   = aggregate_pre_post(df, "NL_AVG_RAD")
    agg_rain = aggregate_pre_post(df, "RAIN_MEAN")

    plot_pre_post_maps(
        gdf_admin3,
        agg_ndvi,
        "NDVI",
        title_base="NDVI by Township",
        outname_plain="maps_ndvi_pre_post_monthly.png",
        outname_treated="maps_ndvi_pre_post_monthly_undp.png",
    )

    plot_pre_post_maps(
        gdf_admin3,
        agg_nl,
        "NL_AVG_RAD",
        title_base="Nightlights (avg_rad) by Township",
        outname_plain="maps_nl_pre_post_monthly.png",
        outname_treated="maps_nl_pre_post_monthly_undp.png",
    )

    plot_pre_post_maps(
        gdf_admin3,
        agg_rain,
        "RAIN_MEAN",
        title_base="Rainfall (mean mm/day) by Township",
        outname_plain="maps_rain_pre_post_monthly.png",
        outname_treated="maps_rain_pre_post_monthly_undp.png",
    )

    print("✅ Python visualization script finished.")


if __name__ == "__main__":
    main()
