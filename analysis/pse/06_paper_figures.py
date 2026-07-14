#!/usr/bin/env python3
"""
analysis/pse/06_paper_figures.py
=================================
Standardized paper figures for Gaza. Writes to outputs/paper_figures/.

fig4c_pse_cs_event_study.png  Clean Callaway-Sant'Anna dynamic ATTs
                              (log luminosity), event window trimmed to
                              -12..+18 months, 95% CI band, unambiguous
                              Gaza labelling. SAR VH supplementary variant
                              saved as supp_pse_cs_event_study_sar_vh.png.

Reads the tracked outputs/pse/tables/cs_did_event_study.csv (no data
re-extraction needed).

Usage: .venv/bin/python analysis/pse/06_paper_figures.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paper_style as ps
ps.apply()

OUT = Path("outputs/paper_figures"); OUT.mkdir(parents=True, exist_ok=True)
ES_FILE = Path("outputs/pse/tables/cs_did_event_study.csv")
WINDOW = (-12, 18)


def plot_one(es, outcome, fname, title, ylabel):
    e = es[(es.outcome == outcome) &
           (es.event_time >= WINDOW[0]) & (es.event_time <= WINDOW[1])].sort_values("event_time")
    if e.empty:
        print(f"  ⚠ no rows for {outcome}"); return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axhline(0, color="grey", lw=1)
    ax.axvline(-0.5, color=ps.ACCENT, lw=1.2, ls="--")
    ax.text(-0.5, ax.get_ylim()[1] if False else e["att"].max(),
            " project start", color=ps.ACCENT, fontsize=9, va="bottom")
    ax.fill_between(e.event_time, e.att - 1.96*e.se, e.att + 1.96*e.se,
                    color=ps.TREAT, alpha=0.15)
    ax.plot(e.event_time, e.att, color=ps.TREAT, lw=1.8, marker="o", ms=4)
    ax.set_xlabel("Months relative to site's project start")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig); print(f"✓ {fname}")


def gaza_triptych():
    import importlib.util
    spec = importlib.util.spec_from_file_location("did01", HERE / "01_did_comprehensive.py")
    did01 = importlib.util.module_from_spec(spec); sys.modules["did01"] = did01
    spec.loader.exec_module(did01)
    import geopandas as gpd
    RD = Path("data/pse/rasters")
    months = [("viirs_pse_202309.tif", "Sep 2023 (pre-conflict)"),
              ("viirs_pse_202311.tif", "Nov 2023 (conflict)"),
              ("viirs_pse_202604.tif", "Apr 2026 (recovery)")]
    paths = [RD / f for f, _ in months]
    if not all(p.exists() for p in paths):
        print("  ⚠ Gaza rasters missing — skip triptych"); return
    sites = did01.load_sites_with_projects()
    lon = sites["longitude"].values; lat = sites["latitude"].values
    try:
        bounds = gpd.read_file("data/pse/boundaries/gaza_admin2.geojson")
        gaza = bounds[bounds.geometry.centroid.x < 34.7]  # Gaza strip only
    except Exception:
        gaza = None
    bbox = (34.15, 34.65, 31.20, 31.65)   # Gaza strip
    ps.raster_triptych(paths, [l for _, l in months], OUT / "fig4a_pse_triptych.png",
                       bbox=bbox, boundary_gdf=gaza, sites_lonlat=None,
                       clip_gdf=None, boundary_color="white", boundary_lw=1.2,
                       suptitle="Gaza: nighttime luminosity, same product as Myanmar")
    print("✓ fig4a_pse_triptych.png")
    gaza_design_map(sites, gaza, bbox)


def gaza_design_map(sites, gaza, bbox):
    """Study-design map: treatment buffers (≤500 m) vs comparison rings (500 m-1.5 km)."""
    import geopandas as gpd
    from shapely.ops import unary_union
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(sites["longitude"], sites["latitude"]),
                           crs="EPSG:4326").to_crs("EPSG:32636")
    treat = unary_union(pts.buffer(500).values)
    ring = unary_union(pts.buffer(1500).values).difference(treat)
    tg = gpd.GeoSeries([treat], crs="EPSG:32636").to_crs("EPSG:4326")
    rg = gpd.GeoSeries([ring], crs="EPSG:32636").to_crs("EPSG:4326")
    if gaza is not None:                     # clip to land — no buffers in the sea
        from shapely.ops import unary_union as uu
        land = uu(gaza.geometry)
        tg = tg.intersection(land); rg = rg.intersection(land)
    fig, ax = plt.subplots(figsize=(5.2, 6.8))
    if gaza is not None:
        gaza.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.6)
    rg.plot(ax=ax, color=ps.CONTROL, alpha=0.45)
    tg.plot(ax=ax, color=ps.UN_BLUE, alpha=0.8)
    ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color=ps.UN_BLUE, alpha=0.8, label="Treatment: ≤500 m of a UNDP site"),
        mpatches.Patch(color=ps.CONTROL, alpha=0.45, label="Comparison: 500 m–1.5 km ring"),
    ], loc="lower left", fontsize=8)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(False)
    ax.set_title("Gaza study design: baseline donut specification")
    fig.tight_layout()
    fig.savefig(OUT / "fig4d_pse_design.png", bbox_inches="tight")
    plt.close(fig); print("✓ fig4d_pse_design.png")


def main():
    gaza_triptych()
    es = pd.read_csv(ES_FILE)
    plot_one(es, "log_viirs", "fig4c_pse_cs_event_study.png",
             "Gaza: Callaway–Sant'Anna dynamic ATTs — log luminosity (baseline donut, 500 m–1.5 km)",
             "ATT, log nighttime luminosity")
    plot_one(es, "sar_vh", "supp_pse_cs_event_study_sar_vh.png",
             "Gaza: Callaway–Sant'Anna dynamic ATTs — SAR VH backscatter (baseline donut)",
             "ATT, SAR VH (dB)")
    print(f"\nAll saved to {OUT}/")

if __name__ == "__main__":
    main()
