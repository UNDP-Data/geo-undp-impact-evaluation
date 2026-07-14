# BUILD: corridor-v8
#!/usr/bin/env python3
"""
analysis/mmr/07_paper_figures.py
=================================
Standardized paper figures for Myanmar (Gaza aesthetics as reference).
Writes to outputs/paper_figures/.

fig2a_mmr_zoom_map.png   Zoomed 3-panel township choropleth around the
                         epicentre (same YoY month), treated townships
                         outlined solid, controls dashed — replaces the
                         washed-out whole-country map.
fig2b_mmr_ts.png         Treated vs control mean log-nightlights, from
                         Oct 2024, quake + intervention lines (Gaza-style).
fig3b_mmr_pretrends.png  Clean pre-trends: monthly treated-control gap in
                         log nightlights, 12 months pre-quake through the
                         post window, baseline 3-8 km spec.

Usage: .venv/bin/python analysis/mmr/07_paper_figures.py
"""
import importlib.util, sys
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

spec = importlib.util.spec_from_file_location("d", HERE / "01_did_buffer_fixed.py")
d = importlib.util.module_from_spec(spec); sys.modules["d"] = d; spec.loader.exec_module(d)

OUT = Path("outputs/paper_figures"); OUT.mkdir(parents=True, exist_ok=True)
EPI_LON, EPI_LAT = 95.96, 20.88
BASE_INNER, BASE_OUTER = 3000, 8000            # baseline spec
YOY_MONTHS = ["2024-04-01", "2025-04-01", "2026-04-01"]
PANEL_LABELS = ["Apr 2024 (pre-earthquake)", "Apr 2025 (post-earthquake)", "Apr 2026 (recovery)"]


def load_all():
    panel = d.load_panel_data()
    gdf, id_col = d.load_boundaries()
    sites = d.load_undp_sites()
    dist = d.calculate_min_distance_to_sites(gdf, sites, id_col)
    gdf["min_dist"] = gdf[id_col].map(dist)
    gdf["group"] = np.where(gdf["min_dist"] <= BASE_INNER, "treated",
                    np.where(gdf["min_dist"] <= BASE_OUTER, "control", "other"))
    return panel, gdf, id_col, sites, dist


def fig2a_zoom_map(panel, gdf, id_col):
    study = gdf[gdf["group"] != "other"].copy()
    study["_zid"] = study[id_col]
    minx, miny, maxx, maxy = study.total_bounds
    pad = 0.6
    bbox = (minx - pad, maxx + pad, miny - pad, maxy + pad)

    nl = panel.pivot_table(index="unit_id", columns="period_start",
                           values="nightlights", aggfunc="mean")
    nl.columns = [str(pd.Timestamp(c).date()) for c in nl.columns]

    # Log scale (radiance is tiny outside cities); vmax from the ZOOM region only
    zoom_ids = set(study["_zid"]) if "_zid" in study.columns else set(study[id_col])
    zoom_vals = pd.concat([np.log1p(nl.loc[nl.index.isin(zoom_ids), m])
                           for m in YOY_MONTHS if m in nl.columns])
    vmin, vmax = 0.0, float(np.nanpercentile(zoom_vals, 96))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6))
    im = None
    for ax, month, label in zip(axes, YOY_MONTHS, PANEL_LABELS):
        g = gdf.copy()
        g["val"] = np.log1p(g[id_col].map(nl[month])) if month in nl.columns else np.nan
        g.plot(ax=ax, column="val", cmap="magma", vmin=vmin, vmax=vmax,
               edgecolor="#555555", linewidth=0.15, missing_kwds={"color": "#222222"})
        study[study.group == "treated"].boundary.plot(ax=ax, color=ps.UN_BLUE, linewidth=1.3)
        study[study.group == "control"].boundary.plot(ax=ax, color="white",
                                                      linewidth=0.9, linestyle="--")
        ax.scatter([EPI_LON], [EPI_LAT], marker="*", s=130, c=ps.ACCENT, zorder=15)
        ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
        ax.set_title(label); ax.set_axis_off()
    import matplotlib.cm as cm
    sm = cm.ScalarMappable(cmap="magma",
                           norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                        fraction=0.045, pad=0.06, shrink=0.55)
    cbar.set_label("log(1 + nighttime radiance, nW/cm²/sr)")
    import matplotlib.lines as mlines
    fig.legend(handles=[
        mlines.Line2D([], [], color=ps.UN_BLUE, lw=1.6, label="UNDP townships (≤3 km of a UNDP site)"),
        mlines.Line2D([], [], color="grey", lw=1.2, ls="--", label="Comparison townships (3–8 km)"),
        mlines.Line2D([], [], marker="*", color=ps.ACCENT, ms=13, ls="None", label="Epicentre (M7.7)"),
    ], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Nighttime luminosity around the epicentre — same calendar month across years", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig2a_mmr_zoom_map.png", bbox_inches="tight")
    plt.close(fig); print("✓ fig2a_mmr_zoom_map.png")


def fig2b_ts(panel, dist):
    df = d.create_buffer_treatment(panel, dist, BASE_INNER, BASE_OUTER,
                                   pd.Timestamp("2025-09-01"))
    df = df[df.period_start >= "2024-10-01"]
    g = (df.assign(grp=np.where(df.is_treated == 1, "Treated", "Control"))
           .groupby(["grp", "period_start"])["log_nightlights"].mean().unstack(0))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(g.index, g["Treated"], color=ps.TREAT, lw=2, label="UNDP townships (≤3 km of a site)")
    ax.plot(g.index, g["Control"], color=ps.CONTROL, lw=2, ls="--", label="Comparison townships (3–8 km)")
    ax.axvline(pd.Timestamp("2025-03-28"), color=ps.ACCENT, lw=1.2)
    ax.text(pd.Timestamp("2025-03-28"), ax.get_ylim()[1], " Earthquake",
            color=ps.ACCENT, va="top", fontsize=9)
    ax.axvline(pd.Timestamp("2025-09-01"), color=ps.GREY, lw=1.2, ls=":")
    ax.text(pd.Timestamp("2025-09-01"), ax.get_ylim()[0], " UNDP infrastructure\n deployment",
            color=ps.GREY, va="bottom", fontsize=9)
    ax.set_ylabel("Mean log nighttime luminosity")
    ax.set_title("Myanmar: UNDP vs comparison townships (baseline 3–8 km specification)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2b_mmr_ts.png", bbox_inches="tight")
    plt.close(fig); print("✓ fig2b_mmr_ts.png")


def fig3b_pretrends(panel, dist, panel_full=None, dist_g=None):
    if panel_full is None: panel_full = panel
    if dist_g is None: dist_g = dist
    df = d.create_buffer_treatment(panel, dist, BASE_INNER, BASE_OUTER,
                                   pd.Timestamp("2025-09-01"))
    df = df[df.period_start >= "2024-10-01"]     # match fig2b window
    gap = (df.assign(grp=np.where(df.is_treated == 1, "T", "C"))
             .groupby(["grp", "period_start"])["log_nightlights"].agg(["mean", "sem"]))
    t, c = gap.xs("T"), gap.xs("C")
    diff = t["mean"] - c["mean"]
    se = np.sqrt(t["sem"]**2 + c["sem"]**2)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axhline(0, color="grey", lw=1)
    ax.fill_between(diff.index, diff - 1.96*se, diff + 1.96*se,
                    color=ps.TREAT, alpha=0.15)
    ax.plot(diff.index, diff, color=ps.TREAT, lw=2, marker="o", ms=3)
    ax.axvline(pd.Timestamp("2025-03-28"), color=ps.ACCENT, lw=1.2)
    ax.axvline(pd.Timestamp("2025-09-01"), color=ps.GREY, lw=1.2, ls=":")
    # Formal pre-trend test: linear interaction on pre-earthquake months (as in 02_pretrends_verified.py)
    import statsmodels.formula.api as smf
    # Test on the FULL pre-earthquake period (Oct 2023 - Mar 2025), matching
    # 02_pretrends_verified.py — a 6-month sub-window is not a valid trends test.
    df_full = d.create_buffer_treatment(panel_full, dist_g, BASE_INNER, BASE_OUTER,
                                        pd.Timestamp("2025-09-01"))
    pre = df_full[df_full.period_start < "2025-03-28"].copy()
    pre["month_idx"] = (pre.period_start.dt.year - 2023)*12 + pre.period_start.dt.month
    m = smf.ols("log_nightlights ~ month_idx * is_treated", data=pre).fit(
        cov_type="cluster", cov_kwds={"groups": pre["unit_id"]})
    p = m.pvalues.get("month_idx:is_treated", np.nan)
    verdict = "parallel (PASS)" if p > 0.10 else "diverging (FAIL)"
    ax.text(0.02, 0.95, f"Pre-trend test (linear interaction,\npre-earthquake months): p = {p:.2f} — {verdict}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec=ps.GREY, alpha=0.9))
    ax.set_ylabel("UNDP − comparison gap, log luminosity")
    ax.set_title("Myanmar pre-trends: UNDP–comparison gap, baseline 3–8 km specification")
    fig.tight_layout()
    fig.savefig(OUT / "fig3b_mmr_pretrends.png", bbox_inches="tight")
    plt.close(fig); print("✓ fig3b_mmr_pretrends.png")




RASTER_DIR = Path("data/mmr/rasters")
YOY_FILES = ["viirs_mmr_202404.tif", "viirs_mmr_202504.tif", "viirs_mmr_202604.tif"]

def fig1a_and_zoom_rasters(gdf, sites):
    paths = [RASTER_DIR / f for f in YOY_FILES]
    if not all(p.exists() for p in paths):
        print("  ⚠ YoY rasters missing in data/mmr/rasters — skip fig1"); return
    if hasattr(sites, "columns"):
        lc = next(c for c in sites.columns if "lon" in c.lower())
        la = next(c for c in sites.columns if "lat" in c.lower())
        lons = sites[lc].to_numpy(); lats = sites[la].to_numpy()
    else:
        lons = np.array([s_["lon"] for s_ in sites]); lats = np.array([s_["lat"] for s_ in sites])
    # Sagaing-Mandalay corridor: study townships excluding the far-south outlier,
    # which otherwise stretches the extent to half the country.
    study = gdf[gdf["group"] != "other"]
    core = study[study.geometry.centroid.y > 18.5]
    minx, miny, maxx, maxy = core.total_bounds
    pad = 0.25
    bbox = (minx-pad, maxx+pad, miny-pad, maxy+pad)
    ps.raster_triptych(paths, PANEL_LABELS, OUT / "fig1a_mmr_triptych.png",
                       bbox=bbox, boundary_gdf=core, sites_lonlat=None,
                       epicenter=(EPI_LON, EPI_LAT),
                       boundary_color="white", boundary_lw=0.5,
                       suptitle="Myanmar: Sagaing–Mandalay corridor, same calendar month across years")
    design_map(gdf, (lons, lats), bbox)
    print("✓ fig1a_mmr_triptych.png")


def design_map(gdf, sites_lonlat, bbox):
    """Study-design map: UNDP vs comparison townships (baseline 3-8 km spec)."""
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    gdf.plot(ax=ax, color="#f2f2f2", edgecolor="#cccccc", linewidth=0.2)
    gdf[gdf.group == "treated"].plot(ax=ax, color=ps.UN_BLUE, alpha=0.75,
                                     edgecolor="white", linewidth=0.4)
    gdf[gdf.group == "control"].plot(ax=ax, color=ps.CONTROL, alpha=0.55,
                                     edgecolor="white", linewidth=0.4)
    ax.scatter(sites_lonlat[0], sites_lonlat[1], s=3, c="#333333", alpha=0.6, linewidths=0)
    ax.scatter([EPI_LON], [EPI_LAT], marker="*", s=150, c=ps.ACCENT, linewidths=0, zorder=11)
    ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
    import matplotlib.patches as mpatches, matplotlib.lines as mlines
    ax.legend(handles=[
        mpatches.Patch(color=ps.UN_BLUE, alpha=0.75, label="UNDP townships (≤3 km of a site)"),
        mpatches.Patch(color=ps.CONTROL, alpha=0.55, label="Comparison townships (3–8 km)"),
        mlines.Line2D([], [], marker="o", color="#333333", ms=4, ls="None", label="UNDP site geocodes"),
        mlines.Line2D([], [], marker="*", color=ps.ACCENT, ms=13, ls="None", label="Epicentre (M7.7)"),
    ], loc="lower left", fontsize=8)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(False)
    ax.set_title("Myanmar study design: baseline 3–8 km specification")
    fig.tight_layout()
    fig.savefig(OUT / "fig1c_mmr_design.png", bbox_inches="tight")
    plt.close(fig); print("✓ fig1c_mmr_design.png")


def fig3a_event_study(panel, dist):
    """TWFE event study around the Sep-2025 deployment (baseline spec):
    monthly treated×month coefficients, ref = Aug 2025."""
    import statsmodels.formula.api as smf
    df = d.create_buffer_treatment(panel, dist, BASE_INNER, BASE_OUTER,
                                   pd.Timestamp("2025-09-01"))
    df = df[df.period_start >= "2024-10-01"].copy()
    df = df.dropna(subset=["log_nightlights"]).reset_index(drop=True)   # keep groups aligned
    df["m"] = df.period_start.dt.strftime("%Y-%m")
    ref = "2025-08"
    ev_cols = []
    for mm in sorted(df["m"].unique()):
        if mm == ref:
            continue
        col = "ev_" + mm.replace("-", "")
        df[col] = ((df["m"] == mm) & (df["is_treated"] == 1)).astype(int)
        ev_cols.append((mm, col))
    formula = "log_nightlights ~ C(unit_id) + C(m) + " + " + ".join(c for _, c in ev_cols)
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["unit_id"]})
    rows = [{"m": mm, "b": model.params[col], "se": model.bse[col]}
            for mm, col in ev_cols if col in model.params]
    es = pd.DataFrame(rows); es["dt"] = pd.to_datetime(es["m"] + "-01"); es = es.sort_values("dt")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axhline(0, color="grey", lw=1)
    ax.axvline(pd.Timestamp("2025-03-28"), color=ps.ACCENT, lw=1.2)
    ax.axvline(pd.Timestamp("2025-09-01"), color=ps.GREY, lw=1.2, ls=":")
    ax.fill_between(es.dt, es.b - 1.96*es.se, es.b + 1.96*es.se, color=ps.TREAT, alpha=0.15)
    ax.plot(es.dt, es.b, color=ps.TREAT, lw=1.8, marker="o", ms=4)
    ax.text(pd.Timestamp("2025-03-28"), ax.get_ylim()[1], " Earthquake",
            color=ps.ACCENT, va="top", fontsize=9)
    ax.text(pd.Timestamp("2025-09-01"), ax.get_ylim()[0], " UNDP deployment",
            color=ps.GREY, va="bottom", fontsize=9)
    ax.set_ylabel("Monthly DiD coefficient, log luminosity")
    ax.set_title("Myanmar event study: rise after deployment, convergence by 2026 (3–8 km spec)")
    fig.tight_layout()
    fig.savefig(OUT / "fig3a_mmr_event_study.png", bbox_inches="tight")
    plt.close(fig)
    es.to_csv(Path("outputs/mmr/results/event_study_monthly.csv"), index=False)
    print("✓ fig3a_mmr_event_study.png (+ event_study_monthly.csv)")


def main():
    panel, gdf, id_col, sites, dist = load_all()
    fig1a_and_zoom_rasters(gdf, sites)
    fig2b_ts(panel, dist)
    fig3a_event_study(panel, dist)
    fig3b_pretrends(panel, dist, panel, dist)
    print(f"\nAll saved to {OUT}/")

if __name__ == "__main__":
    main()
