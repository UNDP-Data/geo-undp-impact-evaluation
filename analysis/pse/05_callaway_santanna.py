#!/usr/bin/env python3
"""
analysis/pse/05_callaway_santanna.py
=====================================
Heterogeneity-robust staggered DiD for Gaza using the Callaway & Sant'Anna
(2021) group-time ATT estimator — referee-proofing for the TWFE results in
01_did_comprehensive.py, whose staggered timing is exactly the setting where
TWFE can be contaminated by negative weighting.

DESIGN
------
- Reuses 01_did_comprehensive.py's own loaders and panel builder (imported by
  path), so treatment definition, donut controls, and raster extraction are
  IDENTICAL to the headline specification. Baseline donut: inner 500 m,
  outer 1.5 km (the paper's primary specification).
- Unit: site_id. Time: integer month index. Group g: the calendar month of the
  site's project_start (first treatment period); never-treated donut controls
  have g = 0 and anchor the comparison group.
- Estimator: doubly robust (Sant'Anna & Zhao 2020) via the `csdid` package
  (Python port of the R `did` package). Unconditional parallel trends
  (no covariates), consistent with reporting the raw-luminosity pre-trend
  violation transparently; SAR VH is the clean-pretrends channel.
- Outcomes: log_viirs (primary), sar_vh (physical reconstruction channel).

OUTPUTS
-------
  outputs/pse/site_panel_baseline.csv          (the exact panel used — reproducibility)
  outputs/pse/tables/cs_did_results.csv        (simple aggregated ATT per outcome)
  outputs/pse/tables/cs_did_event_study.csv    (dynamic ATTs by event time)
  outputs/pse/figures/cs_event_study_<outcome>.png

REFERENCES (add to paper if not present)
----------------------------------------
  Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with
    multiple time periods. Journal of Econometrics, 225(2), 200-230.
  Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust difference-in-differences
    estimators. Journal of Econometrics, 219(1), 101-122.

Usage:
  .venv/bin/pip install csdid   # once
  .venv/bin/python analysis/pse/05_callaway_santanna.py
"""

import io
import sys
import contextlib
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# Import 01_did_comprehensive.py by path (filename starts with a digit)
# =============================================================================
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("did01", HERE / "01_did_comprehensive.py")
did01 = importlib.util.module_from_spec(spec)
sys.modules["did01"] = did01
spec.loader.exec_module(did01)

OUTPUT_DIR = Path("outputs/pse")
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OUTCOMES = ["log_viirs", "sar_vh"]   # primary + clean-pretrends channel
BASE_MONTH = pd.Timestamp("2023-01-01")   # panel start -> month index 1


def month_index(ts):
    return (ts.year - BASE_MONTH.year) * 12 + (ts.month - BASE_MONTH.month) + 1


def get_baseline_spec():
    """Pick the baseline donut (inner 500 m, outer 1.5 km) from 01's own specs."""
    for tup in did01.DONUT_SPECS:
        inner_m, outer_m = tup[0], tup[1]
        key = tup[3] if len(tup) > 3 else ""
        if key == "baseline" or (inner_m == 500 and outer_m == 1500):
            return tup
    raise RuntimeError("Baseline donut spec (500m-1.5km) not found in DONUT_SPECS")


def build_cs_frame(df_long, outcome):
    """Long panel (one row per site-month-indicator) -> csdid frame."""
    d = df_long[df_long["indicator"] == outcome].copy()
    if d.empty:
        return None
    d["date"] = pd.to_datetime(d["date"])
    d["time"] = d["date"].apply(month_index)

    # Group g: month index of project_start for treated; 0 for never-treated
    d["project_start"] = pd.to_datetime(d["project_start"], errors="coerce")
    d["g"] = 0
    tr = d["group"].eq("Treated") & d["project_start"].notna()
    d.loc[tr, "g"] = d.loc[tr, "project_start"].apply(month_index)

    # Numeric ids
    d["uid"] = pd.factorize(d["site_id"])[0] + 1
    frame = d[["uid", "time", "g", "value"]].dropna().rename(columns={"value": "y"})

    # csdid needs a balanced-ish panel; keep units observed in >= 80% of months
    n_t = frame["time"].nunique()
    counts = frame.groupby("uid")["time"].nunique()
    keep = counts[counts >= 0.8 * n_t].index
    frame = frame[frame["uid"].isin(keep)]

    # Drop groups treated before/at panel start or with no pre-period
    frame = frame[(frame["g"] == 0) | (frame["g"] > frame["time"].min() + 1)]
    return frame


def run_cs(frame, outcome):
    from csdid.att_gt import ATTgt
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = ATTgt(yname="y", tname="time", idname="uid", gname="g",
                    data=frame).fit(est_method="dr")
        simple = out.aggte(typec="simple")
        dyn = out.aggte(typec="dynamic")

    att = float(np.asarray(simple.atte["overall_att"]).ravel()[0])
    se = float(np.asarray(simple.atte["overall_se"]).ravel()[0])
    z = att / se if se > 0 else np.nan
    from scipy import stats
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    egt = np.asarray(dyn.atte["egt"]).ravel()
    att_e = np.asarray(dyn.atte["att_egt"]).ravel()
    se_e = np.asarray(dyn.atte["se_egt"]).ravel()

    return {
        "outcome": outcome, "att": att, "se": se, "z": z, "pval": p,
        "pct_effect": 100 * (np.exp(att) - 1) if outcome.startswith("log") else np.nan,
        "n_units": frame["uid"].nunique(),
        "n_treated_groups": frame.loc[frame.g > 0, "g"].nunique(),
        "n_never_treated": frame.loc[frame.g == 0, "uid"].nunique(),
    }, pd.DataFrame({"event_time": egt, "att": att_e, "se": se_e, "outcome": outcome})


def plot_event_study(es, outcome, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0, color="grey", lw=1)
    ax.axvline(-0.5, color="red", lw=1, ls="--", alpha=0.6)
    ax.errorbar(es["event_time"], es["att"], yerr=1.96 * es["se"],
                fmt="o", ms=4, capsize=3, color="#1f4e79")
    ax.set_xlabel("Months relative to site's project start")
    ax.set_ylabel(f"ATT ({outcome})")
    ax.set_title(f"Callaway-Sant'Anna dynamic ATTs — {outcome} (baseline donut 500m-1.5km)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    print("=" * 70)
    print(" GAZA: CALLAWAY-SANT'ANNA GROUP-TIME ATT (baseline donut 500m-1.5km)")
    print("=" * 70)

    sites = did01.load_sites_with_projects()
    if sites is None:
        print("  ❌ sites failed"); return
    acled = did01.load_acled()
    panel_controls = did01.load_district_panel()
    rasters = did01.find_rasters()

    inner_m, outer_m = get_baseline_spec()[0], get_baseline_spec()[1]
    print(f"\n--- Building baseline panel ({inner_m}m-{outer_m}m), identical to 01 ---")
    df = did01.build_analysis_dataset(sites, acled, rasters, inner_m, outer_m, panel_controls)
    if df.empty:
        print("  ❌ empty panel"); return

    panel_path = OUTPUT_DIR / "site_panel_baseline.csv"
    df.to_csv(panel_path, index=False)
    print(f"  ✓ Panel saved: {panel_path} ({len(df):,} rows)")

    results, event_frames = [], []
    for outcome in OUTCOMES:
        frame = build_cs_frame(df, outcome)
        if frame is None or frame.empty:
            print(f"  ⚠ {outcome}: no data"); continue
        print(f"\n--- {outcome}: {frame['uid'].nunique()} units, "
              f"{frame.loc[frame.g>0,'g'].nunique()} treatment cohorts, "
              f"{frame.loc[frame.g==0,'uid'].nunique()} never-treated controls ---")
        try:
            res, es = run_cs(frame, outcome)
        except Exception as e:
            print(f"  ❌ estimator failed for {outcome}: {e}")
            continue
        results.append(res)
        event_frames.append(es)
        plot_event_study(es, outcome, FIGURES_DIR / f"cs_event_study_{outcome}.png")

        pct = f" -> {res['pct_effect']:+.1f}% vs controls" if not np.isnan(res["pct_effect"]) else ""
        stars = "***" if res["pval"] < 0.01 else "**" if res["pval"] < 0.05 else "*" if res["pval"] < 0.1 else ""
        print(f"  Simple aggregated ATT: {res['att']:+.4f}{stars} (SE {res['se']:.4f}, p={res['pval']:.4f}){pct}")

    if results:
        pd.DataFrame(results).to_csv(TABLES_DIR / "cs_did_results.csv", index=False)
        pd.concat(event_frames).to_csv(TABLES_DIR / "cs_did_event_study.csv", index=False)
        print(f"\n  ✓ Saved cs_did_results.csv, cs_did_event_study.csv, figures")
        print("\n  COMPARE: TWFE log_viirs baseline gave +17.6% (gdp_sensitivity.csv).")
        print("  If the CS estimate is same-signed and similar magnitude, the TWFE")
        print("  headline survives heterogeneity-robust estimation — report both.")


if __name__ == "__main__":
    main()
