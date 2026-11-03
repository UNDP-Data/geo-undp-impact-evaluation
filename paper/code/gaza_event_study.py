#!/usr/bin/env python3
"""
Gaza event studies under the fair specifications, with pre-trend tests.

For each outcome, estimates monthly event-time coefficients relative to each
site's own programme start (staggered), reference month -1 omitted, event
time binned to [-K, K] with endpoint bins, SEs clustered by site:

  Spec 4  site FE + month FE x pre-conflict brightness tercile + conflict control
  Spec 5  site FE + month FE + conflict control, conflict-period sample only

Reports the joint pre-trend test, the post-period average, and a simple
sensitivity check (does the post average exceed the largest pre-period
deviation?). Saves coefficients to CSV and a figure for the paper.

Usage (repo root, venv active; needs estimation_panel_500_1500_all.csv):
    python3 gaza_event_study.py
    python3 gaza_event_study.py --debris          # debris-management sites only
    python3 gaza_event_study.py --window 18       # wider event window
"""
import argparse, sys, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--panel", default="outputs/pse/estimation_panel_500_1500_all.csv")
ap.add_argument("--window", type=int, default=12)
ap.add_argument("--outcomes", default="viirs,sar_vh,bsi,ndbi,ndvi")
ap.add_argument("--debris", action="store_true")
a = ap.parse_args()

CONFLICT_START = pd.Timestamp("2023-10-01")
K = a.window
try:
    D = pd.read_csv(a.panel, parse_dates=["period_start", "treat_date"])
except FileNotFoundError:
    sys.exit(f"{a.panel} not found. Run: python3 gaza_export.py --inner 500 --outer 1500 --indicator all")

if a.debris:
    deb = D["intervention_type"].astype(str).str.contains("debris", case=False, na=False)
    keep = set(D.loc[deb & (D.treated == 1), "unit_id"]) | set(D.loc[D.treated == 0, "unit_id"])
    D = D[D.unit_id.isin(keep)].copy()
LABEL = "debris" if a.debris else "all"

OUTCOMES = [o for o in a.outcomes.split(",") if o in D.columns]
D["month"] = D["period_start"].dt.strftime("%Y-%m")
D["site_conflict_events"] = D["site_conflict_events"].fillna(0)
pre_b = D[D.period_start < CONFLICT_START].groupby("unit_id")["viirs"].mean()
D["bright_t"] = D["unit_id"].map(pd.qcut(pre_b.rank(method="first"), 3, labels=False)).fillna(-1).astype(int)
D["tercile_month"] = D["bright_t"].astype(str) + "_" + D["month"]

# event time relative to each treated site's own start; controls carry no event dummies
rel = ((D["period_start"].dt.year - D["treat_date"].dt.year) * 12
       + (D["period_start"].dt.month - D["treat_date"].dt.month))
D["rel"] = rel.clip(-K, K)
BINS = [r for r in range(-K, K + 1) if r != -1]


def add_dummies(df):
    cols = []
    for r in BINS:
        c = f"e{'m' if r < 0 else 'p'}{abs(r)}"
        df[c] = ((df["treated"] == 1) & (df["rel"] == r)).astype(float)
        if df[c].sum() > 0:
            cols.append((c, r))
    return cols


def demean(df, cols, fes, tol=1e-9, maxit=1000):
    X = df[cols].astype(float).copy()
    for _ in range(maxit):
        old = X.values.copy()
        for f in fes:
            X = X - X.groupby(df[f]).transform("mean")
        if np.max(np.abs(X.values - old)) < tol:
            break
    return X


def event_study(df, ycol, fes):
    df = df.dropna(subset=[ycol]).copy()
    dcols = add_dummies(df)
    names = [c for c, _ in dcols] + ["site_conflict_events"]
    Z = demean(df, [ycol] + names, fes)
    m = sm.OLS(Z[ycol].values, Z[names].values).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df["unit_id"])[0]})
    coefs = pd.DataFrame({"rel": [r for _, r in dcols],
                          "beta": m.params[:len(dcols)], "se": m.bse[:len(dcols)]})
    pre_idx = [i for i, (_, r) in enumerate(dcols) if r < -1]
    R = np.zeros((len(pre_idx), len(names)))
    for j, i in enumerate(pre_idx):
        R[j, i] = 1
    wt = m.wald_test(R, scalar=True)
    # linear pre-trend slope test: far more powerful than the joint test against drift
    rs = np.array([r for _, r in dcols if r < -1], dtype=float)
    if len(rs) >= 3:
        cw = (rs - rs.mean()) / ((rs - rs.mean()) ** 2).sum()
        V = m.cov_params()[np.ix_(pre_idx, pre_idx)]
        b = m.params[pre_idx]
        slope = float(cw @ b); se_s = float(np.sqrt(cw @ V @ cw))
        from scipy.stats import norm
        p_slope = float(2 * (1 - norm.cdf(abs(slope / se_s))))
    else:
        slope, p_slope = np.nan, np.nan
    return coefs, float(wt.pvalue), slope, p_slope


SPECS = [("4 brightness x month", ["unit_id", "tercile_month"], None),
         ("5 conflict-period base", ["unit_id", "month"], "conflict")]
rows, plots = [], []
print(f"Gaza event studies ({LABEL} sites) | window +/-{K} months, ref -1 | "
      f"{D.loc[D.treated==1,'unit_id'].nunique()} treated, {D.loc[D.treated==0,'unit_id'].nunique()} controls")
print("=" * 86)
for oc in OUTCOMES:
    is_log = oc == "viirs"
    D["y"] = np.log1p(D[oc].clip(lower=0)) if is_log else D[oc]
    sd = D.loc[D.period_start < CONFLICT_START, "y"].std()
    for sname, fes, restr in SPECS:
        df = D[D.period_start >= CONFLICT_START] if restr == "conflict" else D
        try:
            co, p_pre, slope, p_slope = event_study(df, "y", fes)
        except Exception as e:
            print(f"{oc.upper():7s} spec {sname}: failed ({e})"); continue
        pre = co[co.rel < -1]; post = co[co.rel >= 0]
        post_avg = post.beta.mean()
        pre_max = pre.beta.abs().max() if len(pre) else np.nan
        eff = f"{100*(np.exp(post_avg)-1):+6.1f}%" if is_log else f"{post_avg/sd:+5.2f} SD"
        worst = min(p_pre, p_slope if not np.isnan(p_slope) else 1.0)
        verdict = ("PASS" if worst > 0.10 else "marginal" if worst > 0.05 else "FAIL")
        exceeds = "yes" if abs(post_avg) > pre_max else "no"
        print(f"{oc.upper():7s} spec {sname:24s} joint p={p_pre:5.3f} slope p={p_slope:5.3f} [{verdict:8s}]  "
              f"post avg={post_avg:+.4f} ({eff})  |post| > max|pre|: {exceeds}")
        co["outcome"], co["spec"], co["pre_p"], co["slope_p"] = oc, sname, p_pre, p_slope
        rows.append(co)
        plots.append((oc, sname, co, p_pre))
    print("-" * 86)

out = pd.concat(rows)
csv = f"gaza_event_study_{LABEL}.csv"
out.to_csv(csv, index=False)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(OUTCOMES)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.6 * n), squeeze=False)
    titles = {"viirs": "Nighttime lights, log(VIIRS+1)", "sar_vh": "SAR VH (dB)",
              "bsi": "Bare soil index", "ndbi": "Built-up index", "ndvi": "NDVI"}
    for oc, sname, co, p in plots:
        r = OUTCOMES.index(oc); c = 0 if sname.startswith("4") else 1
        ax = axes[r][c]
        ax.axhline(0, color="grey", lw=0.8); ax.axvline(-0.5, color="red", ls="--", lw=1)
        ax.errorbar(co.rel, co.beta, yerr=1.96 * co.se, fmt="o", ms=3, capsize=2,
                    color="#1f4e79")
        ax.set_title(f"{titles.get(oc, oc)} | {'brightness-adjusted' if c == 0 else 'conflict-period baseline'}"
                     f" | pre-trend p={p:.2f}", fontsize=9)
        ax.set_xlabel("Months relative to programme start", fontsize=8)
    fig.tight_layout()
    png = f"gaza_event_study_{LABEL}.png"
    fig.savefig(png, dpi=180)
    print(f"Saved {csv} and {png}")
except Exception as e:
    print(f"Saved {csv} (figure skipped: {e})")

print("\nReading guide (validated on synthetic data):")
print("  Spec 5 (conflict-period baseline) is the primary identification test: it passed a")
print("  clean case and caught a planted pre-trend drift. Spec 4's tests are less reliable,")
print("  because brightness terciles correlate with treatment and absorb part of any trend.")
print("  Causal-grade: spec 5 PASSES (joint and slope p > 0.10) and spec 4 gives the same sign.")
print("  Otherwise report the estimate as a descriptive difference, not an effect.")
