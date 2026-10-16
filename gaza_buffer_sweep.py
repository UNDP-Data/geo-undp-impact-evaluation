#!/usr/bin/env python3
"""
Gaza buffer sweep: is the identification failure a proximity issue?

For each donut band (inner-outer, metres), builds the estimation panel with the
paper's own functions (via gaza_export.py, skipped if the file already exists),
then for VIIRS and SAR VH reports:
  pooled treated x post under  1 paper exact | 3 site FE | 4 site FE + brightness x month
                               | 5 site FE, conflict-period baseline
  event-study pre-trend tests (joint and slope) under spec 5, the validated diagnostic.

Note: a VIIRS pixel is ~500 m, so for inner radii under ~750 m treated sites and
controls can share pixels. SAR VH (10 m) is the cleaner proximity test.

Usage (repo root, venv active):
    python3 gaza_buffer_sweep.py
    python3 gaza_buffer_sweep.py --bands 250-750,500-1500,1000-3000
Each new band runs the slow raster step once (a few minutes each).
"""
import argparse, subprocess, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--bands", default="250-750,500-1500,750-2000,1000-3000,1500-4000")
ap.add_argument("--window", type=int, default=12)
ap.add_argument("--skip-export", action="store_true")
a = ap.parse_args()
CONFLICT_START = pd.Timestamp("2023-10-01")
K = a.window


def demean(df, cols, fes, tol=1e-9, maxit=1000):
    X = df[cols].astype(float).copy()
    for _ in range(maxit):
        old = X.values.copy()
        for f in fes:
            X = X - X.groupby(df[f]).transform("mean")
        if np.max(np.abs(X.values - old)) < tol:
            break
    return X


def fit(df, y, regs, fes):
    Z = demean(df, [y] + regs, fes)
    return sm.OLS(Z[y].values, Z[regs].values).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df["unit_id"])[0]})


def prep(path):
    D = pd.read_csv(path, parse_dates=["period_start", "treat_date"])
    D["post"] = (D["treat_date"].notna() & (D["period_start"] >= D["treat_date"])).astype(int)
    D["tp"] = D["treated"] * D["post"]
    D["month"] = D["period_start"].dt.strftime("%Y-%m")
    D["district"] = D["district"].astype(str)
    D["site_conflict_events"] = D["site_conflict_events"].fillna(0)
    pre_b = D[D.period_start < CONFLICT_START].groupby("unit_id")["viirs"].mean()
    D["bright_t"] = D["unit_id"].map(pd.qcut(pre_b.rank(method="first"), 3, labels=False)).fillna(-1).astype(int)
    D["tercile_month"] = D["bright_t"].astype(str) + "_" + D["month"]
    rel = ((D["period_start"].dt.year - D["treat_date"].dt.year) * 12
           + (D["period_start"].dt.month - D["treat_date"].dt.month))
    D["rel"] = rel.clip(-K, K)
    return D


def pooled(D, y):
    out = {}
    for key, regs, fes, conflict in [("s1", ["tp", "site_conflict_events"], ["district", "month"], False),
                                     ("s3", ["tp", "site_conflict_events"], ["unit_id", "month"], False),
                                     ("s4", ["tp", "site_conflict_events"], ["unit_id", "tercile_month"], False),
                                     ("s5", ["tp", "site_conflict_events"], ["unit_id", "month"], True)]:
        df = D[D.period_start >= CONFLICT_START] if conflict else D
        df = df.dropna(subset=[y])
        m = fit(df, y, regs, fes)
        out[key] = (m.params[0], m.pvalues[0], m.bse[0])
    return out


def pretrend_s5(D, y):
    df = D[(D.period_start >= CONFLICT_START)].dropna(subset=[y]).copy()
    names, rs = [], []
    for r in [r for r in range(-K, K + 1) if r != -1]:
        c = f"e{'m' if r < 0 else 'p'}{abs(r)}"
        df[c] = ((df["treated"] == 1) & (df["rel"] == r)).astype(float)
        if df[c].sum() > 0:
            names.append(c); rs.append(r)
    m = fit(df, y, names + ["site_conflict_events"], ["unit_id", "month"])
    pre = [i for i, r in enumerate(rs) if r < -1]
    if len(pre) < 3:
        return np.nan, np.nan
    R = np.zeros((len(pre), len(names) + 1))
    for j, i in enumerate(pre):
        R[j, i] = 1
    p_joint = float(m.wald_test(R, scalar=True).pvalue)
    rr = np.array([rs[i] for i in pre], float)
    cw = (rr - rr.mean()) / ((rr - rr.mean()) ** 2).sum()
    V = m.cov_params()[np.ix_(pre, pre)]
    z = (cw @ m.params[pre]) / np.sqrt(cw @ V @ cw)
    return p_joint, float(2 * (1 - norm.cdf(abs(z))))


def fmt(b, p, log):
    star = "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
    return (f"{100*(np.exp(b)-1):+6.1f}%{star:3s}" if log else f"{b:+7.3f}{star:3s}")


rows = []
for band in a.bands.split(","):
    inner, outer = [int(x) for x in band.split("-")]
    path = Path(f"outputs/pse/estimation_panel_{inner}_{outer}_all.csv")
    if not path.exists():
        if a.skip_export:
            print(f"[{band}] panel missing, skipped"); continue
        print(f"[{band}] building panel (slow raster step)...", flush=True)
        r = subprocess.run([sys.executable, "gaza_export.py", "--inner", str(inner),
                            "--outer", str(outer), "--indicator", "all"],
                           capture_output=True, text=True)
        if not path.exists():
            print(f"[{band}] export failed:\n{r.stdout[-600:]}\n{r.stderr[-600:]}"); continue
    D = prep(path)
    rec = {"band": band, "treated": D.loc[D.treated == 1, "unit_id"].nunique(),
           "controls": D.loc[D.treated == 0, "unit_id"].nunique()}
    for oc in ["viirs", "sar_vh"]:
        if oc not in D.columns:
            continue
        D["y"] = np.log1p(D[oc].clip(lower=0)) if oc == "viirs" else D[oc]
        po = pooled(D, "y")
        pj, ps = pretrend_s5(D, "y")
        for k, (b, p, se) in po.items():
            rec[f"{oc}_{k}_b"], rec[f"{oc}_{k}_p"], rec[f"{oc}_{k}_se"] = b, p, se
        rec[f"{oc}_pre_joint_p"], rec[f"{oc}_pre_slope_p"] = pj, ps
    rows.append(rec)
    print(f"[{band}] done", flush=True)

if not rows:
    sys.exit("no bands estimated")
R = pd.DataFrame(rows)
R.to_csv("gaza_buffer_sweep.csv", index=False)

for oc, log, title in [("viirs", True, "NIGHTTIME LIGHTS, log(VIIRS+1)"), ("sar_vh", False, "SAR VH (dB)")]:
    if f"{oc}_s1_b" not in R.columns:
        continue
    print("\n" + "=" * 100)
    print(f"{title}  |  pooled treated x post by band; pre-trend tests under spec 5 (validated diagnostic)")
    print("=" * 100)
    print(f"{'band (m)':12s} {'sites T/C':>10s} {'1 paper':>12s} {'3 site FE':>12s} {'4 bright':>12s} "
          f"{'5 conflict':>12s} {'pre joint p':>12s} {'pre slope p':>12s}  verdict")
    for _, r in R.iterrows():
        pj, ps = r[f"{oc}_pre_joint_p"], r[f"{oc}_pre_slope_p"]
        worst = np.nanmin([pj, ps])
        verdict = "PASS" if worst > 0.10 else "marginal" if worst > 0.05 else "FAIL"
        print(f"{r.band:12s} {str(r.treated)+'/'+str(r.controls):>10s} "
              f"{fmt(r[f'{oc}_s1_b'], r[f'{oc}_s1_p'], log):>12s} {fmt(r[f'{oc}_s3_b'], r[f'{oc}_s3_p'], log):>12s} "
              f"{fmt(r[f'{oc}_s4_b'], r[f'{oc}_s4_p'], log):>12s} {fmt(r[f'{oc}_s5_b'], r[f'{oc}_s5_p'], log):>12s} "
              f"{pj:12.3f} {ps:12.3f}  {verdict}")
print("\nSaved gaza_buffer_sweep.csv")
print("Reading: if pre-trends PASS in some bands (especially tight ones for SAR), proximity")
print("matters and those bands can support a causal estimate. If they FAIL everywhere, the")
print("identification problem is the conflict setting, not the choice of band.")
