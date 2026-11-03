#!/usr/bin/env python3
"""
Part 3 diagnostics: does the log(VIIRS+1) result survive a proper estimator?

Runs, on one panel:
  A. Share of near-zero radiance observations (how much the +1 shift matters)
  B. Three estimators of the same treated x post contrast, same fixed effects
     (unit + month), SEs clustered by unit:
        1. OLS on log(y+1)            <- what the paper currently reports
        2. PPML (Poisson) on raw y     <- proportional effect, no +1 shift, handles zeros
        3. OLS on y / unit pre-mean    <- levels effect as a share of baseline
  C. Event study (monthly, OLS log(y+1) and PPML), plus a linear pre-trend
     extrapolation check: post effect net of the extrapolated pre-trend.
     This is the "first look" version of Rambachan-Roth sensitivity.

Treatment assignment, in order of preference:
  --treat-col NAME             column already in the panel (0/1)
  --treated-ids FILE           csv with a unit_id column listing treated units
                               (optional --control-ids FILE; else all others)
  --sites FILE --boundaries F  compute it: unit centroid distance to nearest
     --inner KM --outer KM     site; treated if <= inner, control if in
                               (inner, outer]; others dropped (donut design)

Usage (Myanmar baseline 3-8 km, computed assignment):
  python3 part3_diagnostics.py --panel outputs/mmr/panel_data.csv \
      --event-date 2025-09-01 --sites data/mmr/external/undp_projects.csv \
      --boundaries data/mmr/boundaries/mmr_admin3.geojson --inner 3 --outer 8 \
      --label myanmar_3_8km

Outputs: part3_<label>_results.csv, part3_<label>_eventstudy.csv, and a
KEY RESULTS block on screen to paste back.
"""
import argparse, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--panel", required=True)
ap.add_argument("--event-date", required=True)
ap.add_argument("--unit-col", default="unit_id")
ap.add_argument("--date-col", default="period_start")
ap.add_argument("--outcome", default="nightlights")
ap.add_argument("--treat-col")
ap.add_argument("--treated-ids"); ap.add_argument("--control-ids")
ap.add_argument("--sites"); ap.add_argument("--boundaries")
ap.add_argument("--inner", type=float); ap.add_argument("--outer", type=float)
ap.add_argument("--id-column", default=None, help="id column in boundaries (default: auto)")
ap.add_argument("--lat-col"); ap.add_argument("--lon-col")
ap.add_argument("--label", default="run")
ap.add_argument("--near-zero", type=float, default=0.5, help="nW threshold for 'near zero'")
ap.add_argument("--treat-date-col", help="per-unit treatment date column (staggered adoption); "
                                         "event-date then only defines rel-time for controls")
ap.add_argument("--ev-window", type=int, default=None,
                help="bin event time into [-K, K] with endpoint bins (fewer coefficients, "
                     "more reliable joint test with few clusters)")
ap.add_argument("--drop-cal-months", default=None,
                help="comma-separated calendar months to drop, e.g. 6,7 (monsoon cloud contamination)")
ap.add_argument("--ref-month", type=int, default=-1,
                help="event-study reference month relative to event (default -1). "
                     "Use e.g. -7 to anchor before an earthquake that precedes programming")
ap.add_argument("--pretrend-end", type=int, default=-1,
                help="last relative month counted in the joint pre-trend test (default -1). "
                     "Use e.g. -7 to test parallel trends on pre-earthquake months only")
a = ap.parse_args()

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError:
    sys.exit("statsmodels missing: pip3 install statsmodels")

# ── load panel ─────────────────────────────────────────────────────────────
d = pd.read_csv(a.panel)
for c in (a.unit_col, a.date_col, a.outcome):
    if c not in d.columns:
        sys.exit(f"column '{c}' not in panel. Columns: {list(d.columns)[:30]}")
d[a.date_col] = pd.to_datetime(d[a.date_col])
d = d[[a.unit_col, a.date_col, a.outcome]].dropna().copy()
d.columns = ["unit", "date", "y"]
d["unit"] = d["unit"].astype(str)
E = pd.Timestamp(a.event_date)
if a.drop_cal_months:
    _drop = {int(x) for x in a.drop_cal_months.split(",")}
    d = d[~d["date"].dt.month.isin(_drop)].copy()
d["month"] = d["date"].dt.to_period("M").astype(str)
if a.treat_date_col:
    _td = pd.read_csv(a.panel, usecols=[a.unit_col, a.treat_date_col]).drop_duplicates(a.unit_col)
    _tdm = dict(zip(_td[a.unit_col].astype(str), pd.to_datetime(_td[a.treat_date_col], errors="coerce")))
    d["tdate"] = d["unit"].map(_tdm)
    d["post"] = ((d["tdate"].notna()) & (d["date"] >= d["tdate"])).astype(int)
else:
    d["tdate"] = E
    d["post"] = (d["date"] >= E).astype(int)

# ── treatment assignment ───────────────────────────────────────────────────
if a.treat_col:
    src = pd.read_csv(a.panel, usecols=[a.unit_col, a.treat_col]).drop_duplicates(a.unit_col)
    tmap = dict(zip(src[a.unit_col].astype(str), src[a.treat_col].astype(int)))
    d["treat"] = d["unit"].map(tmap)
    d = d[d["treat"].notna()]
    how = f"panel column '{a.treat_col}'"
elif a.treated_ids:
    tr = set(pd.read_csv(a.treated_ids).iloc[:, 0].astype(str))
    if a.control_ids:
        co = set(pd.read_csv(a.control_ids).iloc[:, 0].astype(str))
        d = d[d["unit"].isin(tr | co)]
    d["treat"] = d["unit"].isin(tr).astype(int)
    how = f"id lists ({len(tr)} treated)"
elif a.sites and a.boundaries and a.inner and a.outer:
    import geopandas as gpd
    g = gpd.read_file(a.boundaries)
    if g.crs is not None and g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    idc = a.id_column or next(c for c in g.columns if "id" in c.lower())
    cen = g.geometry.centroid            # same as 01_did_buffer_fixed.py (degrees)
    s = pd.read_csv(a.sites)
    latc = a.lat_col or next((c for c in s.columns if "lat" in c.lower()), None)
    lonc = a.lon_col or next((c for c in s.columns if "lon" in c.lower()), None)
    if latc is None or lonc is None:
        sys.exit(f"No coordinate columns in {a.sites}. Columns found: {list(s.columns)}\n"
                 "Point --sites at the geocoded file, or pass --lat-col/--lon-col.")
    s = s.dropna(subset=[latc, lonc])
    slat = np.radians(s[latc].astype(float).values); slon = np.radians(s[lonc].astype(float).values)
    def min_hav_km(lat, lon):
        lat, lon = np.radians(lat), np.radians(lon)
        h = np.sin((slat - lat) / 2) ** 2 + np.cos(lat) * np.cos(slat) * np.sin((slon - lon) / 2) ** 2
        return (2 * 6371.0 * np.arcsin(np.sqrt(h))).min()
    dist_km = np.array([min_hav_km(p.y, p.x) for p in cen])
    assign = pd.DataFrame({"unit": g[idc].astype(str), "dist_km": dist_km})
    assign["treat"] = np.where(assign.dist_km <= a.inner, 1,
                        np.where(assign.dist_km <= a.outer, 0, np.nan))
    assign = assign.dropna(subset=["treat"])
    _panel_ids = set(d["unit"].unique())
    d = d.merge(assign[["unit", "treat"]], on="unit")
    if len(d) == 0:
        sys.exit("Panel unit ids do not match boundary ids.\n"
                 f"  panel sample:      {sorted(_panel_ids)[:4]}\n"
                 f"  boundaries sample: {assign['unit'].head(4).tolist()} (column '{idc}')\n"
                 "  Pass the matching column with --id-column.")
    d["treat"] = d["treat"].astype(int)
    how = f"centroid distance to nearest site: treated <= {a.inner} km, control ({a.inner}, {a.outer}] km"
else:
    sys.exit("No treatment assignment given. Use --treat-col, --treated-ids, or --sites/--boundaries/--inner/--outer.")

d["tp"] = d["treat"] * d["post"]
n_t, n_c = d.loc[d.treat == 1, "unit"].nunique(), d.loc[d.treat == 0, "unit"].nunique()
if n_t == 0 or n_c == 0:
    sys.exit(f"Degenerate assignment: {n_t} treated, {n_c} control units.")

# ── A. near-zero share ─────────────────────────────────────────────────────
share_nz = (d["y"] < a.near_zero).mean()
share_0 = (d["y"] <= 0).mean()
share_nz_t = (d.loc[d.treat == 1, "y"] < a.near_zero).mean()

# ── B. three estimators ────────────────────────────────────────────────────
d["log1y"] = np.log1p(d["y"].clip(lower=0))
pre_mean = d[d.post == 0].groupby("unit")["y"].mean()
d["y_norm"] = d["y"] / d["unit"].map(pre_mean)
d = d[np.isfinite(d["y_norm"])]

def twfe_ols(dep):
    m = smf.ols(f"{dep} ~ tp + C(unit) + C(month)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["unit"]})
    return m.params["tp"], m.bse["tp"], m.pvalues["tp"]

def twfe_ppml():
    X = pd.get_dummies(d[["unit", "month"]], drop_first=True).astype(float)
    X.insert(0, "tp", d["tp"].values); X = sm.add_constant(X)
    yv = d["y"].clip(lower=0).values
    m = sm.GLM(yv, X, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["unit"])[0]})
    return m.params["tp"], m.bse["tp"], m.pvalues["tp"]

b1, s1, p1 = twfe_ols("log1y")
b2, s2, p2 = twfe_ppml()
b3, s3, p3 = twfe_ols("y_norm")
rows = [
    ("OLS log(y+1)  [current paper]", b1, s1, p1, 100 * (np.exp(b1) - 1)),
    ("PPML raw y    [Chen-Roth fix]", b2, s2, p2, 100 * (np.exp(b2) - 1)),
    ("OLS y/pre-mean [levels share]", b3, s3, p3, 100 * b3),
]
res = pd.DataFrame(rows, columns=["estimator", "beta", "se", "p", "implied_pct"])
res["ci_lo_pct"] = [100 * (np.exp(b - 1.96 * s) - 1) if i < 2 else 100 * (b - 1.96 * s)
                    for i, (b, s) in enumerate(zip(res.beta, res.se))]
res["ci_hi_pct"] = [100 * (np.exp(b + 1.96 * s) - 1) if i < 2 else 100 * (b + 1.96 * s)
                    for i, (b, s) in enumerate(zip(res.beta, res.se))]
res.to_csv(f"part3_{a.label}_results.csv", index=False)

print("\n--- B (early print) ---")
for _, r in res.iterrows():
    print(f"   {r.estimator:32s} {r.beta:8.3f} {r.se:7.3f} {r.p:7.3f} {r.implied_pct:9.1f}% "
          f"[{r.ci_lo_pct:6.1f}, {r.ci_hi_pct:6.1f}]")
# ── C. event study + pre-trend extrapolation ───────────────────────────────
_anchor = d["tdate"].fillna(E)
d["rel"] = ((d["date"].dt.year - _anchor.dt.year) * 12 + (d["date"].dt.month - _anchor.dt.month))
if a.ev_window:
    d["rel"] = d["rel"].clip(-a.ev_window, a.ev_window)
REF = a.ref_month
# event-time dummies only exist for treated units, so only their rel values
# have support; control-only rel values would create all-zero columns.
rels = sorted(r for r in d.loc[d["treat"] == 1, "rel"].unique() if r != REF)
for r in rels:
    d[f"ev_{'m' if r < 0 else 'p'}{abs(r)}"] = ((d["rel"] == r) & (d["treat"] == 1)).astype(int)
evcols = [c for c in d.columns if c.startswith("ev_") and d[c].sum() > 0]
fml = "log1y ~ " + " + ".join(evcols) + " + C(unit) + C(month)"
es = smf.ols(fml, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["unit"]})
Xp = pd.get_dummies(d[["unit", "month"]], drop_first=True).astype(float)
for c in evcols: Xp.insert(0, c, d[c].values)
Xp = sm.add_constant(Xp)
try:
    esp = sm.GLM(d["y"].clip(lower=0).values, Xp, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["unit"])[0]})
except Exception as _e:
    print(f"   (PPML event study failed: {_e}; falling back to OLS event study only)")
    esp = es
ev = pd.DataFrame({"rel_month": [(-int(c[4:]) if c[3] == "m" else int(c[4:])) for c in evcols],
                   "ols_log1p_beta": [es.params[c] for c in evcols],
                   "ols_log1p_se": [es.bse[c] for c in evcols],
                   "ppml_beta": [esp.params[c] for c in evcols],
                   "ppml_se": [esp.bse[c] for c in evcols]}).sort_values("rel_month")
ev.to_csv(f"part3_{a.label}_eventstudy.csv", index=False)

def pretrend_check(col):
    pre = ev[(ev.rel_month < 0) & (ev.rel_month <= a.pretrend_end)]; post = ev[ev.rel_month >= 0]
    if len(pre) < 3: return None
    slope, intercept = np.polyfit(pre.rel_month, ev.loc[pre.index, col], 1)
    # joint pre-period test on the chosen window only
    pre_names = [c for c in evcols if c[3] == "m" and -int(c[4:]) <= a.pretrend_end]
    model = es if col.startswith("ols") else esp
    f = model.wald_test(np.eye(len(model.params))[[list(model.params.index).index(n) for n in pre_names]],
                        scalar=True)
    post_avg = post[col].mean()
    extrap = np.mean([slope * r + intercept for r in post.rel_month])
    return dict(pre_slope_per_month=slope, pre_joint_p=float(f.pvalue),
                post_avg_beta=post_avg, post_net_of_trend=post_avg - extrap,
                pre_max_abs=pre[col].abs().max())

pt_ols = pretrend_check("ols_log1p_beta"); pt_ppml = pretrend_check("ppml_beta")

# ── KEY RESULTS ────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"KEY RESULTS  |  label={a.label}  |  panel={a.panel}")
print("=" * 72)
print(f"Assignment: {how}")
print(f"Units: {n_t} treated, {n_c} control | months: {d['month'].nunique()} "
      f"({d['date'].min().date()} to {d['date'].max().date()}) | event: {E.date()}")
print(f"\nA. Near-zero radiance: {100*share_nz:.1f}% of obs < {a.near_zero} nW "
      f"({100*share_nz_t:.1f}% among treated); exact zeros {100*share_0:.1f}%")
print("   -> below ~5%: the +1 shift is cosmetic; above ~20%: it materially biases percentages\n")
print("B. Treated x post, unit + month FE, SE clustered by unit")
print(f"   {'estimator':32s} {'beta':>8s} {'se':>7s} {'p':>7s} {'implied %':>10s} {'95% CI':>18s}")
for _, r in res.iterrows():
    print(f"   {r.estimator:32s} {r.beta:8.3f} {r.se:7.3f} {r.p:7.3f} {r.implied_pct:9.1f}% "
          f"[{r.ci_lo_pct:6.1f}, {r.ci_hi_pct:6.1f}]")
print("   -> if PPML implied % is within ~1/3 of the log(y+1) figure and same sign, the story holds\n")
print(f"C. Pre-trends and trend-extrapolation check (event study, reference month {REF}; "
      f"joint pre-test on months <= {a.pretrend_end})")
for nm, pt in (("OLS log(y+1)", pt_ols), ("PPML", pt_ppml)):
    if pt:
        print(f"   {nm:14s} pre-slope/month={pt['pre_slope_per_month']:+.4f}  joint pre p={pt['pre_joint_p']:.3f}  "
              f"post avg beta={pt['post_avg_beta']:+.3f}  net of extrapolated trend={pt['post_net_of_trend']:+.3f}")
if a.pretrend_end < -1:
    _shock = ev[(ev.rel_month > a.pretrend_end) & (ev.rel_month < 0)]
    print(f"   Shock phase (months {a.pretrend_end+1} to -1, between pre-window and programming):")
    print("   " + "  ".join(f"m{int(r.rel_month):+d}={r.ppml_beta:+.2f}" for _, r in _shock.iterrows()))
    print("   -> strongly negative here = treated units were hit harder (targeting), not a pre-trend violation")
print("   -> joint pre p > 0.10 and 'net of trend' still positive: effect survives the pre-trend concern")
print("   -> joint pre p < 0.05 and 'net of trend' near zero or negative: identification problem is real")
print(f"\nFiles: part3_{a.label}_results.csv, part3_{a.label}_eventstudy.csv")
print("=" * 72)
