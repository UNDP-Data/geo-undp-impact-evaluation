#!/usr/bin/env python3
"""Round 27: per-project PREDICTED ROI + honest language. Run from the repo root."""
import re, sys
p = "app.py"; s = open(p, encoding="utf-8").read(); orig = s
def rep(old, new, n=1):
    global s
    c = s.count(old)
    if c != n: sys.exit(f"ABORT: anchor found {c}x (expected {n}): {old[:70]!r}")
    s = s.replace(old, new)

# ── 1. helper: recovery since any start date, scorecard rule, from the panel ──
rep('def render_copilot_sidebar(config, figures_dir, code):',
'''def recovery_since_start(panel, event_dt, start_dt, threshold=0.10):
    """Nightlight recovery since a given project start, following the scorecard
    rule: units with a >10% decline from the pre-event mean to the post-event
    trough are 'affected'; recovery since start is the share of that decline
    regained between the first month at/after the start date and the latest
    observation, averaged over affected units and capped at 0-100. Returns
    dict(since, overall, n_aff, n_tot) or None when no post-event data."""
    p = panel.dropna(subset=["nightlights"])
    pre = p[p["period_start"] < event_dt].groupby("unit_id")["nightlights"].mean()
    post = p[p["period_start"] >= event_dt].sort_values("period_start")
    if post.empty or pre.empty:
        return None
    trough = post.groupby("unit_id")["nightlights"].min()
    latest = post.groupby("unit_id")["nightlights"].last()
    at_start = post[post["period_start"] >= start_dt].groupby("unit_id")["nightlights"].first()
    df = pd.concat([pre.rename("pre"), trough.rename("trough"), latest.rename("latest"),
                    at_start.rename("at_start")], axis=1).dropna(subset=["pre", "trough", "latest"])
    df = df[df["pre"] > 0]
    df["decline"] = (df["pre"] - df["trough"]) / df["pre"]
    aff = df[df["decline"] > threshold]
    out = {"since": 0.0, "overall": 0.0, "n_aff": int(len(aff)), "n_tot": int(len(df))}
    if len(aff) == 0:
        return out
    span = aff["pre"] - aff["trough"]
    out["overall"] = float(((aff["latest"] - aff["trough"]) / span).clip(0, 1).mean() * 100)
    if aff["at_start"].notna().any():
        out["since"] = float(((aff["latest"] - aff["at_start"]) / span).clip(0, 1).mean() * 100)
    out["since"] = min(out["since"], out["overall"])
    return out


PREDICTED_NOTE = (
    '<div style="background:#1f2a1f;border:1px solid #2ea043;border-radius:6px;padding:.7rem 1rem;'
    'margin:.2rem 0 .8rem 0;font-size:.8rem;color:#c9d1d9;line-height:1.6;">'
    '<b>Predicted, not measured.</b> The figures below project the nightlight recovery observed in '
    'affected areas since each UNDP project began onto a standard nightlight-GDP elasticity and the '
    'pre-shock GDP of those areas. They assume that the recovery observed after a project started is '
    'attributable to it in proportion to its share of UNDP spending, an assumption this dashboard does '
    'not test: recovery reflects every actor on the ground, and the scorecard above is correlational. '
    'Read these as planning-grade orders of magnitude, not as evaluated returns.</div>'
)


def render_copilot_sidebar(config, figures_dir, code):''')

# ── 2. the reactivation block: replace from the expander to the end of the table ──
a = s.find('                with st.expander("Indicative Economic Reactivation", expanded=True):')
b_line = '                            st.dataframe(_tdf, hide_index=True, width="stretch")'
b = s.find(b_line, a)
if a < 0 or b < 0: sys.exit("ABORT: reactivation block anchors not found")
b += len(b_line)
new_block = r'''                # Per-project recovery since each project's OWN start date (scorecard rule).
                _per = {}
                try:
                    _pp = pd.read_csv(resolve_dir("outputs", code) / "panel_data.csv",
                                      usecols=["unit_id", "period_start", "nightlights"],
                                      parse_dates=["period_start"])
                    for _sd in _post["start_date"].dropna().unique():
                        _r = recovery_since_start(_pp, _evt, pd.Timestamp(_sd))
                        if _r: _per[str(pd.Timestamp(_sd).date())] = _r
                except Exception:
                    _per = {}

                with st.expander("Predicted Economic Reactivation (illustrative)", expanded=True):
                    st.markdown(PREDICTED_NOTE, unsafe_allow_html=True)
                    if not _has_nl:
                        st.info('Predictions require nightlight recovery data.')
                    elif _sel_b <= 0:
                        st.info('Predictions require a UNDP project with budget data. Select a project above.')
                    if _has_nl and _sel_b > 0:
                        def _fmt_big(v):
                            if v >= 1e9: return "${:.1f}B".format(v/1e9)
                            if v >= 1e6: return "${:.1f}M".format(v/1e6)
                            if v >= 1e3: return "${:.0f}K".format(v/1e3)
                            return "${:,.0f}".format(v)
                        _cards = [
                            ("Pre-shock GDP", _fmt_big(_nat_gdp), str(_wb_year)),
                            ("Project budget", _fmt_big(_sel_b), ""),
                            ("NL recovery", "{:.0f}%".format(_rec), _rlbl),
                            ("Predicted reactivation", _fmt_big(_prog_react), "selected project"),
                            ("Predicted ROI", "{:,.1f}:1".format(_roi), "illustrative"),
                        ]
                        _cards_html = '<div style="display:flex;justify-content:center;gap:1.5rem;margin:.8rem 0;">'
                        for _lbl, _val, _sub in _cards:
                            _cards_html += (
                                '<div style="text-align:center;min-width:120px;">'
                                + '<div style="font-size:.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;">'
                                + _lbl + '</div>'
                                + '<div style="font-size:1.4rem;font-weight:700;color:#e6edf3;">'
                                + _val + '</div>'
                                + ('<div style="font-size:.65rem;color:#8b949e;">' + _sub + '</div>' if _sub else '')
                                + '</div>'
                            )
                        _cards_html += '</div>'
                        st.markdown(_cards_html, unsafe_allow_html=True)

                        _expl = (
                            '<div style="font-size:.85rem;color:#c9d1d9;line-height:1.7;margin:.8rem 0;">'
                            '<p style="margin:0 0 .6rem 0;"><b>Step 1 - Observed nightlight recovery:</b> '
                            + '{} nightlight recovery of <b>{:.0f}%</b>'.format(_rlbl.capitalize(), _rec)
                            + ' was observed across <b>{:,}</b> affected administrative units.</p>'.format(_n_aff)
                            + '<p style="margin:0 0 .6rem 0;"><b>Step 2 - Pre-shock GDP:</b> '
                            + 'National GDP before the shock was <b>${:,.0f}</b> ({}, {}). '.format(_nat_gdp, _wb_src, _wb_year)
                            + 'The affected area ({:,.0f} of {:,.0f} admin units, <b>{:.0%}</b> of the country) '.format(_n_aff, _n_tot, _cov)
                            + 'is assigned a proportional GDP of <b>${:,.0f}</b>.</p>'.format(_aff_gdp)
                            + '<p style="margin:0 0 .6rem 0;"><b>Step 3 - Predicted reactivation:</b> '
                            + 'Applying the Henderson et al. (2012) elasticity of 0.3, {:.0f}% nightlight recovery '.format(_rec)
                            + 'predicts <b>{:.1f}% GDP reactivation</b>, about <b>${:,.0f}</b> '.format(_gdp_eff, _react)
                            + 'in economic activity across all actors on the ground.</p>'
                            + '<p style="margin:0 0 .6rem 0;"><b>Step 4 - Share attributed to this project:</b> '
                            + 'This project represents <b>{:.1%}</b> of post-event UNDP spending '.format(_bshare)
                            + '(${:,.0f} of ${:,.0f}). '.format(_sel_b, _tot_b)
                            + 'Attributing that share yields a predicted <b>${:,.0f}</b>, '.format(_prog_react)
                            + 'a predicted return of <b>{:,.1f}:1</b> on the project budget.</p></div>'.format(_roi)
                        )
                        st.markdown(_expl, unsafe_allow_html=True)
                        st.caption(
                            "Prediction for rapid screening, not an evaluated return. Recovery since a project's "
                            "start reflects all actors on the ground, not just UNDP; the proportional share is an "
                            "assumption. GDP data: " + _wb_src + " (" + str(_wb_year) + "). Affected-area GDP is a "
                            "proportional share of national GDP by number of affected admin units. The elasticity "
                            "of 0.3 is a global average that may differ locally."
                        )
                    elif _has_nl:
                        st.metric("NL recovery", "{:.0f}%".format(_rec))
                        st.markdown("Recovery detected but no project budget available for a prediction.")
                    else:
                        st.info("Nightlights show no measurable recovery yet. Predictions appear once recovery is detected.")

                    # Per-project table: each project measured since its OWN start date.
                    if "total_budget" in _post.columns and len(_post.dropna(subset=["total_budget"])) > 0:
                        st.markdown('<div style="font-size:.85rem;font-weight:600;color:#e6edf3;margin:.6rem 0 .2rem 0;">'
                                    'By project (recovery measured since each project started)</div>',
                                    unsafe_allow_html=True)
                        _rows = []
                        for _, _pr in _post.dropna(subset=["total_budget"]).iterrows():
                            _b = float(_pr["total_budget"])
                            if pd.isna(_b) or _b <= 0: continue
                            _t = str(_pr.get("title", ""))[:55]
                            _d = _pr["start_date"].strftime("%Y-%m-%d") if pd.notna(_pr["start_date"]) else "-"
                            _bs = _b / _tot_b if _tot_b > 0 else 0
                            _is_sel = (_d == str(_intv.date()))
                            if _is_sel and _nl_un > 0:
                                _rj = _nl_un                                   # scorecard figure, same as the cards
                            else:
                                _rj = float(_per.get(_d, {}).get("since", 0.0))
                            _pr_r = _aff_gdp * (_rj * _EL / 100.0) * _bs if _has_nl else 0.0
                            _pr_roi = _pr_r / _b if (_b > 0 and _pr_r > 0) else 0.0
                            _rows.append({"Project": ("▶ " if _is_sel else "") + _t, "Start": _d, "Budget": _b,
                                          "NL recovery since start": _rj, "Share of UNDP budget": _bs,
                                          "Predicted reactivation": _pr_r, "Predicted ROI": _pr_roi})
                        if _rows:
                            _tdf = pd.DataFrame(_rows)
                            _tdf["Budget"] = _tdf["Budget"].apply(lambda x: "${:,.0f}".format(x))
                            _tdf["NL recovery since start"] = _tdf["NL recovery since start"].apply(lambda x: "{:.0f}%".format(x) if x > 0 else "none yet")
                            _tdf["Share of UNDP budget"] = _tdf["Share of UNDP budget"].apply(lambda x: "{:.0%}".format(x))
                            _tdf["Predicted reactivation"] = _tdf["Predicted reactivation"].apply(lambda x: "${:,.0f}".format(x) if x > 0 else chr(8212))
                            _tdf["Predicted ROI"] = _tdf["Predicted ROI"].apply(lambda x: "{:,.1f}:1".format(x) if x > 0 else chr(8212))
                            st.dataframe(_tdf, hide_index=True, width="stretch")
                            st.caption("▶ marks the project selected above. Recovery since start is computed per project "
                                       "from the panel with the scorecard rule (affected units only, capped at 100%); "
                                       "the selected project shows the scorecard's own figure.")'''
s = s[:a] + new_block + s[b:]

# ── 3. language elsewhere ──
rep('"🔒 This is a protected country with verified results.',
    '"🔒 This is a protected country with published results.')
rep('contains verified causal impact-evaluation results and is ',
    'contains published impact-evaluation results and is ')
rep('''        + "8. Use plain language a policy maker would understand. Avoid academic hedging."''',
'''        + "8. Use plain language a policy maker would understand. Avoid academic hedging." + chr(10)
        + "9. Economic reactivation and ROI figures are PREDICTIONS: observed recovery since a project started, "
          "projected with a fixed elasticity and a proportional-share assumption. Always call them predicted, never measured."''')

# ── 4. About page: economic model + what Myanmar and Gaza actually show ──
a = s.find("### Economic Reactivation Model"); b = s.find("- GDP proportionally distributed across admin units (simplification)")
if a < 0 or b < 0: sys.exit("ABORT: About economic section not found")
b += len("- GDP proportionally distributed across admin units (simplification)")
s = s[:a] + """### Predicted Economic Reactivation (illustrative)

Converts nightlight recovery into a predicted USD figure in 4 steps:

1. **Observed nightlight recovery** since each UNDP project's own start date, computed from the panel with the scorecard rule (affected units only, capped at 100%)
2. **Pre-shock GDP** from World Bank WDI for the year before the event
3. **Affected-area GDP** as the national GDP times the share of affected admin units
4. **Predicted reactivation** as affected GDP times recovery times the Henderson, Storeygard & Weil (2012) elasticity of 0.3, with each project assigned a share in proportion to its budget

The elasticity means 10% nightlight recovery corresponds to roughly 3% GDP; it is a global average and local values differ. The predicted ROI divides each project's assigned share by its budget.

**These are predictions, not measured returns.** They assume the recovery observed after a project started is attributable to it in proportion to its spending, which the dashboard does not test. Recovery reflects every actor on the ground; budgets are allocations, not disbursements; and GDP is distributed proportionally across admin units. Measured effects require the comparison designs described below.""" + s[b:]

a = s.find("### Causal Impact Analysis"); b = s.find("Results for Myanmar and Gaza are available in the Impact Measurement tab.")
if a < 0 or b < 0: sys.exit("ABORT: About causal section not found")
b += len("Results for Myanmar and Gaza are available in the Impact Measurement tab.")
s = s[:a] + """### Impact Evaluation (Myanmar and Gaza)

Where geocoded UNDP project locations are available, the pipeline estimates effects with formal comparison designs: difference-in-differences with unit and month fixed effects, Poisson (PPML) estimation on raw radiance, staggered-adoption event studies with parallel-trend tests, and comparison-distance robustness checks.

- **Myanmar (2025 earthquake).** Townships exposed to UNDP early-recovery programming recovered nighttime luminosity about 19 to 23 percent faster than comparison townships, an estimate that holds across three estimators and passes the reported diagnostics. This is the dashboard's one evaluated effect.
- **Gaza (conflict).** Nighttime lights cannot identify a programme effect: electricity supply is constrained across the territory and comparison locations are not on parallel paths. Radar backscatter shows a small, localized surface change within about 500 m of intervention sites, consistent with physical works, and no economic effect is claimed. The Gaza evidence is descriptive.

Results and diagnostics are in the Impact Measurement tab and in the manuscript "From Space to Impact" (replication package in the repository under `paper/`).""" + s[b:]

# ── 5. latent bug: `out` undefined in run_cmd error branch ──
rep('''                    tail = chr(10).join(log[-15:])
                    if "Pipeline finished" in out or "✓ COMPLETE" in out:''',
'''                    tail = chr(10).join(log[-15:])
                    out = chr(10).join(log)
                    if "Pipeline finished" in out or "✓ COMPLETE" in out:''')

compile(s, p, "exec")
open(p, "w", encoding="utf-8").write(s)
print(f"patched app.py ({len(orig):,} -> {len(s):,} bytes); compiles OK")
