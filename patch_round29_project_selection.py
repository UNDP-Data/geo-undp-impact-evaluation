#!/usr/bin/env python3
"""Round 29: predicted ROI fixes. Run from the repo root.
1. Programme picker selects a PROJECT (by project_id), not a start date, so
   projects sharing a start date are no longer merged.
2. Predictions only for eligible projects: budget > 0, classified post-disaster
   in the register (is_post_disaster), not an umbrella (is_likely_umbrella);
   the budget-share denominator uses the same eligible set (no double counting).
3. Recovery = since the selected project's start only (no fallback to overall).
4. No dollar figures when recovery since start is below 10%.
5. Errors in this section are shown instead of silently hiding it."""
import sys
p = "app.py"; s = open(p, encoding="utf-8").read(); orig = s
def rep(old, new):
    global s
    c = s.count(old)
    if c != 1: sys.exit(f"ABORT: anchor found {c}x (expected 1): {old[:80]!r}")
    s = s.replace(old, new)

# ── 1. picker: carry project_id; match current selection by id, then date ──
rep('''                    _opts = []
                    for _, _r in _post_u.iterrows():
                        _d = _r['start_date'].strftime('%Y-%m-%d')
                        _t = str(_r.get('title',''))[:60]
                        _b = "${:,.0f}".format(_r['total_budget']) if pd.notna(_r.get('total_budget')) else ""
                        _opts.append({"label": _d + " - " + _t + " " + _b, "date": _d})
                    _cur = config['events'].get('intervention_start', event_date)
                    _cidx = 0
                    for _i, _o in enumerate(_opts):
                        if _o['date'] == _cur: _cidx = _i; break''',
'''                    _opts = []
                    for _i, (_, _r) in enumerate(_post_u.iterrows()):
                        _d = _r['start_date'].strftime('%Y-%m-%d')
                        _t = str(_r.get('title',''))[:60]
                        _b = "${:,.0f}".format(_r['total_budget']) if pd.notna(_r.get('total_budget')) else ""
                        _pid = str(_r.get('project_id', '')) if pd.notna(_r.get('project_id', None)) else "row" + str(_i)
                        _opts.append({"label": _d + " - " + _t + " " + _b, "date": _d, "pid": _pid})
                    _cur = config['events'].get('intervention_start', event_date)
                    _cur_pid = str(st.session_state.get('sel_pid_' + code)
                                   or config['events'].get('selected_project_id', '') or '')
                    _cidx = next((i for i, o in enumerate(_opts)
                                  if o['pid'] == _cur_pid and o['date'] == str(_cur)), None)
                    if _cidx is None:
                        _cidx = next((i for i, o in enumerate(_opts) if o['date'] == str(_cur)), 0)''')

rep('''                    _nint = _opts[_pick]['date']
                    if _nint != _cur:''',
'''                    _nint = _opts[_pick]['date']
                    _npid = _opts[_pick]['pid']
                    if _nint == str(_cur):
                        st.session_state['sel_pid_' + code] = _npid
                    if _nint != str(_cur):''')

rep("""                            config['events']['intervention_start'] = _nint""",
"""                            config['events']['intervention_start'] = _nint
                            config['events']['selected_project_id'] = _npid
                            st.session_state['sel_pid_' + code] = _npid""")

# ── 2. eligible set + the selected project row ──
rep('''                _post = _undp[_undp["start_date"] >= _evt].copy()
                if "total_budget" not in _post.columns and "budget" in _post.columns:
                    _post["total_budget"] = _post["budget"]''',
'''                _post = _undp[_undp["start_date"] >= _evt].copy()
                if "total_budget" not in _post.columns and "budget" in _post.columns:
                    _post["total_budget"] = _post["budget"]

                def _truthy(_s):
                    return _s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes", "y"])
                _elig_note = ""
                if "total_budget" in _post.columns:
                    _elig = _post[pd.to_numeric(_post["total_budget"], errors="coerce").fillna(0) > 0].copy()
                else:
                    _elig = _post.iloc[0:0].copy()
                if "is_likely_umbrella" in _elig.columns:
                    _elig = _elig[~_truthy(_elig["is_likely_umbrella"])]
                if "is_post_disaster" in _elig.columns:
                    _pdz = _elig[_truthy(_elig["is_post_disaster"])]
                    if len(_pdz):
                        _elig = _pdz
                    else:
                        _elig_note = ("No project in this country's register is classified as a post-disaster "
                                      "intervention, so all post-event projects with a budget are included.")
                _pid_sel = str(st.session_state.get("sel_pid_" + code)
                               or config["events"].get("selected_project_id", "") or "")
                _intv_s = str(_intv.date())
                _sel_row = None
                if _pid_sel and "project_id" in _post.columns:
                    _m = _post[(_post["project_id"].astype(str) == _pid_sel)
                               & (_post["start_date"].dt.date.astype(str) == _intv_s)]
                    if len(_m): _sel_row = _m.iloc[0]
                if _sel_row is None:
                    _m = _post[_post["start_date"].dt.date.astype(str) == _intv_s]
                    if len(_m): _sel_row = _m.iloc[0]
                _sel_eligible = _sel_row is not None and _sel_row.name in _elig.index''')

# ── 3. recovery since the selected project's start only ──
rep('''                if _nl_un > 0:
                    _rec = _nl_un; _rlbl = "since selected project"
                elif _nl_ov > 0:
                    _rec = _nl_ov; _rlbl = "overall"
                else:
                    _rec = 0; _rlbl = "N/A"''',
'''                _rec = _nl_un; _rlbl = "since selected project"''')

# ── 4. budgets: selected project's own budget; share among eligible projects ──
rep('''                _sel_b, _tot_b = 0.0, 0.0
                if "total_budget" in _post.columns:
                    _tot_b = _post["total_budget"].dropna().sum()
                    _sm = _post[_post["start_date"].dt.date.astype(str) == str(_intv.date())]
                    if not _sm.empty and pd.notna(_sm.iloc[0]["total_budget"]):
                        _sel_b = float(_sm.iloc[0]["total_budget"])
                    elif not _post.dropna(subset=["total_budget"]).empty:
                        _sel_b = float(_post.dropna(subset=["total_budget"]).iloc[0]["total_budget"])''',
'''                _sel_b, _tot_b = 0.0, 0.0
                if "total_budget" in _post.columns:
                    _tot_b = float(pd.to_numeric(_elig["total_budget"], errors="coerce").fillna(0).sum())
                    if _sel_row is not None and pd.notna(_sel_row.get("total_budget")):
                        _sel_b = float(_sel_row["total_budget"])''')

# ── 5. gates: eligibility and minimum signal before any dollar figure ──
rep('''                    if not _has_nl:
                        st.info('Predictions require nightlight recovery data.')
                    elif _sel_b <= 0:
                        st.info('Predictions require a UNDP project with budget data. Select a project above.')
                    if _has_nl and _sel_b > 0:''',
'''                    _MIN_REC = 10.0
                    _show = False
                    if not _has_nl:
                        st.info('Predictions require nightlight recovery data.')
                    elif _sel_b <= 0:
                        st.info('Predictions require a UNDP project with budget data. Select a project above.')
                    elif not _sel_eligible:
                        st.info("This project is not classified as an early-recovery or post-disaster intervention "
                                "in the UNDP project register (or is an umbrella programme whose components are "
                                "listed separately), so no economic prediction is shown for it.")
                    elif _rec < _MIN_REC:
                        st.info("Nightlight recovery since this project started is {:.0f}%, too small to support a "
                                "meaningful prediction. Predictions are shown once recovery since the project's "
                                "start reaches {:.0f}%.".format(_rec, _MIN_REC))
                    else:
                        _show = True
                    if _show:''')

rep('''                    elif _has_nl:
                        st.metric("NL recovery", "{:.0f}%".format(_rec))
                        st.markdown("Recovery detected but no project budget available for a prediction.")
                    else:
                        st.info("Nightlights show no measurable recovery yet. Predictions appear once recovery is detected.")
''', '')

# ── 6. single row = the selected project itself ──
rep('''                    if _has_nl and _sel_b > 0:
                        _sel_rows = _post[_post["start_date"] == _intv] if "start_date" in _post.columns else _post.iloc[0:0]
                        _sel_title = ("; ".join(_sel_rows["title"].astype(str).str[:60])
                                      if "title" in _post.columns and len(_sel_rows) else "Selected programme")''',
'''                    if _show:
                        _sel_title = str(_sel_row.get("title", "Selected programme"))[:90]''')
rep('"Share of post-event UNDP budget"', '"Share of post-event recovery budget"')
rep("'of post-event UNDP spending '", "'of post-event UNDP early-recovery spending '")
rep('''"its prediction; recovery is measured from each programme's own start date.")''',
'''"its prediction; recovery is measured from each programme's own start date.")
                        if _elig_note:
                            st.caption(_elig_note)''')

# ── 7. make failures visible ──
rep('''            except Exception:
                pass

            # DATA SOURCES & AVAILABILITY''',
'''            except Exception as _re_err:
                st.caption("Predicted reactivation unavailable: " + str(_re_err)[:150])

            # DATA SOURCES & AVAILABILITY''')

# ── 8. About page ──
rep("The predicted ROI divides each project's assigned share by its budget.",
    "The predicted ROI divides each project's assigned share by its budget. Only projects classified in the "
    "UNDP register as post-disaster interventions are included, umbrella programmes are excluded to avoid "
    "double counting, and no prediction is shown when recovery since a project's start is below 10%.")

compile(s, p, "exec")
open(p, "w", encoding="utf-8").write(s)
print(f"patched app.py ({len(orig):,} -> {len(s):,} bytes); compiles OK")
