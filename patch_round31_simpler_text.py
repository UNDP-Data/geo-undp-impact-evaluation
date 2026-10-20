#!/usr/bin/env python3
"""Round 31: simpler, shorter text for the predicted ROI section. Run from the repo root.
- one short note at the top (upper bound, not a measured return)
- the four steps rewritten in plain language and moved BELOW the project table
- the repeated caption under the steps replaced by one short source line"""
import sys
p = "app.py"; s = open(p, encoding="utf-8").read(); orig = s

def cut_lines(start_marker, end_marker, new, keep_end_line=False):
    global s
    if s.count(start_marker) != 1: sys.exit(f"ABORT: {s.count(start_marker)}x start marker {start_marker!r}")
    a = s.rfind("\n", 0, s.find(start_marker)) + 1
    b = s.find(end_marker, a)
    if b < 0: sys.exit(f"ABORT: end marker not found {end_marker!r}")
    b = s.rfind("\n", 0, b) + 1 if keep_end_line else s.find("\n", b) + 1
    s = s[:a] + new + s[b:]

def rep(old, new):
    global s
    if s.count(old) != 1: sys.exit(f"ABORT: {s.count(old)}x anchor {old[:70]!r}")
    s = s.replace(old, new)

# 1. the note at the top: short and plain
cut_lines("PREDICTED_NOTE = (", "not as evaluated returns.</div>'",
'''PREDICTED_NOTE = (
    '<div style="background:#1f2a1f;border:1px solid #2ea043;border-radius:6px;padding:.7rem 1rem;'
    'margin:.2rem 0 .8rem 0;font-size:.85rem;color:#c9d1d9;line-height:1.6;">'
    '<b>Upper-bound prediction, not a measured return.</b> It credits UNDP projects with all the recovery '
    'seen since they started, in proportion to their budgets. Many other actors contribute, so the actual '
    'return is likely to be much lower.</div>'
''')

# 2. remove the long steps + caption from above the table
cut_lines("                        _expl = (", "# One row: the selected programme only.", "", keep_end_line=True)

# 3. below the table: short caption, then the steps in plain language, then one source line
cut_lines('st.caption("Shown for the selected programme only.', "st.caption(_elig_note)",
'''                        st.caption("Select another project above to see its prediction.")
                        if _elig_note:
                            st.caption(_elig_note)
                        _src = "World Bank WDI" if str(_wb_src).startswith("World Bank") else "built-in World Bank estimate"
                        _steps = (
                            '<div style="font-size:.85rem;color:#c9d1d9;line-height:1.7;margin:.9rem 0 .3rem 0;">'
                            '<b>How this figure is calculated</b>'
                            '<ol style="margin:.3rem 0 0 1.1rem;padding:0;">'
                            "<li><b>Recovery.</b> Nightlights in the {:,} affected areas regained {:.0f}% of their loss "
                            "since this project started.</li>"
                            "<li><b>Economy at stake.</b> These areas account for about {} of GDP "
                            "({:.0%} of admin units; national GDP {}, {}).</li>"
                            "<li><b>Economic reactivation.</b> With the standard elasticity of 0.3 (Henderson et al., 2012), "
                            "{:.0f}% more light corresponds to {:.1f}% more GDP, about {} across all actors.</li>"
                            "<li><b>This project's share.</b> It is {:.0%} of UNDP's post-event budget ({} of {}), "
                            "so it is credited with {}, a {:,.1f}:1 return on its budget.</li>"
                            "</ol></div>"
                        ).format(_n_aff, _rec, _fmt_big(_aff_gdp), _cov, _fmt_big(_nat_gdp), _wb_year,
                                 _rec, _gdp_eff, _fmt_big(_react),
                                 _bshare, _fmt_big(_sel_b), _fmt_big(_tot_b), _fmt_big(_prog_react), _roi)
                        st.markdown(_steps, unsafe_allow_html=True)
                        st.caption("GDP: " + _src + " (" + str(_wb_year) + "). Affected-area GDP is split evenly "
                                   "across admin units; the elasticity is a global average and may differ locally.")
''')

# 4. card subtitle matches the note
rep('("Predicted ROI", "{:,.1f}:1".format(_roi), "illustrative")',
    '("Predicted ROI", "{:,.1f}:1".format(_roi), "upper bound")')

if "_expl" in s: sys.exit("ABORT: old steps block still present")
compile(s, p, "exec")
open(p, "w", encoding="utf-8").write(s)
print(f"patched app.py ({len(orig):,} -> {len(s):,} bytes); compiles OK")
