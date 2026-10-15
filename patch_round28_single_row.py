#!/usr/bin/env python3
"""Round 28: predicted ROI shown for the selected programme only. Run from the repo root."""
import sys
p = "app.py"; s = open(p, encoding="utf-8").read(); orig = s

# 1. drop the per-project pre-computation (no longer needed)
a = s.find("                # Per-project recovery since each project's OWN start date (scorecard rule).")
b = s.find('                with st.expander("Predicted Economic Reactivation (illustrative)", expanded=True):')
if not (0 < a < b): sys.exit("ABORT: per-project block anchors not found")
s = s[:a] + s[b:]

# 2. replace the all-programmes table with one row for the selected programme
a = s.find("                    # Per-project table: each project measured since its OWN start date.")
end_marker = '"the selected project shows the scorecard\'s own figure.")'
b = s.find(end_marker, a)
if not (0 < a < b): sys.exit("ABORT: table block anchors not found")
b += len(end_marker)
new = '''                    # One row: the selected programme only. Other programmes are not listed,
                    # because the proportional-share model gives identical figures to any
                    # programmes sharing a start date, which reads as an error.
                    if _has_nl and _sel_b > 0:
                        _sel_rows = _post[_post["start_date"] == _intv] if "start_date" in _post.columns else _post.iloc[0:0]
                        _sel_title = ("; ".join(_sel_rows["title"].astype(str).str[:60])
                                      if "title" in _post.columns and len(_sel_rows) else "Selected programme")
                        _row = pd.DataFrame([{
                            "Programme": _sel_title,
                            "Start": str(_intv.date()),
                            "Budget": "${:,.0f}".format(_sel_b),
                            "NL recovery since start": "{:.0f}%".format(_rec),
                            "Share of post-event UNDP budget": "{:.0%}".format(_bshare),
                            "Predicted reactivation": "${:,.0f}".format(_prog_react),
                            "Predicted ROI": "{:,.1f}:1".format(_roi),
                        }])
                        st.dataframe(_row, hide_index=True, width="stretch")
                        st.caption("Shown for the selected programme only. Choose another programme above to see "
                                   "its prediction; recovery is measured from each programme's own start date.")'''
s = s[:a] + new + s[b:]

compile(s, p, "exec")
open(p, "w", encoding="utf-8").write(s)
print(f"patched app.py ({len(orig):,} -> {len(s):,} bytes); compiles OK")
