#!/usr/bin/env python3
"""Round 37: country PDF report. Run from the repo root, after copying
report_pdf.py into pipeline/. Adds a 'Prepare PDF report' button above the
scorecard and a download button once the report is built; keeps the last
predicted-ROI figures so the report can include them."""
import sys
p = "app.py"; s = open(p, encoding="utf-8").read(); orig = s
def rep(old, new):
    global s
    c = s.count(old)
    if c != 1: sys.exit(f"ABORT: anchor found {c}x (expected 1): {old[:80]!r}")
    s = s.replace(old, new)

# 1. report buttons, right-aligned above the scorecard
rep('''    try:
        has_scorecard = render_scorecard()''',
'''    _pdf_key = "pdf_" + code
    _pdf_cols = st.columns([3, 1.1, 1.1])
    with _pdf_cols[1]:
        if st.button("📄 Prepare PDF report", key="prep_pdf_" + code, width='stretch'):
            try:
                from pipeline.report_pdf import build_report
                with st.spinner("Building the report..."):
                    st.session_state[_pdf_key] = build_report(
                        code, config, resolve_dir("outputs", code), resolve_dir("data", code),
                        roi=st.session_state.get("roi_summary_" + code),
                        username=st.session_state.get("username", ""))
            except Exception as _pe:
                st.error("Report could not be built: " + str(_pe)[:150])
    with _pdf_cols[2]:
        if st.session_state.get(_pdf_key):
            st.download_button("⬇️ Download report", data=st.session_state[_pdf_key],
                               file_name=code + "_early_recovery_report.pdf", mime="application/pdf",
                               key="dl_pdf_" + code, width='stretch', type="primary")
    try:
        has_scorecard = render_scorecard()''')

# 2. keep the predicted-ROI figures for the report
rep('''                        st.markdown(_cards_html, unsafe_allow_html=True)''',
'''                        st.markdown(_cards_html, unsafe_allow_html=True)
                        st.session_state["roi_summary_" + code] = {
                            "title": str(_sel_row.get("title", "")) if _sel_row is not None else "",
                            "start": str(_intv.date()), "budget": _sel_b, "rec": _rec, "n_aff": _n_aff,
                            "cov": _cov, "nat_gdp": _nat_gdp, "gdp_year": _wb_year, "aff_gdp": _aff_gdp,
                            "gdp_eff": _gdp_eff, "react_all": _react, "share": _bshare, "tot_budget": _tot_b,
                            "react_project": _prog_react, "roi": _roi}''')

compile(s, p, "exec")
open(p, "w", encoding="utf-8").write(s)
print(f"patched app.py ({len(orig):,} -> {len(s):,} bytes); compiles OK")
