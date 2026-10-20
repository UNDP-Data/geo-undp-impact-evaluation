#!/usr/bin/env python3
"""Round 30: remove the title-based relevance rule introduced in round 29.
Every post-event project with a budget is eligible; the only exclusions are
data rules (umbrella flag in the register, or a UNDP-provided roi_eligible
column). Run from the repo root."""
import sys
p = "app.py"; s = open(p, encoding="utf-8").read(); orig = s

def swap_between(start_marker, end_marker, new, include_end_line=True):
    """Replace from the line containing start_marker to the end of the line
    containing end_marker (or up to the start of that line)."""
    global s
    if s.count(start_marker) != 1: sys.exit(f"ABORT: start marker found {s.count(start_marker)}x: {start_marker!r}")
    a = s.rfind("\n", 0, s.find(start_marker)) + 1
    b = s.find(end_marker, a)
    if b < 0: sys.exit(f"ABORT: end marker not found: {end_marker!r}")
    if include_end_line:
        b = s.find("\n", b) + 1
    else:
        b = s.rfind("\n", 0, b) + 1
    s = s[:a] + new + s[b:]

def rep(old, new):
    global s
    c = s.count(old)
    if c != 1: sys.exit(f"ABORT: anchor found {c}x (expected 1): {old[:80]!r}")
    s = s.replace(old, new)

# 1. eligibility: data rules only
swap_between("# Relevance: 'is_post_disaster'", "case=False, regex=True)]",
'''                # Eligibility is a data rule only. Every post-event project with a budget counts:
                # any project spending (staff, procurement, local purchases) can reach the local
                # economy, and the dashboard does not judge how a project works. A UNDP-provided
                # 'roi_eligible' column, if present in the register, overrides this.
                if "roi_eligible" in _elig.columns:
                    _elig = _elig[_truthy(_elig["roi_eligible"])]
''')

# 2. the message: only data-rule exclusions remain
swap_between("elif not _sel_eligible:", "elif _rec < _MIN_REC:",
'''                    elif not _sel_eligible:
                        st.info("This project is excluded from predictions in the UNDP project register "
                                "(an umbrella programme whose components are listed separately, or marked "
                                "as not eligible), so no prediction is shown for it.")
''', include_end_line=False)

# 3. labels back to all post-event spending
rep('"Share of post-event recovery budget"', '"Share of post-event UNDP budget"')
rep("of post-event UNDP early-recovery spending ", "of post-event UNDP spending ")

# 4. About page sentence
a = s.find("Only projects classified in the UNDP register as recovery-relevant")
b = s.find("is below 10%.", a)
if a < 0 or b < 0: sys.exit("ABORT: About sentence not found")
b += len("is below 10%.")
s = s[:a] + ("Every post-event project with a budget is included, because any project spending (staff, "
             "procurement, local purchases) can reach the local economy. Umbrella programmes flagged in the "
             "register are excluded to avoid double counting, and no prediction is shown when recovery since "
             "a project's start is below 10%, where the satellite signal is too weak to convert into dollars.") + s[b:]

if "_NON_RECOVERY" in s or "no direct link to post-shock" in s:
    sys.exit("ABORT: remnants of the relevance rule are still present")
compile(s, p, "exec")
open(p, "w", encoding="utf-8").write(s)
print(f"patched app.py ({len(orig):,} -> {len(s):,} bytes); compiles OK")
