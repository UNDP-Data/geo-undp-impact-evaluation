#!/usr/bin/env python3
"""Extended Data Table 2 and the valuation figures quoted in the text (Methods, Economic valuation).

For a proportional radiance change r and an elasticity of output with respect to light eps:
    output change      = (1 + r) ** eps - 1
    value over horizon = output change x population x GDP per capita x horizon / 12
    output / cost      = value / programme cost
The point estimate uses the PPML contrast r; the coefficient-only range transforms the endpoints of its
95% confidence interval and holds every other input fixed (it is not a confidence interval for returns).

Run from the repository root, as paper/reproduce.sh does:  python3 paper/code/valuation.py
Writes paper/outputs/tables/valuation_EDtable2.csv and prints the in-text figures.
"""
import csv
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
TABLES = PAPER / "outputs" / "tables"
LOGS = PAPER / "outputs" / "logs"

# ---------------------------------------------------------------- inputs and their sources
# r: PPML contrast for the baseline 3-8 km band and its 95% interval (implied_pct, ci_lo_pct, ci_hi_pct).
#    Source: paper/outputs/tables/part3_myanmar_3_8km_results.csv, row "PPML raw y", written by
#    paper/code/part3_diagnostics.py (Table 1).
RESULTS = TABLES / "part3_myanmar_3_8km_results.csv"

# Population: residents of the 14 treated townships of the baseline 3-8 km band.
#    Source: WorldPop 2020 UN-adjusted 100 m constrained raster (mmr_ppp_2020_UNadj_constrained.tif), zonal sum
#    over the treated townships; analysis/mmr/05_population_worldpop.py, log paper/outputs/logs/worldpop_myanmar.txt.
POPULATION = 3_005_825
TREATED_TOWNSHIPS = 14

# GDP per capita, current US$.
#    Source: World Bank, World Development Indicators, NY.GDP.PCAP.CD, Myanmar, 2023.
GDP_PER_CAPITA = 1233.20

# Programme cost: sum of the total budgets of the four UNDP Myanmar early-recovery projects in the project
#    register (the debris-clearance project records no budget).
COST = 16_783_198

# Horizon: five months (scenario assumption stated in Methods).
HORIZON_MONTHS = 5

# Elasticity of output with respect to light: 0.3 central, 0.15 and 0.5 scenarios, informed by national
#    annual evidence (Henderson, Storeygard and Weil 2012, reference 8).
ELASTICITIES = (0.15, 0.3, 0.5)


def ppml_contrast(path=RESULTS):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["estimator"].startswith("PPML raw y"):
                return tuple(float(row[k]) / 100 for k in ("implied_pct", "ci_lo_pct", "ci_hi_pct"))
    sys.exit(f"PPML row not found in {path}")


def check_population(log=LOGS / "worldpop_myanmar.txt"):
    """Confirm POPULATION against the WorldPop log when it is present."""
    if not log.exists():
        print(f"note: {log} not found; population not re-checked")
        return
    m = re.search(r"3km-8km:\s+(\d+) treated townships, pop ([\d,]+)", log.read_text())
    if not m:
        sys.exit(f"3-8 km line not found in {log}")
    n, pop = int(m.group(1)), int(m.group(2).replace(",", ""))
    if (n, pop) != (TREATED_TOWNSHIPS, POPULATION):
        sys.exit(f"population mismatch: log {n} townships, {pop:,}; script {TREATED_TOWNSHIPS}, {POPULATION:,}")
    print(f"population check: {n} treated townships, {pop:,} (matches {log.name})")


def value(r, eps):
    change = (1 + r) ** eps - 1
    usd = change * POPULATION * GDP_PER_CAPITA * HORIZON_MONTHS / 12
    return change, usd, usd / COST


def main():
    check_population()
    r, lo, hi = ppml_contrast()
    rows = []
    for eps in ELASTICITIES:
        c, v, q = value(r, eps)
        _, vlo, qlo = value(lo, eps)
        _, vhi, qhi = value(hi, eps)
        rows.append({
            "elasticity": eps,
            "output_change_pct": f"{100 * c:.2f}",
            "output_usd_m": f"{v / 1e6:.1f}",
            "output_range_usd_m": f"{vlo / 1e6:.1f}\u2013{vhi / 1e6:.1f}",
            "output_cost_ratio": f"{q:.2f}",
            "ratio_range": f"{qlo:.2f}\u2013{qhi:.2f}",
        })
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "valuation_EDtable2.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"inputs: r = {r:.4f} (95% CI {lo:.4f} to {hi:.4f}); population {POPULATION:,}; "
          f"GDP per capita US${GDP_PER_CAPITA:,.2f}; cost US${COST:,}; horizon {HORIZON_MONTHS} months")
    print("\nExtended Data Table 2")
    for row in rows:
        print("  " + " | ".join(str(v) for v in row.values()))
    c, v, q = value(r, 0.3)
    _, vlo, qlo = value(lo, 0.3)
    _, vhi, qhi = value(hi, 0.3)
    print("\nText (elasticity 0.3):")
    print(f"  a {100 * r:.1f} percent radiance change maps to a {100 * c:.2f} percent output change, approximately "
          f"US${v / 1e6:.1f} million over five months, or {q:.1f} times the stated portfolio denominator.")
    print(f"  Transforming the reported radiance interval of {100 * lo:.1f}\u2013{100 * hi:.1f} percent gives "
          f"US${vlo / 1e6:.1f}\u2013{vhi / 1e6:.1f} million, or {qlo:.1f}\u2013{qhi:.1f} times the denominator.")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
