# Early Recovery — Economic Reactivation & ROI Evidence

Single source of truth for the recovery/ROI figures in the policy key-messages.

## Files
- `key_messages.md` — final key-messages text (the deliverable).
- `key_messages_slides.qmd` — Quarto reveal.js deck displaying the key messages.
- `CALCULATIONS.md` — how every number was computed, linked to source files + scripts.
- `roi_attributed.py` — Myanmar & Gaza ROI (reproduces headline numbers) -> `roi_attributed.json`.
- `roi_calculation.py` — full six-country table -> `roi_results.json`.

## Reproduce the numbers
```
python3 roi_attributed.py     # Myanmar 21-39:1 ; Gaza $0.17/$1/yr
python3 roi_calculation.py    # full country table
```

## Render the deck
```
quarto render key_messages_slides.qmd --to revealjs
# optional PDF:  decktape reveal key_messages_slides.html key_messages_slides.pdf
```

## Honest-use rules
1. **Myanmar** is the only causal claim (30–55% more recovery, p<0.001). The dollar ROI
   (20–40:1) depends on the 3.5M beneficiary count, which is NOT in the repo — confirm it.
2. **Gaza**: employment/cash-for-work is the only sector with a significant positive
   economic signal. ROI scales with the beneficiary count used.
3. **Myanmar denominator** is the total active ER portfolio (~$16.8M), NOT the single
   earthquake project ($1,215,388). Using the smaller figure would overstate ROI ~20x.
4. Syria, Afghanistan, DR Congo, Ukraine are **correlational** — recovery alongside
   investment, not attributable impact.
5. The earlier "$1.13 per $1" sector-multiplier figure could NOT be reproduced from repo
   data and is not used.
