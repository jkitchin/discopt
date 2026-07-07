# V — Default-path re-measure vs BARON after G1+G2 (defaults-only)

**Date:** 2026-07-07
**Branch:** `v-baron-remeasure` (from `origin/main` @ `0308a7d1`)
**Panel:** 61 vendored MINLPLib `.nl` (`python/tests/data/minlplib_nl/`), 60 s time limit,
`gap_tolerance=1e-4`, correctness tol abs=1e-6 rel=1e-4.
**Build:** release (`maturin develop --release`); pounce `.so` = 4.73 MB (release, not debug).
**Run:** discopt **defaults-only, no flags toggled** — measures what a real user gets.
BARON numbers reused verbatim from the 2026-07-06 baseline (only discopt changed).

## What is on the default path now (all merged to `main`, all default-ON)

- ILS cap (#532)
- PSD gate (#537)
- zero-spanning lift (#538)
- heuristic effort governor (#541)
- C-38 false-optimal fix (#536)

## Headline: before → after

| metric | baseline 2026-07-06 | this run 2026-07-07 | delta |
|---|---|---|---|
| **discopt proved-optimal** | **42** | **43** | **+1** (no cert lost) |
| discopt total wall (all 61) | 18.83 min (1130 s) | **18.18 min (1091 s)** | −0.65 min |
| discopt shifted-geomean wall | 4.74 s | 4.63 s | −0.11 s |
| **discopt VIOLATIONS** | **0** | **0** | — |

**Cert change:** GAINED `st_e36` (feasible→optimal). LOST: none. No proved-optimal regression.

## HARD CORRECTNESS GATE — result: PASS (0 violations)

- **discopt VIOLATIONS = 0** on the 61-panel (harness verdict, oracle = MINLPLib `primalbound`).
- **Independent `minlplib.solu` cross-check:** every discopt `status==optimal` objective
  matches its `=best=`/`=opt=` value within tol. **CLEAN** — no C-38/C-40-shape false-optimal.
- **Dual-bound cross-check:** no discopt dual bound crosses the `.solu` oracle. **CLEAN.**
- **`util` reconciliation:** the known C-40 `util` false-optimal is **NOT in this 61-panel**
  (it was caught on a different held-out set). 58/61 panel instances carry a `.solu`
  primal oracle; the 3 without one are the `bchoco06/07/08` class (`known=null` in both
  baseline and this run — n/a, not a cert). The baseline's "0 violations" needs no
  reconciliation for this panel.

## Per-instance wall deltas — where the capabilities moved the needle

**The one real win — governor freeing search budget:**

| instance | base | now | Δ | nodes | note |
|---|---|---|---|---|---|
| **st_e36** | 60.09 s (feasible) | **16.26 s (optimal)** | **−43.8 s** | 883→153 | **new cert** — proved within budget |

**Integer instances — ILS cap (#532):**

| instance | base | now | Δ | nodes |
|---|---|---|---|---|
| nvs06 | 4.07 s | 1.60 s | −2.46 s | 5→5 |
| nvs01 | 5.50 s | 3.09 s | −2.41 s | 17→17 |
| nvs08 | 2.76 s | 1.05 s | −1.71 s | 57→57 |
| st_e38 | 1.22 s | 0.60 s | −0.62 s | 3→3 |

Node counts unchanged on these — the ILS cap trims per-node heuristic work, not the tree.

**Small regressions (noise / governor overhead where it doesn't pay):** m3 +0.86 s,
fac2 +0.94 s, clay0303hfsg +2.70 s, casctanks +1.20 s, several `tspn*`/`heatexch*`
+0.5–1.5 s. All node-neutral or near it; none crosses a cert boundary. These are the
governor/lift paying a small fixed tax on instances where its win does not land.

**Panel-invisible wins (the instrument limitation).** The **PSD gate (#537)** and
**zero-spanning lift (#538)** target the QCQP class. The QCQP probes named in the roadmap
(`nvs17`, `nvs19`, `nvs24`) are **NOT in this 61-panel**, so those two capabilities are
essentially invisible here. The panel-visible improvement is therefore small **by
construction** — the wins concentrate off-panel. This is the known limitation of the
BARON-vs-discopt 61-panel as an instrument for G1+G2, and it is why the headline gain is
+1 cert / −0.65 min rather than something larger.

## The BARON gap

- BARON proves **44**, discopt proves **43**; **40 jointly proved**.
- **BARON proves but discopt does not (4):** `nvs05`, `nvs09`, `tanksize`, `tls2`
  — discopt returns an honest feasible/time-limit incumbent (GAP), not a wrong answer.
- **discopt proves but BARON does not (3):** `chance`, `dispatch` (BARON stops at
  `2 Locally Optimal` — not certified global), `tspn05` (BARON `8 Integer Solution` at 60 s).
- **Wall ratio on the 40 jointly-proved:** discopt/BARON geomean **14.4×** (median 14.6×,
  min 1.40×, max 180.6×). discopt is correct but slower; closing this ratio is the
  remaining performance work, and it lives mostly off this panel.

## Verification

- Release build; pounce `.so` = 4.73 MB (confirmed release, ~10× smaller than a debug build).
- Bound-neutral spot check: `alan` optimal, obj 2.925, 21 nodes — identical to baseline.
- Node counts unchanged on the bound-neutral instances (ILS cap trims work, not tree).
- Raw discopt-only sweep JSON + the BARON-merged JSON under `reports/`.

## Artifacts

- `reports/global_opt_baron_vs_discopt_2026-07-07T11-06-06.json` — raw discopt-only sweep
  (BARON = SKIPPED).
- `reports/v_baron_remeasure_2026-07-07T11-06-06.json` — merged: this run's discopt +
  baseline BARON, verdicts recomputed.
