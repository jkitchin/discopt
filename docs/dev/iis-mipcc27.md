# MILP IIS engine and the MIPcc27 entry program

Branch `feat/iis-mipcc-entry`. The MIP Competition 2027 ("Explainability and Small
Infeasible Subsystems") asks for a small infeasible subsystem of an infeasible MIP, with
optional feasibility certificates; a run scores only if the official checker passes it.
This note records how to run discopt's entry, what was measured, and which hypotheses
were falsified on the way. The competition kit (checker, `smallset` of 50 instances with
Gurobi reference solutions) is not vendored here.

## Running the entry

```bash
MIPCC_TIME_LIMIT=3600 MIPCC_THREADS=8 python -m discopt.mipcc INSTANCE.mps.gz OUT.json
python -m mipcc_iis_checker --solver highs -i INSTANCE.mps.gz -s OUT.json   # the kit's checker
```

`discopt/mipcc.py` reads the MPS with HiGHS straight into arrays (no `Model`), runs the
default deletion-order portfolio through `_iis_milp.milp_iis_arrays`, re-checks every
certificate against the checker's rule with a safety margin (others within 0.5e-5, the
certified element violated by more than 2e-5; the checker uses 1e-5 for both), and writes
the solution atomically. The whole model is written first and stays the answer if the
engine fails or runs out of time; a watchdog exits 0 before the limit. The library API is
`compute_iis(model, include_integrality=True, portfolio=..., portfolio_wait="fast"|"all")`.

## The rule that is not relaxed

Every returned subsystem is proven infeasible twice (#1634): by presolve-on HiGHS and by
an independent second route. The competition checker accepts a presolve-on proof alone;
the owner decided (2026-11-03) to keep the two-route rule anyway. Second routes, tried in
this order inside `_Engine.confirm_infeasible`:

1. a mod-2 (parity) certificate over integer equality rows (`_parity_infeasible`);
2. **the exact reduction** (`discopt/_iis_reduce.py`, below) followed by presolve-free
   HiGHS on the reduced model;
3. presolve-free HiGHS on the raw set;
4. the rational decider (`discopt/_iis_exact.py`) for small sets.

## What changed, with the measurement behind each

| Change | Measured effect (MIPcc27 smallset, official checker PASSED) |
|---|---|
| Entry program (`discopt.mipcc`) | 50 instances at 600 s: 49 pass the checker (no-ip-65059: even the checker cannot prove its full model in 1000 s), 42 complete, 36 at or below Gurobi's size, 39 fully certified |
| Checkpoint 0 gets its own budget (not the 30 s probe cap) | dives and neos-1461051: no answer -> a checker-passing answer |
| Rolling checkpoints inside phases | anytime answers follow the row/bound phases (dives 9500 -> 8706 at 600 s) |
| An unfinished run answers with its smallest confirmed set | fixed dijle answering 8005 instead of 6282 |
| `portfolio_wait="fast"` default (others stop clamp(t, 10, 30) s after the first member completes) | 100-200 s earlier on neos859080 / mod008inf / hole10 / jjliu_clique; jjliu_clique 277 vs 266 |
| Exact reduction as a second-route pre-pass (`_iis_reduce`) | dijle path sets: presolve-free undecided at 300 s -> confirmed in 1-11 s; dingo / dives captured sets: 0.3 s / 0.0 s |
| Confirmation budgets sized from confirmations (R2) | no effect alone (falsified); needed by the reduction route |
| Integrality phase relaxes whole columns (integrality + bounds) | dijle 6282 -> 3961 (reduction) -> **2426** at 600 s (Gurobi 2402); neos-942886 47 -> 38; dell 158 -> 157; misc05inf 126 -> 127 |
| Order flag `i` + `ICB-i` portfolio member | PANEL_RESULT |

dingo at 3600 s (2 workers): 4464 (Gurobi 1867). Its limit is the cost of each removal
probe (12.6 s, the same 7838-node tree every time), not confirmation.

## Soundness of the exact reduction

Five rules over `fractions.Fraction`, each preserving "the feasible set is nonempty" in
both directions with zero objective: propagation with integer rounding; redundant rows
and fixed columns; dominated columns; coefficient tightening of binaries in one-sided rows;
aggregation `u := c0 + c1 x` of a column every row but one prefers in one direction. The
export scales each row to integer coefficients and rounds bounds and sides outward (a
relaxation), and declines (no conclusion) when a magnitude exceeds 1e9. Only infeasibility
is ever concluded. Checked against HiGHS on 599 random models with 0 disagreements;
`python/tests/test_iis_reduce.py` keeps a 150-model version and asserts the route never
concludes on a feasible set.

## Falsified on the way (binding; do not retry without new evidence)

* The final confirmation fails on dijle/dingo/dives because of the 30 s probe cap: with
  80-92 s it still failed.
* Integrality-phase checkpoints give smaller answers: relaxing an integrality turns the
  column's bounds into counted elements; mid-phase sets count more than the base set.
  (A retraction of an earlier claim in the same session.)
* discopt's Rust MILP as a second route: 0 of 3 captured sets decided in 60 s.
* Re-implementing HiGHS presolve as an exact proof by propagation + Farkas: presolve never
  returns `kInfeasible` on these sets; the proof is reduction plus a tree (943-7838 nodes).
* A cut-free exact branch-and-bound as a third route: 69k nodes undecided on a model HiGHS
  proves in 943; 41% of float Farkas rays fail the exact check.
* Presolve-surviving rows as an IIS candidate on dingo: they cover 8% of Gurobi's IIS rows.
* Dropping the reducer's "redundant" rows as one batch on dingo: the rest is feasible
  (redundancy found sequentially is interdependent).

## Out of reach for now

* no-ip-65059: neither HiGHS nor the official checker proves the full model infeasible
  (1000 s); Gurobi's reference carries no certificates either.
* neos-1461051: the reduction does not apply; presolve-free confirmation of its sets takes
  more than the budget; Gurobi's reference is not a proven IIS.
* dingo / dives to Gurobi's size: needs cheaper removal probes (HiGHS exposes no proof
  support to batch deletions with).
