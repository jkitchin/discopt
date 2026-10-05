# #1654 — big-M coefficient tightening on the HiGHS MILP route: the §5 panel (2026-10-05)

**Status: GRADUATED ON INTRODUCTION, default ON.** `DISCOPT_MILP_COEF_TIGHTEN`
(`lp_milp_highs._coef_tighten_enabled`); `=0` restores the pre-#1654 route.

## What the change is

`lp_milp_highs.coefficient_tightened` is the textbook MIP coefficient tightening
(Savelsbergh 1994; Achterberg 2007 §10.1): in a one-sided row `a x <= U` a binary's
coefficient is shrunk by the row's slack at the binary's redundant value, computed
from the declared box of the other columns. Both restrictions (`y = 0`, `y = 1`) are
unchanged, so the **integer-feasible set is exactly the same** and only the LP
relaxation tightens; the slack is reduced by a summation round-off bound
(`8 (k + 2) eps sum|terms|`) before use, which can only weaken the rewrite.

`_coefficient_tightening_rescue` runs it **only when the primary route did not
certify**, so a model the route certifies today is handed to HiGHS unchanged. The
tightened solve goes through the whole certified pipeline (#1295, #1410, #1509,
#1612, #1621, #1634); its incumbent is re-verified on the untightened form, and a
verified point of the primary below its bound refutes it.

The same PR makes a refused HiGHS incumbent (the #1380 integral-realisation check
failing by `tol * M`) re-derivable by the fixed-integer LP over its integers
(`_repair_refused_incumbent`) instead of ending the solve as `error`. That part is
primal-only and not behind the flag.

## Panel

`discopt_benchmarks/scripts/panel_1654_coef_tighten.py OUT.jsonl 4`: three big-M
families (single-machine sequencing with disjunctive precedence, capacitated facility
location, fixed charge) x 4 seeds x M in {1e4, 1e6, 1e8, 1e10}, flag OFF vs ON,
interleaved with the arm order alternating, each solve in a fresh subprocess with a
branch-marker assertion (§8). For M at or above the activity bound every M is the
same MILP, so the oracle is the solve at the family's tight `M_ref`. Every incumbent
is independently re-checked against the model's rows and integrality. Load average
1.3 at the end of the run.

| arm | certified optimal | `error` | `feasible` | soundness violations | total wall |
|---|---|---|---|---|---|
| OFF | 22 / 48 | 25 | 1 | 0 | 20.3 s |
| ON  | **48 / 48** | 0 | 0 | 0 | 25.7 s |

96 executed oracle checks; 0 bounds above the reference, 0 infeasible incumbents,
0 certified wrong values; **certification regressions: none**; gains: 26.

* **Cert-clean:** yes (above).
* **Net-positive:** yes on the class it fires on: 26 `error`/uncertified results
  become certified optima. The +5.4 s total is the rescue's re-solve on the 26
  instances it rescues; on the 22 the primary certifies it never runs.

**In-repo corpus arm.** None of the 66 `python/tests/data/minlplib_nl/` instances
classifies as LP or MILP (scanned with `classify_problem`), so the HiGHS MILP route
and therefore the rescue never run on it: the corpus is byte-identical by
construction, the situation the `DISCOPT_CLOSED_EDGE_ENVELOPES` row of
`flag-retirement-audit.md` records. The 413 tests of the MILP-route regression files
(#1295, #1309, #1320, #1380, #1410, #1509, #1537, #1612, #1621, #1634, #1640, ...)
pass with the flag ON.

## The issue's own repros

* Sequencing (8 jobs): M = 1e6 ... 1e9 all `optimal` 47, bound <= 47 (was `error`
  from M = 1e7).
* #1621 blending: M = 1e8, 1e9, 1e10 all `optimal` 473,958.43 in < 0.5 s (was
  `feasible`, `feasible` at an 11 % worse point, `error`).
