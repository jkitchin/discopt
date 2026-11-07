# #2212 entry experiments: the engine's cold path on the captured LP fixtures

`entry_experiment.py` replays every captured LP fixture in the tree
(`crates/discopt-core/src/{bnb,lp/simplex}/testdata/*_lp.json`, 18 LPs) through the
engine (`solve_lp_warm_csc` via the Python binding, default `SimplexOptions`) under
four starts, and checks each answer against HiGHS and independently of it.
`fixtures.jsonl` is its output on `798c370` (release build, this checkout asserted).
60 executed comparisons.

```
DISCOPT_PROFILE=1 python -u docs/dev/data/2212-cold-path/entry_experiment.py
```

| arm | what it is |
|---|---|
| `warm` | the fixture's captured start basis (the caller's inherited basis); the entry falls back to the cold primal on difficulty, and the counters say which happened |
| `cold` | no start: today's cold path, the two-phase primal |
| `logical` | dual simplex from the fixture's own unit columns as a basis, structurals at their cost-sign bound (the #2202 kernel retry's start) |
| `augmented` | #2163's general form (`ScaledProblem::logical_basis_dual`): append `-e_i` per row fixed at `[b_i, b_i]`, so an all-logical basis always exists |

## The oracle needed fixing first (three instrument findings)

Read before trusting any HiGHS comparison of these fixtures:

1. **HiGHS drops matrix entries below `small_matrix_value` (default 1e-9).** Six fixtures have
   such entries. On `ex1252a` HiGHS stored 582 of 593 nonzeros and called the LP
   **infeasible** (minimum L1 row violation 0.3447). The engine's point satisfies the
   full matrix to an L1 residual of 3.6e-7, and HiGHS with the entries kept agrees with
   the engine. The oracle runs at `small_matrix_value=1e-12`, HiGHS's floor. Three
   fixtures (`adhya1stp`, `c39_surplus_slack`, `tspn12`) still have entries below that,
   down to 5.6e-17, and are marked `oracle_exact: false`.
2. **HiGHS presolve moves `nvs01`'s optimum**: 9.1004 with presolve, 9.0833 without and at
   1e-10 tolerances, which is the engine's value (point residual 4.7e-10). The oracle runs
   presolve off at 1e-10.
3. **`cstr_bigm_dual_rebuild` is infeasible to HiGHS at 1e-10 tolerances and optimal at its
   defaults** (the fixture's recorded `highs_obj` was taken at the defaults). Even the
   oracle cannot say which it is, so that LP is on the feasibility edge.

Every `optimal` answer is also checked without HiGHS: activity-scaled row residual
(≤ 2e-10 on every one), bound violation, and the Neumaier–Shcherbina safe bound from
the returned duals (never above the objective). `|b|`-scaled residuals are misleading
on `qplib2170`: a column parked at 9.999e19, on the 1e20 sentinel, makes a 5e-21 relative
residual read as 0.5. The bound is right there and the point is unusable. That is the
#2125 huge-bound class, not this issue.

**No arm produced a wrong verdict on any fixture.**

## Results

| fixture | HiGHS | warm | cold (today) | logical | augmented |
|---|---|---|---|---|---|
| nous1 (#2202, optimal) | optimal | numerical | numerical | **optimal** | **optimal** |
| nous1 (#2202, infeasible) | infeasible | numerical | numerical | **infeasible** | **infeasible** |
| relu_anderson_cuts_numerical (#2163) | optimal | numerical | numerical | refused | **optimal** |
| relu_anderson_cuts_false_optimal (#2163) | optimal | numerical | numerical | refused | **optimal** |
| qplib2170_cold_fail | optimal | optimal | numerical | **optimal** | refused |
| ex1252a_drive_out_cycle | optimal | numerical | optimal | optimal | optimal |
| bchoco06_unstable_pivot | optimal | **optimal** | numerical | numerical | numerical |
| cstr_bigm_dual_rebuild (#2142) | infeasible (optimal at default tol) | numerical | numerical | numerical | numerical |
| 10 others | — | decided where run | decided | decided where run | decided where run, except `adhya1stp` (numerical; cold decides) |

"refused": no full logical basis from the fixture's own columns (`logical`), or a costed
column whose selected bound is open (`augmented`).

## What it answers

**Q1, inherited start vs cold fallback: both are broken on nous1, and only the fallback matters.**
The infeasible LP's inherited basis is singular (κ₁ 1.5e21, `DualPrepRejectSingular`), so the
warm dual never starts. The optimal LP's inherited basis is accepted (κ₁ 1.3e13), and the dual
hands off to the cold primal four times (`DualColdFallbacks 4`). In both cases the cold primal
then ends `Numerical` with `Phase1EndBoxViolated` + `Phase1DriftDominates` + `LpAuditBoundsFail`:
the #956 T2′ phase-1 drift. The `cold` arm alone fails identically. Fixing the start would
not help; fixing the fallback does.

**Q2, a logical-basis dual as the cold path: decides 16 of 18, today's decides 11.**
The proposed cold path is the augmented logical dual, then the fixture's own logical
basis when that is refused, then the primal as last resort. It decides everything except
`cstr` and `bchoco06_unstable_pivot`, with no wrong verdict. The second is decided by its
warm start, so as a whole entry the engine would leave only `cstr` undecided. The order
matters: neither start dominates the primal alone (`adhya1stp`: augmented `Numerical`,
primal decides), and neither logical form covers the other (`qplib2170` needs the
fixture's columns because a costed column's bound is open; the ReLU LPs need the
augmentation).

**What it does not settle:**
- `cstr`. Nothing decides it, HiGHS's verdict depends on its tolerance, and the augmented
  dual falls back to the primal as well (`DualColdFallbacks 4`). #2212's "done" list asks
  for it. It needs its own look (phase-1 repair, or accepting that its correct answer
  depends on the tolerance).
- Cost. Pivots per solve are here (`iters`); the #1008 throughput panel and the
  equality-rich `max_iter` grind that kept `cold_dual_start` OFF are not measured yet.
- The 40-LP nous1 capture (performance-plan §121) was not regenerated; the two nous1
  fixtures are from it.
