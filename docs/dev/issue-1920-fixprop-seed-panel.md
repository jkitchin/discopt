# #1920: the HiGHS MILP route on ReLU networks (bound, speed, incumbent)

**Status: both flags GRADUATED ON INTRODUCTION, default ON.**
`DISCOPT_MILP_CONCURRENT_CROSS` (`lp_milp_highs._concurrent_cross_enabled`) and
`DISCOPT_MILP_FIXPROP_SEED` (`lp_milp_highs._fixprop_seed_enabled`). `=0` restores the
sequential cross-check or the cold start. The `_withdraw` bound rule is not flagged.

## The issue

The issue minimised a 2-64-64-1 ReLU network embedded with `add_predictor(...,
method="relu_bigm")`. The model is a pure MILP, so it goes to the HiGHS route.

* A 600 s run returned `feasible` at the optimum with bound **-15.30**. Runs of 20 s,
  60 s and 180 s returned bounds of -6.7 to -4.4. A longer run gave a looser bound.
* The same model was certified `optimal` in 427 s in one run, and ended `feasible`
  in another.
* discopt took longer than HiGHS alone (364-371 s through Pyomo + OMLT).
* Every run of 180 s or less had **no incumbent** (`objective=None`).

The weights were not attached, so every measurement here uses random networks of the
same shape (`relu_layers` in `discopt_benchmarks/scripts/panel_1920_milp_route.py`).

## Root causes (measured)

1. **Bound.** HiGHS proves the MILP optimal late in the budget. The #1634
   presolve-free cross-solve then runs on whatever time is left, and reaches no
   verdict. `_withdraw` withdrew the certificate, which is correct. It also replaced
   the bound with the NS-safe ROOT LP bound, throwing away both tree bounds.

   Reproduced on a 2-32-32-1 net. At `time_limit=5`, HiGHS stops on the time limit and
   its tree bound is -2.06. At `time_limit=11`, kOptimal arrives at about 8 s, the
   cross-solve is cut off after about 3 s, and the run returns the root bound -5.82.

2. **Speed.** The cross-solve ran *after* the primary. On three 2-40-40-1 nets:

   | seed | HiGHS alone | Pyomo + OMLT + HiGHS | discopt before | of which the cross-solve |
   |---|---|---|---|---|
   | 0 | 41 s | 72 s | 157 s | 100 s |
   | 1 | 23 s | 31 s | 66 s | 27 s |
   | 2 | 24 s | 19 s | 54 s | 30 s |

3. **Incumbent.** HiGHS's own heuristics find no point on this class.
   * Plain HiGHS on the route's standard form: no incumbent after 60 s (H=64).
   * HiGHS through OMLT's formulation: first incumbent at 53 s, value 0.284.
   * discopt's `relu_bigm` constraints match OMLT's, so the formulation is not the
     cause.

## The fix

1. **`_withdraw` and `_merge_uncertified_cross`.** When the cross-solve ran without a
   verdict, the bound is now `max(root, min(primary tree bound, cross tree bound))`.
   `_weaker_bound` already certifies on that rule: the two configurations fail on
   disjoint instances, so the `min` is valid whenever either one is.

   The same rule now applies to a budgeted exit, where the cross-solve is stopped and
   its bound joins the primary's. Without that, monotonicity could not hold:
   * A budgeted exit would publish the presolved bound alone.
   * A withdrawn optimum would publish the `min`.

   The presolved bound alone is exactly what #1634 found can lie above the optimum:
   seed 965 is 3e-6 above it with the cross-solve undecided. So the consistent rule is
   the `min`, not the presolved bound.

   A cross point is used only after re-derivation: the LP over its own integers,
   verified. This was learned from the panel; see Panel run 1 below.

2. **Concurrent cross-solve** (`DISCOPT_MILP_CONCURRENT_CROSS`). HiGHS releases the
   GIL, so the presolve-free solve runs in a thread beside the primary, from the same
   start and under the same time limit.
   * It is stopped through HiGHS's MIP interrupt callback (`_CrossStop`), at once when
     the primary ends uncertified.
   * When the primary certifies, it stops at the #1634 cap, measured from the start.
   * The verdict rule is unchanged (`_judge_presolve_cross` is the old tail of
     `_cross_check_presolve`).
   * The cross-solve still starts cold (#1733). The user's HiGHS options are carried
     into the thread (`contextvars.copy_context`).

3. **Fix-and-propagate start** (`DISCOPT_MILP_FIXPROP_SEED`). With no user warm start,
   the structural columns are fixed in order at their domain midpoints, with FBBT
   propagation on `A x = b` after each fix. The integers are then re-optimised by the
   fixed-integer LP, and the point is kept only if it verifies.
   * On a ReLU net, fixing the inputs lets propagation decide every neuron. It found a
     point in 0.3-0.5 s on 6/6 nets.
   * Budget: 2 s, or 5% of the time limit if smaller. Work is capped at 50 row visits
     per nonzero.
   * It seeds the primary only. If HiGHS ends with no incumbent (or a worse one), the
     verified start is published as the incumbent.

   Entry experiment: seeding plain HiGHS changed time-to-optimal from 57→66, 37→31
   and 23→20 s on the three 2-40-40-1 nets. That is within HiGHS's run-to-run
   variability, so the start does not slow the solve.

4. **A seeded `kOptimal` with no dual bound is re-solved cold.** The fast lane found
   this after panel run 2: `test_nonlinear_to_pwl::test_outer_rows_never_cut_a_graph_point`
   failed for `sin_plus_sq` and `tanh`.

   The model is a piecewise-linear MILP with its input fixed. Handed the start,
   HiGHS's presolve accepted it and stopped `kOptimal` at 0 nodes with
   `mip_dual_bound = -inf`. The route then certified `optimal` with `bound=None`, and
   `Model.solve` published it uncertified with no bound.

   Cold, the same HiGHS presolve returns a **false** `kInfeasible`. The #1634
   cross-check refutes that and publishes the point with the NS root bound.

   A user's warm start reaches the same state, so this is not specific to the new
   start. `_solve_milp_std` now re-runs such a solve cold
   (`milp/seeded_optimal_without_bound`).

## Panel (`panel_1920_milp_route.py`)

Three configurations per instance, interleaved with the order rotating. Each solve
runs in a fresh subprocess with a branch-marker assertion. The source file's md5 was
asserted unchanged across the run.
* `base`: both flags `=0`.
* `conc`: concurrent cross-solve only.
* `on`: the default.

Arms:
* `relu`: 2-H-H-1 nets, H ∈ {16, 24, 32}, 4 seeds, 300 s. Oracle: a 1601×801 grid
  minimum, an upper bound on the optimum, plus a forward pass at the returned input.
* `relu-tl`: 2-64-64-1, 2 seeds, 20 s and 40 s.
* The #1733 panel's arms with their oracles: the eps family cold/warm, the #1654 and
  #1678 big-M families at M ∈ {1e4 … 1e14}, and the 24 HiGHS check instances.

672 solves, 1264 executed oracle checks. Load average was 0.9-1.6.

Between runs 2 and 3, `_fixprop_start`'s cap became `min(FIXPROP_MAX_SECONDS, ...)`
instead of `min(cap, ...)` with `cap == FIXPROP_MAX_SECONDS`. This is value-identical:
it lets the #1153 budget scanner see the constant ceiling.

### Panel run 1 (falsified a draft)

The first full run found **10 infeasible incumbents**, on the #1654 `sequencing`
family at M ∈ {1e10, 1e14}, in `conc` and `on` alike. Example: objective 11.0 against
an optimum of 31.0, with a model violation of 8.

In that draft, `_merge_uncertified_cross` adopted the presolve-free solve's raw
point, which passed `_verified_mip_point`. That verifier's relative row tolerance
(`FEAS_RTOL · Σ|A_ij||x_j|`) lets a point spend about `1e-9 · M` on a big-M row. The
merge then picks the best objective, which selects exactly those points. The poisoned
11.0 then outranked the #1654 rescue's certified 31.0.

The LP over the same integers gives 31.0 with zero model violation. The merge now
uses only that re-derived point, and takes nothing when no budget is left for the LP.
Regression test: `test_merge_never_adopts_a_tolerance_point_on_huge_big_m`. It fails
at 11.0 on the raw-point draft.

### Panel run 2

This run was on the code before change 4. It was cert-clean with the same counts as
run 3. The fast lane then found change 4's defect, so the panel was run again.

### Panel run 3 (final code)

The md5 of `lp_milp_highs.py` was asserted unchanged from start to end. Load average
at the start was 0.7 (1-min) and 6.1 (15-min), right after a test run. At the end it
was 2.0.

| arm | base certified | conc certified | on certified | wall total base / conc / on |
|---|---|---|---|---|
| relu (12) | 12 | 12 | 12 | **171.0 / 103.7 / 100.1 s** |
| relu-tl (4) | 0, **0 incumbents** | 0, 0 incumbents | 0, **4 incumbents** | 121.1 / 121.1 / 124.7 s |
| eps cold / warm (12 + 12) | 24 | 24 | 24 | 5.2 / 5.2 / 5.0 s |
| #1654 families (60) | 60 | 60 | 60 | 30.4 / 27.8 / 28.6 s |
| #1678 open-box (100) | 63 | 63 | 63 | 36.8 / 36.4 / 36.9 s |
| check set (24) | 24 | 24 | 24 | 51.2 / 46.1 / 48.7 s |

* **Cert-clean: yes.**
  * 0 bounds above a reference or the grid minimum.
  * 0 infeasible incumbents.
  * 0 certified wrong values.
  * 0 certification regressions.
  * 0 certified-objective drift.

  The analyzer flagged one item: 2122, in all three configurations alike. That is
  the known #1678 oracle artefact. The manifest's value is a `mip_rel_gap = 1e-4`
  solve, and the exact optimum is discopt's -187612.944194.
* **Net-positive: yes, on the class.**
  * The 12 ReLU solves certify the same optima in 41% less total wall (median 4.56 →
    3.02 s). In run 2, on the largest nets: 19.1 → 11.8, 45.8 → 21.8, 57.0 → 35.7
    and 19.9 → 10.6 s.
  * Every time-limited 64-wide run now has a verified incumbent, where neither HiGHS
    alone nor OMLT had one.
  * The other arms are unchanged. Node counts changed on 3 of 208 non-ReLU
    instances, all certified identically.
* **Cost, recorded.** Time-limited bounds on `relu-tl` (run 3; run 2 in brackets):

  | instance, limit | base | conc | on |
  |---|---|---|---|
  | relu64/0, 20 s | -5.76 (-5.76) | -6.52 (-6.52) | -6.52 (-7.00) |
  | relu64/0, 40 s | -5.10 (-5.10) | -6.07 (-5.71) | -5.71 (-5.75) |
  | relu64/1, 20 s | -6.47 (-7.16) | -7.18 (-7.18) | -7.18 (-7.18) |
  | relu64/1, 40 s | -5.90 (-5.90) | -5.90 (-5.90) | -5.98 (-5.98) |

  `base`'s bound is the presolved tree bound alone, which nothing confirmed and which
  #1634 showed can be false. The default's bound is the `min` of the two trees, so it
  is looser by up to about 1 on this class. Within each run and configuration it is
  monotone in the limit.

  The same configuration differs between the two runs by up to 0.7 (base,
  relu64/1, 20 s). So the time-limited bound is timing-dependent, and differences
  smaller than that are noise.

  `on` overshoots the time limit by about 1 s more than `base`. That is
  `Model.solve`'s existing post-processing of a published incumbent
  (`_repair_published_incumbent`, dual unpacking; profiled). A run with no incumbent
  skips it.

**In-repo corpus.** As the #1654, #1678 and #1733 panels recorded, none of the 66
`python/tests/data/minlplib_nl/` instances classifies as LP or MILP. That arm is
byte-identical by construction.

## Still open

* After the concurrent change, discopt's certified wall is the slower of the two
  HiGHS configurations. On the three 2-40-40-1 nets: 104 / 39 / 32 s, against 41 /
  23 / 24 s for HiGHS alone and 72 / 31 / 19 s through OMLT. The rest of the gap is
  the presolve-free solve being slower than the presolved one. #1634 measured that
  neither configuration is sound alone, so that solve stays.
* The fix-and-propagate point is a feasible point, not a good one (0.58 against a
  grid minimum of -0.03 on relu64/0). HiGHS did not improve it within 40 s.
