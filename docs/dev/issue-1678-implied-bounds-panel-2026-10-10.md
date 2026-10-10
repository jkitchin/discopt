# #1678 (c): implied bounds before the #1654 big-M tightening. §5 panel (2026-10-10)

**Status: GRADUATED ON INTRODUCTION, default ON.** The flag is `DISCOPT_MILP_IMPLIED_BOUNDS`
(`lp_milp_highs._implied_bounds_enabled`). `=0` restores the #1654 tightening over the
declared box only.

## The failure

Take facility location (3 facilities, 4 customers) with `x_ij <= M y_i` and `x` left at the
default `9.999e19` upper bound. At M = 1e10 and M = 1e14 the HiGHS MILP route returned
`error` ("HiGHS MILP incumbent (kOptimal): row 8 violated by 30"). Raw HiGHS on the
`to_mps` export solves it in 1 node.

**Why the result was `error` and not `feasible`.** HiGHS's incumbent had every `y_i`
near 2e-9, inside its `mip_feasibility_tolerance`. At M = 1e10 that buys 20 units of
row slack, so every facility reads as closed while still serving demand. The
readback/feasibility gates refuse this point, and they are right to: it is infeasible.
The #1654 fixed-integer repair rounds it to all `y = 0`, which is infeasible too. That
leaves no verified point and no trusted bound, so under the route's contract `error` is
the honest status. The error is not hiding a solution discopt could have published.

The route did have a rescue for this class: the #1654 coefficient tightening. That
rescue skips any row with an open-box column (`|bound| >= READBACK_LIMIT`), so it never
fired here. The demand equality implies `x_ij <= d_j`, but nothing derived it.

## The change

`fbbt_box` gains `open_limit` (default `INF`, so every certificate caller is
unchanged). `coefficient_tightened(sf, implied_bounds=True)` first runs that
outward-rounded FBBT with `open_limit = READBACK_LIMIT`. Each non-logical column whose
open side the rows bound gets the derived finite side, and the tightened form
**declares** that box `B`. The integer-feasible set is unchanged for three reasons:

* every feasible point of the model lies in `B`;
* `B` lies inside the declared box;
* over `B`, each rewritten row has the same `y = 0` and `y = 1` restrictions as before.

If the implied box is empty, nothing is rewritten and the route decides the instance.
The rescue still runs only when the primary solve did not certify. Its incumbent is
re-verified on the untightened form.

## Differential bound test and feasible-point sampling

These are in `python/tests/test_1678_implied_bounds.py`, run over 16 generated facility
models. They mix demand equalities and `>=` demand, with and without capacities, all
with open `x`. For each model:

* every binary assignment is enumerated, and the fixed-integer LP is solved on both
  forms. Both forms are always feasible or infeasible together, with equal optimal
  values;
* random-objective vertices of each form, with the logicals re-derived, are feasible
  for the other form;
* the tightened LP relaxation bound is at least the untightened one, and at most the
  true MILP optimum.

The uncapped `>=` variant has no implied bound, so nothing is rewritten.

## Panel

Run as `discopt_benchmarks/scripts/panel_1678_implied_bounds.py OUT.jsonl 4
<repo-with-ref/>`. Flag OFF and ON are interleaved, and the arm order alternates. Every
solve runs in a fresh subprocess, with a branch-marker assertion and a `discopt.__file__`
check (§8). Every incumbent is re-checked independently against the model's rows, box
and integrality. There were 368 solves and 366 executed oracle checks. Load average was
12 to 29 throughout, because other agents were running. Wall times are therefore
reported only as "no measurable difference", not as a speed claim (§9).

| arm | flag | certified | `error` | `feasible` | wall total |
|---|---|---|---|---|---|
| #1654 families (finite boxes), 3 x 4 seeds x M in {1e4, 1e6, 1e8, 1e10, 1e14} | OFF | 60/60 | 0 | 0 | 21.0 s |
| | ON | 60/60 | 0 | 0 | 20.8 s |
| open-box families, 5 x 4 seeds x the same 5 M | OFF | 39/100 | 56 | 5 | 23.8 s |
| | ON | **63/100** | 32 | 5 | 24.5 s |
| HiGHS check set + i1183 (manifest `lp_milp/manifest.json`) | OFF | 22/24 | 0 | 1 | 17.3 s |
| | ON | **23/24** | 0 | 0 | 17.4 s |

Per open-box family:

* `facility_open_eq` (the issue's model) went from 8/20 to 20/20.
* `facility_open_cap` (`>=` demand plus capacities) went from 8/20 to 20/20. In both
  facility families every M >= 1e8 was `error` with the flag OFF.
* `facility_open_ge` (`>=` demand, no capacity), `fixed_charge_open` and
  `sequencing_open` are negative controls with no implied bound. Their results are
  identical in the two arms (8/20, 8/20, 7/20).

Check set: **2122** went from `feasible` to `optimal`. The flag-ON tightening rewrote 258
entries, against 141 with the flag OFF.

* **Cert-clean: yes.**
  * 0 bounds cross a reference.
  * 0 infeasible incumbents.
  * 0 certified wrong values.
  * 0 certification regressions.
  * 0 objective drift between arms that both certified.

  The analyzer flagged three things, and none is a soundness problem:
  * *2122, objective better than the reference, in both arms.* The manifest oracle
    (-187616.11, maximize) is a HiGHS solve at `mip_rel_gap = 1e-4`. Re-solving with
    `mip_rel_gap = 0` gives -187612.944194 (31 nodes, presolve on or off). discopt's
    incumbent is exactly that value, and its certified bound -187608.64 is at or above
    it.
  * *1451, a crash in both arms.* The installed highspy 1.12 cannot read `1451.lp`
    (a constraint named `end`), so the loader fails before discopt runs.
* **Net-positive: yes.** 25 instances that were uncertified became certified: 24 open-box
  facility instances, all of which had been `error`, plus 2122. Neither control arm
  changed. The added wall time is the rescue's re-solve, on the instances it rescues.

## What it does not fix

Big-M rows whose continuous column has **no** row-implied bound still return `error`
from M >= 1e8 in both arms. Examples are `>=` demand with no capacity, fixed charge
`sum x >= D` with open `x`, and sequencing with an open makespan. Optimal solutions do
have `x <= d`, but only by optimality (dual) reasoning, which FBBT does not do. Raw
HiGHS is no oracle on this class: on the MPS export of an uncapped `>=` facility model
(`_family(0, 1e10)` in the test file) it reports `Optimal` 8.39 where the true optimum is 32.57 (all `y` near 1e-9). Here
discopt's `error` refuses a false optimum. It is not hiding a solvable instance.
