# #1733 — logical columns as row bounds on the HiGHS MILP route: the §5 panel (2026-10-17)

**Status: GRADUATED ON INTRODUCTION, default ON.** `DISCOPT_MILP_ROW_FORM`
(`lp_milp_highs._row_form_enabled`); `=0` hands HiGHS the explicit slack columns again.

## The bug

An epsilon-constraint step on a 5-site, 7-customer facility-location MILP (all binary):
minimise cost subject to `emissions <= e* + 1e-6`, where `e*` is the minimum emission.
The enumerated optimum is 363.3962. On `main` @ a9c7fbd:

| solve | status | objective | bound |
|---|---|---|---|
| cold | `feasible` | 403.3962 | 322.6373 (NS root) |
| warm start from the emission-optimal point | **`optimal`** | 433.3962 | **433.3962** |

The warm solve certified a false bound. `discopt.mo.epsilon_constraint` warm-starts
by default, so its payoff table and nadir point were wrong on MILPs.

## Root cause

The route hands HiGHS its standard form `A x = b`. Each inequality row carries a
continuous logical (slack) column with zero cost. Replaying that `StdForm` in highspy:

| form | presolve (route rules) | presolve off |
|---|---|---|
| slack columns (the route) | `kOptimal` **403.3962** | `kInfeasible` |
| same rows as row bounds | `kOptimal` 363.3962 | `kOptimal` 363.3962 |

The only difference is the slack columns. The route's falsifiers did not catch it:

* The #1509 refutation candidates (root LP point, fixed-integer LP, rounded root) all
  sit at or above 403.
* On the cold solve, the #1634 presolve-free cross-solve's `kInfeasible` was "no
  verdict", so the certificate was withdrawn.
* On the warm solve, the cross-solve was handed **the same user seed**. It returned
  the seed as `kOptimal` and *confirmed* the primary.

Entry experiment (before any code): 100 seeds of the issue's generator, with the
right-hand-side shift in {1e-6, 0, 1e-4}. Each instance is replayed in highspy with the
route's options, slack form vs row form, presolve on and off. The oracle is
exhaustive enumeration of all 5^7 assignments in integer arithmetic, since every
emission coefficient is a multiple of 0.2. 1200 executed comparisons:

| shift | form | presolve on: wrong | presolve off: wrong |
|---|---|---|---|
| 1e-6 | slack | **25 / 100** | **8 / 100** |
| 1e-6 | row | 0 / 100 | 0 / 100 |
| 0 | slack / row | 0 / 0 | 0 / 0 |
| 1e-4 | slack | 0 / 100 | 1 / 100 |
| 1e-4 | row | 0 / 100 | 0 / 100 |

"Wrong" covers two outcomes: `kInfeasible` on a feasible model, and `kOptimal` at a
value or bound above the optimum. The #1634 witnesses were also a slack-column
artefact. There, HiGHS's parallel-column rule paired the route's slack `s` with an
integer column.

## The fix

1. `_eliminate_logicals` folds each logical column into its row's bounds before
   `passModel`. Each is continuous, zero-cost and in one row. A column is folded only
   when it is its row's sole candidate, and never a model column below `n_struct`.
   An earlier draft folded the first candidate in each row. On the #1229 replay
   (`n_struct=None`) that folded a model column instead of the slack and returned it
   at the 1e20 sentinel. A logical `a s + r(x) in
   [lo, hi]`, `s in [l, u]` becomes `r(x) in [lo - max(al, au), hi - min(al, au)]`.
   This is exact, and the ends are rounded outward when the subtracted term is
   nonzero. A side of `s` at the finite `INF = 1e20` sentinel opens the matching row
   end. Openness is tested on the bound, not on the product. An earlier draft
   computed `coef * 1e20`. For a #1537-rescaled logical (`|coef| < 1`) that gives a
   *finite* row end HiGHS would treat as binding. The panels below were first run on
   that draft. Every number recorded here is from the rerun on the final code.

   `_lift_logicals` rebuilds the full point from HiGHS's solution: each logical gets
   the value its row determines, clipped to its box. Every point is still verified
   against the untouched `StdForm`, and the user's seed is projected the same way.
2. `_cross_check_presolve` no longer passes the user's warm start to the
   presolve-free cross-solve, so the falsifier is independent of the primary's claim.
   This change is not behind the flag. With `DISCOPT_MILP_ROW_FORM=0` it turns the
   issue's warm 433.40 certificate into a withdrawn one (`feasible`, NS root bound).
3. `_rederived_point`: a cross-solve point that would refute the primary is first
   re-derived by the #1509 fixed-integer LP over its own integers.

   Row form exposed a latent flaw in that falsifier. The verifier admits a row
   violated by `FEAS_TOL = 1e-6`, which equals the refutation yardstick `CERT_ABS`.
   On the #1494 piecewise sweep, the presolve-free solve returned a point that used
   HiGHS's 1e-6 MIP row tolerance on the objective's defining equality. That
   equality carries no logical, so the elimination did not touch it; row form only
   changed HiGHS's search path. The point verified at objective -1e-6 against an
   exact 0, and withdrew 10 of 160 correct certificates.

   The LP over the same integers solves at a 1e-7 primal tolerance and returns the
   best point those integers admit. A genuine refutation therefore survives it, and
   a tolerance one does not. When the LP gives no verified point, the cross-solve's
   own point is kept, so the falsifier never has less evidence than before. Also not
   behind the flag.

4. **HiGHS-version gate** (`ROW_FORM_MIN_HIGHS = 1.12.0`). CI's `highspy==1.10.0` job
   failed on the PR's first push: the #1229 instance came back certified
   `infeasible`.

   Replaying it: HiGHS 1.10.0 and 1.11.0 label the row form of that (feasible)
   instance `kInfeasible`, with presolve on and off. The optimal point satisfies
   every folded row range, so the reduced model is correct; HiGHS is wrong. The
   minimal trigger is eight folded rows, among them a parallel pair and rows over a
   fixed integer column. 1.12.0, 1.13.0, 1.14.0 and 1.15.1 solve it. Older HiGHS
   keeps the slack form (`milp/row_form_skipped_old_highs`).
5. **`_cross_form_infeasible_check`.** The gate alone does not close the hole it
   exposed: the route trusts a HiGHS `kInfeasible` label once the presolve-free
   cross-solve agrees, and that cross-solve uses the same form. HiGHS 1.11.0 labels
   *this issue's* model infeasible on the slack form; main certifies that false
   `infeasible` on 1.11 today, and it fails the #1229 test there too.

   So a HiGHS-only infeasible label (provenance `highs`, not a Farkas proof) is now
   held against one solve of the other form. A verified point there refutes it. The
   point is published with the NS-safe root bound, and certified only if that bound
   closes the gap (the #1509 rule). With no verified point the label stands
   unchanged, so this check can withdraw a certificate but never add one.

   On 1.11 the issue model now comes back `feasible` at the optimum 363.3962,
   uncertified, instead of `infeasible`. The route, #1229 and #1733 test files pass
   on 1.10.0, 1.11.0, 1.12.0 and 1.15.1.

6. **Re-derived incumbent.** CI's "Python fast" job on the PR's first head failed
   `test_piecewise::test_graph_is_exact_at_and_between_breakpoints[incremental]`. It
   published y = f(x) + 1.0e-6, an honest certificate within the 1e-6 feasibility
   tolerance and the 1e-4 gap, but no longer exact.

   It does not reproduce locally on HiGHS's default seed; the search path is
   CPU-dependent. Sweeping `random_seed` 0–29 over the incremental graph's 30 probes
   (900 solves) reproduced it once on the row form (seed 16) and never on the slack
   form. HiGHS's objective beat the #1509 fixed-integer LP over the same integers by
   exactly 1e-6, row tolerance spent on the objective.

   When that difference is within the equality yardstick and the gap still closes
   with it charged twice (the case #1612 leaves alone), the route now publishes the
   LP's verified point (`milp/incumbent_rederived`). The sweep is 0/900 after.

**Tests that pin a falsifier.** Twelve tests exercise a falsifier on a witness where
HiGHS is wrong first:

* the #1634 cross-check alone;
* the #1621 rescale refutation;
* the #1640 tight cross-solve (five tests);
* the #1678 adoption yardstick.

On the row form HiGHS gets those witnesses right, so the falsifier never fires. Those
twelve tests now set `DISCOPT_MILP_ROW_FORM=0`, so they still exercise the mechanism
the falsifiers guard. `test_1733_...::test_falsifier_witnesses_are_right_on_the_row_form`
checks that the default route gets the same models right. The #1634 witnesses and
the #1494 sweep are already checked end to end on the default route by their own
files.

## Panel 1 — the issue's class through `Model.solve`

100 seeds x shift {1e-6, 0, 1e-4} x {cold, warm} x flag, interleaved, each result
checked against enumeration. 1200 executed comparisons. Run on the final code; load
average 1.7 at the start and 1.3 at the end. This was an
in-process run of the same model and oracle as panel 2's `eps` arm
(`panel_1733_row_form.eps_data` / `eps_oracle`), over 100 seeds instead of 4.

| shift | flag | mode | certified | optimum found | **incorrect** | mean wall (sd) |
|---|---|---|---|---|---|---|
| 1e-6 | 0 | cold | 70 | 94 | **2** | 0.060 (0.026) s |
| 1e-6 | 0 | warm | 61 | 89 | **1** | 0.057 (0.028) s |
| 1e-6 | 1 | cold | **100** | 100 | 0 | 0.061 (0.009) s |
| 1e-6 | 1 | warm | **100** | 100 | 0 | 0.060 (0.009) s |
| 0 | 0 / 1 | cold, warm | 100 each | 100 each | 0 | 0.061 s |
| 1e-4 | 0 | cold / warm | 99 / 98 | 100 / 100 | 0 | 0.060 / 0.061 s |
| 1e-4 | 1 | cold / warm | 100 / 100 | 100 / 100 | 0 | 0.061 s |

Rerun after change 5 (cross-form infeasible check): the ON arm is unchanged. In the OFF arm
the optimum is now found on 94 of the 1e-6 cold instances (was 91), because refuted
`infeasible` labels now publish the other form's point.

The three flag-OFF false certificates are seed 44 (cold, 400.30 against 381.20) and
seed 70 (cold and warm, 277.12 against 276.86). Both HiGHS configurations agree on
them, so no falsifier can catch them. That is why the slack form is an opt-out only.

## Panel 2 — graduation panel

`python -u discopt_benchmarks/scripts/panel_1733_row_form.py OUT.jsonl 4 <ref_root>`
covers four arms:

* the issue's class, with 4 seeds x 3 shifts x {cold, warm};
* the three #1654 big-M families;
* the five #1678 open-box families, both with M in {1e4 .. 1e14};
* the 24 LP/MILP check instances of `discopt_benchmarks/data/lp_milp/manifest.json`
  (HiGHS `check/instances`, fetched into `<ref_root>/ref/HiGHS`, plus i1183).

Arms are interleaved with the order alternating. Each solve runs in a fresh subprocess
with a branch-marker assertion (§8), and every incumbent is re-checked against the
model's rows, box and integrality. There were 416 executed oracle checks. Two runs on earlier drafts gave the same counts.
Load average was 5.1 at the start, right after a test run, and 1.9 at the end.

| arm | flag OFF certified | flag ON certified | other changes |
|---|---|---|---|
| eps cold / warm | 11/12, 11/12 | **12/12, 12/12** | seed 3 at 1e-6: `feasible` 348.21 → `optimal` 346.81 (oracle 346.81) |
| #1654 families | 60/60 | 60/60 | none |
| #1678 open-box families | 63/100 | 63/100 | `sequencing_open` at M ∈ {1e10, 1e14}: 8 × `error` → `feasible` with a valid bound |
| check set | 24/24 | 24/24 | 2122 and bell5 incumbents differ inside the gap (below) |

* **Cert-clean: yes.**
  * 0 bounds cross a reference.
  * 0 infeasible incumbents.
  * 0 certified wrong values.
  * 0 certification regressions.

  The analyzer flagged one item, which is the known #1678 oracle artefact. On 2122,
  the manifest oracle (-187616.11, maximize) is a `mip_rel_gap = 1e-4` value. The true
  optimum is -187612.944194, which the #1678 panel doc measured at `mip_rel_gap = 0`.
  The flag-ON incumbent is exactly that value; the flag-OFF incumbent is the worse
  -187616.11. On bell5 both arms certify within the default `gap_tolerance = 1e-4`:
  OFF 8966406.49 and ON 8966635.63, against an oracle of 8966413.71. Both bounds are
  at or below the oracle.
* **Net-positive: yes**, on the class the fix targets: 2 certifications gained in
  panel 2, and in panel 1, 600/600 certified against 528/600, with 3 false
  certificates removed. The open-box `sequencing_open` arm gains 8 answers where OFF
  returned `error`. No arm loses anything. Total wall time is within noise: check
  set 29.6 → 28.6 s, #1654 families 29.9 → 28.1 s, open-box 32.5 → 33.3 s.

**In-repo corpus arm.** As the #1654 and #1678 panels recorded, none of the 66
`python/tests/data/minlplib_nl/` instances classifies as LP or MILP. The HiGHS MILP
route therefore never runs on them, and that arm is byte-identical by construction.
The HiGHS-route regression files (every `python/tests` file that names the route)
pass with the flag ON.

## Not changed

* The #1537 logical-column rescaling (`_solve_milp_scaled`) still runs. With the
  logicals folded into row bounds, it no longer changes what HiGHS sees beyond
  rounding of the row ends. Its #1621 cross-check is then an extra solve that cannot
  disagree for a scaling reason. Removing it is a bound-neutral simplification and is
  left out of this change.
