# #1917: CLAUDE.md §5 panels for the default-OFF GDP flags

One `.jsonl` per flag, written by
`discopt_benchmarks/scripts/issue1917_gdp_flag_panel.py` (its docstring is the
method). Re-derive every verdict with `--report --flag <F> --out <F>.jsonl`; do not
copy a figure out of this README without doing so.

## Method

* **Panel:** the 24-instance `gdp_parity` panel (`gdplib_runner.GDP_PARITY`), under
  the `gdp_method` each flag gates (`FLAG_METHODS` in the script).
* **Fire phase:** each instance is lowered with
  `transformations.create_using("gdp.<method>")` with the flag OFF and ON, and the
  two `model_fingerprint`s are compared. Only instances whose lowering differs are
  solved, because the flag cannot touch the rest. `DISCOPT_GDP_ENUMERATE_CUTOFF`
  acts inside the enumerate algorithm, so every instance is solved for it.
* **Solve phase:** one fresh `gdp_baseline.py --worker` subprocess per
  (instance, arm, rep). The worker builds, converts and solves the model, then
  re-checks the incumbent in Pyomo.
  * 3 reps at 120 s each, OFF and ON interleaved, with the order alternated per rep.
  * Load gate `--max-load 10`: a run waits until the 1-minute load is below 10, and
    each row records its `load1`.
* **Bars:**
  * **cert-clean:** no `incorrect` verdict, no dual bound past a proven oracle,
    and no instance `solved` OFF that is not `solved` ON.
  * **net-positive:** more instances solved, or the same instances solved faster
    or in fewer nodes, and broadly rather than on one instance.

## Which code each row measured

Each solve row records `git_head`. The branch moved while the panels ran, and four
commits change what a row measures:

| commit | time (2026-10-31/11-01) | effect on rows |
|---|---|---|
| `51966cf6` | 21:13 | work budget on `DISCOPT_GDP_DISJUNCT_FBBT` setup |
| `018483fe` | 23:23 | `DISCOPT_GDP_DISJUNCT_FBBT` caps floored at 1e-3 of today's M (round 2) |
| `140da1b1` | 23:43 | POUNCE re-solves without all-fixed rows on `Not_Enough_Degrees_Of_Freedom` (every method's NLP) |
| `064efbf5` | 00:39 | each origin-free hull `t` gets its own column (hull only; see below) |

Within each rep the OFF and ON arms of an instance run back to back on the same
code, so the per-rep A/B comparison is fair. A cross-rep comparison is not: a row
before `140da1b1` can report ERROR or `unknown` on an NLP that a later row solves.

## Files

* **`DISCOPT_GDP_DISJUNCT_FBBT.round1-unfloored.jsonl`:** round 1, before the
  `018483fe` floor. It was cert-clean, and it is kept because it is the
  measurement that motivated the floor. `DISCOPT_GDP_DISJUNCT_FBBT.jsonl` is
  round 2.
* **`DISCOPT_GDP_ENUMERATE_CUTOFF.jsonl`:** 36 error rows, 6 per instance (both
  arms × 3 reps), come from
  `ValueError: enumerate: the GDP has more than max_configurations=10000` on
  biofuel, disease_model, med_term_purchasing, positioning, spectralog and
  stranded_gas. That is enumerate's documented refusal and it happens in both
  arms, so it is not a flag effect.
* **`DISCOPT_GDP_BOUND_PRETRANSFORM.jsonl`:** one syngas OFF row (mbigm, rep 1)
  reads `AttributeError: module 'discopt.solvers.nlp_pounce' has no attribute
  'solve_nlp'`. The worker imported `nlp_pounce.py` while the file was being
  edited for `140da1b1`. This is an instrument artifact, not a solver result.
* **`DISCOPT_HULL_LIFTED_RATIO.jsonl`:** the panel that retired the flag (`f9b9e7a5`).
  It is still the reason the flag was retired: ON gave a false certificate on
  clay0203, and that instance has no origin-free row. **Its hull rows for hda,
  gdp_col and kaibel, in both arms, measured a defective lowering, so they are not
  results.** See the next section.

## Hull rows on the origin instances (hda, gdp_col, kaibel): retracted

`_origin_free_hull_row` (`e9b10ed0`, branch-only) built every `t` of a row before
registering any of them. `Variable.__init__` takes its column from
`len(model._variables)` at construction time, so every `t` of one row got the same
column, and each expression over the second `t` read the first `t`'s value and box.

How it was found, on GDPlib `hda`:
* The hull solve returned `infeasible` at 1 node, while Pyomo LOA, RIC and
  enumerate reach 5965.856 (maximize). With LOA's configuration fixed, the hull
  solve certified 5965.855.
* The root McCormick LP was infeasible at the full lifted point, which is feasible
  to 1.1e-8 and inside the root box.
* A deletion filter reduced the infeasibility to the single row
  `_hull_ratio0_..._rctt[1]: v - y*t == 0`. Its bilinear term was relaxed on
  column 902 (`t0_krct[1]`, box [0.0218, 0.0239]), while `t0_rctt[1]` sits at
  position 903 (box [8.94, 9.78]).
* The lowered hda model had 58 generated `t`s sharing an index.

The same defect certified -0.6 on a 2-variable model whose optimum is -2.6
(`test_1917_hull_origin_free.py::test_two_lifted_columns_in_one_row_solve_to_the_big_m_optimum`).

`064efbf5` fixes the site. It also makes `Model._register_generated_variable` refuse
any variable whose `_index` is not its position, so no pass can repeat the defect
silently. After the fix, hda hull is sound: `time_limit` at 300 s, bound 112572 ≥
5965.856, no incumbent. The post-fix gdp_col and kaibel measurements are in
`docs/dev/gdp-parity-baseline.md`.

The MBIGM_CONDITIONED, DISJUNCT_FBBT and BOUND_PRETRANSFORM panels lower with big-M
or mbigm only, so this defect does not touch them.

## Verdicts (2026-11-01)

Re-derive each with `--report`; all four panels are cert-clean.

| flag | verdict | why |
|---|---|---|
| `DISCOPT_GDP_ENUMERATE_CUTOFF` | graduated (`f9b9e7a5`) | see its audit row |
| `DISCOPT_GDP_BOUND_PRETRANSFORM` | graduated | syngas mbigm 0/3 -> 3/3; wall down on 15 of the 16 cells solved in both arms; SGM 36.15 -> 35.39 s |
| `DISCOPT_MBIGM_CONDITIONED` | retired | no solve gained; SGM 26.88 -> 27.49 s |
| `DISCOPT_GDP_DISJUNCT_FBBT` | retired | one solve gained (syngas 0/3 -> 2/3), slower elsewhere; SGM 34.55 -> 35.73 s |
| `DISCOPT_HULL_LIFTED_RATIO` | retired (`f9b9e7a5`) | not cert-clean |

The retired flags' code is deleted, so re-running their fire phase now finds every
lowering identical; the `.jsonl` files are the record.
