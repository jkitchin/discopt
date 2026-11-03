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

## `batch_processing`: the in-house simplex fails on mbigm's big-M LPs (side finding)

Found while measuring `DISCOPT_MBIGM_LP_COMPONENTS`, but it is about the default
mbigm path, flag OFF. When `gdp.mbigm` lowers `batch_processing`, the in-house
simplex (`get_exact_lp_solver`) returns `numerical` (`SolveStatus.ERROR`) on 448 of
the 450 full big-M LPs. `_compute_big_m_lp` treats a non-OPTIMAL LP as "no LP
bound", which is sound, and so 449 of the 450 M's are the interval M rather than
the LP M. That makes mbigm on this instance close to plain big-M.

With the flag ON, 48 component sub-LPs solve OPTIMAL and 402 are still `numerical`,
for 409 interval M's. Every one of the 48 equals HiGHS's optimum of the full LP
(worst relative 3e-16), so the full LPs are well posed and the failure is the
simplex's.

Reproduce (one process, about 30 s):

```bash
python -u discopt_benchmarks/scripts/issue1917_mbigm_components_lowering.py \
    --phase lower --reps 1 --include batch_processing --out <f>.jsonl
python -u discopt_benchmarks/scripts/issue1917_mbigm_components_lowering.py \
    --phase audit --include batch_processing --out <g>.jsonl
```

Each `lower` row's `lp_status` and `interval_fallback` hold the counts above. The
`audit` row's `audit.checked` / `audit.worst` hold the HiGHS comparison. No issue is
filed yet; the #1917 write-up decides.

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

## `DISCOPT_HULL_CONIC_QUADRATIC` convex-GDP panel (2026-11-01): graduated

Written by `discopt_benchmarks/scripts/issue1917_conic_panel.py`, which states its
method and its pre-registered rule in the header. Re-derive the verdict with
`--report --out <file>`.

* `DISCOPT_HULL_CONIC_QUADRATIC.wide-lowload.jsonl` is the **panel of record**,
  load-gated at `--max-load 8` (load1 4.5-8.0). Its fire and reference rows are
  copied from the ungated file.
* `DISCOPT_HULL_CONIC_QUADRATIC.wide.jsonl` is the same panel run ungated while
  the machine was at load1 9-141. It is kept as recorded. Its walls measure the
  load, and its two "certification lost" rows were starvation: both instances
  certify 3/3 ON in the gated run.
* Corpus: every Pyomo `examples/gdp` model, every GDPlib builder, and the 18
  seeded `benchmarks/convex_gdp_generator.py` instances (`cgdp.<family>_<size>_s<seed>`).
  The flag fires on 29: 11 Pyomo and 18 generated.
* References: SCIP (executable) on Pyomo `gdp.bigm`, 300 s, optimal on all 29.
* Gated result, ON arm: 0 incorrect, 0 crossings, 0 lost, incumbents verified.
  Certified 19 -> 25. SGM wall 39.22 -> 17.36 s (0.443). SGM nodes 68.4 -> 36.7.
  The only incorrect rows belong to OFF: Pyomo clay0303 certified at 28351.42
  against an optimum of 26669.11 (#2088).

## Pre-registered panels of 2026-11-02/03: `DISCOPT_MBIGM_LP_COMPONENTS` graduated, `DISCOPT_SUBNLP_IMPLIED_BOX` retired

Both were run from the `fix/1917-finish` worktree (§8 preflight asserted the tree
and the flag code), one process at a time apart from IMPLIED_BOX's two
prereg-sanctioned panels, every thread env set to 1, load 4-11. Each verdict
script's output is reproduced below unchanged; re-run it on the files to
re-derive.

**`DISCOPT_MBIGM_LP_COMPONENTS` -> GRADUATE** (`issue1917_mbigm_components_verdict.py`):

```
== DISCOPT_MBIGM_LP_COMPONENTS: L 24 instances, A 24 (6796 sub-LPs re-solved), S 24
M's ON vs OFF (first rep both arms lowered): {'equal': 7481, 'smaller': 40, 'larger': 0}
lowering faster (both reps): ['biofuel', 'kaibel']
lowering slower (both reps): []
solved in both reps: {'off': 15, 'on': 15}; incumbents lost: []
node geomean ON/OFF over 15 cells: 1.0
rep0 SGM solve elapsed OFF 33.89 s, ON 31.10 s, ratio 0.918
rep1 SGM solve elapsed OFF 33.99 s, ON 30.95 s, ratio 0.911
cert-clean and net-positive -> GRADUATE
```

The V7 corpus guard (`.corpus_guard.json`) ran 81 checks: 0 `_compute_big_m_lp`
calls on the in-repo corpus, 9 on the `pyomo.eight_process` positive control. Its
first launch died on import (it named a solver helper deleted with
`DISCOPT_CONVEX_ROUTE_GDP_LOWERED`); that run wrote no data, the import was
inlined (`b0d029b9`) and the whole panel was restarted from scratch.

**`DISCOPT_SUBNLP_IMPLIED_BOX` -> RETIRE** (`issue1917_implied_box_verdict.py`, exit 1):

```
== DISCOPT_SUBNLP_IMPLIED_BOX: 122 cells, 192 GDP + 296 corpus solves
feasibility checks on ON incumbents: 280
rep0 SGM wall OFF 15.97 s, ON 15.93 s, ratio 0.998
rep1 SGM wall OFF 15.95 s, ON 15.91 s, ratio 0.998
gained (2): [('gdp', 'gdp_col', 'hull'), ('gdp', 'hda', 'hull')]
lost (1): [('corpus', 'heatexch_gen3', '')]
primal better 0 / worse 2: [] / [('gdp', 'gdp_col', 'big-m'), ('corpus', 'contvar', '')]
node geomean ON/OFF over 85 certified cells: 0.9732729454662137
NOT CERT-CLEAN (K1: stays OFF, root-cause):
  C3 ('gdp', 'spectralog', 'hull') rep0: certified OFF, not ON
```

K1 asks for the C3 violation to be root-caused first. The spectralog hull rows: OFF
rep 0 `optimal` at 119.974 s of the 120 s limit; OFF rep 1, ON rep 0 and ON rep 1
all `feasible` at the limit with the identical bound 12.087709. OFF itself flips
between reps, so it is a time-limit coin flip and not caused by the flag. With C
then holding, the prereg's rule is "C holds and N fails -> retire", and N fails on
three counts: N1 (2 cells gained, bar 3), N2 (heatexch_gen3: OFF incumbent 274378
after 3 nodes, ON none after 31, identical in both reps) and N3 (worse 2 > better
0). Had the C3 violation been flag-caused, K1 would retire it as well, so both
readings give the same outcome.

## Hull follow-ups outside any panel (2026-11-01, `fix/1917-hull-remaining`)

These are single-run measurements, taken on 6d3e2f79 plus this branch, at load
9–12 with every thread env set to 1. Wall times are secondary evidence.

### Objective sense of the two origin instances (checked from the model)

The sense is read from the Pyomo model's active `Objective` and from the converted
discopt model's `_objective.sense`, with no solve.

| instance | sense | hull, 120 s, `DISCOPT_SUBNLP_IMPLIED_BOX=1`: incumbent / bound | OFF | Pyomo hull + SCIP, 300 s (`gdp_baseline_g06.jsonl`) |
|---|---|---|---|---|
| `hda` | **maximize** | 4824.32 / 121140.98 | no incumbent / 121140.98 | 5353.84 / 14308.1 |
| `gdp_col` | **minimize** | 19565.86 / 1000 (25 nodes) | no incumbent / 1000 (3 nodes) | 22355.24 / 17833.2 |

Both bounds are valid but weak; neither one crosses its incumbent. `panel_oracle`
lists both instances as unproven. BARON's open gap on `gdp_col` (9802.14 ..
19548.72) lies above our bound of 1000. On `hda` the ON incumbent is *worse* than
SCIP's, because the objective is maximized. An earlier report said "better"; that
claim is retracted.

### `gdp_col`: pump repair deadline (fixed in c5f98e36) and the residual overrun

Before the fix, a 120 s hull solve returned after 359.0 s (cProfile). Of that,
270.5 s was `_repair_to_feasible` → `pounce.project_to_feasible` (pounce 0.14 takes
no time limit), which ran 10 `solve_qp` calls at 27.05 s each. The repair's callbacks
now raise at the deadline, and pounce calls them between QPs. After the fix the same
solve returned at **132.3 s** (`time_limit`, 3 nodes, bound 1000).

The remaining ~12 s past the limit is **not fixed here**, by decision. The deadline
is checked only between non-interruptible Rust calls, so a call that is already
running finishes first. The candidates are one in-flight `solve_qp` (27 s each on
this model) and the other long Rust calls on the root path: `presolve` took 29.98 s
here, and `fbbt` is another. The post-fix overrun was not attributed to one of these
call by call. A fix would need a deadline argument on the Rust side (pounce's QP,
discopt's presolve and FBBT), not more checks in Python.

### `kaibel`: root setup cost (out of scope for #1917)

hull, 120 s: 3733 vars and 5835 cons after conversion; `time_limit` with 0 nodes, no
bound and no incumbent. The cProfile cumulative times are all root setup, before
B&B:

| call | cumulative |
|---|---|
| `_root_presolve.tighten_root_bounds_with_fbbt` (Rust `fbbt`) | 45.97 s |
| `presolve_pipeline.run_root_presolve` (Rust `presolve`) | 29.85 s |
| `convexity.rules.classify_model` (2 calls) | 20.11 s |
| … of which `certificate.certify_convex` → `interval_ad.interval_hessian` (2153 calls) | 12.58 s |
| `transformations.apply` (2 calls) | 8.07 s |
| `integer_product_reform.reformulate_integer_bilinear` | 5.44 s |

Neither arm of any #1917 flag solves `kaibel`, and Pyomo hull + SCIP reaches only
bound 36000 with no incumbent at 300 s. The cost is in how root FBBT, presolve and
convexity classification scale, not in the hull lowering. It belongs to separate
root-scaling work.
