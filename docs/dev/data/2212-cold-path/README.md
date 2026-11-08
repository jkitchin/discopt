# #2212: the engine's cold path, entry experiments and throughput panel

Scripts (each asserts it loaded this checkout's `discopt`, prints an executed count, §6/§8):

| file | what |
|---|---|
| `entry_experiment.py` → `fixtures.jsonl` | every captured LP fixture in the tree (`crates/discopt-core/src/{bnb,lp/simplex}/testdata/*_lp.json`, 19 LPs) under four starts, checked against HiGHS and independently of it |
| `throughput_panel.py` → `throughput_nl.jsonl`, `throughput_nl.summary.log` | the #1008 question: cold dual from the logical start vs today's cold primal, on root relaxations (`build_uniform_relaxation`, `rlt_lineq` off/on), 3 interleaved reps per LP. `--corpus nl` is the in-repo corpus; `--corpus qplib` is #1008's own QPLIB selection (needs the corpus) |
| `prepare_check_probe.py` → `prepare_check_probe.log` | the corpus solved through `Model.solve`, counting how often the new `PreparedDual::prepare` consistency check fires. **Result: 74 instances, 28,123 accepted warm starts, 0 rejections**, so no engine-internal caller produces an inconsistent start and the check is bound-neutral on the default path |

```
DISCOPT_PROFILE=1 python -u docs/dev/data/2212-cold-path/entry_experiment.py
python -u docs/dev/data/2212-cold-path/throughput_panel.py --corpus nl --out nl.jsonl
python -u docs/dev/data/2212-cold-path/throughput_panel.py --corpus qplib \
    --qplib-dir ~/Dropbox/projects/discopt-minlp-benchmark/qplib --out qplib.jsonl
```

## Retraction (§11): the first version of this README was measured with the wrong basis constants

`1865672` published the entry experiment with `BASIC = 2, AT_UPPER = 1` typed into the script.
The engine's values (`crate::lp::basis`) are `BASIC = 1, AT_UPPER = 2`. Every
logical start it built marked the logicals at-upper and the at-upper structurals basic. The
binding then forced the listed basics to BASIC without clearing the extras, so the engine
received starts with more "basic" columns than rows. Its `logical` and `augmented` columns,
and the "16 of 18" built on them, measured corrupted starts and are withdrawn. The scripts now
read the constants from `basis.rs` and assert them. The verdict checks were unaffected (no
wrong answer was published either way), and the corrected numbers below are better than the
withdrawn ones.

The same mistake found a real engine defect, fixed in this change. `PreparedDual::prepare`
accepted a start whose `col_status` and `basic_vars` disagree. A column marked basic but
unlisted had no slot, and `assemble` indexed `xb` with `-1`: a **panic out of
`solve_lp_warm_csc_py`**, a public binding, on `ex1221`'s root relaxation. `prepare` now
refuses such a start (cold fallback, counter `DualPrepRejectInconsistent`). The regression test
is `inconsistent_start_basis_is_refused_not_run_2212`, on the captured start
(`testdata/ex1221_inconsistent_start_lp.json`). It fails before the fix with the panic and
passes after.

## The oracle (three instrument findings; read before trusting a HiGHS comparison here)

1. **HiGHS drops matrix entries below `small_matrix_value` (default 1e-9).** On `ex1252a` it
   stored 582 of 593 nonzeros and called the LP infeasible. With the entries kept it agrees
   with the engine, whose point satisfies the full matrix to an L1 residual of 3.6e-7. The
   oracle runs at 1e-12, HiGHS's floor. Three fixtures still have smaller entries (down to
   5.6e-17) and are marked `oracle_exact: false`.
2. **HiGHS presolve moves `nvs01`'s optimum** (9.1004 with, 9.0833 without, = the engine's).
   The oracle runs presolve off, at 1e-10 feasibility tolerances.
3. **HiGHS rejects a model with a coefficient above `large_matrix_value` (1e15)** instead of
   solving it (`nvs05`'s relaxation has 1.4e15); the panel raises it to 1e20.

`cstr_bigm_dual_rebuild` is infeasible to HiGHS at 1e-10 and optimal at its defaults, so its
answer depends on the tolerance. `nvs05`'s relaxation has right-hand sides up to 2.9e17. Both
the engine's point (objective 0.99146) and HiGHS's (0.99097) are feasible to double precision:
HiGHS misses a row by 32 absolute, 1e-16 relative. The 5e-4 objective gap is inside that
rounding, and the engine's own Neumaier–Shcherbina evaluation declines to certify, as it should.
The panel's two "WRONG" flags are this LP. No false verdict was established anywhere.

## Entry experiment (19 fixtures, 64 comparisons, no wrong verdict)

| fixture | HiGHS | warm (inherited) | cold (today) | logical | augmented |
|---|---|---|---|---|---|
| nous1 (#2202, optimal) | optimal | numerical | numerical | **optimal** | **optimal** |
| nous1 (#2202, infeasible) | infeasible | numerical | numerical | **infeasible** | **infeasible** |
| relu_anderson_cuts_numerical (#2163) | optimal | numerical | numerical | refused | **optimal** |
| relu_anderson_cuts_false_optimal (#2163) | optimal | numerical | numerical | refused | **optimal** |
| qplib2170_cold_fail | optimal | optimal | numerical | **optimal** | refused |
| bchoco06_unstable_pivot | optimal | optimal | numerical | **optimal** | numerical |
| cstr_bigm_dual_rebuild (#2142) | tolerance-dependent | numerical | numerical | numerical | numerical |
| 12 others | — | decided where run | decided | decided where run | decided where run |

`logical` uses the fixture's own unit columns (the #2202 kernel retry's start); `augmented`
appends `-e_i` per row fixed at `[b_i, b_i]` (#2163's `ScaledProblem::logical_basis_dual`).
"refused": no full logical basis from the fixture's columns, or a costed column whose selected
bound is open.

**Q1, inherited start vs cold fallback.** On both nous1 LPs the inherited basis is bad: singular
(κ₁ 1.5e21, refused at prepare), or accepted at κ₁ 1.3e13 and handed to the cold primal four
times. Either way the cold primal then fails with `Phase1EndBoxViolated` +
`Phase1DriftDominates` + `LpAuditBoundsFail` (#956 T2′). The `cold` arm alone fails the same
way. The fallback is what needs fixing.

**Q2, a logical-basis dual as the cold path.** Augmented, then the fixture's own logical basis,
then the primal decides **18 of 19, everything except `cstr`**. Today's cold path decides 12.
Neither logical form covers the other: `qplib2170` needs the fixture's columns because a costed
column's bound is open, and the ReLU LPs need the augmentation.

## Throughput panel, in-repo corpus (148 root relaxations, 3 reps, load ~1)

`throughput_nl.summary.log`, recomputed from the rows with the corrected cascade accounting
(the dual arm already contains the cold-primal fallback inside `solve_lp_warm_csc`, so it is
the cascade; the first summary counted `hda` twice):

| | cold primal (today) | cold dual from logical start (cascade) |
|---|---|---|
| decided, agreeing with HiGHS | 124 / 148 | **126 / 148** (gains `nvs09` ×2, loses none) |
| wall on the 92 LPs both decide | 0.489 s | **0.339 s (−31 %)** |
| per-LP wall ratio, both decide | — | geomean 0.765, median 0.755, max 6.64 (0.2 → 1.1 ms on `st_testgr3`) |
| LPs over 10 ms (8) | — | faster on all 8, ratio 0.55–0.92 |
| pivots, both decide | — | geomean 0.79× |

Rep-to-rep relative sd: median 2 %, so the direction is not noise (§9). `rlt0`/`rlt1` pairs are
near-identical on this corpus, so these are effectively ~74 distinct LPs.

**No max_iter grind here, and no regression.** The history that kept `cold_dual_start` OFF was
the opposite (primal ground on QPLIB_1157's equality-rich RLT relaxation; dual ~25× faster). This
corpus's LPs are small (largest `hda`, m ≈ 3000; neither arm decides it). #1008's class
(m 3000–8000, QPLIB RLT) needs `--corpus qplib`, which this container cannot run.

**The coverage gap is the logical start, not the dual.** The start is refused on **52 of 148**
LPs, those where a costed column has no bound on the side its cost wants (or is a free
zero-cost column). That is where 18 of the primal's 22 undecided LPs sit (`bchoco06-08`,
`contvar`, `tanksize`, `wallfix`, ...), and where the cascade adds nothing. Both existing
implementations (the kernel's `cold_dual_start_basis`, #2163's stage) refuse the same way.
Reaching them needs a start for open costed columns: a big-M box, or HiGHS's free-column
handling.

## Throughput panel, QPLIB (#1008's class: 26 root relaxations, 3 reps, load ~3.5)

`throughput_qplib.jsonl`, `throughput_qplib.summary.log`: `--corpus qplib` at `8108333`, the
script's own selection (usable oracle, continuous, nonconvex, `nvars <= 400`, top 12 by RLT lift,
plus QPLIB_1157), `rlt_lineq` off/on. Measured on macOS (14 cores) beside another session's pytest
run: load 3.0 at start and 4.0 at end, with both arms under the same contention because they are
interleaved. `1f73343` was measured too, and its status, objective, pivots and verdict are
identical on every LP. Its wall times are within 4 % (primal 155.2 s, cascade 115.1 s).

| | cold primal (today) | cascade (dual from logical start, else primal) |
|---|---|---|
| decided, agreeing with HiGHS | 19 / 26 | **23 / 26** (gains 4, loses none) |
| contradicting HiGHS | 0 | 0 (50 executed checks) |
| total wall, all 26 LPs | 160.9 s | **117.0 s (−27 %)** |
| per-LP wall ratio, both decide (19) | — | geomean **0.374**, median 0.476, min 0.084, max 1.43 |
| pivots, both decide | — | geomean 0.375× |

**The four gains are all RLT relaxations where the primal fails:** `QPLIB_1535_rlt1` and
`QPLIB_1703_rlt1` (primal `Numerical` after ~25 s, dual optimal in 10.1 s and 8.1 s),
`QPLIB_1619_rlt1` (primal hits `max_iter` at 22.5 s, dual 2.2 s) and `QPLIB_1157_rlt1`, which is
#1008's own case (primal hits `max_iter` at 11.1 s, dual 0.74 s). The other large wins are
`rlt0`/`rlt1` pairs that both arms decide: `QPLIB_1967` 6.0 → 0.51 s, `QPLIB_1886` 1.05 → 0.11 s.

**What it costs:**
- `QPLIB_1745_rlt0` is the only LP both decide where the dual is slower, 1.43× (0.51 → 0.72 s,
  2,327 → 3,371 pivots).
- `QPLIB_1451_rlt1` is decided by neither arm, and the cascade pays for both: the dual reaches
  `max_iter` and then the primal does too (39.8 s → 83.4 s). That is the "a failing LP pays the
  attempt twice" cost from the audit row, measured.
- `QPLIB_8784` (×2) is refused because it has an open costed bound. The primal returns
  `unbounded` in 0 pivots, which agrees with HiGHS, but the panel counts only optimal and
  infeasible verdicts as decided.

**This is the net-positive LP-level evidence the in-repo corpus lacked.** It is root-relaxation
throughput, not the §5 whole-solve panel: the three corpus panels above still show certified
59 → 59. Graduation needs a certificate, node or wall gain on whole solves that include this class.

## What is still open for #2212

- `cstr` (tolerance-dependent; nothing decides it).
- ~~The QPLIB throughput run~~: done (section above), 23 vs 19 decided, −27 % wall.
- A §5 whole-solve panel over QPLIB instances of this class (the corpus panels never reach it).
- A start for the 52 refused LPs.
- The 40-LP nous1 capture (§121) was not regenerated.

## §5 corpus panel for `DISCOPT_LP_LOGICAL_DUAL_COLD` (`panel/`)

`issue2115_graduation_panel.py --flag DISCOPT_LP_LOGICAL_DUAL_COLD --deterministic --max-nodes 2000
--time-limit 120 --assert-unfired-identical --jobs 2`, 74 in-repo instances, each arm in its own
process, on `c9d6e1c`'s release build.

| | OFF | ON |
|---|---|---|
| violations (cert-clean) | — | **0** of 774 checks; 21 unfired instances bit-identical (3 skipped at the wall) |
| certified | 59 | 59 |
| nodes, all instances | 6839 | 6578 (−3.8 %) |
| wall, all instances | 2254.7 s | 2253.1 s |
| `LpVerdictNumerical` | 21,407 | 18,089 (−15 %) |
| kernel `NodeLpSlackRetry` (#2202) | 1 (decided) | **0** |
| cascade attempts / own decided / augmented decided / refused / fell back | — | 12,387 / 6,709 / 183 / 4,960 / 535 |

Largest node changes: `ex8_5_3` 219 → 141 (28 s → 15 s), `nvs02` 301 → 183, `nvs12` 321 → 201;
`tanksize` 461 → 529. One bound looser, `hda`: both arms stop at the 120 s wall, and ON reaches 1
node against 3 (bound −4.66e8 vs −6996). 13 of its 21 attempts are not certified, and each failed
attempt on its ~3000-row LPs is paid twice, by the scaled cold solve and by the #649 unscaled
retry, before the primal runs.

**Verdict: cert-clean, not yet net-positive. The flag stays OFF.** The mechanism works as
measured on the fixtures: fewer `Numerical` LPs, and the kernel's slack retry no longer has
anything to do. But on this corpus it buys no certificate, and on `hda` the cost of a failed
attempt shows up as a looser bound. The next step is to make a failed attempt cheap, then re-run
this panel: one attempt per cold solve instead of one per scaled and unscaled retry, and a pivot
budget for the attempt on large LPs.

### Second panel: a failed attempt made cheap (`30c9b84`, `panel/corpus_cheap_attempt.*`)

Hypothesis: `hda`'s loss is each failed attempt being paid twice (the scaled solve and the #649
unscaled retry), so making a failed attempt cheap restores it. Change: one attempt per
caller-level solve (`cold_scope`) and a pivot budget `min(max_iter, 2(m+n)+1000)` (over 3x the
largest the 121 decided LPs used). Kill criterion, set before the run: decided attempts falling
noticeably below the first panel's 6,892 means the budget is too tight. Same protocol, same corpus.

| | first panel | cheap attempt |
|---|---|---|
| violations | 0 | 0 |
| certified OFF → ON | 59 → 59 | 59 → 59 |
| nodes ON (OFF ≈ 6,845) | 6,578 | 6,604 |
| attempts / decided / refused / fell back | 12,387 / 6,892 / 4,960 / 535 | 9,867 / 6,719 / 2,833 / 315 |
| `LpVerdictNumerical` ON (OFF ≈ 21,450) | 18,089 | 17,136 |
| `hda` | 1 node vs 3, bound −4.66e8 vs −6996 | **unchanged** |

**The hypothesis is falsified.** `hda`'s attempts fell from 21 to 16 and its loss did not move.
Decided attempts fell 2.5 % (6,892 → 6,719), at the edge of the kill criterion. The cause is
not attributed: the budget, or a skipped unscaled repeat that would have decided. The panel
did not record `LogicalDualColdSkippedRepeat` (it was missing from the panel's counter list,
added since), so this run cannot say which.

**What `hda` is** (`panel/hda_300s.*`, root solve with a 300 s limit, both arms). Both end at
**3 nodes with the same bound (−6995.66)**. The flag-on arm runs twice the LPs in the same wall
(392 LP verdicts against 196), decides more of them (44 optimal against 28), and does twice the
phase-1 work (719k pivots against 339k). Neither arm stops at its node cap: a pre-node stage
issues LPs until the time limit, and with more of them decided it keeps doing more. So the
120 s "looser bound" is a wall-budget effect of the flag doing *more* work, not a quality loss,
and no per-attempt cost change could have fixed it.

**Verdict after both panels: cert-clean, not net-positive on the in-repo corpus.** It decides
more LPs (`Numerical` −20 %), makes the kernel's slack retry redundant, and saves nodes on some
instances (`ex8_5_3` −36 %), but certifies nothing the OFF arm does not, and wall is flat. The
flag stays OFF. What could still show a benefit: the QPLIB run (#1008's large equality-rich LPs,
where the cold primal ground to `max_iter`), and covering the LPs whose start is refused.

### Third panel: the cascade offered from `solve_lp_cols_warm` too (`210f400`, `panel/corpus_warm_hook.*`)

The 40-LP nous1 replay (`nous1_40/`, below) showed 13 of 40 undecided LPs never reaching the
cascade: their inherited basis is singular, the dual refuses it, and `solve_lp_cols_warm` runs its
own two-phase fallback directly. The hook there (flag ON only) took the replay from 27/40 to
**39/40** (flag OFF: 0/40; no wrong verdict in either arm). The same panel protocol on that code:

| | OFF | ON |
|---|---|---|
| violations | — | **0** (773 checks; 20 unfired instances bit-identical) |
| certified | 59 | 59 |
| bounds looser / tighter | — | **0 / 0** (`hda` back to 3 nodes and the OFF bound) |
| nodes | 6,839 | 6,607 (−3.4 %) |
| wall | 2,244.1 s | 2,237.7 s |
| `LpVerdictNumerical` | 21,463 | **16,326 (−24 %)** |
| kernel `NodeLpSlackRetry` | 1 | 0 |
| attempts / own / augmented / refused / fell back / repeat skipped | — | 9,769 / 6,121 / 140 / 3,047 / 461 / 4,431 |

**No regression anywhere on the corpus, no certificate gained.**

### #2163's third `ScaledProblem` stage (`stage3_probe.*`)

The four #2163 networks on the Rust MILP engine: flag OFF the stage fires on `scaled_numerical` and
`cuts_b` (once each, deciding); flag ON it fires **0** times, the cascade's augmented start deciding
those LPs first. All four certify the same optimum. With the kernel's slack retry also at 1 → 0,
neither caller-side retry #2212 names has anything left to do on its own cases.

### A false infeasibility certificate in the spatial tree, found by the flag-ON smoke run

`test_monomial_straddle_kernel` failed with the flag ON: the native tree answered `infeasible` for
`s = x**6` with `x` pinned at an endpoint of a straddling box, a feasible problem. The node LPs were
all correct (checked against HiGHS). The tree itself was wrong: emptying its worklist with no
incumbent, it returned `Infeasible` even after closing a region that was only too narrow to split
(`min_box_width`), never proven empty. `s = 1e6` cannot pass the absolute McCormick tolerance as an
incumbent there. With the flag OFF the same happens at 5,000 nodes (in ~260), so this is a
pre-existing default-path defect; the flag only reached the narrow box inside the test's 200-node
cap. Fixed in `4c04c6b` (`Exhausted` with the closed bound unless every region was proven empty),
with a regression test. `Model.solve`'s kernel route already declined a kernel `infeasible`, so no
published `Model.solve` result was affected; `solve_spatial_tree_py` returned it.

### `cstr`: accepted as pinned (owner decision, 2026-11-08)

#2212's "done" list asked for `cstr_bigm_dual_rebuild` to be decided by the engine. Nothing decides
it (inherited start, cold primal, own logical, augmented), and HiGHS's own verdict depends on its
tolerance: infeasible at 1e-10, optimal 0.2397 at its defaults. The owner accepted the LP as
tolerance-dependent and pinned undecided (`cstr_node_lp_stays_undecided_with_the_flag_on_2212` in
`lp/simplex/cold_dual.rs`, and the existing `captured_cstr_bigm_node_lp_stays_undecided_2142`),
rather than a criterion this issue must meet.

## §5 QPLIB whole-solve panel: retirement

The owner ran the whole-solve panel the audit row said was still owed, on this branch's `70ae79b`
(`panel/qplib44.summary.log`; the rows, `qplib44.jsonl`, are with the reviewer). It covered all 44
instances matching the QPLIB filter: usable oracle, continuous, nonconvex, `nvars <= 400`,
lift >= 300. That is the full class `throughput_panel.py --corpus qplib` sampled 13 of. Arms ran in
separate processes, interleaved, on macOS at load 1.5 to 6.6. `qplib.solu` has only `=best=` values,
so the bound checks are against best-known values.

| | OFF | ON |
|---|---|---|
| violations (cert-clean) | — | **0** of 326 checks; fired on 41 of 44 |
| certified | 0 | 0 |
| nodes | 44,048 | 44,105 (+0.1 %) |
| wall | 4,170.4 s | 4,208.0 s (+0.9 %) |
| bounds tighter / looser | — | 14 / 10 (sign test over the 27 that differ: 15 better, 12 worse, p = 0.70) |
| gap to `=best=`, geomean ON/OFF (31 with a bound in both) | — | 0.964 (median 1.000) |
| no bound in either arm by 120 s | 13 | 13 |

**What the instances show:**
- The wins sit where the root LPs got faster and the 120 s wall binds: `QPLIB_1619` 1 → 22 nodes,
  `QPLIB_1423` 169 → 479, `QPLIB_3318` 32 → 105, `QPLIB_1157` 175 → 207.
- The losses sit at the node limit, where wall does not bind: a looser bound at the same 2,000
  nodes on 7 of 13 (`1886`, `1913`, `1922`, `1931`, `6310`, ...). That reads as a different optimal
  root vertex steering branching, not as LP cost.
- Failed attempts cost nothing visible: `QPLIB_2635` made 3,796 attempts and `QPLIB_2967` 2,201,
  all undecided, with nodes and wall unchanged.
- 13 of 44 never leave the root in either arm. The cascade fires and decides there, but the root
  stage is wall-limited for another reason.

**Verdict: RETIRED (owner decision, 2026-11-08).** Four §5 panels: three in-repo corpus panels
(certified 59 → 59) and this one. All are cert-clean; none is net-positive. The flag, `cold_dual.rs`,
its hooks and its counters are deleted. The LP-level evidence stays recorded here: 18/19 fixtures,
nous1 39/40, QPLIB root LPs 23 vs 19 decided at −27 % wall, both caller-side retries subsumed. It
is not wrong; it does not move whole solves. Kept: the fixtures, the scripts (the arm scripts now
refuse `DISCOPT_LP_LOGICAL_DUAL_COLD=1`; reproduce from `70ae79b`), and the two default-path fixes
found along the way.

#2212 stays open for the root cause the panels point at instead:
- the 13 of 44 QPLIB instances whose root stage never completes in 120 s, with or without the flag;
- the 52 of 148 root relaxations where a sign-matched logical start does not exist.
