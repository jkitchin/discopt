# GDP parity baseline: open-solver profile O (#1775 G0.6)

This is the entry experiment for the spec §1.2 gate: the gap table and the per-family
targets. The full 24-instance `gdp_parity` panel was run under 12 arms at 300 s per
run, 288 runs in all. The raw rows are in
[`data/gdp_baseline_g06.jsonl`](data/gdp_baseline_g06.jsonl), and the report is
reproduced with:

```bash
python discopt_benchmarks/scripts/gdp_baseline.py --out docs/dev/data/gdp_baseline_g06.jsonl --report
```

## Setup

* **Panel:** the 17 GDPlib PR58 instances plus the 7 Pyomo GDPopt test models (spec
  §1.2). Each is scored against `gdplib_runner.panel_oracle()`, which has 17
  certified values and 7 marked `unproven`.
* **Pyomo arms:**
  * `gdp.bigm` and `gdp.hull`, each solved by SCIP 10 (pyscipopt);
  * GDPopt LOA, GLOA, LBB, RIC and enumerate, with CBC as the MIP solver, Ipopt
    3.11.9 as the NLP solver and SCIP as the MINLP / global NLP solver.

  The solvers are installed by `scripts/install_open_gdp_solvers.sh`.
* **discopt arms:** `gdp_method` set to `big-m` (the default route), `hull`, `mbigm`,
  `auto` or `loa`. All discopt rows are at `5f8a174c`. Every discopt incumbent is
  also checked in the original Pyomo model under Pyomo's semantics.
* **Provenance.** The existing rows record `discopt.__file__` only, not a commit or a
  host; the `5f8a174c` above is the author's record of the run. From now on each row
  also records `git_head` (`git rev-parse HEAD` in the tree `discopt` was imported
  from, or None with `git_head_error` saying why) and `hostname`.
  * `5f8a174c` predates `cf5ffb21`, which changed the fast-API placeholder objective.
    That change cannot affect these rows. The converted GDPlib models have no
    `_builder_linear_blocks` and carry their real objective expression, not a
    placeholder (checked on med_term_purchasing, disease_model, modprodnet and
    small_batch).
  * An interleaved control, run by the PR's reviewer, solved med_term_purchasing
    big-m, disease_model big-m and modprodnet hull on `5f8a174c` and on `cf5ffb21`
    or later. Node counts and objectives were identical.
* **Scoring** (`gdp_baseline.score`), at the repository's tolerance (abs 1e-6,
  rel 1e-4):
  * `solved` -- claims optimal and agrees with the oracle;
  * `incorrect` -- one of: claims optimal and disagrees, an incumbent better than
    the optimum, an infeasibility claim on a known-feasible instance, or a point
    that violates the Pyomo model;
  * `claimed_unverified` -- claims optimal where the oracle is unproven;
  * "known-feasible" means a certified oracle, or a discopt incumbent that is
    *verified* feasible in the Pyomo model (`gdp_baseline.verified_feasible`). A
    verified incumbent passed at least one check, has no row, count or logic
    violation, and was also checked against variable bounds and integrality
    (within 1e-5, before rounding).
    * The existing rows predate the bound and integrality checks, so none of
      their incumbents counts as verified.
    * That changes no verdict. The panel's only infeasibility claim (GDPopt LBB
      on kaibel) is on an instance with no discopt incumbent.
  * `unsolved` -- anything else, including time-limit kills and exit -9, which is
    recorded as "killed (SIGKILL; presumed OOM)". Rows written before this label
    say "out of memory". The kernel log was not checked for each one.
* **SGM** uses a 1 s shift, and every run that is not solved counts at 300 s.

### Deviations from the spec, stated so the gate run (G6.1) can close them

* **Pyomo arms do not go through GDPlib's `benchmark.py`.** It has GAMS profiles
  only, so `gdp_baseline.py` configures profile O directly.
* **GDPopt's MIP solver is CBC, not HiGHS.** Pyomo's HiGHS interfaces raise on the
  dual suffix GDPopt attaches (`NoDualsError`).
* **`gdpopt.ldsda` was not run.**
* **One repeat, not three.** The gate's 3 repeats and bootstrap CI belong to G6.1.
* **Concurrency.** The first 170 runs were serial. The last 118 ran 3 at a time on 4
  cores (`--jobs 3`, recorded per row as `jobs`), because the cloud container cannot
  hold an unattended 8 h serial job.
  * **Control:** 35 already-measured (instance, arm) pairs were re-run at 3 jobs.
    Status and verdict were identical in all 35. That does **not** show that
    parallelism changes no verdict on memory-heavy instances. All 8 stranded_gas
    SIGKILLs (see Errors) came from the jobs=3 phase. There, `drive()` queued
    instance-major, so the 3 concurrent runs were usually arms of the *same*
    instance, with the same memory footprint. `drive()` now never runs two arms of
    one instance at once. Whether stranded_gas survives serially is unmeasured. Wall times were higher by a factor
    of **1.20** (geometric mean over the 27 pairs ≥ 1 s; sd of the log ratio 0.14;
    range 1.00–1.87).
  * **Effect:** the timings of the parallel rows are inflated by about that much, on
    both sides. Verdicts near the 300 s limit could move. The SGM ratios below mix
    the two regimes, so treat them as ±20 %.
  * No run started above the load gate (max 1-minute load 6.7, against the gate of 16).

## Results

```
arm                  n solved incorr unver   SGM(s) median_solved  total(s)
pyomo-bigm          24     17      0     1    10.60          0.34    2343.8
pyomo-hull          24     17      0     0    10.51          0.34    2401.1
pyomo-loa           24     11      2     3    35.85          1.90    3933.4
pyomo-gloa          24     11      3     0    38.82          1.44    3971.9
pyomo-lbb           24     14      0     0    34.30          4.25    3330.3
pyomo-ric           24     11      2     3    34.63          1.65    3932.3
pyomo-enumerate     24     10      1     2    43.01          0.98    4251.9
discopt-big-m       24     14      0     0    58.12         22.94    3524.9
discopt-hull        24     12      0     0    73.88         31.96    4143.0
discopt-mbigm       24     14      0     0    57.67         22.65    3525.6
discopt-auto        24     14      0     0    58.12         23.29    3528.9
discopt-loa         24      8      0     0    57.57          0.54    4855.2
```

Per run: S solved, X incorrect, ? claimed_unverified, - unsolved, E error.

| instance | oracle | P:bigm | P:hull | P:loa | P:gloa | P:lbb | P:ric | P:enum | d:big-m | d:hull | d:mbigm | d:auto | d:loa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| batch_processing | baron | S | S | - | E | - | - | - | - | - | - | - | - |
| biofuel | unproven | - | - | - | - | - | - | - | - | - | - | - | - |
| cstr | baron+scip | S | S | X | X | S | X | X | - | - | - | - | - |
| disease_model | highs | S | S | S | S | S | S | - | S | S | S | S | S |
| ex1_linan_2023 | baron+scip | S | S | S | S | S | S | S | S | S | S | S | S |
| gdp_col | unproven | - | - | ? | - | - | ? | - | - | E | - | - | - |
| hda | unproven | ? | - | ? | - | - | ? | ? | - | E | - | - | - |
| kaibel | unproven | - | - | - | - | - | - | - | - | E | - | - | - |
| med_term_purchasing | highs | S | S | X | X | - | X | - | S | S | S | S | - |
| methanol | unproven | - | - | ? | - | - | ? | ? | - | - | - | - | - |
| mod_hens | unproven | - | - | - | - | - | - | - | - | - | - | - | - |
| modprodnet | baron+scip | S | S | S | S | S | S | S | S | - | S | S | - |
| positioning | baron+scip | S | S | - | - | S | - | - | S | S | S | S | S |
| small_batch | baron+scip | S | S | S | S | S | S | S | S | S | S | S | S |
| spectralog | baron+scip | S | S | E | E | - | E | - | S | - | S | S | - |
| stranded_gas | unproven | - | - | - | - | - | - | - | - | E | - | - | E |
| syngas | baron+scip | S | S | S | S | S | S | - | - | - | - | - | - |
| pyomo.eight_process | baron | S | S | S | S | S | S | S | S | S | S | S | S |
| pyomo.eight_process_logical | baron | S | S | S | S | S | S | S | S | S | S | S | S |
| pyomo.strip_packing_concrete | highs | S | S | S | S | S | S | S | S | S | S | S | S |
| pyomo.constrained_layout_clay0203 | baron+scip | S | S | - | S | S | - | S | S | S | S | S | - |
| pyomo.ex_633_trespalacios | baron+scip | S | S | S | S | S | S | S | S | S | S | S | - |
| pyomo.nonconvex_hen | baron+scip | S | S | S | X | S | S | S | S | S | S | S | - |
| pyomo.jobshop_small | highs | S | S | S | S | S | S | S | S | S | S | S | S |

### Correctness

**Gate item 1 holds for every discopt family: `incorrect_count = 0`.** All 8 incorrect
runs are GDPopt false certificates, each claiming `optimal` at a value the oracle
refutes:

| instance | arms | claimed | certified |
|---|---|---|---|
| cstr | LOA, GLOA, RIC, enumerate | 3.1302 | 3.0620 |
| med_term_purchasing (max) | LOA | 6781.50 | 6797.54 |
| med_term_purchasing (max) | GLOA, RIC | 6793.46 | 6797.54 |
| pyomo.nonconvex_hen | GLOA | 132290.4 | 114384.8 |

### Errors

* **discopt hull refuses 4 instances** with `HullPerspectiveOriginError`: gdp_col,
  hda, kaibel and stranded_gas. Each has a disjunct body with `log` or a division
  that is not finite at the origin, so the eps-perspective cannot cancel its residual
  there. This is a loud refusal, not a wrong answer, and it is hull's largest
  capability gap on this panel.
  * **Update 2026-10-31 (#1917):** such a row is now lowered by the eps-free lifted
    perspective (`_origin_free_hull_row`: `v - y t = 0`, `t` in the disjunct's own
    box, row `y g(t) <sense> y rhs`), which needs no `g(0)`, whenever `g` has a finite
    interval enclosure on that box; the refusal stands otherwise (e.g. `1/x` or
    `log x` on a box reaching 0). Verified on synthetic `log`/`1/x` disjuncts
    (differential battery, hull = big-M).
  * **Measured 2026-11-01, superseding the "not yet measured" this bullet used to
    carry (§11).** These are single 120 s hull solves on GDPlib master, at host load
    about 15, so the walls are not measurements.
    * stranded_gas still refuses: its `log` rows are unbounded on their own box, as
      predicted.
    * gdp_col and kaibel lower and end at `time_limit`. gdp_col reaches bound 1000
      with no incumbent; kaibel reaches neither a bound nor an incumbent.
    * hda lowers and ends at `time_limit` at 300 s with bound 112572 ≥ 5965.856
      (maximize) and no incumbent.
    * **The first measurement of these instances was wrong.** The origin-free row
      gave every `t` of a row the same column, and hull returned a false
      `infeasible` on hda. Fixed in `064efbf5`; the evidence is in
      `data/1917-flag-panel/README.md`.
    * Status: three of the four refusals became sound lowerings, and none of the
      three solves within the limit yet. That is a performance gap. It is not a
      parity result.
* **discopt LOA on stranded_gas** refuses because a row inside a disjunct has no
  finite lower bound for big-M.
* **Killed (SIGKILL; presumed OOM), exit -9: 18 runs, scored `unsolved`.**
  * discopt, 5 runs: biofuel hull and LOA (serial phase), and stranded_gas big-m,
    mbigm and auto (jobs=3 phase).
  * GDPopt enumerate, 9 runs: biofuel, disease_model, kaibel, med_term_purchasing,
    mod_hens, positioning, spectralog, stranded_gas and syngas.
  * GDPopt LOA, GLOA, LBB and RIC on stranded_gas, 4 runs.
  * All 8 stranded_gas kills (4 GDPopt, enumerate and 3 discopt) ran in the jobs=3
    phase, where arms of the same instance were co-scheduled. They may be
    artifacts of that phase.
* **GDPopt:**
  * LOA, GLOA and RIC raise on spectralog ("unbounded MILP discrete problem");
  * GLOA trips an `AssertionError` in Pyomo's FBBT on batch_processing.

## Gap table

Pyomo-only and discopt-only list the instances solved by one side but not the other.
"Both-solved SGM" is the SGM over the instances both sides solve.

| family | discopt solved / SGM | Pyomo solved / SGM | SGM ratio | Pyomo-only | discopt-only | both-solved SGM d / P |
|---|---|---|---|---|---|---|
| big-m | 14 / 58.1 | 17 / 10.6 | 5.48 | batch_processing, cstr, syngas | -- | 17.5 / 0.80 |
| hull | 12 / 73.9 | 17 / 10.5 | 7.03 | + modprodnet, spectralog | -- | 17.6 / 0.63 |
| loa | 8 / 57.6 | 11 / 35.9 | 1.61 | modprodnet, syngas, ex_633, nonconvex_hen | positioning | 1.30 / 1.89 |
| gloa, lbb, ric, enumerate | no discopt route (G4.2–G4.4) | 11 / 38.8, 14 / 34.3, 11 / 34.6, 10 / 43.0 | -- | | | |
| best-of | 14 / 30.6 | 17 / 9.4 | 3.26 | | | |
| default (`big-m`) vs Pyomo best | 14 / 58.1 | 17 / 9.4 | 6.18 | | | |

What the gap is made of:

* **big-M and hull: the per-node engine, not the formulation.** On the instances both
  sides solve, discopt's SGM is about 22x SCIP's (17.5 s vs 0.80 s), and the three
  Pyomo-only instances are 300 s time-outs for discopt.
  * batch_processing has no incumbent at all.
  * cstr ends with an incumbent of 3.359 against a bound of 0.108; the certified
    optimum is 3.062.
  * This is a dual-side gap (bound strength and node throughput) on reformulated
    MINLPs. GDP structure does not explain it.
  * `mbigm` and `auto` change nothing on this panel: they solve the same 14 at the
    same SGM.
* **hull** adds the 4 refusals above and loses modprodnet and spectralog to the
  time limit.
* **LOA:**
  * on the instances both sides solve, discopt is faster (SGM 1.30 s vs 1.89 s) and
    solves positioning, which Pyomo does not;
  * discopt fails to certify 4 instances that Pyomo LOA solves: 2 nonconvex Pyomo
    models, where Pyomo's LOA is a local method that happens to be right, and
    modprodnet and syngas.

## Per-family targets (gate §1.2, profile O)

A family meets the gate when it has `incorrect = 0`, solved ≥ Pyomo's, and
SGM(discopt) / SGM(Pyomo) ≤ 1.0 (G6.1 adds 3 repeats and the bootstrap CI). No family
meets it today.

| family | target solved | target SGM (s) | today | gap to close |
|---|---|---|---|---|
| big-m (default) | ≥ 17 | ≤ 10.6 | 14 / 58.1 | 3 instances (batch_processing, cstr, syngas); 5.5x SGM |
| hull | ≥ 17 | ≤ 10.5 | 12 / 73.9 | the 4 origin refusals and 5 time-outs; 7.0x SGM |
| loa | ≥ 11 | ≤ 35.9 | 8 / 57.6 | 4 instances; 1.6x SGM |
| gloa | ≥ 11 | ≤ 38.8 | none | the route (G4.3) |
| lbb | ≥ 14 | ≤ 34.3 | none | the route (G4.4) |
| ric | ≥ 11 | ≤ 34.6 | none | the route (G4.2) |
| enumerate | ≥ 10 | ≤ 43.0 | none | the route (G4.2) |

For the G5 entry experiments:

* The big-M and hull gaps sit in the B&B engine's throughput and bound on the
  reformulated model. That is where G5.1 (disjunct-local FBBT/OBBT) and G5.2 (k-way
  branching) have to show a measured effect.
* The hull origin refusals need a hull form that never evaluates the body at 0 for a
  de-selected disjunct. G2.2's perspective variants do not obviously provide one,
  since the eps-perspective still evaluates g at 0 when y = 0, so this is not yet
  scoped to a task. (Superseded 2026-10-31: see the `_origin_free_hull_row` update under Errors.)
* LOA's gap is 4 instances, not speed.

## G4.1 part 2: per-cut big-M on LOA's OA cuts — measured, not adopted (2026-10-31, #1917)

**Hypothesis.** Spec §5.3 asks for OA cuts "big-M'd onto that disjunct's indicator"
(Pyomo `_add_cuts_to_discrete_problem`) instead of linearised through the lowered
row's `M(1 - y)`. For a cut `a x + b y <= c` with `y` binary this is classical
coefficient strengthening: if the cut has slack `d > 0` at `y = 0` over the whole
box, replace `(b, c)` by `(b - d, c - d)`. That is exact at `y in {0, 1}` and tighter
in between. Kill criterion: no fewer master iterations and no better bound on the
instances it fires on.

**Experiment.** `docs/dev/data/1917-loa-cut-strengthening/loa_strengthen.py` patches
`LOA._cuts_at` to strengthen every new cut over the lowered box (outward-rounded).
It runs `gdp_method="loa"` OFF/ON interleaved, at 120 s, on the 7 Pyomo GDPopt
models (the only `gdp_parity` instances available here; see
`1917-flag-panel-preview/README.md`). Main `4517c7b`, default master
(`DEFAULT_MASTER_BACKEND="auto"`, the in-house MILP).

| instance | coefs tightened | OFF it / status | ON it / status |
|---|---|---|---|
| eight_process | 1825 | 3 / optimal | 25 / feasible, **no bound** |
| eight_process_logical | 1475 | 3 / optimal | 25 / feasible, **no bound** |
| constrained_layout_clay0203 | 1764 | 45 / optimal (39 s) | 48 / time limit, **no bound** |
| nonconvex_hen | 672 | 28 / optimal | 28 / optimal (identical) |
| strip_packing_concrete, jobshop_small | 48, 12 | 1 / optimal | identical |
| ex_633_trespalacios | 0 (no convex row) | 5 / optimal | identical |

**Why.**
- The strengthening is valid. `loa_strengthen_check.py` evaluated 876 checks of the
  OFF run's master points against the strengthened cuts, and **0** were cut off.
- With strengthened cuts every master on eight_process ends `iteration_limit`
  (25 of 25), against 3 of 3 `optimal` without them (`loa_master_status.py`). With
  no master bound, LOA walks the configurations blind.
- On the HiGHS master (`milp_solver="highs"`) the two arms are identical (3
  masters, same bound): HiGHS's presolve already does this strengthening.

**Verdict: falsified on this slice.** The change is neutral where the master
presolves and harmful where it does not, so it is not adopted. The other G4.1
part 2 pieces (equality relaxation, slack penalty) give **no** valid LB (spec
§5.3, §5.4), so they can only help where LOA fails to converge. All 7 available
models certify under LOA, so there is no instance to measure them on: blocked
on the full panel.

**Side finding.** The in-house MILP master hits its iteration limit on
coefficient-strengthened OA rows that HiGHS solves at once. That is a
master-robustness defect outside #1917's scope; the probe above reproduces it.

## G5.2: native k-way disjunction branching — entry experiment, killed on the available slice (2026-10-31, #1917)

Plan §7: count big-M B&B branches on selector binaries, then compare node counts
against disjunction-wise branching prototyped in the Python LBB. Kill: median
node-count ratio >= 0.95; then k-way stays an LBB-only route.

Measured on the 5 Pyomo GDPopt models with a non-trivial tree (main `4517c7b`,
120 s). Branch counts come from the instrumentation-only `_R3A_BRANCH_COUNT_SINK`.

| instance | big-M nodes | branches on selectors / all | LBB nodes | ratio |
|---|---|---|---|---|
| eight_process | 33 | 15 / 16 | 59 | 1.79 |
| eight_process_logical | 25 | 12 / 12 | >= 64 (time limit) | >= 2.6 |
| constrained_layout_clay0203 | 135 | 67 / 67 | >= 237 (time limit) | >= 1.8 |
| ex_633_trespalacios | 13 | 1 / 6 | 5 | 0.38 |
| nonconvex_hen | 3 | 1 / 1 | 13 | 4.3 |

**Median ratio about 1.8: killed.** k-way branching stays LBB's.

Two caveats:
- LBB's node relaxation is also weaker: it drops the nonlinear rows of undecided
  disjunctions. So this is the plan's experiment, not a pure test of the branching
  rule.
- Big-M B&B already branches almost only on selectors. For a two-disjunct
  disjunction that *is* a 2-way disjunction branch, so a Rust k-way rule could
  differ only on disjunctions with three or more disjuncts (clay0203's four-way
  non-overlap, positioning).

Re-open only with a Rust prototype measured on such instances.

That run also surfaced a HiGHS false-optimum on the `lp_nlp_bb` master:
`docs/dev/data/1917-highs-false-milp-optimum/README.md`.

## G5.4: advisor extension — closed, nothing to advise on (2026-11-01, #1917)

Plan §7 scopes G5.4 as extending `gdp_advisor` to consider G2.2 and G2.3, with
the kill rule that it graduates only if `auto` beats `big-m` net-positive.
- G2.2 (hull local vars) was retired with no measurement.
- G2.3 (`DISCOPT_MBIGM_CONDITIONED`) was retired by its §5 panel (below).

With both options gone, G5.4 has nothing left to route to and closes.

## §5 panels for the GDP flags (2026-11-01, #1917)

Full `gdp_parity` panel (24 GDPlib instances from git), 3 reps at 120 s, OFF/ON
interleaved, load-gated. Data and method: `data/1917-flag-panel/`. Every panel was
cert-clean: 0 incorrect, 0 dual bounds past an oracle, 0 instances lost.

| flag | cells fired | solved gained | SGM wall OFF -> ON | verdict |
|---|---|---|---|---|
| `DISCOPT_GDP_BOUND_PRETRANSFORM` (G2.4) | 32 (big-m + mbigm) | syngas mbigm 0/3 -> 3/3 | 36.15 -> 35.39 s; wall down on 15 of the 16 cells solved in both arms (modprodnet big-m 8.2 -> 8.3 s the exception) | **graduated** (default ON) |
| `DISCOPT_MBIGM_CONDITIONED` (G2.3) | 21 | none | 26.88 -> 27.49 s; med_term 4.9 -> 10.6 s | **retired** |
| `DISCOPT_GDP_DISJUNCT_FBBT` (G5.1) | 17 | syngas 0/3 -> 2/3 | 34.55 -> 35.73 s; positioning 26.4 -> 39.4 s, med_term 4.5 -> 8.7 s | **retired** (gain on one instance only) |
| `DISCOPT_GDP_ENUMERATE_CUTOFF` (G4.2) | all | see audit row | see audit row | **graduated** (`f9b9e7a5`) |

BOUND_PRETRANSFORM costs nodes on three cells (spectralog 411 -> 651,
nonconvex_hen 3 -> 5, med_term_purchasing mbigm 57 -> 95) while still cutting their
wall; the gain is not confined to one instance. Graduating it surfaced two gaps,
both closed in the same PR and both bound-neutral on the panel (no converted
instance has an array-valued disjunct row or a user `big_m`): array rows are now
pretransformed (otherwise the two #1215 idioms lower differently), and a user's
`big_m` is no longer silently replaced. A third was a false certificate: a block
row whose `Constraint` object an `if_then` also wraps was dropped from both, so the
implication disappeared (smoke caught it; objective 1 against a true 5). It was
latent on `main` under `=1`, and the panel could not see it, since no converted
instance shares a constraint between indicator rows (16,648 references, 0 shared).
Rows are now dropped only as members of their own block disjunct.

Side findings from the panels:
- `small_batch` under `enumerate` returned `unknown`: every configuration NLP was
  refused by POUNCE with `Not_Enough_Degrees_Of_Freedom` (more all-fixed equality
  rows than free columns). `140da1b1` re-solves without the all-fixed rows when they
  all hold; enumerate now certifies 167427.657 (`test_1917_nlp_constant_rows.py`).
- The mbigm lowering exceeds 600 s on `biofuel` and `stranded_gas`, and the big-M
  setup on `stranded_gas` takes about 120 s. Both are open performance gaps.

## cstr big-M: per-node propagation and OBBT entry experiments, both killed (2026-11-02, #1917)

The question was why the native spatial kernel's dual bound on cstr big-M (optimum
3.0620145717261655) climbs so slowly when SCIP certifies the same model in under a
minute. The two candidates were per-node reductions (propagation and OBBT). Both
were measured on the kernel spec that `Model.solve` captures (lifted big-M, replayed
through `_rust.solve_spatial_tree_py`, single thread). Both are rejected.

### Propagation premise: falsified

Pre-registered hypothesis: the kernel misses a box reduction that SCIP's propagation
finds, and that is what keeps the bound low.

* **Not the formulation.** SCIP solved on a pyscipopt model built to equal the
  kernel spec exactly reaches 1.159 at 5000 nodes, and 1.183 on Pyomo's own model.
  With nonlinear propagation off it reaches 0.111. The kernel default reaches
  0.2152 at 5000 nodes.
* **Not propagation.** I captured 94 post-propagation kernel boxes and ran SCIP's
  propagation from the pre-propagation box. SCIP tightened further on only 3 boxes
  (nodes 900, 920, 1460). It cut off 2 boxes, both of which the kernel LP already
  proves infeasible. It never loosened a bound.
* **Not the LP.** On the 81 equal boxes, SCIP's LP bound minus the kernel's has
  median -4e-9 and max 0.018.

So the node-indexed gap is in tree search (branching variable, branching point and
node selection), not in per-node reductions. The per-node nonlinear FBBT item (F3)
was killed by the same measurement.

### OBBT at the gate's wall-clock limit: killed

Pre-registration, copied verbatim from the file written before any run:

```
Pre-registered 2026-11-01 before any run (bigm-gap, #1917 cstr big-M OBBT entry experiment).
Instance: cstr big-M lifted kernel spec (spec_cstr.pkl, the one in replay.py), replayed through
_rust.solve_spatial_tree_py with time_limit_s=120, max_nodes=10**9, bound_time_extension_s=None.
Arms: (a) default; (b) run_obbt=True, every node; (c) run_obbt=True, obbt_every=10
  (THROTTLE RULE, fixed now: sweep at root and at every node whose id % 10 == 0; children inherit
  the tightened box). Chosen over depth<=10 because the kernel tracks no depth; not tuned.
Order: interleaved a,b,c,a,b,c (2 reps), one process at a time; uptime recorded before each run.
Report: bound, status/certified, nodes, LPs at 120 s.
PASS: (b) or (c) reaches bound >= 0.59 (or certifies) in BOTH reps, with bound <= 3.062 (cert-clean).
KILL: no arm beats (a)'s 120 s bound by >= 2x (in both reps).
Caveat fixed now: kernel-only wall; the Model.solve spec build / presolve time is excluded.
```

Setup:

* Build: .so sha `e9d1f536`, which is a8a4be0e plus an experiment-only `obbt_every`
  knob.
* Probe check at 30 nodes: the LP counts were a 83, b 4643, c 700, which shows the
  throttle fired.
* Load average during the runs: 6.8–10.2, from another panel on the machine.

Results at 120 s (none of the six runs certified, all are cert-clean, and every run
found the incumbent 3.0620145717261655):

| arm | rep | bound | nodes | LPs | undecided |
|---|---|---|---|---|---|
| a default | 1 | 1.7756 | 31031 | 30833 | 1152 |
| b OBBT every node | 1 | 0.3698 | 336 | 40331 | 1 |
| c OBBT every 10th | 1 | 0.9724 | 3520 | 37547 | 0 |
| a default | 2 | 1.7801 | 31531 | 31325 | 1175 |
| b OBBT every node | 2 | 0.3698 | 335 | 40178 | 1 |
| c OBBT every 10th | 2 | 0.9579 | 3430 | 36709 | 0 |

**Verdict: KILL.** No OBBT arm comes close to 2x the default. Both are worse than the
default at the same wall clock: full OBBT reaches 0.21x and the throttle 0.54x.

The PASS line was not crossed by either OBBT arm. (The default's own 1.78 is above
0.59, but that is because the 0.59 threshold was written against a node-indexed
baseline, not a 120 s one.)

The node-indexed advantage OBBT showed earlier (1.675 at 5000 nodes, against 0.215
for the default) is real. It costs about 120x more LPs per node, so at a fixed wall
clock the default's raw node throughput wins.

No flag is added. The `obbt_every` knob lived only in the experiment build and is
not committed.

What remains is a node-efficiency gap of roughly 5x against SCIP at 5000 nodes, and
it is the next target (branching variable and point, compared against SCIP on the
same box sequence). The default's 1152–1175 undecided node LPs per 120 s are also
worth a look. How an undecided node contributes to the global bound has not been
measured here.

## G6.1 gate run (2026-11-04)

Data: `docs/dev/data/gdp_g61/`. Score: `python discopt_benchmarks/scripts/gdp_g61_score.py`.
Build: `d032897d` (main) plus this branch's harness changes; Pyomo 6.10.0; SCIP 10.0.2;
24 instances × 14 arms; 300 s limit, 120 s grace; `--jobs 4` on a 14-core Mac mini.

### Result: no family passes all three bars

| family | solved d / p | incorrect d / p | SGM d (s) | SGM p (s) | ratio | 95% CI | per-repeat ratios |
|---|---|---|---|---|---|---|---|
| big-m | 15 / 15 | 0 / 2 | 26.6 | 13.0 | 1.97 | [1.06, 3.53] | 1.98 1.97 1.97 |
| hull | **14 / 16** | 0 / 0 | 35.6 | 10.5 | 3.19 | [1.59, 6.18] | 3.23 3.16 3.18 |
| loa | 13 / 12 | 0 / 2 | 22.6 | 32.3 | **0.71** | [0.38, 1.15] | 0.70 0.71 0.72 |
| gloa | 13 / 11 | 0 / 3 | 23.2 | 30.7 | **0.76** | [0.41, 1.22] | 0.75 0.78 0.77 |
| ric | 12 / 12 | 0 / 2 | 38.6 | 30.9 | 1.24 | [1.02, 1.57] | 1.23 1.24 1.25 |
| enumerate | 10 / 10 | 0 / 1 | 47.8 | 39.0 | 1.22 | [0.97, 1.61] | 1.21 1.22 1.22 |
| lbb | 15 / 15 | 0 / 0 | 45.7 | 31.7 | 1.43 | [0.82, 2.39] | 1.42 1.43 1.43 |

* **Correctness: passes everywhere.** discopt has 0 incorrect and 0 unverified claims in
  every family and every repeat. Pyomo has 10 incorrect answers across five families.
* **Solved count: passes in six families.** hull is short by 2: cstr, batch_processing and
  syngas solve under Pyomo hull and not under discopt (tracked in #2142, #2144, #2143);
  discopt hull solves positioning, which Pyomo does not.
* **SGM ratio: passes in no family.** The spec's bar 3 (#1775 §1.2) is a conjunction:
  ratio ≤ 1.0 **and** the upper end of the bootstrap 95% CI ≤ 1.10. loa (0.71, upper
  1.15) and gloa (0.76, upper 1.22) meet the first clause and fail the second; every
  other family fails the first. The CI is wide because it resamples 24 heterogeneous
  instances, not because the repeats disagree (per-repeat ratios differ by ≤ 0.07), so
  more repeats would not narrow it. loa and gloa are faster at the point estimate,
  not demonstrated faster.
  *Correction (2026-11-05):* this bullet first read "if the bar is the point estimate
  ≤ 1.0, loa and gloa pass the whole gate … the spec does not say which", testing the
  CI's upper end against 1.0. That was wrong: the spec states both clauses and the 1.10
  threshold. The bar is applied as written; it was not re-chosen after the results.
* **big-m and hull lose on small instances.** Pyomo + SCIP solves most of the pyomo.*
  examples in 0.1–0.2 s; discopt takes 2–11 s (eight_process hull 7.6 s vs 0.18 s).
  That points at per-solve fixed cost, not search, and is the next thing to profile.

### How it was run, and deviations from the spec

* **Repeats.** Repeat 1 ran all 336 (instance, arm) pairs. Repeats 2 and 3 re-ran only the
  174 pairs that both sides solved in repeat 1; a run that is not solved counts at 300 s
  in the SGM whatever its wall, so repeating it measures nothing. No verdict changed in the
  re-runs. 93% of a full repeat's time was time-outs (an estimated 11–13 h for three full
  repeats against about 4.5 h for this design).
* **Concurrency.** `--jobs 4`. A serial control (two pairs per family, 28 rows, jobs=1)
  measured jobs=4 walls at 1.11× serial (sd of log 0.08, n 60): discopt 1.09×, Pyomo 1.13×.
  The inflation is symmetric, so it does not move the ratios.
* **Global solver.** Pyomo's `scip_direct` is absent in Pyomo 6.10.0 here, so the bigm/hull
  arms use the SCIP executable (`GDP_BASELINE_GLOBAL=scip`, recorded per row). That adds
  `.nl` write and process start to Pyomo's walls, which if anything favours discopt.
* **Pyomo 6.10 hull** rejects `well_defined_points_heuristic_solver`;
  `gdp_baseline.py` now passes only the options the transformation's CONFIG declares.
* **Pyomo hull errors** on gdp_col (math domain error), hda, kaibel and stranded_gas;
  they count as not solved for Pyomo.
* The driver scripts in the data directory hard-code the session's scratch paths; they
  are kept as the record of what ran.

## Capability matrix: #1775 §1.1, current state (2026-11-06, #2182)

#1775's spec §1.1 lists what Pyomo 6.10.0's GDP offers, with discopt's state on
`c2851b03` (2026-10-18). This is that table re-derived against `main` @ `63910b2`. Each
entry names the code or audit row that holds it. Every *done* row is sound in the §1.1
sense: it is rigorous, or it reports that it is not (no `optimal`, `bound=None`).

| # | Pyomo 6.10.0 | discopt @ `63910b2` | state |
|---|---|---|---|
| A1 | `gdpopt.loa` | `gdp_method="loa"` (`gdp/algorithms/loa.py`); D1 fixed (#1773); set-covering and other inits (G3.3); global subproblems by default. Equality relaxation and slack penalty not built (see "G4.1 part 2" below). | done, minus two options |
| A2 | `gdpopt.gloa` | `gdp_method="gloa"` (`gloa.py`, #1880) | done |
| A3 | `gdpopt.lbb` (+ `check_sat`) | `gdp_method="lbb"` (`lbb.py`, #1897); `check_sat` is native logic propagation (`gdp/propagate.py`, #1890), no z3 | done |
| A4 | `gdpopt.ric` | `gdp_method="ric"` (`ric.py`, #1887) | done |
| A5 | `gdpopt.enumerate` | `gdp_method="enumerate"` (`enumerate.py`, #1887); refuses above `max_configurations` rather than truncating; incumbent cutoff default ON (`DISCOPT_GDP_ENUMERATE_CUTOFF`) | done |
| A6 | `gdpopt.ldsda` | `gdp_method="ldsda"` (`ldsda.py`, #1900) | done |
| A7 | init `set_covering`, `max_binary`, `fix_disjuncts`, `custom_disjuncts`, `no_init` | `gdp/algorithms/init.py` (#1865), all five | done |
| T1 | `gdp.bigm` with a user `bigM` | `either_or`/`disjunction(big_m=...)`, LP-validated (`gdp/user_big_m.py`, #1851) | done |
| T2 | `gdp.hull`: nesting, `LocalVars`, 3 perspectives, EPS | nesting done (#1821); `Disjunct.local_vars` exists (G1.2); hull's use of local vars and the three perspective variants (G2.2) **not built**, skipped by owner decision 2026-10-20 | partial (owner skip) |
| T3 | `gdp.mbigm` conditioned on each disjunct | `gdp.mbigm` exists; the conditioned form (`DISCOPT_MBIGM_CONDITIONED`) was **retired** by its §5 panel (cert-clean, not net-positive; audit row). `DISCOPT_MBIGM_LP_COMPONENTS` graduated. | capability retired on measurement |
| T4 | `gdp.cuttingplane` | `gdp.cuttingplane` (#1853), linear / certified-convex disjuncts only | done |
| T5 | `gdp.partition_disjuncts` + `gdp.between_steps` | both registered (#1857) | done |
| T6 | `gdp.bound_pretransformation` | `gdp.bound_pretransformation` (#1837); automatic use graduated default ON (`DISCOPT_GDP_BOUND_PRETRANSFORM`) | done |
| T7 | `gdp.binary_multiplication` | registered (#1838), OR and nesting included | done |
| T8 | `gdp.fix_disjuncts`, `gdp.transform_current_disjunctive_state` | both registered (G1.3, `gdp/state.py`) | done |
| T9 | `basic_step` | `gdp.basic_step` (#1860) | done |
| T10 | `gdp.bilinear` | `gdp.bilinear` (#1860) | done |
| T11 | `compute_disj_var_bounds` (per-disjunct FBBT/OBBT) | `DISCOPT_GDP_DISJUNCT_FBBT` **retired** by its §5 panel (audit row); G5.1 otherwise unbuilt | capability retired on measurement |
| M1 | OR (at-least-one) disjunctions | `SelectorCardinality.AT_LEAST_ONE` for big-M and binary multiplication (#1795) | done |
| M2 | nested disjunctions in `Disjunct` | #1789 | done |
| M3 | disjunct-local variables | `Disjunct.local_vars` (#1789) | done |
| M4 | logic over every indicator | eager indicators, `logical()` over them (#1782) | done |
| M5 | factorable `logical_to_disjunctive` | **killed** at its entry experiment (G2.10: max clause growth 3.0x < 10x); CNF + one-way Tseitin stays | not built, on measurement |
| I1 | Pyomo GDP model in, structure intact | `discopt.pyomo.gdp.from_pyomo_gdp` / `convert_pyomo_gdp` (#1800) | done |

So 19 of the 23 rows are done, A1 among them minus two options whose fate is recorded
below. Of the other four, T2's local-variable and perspective half is an owner skip, T3
and T11 shipped and were retired on their panels, and M5 was killed by its entry
experiment.

## #2182 task 2: the big-M / hull per-solve fixed cost (2026-11-06)

Data and probes: `docs/dev/data/2182-root-ski/`. Host: a 4-core Linux VM, main `63910b2`,
Python 3.12, Pyomo 6.10.1, highspy 1.15.1. The G6.1 gate ran on three cores throughout, so
every probe below was pinned to the fourth. No timing here is comparable to the Mac
numbers above; only the within-run ratios are.

### Where the time goes: the issue's premise holds for big-M, only partly for hull

`root_split.jsonl` splits each solve of the 10 smallest both-solved instances at
discopt's own root snapshot (`SolveResult.root_time`):

* **big-M:** the root is 79-90 % of the solve on small_batch, ex1_linan_2023,
  ex_633_trespalacios and nonconvex_hen (3-13 nodes), and 50-64 % on eight_process,
  eight_process_logical and modprodnet. That is per-solve fixed cost, as the issue says.
* **hull:** 87-88 % on small_batch and ex1_linan_2023, but 30-65 % on the pyomo.* models.
  Hull trees run 2-4x the big-M node counts there (eight_process 59 vs 27, ex_633 45 vs
  13), so a good part of hull's gap is search.

A cProfile of eight_process hull (`probes/phase_probe.py`, `probes/children.py`) shows no
single setup phase. The solve divides among primal heuristics 23 %, node FBBT 19 % (mostly
the Python nonlinear tightener, 269 calls over 59 nodes), node LPs 17.5 %, root OBBT 15 %
(1246 LP probes on 44 variables) and the GDP-config constructor 6 %. On the smallest
instance (ex_633 big-M, 6 variables) **one** component dominates: the GDP-config
constructor, 1.53 s of a 2.64 s solve.

### H1: the constructor wastes its grant after its last improvement. Killed

`probes/cfg_probe.py` wraps `one_hot_config_subnlp`, timestamping each feasible point and
recording each sub-NLP's integer assignment (`cfg_probe.jsonl`). On ex_633 big-M the wave
found the final optimum at 0.022 s; the dive then spent 1.45 s on 53 more sub-NLPs, every
one feasible and none better, over the model's **4** configurations, so the dive covered
the space about 14 times. Two fixes follow, and the class measurement kills both:

* **Stop the dive once every configuration has been seen.** Kill criterion, stated before
  the class run: fires with ≥ 0.5 s left on fewer than 5 of the 15 instances. It fires on
  **1**. The others have 27 to 3.4e7 configurations and cover 8-49 of them in the grant.
* **Stop on stagnation.** The final point came late on four instances: 1.503 s of 1.5
  (eight_process_logical big-M), 1.34 s (ex1_linan_2023 hull), 1.38 s (clay0203 big-M)
  and 1.1 s (positioning). Any stop rule short of the grant loses those.

A memo of repeated configurations alone saves no wall: the dive is deadline-bound, so the
time would go to more restarts.

### H2: the ski-rental rule for the root share. Entry passes

`DISCOPT_GDP_CONFIG_DEFER` already holds back the rest of the grant until the search has
run as long as that rest. H2 applies the same rule to the root share, so the share is
`min(1.5 s, time the solve spent before the constructor)`. Kill criterion, stated
before the run: any change in a certified objective or a lost certificate, an SGM gain
< 5 %, or a median node ratio > 1.10.

Entry A/B (`entry.jsonl`, `entry.summary.txt`): the 15 instances × {big-M, hull} × 3
reps, OFF/ON interleaved in fresh processes on one pinned core, the ON arm patching
`_gdp_config_root_until` in-process.

* 90 pairs. 0 certified-objective changes, 0 certifications lost or gained. Median node
  ratio 1.000; positioning alone changes its tree (307 -> 323 nodes) and still gets
  faster.
* SGM (1 s shift) **10.14 -> 8.99 s (-11.4 %)**: small_batch 0.37/0.46x, ex_633 big-M
  0.44x, nonconvex_hen big-M 0.55x, ex1_linan_2023 0.56/0.59x, eight_process big-M 0.74x,
  clay0203 hull 0.89x, positioning 0.84x. Cells whose pre-constructor root exceeds 1.5 s
  keep the full share and are unchanged within noise (0.97-1.05x).
* While the A/B ran, the regression test briefly used its core for ~11 s. The interleaved
  arm order keeps that roughly symmetric.

It ships as `DISCOPT_GDP_CONFIG_ROOT_SKI`. It was added default-OFF with its §5 panel
pre-registered (`prereg.md` and `verdict.py`, committed before the first panel row).
**The panel passed, and the flag is graduated (default ON, `=0` opts out).**
`verdict.txt` has the run: 192 GDP and 296 corpus solves, 0 C violations. GDP SGM
ON/OFF was 0.799 and 0.818 per rep over the 24 cells both arms certify. Node ratio
1.004 (GDP) and 1.001 (corpus); corpus SGM 1.002 / 1.010.

This does not close the big-M/hull gap. At best it takes ~1.4 s off a 2-4 s solve that
Pyomo + SCIP finishes in 0.1-0.2 s. The remainder is distributed across root OBBT, the
Python node FBBT and the root heuristics, with no single owner, and hull adds search.

### Side finding: med_term_purchasing big-M loses its certificate with HiGHS ≥ 1.13

In every run here, med_term_purchasing big-M ended `feasible`, uncertified, at 41 nodes
and ~8 s: incumbent 6797.5397 (the optimum), bound 8359.0. In #2149 it was solved in
3.06 s. The cause is the HiGHS version, not a discopt commit (`medterm_highs_version.txt`).
With highspy 1.11.0 the same build certifies (bound 6798.149). With 1.13.0, 1.13.1, 1.14.0
and 1.15.1 the route declines. Its fixed-integer LP check puts the NS safe bound 0.00153
below the objective, against a 7.8e-6 threshold, so the reported bound falls back to the
root NS bound. Restoring `lp_milp_highs.py` from `1f16c936`, `aa12eb4~1`, `aa12eb4` or
`f0de847`, or setting `DISCOPT_MILP_REFUTED_TIGHT_RESOLVE=0`, changes nothing. The decline
is sound (it refuses, never mis-certifies) but costs a certificate on the default install
(`highspy>=1.11` resolves to 1.15.1). It is a lead for the LP/MILP route's certificate,
outside #2182.

**Fixed by #2183 (2026-11-06).** The decline was the check treating a merely-feasible LP
as having no verdict. Its NS bound (-6797.541) already sat on the safe side of the MILP
bound (-6798.199), so no point on that integer assignment could refute it. The route now
accepts that verdict. With highspy 1.15.1, med_term big-M certifies again: obj 6797.5397,
bound 6798.1986. See the 2026-11-06 entry in `lp-milp-highs-routing-plan.md`.

## #2182 task 3: the leftovers, closed or retired (2026-11-06)

Every measurement here is from the #2182 G6.1 re-run's repeat 1 (`docs/dev/data/gdp_g61_2182/rep1.jsonl`,
main `63910b2`, 300 s, all 24 instances × 14 arms; see the G6.1 section below for the setup).

* **G4.1 part 2 (LOA equality relaxation, slack penalty): retired, not built.** Per spec
  §5.3 / §5.4 a slack-penalized master gives no valid LB (`bound=None`), so neither option
  can turn an unsolved instance into a certified one. Both §1.2 bars that LOA could miss
  count certified solves: bar 2 directly, bar 3 because an unsolved run scores at the 300 s
  limit. Measured: discopt-LOA leaves 11 of 24 unsolved (batch_processing, biofuel, cstr,
  gdp_col, hda, kaibel, med_term_purchasing, methanol, mod_hens, spectralog, stranded_gas).
  6 already have an incumbent. The other 5 are all the options could change, and only by
  adding an incumbent, which no bar counts. LOA's bar-2 standing is 13 vs Pyomo's 11
  without them. What would reopen this: a gate that scores primal quality on unsolved
  instances.
* **G4.2 (RIC + enumerate): re-measured, done-when still not met.** The done-when (spec
  §9) is "certify every panel instance that has ≤ 10⁴ configurations within the budget."
  With `DISCOPT_GDP_ENUMERATE_CUTOFF` graduated, enumerate refuses 6 instances above 10⁴
  configurations (biofuel, disease_model, med_term_purchasing, positioning, spectralog,
  stranded_gas). Of the 18 eligible it certifies **9 within 60 s** and 10 within 300 s
  (clay0203 at 128 s), 0 incorrect. Against the old "7 of 14 eligible within 60 s", that
  is more instances, and the same half. The 8 eligible instances it does not certify
  (batch_processing, cstr, gdp_col, hda, kaibel, methanol, mod_hens, syngas) end at the
  limit with open configuration subproblems. RIC certifies 11 of 24. Both match Pyomo's
  counts on this run (enumerate 10 / 10, RIC 11 / 11), so G4.2's remaining gap is the
  global-subproblem cost on the large GDPlib models, the same as big-M's. It is not an
  enumeration defect.
* **`stranded_gas` under hull: no longer refuses; no incumbent in 300 s.** discopt hull
  ends `time_limit` at 304 s with no incumbent and no bound, and so does discopt big-M
  (306 s). Pyomo hull, which errored in #2149, now reaches its time limit (373 s) with
  bound -99072.4 and no incumbent. No arm of either tool solves it.
* **The §1.1 capability matrix:** done, in the section above.
* **G5.4 (advisor extension): retired.** Already closed in writing on 2026-11-01 (the G5.4
  section above): both options it would have routed to are gone, `DISCOPT_MBIGM_CONDITIONED`
  retired by its panel and G2.2 skipped by the owner.

## #2182 task 1: G6.1 re-run on `main` (2026-11-06)

Data: `docs/dev/data/gdp_g61_2182/`. Score: `python discopt_benchmarks/scripts/gdp_g61_score.py
docs/dev/data/gdp_g61_2182`. Build `63910b2` (main) in a worktree, every row's `discopt` path
asserted inside it. Same design and scorer as #2149: repeat 1 runs all 24 instances × 14 arms;
repeats 2-3 re-run the 172 pairs that both sides solved; 300 s limit, 120 s grace. Host: a 4-core
Linux VM, Python 3.12, Pyomo 6.10.1, SCIP 10.2.0 (`scip_direct` and the executable), Ipopt
3.11.9, CBC, highspy 1.15.1. Cores 0-2 ran the gate (`--jobs 3`, `taskset`); core 3 ran the
#2182 task-2 experiments.

### Result: no family passes all three bars

| family | solved d / p | incorrect d / p | SGM d (s) | SGM p (s) | ratio | 95% CI | per-repeat ratios |
|---|---|---|---|---|---|---|---|
| big-m | 15 / **17** | 0 / 0 | 38.58 | 10.59 | 3.42 | [2.15, 5.81] | 3.44 3.42 3.39 |
| hull | 15 / **17** | 0 / 0 | 47.09 | 10.88 | 4.05 | [2.38, 7.17] | 3.97 4.10 4.08 |
| loa | 13 / 11 | 0 / 2 | 35.21 | 37.44 | **0.94** | [0.51, 1.54] | 0.90 0.97 0.96 |
| gloa | 13 / 11 | 0 / 3 | 34.57 | 40.00 | **0.87** | [0.47, 1.46] | 0.87 0.85 0.88 |
| ric | 11 / 11 | 0 / 2 | 59.38 | 35.32 | 1.66 | [1.25, 2.31] | 1.67 1.66 1.66 |
| enumerate | 10 / 10 | 0 / 1 | 65.37 | 42.92 | 1.51 | [1.10, 2.18] | 1.51 1.51 1.52 |
| lbb | 13 / **14** | 0 / 0 | 69.41 | 35.55 | 1.93 | [1.40, 2.74] | 1.94 1.93 1.91 |

* **Bar 1 passes everywhere.** discopt has 0 incorrect and 0 unverified claims, in every
  family and repeat. Pyomo has 8 incorrect answers.
* **Bar 2 fails for big-m, hull and lbb.**
  * big-m: discopt gained cstr (#2142) and lost med_term_purchasing. Pyomo gained cstr and
    eight_process_logical, which it scored `incorrect` in #2149.
  * hull: discopt gained syngas (#2143). Pyomo now also solves positioning and spectralog.
  * lbb: discopt lost med_term_purchasing and spectralog; spectralog is near the limit on this host.
  * The three instances #2182 named (cstr #2142, syngas #2143, batch_processing #2144) did
    close the hull gap discopt had, but Pyomo's gains here outran them.
* **Bar 3 fails everywhere.** loa and gloa pass the point estimate and fail the CI clause
  (upper 1.54, 1.46 > 1.10), as in #2149. The per-repeat ratios differ by ≤ 0.07, so the
  width of the CI comes from instance heterogeneity, not from noise.

### Why the ratios are worse than #2149's, and what the cause is not

Every ratio is above #2149's (big-m 1.97 → 3.42, loa 0.71 → 0.94). Three controls separate
the causes.

* **Not a discopt regression.** `old_vs_new_same_host.jsonl` runs `1f16c936` (the #2149
  build, rebuilt here) and `main` on the same host, interleaved, one run per pinned core,
  over the 90 (family, instance) cells `main` solved. On the 86 cells both solved, the
  geometric-mean wall new/old is **0.95** (sd of log 0.27): hull 0.80, lbb 0.83, big-m 0.89,
  loa 1.05, gloa 1.04, ric 1.08, enumerate 1.07. The three status changes are all gains:
  cstr under big-M and LBB, syngas under hull.
* **The concurrency penalty is asymmetric here.** The jobs=1 serial control (`serial.jsonl`,
  78 runs) puts `--jobs 3` walls at 1.63x serial for discopt and 1.20x for Pyomo. discopt's
  workers are multithreaded (about 1.1-1.3 cores each), and three of them on three cores
  oversubscribe; Pyomo's subsolvers are single-threaded. #2149's 14-core host had room, and
  measured 1.09x / 1.13x. That alone inflates every ratio here by about 1.36x. Big-m's
  corrected ratio is about 2.5, still above 1.0.
* **Pyomo's arms are stronger.** The harness default `scip_direct` is available here; #2149
  ran the SCIP executable because its Pyomo lacked it. `control_pyomo_scipexe.jsonl` re-runs
  Pyomo big-m and hull on the executable. It solves 16 / 15 (vs 17 / 17 direct). So **bar 2
  for hull passes like-for-like (15 vs 15)**: the direct interface adds positioning and
  spectralog. Pyomo's two `incorrect` big-m answers in #2149 (cstr, eight_process_logical)
  come out correct here on either interface, so they belonged to SCIP 10.0.2, not to the
  `.nl` route.

### Per family: the measured lead for each miss

* **big-m, hull:** per-solve fixed cost on small instances plus search on hull (task 2
  above); `DISCOPT_GDP_CONFIG_ROOT_SKI` takes the constructor's share. Bar 2 for big-m also
  loses med_term_purchasing to the HiGHS ≥ 1.13 certificate decline (task 2's side finding).
* **lbb:** med_term_purchasing and spectralog, both solved in #2149. spectralog is near the
  limit on this host. Whether med_term's LBB loss has the same HiGHS cause has not been
  tested; a highspy 1.11.0 run of `discopt-lbb` would decide it.
* **ric, enumerate:** the per-configuration global subproblems on the large GDPlib models
  (task 3, G4.2).
* **loa, gloa:** bar 3's CI clause only. The point estimates are 0.94 and 0.87.

### Deviations from the spec and from #2149, stated

* `--jobs 3` on 4 cores (#2149: jobs 4 on 14), with the asymmetric inflation above.
* Pyomo 6.10.1 (#2149: 6.10.0). Pyomo hull no longer errors on gdp_col, hda, kaibel or
  stranded_gas; it reaches its limit instead.
* highspy 1.15.1, which costs med_term_purchasing under discopt big-M and LBB.
* Pyomo enumerate on med_term_purchasing was OOM-killed (11.9 GB RSS in a 16 GB container),
  as in #2149 (`killed`); it counts as unsolved.
* No row was load-gated (`--max-load 16`); the 1-minute load ran 2.4-8.5.

## #2182 close-out: the decision (2026-11-06)

**Decision (owner, 2026-11-06): #2182 closes, and with it the GDP parity effort (#1775 →
#1917 → #2182), with the §1.2 performance gate recorded as unmet.** The spec is not
edited; the bar stands as written and is not met. What was reached, and why the rest is
not GDP parity work:

* **Correctness parity: exceeded.** discopt has 0 incorrect answers in every family in
  both full gate runs (#2149 and this one). Pyomo has 10 and 8, including false optima
  from its LOA, GLOA and RIC.
* **Capability parity: met**, with documented exceptions (the capability matrix above:
  19 of 23 rows; T2's local-variable half skipped by the owner, T3 and T11 retired on
  their panels, M5 killed at its entry experiment).
* **Performance parity: not met.** loa and gloa beat Pyomo at the point estimate (0.94,
  0.87) and on solved count (13 vs 11), and miss only the CI clause. big-M and hull lose
  on small instances where Pyomo + SCIP takes 0.1-0.2 s. The cause is measured (task 2,
  `performance-plan.md` §120): per-solve and per-node overhead, spread across components,
  in the **Python** tree. The native kernel declines every lowering measured, for five
  distinct reasons. That is kernel coverage, a general performance question, and it
  continues in `performance-plan.md`, not as GDP parity work.
* **Why the gate is not re-run again.** Between #2149 and this run discopt got slightly
  faster on the same host (new/old 0.95), yet big-M's ratio moved 1.97 -> 3.42 on the
  environment alone: host, concurrency, Pyomo's SCIP interface and version. The CI clause
  needs a point estimate of roughly 0.65-0.7 on this 24-instance panel. Re-running it
  measures the machine more than the solver.

**Carried forward**, outside #2182:

* #2183: the HiGHS ≥ 1.13 fixed-integer-LP certificate decline (med_term_purchasing
  big-M). A sound refusal of a certificate, with a proposed rigorous fix.
* `performance-plan.md` §120: kernel coverage of GDP lowerings (`exp` / `log` atoms,
  lifted perspective columns, box-dependent big-M rows) as the next entry experiment.
* G4.2's done-when (enumerate / RIC certify every ≤ 10⁴-configuration instance) stays
  unmet. Its remaining instances are the large GDPlib models, where the per-configuration
  global subproblem is the cost, the same kernel-coverage question.
