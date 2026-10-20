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
  scoped to a task.
* LOA's gap is 4 instances, not speed.
