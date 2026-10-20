# GDP parity baseline: open-solver profile O (#1775 G0.6)

**Status: partial, 14 of 24 panel instances.** The run stopped because the cloud
container it ran in shuts down shortly after a session goes idle, so it cannot hold
an 8-hour job. The results so far are in
[`data/gdp_baseline_g06.jsonl`](data/gdp_baseline_g06.jsonl), and the command below
resumes from them. G0.6 is not done until all 24 instances are measured and the
gap table and per-family targets below are final.

## Finishing the run

On a Linux machine with root (Ubuntu 24.04 is the tested target):

```bash
sudo bash discopt_benchmarks/scripts/install_open_gdp_solvers.sh   # profile O solvers
cp docs/dev/data/gdp_baseline_g06.jsonl /tmp/base.jsonl
python -u discopt_benchmarks/scripts/gdp_baseline.py --out /tmp/base.jsonl \
    --time-limit 300 --grace 120          # skips the 170 recorded runs
python -u discopt_benchmarks/scripts/gdp_baseline.py --out /tmp/base.jsonl --report
```

Keep the machine otherwise idle: every run records its 1-minute load, and a run
started above `--max-load` (default 16) is marked `load_gated` and is not a timing
claim (CLAUDE.md §9).

**Mixed discopt versions.** The discopt rows already recorded were measured on
`5f8a174c`, which is before `cf5ffb21` changed the fast-API placeholder objective.
That change is not expected to move these results, because the GDPlib/Pyomo
conversion does not use the fast API, but that expectation is untested. To measure
every discopt row on one version, delete the `discopt-*` rows before resuming. That
adds about 120 runs.

## What is measured

The suite is `gdp_parity`: the 17 GDPlib instances of GDPlib PR 58, plus 7 Pyomo
GDPopt test models (spec §1.2). The 12 arms are interleaved per instance, each run is
a subprocess with a hard kill at `time_limit + grace`, and the time limit is 300 s:

* **Pyomo:** `gdp.bigm` and `gdp.hull` solved by SCIP (pyscipopt), and GDPopt
  LOA / GLOA / LBB / RIC / enumerate, with CBC as the MIP solver, Ipopt as the NLP
  solver and SCIP as the MINLP / global NLP solver.
* **discopt:** `gdp_method` set to `big-m` (the default route), `hull`, `mbigm`,
  `auto` or `loa`. Every discopt incumbent is also checked in the original Pyomo
  model under Pyomo's GDP semantics.

Scoring is `gdp_baseline.score`, against `gdplib_runner.panel_oracle()` at the
repository's tolerance (abs 1e-6, rel 1e-4):

* `solved` -- claims optimal and agrees with the oracle;
* `incorrect` -- one of: claims optimal and disagrees, an incumbent better than the
  optimum, an infeasibility claim on a known-feasible instance, or a point that
  violates the Pyomo model;
* `claimed_unverified` -- claims optimal where the oracle is unproven;
* `unsolved` -- anything else, including time-limit kills and out-of-memory kills
  (exit -9).

SGM uses a 1 s shift, and every run that is not solved counts at the time limit.

## Partial results (14 instances; not final)

The full report, from `--report` on the partial file:

```
arm                  n solved incorr unver   SGM(s) median_solved  total(s)
pyomo-bigm          14      8      0     1    29.25          1.72    2024.5
pyomo-hull          14      8      0     0    27.10          3.42    2080.6
pyomo-loa           15      4      2     3    93.12          1.90    3314.6
pyomo-gloa          15      4      2     0    90.58          1.52    3313.9
pyomo-lbb           14      6      0     0    79.54         59.13    2625.7
pyomo-ric           14      4      2     3    84.38          1.97    3014.4
pyomo-enumerate     14      3      1     2   108.43          0.95    3306.4
discopt-big-m       14      6      0     0   103.42         39.28    2600.9
discopt-hull        14      5      0     0   111.48         16.13    2865.6
discopt-mbigm       14      6      0     0   102.03         38.02    2602.0
discopt-auto        14      6      0     0   102.94         40.38    2601.3
discopt-loa         14      4      0     0    83.53          0.71    3053.5
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

Observations, to be confirmed on the full panel:

* **No incorrect discopt run.** Every incorrect run is a GDPopt false certificate:
  * on cstr, LOA, GLOA, RIC and enumerate all claim `optimal` at 3.1302 against the
    certified 3.0620;
  * on med_term_purchasing (a maximization), LOA claims 6781.50, and GLOA and RIC
    claim 6793.46, against 6797.54.
* **discopt's gap to Pyomo's bigm/hull with SCIP:** on the big-M family discopt
  solves 6 of 14 instances against 8 for Pyomo, and its SGM is 3.5x Pyomo's. The
  two instances it does not solve are batch_processing and cstr, both at the
  300 s limit.
* **discopt hull** refuses gdp_col, hda and kaibel with
  `HullPerspectiveOriginError`: a disjunct body is not finite at the origin. That is
  a loud refusal, not a wrong answer, and it is the next hull capability gap after
  G2.1.
* **discopt LOA** solves 4 instances, with a lower SGM than Pyomo LOA, and has no
  false certificate.

Per-family targets (spec §1.2) are set once the panel is complete.
