# External Solver Setup (local benchmarking)

discopt benchmarks compare against external MINLP/MILP solvers via their AMPL
`.nl` interface. The harness shells out, captures stdout, and parses status /
objective / bound / nodes (`benchmarks/runner.py:_build_command`,
`_parse_external_output`). This note records the working local setup on macOS
(osx-arm64) so the comparison suite is reproducible.

## Which build to use (the `.nl` gotcha)

A solver binary only works here if it can **read `.nl` files**. Several distributed
builds cannot, so the binary that matters is not always the obvious one:

| Solver  | Use this binary                                                        | Version  | Reads `.nl`? |
|---------|-----------------------------------------------------------------------|----------|--------------|
| SCIP    | `/opt/homebrew/bin/scip` (`brew install scip`)                         | 10.1.0   | ✅ (AMPL/MP reader) |
| Couenne | `/Applications/AMPL/couenne` (COIN-OR ASL build)                       | 0.5.8    | ✅ |
| HiGHS   | `/Applications/AMPL/highs` (ASL build)                                 | 1.11.0   | ✅ |

**Do not** use conda-forge HiGHS (1.14.0) for `.nl` work — its CLI errors with
"Model file not supported" and only reads `.mps`/`.lp`. Couenne is **not**
available on conda-forge osx-arm64 at all; the AMPL-bundled COIN-OR build is the
source. The AMPL-bundled open-source COIN-OR solvers (couenne, bonmin, the ASL
highs) need **no AMPL license**.

Paths are configured in `config/benchmarks.toml` under `[solvers]`.

## Installing

```bash
brew install scip   # SCIP 10.1.0 with SoPlex; reads .nl
```

SCIP came from a conda-forge `discopt-bench` env until 2026-11-06, when that env
(and miniforge) was found deleted; `[solvers.bonmin]` still names it, so Bonmin
fails the preflight below until it is reinstalled. HiGHS/Couenne/BARON-demo are
taken from the AMPL bundle in `/Applications/AMPL/` per the table above.

## Preflight: a run refuses to start on a broken solver

Before any instance runs, `run_benchmarks.py` makes every external solver solve a
probe with a known optimum (`BenchmarkRunner.preflight_external_solvers`,
`PREFLIGHT_PROBES` in `benchmarks/runner.py`) and exits 1, naming each failing
solver and its configured path, if any of them cannot. It catches a missing
binary, a binary that cannot read `.nl` (it answers with a usage message and
rc=0), a demo license (both probes are larger than 10 variables and 10
constraints) and a wrong answer. This replaced a failure mode where a deleted
binary produced a column of ERROR rows that the summary reported as `0/50 solved`.

- `minlp` (default): MINLPLib `flay03m`, optimum 48.98979486. Convex, so local
  Bonmin is exact on it.
- `milp`: `data/preflight/probe_milp.nl`, a 5x5 assignment with one binding side
  row, optimum 16 by brute force (`data/preflight/make_probe_milp.py`). For an
  LP/MILP-only solver, set `preflight_probe = "milp"` in its `[solvers.*]` table
  (HiGHS does).

A `--solvers` name with no `[solvers.*]` entry is also refused. It used to fall
back to an internal solver, which runs discopt under the unknown name.

## Invocation details

- **SCIP** uses the batch `-c` interface with an explicit time limit:
  `scip -c "set limits time T" -c "read f.nl" -c "optimize" -c "display solution"
  -c "display statistics" -c "quit"`. Parsed from `SCIP Status`, `Primal Bound`,
  `Dual Bound`, `Solving Nodes`.
- **HiGHS / Couenne** take the `.nl` path as a positional argument.

## Objective-sense convention (important)

Couenne (and COIN AMPL solvers) report `Lower bound:` / `Upper bound:` in their
**internal minimization** sense: for a maximization model they solve `min g = -f`.
The harness reads the objective-sense marker from the `.nl` header
(`O<k> 1` = maximize) via `_nl_is_maximize` and **un-negates** for max problems:
`objective = -upper_internal`, `bound = -lower_internal`. SCIP and the ASL HiGHS
already report in the model's original sense, so no flip is applied there.

## Smoke check

```python
from benchmarks.runner import BenchmarkRunner, BenchmarkConfig
r = BenchmarkRunner(BenchmarkConfig(suite_name="adhoc"))
# /tmp/milp.nl is a small max-objective MILP with optimum 6.
# scip / couenne / highs should all report status=optimal, objective=6.0.
```
