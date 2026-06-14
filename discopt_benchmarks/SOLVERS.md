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
| SCIP    | `…/miniforge/base/envs/discopt-bench/bin/scip` (conda-forge)          | 10.0.2   | ✅ (AMPL/MP reader) |
| Couenne | `/Applications/AMPL/couenne` (COIN-OR ASL build)                       | 0.5.8    | ✅ |
| HiGHS   | `/Applications/AMPL/highs` (ASL build)                                 | 1.11.0   | ✅ |

**Do not** use conda-forge HiGHS (1.14.0) for `.nl` work — its CLI errors with
"Model file not supported" and only reads `.mps`/`.lp`. Couenne is **not**
available on conda-forge osx-arm64 at all; the AMPL-bundled COIN-OR build is the
source. The AMPL-bundled open-source COIN-OR solvers (couenne, bonmin, the ASL
highs) need **no AMPL license**.

Paths are configured in `config/benchmarks.toml` under `[solvers]`.

## Conda environment

```bash
conda create -n discopt-bench -c conda-forge scip=10.0.2 highs bonmin
```

(SCIP is the one solver consumed from conda-forge. HiGHS/Couenne are taken from
`/Applications/AMPL/` per the table above.)

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
