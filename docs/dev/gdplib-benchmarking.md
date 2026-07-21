# Benchmarking discopt against GDPlib (issue #823)

**Status:** implemented. Runner + tests + optional-dependency extra landed;
`jobshop` verified end-to-end against the HiGHS oracle. This doc is the "figure out
how" writeup requested in [#823](https://github.com/jkitchin/discopt/issues/823)
(suggested by David Bernal Neira in [#819](https://github.com/jkitchin/discopt/issues/819)).

## What GDPlib is

[GDPlib](https://github.com/SECQUOIA/gdplib) (SECQUOIA) is an open library of
**Generalized Disjunctive Programming** models — process-synthesis and design
problems written in **Pyomo GDP** (`Disjunct` / `Disjunction` / `LogicalConstraint`
/ Boolean vars). It is the GDP analogue of MINLPLib: an educational and
benchmarking corpus. Most instances are nonlinear (reactor/column/network design);
a handful are linear (jobshop, scheduling, capacity-expansion).

## The bridge: how a GDPlib model reaches discopt

GDPlib models are Python-built Pyomo objects, not a file format discopt reads. But
discopt already ships a Pyomo solver plugin (`discopt.pyomo`, `pip install
discopt[pyomo]`) that round-trips a Pyomo model through a temporary AMPL `.nl` file
and solves it in-process. A Pyomo GDP model is not directly `.nl`-writable, so we
first lower the disjunctions to a standard mixed-integer (non)linear program with
Pyomo's built-in GDP transformations. The full path:

```python
import pyomo.environ as pyo
from pyomo.core import TransformationFactory
import discopt.pyomo                       # registers SolverFactory('discopt')
from gdplib.jobshop import build_model

m = build_model()                          # Pyomo GDP model
TransformationFactory('gdp.bigm').apply_to(m)   # or 'gdp.hull'
res = pyo.SolverFactory('discopt').solve(m, options={'time_limit': 300})
print(res.solver.termination_condition, pyo.value(m.makespan))
```

`gdp.bigm` (big-M) and `gdp.hull` (convex-hull / perspective) are the two standard
reformulations. Both round-trip cleanly through the bridge; on `jobshop` they
certify the same optimum (11.0). discopt also has a *native* GDP modeling API
(`either_or`, `make_disjunct`, `m.logical(...)`, see `python/tests/test_gdp.py`),
but the Pyomo-reformulation path is what lets us benchmark the existing corpus
without hand-porting 20+ models.

## Installation caveats (learned the hard way)

- **Install GDPlib from source, not PyPI.** The PyPI wheel omits the model data
  files (`.dat`, `.xlsx`, `.txt`), so most builders raise `FileNotFoundError`. Use:
  ```bash
  pip install "gdplib @ git+https://github.com/SECQUOIA/gdplib.git"
  ```
  or `pip install discopt-benchmarks[gdplib]`, which pins the source install plus
  `pyomo` and `highspy`.
- **Some builders need an external solver at *build* time** (`mod_hens`, `kaibel`
  call ipopt; `reverse_electrodialysis` calls GAMS). Without those tools the build
  raises — the runner records this as an `ERROR` run rather than crashing the sweep.
- A few `logical`-subpackage models depend on Pyomo APIs that drift across versions
  (`BooleanVarData.set_binary_var`); pin a compatible Pyomo if you hit that.

## The runner

`discopt_benchmarks/benchmarks/gdplib_runner.py` automates the pipeline and emits
`metrics.SolveResult` / `InstanceInfo` objects, so GDPlib results flow through the
same correctness/reporting machinery as the MINLPLib and CUTEst runners.

```bash
cd discopt_benchmarks

# list the models discovered in your install
python -m benchmarks.gdplib_runner --list

# solve one model with both reformulations, 60s each, cross-check vs HiGHS
python -m benchmarks.gdplib_runner --models jobshop --methods bigm hull --time-limit 60

# sweep the small models only, write results JSON
python -m benchmarks.gdplib_runner --max-variables 500 --time-limit 120 --output reports/gdplib.json
```

The CLI exits nonzero if any run trips a soundness flag, so it can gate CI.

## Correctness strategy (the non-negotiable part)

Per `CLAUDE.md` §1 the product is the *certificate*, so a benchmark that only
measures speed is not enough — every run is checked against an independent oracle,
selected most-trusted-first:

1. **HiGHS** (`appsi_highs`) on the **linear** subset — the reformulation is an MILP,
   so HiGHS is an exact, independent oracle. discopt's certified objective must match.
2. **SCIP** (`pyscipopt`, which bundles SCIP) on the **nonlinear** subset — a global
   MINLP solver reading the *same* AMPL `.nl` discopt solves. SCIP's objective is
   used **only when SCIP proves global optimality** (status `optimal`, gap ≈ 0); an
   unconverged SCIP run yields an incumbent/bound, never a certified optimum, so it
   is discarded rather than trusted.
3. **`reference_optima()`** — the SCIP-certified values, baked in as regression
   anchors and used offline / when `pyscipopt` is absent.

Three flags are raised and surfaced loudly, never masked:
- **impossible incumbent** — *any* feasible incumbent (even `FEASIBLE`, not
  `OPTIMAL`) beating the oracle optimum: it violates a constraint (false primal, cf.
  #815). The most dangerous case, and it does not require an `OPTIMAL` claim.
- **false optimum** — `OPTIMAL` but the certified objective disagrees with the oracle.
- **bound crossing** — the dual bound sits on the wrong side of the optimum.

Any of these is a hard failure — `incorrect_count` must stay 0, and the CLI exits
nonzero. BARON / Couenne could be added the same way SCIP was, behind the oracle hook.

## Model inventory (source install, this environment)

Sizes are the *reformulated* (`gdp.bigm`) model; `class` is discopt-relevant
(linear ⇒ HiGHS-checkable). `ERR` = build needs an external tool not present here.

| model | vars | disj | class | notes |
|---|---:|---:|---|---|
| jobshop | 4 | 3 | linear | ✓ verified: bigm & hull = **11.0**, matches HiGHS |
| ex1_linan_2023 | 2 | 2 | nonlinear | tiny but nonconvex |
| positioning | 6 | 25 | nonlinear | nonconvex distance |
| small_batch | 19 | 9 | nonlinear | |
| cstr | 56 | 10 | nonlinear | |
| spectralog | 68 | 30 | nonlinear | |
| methanol | 247 | 4 | nonlinear | |
| batch_processing | 270 | 9 | nonlinear | |
| syngas | 321 | 23 | nonlinear | |
| water_network | 335 | 5 | nonlinear | |
| gdp_col | 412 | 15 | nonlinear | distillation column |
| multiperiod_blending | 474 | 126 | nonlinear | |
| modprodnet | 486 | 1 | nonlinear | + distributed/quarter variants |
| med_term_purchasing | 949 | 72 | linear | HiGHS-checkable |
| pandemic | 999 | 111 | nonlinear | |
| hda | 1146 | 6 | nonlinear | |
| disease_model | 1198 | 26 | linear | HiGHS-checkable |
| grid | 12525 | 12500 | linear | large scheduling MILP |
| biofuel | 36324 | 252 | nonlinear | large |
| stranded_gas | 57618 | 96 | nonlinear | large |
| kaibel, mod_hens | — | — | ERR | build needs ipopt |
| reverse_electrodialysis | — | — | ERR | build needs GAMS |

## discopt vs SCIP (12 small models, big-M, 60 s each)

Both solvers read the **same** big-M `.nl`, so this is a fair head-to-head. SCIP 10
(via `pyscipopt`) is the oracle; its proven optima seed `reference_optima()`.

| model | discopt status | discopt obj | SCIP status | SCIP obj (opt) |
|---|---|---:|---|---:|
| jobshop | optimal | 11.0 | optimal | 11.0 |
| ex1_linan_2023 | optimal | −0.9996 | optimal | −0.9996 |
| positioning | feasible | 2570.62 | optimal | −8.06414 |
| cstr | feasible | 4.06191 | optimal | 3.05431 |
| small_batch | time_limit | — | optimal | 167427.65 |
| spectralog | time_limit | — | optimal | 12.0893 |
| syngas | time_limit | — | optimal | 4669.02 |
| water_network | time_limit | — | optimal | 348337.04 |
| modprodnet | time_limit | — | optimal | 3592.92 |
| methanol | time_limit | — | timelimit | −1793.43 (gap ~130%) |
| batch_processing | time_limit | — | timelimit | 679365 (gap 3%) |
| gdp_col | time_limit | — | timelimit | 20355.1 (gap ~110%) |

**SCIP: 9/12 proven optimal** (all ≤ 41 s). **discopt: 2/12 optimal, 2 loose-feasible,
8 no incumbent.** The three SCIP left unproven are *not* seeded.

## Findings & limitations

- **Sound, but far slower/weaker than SCIP on this set.** Every discopt result is on
  the correct side of SCIP's bound — where both prove optimality they agree to the
  digit; discopt's feasible incumbents (positioning, cstr) sit *above* the min optimum
  as a valid incumbent must. **Zero soundness violations.** This is the
  certification-gap story, quantified.
- **Nonlinear GDPlib instances are genuinely hard for discopt today** and are good
  stress material, not CI fodder: several hit the time limit and overrun it (the
  known [#814](https://github.com/jkitchin/discopt/issues/814) time-limit-overrun
  behavior — e.g. `small_batch` ran well past a 5 s budget; SCIP by contrast stops
  cleanly at its limit). Expect `feasible`/`time_limit`, not `optimal`, on the
  nonlinear corpus.
- **The oracle gap is now closed for the small nonlinear subset** via SCIP; the
  larger models (biofuel, stranded_gas, grid) still need a longer SCIP budget to
  certify.

## Suggested next steps

1. Extend the SCIP-certified `reference_optima()` to the larger models with a longer
   budget; optionally add BARON/Couenne behind the same oracle hook for redundancy.
2. Wire a curated `gdplib_small` suite into the nightly correctness lane (linear
   models + short-limit nonlinear feasibility), gating on `incorrect_count == 0`.
3. Optionally add a direct **Pyomo GDP → discopt native GDP** converter so discopt's
   own disjunction/big-M/hull machinery is exercised instead of Pyomo's — a stronger
   test of discopt's GDP path than the reformulate-then-solve route.
