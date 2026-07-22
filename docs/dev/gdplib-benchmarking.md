# Benchmarking discopt against GDPlib (issue #823)

**Status:** implemented. Pyomo-bridge runner + a native-discopt subset + tests +
optional-dependency extra landed; `jobshop` verified end-to-end against the HiGHS
oracle. The discopt-vs-SCIP table was **re-run 2026-07-22** after recent performance
work (5/12 now proven optimal, up from 2/12 — see below), and the HiGHS oracle was
hardened to require a *proven* optimum (review finding #1). This doc is the "figure
out how" writeup requested in [#823](https://github.com/jkitchin/discopt/issues/823)
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
certify the same optimum (11.0). The Pyomo-reformulation path is what lets us
benchmark the *whole* corpus without hand-porting 20+ models — but it means
**Pyomo**, not discopt, lowers the disjunctions. To also exercise discopt's *own*
disjunction machinery, a curated subset is additionally rebuilt in discopt's native
GDP API (`either_or`, `make_disjunct`/`add_disjunction`) — see
[Native discopt models](#native-discopt-models) below.

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

## Native discopt models

`discopt_benchmarks/benchmarks/gdplib_native.py` rebuilds a curated subset of GDPlib
problems **directly in discopt's native modeling API** instead of going through
Pyomo's `gdp.bigm`/`gdp.hull`. The runner above measures discopt *solving a model
Pyomo lowered*; the native path measures discopt lowering the disjunctions **itself**
(its in-house big-M/hull + the integrality-aware FBBT that branches on the selector
binaries). Two independent lowerings of the same math cross-check each other, and the
native builders need **only discopt** — not pyomo or gdplib — since the model is
transcribed from the GDPlib source, not imported.

```bash
cd discopt_benchmarks
python -m benchmarks.gdplib_native --list                 # native models
python -m benchmarks.gdplib_native --time-limit 120        # solve + soundness-check all
```

Each native builder is **tested to reach the same SCIP-certified optimum** as its
Pyomo-bridged counterpart (`reference_optima()`); a GDP's optimum is
reformulation-independent, so an equivalent native encoding of the feasible set is
faithful iff it certifies that optimum. Ported so far:

| native model | structure exercised | certified optimum |
|---|---|---:|
| `jobshop` | 2-way linear ordering disjunctions | 11.0 |
| `ex1_linan_2023` | two xor grid disjunctions + excluded disjunct, nonconvex objective | −0.9996 |
| `small_batch` | k-way unit-count disjunction per stage, `exp` objective | 167427.65 |

Not yet ported (candidates, with the transcription reason they were deferred):
`cstr` (944-LOC reactor superstructure with recycle), `positioning` (25×5 embedded
data tables), and the larger process models (`syngas`, `water_network`, `methanol`,
`modprodnet`, `batch_processing`, `gdp_col`). Port from the gdplib source and add a
certified-optimum test before listing them. This is the concrete, verified form of
the "exercise discopt's own GDP path" follow-up — narrower than a full Pyomo-GDP →
native converter, but every model is checked against an independent optimum.

## Correctness strategy (the non-negotiable part)

Per `CLAUDE.md` §1 the product is the *certificate*, so a benchmark that only
measures speed is not enough — every run is checked against an independent oracle,
selected most-trusted-first:

1. **HiGHS** (`appsi_highs`) on the **linear** subset — the reformulation is an MILP,
   so HiGHS is an exact, independent oracle. Used **only when HiGHS proves optimality**
   (termination `optimal`) within a bounded time limit; an interrupted MILP yields a
   bare incumbent that, if trusted, would flag discopt's *correct* optimum as an
   impossible incumbent, so it is discarded (issue #823 review, finding #1 — the
   HiGHS gate now mirrors the SCIP one). discopt's certified objective must match.
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

## discopt vs SCIP vs BARON (12 small models, big-M, 60 s each)

Re-run **2026-07-22** after the recent performance work. discopt and SCIP
(`pyscipopt`) read the same big-M `.nl`; BARON is run via GAMS (`optcr=0`) as an
independent third global solver. `reference optimum` is the value **both BARON and
SCIP prove** (see the cstr correction below). Reproduce discopt/SCIP with
`python scripts/reeval_gdplib.py`.

Legend: **opt** = proven optimal (gap 0); *feas* = feasible incumbent, not proven;
*inc* = incumbent only (time limit, gap open); — = no incumbent.

| model | reference opt | discopt | s | SCIP (pyscipopt) | s | BARON (GAMS) | s |
|---|---:|---|---:|---|---:|---|---:|
| jobshop | 11.0 | **opt** 11.0 | 0.1 | opt 11.0 | 0.0 | opt 11.0 | 1.2 |
| ex1_linan_2023 | −0.9996 | **opt** −0.9996 | 8.7 | opt −0.9996 | 0.0 | opt −0.9996 | 0.9 |
| positioning | −8.06414 | **opt** −8.06414 | 10.9 | opt −8.06414 | 0.5 | opt −8.06414 | 1.2 |
| small_batch | 167427.65 | **opt** 167427.65 | 11.4 | opt 167427.65 | 0.0 | opt 167427.66 | 0.9 |
| modprodnet | 3592.92 | **opt** 3592.92 | 25.3 | opt 3592.92 | 0.1 | opt 3592.92 | 1.1 |
| cstr | **3.06201** | *feas* 3.06202 | 61.2 | **opt 3.05431 ⚠ false** | 16.4 | opt 3.06201 | 8.7 |
| spectralog | 12.0893 | *feas* 12.0893 | 60.4 | opt 12.0893 | 8.9 | opt 12.0893 | 1.4 |
| water_network | 348337.04 | *feas* 348337.04 | 63.2 | opt 348337.04 | 6.3 | opt 348337.04 | 22.8 |
| syngas | 4669.02 | — | 89.0 | opt 4669.02 | 4.1 | opt 4669.02 | 1.7 |
| batch_processing | 679365.33 | — | 63.2 | *inc* 679365 | 60.2 | opt 679365.33 | 41.9 |
| methanol | *(unproven)* | *feas* −1793.43 | 61.8 | *inc* −1574.57 | 60.0 | *inc* −1793.43 | 61.9 |
| gdp_col | *(unproven)* | *feas* 20100.3 | 65.9 | *inc* 22283.5 | 60.0 | *inc* 20008.7 | 61.2 |

**Score (12 models): discopt** — 5 proven optimal, 3 more *reach* the reference
optimum without closing the gap (cstr, spectralog, water_network), 1 matches the
best-known unproven incumbent (methanol), 2 no incumbent (syngas, batch_processing),
1 slightly-worse incumbent (gdp_col). **SCIP (pyscipopt)** — 9 proven (one, cstr, is
a *false* optimum, see below), 3 time-limit incumbents. **BARON** — 10 proven
(everything except methanol/gdp_col, where it returns the best incumbent). **0 discopt
soundness violations.** Prior baseline (PR #825) was discopt 2/12 optimal — the perf
work moved 5 models into proven-optimal and 3 more onto the true optimum.

### ⚠ Correctness finding: a false SCIP optimum on cstr (and a wrong seed it produced)

The `_solve_with_scip` oracle (pyscipopt reading a hand-written `.nl`) reported
**cstr = 3.0543 with gap 0** — but that is *below* the true minimum. Two independent
solvers through a solution-loadable path (**BARON and SCIP, both via GAMS**) prove
**3.0620** with a pyomo-verified feasible point (max constraint violation ~1e-6), and
discopt's own incumbent agrees at 3.0620. So pyscipopt-via-`.nl` solved a
mis-encoded/relaxed cstr yet certified it — a classic false optimum. Consequences and
actions:

- `reference_optima()["cstr"]` is corrected `3.0543118 → 3.0620073` (BARON-proven).
  Every *other* seed was re-verified against BARON's independent proof and confirmed.
- discopt's cstr result was therefore **correct all along** (it found the true 3.0620,
  just didn't prove it) — the earlier "feasible-loose" label was an artifact of the
  false reference, now fixed.
- **The `pyscipopt`-`.nl` oracle path can seed a below-true value**, which as a
  minimize oracle can *mask* a real discopt false primal — a soundness-surveillance
  hole. Hardening it (use the GAMS path that loads a pyomo-verifiable solution, or
  cross-check every seed against BARON) is the priority oracle follow-up on #823.

## Findings & limitations

- **Sound throughout, and now competitive on the small subset.** Where discopt proves
  optimality it agrees with BARON to the digit; its feasible-only incumbents sit on
  the valid side of the true optimum. **Zero discopt soundness violations** — the
  certification-gap story has narrowed substantially since PR #825, without weakening
  any check. (The one false optimum on the sweep was SCIP's, not discopt's.)
- **A strong primal on the hardest two.** On methanol and gdp_col — which neither SCIP
  nor BARON certifies in 60 s — discopt matches BARON's best incumbent on methanol and
  is within 0.5 % on gdp_col, both far ahead of SCIP's incumbent. No certified optimum
  exists yet, so this is "strong incumbent, still unproven"; a longer BARON/SCIP run to
  certify is the priority.
- **The remaining gap is dual-side, not primal, on most of the set.** cstr, spectralog
  and water_network *find* the reference optimum but do not prove it in 60 s (a bounding
  gap, [#818](https://github.com/jkitchin/discopt/issues/818)); only syngas and
  batch_processing still find no incumbent (the [#817](https://github.com/jkitchin/discopt/issues/817)
  primal gap). Mild time-limit overruns persist (most within a few s of 60 s; syngas
  worst at 89 s) — the [#814](https://github.com/jkitchin/discopt/issues/814) behavior.
- **The oracle gap is closed for the small nonlinear subset** via BARON+SCIP; the
  larger models (biofuel, stranded_gas, grid) still need a longer budget to certify.

## Suggested next steps

1. Extend the SCIP-certified `reference_optima()` to the larger models with a longer
   budget; optionally add BARON/Couenne behind the same oracle hook for redundancy.
2. Wire a curated `gdplib_small` suite into the nightly correctness lane (linear
   models + short-limit nonlinear feasibility), gating on `incorrect_count == 0`.
3. Grow the native-discopt subset (`gdplib_native.py`) beyond the current three —
   `cstr`, `positioning`, and the larger process models — porting each from source
   with a certified-optimum test. The hand-ported route already exercises discopt's
   own disjunction/big-M/hull machinery; a general **Pyomo GDP → discopt native GDP**
   converter would generalize it to the whole corpus and remains the stronger,
   optional end goal.
4. Certify methanol and gdp_col (a longer SCIP/BARON budget) — discopt already returns
   a better incumbent than SCIP's 60 s incumbent on both, but neither is proven, so
   they need an independent optimum before they can anchor `reference_optima()`.
