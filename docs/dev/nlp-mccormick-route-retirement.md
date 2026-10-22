# Retirement of the McCormick-"nlp" relaxation route (#1873)

**Decision (2026-10-22, confirmed by the owner):** retire the route and everything that
existed only to serve it. That is 24 modules, the `relaxation_compiler` subtree and the
symbolic domain packs that fed it. Four `solve()` options are now **refused** with a
`ValueError` that points here:

- `mccormick_bounds="nlp"`
- `partitions` (any nonzero value)
- `relaxation_arithmetic` (anything but `"mccormick"`)
- `use_learned_relaxations=True`

This follows CLAUDE.md §5's third outcome. The route was not behind a flag, but it is
the same question: an alternative bounding path that measured no benefit. The
parameters stay in the signature so a caller gets a reason, not a `TypeError` or a
silent no-op.

It came out of #1873 (JAX as an optional extra): these 24 modules were most of the
module-scope `import jax` sites, and the first inventory called them "unreachable". That
was wrong (see *Retraction*). The question the owner asked was whether we need them, so
it was measured.

## What the route was

When `_mc_mode == "nlp"` survived to the node loop, the solver compiled the objective and
constraints into compositional McCormick relaxations (`relaxation_compiler`, on the JAX
primitives in `mccormick.py`/`envelopes.py`, optionally `piecewise_mccormick`, the
OA/Taylor/Chebyshev/ellipsoidal arithmetics, or ICNN-learned envelopes). It then solved an
NLP over the relaxation (`mccormick_nlp`), every 5th iteration, and max-combined the
result into the node bound.

It could only survive to the node loop in one way. Under the default
`mccormick_bounds="auto"` a convex model starts at `"none"`, and the #120 soundness guard
demotes every lp→nlp fallback on a nonconvex model to alphaBB. So it took **a convex model
plus an explicit `mccormick_bounds="nlp"`/`"lp"`**. On top of that, convex MINLPs
auto-route to `nlp-bb`, which never reads `mccormick_bounds`, so it also took
**`nlp_bb=False`**.

For a convex model, the McCormick relaxation of the objective is ≤ the objective, so its
minimum cannot exceed a correctly solved node NLP. In theory, the route can never tighten
the node bound. That was the hypothesis; the measurement below tests it.

## Measurements

All on this branch, Linux, 4 cores, in-repo corpus (`python/tests/data/minlplib_nl/`, 66
instances). Each run is a fresh interpreter. The probe counts `compile_objective_relaxation`
calls, McCormick-NLP solves and finite results, so a run where the route never fired is
visible as such (CLAUDE.md §6).

**1. Reachability.** Under `mccormick_bounds="nlp"` alone, the route fired on **0** of the
first 5 instances screened, including the convex MINLP `alan` (it went to `nlp-bb`). A small
convex MINLP confirmed the mechanism: `compile_calls 0` with `nlp-bb`, `1` with
`nlp_bb=False`. With `nlp_bb=False` added, it fired on **7 of 66** instances: `nvs03`,
`flay02m`, `flay03m`, `m3`, `fac2`, `tls2`, `cvxnonsep_psig40r`. Two more,
`cvxnonsep_nsig30` and `cvxnonsep_psig30`, produced no row at all (see 3).

**2. Ablation: does its bound change anything?** The route was run as-is (`nlp`) and with
both McCormick-NLP solvers forced to return `-inf` (`nlp_ablate`). Everything else is
identical, including the compile and the solves. `time_limit=60`.

| instance | default | nlp | nlp_ablate | MC solves / finite |
|---|---|---|---|---|
| nvs03 | optimal, 3 nodes | optimal, 3 nodes, bound 16.0 | optimal, 3 nodes, bound 16.0 | 1/1 |
| flay02m | optimal, 7 | optimal, 7 | optimal, 7 | 1/1 |
| flay03m | optimal, 115 | optimal, 115 | optimal, 115 | 17/17 |
| m3 | feasible, 47 | feasible, 47 | feasible, 47 | 11/10 |
| fac2 | feasible, 39 | feasible, 39 | feasible, 39 | 10/17 |
| cvxnonsep_psig40r | optimal, 95 | optimal, 95 | optimal, 95 | 23/40 |
| tls2 | time_limit, 251 | time_limit, 221 | time_limit, 237 | 18/18 |

Bounds are identical across all three arms on every row (`tls2`: 0.9552466284 in all
three). On `tls2` node counts differ only because the run hits the time limit.

The same ablation was repeated for every variant on the 6 terminating instances:
`partitions=4` and `relaxation_arithmetic` ∈ {`alphabb`, `taylor`, `ellipsoidal`,
`chebyshev`}. That is 30 nlp/nlp_ablate pairs. Status, bound and node count were identical
in all 30. Three `fac2` pairs differ only in the *incumbent's* last digits (1e-16 relative;
6e-10 in one). The incumbent varies the same way between the default and `nlp` arms, so
it is heuristic/timing noise, not the McCormick bound.

**Result: in 36 of 36 instance/variant pairs the route's bound changed nothing**, though it
was computed, and finite on most calls. The hypothesis held.

**3. Cost.** These are single sequential runs, *not* interleaved and with no spread, so
under CLAUDE.md §9 they are indicative and not a timing claim. The decision does not rest
on them.

- On the 6 terminating instances `nlp` took 1.4–3.3× the default's wall time and
  2.3–3.6× its peak RSS (default 80–116 MB, `nlp` 248–399 MB).
- `cvxnonsep_nsig30`, `time_limit=10`: the default certifies optimal in **6.2 s** (293
  nodes). `nlp` ran **270 s** (27× over its limit) and ended `time_limit` with **no
  incumbent**.
- `cvxnonsep_psig30`, `time_limit=10`: the default certifies optimal in **1.8 s at 106 MB
  peak**. `nlp` reached **13.6 GB** and was **killed by the kernel (OOM)**.

Those last two are orders of magnitude, not noise.

**4. `use_learned_relaxations`** could never take effect. No pretrained models ship
(`discopt/_relax/pretrained/` does not exist), so `load_pretrained_registry()` always
returned an empty registry and the solve warned and fell back to standard McCormick. It
also only ever affected this route.

## Per-module resolution

Every module with a module-scope `import jax` before #1873 (37), plus `chebyshev_model`
(not JAX, orphaned by the retirement). "Importers" comes from the AST import graph.
`test_relax_module_status.py` enforces that every remaining `_relax` module is either
production-imported or declared in `module_status.py`, and that none of the retired ones
comes back (`RETIRED_1873`).

### Kept (15), each with a reason

| Module(s) | Why it stays |
|---|---|
| `nlp_evaluator`, `dag_compiler`, `sparse_jacobian`, `sparse_hessian` | The JAX NLP evaluator. It is the only evaluator for constructs the tape cannot lower: permanently the `dm.custom` family, and today matrix norms and trainable-NN surrogates (`docs/dev/jax-optional-inventory.md` §1). It is also the `DISCOPT_NLP_EVAL=jax` opt-out §5 requires, and the oracle in 85 test files. |
| `mcbox`, `mccormick_subgradient`, `multivariate_mccormick` | The reduced-space engine: the only route that certifies a `dm.custom` model globally (P3.1, #713). `mccormick_subgradient`'s one import from `relaxation_compiler` (a wrapper of `_flat_index.resolve_scalar_slot`) now goes to `_flat_index` directly. |
| `differentiable`, `differentiable_lp`, `differentiable_qp`, `pounce_layer`, `parametric` | Public differentiable-optimization features (`differentiable_solve`, the POUNCE layers, `discopt.parametric`/`sensitivity`), each with a notebook. |
| `symbolic.signed_signomial` | Reference oracle. The production numpy signomial engine (`convexity.signomial_global`) is pinned to it pointwise by three test files. Declared `tooling` in `module_status.py`. |
| `symbolic.verification`, `symbolic.runtime` (+ the sympy-side `codegen`, `envelope_deriver`) | The derive → code-generate → verify toolkit of the published `symbolic_envelopes` notebook. No solver path runs its output, as that notebook now says. |

### Retired (24)

| Module(s) | Why |
|---|---|
| `relaxation_compiler`, `batch_evaluator`, `mccormick_nlp`, `mccormick_evaluator` | The route itself (measurements above). |
| `mccormick`, `envelopes`, `alphabb`, `piecewise_mccormick`, `oa_relax`, `polyhedral_oa`, `taylor_model`, `ellipsoidal_arith`, `chebyshev_model` | Imported only by the route (`relaxation_compiler` and its subtree). The default engine has its own numpy envelopes (`uniform_relax._UNIVARIATE_FN`, `milp_relaxation`); rigorous alphaBB lives in `_alphabb_rigorous`. |
| `learned_relaxations`, `icnn`, `icnn_trainer` | `use_learned_relaxations` never took effect (no shipped models) and fed only the route. |
| `symbolic.domains.{chemeng,gas,power}`, `symbolic.registry` | Hand-written JAX closures whose only solver consumer was `relaxation_compiler`. The numpy engine relaxes these atoms natively (`entropy`/`x log x`, `sin`, `cos`, `exp`, ...). |
| `symbolic.patterns`, `symbolic.gp_hull` | A design catalogue with no consumer. `gp_hull` was tested only against itself, unlike `signed_signomial`. |
| `symbolic.certified_learned` | Certified ICNNs for the retired learned route. |

### Retraction

The first version of `docs/dev/jax-optional-inventory.md` §3 listed most of these as having
"no importer outside other JAX modules and tests", which read as unreachable. That was wrong.
Most were reachable from `solver.py` through `relaxation_compiler`, behind the option
combination above. The corrected statement is this document.

## What changed alongside

- **Solver:** `solver._refuse_retired_relaxation_options` runs at the top of both
  `Model.solve` (before any route, native kernel or convex kernel can take the model) and
  `solve_model`. The lp→nlp fallbacks set `"none"` directly and keep the #120/#1112
  message via `_lp_downgraded`. That is the same outcome the #120 guard produced for
  nonconvex models; for convex models the ablation shows `"none"` matches on node count.
- **Option plumbing:** the ignored-option tables, `_native_kernel_ignored_levers`'
  `partitions`, and the CLI's `--partitions` are gone.
- **LLM advisor/diagnosis/prompts and the shipped skills** (`convex-relaxation-expert`,
  `/diagnose`, `/debug`, `/reformulate`) recommended `partitions=4`, which would now raise.
  They point at `solver="amp"` (adaptive McCormick partitioning) instead, and
  `suggest_solver_params` drops LLM-suggested keys outside its documented set.
- **Tests:**
  - deleted where they tested only retired code: 25 files plus `relaxation_harness.py`, and
    4 `discopt_benchmarks` regression tests;
  - mixed files keep their live parts: `test_multivariate_mccormick`,
    `test_relax_compiler_convexity_units`, `test_1520_relax_misc_handlers`,
    `test_alphabb_bound_soundness`, `test_amp_integration` and others.

  The envelope bug cases the deleted tests pinned (C-19 tan pole, C-23 division, C-24
  infinite bounds, C-32 asin/acos, #1493 negative powers) already have default-engine
  regressions: `test_1277_univariate_envelopes`, `test_1493_extended_division`,
  `test_1118_power_singular_tangent`, `test_1510_closed_edge_envelopes` and
  `test_1111`/`test_1115`.
- **Docs:**
  - notebooks `advanced_features`, `tutorial_minlp`, `solver_internals`,
    `symbolic_envelopes` and `llm_integration` were rewritten, and their changed cells
    re-executed;
  - `docs/design/relaxation-catalog.md` was re-audited (below);
  - `mecp_crossing_points.md` was updated.
- **Scripts:** `discopt_benchmarks/scripts/p1a2_gibbs_entry/joint_alphabb.py` (an archived
  entry-experiment harness built on `_relax.alphabb`) is deleted; its record,
  `p1a2-gibbs-log-sum-oa-entry-2026-07-11.md`, stays as history. Arm A of
  `scripts/entry_1245_parameter_resolve_cost.py` now reports the compiler retired instead of
  crashing. A counter in `scripts/decomp1_cert_effort_run.py` that would have silently
  no-oped behind `except Exception: pass` is removed.

## A gap this surfaced: `sign`

The relaxation catalog claimed every API function had a relaxation rule. It was audited
against the retired compiler. Re-measured against the default engine (each of the 24
univariate API functions in `min f(x) - x*y`, default `solve()`): **23 of 24 certify; `sign`
does not** (`feasible`, `bound=None`). Its only envelope, `relax_sign`, lived in the
retired compiler, which never ran on the default path. So this is **not a regression from
the retirement**: the default path never had it. The catalog now says so. Adding a `sign`
envelope to the numpy engine is separate work.

## What would change this decision

A model class where an NLP over a compositional relaxation gives a node bound the LP-form
relaxation, alphaBB and the convex NLP relaxation do not. For convex models that is ruled
out by construction (see *What the route was*). For nonconvex models the route was never
sound (#120). The piecewise and alternative-arithmetic envelopes could still be worth having
as **rows of the LP relaxation**. That would be new work in `milp_relaxation`, measured on
its own panel, and not a revival of this route.
