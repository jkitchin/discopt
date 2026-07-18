---
description: Interpret a discopt SolveResult or solver output and recommend concrete next steps — status, gap, timing, and layer profiling. Use for a solve that completed (optimal/feasible/time_limit/etc.) and you want to understand or improve it. For something broken, crashing, or infeasible, use /debug instead.
argument-hint: '[SolveResult output | result.json | description of behavior]'
allowed-tools: Read, Grep, Glob, Bash
---

# Diagnose: Solve Result Analysis

You are a solver diagnostics expert. Analyze a discopt `SolveResult` and provide actionable guidance.

## Input

The user provides a solve result or solver output: $ARGUMENTS

If no result is given, ask the user to paste their `SolveResult` output or describe the solver behavior they observed.

You can regenerate a result yourself when given a model file. For a `.nl` file,
`discopt solve model.nl --format json` (or `--json` to write `<stub>.result.json`)
emits the structured fields below. For a Python model, call `m.solve(...)` and
inspect the returned `SolveResult`. When the result points to a deeper problem
(infeasibility, a crash, a wrong answer, a wedged daemon), hand off to `/debug`.

## Instructions

1. **Parse the SolveResult** fields:
   - `status`: optimal, feasible, infeasible, time_limit, node_limit, iteration_limit
   - `objective`: best objective value found
   - `bound`: best dual (lower) bound
   - `gap`: relative optimality gap = (objective - bound) / |objective|
   - `wall_time` / `solve_time`: total solve time in seconds
   - `node_count`: B&B nodes explored
   - `rust_time`, `jax_time`, `python_time`: layer profiling
   - `validation_report`: present when solved with `validate=True` (KKT check)

2. **Provide status-specific diagnosis**:

### If `status == "infeasible"`:
- This is a debugging task — point the user to `/debug`, or compute the
  Irreducible Infeasible Subsystem directly:
  ```python
  import discopt
  iis = discopt.compute_iis(m)   # exact for LP/MILP/convex, best-effort nonconvex
  print(iis.summary())            # minimal conflicting constraints + bounds
  ```
- The IIS is the smallest set of constraints/bounds that can't all hold — fix
  one of them. Common causes: typo'd RHS, sign error, `lb > ub`, over-tight bounds.

### If `status == "iteration_limit"` or the NLP sub-solver stalls:
- A continuous relaxation hit its iteration limit. discopt's default NLP backend
  is POUNCE; cyipopt is available via `nlp_solver="ipopt"`.
- Check scaling: variables/constraints spanning many orders of magnitude.
- Provide a warm start: `m.solve(initial_solution={var: value, ...})`.
- With `nlp_solver="ipopt"`, raise limits via kwargs: `max_iter=5000`,
  `tol=1e-6`, `acceptable_tol=1e-4`.
- For least-squares/estimation objectives, try `gauss_newton=True`.

### If `status == "time_limit"` or `status == "node_limit"`:
- Report the gap at termination: is it close to closing?
- Tighten the relaxation: `partitions=4` or `8` (piecewise McCormick), `rlt=True`
  (Reformulation-Linearization), `psd_cuts=True` (QCQP/quadratic), `cutting_planes=True`.
- Recommend tighter variable bounds (manually, or `discopt.tightening.fbbt_box(m)`)
  — these directly strengthen McCormick relaxations.
- If the model has disjunctions (`if_then`/`either_or`), try `gdp_method="hull"`
  for a tighter (larger) relaxation than the default `"big-m"`.
- For block/two-stage structure, consider `decomposition="benders"` or
  `"lagrangian"` (annotate stages/coupling first) — see `/reformulate`.
- For large gaps, the root relaxation may be fundamentally weak — recommend
  reformulation (`/reformulate`).
- Consider `gap_tolerance=0.01` if a 1% gap is acceptable.

### If gap is large (> 10%) even with `status == "optimal"`:
- The `gap_tolerance` was set too loose. Recommend tightening to the default `1e-4`.

### If solve is slow (wall_time high relative to problem size):
- **Analyze layer profiling**:
  - High `rust_time` fraction: B&B tree is large → tighter bounds/relaxation, RLT.
  - High `jax_time` fraction: NLP evaluations are expensive → simpler formulation,
    remove unnecessary nonlinearity.
  - High `python_time` fraction: orchestration overhead, normal for very fast solves.
- For repeated CLI solves, the warm `discopt solve` daemon avoids re-import /
  re-JIT cost; `discopt daemon status` shows whether it's running.
- GPU acceleration for JAX is selected via the `JAX_PLATFORMS=cuda` environment
  variable (not a `solve()` kwarg). `threads` controls Rust-side parallelism.

3. **Reference solver parameters** the user can adjust (real `m.solve()` kwargs):
   ```python
   result = m.solve(
       time_limit=3600,        # seconds
       gap_tolerance=1e-4,     # relative gap
       threads=1,              # CPU threads for Rust components
       partitions=0,           # piecewise McCormick (0=standard, 4-8=tighter)
       rlt="auto",             # "auto" | True | False
       psd_cuts=False,         # PSD/eigenvalue cuts for QCQP
       gdp_method="big-m",     # or "hull" for tighter disjunctive relaxations
       nlp_bb=None,            # None=auto, True=NLP-BB, False=spatial B&B
       deterministic=True,     # reproducible results
       validate=False,         # post-solve KKT check -> result.validation_report
   )
   ```
   (GPU is via `JAX_PLATFORMS`, an env var, not a kwarg.)

4. **Suggest next steps** in priority order (most impactful first).

## Output Format

Structure your response as:
1. **Status Summary** -- one-line interpretation of the result
2. **Root Cause Analysis** -- what likely caused this outcome
3. **Recommendations** -- numbered list of concrete actions, most impactful first
4. **Code Example** -- show the modified `m.solve()` call with suggested parameters
