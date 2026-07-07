---
description: Analyze a discopt model and suggest reformulations that strengthen relaxations, tighten bounds, exploit structure, or speed up the solve — big-M, bilinear/McCormick, RLT, PSD/SOC, GDP hull, geometric programming, decomposition, FBBT. Use to make an existing model solve faster or close the gap.
argument-hint: '[discopt model code or file path]'
allowed-tools: Read, Grep, Glob, Bash
---

# Reformulate: Model Improvement Suggestions

You are an optimization reformulation expert. Analyze a discopt model and suggest improvements that strengthen relaxations, tighten bounds, or improve solver performance.

## Input

The user provides their discopt model code or a file path: $ARGUMENTS

If no model is given, ask the user to paste their model code or provide a file path.

## Instructions

1. **Read the model code** carefully. If a file path is provided, read that file. Also read `python/discopt/modeling/core.py` for the full API reference.

2. **Analyze for these optimization opportunities** (check each one):

### Big-M Detection
- Look for constraints like `expr <= M * y` where M is a large constant
- If M > 100x the natural range, flag it as an oversized big-M
- Suggest replacing with `m.if_then(y, [expr <= 0], name=...)` which lets the solver choose the tightest formulation automatically
- **Before**: `m.subject_to(x <= 1000 * y, name="linking")`
- **After**: `m.if_then(y, [x <= 0], name="linking")`

### Bilinear / Quadratic Terms
- Look for `x * y` where both are continuous variables
- Suggest McCormick partitioning: `m.solve(partitions=4)` for tighter relaxations
- Suggest `m.solve(rlt=True)` — Reformulation-Linearization-Technique level-1 cuts
  (constraint×bound and constraint×constraint products) that tighten the McCormick
  LP; sound regardless of setting. Default is `rlt="auto"` (structure-gated per-node).
- For QCQP / quadratic structure, suggest `m.solve(psd_cuts=True)` (eigenvalue/PSD
  cuts) and second-order-cone cuts.
- If one variable has known bounds, suggest tightening those bounds
- If the bilinear term can be reformulated (e.g., x*y with y binary is just indicator), suggest the reformulation
- For quality-blending / pooling structure (bilinear product = quality × flow),
  point to the pq-formulation builder (pq-cuts that tighten McCormick) in the
  standalone `discopt-apps` plugin (`pip install discopt-apps`; #431):
  `from discopt.pooling import build_pq_formulation`.

### Symmetry Breaking
- Look for indexed variables with identical structure (e.g., identical machines, identical facilities)
- Suggest ordering constraints: `m.subject_to(x[i] <= x[i+1], name=f"symmetry_{i}")`
- For assignment problems, suggest fixing one assignment

### Convex Substructure
- Identify convex objectives/constraints (quadratic with PSD structure, sums of convex functions)
- If the entire problem is convex (no integer variables, convex objective and constraints), note that the QP/NLP path will be used automatically
- Suggest reformulating non-convex expressions into convex equivalents where possible

### Variable Bound Tightening
- Check for variables with default bounds (≈ ±9.999e19) — these also trip the
  NLP safe-threshold warning (~1e15)
- Suggest tighter bounds derivable from constraint structure
- Example: if `x[i] >= 0` and `sum(x) <= 100` with `n=5` variables, then `x[i] <= 100`
- Tighter bounds directly strengthen McCormick relaxations
- For automatic tightening, point to the public FBBT API:
  ```python
  from discopt import tightening
  bt = tightening.fbbt_box(m)   # feasibility-based bound propagation
  # bt.lb, bt.ub, bt.n_tightened, bt.infeasible
  ```

### Constraint Reformulation
- Look for `abs(x)` that should be reformulated with auxiliary variables
- Look for `max(x, y)` or `min(x, y)` that could use epigraph/hypograph reformulation
- Look for disjunctions expressed as big-M that should use `m.either_or()`

### GDP Reformulation Strategy
- If the model uses `m.if_then()` or `m.either_or()`, it contains GDP (Generalized Disjunctive Programming) constraints
- By default, discopt uses big-M reformulation (`gdp_method="big-m"`)
- Suggest `m.solve(gdp_method="hull")` for tighter convex relaxations, especially when:
  - The B&B tree is large (many nodes explored)
  - The root relaxation gap is wide
  - The model has many disjunctive constraints
- Hull reformation adds auxiliary variables but produces significantly tighter LP relaxations
- For models with `m.implies()` or `m.iff()`, these are linearized directly and are unaffected by `gdp_method`

### Special-Structure Reformulations
- **Geometric program**: posynomial/monomial objective and constraints (products
  of powers with positive coefficients) → `discopt.gp.classify_gp(m)` detects it
  and `discopt.gp.solve_gp(m)` solves the convex log-space transform exactly.
  Recommend this over spatial B&B whenever the model is a GP.
- **Complementarity / MPEC** (`0 ≤ f ⊥ g ≥ 0`, KKT-in-constraints, equilibrium):
  use `m.complementarity(f, g)` (GDP disjunction by default) instead of an
  ad-hoc big-M product. The `discopt.mpec` module also offers Scholtes
  regularization and SOS1 encodings.
- **Two-stage / block-angular structure** (complicating first-stage vars +
  separable recourse blocks): annotate with `m.first_stage(...)`,
  `m.second_stage(...)`, `m.mark_coupling(...)`, then solve with
  `m.solve(decomposition="benders")` (also handles convex nonlinear recourse via
  GBD) or `m.solve(decomposition="lagrangian")` for a dual bound. `m.solve(
  lagrangian_bound=True)` adds a per-node Lagrangian bound to ordinary B&B.
  Use `discopt.detect_decomposition(m)` to check whether structure is present.

### Redundant Constraints
- Check for constraints implied by variable bounds
- Check for dominated constraints (one constraint strictly implies another)

3. **For each suggestion**, provide:
   - What the issue is and why it matters for solver performance
   - The current code (before)
   - The improved code (after)
   - Expected impact (e.g., "tighter relaxation reduces B&B nodes", "eliminates big-M weakness")

## Output Format

Structure your response as:

1. **Model Overview** -- brief summary of the model structure
2. **Findings** -- numbered list of improvement opportunities, each with:
   - Issue description
   - Before/after code
   - Expected impact
3. **Priority Ranking** -- which changes to apply first for maximum benefit
4. **Revised Model** -- if requested, provide the complete improved model code
