# Benchmark Diagnosis: MILP, MIQP, and MINLP Issues

**Date**: 2026-02-20
**Platform**: macOS ARM64 (Apple M4 Pro), Rust 1.84, Python 3.12

## Finding 1: MILP — Constraint-Violating Incumbents (Solver Bug)

**Root cause: The MILP B&B tree accepts LP relaxation solutions as
integer-feasible incumbents without checking constraint satisfaction.**

The Rust B&B tree only verifies integrality of integer/binary variables.
It does NOT verify that the LP solution satisfies the original constraints
(converted to equalities with slacks). When the LP IPM at a node returns a
solution where binary variables happen to be near-integer but continuous
variables violate constraints, the tree accepts it as a valid incumbent.

### Proof: `milp_fixed_charge_4`

```
Status: optimal
Objective: 21.83  (should be 33.0)
Nodes: 11

Solution:
  y0 = 1.000 (binary, OK)
  y1 = 0.000 (binary, OK)
  y2 = 0.000 (binary, OK)
  y3 = 0.000 (binary, OK)
  x0 = 10.00, x1 = 0.77, x2 = 0.39, x3 = 1.13

Constraint violations:
  demand: x0+x1+x2+x3 = 12.29  (should be >= 15)  VIOLATED
  link_1: x1 = 0.77 <= 10*y1 = 0                   VIOLATED
  link_2: x2 = 0.39 <= 10*y2 = 0                   VIOLATED
  link_3: x3 = 1.13 <= 10*y3 = 0                   VIOLATED
```

Binary variables pass integrality check, but the constraint violations are
massive (0.4 to 2.7 absolute). The solver reports "optimal" with an
objective below the LP relaxation lower bound (21.83 < 28.0).

### Proof: `milp_bin_packing_8x3`

```
Status: optimal
Objective: 1.0  (should be 3.0, need ceil(35/15) = 3 bins)
Nodes: 391

Bins used: y = [1, 0, 0]  (only 1 bin "open")
  Items 0,1,7: NOT assigned to any bin (sum ≈ 0)   VIOLATED
  Bin 1: load=3, but y1=0, capacity=0               VIOLATED
  Bin 2: load=4, but y2=0, capacity=0               VIOLATED
```

### All 4 INCORRECT MILP instances have the same root cause

| Instance               | Solver Obj | Correct | Issue                         |
|------------------------|------------|---------|-------------------------------|
| milp_bin_packing_8x3   | 1.0        | 3.0     | Items unassigned + closed-bin load |
| milp_capital_budgeting_8| -35.0     | -33.0   | Budget constraint violated     |
| milp_fixed_charge_4    | 21.83      | 33.0    | Demand + linking violated      |
| milp_warehouse_2x4     | 76.74      | 87.0    | Demand + capacity violated     |

**The reference values in the benchmark ARE correct.** The solver is wrong.

### Bug chain

1. B&B tree branches on a binary variable, tightening its bounds
2. LP IPM at that node converges, but with constraint violations
   (the IPM tolerance allows small violations that cascade through
   linked constraints like `x_i <= M * y_i`)
3. Binary variables happen to be near-integer in the LP solution
4. B&B tree checks integrality: PASS (binaries are 0/1 within tol)
5. B&B tree does NOT check constraint satisfaction: accepts as incumbent
6. Incumbent has obj below LP lower bound (constraint violation = free lunch)
7. Tree prunes better branches using this invalid incumbent

### Fix: `_solve_milp_bb` in `solver.py` (lines 2135-2322)

Add a constraint feasibility check before passing solutions to the tree.
Evaluate all original constraints at the candidate solution. Only pass
`result_feas[i] = True` if max constraint violation < tolerance (e.g. 1e-5).

```python
# After extracting result_sols[i] = state.x[:n_vars]:
if _check_integer_feasible(result_sols[i], int_offsets, int_sizes):
    # Also check constraint satisfaction
    viol = _max_constraint_violation(model, result_sols[i])
    result_feas[i] = (viol < 1e-5)
```

Alternatively, fix in the Rust B&B tree: `process_evaluated()` should verify
`Ax = b` (or at least `||Ax - b|| < tol`) before accepting an incumbent.

## Finding 2: MIQP — No Issue

**ripopt is never called for MIQP problems.** The solver correctly classifies
MIQP and routes to `_solve_miqp_bb`, which uses a JAX QP IPM (`qp_ipm_solve`).
The `nlp_solver` parameter is completely ignored.

All MIQP instances solve correctly and quickly. The `ripopt: MaxIter diag:`
messages visible in benchmark stderr are from MINLP runs, not MIQP.

**No fix needed for MIQP.**

## Finding 3: MINLP — Genuine ripopt Convergence Issue

ripopt IS used as B&B node solver for MINLP problems. The `minlp_nvs03`
instance shows a real convergence issue:

| Instance    | ipopt  | ripopt   | ipm    |
|-------------|--------|----------|--------|
| minlp_nvs03 | 0.25s  | **421s** | 2.37s  |

ripopt returns only "feasible" (not optimal) with obj=17.0 vs correct 16.0.
The MaxIter diagnostics show complementarity stalling at co ~ 1e-2 to 1e-1
(threshold 1e-3) on B&B node subproblems.

### ripopt MaxIter Pattern (from MINLP node solves)

```
pr=6.74e-9  — primal feasible (good)
du=0.00e0   — dual feasible (good)
co=5.72e-2  — complementarity 57x above 1e-3 threshold (STUCK)
mu=5.20e-2  — barrier parameter tracking complementarity (STUCK)
ac=0        — no accepted steps (STUCK)
```

### Suggested ripopt Fixes

1. **Acceptable convergence path**: Accept solutions where co < 1e-2 after
   N consecutive near-feasible iterations. B&B only needs approximate bounds,
   not high-accuracy complementarity.

2. **Aggressive mu reduction**: When pr and du are satisfied, reduce mu by
   factor >= 0.2 per iteration regardless of complementarity progress.
   Currently mu tracks co closely, which creates a deadlock.

3. **Fixed-variable handling**: When lb[i] == ub[i] (B&B fixes integer var),
   eliminate the variable from the barrier formulation rather than creating
   degenerate barrier terms `log(x - lb) + log(ub - x)` with lb=ub.
   This likely causes the singular behavior.

4. **Warm-start from parent node**: Pass parent's NLP solution as starting
   point for child nodes. Currently each node starts from scratch.

## Finding 4: Infrastructure Issues

- **Infeasibility detection**: `lp_infeasible` and `milp_infeasible` return
  `unknown` / `optimal` instead of `infeasible`. The LP IPM doesn't detect
  infeasibility — only HiGHS does. Fix: add infeasibility detection to the
  LP IPM (check for diverging dual variables or empty feasible region).

- **Unbounded detection**: `lp_unbounded` returns `optimal` with obj=0.
  Same issue — IPM-based solvers don't detect unboundedness. Fix: check for
  diverging primal variables or unbounded ray.

## Summary

| Category | Bug?          | Root Cause                                    | Fix Location           |
|----------|---------------|-----------------------------------------------|------------------------|
| LP       | Minor         | No infeasible/unbounded detection             | `lp_ipm.py`           |
| QP       | No            | —                                             | —                      |
| MILP     | **Critical**  | Constraint-violating incumbents accepted      | `solver.py` or Rust B&B |
| MIQP     | No            | —                                             | —                      |
| MINLP    | **Moderate**  | ripopt complementarity stalling at nodes      | ripopt repo            |
