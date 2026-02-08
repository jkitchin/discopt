# Phase 3 Mathematical Foundations -- Validation Report

Author: opt-expert
Date: 2026-02-08

This document validates the mathematical correctness of Phase 3 algorithms
(piecewise McCormick, alphaBB, RLT cuts, OA cuts, GNN branching) and records
requirements that implementations MUST satisfy.

---

## 1. Baseline Review: Existing McCormick Relaxations

Reviewed `mccormick.py` and `relaxation_compiler.py`. Key findings:

**Correct:**
- Bilinear McCormick envelopes (lines 41-56 of mccormick.py) correctly implement
  the four standard McCormick inequalities for w = x*y.
- Convex functions (exp, x^2, |x|): cv = f(x), cc = secant. Correct.
- Concave functions (sqrt, log, log2, log10): cv = secant, cc = f(x). Correct.
- Addition/subtraction/negation propagation is exact. Correct.
- Odd-power relaxation handles three regimes (nonneg/nonpos/mixed). Correct.
- Secant helper degeneracy guard (lb approx ub) at line 33. Sound.

**Observations on the relaxation compiler:**
- The compiler uses `mid = 0.5*(cv+cc)` as the evaluation point for composed
  relaxations. This is a standard approach (Tsoukalas & Mitsos 2014) that
  maintains soundness: since cv <= f(x) <= cc, using mid as the point at which
  to evaluate the next relaxation primitive produces valid (though potentially
  loose) envelopes.
- The bilinear relaxation in the compiler (line 166-173) passes cv/cc as the
  bound arguments to `relax_bilinear`. This is correct: cv_l and cc_l are valid
  lower/upper bounds on the left factor's relaxed value.

**Potential issue for Phase 3:**
- The `relax_div` function (lines 119-126) passes `recip_cv` (not `recip_cc`)
  as the "y" argument to `relax_bilinear`, but uses `recip_lb_sorted`/
  `recip_ub_sorted` as bounds. Since `recip_cv <= 1/y <= recip_cc`, and the
  bounds are computed from the original y-bounds, this is sound only when
  y > 0 or y < 0 throughout (which is enforced by the docstring).

---

## 2. Piecewise McCormick Relaxations

### Theory

Standard McCormick relaxations of bilinear terms w = x*y on [x_L, x_U] x
[y_L, y_U] converge to the convex envelope only as the variable bounds shrink
to zero width. Piecewise McCormick (Bergamini et al. 2005, Castro 2015)
partitions one variable's domain into k sub-intervals and takes the tightest
envelope across partitions.

### Convexity proof sketch

Let P_1, ..., P_k be a partition of [x_L, x_U] into sub-intervals
[x_L^i, x_U^i]. For each partition i, define:

  cv_i(x, y) = max(x_L^i * y + x * y_L - x_L^i * y_L,
                    x_U^i * y + x * y_U - x_U^i * y_U)

Each cv_i is the pointwise max of two affine functions in (x, y), hence convex.

The piecewise underestimator is:

  cv_pw(x, y) = max_{i: x in P_i} cv_i(x, y)

Since x belongs to exactly one partition at any point, cv_pw selects a single
convex function -- BUT the resulting function may not be convex globally because
different partitions are active in different regions.

**CRITICAL CORRECTION:** The piecewise McCormick relaxation is NOT simply
max over partitions evaluated at the point. The correct formulation uses
auxiliary binary variables lambda_i indicating which partition is active, plus
disaggregated variables (x_i, y_i) for each partition. This is an LP/MILP
reformulation, not a pointwise max.

For a **continuous relaxation** (without binaries), the approach is:

  cv_pw(x, y) = max_i { cv_i(x, y) }   (pointwise max over all partitions)

This IS convex because it is the pointwise maximum of finitely many convex
(affine) functions. The key insight: each cv_i is globally defined (not just on
partition i), and the pointwise max of convex functions is convex.

**Convergence:** As k -> infinity, the partition widths -> 0 and the McCormick
envelopes on each sub-interval converge to the true bilinear function. The rate
is O(1/k^2) for the relaxation gap (Hasan 2018).

### Edge cases for implementation

1. **Unbounded domains:** If x_L = -inf or x_U = +inf, partitioning is
   impossible. Must apply FBBT or user-supplied finite bounds first.
2. **Degenerate intervals:** If x_L^i = x_U^i for some partition, the bilinear
   term degenerates to a linear function. Handle gracefully.
3. **Partition count:** k should be configurable. k = 2-4 often gives good
   gap reduction for the cost. Diminishing returns for large k.
4. **Which variable to partition:** Partition the variable with the wider range,
   or allow user/heuristic selection.
5. **JAX compatibility:** The partition selection (which partition is active)
   involves a conditional. Use `jnp.where` chains or segment-based computation
   to keep the function JIT-compatible.

### Implementation requirements

- MUST: cv_i must use the sub-interval bounds [x_L^i, x_U^i] but the full y
  bounds [y_L, y_U].
- MUST: The overall envelope is max over all i of cv_i. This is convex.
- MUST: Handle k=1 as a no-op (reduces to standard McCormick).
- SHOULD: Support partitioning either x or y (choose the wider-range variable).

---

## 3. alphaBB Convex Underestimators

### Theory

For a twice-differentiable function f(x) on a box [lb, ub] in R^n, the alphaBB
underestimator (Androulakis, Maranas, Floudas 1995) is:

  f_alpha(x) = f(x) + sum_i alpha_i * (lb_i - x_i) * (ub_i - x_i)

The perturbation term alpha_i * (lb_i - x_i) * (ub_i - x_i) is a concave
quadratic in x_i that:
- Equals zero when x_i = lb_i or x_i = ub_i (vanishes at bounds)
- Has maximum magnitude alpha_i * (ub_i - lb_i)^2 / 4 at the midpoint

### Convexity guarantee

**Claim:** f_alpha is convex if alpha_i >= max(0, -lambda_min(H_f) / 2) for all i,
where H_f is the Hessian of f and lambda_min is its smallest eigenvalue over
the domain.

**Proof sketch:**

The Hessian of the perturbation term is:

  H_pert = diag(-2*alpha_1, -2*alpha_2, ..., -2*alpha_n)

So H_{f_alpha} = H_f + diag(-2*alpha_i).

Wait -- the perturbation is alpha_i * (lb_i - x_i)(ub_i - x_i) =
alpha_i * (lb_i*ub_i - (lb_i + ub_i)*x_i + x_i^2).

The second derivative w.r.t. x_i is 2*alpha_i (positive!). So:

  H_{f_alpha} = H_f + diag(2*alpha_1, ..., 2*alpha_n)

For f_alpha to be convex, we need H_{f_alpha} >= 0 (positive semidefinite).

By Gershgorin or eigenvalue shifting: if alpha_i >= max(0, -lambda_min(H_f)/2),
then the minimum eigenvalue of H_{f_alpha} >= lambda_min(H_f) + 2*alpha_i >= 0.

**IMPORTANT CORRECTION:** The standard result uses a UNIFORM alpha:

  alpha >= max(0, -lambda_min(H_f) / 2)

where lambda_min is the global minimum eigenvalue of H_f over the entire box.
Using per-variable alpha_i is a refinement (interval Hessian approach) that
requires bounding each diagonal element of H_f plus off-diagonal contributions.

The simpler uniform-alpha approach is:

  f_alpha(x) = f(x) + alpha * sum_i (lb_i - x_i)(ub_i - x_i)

with alpha = max(0, -lambda_min(H_f) / 2).

### Bound vanishing property

At any vertex of the box where x_i in {lb_i, ub_i} for all i:

  f_alpha(x) = f(x) + 0 = f(x)

So the underestimator is tight at all box vertices. More generally, for any
face where x_j = lb_j or x_j = ub_j, the perturbation in direction j vanishes.

This means f_alpha(x) <= f(x) for all x in [lb, ub], with equality at vertices.

**Proof that f_alpha <= f:** Each term alpha_i*(lb_i - x_i)*(ub_i - x_i) <= 0
for x_i in [lb_i, ub_i] (since (lb_i - x_i) <= 0 and (ub_i - x_i) >= 0, or
vice versa, so the product is <= 0). Thus f_alpha(x) = f(x) + (nonpositive) <= f(x).

### Sampling-based alpha estimation

Computing lambda_min(H_f) exactly over a box is itself a global optimization
problem. A practical approach samples the Hessian at multiple points and takes:

  alpha = max(0, -min_sample(lambda_min(H_f(x_s))) / 2) * safety_factor

**CRITICAL:** The sampling-based estimate MUST be conservative (overestimate
alpha). If alpha is too small, f_alpha may not be convex, breaking the
relaxation's validity.

Requirements:
- MUST: Use safety_factor >= 1.0 (recommend 1.5-2.0).
- MUST: Include box vertices in the sample set (Hessian extremes often at
  boundaries).
- SHOULD: Use interval arithmetic on the Hessian for a rigorous bound when
  feasible.
- MUST: Recompute alpha when bounds change (tighter bounds -> smaller alpha
  -> tighter relaxation).
- SHOULD: Use JAX's `jax.hessian` for automatic Hessian computation.

### Edge cases

1. **Linear functions:** H_f = 0, so alpha = 0 and f_alpha = f. Correct.
2. **Already convex functions:** lambda_min >= 0, so alpha = 0. Correct.
3. **Near-singular Hessian:** lambda_min very close to 0. Use a small positive
   alpha for numerical safety.
4. **Large domains:** The perturbation magnitude scales as alpha * (ub-lb)^2/4,
   which can be very large. alphaBB works best on small subproblems within B&B.

---

## 4. RLT (Reformulation-Linearization Technique) Cuts

### Theory

For a bilinear term w = x*y with x in [x_L, x_U] and y in [y_L, y_U], the
RLT cuts are precisely the four McCormick inequalities:

  w >= x_L*y + x*y_L - x_L*y_L    (1)
  w >= x_U*y + x*y_U - x_U*y_U    (2)
  w <= x_U*y + x*y_L - x_U*y_L    (3)
  w <= x_L*y + x*y_U - x_L*y_U    (4)

These define the convex hull of {(x, y, xy) : x in [x_L, x_U], y in [y_L, y_U]}.

### Validity proof

Consider inequality (1): w >= x_L*y + x*y_L - x_L*y_L.

Rearranging: w - x_L*y - x*y_L + x_L*y_L >= 0, i.e.,
(x - x_L)(y - y_L) >= 0.

Since x >= x_L and y >= y_L, both factors are nonneg, so the product is >= 0.
Valid for ANY (x, y) in the box. QED.

Similarly:
- (2): (x_U - x)(y_U - y) >= 0. Valid since x <= x_U and y <= y_U.
- (3): (x_U - x)(y - y_L) >= 0. Valid.
- (4): (x - x_L)(y_U - y) >= 0. Valid.

### Key property: NO convexity assumption needed

RLT cuts are valid linear inequalities for the set {(x, y, w) : w = xy,
x in [x_L, x_U], y in [y_L, y_U]}. They do not require f to be convex or
concave. They are always valid.

### Implementation requirements

- MUST: Cuts are added as linear constraints to the LP relaxation.
- MUST: Auxiliary variable w_ij must be introduced for each bilinear term x_i*x_j.
- SHOULD: Identify bilinear terms during preprocessing (expression analysis).
- SHOULD: Tighten bounds via FBBT before generating RLT cuts (tighter bounds
  -> tighter relaxation).
- MUST: Update cuts when bounds change during B&B (the cut coefficients depend
  on variable bounds).

---

## 5. Outer Approximation (OA) Cuts

### Theory

For a convex constraint g(x) <= 0, a tangent (OA) cut at point x_0 is:

  g(x_0) + nabla g(x_0)^T (x - x_0) <= 0

This is valid because g is convex, so g(x) >= g(x_0) + nabla g(x_0)^T (x - x_0)
for all x. Therefore any x satisfying g(x) <= 0 must also satisfy the cut.

### CRITICAL: Convexity requirement

**OA cuts are ONLY valid for convex constraints.**

For a nonconvex g, the tangent plane can cut off feasible points. Example:
g(x) = -x^2 + 1 (concave). At x_0 = 0: g(0) = 1, g'(0) = 0, so the OA cut
is 1 <= 0, which is infeasible -- but g(x) <= 0 is feasible for |x| >= 1.

**Implementation MUST enforce one of:**
1. Only generate OA cuts for constraints verified to be convex, OR
2. Apply convexification (alphaBB) to nonconvex constraints first, then
   generate OA cuts on the convexified version, OR
3. Use OA cuts only within a convex relaxation framework where the constraint
   functions have been replaced by their convex relaxations.

### Implementation requirements

- MUST: Check convexity before generating OA cuts. If the constraint is
  nonconvex, refuse or convexify first.
- MUST: Use JAX autodiff (`jax.grad`) for gradient computation.
- SHOULD: Add OA cuts iteratively (solve LP, find most violated constraint,
  add cut, re-solve).
- SHOULD: Limit the number of OA cuts per node to avoid LP bloat.
- MUST: For equality constraints h(x) = 0 where h is nonlinear:
  - If h is convex: add OA cut h(x_0) + nabla h(x_0)^T (x - x_0) <= 0
  - If h is concave: add OA cut h(x_0) + nabla h(x_0)^T (x - x_0) >= 0
  - If h is neither: do NOT use OA cuts directly. Use RLT or alphaBB.

### Interaction with McCormick relaxations

In the current architecture, McCormick relaxations produce convex underestimators
for the objective and constraint bodies. OA cuts can be generated on these
RELAXED functions (which are convex by construction). This is the recommended
approach: it avoids the need for separate convexity verification.

---

## 6. GNN Branching Policy

### Background

Gasse et al. (2019, NeurIPS) proposed learning branching policies via GNNs
on the bipartite graph representation of LP relaxations. The graph has:
- Variable nodes (features: LP value, reduced cost, objective coefficient, etc.)
- Constraint nodes (features: slack, dual value, RHS, etc.)
- Edges from variable i to constraint j if variable i appears in constraint j

The GNN predicts a score for each candidate branching variable; the variable
with the highest score is selected.

### Mathematical validity

GNN branching is a HEURISTIC -- it does not affect correctness of B&B, only
efficiency. As long as:
1. The selected variable is a valid branching candidate (integer variable with
   fractional value), AND
2. The B&B tree exploration is complete (all nodes eventually processed or
   pruned)

...then correctness is preserved regardless of branching quality.

### Key risks

1. **Training distribution mismatch:** Models trained on small instances may
   perform poorly on larger ones. The graph structure changes qualitatively
   with problem size.
   - Recommendation: Include problem-size features (num_vars, num_constraints,
     density) as global graph features.
   - Recommendation: Train on a range of problem sizes.

2. **Feature normalization:** LP values and dual values can vary by orders of
   magnitude across different problems.
   - Recommendation: Normalize features per-instance (e.g., divide by range).

3. **Inference latency:** GNN inference must be fast enough that the overhead
   doesn't negate the tree-size reduction.
   - Recommendation: Use small GNNs (2-3 layers, hidden dim 64).
   - Recommendation: Batch inference across multiple candidate variables.

4. **Fallback:** If the GNN produces degenerate scores (all equal, NaN, etc.),
   fall back to most-fractional branching (the existing strategy in branching.rs).

### Implementation requirements

- MUST: Fall back to most-fractional if GNN is unavailable or produces invalid output.
- MUST: Only select from valid branching candidates (integer variables with
  fractional values, same filter as `select_branch_variable`).
- SHOULD: Use the bipartite graph representation (Gasse et al. 2019).
- SHOULD: Include problem size as a feature to mitigate distribution shift.
- SHOULD: Cap inference time; if GNN is too slow, fall back to heuristic.

---

## 7. Review of Existing Branching Code (branching.rs)

The existing `select_branch_variable` in `branching.rs` implements
most-fractional branching:

- **Correct:** Fractionality score `0.5 - |frac - 0.5|` is maximized at
  frac = 0.5, which is the most ambiguous value. This is the standard
  most-fractional heuristic.
- **Correct:** Integrality tolerance of 1e-5 matches the project's tolerance
  specification in CLAUDE.md.
- **Correct:** `create_children` uses floor(val) for left (x <= floor) and
  floor(val)+1 for right (x >= ceil). Standard binary branching.
- **Correct:** `is_integer_feasible` uses the same tolerance for consistency.

**Note for GNN integration:** The GNN branching policy should replace or
wrap `select_branch_variable` but MUST maintain the same interface: given a
solution and variable info, return an `Option<BranchDecision>`. The
`create_children` function remains unchanged.

---

## 8. Summary of Critical Requirements

| Algorithm | Critical Requirement | Risk if Violated |
|-----------|---------------------|------------------|
| Piecewise McCormick | Compute max over all partition envelopes | Invalid (non-convex) relaxation |
| Piecewise McCormick | Finite bounds required | Cannot partition infinite domain |
| alphaBB | alpha >= max(0, -lambda_min/2) | Non-convex underestimator |
| alphaBB | Sampling must OVERESTIMATE alpha | Unsound relaxation |
| alphaBB | Perturbation uses (lb-x)(ub-x) form | Fails to vanish at bounds |
| RLT | Update cuts when bounds change | Stale/invalid cuts |
| OA | ONLY apply to convex constraints | Cuts off feasible region |
| OA | Or apply to McCormick-relaxed (convex) functions | |
| GNN | Fall back to most-fractional on failure | B&B stalls or crashes |
| GNN | Only branch on valid candidates | Incorrect tree |

---

## 9. References

- Androulakis, Maranas, Floudas (1995). alphaBB: A global optimization method
  for general constrained nonconvex problems. J. Global Optim.
- Bergamini, Aguirre, Grossmann (2005). Logic-based outer approximation for
  globally optimal synthesis of process networks. Comput. Chem. Eng.
- Castro (2015). Tightening piecewise McCormick relaxations for bilinear
  problems. Comput. Chem. Eng.
- Gasse, Chetelat, Ferroni, Charlin, Lodi (2019). Exact combinatorial
  optimization with graph convolutional neural networks. NeurIPS.
- Hasan (2018). An edge-concave underestimator for the global optimization of
  twice-differentiable nonconvex problems. J. Global Optim.
- McCormick (1976). Computability of global solutions to factorable nonconvex
  programs. Math. Prog.
- Tsoukalas, Mitsos (2014). Multivariate McCormick relaxations. J. Global Optim.
