# Robust Optimization Roadmap

Current state: v1 with box, ellipsoidal, and polyhedral static robust
counterparts, plus affine decision rules for two-stage adjustable RO.
Verified against Ben-Tal et al. (2009) Example 1.1.1.

## Known Limitations

**Solver convergence.** The LP/QP IPMs now use gradient-based scaling,
a mu error gate, and have the z-recovery removed, which improved
convergence on poorly-scaled problems (e.g., Ben-Tal drug example).
However, the NLP IPM lacks scaling, and variables without explicit
upper bounds still converge slowly due to the default bound creating
a distant barrier term. The ellipsoidal reformulation produces SOCP
constraints solved as general NLP; a dedicated SOCP solver would be
faster and more reliable.

**ADR practical value.** The affine decision rule implementation is
mathematically correct but no example yet demonstrates a strict improvement
over static robust. For simple two-variable LPs with single-parameter
uncertainty, ADR provably cannot beat static. A multi-constraint or
multi-period example is needed to demonstrate the value.

**Expression tree complexity.** The coefficient extraction approach
(substitute p with p_bar+1, subtract) creates verbose unsimplified
expression trees. An expression simplifier would improve solver performance
on reformulated models.

**Mixed-sign vector constants (fixed, #1831).** Sign tracking used
`np.sign(np.sum(value))` for a constant vector, so a mixed-sign multiplier
certified non-robust points. Constant multipliers (`*`, `/`, `@`) are now
signed element-wise; a mixed-sign one is split into its positive and negative
parts. A parameter element that enters one row through several coefficients of
opposite sign is worst-cased per occurrence -- sound, possibly conservative.

## Future Work (roughly by priority)

1. **SOCP solver interface.** Route ellipsoidal reformulations to MOSEK or
   ECOS instead of the general NLP IPM. Would also enable direct handling
   of the absolute-value auxiliary constraints from bilinear ADR terms.

2. **Multi-period ADR example.** Inventory management over T periods with
   demand uncertainty, where the recourse at period t responds to demands
   observed in periods 1..t-1. This is the canonical case where ADR
   strictly beats static robust.

3. **Distributionally robust optimization (DRO).** Moment-based ambiguity
   sets (mean and covariance known) and Wasserstein ambiguity sets
   (distributional ball around empirical distribution). See Delage &
   Ye (2010), Mohajerin Esfahani & Kuhn (2018).

4. **Piecewise-linear and polynomial decision rules.** Extend beyond
   affine rules for better approximation of the optimal recourse. See
   Georghiou et al. (2015).

5. **Multi-stage ARO.** Nested information revelation with decision rules
   at each stage. Requires careful handling of non-anticipativity
   constraints.

6. **Expression simplification.** Simplify expression trees after
   coefficient extraction and parameter substitution to reduce solver
   burden. Constant folding, zero elimination, and common subexpression
   detection.

7. **Integer first-stage with continuous recourse.** The ADR currently
   requires continuous recourse variables. Supporting binary/integer
   first-stage with continuous affine recourse would enable facility
   location and network design under uncertainty.

8. **Sparse polyhedral duality.** The current LP duality implementation
   introduces dual variables for every polytope row per constraint.
   Exploiting sparsity in A^T lambda = coeff(x) would reduce model size
   for large uncertainty sets.

9. **Automatic uncertainty set selection.** Given historical data, fit
   box/ellipsoidal/polyhedral sets with specified coverage probability.
   See Bertsimas et al. (2018) "Data-driven robust optimization."

## IPM Solver Improvements

These are general solver improvements discovered during RO development
that affect all problem types, not just robust optimization.

10. **Route LPs to HiGHS when available.** The custom LP IPM is a
    research implementation. HiGHS is a production simplex/IPM that
    handles edge cases (degeneracy, unbounded variables, poor scaling)
    robustly. The solver already uses HiGHS for QP; extending it to
    LP would give production-quality LP solving immediately. The
    custom IPM would remain as a fallback when HiGHS is not installed.

11. **Acceptable convergence for LP/QP IPMs.** The NLP IPM has an
    "acceptable" convergence path (code 2) that triggers after 15
    consecutive iterations within a loose tolerance. The LP and QP
    IPMs lack this: they only return optimal or iteration_limit.
    Adding acceptable convergence would catch cases where the solution
    is correct but formal optimality is not certified within max_iter.

12. **NLP IPM scaling.** The LP/QP IPMs now have gradient-based
    scaling (scaling.py). The NLP IPM needs the same treatment, but
    it is harder because scaling factors depend on the evaluation
    point and the steepest-descent recovery path interacts with the
    scaling. Requires evaluating the gradient and Jacobian at the
    initial point, computing row scaling factors, and wrapping the
    evaluator functions.
