# Rubric — Lesson 1

**Total: 100 points** (Exercises: 70, Writing: 30)

## Exercises (70 pts)

### Exercise 1 — Classification (10 pts)
- (4) All four problems classified correctly.
- (3) Justification names a specific structural feature.
- (3) Subtle cases noted (e.g., (a) is a convex SDP/SOCP, not LP).

### Exercise 2 — Diet with extras (15 pts)
- (5) Four food variables + binary indicators.
- (5) "≥ 2 foods" constraint via $\sum z_i \ge 2$ with $x_i \le M z_i$.
- (3) Solves to optimal, prints purchased foods + cost.
- (2) Numerical answer matches reference within 1e-3.

### Exercise 3 — LP relaxation gap (15 pts)
- (4) LP relaxation correctly solved (continuous $[0,1]$).
- (4) MILP correctly solved.
- (3) Gap formula correct.
- (4) Discussion connects gap to B&B difficulty.

### Exercise 4 — Six minima (15 pts)
- (5) Grid of ≥ 25 starts.
- (5) Six distinct local minima recovered after deduplication.
- (5) Two global minima identified at $f^\star \approx -1.0316$.

### Exercise 5 — Read the log (15 pts)
- (5) Primal residual = constraint violation.
- (5) Dual residual = ‖∇L‖.
- (5) Complementary slackness = $\lambda_i g_i(x) = 0$.

## Writing (30 pts)

- **Clarity (10):** explicit variables/objective/constraints, clean prose.
- **Technical correctness (10):** classification correct *and* justified;
  no false statements about complexity.
- **Citations (5):** at least one recommended reference; all keys resolve
  in `docs/references.bib`.
- **Engagement (5):** the surprise reflection is specific.

**Pass threshold: 70 / 100.**
