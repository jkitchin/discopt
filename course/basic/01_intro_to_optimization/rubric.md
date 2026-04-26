# Rubric — Lesson 1

**Total: 100 points** (Exercises: 70, Writing: 30)

## Exercises (70 pts)

### Exercise 1 — Classification (10 pts)
- (4) All four problems classified correctly.
- (3) Justification names a specific structural feature.
- (3) Subtle cases noted (e.g., (a) is a convex SDP/SOCP, not LP).

### Exercise 2 — Diet with eggs (15 pts)
- (6) Four continuous food variables with $0 \le x_i \le 8$ (per-food cap).
- (5) Cost vector and nutrient matrix correctly extended to 4 foods.
- (2) Solver returns `OPTIMAL`; prints purchased quantities and cost.
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

### Exercise 5 — Read the solver result (15 pts)
- (4) Reports status, objective, iteration/node count, wall time for the diet LP.
- (5) Constructs an *infeasible* variant (e.g., $b_3 = 10^{12}$); solver returns `INFEASIBLE`.
- (5) Constructs an *unbounded* variant (e.g., drop bounds + minimize $-\sum x$); solver returns `UNBOUNDED`.
- (1) Briefly notes that "unexpectedly unbounded" usually means a missing bound.

## Writing (30)

- **Concept coverage (10):** the response names ≥ 3 specific concepts from
  this lesson with a one-line definition or formula each, AND references at
  least one numerical result the student produced in the exercise notebook
  (objective value, runtime, gap, plot point).
- **Technical correctness (10):** every claim about complexity, an algorithm,
  or a solver's behaviour is either a citation or a derivation; nothing
  factually wrong per the assessor's checklist for this lesson.
- **Citations (5):** ≥ 2 `{cite:p}` keys resolve in `docs/references.bib`,
  at least one of which is a recommended reference for this lesson; no
  invented or unverifiable keys.
- **Engagement (5):** the response cites a specific surprise, equation,
  measurement, or solver-log line the student personally observed — not a
  generic platitude.

