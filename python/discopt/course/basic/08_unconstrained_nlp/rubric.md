# Rubric — Lesson 8

**Total: 100**

## Exercises (70)
- E1 (15): both methods implemented; iteration counts compared (Newton ≪ GD).
- E2 (15): backtracking implemented; backtracking-iter count reported.
- E3 (15): BFGS works; B → H_true on quadratic.
- E4 (10): Cauchy point implemented; $\Delta$ adaptation visible.
- E5 (15): the response correctly identifies that $f = x^2 - y^2$ has no minimum (saddle + unbounded along $\pm y$); GD diverges to $-\infty$ from a non-axis start; Newton stalls/oscillates on the indefinite Hessian; BFGS behaviour traced to its initial $B_0$.

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

