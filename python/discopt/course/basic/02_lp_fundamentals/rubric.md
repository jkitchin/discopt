# Rubric — Lesson 2

**Total: 100** (Exercises: 70, Writing: 30)

## Exercises
- E1 (10): Standard-form conversion correct. Specifically: (i) the free variable $x_1$ is split into $x_1^+ - x_1^-$ with both $\ge 0$; (ii) one slack added per inequality (and the $\ge$ inequality flipped to $\le$ before slack-adding); (iii) the resulting $A x = b, x \ge 0$ system has the same primal optimum as the original LP, verified numerically.
- E2 (15): LP solved; primal/dual objectives equal within 1e-9; identifies tight constraints.
- E3 (15): KM correctly built; objectives are $100^{n-1}$.
- E4 (15): Cost change equals $y_3 \cdot \Delta b_3$ within solver tol.
- E5 (15): Three LPs with correct statuses; brief explanation references duality / Farkas.

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

