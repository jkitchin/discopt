# MO Module Review — Correctness, Thoroughness, Performance, and SOTA Assessment

**Date:** 2026-07-03
**Scope:** `python/discopt/mo/` (`scalarization.py`, `nbi.py`, `pareto.py`,
`indicators.py`, `utils.py`, `__init__.py` — 1,783 lines) and its tests
(`test_mo_scalarization.py`, `test_mo_nbi.py`, `test_mo_pareto.py`,
`test_mo_indicators.py`).
**Method:** Full read of all six files; every suspected defect either reproduced
numerically against the installed package or verified/refuted by direct computation.
Baseline: fast mo tests green (30 passed); slow scalarization/NBI suite result
recorded in §5.

This module is in notably better shape than its peers reviewed earlier
(`docs/dev/dae-module-review.md`, `docs/dev/modeling-module-review.md`): the sense
handling, dominance filtering, and all four quality indicators were checked and came
out **correct**, several of them exactly. The findings are one method-fidelity bug,
a set of methodological gaps against the literature the docstrings themselves cite,
and test blind spots.

---

## 1. Summary of findings

> **✅ RESOLVED MO1** — `python/tests/test_display_cluster.py` (this PR). The
> quasi-normal is extracted into a `_quasi_normal(phi)` helper computing the
> Das–Dennis `n̂ = -Φe = -phi.sum(axis=0)` (negated sum of the anchor rows), fixing
> the `axis=1` slip. Verified fails-before/passes-after; the k=2 case is unchanged
> (symmetric payoff). The full finding text is preserved below.

| # | Severity | Component | Finding |
|---|----------|-----------|---------|
| MO1 | **P1 method-correctness** — ✅ RESOLVED | `nbi.py:128` | NBI quasi-normal computed with the **wrong axis** (`phi.sum(axis=1)` instead of `axis=0`) — deviates from the cited Das–Dennis (1998) formula for k ≥ 3; invisible at k = 2 by symmetry [CONFIRMED] |
| MO2 | P2 method-fidelity | `scalarization.py` | The `epsilon_constraint` implementation is **AUGMECON, not AUGMECON2**: the bypass/jump acceleration that defines AUGMECON2 is absent, and the payoff table is not built lexicographically as Mavrotas prescribes [BY INSPECTION] |
| MO3 | P2 | `pareto.py` | `filtered()` keeps **duplicate points** (identical objective vectors from different scalarization parameters survive weak-dominance filtering) [CONFIRMED] |
| MO4 | P2 (doc/API) | `indicators.py` | Default hypervolume reference is **front-dependent** — comparing two fronts via `front.hypervolume()` silently compares against different references [CONFIRMED: 2.16 vs 0.08 for nested fronts] |
| MO5 | P2 perf | `utils.py`, sweep loops | `evaluate_expression` recompiles (JAX trace) each objective per call — k compilations per accepted point per sweep, plus k² for the payoff table, of the *same* k expressions |
| MO6–MO10 | P2–P3 | various | Robustness/API gaps (§3) |

Checked and found **correct** (no action):

- **Sense handling end-to-end** [VERIFIED numerically]: `weighted_sum` with
  `senses=["max","max"]` produces a front *bit-identical* to the equivalent
  min-form problem; `epsilon_constraint` with mixed `["min","max"]` senses recovers
  the analytic Pareto curve of the test QP to machine precision (max deviation 0.0).
  This is the hardest thing to get right in a scalarization layer and it is right —
  the sign conventions in `_as_senses`/`signs`/`span`, the flipped ε-inequalities,
  and the reversed ε-grids for max-sense objectives all compose correctly.
- **AUGMECON slack/penalty algebra**: `sign·f + s ≤ sign·ε` with the `−δ·Σ s/r`
  reward makes the inequality bind, reproducing the equality form; δ-normalization
  by objective ranges matches Mavrotas.
- **Weighted-Tchebycheff form** (aug. `t ≥ w·g` constraints + ρ-augmentation) and
  **NNC** (normals `N_i = anchor_last − anchor_i`, `N·(g − X^p) ≤ 0` cuts,
  minimize last normalized objective) both match their cited references.
- **NBI CHIM target** `b = Φᵀw` is correct for the code's row layout — which is
  what makes MO1's axis slip on the *other* Φ-product identifiable as a bug.
- **Dominance filter** (`filter_nondominated`): the "skip already-dominated
  dominators" optimization is safe by transitivity; equal points cannot eliminate
  each other (no strict component), so no false removals.
- **All four indicators**: 2-D hypervolume staircase, HSO recursion (slice
  bookkeeping, final slice, per-slice nondominance pruning), Monte-Carlo HV
  (sampling box provably contains the dominated region; deterministic default rng),
  IGD, additive-ε, and both spread variants match their textbook definitions.
- `ideal_point` fails **loudly** (RuntimeError) when an anchor solve fails, and
  restores the model objective in a `finally`. Parameter mutation between sweep
  solves is safe: `compile_expression` snapshots parameter values at compile time
  and the sweeps recompile via fresh solves (the params-threaded compile path keeps
  the XLA cache warm across value changes).

---

## 2. The confirmed method bug

### MO1. NBI quasi-normal uses the wrong matrix axis (k ≥ 3)

`nbi.py` stores the normalized payoff in row layout — `phi[i, j]` = objective *j*
at anchor *i* — and correctly computes CHIM points as `b = phi.T @ w`. But the
quasi-normal is

```python
n_hat = -phi.sum(axis=1)     # sums the components WITHIN each anchor
```

Das–Dennis define `n̂ = −Φe`, the negated **sum of the anchor payoff vectors**,
which in this layout is `-phi.sum(axis=0)`. The two agree only when `phi` is
symmetric — which the ideal/nadir-normalized payoff always is for **k = 2**
(`[[0,1],[1,0]]`), exactly the only case the test suite exercises.

**Verification.**
- Pure math: for a representative asymmetric k=3 normalized payoff, the two
  directions differ by 8.6°.
- End-to-end (tri-objective quadratic, asymmetric centers, 10 CHIM weights, both
  variants solved to completion): both produce 10 feasible nondominated boundary
  points, but the *sampled fronts differ* — max nearest-point distance 0.604
  between the two point clouds in normalized objective space. So the returned
  points are valid Pareto candidates either way (the equality constraint always
  lands on the attainable boundary, and the dominance filter guards the output);
  what breaks is the **Das–Dennis placement/uniform-spacing property** the method
  exists to provide, and which the module's own docstring promises.

**Fix:** `n_hat = -phi.sum(axis=0)` (one token), plus a regression test at k = 3
asserting `n_hat == -(sum of anchor rows)` on an asymmetric payoff, and an
end-to-end tri-objective test (the suite currently has none — §5).

---

## 3. Methodological and robustness gaps

### MO2. "AUGMECON2" is actually AUGMECON

The front tag and docstring say AUGMECON2 [Mavrotas 2009/2013], and the augmented
slack objective with range normalization is implemented correctly. But the two
things that *distinguish* AUGMECON2 from AUGMECON are missing:

1. **The bypass (jump) acceleration**: using the slack value of the just-solved
   subproblem to skip ε-grid points that provably yield the same solution. Without
   it the sweep always performs `n_points^(k-1)` solves; with it, dense grids on
   flat front regions are skipped wholesale (the headline 2–10× saving of the
   AUGMECON2 paper).
2. **Lexicographic payoff table**: Mavrotas prescribes lexicographic optimization
   for the anchors so the payoff rows are Pareto-optimal. `ideal_point` does plain
   single-objective solves; with alternative optima, `f_j` at an anchor can be
   arbitrarily worse than the Pareto-worst, inflating the nadir estimate and the
   ε-grid (wasted solves; and for k ≥ 3 the payoff-table nadir can also
   *underestimate*, truncating the grid and missing front regions — the classic
   payoff-table failure mode, [Miettinen 1999]).

Fix: either implement both (rename stays) or rename the tag/docs to `augmecon`
(honest-labeling per house philosophy). The lexicographic payoff is the more
valuable of the two for correctness of *coverage*; the bypass is the perf win.
Both are well-specified in the cited papers.

### MO3. Fronts accumulate duplicate points

`filtered()` removes only strictly dominated points; identical objective vectors
(the same optimum found at several weights — routine for anchors and flat regions)
all survive [CONFIRMED: 3 identical + 1 distinct + 1 dominated → `filtered().n == 4`].
Consequences: inflated `front.n`, distorted `spread()` (zero-distance neighbors),
misleading summaries. Fix: tolerance-based dedup in `filtered()` (keep first
occurrence; `np.allclose` on objective vectors with a documented tol), preserving
`scalarization_params` of the kept representative.

### MO4. Default hypervolume reference is front-dependent

`front.hypervolume()` with no reference derives one from the front's own
nadir/worst + 1 % margin. Two fronts for the *same problem* therefore get
different references and **incomparable** hypervolumes [CONFIRMED: nested fronts
scored 2.16 vs 0.08 under defaults]. The natural use of the module (comparing
`weighted_sum` vs `nbi` fronts, as the `__init__` docstring suggests) hits this
silently. Fix: document loudly; add
`hypervolume(fronts..., reference=None)`-style shared-reference helper or a
`ParetoFront.compare(other)` that constructs one common reference from the union.

### MO5. Repeated JAX recompilation in sweeps

`evaluate_expression` calls `compile_expression` per invocation;
`_collect_objectives_at_x` calls it k times per accepted point, `_payoff_matrix`
k² times. Each is a fresh DAG walk + trace of expressions that never change during
a sweep. Fix: compile the k objectives once per sweep (via
`compile_expression_params`, which is parameter-value-agnostic) and reuse; this
also removes the per-point Python DAG-walk cost. Measure before/after on a
21-point sweep per the house rule.

### MO6–MO10 (smaller, [BY INSPECTION])

- **MO6.** `weighted_sum(ideal=...)` without `anchors` silently *recomputes* both
  (the provided ideal is discarded) — surprising and costs k solves; accept
  ideal-only when `normalize=False`, or document.
- **MO7.** No overall time budget: a sweep issues `n_points^(k-1)` solves each with
  the full per-solve `time_limit` (default 3600 s) — a 20-point bi-objective sweep
  can legitimately take 20 hours with no warning and no partial-result return.
  Add `total_time_limit` (stop the sweep, return the partial front with a status
  note) and warn when `n_points ** (k-1)` exceeds a threshold.
- **MO8.** Scalarizer side effects accumulate: every call leaves its aux
  slacks/`t`/parameters on the model permanently (documented), so comparing three
  methods on one model triples the dangling variables; there is no cleanup API
  because the modeling layer has no variable/constraint removal (see
  modeling-review Phase 4 — this is a concrete consumer for it). Also
  `del model._constraints[saved_n_cons:]` in the `finally` assumes nothing else
  appended constraints during the sweep — brittle if callbacks/reformulations ever
  mutate `_constraints` at solve time; slice out exactly the constraints added by
  identity instead.
- **MO9.** `_payoff_matrix` takes a `senses_list` argument it never uses (dead
  parameter — confusing given how sense-critical this code is).
- **MO10.** Parameter-carrying constraints are dropped from AMP's MILP relaxation
  with a per-constraint warning ("cannot be linearized safely: Cannot linearize
  Parameter"), observed live during the NBI k=3 runs. Sound (dropping rows only
  loosens a relaxation) but it means every ε-constraint/NBI/NNC/Tchebycheff
  subproblem solved via AMP runs with its *defining* scalarization constraint
  absent from the relaxation — bounds are weak exactly where the sweep needs them.
  Worth a targeted fix in the AMP linearizer: a `Parameter` is a constant at
  relaxation-build time; substitute its current value (and rebuild per sweep
  iteration, or thread it as a bound shift) instead of omitting the row.

---

## 4. Performance notes

- MO5 (recompilation) is the dominant per-point overhead; the solves themselves
  dominate wall time, so this matters most for cheap subproblems (LPs/QPs).
- `_simplex_lattice` builds the Das-Dennis grid by recursive enumeration —
  exponential in k but k ≤ 5 in practice; fine.
- Sweeps are strictly sequential because warm starts chain solve i → i+1. That is
  a reasonable default, but the subproblems are independent; a `parallel=True`
  option (thread pool over grid points, warm-starting each from the nearest anchor
  instead of the previous point) is the standard way tools scale sweeps. Requires
  model-copy support (again modeling-review Phase 4) since scalarizers mutate the
  model.

---

## 5. Test coverage gaps

Baseline: fast suite (indicators/pareto) 30 passed; slow suite
(scalarization/NBI) — see repository CI record for this branch date
(all previously passing; re-run performed during this review).

1. **No k ≥ 3 test anywhere** in the module's suite — MO1 is invisible at k = 2 by
   symmetry, and every solver-backed test uses the same two bi-objective toys. Add
   the asymmetric tri-objective QP used in this review (analytic front unknown but
   anchor/payoff quantities are checkable, and the n̂ formula is directly
   assertable).
2. **No `senses="max"` or mixed-sense solver test.** The sense math is currently
   correct (verified here) but nothing pins it; the min↔max bit-identity check in
   §1 is cheap and exact — add it.
3. **No duplicate-point test** (MO3) and no shared-reference hypervolume
   comparison test (MO4).
4. **No test of the AUGMECON payoff/grid coverage** on a problem where
   non-lexicographic anchors actually differ (alternative optima at an anchor).
5. Indicators are tested (good coverage of HV 2-D/3-D/MC, IGD, spread, ε) — the
   one missing case is HSO with duplicate coordinates and a point exactly on the
   reference boundary.

---

## 6. SOTA assessment

### 6.1 What the module is

A **scalarization-sweep layer over a global MINLP solver** — the classic a
posteriori "generate the front by repeated single-objective solves" architecture,
with the five canonical methods (weighted sum, ε-constraint/AUGMECON, augmented
weighted Tchebycheff, NBI, NNC), payoff-table utilities, dominance filtering, and
the four standard quality indicators. The method selection is exactly the right
canon for exact solvers (matches Pyomo's community practice, GAMS's
`$batinclude`-style sweeps, and MATLAB's paretosearch-adjacent workflows), the
guidance docstring ("when to use which") is accurate, and — unusually — the
**augmentation terms that guarantee strict Pareto optimality are implemented
correctly** in both AUGMECON and Tchebycheff forms, which many ad-hoc
implementations omit.

Because every subproblem is solved by a *global* solver, the returned points carry
per-point global-optimality status — a genuine differentiator versus the
metaheuristic ecosystem (pymoo/NSGA-II etc.), which offers no such guarantee. For
nonconvex MINLP fronts this "exact scalarization" architecture *is* the current
state of practice.

### 6.2 Where it stands vs. the state of the art

- **vs. pymoo / evolutionary MO**: different paradigm (approximate, derivative-free,
  population-based). discopt.mo wins on certification and constraint handling;
  pymoo wins on many-objective (k > 4) scale and indicator/algorithm breadth. The
  module's k ≤ 3-4 sweet spot is appropriate and honestly documented (grid growth
  is `n^(k-1)`).
- **vs. the exact-MO literature**: the missing tier is **criterion-space search
  algorithms** — bi-objective MILP methods (balanced box, ε-tabu,
  Boland-Charkhgard-Savelsbergh) and MOMILP branch-and-bound that compute the
  *entire nondominated set exactly* rather than sampling it on a grid. For
  MILP-class models these are strictly stronger than any scalarization sweep and
  are the research SOTA. A natural discopt fit given the in-house B&B, but a
  large work item — record as a Phase-3 direction, entry experiment first.
- **vs. AUGMECON2/interactive practice**: MO2's two gaps (bypass, lexicographic
  payoff) are the concrete distance from the ε-constraint state of practice. No
  interactive/reference-point methods (NIMBUS-style) — reasonable scope cut.
- **Indicators**: exact HV to k=3 + MC beyond is the standard tier; SOTA exact HV
  (WFG/QuickHV) only matters for k ≥ 4 at large n — the MC fallback is an
  acceptable answer, and it is deterministic by default (good for CI).
- **Warm-start chaining** across the sweep is a nice solver-aware touch most
  wrappers lack.

### 6.3 Verdict

The mo module is a **clean, literature-faithful (with the two exceptions above),
correctly sense-handled implementation of the classic exact-scalarization
architecture** — the right design for a global solver, with the strongest part
being what's usually botched (sense conventions, augmentation terms, indicator
math) and the weak spots being exactly where the test suite is blind (k ≥ 3, max
senses). Fix MO1 (one line), decide the AUGMECON2 question honestly (implement or
rename), add dedup and the shared-reference comparison helper, and the module is
at the state of practice. The step *beyond* — exact criterion-space methods for
MOMILP — is the one genuinely SOTA-level opportunity and should get its own design
round.

---

## 7. Implementation plan (for Opus)

House rules per CLAUDE.md apply (feature branch + PR, task ID in title, regression
test failing-before/passing-after, run `pytest python/tests/test_mo_*.py` fast+slow
and `-m smoke`; state results in the PR).

### Phase 1 — method correctness (PR `fix(mo): MO-1..MO-2`)

| ID | Task | Files | Acceptance criteria |
|----|------|-------|---------------------|
| MO-1 | NBI quasi-normal: `phi.sum(axis=1)` → `phi.sum(axis=0)` | `nbi.py:128` | Unit test: on an asymmetric k=3 payoff, `n_hat == -phi.sum(axis=0)` (fails on main); end-to-end tri-objective NBI test (asymmetric quadratic toy from this review, `time_limit≈20`/solve) returns ≥ 8 nondominated points; k=2 tests unchanged (symmetric ⇒ bit-identical fronts) |
| MO-2a | Lexicographic payoff table: after each anchor solve, re-solve the remaining objectives lexicographically (fix `f_i` at its optimum via a tolerance constraint, optimize `f_j`) behind `payoff="lexicographic"` (default) with `"simple"` opt-out | `utils.py` (`ideal_point`/`nadir_point`), callers | Test with a deliberate alternative-optimum anchor where simple and lexicographic payoffs differ; nadir from lexicographic matches hand-computed value |
| MO-2b | Either implement the AUGMECON2 bypass (slack-based grid jumping, k=2 first) or rename tag/docs to `augmecon` | `scalarization.py` | If implemented: identical front to the non-bypassed sweep on the suite's toys with strictly fewer solves (assert solve-count drop); if renamed: docs/tag consistent, changelog note |

### Phase 2 — API robustness (PR `fix(mo): MO-3..MO-8`)

| ID | Task |
|----|------|
| MO-3 | Tolerance dedup in `ParetoFront.filtered()` (default `tol=1e-8` on objective vectors; keep first) + test |
| MO-4 | Shared-reference hypervolume: `hypervolume(front, reference=...)` docs warning + `common_reference(*fronts)` helper (union-worst + margin) + comparison test |
| MO-5 | Compile objectives once per sweep via `compile_expression_params`; thread through `_collect_objectives_at_x`/`_payoff_matrix`; measure per-point overhead before/after on a 21-point sweep |
| MO-6 | Honor a provided `ideal` without `anchors` where possible; document the recompute otherwise |
| MO-7 | `total_time_limit=` on all five sweeps (return partial front, record `front.method + "/truncated"` or a `completed: bool` field) + warning when the grid exceeds ~200 subproblems |
| MO-8 | Replace `del model._constraints[saved_n_cons:]` with identity-based removal of exactly the constraints each scalarizer added; drop the dead `senses_list` arg (MO-9) |

### Phase 3 — solver-integration and SOTA direction (design first)

1. **MO-10 (AMP linearizer)**: substitute `Parameter` current values when building
   the MILP relaxation instead of omitting the row — benefits every
   parameter-swept workflow, not just mo. Bound-changing: verify with the
   differential bound protocol (new relaxation bound ≥ old, ≤ oracle) per
   CLAUDE.md §5.
2. **Parallel sweeps** once model-copy lands (modeling-review Phase 4 dependency).
3. **Exact bi-objective MILP criterion-space search** (balanced-box or
   full-2-split) as a new `discopt.mo.exact` entry point — the genuine SOTA step.
   Needs its own design doc + entry experiment (compare against an AUGMECON sweep
   on bi-objective knapsack/assignment instances: nondominated-set completeness
   and wall time).
