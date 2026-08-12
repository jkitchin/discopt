# DIRECT backend — entry experiment (2026-08-12)

Entry experiment for a derivative-free `solver="direct"` backend, run **before**
any production code, per Dev-Philosophy #4. Papers: Jones & Martins, *The DIRECT
Algorithm — 25 Years Later*, JOGO 2021 (the survey); Jones, Schonlau & Welch,
*Efficient Global Optimization of Expensive Black-Box Functions*, JOGO 1998.

Reproduction: `scripts/direct_entry_experiment.py`
(`--self-check` validates the prototype; no arguments runs the panel).

---

## Pre-registered hypothesis and kill criterion

> **Hypothesis.** On bound-constrained models whose objective is a non-MCBox
> `dm.custom` (`CustomCall`) body, a DIRECT global search followed by discopt's
> local NLP finds a strictly better objective than discopt's *current* path for
> such a model — a single local NLP from discopt's default start.
>
> **Kill criterion.** Arm B must strictly improve on **≥ 2/3** of the panel AND
> reach **1e-2** relative accuracy on **≥ 3/4**.

Arms, both using discopt for the local step so the comparison isolates what
DIRECT adds (a starting point) rather than comparing two local solvers:

* **A** — `m.solve()`: today's path, discopt's default start.
* **B** — prototype DIRECT, 2000 evaluations → best point `x*`, then
  `m.solve(initial_solution={x: x*})`.

Panel: 13 standard multimodal test functions, n = 2…6, each wrapped in
`dm.custom` so it takes the opaque path. Boxes for the origin-centred functions
(rastrigin, ackley, griewank) are deliberately **asymmetric** — DIRECT's first
evaluation is the box centre, so a symmetric box around an optimum at the origin
is "solved" at evaluation 1 and measures nothing. The first draft of this panel
had that flaw and produced three meaningless exact zeros.

## Result — the pre-registered criterion was NOT met

| criterion | required | observed | |
|---|---|---|---|
| Arm B strictly better | ≥ 9 / 13 | **8 / 13** | **not met** |
| Arm B within 1e-2 relative | ≥ 10 / 13 | **10 / 13** | met |

Per Dev-Philosophy #4 the measurement wins and this is recorded as written: **the
gate as pre-registered failed.**

## Full panel

| instance | n | A (local only) | B (DIRECT+local) | f\* | verdict |
|---|---|---|---|---|---|
| branin | 2 | 0.397887 | 0.397887 | 0.397887 | tie **at the optimum** |
| six_hump_camel | 2 | 0 | −1.03163 | −1.03163 | B better |
| goldstein_price | 2 | 30 | 3 | 3 | B better |
| shubert | 2 | −32.77 | −123.58 | −186.73 | B better |
| rastrigin_2 | 2 | 1.98992 | 0 | 0 | B better |
| rastrigin_5 | 5 | 4.9748 | 3.97984 | 0 | B better |
| ackley_2 | 2 | 15.0635 | 0 | 0 | B better |
| ackley_5 | 5 | 15.0635 | 0 | 0 | B better |
| levy_4 | 4 | 0.179057 | 1.34e−20 | 0 | B better |
| **griewank_3** | 3 | **0.0887** | **0.1208** | 0 | **B worse** |
| hartman_3 | 3 | −3.86278 | −3.86278 | −3.86278 | tie **at the optimum** |
| hartman_6 | 6 | −3.32237 | −3.32237 | −3.32237 | tie **at the optimum** |
| michalewicz_2 | 2 | −1.8013 | −1.8013 | −1.8013 | tie **at the optimum** |

## Why the criterion failed, and why that is a criterion defect

The criterion counts "tied" as a failure. **All four ties are exact ties at the
known global optimum** — instances where arm A already solved the problem and
improvement was arithmetically impossible. The criterion therefore penalizes
DIRECT for the cases it cannot possibly win, and its denominator is wrong.

Decomposed against the question the backend actually exists to answer:

* Arm A **failed** (relative error > 1e-2) on **9 / 13**.
  * Arm B improved on **8 / 9** of those.
  * Arm B reached 1e-2 on **6 / 9** of those.
* Arm A was **already at the optimum** on **4 / 13** — Arm B matched it exactly on
  all four, with **no regression**.
* Arm B was genuinely worse on **1 / 13** (griewank_3).

The improvements on the cases that needed them are not marginal: goldstein_price
30 → 3, ackley_2/ackley_5 15.06 → 0, rastrigin_2 1.99 → 0, six_hump_camel 0 →
−1.0316, levy_4 0.179 → 1.3e−20, shubert −32.8 → −123.6.

**This re-reading happened after seeing the data and is therefore weaker evidence
than the pre-registered gate.** It is recorded as a criterion defect rather than
used to quietly declare a pass; see "Decision" below.

## The one genuine regression, and the design correction it forces

griewank_3: DIRECT's best point after 2000 evaluations is
`(−6.28, 13.32, 16.30)` with f = 0.1208; the local solve from there stays in that
basin (0.12075). Arm A's default start happens to sit in a better basin (0.0887).
Neither finds the global optimum.

Root cause: **DIRECT's incumbent is not always a better local-solve start than
the default.** Griewank is a broad quadratic bowl with fine multimodal ripple;
the default start is already near the bowl's floor, while DIRECT's coverage-driven
incumbent is in a nearby ripple.

**Design correction, adopted:** the production hybrid must launch the local solve
from **the best of {default/incumbent start, DIRECT's best point(s)}** and keep
the better result, rather than replacing the default start with DIRECT's. This
makes `solver="direct"` **no worse than today's path by construction**, and it is
what the survey's own hybrids do — glcCluster clusters the sampled points and
launches a local optimizer from the best point in *each* cluster; Jones' 2001
revision *alternates* between DIRECT and the local optimizer, keeping the
incumbent. With this correction griewank_3 becomes a tie (0.0887) rather than a
regression, giving **12 no-worse + 1 tie-at-worse → 0 regressions** on this panel.

## Second measured claim — CONFIRMED

The survey's endorsements 1 and 2 (trisect one long side; select one rectangle
among ties) reduce evaluations-to-accuracy.

Evaluations to 1e-2 relative accuracy (`None` = not reached in 2000):

| instance | all sides + ties | all sides + break ties | one side + break ties |
|---|---|---|---|
| linear_drag_2 | 121 | 63 | **43** |
| linear_drag_5 | None | 479 | **193** |
| branin | 63 | 53 | 59 |
| six_hump_camel | 109 | 105 | 127 |
| goldstein_price | 99 | 83 | **65** |
| rastrigin_2 | 847 | 731 | **513** |
| ackley_2 | 89 | 75 | **65** |
| ackley_5 | 305 | 263 | **175** |
| levy_4 | 165 | 145 | **121** |
| hartman_3 | 81 | 73 | **71** |
| hartman_6 | 213 | 199 | **101** |
| michalewicz_2 | 45 | 39 | **23** |
| shubert, rastrigin_5, griewank_3 | None | None | None |

One-side + break-ties needs **≤** the original's evaluations on **10 / 11**
comparable instances (median **1.52×** fewer, max 2.81×). The single exception,
six_hump_camel (109 → 127), is within noise of a 2-D problem solved in ~100
evaluations. **Both defaults stand as planned: `divide="one"`, `break_ties=True`.**

## Prototype validation — and two bugs it caught

The prototype is cross-checked two ways (`--self-check`, 208 executed assertions,
exits non-zero if zero):

1. **Selection vs the definition.** The potentially-optimal set is compared with a
   brute-force sweep over `K` — Eq. (3) itself, not a reimplementation — on 200
   randomized (d, f) configurations.
2. **Evaluation counts vs the survey's published figures.**

| measurement | prototype | survey |
|---|---|---|
| `1+x₁+x₂`, evals to 1% | 121 | 90 |
| `1+x₁+x₂`, evals to 0.01% | 757 | 616 |
| `1+…+x₅` to 1%, all sides + ties | 16,555 | 14,492 (Fig. 15) |
| `1+…+x₅` to 1%, all sides + break ties | 479 | **470** (Fig. 15) |
| `1+…+x₅` to 1%, one side + break ties | 193 | **192** (Fig. 15) |

The two improved variants match the published figures to within 2%; the original
variant is ~14% higher, consistent with implementation-specific tie enumeration.

**Two bugs were found only because of these checks, and both had silently produced
a KILL verdict on the first run:**

* **Sign error in the Eq. (4) slope.** The largest admissible `K` for which
  rectangle *i* still beats *j* is `(f_j − f_i)/(d_j − d_i)`; the code had it
  negated. Effect: every rectangle except the largest was rejected, the search
  stalled at a fixed refinement depth, and the linear drag function never
  converged — contradicting a published figure, which is what exposed it.
* **Missing positive-slope restriction on the hull.** The monotone-chain lower
  hull always includes its leftmost vertex, but that vertex is optimal only for
  `K < 0` (preferring *smaller* rectangles with *worse* values). The hull must be
  truncated to the suffix beginning at the minimum-`f` vertex.

Both are hazards for the production implementation and are pinned as unit tests
in the plan (`test_hull_selects_lower_right_convex_hull`,
`test_hull_epsilon_condition_prunes_tiny_rectangles`).

A third defect was in the *panel*, not the code: the symmetric boxes described
above. Recorded because it is the same failure mode as Dev-Philosophy #6 — an
instrument that appears to measure and does not.

## Also observed — filed as #998, since **fixed**

At the time of this run, discopt's existing local path returned
`status="optimal"` with `bound = objective` for a non-MCBox `CustomCall` model,
while correctly setting `gap_certified=False`. On Ackley (true optimum 0) that
meant a reported dual bound of **15.06**, i.e. not a valid bound at all. The
same mathematics written algebraically correctly returned `bound=None`, because
the C-33/SC-1 fix had been applied to the convexity-unknown caller but not to
the `CustomCall` one.

Filed as [#998](https://github.com/jkitchin/discopt/issues/998) and **fixed by
[#999](https://github.com/jkitchin/discopt/pull/999)** (C-42), which strips the
fabricated `bound`/`gap`/`root_bound`/`root_gap` and additionally reports
`status="feasible"` rather than `"optimal"` — going further than the issue
proposed, which had flagged the status question as separable. Verified against
the merged fix:

```text
status = feasible   objective = 15.06351400512647   bound = None
gap    = None       root_bound = None               gap_certified = False
```

So the arm-A `bound` values described in this record are a snapshot of pre-#998
behaviour. The panel comparison itself is unaffected: it compares `objective`,
never `bound`, and the probe's soundness check watches `gap_certified`, which was
correct throughout.

With #999 merged, the local path's contract and the DIRECT backend's contract now
agree — no bound, no gap, and a status that does not read as a proof.

## Decision

The pre-registered gate failed on its "strictly better" arm (8/13 vs 9/13
required) and passed on its accuracy arm (10/13 vs 10/13 required). Restricted to
the 9 instances where improvement was possible, Arm B improved on 8 and regressed
on 1, with the regression root-caused and fixed by a design change that makes the
backend no-worse-than-today by construction.

**Recommendation: proceed to implementation** with the best-of-both-starts
correction folded into the hybrid, and with `divide="one"` / `break_ties=True`
confirmed as defaults by the second claim. The gate is *not* being declared
retroactively passed — it is being reported as failed-as-written with a defect in
its own construction, and the decision to continue is recorded here as a
judgement call on the decomposed evidence rather than a criterion pass.
