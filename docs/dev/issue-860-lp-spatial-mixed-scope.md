# Issue #860 — widening the LP-per-node engine to mixed-integer and MAXIMIZE models

*Status: implemented. Engine scope widened unconditionally (Panel A: cert-clean, 33
newly reachable instances get a verified incumbent). The default-path fallback reserve
for the newly in-scope class stays behind `DISCOPT_LP_SPATIAL_MIXED`, **default OFF**:
its graduation panel is cert-clean but not net-positive (§4, Panel B).*

The LP-per-node spatial engine (`_relax/lp_spatial_bb.py`) was gated to **pure-integer,
MINIMIZE** models. Issue #860 records the consequence: three instances named in #844
(`rsyn0805m04hfsg`, `gastrans582_*`, `gastrans040`) return no incumbent and cannot
even be *offered* to the engine, so no primal heuristic inside it can help.

## 0. What the gate actually cost — measurement before design

The gate probes named in the issue are not vendored in this repo (the full MINLPLib
snapshot lives outside it, and this environment has no network route to
minlplib.org), so both experiments below ran over the **in-repo corpus**:
`python/tests/data/minlplib_nl` (66 instances) ∪ `python/tests/data/minlplib`
(81, 53 new) = 119 distinct instances. That corpus is dominated by exactly the class
under discussion — **71 of the 119 are mixed** (continuous + integer) and 5 are
MAXIMIZE, including `syn05m` / `syn05hfsg`, the same `syn`/`rsyn` family as the
issue's maximize probe.

**Round 1** (`scratchpad/issue860_entry_experiment.py`) measured the engine's
*existing* gate on that corpus and produced the finding that reframed the work:

| blocker on the 71 mixed instances | count |
|---|---|
| root box has an infinite endpoint → `_is_in_scope` declines | **31** |
| `IncrementalMcCormickLP` declines (term types its closed-form patch does not map) | 28 |
| reach a root LP at all | **11** |

Mixed-ness was *not* the binding constraint. Real MINLPs leave continuous columns
unbounded above, and the engine's all-variables-finite test — inherited from the
pure-integer step, where it is nearly free — rejected them before anything else could
run. A widening that only relaxed the variable-mix test would have been close to a
no-op on the real class (the #727 RLT lesson: a mechanism validated on the wrong
proxy).

## 1. Entry experiment (CLAUDE.md §4) — the gate this issue proposes

**Hypothesis.** For mixed-integer and MAXIMIZE MINLPs, an LP-per-node McCormick
relaxation over the continuous+integer box produces (a) a valid finite dual bound and
(b) verifiable feasible points, so widening the gate is worth the engine work.

**Kill criterion (from the issue).** No usable incumbent anywhere in the mixed class
at any budget ⇒ mixed-integer LP-per-node is not the lever.

**Method** (`scratchpad/issue860_entry_experiment_v2.py`), at the root box only — no
tree, no dive, no pump:

* accept a partially infinite box — the *cold* `build_milp_relaxation` already
  discards any row whose payload is non-finite (`uniform_relax._Builder.add_row`), so
  it self-guards and merely loosens; the *incremental* patch does not, so it is used
  only when every mapped product factor is finite;
* `round` — integers rounded, continuous left at their LP values: feasible for the
  TRUE nonlinear constraints?
* `complete` — integers rounded and FIXED, continuous re-solved by the node LP:
  feasible for the TRUE constraints? (the mixed generalization of the pure-integer
  engine's collapsed-box primal)

**Result — the kill criterion did NOT fire.**

| metric (71 mixed instances) | current gate | widened gate |
|---|---|---|
| finite root McCormick bound | 11 | **46** |
| …of which served by the cold builder | 0 | 35 |
| …of which have an infinite root endpoint | 0 | 12 |
| verified feasible point from one-shot rounding | 4 | 7 |
| verified feasible point from round + LP completion | 4 | **13** |

The 13: `cvxnonsep_psig30`, `ex1225`, `gbd`, `gkocis`, `nvs08`, `st_miqp5`,
`ex1223a`, `gear2`, `gear3`, `gear4`, `nvs20`, `prob10`, `st_e27` — each a genuinely
feasible point under an independent evaluator, from a single LP re-solve.

**Maximize.** All 5 in-repo maximize instances have *no valid objective bound* over
their raw root box (`_objective_bound_valid=False`: the objective is unbounded above
there). That is not a maximize bug — it is the infinite box again. Root OBBT
finitizes every infinite upper bound in 0.2–0.4 s and the McCormick LP then returns a
valid bound:

| instance | inf. upper bounds before → after OBBT | minimize-equivalent root bound |
|---|---|---|
| `syn05m` | 13 → 0 | −1165.78 |
| `syn05hfsg` | 35 → 0 | −1335.50 |

So on this family the blocker is the box, and OBBT removes it. This is why the engine
now runs root OBBT whenever the box is infinite, regardless of `use_obbt`.

## 2. What changed

### 2.1 Scope (`lp_spatial_bb._is_in_scope`)

From "every variable integer AND MINIMIZE" to "**at least one integer variable**, any
objective sense, any continuous mix". Pure-continuous models stay out: the engine's
convergence argument is that branching the *integers* drives the lifted products
exact, and with no integer variable it degenerates to plain spatial bisection, which
the default NLP-per-node path already does. `_is_in_scope(model, mixed=False)`
restores the old gate for callers rolling the widening out behind a flag.

### 2.2 Objective sense

The engine runs entirely in **minimize-equivalent** space. Both producers already
work there — `uniform_relax` negates a MAXIMIZE objective when it builds the LP cost,
and `NLPEvaluator` negates it when it evaluates a point — so the LP bounds and the
verified incumbents were already commensurable; nothing inside the loop needed a sign
at all. `sgn` is applied exactly once, on the reported `objective` / `bound`. A valid
lower bound on `−f` is a valid **upper** bound on `f`, so `bound ≤ incumbent` becomes
`bound ≥ incumbent` for a maximize — the maximize form of "the dual bound never
crosses the incumbent". `gap` is a ratio of absolute differences and is
sign-invariant.

### 2.3 Continuous variables

* **Branching.** Integers branch integrally (`floor`/`ceil`) and split spatially into
  the disjoint `[lb, mid]` / `[mid+1, ub]`; continuous variables bisect at the box
  midpoint into the *overlapping* `[lb, mid]` / `[mid, ub]` — the children must share
  the midpoint or a feasible point could fall between them. Both only shrink boxes,
  so every child relaxation stays a valid outer approximation of its subtree.
* **Unbranchable nodes.** A candidate narrower than `_MIN_BRANCH_WIDTH` (1e-9, the
  same constant as the collapsed-box test, so the two agree by construction) is not
  branchable; such a node folds into `unresolved_lb` and can never yield an
  optimality proof.
* **Primal.** New `complete()`: fix the integers at their rounded values, re-solve the
  node LP for the continuous coordinates, verify the completed point against the
  ground-truth evaluator. On a pure-integer model this degenerates to the existing
  collapsed-box primal. The LP completion is still a *relaxation* of the continuous
  restriction, which is exactly why its point is only a candidate — it is discarded
  unless the evaluator accepts it. Run once at the root, then every 16th node.
* **Feasibility pump** acts on the integer coordinates only; the continuous ones carry
  no rounding distance to close and are left to the LP.

### 2.4 Infinite root boxes

Accepted, by two independent routes:

* root OBBT runs whenever the box is infinite (see §1) and usually finitizes it;
* whatever remains infinite is handled by the **cold** builder, which drops rows it
  cannot make finite. The **incremental** patch writes closed-form envelope
  coefficients straight from the box endpoints with no row-level guard, so it is
  declined when any column it patches is infinite
  (`IncrementalMcCormickLP.box_is_patchable`, backed by the new `box_dependent_cols`;
  `_validate` guarantees that column set is complete, since an unmapped box-dependent
  row makes `ok` False). Branching only shrinks boxes, so a patchable root stays
  patchable at every node.

### 2.5 Cold node builds are now deadline-bounded

`_relax_bound` takes a `deadline` and passes it to both uninterruptible halves — the
DAG re-walk (`build_deadline`, which keeps the row prefix: a weaker but still valid
outer relaxation) and the LP solve. The widened scope routes the mixed class largely
through this path, on models an order of magnitude larger than the pure-integer
instances the engine started on, where a single cold node could otherwise outlive the
whole engine budget.

## 3. A soundness bug found on the way (default path, not opt-in)

`IncrementalMcCormickLP` solves `min c·x`, but the relaxation's objective is
`c·x + obj_offset` — the constant the cold `MilpRelaxationModel.solve` adds back. The
incremental path dropped it, so its node bound sat on a different origin from the cold
build's. For a positive constant that is merely weak; for a **negative** one it is a
dual bound *above* the true node optimum, i.e. the false-fathom class.

Measured on a bilinear integer node whose true McCormick optimum is −92.0:

| objective constant | cold node bound | fast node bound (before) | fast node bound (after) |
|---|---|---|---|
| 0 | 8.0 | 8.0 | 8.0 |
| −100 | **−92.0** | **+8.0** (unsound) | −92.0 |
| +100 | 108.0 | 8.0 (weak) | 108.0 |

This is on the **default** spatial path (`mccormick_lp`'s incremental fast path since
cert:T1.3), not only the opt-in engine. Fixed by carrying `obj_offset` on the
structure and adding it back at every bound-returning exit — including the
certificate's `safe_bound`, so a consumer reading either one gets the same origin.
`c_override` solves (the feasibility pump) are surrogates, not bounds, and are
deliberately excluded. `_validate` now also asserts the offset is box-independent, so
a shape where it were not would set `ok=False` and fall back rather than mis-report.

Regressions: `test_incremental_mccormick_node.py::test_incremental_node_bound_includes_objective_constant`
(fails before, passes after) and `::test_incremental_solve_bound_matches_cold_builder_with_constant`.

## 4. Verification

Harness `scratchpad/panel860.py`, scoring `scratchpad/panel860_analyze.py`, 20 s
budget, one fresh model per instance.

### Panel A — engine soundness on the widened class

Every in-repo instance the engine now accepts (89: the 70 newly in scope plus the 19
the old gate already served), run through `solve_lp_spatial_bb` directly. The checks
need no external oracle: every reported incumbent independently re-verified feasible
with its objective re-evaluated by a fresh evaluator; `bound ≤ objective` for a
minimize and `bound ≥ objective` for a maximize; the bound never crossing the best
verified feasible point found by any run of that instance; and no `status="optimal"`
run beaten by another run's verified incumbent (which would be a false optimality
certificate).

| | newly in scope | legacy in scope |
|---|---|---|
| instances | 70 | 19 |
| engine declined | 15 | 0 |
| errors | **0** | **0** |
| verified incumbent | **33** | 17 |
| certified optimal | 12 | 14 |
| **unsound bounds / incumbents** | **0** | **0** |
| **false optimality certificates** | **0** | **0** |

**Cert-clean.** 33 instances that the engine previously refused outright now yield a
verified feasible point, 12 of them with an optimality certificate. The legacy
pure-integer class is unchanged — nothing declined, nothing lost.

Both maximize instances behave as the soundness argument requires — the reported
bound is an *upper* bound sitting above the incumbent:

| instance | status | objective | bound | incumbent independently feasible |
|---|---|---|---|---|
| `syn05m` | feasible | 837.732 | 868.110 | yes |
| `syn05hfsg` | feasible | 837.732 | 868.110 | yes |

(`bchoco06/07/08`, the other three maximize instances, decline: their root LP exits at
`iteration_limit` on an ill-conditioned equilibrated system. A conditioning problem,
not a scope one.)

### Panel B — graduation gate for `DISCOPT_LP_SPATIAL_MIXED`

The flag governs whether the *default* path reserves 35% of its budget for the #844
no-incumbent fallback on mixed / maximize models. That reserve is the sole risk of the
widening (an in-scope model hands budget to a pass that may find nothing), which is
why it is a separate, measured decision from the engine's capability. Run over the 70
newly in-scope instances — the other 49 take a bit-identical path under either
setting, since the reserve gate is the flag's only consumer and both gates agree
there. Graduation needs both bars of CLAUDE.md §5: cert-clean **and** net-positive.

**Verdict: cert-clean, but NOT net-positive. The flag stays default-OFF.**

| bar | result |
|---|---|
| certification regressions (`gap_certified` True→False) | **0** |
| `incumbent_verification_failed` | **0** |
| unsound bound / false optimality certificate | **0** |
| objective improved | 1 — `ex1252a` 183660.35 → **149530.99** (minimize; both independently feasible) |
| incumbents **gained** | 1 — `tspn12` 262.647 (independently feasible) |
| incumbents **LOST** | **2 — `tls2`, `st_e31`** |

| instance | flag off | flag on |
|---|---|---|
| `tspn12` | time_limit, no incumbent (30.6 s) | feasible **262.647** (9.3 s) |
| `ex1252a` | feasible 183660.35 (24.5 s) | feasible **149530.99** (14.9 s) |
| `tls2` | feasible **11.30** (20.1 s) | time_limit, **no incumbent** (13.6 s) |
| `st_e31` | feasible **−2.00** (22.2 s) | time_limit, **no incumbent** (14.8 s) |

Soundness is untouched — the widening cannot produce a bad answer, exactly as Panel A
showed. What it cannot do is *pay for itself on the default path*: one incumbent
gained and one improved against two lost. Losing an incumbent is strictly worse than
improving one that already exists, so the net is negative and the flag does not
graduate. This is the `DISCOPT_CUT_INHERIT` lesson again (CLAUDE.md §5): sound ≠
helpful, and a cert-clean but harmful flag stays off with its measurement recorded.

**Mechanism of the two losses**, which is the useful part of the result. The reserve
hands 35% of the budget to the fallback *before* knowing whether the engine can serve
the model. On `tls2` and `st_e31` the primary then runs out of time at 65% and returns
nothing — and the fallback declines anyway, because it passes `require_incremental=True`
and both models have infinite root boxes, which decline the incremental structure
(§2.4). The reserve is pure loss there: budget taken from a path that was going to
succeed, given to a path that never runs.

Do not read the 0.747 wall ratio as a win. The flag-on runs are faster mostly *because*
they gave up earlier — `tls2` 20.1 s → 13.6 s and `st_e31` 22.2 s → 14.8 s are exactly
the two instances that stopped producing an answer.

**The concrete follow-up** this points at: make the reserve conditional on the engine
actually being able to build (probe buildability, or relax `require_incremental` for
the mixed class now that cold node builds are deadline-bounded per §2.5), so a model
the fallback will decline never pays for it. That is a separate change with its own
panel, not a tweak to this one.

### A pre-existing soundness flag, not from this work

Panel B's independent verifier rejects `nvs22`'s reported incumbent at tolerance 1e-5
under **both** flag settings. It is not caused by the flag, and not by this branch:
run against the base commit `ac9f5cf` the result is byte-identical — objective
`6.058219942618198`, `gap_certified=True`, maximum constraint violation
`2.641e-4` on row 2. That sits above the 1e-5 the panel checks and below the
deliberately loose 1e-3 of discopt's own final `incumbent_verification_failed` guard,
so nothing flags it today. Worth its own investigation; out of scope here.

## 5. Known limits (not addressed here)

* **28 of the 71 mixed instances still decline the incremental structure** — term
  types its closed-form patch does not map (univariate `exp`/`log`, trilinear,
  `monomial^p` for `p≥3`, …). They are served by the cold builder, correctly but far
  more slowly. Extending the patch table term-family by term-family is
  `certification-gap-plan.md` T1.2 and is deliberately out of scope here.
* Because the #844 fallback passes `require_incremental=True`, mixed models that fall
  to the cold path decline that fallback even with the flag on. Relaxing it is a
  budget-overrun question (the `ball_mk2_30` measurement), now partly de-risked by the
  deadline-bounded cold build in §2.5, but it needs its own panel.
* A mixed model rarely reaches a *certified* optimum: the exact-leaf test requires the
  whole box collapsed, and a live continuous dimension keeps a node unresolved. That
  is sound and honest — the fallback exists to supply a missing incumbent, and the
  dual bound remains the primary path's job — but it means the widening is a primal
  gain, not a certification one.
* `test_lp_spatial_bb.py::test_matches_brute_force` fails on 3 of its 4 parameter sets
  **before and after** this change (identical set, verified against the base commit):
  the engine finds the true optimum but reports `feasible` rather than `optimal`. It
  is the documented post-#636 conservatism — `a*b` is lifted via univariate squares,
  so it never appears in the engine's `info` product map, `_worst_product_var` cannot
  see it, and no node short of a fully collapsed box is treated as exact. A
  pre-existing tightness gap, unrelated to this issue.
