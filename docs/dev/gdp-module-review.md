# GDP Module Review — Soundness, Correctness (#413 `[ ] gdp?`)

**Date:** 2026-07-05
**Scope:** the disjunctive-programming / GDP reformulation path —
`python/discopt/_jax/gdp_reformulate.py` (big-M / MBigM / hull reformulation of
disjunctions, indicators, SOS, and propositional-logic constraints),
`python/discopt/solvers/gdpopt_loa.py` (logic-based outer approximation),
`python/discopt/_jax/gdp_advisor.py` (big-M-vs-hull advisor), and the GDP-relevant
parts of the modeling layer (`_DisjunctiveConstraint` / `_IndicatorConstraint` /
`_SOSConstraint` / `_LogicalConstraint`, `either_or` / `if_then` / `sos1` / logic).
`decomposition/ir/reformulation.py` was inspected and is out of GDP scope (it wraps
the Benders/GBD/Lagrangian drivers; no disjunction handling — covered by
`decomposition-module-review.md`).

**Method:** every load-bearing function read and its reformulation math traced,
then a **differential soundness harness** (`no feasible disjunctive point cut` ∧
`no infeasible point admitted`) run over a battery of disjunction shapes (two-way,
multi-constraint, multi-var, equality, three-way, nonlinear-quadratic, unbounded)
for big-m / mbigm, an **exhaustive truth-table check** of every propositional-logic
encoding (and/or/implies/equiv/nested/DeMorgan/atleast/atmost/exactly/xor/Tseitin),
a hull-perspective admits-true-optimum check, and end-to-end solves comparing
big-m / hull / mbigm optima.

**Bottom line: two real soundness bugs found and fixed, both confirmed with a
repro that produces a wrong certificate; the rest of the GDP path is sound.**
GDP-1 is a **P0 default-path false-infeasible certificate** (too-small big-M on a
large-but-finite variable bound); GDP-2 is an mbigm dimension-mismatch crash. The
hull/perspective reformulation, the SOS-cardinality encoding, the full
propositional-logic linearization (incl. the one-directional Tseitin), the LOA
convexity gating + C-35 no-good discipline, and the advisor are all sound.

---

## 1. Findings

Legend: severity per CLAUDE.md (P0 = default-path false certificate).

| ID | Severity | Loc | Finding | Status |
|----|----------|-----|---------|--------|
| **GDP-1** | **P0 — default-path false certificate** | `gdp_reformulate.py:_compute_big_m` (`:189-211`), SOS `:1382-1386` | **Too-small big-M cuts feasible points of the active disjunct → false `infeasible`.** | ✅ FIXED |
| **GDP-2** | P1 — hard crash on opt-in path | `gdp_reformulate.py:_compute_big_m_lp` (`:317-324`) | **mbigm builds the LP `bounds` list from the *mutated* model (with aux selectors) → longer than the `c`/`A` columns → exact-LP backend panics.** | ✅ FIXED |

### GDP-1 — too-small big-M on a large-but-finite bound (P0)

`_compute_big_m` treated **any** variable bound with `abs ≥ _INF_THRESH = 1e15` as
"effectively infinite" and substituted `_DEFAULT_BIG_M = 1e4`. discopt's *default*
continuous bounds are ±~1e20 — **finite**, but above that threshold. So a variable
left at its default bounds got `M = 1e4·1.01 = 10100`, which is **not a valid
over-estimate** of the disjunct body. A big-M that is too small turns the inactive
disjunct's *relaxed* constraint into a *real* one and cuts feasible points of the
**active** disjunct.

**Repro (confirmed on `origin/main`, default `gdp_method="big-m"`):**

```python
m = Model("gdp1"); x = m.continuous("x")          # default bounds ±~1e20 (finite)
m.minimize((x - 25000)**2)
m.either_or([[x <= 2], [x >= 20000]])             # true optimum: x=25000, obj=0
m.solve(gdp_method="big-m")                        # BEFORE: status="infeasible" (WRONG)
```

The `x<=2` disjunct lowers to `x - 2 <= M(1-y0)`; with `M=1e4` and `y0=0`
(disjunct-1 relaxed) that is `x <= 10002`, which forbids the entire `x>=20000`
disjunct → the solver reports the feasible model **infeasible**. The identical
defect lived in the SOS linking big-M (`x_i <= ub_i·z_i` clamped `ub_i` to
`_DEFAULT_BIG_M` on a non-finite bound, capping the variable below its true range).

**Fix (correctness-first).** M *must* over-estimate the body over the true domain:
- **Finite bound (even huge, e.g. 1e20):** it *is* a valid M — use it as-is. A large
  M weakens the LP relaxation, but that is a *performance* cost; correctness wins.
  We never shrink a valid finite bound to a smaller `default`.
- **Truly infinite bound (`±inf`):** no valid finite M exists → **refuse loudly**
  (`ValueError`) telling the user to add a finite bound or use `method="hull"` (which
  needs no big-M). Silently substituting `default` is unsound and is removed.

SOS gets the same treatment: a finite (large) bound is a valid linking cap; a truly
infinite one refuses loudly instead of clamping.

After the fix the repro solves to `x=25000, obj=0, status="optimal"`; the
differential harness reports 0 cut / 0 admitted across the whole battery.

### GDP-2 — mbigm LP-bounds/columns dimension mismatch (crash)

`_compute_big_m_lp` precomputes the LP matrices (`c_vec`, `A_ub`, `A_eq`, `n_vars`)
from the **original** model, but then builds the per-column `bounds` list by
iterating `model._variables` on the **mutated** `new_model` — which already carries
the disjunction's selector binaries. So `len(bounds) > n_vars`, and the exact LP
backend panics on the dimension mismatch (`copy_from_slice: source (3) vs dest (1)`),
breaking **every** mbigm/`auto` disjunction that adds a selector (i.e. all of them).

This is a hard crash, not a false certificate (the panic aborts the solve rather than
returning a wrong answer), but mbigm/auto were entirely non-functional for
disjunctions. **Fix:** the aux vars are always appended *after* the originals, so the
first `n_vars` scalar columns are exactly the original variables `c_vec` indexes —
truncate `bounds` to `n_vars`. After the fix mbigm reformulates + solves and matches
the big-m optimum on the differential battery, and `auto` mode works.

---

## 2. Verified SOUND (not findings — do not re-audit without new evidence)

- **Big-M disjunction / indicator / nested reformulation.** With a valid M (post
  GDP-1), `body <= M(1-y)` / `body >= -M(1-y)` / the `==` two-sided split, the
  selector `sum == 1`, and the nested `sum(inner) == parent` linkage exactly
  represent the disjunctive region — 0 cut / 0 admitted over two-way,
  multi-constraint, multi-var, equality, three-way, and nonlinear-quadratic
  disjunctions. `Constraint.rhs` is always 0 (normalized), so the `rhs=0` bodies are
  correct.
- **Hull / perspective reformulation.** Disaggregated copies with `x == Σ_k v_{j,k}`,
  bound-linking `dlb·y_k ≤ v_{j,k} ≤ dub·y_k`, linear substitution scaling the
  constant by `y_k`, and the ε-clamped perspective `f(v/ỹ)·ỹ` for nonlinear bodies
  admit the true optimum with the natural selector/disagg assignment (checked on a
  nonlinear disjunct at its optimum: 0 violations). The ε-clamp residual at the
  integer faces is documented (`:994-1006`) and absorbed by the feasibility tolerance
  (issue #27a); it does not cut. Linear-hull solves match big-m/mbigm optima.
  (Nonlinear-hull *convergence* can be slow — a numerical/performance property of
  the perspective, not a soundness defect.)
- **SOS1 / SOS2 cardinality.** Linking `x_i ≤ ub_i·z_i`, `x_i ≥ lb_i·z_i` (only when
  `lb_i<0`), `Σz ≤ 1` (SOS1) / `Σz ≤ 2` + non-adjacency `z_i+z_j ≤ 1` for `|i−j|>1`
  (SOS2) are the standard sound encoding. The infinite-bound clamp was the GDP-1
  sibling and is now a loud refusal.
- **Propositional-logic linearization.** NNF (De Morgan, implies/equiv rewrite) →
  CNF → `Σ pos + Σ(1−neg) ≥ 1` clauses, and the cardinality specials
  (AtLeast/AtMost/Exactly) are exactly sound on an **exhaustive truth table** over
  and/or/implies/equiv/nested/DeMorgan/xor/atleast/atmost/exactly. The
  **one-directional Tseitin** (`aux → sub-clause`, defining clauses pushed onto
  `model._constraints`) is sound: it only *forces* the formula true when the aux is
  forced (never adds a constraint that cuts an original-variable feasible point), and
  the missing reverse direction only leaves `aux` free — projected out. Verified 0
  mismatches on `(A&B)|C` and `A→(B&C)` including the model-pushed defining clauses,
  and end-to-end (`(b0&b1)|b2` minimizes to `b2=1`, all 3 clauses present).
- **LOA / logic-based OA** (`gdpopt_loa.py`). OA cuts are emitted only for
  constraints/objective classified convex (`oa_convexity.constraint_mask`,
  `objective_is_convex`); the master `bound` is trusted only when
  `master_bound_valid` (linear or convex objective) and comes from the master *dual*
  bound, never the incumbent objective; the no-good cut is applied **only** on a
  *rigorous* infeasibility verdict (all-integer-fixed single-point violation), and a
  non-rigorous NLP failure downgrades certification and stops soundly — this is the
  **C-35 = OA-1** fix, confirmed present on main (`:159-333`,
  `_fixed_subproblem_rigorously_infeasible`) with `test_c35_oa_nogood_nlp_failure.py`
  green.
- **Advisor** (`gdp_advisor.py`). Analysis-only: it emits a per-disjunction
  method *recommendation* consumed by `reformulate_gdp(method="auto")`; every branch
  returns one of the three sound methods, nested/non-`Constraint` items force big-m
  (never silently dropped), unbounded-M forces big-m, and `auto` still respects a
  disjunction-local `method` override. It cannot mis-handle or drop a disjunction —
  the worst case is a *suboptimal-but-sound* method choice.

## 3. Cross-reference — the three ALREADY-FIXED items hold on main

- **C-34** (even-power bound over a zero-straddling base, `_bound_expression`
  `:504-525`): present — `p_int % 2 == 0 and left_lo < 0 < left_hi ⇒ [0, max(lb^p, ub^p)]`,
  generalized past the p==2 case. Holds.
- **X-2 residual** (`_compute_big_m_lp` element-0 box collapse, `:307-324`): present —
  per-flat-element bounds (`lb_flat[i]`/`ub_flat[i]`). Holds; GDP-2's fix is *adjacent*
  (it truncates that same loop to `n_vars` — it does not regress the per-element read).
- **OA-1 = C-35** (`gdpopt_loa` unconditional no-good on non-rigorous NLP failure):
  present and green (see §2). Holds.

---

## 4. Verification

- **Differential GDP soundness harness** (no-feasible-cut ∧ no-infeasible-admitted):
  0/0 across the big-m and mbigm batteries after the fixes.
- **New regressions** `python/tests/test_gdp1_gdp2_bigm_soundness.py` (5 tests) +
  rewritten `test_gdp.py::TestComputeBigM::{test_large_finite_bounds_use_true_bound,
  test_truly_infinite_bounds_refuse_loudly}` — all **fail-before** (verified by
  stashing the `gdp_reformulate.py` fix: 5 failed) / **pass-after**.
- Targeted suite `pytest -k "gdp or disjunct or reformulat or oa or loa or bigm or
  big_m or hull or indicator or sos or logic or c35"`: **396 passed**, 2 skipped.
- `pytest -m smoke`: **535 passed**. Adversarial
  `pytest -m slow test_adversarial_recent_fixes.py`: **10 passed**.
- ruff + ruff-format + mypy (`--python-version 3.10`) clean on the changed files
  (the residual mypy errors are pre-existing, in other modules). No Rust touched.
- **Cert-baseline neutrality: EXACT by construction.** `reformulate_gdp` returns the
  input model **unchanged** when it has no GDP constraints (`:79-87`); all 41
  cert-baseline instances are MINLPLib `.nl` files with no disjunction/indicator/SOS/
  logic constructs, so `_compute_big_m` / `_compute_big_m_lp` are never reached —
  node_count / objective / status are bit-identical. `incorrect_count == 0`.

## 5. Plan / follow-ons (not soundness)

- Nonlinear-hull convergence is slow (perspective + ε-clamp); a performance item, not
  a certificate risk. Big-m/mbigm are the robust default for nonlinear disjuncts.
- A large valid big-M (from a variable left at ±1e20 defaults) weakens the LP
  relaxation — the sound-but-loose regime GDP-1 now correctly prefers over a wrong
  answer. Users wanting a tight relaxation should supply real bounds (or `method="hull"`);
  the loud refusal on truly-infinite bounds now nudges them to.
