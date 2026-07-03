# RO Module Review — Correctness, Thoroughness, Performance, and SOTA Assessment

**Date:** 2026-07-03
**Scope:** `python/discopt/ro/` (`uncertainty.py`, `counterpart.py`, `affine_policy.py`,
`formulations/{_common,box,ellipsoidal,polyhedral}.py` — 2,195 lines) and its tests
(`test_robust_counterpart.py`, `test_robust_solve.py`, `test_robust_uncertainty.py`,
`test_affine_decision_rule.py` — 1,520 lines).
**Method:** Full read of every file; each suspected defect reproduced end-to-end or
verified by direct computation. Baseline: **89 tests passed** (5.2 s).

The module ships an unusually candid `ROADMAP.md` that already names some of these
limitations. The review's central conclusion is sharper than the roadmap's: several
of the "known limitations" are not future-work items but **silent soundness holes in
the shipped behavior** — a robust counterpart that is not robust, with no error and
no warning. For a module whose entire product is a guarantee ("feasible for *every*
realization"), silent non-robustness is the P0 failure mode, exactly analogous to a
wrong bound in the global solver.

---

## 1. Summary of findings

| # | Severity | Component | Finding |
|---|----------|-----------|---------|
| RO-1 | **P0 soundness** | `formulations/_common.py` (box path) | Sign-tracking assumes the parameter's variable coefficient is **nonnegative**; with a variable that can be negative the "robust" counterpart **under-protects** — returned solution violates the constraint at in-set realizations [CONFIRMED] |
| RO-2 | **P0 soundness** | `formulations/ellipsoidal.py` | Only the literal `p @ x` / `x @ p` MatMul pattern is robustified; every other appearance of the parameter (`p * x`, `dm.sum(p*x)`, RHS `h(x) <= p`) is **left at nominal with zero penalty — silently non-robust** [CONFIRMED: constraint object provably unchanged; scalar case returns the nominal optimum] |
| RO-3 | **P0 method** | `uncertainty.py:budget_uncertainty_set` + `polyhedral.py` | The stored `A,b` has only the all-plus/all-minus budget facets (2 of the 2^k needed); the compact `_is_budget/_delta/_gamma` attributes are **never read** by any formulation — the "Bertsimas–Sim" counterpart protects against a strict superset, **2× over-conservative** in mixed-sign directions at k=2 [CONFIRMED: support of [1,−1] = 2.0 vs true 1.0] |
| RO-4 | **P1 soundness** | `_common.py` | Division/power propagate sign **without flipping**: `c/p` substitutes the wrong bound (non-conservative); non-monotone dependence (`p**2` on a straddling interval) picks the wrong endpoint [BY INSPECTION — mechanism unambiguous] |
| RO-5 | P1 soundness | `polyhedral.py:_eval_constant_expr` | Unknown node types (SumExpression, MatMul, FunctionCall, IndexExpression) **evaluate to 0.0 silently** — a constant-coefficient constraint written with any of those shapes gets **zero protection** [BY INSPECTION] |
| RO-6 | P1 | `_common.py` (documented) | `np.sign(np.sum(value))` collapses constant **vectors** to one scalar sign — mixed-sign coefficient vectors get the wrong bound per-component (documented in ROADMAP, but ships silent) |
| RO-7 | P1 | `box.py`, `polyhedral.py`, `affine_policy.py` | Arbitrary caps silently restrict the model: `t_ub = max|bound|+1` on abs-value auxiliaries, `lam_ub = Σ|b|+100` on LP duals, `|Y_j| ≤ ub−lb` on ADR policy columns — each can cut the true optimum on non-unit-scaled problems with no warning |
| RO-8 | P2 | box/ellipsoidal paths | Equality constraints containing uncertain parameters are "robustified" by substitution/penalty — semantically meaningless for continuous uncertainty; should refuse loudly |
| ADJ-1 | **P0 (adjacent, solver layer)** | extraction layer (`_jax/problem_classifier.py` neighborhood) | `maximize` + a `dm.sum(c * x) <= b` constraint returns the **wrong certified optimum (0 instead of 4)**; `minimize(−·)`, scalar, MatMul, and `sum(x)` forms are all correct. Discovered while controlling RO-2; same layer as modeling-review M1 [CONFIRMED, minimal repro in §5] |

Checked and found **correct**:

- **Polyhedral LP-duality core** (variable coefficients path): the ∃λ weak-duality
  encoding, the sign conventions for both maximize and minimize worst cases
  (`Aᵀλ = ±coeff`, `±bᵀλ` penalty), and the optimizer pressure directions are all
  right; `_support_function_lp` applies its IPM margin in the conservative
  direction and its ImportError fallback over-covers. This is the module's best
  code.
- **Box path on its blessed domain** (affine in p, nonnegative variable
  coefficients, single-sign constant vectors): worst-case substitution and the
  minimize/maximize objective conventions are correct — consistent with the
  Ben-Tal Example 1.1.1 validation the roadmap cites.
- **Box bilinear path** (ADR-generated `Y·ξ` terms): the coefficient-extraction +
  `|coeff|`-linearization is the right construction, and the aux-variable pressure
  directions bind correctly in both objective senses (modulo RO-7's cap).
- **Ellipsoidal math for the MatMul pattern**: verified numerically — robust
  optimum 2.3431 matches the analytic `x₁+x₂+ρ‖x‖ ≤ 4` solution.
- **ADR construction**: intercept/policy-column creation, ξ-perturbation
  expressions, identity-based substitution (including `y[i] → (affine)[i]`), the
  robustified bound constraints on the affine expression, and variable retirement
  are all coherent; the class correctly defers all uncertainty handling to
  `RobustCounterpart`.
- API hygiene: mixed-set-type refusal, double-`formulate()` guard, non-`Constraint`
  entries passed through untouched everywhere.

---

## 2. The soundness holes (P0/P1)

### RO-1. Sign-tracking is unsound for sign-indefinite variable coefficients

`_common.py`'s traversal decides upper-vs-lower bound from the *constant* multipliers
along the path only. For `p * x` it explicitly declares (in `box.py`'s
`_has_bilinear_param_var` docstring) that "the parameter IS the coefficient and
sign-tracking can determine its worst-case value directly" — true only when **x ≥ 0**.

**Reproduction:** `maximize x s.t. p·x ≤ −1`, `x ∈ [−10,10]`, `p̄=1, δ=0.5`.
True robust optimum: `x = −2` (binding at `p = 0.5`). The module returns
**`x = −0.667`**, and at the in-set realization `p = 0.5` the constraint value is
**−0.333 > −1 — violated**. The counterpart is not a counterpart.

**Fix:** the correct machinery already exists in the same file — the bilinear
`|coeff|`-linearization path handles sign-indefinite coefficients by construction.
Route through it whenever the parameter's coefficient involves a variable that is
not provably nonnegative (`lb ≥ 0` check per flat component); keep sign-tracking
as the fast path only for provably-nonnegative coefficients. Regression test: the
repro above must return −2.

### RO-2. Ellipsoidal robustification silently no-ops outside `p @ x`

`_extract_penalties` adds the SOC penalty only when the parameter is the *immediate*
child of a `MatMulExpression`. In every other position — elementwise `p * x`,
`dm.sum(p * x)`, scalar `p·x`, or RHS uncertainty `h(x) ≤ p` (which the module
docstring explicitly claims to handle) — the Parameter branch substitutes the
nominal value and returns **no penalty**.

**Reproduction:** scalar `p·x ≤ 4`, ρ=0.5: after `formulate()` the constraint object
is *literally unchanged* (`(param(p) * x) − 4 ≤ 0`) and the solve returns the
nominal optimum 4.0 (true robust: 2.667). The vector `dm.sum(mu*x)` form is likewise
untouched. The user asked for a robust model and got the nominal one, silently.

**Fix:** per house philosophy, refuse loudly first: after `build()`, run
`_contains_uncertain_param` over all constraint bodies and the objective; any
remaining uncertain parameter ⇒ `NotImplementedError` naming the constraint and the
supported patterns. Then extend coverage: elementwise `p*x` + `sum` reduces to the
same SOC penalty (coefficient extraction like the polyhedral path, penalty
`ρ‖Σ^{1/2}coeff(x)‖`), and scalar/RHS cases are one-liners. The same
"no-silent-leftover-parameters" post-check belongs in the **box and polyhedral**
builders too — it converts every unknown-pattern gap in this module from silent to
loud in one stroke.

### RO-3. `budget_uncertainty_set` is not the Bertsimas–Sim set

The H-representation stored is `box ∩ {|Σξ_j/δ_j| ≤ Γ}` — the all-plus and
all-minus facets only — while the true set is `box ∩ {Σ|ξ_j|/δ_j ≤ Γ}` (2^k facets
or a lifted formulation). The compact attributes (`_delta`, `_gamma`, `_is_budget`)
that the constructor stores "for the formulation to use" are **read by nothing**
(grep-verified). Consequence: a strict superset, hence sound but over-conservative —
support of direction `[1,−1]` at k=2, δ=1, Γ=1 is **2.0 vs the true 1.0**
[CONFIRMED]. The docstring's "encodes the full polyhedral representation" is false
for k ≥ 2, and the "price of robustness" Γ-interpolation the API promises does not
hold in mixed-sign directions.

**Fix:** use the standard lifted representation — variables `(ξ, u)` with
`−u ≤ ξ/δ ≤ u`, `Σu ≤ Γ`, `u ≤ 1` — which is linear and dualizes exactly through
the existing polyhedral machinery (the dual just gains the `u`-rows); or special-case
`_is_budget` in the polyhedral formulation with the closed-form Bertsimas–Sim
protection term. Regression test: support function of mixed-sign directions equals
the hand-computed B–S value; end-to-end price-of-robustness monotonicity in Γ with
the correct Γ=0 (nominal) and Γ=k (box) endpoints.

### RO-4 / RO-5 / RO-6. Remaining silent-wrong paths

- **RO-4**: `/` and `**` propagate sign unflipped — `c/p` (maximize) substitutes
  `p_upper`, giving the *smallest* value: non-conservative. Non-monotone `p**2`
  over a straddling interval picks one endpoint. Fix: refuse (`NotImplementedError`)
  on any uncertain parameter under `/`, `**`, or a non-monotone `FunctionCall` —
  monotone functions (`exp`, `log`, `sqrt`) may keep substitution with a
  documented monotonicity table.
- **RO-5**: `_eval_constant_expr` returns `0.0` for unrecognized nodes — a silent
  zero-protection sink on the polyhedral constant-coefficient path (and its
  `Constant` case already does the `np.sum` vector collapse). Fix: evaluate via
  the existing DAG compiler (`compile_expression` on a variable-free expression)
  or raise on unknown nodes.
- **RO-6**: the documented `np.sign(np.sum(v))` vector collapse. Per-component
  sign handling falls out of the same coefficient-extraction fix as RO-1; until
  then, at minimum warn when a mixed-sign constant multiplies an uncertain
  parameter (`np.any(v>0) and np.any(v<0)` is one line).

### RO-7. Arbitrary magnitude caps silently reshape the model

Three places invent finite bounds to placate the IPM: `t_ub = max|var bound|+1`
(box abs-value auxiliaries — but `coeff(x)` can exceed any variable bound, e.g.
`100·x`), `lam_ub = Σ|b|+100` (polyhedral duals — needed λ scales with `|coeff(x)|`,
unrelated to `b`), and `|Y_j| ≤ max(ub−lb)` (ADR policy slopes — the optimal slope
for small-δ uncertainty legitimately exceeds the recourse range). Each cap, when
binding, silently cuts the feasible/policy space: over-conservative results with no
diagnostic. Fix: derive bounds from the actual expressions where cheap (interval
evaluation of `coeff(x)` — the machinery exists in
`_jax.convexity.interval_eval`), else leave unbounded and let the solver's
large-bound warning fire; if a cap is kept, check it at the solution (`t` or `λ`
within 1e-6 of its cap ⇒ warn loudly).

---

## 3. Smaller items

- **RO-8**: equality constraints with uncertain parameters are transformed
  meaninglessly (box substitutes one worst case; ellipsoidal would add a one-sided
  penalty). Robust equalities with continuous uncertainty are infeasible except in
  degenerate cases — refuse loudly.
- **RO-9**: `RobustCounterpart` requires uniform set *types* and tells users to
  apply two counterparts sequentially for mixed uncertainty — but sequential
  application is only correct because each pass eliminates its own parameters;
  worth a test (none exists) since pass-2 re-walks pass-1's rewritten expressions.
- **RO-10**: builder-resident constraint rows (`m.constraint` fast path) are
  invisible to all three formulations — same class as GP-1/M12. Parameters cannot
  appear in builder rows (numeric-only API), so today this only means such rows
  are correctly left alone; add the shared `_has_builder_only_rows()` guard anyway
  once it lands (gp-review GP-1) for uncertain-*variable*-coefficient future work.
- **RO-11** (perf): the ROADMAP's own items are accurate — coefficient extraction
  by substitute-and-subtract builds duplicate unsimplified trees (each `coeff_j`
  embeds two full copies of the constraint body; the polyhedral path does this k
  times per constraint → O(k·|expr|) DAG blowup), and the polyhedral dual pass
  introduces `n_rows` λ per (constraint × parameter) with no sparsity. Real, but
  secondary to the soundness work.

---

## 4. Test coverage gaps

The 89-test suite validates the blessed patterns well (Ben-Tal 1.1.1, budget-set
*construction*, ADR mechanics, dual-variable bookkeeping). Every confirmed finding
sits in an untested region:

1. **No negative-domain variable** appears in any box test (RO-1).
2. **No non-MatMul parameter pattern** in any ellipsoidal test (RO-2) — and no test
   asserts the *absence of leftover uncertain parameters* after `formulate()`,
   which would have caught RO-2 across all three formulations at once. That
   invariant check is the single highest-value test to add.
3. **No support-function or price-of-robustness test** for budget sets in
   mixed-sign directions (RO-3); existing budget tests exercise construction and
   nonnegative-coefficient models only.
4. No division/power/nonmonotone-function uncertainty test (RO-4); no
   constant-coefficient polyhedral constraint written with Sum/MatMul shapes
   (RO-5); no mixed-sign constant vector (RO-6); no scale-stress test that would
   trip the caps (RO-7); no uncertain-equality refusal test (RO-8); no
   sequential mixed-set test (RO-9).
5. A general pattern worth adopting from `test_gp_hull.py`: **randomized soundness
   fuzz** — sample models from the supported grammar, formulate, solve, then check
   the returned x against N random in-set realizations of every parameter. That
   one harness would have caught RO-1, RO-2, RO-4, RO-5, and RO-6.

## 5. Adjacent solver-layer finding (ADJ-1)

Discovered while building the RO-2 control, independent of this module:

```python
x = m.continuous("x", shape=(2,), lb=0, ub=10)
m.subject_to(dm.sum(np.array([1.,1.]) * x) <= 4)
m.maximize(dm.sum(x))
m.solve()        # -> "optimal", objective 0.0 at x=(0,0); true optimum 4.0
```

Shape isolation: `x[0]+x[1] <= 4`, `c @ x <= 4`, and `dm.sum(x) <= 4` all solve
correctly under maximize; only the `SumExpression(c * x)` body shape fails — and
`minimize(-dm.sum(x))` with the same constraint is correct, so it is a
sense-handling defect on whatever extraction path that body shape selects. Same
layer as modeling-review **M1** (`_jax/problem_classifier.py` extraction); fold
this repro into the M1 fix's regression suite. Until M1/ADJ-1 land, RO test results
involving vectorized bodies + maximize are not trustworthy evidence of RO-layer
correctness.

---

## 6. SOTA assessment

### 6.1 What it is

A static-RO + affine-decision-rule layer over the discopt solver: the three
canonical uncertainty sets, worst-case reformulations (substitution / SOC penalty /
LP duality), and ARO via ADR with correct deferral of the bilinear terms to the
box linearization path. The *architecture* — uncertainty sets as first-class
objects bound to `Parameter`s, formulation strategies behind one `RobustCounterpart`
entry point, ADR as a composable pre-pass — matches the reference designs (RSOME,
ROmodel for Pyomo, JuMPeR/BilevelJuMP lineage) and is the right shape to grow.

### 6.2 Against the state of the art

- **vs. RSOME (Chen & Xiong) / ROmodel / PyROS**: those tools handle *general
  affine* uncertain constraints — arbitrary linear appearance of parameters, both
  sides, any coefficient sign — because they extract coefficients algebraically
  rather than pattern-matching the DAG. That is exactly the gap RO-1/RO-2 expose:
  the module robustifies *patterns*, not *the affine dependence itself*. The fix
  direction (coefficient extraction everywhere, already half-built in the
  polyhedral path) is also the SOTA-alignment direction.
- **PyROS** (Pyomo) is the interesting contrast for *this* repo: it does
  cutting-set robust optimization for **nonlinear** models with a global solver in
  the loop — which discopt uniquely has in-house. A cutting-set loop (solve master
  at scenario subset → globally maximize constraint violation over ξ ∈ U → add
  scenario, repeat) would give *certified* robust feasibility for general
  nonconvex `g(x, ξ)` — beyond what RSOME-class reformulation tools can do, and a
  natural flagship use of the global engine. This is the module's genuine
  SOTA-plus opportunity (and it makes RO-4's "nonlinear in p" limitation moot for
  models routed to it).
- **Missing standard tiers**, roughly in the roadmap's own (correct) order: DRO
  (moment / Wasserstein ambiguity), multi-period ADR with information-basis
  restriction (non-anticipativity), data-driven set calibration. The roadmap is a
  good plan; its priority list should be re-ordered to put the §2 soundness holes
  *before* all of it.

### 6.3 Verdict

The architecture and the duality/ADR cores are sound and well-referenced, and the
project's own ROADMAP shows accurate self-knowledge of the limitations. But the
shipped default behavior violates the module's contract in silent ways: a box
counterpart that under-protects for sign-indefinite variables (RO-1), an
ellipsoidal counterpart that quietly returns the nominal model for most input
shapes (RO-2), and a budget set that is not Bertsimas–Sim (RO-3). None of these are
exotic inputs. The single most important change is cultural-mechanical: **every
formulation must end with a "no uncertain parameters remain" assertion** — that
converts this module's entire silent-failure class into loud errors, after which
the coverage can grow to RSOME parity pattern by pattern, and the cutting-set
global-RO direction can make discopt's RO offering genuinely differentiated.

---

## 7. Implementation plan (for Opus)

House rules per CLAUDE.md (feature branch + PR, task IDs, failing-first regression
tests; run `pytest python/tests/test_robust*.py test_affine_decision_rule.py`,
`-m smoke`, adversarial suite; report results in PR).

### Phase 1 — stop the silent failures (PR `fix(ro): RO-0..RO-2`)

| ID | Task | Acceptance criteria |
|----|------|---------------------|
| RO-0 | **Leftover-parameter assertion**: after every `build()`, walk all constraint bodies + objective with `_contains_uncertain_param`; any hit ⇒ `NotImplementedError` naming the constraint and supported patterns | Ellipsoidal `p*x` / `sum(p*x)` / RHS repros raise instead of returning nominal (fail-silent on main); all 89 existing tests still pass |
| RO-1 | Box: route sign-indefinite variable coefficients through the existing `\|coeff\|`-linearization (nonnegativity check per flat component); keep sign-tracking only for provably-safe cases | `maximize x s.t. p·x ≤ −1` repro returns −2 (returns −0.667 on main); randomized soundness fuzz (§4.5, N=200 realizations) green on the box grammar |
| RO-2 | Ellipsoidal: coefficient-extraction-based SOC penalty for scalar, elementwise+sum, and RHS patterns (post-RO-0 these currently raise) | The three repros return the analytic robust optima (scalar: 2.667; vector: 2.343); MatMul path bit-identical to today |

### Phase 2 — method fidelity + caps (PR `fix(ro): RO-3..RO-7`)

| ID | Task |
|----|------|
| RO-3 | Budget set: lifted `(ξ,u)` polyhedron (or `_is_budget` closed form in the polyhedral dual); support-function tests in mixed-sign directions; Γ-endpoint tests (Γ=0 nominal, Γ=k box); delete or use the dead `_delta/_gamma/_is_budget` attributes |
| RO-4 | Refuse `/`, `**`, non-monotone `FunctionCall` over uncertain params; monotone table for `exp/log/sqrt` |
| RO-5 | Replace `_eval_constant_expr` with a `compile_expression` call (raise if variables present); delete the silent-zero fallback |
| RO-6 | Mixed-sign constant × uncertain param: correct per-component handling via the RO-1 extraction machinery (or warn until it lands); remove the ROADMAP caveat once fixed |
| RO-7 | Replace the three magnitude caps with interval-derived bounds (`_jax.convexity.interval_eval`) or unbounded+solver-warning; add at-cap detection warnings; scale-stress regression test (coefficients ×1e4) |
| RO-8/9 | Refuse uncertain equalities; add the sequential mixed-set test |

### Phase 3 — SOTA direction (design doc + entry experiment first)

1. **Cutting-set robust optimization with the global solver as the pessimization
   oracle** (PyROS-style, but with certified pessimization): master/adversary loop,
   ξ-feasibility certified by the global engine. Entry experiment: a small
   nonconvex-`g` corpus where reformulation is impossible; kill criterion:
   pessimization oracle cost makes the loop slower than naive scenario sampling at
   equal guarantee.
2. Multi-period ADR example + information-basis restriction (roadmap item 2/5) —
   also resolves the roadmap's "no example where ADR beats static" gap.
3. DRO (moment sets first — they reduce to SOC/SDP counterparts the solver can
   already express; Wasserstein after).
4. Fold **ADJ-1** into the modeling-review M1 extraction fix with the §5 repro as
   a regression test.
