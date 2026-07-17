# GP Module Review — Correctness, Thoroughness, Performance, and SOTA Assessment

**Date:** 2026-07-03
**Scope:** `python/discopt/gp/__init__.py` (454 lines), its recognizer
`python/discopt/_jax/convexity/posynomial.py` (300 lines), and the GP dispatch in
`python/discopt/solver.py` (explicit `solver="gp"` + the **auto-GP fast path** that
fires inside every default `Model.solve()`). Tests: `test_gp.py`, `test_gp_corpus.py`,
`test_gp_hull.py`.
**Method:** Full read of all three layers; every finding below reproduced end-to-end
against the installed package. Baseline: 47 tests passed (fast suite, 9.6 s).

Because GP is auto-detected on the *default* solve path, any classification or
result-mapping defect here is a defect of plain `m.solve()` for GP-shaped models —
the stakes are solver-wide, not module-local.

---

## 1. Summary of findings

| # | Severity | Component | Finding |
|---|----------|-----------|---------|
| GP-1 ✅ RESOLVED 1dc3278 | **P0 correctness** | `gp/__init__.py:classify_gp` + auto path | Fast-path constraint families (`m.constraint(...)`, `add_linear_constraints`) live only in the Rust builder and are **invisible to `classify_gp`** — the auto-GP path solves the model **without them** and certifies the wrong optimum [CONFIRMED: 0.1 vs true 0.5]. **FIXED (X-1):** `classify_gp` returns `None` when `model._has_builder_only_rows()`, so the model falls back to spatial B&B; repro now solves 0.5. Regression: `test_x1_builder_resident_rows.py::test_gp1_*`. **CORRECTION (#681, 2026-07-17):** the "spatial B&B sees the builder rows" premise held only for *single-variable* fast rows (FBBT folds them into bounds — why the GP-1 `xv<=2` repro passes); the JAX spatial-B&B evaluator + McCormick relaxer read only `model._constraints`, so *multi-variable* builder rows were still dropped → a false optimum on the core nonlinear path. Now `solve_model` materializes builder linear rows into `model._constraints` before the nonlinear path (`Model._materialize_builder_linear_rows`). Regression: `test_issue_681_builder_rows_nonlinear.py`. |
| GP-2 ✅ RESOLVED | **P1** | `gp/__init__.py:solve_gp` | Certified GP optima are returned with **`gap_certified=False`** (and non-optimal statuses carry a log-space `gap` with `bound=None`) [CONFIRMED] **[see STATUS below]** |
| GP-3 ✅ RESOLVED | P2 recognition | `posynomial.py` | No distribution of products over sums: `2*(h*w + h*d) <= 10` is refused — the textbook Boyd box-volume GP written naturally **silently misses the fast path** [CONFIRMED] **[see STATUS below]** |
| GP-4 ✅ RESOLVED | P2 recognition | `posynomial.py` | `dm.sum(..., over=set)` (`SumOverExpression`) is refused — the natural indexed form of a posynomial never classifies [CONFIRMED] **[see STATUS below]** |

> **STATUS (2026-07-05, PR `fix(gp): GP-2/GP-3/GP-4`, #413, branch
> `fix-gp-cluster-gp2-gp3-gp4`).** GP-1 resolved earlier under root finding X-1.
> GP-2/GP-3/GP-4 now resolved.
>
> - **GP-2 (P1) — RESOLVED.** `solve_gp` builds the `SolveResult` in one shot with
>   its final fields, mapping the convex optimum back to x-space as `bound`, so
>   `SolveResult.__post_init__` validates the real bound rather than silently
>   downgrading `gap_certified` off a transient `bound=None`. **Certification is
>   rigorous and claimed ONLY when** (a) the model classified as an *exact* GP
>   (`gp is not None` ⇒ exact posynomial/monomial transform, no approximation) AND
>   (b) the convex solve returned `"optimal"` with (c) a finite recovered
>   objective. Any other status → `bound=None`, `gap=None`, `gap_certified=False`,
>   `convex_fast_path=False` (a sound under-claim; never a fabricated/log-unit
>   bound). Soundness argument: a GP is a *convex* program in `y = log x` and the
>   transform here is exact, so a certified-optimal convex solve *is* a certified
>   global optimum with zero gap — the claim is a theorem, not a heuristic. Also
>   attaches `_model` (GP-6 rider). Tests: `TestGP2Certification` (two positive
>   cases fail on main; two negative cases pin no-over-claim).
> - **GP-3 (P2) — RESOLVED.** `_flatten_sum_terms` distributes `const*(sum)` /
>   `monomial*(sum)` (`(Σaᵢ)(Σbⱼ)=Σaᵢbⱼ`, exactly posynomial-preserving) under a
>   ≤64-term budget (`_MAX_DISTRIBUTE_TERMS`); over-budget → refuse (spatial B&B
>   fallback, sound). Factored Boyd box-volume GP now classifies + solves to the
>   same certified optimum as the expanded form. Negatives: signomial-in-product
>   and oversized product-of-sums refuse. `TestGP3Distribution`.
> - **GP-4 (P2) — RESOLVED.** `_flatten_sum_terms` recurses into
>   `SumOverExpression.terms`; the top-level rejection in `is_posynomial` is
>   removed. Indexed `dm.sum(..., over=set)` posynomials classify + solve (in the
>   objective and in constraints). Negatives: indexed-`sin` and indexed-signomial
>   refuse. `TestGP4IndexedSum`.
>
> **Cert-baseline neutrality:** zero of the 41 `cert-baseline.jsonl` instances
> classify as GP before OR after (all carry integer / non-positive-lb variables →
> `classify_gp` early-bails), so the changed recognizer is never reached for any
> of them — routing, `node_count`, `objective`, `status`, `gap_certified`
> provably unchanged; `incorrect_count == 0`.
| GP-5 | P2 | `solver.py` auto-path guard | The auto-path guard checks incumbent/node/iteration callbacks but **not `lazy_constraints`**; a GP-shaped model with a lazy-constraint generator auto-routes to GP and silently ignores it (explicit `solver="gp"` at least warns). `initial_solution` is likewise silently dropped on both GP paths [BY INSPECTION] |
| GP-6–GP-9 | P3 | various | Result-object and hygiene items (§4) |

Checked and found **correct** (no action) — and this module deserves credit for it:

- **The log-space reformulation math is right**: monomial → affine (`aᵀy + log c`),
  posynomial → Σexp(affine), division by the rhs monomial, monomial equality →
  affine equality, monomial min/max via the affine log. End-to-end, the expanded
  box-volume GP (maximize `h·w·d`) solves via the fast path to **5.590170**,
  matching spatial B&B (`solver="bb"`) to 1e-6 — the maximize-sense sign handling
  is verified correct.
- **The recognizer is sound (conservative by construction)**: every acceptance path
  validates signs; non-integer powers of negative bases and non-finite coefficients
  are rejected; `lb > 0` is enforced both per-leaf and model-wide; integer/binary
  variables and non-algebraic constraint kinds (indicator, disjunctive, SOS,
  logical) are rejected; vector variables are rejected (so the modeling-review M1
  broadcast-aggregation trap cannot leak into GP classification); `SumOverExpression`
  is rejected *soundly* (GP-4 is a completeness gap, not a soundness one).
- **The x-space objective is recomputed from the recognized structure** rather than
  trusted from the log-space solver — the right call, especially for monomial
  objectives where the log model optimizes `log f`.
- **`is_log_convex` / x-space-convexity separation**: the docstring explicitly keeps
  log-space convexity out of the x-space verdict because folding it in would
  mis-gate the convex fast path — a soundness hazard reasoned about *in advance*
  and pinned by `test_gp_corpus` (`is_log_convex=True` while
  `classify_model=False`). Exemplary discipline.
- **Explicit `solver="gp"` has a model ignored-options warning** enumerating every
  B&B option the path cannot honor; `solve_gp` refuses streaming loudly; the
  infeasibility certificate transfer (infeasible log model ⇒ infeasible GP) is
  sound because the reformulation is an equivalence on the declared box.

---

## 2. P0: builder-resident constraints are dropped by the auto-GP path

`classify_gp` inspects `model._constraints` only. Constraint families emitted by
`Model.constraint(...)` via the linear fast path (and rows from
`add_linear_constraints`) live **only in the Rust builder**
(`model._builder_linear_blocks`); `classify_gp` never looks there.

**Reproduction** (default solve path, no explicit solver):

```python
s  = m.set("s", ["a"])
xv = m.continuous("xv", over=s, lb=0.5, ub=10)
m.minimize(1 / xv["a"])                       # monomial -> GP-shaped
m.constraint(s, lambda i: xv[i] <= 2, "cap")  # fast path: builder-only rows
m.solve()   # -> status "optimal", objective 0.1, xv = 10
```

The cap `xv <= 2` is silently dropped: true optimum **0.5**, returned **0.1**,
certified optimal. The family's `fast=True` flag and `len(m._constraints) == 0`
confirm the mechanism. This is the same failure *class* as the modeling review's
M12 (builder rows invisible to Python-side introspection), here escalated to a
wrong certificate on the **default path**.

**Fix (minimal, safe):** `classify_gp` returns `None` whenever the model has
builder-resident rows — `model._builder_linear_blocks`, a non-`None`
`_builder_linear_objective` / `_builder_quadratic_objective`, or any state in
`model._builder` not mirrored in `_constraints`. Introduce one shared predicate on
`Model` (e.g. `_has_builder_only_rows()`) so future classifiers (convexity,
problem-class) can use the same guard — an audit of those classifiers for the same
blind spot should ride along in the PR. **Regression test:** the repro above must
return 0.5 (via B&B) and `classify_gp` must return `None`; plus the mirrored
`add_linear_constraints` variant.

(A later enhancement may *incorporate* builder rows instead of refusing — a linear
row is GP-compatible only in special sign patterns, so refusal is the correct
default.)

---

## 3. P1: certified GP optima report `gap_certified=False`

`solve_gp` constructs `SolveResult(status=...)` **first** — at which point
`SolveResult.__post_init__` sees `bound=None` and (correctly, per its soundness
guard) downgrades `gap_certified` to `False` — and only *then* assigns
`result.bound = result.objective; result.gap = 0.0`. The flag is never restored:

```
status=optimal  obj=0.25  bound=0.2499...  gap=0.0
gap_certified=False   convex_fast_path=True      # <- contradiction
```

A GP solve is precisely a zero-gap certified global optimum (the module's own
comment block says so); benchmark gates that count certified solves will miscount
every GP-fast-path instance as uncertified. Inverse of the usual hazard
(under-claiming, not unsound) but a results-integrity bug all the same.

Also in the same block: for **non-optimal** statuses `result.gap` is copied from
the log-space solve (log-objective units) while `bound` is `None` — a gap with no
bound in the wrong units.

**Fix:** build the `SolveResult` in one shot with its final fields
(`SolveResult(status=..., objective=..., bound=..., gap=0.0, x=..., ...)`) so
`__post_init__` validates the *actual* values; drop the log-space `gap` for
non-optimal statuses (set `None`); carry over `node_count=0` and the timer fields.
**Tests:** pin `gap_certified is True` + `gap == 0.0` on a solved GP (fails on
main); pin `gap is None` on a time-limited GP.

---

## 4. Recognition-breadth and hygiene items

**GP-3. Products over sums are not distributed.** `2*(h*w + h*d)` parses as a
single `*` term whose right factor is a sum → `_parse_monomial` → `None` → the
whole model silently loses GP status and routes to spatial B&B (correct answer,
none of the GP benefits, no indication why). Confirmed: the factored Boyd
box-volume GP is "NOT GP", its expanded form is GP and fast-paths. Fix: in
`_flatten_sum_terms` (or a pre-pass), distribute `const * (sum)` and
`monomial * (sum)` with a bounded expansion budget (e.g. ≤ 64 resulting terms) —
distribution of a monomial over a posynomial is exactly posynomial-preserving, so
this is soundness-neutral by construction.

**GP-4. `SumOverExpression` is refused.** `dm.sum(lambda i: 1/x[i], over=s)` — the
API's own preferred aggregation — never classifies; the manually expanded identical
model does [CONFIRMED]. Fix: have `_flatten_sum_terms` recurse into
`SumOverExpression.terms` (they are ordinary expressions); remove the top-level
defensive rejection in `is_posynomial` at the same time. One-case change + tests.

**GP-5. Auto-path guard gaps.** `_has_bb_callbacks` omits `lazy_constraints`, so a
GP-shaped model with a lazy-constraint generator silently solves without it — the
generator's cuts are part of the model semantics, so this is a (narrow) correctness
hole, not just an option drop. Add it to the guard. Separately, `initial_solution`
is computed by `Model.solve` and then silently unused by both GP paths — harmless
for correctness (convex), but either forward it as the NLP start or include it in
the ignored-options warning.

**GP-6.** `solve_gp` called directly (public API) returns a result with no `_model`
attached, so `.gradient()`/`.explain()` degrade; attach `gp.original`.

**GP-7.** `_var_offset` falls back to name matching (`v.name == target.name`) —
inherits the modeling review's cross-model aliasing hazard (M3); once M3's
ownership validation lands, tighten this to identity-only.

**GP-8.** `_split_signed_monomials` silently drops terms with `|coeff| ≤ 1e-12` —
a tolerance-based model edit. Negligible in practice; document it next to `_TOL`
or raise the refusal instead (house style prefers the latter).

**GP-9.** Parameters are accepted as coefficients/exponents and their values are
**baked at classification time**. Because both GP entry points re-classify per
call, parameter sweeps behave correctly today — but nothing tests
`result.gradient(param)` through the GP path, and the baking is undocumented. Add
a test or document the snapshot semantics.

**Performance:** nothing concerning — classification is a linear DAG walk with
early bail-outs (`classify_gp` exits on the first integer variable or
non-positive bound, as the solver comment promises), and the auto-path probe cost
on non-GP models is negligible. The log model build is O(terms).

---

## 5. Test coverage gaps

Baseline: 47 passed (fast). Existing coverage is good on classification
polarity (signomial/integer/nonpositive refusals), the log-model structure, the
corpus auto-routing, and the log-convex/x-convex separation. Missing:

1. **Builder-resident constraints** (GP-1) — no test builds a GP-shaped model with
   `m.constraint(...)` or `add_linear_constraints`.
2. **`gap_certified` / result-field pinning** (GP-2) — no test inspects the
   certification fields of a GP result.
3. **Recognition breadth** (GP-3/GP-4) — corpus models are all hand-expanded;
   add factored and indexed-sum forms (fail before the Phase-2 fixes, pass after).
4. **Time-limited / infeasible GP results** — gap/bound field semantics.
5. **`result.gradient` through the GP path** (GP-9).
6. `test_gp_hull.py` covers the adjacent `monomial_log_envelope` soundly (random
   soundness fuzz — good pattern); it is relaxation-layer, not this module.

---

## 6. SOTA assessment

### 6.1 What it is

Whole-model **standard-form GP recognition + log-space convex reformulation +
auto fast path** — the classic pattern of GPkit and of CVXPY's DGP entry point,
integrated so that `m.solve()` transparently upgrades GP-shaped models from
spatial B&B to a single convex NLP with a global certificate. The engineering
around the edges is unusually careful: the deliberate x-space/log-space verdict
separation, the conservative-by-construction recognizer, and the ignored-options
warning on the explicit path are all things comparable layers commonly get wrong.

### 6.2 Against the state of the art

- **vs. CVXPY DGP (Agrawal et al. 2019)** — the reference point the module itself
  cites. DGP is a *compositional* log-log curvature ruleset: it certifies
  log-log-convexity through `max`, `exp`, `log` of monomials, ratios, powers of
  posynomials, etc., far beyond standard form. This module is whole-model
  standard-form only (its docstring names per-expression log-curvature lattice
  propagation as future work). For the standard-form subset it is at parity; in
  breadth it is well behind DGP — and GP-3/GP-4 show it is currently behind even
  on *syntactic* breadth for standard-form models.
- **vs. GPkit** — GPkit's main extra is **signomial programming** (sequential GP
  approximation, local). Here discopt has a genuinely differentiated opportunity:
  signomials (posynomial − posynomial) are exactly the nonconvex structure a
  *global* solver can handle rigorously, and the repo already ships
  `monomial_log_envelope` (`_jax/symbolic/gp_hull.py`, soundness-fuzzed) — i.e.
  the relaxation primitive for exploiting log-structure *inside* B&B exists. A
  "global signomial programming" capability (GP-informed log-space envelopes as
  B&B relaxations, certified — vs. GPkit's heuristic SP loop) would be beyond
  current tool SOTA, not just at it.
- **Exponential-cone solvers**: MOSEK-class handling of GP via the exp cone is the
  conic-world SOTA; solving the Σexp NLP with the in-house IPM is a legitimate
  equivalent here and not a gap.
- **GP duals**: the classical GP dual (weight/sensitivity interpretation) is not
  surfaced; low priority but a known nicety in dedicated GP tools.

### 6.3 Verdict

A **small, sound, well-integrated GP layer whose math checks out end-to-end** —
let down by one integration hole (GP-1: builder-resident constraints, wrong
certified answers on the default path), one results-integrity slip (GP-2:
certified optima labeled uncertified), and a recognizer whose syntactic reach is
narrower than its own cited references (GP-3/GP-4: the textbook GP in factored or
indexed form silently misses the path). Fix those four and it is state of practice;
the signomial-global direction (§6.2) is the SOTA-plus opportunity and fits the
repo's global-certification identity better than chasing full DGP breadth.

---

## 7. Implementation plan (for Opus)

House rules per CLAUDE.md (feature branch + PR, task IDs in titles, regression
tests failing-before/passing-after; run `pytest python/tests/test_gp*.py`,
`-m smoke`, and the adversarial suite; state results in the PR).

### Phase 1 — correctness (PR `fix(gp): GP-1..GP-2`)

| ID | Task | Files | Acceptance criteria |
|----|------|-------|---------------------|
| GP-1 | `classify_gp` → `None` when the model carries builder-resident rows/objective; add shared `Model._has_builder_only_rows()`; audit `classify_model`/`classify_problem` for the same blind spot in the PR description | `gp/__init__.py`, `modeling/core.py` | Repro returns 0.5 via B&B (fails on main at 0.1); `add_linear_constraints` variant likewise; corpus tests unchanged |
| GP-2 | Construct the GP `SolveResult` with final fields in one shot; `gap=None` on non-optimal statuses; attach `_model` (GP-6); forward timers | `gp/__init__.py:403-443` | `gap_certified is True` and `gap == 0.0` pinned on a solved GP (fails on main); time-limited GP has `gap is None`; `benchmarks` certified-count unaffected for non-GP instances |

### Phase 2 — recognition breadth + guard hygiene (PR `feat(gp): GP-3..GP-5`)

| ID | Task |
|----|------|
| GP-3 | Distribute `const*(sum)` / `monomial*(sum)` in the flattener with an expansion budget (≤64 terms; over-budget → refuse as today). Soundness-neutral by construction — state the argument in the PR |
| GP-4 | Recurse into `SumOverExpression.terms` in `_flatten_sum_terms`; drop the top-level rejection. Tests: factored box-volume GP and indexed-sum GP now classify and fast-path (both fail on main) |
| GP-5 | Add `lazy_constraints` to the auto-path guard; forward `initial_solution` to the log-model solve (map `x0 → log x0`) or add it to the ignored-options warning |
| GP-8/9 | Refuse (or document) the 1e-12 term drop; test or document parameter-snapshot semantics incl. `result.gradient` through the GP path |

### Phase 3 — SOTA direction (design doc + entry experiment first)

1. **Global signomial programming**: use `monomial_log_envelope` to build log-space
   relaxations of signomial constraints inside spatial B&B (bound-changing —
   differential bound protocol per CLAUDE.md §5, feature-flagged, default-off).
   Entry experiment: root-gap and node-count on a small signomial corpus (e.g.
   MINLPLib `ex*`/`gptest`-style instances) with vs. without the envelope cuts;
   kill criterion: no measurable root-gap reduction on ≥ half the corpus.
2. **Log-log curvature lattice** (the docstring's own future-work item) for
   DGP-style breadth — only after Phase 2 shows standard-form recognition is no
   longer the binding constraint.
