# Validation (Examiner) Module Review — Correctness and Coverage

**Date:** 2026-07-03
**Scope:** `python/discopt/validation/` (`examiner.py` — 927 lines, `__init__.py`)
and `test_examiner.py`.
**Method:** Full read of `examiner.py`; behavior verified end-to-end — confirmed the
checks *catch* infeasible points and lying objectives, then probed the coverage
boundary. Baseline: **16 passed, 1 skipped** (2.3 s).

The examiner is discopt's post-solve KKT validator (GAMS-Examiner-style, pure
Python): re-evaluate at the returned point, check the seven first-order KKT
residuals + integrality + objective consistency, recovering multipliers via LSQ
when the solver doesn't expose them. It is invoked by `Model.solve(validate=True)`
and by the benchmark suite's `assert_examined`. Its job is to be the last line that
catches a wrong `SolveResult` — so a blind spot here is a blind spot in the
project's own correctness safety net.

---

## 1. Summary of findings

| # | Severity | Component | Finding |
|---|----------|-----------|---------|
| VAL-1 ✅ RESOLVED 1dc3278 | **P1 coverage hole** | `examiner.py:175` (`_row_metadata` via `NLPEvaluator`) | The examiner validates only Python-side `model._constraints`; **builder-resident (fast-API) rows are invisible**, so a point grossly violating a `m.constraint(...)` / `add_linear_constraints` floor is certified **PASS** [CONFIRMED: infeasible `xv=[0,0]` against a `xv>=3` fast-path floor → `passed=True`, evaluator reports 0 constraints]. **FIXED (X-1):** a new `primal_con_feas (builder rows)` check evaluates the builder linear rows directly (via `export._common.iter_builder_linear_rows`) in the examiner's flat-variable frame; the violating point is now `passed=False`. Additive only — no existing check relaxed. Regression: `test_x1_builder_resident_rows.py::test_val1_*`. |
| VAL-2 | P2 | `examiner.py:810-843` | An LSQ dual-recovery exception reports stationarity/primal_cs/dual_cs as **FAILED**, so a numerical hiccup in recovery makes `assert_examined` reject a genuinely-correct point (false alarm in the benchmark gate). Should mark recovery *inconclusive*, not failed [BY INSPECTION] |
| VAL-3 | P3 | `examiner.py:242` | `obj_consistency` uses only the objective's own magnitude for its relative tolerance (`obj_tol + 1e-4·|obj|`); for an objective near 0 whose *terms* are large with cancellation, re-eval float noise can exceed the tiny tolerance. Rare, but a scale-by-terms floor would harden it [BY INSPECTION] |

Checked and found **correct** — and this is a genuinely well-built validator:

- **It catches wrong points** [CONFIRMED]: a real optimum passes (merit 2e-14); an
  infeasible point fails on `primal_con_feas`; a feasible point with a
  tampered objective fails on `obj_consistency`. The three core failure modes the
  validator exists to detect all fire correctly.
- **Objective sense handling** is right: `evaluate_objective` returns min-form and
  is negated for MAXIMIZE before comparing to the reported objective; the relative
  tolerance matches CLAUDE.md's `rel=1e-4`.
- **GDP handling** is correct and thoughtful: disjunctive/indicator/SOS/logical
  models are reformulated via `reformulate_gdp` before KKT (stationarity is only
  well-defined on the smooth reformulation the solver actually optimized), and
  `result.x` already carries the auxiliary/disaggregated variables. Verified an
  `if_then` model examines cleanly (13 checks).
- **Dual-side rigor**: it uses solver-supplied multipliers when present (true
  direct check) *and* independently recovers them from the active set via
  sign-bounded LSQ, then cross-checks the two with a permissive tolerance — a
  genuinely strong design. Integer columns are dropped for the continuous KKT
  system (matching Examiner's fix-and-recheck), sign conventions per row sense are
  handled consistently between the solver-dual and recovered paths (including the
  `>=`-row flip and un-flip), and complementary slackness is checked on both the
  primal (`λ·|x−bound|`) and dual (`|μ|·|body−rhs|`) sides.
- **Fail-safe structure**: `examine` never raises (except the deliberate
  `x is None` guard), degrades to a merit score, reports per-row worst violators,
  and separates unscaled from Examiner-scaled constraint feasibility.
- It is used as **advisory** on the `solve(validate=True)` path (errors swallowed,
  attached to `result.validation_report`) — it does **not** gate the certificate,
  so VAL-1's blind spot cannot silently upgrade a wrong solve to "certified" there.
  Its gate role is confined to the benchmark suite's `assert_examined`.

---

## 2. VAL-1: the fast-API blind spot

`_row_metadata` iterates `evaluator._source_constraints`, and `NLPEvaluator` is
built only from `model._constraints`. Constraints emitted by the fast path
(`Model.constraint(...)` when it lowers to `add_linear_constraints`, and direct
`add_linear_constraints` calls) live in `model._builder_linear_blocks` / the Rust
builder and never enter `_constraints`. Confirmed: `NLPEvaluator.n_constraints`
is **0** for a two-row fast-API model, and `examine` on a point violating that
model's `xv >= 3` floor returns `passed=True` with only variable-bound,
objective, and (vacuous) KKT checks run.

Why this matters more here than in the export/GP reviews: the examiner is the
*validator*. Its entire premise is "re-check the returned point against the
model." Silently checking a *subset* of the model means an infeasible incumbent on
any fast-API-constructed model (the recommended path for large linear
families — DAE collocation, network flow, assignment) passes examination. In the
benchmark gate (`assert_examined`), that is a wrong point the safety net waves
through.

**Fix:** build the examiner's row set from the *same* complete source the `.nl`
exporter already uses. `export/nl.py:_decompose_builder_blocks` reconstructs every
`(A, x, sense, b, name)` builder block into scalar rows; the examiner needs the
identical reconstruction for feasibility (and, ideally, the Jacobian rows for
KKT). The clean fix is a shared `iter_all_rows(model)` (proposed in the export
review as `export/_common.py`) that both the exporter and the examiner consume, so
"what constitutes the model's constraints" has one definition. Minimal interim
fix: in `examine`, append builder-block feasibility rows to the primal-constraint
check even before the KKT machinery understands them — a violated builder row must
never pass. **Regression test:** the §1/C2 repro (infeasible point vs a fast-path
floor) must FAIL examination.

---

## 3. VAL-2 / VAL-3: robustness of the verdict

**VAL-2.** When `lsq_linear` throws (singular active-set system, conditioning),
`_recover_and_check_kkt` returns three `passed=False` checks. Since
`report.passed = all(checks)`, `assert_examined` then raises on a point whose only
sin was that *multiplier recovery* was numerically hard — not that the point is
wrong. A validator should distinguish "this point violates KKT" from "I could not
recover multipliers to check KKT." Fix: on recovery failure, emit an
`inconclusive`/skipped result (or a warning-level check that doesn't flip
`passed`), and surface it in the summary — never fail-closed on a tooling error.
Feasibility and objective checks (which don't need recovery) already stand on
their own.

**VAL-3.** `obj_consistency`'s tolerance `obj_tol + 1e-4·|result.objective|` keys
off the *result* magnitude. For an objective that is ~0 by cancellation of large
terms (e.g. `Σ(aᵢ − bᵢ)` with large aᵢ,bᵢ), re-evaluation float error scales with
the *terms*, not the near-zero result, and can exceed the tolerance → spurious
FAIL. Add a terms-magnitude floor (e.g. scale by `max(|obj|, |re_obj|, unit)` or a
small absolute epsilon tied to the evaluator's condition). Low frequency; worth a
one-line hardening since a false objective-mismatch is exactly the kind of alarm
that erodes trust in the gate.

---

## 4. Test coverage gaps

`test_examiner.py` (16 tests) covers scalar/vector KKT, integrality, objective
consistency, solver-vs-recovered duals, and the GDP path well. The gaps map to the
findings:

1. **No fast-API-constructed model** in any examiner test (`grep` for
   `m.constraint`/`add_linear_constraints`/`over=` returns 0) — VAL-1 is
   completely untested. Add: a fast-path model where the examiner must FAIL an
   infeasible point and PASS the true optimum (both currently mis-behave / are
   untested).
2. No test forcing LSQ recovery failure (VAL-2) — a rank-deficient active set that
   should report *inconclusive*, not *failed*.
3. No near-zero-objective-with-large-terms case (VAL-3).

The existing "catches a wrong point" coverage is good on Python-side constraints;
the harness pattern to add is: solve, then *corrupt* `result.x`/`result.objective`
and assert examination FAILS — which pins the validator's discriminating power
directly (this review used exactly that pattern to confirm VAL-1).

---

## 5. Assessment

The examiner is the strongest-engineered module in this review series after
`nl.py`: mathematically careful (dual sign conventions, GDP reformulation,
solver-vs-recovered cross-check), fail-safe, and — verified here — actually
discriminating between good and bad points on the constraint set it sees. Its one
material weakness is that it doesn't see the *whole* model: fast-API constraints
are invisible, so on the very models that use the fast path (large structured
linear families) the validator silently checks a subset. That it's advisory on the
`solve()` path limits the blast radius, but it is a genuine gate in the benchmark
suite, and a correctness safety net with a hole in it is exactly the thing this
repository's philosophy says to close. Fixing VAL-1 by unifying "the model's
rows" with the exporter's already-correct reconstruction closes the hole and
removes a definition that is currently duplicated (and divergent) across the
codebase.

---

## 6. Implementation plan (for Opus)

### Phase 1 — close the blind spot (PR `fix(validation): VAL-1`)

- Introduce (or reuse, if the export review's `export/_common.py:iter_all_rows`
  lands first) a single "all constraint rows including builder blocks" source, and
  build the examiner's `_row_metadata` + feasibility check from it. Extend the KKT
  Jacobian rows to include builder blocks where available; at minimum, add builder
  rows to the primal-feasibility check so a violated fast-path row can never pass.
- **Acceptance:** infeasible point vs a `m.constraint(...)` floor FAILS
  examination (currently PASSES); the fast-path model's true optimum PASSES; all
  16 existing tests unchanged.

### Phase 2 — verdict robustness (PR `fix(validation): VAL-2..VAL-3`)

- VAL-2: LSQ-recovery failure → `inconclusive` result that does not flip
  `report.passed`; surfaced in `summary`. Test with a rank-deficient active set.
- VAL-3: terms-magnitude floor in the `obj_consistency` tolerance. Test with a
  near-zero objective built from large cancelling terms.

### Cross-reference

VAL-1 is the fourth instance of the builder-resident-rows blind spot found in this
review series (modeling M12, gp GP-1, export EX-2). These should be fixed together
behind one shared `iter_all_rows(model)` / `_has_builder_only_rows(model)`
primitive on `Model`, so classification, export, and validation share one
definition of "the model's constraints" — the divergence is the root cause, and
patching each consumer separately invites the fifth instance.
