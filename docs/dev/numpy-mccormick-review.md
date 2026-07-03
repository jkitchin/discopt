# _numpy McCormick Relaxation Review — Soundness

**Date:** 2026-07-03
**Scope:** `python/discopt/_numpy/mccormick.py` (352), `relaxation_compiler.py`
(476) — numpy McCormick/factorable convex relaxations. The soundness of the
`asin`/`acos` primitive is **shared with the live JAX layer**
(`_jax/mccormick.py`), so a finding here reaches the shipping path.
**Method:** Delegated soundness fuzz (~3.4M primitive samples + composite checks,
harness modeled on `test_gp_hull.py::test_soundness`); the two findings
re-confirmed here, including on the **live JAX primitive**.

A convex relaxation is valid only if, over the whole box, the convex
under-estimator is ≤ the true function and the concave over-estimator ≥ it. An
unsound envelope (one that crosses the function) cuts the true optimum → an invalid
dual bound → risk of certifying a wrong global optimum (CLAUDE.md §1 forbids this).

**Verdict: the primitive math is sound to machine epsilon EXCEPT `asin`/`acos`,
which are provably inverted — and that inversion is present in the live JAX
primitive too. The numpy *compiler* is not a relaxation at all (its leaf drops the
box).** Both are currently *latent* (masked in today's calling paths) but each is a
direct route to an invalid bound the instant the path is exercised.

---

## 1. Findings

| # | Severity | Loc | Finding |
|---|----------|-----|---------|
| NM-1 | **P0 soundness (live)** | `_numpy/mccormick.py:210-241` **and** `_jax/mccormick.py:474-524` | `relax_asin`/`relax_acos` have **inverted curvature regimes**. `arcsin''(x)=x(1−x²)^{−3/2}` so arcsin is **convex on [0,1]**, but the code sets `is_concave = lb≥0` and swaps cv/cc. The convex under-estimator sits **above** the function → cuts feasible points → invalid bound. **Present in the live JAX primitive** [CONFIRMED both: `relax_asin(0.5)` on `[0.1,0.9]` returns cv=0.609968 > true=0.523599, gap 0.086] |
| NM-2 | **P0 soundness (latent)** | `relaxation_compiler.py:164-181` | The numpy compiler's `Variable` leaf returns `(x_cv, x_cc)` and **never reads the box `lb/ub`**. Under the real calling convention (`relax_fn(x, x, lb, ub)` → leaf sees `(x,x)`), the whole tree collapses and returns the **exact nonconvex function** as its own "relaxation" [CONFIRMED: `x*y` on `[1,5]²` at (3,3) → numpy cv=cc=9.0 (the true product); JAX cv=5.0 (real underestimator)] |
| NM-3 | P3 robustness | `_numpy/mccormick.py` `_secant`/primitives | No `lb>ub` guard — an inverted box yields a finite envelope from a negative-width secant with no loud refusal (`lb==ub` degeneracy *is* handled correctly). Callers currently pass valid boxes [SUSPECTED] |

### Reachability (why these are latent today)
- **NM-1**: the soundness suite evaluates on the diagonal `relax_fn(x, x, lb, ub)`,
  where a univariate-of-a-bare-variable relaxation collapses to a degenerate
  interval (`_secant` → `f(x)`), so the buggy branch is never exercised; and
  `asin(x*y)` routes through the sound multivariate bilinear path. But any model
  with `asin`/`acos` of a variable over a non-degenerate box — or fixing NM-2 —
  makes it live immediately. The docstrings ("asin is convex on [-1,0]") are
  themselves mathematically wrong in both files.
- **NM-2**: `solver.py:5031,5259` compile the numpy fns and pass them as
  `obj_relax_fn_numpy=`/`con_relax_fns_numpy=` into `solve_mccormick_batch`, but
  those params are **never consumed** (threaded through and dropped; the POUNCE
  evaluator is built only from the JAX `obj_relax_fn`). So the numpy compiler is
  compiled-but-unused today — the moment it is wired in, NM-2 makes its bounds
  unsound and re-enables NM-1.

## 2. Verified SOUND (fuzz coverage)

~25–30 points on random boxes (negative, zero-crossing, wide, degenerate `lb==ub`),
`under ≤ true ≤ over` at tol 1e-7; max violation among these **9.6e-16** (cos; rest
exactly 0):

- Univariate `exp, log, log2, log10, sqrt, square, abs, sin` (narrow + wide ≥2π),
  `cos, tan, atan, sinh, cosh, tanh, sigmoid, softplus` — clean.
- `relax_pow` n=2..6 (even convex + odd zero-crossing three-case) — clean.
- `relax_bilinear` — all **nine** sign combinations of (x,y); the `both_nonneg`
  tightening clamp holds — clean.
- `relax_div` — y strictly positive and strictly negative — sound (the mislabeled
  reciprocal concavity is harmless because `relax_div` passes the *exact*
  reciprocal value as the bilinear point).
- `lb==ub` degeneracy handled correctly (`_secant` → `f(x)`; 0 failures).

Composite compiler paths return 0 crossings but are **vacuous under NM-2's diagonal
collapse** (they return the exact value), so they do not evidence compiler
soundness — only NM-2's fix + re-fuzz will.

## 3. Plan (for Opus)

**Phase 1 — `fix(correctness): NM-1` (live JAX + numpy).** Swap the regime flags so
`lb≥0` is the **convex** case for `asin` (mirror for `acos`); correct both
`mccormick.py` files and the docstrings. **Acceptance:** off-diagonal soundness fuzz
(`relax_asin(x, lb, ub)` with `lb<ub` across `[-1,1]` sub-boxes) shows
`cv ≤ arcsin ≤ cc` with zero crossings — the repro (cv=0.610 > 0.524) must flip to
sound. Extend the relaxation soundness harness to sample **off-diagonal**
(`x_cv ≠ x_cc`, and univariate over non-degenerate boxes) so this class can never
hide again — the diagonal-only convention is exactly what masked NM-1.

**Phase 2 — `fix(correctness): NM-2`.** The numpy `Variable` leaf must return the
box bounds `(lb[offset], ub[offset])` like the JAX compiler. **Acceptance:** `x*y`
on `[1,5]²` at (3,3) gives numpy cv=5.0 (matching JAX), not 9.0; then re-run the
composite fuzz (now non-vacuous) — every primitive/composite `under ≤ true ≤ over`.
Do **not** wire the numpy backend into the live solve path until both NM-1 and NM-2
are green, since NM-2's fix makes NM-1 live.

**Phase 3 — NM-3.** Loud refusal on `lb>ub` in `_secant`/primitives.

**Priority:** NM-1 is **Tier 1** — it is an unsound envelope in the *live JAX*
relaxation layer; a model using `asin`/`acos` over a real box can get an invalid
dual bound. Verify whether any test/benchmark instance uses `asin`/`acos` (if so
the risk is not merely latent). NM-2 is Tier 1 for *activation-gating* — keep the
numpy backend disabled until fixed.
