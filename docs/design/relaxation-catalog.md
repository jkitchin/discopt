# Relaxation Catalog — Coverage for Global Optimality Certification

**Status:** living document · **Last audited:** 2026-06-16 · **Scope:** convex/concave
relaxations, interval/FBBT bound propagation, and convexity detection used to compute
valid lower bounds in spatial branch-and-bound.

This document catalogs every relaxation, operator, and function for which discopt can
currently produce a rigorous convex relaxation (and hence a valid bound for global
optimality certification), and records the gaps relative to state-of-the-art global
solvers (BARON, Couenne, SCIP, ANTIGONE).

A relaxation is "covered" only if **all** of the following hold along the global path:
1. the modeling API can build the node (`dm.*` / operator),
2. the node round-trips into the Rust IR (`MathFunc`) for FBBT + B&B, and
3. the JAX relaxation compiler has a rule that returns a valid `(cv, cc)` pair.

A break in any link means the function is only usable in a *local* NLP path, not in a
certified global one. Several functions below are covered in (1) and (3) but **not** (2)
— see [Gaps](#5-gaps-and-deficiencies-vs-sota).

---

## 1. Operator relaxations (arithmetic core)

Source: `python/discopt/_jax/mccormick.py`. Convention: each `relax_*` returns
`(cv, cc)` with `cv ≤ f ≤ cc`; convex parts are exact, concave parts use secant lines
(`_secant`, mccormick.py:23).

| Operator | Node | Function | Underestimator | Overestimator | Notes |
|---|---|---|---|---|---|
| `x + y` | `BinaryOp("+")` | `relax_add` | exact | exact | additive, tight |
| `x - y` | `BinaryOp("-")` | `relax_sub` | exact | exact | tight |
| `-x` | `UnaryOp("neg")` | `relax_neg` | exact | exact | tight |
| `x * y` | `BinaryOp("*")` | `relax_bilinear` | max of 2 affine | min of 2 affine | **McCormick envelope = convex hull** of bilinear term |
| `x / y` | `BinaryOp("/")` | `relax_div` | via `x·(1/y)` | same | requires `0 ∉ [y_lb, y_ub]`; reciprocal is convex/concave by sign |
| `1/y` | (internal) | `_relax_reciprocal` | `1/y` or secant | secant or `1/y` | sign-split |
| `x ** n` (int) | `BinaryOp("**")` | `relax_pow` / `relax_power_int` | even: `x^n`; odd: tangent/piecewise | secant / piecewise secants | odd powers handle the inflection at 0 with a 3-regime split |
| `x ** α` (frac) | `BinaryOp("**")` | inline + `relax_signomial` | `0<α<1` concave, `α>1` convex | secant | requires `x_lb > 0`; general `x^y → exp(y·log x)` |
| `abs(x)` | `UnaryOp("abs")` | `relax_abs` | `|x|` | secant if `0 ∈ int`, else exact | convex; exact when interval excludes 0 |

The bilinear and reciprocal envelopes are the exact convex hulls — this is the same
foundation BARON/Couenne use for factorable programming. Trilinear and higher products
are **not** exact hulls (see §3).

---

## 2. Univariate function relaxations

Source: `mccormick.py` + `envelopes.py`. Dispatched by the relaxation compiler
(`relaxation_compiler.py`, tables `_univariate_relax` at line 698 and `_envelope_relax`
at line 783).

### Covered with dedicated envelopes

| Function | `dm.*` builder? | Rust `MathFunc`? | Relaxation | Curvature handling |
|---|:--:|:--:|---|---|
| `exp` | ✅ | ✅ | `relax_exp` | convex: cv exact, cc secant |
| `log`, `log2`, `log10` | ✅ | ✅ (log2/log10 too) | `relax_log[2/10]` | concave: cc exact, cv secant |
| `sqrt` | ✅ | ✅ | `relax_sqrt` | concave |
| `x²` (`x**2`) | ✅ | ✅ | `relax_square` | convex |
| `sin`, `cos` | ✅ | ✅ | `relax_sin` / `relax_cos` | regime-based (concave/convex/mixed); `[-1,1]` if width ≥ 2π |
| `tan` | ✅ | ✅ | `relax_tan` | inflection at k·π, 3-regime |
| `atan` | ❌ (import only) | ✅ | `relax_atan` | convex/concave by sign |
| `asin`, `acos` | ❌ (import only) | ✅ | `relax_asin` / `relax_acos` | 3-regime |
| `sinh` | ❌ (import only) | ✅ | `relax_sinh` | convex/concave by sign |
| `cosh` | ❌ (import only) | ✅ | `relax_cosh` | convex |
| `tanh` | ✅ | ✅ | `relax_tanh` | concave/convex by sign |
| `sigmoid` | ✅ | ❌ | `relax_sigmoid` | 3-regime logistic |
| `softplus` | ✅ | ❌ | `relax_softplus` | convex |
| `sign` | ✅ | ✅ | `relax_sign` | constant bounds per sign regime (discontinuous; loose) |
| `asinh` | ✅ | ❌ | `relax_asinh` (envelopes) | sign-split |
| `acosh` | ✅ | ❌ | `relax_acosh` (envelopes) | concave |
| `atanh` | ✅ | ❌ | `relax_atanh` (envelopes) | sign-split |
| `erf` | ✅ | ❌ | `relax_erf` (envelopes) | inflection at 0 |
| `log1p` | ✅ | ❌ | `relax_log1p` (envelopes) | concave |

**Read the last two columns together.** Twelve functions are covered three ways
(builder + IR + relaxation) and are fully certifiable. The `atan/asin/acos/sinh/cosh`
rows are in the Rust IR (so importable from `.nl`/Pyomo and certifiable) but have **no
`dm.*` builder** — you cannot write them in the native Python API. The
`sigmoid/softplus/asinh/acosh/atanh/erf/log1p` rows are the reverse: native builder +
JAX relaxation exist, but they are **not in the Rust `MathFunc` enum**, so they raise
`Unknown MathFunc` at `crates/discopt-python/src/expr_bindings.rs:1134` and cannot enter
the FBBT/B&B core. See §5.

### Bivariate / n-ary

| Function | Builder | Relaxation | Notes |
|---|---|---|---|
| `min(x,y)` | `dm.minimum` | `relax_min` | concavity-preserving form (issue #27a); not naive pointwise min |
| `max(x,y)` | `dm.maximum` | `relax_max` | convexity-preserving form |
| `if_else` | `dm.if_else` | lowers to max/min + aux var | GDP/big-M |

---

## 3. Composite / multivariate relaxations

Source: `envelopes.py`, `multivariate_mccormick.py`, `relaxation_compiler.py` dispatch.

- **Trilinear `x·y·z`** — `relax_trilinear` (loose single ordering) and
  `relax_trilinear_exact` (permutation-symmetric nested McCormick over all 3 orderings,
  `cv=max`, `cc=min`). Auto-detected via `_try_extract_trilinear_chain`. **Explicitly not
  the true trilinear convex hull** (envelopes.py docstring cites Rikun 1997 /
  Meyer–Floudas 2004 as missing facet families).
- **Signomial / multilinear `∏ xᵢ^{aᵢ}`** — `relax_signomial_multi` via
  `exp(Σ aᵢ·log xᵢ)` log-space composition; requires all `lb > 0`. Auto-detected via
  `_try_extract_signomial_factors`.
- **Fractional `x/y`** — `relax_fractional`.
- **Joint `log(x+y)`, `exp`-bilinear** — `relax_log_sum`, `relax_exp_bilinear`.
- **Tsoukalas–Mitsos (2014) multivariate McCormick** — `multivariate_mccormick.py`,
  `_COMPOSITION_RULES` for `exp, log, log2, log10, sqrt, softplus, abs, tanh, atan,
  sigmoid, sinh`. This is the tighter univariate-composition rule, preferred over the
  legacy midpoint composition. The sigmoidal spanning-zero case is sound but loose
  (M1 follow-up, issue #51).

---

## 4. Relaxation strategies / backends

Selected via the `arithmetic` / `mode` parameters of the relaxation compiler:

| Backend | File | What it does | Op coverage |
|---|---|---|---|
| **McCormick** (default) | `mccormick.py`, `envelopes.py` | factorable convex/concave envelopes | full set above |
| **TM2014 multivariate** | `multivariate_mccormick.py` | tighter univariate composition | 11 ops |
| **Piecewise McCormick** | `piecewise_mccormick.py` | partitioned envelopes (`partitions>0`) | bilinear, exp, log, sqrt, square, sin, cos, tan |
| **α-BB** | `alphabb.py` | `f − Σαᵢ(xᵢ−lb)(ub−xᵢ)`, α from min Hessian eigenvalue (eigenvalue or Gershgorin) | general C² fallback — **not auto-dispatched** by the compiler (standalone API) |
| **Outer approximation / Chebyshev / Taylor / ellipsoidal** | `oa_relax.py`, `polyhedral_oa.py`, `chebyshev_model.py`, `taylor_model.py` | affine cuts on a static reference box | exp, log, log2, log10, sqrt, sin, cos, tan, atan, asin, acos, sinh, cosh, tanh |
| **Learned (ICNN)** | `learned_relaxations.py`, `icnn.py` | input-convex NN relaxations | {bilinear, exp, log, sqrt, sin} with McCormick fallback |

---

## 5. Gaps and deficiencies vs SOTA

### A. Broken global path: native functions that can't reach the B&B core

Seven functions have a Python builder **and** a JAX relaxation but are **absent from the
Rust `MathFunc` enum** (`crates/discopt-core/src/expr.rs`), so `convert_expr` rejects
them at `expr_bindings.rs:1134`:

> `asinh`, `acosh`, `atanh`, `erf`, `log1p`, `sigmoid`, `softplus`

Consequence: a model using these builds and can be relaxed in pure-JAX contexts, but
cannot enter FBBT or the B&B tree — **no global certificate**. This is the single most
impactful gap because it is silent (the relaxation exists, so it looks supported) and
because `sigmoid`/`softplus`/`erf` are common in ML-embedded MINLPs, a stated use case
(`discopt.nn`). **Fix:** add the seven variants to `MathFunc`, the PyO3 match, and the
FBBT forward/backward interval rules.

### B. No relaxation rule (hard error) for prod / norm

- `dm.prod` → `FunctionCall("prod")` and `dm.norm` → `FunctionCall("norm2"|"normN")`
  have **no entry** in the relaxation compiler and raise
  `ValueError("Unknown function")` (`relaxation_compiler.py:892`). `prod` is only
  relaxed when it happens to decompose into a `*`-tree. `norm` has no relaxation at all
  (FBBT gives only the loose `[0,∞)`), and `normN` for `N≠2` isn't even in the Rust IR.
  BARON/Couenne handle norms and general multilinear products natively.

### C. Symmetric API/IR asymmetry

- `atan`, `asin`, `acos`, `sinh`, `cosh` have relaxations and Rust IR support but **no
  `dm.*` builder** — unreachable from native Python (only via `.nl`/Pyomo import). Pure
  ergonomics gap; add the five free functions.

### D. Relaxation tightness gaps vs BARON

- **Trilinear / general multilinear:** nested McCormick, not the exact convex hull (no
  Rikun/Meyer–Floudas facets, no edge-concave underestimators). BARON ships the true
  multilinear hull.
- **α-BB not wired in:** the general C² fallback exists but the compiler never dispatches
  to it, so a nonconvex C² term with no McCormick rule has no automatic relaxation. SOTA
  solvers always have a general fallback.
- **No automatic relaxation tightening loop:** OBBT (`obbt.py`) and nonlinear bound
  tightening exist, but there is no range-reduction / probing / duality-based bound
  tightening loop coordinated with the relaxation the way BARON does. (Components are
  present; orchestration is the gap.)
- **`sign` and discontinuous functions** get only constant per-regime bounds (very
  loose). No special-ordered-set or indicator reformulation.

### E. Bound propagation (FBBT) coverage gaps

Source: `crates/discopt-core/src/presolve/fbbt.rs`.

- **Forward** intervals cover the arithmetic ops + Exp/Log/Log2/Log10/Sqrt/Sin/Cos/
  Atan/Sinh/Cosh/Asin/Acos/Tanh/Abs. **`Tan` → `entire()`** (no analytic bound),
  **`Norm2` multi-arg → `[0,∞)`**, **`MatMul` → `entire()`**.
- **Backward** propagation inverts only `Add/Sub/Mul/Div/Pow`, `Neg/Abs`, and among
  functions **only `Exp/Log/Sqrt`** — so constraint-driven domain reduction through
  `sin/cos/tan/tanh/...` is a no-op. BARON propagates through its full function set.

### F. Convexity detection coverage

Source: `python/discopt/_jax/convexity/`. The DCP curvature lattice
(`lattice.py:unary_atom_profile`) assigns curvature only to `exp, log/log2/log10, sqrt,
abs, cosh, sinh, tanh`. Everything else (`tan, atan, asin, acos, asinh, acosh, atanh,
erf, log1p, sigmoid, softplus, sign, min, max, prod, norm`) returns `UNKNOWN` — so the
convex fast-path under-recognizes convex models that use those atoms and falls back to
spatial B&B unnecessarily.

---

## 6. Prioritized recommendations

1. **Close the Rust IR gap (A).** Add `Asinh, Acosh, Atanh, Erf, Log1p, Sigmoid,
   Softplus` to `MathFunc`, the PyO3 string match, FBBT forward intervals, and (at least
   forward) the convexity profiles. Highest impact, removes a silent correctness-of-scope
   hole for ML-embedded models.
2. **Add prod/norm relaxations (B).** A relaxation rule for `norm2` (convex — it's just a
   convex atom) and general `prod` would close two hard-error paths.
3. **Auto-dispatch α-BB (D).** Wire `alphabb.py` in as the default fallback for C² terms
   with no McCormick rule, so the solver always has a valid relaxation.
4. **Extend FBBT backward propagation (E)** to the full function set, and give `Tan` an
   analytic forward interval.
5. **Expose `atan/asin/acos/sinh/cosh` as `dm.*` builders (C).** Cheap ergonomics win.
6. **Tighter multilinear hulls (D)** — true trilinear/edge-concave envelopes — for
   parity with BARON on signomial/multilinear problems.
