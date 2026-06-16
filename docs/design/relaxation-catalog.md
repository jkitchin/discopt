# Relaxation Catalog — Coverage for Global Optimality Certification

**Status:** living document · **Last audited:** 2026-06-16 (post #139/#148/#149 + the
relaxation-coverage work on this branch) · **Scope:** convex/concave relaxations,
interval/FBBT bound propagation, convexity detection, and the dual-bound soundness rules
used to certify global optima in spatial branch-and-bound.

This document catalogs every relaxation, operator, and function for which discopt can
produce a rigorous convex relaxation (and hence a valid bound for global optimality
certification), and records the gaps relative to SOTA global solvers (BARON, Couenne,
SCIP, ANTIGONE).

A function is **fully certifiable** only if all three hold along the global path:
1. the modeling API can build the node (`dm.*` / operator),
2. the node round-trips into the Rust IR (`MathFunc`) for FBBT + B&B, and
3. the JAX relaxation compiler has a rule that returns a valid `(cv, cc)` pair.

As of this audit **every function the modeling API exposes satisfies all three** — the
former (1)/(2) asymmetries (gaps A and C) are closed. What remains are *tightness* gaps
(§5) and the curvature/FBBT coverage notes.

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
| `x / y` | `BinaryOp("/")` | `relax_div` | via `x·(1/y)` | same | requires `0 ∉ [y_lb, y_ub]`; reciprocal convex/concave by sign |
| `1/y` | (internal) | `_relax_reciprocal` | `1/y` or secant | secant or `1/y` | sign-split |
| `x ** n` (int) | `BinaryOp("**")` | `relax_pow` / `relax_power_int` | even: `x^n`; odd: tangent/piecewise | secant / piecewise secants | odd powers: 3-regime split at the inflection |
| `x ** α` (frac) | `BinaryOp("**")` | inline + `relax_signomial` | `0<α<1` concave, `α>1` convex | secant | requires `x_lb > 0`; general `x^y → exp(y·log x)` |
| `abs(x)` | `UnaryOp("abs")` | `relax_abs` | `|x|` | secant if `0 ∈ int`, else exact | convex; exact when interval excludes 0 |

The bilinear and reciprocal envelopes are the exact convex hulls — the same factorable
foundation BARON/Couenne use. Trilinear and higher products are **not** exact hulls (§3, §5.D).

---

## 2. Univariate function relaxations

Source: `mccormick.py` + `envelopes.py`. Dispatched by `relaxation_compiler.py`
(`_univariate_relax` and `_envelope_relax` tables). **Every row below is now covered all
three ways** (builder ✓, Rust IR ✓, relaxation ✓) and is fully certifiable.

| Function | Relaxation | Curvature handling |
|---|---|---|
| `exp` | `relax_exp` | convex: cv exact, cc secant |
| `log`, `log2`, `log10` | `relax_log[2/10]` | concave: cc exact, cv secant |
| `sqrt` | `relax_sqrt` | concave |
| `x²` (`x**2`) | `relax_square` | convex |
| `sin`, `cos` | `relax_sin` / `relax_cos` | regime-based; `[-1,1]` if width ≥ 2π |
| `tan` | `relax_tan` | inflection at k·π, 3-regime |
| `atan` | `relax_atan` | convex/concave by sign |
| `asin`, `acos` | `relax_asin` / `relax_acos` | 3-regime |
| `sinh` | `relax_sinh` | convex/concave by sign |
| `cosh` | `relax_cosh` | convex |
| `tanh` | `relax_tanh` | concave/convex by sign |
| `sigmoid` | `relax_sigmoid` | 3-regime logistic |
| `softplus` | `relax_softplus` | convex |
| `sign` | `relax_sign` | constant bounds per sign regime (discontinuous; loose) |
| `asinh` | `relax_asinh` (envelopes) | sign-split |
| `acosh` | `relax_acosh` (envelopes) | concave |
| `atanh` | `relax_atanh` (envelopes) | sign-split |
| `erf` | `relax_erf` (envelopes) | inflection at 0 |
| `log1p` | `relax_log1p` (envelopes) | concave |

**IR/builder status (now complete).** The Rust `MathFunc` enum
(`crates/discopt-core/src/expr.rs`) carries the full set — including `Asinh, Acosh,
Atanh, Erf, Log1p, Sigmoid, Softplus` (added to close gap A) and `Norm1, Norm2, NormInf`.
The modeling API exposes builders for all of them, including `atan/asin/acos`
(`modeling/core.py`) and `sinh/cosh` (added to close gap C).

### Bivariate / n-ary

| Function | Builder | Relaxation | Notes |
|---|---|---|---|
| `min(x,y)` | `dm.minimum` | `relax_min` | concavity-preserving (issue #27a); not naive pointwise min |
| `max(x,y)` | `dm.maximum` | `relax_max` | convexity-preserving |
| `if_else` | `dm.if_else` | lowers to max/min + aux var | GDP/big-M |
| `prod(x)` | `dm.prod` | recursive McCormick fold (corner-product interval tracking) | valid for any factor sign; not the exact hull |
| `norm(x, ord)` | `dm.norm` | norm-equivalence envelopes `max_i|x_i| ≤ ‖x‖_p ≤ Σ_i|x_i|` | L1/L2/L∞ in the IR (certified); other orders JAX-path only |

---

## 3. Composite / multivariate relaxations

Source: `envelopes.py`, `multivariate_mccormick.py`, `relaxation_compiler.py`,
`factorable_reform.py`, `milp_relaxation.py`.

- **Trilinear `x·y·z`** — `relax_trilinear` (single ordering) and `relax_trilinear_exact`
  (permutation-symmetric nested McCormick). Auto-detected via
  `_try_extract_trilinear_chain`. **Not the true trilinear convex hull** (Rikun 1997 /
  Meyer–Floudas 2004 facet families are not encoded).
- **Signomial / multilinear `∏ xᵢ^{aᵢ}`** — `relax_signomial_multi` via
  `exp(Σ aᵢ·log xᵢ)`; requires all `lb > 0`. Auto-detected via
  `_try_extract_signomial_factors`.
- **Repeated-factor lifting (PR #148)** — `factorable_reform.py` lifts mixed
  repeated-factor monomials (e.g. `x²·y → w·y`, `w == x²`) to bilinear form so the
  existing pipeline relaxes them; `milp_relaxation.py`
  (`_collect_lifted_bilinear_products`, `_collect_lifted_higher_products`,
  `_choose_trilinear_pair`) then collects the lifted bilinear/trilinear products into the
  LP relaxation. This is sound and compositional but still **not** the exact multilinear
  hull.
- **Fractional `x/y`** — `relax_fractional`; sign-definite denominator clearing in
  `factorable_reform.py` (issue #130).
- **Joint `log(x+y)`, `exp`-bilinear** — `relax_log_sum`, `relax_exp_bilinear`.
- **Tsoukalas–Mitsos (2014) multivariate McCormick** — `multivariate_mccormick.py`,
  `_COMPOSITION_RULES` for `exp, log, log2, log10, sqrt, softplus, abs, tanh, atan,
  sigmoid, sinh`. Tighter univariate composition; sigmoidal spanning-zero case sound but
  loose (issue #51).

---

## 4. Relaxation strategies / backends

| Backend | File | What it does | Notes |
|---|---|---|---|
| **McCormick (NLP)** (default) | `mccormick.py`, `envelopes.py`, `relaxation_compiler.py` | compositional cv/cc envelopes (JAX) | midpoint or convex-NLP solve; **valid dual bound only for convex models** (see §6) |
| **McCormick LP** | `mccormick_lp.py`, `milp_relaxation.py` | polyhedral **outer** approximation with **lifted bilinear aux columns**, solved as an LP | rigorous valid dual bound; tighter than the midpoint evaluator; default when the model has relaxable nonlinearity |
| **α-BB (rigorous)** | `alphabb.py` | `f ∓ Σαᵢ(xᵢ−lb)(ub−xᵢ)`, α from a **sound interval Hessian + interval-Gershgorin** bound | now auto-dispatched as a fallback and selectable via `arithmetic="alphabb"`; the rigorous α replaces the older sampling estimator |
| **TM2014 multivariate** | `multivariate_mccormick.py` | tighter univariate composition | 11 ops |
| **Piecewise McCormick** | `piecewise_mccormick.py` | partitioned envelopes (`partitions>0`) | bilinear, exp, log, sqrt, square, sin, cos, tan |
| **Outer approx / Chebyshev / Taylor / ellipsoidal** | `oa_relax.py`, `polyhedral_oa.py`, `chebyshev_model.py`, `taylor_model.py`, `ellipsoidal_arith.py` | affine OA cuts on a reference box (with remainder bounds) | exp, log, log2, log10, sqrt, sin, cos, tan, atan, asin, acos, sinh, cosh, tanh |
| **Learned (ICNN)** | `learned_relaxations.py`, `icnn.py` | input-convex NN relaxations | {bilinear, exp, log, sqrt, sin} with McCormick fallback |

**Bound tightening & cuts** (part of the global story, complementary to relaxations):
`obbt.py` (optimality-based bound tightening via per-variable LPs),
`nonlinear_bound_tightening.py` (pattern rules: `x**p` ranges, reciprocal bounds),
`crates/discopt-core/src/presolve/fbbt.rs` (forward/backward interval FBBT),
`cutting_planes.py` (RLT / OA / lift-and-project cuts), `cover_cuts.py` (knapsack cover
cuts), `monotonicity.py` (per-expression monotonicity for DCP composition).

---

## 5. Soundness of the dual bound (certification rules)

Certifying a global optimum requires the reported lower bound to be a **rigorous valid
dual bound**, not merely a feasible value. Two guards enforce this:

- **McCormick-`nlp` bound is valid only for convex models (issue #120).** The NLP bound
  solver evaluates the compiled relaxation at `x_cv == x_cc`, where every McCormick rule
  is *tight*, so it effectively minimizes the original objective *locally*. For a
  nonconvex model that local optimum can lie **above** the true optimum and be falsely
  certified as a lower bound. `solver.py` (~line 2386) detects `_mc_mode == "nlp" and not
  _model_is_convex` and falls back to the rigorous α-BB underestimator / `"none"`. The
  **LP** path (`mccormick_lp.py`) is a polyhedral *outer* approximation and is therefore
  always a valid bound (and an infeasible LP is a rigorous fathom).
- **`gap_certified` guard (PR #139).** `SolveResult.__post_init__`
  (`modeling/core.py` ~line 1143) downgrades `gap_certified → False` and clears
  `bound`/`gap` whenever a result claims certification but `bound` is `None` or non-finite
  (e.g. a time-limit termination with a `−∞` lower bound). `status="infeasible"` is
  exempt (it certifies infeasibility, not a gap).

**Practical rule:** the rigorous dual-bound paths are **LP** (default for relaxable
nonlinearity), **α-BB**, and the **spatial B&B** that branches on them; the convex-NLP
bound is rigorous only when the model is convex.

---

## 6. Gaps and deficiencies vs SOTA

### A. Native functions unreachable by the B&B core — RESOLVED

`asinh, acosh, atanh, erf, log1p, sigmoid, softplus` are now in the Rust `MathFunc`
enum with stable evaluation, FBBT forward intervals, and PyO3 conversion both ways, so
they round-trip into FBBT + B&B and are fully certifiable. (Also fixed: the `.nl` parser
previously approximated `atanh→atan`, `asinh→asin`, `acosh→acos`.)

### B. Hard error for prod / norm — RESOLVED

`dm.prod` and `dm.norm` are relaxed (prod via recursive McCormick; norm via
norm-equivalence envelopes). The Rust IR carries `Norm1/Norm2/NormInf`, so **L1, L2, L∞
are globally certified**; other integer orders are JAX-path only. A latent bug was fixed:
the Rust evaluator for vector `norm2`/`prod` was a scalar stub returning **NaN** for array
arguments — it now reduces over the components.

### C. API/IR builder asymmetry — RESOLVED

`atan/asin/acos` (PR-side) and `sinh/cosh` (this branch) now have `dm.*` builders, so the
functions that already had relaxations + IR support are writable from native Python.

### D. Relaxation tightness gaps vs BARON — PARTIAL

- **Trilinear / general multilinear:** still nested/recursive McCormick, **not** the exact
  convex hull (no Rikun/Meyer–Floudas facets, no edge-concave underestimators). PR #148
  added repeated-factor *lifting* and the LP path carries lifted bilinear aux variables,
  so the infrastructure for RLT-style tightening now exists, but the literal hull facets
  are not yet generated. See §7.
- **α-BB now wired in (resolved sub-item):** auto-dispatched as a fallback and selectable
  via `arithmetic="alphabb"`, with a rigorous interval-Hessian α.
- **No automatic relaxation tightening loop:** `obbt.py` and nonlinear bound tightening
  exist but there is no range-reduction / probing / duality-based loop coordinated with
  the relaxation the way BARON does. (Components present; orchestration is the gap.)
- **`sign` and discontinuous functions** get only constant per-regime bounds (loose); no
  SOS / indicator reformulation.

### E. Bound propagation (FBBT) coverage — IMPROVED

Source: `crates/discopt-core/src/presolve/fbbt.rs`.

- **Forward** now includes an analytic branch-aware **`Tan`** (finite within a single
  branch, `entire()` only when an asymptote is crossed) in addition to the arithmetic ops
  and Exp/Log/Log2/Log10/Sqrt/Sin/Cos/Atan/Sinh/Cosh/Asin/Acos/Tanh/Abs and the new
  Asinh/Acosh/Atanh/Erf/Log1p/Sigmoid/Softplus. Norms → `[0,∞)` (sound; the array collapses
  to one node interval, so tightening comes from the relaxation, not FBBT). `MatMul → entire()`.
- **Backward** now inverts, besides `Add/Sub/Mul/Div/Pow`, `Neg/Abs`, the monotone set
  `Exp, Log, Log2, Log10, Log1p, Sqrt, Sinh, Asinh, Tanh, Atanh, Atan, Asin, Acos, Acosh,
  Cosh, Sigmoid, Softplus, Tan` (Tan within the forward input's branch). Remaining no-ops:
  periodic `Sin/Cos`, `Erf` (no std `erfinv`), and `Sign/Min/Max/Prod/Norm`.

### F. Convexity detection coverage — MOSTLY CLOSED (PR #149)

`convexity/lattice.py:unary_atom_profile` now assigns curvature to **17 atoms**:
`exp, log, log2, log10, sqrt, abs, cosh, sinh, tanh, asin, atanh, atan, asinh, erf, acos,
acosh, log1p`. Interval evaluation (`convexity/interval_eval.py`) additionally covers
`sin, cos, tan`. **Still `UNKNOWN`:** `sin, cos, tan` (no curvature profile — periodic),
and `sigmoid, softplus, sign, min, max, prod, norm`. The convex fast-path therefore still
under-recognizes models using those atoms, but the previously-large gap (which excluded
all inverse-trig/hyperbolic and `erf/log1p`) is closed.

---

## 7. Remaining work (prioritized)

1. **Exact multilinear hull facets (D).** The lifting infrastructure now exists (#148, LP
   path with lifted bilinear aux columns), so the next step is to *generate the
   Rikun/Meyer–Floudas facets* on those lifted variables in the LP relaxation, rather than
   relying on recursive bilinear envelopes. Note: in the compositional NLP/value-evaluator
   this is not expressible — at the box-midpoint linearization point recursive McCormick is
   order-invariant (verified: `relax_trilinear_exact`'s three orderings coincide at the
   midpoint, so permutation-merging yields no tightening). Tightening must happen on the
   **LP** path with explicit lifted variables.
2. **Curvature profiles for the remaining atoms (F).** `softplus` is globally convex and
   `sigmoid` is sign-split (convex for x<0, concave for x>0, like `tanh`); both are smooth
   and easy to add. Treating `sin/cos/tan` piecewise would further widen the convex
   fast-path.
3. **Range-reduction / OBBT orchestration loop (D).** Coordinate `obbt.py` + nonlinear
   bound tightening with the relaxation in a probing loop.
4. **FBBT backward for `Erf`** (needs an `erfinv`) and a periodic-aware inverse for
   `Sin/Cos`.
5. **General `norm{p}` certification (`p ∉ {1,2,∞}`).** Would need a payload-carrying
   `MathFunc` variant (e.g. `NormP(u32)`); low priority — these orders are rarely modeled.

## 8. Implementation status (this branch)

Implemented and verified here: gaps **A**, **B**, **C** closed; **D** α-BB sub-item closed
(rigorous, auto-dispatched); **E** forward Tan + backward monotone-function set; **F**
context updated (closed by #149). The `prod` path uses the proper recursive-McCormick fold;
L1/L2/L∞ norms are certified and a NaN vector-eval bug in the Rust IR was fixed. Item 1 of
§7 (exact multilinear hull) was investigated and deferred to the LP/lifted path; the rest
of §7 is open.
