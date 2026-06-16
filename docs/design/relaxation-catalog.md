# Relaxation Catalog — Coverage for Global Optimality Certification

**Status:** living document · **Last audited:** 2026-06-16 (post #139/#148/#149 + the
relaxation-coverage work on this branch: full-function-set IR, prod/norm, rigorous α-BB,
FBBT extensions, convexity-lattice closure, and the exact bilinear→multilinear RLT hull) ·
**Scope:** convex/concave relaxations, interval/FBBT bound propagation, convexity detection,
and the dual-bound soundness rules used to certify global optima in spatial branch-and-bound.
**Bottom line:** relaxation *tightness* is now at SOTA parity for the factorable/polynomial
core; the remaining distance to BARON/SCIP is bound-tightening orchestration (§7).

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

Two regimes share this section: the compositional **NLP/value-evaluator** (`envelopes.py`,
`multivariate_mccormick.py`) and the **LP** relaxation (`milp_relaxation.py`). The LP path
carries explicit lifted product variables and now produces the **exact multilinear hull**
(§6.D); the NLP-path envelopes below are the compositional (looser) counterparts.

- **Trilinear / multilinear `∏ xᵢ` (NLP path)** — `relax_trilinear` (single ordering) and
  `relax_trilinear_exact` (permutation-symmetric nested McCormick), auto-detected via
  `_try_extract_trilinear_chain`, and the recursive `prod` fold. These are sound but **not**
  the convex hull (at the box-midpoint linearization point recursive McCormick is
  order-invariant). The exact hull lives on the **LP path** — see §6.D.
- **Signomial / multilinear `∏ xᵢ^{aᵢ}`** — `relax_signomial_multi` via
  `exp(Σ aᵢ·log xᵢ)`; requires all `lb > 0`. Auto-detected via
  `_try_extract_signomial_factors`.
- **Repeated-factor lifting (PR #148)** — `factorable_reform.py` lifts mixed
  repeated-factor monomials (e.g. `x²·y → w·y`, `w == x²`) to bilinear form so the pipeline
  relaxes them; `milp_relaxation.py` (`_collect_lifted_bilinear_products`,
  `_collect_lifted_higher_products`, `_choose_trilinear_pair`) collects the lifted products
  into the LP relaxation, where the RLT cuts (§6.D) then make the **distinct-factor** product
  exact. (Repeated factors x², x³, … are themselves relaxed by the monomial-secant path.)
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

### D. Relaxation tightness gaps vs BARON — multilinear hull now exact (n ≤ cap)

- **Multilinear `∏ xᵢ` (n distinct factors): exact convex/concave hull implemented (this
  branch).** The LP path (`milp_relaxation.py`) materializes a lifted column `w_T` for every
  sub-product `T` and emits the RLT bound-factor product cuts: for every factor-subset `S`
  with `|S| ≥ 2`, the `2^{|S|}` degree-`|S|` products `∏_{i∈S}(sᵢxᵢ+cᵢ) ≥ 0`
  (`(x−xL)→(+1,−xL)`, `(xU−x)→(−1,xU)`). The `|S|=2` cuts are the McCormick envelopes; the
  `|S|≥3` cuts tie the higher products together. The full family is the exact hull (Rikun
  1997 / Meyer & Floudas 2004) — verified to match the box-vertex convex envelope to ~1e-14
  for n=3 and n=4. Each cut is a product of nonnegative bound factors, so it is individually
  valid (tightens, never invalidates). Capped at `DISCOPT_MULTILINEAR_RLT_MAX` factors
  (default 4) to bound the `2^n` growth; larger products keep the loose recursive chain.
  Gated by `DISCOPT_TRILINEAR_RLT` (default on).
- **Remaining multilinear gaps:** products with more than `DISCOPT_MULTILINEAR_RLT_MAX`
  factors fall back to recursive McCormick; `edge-concave` underestimators (Tardella; an
  alternative tighter family for some structures) are not generated; and the cuts are added
  densely rather than separated on demand (a cut-on-violation loop would scale to higher n).
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

### F. Convexity detection coverage — RESOLVED (for smooth unary atoms)

`convexity/lattice.py:unary_atom_profile` now assigns curvature to **19 atoms**:
`exp, log, log2, log10, sqrt, abs, cosh, sinh, tanh, asin, atanh, atan, asinh, erf, acos,
acosh, log1p` (PR #149) plus `softplus` (globally convex) and `sigmoid` (sign-split, joins
the `atan/asinh/erf` group) added on this branch, with matching sound interval enclosures
in `convexity/interval.py` / `interval_eval.py`. **Every smooth unary atom that has a
definable global or sign-conditioned curvature is now covered.** The remaining
`unary_atom_profile` returns `None` only for atoms where that is *correct*: `sin, cos, tan`
(periodic — no sign-based curvature; the interval-Hessian certificate handles them
region-wise) and `sign` (discontinuous). `min, max, prod, norm` are n-ary, not unary atoms;
`norm` is convex and could be added to the DCP *composition* rules (`rules.py`) as a future
enhancement.

---

## 7. Distance to SOTA — status

The **relaxation layer** is now broadly SOTA-competitive: exact bilinear/trilinear/
multilinear hulls (RLT), factorable McCormick over the full function set, a rigorous
interval-Hessian α-BB fallback, the OA/Chebyshev/Taylor/ellipsoidal cut family, two
convexity-detection tracks (x-space + GP, §9), and FBBT in both directions.

**Done (this branch):**

1. **OBBT / range-reduction loop — DONE.** `obbt_tighten_root` runs a looped OBBT over the
   McCormick *relaxation* (not just linear rows) with rounds, an incumbent cutoff, and
   soundness guarantees; the periodic Phase-C tightening now routes through it.
2. **Duality-based bound tightening (DBBT) — DONE.** `dbbt_on_relaxation` reads the relaxation
   LP's reduced costs to bound how far each variable moves from its pressed bound — one LP
   tightens all variables, interleaved into the OBBT loop. Needs an exact dual-exposing oracle
   (HiGHS); no-ops soundly otherwise.
6. **Convexity-detection completeness — DONE.** `norm(affine)` is now a CONVEX DCP atom
   (`rules.py`); `sin/cos/tan` are region-aware in the interval-Hessian certificate
   (`interval_ad.py`) — convex/concave on constant-curvature boxes, sound abstention when
   indefinite.
7. **FBBT backward for `Erf` + periodic `Sin/Cos` — DONE.** `erfinv` (Winitzki + Newton) and
   single-monotone-piece inversion for sin/cos, validated sound on a 1196-case sweep.
8. **General `norm{p}` (`p ∉ {1,2,∞}`) — DONE.** `MathFunc::NormP(u32)` closes the IR
   round-trip; arbitrary integer orders certify end to end.

**Remaining (substantial / lower value):**

3. **Uncapped multilinear via on-demand separation.** The exact multilinear hull is dense
   (`2^n` columns/cuts) and capped at `DISCOPT_MULTILINEAR_RLT_MAX` (default 4). A
   cut-on-violation separator would scale it past the cap. *Sound but needs a separation-loop
   refactor of the single-shot LP solve flow* — the cuts are always valid (RLT bound-factor
   products), so the work is integration, not correctness.
4. **Edge-concave / vertex-polyhedral underestimators** (Tardella; Hasan 2018). *Research-grade
   new relaxation family.* Note: multilinear monomials are already edge-concave and their
   vertex envelope is exactly what RLT (§6.D) produces — the open value is *other* edge-concave
   structures, needing a general edge-concavity detector + vertex-envelope construction.
5. **Discrete / `sign` / indicator handling.** *Finding: sign's continuous convex/concave hull
   over a zero-spanning box is already exact* (constant `±1`, since sign equals `−1` across all
   of `[lb, 0)`), so `relax_sign` is not loose in the continuous sense. Tightening genuinely
   requires a *discrete* binary-indicator (SOS1/big-M) reformulation — niche (sign is rarely
   modeled) and ill-posed at `x=0` (`sign(0)=0` vs a `±1` indicator). Deferred as low value.

## 8. Implementation status (this branch)

Implemented and verified here: gaps **A**, **B**, **C** closed; **D** — α-BB wired in
(rigorous, auto-dispatched) **and the exact convex/concave hull for bilinear → multilinear
(n ≤ cap) products** via RLT bound-factor cuts (§6.D), validated to match the box-vertex
envelope to ~1e-14; **E** forward Tan + backward monotone-function set; **F** closed for smooth
unary atoms — `softplus`/`sigmoid` curvature profiles + sound interval enclosures on top of
#149 (verified sound: a zero-spanning `sigmoid` and a concave `-softplus` are *not*
misclassified convex). The `prod` path uses the proper recursive-McCormick fold; L1/L2/L∞ norms
are certified and a NaN vector-eval bug in the Rust IR was fixed.

Also delivered (the §7 SOTA items): **1–2** OBBT-on-relaxation loop + duality-based bound
tightening; **6** norm-DCP atom + region-aware trig certificate; **7** FBBT backward for
erf/sin/cos; **8** general `norm{p}` via `NormP(u32)`. What remains is **3** (uncapped
multilinear via on-demand separation — needs a separation-loop refactor), **4** (edge-concave
underestimators — research-grade), and **5** (discrete `sign`/indicator — low value; sign's
continuous hull is already exact). The relaxation tightness itself is at parity for the
polynomial/factorable core; the residual distance to SOTA is bound-tightening *orchestration*
breadth and those three specialized items.

---

## 9. Complementary track: log-space / geometric programming (not cataloged above)

This catalog covers **x-space** relaxations and convexity detection. discopt has a second,
deliberately separate **log-space (`y = log x`) / geometric-programming** track that a reader
should know exists; it is not part of the mechanisms above:

- **`discopt.gp`** — `is_log_convex(model)` / `classify_gp` (shipped in #113, follow-up to the
  closed #111): a whole-model GP recognizer that auto-routes a posynomial/monomial program
  through the exact log-space convex NLP and maps the optimum/bound/gap back to `x`.
- **Open GP follow-ups:** #114 (mixed-sign **signomial** global solver — the log-space
  counterpart to the x-space `relax_signomial_multi` in §3), #115 (**per-expression
  log-curvature lattice** — the log-space analogue of the x-space curvature lattice in §6.F,
  required by the issue to stay *strictly separate* from it), and #116 (**y-space
  branching/bound-tightening** for GP-structured MINLPs — a GP-specific bounding backend
  alongside §4).

These give discopt two convexity-detection lattices (x-space and log-space) and two relaxation
regimes; only the x-space side is cataloged here.
