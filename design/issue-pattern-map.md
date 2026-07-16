# Issue → relaxation-pattern map

A triage of open issues against the relaxation/cut **pattern catalog**
(`design/relaxation-patterns.md`). For each pattern-addressable issue: the
nonconvex structure, the pattern(s) that apply, current coverage, the residual
gap, and a comment-ready recommendation. Issues that are pure search /
LP-throughput / infrastructure are listed at the end as *not* pattern-addressable.

Status key: **covered** (a done pattern applies), **partial** (atom covered, gap
elsewhere), **new-pattern** (needs a pattern not yet implemented), **search/infra**
(not a relaxation-pattern problem).

---

## Pattern-addressable issues

### #15 — Gas-network Weymouth chain — **covered**
- Structure: `f^2 = C(p_in^2 − p_out^2)` chain + compressor ratio + power obj.
- Pattern: **P5 square-difference network** (+ the auto recognizer).
- Status: done — `recognize_and_inject` closes 66.7%→0% automatically.

### #189 — fractional-power / general nonconvex (cvxnonsep_nsig30, clay0303hfsg, chakra, demo7) — **partial**
- Structure: signomial / fractional-power terms `x^p` and products `∏ x_j^{a_j}`.
- Findings (measured): the **univariate `x^p` envelopes are already tight** in the
  engine across regimes — `x^0.5, x^0.7` (concave), `x^1.5, x^2.5, x^-1, x^-0.5`
  (convex), all sound, tight convex hull. So the issue's "tighter `x^p` envelope"
  lever is **largely already in place** for univariate powers.
- Residual gap: (a) **multivariate signomial** products `∏ x_j^{a_j}` relaxed
  compositionally are loose → need the **G-convex / log-domain joint hull**
  (P11/P14, new); (b) **OBBT** on the variables feeding power terms (→ #208) to
  shrink the boxes the tight envelopes act on.
- Recommendation: the univariate part is closed; route remaining looseness to the
  signomial-hull pattern (P11) and OBBT (#208). Add `cvxnonsep_nsig30` /
  `clay0303hfsg` as `gap_certified` regression tests (both vendored).

### #188 — kall_congruentcircles_c51: Euclidean separation — **partial / search**
- Structure: bilinear objective `x_i·x_j` (**P3**) + separation constraints
  `(x_i−x_j)^2 + (y_i−y_j)^2 >= 1` (convex-LHS `>=`, nonconvex feasible region).
- Pattern lever: expand `(x_i−x_j)^2 = x_i^2 − 2x_ix_j + x_j^2` and apply the
  **P3 bilinear hull** + RLT on the squared differences (a *squared-difference /
  Euclidean-separation* pattern, new). Tighter bilinear relaxation is one of the
  owner's named levers.
- Honest caveat: the dominant gap is **combinatorial multimodality** (global
  search / feasibility pump), not a single missing envelope. Pattern work helps
  the bound but won't alone reach the global optimum.

### #114 — signomial (mixed-sign) global solver (SGO) — **new-pattern**
- Structure: signomial terms `±c·∏ x_j^{a_j}` with **mixed signs**.
- Pattern: **P7 posynomial** covers only the positive (posynomial) terms. Signed
  signomials are difference-of-convex in the log domain (posynomial − posynomial);
  the negative terms need a concave-overestimator treatment. → new **signed
  signomial** pattern (P11).
- Recommendation: implement signed-signomial decomposition on top of P7's
  log-convex monomial value; cite Lundell & Westerlund (SGO).

### #181 — G-convexity / convex-transformable relaxations — **new-pattern**
- Structure: terms convex after a variable transform (log, power, exp).
- Pattern: **P7** is the canonical special case (log transform of monomials);
  the issue asks for the *general* framework (Khajavirad-Michalek-Sahinidis 2012).
  → general **G-convexity detection + transform** pattern (P11/P12).
- Recommendation: generalize the log-domain pattern to a transform registry; the
  log-curvature lattice (#115) is the detection layer for it.

### #115 — per-expression log-curvature lattice — **new-pattern (detection)**
- Structure: detect log-convex / log-concave subexpressions.
- Pattern: the **detection layer** that powers P7/P11 (which terms are GP-convex).
  Mirrors the existing convexity lattice but in log-space.

### #116 — y-space (log) branching/bound-tightening for GP — **search/infra (GP)** — **constraint lift done (P16)**
- The branching/OBBT counterpart of the GP patterns; pairs with P7/P11 but is a
  presolve/branching feature, not an envelope.
- **P16 (`inject_gp_constraint_cuts`)** implements the *constraint-level* y-space
  substitution: a posynomial `<=` constraint `P(x) <= b` (all `c_k>0`, exponents
  `a_kj>=0`) is convex after `u_j=log x_j` → `Σ_k exp(s_k) <= b`. The pass adds the
  convex link `u_j <= log x_j`, log-domain bounds, and OA tangent cuts
  `Σ_k exp(s_k0)(1+s_k−s_k0) <= b`. Sound (no feasible `x` removed: at
  `u_j=log x_j` the cut underestimates `P(x)`), value-preserving, and a no-op on
  `>=`/affine/negative-exponent/signomial rows. **Honest:** does **not** fire on
  `cvxnonsep_nsig30` (its single constraint is a *signomial* — a negative term —
  so it needs the signed-signomial DC lift, P11, not the posynomial lift); and on
  synthetic posynomial-constraint GPs discopt's native McCormick relaxation is
  already root-tight (≈3 nodes), so the lift is correct but exposes no measurable
  root gap there. Its value is on instances where the *separable* box relaxation of
  a coupled posynomial constraint is genuinely loose.

### #231 — complementarity `x·y = 0`, `x,y >= 0` — **new-pattern**
- Structure: complementarity / disjunctive `x·y = 0`.
- Pattern: new **complementarity** pattern — `xy = 0` with `x,y>=0` is the
  disjunction `x=0 ∨ y=0`; the bilinear `w=xy` McCormick hull plus the equality
  `w = 0` gives a valid relaxation, and the disjunction drives branching.
- Recommendation: add an IR node (per the issue) + the `w=xy, w=0` cut; provable.

### #187 — autocorr_bern: products of binaries — **new-pattern (exact)** — **done**
- Structure: products of **binary** variables `∏ b_i` (Fortet/Glover).
- Pattern: new **Fortet/Glover linearization** — *exact* (not a relaxation):
  `z = ∏_i b_i ⟺ z <= b_i ∀i, z >= Σ b_i − (n−1), z >= 0`.
- Recommendation: highest-value clean win — exact, provable, and avoids the
  full-DAG Jacobian XLA blowup the issue flags. Implement first.
- **Shipped** as the auto-firing presolve pass
  `discopt._jax.binary_multilinear_reform` (`solve_model` adopts it under the
  pure-MILP guard and routes to the MILP engine): per-monomial Fortet rows for
  the general binary-multilinear case, plus an exact **integer-point secant
  envelope** for objective-pressure squares of integer-valued forms
  (`y == E`, `t >= (u+v)·y − u·v` over the attainable value grid — the
  autocorr `Σ_k C_k²` structure), which keeps the MILP ~10x smaller than flat
  expansion. `{0,1}`-bounded INTEGER columns count as binary (from_nl typing).
  Certified in `python/tests/test_binary_multilinear_reform.py`.

### #219 — transcendental over unbounded/implied-bound domains — **search/infra (bounds)**
- Structure: `exp/log/...` whose argument has no finite bound → dropped from LP.
- Pattern lever: **FBBT / implied-bound propagation** to supply a finite argument
  box so the existing envelopes (P1/P2) can fire. A bound-propagation pass, not a
  new envelope.

### #208 — OBBT-on-auxiliaries — **search/infra (presolve)** — **decision-gate measured + reverse-FBBT built (opt-in)**
- Rebuild McCormick from OBBT-tightened aux bounds. Not a pattern itself, but it
  *tightens every envelope pattern* (P3/P4/P5/P7…) by shrinking boxes. Directly
  amplifies #189, #188.
- **Decision gate (done).** `obbt.measure_discarded_aux_tightening` +
  `design/measure_aux_obbt.py` measure the aux tightening currently discarded.
  On the vendored corpus (structural-only): of 32 instances with aux columns, 25
  tighten, 18 have mean aux reduction > 10% (corpus mean of per-instance means
  34.6%; several ~99% — nvs03/10/11/12/13/15, ex1221). This **flips** the issue's
  gate — the earlier gear4-only "no-op" was an integrality-needle outlier.
- **Part 2 built (opt-in, `obbt_tighten_root(cascade_aux=True)`).** The generic
  aux-column *intersection* alone is a **no-op** (reproduces the issue's Part-1
  finding — a captured aux bound is self-implied within the same LP). The real
  lever is **reverse FBBT** (`obbt.reverse_fbbt_from_aux`): propagate the
  OBBT-tightened aux bound back through the term definition
  (`w=a·b ⟹ a∈[w]/[b]`; `w=a^p ⟹ p`-th root), recovering the hyperbolic/root
  bounds the linear rows can't express. **Honest status:** sound (every corpus
  optimum preserved exactly) and it measurably shrinks the root box (nvs11/12/13
  box log-volume −4 nats), but across the tested corpus it gave **no net
  node-count / wall reduction** and slightly regressed nvs13 (39→53 nodes). It
  therefore does **not** meet #208's "the extra OBBT cost must pay for itself"
  acceptance bar, so it ships **default-off** behind `cascade_aux`, pending a
  targeted-budget A/B. The root-box reduction lands on integer-heavy instances
  whose bottleneck is the integer search, not the continuous relaxation box.

---

## Not pattern-addressable (search / LP throughput / infrastructure)

- **#236** GNN branching (ML training/weights).
- **#229** spatial-B&B node LP ~750× slower than HiGHS (FT-update / factorization).
- **#196** ex1252 gap closure (node throughput, SOS1 selectors).
- **#194** spatial branching on nonlinear-term integers; unbounded-var policy.
- **#186** ex1263a/trim-loss: per-node McCormick MILP solve speed.
- **#163** benchmark harness setup.
- **#87** restore test coverage to 85%.
- **#27** user-defined functions (JuMP.register equivalent).

---

## New patterns this review surfaces (added to the catalog)

| Pattern | Issues | Kind | Status |
|---|---|---|---|
| P10 Fortet/Glover binary-product linearization | #187 | exact | **done** (`patterns.binary_product_linearization`) |
| P11 Signed signomial (log-domain DC) | #114, #181, #189 | relaxation | **done** (`signed_signomial.signed_signomial_dc_envelope`) |
| P12 Complementarity `x·y=0` | #231 | relaxation + disjunction | **done** (`patterns.complementarity_cut`) |
| P13 Squared-difference / Euclidean separation | #188 | relaxation (RLT) | roadmap |
| P14 Multivariate signomial joint hull (GP log-lift) | #189, #181 | relaxation | **done** (`gp_hull.monomial_log_envelope`) |
| log-curvature classifier (detection layer) | #115, #181 | analysis | **done** (`log_curvature.log_curvature`) |
| (presolve) OBBT-on-aux, transcendental FBBT | #208, #219 | presolve | amplifier |

## Auto-firing detectors wired into `cut_recognizer`

`inject_all_patterns(model)` runs every detector in turn and returns
`{pattern: count}`; each is sound and a no-op on non-matching models:

| Detector | Issue | What it does |
|---|---|---|
| `recognize_and_inject` (square-diff network) | #15 | gas/water Weymouth chain → objective aux + `u >= K·h(w)` |
| `inject_binary_products` (Fortet/Glover) | #187 | objective `coef·∏ b_i` (n>=3) → aux `z` + Fortet linearization (tighter than nested McCormick); value-preserving |
| `inject_complementarity` | #231 | `x·y = 0` (or `<= 0`), `x,y>=0` → cut `x/x_ub + y/y_ub <= 1` |
| `inject_gp_cuts` (GP log-lift) | #189, #181 | objective monomial `c·∏ x_j^{a_j}` (n>=2, a_j>0) → `u_j <= log x_j` lift + aux `t` + tangent cuts `t >= exp(s0)(1+s-s0)` |
| `inject_gp_constraint_cuts` (P16 constraint GP) | #116 | posynomial `<=` constraint `Σ_k c_k∏ x_j^{a_kj} <= b` (a_kj>=0) → `u_j <= log x_j` lift + OA cuts `Σ_k exp(s_k0)(1+s_k−s_k0) <= b` |
| `inject_signed_signomial_constraint_cuts` (P17 DC) | #114, #116 | signed-signomial `<=` constraint with a negative term → DC log-lift: OA tangents under `Pplus`, affine secant over `Pminus`, cut `T_plus − S_minus <= b` (any real exponents; both senses) |

Bilinear (P3) and linear-fractional (P4) are intentionally **not** wired: the
relaxation compiler already emits those envelopes, so a detector would be
redundant.

**GP log-lift — validated, and it does NOT apply to #189's instances (honest).**
`inject_gp_cuts` is *sound* (cut never exceeds the true monomial over 20k+
feasible points) and *value-preserving*, but it targets **objective** monomials
`c·∏ x_j^{a_j}` (n>=2). Measured on the two vendored #189 instances it fires
**nothing**:

| Instance | structure | `inject_all` | baseline |
|---|---|---|---|
| `cvxnonsep_nsig30` | linear objective; signomial in 1 *constraint* | `{}` (all 0) | already optimal+certified, 6.2 s |
| `clay0303hfsg` | separable-quadratic + bilinear layout, big-M binaries | `{}` (all 0) | optimal+certified, 52 s |

Conclusions: (1) `cvxnonsep_nsig30` already certifies fast — no gap to close —
and its signomial is a *constraint*, so an objective-monomial pass can't apply; a
**constraint-level GP reformulation** (the #116 y-space substitution) would be
needed. (2) `clay0303hfsg` is a different structure (separable quadratics +
bilinear); its looseness is an OBBT / separable-quadratic-cut problem (#208), not
one of the implemented patterns. The earlier "GP closes the relaxation part of
#189" hope is **retracted** — the implemented pass does not help these
instances.

**Constraint-level passes P16/P17 — built, and the honest measurement.** The
constraint-level GP (P16, `inject_gp_constraint_cuts`) and signed-signomial DC
(P17, `inject_signed_signomial_constraint_cuts`) passes now close the "objective
only" gap above. On inspection `cvxnonsep_nsig30`'s single constraint is **one
negative monomial** `−0.2·∏ x_j^{a_j} ≤ −1` (i.e. the concave bound
`0.2·∏ x_j^{a_j} ≥ 1`), not a posynomial — so P16 correctly skips it and **P17
fires** (it is a degenerate signed signomial). Measured before/after:

| Instance | P16 | P17 | obj | bound | nodes | note |
|---|---|---|---|---|---|---|
| `cvxnonsep_nsig30` | skip | **fires (1)** | 130.6287 (unchanged) | 130.6287 (unchanged) | 263 → 263 | sound + optimum-preserving, but **no bound/node gain** — discopt already relaxes this single concave-monomial constraint tightly |
| synthetic posynomial `≤` GP | fires | skip | unchanged | root-tight (~3 nodes) | — | sound; no measurable root gap (native McCormick already tight) |

Honest conclusion: P16/P17 are **sound, optimum-preserving, and fire on the right
structures**, but on the vendored corpus instances discopt's native relaxation is
already tight enough that the lift adds no measurable bound/node improvement. Their
payoff is on instances where the *separable* box relaxation of a coupled
posynomial/signomial constraint is genuinely loose; the corpus instances here are
not that case. No correctness claim is overstated.

## Fix-and-comment workflow

As each pattern lands: (1) implement + prove + certify in `symbolic/patterns.py`
and `test_patterns.py`; (2) wire its detector into `cut_recognizer` where it
auto-fires; (3) comment on the mapped issue(s) with the measured before/after and
a soundness note. This document is the index to drive that loop.
