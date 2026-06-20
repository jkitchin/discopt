# Relaxation & structured-cut pattern catalog

A catalog of **recognizable structural patterns** for which `discopt` can derive
a sound relaxation or cut, organized so each entry is independently *provable*.
Each pattern lists the **fields** it appears in, the **structural template**, the
**relaxation/cut**, a **correctness proof** (theorem + argument + citation), and
its **implementation status**.

Soundness convention throughout: a *relaxation* of a term `f` over a box gives
`cv <= f <= cc` with `cv` convex, `cc` concave; a *cut* is a valid inequality
implied by the constraints. Every implemented pattern is certified numerically by
the same theorem-style sampling gate used across the relaxation suite
(`verify_envelope` / `verify_cut`), in addition to the analytic proof below.

Legend — Status: **done** (implemented + certified test), **engine** (covered by
the envelope engine), **roadmap** (proof given here, implementation pending).

---

## Tier 1 — Univariate atom envelopes

### P1. Convex / concave univariate atom — **engine**
- **Fields:** ubiquitous (any NLP/MINLP).
- **Template:** a term `g(x)` with `g'' >= 0` (convex) or `g'' <= 0` (concave) on
  the box: `x^2, exp, x*log x` (convex); `log, sqrt, x^a (0<a<1)` (concave).
- **Relaxation:** convex `g` → `(cv, cc) = (g, secant)`; concave → `(secant, g)`.
- **Correctness:** for convex `g`, `g <= ` its chord (Jensen) ⇒ `secant >= g` is a
  valid concave overestimator; `cv = g` is convex and tight. Mirror for concave.
  *McCormick (1976).*
- Implemented in `symbolic.derive_envelope` / `mccormick.py`.

### P2. Single-inflection univariate atom — **done**
- **Fields:** gas friction (`f|f|`), kinetics (Arrhenius), ML activations
  (sigmoid/tanh), odd powers (`x^3`).
- **Template:** one inflection `c`: concave-then-convex (e.g. `x|x|`) or
  convex-then-concave (e.g. sigmoid).
- **Relaxation:** tangent/secant construction — on a straddling box one side
  follows `g`, the other a line through a box endpoint tangent to `g`, with a
  secant fallback when the tangent leaves its branch.
- **Correctness:** the convex envelope of a concave-then-convex `g` over `[a,b]`
  is `g` on `[t,b]` and the line through `(a,g(a))` tangent at `t` on `[a,t]`,
  where `g'(t)(t-a) = g(t)-g(a)`; convex by construction (linear then convex,
  slopes match at `t`) and `<= g` (the chord lies under `g` on the concave part).
  Mirror for the concave envelope. *Tawarmalani & Sahinidis (2002), §convex
  extensions.*
- Implemented in `symbolic.runtime.single_inflection_envelope`.

---

## Tier 2 — Bivariate product / ratio terms

### P3. Bilinear `x·y` (McCormick hull) — **done**
- **Fields:** pooling/blending (refinery), robust optimization, portfolio
  covariance, process flowsheets, bilinear matrix inequalities.
- **Template:** product of two box-bounded variables.
- **Relaxation:**
  `cv = max(x_L y + x y_L − x_L y_L, x_U y + x y_U − x_U y_U)`,
  `cc = min(x_U y + x y_L − x_U y_L, x_L y + x y_U − x_L y_U)`.
- **Correctness:** each inequality is a bound-factor product, e.g.
  `(x − x_L)(y − y_L) >= 0 ⇒ xy >= x_L y + x y_L − x_L y_L`; the four such products
  give the two under- and two over-estimators. `cv` is a max of affines (convex),
  `cc` a min of affines (concave). These four facets are exactly the convex hull
  of `{(x,y,xy)}` over the box. *Al-Khayyal & Falk (1983); McCormick (1976).*
- `mccormick.relax_bilinear`; certified in `test_patterns.py`.

### P4. Reciprocal `1/y` and linear-fractional `x/y` (`y>0`) — **done**
- **Fields:** efficiency/DEA, fractional programming (Charnes–Cooper), blending
  ratios, chemical equilibrium.
- **Template:** `1/y` with `0 not in [y_L, y_U]`; `x/y` with `x >= 0`.
- **Relaxation:** `1/y` convex (`y>0`) → `(cv, cc) = (1/y, secant)`.
  `x/y` via the lift `z = 1/y` (relaxed by the reciprocal envelope) and the
  bilinear hull of `x·z` over `z in [1/y_U, 1/y_L]`.
- **Correctness:** `(1/y)'' = 2/y^3 > 0` for `y>0` ⇒ convex ⇒ P1. For `x/y`,
  `x/y = x·z` with `z = 1/y`; the bilinear hull (P3) of `x·z` is valid for any `z`
  in its range, and `z`'s range follows from `1/y` monotone decreasing. Tighter
  closed-form hulls exist. *Tawarmalani & Sahinidis (2001), "Convex extensions
  and envelopes of l.s.c. functions."*
- `mccormick.relax_div`; lifted form certified in `test_patterns.py`.

---

## Tier 3 — Structured coupling cuts (constraint-chain / RLT)

### P5. Square-difference network coupling — **done**
- **Fields:** gas networks (Weymouth `f^2 = C(p_in^2 − p_out^2)`), water networks
  (Hazen–Williams), district heating, any pressure/flow + ratio network.
- **Template:** a chain of square-difference equalities plus a ratio equality
  `p_out = r · p_in`, with a bound-propagated terminal pressure.
- **Cut:** eliminate intermediates to get `r >= sqrt(phi(flow))`, then for an
  objective term `flow · (r^k − 1)` a univariate convex underestimator
  `h(flow) = flow · (sqrt(phi(flow))^k − 1)_+`.
- **Correctness:** each elimination is an exact equality substitution; the
  terminal bound is substituted at the extreme that *increases* the target
  (verified by the sign of the derivative), so `r >= sqrt(phi(flow))` is a valid
  lower bound; `term = flow·(r^k−1) >= flow·(max(1,sqrt(phi))^k − 1)_+ = h` since
  `flow >= 0` and `r^k − 1` is increasing; `h` is verified convex. Closes the
  issue-#15 gas gap 66.7%→0%. *RLT, Sherali & Adams (1990); reduction-constraint
  elimination.*
- `symbolic.constraint_cuts` + `symbolic.cut_recognizer`.

### P6. RLT cut for sum-to-constant bilinears — **done**
- **Fields:** pooling/blending (mass balance `Σ_j x_j = C`), refinery, supply
  chain, transportation with conservation.
- **Template:** a conservation equality `Σ_j x_j = C` (`x_j >= 0`) together with
  bilinear products `w_{ij} = x_i x_j` appearing in the model.
- **Cut:** `Σ_j w_{ij} = C · x_i` (a linear equality among the bilinear auxes).
- **Correctness:** multiply the valid equality `Σ_j x_j − C = 0` by `x_i >= 0`:
  `Σ_j x_i x_j = C x_i ⇒ Σ_j w_{ij} = C x_i`. This RLT equality is implied by the
  constraints and is generally *not* implied by the per-term McCormick hulls, so
  it tightens the relaxation. *Sherali & Adams (1990), Reformulation-Linearization
  Technique.*
- `symbolic.patterns.rlt_sum_constant_cut`; certified in `test_patterns.py`.

---

## Tier 4 — Transform-linearizable

### P7. Signomial / posynomial monomial `c·∏_j x_j^{a_j}` (`x_j>0`) — **done**
- **Fields:** geometric programming — analog/RF circuit design, structural
  design, chemical process design, epidemiology rate models.
- **Template:** a monomial term with positive variables and real exponents.
- **Relaxation:** substitute `u_j = log x_j`; the monomial becomes
  `exp(log c + Σ_j a_j u_j)`, **convex in `u`** ⇒ a convex underestimator in the
  log-domain (the basis of GP convexification).
- **Correctness:** `exp` is convex and increasing; `log c + Σ a_j u_j` is affine
  in `u`; composition of a convex increasing function with an affine map is convex
  *(Boyd & Vandenberghe, §3.2)*. A posynomial (sum of such monomials with positive
  coefficients) is therefore convex in `u`. *Boyd et al. (2007), "A tutorial on
  geometric programming."*
- `symbolic.patterns.posynomial_logconvex`; convexity certified in
  `test_patterns.py`.

### P16. Constraint-level GP reformulation `P(x) ≤ b` (`x_j>0`) — **done** (#116)
- **Fields:** geometric programming feasibility regions — the constraint analog of
  P7/P14, in chemical process, structural and RF circuit design.
- **Template:** a `≤` constraint whose body is a posynomial
  `Σ_k c_k·∏_j x_j^{a_kj}` with `c_k>0` and **non-negative** exponents `a_kj`.
- **Relaxation:** substitute `u_j = log x_j`. Each monomial becomes `exp(s_k)` with
  `s_k = log c_k + Σ_j a_kj u_j` affine in `u`, so the constraint reads
  `Σ_k exp(s_k) ≤ b` — convex in `u`. Inject the **convex** link `u_j ≤ log x_j`
  (hypograph of the concave `log`), log-domain bounds, and outer-approximation
  tangent cuts `Σ_k exp(s_k0)(1 + s_k − s_k0) ≤ b` over a grid of expansion points.
- **Correctness (no feasible `x` removed):** with `a_kj ≥ 0`, `u_j ≤ log x_j` gives
  `s_k ≤ log(monomial_k)`, hence `Σ_k exp(s_k) ≤ P(x)`; so for any feasible `x` the
  choice `u_j = log x_j` satisfies the link with equality and makes every OA cut
  `Σ_k exp(s_k0)(1+s_k−s_k0) ≤ Σ_k exp(s_k) = P(x) ≤ b` hold. `>=`, affine,
  negative-exponent and signomial (negative-coefficient) rows are skipped — those
  need the signed-signomial DC lift (P11), not this single-direction link.
- `cut_recognizer.inject_gp_constraint_cuts`; soundness, firing, optimum-
  preservation and skip-cases certified in `test_cut_recognizer.py`.

### P17. Constraint-level signed-signomial DC `s(x) ≤ b` (`x_j>0`) — **done** (#114/#116)
- **Fields:** signomial global optimization — chemical process synthesis, structural
  design, RF/analog circuits, anywhere a constraint mixes posynomial growth with
  negative (subtractive) signomial terms.
- **Template:** a `≤` (or sign-flipped `≥`) constraint whose body is a *signed*
  signomial `s(x) = Σ_k σ_k c_k·∏_j x_j^{a_kj}`, `c_k>0`, `σ_k=±1`, with at least
  one `σ_k = −1` term (pure posynomials go to P16) and any real exponents.
- **Relaxation:** lift `u_j = log x_j`; `s(u) = Pplus(u) − Pminus(u)` is a
  **difference of convex** functions (each monomial `exp(s_k)`, `s_k` affine, is
  convex). Relax convexly by **under**-estimating `Pplus` with OA tangents
  `T_plus(u) = Σ_{+} exp(s_k0)(1+s_k−s_k0)` and **over**-estimating `Pminus` with
  its affine box **secant** `S_minus(u) = Σ_{−}[exp(s_k^lo)+slope_k(s_k−s_k^lo)]`,
  giving the linear cut `T_plus(u) − S_minus(u) ≤ b` plus the convex link
  `u_j ≤ log x_j`.
- **Correctness:** at `u_j = log x_j` each monomial is exact, the tangent
  underestimates the convex `Pplus` and the chord overestimates the convex
  `Pminus`, so `T_plus − S_minus ≤ Pplus(x) − Pminus(x) = s(x) ≤ b`; the witness
  `u = log x` satisfies link + every cut, so no feasible `x` is removed. Holds for
  **any** real exponents (the bound directions are exponent-sign-independent),
  generalizing P16. *Lundell & Westerlund, signomial global optimization (SGO);
  Khajavirad, Michalek & Sahinidis (2012), convex-transformable intermediates.*
- `cut_recognizer.inject_signed_signomial_constraint_cuts`; soundness (40k-point
  sampling), firing, optimum-preservation, both-sense and skip-case tests in
  `test_cut_recognizer.py`. The secant scales linearly (no `2^n` corner blowup).

---

## Tier 5 — Domain-specific (roadmap, proven here)

### P8. AC power flow `V_i V_j cos(θ_ij)` / `sin(θ_ij)` (QC relaxation) — **roadmap**
- **Fields:** AC optimal power flow, state estimation, power-system planning.
- **Template:** products of voltage magnitudes with cos/sin of an angle
  difference, over `|θ| <= θ̄ < π/2`.
- **Relaxation:** relax `cos θ` (concave on `(−π/2,π/2)`) and `sin θ`
  (single-inflection) by their envelopes (already in `domains/power.py`), `w=V^2`
  by P1, and the product `V_i V_j (cos)` by recursive bilinear hulls.
- **Correctness:** concavity of `cos` on `(−π/2,π/2)` (`cos'' = −cos < 0`); the
  product envelope follows from composing the P3 bilinear hull with the trig
  envelope (valid since each factor's relaxation is valid and the composition of
  valid relaxations is valid). *Coffrin, Hijazi & Van Hentenryck (2016), "The QC
  Relaxation."*
- Univariate pieces implemented (`domains/power.py`); the simultaneous trilinear
  hull is the multivariate Phase-6 frontier.

### P9. Logarithmic-mean temperature difference (LMTD) — **roadmap**
- **Fields:** heat-exchanger network synthesis, process integration.
- **Template:** `ΔT_lm = (ΔT_1 − ΔT_2) / ln(ΔT_1/ΔT_2)` (and Chen/Paterson
  approximations).
- **Relaxation:** LMTD is concave and monotone in each `ΔT_i` on the positive
  orthant; bound by tangent planes (under) and the secant hyperplane (over).
- **Correctness:** the log-mean is a concave, symmetric, positively-homogeneous
  function of `(ΔT_1, ΔT_2)` on `R_{>0}^2`; concavity gives the tangent-plane
  overestimator and chord underestimator. *Mistry & Misener (2016).*
- Roadmap (bivariate; pairs with the Phase-6 multivariate verifier).

---

## Tier 6 — Combinatorial / disjunctive

### P10. Fortet/Glover product-of-binaries `z = ∏_i b_i` — **done**
- **Fields:** 0-1 polynomial programming, autocorrelation/QUBO, boolean products.
- **Template:** a product of binary variables.
- **Linearization (exact):** `z <= b_i ∀i`, `z >= Σ_i b_i − (n−1)`, `z >= 0`,
  equivalently `cv = max(0, Σ b_i − (n−1))`, `cc = min_i b_i`.
- **Correctness:** if all `b_i = 1`, `Σ b_i − (n−1) = 1` ⇒ `z = 1`; if some
  `b_k = 0`, `z <= b_k = 0` ⇒ `z = 0`; so `z = ∏ b_i` exactly at binary vertices.
  For `n = 2` these are the bilinear hull on `[0,1]^2`. *Fortet (1960); Glover &
  Woolsey (1974).* Issue #187.
- `patterns.binary_product_linearization`; certified in `test_patterns.py`.

### P12. Complementarity `x·y = 0`, `x,y >= 0` — **done**
- **Fields:** MPEC/equilibrium, contact mechanics, KKT systems, disjunctive.
- **Template:** a complementarity product equal to zero.
- **Cut:** `x/x_ub + y/y_ub <= 1`.
- **Correctness:** McCormick `xy >= x_ub y + x y_ub − x_ub y_ub`; with `xy = 0`,
  `x_ub y + x y_ub <= x_ub y_ub`; divide by `x_ub y_ub > 0`. The cut is implied by
  the disjunction `x=0 ∨ y=0` and excludes interior points with `xy > 0`.
  *Sherali & Adams (1990).* Issue #231.
- `patterns.complementarity_cut`; certified in `test_patterns.py`.

---

## How a pattern is "proven correct" here

1. **Analytic proof** — the entry above gives the validity argument (bound-factor
   product, exact elimination, convexity-by-composition, or RLT multiplication)
   plus the literature theorem.
2. **Numerical certification** — `test_patterns.py` / `verify_*` sample the box or
   feasible manifold and assert containment (`cv <= f <= cc`) and curvature
   (Jensen) within tolerance, the same gate the rest of the relaxation suite uses.

A pattern is only marked **done** when both hold.
