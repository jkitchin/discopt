# SymPy-driven envelopes & relaxations + certified learned envelopes — design / plan

**Status:** Phase 1 implemented and tested (`python/discopt/_jax/symbolic/`). Later
phases planned. Branch: `claude/sympy-envelopes-relaxations-4d3vwz`.

**Goal:** use SymPy as a *design-time* engine to derive, verify, and code-generate
tight convex/concave envelopes for nonlinear atoms that the hand-written McCormick
library (`discopt._jax.mccormick`, `_jax.envelopes`) does not yet cover — focused on
terms from **chemical-engineering, gas-network, and electrical-grid** optimization —
and to extend this with **certified learned (ML) envelopes** that exploit
guaranteed-convex / guaranteed-monotone neural networks.

## Design principles

1. **SymPy never runs on the solver hot path.** It produces closed-form envelope
   formulas and emits pure-JAX closures matching the univariate primitive contract
   `relax_fn(x, lb, ub) -> (cv, cc)`. SymPy is an opt-in `[sympy]` extra.
2. **Soundness is non-negotiable.** Every derived/learned atom must satisfy
   `cv(x) <= f(x) <= cc(x)` with `cv` convex and `cc` concave, and pass the
   theorem-style certification gate (`verify_envelope`) before registration.
3. **Reuse the existing stack.** Generated closures plug into the relaxation
   compiler and `McCormickRelaxationEvaluator` unchanged.

## Architecture

```
python/discopt/_jax/symbolic/
  envelope_deriver.py   # univariate curvature analysis + envelope synthesis  [DONE]
  codegen.py            # SymPy -> JAX (x,lb,ub)->(cv,cc) closures             [DONE]
  verification.py       # soundness + curvature + tightness certification     [DONE]
  certified_learned.py  # certify guaranteed-convex ML nets as envelopes      [Phase 8]
  registry.py           # register generated atoms into relaxation_compiler   [Phase 2]
  structured.py         # bi/tri-variate structured envelopes                 [Phase 6]
  domains/
    gas.py              # Weymouth f|f|, compressor terms                     [Phase 3]
    power.py            # cos/sin envelopes, V_iV_j(cos/sin), QC atoms        [Phase 4]
    chemeng.py          # LMTD, Arrhenius, Langmuir, entropy                  [Phase 5]
```

## Phase 1 — Foundations (DONE)

Univariate engine validated against existing atoms.

- **Curvature classification** (`Curvature`): CONVEX, CONCAVE, CONCAVO_CONVEX,
  CONVEXO_CONCAVE. Detects smooth inflections (real roots of `f''`) **and**
  non-smooth kinks (zeros of `Abs`/`sign`/`Heaviside` arguments) so the Weymouth
  term `x|x|` is correctly CONCAVO_CONVEX rather than silently CONVEX — a soundness
  trap that pure `solve(f''==0)` falls into.
- **Tangent-point construction** for single-inflection envelopes: solves
  `f'(t)(t-e) = f(t) - f(e)` in closed form via a positive-dummy side substitution
  (`endpoint = c ± u`, `u > 0`) so SymPy resolves the signs of non-smooth atoms.
  Closed forms recovered: `x^3 -> -a/2`, `x|x| -> e(1-√2)`, `x^5`.
- **Codegen** uses SymPy's `"jax"` printer so `sign`/`Abs` render to `jax.numpy`
  (not Python conditionals, which break under `vmap`). All box branching via
  `jnp.where` → one compiled closure serves every B&B node.
- **Verification**: randomized-box sampling for containment + Jensen-inequality
  curvature checks + tightness (gap) reporting.
- **Tests** (`python/tests/test_symbolic_envelope_deriver.py`, 27): classification,
  soundness, parity with `relax_square/exp/log/sqrt`, closed-form tangent points,
  jit/vmap, graceful rejection of DiracDelta-gradient forms.

## Phase 2 — Registration & round-trip

Wire generated closures into `relaxation_compiler` (a `registry.py` keyed by atom),
prove an atom flows end-to-end through `Model.solve()`. Reuse `relaxation_harness.py`.

## Phase 3 — Gas networks (first domain pack)

Weymouth `f|f|` (single-inflection, done at the engine level), compressor power /
ratio terms, plus a small pipe-network benchmark instance. Decision (confirmed with
user): gas networks first.

## Phase 4 — AC power / OPF

`cos`/`sin` envelopes over `[-θ̄, θ̄]`, `w = V^2`, and structured `V_iV_j cos/sin`
envelopes (QC relaxation, Coffrin et al.). Cross-check tightness vs. a known QC bound.

## Phase 5 — Chemical engineering

LMTD (log-mean ΔT), Arrhenius `exp(-E/RT)`, Langmuir `x/(1+x)`, entropy refinements.

## Phase 6 — Structured multivariate envelopes

SymPy-derived vertex-polyhedral / edge-concave facets for trilinear and product
terms (generalizes `relax_trilinear_exact`).

## Phase 7 — Docs & dissemination

`docs/notebooks/symbolic_envelopes.ipynb` with `{cite:p}` citations + new
`references.bib` entries (McCormick 1976; Tsoukalas & Mitsos 2014; Coffrin et al. QC
relaxation; Misener & Floudas). Update `_toc.yml`; rebuild with zero warnings.

## Phase 8 — Certified learned (ML) envelopes

**Motivation.** discopt already ships ICNN-based learned relaxations
(`_jax/icnn.py`, `_jax/learned_relaxations.py`): a pair of Input Convex Neural
Networks gives a convex `cv` and concave `cc` *by construction*. The gap to a *valid
relaxation* is the **bound**: convexity ≠ `cv <= f`.

**Key finding (prototype).** The existing wrapper enforces soundness by clamping
with the true value: `cv = min(cv_pred, f(x))`, `cc = max(cc_pred, f(x))`. This is
pointwise-bounding but **destroys convexity** — `min(convex, nonconvex f)` is not
convex — so the clamped output is not a sound *convex* relaxation for a
lower-bounding subproblem. `certified_learned.py` demonstrates this and provides the
fix: a **constant certified margin** `cv_pred - δ_lo`, `cc_pred + δ_hi` (margin
constant in `x` for a given box) which **preserves convexity** while restoring
soundness.

**Workstream.**
- (a) **Symbolic targets.** Train ICNNs to emulate the *exact* convex envelopes the
  symbolic engine computes, across the box family `(lb, ub)` → sound-by-target,
  tighter-than-McCormick, no per-node re-derivation.
- (b) **Guaranteed monotone/convex surrogates** for domain black-box terms
  (compressor maps — monotone in flow/speed → exact interval image `[g(lb),g(ub)]`;
  AC loss / power-flow surrogates — convex NN → certified convex relaxation;
  thermodynamic property models — monotone in T / convex in composition). Embedded
  via `discopt.nn`, relaxed *tightly because of* the guaranteed structure.
- (c) **Certification layer.** Bound worst-case under/over-estimation with the
  outward-rounded interval arithmetic in `_jax/convexity/interval.py` + a Lipschitz
  bound, subtract a sound constant margin (preserving convexity), track the gap.
  The same `verify_envelope` gate guards every learned atom before registration.

**Synergy.** Symbolic = exact but closed-form-limited; guaranteed ML = flexible /
high-dimensional but needs a certificate. Used together: symbolic + interval
machinery *certifies* the ML relaxation; ML *generalizes* the envelope across the
box family and into dimensions without closed forms. "Learned, then certified."

**Honest caveat.** Sampling-based margins are certified *over the sampled boxes* with
a safety factor — a bug-catching gate, not a proof. The rigorous version uses the
interval/Lipschitz path (tracked as a refinement of the certification layer).
