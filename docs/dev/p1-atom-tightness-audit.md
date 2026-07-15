# P1 entry experiment — per-atom factorable-envelope tightness audit (#632)

**Status:** measurement + report (2026-07-12) · **Owner issue:** #632 · **Plan:**
`docs/dev/avm-canonicalization-plan.md` §0′ (P1 — tight factorable per-atom
envelopes) · **Reproduce:**
`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python discopt_benchmarks/scripts/p1_atom_tightness_audit.py --census`
· **This is a MEASUREMENT stage — no solver/relaxation code was changed.**

## 0. Thesis under test (P1) and what this audit decides

P1's thesis (plan §0′): the tightness the deleted per-node composite 1-D hull
(H-UNI) chased — nvs09's root gap — should be recovered the SOTA way, by making
each **factorable atom's** envelope tight and composing them via the
auxiliary-variable/McCormick method, not by building a bespoke per-node 1-D hull.
This audit **measures, per atom, where discopt's current default factorable
relaxation is loose**, so the highest-leverage atoms are tightened first.

**Headline finding — the current default gap is not "loose envelopes"; for the
highest-leverage atom classes there is *no envelope at all*.** discopt's default
path applies near-exact envelopes to every *base* atom (§2), but on two composite
classes it does not atomize — it drops the whole objective to a separable-interval
or feasibility fallback, producing either a bound of 0 (all tightness lost) or no
finite bound at all. Both classes are exactly nvs09's objective (§4). The P1
canonical-AVM cutover (plan R1.1/R1.2) is therefore the lever: **just atomizing
these composites recovers most of the gap with no new envelope math** (§3), and a
tighter composition then closes the residual.

## 1. Method (all numbers in-container, no external solver, no scipy)

- **Root bound** = the LP relaxation optimum from discopt's **own in-house Rust
  simplex** via `MccormickLPRelaxer(model).solve_at_node(lb, ub)` — the same
  engine `gen_claim_baseline.py` uses; **not** scipy/HiGHS (plan §3.1, ledger
  R0.3).
- **True optimum** = analytic where closed-form, else a deterministic 1-D fine
  scan (`400 001` points), else exact **vertex enumeration** for multilinear /
  signed-monomial atoms (their extrema over a box are attained at a corner).
- **`fb` column** = the objective could not be linearized and the relaxation fell
  back to the separable-interval / feasibility path (`build_milp_relaxation`'s
  "could not linearize the objective" warning fired) ⇒ **no genuine per-atom
  envelope was applied**. `bound = None` ⇒ no finite bound at all.
- `abs_gap = true_opt − bound` (≥ 0 for a sound min-sense relaxation; the script
  flags `bound > true_opt + 1e-6` as UNSOUND — none were).
- **Forcing interior optima.** Multilinear/product/division atoms are exact on box
  *vertices* (McCormick is vertex-exact), and single-box extrema of these atoms
  are always at a vertex — so a plain min shows gap 0 regardless of envelope
  quality. To expose the real envelope looseness we add **one linear equality**
  (a sum slice, e.g. `Σxᵢ = c`) that pushes the optimum to the interior; the
  residual gap there is the envelope's true looseness. This is precisely how
  product looseness manifests inside real models.

## 2. Base atoms are near-exact (confirmed) — the looseness is not here

Every base univariate atom, in **both** directions (min exposes the convex
underestimator; max exposes the secant), over representative boxes:

| atom | box | true opt | root bound | abs gap | fb |
|---|---|---|---|---|---|
| `x**2`, `x**3`, `x**4` | [-2,3]/[-2,2] | — | — | ≤ 1e-7 | . |
| `exp(x)` | [-1,2] | ±0.36788 / -7.38906 | =true | ≤ 1e-8 | . |
| `log(x)` | [0.5,5] | -0.69315 / -1.60944 | =true | ≤ 1e-8 | . |
| `sqrt(x)` = `x**0.5` | [0.5,5] | 0.70711 / -2.23607 | =true | ≤ 1e-8 | . |
| `1/x` | [0.5,5] | 0.20000 / -2.00000 | =true | ≤ 1e-8 | . |
| `x**-2` | [0.5,5] | 0.04000 / -4.00000 | =true | ≤ 1e-8 | **Y** |
| `x**2.5` | [0.5,5] | 0.17678 / -55.90170 | =true | ≤ 1e-7 | . |
| `x**0.2` | [0.5,5] | 0.87055 / -1.37973 | =true | ≤ 1e-8 | . |
| `x*y` (min/max) | [1,4]² | 0.25 / -4 | =true | ≤ 1e-8 | — |
| `x/y` (min/max) | [1,4]² | 0.25 / -4 | =true | ≤ 1e-8 | **Y** |
| `x**2*y` (min/max) | [1,3]² | 1 / -27 | =true | ≤ 1e-8 | . |

**Reading:** the secant + tangent envelopes for `exp/log/sqrt/1/x`, even/odd/
negative/fractional powers, bilinear, monomial, and division are all
machine-precision-exact on a plain box, both directions. (`x**-2` and `x/y`
carry `fb=Y` because the default path routes them through the separable /
fractional fallback rather than a dedicated envelope, but the fallback happens to
be exact at their vertex optima — a latent gap only, see Target 4/5.) **Base
envelopes are not a leverage target.**

## 3. Composition is where tightness is lost — the crux (three-tier evidence)

| composite (objective) | box | true opt | root bound | abs gap | rel gap | fb |
|---|---|---|---|---|---|---|
| `(log(x-2))**2` | [3,9] | 0.00000 | -0.00000 | 0.0* | — | Y |
| **`(log(x-2))**2 + (log(10-x))**2`** (nvs09 per-var) | [3,9] | **3.66667** | **-0.00000** | **3.66667** | **0.786** | Y |
| `exp(-2*(x-1)**2)` | [-1,3] | 0.00034 | **None** | **no bound** | — | Y |
| `x**3 - 3*x` (nonconvex + linear tilt) | [-2,2] | -2.00000 | -10.00000 | 8.00000 | 2.667 | . |
| `(x**2-1)**2` | [-2,2] | 0 | -0.0 | 0.0 | — | . |
| `sin(x)**2` | [0,π] | 0 | -0.0 | 0.0 | — | . |

\* the single square's min is 0, so its gap is 0 *by coincidence* even though no
envelope was applied — the pair below removes the coincidence.

### 3.1 The nvs09 per-variable composite: default → AVM → exact

The decisive measurement. `(ln(x−2))² + (ln(10−x))²` on [3,9], true min **3.66667**:

| tier | how | root bound | loss vs true | fb |
|---|---|---|---|---|
| **default** (today) | no atomization: `log**2` seen as `log*log` product ⇒ separable interval (each square ≥ 0 independently) | **0.00000** | 3.66667 (100%) | Y |
| **AVM factorable** | introduce `w = ln(·)` (exact ln envelope), then `w²` (exact square envelope), McCormick-composed | **1.89328** | 1.77339 (48.4%) | . |
| **exact 1-D hull** (deleted H-UNI) | analytical per-atom 1-D envelope | 3.66667 | 0 | — |

The AVM tier is measured in-container by building the decomposed model explicitly
(`w1==log(x-2)`, `w2==log(10-x)`, `min w1²+w2²`) — the relaxer then applies **real
envelopes** (`fb=.`). **Atomizing alone recovers 1.89328 of 3.66667 = 51.6% per
variable, with zero new envelope math** — the whole point of the P1 canonical AVM.
The residual 1.77339 (48.4%) is the *composition-tightness* target: `w1²` and
`w2²` are relaxed independently and the shared-`x` coupling survives only through
the two `ln` secants, so the LP floor is `Σ secant(x)²`, minimized at `x=6` to
`2·(ln7/6·3)² = 1.893` — exactly the shared-secant value the lever-a analysis
predicted (`docs/dev/lever-a-root-tightness-plan.md` §3).

### 3.2 Multilinear products: recursive McCormick is catastrophic on wide boxes

Positive product maximized on a sum-slice (interior optimum), true max = equal
split (AM-GM):

| atom | arity / box | true opt | root bound | abs gap | rel gap |
|---|---|---|---|---|---|
| `x*y \| x+y=5` | n=2, [1,4] | -6.25 | -8.50 | 2.25 | 0.310 |
| `x*y*z \| Σ=6` | n=3, [1,4] | -8 | -22.00 | 14.0 | 1.556 |
| `prod5 \| Σ=25` **NARROW** | n=5, [4,6] | -3125 | -4400 | 1275 | 0.408 |
| `prod5 \| Σ=25` **WIDE** | n=5, [1,10] | -3125 | **-44445** | **41320** | **13.22** |

**Reading:** the recursive-pairwise-McCormick gap grows with **both** arity (n=2
→ n=5) and box width (NARROW rel 0.41 → WIDE rel **13.2**, a 32× worse relative
gap on the same arity). This is the known SOTA weakness and the log-space lever's
target (§5 Target 3/4).

## 4. nvs09 anchor and root-gap attribution

nvs09: `min Σᵢ₌₁¹⁰[(ln(xᵢ−2))² + (ln(10−xᵢ))²] − (∏ᵢxᵢ)^0.2`, 10 integer vars
∈ {3..9}; reference optimum **−43.134** (`minlplib.solu`, cited in lever-a §1).

| probe | reference | default root bound | fb |
|---|---|---|---|
| **nvs09 full** (`nvs09.nl`, default LP) | opt −43.134 | **None (no finite bound)** | Y |
| nvs09 squares-only (10 vars, [3,9]) | 36.66673 | **-0.00000** (loses all 36.667) | Y |
| nvs09 `(∏xᵢ)^0.2` term ([3,9]¹⁰) | 9 (min) / 81 (max) | **None (both directions)** | Y |

**Attribution — nvs09's entire root gap is two missing atom envelopes:**

1. **`(∏xᵢ)^0.2` — fractional power of a 10-way positive product — is the
   *blocker*.** It produces **no finite bound in either direction**, which alone
   makes the whole nvs09 objective unbounded below on the default path (`bound =
   None`). SOTA fix: the log-space / exponential transform (Target 3).
2. **The 20 `(ln(xᵢ−2))²` / `(ln(10−xᵢ))²` square atoms** contribute the full
   **36.667** of achievable bound but the default separable-interval fallback
   recovers **0** of it (each square's interval min is 0). Atomizing recovers
   ≥ 1.893/var ≈ **18.9 total**; tightening the composition targets the rest
   (Target 1 + 2).

So nvs09 is a clean, faithful probe of exactly the P1 atom classes — not a special
case. **Handoff:** the *end-to-end certified* nvs09 root gap and the global50 /
BARON side-by-side require the local host (BARON absent in-container; `minlplib.solu`
oracle on the user's host) — see §6.

## 5. Corpus frequency and the prioritized target list

**Frequency signal (62 vendored `.nl`, `--census`):** **9/62 (15%)** drop the
objective to the fallback path on the default relaxation — `fac2, heatexch_gen2,
heatexch_gen3, nvs06, nvs09, tspn05, tspn08, tspn10, tspn12`; **16/62 (26%)**
produce no finite root bound. The missing/loose atom classes below are corpus-wide,
not nvs09-specific.

Leverage = measured gap × frequency. Targets ranked; each states the hypothesis,
the evidence in this audit, the proposed SOTA-standard envelope, and a kill
criterion (house style, `performance-plan.md` §6).

> **Target 1 — Atomize composite univariate `f(x)^p` / `call·call` (the P1 AVM
> cutover). [ENABLING, highest leverage.]** *Hypothesis:* the default path never
> atomizes `pow(FunctionCall, p)` / `call*call`, so it loses 100% of these atoms'
> bound; performing the canonical AVM decomposition (aux `w=f(x)` with its exact
> envelope, then the outer atom over `w`) recovers most of it with **no new
> envelope math**. *Evidence:* §3.1 — default 0.0 vs AVM-composed **1.89328** vs
> true 3.66667 (recovers 51.6%/var; ≈18.9 on nvs09's 10 vars), measured
> in-container. *SOTA envelope:* the R1.1/R1.2 canonical DAG + atomizer +
> univariate dispatcher (plan §2). *Expected win:* nvs09 gains ≥ 1.89/var of bound
> purely from atomization; the 9 fallback-corpus instances get a genuine objective
> envelope. *Kill:* if the AVM-composed bound does not exceed the separable-
> interval bound on the nvs09 per-var probe — already falsified-safe (measured
> +1.89328).

> **Target 2 — Tighten the composition residual (simultaneous outer-atom
> envelope + lazy OA cuts). [highest *tightness* leverage after Target 1.]**
> *Hypothesis:* AVM-McCormick leaves 48.4%/var on the nvs09 composite because the
> outer squares are relaxed independently and the shared-`x` coupling survives
> only through the inner secants (LP floor = `Σ secant²`); refining the outer
> convex atom with gradient/OA cuts at the LP solution (plan P2) and/or a tighter
> secant/tangent set over the actual inner range closes it. *Evidence:* §3.1 —
> residual 1.77339/var between AVM (1.89328) and exact (3.66667). *SOTA envelope:*
> Kelley outer-approximation cuts on the convex outer atom added lazily
> (plan P2), tangent/secant refinement, or the rigorous analytical 1-D per-atom
> envelope (lever-a §0.1 construction, already built flag-gated) used as the
> univariate atom's envelope. *Expected win:* per-var bound → 3.667, i.e. nvs09
> root → the −43.134 certificate (combined with Target 3). *Kill:* if OA cuts +
> tighter outer envelope do not raise the per-var bound above 1.89328 toward
> 3.66667.

> **Target 3 — Log-space (exponential-transform) envelope for `(∏xᵢ)^a`,
> positive vars. [nvs09 blocker; unblocks a finite bound.]** *Hypothesis:* a
> fractional/integer power of a positive multilinear product has no finite default
> bound; the log-space relaxation (`s = a·Σ ln xᵢ` with exact `ln` envelopes,
> `t = exp(s)` with the exact `exp` envelope) yields a finite, tight bound and
> removes the recursion. *Evidence:* §4 — `(∏xᵢ)^0.2` over [3,9]¹⁰ = **None** both
> directions (the nvs09 unboundedness cause); §3.2 — recursive McCormick is 13.2×
> rel-loose on the wide 5-product. Strict positivity holds on nvs09 ([3,9]).
> *SOTA envelope:* BARON functional transform / Maranas–Floudas signomial
> relaxation (the H-LOG chain, plan §2.2 taxonomy), strict-positivity gated on the
> FBBT root box. *Expected win:* the nvs09 `−(∏)^0.2` term gains a finite bound in
> [−81,−9]; wide-box products tighten dramatically. *Kill:* if the log-space bound
> does not beat recursive McCormick on `prod5 WIDE` (must, per lever-a §3).

> **Target 4 — Simultaneous multilinear convex envelope (replace chained pairwise
> McCormick). [wide-box products beyond the positive case.]** *Hypothesis:*
> recursive pairwise McCormick compounds looseness with arity and box width; the
> per-term simultaneous multilinear hull (exact facets for tri-/multilinear) is
> tighter. *Evidence:* §3.2 — n=2 rel 0.31 → n=3 rel 1.56 → n=5 wide rel 13.2.
> *SOTA envelope:* simultaneous trilinear/multilinear convex envelope (the
> `relax_signomial_multi`/`envelopes.py` machinery the catalog scopes) or the
> log-space transform when all factors are positive. *Expected win:* close a large
> fraction of the 41320 wide-product gap. *Kill:* if the simultaneous envelope
> ties pairwise McCormick on the trilinear interior slice.

> **Target 5 — Two-piece univariate hull for odd/nonconvex powers over
> sign-straddling boxes. [lower frequency, clean fix.]** *Hypothesis:* an odd
> power over a box that straddles 0 gets a loose single-piece convex
> underestimator; the exact concave∪convex two-piece hull (or one spatial branch
> at 0) makes each sub-box's monomial envelope exact. *Evidence:* §3 — `min
> x³−3x` on [−2,2] = **−10** vs true −2 (gap 8, rel 2.67); the looseness is the
> `x³` underestimator over [−2,0]. *SOTA envelope:* extend the existing
> definite-curvature monomial-secant kernel to the two-piece hull, or branch-and-
> reduce split at the inflection. *Expected win:* exact on each definite-curvature
> half after one split. *Kill:* if splitting at 0 does not make each half's
> monomial envelope exact (must — each half is convex or concave).

**Sequencing:** Target 1 is the enabling prerequisite (atomize before you can
tighten a composed atom) and is exactly the R1.1/R1.2 cutover already scaffolded.
Target 3 unblocks nvs09's finite bound. Targets 2 + 3 together are the nvs09
certificate. Targets 4/5 are corpus-wide product/power tightness.

## 6. What needs the local-host oracle (handoffs)

All bounds/gaps above are in-container and reproducible. The following are
**out-of-container** and are handed off, not claimed here:

- **End-to-end certified nvs09 root gap vs `minlplib.solu`** and the certified
  objective — the oracle file lives on the user's host
  (`~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu`); in-container we use
  the −43.134 constant cited in the plan. (Ledger R0.5 already flags nvs09's
  in-container certification as environment-sensitive.)
- **BARON / global50 side-by-side** (plan §0′ measurement of record,
  `global_opt_baron_vs_discopt.py --time-limit 60`): BARON is not installed
  in-container and the global50 corpus is on the user's host. The per-atom targets
  above must be re-measured end-to-end there before any envelope graduates.
- **Full MINLPLib census** (beyond the 62 vendored `.nl`): the frequency numbers
  in §5 are the vendored subset; the full-snapshot frequency is a host-side sweep
  (plan R3.3).

## 7. Reproduction

```bash
cd discopt && source .venv/bin/activate
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python discopt_benchmarks/scripts/p1_atom_tightness_audit.py --census
```

Deterministic (fixed grids, vertex enumeration, no randomness/timestamps). The
script also writes structured results with `--json <path>`. The census (~2 min)
is gated behind `--census`; the atom + nvs09 sections run in seconds.
