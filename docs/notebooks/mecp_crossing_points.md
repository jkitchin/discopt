# Minimum-Energy Crossing Points (MECP / MECI)

A **minimum-energy crossing point** is the lowest-energy molecular geometry at
which two electronic potential energy surfaces are degenerate. It plays the
role in spin-forbidden and photochemical reactions that a transition state
plays in ordinary chemistry: in nonadiabatic transition state theory the rate
depends on the crossing energy through a Boltzmann factor
{cite:p}`Lykhin2016`, so the *lowest* crossing point is the one that sets the
rate.

Two cases, distinguished by whether the two states can couple:

| | States | Degeneracy conditions | Seam dimension |
|---|---|---|---|
| **MECP** (spin crossing) | different multiplicity — e.g. singlet/triplet | `E₁ = E₂` only | 3N−7 |
| **MECI** (conical intersection) | same multiplicity | `E₁ = E₂` **and** interstate coupling `H₁₂ = 0` | 3N−8 |

Written as an optimization problem this is a small, dense, continuous,
nonconvex NLP with one or two nonlinear equality constraints:

$$
\min_x\; E_1(x) \quad \text{s.t.} \quad E_1(x) - E_2(x) = 0,
\qquad x \in [x^L, x^U]
$$

which is exactly what discopt's spatial branch-and-bound is for. What discopt
adds over the established MECP algorithms — gradient projection
{cite:p}`Bearpark1994`, the Lagrange–Newton method {cite:p}`Yarkony1993`, the
smooth penalty method {cite:p}`Levine2008`, branching-plane updating
{cite:p}`Maeda2010` — is a **certificate**: all of those are local methods, and
the state of the art for finding *multiple* crossing points is stochastic
exploration {cite:p}`Lindner2019`. None of them can tell you the crossing point
you found is the lowest one.

```{warning}
How you write the problem matters enormously — far more than for a local
solver, where all the forms below are equivalent. The same physics spans about
four orders of magnitude in tractability depending on the formulation. The two
mistakes to avoid are in "Two things not to do" below: the penalty objective,
and free Cartesian coordinates.
```

## A spin-crossing MECP

For states of different multiplicity the interstate coupling vanishes by spin
symmetry, so the adiabatic surfaces *are* the diabatic ones and the problem
above needs no reformulation. Here are two Morse-oscillator surfaces over the
three bond distances of a triatomic — the singlet deeper and shorter-bonded,
the triplet shallower and longer-bonded:

```python
import discopt.modeling as dm

D1, a1, b1 = 4.0, 1.55, 1.15      # lower state:  depth, range, r_eq
D2, a2, b2, dE = 2.5, 1.00, 1.66, -0.40   # upper state, offset down
TILT = [0.10, -0.06, 0.03]        # identical in both states (see note below)

def surface(d, D, a, b, dE):
    """Sum of Morse oscillators over the bond distances ``d``."""
    val = dE
    for k, dk in enumerate(d):
        val = val + D * (1.0 - dm.exp(-a * (dk - b))) ** 2 + TILT[k] * dk
    return val

m = dm.Model("mecp_triatomic")
d = [m.continuous(f"d{k}", lb=0.70, ub=3.20) for k in range(3)]

W1 = surface(d, D1, a1, b1, 0.0)
W2 = surface(d, D2, a2, b2, dE)

m.minimize(W1)                    # minimize the energy ...
m.subject_to(W1 - W2 == 0)        # ... on the crossing seam

result = m.solve(time_limit=60)
print(result.status, result.objective, result.bound, result.gap_certified)
print({k: float(v) for k, v in result.x.items()})
```

```text
optimal 0.8199959475426919 0.8199170904357771 True
{'d0': 1.3244686148294158, 'd1': 1.3410473365240474, 'd2': 1.33158614366199}
```

Five branch-and-bound nodes, under a second, `gap_certified=True`. That last flag is
the whole point: the reported gap is a proof, so this is the lowest crossing
point *anywhere in the box*, not merely the one nearest the starting guess.

```{note}
`TILT` is added to **both** surfaces. A term common to both states cancels
from the crossing condition `W₁ − W₂ = 0` while still changing the energy
*along* the seam — a convenient way to break a model's permutation symmetry
without moving the seam. Physically it plays the role of a substituent effect
or an external field.
```

## Why the certificate is worth having

Consider a spin-crossing model whose lower state has two conformer wells (an
inversion or torsion coordinate), so the seam has two low-lying basins at
different energies — the ordinary situation for a real molecule. A brute-force
grid-and-bisect search over the 2-coordinate version finds the global crossing
point at `E = 1.270685` and a second one at `E = 1.548537`.

Running the two standard local algorithms from 200 uniformly random starting
geometries:

| Method | reached the seam | found the **global** basin |
|---|---|---|
| Smooth penalty {cite:p}`Levine2008` (σ escalated 0.5→128, L-BFGS-B) | 200/200 | **127/200 (63.5 %)** |
| Direct constrained (SLSQP on `min W₁ s.t. W₁−W₂ = 0`) | 199/200 | **78/199 (39.2 %)** |

Both reliably reach the seam. Neither reliably reaches the *bottom* of it: a
local search reports a crossing point 0.278 too high in 37 % to 61 % of runs,
and SLSQP sometimes lands in a third basin higher still. discopt returns the
global basin with a proof in 0.5 s and 5 nodes.

If the energy unit is read as eV, that 0.278 basin error is a factor of
`exp(0.278 / 0.02585) ≈ 4.6 × 10⁴` in a nonadiabatic-TST rate constant at
300 K. A local MECP optimizer that converges to the second-lowest crossing
point does not give a slightly worse answer — it gives a rate wrong by four to
five orders of magnitude.

## Two things not to do

### 1. Do not hand the solver the penalty objective

The penalty method {cite:p}`Levine2008` is the standard MECP algorithm in
quantum chemistry, and it minimizes an unconstrained objective:

$$
F_\sigma(x) = \tfrac{1}{2}\big(E_1 + E_2\big)
  + \sigma\, \frac{(\Delta E)^2}{|\Delta E| + \alpha}
$$

discopt can express this — `dm.abs` and division are both supported — and it is
exactly the wrong thing to give a relaxation-based solver. The numerator is
squared and the denominator's lower bound is the small smoothing parameter `α`,
so the McCormick relaxation of that quotient is enormous. On the
**two-variable** model above:

| σ, α | status | objective | dual bound | relative gap |
|---|---|---|---|---|
| 3.5, 0.025 | `feasible` (300 s limit) | 3.3125 | −11177.6 | 3.4 × 10³ |
| 20, 0.005 | `feasible` (300 s limit) | 10.5175 | −212877.3 | 2.0 × 10⁴ |

Compare 5 nodes and a certificate for the constrained form. Use the
constrained form.

The penalty objective *is* the right choice in one place: `solver="direct"`,
the derivative-free search, needs an unconstrained problem. There it works
well — but it returns no bound, and reports `gap_certified=False`.

### 2. Do not use free Cartesian coordinates

Production MECP codes work in 3N Cartesians and project out the six
translation and rotation modes. For a local optimizer that redundancy is
harmless. For branch-and-bound it is fatal: every translation and rotation of
the MECP is *also* an MECP, so the set of global optima is a 6-dimensional
manifold, no finite subdivision isolates an optimum, and the dual bound never
closes.

The same triatomic MECP, three ways:

| Coordinates | free vars | certified | dual bound | nodes | wall |
|---|---|---|---|---|---|
| internal (3 bond distances) | 3 | **yes** | 0.819917 | 5 | 0.96 s |
| free Cartesian (9, 6 redundant) | 9 | **no** — 93 % gap at 180 s | **−0.109858** | 341 | 180 s |
| gauge-fixed Cartesian | 3 | **yes** | 0.819964 | 329 | 14.6 s |

All three find the identical geometry and energy — this is purely about
certifiability. Gauge fixing (atom 0 at the origin, atom 1 on the +x axis,
atom 2 in the xy half-plane) restores it, at 66× the nodes of the
internal-coordinate form because the energies then reach the coordinates
through `sqrt` of a sum of squares rather than directly.

**Pose MECP problems in internal coordinates, or in a gauge-fixed Cartesian
frame.**

## Conical intersections: avoiding the radical

For two states of the same multiplicity the adiabatic energies are the
eigenvalues of a 2×2 matrix,

$$
E_\pm = \frac{W_{11}+W_{22}}{2} \pm
  \sqrt{\left(\frac{W_{11}-W_{22}}{2}\right)^{2} + W_{12}^{2}}
$$

and the radical vanishes at the solution, so `E_±` are not differentiable to
any order exactly where the answer is. That non-differentiability is the reason
the field's algorithms are built the way they are, and the modern answer is to
work with *smooth* quantities instead {cite:p}`Galvan2023,Richings2023`.

discopt has `dm.sqrt`, so the adiabatic form is expressible and does solve. But
two reformulations are better.

### Diabatic form — the cheapest

Degeneracy of a 2×2 symmetric matrix is `W₁₁ = W₂₂` **and** `W₁₂ = 0`. In a
diabatic representation the radical disappears entirely:

```python
m = dm.Model("meci_diabatic")
qt = m.continuous("qt", lb=-30, ub=30)   # tuning mode
qc = m.continuous("qc", lb=-30, ub=30)   # coupling mode

# Two-state linear vibronic coupling Hamiltonian (Koeppel/Domcke/Cederbaum)
common = 0.5 * 0.020 * qt**2 + 0.5 * 0.015 * qc**2
W11 = 0.00 + 0.12 * qt + common
W22 = 0.30 - 0.18 * qt + common
W12 = 0.08 * qc

m.minimize(0.5 * (W11 + W22))
m.subject_to(W11 - W22 == 0)
m.subject_to(W12 == 0)

result = m.solve(time_limit=60)
print(result.status, result.objective, result.gap_certified, result.node_count)
```

```text
optimal 0.13 True 0
```

Zero nodes: the convexity detector disposes of it at the root. This model's
MECI is analytic (`qt = 1`, `qc = 0`, `E = 0.13`), and that is what comes back.

For comparison, the same problem posed adiabatically — `min E₋ s.t. E₊−E₋ = 0`
with `dm.sqrt` — also certifies and also lands on `0.13`, but takes 1.53 s
instead of 0.03 s. The radical costs a 50× wall-clock factor for the same
answer, because the tangent facet of `sqrt` at zero has infinite slope and is
dropped from the envelope.

### Characteristic-polynomial form — no diabatization needed

The coefficients of the characteristic polynomial of the potential matrix are
smooth through the intersection even though its eigenvalues are not
{cite:p}`Richings2023`:

$$
T(x) = E_1 + E_2 = W_{11}+W_{22}, \qquad
D(x) = E_1 E_2 = W_{11}W_{22} - W_{12}^{2}
$$

Degeneracy is the vanishing of the discriminant, and at degeneracy
`E₁ = E₂ = T/2`:

```python
m.minimize(0.5 * T)
m.subject_to(T**2 - 4 * D <= 0)     # (E₁-E₂)² ≤ 0, so ≤ is exact here
```

The inequality is valid because `T² − 4D = (E₁−E₂)² ≥ 0` identically, and it
solves faster than the equality version (0.85 s vs 2.48 s for the same 7-node
tree). This form needs no diabatization at all — `T` and `D` are computable
from the adiabatic energies directly, which is why they are what the ML-seam
literature fits.

```{warning}
The characteristic-polynomial form is **not** free, and on a spin crossing it
is a bad trade: squaring the surfaces widens the relaxation. On the two-basin
model where `W₁ − W₂ = 0` certified in 5 nodes and 0.5 s, `T² − 4D ≤ 0` did not
certify in 300 s (3353 nodes, 1 % gap). Use the plain difference whenever the
coupling vanishes; reach for the characteristic polynomial only for a
same-spin conical intersection.
```

## How large a molecule?

A molecule with N atoms has 3N−6 internal coordinates. Certified solve cost on
the two-basin spin-crossing model, 300 s limit:

| coordinates | 3N−6 ⇒ | nodes | wall | certified |
|---|---|---|---|---|
| 3 | triatomic | 47 | 1.5 s | yes |
| 6 | 4 atoms | 1113 | 43 s | yes |
| 8 | — | 7709 | 194 s | yes |
| 9 | 5 atoms | 13585 | 300 s | no — 0.6 % gap when the clock ran out |

Cost grows about ×2.7 per added coordinate, which is the expected exponential
for spatial branch-and-bound. So certified global MECP is available outright
for triatomics through roughly 5-atom systems, and for larger molecules only in
a reduced set of active coordinates with the rest frozen. That is how MECP
scans on large systems are done anyway — but the certificate then applies to
the reduced subspace, not the full geometry, and should be described that way.

## Several states: which pair crosses lowest?

A molecule usually has more than two low-lying states, and the crossing that
matters is the lowest one over *any* pair. That is a disjunction, so the
problem becomes an MINLP — binaries choose the active pair:

```python
m.minimize(E)
m.subject_to(sum(y) == 1)                     # exactly one pair is active
for p, (i, j) in enumerate(PAIRS):
    slack = BIGM * (1 - y[p])
    m.subject_to(w[i] - w[j] <= slack)        # y_p=1 => states i,j degenerate
    m.subject_to(w[j] - w[i] <= slack)
    m.subject_to(E - w[i] <= slack)           # y_p=1 => E is that energy
    m.subject_to(w[i] - E <= slack)
```

Over three states (three candidate pairs, two coordinates), against the obvious
baseline of enumerating the pairs as separate continuous solves:

| Route | energy | certified | nodes | wall |
|---|---|---|---|---|
| enumeration (3 separate solves) | 1.270684 | yes | 11 | 2.14 s |
| disjunctive MINLP (1 solve) | 1.270684 | yes | 51 | 5.70 s |

Both certify and agree. **At three states, enumeration wins** — three tiny
independent problems beat one MINLP carrying binaries and big-M slack for all
of them. The disjunctive formulation is the interesting one when the number of
states grows, since pairs grow as O(K²) and a single tree can prune a whole
pair on a bound instead of solving it; note that in the run above one pair (the
two upper states) never crosses inside the box at all, and enumeration spends a
solve finding that out. Where the crossover lies has not been measured.

```{note}
The example uses a hand-rolled big-M taken from a grid maximum, validated after
the fact by checking the selected pair really is degenerate at the solution
(`|W₀−W₁| = 5.3e-15`). For production use, derive the constant from interval
bounds, or hand the disjunction to discopt's GDP path via `gdp_method` instead
of writing big-M by hand.
```

## Real surfaces are black boxes

Everything above assumes an algebraic expression for the energies. Real `E₁`
and `E₂` come from an electronic-structure program, so a certified answer is
necessarily a statement about a **surrogate**, not about the true surface.
Three routes, in decreasing order of what they give you:

1. **Write the surrogate symbolically.** If the fit has a known functional form
   — Morse, polynomial, Gaussian, permutationally-invariant-polynomial — build
   it from `dm.*` primitives, optionally packaged with {func}`dm.udf
   <discopt.modeling.udf>`. This is the only route that certifies efficiently
   (47 nodes / 1.6 s on a 3-coordinate problem), and `dm.udf` is exactly as
   fast as inline primitives because it is a documented pass-through.

2. **A trained network via {mod}`discopt.nn`.** Fitting two tanh networks to
   the two surfaces and embedding them with
   {func}`~discopt.nn.predictor.add_predictor` models the problem correctly and
   returns a good geometry, but does **not** currently certify: two 12×12 nets
   over three coordinates gave a dual bound of −20.6 against an optimum near
   1.6 after 601 s. Useful for the answer, not for the proof.

3. **`solver="direct"` on a genuinely opaque body.** Wrap the
   electronic-structure call with {func}`dm.custom <discopt.modeling.custom>`
   and use the derivative-free global search (see
   [direct_global](direct_global.md)). Fast and good in practice — 1.2 s to
   within 1.4 × 10⁻⁴ of the certified optimum on the model above — with no
   bound, honestly reported as `gap_certified=False`.

Note what a surrogate certificate does and does not mean: it certifies the
global optimum *of the fit*. Turning that into a statement about the true
surface needs a validated error bound `|E_true − E_surrogate| ≤ ε`, so that
optimizing the surrogate ± ε brackets the true MECP. discopt has rigorous
remainder machinery for factorable functions
({mod}`discopt._relax.taylor_model`, {mod}`discopt._relax.chebyshev_model`) but
nothing that produces such a bound for a data-fitted surrogate. That gap is
open.

## Summary

| If you have | Use | Certified? |
|---|---|---|
| Two spin-different surfaces, algebraic | `min W₁ s.t. W₁ − W₂ = 0` | yes, cheaply |
| Same-spin, diabatic representation available | `min (W₁₁+W₂₂)/2 s.t. W₁₁−W₂₂ = 0, W₁₂ = 0` | yes, cheapest |
| Same-spin, adiabatic energies only | `min T/2 s.t. T² − 4D ≤ 0` | yes, more expensive |
| A numerical tolerance on the gap | `min W₁ s.t. \|W₁−W₂\| ≤ tol` | yes |
| A trained-network surrogate | {func}`~discopt.nn.predictor.add_predictor` | no — good answer, no proof |
| A true black box | `solver="direct"` on the penalty objective | no, by design |
| — | the penalty objective in branch-and-bound | **no — do not** |
| — | free Cartesian coordinates | **no — do not** |

The measurements behind every number on this page, and the gaps that remain,
are in `docs/dev/mecp-readiness-2026-08-22.md`; the probe scripts are in
`scratchpad/mecp/`.
