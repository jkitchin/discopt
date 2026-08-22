# MECP / MECI optimization: problem class and discopt readiness

**Date**: 2026-08-22
**Status**: research assessment. No solver behaviour was changed.
**Probes**: `scratchpad/mecp/` (`mecp_models.py`, `exp1_formulations.py`,
`exp2_scaling_and_local.py`, `exp3_surrogate_mecp.py`, `exp4_coordinates.py`,
`exp5_charpoly.py`, `exp3b_reduced_space.py`, `exp3c_bound_setting.py`). Every
script ends with an
executed-check count and exits non-zero if it checked nothing (per CLAUDE.md
§6). Model surfaces are written once against an injected `exp`, so the numpy
oracle and the discopt expression tree cannot diverge.

---

## 0. Summary

A minimum-energy crossing point (MECP) is the lowest-energy geometry at which
two electronic potential energy surfaces are degenerate. In the standard
formulation it is a **small, dense, continuous, nonconvex NLP with one or two
nonlinear equality constraints** — a shape discopt handles natively. The
measured result is that discopt already solves this problem class to a
**certified global optimum**. No established MECP method returns a bound:
gradient projection, penalty, branching-plane updating and Lagrange–Newton are
all local, and the state of the art for finding *multiple* crossing points is
stochastic exploration (multistate metadynamics {cite:p}`Lindner2019`).

The three findings that matter:

1. **The formulation choice dominates everything.** The same physical problem
   is either trivial or hopeless for discopt depending on how it is written.
   The constrained forms (diabatic, gap-band, direct difference) certify in
   0–47 nodes. The *penalty* objective that the field actually uses
   {cite:p}`Levine2008` does not certify at all — it timed out at a relative
   gap of 3.4 × 10³ on a 2-variable problem. The characteristic-polynomial
   form sits in between: cheap on a conical intersection, but 600× worse than
   the difference form on a spin crossing (§3.5).
2. **Certified global search buys a real, quantifiable answer.** On a
   two-basin spin-crossing model, discopt certifies the global MECP while the
   two standard local algorithms land in the wrong basin from most starting
   geometries. Since the MECP energy enters a nonadiabatic-TST rate constant
   exponentially {cite:p}`Lykhin2016`, that basin error is a rate error of
   several orders of magnitude.
3. **The ceiling is dimensional, and it is the expected one.** Certified
   global solves are comfortable up to ~8 internal coordinates and cost
   roughly ×2.7 per added coordinate. That covers triatomics through
   ~5-atom molecules exactly, and larger systems only in a reduced
   active-coordinate subspace.

The gap between this and a usable tool for chemists is not the solver core.
It is (a) that real energies are black boxes, so a surrogate is mandatory and
its error is currently uncertified, and (b) missing surface: no coordinate
handling, no symmetry handling, no MECP-shaped API.

---

## 1. The problem class

### 1.1 Definition

Let `E_1(x)` and `E_2(x)` be two adiabatic electronic energies as functions of
nuclear geometry `x`. The crossing seam is `{x : E_1(x) = E_2(x)}` and the
MECP is its lowest point:

```
min  E_1(x)   s.t.   E_1(x) - E_2(x) = 0
```

Two physically distinct cases:

| Case | States | Degeneracy conditions | Seam dimension |
|---|---|---|---|
| **MECP** (spin crossing) | different multiplicity (e.g. singlet/triplet) | `E_1 = E_2` only — the interstate coupling vanishes by spin symmetry | 3N−7 |
| **MECI** (conical intersection) | same multiplicity | `E_1 = E_2` **and** the interstate coupling `H_12 = 0` | 3N−8 |

The MECP case is the easier one and the more common one in thermal chemistry:
because the coupling vanishes, the adiabatic surfaces *are* the diabatic ones,
so `min E_1 s.t. E_1 = E_2` is the exact problem with no reformulation needed.
The MECI case carries a second condition and, crucially, a topological
obstruction (§1.3).

### 1.2 Why the *global* MECP matters

In nonadiabatic transition state theory the MECP plays the role the transition
state plays in ordinary TST: the rate depends on the crossing energy through a
Boltzmann factor {cite:p}`Lykhin2016`. An error of 0.278 in the MECP energy —
exactly the basin error measured in §3.2 — is a factor of
`exp(0.278/0.02585) ≈ 4.6 × 10⁴` in the rate at 300 K if the model's energy
unit is read as eV. A local MECP optimizer
that converges to the second-lowest crossing point does not give a
"slightly worse" answer; it gives a rate that is wrong by four to five orders
of magnitude.

Multiple crossing points between the same pair of states are the norm rather
than the exception, which is precisely why automated *exploration* methods
exist {cite:p}`Lindner2019`. None of them certifies that the lowest one has
been found.

### 1.3 The three structural difficulties

**(a) The surfaces are black boxes.** `E_1`, `E_2` come from an
electronic-structure program. There is no algebraic expression to relax, so
deterministic global optimization is impossible *on the true surfaces*. Any
certified result is necessarily a statement about a surrogate.

**(b) The adiabatic surfaces are non-differentiable exactly at the answer.**
Near a conical intersection the two adiabats are the eigenvalues of a 2×2
matrix,

```
E_± = (W_11 + W_22)/2 ± sqrt( ((W_11 - W_22)/2)² + W_12² )
```

so the radical vanishes at the solution and `E_±` are not differentiable to
any order there. This is the reason the field's algorithms are built the way
they are: gradient projection {cite:p}`Bearpark1994` and the penalty method
{cite:p}`Levine2008` both exist to avoid differentiating through the cone.
The modern answer is to fit *smooth* surrogate quantities instead: a
pseudodiabatic three-surface model {cite:p}`Galvan2023`, or the coefficients of
the characteristic polynomial of the potential matrix {cite:p}`Richings2023`.
That second idea turns out to be exactly the reformulation a factorable global
solver wants (§2.3).

**(c) Redundant coordinates.** Production codes work in 3N Cartesians and
project out 6 translation/rotation modes. For a local optimizer that is
harmless. For branch-and-bound it means the set of global optima is a
6-manifold, so no box ever isolates a unique optimum (§3.4).

### 1.4 Established algorithms and what they optimize

| Method | Formulation | Needs coupling vector? |
|---|---|---|
| Lagrange–Newton {cite:p}`Yarkony1993` | KKT of `min E_1 s.t. E_1-E_2=0` (+ `H_12=0`) | yes |
| Gradient projection {cite:p}`Bearpark1994` | composite gradient: gap-reduction + projected `∇E` | yes |
| Effective gradient {cite:p}`Harvey1998` | as above, spin-crossing specialization | no (spin crossing) |
| Smooth penalty {cite:p}`Levine2008` | `min (E_1+E_2)/2 + σ·ΔE²/(ΔE+α)`, σ escalated | no |
| Branching-plane update {cite:p}`Maeda2010` | replaces the coupling vector with an updated second direction | no |
| Multistate metadynamics {cite:p}`Lindner2019` | biased dynamics along a gap collective variable | no |

Every row is local except the last, which is stochastic. **Nothing in this
list returns a bound.**

---

## 2. Formulation map: what is inside discopt's expression class

discopt relaxes *factorable* expressions — arithmetic plus a fixed intrinsic
set (`exp`, `log`, `sqrt`, powers, trigonometric and hyperbolic functions,
`abs`, `min`/`max`) — over a box, and branches spatially. The question is
therefore not "can discopt do MECP" but "which way of writing MECP lands in
that class with a useful relaxation".

### 2.1 Diabatic form (smooth, no radical)

For a spin crossing (`W_12 ≡ 0`) the problem is already smooth:

```
min W_1(x)   s.t.   W_1(x) - W_2(x) = 0
```

For a conical intersection, working in a diabatic representation removes the
radical entirely, because degeneracy of a 2×2 symmetric matrix is
`W_11 = W_22` **and** `W_12 = 0`:

```
min (W_11 + W_22)/2   s.t.   W_11 - W_22 = 0,   W_12 = 0
```

Both are polynomial/exponential in the coordinates and fully factorable.

### 2.2 Adiabatic form (radical vanishing at the solution)

```
min E_-(x)   s.t.   E_+(x) - E_-(x) = 0
```

Expressible (discopt has `sqrt`), but the relaxation is weaker: see §2.5 for
the measured cost and §4.2 for the mechanism.

### 2.3 Characteristic-polynomial form — the representation-free bridge

The coefficients of the characteristic polynomial of the potential matrix are
the symmetric functions of its eigenvalues, and unlike the eigenvalues
themselves they are smooth through the intersection {cite:p}`Richings2023`:

```
T(x) = E_1 + E_2 = W_11 + W_22            (trace)
D(x) = E_1 · E_2 = W_11·W_22 - W_12²      (determinant)
```

Degeneracy is the vanishing of the discriminant, and at degeneracy
`E_1 = E_2 = T/2`, giving

```
min T(x)/2   s.t.   T(x)² - 4 D(x) = 0
```

This is smooth, polynomial in `(T, D)`, needs **no diabatization**, and is
exactly the pair of quantities the ML-seam literature already fits. It also
generalizes to more states via the discriminant of the characteristic
polynomial.

It has one property worth stating plainly, because it cuts *for* the global
approach: `T² - 4D = (E_1-E_2)² ≥ 0` identically, so
`∇(T²-4D) = 2(E_1-E_2)∇(E_1-E_2)` **vanishes at every feasible point**. LICQ
fails on the entire feasible set, so Newton-type local methods lose their
convergence theory on this formulation. A relaxation-based solver does not need
LICQ — it needs valid bounds. Measured consequences in §3.5.

Because the discriminant is nonnegative by construction, the equality may be
written as an inequality `T² - 4D ≤ 0`, which is the better-conditioned form.

### 2.4 Penalty form (what the field actually minimizes)

```
min (W_1+W_2)/2 + σ · ΔE² / (|ΔE| + α)
```

Expressible — discopt has `abs` and division — and box-constrained only. But
as an object to *relax* it is close to the worst case: a squared numerator over
a denominator whose lower bound is the small smoothing parameter `α`. Measured
in §3.1: it does not certify.

### 2.5 Measured comparison (LVC conical intersection, analytic oracle)

Two-state, two-mode linear vibronic coupling model {cite:p}`Koppel1984`, whose
MECI is analytic: `q_t = 1`, `q_c = 0`, `E = 0.13`.
(`scratchpad/mecp/exp1_formulations.py`)

| Formulation | status | objective | certified | nodes | wall |
|---|---|---|---|---|---|
| diabatic, two equalities | optimal | 0.130000 | **yes** | **0** | 0.03 s |
| adiabatic, `E_+-E_- = 0` | optimal | 0.129998 | yes | 1 | 1.53 s |
| adiabatic, `E_+-E_- ≤ 10⁻⁴` | optimal | 0.129948 | yes | 3 | 0.87 s |

All three land on the analytic answer with a certified bound. The diabatic form
is free — the convexity detector disposes of it at the root with no branching —
while the radical costs a 50× wall-clock factor for the same answer.

---

## 3. Measured readiness

Models used (all in `scratchpad/mecp/mecp_models.py`, each written once
against an injected `exp` so the numpy oracle and the discopt expression tree
cannot diverge):

* **LVC** — linear vibronic coupling {cite:p}`Koppel1984`, analytic MECI.
* **Morse** — two sums of Morse oscillators in bond-length coordinates with
  different depths and equilibrium lengths: a spin-crossing model.
* **TwoWell** — the same, but with a double-well (inversion/torsion) coordinate
  on the lower state, so the lower surface has two conformer wells and the seam
  has two energetically distinct low-lying basins. A tilt term identical in
  both states breaks the degeneracy between them without touching the seam.

### 3.1 Correctness against independent oracles

| Instance | oracle | oracle E | discopt E | certified bound | nodes |
|---|---|---|---|---|---|
| LVC MECI (diabatic) | analytic | 0.130000 | 0.130000 | 0.130000 | 0 |
| Morse MECP, n=2 | grid + bisection (2904 crossings) | 0.4734154 | 0.4734150 | 0.4734150 | 5 |
| TwoWell MECP, n=2 | grid + bisection (2642 crossings) | 1.2706849 | 1.2706840 | 1.2706760 | 5 |

No certified bound anywhere in the campaign exceeded a known feasible seam
point — the soundness check every probe applies.

The penalty formulation (§2.4) is the one failure, and it is a formulation
failure rather than a solver failure:

| σ, α | status | objective | dual bound | rel. gap | nodes |
|---|---|---|---|---|---|
| 3.5, 0.025 | feasible (300 s limit) | 3.3125 | −11177.6 | 3.4 × 10³ | 571 |
| 20, 0.005 | feasible (300 s limit) | 10.5175 | −212877.3 | 2.0 × 10⁴ | 735 |

The relaxation of `ΔE²/(|ΔE|+α)` over a box where the denominator can approach
`α` is enormous, so the dual bound is useless. **Recommendation: never hand a
global solver the penalty form; use a constrained form.**

### 3.2 Global versus local (the headline result)

TwoWell, n=2. Grid oracle: global MECP at `E = 1.270685`; second basin at
`E = 1.548537` (0.278 higher). discopt returns the global basin, certified,
in 5 nodes and 0.5 s.

Local baselines on the identical model, 200 uniformly random starting
geometries each (`exp2_scaling_and_local.py` Part B):

| Method | reached the seam | found the **global** basin | distinct energies converged to |
|---|---|---|---|
| Levine/Coe/Martinez penalty (σ escalation 0.5→128, L-BFGS-B) | 200/200 | **127/200 (63.5 %)** | 2 |
| Direct constrained (SLSQP on `min W_1 s.t. W_1-W_2=0`) | 199/200 | **78/199 (39.2 %)** | 3 |

Both local methods reliably *reach* the seam; neither reliably reaches the
lowest point of it. A local MECP search on this model reports a crossing point
0.278 too high in 36.5 % (penalty) to 60.8 % (SLSQP) of runs. SLSQP also
converged to a third, still higher basin (2.2198).

At 300 K that basin error is a nonadiabatic-TST rate error of
`exp(0.278/0.02585) ≈ 4.6 × 10⁴` if the model's energy unit is read as eV.
discopt returns the correct basin with a proof, every time, in 0.5 s.

### 3.3 Dimension scaling

TwoWell spin-crossing MECP, exact equality form, 300 s limit per instance.
`n` is the number of internal coordinates; a molecule with N atoms has 3N−6.

| n | 3N−6 ⇒ atoms | status | objective | dual bound | rel. gap | certified | nodes | wall |
|---|---|---|---|---|---|---|---|---|
| 2 | — | optimal | 1.270684 | 1.270676 | 6.8e−06 | **yes** | 5 | 0.54 s |
| 3 | triatomic | optimal | 1.558620 | 1.558547 | 4.7e−05 | **yes** | 47 | 1.48 s |
| 4 | — | optimal | 1.814815 | 1.814639 | 9.7e−05 | **yes** | 231 | 5.63 s |
| 5 | — | optimal | 2.161454 | 2.161265 | 8.7e−05 | **yes** | 479 | 14.56 s |
| 6 | 4 atoms | optimal | 2.471629 | 2.471411 | 8.8e−05 | **yes** | 1113 | 43.04 s |
| 7 | — | optimal | 2.828308 | 2.828029 | 9.8e−05 | **yes** | 3289 | 73.30 s |
| 8 | — | optimal | 3.157964 | 3.157651 | 9.9e−05 | **yes** | 7709 | 193.98 s |
| 9 | 5 atoms | feasible | 3.515116 | 3.494239 | 5.9e−03 | no (300 s cap) | 13585 | 300.02 s |

Cost grows by roughly ×2.7 per added coordinate (nodes ×2.3–4.6). The n=9
instance was *close* — 0.6 % relative gap when the clock ran out — so the
practical certified ceiling at a 5-minute budget is 8–9 internal coordinates,
and n=9–11 is reachable with a longer budget.

Two secondary observations from the same sweep:

* discopt's incumbent beat the stochastic seam oracle by a margin that grows
  with dimension (−0.006 at n=3, −1.19 at n=8). Random sampling plus Newton
  projection degrades exactly as one expects in higher dimension; the
  deterministic search does not. This is the same failure mode a chemist would
  hit with a stochastic seam explorer.
* No soundness violation anywhere: no certified bound ever exceeded a known
  feasible seam point, across all 9 instances.

**Reading this for real molecules.** 3N−6 means a triatomic is n=3 and a
6-atom molecule is n=12. So certified global MECP is available *outright* for
triatomics through roughly 5-atom systems, and for larger molecules only in a
reduced set of active coordinates with the rest frozen — which is how MECP
scans on large systems are done anyway, but the certificate then applies to
the reduced subspace, not the full geometry. That limitation must be stated
whenever the word "global" is used.

### 3.4 Coordinate systems

A triatomic MECP with two Morse-sum states, written three ways over the same
energy function (`exp4_coordinates.py`, 180 s limit):

| Coordinates | free vars | status | objective | dual bound | certified | nodes | wall |
|---|---|---|---|---|---|---|---|
| A: internal (3 bond distances) | 3 | optimal | 0.819996 | 0.819917 | **yes** | 5 | 0.96 s |
| B: free Cartesian (9, 6 redundant) | 9 | feasible | 0.819996 | **−0.109858** | **no** (93 % gap) | 341 | 180 s |
| C: gauge-fixed Cartesian | 3 | optimal | 0.819996 | 0.819964 | **yes** | 329 | 14.62 s |

All three recover the identical geometry (bonds 1.32447, 1.33159, 1.34105) and
the identical energy, so this is purely about certifiability.

**Free Cartesian coordinates cannot be certified, and this is structural, not a
solver defect.** Every translation and rotation of the MECP is also an MECP, so
the optimal set is a 6-dimensional manifold; no finite subdivision isolates an
optimum, and the dual bound stalls (−0.11 against a true optimum of 0.82). The
gauge-fixed variant — atom 0 at the origin, atom 1 on +x, atom 2 in the xy
half-plane, the standard 3N−6 gauge — recovers certifiability. It costs 66× the
nodes of the internal-coordinate form because the energies then reach the
coordinates through `sqrt(Σ(Δx)²)` compositions rather than directly.

**Requirement, to be documented for users: pose MECP problems in internal
coordinates (bond distances/angles) or in a gauge-fixed Cartesian frame. Never
in free Cartesians.**

### 3.5 Characteristic-polynomial form and the LICQ failure

(`exp5_charpoly.py`) On the LVC conical intersection with its analytic answer:

| Form | status | objective | dual bound | certified | nodes | wall |
|---|---|---|---|---|---|---|
| CP equality `T²−4D = 0` | optimal | 0.130000 | 0.129918 | **yes** | 7 | 2.48 s |
| CP inequality `T²−4D ≤ 0` | optimal | 0.130000 | 0.129918 | **yes** | 7 | 0.85 s |

Both hit the analytic optimum with a sound bound, and the returned point is
genuinely degenerate (`E₊−E₋ = 4.4e−07`). The inequality form is 3× faster for
the same tree, and is the one to prefer.

**The LICQ prediction was only half right, and the correction matters.** The
claim in §2.3 — that `∇(T²−4D)` vanishes on the whole feasible set, so
Newton-type methods lose their theory — is mathematically correct, and SciPy
reports the symptom directly (`delta_grad == 0.0` warnings from
`trust-constr`). But the consequence is algorithm-dependent, not universal:

| Local method on the CP form | reached the seam | median energy error there |
|---|---|---|
| SLSQP | **32/60 (53 %)** | 2.5e−06 |
| trust-constr | **60/60 (100 %)** | 2.7e−10 |

SLSQP fails on nearly half its starts; trust-constr copes with the degeneracy
entirely. So "the CP form defeats local solvers" is **false as stated** and is
retracted here: it defeats *some* of them. The honest version is that the CP
form is safe for a global relaxation-based solver, hostile to SLSQP, and fine
for a well-implemented interior/trust-region method.

**The CP form is not free, and on a spin crossing it is a bad trade.** On the
two-basin exponential model, where the direct difference `W_1−W_2 = 0`
certified in 5 nodes / 0.5 s, the CP form did **not** certify in 300 s:

| Form, TwoWell n=2 | status | objective | dual bound | rel. gap | nodes | wall |
|---|---|---|---|---|---|---|
| direct difference `W_1−W_2 = 0` | optimal | 1.270684 | 1.270676 | 6.8e−06 | 5 | 0.50 s |
| CP inequality `T²−4D ≤ 0` | feasible | 1.270654 | 1.257837 | 1.0e−02 | 3353 | 300 s |

Squaring the surfaces (`T²`, `W_1·W_2`) widens the McCormick relaxation
enormously compared with a linear difference of the same two surfaces.

**Guidance: use the difference form whenever the interstate coupling vanishes
(any spin crossing). Reach for the CP form only for a same-spin conical
intersection, where the alternative is the radical or an explicit
diabatization.**

### 3.6 Surrogate routes (the realistic ab-initio workflow)

The true surfaces are black boxes, so a certified answer is necessarily a
statement about a surrogate. Four routes exist; they differ enormously.
(`exp3_surrogate_mecp.py`, `exp3b_reduced_space.py`; TwoWell n=3, whose
certified reference optimum is 1.558620.)

| Route | how the surfaces are written | status | dual bound | certified | nodes | wall |
|---|---|---|---|---|---|---|
| symbolic (`dm.exp`) | factorable expression | optimal | 1.558547 | **yes** | 47 | 1.97 s |
| `dm.udf` | symbolic body, `dm.*` primitives | optimal | 1.558547 | **yes** | 47 | 1.60 s |
| `dm.custom` + MCBox-dispatching `exp` | opaque, reduced-space relaxation | feasible | 1.446551 | no (7.2 % gap) | 299 | 300 s |
| `dm.custom` + raw `jnp.exp` | opaque, local NLP only | feasible | — | no (no bound) | 0 | 0.70 s |
| `solver="direct"` | fully opaque, derivative-free | feasible | — | no (by design) | — | 1.2 s |
| `discopt.nn` (two tanh 12×12 nets) | trained surrogate as constraints | feasible | −20.56 | no (gap 13.7) | 10171 | 601 s |

All routes agree on the optimum (1.558620) — a useful cross-validation — but
only the symbolic ones certify.

Four things follow.

1. **Write the surfaces symbolically.** `dm.udf` is exactly as fast as inline
   `dm.exp` (it is a documented pass-through) and is the right way to package a
   PES for reuse. The 150× penalty for going opaque is not worth paying when
   the surrogate's functional form is known — and for Morse/polynomial/Gaussian
   PES fits it always is.
2. **The reduced-space route engages but does not close the gap.** A first
   attempt reported "no certificate" for both `dm.custom` variants; that probe
   was wrong — it passed the whole coordinate vector as one argument, and a
   non-scalar leaf is a documented disqualifier, so the MCBox path was never
   exercised. Retested with scalar arguments it does engage (299 nodes, a real
   bound), but it announces `McCormick 'nlp' objective bound is not a valid
   dual bound for nonconvex models (issue #120); falling back to the alphaBB
   underestimator` — and alphaBB is too weak here to certify in 300 s.
3. **`solver="direct"` is the right tool for a genuinely black-box PES**, and it
   performed well: 1.2 s to a point 1.4e−04 off the certified optimum, on the
   penalty objective (DIRECT needs an unconstrained form, which is the one
   place the penalty formulation is the correct choice). It returns no bound
   and correctly reports `gap_certified=False`.
4. **The `discopt.nn` route models the problem correctly but does not certify
   it.** Two tanh networks (12, 12) fitted to RMSE 0.116/0.095 over 3
   coordinates gave a dual bound of −20.6 against an optimum near 1.6 after
   601 s. The geometry it returned was good — 1.570 against a true MECP of
   1.559, i.e. the surrogate's *optimum* was ~10× more accurate than the
   surrogate's RMSE — so the modeling path is sound and the answer is useful,
   but the certificate is not available at this network size. **This is the one
   place where genuine performance work would change what is possible.**

Note what the certificate means on a surrogate even when it is obtained: it
certifies the global optimum *of the fit*, not of the true surface. Closing
that gap rigorously needs a validated error bound `|E_true − E_surrogate| ≤ ε`,
which would let `min` over the surrogate ±ε bracket the true MECP. discopt has
rigorous-remainder machinery for *factorable* functions
(`_relax/taylor_model.py`, `_relax/chebyshev_model.py`) but nothing that
produces such a bound for a data-fitted surrogate. That is a research gap, not
an implementation gap.

---

## 4. Gaps

Ordered by whether they block use.

### 4.1 Not gaps — things that already work

* Nonconvex equality constraints on exponential/polynomial surfaces: certified,
  0–47 nodes on every well-posed formulation tested.
* The full intrinsic set an MECP formulation needs (`exp`, `sqrt`, powers,
  `abs`, division, trigonometric functions) is present in the modeling API, the
  Rust IR, and the relaxation layer.
* Soundness: zero violations in this campaign. Every certified bound stayed at
  or below an independently established feasible seam point, on 9 scaling
  instances plus every formulation variant.
* Black-box fallback: `solver="direct"` exists, works, and honestly reports
  `gap_certified=False`.

### 4.2 `sqrt`'s endpoint tangent is dropped at zero — general, small, real

`_UNIVARIATE_FN` in `_relax/uniform_relax.py:354` carries `sqrt`'s derivative as
`lambda t: 0.5/np.sqrt(t)`, which is `+inf` at `t = 0`. `_emit_1d`'s
`_tangent_row` checks `_finite(g, gp)` and returns without emitting, so the
result is **sound** — the facet is simply lost and the relaxation is looser.
The reason it matters here is that for a conical intersection posed
adiabatically, `t = 0` *is* the solution: the radical vanishes exactly at the
MECI. The measured cost was a 50× wall-clock factor against the diabatic form
for the same answer (§2.5), and a `RuntimeWarning: divide by zero` on every such
solve.

The same shape affects `asin`/`acos` at ±1, `acosh` at 1, and `log` at 0. The
fix is general: when the endpoint derivative is non-finite, place the tangent at
an interior point instead of dropping the facet. This is a bound-changing
change, so it needs the flag plus corpus-wide differential panel treatment from
CLAUDE.md §5 — not a quick patch.

### 4.3 Certified relaxation of trained-network surrogates — the real
performance item

§3.6 measured a dual bound of −20.6 against an optimum near 1.6 for two
12×12 tanh networks over a 3-dimensional box, after 601 s. Any workflow that
learns a PES from ab-initio data and then wants a *certified* MECP runs into
this. It is the same problem the `nn` module faces generally, so a fix would
not be MECP-specific.

### 4.4 Reduced-space bound falls back to alphaBB, and the suggested remedy is
inert

The `dm.custom` MCBox path reports `McCormick 'nlp' objective bound is not a
valid dual bound for nonconvex models (issue #120); falling back to the alphaBB
underestimator. Use mccormick_bounds='lp' for a valid spatial relaxation on
models with continuous variables.` and alphaBB was too weak to certify a
3-coordinate MECP in 300 s where the factorable form took 2 s. Since the
documented selling point of that path is DOF-only branching — exactly what MECP
wants as molecules grow — this matters.

**Passing the setting the message recommends does not help**
(`exp3c_bound_setting.py`):

| Route | `mccormick_bounds` | dual bound | rel. gap | nodes | certified |
|---|---|---|---|---|---|
| `dm.custom` MCBox | `auto` | 1.468451 | 5.8e−02 | 293 | no |
| `dm.custom` MCBox | `lp` | 1.446551 | 7.2e−02 | 297 | no |
| `discopt.nn` 12×12 | `auto` | −20.596426 | 1.4e+01 | 4833 | no |
| `discopt.nn` 12×12 | `lp` | −20.596426 | 1.4e+01 | 5041 | no |

The `issue #120` message **reprints verbatim even when `lp` is passed**, the
bound is unchanged on the `nn` path and slightly *worse* on the CustomCall
path. So on these two paths the documented remediation is inert. That is a
narrower and more actionable finding than "alphaBB is weak": either the setting
is not reaching the relaxation builder on the CustomCall/`nn` paths, or the
message is advising a setting that does not apply there. Worth checking against
issue #120 before any tightness work.

A single-hidden-layer net (5 units) gave a far less negative bound (1.4909 vs
−20.60), so bound quality does scale strongly with network width — but that fit
was too poor (train RMSE 1.25 on `W_1`) for its optimum to be meaningful, so it
sizes the tightness effect without being a fair optimization comparison.

### 4.5 No symmetry handling

MECP problems on molecules with equivalent atoms have permutationally
equivalent crossing points, and a seam basin appears once per equivalent
labelling. `grep` finds symmetry detection only in `llm/reformulation.py`
(`_detect_symmetry`, an advisory); the solver core has no orbit/symmetry
reduction. For a symmetric molecule the B&B tree therefore carries every
symmetric copy of the optimum. Not measured here — the models were deliberately
symmetry-broken by the tilt term — but it should be measured before claiming
readiness on real molecules.

### 4.6 Missing surface (the cheap, high-value work)

* **No coordinate handling.** No `.xyz` reader, no internal-coordinate
  (Z-matrix / redundant internal) construction, no gauge fixing. Given §3.4,
  users *must* get this right or they get no certificate, and right now they
  must do it by hand.
* **No MECP-shaped API.** Every experiment here hand-built
  `min W_1 s.t. W_1 - W_2 = 0`. A thin helper — take two surface callables plus
  a coordinate box, emit the right formulation, warn on free Cartesians, refuse
  the penalty form — would encode all the guidance in this document.
* **`mcbox` has no trigonometric intrinsics.** It carries `exp`, `log`, `sqrt`,
  powers, `softplus`, `abs`, `tanh`, `atan`, `sigmoid`, `sinh` — but not `cos`,
  `sin`, or `acos`. ANI-style ML potentials {cite:p}`Smith2017` are dense
  feedforward networks on top of atom-centered descriptors built from Gaussians
  *and angular terms in `cos θ`*, so such a model is expressible in the
  full-space factorable path but cannot go through the reduced-space path. Worth
  noting if the DOF-only branching route is ever pursued for ML potentials.
* **No chemistry in the bibliography before this change.** `docs/references.bib`
  had 220 entries, none on conical intersections or crossing points; the
  Crucible knowledge base likewise. Eleven entries were added here.

### 4.7 Recommended next steps, in order

1. **A documentation notebook** (`docs/notebooks/`) carrying the formulation
   guidance from §2 and §3: the difference form for spin crossings, the
   diabatic or CP form for conical intersections, internal or gauge-fixed
   coordinates, and an explicit "do not hand the solver the penalty objective".
   That single page prevents the two 10⁴-scale mistakes a chemist would
   otherwise make. *Not written in this change: CLAUDE.md requires a
   zero-warning `jupyter-book build` for a new notebook, and jupyter-book is
   not installed in the container this work was done in.*
2. **The `sqrt` endpoint-tangent fix** (§4.2), with the flag + panel discipline.
3. **A symmetry measurement** on a permutation-symmetric MECP model, to size
   §4.5 before deciding whether it needs work.
4. **Coordinate helpers** (§4.6) if xyz input is wanted.
5. **Certified NN-surrogate relaxations** (§4.3) — the largest piece, and the
   one that would unlock certified MECP on learned ab-initio surfaces.

---

## 5. References

Bibliography entries added to `docs/references.bib` in this change:
`Yarkony1993`, `Bearpark1994`, `Harvey1998`, `Levine2008`, `Maeda2010`,
`Koppel1984`, `Lindner2019`, `Galvan2023`, `Richings2023`, `Lykhin2016`,
`Smith2017`.
