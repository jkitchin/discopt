# DAE Module Review — Correctness, Thoroughness, Performance, and SOTA Assessment

**Date:** 2026-07-03
**Scope:** `python/discopt/dae/` (`polynomials.py`, `collocation.py`, `finite_difference.py`,
`mol.py`, `__init__.py`), tests (`python/tests/test_dae.py`, `test_dae_fd_mol.py`),
and `docs/notebooks/tutorial_dae.ipynb`.
**Method:** Full read of all ~2,200 module lines and ~2,000 test lines; every suspected
defect was then *reproduced numerically* (standalone math checks plus end-to-end solves
against the installed package). The existing fast suite (82 tests, `-m "not slow"`) passes
on `main` — all findings below are latent, not regressions.

This document is both the review record and the executable plan (§6) for fixing what was
found. Findings marked **[CONFIRMED]** were reproduced with a failing numerical experiment;
findings marked **[BY INSPECTION]** are unambiguous from control flow.

---

## 1. Summary of findings

| # | Severity | Component | Finding |
|---|----------|-----------|---------|
| C1 | **P0 correctness** | `mol.py` | Left Neumann BC has a **sign error**; any nonzero-flux left BC yields a wrong PDE solution [CONFIRMED] |
| C2 | **P0 correctness** | `collocation.py` | `integral()` is wrong by **exactly 2×** for Radau `ncp=1` [CONFIRMED] |
| C3 | **P0 correctness** | `collocation.py`, `finite_difference.py` | Two silent no-dynamics paths: (a) `add_second_order_state` without an RHS, (b) RHS dict key typo — both build a model with **no dynamics constraints**, no error [CONFIRMED] |
| C4 | P1 | `mol.py` | `npts=1` crashes with an opaque `ValueError` [CONFIRMED] |
| C5 | P1 | `finite_difference.py` | Forward Euler + algebraic DAE: off-by-one index alignment between `z_k` and the grid point where the ODE is evaluated [BY INSPECTION] |
| R1–R7 | P1–P2 | all | Missing validation and API guards (§3) |
| A1–A4 | P2 | `finite_difference.py`, `mol.py` | API gaps: no `FDBuilder.integral()` (promised by `MOLBuilder.time_builder` docstring), no Robin BCs, first-order Neumann treatment, no FD interpolating `least_squares` (§3) |
| P1–P4 | P2 perf | `collocation.py`, `finite_difference.py` | Scalar Python-loop expression construction in `integral()`/`least_squares()`/all of `FDBuilder`; scipy `quad` in a loop per `integral()` call (§4) |
| T1–T8 | tests | — | Coverage gaps that let C1–C5 through (§5) |

What was checked and found **correct** (no action needed):

- Hardcoded Radau IIA roots (`ncp=1..5`): verified via quadrature exactness to degree
  `2·ncp−2`; max error ≤ 2.2e-16. Machine precision.
- `_lagrange_deriv` / `_lagrange_deriv2`: verified against numerical differentiation.
- The vectorized collocation constraint assembly (`x @ Aᵀ = h·f`), Radau and Legendre
  continuity constraints, non-uniform elements, `state_at` interpolation, and the
  interpolating `least_squares` path are all correct and well-tested.
- `time_points()` float-dedup via `np.unique`: no failures across 162 configurations,
  though it is theoretically fragile (see R6).

---

## 2. Confirmed correctness bugs (P0)

### C1. Left Neumann boundary condition sign error — `mol.py:433`

`_get_left_value` reconstructs the boundary value for the leftmost interior point as:

```python
# bc_val is the OUTWARD normal derivative = -du/dz at the left boundary
return u[0] - dz * bc_val        # WRONG
```

First-order Taylor from the first interior point: `u(z0) ≈ u[0] − dz·(du/dz)`. With the
documented convention `bc_val = du/dn = −du/dz`, that is `u(z0) = u[0] + dz·bc_val`.
The sign is flipped. The right boundary (`_get_right_value`) is correct, which makes the
asymmetry easy to verify.

**Reproduction.** For `u(z) = z` (so `du/dz = 1`, left `bc_val = −1`), the code
reconstructs the boundary value as `0.333` instead of `0.0`, giving `∂²u/∂z² = 12`
where the true value is 0 and `∂u/∂z = 0` where the true value is 1. End-to-end: heat
equation `u_t = u_zz`, left Neumann `du/dn = −1`, right Dirichlet `u(1)=1`, initial
`u = z` — the exact solution is the steady state `u(z,t) = z`, but the solve drifts to
max error **0.81**.

Why tests missed it: every Neumann test in the suite uses flux **0.0**, for which the
sign is invisible.

**Fix:** `return u[0] + dz * bc_val`. Regression test: the steady-state-preservation
experiment above (asserts max error < discretization tolerance), plus a mirrored
right-boundary test to lock the convention on both sides.

### C2. `integral()` under-integrates by 2× for Radau `ncp=1` — `collocation.py:882-908`

`_quadrature_weights` builds weights by integrating the Lagrange basis over the **full
node set** `[0, τ₁, …, τ_ncp]` but then drops the weight of node 0. That is only valid
when the node-0 weight is zero, which holds iff the Radau rule on the collocation points
alone is exact to degree ≥ ncp, i.e. `2·ncp−2 ≥ ncp` ⇔ **ncp ≥ 2**. For `ncp=1` the
dropped node-0 weight is 0.5:

```
radau ncp=1: weights=[0.5]  sum=0.5   # should sum to 1.0
radau ncp=2..5: sums = 1.0            # node-0 weight happens to be 0
```

**Reproduction (end-to-end):** `dae.integral(lambda t,s,a,c: 1.0)` over `[0, 3]` with
`ncp=1` returns **1.5**. Any Lagrange/integral-cost optimal-control problem run at
`ncp=1` silently optimizes half the true cost.

Why tests missed it: `TestIntegral` only exercises `ncp=3`.

**Fix:** compute the Radau quadrature weights on the **collocation points only** (an
`ncp×ncp` Vandermonde solve against moments `1/(k+1)` — exact, no `scipy.integrate.quad`
needed, and correct for all ncp including 1). Regression tests: `sum(w) == 1` and
`∫t^k` exactness up to degree `2·ncp−2` for ncp = 1..5, plus the end-to-end
integral-of-constant test parameterized over ncp = 1..5 and both schemes.

### C3. Silent no-dynamics models — `collocation.py:471-496`, `finite_difference.py:256-341`

Two distinct paths produce a model with **zero dynamics constraints and no error**:

(a) **Missing RHS with second-order states.** `_add_collocation_constraints` raises
"No ODE RHS set" only when `rhs_fn is None and not self._second_order_info`. If the user
calls `add_second_order_state(...)` but forgets `set_second_order_ode(...)`, the guard
passes (info is non-empty) and the very next line `if rhs_fn is None: return` silently
skips all dynamics. Reproduced: the model contains only 8 continuity constraints, no
collocation equations. Identical logic bug in `FDBuilder.discretize`.

(b) **RHS dict key mismatch.** The constraint loop does
`if sv.name not in derivs: continue` and never checks for unknown keys. A typo
(`{"X": ...}` for state `"x"`) silently drops **all** dynamics; a partially specified
dict silently leaves some states unconstrained (they float freely, coupled only by
continuity). Reproduced: 4 constraints (continuity only) instead of 19.

For a repository whose product is a *certificate*, a silently under-constrained
transcription is the worst failure mode: the solver will happily return a "globally
optimal" answer to the wrong problem.

**Fix (both builders, one shared validator):** after calling the user RHS, require
`set(derivs.keys()) == {sv.name for sv in states}` minus algebraic-only names — raise
`ValueError` naming both the missing and the unknown keys. In case (a), tighten the
guard: `if self._second_order_info and self._second_order_rhs is None: raise`. If
intentionally-undriven states are ever needed, that should be an explicit opt-in
(e.g. `add_state(..., quadrature=False)` style flag), not a silent default —
per the development philosophy, refuse loudly rather than approximate silently.

---

## 3. Robustness and API-completeness issues (P1–P2)

**C4. `MOLBuilder` with `npts=1` crashes** (`ValueError: setting an array element with a
sequence`). Root cause: with one interior point the field maps to a state with
`n_components=1`, so `states[name]` is a single vectorized expression rather than a list,
and `mol.py`'s `u[k]` indexing then means something entirely different. Fix: either
handle `n_components=1` in the MOL RHS wrapper or raise a clear `ValueError` at
construction (`npts >= 2 required`). Given a 1-point interior grid is numerically useless,
a loud refusal is fine.

**C5. Forward Euler + algebraic DAE off-by-one.** Algebraic variables are declared to
live at interior grid points (`z[k] ↔ grid point k+1`; the algebraic residual is enforced
at points 1..nfe). The backward and central branches use `_algebraic_dict_at(k-1)` at
grid point `k` — consistent. The **forward** branch (`finite_difference.py:284-302`) uses
`_algebraic_dict_at(k)` while evaluating the RHS at grid point `k`, i.e. it feeds
`z(t_{k+1})` into `f(t_k, x_k, ·)`. The residual is also never enforced at `t_0`, where
explicit Euler actually needs `z`. Fix options: (i) re-index algebraics to grid points
0..nfe−1 for the forward method, or (ii) refuse `method="forward"` when
`set_algebraic` is used (explicit Euler for DAEs is ill-posed anyway; a loud refusal is
defensible). Either way, add a test.

**R1. `ContinuousSet` has no validation.** `nfe=0` → division by zero downstream;
`bounds=(1, 0)` → negative element widths → nonsense model, silently. Add
`__post_init__` checks: `nfe >= 1`, `bounds[1] > bounds[0]`, `ncp >= 1`,
`scheme in {"radau", "legendre"}` (currently a bad scheme only fails later inside
`collocation_matrix`).

**R2. Mutations after `discretize()` are silently ignored.** `add_state`, `add_control`,
`set_ode`, etc. can be called after `discretize()` and do nothing. Guard each mutator
with `if self._discretized: raise RuntimeError(...)` in both builders (and `add_field` /
`set_pde` / `add_control` in `MOLBuilder`).

**R3. Duplicate variable names are not checked.** `add_state("x")` twice, or a state and
an algebraic sharing a name, silently overwrites entries in `self._vars`. Validate at
declaration time.

**R4. `MOLBuilder._compute_initial` does not validate array length** against `npts` —
a wrong-length array surfaces as a cryptic numpy broadcast error inside `add_state`.
Validate and raise with the expected shape.

**R5. `align_time_grid` silently loses measurements.** If two measurement times snap to
the same boundary, the second overwrites the first; if a snapped boundary collides with
another, `np.unique` silently reduces `nfe`. Both violate the function's contract
("ensures that measurement times fall exactly on element boundaries"). Fix: detect and
either raise or warn with the list of unplaced measurement times; document that `nfe`
may shrink.

**R6. `time_points()` dedup is exact-equality.** `tp[i, ncp]` is computed as
`eb[i] + h_i·1.0` while the next element start is `eb[i+1]`; IEEE does not guarantee
equality. It happened to hold in all 162 configurations tested, but a tolerance-based
dedup (or constructing the last node as `eb[i+1]` directly, which is exact and cheaper)
removes the fragility. Related: `extract_solution` for the **Legendre** scheme skips
`k=0, i>0` nodes as "duplicate element boundary", but for Legendre those are *not*
duplicates (no node at τ=1) — real element-boundary values are dropped and the output
grid disagrees with `time_points()`. Make the skip Radau-only.

**A1. `FDBuilder.integral()` does not exist**, yet `MOLBuilder.time_builder`'s docstring
advertises "Useful for accessing `integral()`, `least_squares()`". Any
`time_method="finite_difference"` user following the docs hits `AttributeError`.
Implement trapezoid (or method-consistent rectangle) quadrature on the FD grid.

**A2. `FDBuilder.least_squares` has no `interpolate` option and there is no
`FDBuilder.state_at`.** After #94 fixed the node-snapping bias for collocation, the FD
path still silently snaps to the nearest grid point. Piecewise-linear interpolation
between grid points is the natural FD analogue and cheap to add.

**A3. Neumann treatment is first-order; no Robin BCs.** The ghost-value reconstruction
`u_b = u[0] ± dz·g` is O(dz) while the interior stencil is O(dz²), degrading global
spatial accuracy to first order whenever a Neumann BC is present. A one-sided
second-order reconstruction (`u_b = (4u[0] − u[1] ∓ 2·dz·g)/3`-style, derived for the
half-offset grid) restores O(dz²). Robin (`a·u + b·du/dn = c`) is a straightforward
generalization and is table stakes in PDE-constrained optimization (it subsumes both
current types).

**A4. API inconsistency in RHS `t` argument.** `set_ode`/`set_algebraic` callbacks
receive `t` as a vectorized `Constant` of shape `(nfe, ncp)`, while `integral()` and the
FD builder pass floats. This works because of operator overloading but is undocumented
and surprising (`np.exp(t)` works in one and not the other). Document it in the
docstrings of `set_ode`/`set_pde`/`integral`, and state that `dm.*` ufuncs must be used.

---

## 4. Performance issues

None of these affect correctness; all are O(model-build-time), not solve-time — but for
large transcriptions (the realistic regime: `nfe`×`ncp`×`n_states` in the thousands)
model build time and DAG size are dominated by exactly these paths.

**P1. `integral()` builds a scalar left-fold chain.** It calls the integrand
`nfe·ncp` times with scalar dicts and folds terms with repeated `+`, producing an
O(n)-deep unbalanced expression tree. The module already vectorizes collocation
constraints (one `MatMulExpression` per state); `integral()` should reuse
`_build_vec_dicts()` and emit `SumExpression((h ⊗ qw) * f_vec)` — one integrand call,
one flat reduction node. The core already has everything needed (`SumExpression`,
`Constant` broadcasting).

**P2. `_quadrature_weights` runs `scipy.integrate.quad` in a Python loop, per call,
with imports inside the loop.** The weights are an `ncp×ncp` Vandermonde solve (see C2's
fix — the correct fix and the fast fix are the same change). Move to
`polynomials.py` as `radau_quadrature_weights(ncp)` with an lru_cache, next to its
siblings, and cross-test against the current `quad`-based values for ncp ≥ 2.

**P3. `least_squares(interpolate=True)` recomputes `_element_points()` per observation**
(via `state_at`) and builds each residual as an (ncp+1)-term scalar chain. Cache
`_element_points()` on the builder (it is immutable after construction), and optionally
assemble all observations as a single `(n_obs, nfe·(ncp+1))` interpolation-matrix
`MatMulExpression` — same structure the collocation constraints already use.

**P4. `FDBuilder` is entirely scalar.** The backward/forward/central loops call the user
RHS `nfe` times and emit `nfe` scalar constraints per state; `MOLBuilder` with
`time_method="finite_difference"` multiplies this by `npts` spatial points. The
collocation builder was vectorized precisely to avoid this (per its own docstrings).
Vectorize the FD stencils with slices — e.g. backward:
`(x[1:] − x[:-1]) / h == f(t[1:], x[1:], …)` — one RHS call, one vector constraint per
state. This is mechanical and makes the two builders symmetric.

---

## 5. Test coverage gaps (why the bugs survived)

1. **All Neumann tests use zero flux** → C1 invisible. Add nonzero-flux tests on both
   boundaries with an exact steady-state solution.
2. **`integral()` only tested at `ncp=3`** → C2 invisible. Parameterize over
   `ncp ∈ {1..5}` × `{radau, legendre}`.
3. **No test constructs a builder with a missing/typo'd RHS key** → C3 invisible. Add
   negative tests asserting `ValueError`.
4. **No MOL accuracy test against an analytical PDE solution** — existing tests assert
   only "solves" and "energy decays". Add the standard heat-equation decay-rate check
   (`u(z,t) = e^{−απ²t} sin(πz)`, assert pointwise error) so spatial-stencil and BC
   regressions are caught quantitatively.
5. **Radau roots for `ncp ∈ {4,5}` never checked against reference values** (only
   sortedness). Add the quadrature-exactness check used in this review (degree
   `2·ncp−2`), which pins root precision without hardcoding literature decimals.
6. **No Legendre-scheme tests for `extract_solution`/`time_points` consistency** (R6).
7. **No FD forward/central + algebraic tests** (C5).
8. **No `ContinuousSet` validation tests** (R1) and no post-`discretize()` mutation
   tests (R2).

---

## 6. Implementation plan (for Opus)

House rules apply to every task: feature branch + PR per phase, task ID in the title
(`fix(dae): D-1 …`), each fix ships with a regression test that **fails before and passes
after**, run `pytest python/tests/test_dae.py python/tests/test_dae_fd_mol.py` (fast +
slow) plus `pytest -m smoke`. All changes here are in the Python modeling layer — the
solver core is untouched, so no bound-neutrality panel is required; the verification
regime is the module's own numerical-accuracy tests. Do not weaken any existing test
tolerance. Baseline before starting: fast suite green (82 passed, verified 2026-07-03).

### Phase 1 — P0 correctness (one PR, `fix(dae): D-1..D-3`)

| ID | Task | Files | Acceptance criteria |
|----|------|-------|---------------------|
| D-1 | Flip left-Neumann ghost sign: `u[0] - dz*bc_val` → `u[0] + dz*bc_val` | `mol.py:433` | New steady-state test (`u_t = u_zz`, left `du/dn=−1`, right Dirichlet 1, init `u=z`): final-time max error < 1e-6 with backward-FD time. Mirrored right-boundary test passes unchanged. Fails on current main with error ≈ 0.81. |
| D-2 | Replace `_quadrature_weights` with `polynomials.radau_quadrature_weights(ncp)`: Vandermonde solve on collocation points, `lru_cache`, no scipy | `collocation.py:882-911`, `polynomials.py` (+ export) | `sum(w)==1` and `∫₀¹ t^k` exact to deg `2·ncp−2`, ncp=1..5. `integral(1)` over `[0,T]` returns `T` at every `ncp ∈ {1..5}` × both schemes (parameterized end-to-end test; ncp=1 currently returns `T/2`). Weights for ncp≥2 match old quad-based values to 1e-12 (bound-neutral for previously-correct configs). |
| D-3 | Validate RHS dict keys + missing-RHS guard in **both** builders (shared helper): unknown keys → `ValueError` listing them; states without derivatives → `ValueError`; `add_second_order_state` present but `set_second_order_ode` absent → `RuntimeError` at `discretize()` | `collocation.py`, `finite_difference.py` | Negative tests: typo key raises naming the key; partial dict raises naming the missing state; forgotten second-order RHS raises. All existing tests (incl. mixed first+second order) still pass. |

### Phase 2 — P1 robustness (one PR, `fix(dae): D-4..D-9`)

| ID | Task | Files |
|----|------|-------|
| D-4 | `MOLBuilder`: raise `ValueError("npts >= 2 required")` at construction (or fix the scalar path if trivial — refusal preferred over a numerically useless config) | `mol.py` |
| D-5 | Forward-Euler + algebraic: refuse loudly (`set_algebraic` with `method="forward"` → `ValueError` explaining explicit Euler is inconsistent for DAEs) — per philosophy, a refusal beats a silent off-by-one; leave an issue if someone later wants re-indexed forward DAE support | `finite_difference.py` |
| D-6 | `ContinuousSet.__post_init__` validation: `nfe>=1`, `t0<tf`, `ncp>=1`, known scheme | `collocation.py` |
| D-7 | Post-`discretize()` mutation guards on all `add_*`/`set_*` in all three builders; duplicate-name check across states/algebraics/controls | all three |
| D-8 | `align_time_grid`: detect unplaced/colliding measurements → warn (with the times) and document that `nfe` may shrink; `_compute_initial` array-length validation | `collocation.py`, `mol.py` |
| D-9 | Legendre `extract_solution`: make the boundary-node skip Radau-only; build `tp[i, ncp]` for Radau as `eb[i+1]` exactly (kills the float-dedup fragility) | `collocation.py` |

### Phase 3 — API completeness (one PR, `feat(dae): D-10..D-12`)

| ID | Task | Notes |
|----|------|-------|
| D-10 | `FDBuilder.integral()` (trapezoid on the grid) + docstring alignment with `MOLBuilder.time_builder` | Test: `∫1 = T`, `∫t = T²/2` to trapezoid accuracy |
| D-11 | `FDBuilder.state_at()` (piecewise-linear) and `least_squares(..., interpolate=True)` default, mirroring the collocation API (#94 parity) | Same pinned-values test style as `TestLeastSquaresInterpolation` |
| D-12 | Robin BCs (`a·u + b·du/dn = c`, subsumes Dirichlet/Neumann) + second-order one-sided Neumann/Robin reconstruction | Convergence test: observed spatial order ≥ 1.9 with a nonzero-flux Neumann BC (currently ~1) |

### Phase 4 — performance (one PR, `perf(dae): D-13..D-15`)

| ID | Task | Measurement required in PR |
|----|------|---------------------------|
| D-13 | Vectorize `integral()` via `_build_vec_dicts` + `SumExpression`; cache `_element_points()` | Build-time + DAG-node-count before/after at `nfe=100, ncp=3`; objective values identical to 1e-12 on the Phase-1 parameterized integral tests |
| D-14 | Vectorize `FDBuilder` stencils (slice arithmetic, one RHS call) | Same-solution check (`node values identical to 1e-10`) vs scalar path on exp-decay, all three methods; build-time at `nfe=1000` |
| D-15 | Optional: matrix-form `least_squares` (single MatMul over an interpolation matrix) | Only if D-13 profiling shows it matters; otherwise record as not-worth-it in this doc (falsification-log style) |

Phase-4 changes are *transcription-neutral*: assert generated constraint counts and
solved trajectories are bit-identical (or ≤1e-10) against the scalar implementation
before deleting it.

### Phase 5 — SOTA-gap enhancements (separate design discussion, not one PR)

Ordered by value-to-effort for this repo specifically (see §7): (1) interval
bound-propagation through elements to auto-tighten default ±1e20 state bounds — directly
strengthens the global solver's McCormick relaxations on transcribed problems; (2)
simulation-based initialization (`solve_ivp` rollout → initial point for the NLP/B&B);
(3) free-final-time support via time scaling; (4) error-estimate-driven mesh refinement
(Betts-style local error on a fixed refinement loop); (5) piecewise-linear controls.
Each needs its own entry-experiment + kill criterion per the development philosophy
before implementation.

---

## 7. SOTA assessment — solvers, collocation methods, and where this module stands

### 7.1 The transcription core is the standard, correct method — not a weak point

The module implements **orthogonal collocation on finite elements (OCFE) with Radau IIA
points**, transcribing the DAE into an all-at-once NLP ("simultaneous" / full
discretization). This is precisely the method of record for equation-oriented dynamic
optimization — Biegler's *Nonlinear Programming* (2010) canon, and what Pyomo.DAE,
gPROMS, GEKKO, and DynoPy use as their default. The specific choices are the right ones:

- **Radau IIA as default.** Stiffly accurate and L-stable, order `2s−1`, includes the
  right endpoint — which (a) gives C⁰ continuity for free, (b) makes index-1 algebraic
  variables well-defined at element ends, and (c) avoids the weak instability
  Gauss-Legendre collocation exhibits on stiff/high-index problems. This is the same
  default as Pyomo.DAE and the collocation mode of CasADi-based tools. Verified here to
  machine precision (roots, differentiation matrices, continuity weights) and the test
  suite confirms the expected `O(h^{2s−1})` superconvergence.
- **Gauss-Legendre as an option** with explicit interpolated continuity — correct and
  standard (higher quadrature order, no stiff accuracy).
- **Vectorized constraint emission** (`x @ Aᵀ = h·f` as one MatMul per state) is a
  genuinely good implementation choice — most academic transcription layers (including
  Pyomo.DAE) emit scalar constraints and pay for it in model-build time.
- Element-boundary alignment to measurements (`align_time_grid`), exact-order
  polynomial evaluation at arbitrary times (`state_at`), and interpolating least-squares
  are exactly the right primitives for parameter estimation, and ahead of what
  Pyomo.DAE gives you out of the box.

So: **as a fixed-mesh simultaneous-collocation transcription layer, the design is
state of the practice**, feature-comparable to Pyomo.DAE (the most widely used
open-source equivalent), and the P0 items above are implementation bugs, not
method-level deficiencies.

### 7.2 Gaps relative to state-of-the-art *dynamic optimization systems*

Measured against the best current tools (CasADi/acados ecosystem, GPOPS-II-style
hp-adaptive pseudospectral packages, gPROMS/Betts-class industrial transcription), the
missing capabilities, in rough order of practical importance:

1. **No adaptive mesh refinement / error estimation.** SOTA optimal-control codes
   (GPOPS-II's hp-adaptive Legendre-Gauss-Radau, Betts' SOS, recent ph/hp schemes)
   estimate local discretization error and refine elements or raise polynomial degree
   automatically; they also place element boundaries at control switching points, which
   fixed meshes smear. This module is fixed-mesh with manual `element_boundaries`. This
   is the single largest method-level gap for optimal control use.
2. **No initialization strategy.** Simultaneous collocation NLPs are only as good as
   their initial point; every serious tool provides a simulation rollout (Pyomo.DAE's
   `Simulator` via CasADi/scipy) or element-wise integration warm start. Here all
   variables start from the solver default. For discopt's *global* B&B this matters
   less for the certificate but a lot for time-to-first-incumbent.
3. **Piecewise-constant controls only.** No piecewise-linear/quadratic control
   parameterization, no control continuity or rate-of-change constraints across
   elements — standard options elsewhere.
4. **No free final time / time-scaling support.** Minimum-time problems require the
   user to hand-roll the `t = t_f·τ` transformation. Every SOTA OCP layer automates it.
5. **Index-1 only, by assumption, unchecked.** No structural index analysis
   (Pantelides), no index reduction, no consistency checking of initial conditions for
   algebraic variables. gPROMS and Modelica-world tools (JModelica heritage, Dymola)
   do this automatically. At minimum the restriction should be documented and, ideally,
   detected (a singular algebraic Jacobian at the initial point is cheap to probe).
6. **Path constraints only at collocation nodes.** Inter-node violations are a known
   failure mode of direct collocation; SOTA mitigations (extra check points, ε-relaxed
   bounds between nodes) are absent — though `state_at` provides the right primitive to
   build them.
7. **MOL layer is basic.** 1-D only, second-order central differences only, first-order
   Neumann treatment (A3), no Robin BCs, and no upwinding/flux limiting — central
   differences on convection-dominated PDEs (`fz`-heavy right-hand sides) will produce
   spurious oscillations. PDE-constrained-optimization SOTA (even at the "simple MOL"
   tier) offers at least first-order upwind for advection terms and second-order
   boundary treatment throughout. Fine for diffusion-dominated teaching problems;
   not for transport.
8. **No multiple shooting.** Direct collocation is the right default for a global
   solver (everything stays algebraic), so this is a deliberate and defensible scope
   choice, not an oversight — noted only because "DAE solver SOTA" comparisons usually
   include it (acados/MUSCOD-II lineage for real-time NMPC).

### 7.3 The discopt-specific angle: global optimization of transcribed dynamics

This module's real differentiator is its context: it feeds a **global** MINLP solver,
whereas Pyomo.DAE/CasADi target local NLP solvers (Ipopt). Two consequences worth
acting on (Phase 5):

- **Default state bounds of ±1e20 are hostile to the relaxation layer.** McCormick and
  factorable relaxations need finite boxes; unbounded intermediate states make the root
  relaxation vacuous. Because the transcription knows the dynamics, it can propagate
  interval bounds element-by-element (a crude interval Euler/Radau sweep) and install
  finite, valid state bounds automatically — turning the DAE layer from a passive
  transcriber into an active relaxation-strengthener. None of the mainstream local-NLP
  transcription layers do this because they don't need it; for discopt it is likely the
  highest-value SOTA-plus feature. The literature basis is the validated-bounding line
  of work (differential inequalities / Taylor-model reachability — Scott & Barton,
  Chachuat and coworkers) used by global dynamic-optimization codes; even the cheap
  interval version is a meaningful step. Entry experiment: measure root-node gap on a
  small optimal-control MINLP with and without propagated bounds.
- **Deterministic global dynamic optimization** (the Chachuat/Barton line: spatial B&B
  directly on the ODE with validated relaxations of the solution map) is the *other*
  SOTA paradigm. Full-transcription-then-global-MINLP — what this module enables — is
  the standard practical alternative and scales better in state dimension; the module
  is the right architecture for this repo, and no change of approach is recommended.

### 7.4 Verdict

- **Methodologically**: Radau-IIA OCFE with the implemented order/continuity structure
  *is* the state-of-the-art transcription family; the module is a correct-by-design,
  well-vectorized member of it, on par with Pyomo.DAE's feature set for fixed meshes
  and ahead of it on parameter-estimation ergonomics (`state_at`, interpolating least
  squares, measurement-aligned grids).
- **Implementation-wise**: three P0 bugs (§2) currently undermine that standing —
  a wrong answer with any nonzero left Neumann flux, a 2× wrong integral objective at
  `ncp=1`, and silently dynamics-free models on two easy-to-hit user mistakes. These
  are all small, well-localized fixes (Phase 1).
- **System-wise**: it is a solid fixed-mesh transcription layer, not yet a SOTA
  dynamic-optimization *system* — no mesh adaptation, no initialization, no free final
  time, index-1-unchecked, and a basic MOL layer. The single most valuable direction is
  not to chase GPOPS-style adaptivity but to exploit the global-solver context:
  automatic interval state bounds for tighter relaxations (§7.3), which no comparable
  open-source transcription layer provides.
