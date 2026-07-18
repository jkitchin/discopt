# Issue #671 — GSW LP iterative-refinement kernel (the precision layer above feral)

Follow-up to the #671 entry experiment (`issue-671-hda-exact-lp-entry-experiment-2026-07-17.md`,
PR #708), which **CONFIRMED** that hda's loose dual bound (−1.80e10, candidate A / #662)
is a *precision artifact*, not a relaxation property: solving the same root
McCormick LP with high-precision residuals recovers the true root value ≈ −6.47e4
(≈5.4 orders tighter). The confirmed lever is **LP iterative refinement**
(Gleixner–Steffy–Wolter 2016): solve in double, compute the residual against the
original data in higher-than-double precision, scale it up, re-solve a correction
LP in double, iterate.

This doc records the **kernel** that lever needs, landed as a self-contained,
cargo-tested Rust module, and the integration + factorization-hardening plan for
turning it into hda's tight production bound.

## What landed (this change)

Two parts: the reusable **kernel** (`crates/discopt-core/src/lp/simplex/refine.rs`)
and the **shipped hda bound** (the τ-regularized-resolve path wired into the
numerical-failure branch behind a default-OFF flag — see "Integration" below). The
kernel is a precision layer *above* the double-precision simplex. Three pieces:

1. **High-precision residual primitive** — `residual_dd` / `dot_dd`. Pure-Rust
   *double-double* (≈106-bit) accumulation via the `two_sum`/`two_prod` error-free
   transforms (the latter using a fused multiply-add). No new dependencies. This
   is the GSW primitive: it computes `b − A x` (and `c − Aᵀ y`) to full float64
   accuracy even when the individual products dwarf the true residual — the case
   that makes a naive float64 dot return noise, or `0`.

2. **GSW primal-dual refinement loop** — `refine(...)`, generic over a
   `CorrectionSolver` (the fixed-`A` inner double solver — in production, the
   warm-started node simplex). Each round:
   - computes the primal residual `b − A x*` and dual residual `c − Aᵀ y*` in
     double-double,
   - scales each up toward O(1) (`Δp`, `Δd`),
   - solves a correction subproblem with the **same `A`** and shifted
     objective/rhs/bounds in double,
   - folds the scaled-down correction back into the incumbent `(x*, y*)`.

   The incumbent is kept **exactly box-feasible every round** because the
   correction's bounds are `Δp·(l − x*) ≤ x̄ ≤ Δp·(u − x*)`, so
   `x* + x̄/Δp ∈ [l, u]` identically. Convergence is geometric (≈16 digits/round
   for a solver returning ≈1e-16-accurate corrections).

3. **Neumaier–Shcherbina safe bound** — `ns_safe_bound(y, …)`, the same
   weak-duality lower bound `g(y) = bᵀy + Σ_j min_{x_j∈box}(c − Aᵀy)_j x_j` the
   MILP boundary already applies to candidate A's drifted dual
   (`milp_simplex._safe_lp_lower_bound_std`) and that `primal::safe_bound` uses in
   its tests. `g(y)` is valid for **any** `y`, so a refined dual can only *tighten*
   the certificate — never lift a bound above the true optimum. Refinement improves
   the multiplier fed into this function; it changes no guard, tolerance, or
   fallback.

### Why the precision has to live here, not inside feral

feral's *existing* double-precision iterative refinement (`with_numeric_focus`)
was already falsified on hda (issue body: "byte-identical, no effect — residual
computed in double can't recover ~1e14 loss"). GSW is a **different layer**: the
residual is computed against the original problem data at *higher-than-double*
precision (`residual_dd`), then a **double** correction solve consumes it. Only the
residual is high-precision. That is exactly what this module adds, and exactly what
a feral-internal double-precision refine cannot.

## Cargo-validated results (`cargo test -p discopt-core --lib refine`, 12 tests)

| test | what it proves |
|---|---|
| `two_sum_is_exact`, `two_prod_is_exact` | the error-free transforms are exact (the precision floor of the whole layer) |
| `residual_dd_survives_catastrophic_cancellation` | `1e16 + 1 − 1e16`: naive float64 → `0`, `residual_dd` → the exact `1` |
| `residual_dd_recovers_arrhenius_scale_product_error` | at hda's `6.3e10 × 1.96e-13` Arrhenius magnitudes, `residual_dd` recovers the sub-ulp product tail naive drops to `0` |
| `dot_dd_beats_naive_on_cancellation` | double-double dot resolves a cancellation naive gets wrong |
| `refine_exact_solver_reproduces_optimum_bound` | the GSW loop + `ns_safe_bound` reproduce the true LP optimum |
| `refine_extracts_accuracy_below_the_inner_solvers_grid` | given an inner solver capped at **1e-4** accuracy, refinement drives the true residual **< 1e-9** — the core GSW guarantee (accuracy past the inner solver's floor), the analogue of escaping HiGHS's ~1e-7 feasibility-tolerance floor on hda |
| `refine_tightens_a_drifted_dual_on_an_ill_conditioned_lp` | on a 1e8-range LP, a single lossy solve gives a **drifted dual → loose** safe bound; refinement tightens it to the optimum (candidate A → tight, in miniature) |
| `ns_safe_bound_is_never_above_optimum_for_arbitrary_dual` | soundness: `g(y) ≤ opt` for arbitrary `y` — the property candidate A relies on |

Full suite green: `cargo test -p discopt-core --lib` → **470 passed**; `cargo
clippy -p discopt-core --lib` clean.

The `…grid` and `…drifted_dual` tests are the honest model of hda: the obstacle
there is **not** float64 arithmetic cancellation in a single residual (hda's
balancing terms are ~1e-11, only ~1 digit of cancellation) — it is a **fixed
tolerance floor** (HiGHS's ~1e-7; feral's growth guards) that a single double
solve cannot see under. GSW's scale-up-and-resolve is what escapes that floor, and
that is precisely what these two tests exercise with a deliberately floor-limited
inner solver.

## Integration (landed, default-OFF, failure-triggered) — hda resolved end-to-end

The kernel is wired into the numerical-failure branch that already computes
candidate A, behind a new **default-OFF** flag `DISCOPT_LP_ITERATIVE_REFINEMENT`
(`SolverTuning.lp_iterative_refinement`).

### How the factorization blocker was sidestepped

The issue flags that GSW alone does not rescue hda without a working factorization
on near-singular bases: feral FT-`Growth`-fails on hda's root LP, so it returns
`numerical` with a *drifted* dual, not a clean one. Measurement on the real
exported root LP (`inhouse_regularized.py`) showed the way through:

| τ (RHS regularization) | in-house status | NS bound g(y) |
|---|---|---|
| 0 (plain) | numerical | −2.15e15 (garbage) |
| 1e-3 | numerical | −7.6e9 |
| **3e-3** | **optimal** | **−64735.08** |
| 1e-2 | numerical | −64736.20 |
| 2e-2 | optimal | −69296 |
| ≥5e-2 | optimal | progressively looser |

A **small RHS regularization moves the optimum off the near-singular configuration
so feral certifies the well-conditioned neighbour** (τ=3e-3 → `optimal`) and hands
back a *good* dual. The key soundness lever: the NS safe bound `g(y)` is valid for
**any** `y`, so it is evaluated against the **original** `b` (never `b+τ`) — the
regularization changes only the *tightness* of the recovered multiplier, never the
*soundness* of the bound. Because every `g(y)` is a valid lower bound, the reported
bound is the **max over a geometric τ-sweep and candidate A**, which is rigorously
safe (max of valid lower bounds) and robust to where the usable τ-window sits.

This needs **no external solver** (the #517 HiGHS rescue is not resurrected) and
**no feral factorization change** — it is entirely the "layer above feral" half.
The double-double residual / GSW `refine` kernel remains the principled primitive
(and the more general future path once a rank-revealing LU lets feral return
consistent approximate solutions to refine); the τ-regularized-resolve schedule is
the regularization that makes feral's *current* factorization return usable duals
today.

### Measured result (end-to-end, `dm.from_nl("hda.nl").solve()`)

| | reported dual bound | gap to opt (−5964.53) |
|---|---|---|
| flag OFF (candidate A, unchanged) | **−1.80e10** | 1.80e10 |
| **flag ON (#671)** | **≈ −6.45e4** (−64473 at 90 s) | **5.85e4** |

**~5.5 orders of magnitude tighter, sound** (−64473 ≤ opt −5964.53), the issue's
target. Flag OFF is byte-identical to today (candidate-A floor `−18016528426.5`
reproduced exactly before and after the change) — bound-neutral by construction.

### Placement / cost

Fires **only** on the numerical-failure path (`numerical`/`iter_limit` node LPs),
never the hot per-node engine. Each trigger costs one in-house LP solve per τ in
`_REFINE_TAUS` (7). On hda this ~doubles wall time (73 s → 141 s at the same
90 s budget) because many nodes fail numerically — acceptable for a **default-OFF,
research-scale** lever whose job is the certificate, not throughput. Instances
whose node LPs solve cleanly never trigger it (test:
`test_inert_on_cleanly_certifying_instances`).

### Verification (this change)

- `cargo test -p discopt-core --lib` (kernel): 470 passed; clippy + fmt clean.
- `pytest python/tests/test_issue_671_lp_iterative_refinement.py -m "not slow"`:
  fast soundness/plumbing tests pass (helper returns a sound, tight bound on an
  ill-conditioned LP; flag defaults OFF).
- `-m slow`: hda ON → tight sound bound (> −1e7, ≤ opt); hda OFF → unchanged
  candidate-A floor (< −1e7); alan/ex1221 byte-identical ON vs OFF.
- Candidate-A regression suite (`test_issue_517_*`, `test_issue362_*`) still green
  — the failure branch is unchanged with the flag OFF.

### Factorization hardening (the feral-touching half) — built, validated, wired; does NOT rescue hda's bound

The second half of the issue: make feral's LU survive the near-singular bases so a
node LP can return `optimal` at τ=0. feral exposes `LuSingularAction::PerturbToEps
{ abs_floor }`, which floors *only* the singular pivots (a localized regularization
→ a nearby `B'`, not a uniform `B + εI`) and completes instead of aborting.

**Primitive — CONFIRMED** (`crates/discopt-core/src/lp/simplex/regularized_lu.rs`,
7 cargo tests). On near-singular / exactly-singular bases where feral's default
`Fail` aborts, `PerturbToEps` + double-double refinement recovers the solve to
residual ~0 (growth 1.0); the boundary case (solution loading the near-singular
direction) recovers too, provided `abs_floor` sits below the genuine small pivots.
The capability lives on `FeralLU::with_singular_perturb`; the double-double
refinement (`dd_refined`, sparse-aware `residual_matvec_dd`) is what recovers
accuracy past the perturbed factor's error (feral's own double-precision refine was
falsified on this class).

**Wiring — safe, default-OFF** (flag `DISCOPT_LP_FACTORIZATION_HARDENING`). Mirrors
the #85 dense-retry: a thread-local `hardening_active()` makes `node_feral_lu()`
build a hardened factor; `dual::solve_lp_warm_csc` re-runs the solve under
`with_hardening_active` once a strict solve exits `Numerical`/`IterLimit`. The
re-solve is bounded (`8·(m+n)+2000` pivots **and** a 3 s wall deadline) so it can
only *rescue*, never stall. Flag OFF ⇒ byte-identical (measured: hda bound
`−18016528426.5` unchanged; `node_lu_is_plain_when_hardening_inactive`).

**Measured end-to-end on hda — the hardening does NOT rescue the bound.** With the
flag ON, hda's reported bound stays at candidate A's `−1.80e10` (the hardened
re-solve bails at its bound and falls through). The reason is decisive and worth
recording (Dev-Philosophy #4): **`PerturbToEps` unblocks the *factorization*, but
hda's McCormick LP is still too *degenerate to pivot* to optimality** — the simplex
grinds the same near-singular vertex geometry a completed factor does not change.
Contrast the **RHS regularization** lever (`DISCOPT_LP_ITERATIVE_REFINEMENT`, shipped
above), which *does* give hda its tight `−6.45e4`: perturbing the RHS moves the
optimum *off* the degenerate vertex, so the simplex reaches a clean optimum. So for
**hda's bound, RHS regularization is the working lever**; factorization hardening is
a correct, safe, validated primitive that may rescue *other*, less-degenerate
numerical-failure instances (where a completed factor lets the simplex finish) but
is not what hda needs. It stays default-OFF pending a corpus-wide panel that finds
the instances it *does* help.

### Remaining principled follow-up

The GSW `refine` kernel would supersede the τ-sweep once feral gains a **true
rank-revealing LU** *and* the degenerate-pivot problem above is addressed (a
completed factor is necessary but not sufficient — hda shows the vertex geometry,
not just the linear algebra, is the obstacle). An **exact-rational** solve of the
small failing sub-block (E2 core was 2×3) remains the gold-standard feasibility
certificate. Both are orthogonal to the delivered τ-regularized tight bound.

## Verification regime (Dev-Philosophy #5)

- **Bound-neutral with the flag OFF (the default):** the refinement block is guarded
  by `lp_iterative_refinement`, so with the flag OFF `node_count` and certified
  `objective` are exactly unchanged on every already-solving instance — the
  numerical-failure branch is byte-identical to today (candidate-A floor
  `−18016528426.5` reproduced exactly before/after the change). `cargo test
  -p discopt-core` green (470).
- **With the flag ON:** fires only on numerical-failure nodes. The NS bound is sound
  for any dual (evaluated against the original `b`), and the reported bound is the
  max over the τ-sweep and candidate A, so it is never looser than candidate A and
  never above the optimum; incumbents remain independently feasibility-verified.
  Acceptance met: hda reports a **tight** finite dual bound ≈ −6.45e4 (vs candidate
  A's −1.80e10; true root McCormick value ≈ −6.47e4), sound (≤ opt −5964.53).
- **Corpus-wide differential panel (graduation gate, before any default-ON):** not
  run here (the in-repo run needs the full MINLPLib corpus and a longer budget). The
  flag stays **default-OFF** pending that panel — this change delivers the sound,
  tight bound under the opt-in flag, per the bound-changing regime.

## Scope honesty

This tightens the **certificate**. It does **not** close hda's optimality gap: the
residual between −6.47e4 and opt −5964.53 is the genuine McCormick relaxation gap
at the root box (branch-and-reduce / a stronger relaxation — an orthogonal effort),
not anything LP precision can touch.

## References

- #517 (first finite bound), #662 (candidate A, merged), #664 (superseded),
  #708 (entry experiment, docs-only).
- Gleixner, Steffy, Wolter, *Iterative Refinement for Linear Programming*,
  INFORMS J. Comput. 28(3), 2016.
- `crates/discopt-core/src/lp/simplex/refine.rs`,
  `docs/dev/issue-671-hda-exact-lp-entry-experiment-2026-07-17.md`,
  `docs/dev/hda-no-bound-simplex-robustness-2026-07-16.md`.
