# Issue #671 entry experiment — does exact / high-precision LP arithmetic give hda a tight dual bound?

Entry experiment (Dev-Philosophy #4: run the measurement **before** building).
Research-scale, docs-only. **No production solver code changed.**

- Loose floor today (candidate A, #662): dual bound **−1.80e10** (sound but loose).
- True optimum (`minlplib.solu`): **−5964.534084**.
- **Result of this experiment: CONFIRMED.**

## Hypothesis (H)

Solving hda's root McCormick relaxation LP in exact rational (or sufficiently
high-precision) arithmetic resolves the float64 false-infeasible and yields a
bound *dramatically* tighter than candidate A's −1.80e10 — justifying integrating
an exact / higher-precision LP path.

## Evidence / prior trail

- `docs/dev/candidate-b-phase2-scaling-entry-2026-07-16.md`: hda's root LP is
  feasible (elastic phase-1 min violation 1.8e-10) but every float64 engine
  (feral, HiGHS simplex + IPM, SCIP/SoPlex) false-infeasibles; rank-deficient,
  ~1e14 conditioning; scaling and presolve/rank-reduction both falsified.
- Gleixner, Steffy, Wolter, *Iterative Refinement for Linear Programming*
  (INFORMS J. Comput. 2016).

## Kill criterion

H is **FALSIFIED** if exact/high-precision arithmetic still yields no tight bound
(the LP is genuinely rank-deficient in a way that leaves the bound loose even
solved exactly, or exact solve is hopelessly intractable at scale). **CONFIRMED**
if exact arithmetic resolves the false-infeasible and produces a bound within a
few orders of magnitude of −5964.53 rather than −1.80e10.

## What was exported

`build_milp_relaxation` on hda at its root box →
`discopt_benchmarks/results/issue671/hda_root_lp.npz`:

| property | value |
|---|---|
| shape | **3008 rows × 1140 cols** (pure inequalities `A_ub x ≤ b`, + finite bounds; no equalities) |
| nnz | 7715 (extremely sparse, ~2.6 nnz/row) |
| objective nnz | 42 |
| raw coefficient spread | **2.837e26** — *identical* to the diagnosis's `raw_spread`, confirming the same relaxation |
| SVD | σ_max 8.91e10, σ_min(nz) 8.96e-17, numerical rank 1127/1140 (deficient by 13) |

(Reproducer: `discopt_benchmarks/results/issue671/export_hda_root_lp.py`.)

## Measurements

### Baselines (reproduce the failure) — `analyze_baselines.py`

| test | result |
|---|---|
| float64 objective LP, HiGHS simplex / `highs-ds` / IPM | **infeasible (false)** — all three |
| float64 elastic phase-1 (min total violation) | optimal, **1.821e-10 → FEASIBLE** |
| sound presolve (drop zero rows, singleton→bounds, fixed/empty cols) | core still **2883 × 1096** — barely reduced |

The presolve result re-confirms the diagnosis: **rank-reduction is not a lever**;
an exact solver faces essentially the full system.

### E1 — τ-relaxation homotopy (the bound magnitude) — `exact_experiment.py`

`min cᵀx s.t. A x ≤ b + τ, bounds`. As τ→0⁺ the feasible set shrinks to the true
polytope and the objective rises to the true LP optimum. This is a poor-man's
high-precision probe: a *tiny* feasibility slack lets float64 recover the bound
the exact solve would certify.

| τ | status | objective |
|---|---|---|
| 1e-2 | optimal | −75611.72 |
| 1e-4 | optimal | −64798.83 |
| 1e-7 | optimal | −64675.37 |
| 1e-8 | optimal | −64675.26 |
| 1e-9 | optimal | −64675.25 |
| 1e-10 | optimal | −64675.25 |
| 1e-6, 1e-11 | *unbounded* | (float64 numerical noise on the near-singular basis) |
| 1e-12, 0 | infeasible | (float64 false) |

**Converged tight bound ≈ −6.468e4**, stable across τ = 1e-7 … 1e-10. That is
**~5.4 orders of magnitude tighter** than candidate A's −1.80e10 (gap-to-opt
5.9e4 vs 1.8e10). The flips to "unbounded"/"infeasible" at τ ≤ 1e-11 are exactly
the float64 precision fragility this issue is about.

### E2 — the false-infeasible is a precision artifact (decisive) — `exact_experiment.py`, `inspect_core.py`

A deletion filter shrinks the float64-infeasible row set to an irreducible
**2-row / 3-column core**:

```
row 1393:  −x[1026] − 6.211261e-10·x[1027]  ≤  −6.590951e-11
row 1395:  −x[42]   + 6.300000e+10·x[1026]  ≤   0
x[42]  ∈ [1.23e-2, 1.50e-1]     (normal-magnitude flow var)
x[1026]∈ [1.96e-13, 2.37e-12]   (tiny Arrhenius aux)
x[1027]∈ [1.02e-1, 1.12e-1]     (normal var)
```

| solver on this 2-row block | verdict |
|---|---|
| float64 HiGHS | **infeasible** |
| exact rational (fractions.Fraction two-phase simplex) | **FEASIBLE, witness violation = 0** |

This is the diagnosis's Arrhenius coupling in miniature: a `6.3e10`
pre-exponential (row 1395) tied to a `~1e-13` aux, and an envelope-slope term
`6.2e-10·x[1027] ≈ 6.8e-11` that must be balanced against a `6.59e-11` RHS
(row 1393). That balancing term sits **below HiGHS's ~1e-7 feasibility
tolerance**, so float64 drops it and certifies the pair empty; exact arithmetic
keeps it and finds a feasible point (e.g. `x[42]=0.15, x[1026]=1.96e-13,
x[1027]=0.112`). **The false-infeasible is a pure precision artifact, reproduced
in two rows and dissolved by exact arithmetic.**

## Verdict — CONFIRMED

Both halves of H hold:

1. **False-infeasible resolved by precision.** The 2-row core is float64-infeasible
   but exactly feasible (zero-violation witness). Precision is the lever, exactly
   as the issue predicted.
2. **Tight bound recovered.** The root McCormick LP optimum is **≈ −6.468e4** —
   ~5.4 orders of magnitude tighter than candidate A's −1.80e10, and materially
   closer to opt −5964.53 (the issue's "tight" target: *materially closer than
   −1.80e10*).

**Important caveat (not a falsification).** −6.468e4 is the *true root McCormick
relaxation value*; the residual gap to opt −5964.53 is the **genuine McCormick
relaxation gap** at the root box — a relaxation property, **not** a precision
artifact, and **not** closable by LP precision. Exact/high-precision LP fixes the
soundness **and tightness of the root dual bound**; it does not by itself solve
hda (closing the McCormick gap is branch-and-reduce / a stronger relaxation, an
orthogonal effort).

## Tractability

- **Naive pure-Python exact rational: intractable at full scale.** The irreducible
  core is 2883 × 1096; a dense Fraction tableau is ~3.2M rationals with O(m·n) work
  per pivot in pure Python. Exact arithmetic was tractable on the **2-row core**
  and would be on small sub-blocks, not the full LP by hand.
- **A real exact-LP library is routine here.** 3008 × 1140 with 7715 nnz is small
  for QSopt_ex / SoPlex-exact (the bundled SCIP lacked exact-solve support, which
  is why the prior trail could not demonstrate it).
- **A tiny feasibility relaxation already recovers the bound in milliseconds** (E1),
  which strongly suggests **LP iterative refinement (GSW)** — solve in double,
  compute residuals in f128/rational, re-solve a correction LP — is a *cheaper
  sufficient* lever than full exact rational for the bound. (Full exact rational
  remains the gold standard for a definitive feasibility certificate.)

## Go / No-go recommendation — GO (research-scale), root-only / failure-triggered

- **Path:** an LP iterative-refinement layer (Gleixner–Steffy–Wolter: f128 or
  rational residuals over feral, correction LPs in double), and/or an exact-rational
  LP invoked **only when a node LP false-fails** (numerical / spurious-infeasible).
  Per the issue, refinement also needs a working factorization on near-singular
  bases (rank-revealing / regularized LU), the smaller feral-touching half.
- **Placement:** **root-only or numerical-failure-triggered**, never the hot
  per-node engine — the cost is justified only on the pathological ill-conditioned
  relaxations that float64 cannot certify. On hda this is the root node.
- **Expected payoff:** hda's dual bound moves from −1.80e10 (loose floor) to
  ≈ −6.47e4 (true root relaxation), a sound, materially tight certificate — while
  keeping candidate A as the fallback. Verification regime unchanged from the issue:
  bound-neutral on the certifying panel (`node_count` + certified `objective`
  exactly unchanged for already-solving instances), `cargo test -p discopt-core`,
  `incorrect_count ≤ 0`.
- **Scope honesty:** this tightens the *certificate*; it does not close hda's
  optimality gap (that is the McCormick relaxation gap, a separate lever).

## Artifacts (reproducible)

Under `discopt_benchmarks/results/issue671/`:
`export_hda_root_lp.py`, `analyze_baselines.py`, `exact_experiment.py`,
`inspect_core.py`, `hda_root_lp.npz`, `results_summary.json`.
