# RLT-1 Lagrangian dual — scoping plan (issue #661)

**Status:** scoping / entry-experiment GO. Not yet implemented behind a flag.
**Owner tracking issue:** #661 (qap global dual bound).
**Prereqs already shipped:** exhaustive RLT-1 root bound (#675), NS-safe rigorous
return (PR #679), set-partitioning exclusion presolve (PR #679). See
`docs/dev/sparse-milp-plan.md` §RLT1.

## 0. Why this, why now (the measurements that force decomposition)

The RLT-1 relaxation is strong enough — it lifts qap's root bound from McCormick's
~0 to ~352 891 (91 % of the 388 214 optimum, well above the published dual
149 106). The blocker is **producing that bound at qap scale, rigorously, in a
practical budget.** Two routes have now been measured and *falsified* as
graduation levers (both recorded in `sparse-milp-plan.md` §RLT1, §4 discipline):

1. **Monolithic exact simplex** — the only rigorous oracle (POUNCE's IPM does not
   converge on these degenerate LPs: ~25 iters in 90 s on a 2778×666 RLT LP;
   HiGHS is removed, #356). Its solve time grows **~10–20× per unit `n`**
   (n=4→7: 0.02→38.7 s); qap is n=15, eight steps past n=7 → intractable.
2. **Bound-neutral structural presolve** — real but only a **constant ~2.4×**
   speedup; it shifts the exponential wall one notch, does not break it.

The wall is the *coupling*: the RLT product rows `Σ_k a_k X_{p,k} = β x_p` tie the
otherwise near-separable McCormick structure together and create the massive
primal degeneracy the simplex chokes on. **Decomposition is the only route that
attacks the wall rather than the constant** — dualize the coupling rows and never
form the monolithic LP.

## 1. The construction

Write the RLT-1 LP as

    min  cᵀz
    s.t. z ∈ P_McC := { McCormick bound-factor rows ∧ model linear rows ∧ z∈[0,1] }   (cheap, sparse)
         C z = 0                                                                       (RLT coupling rows)

where `z = (x, X)` is the lifted vector and `C z = 0` collects the (non-trivial,
post-presolve) RLT product identities. Lagrangian-relax **only** the coupling
rows with free-sign multipliers `μ`:

    g(μ) := min_{z ∈ P_McC} ( c + Cᵀμ )ᵀ z .

**Soundness (the whole point).** For *every* `μ`, weak duality gives
`g(μ) ≤ (RLT-1 LP optimum) ≤ (true integer optimum)`. So *any* `μ` yields a valid
global lower bound — no convergence needed for correctness, only for tightness.
`P_McC` is bounded (`z∈[0,1]`) so each `g(μ)` is finite. The inner minimizer's
coupling residual `C z*` is a subgradient of the concave `g`, so projected
subgradient / bundle ascent on `μ` drives `g(μ) ↑ RLT-1 optimum` (LP strong
duality: the dual optimum equals the primal RLT-1 optimum).

**Rigor of each inner solve.** The inner LP is solved for a bound, so we apply the
**Neumaier–Shcherbina safe bound** (`obbt._ns_safe_lp_lower_bound`) to the inner
solve's duals — already wired for the RLT path — making `g(μ)` rigorous even from
an inexact/ill-conditioned inner solve. `P_McC` is exactly the **sparse McCormick
LP** the sparse-MILP thread already solves in ~0.1 s at qap scale (T0–T12), so an
iteration is cheap and never densifies.

## 2. Entry experiment — done (synthetic), and the honest caveats

Prototype (`subgradient ascent over P_McC`, NS-safe inner bound, exclusion
presolve applied) on synthetic Koopmans–Beckmann QAPs:

| n | RLT-1 opt (monolithic) | Lagrangian best | % of RLT-1 | iters | Lag time |
|---|------------------------|-----------------|-----------|-------|----------|
| 5 | 1406.0 (0.1 s)         | 1406.0          | 100 %     | 65    | 0.5 s    |
| 6 | 2518.0 (0.8 s)         | 2518.0          | 100 %     | 108   | 2.6 s    |
| 7 | 2678.7 (15.8 s)        | 2675.9          | ~100 %    | 200   | 14.6 s   |

The dual **reaches the RLT-1 bound** through ~10²  cheap rigorous inner solves.
**Two caveats that bound this evidence — the real entry experiment must remove
both:**

- **Optimistic step rule.** The prototype used a Polyak step
  `t = (g* − g(μ)) / ‖Cz*‖²` seeded with the *known* target `g*` (the monolithic
  RLT-1 optimum). In production the target is unknown; a target-free rule (bundle
  method, or Polyak with an estimated/adaptive target, or diminishing step) will
  need **more** iterations. Iteration count is the primary risk to quantify.
- **qap-scale is extrapolated, not measured.** n=7's ~15 s matched monolithic here
  only because n=7's monolithic is already fast; the advantage is at qap scale
  where monolithic is >25 min but each inner McCormick solve is ~0.1 s (⇒ ~10²
  iters ≈ 10–20 s, *if* iteration count holds). qap's real `.nl` is not in the
  in-repo corpus, so this is unverified.

## 3. Kill criterion for the implementation stage (§4)

Before shipping (even flag-gated), on the **real** qap root box with the sparse
McCormick inner oracle (`DISCOPT_SPARSE_LARGE_LP`) and a **target-free** step rule:

- **GO** iff the rigorous `g(μ)` moves **meaningfully off 0 toward the oracle dual
  149 106** (ideally toward the 352 891 RLT-1 gauge) within an iteration budget
  whose total cost is **well under** the monolithic simplex (< a few minutes), on
  qap **and ≥1 other indefinite binary QP** (generality, §2).
- **KILL** iff a target-free rule stalls far below the bound, or the per-iteration
  inner solve is not fast at qap scale (densification / degeneracy leak), or the
  cost is comparable to the monolithic solve. Record and re-scope (next candidate:
  bundle-level stabilization, or the Shor SDP / global moment-cut directions in
  #661).

## 4. Soundness & graduation gates (non-negotiable, CLAUDE.md §1/§5)

- `g(μ) ≤ RLT-1 opt ≤ true opt` for any `μ` (weak duality) — the bound can only be
  an under-estimate. NS-safe on each inner solve dominates inner float error.
- On qap: bound must stay `≤ 388 214` (validated vs `minlplib.solu`).
- `incorrect_count == 0` on the global50 panel; joined into
  `_root_relaxation_lower_bound` via `max` so it can only *raise* the surfaced
  bound.
- Default-off flag (`DISCOPT_RLT1_LAGRANGIAN` or fold into the RLT-1 lever);
  graduates only after consecutive nightly-green per §5.

## 5. Implementation sketch (once GO confirmed)

1. Factor `build_rlt1_lp` into `(P_McC, C, c, offset)` (split builder; the
   prototype already does this) so the coupling matrix is first-class.
2. `rlt1_lagrangian_lower_bound(...)`: subgradient/bundle loop over the split,
   each step a sparse McCormick solve + NS-safe bound; return `max_μ g(μ)`.
3. Target-free step rule + stopping (small residual, stalled improvement, or
   iteration/time budget). Warm-start `μ` across the loop.
4. Wire as a candidate in `_root_relaxation_lower_bound` (max), flag-gated.
5. Tests: differential bound (≥ McCormick, ≤ true opt), no feasible point cut,
   rigor of each `g(μ)`, eligibility no-ops — mirroring `test_rlt_root_bound.py`.
