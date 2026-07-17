# RLT-1 Lagrangian dual — scoping plan (issue #661)

**Status:** implemented behind a default-off flag (`DISCOPT_RLT1_LAGRANGIAN`);
target-free convergence de-risked on synthetic QAPs. Remaining gate: the qap-scale
entry experiment on the real instance with the sparse inner oracle (§3), needed
before default-on.
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

**Target-free step rule — de-risked (was the primary open risk).** The prototype
used a Polyak step seeded with the *known* target `g*`. The shipped implementation
uses an **adaptive target level** (`level = g_best + δ`, `t = (level − g)/‖s‖²`,
with `δ` halving on a stall and growing on a run of improvements) — no external
upper bound, self-tuning from a scale-only initial `δ`. Measured against the
monolithic RLT-1 optimum on synthetic QAPs (`rlt1_lagrangian_lower_bound`,
`max_iter=400`): **100 % of the monolithic bound**, sound (≤ true optimum) in every
case:

| n | monolithic RLT-1 | Lagrangian (target-free) | % | sound |
|---|------------------|--------------------------|---|-------|
| 4 | 980.0  | 980.0  | 100 % | ✅ |
| 5 | 1406.0 | 1406.0 | 100 % | ✅ |
| 6 | 2518.0 | 2518.0 | 100 % | ✅ |
| 7 | 2678.7 | 2678.7 | 100 % | ✅ |

A comparison of rules (300-iter budget) confirmed the choice: adaptive-level 100 %
across the board, Held-Karp-with-UB 97–100 %, plain diminishing only 77–88 %.

**Still open — qap-scale is extrapolated, not measured.** At these small `n` the
Lagrangian wall-clock (~26 s at n=7) is no better than the monolithic solve,
because the *inner* solve here uses the exact simplex on a not-yet-huge McCormick
LP; the advantage is only at qap scale, where the monolithic solve is >25 min but
each inner McCormick solve is ~0.1 s with the sparse driver (⇒ ~10² iters ≈ 10–40 s
*if* iteration count holds). qap's real `.nl` is not in the in-repo corpus, so this
is the remaining gate before default-on (§3).

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

## 5. Implementation — done (behind `DISCOPT_RLT1_LAGRANGIAN`, default off)

1. `build_rlt1_split` (`python/discopt/_jax/rlt.py`) returns `RLT1Split`
   `(c, A_in, b_in, C, offset, …)` — the inner McCormick polytope `P_McC` and the
   coupling `C z = 0` as first-class matrices, sharing the eligibility gate,
   exclusion presolve, pair lift, and objective with `build_rlt1_lp`.
2. `rlt1_lagrangian_lower_bound(...)`: adaptive-target-level subgradient ascent over
   the split, each step a sparse McCormick solve made rigorous by
   `obbt._ns_safe_lp_lower_bound`; returns `max_μ g(μ)`.
3. Target-free `δ` rule with two-sided adaptation + stopping (residual `‖Cz‖→0`,
   iteration budget `rlt1_lagrangian_max_iter`, time budget). `μ` carried across
   the loop.
4. Wired in `solver.py::_root_relaxation_lower_bound` as a `max` candidate,
   gated by `SolverTuning.rlt1_lagrangian`.
5. Tests (`python/tests/test_rlt_root_bound.py`): reaches the monolithic bound and
   `≤` true optimum; no feasible point cut (`A_in z ≤ b_in ∧ C z = 0` at every
   assignment); split matches the monolithic feasible region; eligibility no-op
   without equalities; flag default-off; root-wiring soundness.

**Remaining before default-on:** the §3 qap-scale entry experiment on the real
instance with the sparse inner oracle, then consecutive nightly-green (§4).
