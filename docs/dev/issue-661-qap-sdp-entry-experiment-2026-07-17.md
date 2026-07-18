# Issue #661 — qap Shor-SDP entry experiment (2026-07-17)

**Type:** entry experiment / feasibility spike (CLAUDE.md §4 — run the experiment
*before* building). **No production solver code was changed.** All measurement is a
local scratch script (`discopt_benchmarks/results/issue661/qap_shor_sdp_repro.py`)
using `cvxpy 1.8.1` + `SCS 3.2.11` / `CLARABEL` on the extracted qap Q/constraints.

## Hypothesis (H)

The **Shor SDP** relaxation of qap gives a dual bound **much larger than the
McCormick ~0** — ideally near the best-known dual 149106 — justifying the cost of
integrating an SDP solve.

## Evidence motivating it (measured, `docs/dev/sparse-milp-plan.md` §T9)

- qap McCormick LP root bound = **~0** (indefinite `x'Qx`, x∈[0,1]).
- Local (≤6-var) moment/PSD cuts are redundant with McCormick at the McCormick
  vertex (0 of 223 pairwise minors violated) — only a **global** moment/PSD
  constraint can bind. This is exactly the Shor SDP `M=[[1,xᵀ],[x,X]]⪰0`.
- RLT-1 (LP, no SDP) already reaches **352890.9** (issue #661 / sparse-milp-plan
  "RLT1" entry). So the entry-experiment question is specifically: **does the SDP
  beat/complement RLT-1, and is it tractable at qap's n=15 (226×226 moment matrix)?**

## Kill criterion (pre-registered)

- **FALSIFIED** if the Shor bound is still ~0 / McCormick-level, **or** intractable
  at qap scale (226-dim moment matrix) within minutes.
- **CONFIRMED** if Shor (or Shor+RLT1) is substantially > 0 (say **> 50000**), sound
  (`≤ 388214`).

## Experiment

Two relaxations were built and solved on both a brute-forceable synthetic
Koopmans–Beckmann QAP (n=4, 5) and the real `qap.nl` (n=15, 225 binaries, 30
assignment equalities):

1. **Plain Shor** — `min ⟨Q,X⟩ s.t. M⪰0, diag(X)=x, x∈[0,1], A_eq x = b_eq`
   (assignment on x only; the formulation named literally in the issue).
2. **Strong Shor (ZKRW-style)** — plain Shor **plus** the lifted-assignment RLT rows
   (`A_eq·X = b_eq·x`), the McCormick box on X (`0 ≤ X_ij ≤ min(x_i,x_j)`,
   `X_ij ≥ x_i+x_j−1`), and the **gangster** constraints (`X_ij = 0` for two binaries
   in the same assignment row/column — they cannot both be 1).

Q, c, offset and A_eq/b_eq were extracted through the *existing* production helpers
(`_reconstruct_quadratic_objective`, `_extract_linear_constraints`) so the model is
identical to what the solver sees.

## Results

### Synthetic QAP (brute-force optimum available)

| n | optimum | McCormick | plain Shor | strong Shor (RLT1+gangster) |
|---|--------:|----------:|-----------:|----------------------------:|
| 4 | 490     | 0         | **−270**   | **490.0** (exact)           |
| 5 | 771     | 0         | **−668**   | **771.0** (exact)           |

Plain Shor is **worse than McCormick** (negative). Strong Shor recovers the **exact
optimum** — i.e. the SDP+gangster relaxation is tight on small QAPs, and the fix is
general (not a qap special-case).

### Real qap (n=15, 225 binaries)

| relaxation | solver | bound | status | time |
|---|---|---:|---|---:|
| plain Shor | SCS | **unbounded (−∞)** | unbounded | 0 s |
| plain Shor | Clarabel | — | did not finish | >300 s |
| **strong Shor (RLT1+gangster)** | **SCS** | **377098** | optimal | **86 s** |
| strong Shor (RLT1+gangster) | Clarabel | 357858 | user_limit (not converged) | 327 s |

Reference ladder for qap: McCormick **~0** · published dual **149106** · RLT-1 LP
**352891** · **strong-Shor SDP 377098** · optimum **388214**.

- **Plain Shor is unbounded** on qap: with only PSD + diag + assignment-on-x, the
  indefinite objective runs to −∞ over the cone `X ⪰ xxᵀ`. The bound comes entirely
  from the McCormick box + lifted-assignment + gangster constraints.
- **Strong Shor = 377098**, i.e. **97.1 % of the optimum**, above the RLT-1 LP gauge
  (352891) and **~2.5× the published dual** (149106). Sound: `377098 ≤ 388214`.
- **SCS (first-order) converges in 86 s**; interior-point (Clarabel) does not finish
  (its 327 s iterate of 357858 is still climbing and confirms the same >350k
  ballpark). First-order is the right tool at this scale.

## Verdict — **CONFIRMED**, with a formulation caveat

H is **confirmed for the strong (ZKRW-style) Shor SDP** — 377098 clears the >50000
kill threshold by ~7× and is the tightest known root bound for qap — but **plain Shor
as literally posed in the issue is FALSIFIED** (unbounded / negative; useless). The
lever is the SDP *plus* the lifted-assignment RLT rows and gangster constraints, not
the bare moment-matrix PSD.

## Caveats (soundness & cost)

- The SCS value is a **first-order approximate** optimum (eps=1e-5), **not yet a
  rigorous lower bound**. It sat below the optimum in every run, but production use
  must derive a **safe dual bound from the SDP dual** (the SDP analogue of the
  Neumaier–Shcherbina safe bound already used on the RLT-1 LP, `obbt._ns_safe_lp_lower_bound`).
- 86 s is fine for a **root** cut; it is **not** a per-node cost.
- Interior-point SDP does **not** scale here — a first-order/ADMM SDP (SCS-class) is
  required.

## Go / no-go recommendation

**Conditional GO** on relaxation strength, scoped narrowly:

- Add an **optional, root-only, default-off** strong-Shor SDP bound (flag, §5
  bound-changing), joined by `max` with the existing McCormick / PSD / RLT-1
  candidates in `solver.py::_root_relaxation_lower_bound` — it can only raise the
  bound.
- **Solver:** a first-order SDP (SCS-class). **Constraints:** PSD moment matrix +
  `diag(X)=x` + assignment + **lifted-assignment RLT** + McCormick box on X +
  **gangster** (all derivable from structure the RLT-1 path already extracts; stated
  generally per §2, keyed to set-partitioning equalities, not to "qap").
- **Expected bound:** ~97 % of optimum on qap (377k), exact on small QAPs.
- **Blocking requirement before default-on:** a rigorous safe-dual post-process on
  the SDP dual (no first-order value surfaced as a bound without it), plus the §5
  differential/feasible-point gates and the global50 `incorrect_count == 0` panel.

## Reproduction

`discopt_benchmarks/results/issue661/qap_shor_sdp_repro.py` (synthetic + real qap);
raw numbers in `qap_shor_sdp_summary_2026-07-17.json`, `qap_shor_plain.json`,
`qap_shor_rlt1_gangster.json` in the same directory.

## Production integration (2026-07-18)

The conditional GO above landed as `python/discopt/_jax/shor_sdp.py`:

- **Strong Shor only** (PSD moment matrix + `diag(X)=x` + model rows +
  lifted-equality RLT + McCormick box on X + gangster), assembled generally from
  the model's linear equalities via the same extraction helpers as the RLT-1 path
  (`_extract_linear_constraints`, `_reconstruct_quadratic_objective`,
  `_mutually_exclusive_pairs`) — no problem-name keying. The falsified plain-Shor
  formulation was not implemented.
- **Root-only, default-off** (`DISCOPT_SHOR_SDP_ROOT_BOUND`), joined by `max` with
  the other candidates in `solver.py::_root_relaxation_lower_bound`; guards
  `DISCOPT_SHOR_SDP_MAX_DIM` (default 400) and `DISCOPT_SHOR_SDP_TIME_LIMIT`
  (default 120 s). SCS is an optional dependency (`discopt[sdp]`); missing solver
  is a sound no-op.
- **The blocking safe-dual requirement is implemented**
  (`shor_sdp_safe_dual_bound`): the surfaced value is never the first-order
  objective but the rigorous weak-duality bound recomputed from the returned
  multipliers — inequality multipliers clamped `>= 0`, dual slack matrix
  `S = C + Σ y1 A + Σ y2 G` assembled in float64 with magnitude-scaled margins,
  and the eigenvalue shift `min(0, λ_min(S) − margin) · tr_ub(M)` with
  `tr(M) = 1 + Σ x_i <= n+1`. Valid for *any* multipliers, so solver convergence
  affects tightness only (the NS-safe property, lifted to SDP).
- **Measured** (`python/tests/test_shor_sdp_root_bound.py` + local runs): the safe
  bound recovers ≥ 99.99 % of the brute-force optimum on synthetic
  Koopmans–Beckmann QAPs n=4/5/6 (e.g. n=6: 2517.99 vs 2518, 0.1 s), never crosses
  it, and survives adversarial dual perturbation. Real-qap root measurement stands
  at the entry experiment's 377098 (86 s, SCS).
- **Default-on graduation** still requires the §5 corpus-wide differential panel
  (cert-clean + net-positive) — not attempted here.
