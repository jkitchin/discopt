# Decomposition Module Review — Soundness, Correctness, SOTA

**Date:** 2026-07-03
**Scope:** `python/discopt/decomposition/` (34 files) — Benders (classical +
GBD/Geoffrion), Lagrangian (subgradient + bundle), the Lagrangian B&B node-bounder,
structure detection, and the decomposition advisor.
**Method:** Delegated verification — all load-bearing files read, cut/bound math
traced, then ~400 randomized adversarial differential tests (decomposition vs.
monolithic `Model.solve` and brute-force oracles). The S1 edge re-confirmed here.
33 decomposition tests pass.

**Bottom line: the decomposition layer is unusually sound — no false-optimal and no
invalid *certified* bound was producible across ~400 instances.** The cut/bound
constructions use complete-dual (Benders) and projected-multiplier Lagrangian-dual
(GBD) forms that stay valid for any dual-feasible point — robust to inexact
interior-point primals. Findings are latent edge cases, none a demonstrated false
certificate.

---

## 1. Findings (all SUSPECTED; none a demonstrated false certificate)

> **◻︎ DC-S1 — NOT-REPRODUCED on the current tree (already fixed by PR #409).**
> The finding below was written against the pre-#409 solver. PR #409 ("Decomposition
> module remediation", 2026-07-03 — the same day as this review) added BOTH guards
> DC-S1 asks for: (1) distinct `unbounded`-recourse detection — an unbounded-below
> block returns `status="unbounded"`, `bound=None` (`benders/solver.py:489-490,574-575`),
> not a `_ETA_FLOOR` bound; and (2) the T0.5 eta-floor-withholding guard
> (`:632-639`) — if any master `eta` still rests on `_ETA_FLOOR` at the final master
> solve, the `bound` is **withheld** (`bound=None`), covering the "bounded problem
> whose true optimum is below −1e12" sub-case. Re-verified (#413) on the exact repro
> `min y−x, x∈[0,1e30], x≥y` → `status="unbounded", bound=None, gap_certified=False`,
> and on the bounded-below-floor case `min y+x, x∈[−5e12,0]` → `bound=None` withheld,
> incumbent −5e12 correct — **no invalid populated `bound` in either.** No solver code
> changed; regression tests `test_benders_soundness.py::test_dcs1_*` added to pin it
> (alongside the pre-existing `test_c3_unbounded_recourse_reported`). DC-S2/DC-S3
> remain as carded (defensive P3s).

| # | Severity | Loc | Finding |
|---|----------|-----|---------|
| DC-S1 | P1 latent — ◻︎ NOT-REPRODUCED (fixed by #409) | `benders/solver.py:294,280-333,447-453,60` | **Unbounded recourse yields a populated, invalid `bound`.** An unbounded-below recourse LP is treated like infeasible (feasibility cuts), so the solve exits `iteration_limit` with `objective=None` and `bound = _ETA_FLOOR (-1e12)`. When the true optimum is below −1e12 the reported `bound` **exceeds** the true optimum (violates `bound ≤ true opt`). Not a false certificate (`status≠optimal`, `gap_certified=False`), and the floor is documented as an assumption — but there is **no runtime detection** of unbounded recourse and no guard withholding `bound` when the assumption breaks [CONFIRMED pre-#409: `min y−x, x∈[0,1e30], x≥y` → `bound=-1e12`, true ≈ −1e30]. **RESOLVED by PR #409** (see banner above): unbounded recourse now reports `status="unbounded"`/`bound=None`, and the T0.5 guard withholds the bound whenever `eta` sits on the floor. |
| DC-S2 | P3 defensive | `benders/solver.py:415-419,455-456` | If the master MILP returns `INFEASIBLE` mid-loop after an incumbent exists, it drops the incumbent and returns `status="infeasible", gap_certified=True`. Unreachable with valid cuts (every cut keeps `(x̂,Q(x̂))` master-feasible), so only a backend misreport triggers it — but no assertion guards `best_ub=inf` before declaring infeasible |
| DC-S3 | P3 consistency | `lagrangian/solver.py:238-266` | The `gnorm2<1e-16` early exit can set `status="optimal", gap_certified=True` with `objective=None` if primal recovery failed at that iterate. The `bound` is numerically valid (residual≈0 ⇒ feasible+optimal subproblem), so not an invalid bound — but "optimal" with no returned primal is inconsistent |

## 2. Verified sound (not bugs)

- **Benders complete-dual cut** (`_dual_const`, `_add_opt_cut`, `_add_feas_cut`):
  sign conventions and the `|bound|≥_BIG` sentinel guard correct; validated on
  active-bound and equality-coupling adversarial cases.
- **GBD** (`gbd.py`): projected multipliers are dual-feasible for discopt's
  one-sided/equality constraints (no ranged constraints exist), so `μ·g ≤ 0` and
  the closed-form box term keep the anchor ≤ primal. Convexity gate **withholds**
  `bound` on nonconvex recourse (verified: `bound=None, gap_certified=False`);
  unbounded-descent recourse downgrades to heuristic.
- **Lagrangian**: `L(λ) = sub_lb − λ·r_c + offset` uses the *rigorous* subproblem
  dual bound; subgradient sign/step and bundle/Kelley model correct; equality
  coupling split into two `≤` rows with `λ≥0` is valid.
- **Lagrangian B&B node hook** (`node_bounder.py`): minimization-only gate,
  combined via `max()` with `result_lbs` (never weakens); the `< 1e29` sentinel
  gate correctly means "real bound"; objective-offset/mixed-continuous alignment
  verified end-to-end — no invalid global bound.
- **Structure detection / advisor**: auto-detection can only *weaken* Lagrangian
  bounds (any dualized set gives a valid relaxation); Benders/GBD reject
  integer-in-recourse **loudly** (`NotImplementedError`); the advisor is
  analysis-only — `Model.solve()` never silently applies a decomposition (opt-in
  `decomposition=`/`lagrangian_bound=` only); `assert_sound()` refuses `HEURISTIC`.

## 3. SOTA

At or above the correctness bar of commercial/open-source Benders. CPLEX/Gurobi
generate cuts from LP subproblem duals (classical, exact for MILP-with-continuous-
recourse); discopt's **complete-dual** cut (row duals + reduced-cost bound terms)
is a deliberate hardening that stays valid under interior-point/analytic-center
duals and inexact primals — a real robustness edge on non-simplex backends. The GBD
path mirrors Geoffrion (1972) but anchors on the Lagrangian-dual value rather than
the NLP primal (stronger than textbook GBD), correctly gated on a convexity
classifier. Gaps vs. PySP/mpi-sppy are **performance/scale, not correctness**: no
scenario-block exploitation or parallel subproblem solves in the bounding loop
(the `parallel/` layer exists but drivers solve serially), no multi-cut/level-bundle
stabilization, and no in-tree branch-and-Benders-cut (cuts regenerated per
standalone solve rather than embedded as lazy constraints in one global tree).

## 4. Plan (for Opus)

Single PR `fix(decomposition): DC-S1..DC-S3` — the priority is **DC-S1**: detect
`UNBOUNDED` recourse status distinctly and report `status="unbounded"` / `bound=None`
instead of the `_ETA_FLOOR` bound (also guard any bounded problem whose true optimum
is below −1e12). Add a guard asserting `best_ub` is infinite before declaring
`infeasible` (DC-S2), and either recover a primal or report a non-"optimal" status
when the primal is absent (DC-S3). Each with an adversarial regression test
(unbounded recourse; post-incumbent master-infeasible; residual-zero-no-primal).
The performance/scale items (parallel subproblems, multi-cut, branch-and-Benders-cut)
are separate design work, not correctness.
