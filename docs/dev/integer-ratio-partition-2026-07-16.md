# Integer-ratio partition bound (issue #309) — entry experiments and results

**Date:** 2026-07-16
**Issue:** #309 (gear4-class ratio-of-integer-products: certifies but node-heavy)
**Feature flag:** `DISCOPT_INTEGER_RATIO_PARTITION=1` (default OFF — bound-changing
per CLAUDE.md verification regime 2; graduation pending nightly runs)
**Code:** `python/discopt/_jax/integer_ratio.py` (partitioner),
`uniform_relax.py` (`ratio_map` registration), `mccormick_lp.py`
(`solve_at_node` max-combine hook), `solver.py` (flag-gated wiring).

All measurements in this doc were made in a clean container on this branch
(gear4 built via the modeling API from the MINLPLib formulation; the in-repo
corpus has no gear4.nl). Baseline on this machine: **2487 nodes / 56.2 s,
root bound −1e-9**, certifies 1.6434 (issue reports 5921–6045 nodes on other
hardware; the frozen root bound reproduces exactly).

## 1. Why every earlier lever measured zero (context from #309)

The issue's comments had already falsified: branching quality (bound frozen at
0 — nothing to fathom), OBBT (3%), product-column integrality (0), McCormick
ratio linking (0), aux-column spatial branching (0 — identical node count).
The c-MIR/aggregation separator track was separately measured NO-GO
(`cut-engine-entry-2026-07-06.md`).

## 2. Entry experiment — the achievable-ratio lattice (GO)

Hypothesis: for `q = (i1·i2)/(i3·i4)` with integer factors in [12,60], the
*achievable* values of `q` form a sparse rational set, and the LP's frozen
point `q* = R/1e6` sits in a genuine hole of that set.

Measured (1.2 ms enumeration, 893 distinct products):

| quantity | value |
|---|---|
| exact min deviation over root box | **1.643428 = the gear4 optimum** (a=304, b=2107) |
| hole around `q*`: nearest achievable below / above | dev 29.81 / **1.643** |
| candidates with deviation < 45.36 (root incumbent) | **18** |
| min deviation for UNSTRUCTURED integer pairs (a,b) ∈ [144,3600]² | 0.698 |

The last row explains the falsified product-integrality lever: plain integer
pairs approximate the target far too well; the *product structure* is what
makes the set sparse. GO.

## 3. Which relaxation the pieces must be built on (measured)

- **Cleared form (the solve-time reform)**: the factorable reform clears the
  division, so the deviation reaches the objective through `s·(i3·i4)` /
  `t·(i3·i4)` trilinear lifts.
  - The cleared-form lifted LP carries **three unlinked copies of `i3·i4`**
    (two trilinear-stage auxes + the standalone bilinear), each free to sit on
    a different face of the shared McCormick envelope; the LP hides the
    residual there (`q*` = 0.336, not `R/1e6`) and hole rows are inert
    (bound stays 0).
  - Equality-linking the duplicates pins `q* = R/1e6` exactly, and one hole
    disjunction lifts the root bound 0 → **0.455** — but the trilinear
    envelopes still dilute the deviation by ≈ `w23/w23_ub` (0.455 =
    1.643·998/3600). End-to-end (monkeypatch): **1133 nodes**, wall flat.
  - Duplicate-merging alone (no holes) measured **node-neutral** (2497 vs
    2487) — the consistency changes where the LP hides slack, not the bound.
    Not shipped; a possible separate tightening with its own graduation.
- **Pre-reform form (division kept)**: `build_milp_relaxation` lifts the
  quotient to its own column; the hole disjunction is then a plain **column
  bound** on `q` and the deviation reaches the objective **undiluted**:
  piece bounds 29.81 / 1.642 at the root → root bound = the optimum.
  **This is what shipped**: the partitioner builds its piece LPs from the
  pre-reform model (a valid outer relaxation of every node — the node box is
  intersected on the original columns only).

## 4. End-to-end results (gear4, this container, `time_limit=150–180`, tol 1e-4)

| config | nodes | wall | root bound | certifies |
|---|---:|---:|---:|:--:|
| baseline (`main` + this branch, flag off) | 2487 | 56.2 s | −1e-9 | yes |
| cleared-form dive (prototype, not shipped) | 1133 | 62.9 s | 0.455 | yes |
| **pre-reform dive (shipped, flag ON)** | **695** | **47.8 s** | **1.6431** | yes |

Node count −72%, wall −15%, root bound unfrozen from 0 to (optimum − 2.9e-4).

## 5. Known residual: the LP certificate margin (follow-up, not this feature)

The piece LP's raw optimum at the optimal hole edge is exactly the true
deviation (1.6434284739…), but the **rigorous Neumaier–Shcherbina safe bound**
from the Rust simplex duals loses a *bit-identical* 2.8856e-4 on this
1e6-scaled row — measured invariant to piece width, box depth, `s,t` widths,
and geometric-mean equilibration, so it is the dual vector's residual, not
conditioning of the piece. Consequence: the dive bound (1.64314) sits 1.25e-4
below the 1e-4-relative certification threshold at the optimum (1.64326), so
nodes containing the optimal ratio never fathom by gap — certification happens
by exhaustion. Measured ceiling: with the optimal incumbent injected at node 1
and the dive active, gear4 still needs > 2k nodes because *no* node beats the
threshold. **Sharpening the safe bound (e.g. one step of iterative refinement
of the duals before the NS evaluation) would convert gear4 into an
essentially root-solved instance** (root bound = optimum, all children fathom
vs. the incumbent). That is certificate-layer work with its own soundness
gate; tracked in the #309 thread. (POUNCE's `bound` shows margin 0 on the same
piece LP, but its rigor contract is not established — not used.)

A second residual: the dive currently returns bound only. Its enumeration
also produces the best achievable candidate *witness* (factor assignment);
injecting a feasibility-checked witness as an incumbent at the root would
remove the incumbent-latency half of the tree (primal work, same class as
#287/#281).

## 6. Soundness

- Achievable set from **outward-rounded** integer node bounds (superset of
  feasible factor values → every hole genuinely empty of feasible ratios).
- Piece intervals padded outward by a relative 1e-12 against float rounding of
  `a/b` (hole widths here ~1e-5; pad costs ~1e-6 of bound).
- Every piece LP is the model's own lifted McCormick relaxation of the node
  box (pre-reform variable space, original columns only) — a valid outer
  relaxation; the node bound is the min over a partition of a valid
  disjunction. `max(engine, dive)` per node.
- Abstains (returns the engine bound) on: missing `ratio_map` column, any LP
  failure, enumeration cap, empty integer range, all-pieces-infeasible (the
  engine remains the fathoming authority), budget/deadline.
- Detection gates (all must hold): quotient matches `extract_ratio_of_products`,
  all factors scalar integer/binary with finite bounds, ≤ 2 factors per side,
  denominator box strictly positive, root enumeration within cap.
- Regression tests: `python/tests/test_integer_ratio_partition.py` (detection
  gates, root-bound unlock, brute-force soundness on random sub-boxes,
  abstention, default-off, slow end-to-end certification with node ceiling).

## 7. What was run

- `pytest python/tests/test_integer_ratio_partition.py -m "not slow"` — 9 passed.
- `pytest python/tests/test_ratio_of_products_relaxation.py
  python/tests/test_nested_division_soundness.py` — 28 passed (default path).
- `pytest python/tests -m smoke` — see PR description.
- gear4 flag-ON / flag-OFF end-to-end (this doc §4).
