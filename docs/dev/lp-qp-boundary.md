# The LP/QP Rust–Python boundary

Status: **binding contract** (§0) plus the audit that motivated it (§2).
Written 2026-09-12. Consolidation is tracked by issue #1230 (§3 is its task list).
Scope: the LP, QP, MILP and MIQP solve paths only — the
MINLP/spatial paths have their own producer contract (`_relax/spatial_producer.py`,
`solvers/_convex_kernel.py`) and are out of scope here — their separation is
tracked by issue #1231.

## §0 The line (binding)

> **Python turns a `Model` into a problem. Rust turns a problem into a certified
> result. Nothing that enters the certificate is computed in Python.**

Operationally, for every LP/QP/MILP/MIQP solve:

| Python owns (the *producer*) | Rust owns (the *solver*) |
|---|---|
| problem classification (`classify_problem`) | scaling / equilibration |
| extraction of `c, Q, A, b`, bounds, integrality | standard-form assembly (slacks, row senses) |
| variable flattening and column layout | LP/MILP/QP presolve |
| objective sense normalization | simplex / IPM, basis management, warm start |
| the option/tuning dict | cut separation **and** selection |
| unmarshaling a returned result into `SolveResult` | the safe dual bound (Neumaier–Shcherbina) |
| user callbacks | feasibility / KKT verification of its own output |
| logging, telemetry, deadlines | status determination (optimal / infeasible / unbounded) |

**The decision rule, stated so it is testable:** if a function's output can change
when floating-point arithmetic changes — rounding mode, summation order, a
margin constant — it belongs in Rust. Python may *read* numbers the solver
produced; it may not produce numbers the certificate depends on.

Two corollaries that are the whole point of the rule:

1. **One crossing per solve, not per node.** The producer runs once. A node LP is
   never assembled, scaled, decomposed or repaired across the FFI.
2. **The certificate has exactly one implementation.** There is one safe-bound
   function in the tree. Not one per backend, not one per call site.

### What the producer is allowed to compute

Structural facts that are exact by construction and carry no floating-point
error: array shapes, column offsets, integrality masks, which constraints exist
and what sense they were *declared* with. The producer may not *re-derive* any
of these from a matrix it just built (see V3).

## §1 Why this line and not another

The obvious alternative — Python assembles the node LP, Rust solves it — is the
line the tree has today, and it is the wrong one for three reasons that are
already measured in this repo:

* **Cost.** `bnb/spatial_kernel.rs`'s header records the entry experiment: a
  Python-assembled node on `tanksize` is ~1352 ms, of which ~99.99 % is
  orchestration that vanishes when the node stays in Rust.
* **Duplication.** A per-node boundary forces every numeric that touches a node
  to exist on both sides, because both sides run nodes. §2 is the bill.
* **Certification.** A bound is only as sound as its weakest implementation. Five
  implementations means five chances to be wrong and no single place to fix it.

## §2 Audit: where the line is crossed today (measured 2026-09-12)

### V1 — the safe dual bound has five production implementations

The Neumaier–Shcherbina bound `g(y) = bᵀy + Σ_k min_{z_k∈[l,u]} (c−Aᵀy)_k z_k` is
the certificate. It exists at:

| # | location | form | precision | error margin | open-bound handling |
|---|---|---|---|---|---|
| 1 | `_relax/obbt.py:60` `_ns_safe_lp_lower_bound` | `A_ub x ≤ b`, `n_eq` tail | float64 | magnitude-scaled | `rc_snap_tol` snap |
| 2 | `solvers/milp_simplex.py:365` `_safe_lp_lower_bound_std` | `Az = b` | float64 | `1e-9·(1+\|bᵀy\|+Σ\|contrib\|)` | FBBT recovery, then abstain |
| 3 | `solvers/milp_simplex.py:137` `_safe_lp_lower_bound_sharp` | `Az = b` | float64 | provable forward-error | abstain on sign-uncertain |
| 4 | `lp/simplex/refine.rs:217` `ns_safe_bound` | `Az = b` dense | double-double | **none** | abstain |
| 5 | `lp/simplex/refine.rs:268` `ns_safe_bound_csc` | `Az = b` CSC | double-double | **none** | abstain |

Two more sit in the same family and cross the same line:

* `solvers/milp_simplex.py:431` `_safe_lp_lower_bound` — a dispatcher that picks
  #2 or #3 off `SolverTuning.ns_sharp_margin`, so *which* rigor model applies is a
  runtime tuning decision made in Python.
* `solvers/milp_simplex.py:461` `_refined_safe_bound_regularized` — a Python
  implementation of the regularized dual sweep whose Rust twin is
  `lp/simplex/refine.rs` itself. The GSW refinement idea therefore exists on both
  sides too.

(`_relax/shor_sdp.py:313` `shor_sdp_safe_dual_bound` matches the same detector but
is a different problem class — SDP, not LP — and is out of scope for this contract.
`lp/simplex/primal.rs:3589` is a `#[cfg(test)]` reference implementation: the Rust
tests re-derive the formula rather than calling the production one, so a defect in
#4/#5 would not be caught by its own test.)

**Measured divergence.** `scratchpad/ns_diff.py` (180 random LPs, three
conditioning regimes, HiGHS duals via `scipy.linprog`, same LP handed to #1 and
#2 in their respective forms):

```
executed comparisons: 132
abstained (None): obbt=47  milp_simplex=35
UNSOUND (g > p*): obbt=0  milp_simplex=0
disagreements (rel > 1e-9): 111
  disagreement rel: median=1.10e-09 max=2.42e-03
```

Read this carefully, because it says two different things:

* **Neither is unsound.** 0/132 violations of `g ≤ p*`. This is not a live
  false-certificate bug and must not be reported as one.
* **They are not the same function.** They disagree on 111/132 (84 %), and the
  three largest disagreements — 2.4e-3, 8.4e-4, 6.3e-4 relative — are all in the
  **ill-conditioned** regime, i.e. exactly the class the NS machinery exists to
  serve. The abstention rates differ too (47 vs 35): on ~12 LPs one path
  certifies a bound and the other returns nothing.

So which dual bound a node gets depends on which code path reached it. That is
the cost of V1 stated precisely, and it is a bound-quality and
reproducibility defect rather than a soundness one.

**Unverified concern, recorded as a hypothesis, not a finding.** #4/#5 compute
`bᵀy` and each `(Aᵀy)_j` in double-double but then accumulate the `n` box terms
in plain `f64` (`g += rc * l[j]`, `refine.rs:236-249`) and subtract **no**
margin, where #1–#3 evaluate in float64 and subtract one. Whether the residual
`O(n·ulp)` accumulation can push `g` above `p*` on a wide-ranged LP is untested.
The falsifying experiment: evaluate #4 against an exact (rational or DD-throughout)
reference on the `hda`/`qap`-class LPs and count `g > p*`. Until that runs this is
a question, not a defect.

### V2 — LP equilibration has three implementations, two of them in Python

* Python: `_relax/milp_relaxation.py:182` `equilibrate_relaxation_lp` (20 iterations,
  row+column)
* Python: `_relax/obbt.py:185` `_equilibrate_rows` (row-only,
  `d_i = 1/max(‖A[i,:]‖_∞, |b_i|, 1)`, for OBBT projection LPs)
* Rust: `lp/simplex/scaling.rs:288` `equilibrate`

`milp_relaxation.py:749` picks between the first and the third: `if backend !=
"simplex" and self._A_ub is not None:` — so the *same relaxation LP* is scaled by
different code depending on which engine will solve it, and an OBBT projection of
that same LP is scaled by a third. Scaling changes which vertex the simplex lands
on and therefore the reported bound, so this is a second source of path-dependent
numbers.

### V3 — the standard form is built in Python, then re-derived from itself

`solvers/lp_simplex.py:197-221` assembles `[A_ub | I ; A_eq | 0 I]` as CSC in
Python, including the slack bound vectors. `solver.py:19349`
`_decompose_eq_slack_form` (and its sparse twin at `:19289`) then *reverses* that
transform — and does so by **numerically inspecting the slack block**: a row is
classified as an inequality iff its slack entry has `abs(v) > 1e-15`, with the
sign used to orient the row.

This is called at **12 sites**, all in `solver.py` (`:19379` — the sparse twin —
then `:19641, :20187, :20376, :20702, :20834, :20916, :21386, :22447, :22816,
:23601, :24073`). Constraint sense is a structural fact the producer knew exactly
and discarded; re-deriving it from float comparisons is the boundary inverted.
`_solve_qp_matrix` additionally densifies via `_dense_A` before decomposing
(`solver.py:20187`).

### V4 — Python repairs the solver's numerical output

* `solvers/lp_simplex.py:46,269-273` — `_BOUND_SNAP_TOL = 1e-3`; solutions
  outside their box by up to 1e-3 are snapped back onto it in Python.
* `solvers/lp_pounce.py:56,202-213` — `_BOUND_SNAP_TOL = 1e-7`, a *different*
  repair wearing the same name: a `lb > ub` inversion up to 1e-7 is collapsed to
  its midpoint on the way **in**, because POUNCE rejects an inverted bound that
  HiGHS would have presolved away. One name, two directions, tolerances four
  orders of magnitude apart — a reader who learns one meaning will misread the
  other.
* `solver.py:20884` `_solve_node_lp_simplex` — re-checks the returned point
  against its own rows and bounds in Python and rejects it.
* `solver.py:~20245` `_solve_qp_matrix` — re-checks primal feasibility and a KKT
  residual in Python and refuses the point.

Each guard is individually defensible (they were added against real
false-certificate incidents) and **none should be deleted before its Rust
equivalent exists**. But a solver that cannot be trusted to validate its own
output has the validation on the wrong side of the boundary: the check belongs
next to the arithmetic it is checking.

## §3 Migration order (tracked by #1230)

Ordered by (certificate risk × cost to fix), not by size:

1. **V1 → one implementation.** Land the single sound NS evaluation in Rust,
   expose it, and make #1–#3 thin callers of it. This is **bound-changing** — the
   measurement above shows the reported bound moves by up to 2.4e-3 relative — so
   it needs the CLAUDE.md §5 bound-changing regime: differential bound test plus a
   corpus-wide flagged panel, cert-clean and net-positive, before the default flips.
   Run the V1 hypothesis experiment (above) first; it may change the target.
2. **V3 → delete the round trip.** Have the producer emit sense/row metadata
   alongside the matrices so nothing is re-derived, and let Rust own standard-form
   assembly. This is **bound-neutral** and should be verified as such: identical
   `node_count` and objective on a certifying panel. Cheapest large win — it
   removes 12 call sites and two functions.
3. **V2 → Rust only.** Drop `equilibrate_relaxation_lp` once every LP consumer
   routes through an engine that scales internally. Bound-changing for the
   non-simplex backends; bound-neutral for `backend="simplex"`.
4. **V4 → move the guards down.** Port each check into the engine that produces
   the point, keeping the Python guard until its replacement is proven, then
   removing it in the same PR that proves it. Never remove a guard and add its
   replacement in separate PRs.

## §4 Enforcement

`python/tests/test_lp_qp_boundary.py` is a ratchet over §2: it pins the current
inventory of boundary violations and fails when one **grows**. It does not
require the built `_rust` extension (it reads source), so it runs in any
environment.

Shrinking the inventory is the goal, and the test tells you so — removing a
violation fails the test with an instruction to update the pinned counts in the
same PR. The counts only ever go down.
