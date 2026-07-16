# Sparse MILP path for the Rust B&B driver

## §0 Loop protocol (binding — this doc is loop-executable)

Execute the checklist in **§Tasks** strictly in order, one task per loop iteration.
Each iteration:

1. Pick the **first unchecked** `[ ]` task in §Tasks. Do only that task.
2. Implement it. Touch only what the task names; keep the dense `solve_milp` intact
   as the reference oracle (do NOT delete it) until T5 is green.
3. **Verify before committing** — a task is not done until:
   - `cargo build -p discopt-core` and `cargo build -p discopt-python` succeed;
   - `cargo test -p discopt-core` passes (and the differential harness from T0 once
     it exists);
   - the task's own **Done** criterion below holds.
   If verification fails, fix it in the same iteration; do not advance.
4. Commit with `feat(sparse-milp): Tn <title>` (or `test:`/`docs:`), then check the
   box `[x]` in this file **in the same commit** and record the one-line result.
5. **Kill criterion:** if a task reveals the plan is wrong (e.g. a node solve DOES
   read `LpView.a`, or `from_csc` can't represent a needed op), STOP, write the
   falsification under the task, and re-scope §Tasks before continuing — the
   measurement wins (CLAUDE.md §4).

**Never advance on red.** The invariant (§Invariants) is bit-identical-to-dense;
any node-count or certified-objective drift on the T0 panel is a bug in the CSC
assembly, never a new relaxation. Stop and root-cause.

## Tasks

- [ ] **T0 — differential harness.** Rust test `bnb::milp_driver` (or a new
  `sparse_milp_diff` test module): a panel of small MILPs — pure-LP, 1–3 binaries,
  infeasible, unbounded, a cuts-firing instance — each solved via dense `solve_milp`
  and (stubbed for now) asserted against itself; wire the CSC path in at T1.
  **Done:** panel builds + passes against the dense path; committed.
- [ ] **T1 — CSC-carrying driver core.** `solve_milp_csc(csc: SparseCols, m, n, c,
  l, u, b, obj_const, opts)` that threads `csc` where `SparseCols::from_dense(&a_w)`
  is today and passes zero-length `.a` to node `LpView`s. Root solve / cuts /
  scaling may still build a dense `a_w` from the CSC *internally* for now (correct,
  not yet memory-fixed). **Done:** T0 panel bit-identical dense vs CSC path.
- [ ] **T2 — sparse root cold solve.** Replace `solve_lp_root`'s
  `dual_slack_basis(lp.a,..)`+`solve_lp` with a CSC cold two-phase (adapt the
  `solve_lp_cols_*` warm-fallback family to a cold entry). **Done:** root bound
  unchanged on the T0 panel; no dense `a_w` touched on the root path.
- [ ] **T3 — sparse scaling + cut-row append.** `Scaling::from_matrix` CSC variant;
  GMI cut rows appended to the CSC (grow col_ptr/row_idx/vals) instead of `a_w`.
  **Done:** cuts-firing panel instance bit-identical; `a_w` no longer constructed.
- [ ] **T4 — Python binding + routing.** `solve_milp_csc_py(col_ptr, row_idx, vals,
  m, n, c, l, u, int_cols, ...)`; route `solvers/milp_simplex.py` and the
  `MilpRelaxationModel` MILP path to pass the relaxation's existing scipy CSC
  through — delete the `A.toarray()`/dense `a_std` assembly. **Done:** a small
  binary-QP solves end-to-end via the CSC binding, bit-identical bound to dense.
- [ ] **T5 — end-to-end + certificate gate.** `_root_relaxation_lower_bound(qap)`
  and a full `solve_model(qap)` honor `time_limit` (no 73 GB, no overrun);
  `incorrect_count == 0` on the global50 panel; node counts bit-identical to dense
  on the T0 panel. **Done:** qap honors a real budget; certifying panels green;
  retire the dense path only after consecutive nightly greens (separate task).

## Problem (measured on qap — a 225-binary Quadratic Assignment Problem)

`solve_milp_py` (the Rust MILP entry) takes a **dense** `a: PyReadonlyArray2`, and
`bnb::milp_driver::solve_milp` carries a **dense** working matrix
`a_w = lp.a.to_vec()` (milp_driver.rs:351). qap's McCormick relaxation LP is
`85 756 × 21 649` with **172 292 nonzeros** (~2/row), but the dense form is
`a_std = zeros(85 756, 107 405)` ≈ **73 GB**. Consequences:

* Python side (`solvers/milp_simplex.py:solve_milp`): `A.toarray()` + `np.eye(m)`
  build the dense `a_std` — the ~12.9 s "marshaling".
* Rust side: `a_w = lp.a.to_vec()` copies it; `SparseCols::from_dense(&a_w)`
  (milp_driver.rs:650) iterates the whole dense matrix; the solve overruns its
  `time_limit_s` (measured `_root_relaxation_lower_bound(qap, tl=5)` > 120 s for a
  ~1 s budget).

The sparse LP path (`solve_lp_warm_csc_py`) is fine (qap LP relaxation solves in
0.5 s) but is **LP-only**; there is no sparse MILP.

## Key facts that make this bounded (verified in source)

1. `SparseCols::from_csc(col_ptr, row_idx, vals)` exists (lp/simplex/sparse.rs:69)
   — build the driver's matrix directly from CSC, no dense intermediate.
2. The **per-node** solver `solve_lp_warm_scaled_csc` (lp/simplex/dual.rs:172) uses
   only the `SparseCols` for the matrix and `LpView` for `c/l/u/b` — it never reads
   `LpView.a`. So node LP solves are ALREADY fully sparse; the driver just also
   keeps a redundant dense `a_w`.

So the ONLY blocker is the dense `a_w`. Its consumers in `milp_driver.rs`:

| line(s) | use of dense `a_w` | sparse replacement |
|---|---|---|
| 351 | `a_w = lp.a.to_vec()` working matrix | carry a `SparseCols` (from `from_csc`) |
| 399 | per-row structural-nnz counts | CSC column/row counts |
| 441, 1042, 1086 | `LpView{ a: &a_w, .. }` for solves | node solves ignore `.a`; pass `&[]`/dummy; root needs a CSC root solve |
| 508, 828 | GMI cut rows appended to `a_w` | append rows to the CSC (grow col_ptr/row_idx/vals) |
| 636–650, 657 | `Scaling::from_matrix(&a_w)`, `SparseCols::from_dense(&a_w)` | scaling from CSC; drop `from_dense` |
| 2124 (`solve_lp_root`) | cold root solve via `dual_slack_basis(lp.a,..)` + `solve_lp` | CSC cold root solve |
| 1142, 1252 | reduced-cost fixing reads `a_w` | already has `csc_rc`; use it |

## Plan (incremental; each step compiles + `cargo test -p discopt-core` green)

0. **Harness first.** Add a Rust test: a small sparse MILP solved via the current
   dense `solve_milp` AND the new CSC path must return the identical status /
   objective / bound / node count on a panel (pure-LP, a few binaries, infeasible,
   cuts-firing). This is the bit-identical gate — the CSC path is a *representation*
   change, not an algorithm change, so any drift is a bug.
1. **CSC-carrying driver.** Introduce `solve_milp_csc(csc: SparseCols, m, n, c, l,
   u, b, obj_const, opts)`. Internally keep today's dense `solve_milp` as the
   reference; the CSC one threads `csc` where `from_dense(&a_w)` was and passes a
   zero-length `.a` to the node `LpView`s (verified unused).
2. **Sparsify the root cold solve** (`solve_lp_root`): a CSC cold two-phase (the
   `solve_lp_cols_*` family already exists as the warm fallback — expose/adapt a
   cold entry).
3. **Sparsify scaling + cut-row appends** (CSC row append; `Scaling::from_matrix`
   CSC variant).
4. **Python binding** `solve_milp_csc_py(col_ptr, row_idx, vals, m, n, c, l, u,
   int_cols, ...)` + route `solvers/milp_simplex.py` (and the `MilpRelaxationModel`
   MILP path) to pass the relaxation's existing `scipy.sparse` CSC straight through
   — deleting the `A.toarray()` / dense `a_std` assembly.
5. **End-to-end**: `_root_relaxation_lower_bound(qap)` and a full `solve_model(qap)`
   honor `time_limit`; `incorrect_count == 0` on the global50 panel (certificate
   invariant) and node counts bit-identical to the dense path on the step-0 panel.

## Invariants (CLAUDE.md §5, §1)

* Bit-identical to the dense path on every certifying instance — this is a pure
  representation change (same pivots, same B&B tree, same cuts). Node count and
  certified objective must be EXACTLY unchanged; any drift means the CSC assembly
  diverged and is a bug.
* Never weaken a validation to make it pass. The dense path stays as the reference
  oracle in the test harness (not deleted) until the CSC path is green on
  consecutive nightlies.
