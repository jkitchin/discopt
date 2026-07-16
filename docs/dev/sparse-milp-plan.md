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

- [x] **T0 — differential harness.** `#[cfg(test)] mod sparse_milp_diff` in
  `milp_driver.rs`: a 6-case panel (pure-LP, binary knapsack, general integer,
  infeasible, unbounded, cuts-firing knapsack) with a `Case` owning its data, a
  dense-oracle runner, a `Case::csc()` (from_dense) for T1, and an `assert_same`
  bit-identical gate (status + node-count exact, obj/bound tight). Tests:
  reference-values, determinism (dense re-solve bit-identical incl. lp_iters), and
  CSC-nnz-round-trips-dense. **Done ✓** — 3 tests green; full `cargo test
  -p discopt-core` 449+ pass.
- [x] **T1 — CSC-carrying driver core.** Added `pub fn solve_milp_csc(csc, m, n, c,
  l, u, b, obj_const, opts) -> MilpResult` (the entry the Python binding calls at T4)
  + a `csc_to_dense` bridge. **T1 keeps a dense `a_w` internally** by reconstructing
  the matrix at entry and calling the reference `solve_milp` — so it is provably
  bit-identical but does NOT yet fix the memory blow-up (that is T2/T3, which remove
  the densification). Note: threading the CSC *through* the batched-cuts loop is
  entangled with per-batch scaling (`Scaling::from_matrix(&a_w)` →
  `SparseCols::from_dense(scaled a_w)`) and cut-row appends, so it is correctly
  deferred to T2/T3 rather than forced here. **Done ✓** — new gate
  `csc_entry_matches_dense_on_panel` (status/nodes/obj/bound/incumbent all identical)
  green; full `cargo test -p discopt-core` 450 pass; `discopt-python` builds.
### Gate strengthening (done during T2 setup — §0 kill/re-scope)

The T1 dense-vs-CSC test (`csc_entry_matches_dense_on_panel`) is insufficient as the
driver-wide gate: T2/T3 change the driver internals for BOTH entry points, so after
conversion "dense" and "CSC" move together and that test can't catch a regression
vs the *original* behavior. Added **`driver_matches_golden`** — golden status / obj /
bound / **node count** / **lp_iters** captured from the pre-conversion driver.
`lp_iters` is the sensitive discriminator (a different root-solve pivot path drifts
it even at a single B&B node). Panel note: the current cases all solve at the ROOT
(nodes=1), so `lp_iters` is the real bit-identity signal for T2; consider adding a
genuinely-branching instance before/with T3.

**Invariant decision (user, option A): STRICT bit-identical.** Reproduce
`solve_lp_root` pivot-for-pivot in CSC (keep the `lp_iters`/node-count golden). The
falsification (`solve_lp_cols` skips `ScaledLp`) is resolved by *porting* the dense
mechanisms — de-risked below, they're all bit-identical ports, not rewrites:
- `dual_slack_basis` ALREADY does `SparseCols::from_dense(a)` at its top then uses
  only `sp.col(j)`; switch its signature to `&SparseCols` and drop the densify —
  identical result.
- `Scaling::from_matrix` ALREADY delegates to `equilibrate(&SparseCols,..)`; add
  `Scaling::from_cols(&SparseCols,m,n)` (same min/max trigger over `vals`, same
  `equilibrate`) — identical factors. `Scaling::scale_cols(&mut SparseCols)` exists.
- A CSC `ScaledLp` mirror: `Scaling::from_cols` → `scale_cols` + scale c/l/u/b →
  `solve_lp_cols` → unscale x/dual/ray (exactly what dense `solve_lp` does).

- [x] **T2 — bit-identical CSC root solve.** (a) `dual_slack_basis` → `&SparseCols`
  (it already built one internally); (b) reused the existing `Scaling::from_sparse`
  (identical factors to `from_matrix`); (c) added `solve_lp_cols_scaled` — a
  sparse-native cold solve that mirrors `solve_lp`'s `ScaledLp` (from_sparse →
  scale_cols + scale c/l/u/b → `solve_lp_cols` → unscale x/dual/ray); (d) added
  `solve_lp_root_csc` (dual-slack warm via `solve_lp_warm_scaled_csc`, `solve_lp_cols_scaled`
  cold fallback) and **wired it into the root-cuts loop** (CSC from `from_dense(a_w)`
  per round for now). `solve_lp_root` retained as `#[allow(dead_code)]` differential
  oracle. **Coupling:** dense `a_w` still built (cuts append to it; the loop derives
  the CSC from it) — a_w removal is T3. **Done ✓** — `driver_matches_golden` green
  incl. `lp_iters`; `solve_lp_root_csc_matches_dense` confirms bit-identity on a
  well-conditioned AND an ill-conditioned (1e8-range, equilibration-firing) LP; full
  `cargo test -p discopt-core` 452 pass; `discopt-python` builds.
- [x] **T3a — CSC cut-augmentation primitive.** Added `augment_cols_with_cuts`
  (CSC analogue of `augment_with_cuts`: append `k` cut rows + `k` surplus-slack
  columns, O(nnz+cut_nnz), no dense). `Scaling::from_matrix`'s CSC form already
  exists (`Scaling::from_sparse`, used in T2). **Done ✓** — `csc_augment_matches_dense_augment`
  proves it's nonzero-for-nonzero identical to `from_dense(augment_with_cuts(..))`;
  453 core tests green.
**T3b scope correction (user: full per-node CSC port).** `a_w` (dense) is read by
the WHOLE per-node engine via `node_lp.a`/`prop_lp.a`/`ctx.sa`, not "a few
consumers": `struct_nnz`, `try_rounding`, `farkas_safe_bound`, `separate_cover`,
`strong_branch`, and node propagation (FBBT). The panel solves at the root (nodes=1)
so it can't gate these — instead **keep each dense function as the oracle and add a
DIRECT differential unit test** (`fn_dense(dense,..) == fn_csc(csc,..)` on crafted
matrices + fractional points). All ports are bit-identical by the same argument: a
structural `0.0` adds exactly, and CSC preserves ascending row order, so `Aᵀy`
(`csc.dot`) and row activities match the dense sums term-for-term.

- [x] **T3b1 — trivial ports.** `struct_nnz` → `col_ptr[ns]`; `try_rounding` and
  `farkas_safe_bound` → `&SparseCols` (column-iteration into `act/lo/hi`;
  `csc.dot(j,y)`). Keep dense versions as oracles. **Done:** differential unit tests
  (dense == csc) green.
- [x] **T3b2 — `separate_cover` → CSC** (+ its dense oracle + differential test on a
  crafted fractional node).
- [x] **T3b3 — `strong_branch` → CSC.** Finding (mini-falsification): `strong_branch` is ALREADY CSC-based — it solves only via `PreparedDual::prepare`/`solve_lp_warm_scaled_csc`, which take `ctx.csc` and never read `LpView.a`, and it has no other matrix access. So no port is needed; its two `a: ctx.sa` fields were vestigial → set to `&[]` (removing its dense dependency). **Done ✓** — full `cargo test -p discopt-core` (456) green with the empty `.a` (runtime proof it's unread); the integration gate where `strong_branch` actually fires is T3b6.
- [x] **T3b4 — FBBT `tighten_bounds` → CSC.** Refactored FBBT so the matrix touch
  (gather a row's nonzeros) is split from the shared per-row propagation `fbbt_row`
  (activity ranges + infinity bookkeeping + integer rounding). Dense scans the row;
  new `tighten_bounds_csc` builds per-row nonzeros once from the CSC (reused across
  rounds). Bit-identical (zeros skipped; per-column tightening order-independent as
  the sums are fixed from loop 1). **Done ✓** — 191 existing presolve tests gate the
  dense refactor; new `csc_matches_dense_fbbt` (tightening + infeasible box) gates the
  port; 457 core tests green.
- [x] **T3b5 — driver rewire + remove `a_w`.** DONE — the MILP driver is fully sparse.
  Built csc_w before presolve; presolve→tighten_bounds_csc; root-cuts loop→solve_lp_root_csc
  + separate_cover_csc + separate_gomory_cols + augment_csc_with_cuts; per-batch scaling via
  Scaling::from_sparse + scale_cols (csc_batch scaled, csc_rc unscaled); NodeCtx drops a_w/sa;
  per-node consumers → CSC ports (FBBT/rounding/cover→csc_rc, farkas→csc); dense cold
  fallbacks solve_lp_scaled→solve_lp_cols(ctx.csc). refactored separate_gomory→separate_gomory_cols.
  **Done ✓** — driver_matches_golden AND branching_golden (nodes=13) green; 458 core tests +
  discopt-python build. Dense oracles retained (allow(dead_code)) for the differential tests.
  ORIGINAL: Carry the working matrix as
  `SparseCols` through `solve_milp_hooked`: build base CSC, append cuts via
  `augment_cols_with_cuts` (+ the `b/c/l/u/is_int` appends), scaling via
  `Scaling::from_sparse`+`scale_cols`, `NodeCtx` carries `csc`/`csc_rc` (drop `a_w`,
  `sa`; ignored `LpView.a` fields → `&[]`), switch all consumers to the CSC ports.
  Delete `a_w = lp.a.to_vec()` and every dense-matrix site. **Done:**
  `driver_matches_golden` green (incl. `lp_iters`); all T3b differential tests green;
  `a_w` no longer constructed; dense-entry path bit-identical.
- [x] **T3b6 (gate strengthening) — branching integration golden.** DONE FIRST (gate-first, before T3b5). Small 6-binary knapsack solved with heuristics off + no root GMI (forces a 13-node tree) but node cover separation + strong branching + FBBT propagation ON — so `separate_cover`/`strong_branch`/`tighten_bounds` (and their CSC ports) run end-to-end. Golden: Optimal, obj −16, **nodes=13, lp_iters=39**. This gates the T3b5 rewire (the nodes=1 panel can't). **Original T3b6 note:** Add ≥1 small MILP
  that genuinely branches (source from the `.nl` corpus or craft with cuts/heuristics
  off) so `driver_matches_golden` exercises `separate_cover`/`strong_branch`/
  propagation end-to-end, not just nodes=1. Capture status/obj/nodes/`lp_iters`.
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
