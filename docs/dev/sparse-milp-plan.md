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
- [x] **T4 — Python binding + routing.** RUST HALF: `solve_milp_hooked` takes a
  `SparseCols` (dense `solve_milp`/`solve_milp_py` build it from `lp.a`; no internal dense
  copy). Added `solve_milp_csc_py` (col_ptr/row_idx/vals input → `from_csc` → the sparse
  driver, NEVER densified) + a shared `run_milp_hooked`, registered in lib.rs. core 458
  tests green. PYTHON HALF (this iter): `solvers/milp_simplex.py::solve_milp` now builds
  the standard-form `[A_ub | I]` as a **scipy CSC** (`sp.hstack`, `sort_indices`) and calls
  `solve_milp_csc_py` — the dense `a_std = np.zeros((m, n+m))` + `np.eye(m)` + `A.toarray()`
  assembly is deleted. `MilpRelaxationModel.solve(backend="simplex")` already passes its
  sparse `_A_ub` straight through (it is a CSR, ~2 MB for qap — verified). **Done:** all 39
  `test_milp_simplex.py` tests green (incl. the pure-LP-machinery-off gate, respied on the
  CSC binding); `pytest -m smoke` 653 green; adversarial suite 10 green — the routing is
  bound-neutral (no node-count / objective drift on the certifying panels).
- [x] **T5 — end-to-end measurement + FALSIFICATION (re-scope).** The MILP driver is now
  fully sparse and materializes no dense `m×(n+m)` matrix (T0–T4 done, MILP path
  bit-identical). **But the plan's premise — "making the Rust MILP driver sparse stops
  qap's ~73 GB densification" — is FALSIFIED by measurement.** Entry experiment (qap.nl,
  RSS watchdog, `/usr/bin/time -l`):
  - qap's `MilpRelaxationModel._A_ub` is **already a sparse CSR** (85756×21649, ~2 MB) — not
    dense. The 73 GB figure was the *hypothetical* dense driver footprint, never actually
    allocated (the `_MAX_RELAX_DENSE_CELLS = 1e8` guard in `mccormick_lp.py` declines qap's
    node LP at `(n_cols+n_rows)·n_rows = 9.2e9 > 1e8` **before** any solve).
  - Measured peak RSS on a full `solve(time_limit=25s)` is **~28–30 GB** — and it is the
    **same ~30 GB whether the guard is ON (every node LP declined, no simplex solve) or
    OFF**. So qap's real memory ceiling is **independent of the LP solve and of the Rust
    driver**: it is the **Python McCormick relaxation build** for the 85,756-row lift
    (21,424 bilinear envelopes), a transient the sparse driver does not touch.
  - **Consequences / re-scope:** (1) The sparse-driver work is correct and necessary but does
    **not** by itself make qap tractable. (2) **Next task:** root-cause and fix the Python
    McCormick relaxation build memory for large lifts (the ~30 GB transient).

- [x] **T6 — ROOT-CAUSED + FIXED the ~30 GB (commit `fad92ab6`).** Stack-sampling the relaxer
  build (`faulthandler.dump_traceback` triggered by an RSS watchdog) placed the entire blowup
  in `IncrementalMcCormickLP._build_structure`, **not** the cold build and **not** `_A_ub`:
  - The incremental per-node fast path stores **`base_A` DENSE** (`rows×cols`): qap's
    85756×21649 lift is `.todense()` → **14.85 GB**, then `self.base_A = A.copy()` holds a
    **second 14.85 GB**, and `_patch()` does `A = self.base_A.copy()` — a **fresh 14.85 GB
    dense array on every node**. The fast path is `O(rows·cols)` memory, per node.
  - **Fix:** guard the incremental structure by dense-cell budget
    (`_MAX_INCREMENTAL_DENSE_CELLS = 1e8`, ~0.8 GB) **before** the `.todense()`; above it,
    raise → caught by the existing `__init__` fallback (`ok=False`) → `solve_at_node` uses the
    sparse per-node cold build. Sound: the fast path is only an accelerator whose rows are
    `_validate`d bit-identical to the cold build, so declining it changes speed, never the
    bound. General (mirrors `_MAX_RELAX_DENSE_CELLS`), not instance-keyed. Regression test
    `test_incremental_declined_when_lift_too_large_for_dense`; 230 mccormick/incremental +
    653 smoke green.
  - **Measured (peak RSS):** relaxer construction 30 GB → **0.38 GB**; full solve with
    production guards 30 GB → **0.86 GB**; full solve with `_MAX_RELAX_DENSE_CELLS` lifted so
    the sparse simplex actually solves qap's 85756-row node LP → **0.88 GB, bound sound**
    (−1e-9 ≤ oracle 388214). **The basis-LU does NOT blow memory** — falsifying the T5
    "unquantified fill-in risk" worry. The remaining issue is per-node LP **time** (8 s budget
    → ~20 s wall on the 85k-row McCormick LP), now a **profiling target**, not a memory
    blocker. The `_MAX_RELAX_DENSE_CELLS` guard is still left intact by default (relaxing it —
    to let qap earn a real McCormick LP bound instead of alphaBB — is a **bound-changing**
    change per §5: needs the differential-bound test + feature flag + nightly greens, a
    separate task). Certificate gate unaffected. Retire dense oracles only after nightly-green.

- [ ] **T7 — relax `_MAX_RELAX_DENSE_CELLS` for the sparse backend: ENTRY EXPERIMENT KILLED
  IT (§4).** Before implementing the flag, measured the payoff on qap's root box (guard lifted,
  `backend="simplex"`):
  - **McCormick LP is now cheap and memory-safe:** `solve_at_node(root)` → **optimal in ~2 s,
    0.6 GB** (all three of budget=10/30/60 s converge to the same vertex). The sparse work
    (T0–T4) + the T6 incremental fix delivered a working large-lift LP path.
  - **…but the bound is useless.** McCormick-LP root bound = **−1e-9 ≈ 0** vs the oracle DUAL
    bound **149106**. Expected: McCormick envelopes on an indefinite binary `x'Qx`
    (eig ∈ [−330k, +953k], x ∈ [0,1]) are trivially loose — each lifted product drops to its
    independent lower envelope, so the LP minimum is ~0 and fathoms nothing.
  - **PSD strengthening does not rescue it AND re-densifies.** `_root_relaxation_lower_bound(
    …, psd_cuts=True)` → bound still **−1e-9**, **wall 64 s, RSS 46 GB** (vs 17 s / 0.6 GB
    without PSD). So the moment/clique path (`psd_strengthen_relaxation_bound`) has its own
    dense blowup, same class as T6 — and even so it buys no bound here.
  - **Verdict:** relaxing the guard **by default is a regression** for qap (≥2 s/node LP for a
    ~0 bound, no fathoming) and **helps no measured instance**, so shipping a default-off flag
    would be a near-dead flag (§3). NOT implemented. The guard stays as-is. qap's real dual
    bound (→149106) needs a fundamentally stronger relaxation (RLT-2 / tailored QAP / SDP),
    not a McCormick-LP guard flip. **Two concrete follow-ups surfaced, neither a guard flip:**
    (a) the PSD path's 46 GB densification (a real memory bug, T6-style fix) — **FIXED in T8
    below**; (b) per-node LP time on large lifts (profiling target from T6).

- [x] **T8 — FIXED the PSD/OBBT exact-LP oracle 46 GB densification.** Root cause: the exact-LP
  oracle `solvers/lp_simplex.py::solve_lp` (returned by `get_exact_lp_solver`, called by
  `psd_strengthen_relaxation_bound`, OBBT, DBBT, Benders) is a **separate** dense entry from the
  T4b `solve_milp` — it did `A_ub.toarray()` **and** built `a_std = np.zeros((m, n+m))`; for
  qap's 85756-row PSD-strengthened relaxation that standard form is `85756 x 107405 ~ 9.2e9
  cells ~ 73 GB` (observed ~46 GB peak before the OS killed/partial-committed it). **Fix:**
  rewrote `solve_lp` to assemble the standard form `[A_ub | I_ub | 0 ; A_eq | 0 | I_eq]`
  **sparsely** (`sp.vstack`/`sp.hstack` + `sort_indices`) and call the CSC-native
  `solve_lp_warm_csc_py` (which already returns the row duals), reconstructing reduced costs
  with sparse `Aᵀy`. Bound-neutral marshaling (§5): objective matches HiGHS to 3e-16 and
  dense-input == sparse-input on a 40-LP panel; row-dual/reduced-cost KKT identities preserved
  (`test_simplex_lp.py` +2 tests, 344 LP/OBBT/DBBT/PSD tests, 653 smoke, 10 adversarial green).
  **Measured (qap root, peak RSS):** `_root_relaxation_lower_bound(psd_cuts=True)` **46 GB →
  0.69 GB** (64 s → 15 s). Regression tests: `test_solve_lp_routes_through_sparse_csc_binding`
  (spies that the dense binding is never called) + `test_solve_lp_does_not_densify_large_sparse_lp`
  (tracemalloc peak ≪ the dense `m×(n+m)`). NB: PSD still buys qap no bound (−1e-9) — that is a
  separate *effectiveness* question, resolved in T9 below.

- [x] **T9 — PSD effectiveness: root-caused the "no bound" AND fixed a real inertness bug.**
  Diagnosed why `psd_strengthen_relaxation_bound` never tightens qap (two layers):
  - **Primary bug (FIXED):** the moment separator's `_diag_col`/`_lifted_cliques` required a
    lifted **square** column `X_ii = x_i²` (`monomial`/`univariate_square`), which pure products
    of *distinct* binaries (the whole QAP objective) never create → **0 cliques → 0 cuts → PSD
    completely inert** on every binary-product model, not just qap. But for a **binary** `x_i`,
    `x_i² = x_i` at every feasible point, so the moment diagonal `X_ii` *is* the original `x_i`
    column. Fix: `discopt/_jax/model_utils.py::binary_flat_cols` (integer-typed vars on `[0,1]`)
    threaded as `binary_vars` through `_diag_col`→`_moment_blocks_for_set`→`_lifted_cliques`→
    `separate_psd_cuts_on_relaxation`→`psd_strengthen_relaxation_bound`, and wired at both
    callers (`solver.py` root bound, `mccormick_lp.py` per-node). Sound only for binaries
    (continuous `X_ii = x_i² ≠ x_i` still requires the real square lift — preserved, see
    `test_no_op_on_model_without_lifted_squares`). On qap: cliques go 0 → 223 (k=2) / 219 (k=4)
    / 215 (k=6).
  - **Deeper limitation (measured, not a bug):** even with cliques, **0 of qap's 223 pairwise
    moment minors are violated** at the McCormick vertex (λ_min = 0) — pairwise binary moment
    cuts are redundant with McCormick, and qap's vertex also satisfies the k≤6 minors, so the
    bound stays ~0. Closing qap needs the **global** (full 226×226) moment/Shor SDP, which local
    clique separation cannot supply — a separate, larger effort.
  - **The fix is verified effective + sound on the class it targets** via a constructed binary
    QP (`min x0x1+x0x2+x1x2 s.t. Σx ≥ 1.5`, true min 1): the k=3 moment matrix at `x=(½,½,½),
    X=0` has eigenvalue −¼, so **PSD strengthening lifts the bound 0 → 0.356** (was 0 cuts before
    the fix), staying valid (≤ 1); feasible-point sampling confirms no cut removes an integer
    point (`test_binary_products_get_moment_cuts_via_diagonal_shortcut`). 158 PSD, 653 smoke, 10
    adversarial green; continuous QCQP PSD path unchanged (reaches −1.0 exactly).

- [x] **T10 — per-node LP-time profiling: the LP is NOT the bottleneck; the cold rebuild is.**
  cProfile of `MccormickLPRelaxer.solve_at_node` on qap (guard lifted, 4 node solves, warm):
  **3.66 s/node**, split — **cold `build_milp_relaxation` 3.16 s (86%)** vs the sparse LP solve
  `solve_lp_warm_csc_py` **0.49 s (13%)**. The "per-node LP time" is small; the cost is
  **rebuilding the 85 756-row McCormick relaxation from scratch every node** (qap can't use the
  incremental fast path — its dense `base_A` is declined in T6). Inside the build, the top Python
  cost was **1.54 M `BinaryOp` constructions per 4 nodes**, each calling
  `numpy.broadcast_shapes` on (almost always) scalar `()` shapes — ~2.5 s of the 12.6 s 4-node
  build spent on shape bookkeeping alone.
  - **Fix (bound-neutral, committed):** scalar/equal-shape fast path in `core._broadcast_shapes`
    (equal shapes → themselves; scalar → the other operand; differing shapes still defer to numpy,
    which also validates). Verified identical output to numpy on a broadcast panel (incl. the
    mismatch-raises case). **Per-node 3.66 s → 3.05 s (−17%)**, build 3.16 → 2.55 s/node; 653
    smoke + 96 shape/broadcast tests green.
  - **The remaining bottleneck (biggest lever, not yet taken):** the 2.55 s/node cold rebuild
    itself. `distribute_products` (0.64 s/node) + `_build_product`/`_emit_mccormick` dominate. The
    real win is to **make `IncrementalMcCormickLP` sparse** (the T6 follow-up) so qap can patch
    coefficients per node instead of rebuilding — that would collapse the ~2.5 s build to a patch,
    leaving mostly the 0.49 s LP solve. Larger change (bit-identity contract vs the cold build).

- [x] **T11 — sparse `IncrementalMcCormickLP`: 14.7× per-node on qap, 30 GB → 0.76 GB.** Made the
  incremental fast path fully sparse so large-lift models patch coefficients per node instead of
  rebuilding:
  - `base_A` is now **CSR** (was `.todense()`d, ~14.85 GB/copy on qap). The McCormick product
    rows have a **stable sparsity pattern** (columns fixed at `{factors, aux}`, only the box-
    dependent *values* change), so `_build_structure` precomputes each product row's data-index
    span + target-column positions, and `_patch` copies the base `.data` (~nnz floats) and
    overwrites only those entries — O(nnz)/node, not O(rows·cols). `_full_build` keeps the matrix
    sparse; the row-mapping loop is now O(nnz) via a CSC aux-column lookup (was O(rows·products)
    ~1.8e9 on qap); `_rowset` compares sparse row-sets order-free without densifying; `assemble`
    uses sparse vstack. Guard is now **nnz-based** (`_MAX_INCREMENTAL_NNZ = 5e7`), replacing the
    T6 dense-cell decline.
  - **Bit-identical:** the built-in `_validate` gate (patched vs cold-built row-set on 6 sign-
    diverse boxes) still passes — 231 mccormick/incremental + 653 smoke + 10 adversarial green.
    New `test_incremental_structure_is_sparse_and_patch_matches_dense` asserts `base_A` is sparse
    and the sparse patch equals an independent dense patch.
  - **Measured (qap, guard lifted so the LP solves):** per-node **1.566 s (cold rebuild) → 0.107 s
    (sparse patch)**, **14.7×**, same bound (−1e-9), RSS **0.76 GB** (was ~30 GB). Construction
    (incl. `_validate`'s 6 cold builds) ~13 s, one-time.
  - **Fixup:** `lp_spatial_bb._separate_node_cuts` (a dense-only GMI/crossover cut separator that
    always received a dense `A`) now densifies its `A` at entry — this bounded per-node cut path
    was never viable for a large lift; the node LP solve stays sparse.
  - **Production note:** the 14.7× is *realized* only where the node LP actually solves. For qap
    the `_MAX_RELAX_DENSE_CELLS` fast-path guard (`mccormick_lp.py:766`) still declines the solve
    (T7: qap's McCormick bound is ~0, so lifting it is a separate flagged, bound-changing
    decision). This change removes the memory blowup and makes the fast path *available* for
    large lifts — the enabler that lifting that guard would need.

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
