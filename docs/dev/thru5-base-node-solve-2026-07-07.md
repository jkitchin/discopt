# THRU-5 — base per-node solve decomposition (2026-07-07)

**Task.** THRU-3 (`thru3-node-decomp-2026-07-07.md`) stripped both per-node
separation loops and found the residual per-node floor is the **base node solve**
(`solve_milp`/`solve_lp_warm_std`, ~48% of leaf samples on nvs24), and global50
shows discopt ~13× slower than BARON on jointly-proved instances — pure per-node
throughput. THRU-5 measures what that base node solve actually spends its wall on,
whether the THRU-2b warm path is taken, and whether **any** of it is reducible
with a bounded, bound-neutral change — or whether the gap is a fundamental
per-operation speed gap that needs an engine-level effort (a valid, honest KILL).

**Verdict: KILL (honest). The base node solve is NOT reducible by a bounded
single-PR change.** The per-node wall is dominated by the **in-house LU kernel
refactorizing on nearly every simplex pivot** (47–66 % of the node-LP wall), and
that refactorization is forced by the **product-form (FT) basis update returning
an error 89–100 % of the time** on the wide/dense-column lifted-McCormick basis.
This is a limitation of the `feral` LU backend's update kernel on dense-spike
bases, not a cadence knob, a Python-marshaling cost, or a dense-rebuild — all of
which the measurement rules out. Closing it needs an engine-level LU/basis-update
change (or a production sparse-LP backend), which is out of a single PR's scope.

Build: release (`maturin develop --release`), pounce `_pounce.abi3.so` = 4.73 MB.
Env: `PYTHONPATH=<wt>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`. Separation forced
OFF (both_off regime: `DISCOPT_SQUARE_SEPARATE=0` + `_psd_cuts`/`_rlt_cuts` = False)
to isolate the base solve, matching THRU-3's 309-node nvs24 arm. Instances profiled
are exactly those whose node bound goes through the **in-house simplex LP** path
(the ones the "13× vs BARON" claim is about): **nvs24, nvs19** + two global50
controls, **st_e36** (the worst jointly-proved ratio, 206× at TL60, 155 nodes) and
**ex1224** (a fast LP-light contrast). fac2/m3/flay03m/clay0303hfsg/cvxnonsep_* were
checked and excluded: their node bound is an NLP (pounce), not the simplex LP, so
they never exercise the base solve THRU-5 targets.

## Instrumentation added (Rust, pure profiling — no-op without `DISCOPT_PROFILE`)

The existing `DISCOPT_PROFILE` phase profiler covered only the integer MILP driver
(`solve_milp_py`), which fires once at the root; the **per-node warm-LP path**
(`solve_lp_warm_csc_py`, the dominant per-node cost when the node bound is a pure
lifted LP — the default `node_bound_mode="lp"`) was invisible. THRU-5 adds, all
guarded by the existing `enabled()` no-op gate:

* `crates/discopt-core/src/lp/simplex/dual.rs` — three warm-dual phases
  (`DualPrepare` = one-time basis factorize + dual-feasibility verify;
  `DualRecompute` = exact xB / reduced-cost seed+refresh; `DualPivotLoop` = the
  dual pivot loop).
* `crates/discopt-python/src/lp_bindings.rs` — `init_from_env()` + `dump()` in
  `solve_lp_warm_csc_py`, so pure-LP-only instances (which never call the MILP
  driver) surface their per-node profile too.
* `crates/discopt-core/src/profile.rs` + `primal.rs` — split the primal
  refactorization trigger into three counters (`RefacFtFail` = the FT update
  returned Err → forced refactor; `RefacCap` = the hard 48-update cap;
  `RefacWorkGate` = the adaptive work gate). This is the counter that produced the
  root-cause finding.

The Python-side decomposition (harness) times the two node-LP entrypoints
(`solve_milp`, `solve_lp_warm_std`) and their internal sections (sparse build /
Rust call / NS safe-bound / Farkas) by wrapping them; it is measurement-only and
not committed.

## Part 1 — per-node bucket breakdown (separation OFF, TL 30 s)

**First cut — where the node wall goes (Python entrypoints).** The default node
bound is a pure lifted-McCormick **LP** (`node_bound_mode="lp"` drops integrality
per node), so the per-node solve is `solve_lp_warm_std` (sparse CSC, warm-start
threaded), **not** the dense `solve_milp`. `solve_milp`'s dense
`a_std = zeros((m, n+m)) + eye(m)` fires only **once** (the root); it is NOT the
per-node path. Inside `solve_lp_warm_std`, decomposed precisely:

| section (nvs24, 228 node-LP calls) | % of LP-path wall |
|---|---:|
| **Rust dual/primal simplex solve** | **98.9 %** |
| sparse CSC build (`hstack`) + marshal | 0.1 % |
| NS safe-bound helper (Python) | 0.4 % |
| Farkas / result build | 0.5 % |

→ **The dense-rebuild, marshaling, and safe-bound hypotheses (a)/(b)/(c) are all
falsified.** The lifted matrix is already sparse (density ≈ 0.0098, ~20 k nnz),
built once, warm-basis threaded; the per-node Python overhead is <2 %. The entire
base-solve cost is **inside the Rust simplex call**.

**Second cut — inside the Rust simplex (phase profile, per-node dumps only, root
MILP dump excluded).** Buckets are % of the node-LP phase time:

| instance | wall (s) | nodes | nps | node-LP %wall | **Refactorize** | FtUpdate | AlphaFtran | PriceBtran | PriceSweep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **nvs24** | 31.2 | 269 | 8.6 | 64 % | **55.8 %** | 18 % | 12 % | 11 % | 3 % |
| **nvs19** | 30.2 | 377 | 12.5 | 68 % | **47.2 %** | 24 % | 14 % | 11 % | 4 % |
| **st_e36** | 23.9 | 155 | 6.5 | 74 % | **66.2 %** | 10 % | 11 % | 12 % | <1 % |
| **ex1224** | 2.1 | 53 | 25.1 | 9 % | **30.8 %** | 27 % | 16 % | 14 % | 11 % |

**Refactorization is the single dominant sub-bucket of the base node solve**
(47–66 % of the node-LP wall on the three LP-heavy instances). FtUpdate (the
product-form update itself), AlphaFtran (ftran), and PriceBtran (btran) split the
rest — all standard revised-simplex per-pivot work.

**Is the THRU-2b warm path taken per node?** Largely **no**, and this is the
mechanism. Per node the driver *attempts* a warm **dual** re-solve
(`DualPrepare` = 102–371 factorize+verify attempts per run), but the dual pivot
loop (`DualPivotLoop`) runs only **24–37 times** — i.e. **~90 % of warm-dual
prepares are rejected** (the warm basis is singular or dual-infeasible after the
per-node relaxation rebuild changes the column layout / objective sign) and fall
through to the **cold primal** path (`solve_lp_cols_warm` → primal simplex:
`Phase2Pivots` = 12 k–150 k, `DualWarmSolves` = 24–37). The warm dual, when it
*is* taken, is cheap (`DualPivotLoop` ≈ 1 % of wall). The cost is the cold-primal
fallback, and within it, the refactorizations.

## Root cause — the FT (product-form) basis update fails on the McCormick bump

Splitting the primal refactorization trigger (`RefacFtFail` / `RefacCap` /
`RefacWorkGate`):

| instance | Refactorizations | **RefacFtFail** | RefacCap | RefacWorkGate |
|---|---:|---:|---:|---:|
| nvs24 | 27 887 | **26 819 (96.2 %)** | 1 068 | 0 |
| nvs19 | 33 668 | **30 707 (91.2 %)** | 2 945 | 16 |
| st_e36 | 42 144 | **42 143 (100.0 %)** | 1 | 0 |
| ex1224 | 726 | **643 (88.6 %)** | 79 | 4 |

The refactorizations are **89–100 % `RefacFtFail`**: the Forrest–Tomlin-style
product-form LU **update returns `Err`** (numerical bump breakdown) and the caller
is forced to refactorize from scratch. On st_e36 this happens on **every single
pivot** (`Refactorize` count 42 143 ≈ `Phase2Pivots`-scale, i.e. the basis is
refactorized once per pivot). The FT update failure is raised inside the external
`feral` LU backend (`FeralLU::update` → `lu.update(...).map_err(...)` in
`crates/discopt-core/src/lp/simplex/linsolve.rs`), triggered by the **wide,
dense-structural-column lifted-McCormick basis** (the McCormick envelope columns
are dense, so a single basis-column swap produces a non-localized spike the
product-form update cannot represent stably).

Contrast to frame the gap: nvs24 does ~6.7 s/node while BARON solves the *entire*
instance in 0.049 s. The per-node cost is a refactorization storm — thousands of
O(nnz)/O(m·bump) refactorizations per node — where a decades-tuned sparse-LP
engine (BARON's) does a handful of stable FT updates and one refactorize.

## Part 2 — decision: KILL, with a falsified entry prototype

**Entry prototype (falsified before shipping, per CLAUDE.md §4).** The one
candidate that looked bounded-and-bound-neutral was the adaptive refactorization
**work gate** (`ft_update_work() > factor_nnz()`), whose own doc-comment asserts
bound-neutrality ("a refactorization yields the same basis inverse … pivots and
the optimum are unchanged"). Hypothesis: the gate over-refactorizes on the wide
McCormick bump, so raising its threshold would cut refactorizations at
byte-identical node_count/objective. **Experiment (entry, run before
implementing the roadmap):** a process-wide `DISCOPT_REFAC_WORK_MULT` multiplier
on the gate threshold, tested at 1× / 4× / 8× / 16× under a fixed node cap.

**Result — the multiplier is completely inert:**

| instance (maxnodes 120) | mult | nodes | refac | wall (s) | obj |
|---|---:|---:|---:|---:|---:|
| st_e36 | 1× | 123 | 33 677 | 20.1 | −246.0 |
| st_e36 | 4× | 123 | 33 677 | 20.5 | −246.0 |
| st_e36 | 16× | 123 | 33 677 | 20.3 | −246.0 |
| nvs24 | 1× | 135 | 14 117 | 12.9 | −1031.8 |
| nvs24 | 16× | 135 | 14 117 | 12.5 | −1031.8 |

node_count, refac count, and objective are byte-identical across the multiplier
(the change *is* bound-neutral, as its doc claims — good), but refactorizations do
**not** move, because the work gate accounts for essentially **0** of the
refactorizations (`RefacWorkGate` = 0–16 vs 26 k–42 k `RefacFtFail`). The gate was
the wrong lever; the trigger-split counter (added, kept) is what proved it. The
multiplier is a dead knob and was **removed** (§3, no dead flags).

**Why no other bounded lever qualifies.**
* *Reduce refactor cadence directly* (48-cap, or the gate) — inert here; the
  refactorizations are `RefacFtFail`, forced by the update kernel, not cadence.
* *Make the warm-dual path taken more often* — the warm dual is rejected ~90 % of
  the time because the per-node relaxation is rebuilt with a changed column layout
  and the OBBT probes flip the objective sign (dual-infeasible). Threading a
  persistent factorization across nodes / batching the OBBT objective directions
  is a real direction, but it changes the LP solve path (roundoff → which OBBT
  bounds tighten → the tree), so it is **bound-changing**, not a bound-neutral
  mechanics fix, and larger than one PR.
* *Fix the FT update itself* — an engine-level LU-kernel change (below).

Shipping any of these to "move nodes/s" would either not move it (falsified) or
change the certificate path. Per the task's kill criterion and CLAUDE.md §1/§4,
THRU-5 **does not yield a scoped, bound-neutral per-node speedup**; it ships the
measurement + the instrumentation that localizes the real lever.

## Engine-level roadmap (ranked) — what an out-of-PR effort must target

1. **Make the basis update tolerate the McCormick dense-column spike (the 89–100 %
   `RefacFtFail`).** This is the single biggest lever: on st_e36 every pivot
   refactorizes because the product-form update breaks down. Options: (a) a
   Forrest–Tomlin update with proper spike handling / partial refactorization in
   the `feral` kernel; (b) a Suhl–Suhl / Bartels–Golub update robust to dense
   eta-vectors; (c) block/hybrid handling that keeps the dense McCormick columns
   out of the eta chain. Success metric: `RefacFtFail` per pivot → O(1/48), i.e.
   the 48-update cap becomes the dominant trigger instead of FT-fail.
2. **Cut the dense-column count in the lifted relaxation** (upstream, structural).
   The FT breakdown is driven by dense McCormick-envelope columns; a sparser lift
   (or column-localized envelope) would shrink the bump. Bound-changing —
   needs the differential-bound regime.
3. **Persist one factorization across a node's OBBT probe fan** (the ~90 % warm
   rejection). OBBT solves one LP per variable-direction against a *fixed* matrix,
   changing only the objective — the textbook `PreparedDual` reuse case — yet each
   probe currently cold-refactorizes. A batched "factor once, reoptimize per
   objective" OBBT entry is a real speedup, but changes which bounds tighten
   (bound-changing).
4. **A production sparse-LP backend** for the node LP (the "engine swap" the
   in-house simplex was meant to avoid). Largest scope; only if 1–3 stall.

Ranked #1 first because it is the measured dominant cost and its fix is confined
to the LU kernel (no bound change if the update reproduces the same inverse), so
it is the closest to a future bound-neutral engine PR.

## §3 — what shipped

Pure **instrumentation** (all no-ops when `DISCOPT_PROFILE` is unset; the
production hot path is a single relaxed atomic-bool load, unchanged):

* `dual.rs`: `DualPrepare` / `DualRecompute` / `DualPivotLoop` phase timers.
* `lp_bindings.rs`: `init_from_env()` + `dump()` in the warm-LP binding so the
  per-node LP profile is visible (previously only the root MILP driver was).
* `profile.rs` + `primal.rs`: `RefacFtFail` / `RefacCap` / `RefacWorkGate`
  refactor-trigger split counters — the load-bearing evidence for the KILL.

The falsified `DISCOPT_REFAC_WORK_MULT` prototype was **removed** (dead knob).

## Gates

* `cargo test -p discopt-core`: **426 passed, 0 failed** (+ 4 presolve-determinism
  + 1 doctest).
* `pytest -m slow python/tests/test_adversarial_recent_fixes.py`: **10 passed**.
* `rustfmt --check` on the four touched files: clean. `cargo clippy` on
  discopt-core/discopt-python: no new warnings (pre-existing too-many-args on the
  binding shims only).
* **Cert neutrality (41-panel, `check_cert_neutrality.py`):** the branch reports 4
  violations (`nvs02` obj/nodes, `nvs12` status, `nvs13` nodes) — but re-running
  the check on a **clean `origin/main` build** (this PR's 4-file diff stashed)
  reproduces the **identical 4 violations**, so they are **pre-existing baseline
  drift on `main`, NOT this change** (the nvs13 19→49 drift is the same one THRU-3
  documented). This PR's diff is pure `DISCOPT_PROFILE`-gated instrumentation and
  is byte-identical to main on the search: with profiling off it cannot alter a
  pivot, a bound, or a node.
* `pytest -m smoke`: 616 passed, 6 failed — all 6 in `test_amp.py`
  (trig/piecewise/square relaxation-strength assertions, unrelated to the LP
  kernel); confirmed to fail **identically on clean `origin/main`** in this scratch
  venv (pre-existing / environment, not THRU-5).
