# PF3 SPIKE — branching quality (spatial branch-point + reliability threshold)

Status: **DONE → KILL (both axes)** on this spike set (2026-07-14). Measurement
only; prototype lives in an **isolated worktree** and is NOT pushed. Companion:
`docs/dev/sota-proof-plan.md` §2 PF3.

## 1. What discopt does TODAY (read from the code)

Branch VARIABLE and branch POINT are chosen in
`crates/discopt-core/src/bnb/branching.rs` and driven by
`crates/discopt-core/src/bnb/tree_manager.rs::process_evaluated`
(the Rust `PyTreeManager` IS the global spatial B&B engine — `solver.py` marshals
node LPs to it; there is no separate Python spatial node loop for branch
selection).

**Integer branching** (`process_evaluated` step 3, tree_manager.rs:451):
- Variable: pseudocost **product score** `(1e-6+d)*(1e-6+u)` when
  `use_pseudocosts` (default true, `select_branch_variable_pseudocost`,
  branching.rs:207); most-fractional (`select_branch_variable`, branching.rs:155)
  as fallback / when pseudocosts disabled.
- **Reliability threshold = 8, HARDCODED** (tree_manager.rs:216). Variables with
  `< 8` pseudocost observations are returned as `unreliable_candidates` for the
  Python orchestrator's optional strong branching (`_strong_branch_lp`,
  solver.py:1938). No setter, no env var, no solve kwarg — only a getter
  (`get_reliability_threshold`, exposed as `tree.reliability_threshold()`).
- Branch POINT: `val.floor()` → children `x ≤ floor(v)`, `x ≥ ceil(v)`
  (branching.rs:187/243). Standard.

**Spatial branching** (nonconvex mode, tree_manager.rs:524, when an
integer-feasible node still has an open convex-relaxation gap):
- Variable: **longest-edge in normalized coordinates** —
  `select_spatial_branch_variable` (branching.rs:413) picks the continuous var
  with the largest `width/global_width`; `select_spatial_integer_branch_variable`
  (branching.rs:513) does the same for integer columns in nonlinear terms; the
  two compete on relative width (integer wins ties). Functionally-dependent
  continuous outputs are deprioritized to a completeness fallback.
- Branch POINT: **BOX MIDPOINT, HARDCODED** —
  `mid = 0.5*(node_lb[idx]+node_ub[idx])` (branching.rs:478); integer partition
  uses `floor(midpoint)` (branching.rs:550). This is the blind-bisection point
  the PF3 premise flags (BARON/SCIP branch near the LP relaxation solution). The
  node relaxation solution (`result.solution`) is in scope at the branch site but
  is **not consulted** for the point. No env var / kwarg.

**Tunability without a shared-code edit: NONE.** Neither the spatial branch point
nor the reliability threshold is reachable via env/kwarg/module-constant. Both
required an isolated worktree Rust edit to measure.

## 2. Prototype (isolated worktree only, env-gated, NOT pushed)

Added two env gates (default OFF reproduces shipped behavior byte-for-byte):
- `DISCOPT_SPATIAL_BRANCH_POINT` = `midpoint`(default)/`lp`/`lpmid` —
  `branching.rs::spatial_branch_point_rule` + `adjust_spatial_branch_point`
  (clamp LP value into `[lb+0.1w, ub-0.1w]` so neither child is a sliver); wired
  at both continuous spatial-branch sites in tree_manager.rs.
- `DISCOPT_RELIABILITY_THRESHOLD` = u32 (default 8) — read at PyTreeManager
  construction (tree_manager.rs:216).

Built with `cargo build --release -p discopt-python`; the `.so` was copied into
the worktree package and run via `PYTHONPATH=<worktree>/python` so the shared
`.venv` and the main tree (PF1's workspace) were never touched.

Method note: with the branch code UNSET, fac2 (proved) moved 101→69 nodes between
a jobs=5 and jobs=1 run — **node counts on this solver are NOT deterministic under
wall-time budgets + CPU contention** (solver.py has wall-fraction-budgeted root
heuristics, e.g. RENS #281). All comparison runs below therefore use `jobs=1`
(serial, matched contention) and lean on the robust signals: proved-node-
count/wall and **dual-bound-at-timeout** (better branching ⇒ tighter bound sooner).

## 3. Spatial branch-POINT results (jobs=1, tl=40s)

| instance | outcome | midpoint | lp | lpmid | bound (all rules) |
|---|---|---|---|---|---|
| fac2 | **proved** | 69 n / 17.7 s | 69 n / 17.7 s | 69 n / 15.3 s | 3.31837e8 (exact) |
| nvs05 | feasible (tl) | 323 n | 303 n | 245 n | **1.352 — frozen** |
| tspn05 | feasible (tl) | 37 n / b=178.27 | 37 n / b=177.83 | 37 n / b=178.05 | flat (lp slightly worse) |
| heatexch_gen1 | timeout | 3 n | 3 n | 3 n | 100500 |
| heatexch_gen2 | timeout | 31 n | 15 n | 15 n | 218690 |
| bchoco06 | timeout | 7 n | 7 n | 7 n | (none) |

Soundness: **0 violations** across every run (dual bound never crossed the
incumbent objective).

Reading:
- **fac2** (the only proved instance): byte-identical tree (69 nodes, 17.7 s)
  under all three rules → its search is integer/MILP-branch-dominated; the spatial
  branch point never materially engages.
- **nvs05**: node count falls (323→245) but the **dual bound is unchanged (1.352)**
  and the incumbent is unchanged — fewer nodes = more per-node time with **zero
  extra proof progress**. The bound is frozen because the McCormick envelope on
  the integer product `i1*i2` is loose (a **PF4 root-gap** problem), not a
  branch-point problem.
- **tspn05**: same 37 nodes, bound flat (lp is marginally *worse*).
- **heatexch_gen1/2, bchoco06**: 3–31 nodes explored in 40 s → the bottleneck is
  **per-node relaxation cost (PF2 territory)**, not branch-point quality. Fewer
  nodes under lp/lpmid at identical bound = slower nodes, no gain.

No config cut nodes ≥2× on a class while holding/improving the bound; no proof
gained; the one proved instance is invariant. **Branch-POINT axis: KILL.**

## 4. Reliability-threshold results (jobs=1, tl=40s)

`DISCOPT_RELIABILITY_THRESHOLD` ∈ {1, 4, 8(default), 16}:

| instance | rt=1 | rt=4 | rt=8 (default) | rt=16 |
|---|---|---|---|---|
| fac2 (proved) | 69 n / 17.3 s | 69 n / 17.2 s | 69 n / 17.7 s | 69 n / 17.8 s |
| nvs05 | 321 n, b=1.352 | 321 n, b=1.352 | 323 n, b=1.352 | 321 n, b=1.352 |
| tspn05 | 35 n, b=178.27 | 37 n, b=178.27 | 37 n, b=178.27 | 37 n, b=178.27 |
| bchoco06 | 3 n | 7 n | 7 n | 7 n |

**Completely inert.** fac2's proved tree is byte-identical (69 nodes) at every
threshold; nvs05/tspn05 bounds are flat. The reliability/strong-branching path
barely fires on these instances — the spatial ones branch spatially (not via the
integer pseudocost selector the threshold governs), and fac2's small integer tree
is threshold-insensitive. bchoco06's 3-vs-7 is per-node-cost noise (node-starved).
Soundness: **0 violations**. **Reliability-threshold axis: KILL.**

## 5. Verdict

**KILL** for PF3 on this spike set, per the plan's criterion (GO = an axis cuts
nodes ≥2× on a class or gains a proof without losses; neither axis does).

The premise — "blind midpoint is costing spatial tree size vs BARON" — is **not
supported by these instances**. Their timeouts are explained by two *other* PF
items, not by branch-point choice:
- relaxation tightness at a frozen bound (nvs05, tspn05) → **PF4**;
- per-node relaxation cost starving the tree to 3–31 nodes/40 s (heatexch*,
  bchoco*) → **PF2**.

The branch-point machinery is sound (0 cut points, 0 bound crossings) but inert
here. Any future PF3 attempt should first find a *class where the tree is large
and the bound is moving* (branch-point can only help where many nodes actually get
explored and the dual bound is climbing) — none of the vendored spike instances
are in that regime at tl=40s.

### If a later run finds such a class (what the full item WOULD change)
- Function: `branching.rs::select_spatial_branch_variable` (and the integer
  sibling) would take the node solution and return an LP-anchored point
  (`adjust_spatial_branch_point`) instead of the hardcoded `mid` at
  branching.rs:478/550; default flipped only behind the PF0 differential +
  feasible-point + panel gate.
- **PF1 conflict check:** PF1's landing edits `solver.py`'s node loop (wiring the
  FBBT/reduce kernel near solver.py:6155 and `node_depth`) — a **different file
  and layer** from this Rust branch-point change. They do **not** overlap; PF3
  (if it ever graduates) and PF1 can land independently in either order. The only
  shared surface is the PF0 harness both must pass.
