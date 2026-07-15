# SOTA-P3 Branch-and-reduce: per-node probing (issue #632)

**Date:** 2026-07-13 · **Base:** `main` @ `9937ff7` (worktree
`worktree-agent-a1659981355eb9dc8`) · **Prototype workstream** — sound-by-
construction, full-corpus benchmarking deferred to the assembled pass.

> This is the §10 ledger entry for the P3 (branch-and-reduce) workstream of the
> #632 roadmap. P3 = "OBBT + FBBT + probing to shrink boxes so the factorable
> envelopes tighten." It records what shipped, what is prototype-level, the
> soundness argument, and the remaining local-host items.

## 0. TL;DR — what P3 needed vs. what was already present

Auditing the tree, **FBBT and OBBT were already wired at B&B nodes** and only
**probing was root-only**:

| P3 leg | Pre-existing state | This workstream |
|---|---|---|
| **FBBT** at nodes | LIVE — `bnb/in_tree_presolve.rs` (`fbbt_with_cutoff`, depth-strided) + `_jax/node_reduce.py` cutoff-FBBT (cert:T2.4b) | reused; now also folds probing |
| **OBBT** at nodes | LIVE — per-node OBBT (`_jax/obbt.py obbt_tighten_root`, structural/dependent-var gate) + free DBBT / integer RC-fixing in `reduce_node` | untouched (Python LP interface, correct owner) |
| **Probing** | ROOT-ONLY — `presolve/probing.rs::probe_binary_vars` via the presolve pipeline (`_jax/presolve_pipeline.py`, binaries only) | **NEW: per-node probing kernel + wiring** |

So the concrete P3 gap closed here is **per-node probing** (the missing triad
leg), plus a unified public branch-and-reduce entry point.

## 1. What landed

1. **`crates/discopt-core/src/presolve/probing.rs` — `probe_node_bounds()`**
   A sound, box-aware, budget-capped per-node probing kernel over **binary AND
   general-integer** variables:
   - binary `y`: fix to 0 and 1, run cutoff-aware FBBT; if exactly one branch is
     infeasible the other value is forced (permanent fix + intersect the
     surviving branch's implied bounds); both infeasible ⇒ node infeasible; both
     feasible ⇒ tighten every *other* variable to the interval hull of the two
     branches (a sound implication cut);
   - integer `x ∈ [l,u]`: peel proven-infeasible endpoints — `x=l` infeasible ⇒
     `x ≥ l+1`, `x=u` infeasible ⇒ `x ≤ u-1` (up to `MAX_INT_PEEL=8` per endpoint).
   Contracts only on `Interval::is_empty_beyond(FEAS_TOL)` proof (eps-scale
   inverted intervals are treated as noise, never proof). Budget: `max_vars`
   discrete vars + optional `deadline`.

2. **`crates/discopt-core/src/bnb/in_tree_presolve.rs`** — `InTreePresolveOptions`
   gains `probing` / `probe_max_vars`; `run_in_tree_presolve` runs probing after
   FBBT and folds its subset box back (intersect-only; empty ⇒ infeasible).
   Default `probing=false` keeps the FBBT-only path byte-neutral.

3. **`crates/discopt-python/src/expr_bindings.rs`** — `PyModelRepr.in_tree_presolve`
   exposes `probing` / `probe_max_vars` (+ the existing `incumbent`).

4. **`python/discopt/solver.py`** — `DISCOPT_NODE_PROBING` (default OFF) and
   `DISCOPT_NODE_PROBE_MAX_VARS` (default 32) route into the in-tree node reducer,
   feeding `tree.incumbent()` as the cutoff so FBBT + probing are optimality-aware.
   A proven-infeasible in-tree box now **fathoms** the node (`node_infeasible_mask`)
   instead of being dropped. All of this is gated behind `in_tree_presolve_stride`
   (default 0), so the default solve path is unchanged.

5. **`python/discopt/tightening.py`** — public `probe_box(model, …)`: the LP-free
   branch-and-reduce reduction (FBBT + per-node probing), mirroring `fbbt_box`,
   at least as tight, returning a `BoundTightening`.

## 2. Soundness argument

Every contraction is feasibility/optimality-justified and applied as an
intersection (never loosens):

- **FBBT** derives only valid bounds (outward-rounded interval arithmetic).
- **Probing** contracts a discrete domain *only* when a tentative fixing is
  *proven* infeasible (`is_empty_beyond(FEAS_TOL)`): if `x=v` admits no feasible
  point, no feasible point has `x=v`, so removing it is sound; for a binary this
  forces the sibling, and the surviving branch's FBBT result is a valid bound to
  intersect. The both-feasible hull cut is sound because every feasible point
  lies in `res0 ∪ res1 ⊆ hull(res0,res1)`.
- **Cutoff** (`incumbent`): the incumbent objective is a valid upper bound on the
  optimum, so `objective ⋈ incumbent` FBBT/probing only discards points that
  cannot improve the incumbent — sound inside branch & bound.

A reduced box therefore always contains the entire feasible region (and the
optimum). No feasible point is cut; the oracle is never crossed.

## 3. Prototype-level / stubbed

- Integer probing peels **endpoints only** (interior-hole domains, e.g. `x²≥2`
  removing `x=0` from `[-3,3]`, are not split — interval FBBT cannot represent
  the gap). Endpoint peeling is capped at `MAX_INT_PEEL=8` per endpoint per call.
- Per-node probing budgets (`max_vars`, per-node/cumulative time) are prototype
  constants, not a tuned scoring policy; a SOTA scheduler would score variables
  (width × pseudo-cost) and probe only the most promising, like the existing
  `DISCOPT_OBBT_TOPK` de-gate for OBBT.
- The in-tree binding operates at variable-**block** granularity (exact for
  scalar variables; array blocks patch element 0 and rely on the
  intersect-with-original floor for soundness) — same limitation as `node_reduce`.
- Probing is **default-OFF** (extra O(discrete) FBBT solves/node); it is not yet
  promoted to the default path.

## 4. Verification done in-container (build/construction correctness)

- `cargo test -p discopt-core`: **445 passed** (new: 4 `probe_node_bounds` tests
  — fixes binary at node box, peels `x·x≤5` integer to `x≤2` while retaining the
  optimum `x=2`, detects both-branch infeasibility, no-op when slack; 2
  `in_tree_presolve` probing tests — fixes `b=1` at node, off-by-default byte-
  neutral).
- `maturin develop --release` clean; `probe_box` sanity: fixes `b=1` / peels
  integer `y≤2` while **retaining the known optimum** in both; end-to-end
  `m.solve(in_tree_presolve_stride=1)` with `DISCOPT_NODE_PROBING=1` returns the
  correct optimum (obj 0.0).

## 5. Remaining heavy-math / local-host items (deferred)

- Full-corpus `incorrect_count` sweep + at-least-as-tight differential bound test
  + feasible-point sampling on MINLPLib (`minlplib.solu` oracle) — **local-host**
  (corpus + BARON absent in-container).
- BARON side-by-side on the nvs05 / global50 branch-and-reduce panel to measure
  node-count reduction from per-node probing.
- Tuning the probing budget/scoring and deciding default-on promotion (needs the
  above measurements); consider a top-k scored probing de-gate mirroring
  `DISCOPT_OBBT_TOPK`.
- Interior integer-domain splitting (branch-on-probe) if measurements show
  endpoint-only peeling leaves value on the table.
