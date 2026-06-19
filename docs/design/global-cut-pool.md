# Design: Global Cut Pool — make strong relaxation cuts reach the per-node bound

**Status:** proposed
**Author:** (drafted with Claude)
**Date:** 2026-06-19
**Tracking the gap:** certifying the `nvs*` dense indefinite integer-QP instances
(nvs17, nvs24, nvs09, nvs18, nvs19, nvs23, …) that SCIP closes in ~1–4 s / tens of
nodes and discopt cannot close in 60 s / thousands of nodes.

---

## 1. Executive summary

The `nvs*` certification gap is **not a missing relaxation** — the relaxation that
closes these problems already exists in discopt. It is an **orchestration gap**:
the strong cuts (PSD / moment cuts, RLT) are computed in one code path and never
reach the path that actually bounds the search nodes.

Measured on `nvs17` (7 integer vars in `[0,200]`, indefinite quadratic
constraints, optimum −1100.4):

| relaxation | bound | note |
|---|---|---|
| McCormick (term-wise) | −65842 | what the search actually uses |
| McCormick + PSD cuts (`solve_at_node`) | **−2453** | 27× tighter, computed in <0.5 s |
| full Shor SDP | **−1104.7** | essentially closes it |

The PSD-cut bound (−2453, and −1104.7 for full Shor) is strong enough that, with
branching, certification is in SCIP's regime. But the search never sees it. This
doc specifies a **global cut pool**: separate the policy-chosen cuts once at the
root, persist them as permanent rows, and feed them into whichever per-node
bounding path the dispatch selects — so every node inherits them.

---

## 2. Why the cuts don't reach the search (root cause)

Three layers of fragmentation, all confirmed by instrumentation (wrapping
`MccormickLPRelaxer.solve_at_node` during a real `nvs17` solve, and direct
relaxer probes):

### 2a. Two per-node bounding paths; cuts live in only one
- **Path A — spatial McCormick LP** (`solver.py:~3898`): bounds each node by
  `_mc_lp_relaxer.solve_at_node(...)`, which runs the cut-separation chain
  (`_separate_multilinear` → `_separate_edge_concave` → `_separate_univariate_square`
  → `_separate_psd` → `_separate_rlt`). **Cuts apply here.**
- **Path B — compiled-relaxation / NLP-BB** (`solver.py:~3854`): bounds each node
  through the compiled `_mc_obj_relax_fn` / `_mc_con_relax_fns` (JAX/numpy batch
  McCormick evaluators) and the Rust tree. **No cut separation.**

`nvs17` runs on **Path B**: instrumentation shows `solve_at_node` is called
**exactly once** (the root usefulness probe) across a 2621-node solve. So the PSD
cuts are computed in the probe (Path A code) and discarded; all 2621 search nodes
bound through Path B's plain McCormick (−65842 → −16966).

### 2b. Multiple un-synced relaxer instances
The cut policy (`_apply_auto_cut_policy`, `solver.py:1727`) mutates
`_mc_lp_relaxer._psd_cuts`/`._rlt_cuts`. But other code paths construct *fresh*
`MccormickLPRelaxer(model)` instances with default `psd_cuts=False` (e.g. the
root-bound probe at `solver.py:1844`). The policy's configuration does not
propagate to them — the single `solve_at_node` call observed for `nvs17` ran with
`_psd_cuts=False`.

### 2c. No cut persistence (rebuild-per-node)
Even on Path A, `solve_at_node` *rebuilds the relaxation from scratch every node*
(`mccormick_lp.py:168`) and re-separates cuts under a shared per-node deadline,
with PSD running 4th in the chain. There is no pool of root cuts inherited by
children — every node pays separation again (and can be starved).

### 2d. The orchestrator is a coarse one-shot heuristic
`_apply_auto_cut_policy` picks **one** family by structure + size:
```
n_vars > 40            -> no cuts
has_linear_constraints -> RLT  (PSD off)
else                   -> PSD  (RLT off)
```
It correctly picks PSD for `nvs17`, but it (i) does not account for *how*
indefinite the quadratic is, (ii) does not verify the cuts actually tighten, and
(iii) selects on a relaxer the search may not use.

---

## 3. Goals / non-goals

**Goals**
- A **global cut pool**: cuts separated once at the root become permanent,
  globally-valid rows inherited by every node, in the bounding path the dispatch
  actually uses.
- Make the policy-chosen cut family (PSD for box/QCQP-without-linear, RLT
  otherwise) reach Path B (and stay applied on Path A).
- Certify the `nvs*` set within the existing time budget, with **zero** loss of
  soundness.

**Non-goals (for the first iteration)**
- Per-node cut separation / a full cutting-plane loop at every node (round-robin,
  cut aging, pool management à la SCIP). Root-only separation + inheritance is the
  high-value first step.
- A new relaxation family. The relaxation (PSD/Shor, RLT) already exists.
- Moving cut separation into Rust. The first cut pool is assembled in Python and
  *passed* across the boundary as extra rows.

---

## 4. Soundness preconditions (must hold before any of this)

1. **Cuts are globally valid.** PSD moment cuts (`vᵀMv ≥ 0`) and RLT
   (constraint×bound products) are valid at *every* feasible point, independent of
   the node box, so a root cut is sound to inherit at any descendant. (RLT bound
   factors use the *root* box; that only makes the cut *weaker* deeper in the
   tree, never invalid.)
2. **False-infeasible guard is in place.** PSD/RLT cuts widen the LP coefficient
   spread; the Rust simplex can then false-infeasible a feasible LP, which would
   unsoundly prune a node. This is already fixed (re-verify an `infeasible`
   verdict on an ill-conditioned LP via exact equilibration — commit `b29dfb8`).
   **This cut pool must not be enabled on any path that lacks that guard.**
3. **No false certification.** A node whose cut-augmented LP is genuinely
   infeasible may be fathomed; a node that times out must leave its bound at −∞
   (unpruned) and decertify the gap — never fabricate `optimal`.

---

## 5. Proposed architecture

### 5a. Cut-pool object
A small, sound, serialisable container produced once at the root:
```
RootCutPool:
    rows:  list[(coeffs: np.ndarray[n_total], rhs: float, sense: ">=")]
    n_total: int                     # lifted column count the cuts are stated over
    column_map: varmap               # which lifted columns each coeff index means
    provenance: {"psd": k, "rlt": m} # for logging / A-B
```
All rows are `coeffs · z ≥ rhs` over the *lifted* column space `z = (x, X, …)`.

### 5b. Separation (root, once)
At the root, after the relaxation is built and solved:
1. Configure the family from `_apply_auto_cut_policy` on the **canonical** relaxer.
2. Run the existing `_separate_psd` / `_separate_rlt` rounds with a **generous
   root budget** (the root bound prunes the whole tree; it is worth seconds).
3. Capture every appended row into `RootCutPool` (they are already assembled as
   `A_ub` rows in `solve_at_node`; expose them instead of discarding).

### 5c. Inheritance (every node, both paths)
- **Path A (`solve_at_node`):** accept an optional `inherited_cuts: RootCutPool`;
  append its rows to the per-node `milp._A_ub`/`_b_ub` *before* solving, and skip
  (or reduce) re-separation. Cuts stated over the root lifted space remain valid
  on the node sub-box.
- **Path B (compiled relaxation / Rust tree):** this is the harder half and the
  reason this needs a spec. Options, in increasing effort:
  1. **Route nonconvex-QCQP-with-cuts to Path A.** If the policy selects a cut
     family, force the node bound through `solve_at_node` (which handles cuts)
     instead of the cut-less compiled path. Smallest change; may cost per-node
     speed (Python LP vs compiled eval) — measure.
  2. **Inject pool rows into the Rust node relaxation.** Pass the pool's rows to
     the Rust tree's per-node LP (the relaxation matrix the Rust driver solves),
     so Path B's nodes inherit them without Python separation. Requires a
     Python→Rust row-append seam on the node relaxation.
  3. **Compile the cuts as extra constraint relaxations** alongside
     `_mc_con_relax_fns` so the batch evaluator includes them.

### 5d. Orchestration changes
- Unify relaxer construction so the policy-configured relaxer (and its cut pool)
  is the single source the bounding path consumes — eliminate fresh default-flag
  `MccormickLPRelaxer(model)` instances on the bounding path.
- Extend `_apply_auto_cut_policy` to also decide *root separation budget* and
  whether to enable the pool, gated on the same structure + size signals (and,
  optionally, a cheap "is the quadratic indefinite enough to need PSD?" check via
  the constraint/objective eigenvalues already computed by the convexity pass).

---

## 6. Phasing

- **P1 — Path A cut pool (prototype, this doc's follow-on).** Add
  `inherited_cuts` to `solve_at_node`; separate once at root; force the
  PSD-policy `nvs*` class onto Path A; measure certification. *Decisive
  proof-of-value with the smallest surface.*
- **P2 — strengthen separation toward Shor** (−2453 → −1104): eigenvector-based
  PSD cuts (separate on the most-negative eigenvector of the current `M`), more
  root rounds, RLT×PSD coupling.
- **P3 — Path B injection** (Rust node-relaxation row-append seam) so the cut pool
  works without forcing Path A, recovering Path B's per-node speed.
- **P4 — pool management:** light per-node re-separation / cut aging if root-only
  proves insufficient deeper in the tree.
- **P5 — validate:** all `nvs*` certify; 0 incorrect; phase1 no-regression;
  the false-infeasible guard stress-tested with the wider spread.

---

## 7. Open questions

- Does the `nvs*` family route to Path B by classification (nonconvex MINLP →
  NLP-BB / Rust tree) or by a heuristic? P1 needs to know the exact gate to force
  Path A.
- Per-node speed on Path A for these instances: the Python LP + inherited cuts vs
  Path B's compiled eval. If Path A is too slow per node, P3 (Rust injection) is
  required, not optional.
- How tight is "root-only" inheritance deep in the tree? RLT bound factors stated
  on the root box weaken as the box shrinks; PSD moment cuts do not. May need P4.
- Interaction with OBBT: a stronger (cut-augmented) relaxation makes OBBT shrink
  boxes far more (OBBT plateaued at `[0,19]` on plain McCormick). Composing the
  cut pool with OBBT is likely multiplicative — worth measuring in P1.

---

## 7b. Status / results (P1, P2 landed; P3 blocked on dispatch)

**P1 — cut-pool mechanism (landed, commit on the feature branch).**
`MccormickLPRelaxer.solve_at_node` gained `out_cuts` (capture the separated rows
once at the root), `inherited_cuts` (append the pool at a node, column-aligned),
and `separate` (skip per-node re-separation). Measured on nvs17: a node inherits
the 80-row pool and reproduces the root bound in **~16 ms vs ~0.5 s** to
re-separate — the ~30× per-node speedup that makes pool-everywhere viable. Purely
additive; default args = prior behaviour; 31 relaxation/soundness tests pass.

**P2 — near-Shor root pool (landed).** The spectral cutting plane converges to
Shor only with many rounds, and `_separate_psd` was capped at 8. Made it a
parameter (`psd_max_rounds`, threaded through `solve_at_node`). On nvs17:
8 rounds → −2453; ~150 rounds → −1221 (1.4 s); ~400 rounds → **−1114** (Shor is
−1104.7), 1235 cuts. A standalone cut-pool B&B (P1 inheritance + P2 root pool)
then drove nvs17 from the frozen −65842 to **14 nodes, incumbent −1100 (= opt),
bound −1110, 0.9 % gap** — SCIP's node regime (SCIP ~75). Not formally certified
only because the 0.9 % gap is just over 1e-4 (needs full Shor −1104 or a few more
nodes), and the 1235-cut pool makes nodes heavy (~7 s) → motivates a cut cap.

**P3 — solver integration: BLOCKED on dispatch (the real remaining work).**
A first hookup (separate the root pool after the usefulness probe; pass
`inherited_cuts`/`separate=False` to the per-node `solve_at_node` calls at
`solver.py:~3918` and `~4180`; cap the pool at `_ROOT_CUT_POOL_MAX`) was written
and then reverted because it **does not reach the integer `nvs*` class**: an
end-to-end `m.solve` left nvs17's bound at −65843 (unchanged McCormick) and the
"Root PSD cut pool" log never fired. Root cause — the integer `nvs*` search
**dispatches away from the Python McCormick-LP node path entirely** (it never
enters the `_mc_mode == "lp"` relaxer-setup block; its 2621 nodes are bounded on a
different, cut-less path — the Rust integer B&B / compiled-relaxation route). So
the hookup is correct for *continuous* nonconvex QCQP that uses the spatial
McCormick-LP loop, but the integer class needs one of:

1. **Route the integer nonconvex-QCQP class through the McCormick-LP node path**
   (so the per-node bound is `solve_at_node` + inherited pool). Smallest change if
   the dispatch gate can be widened; must confirm the per-node LP cost is
   acceptable vs the Rust path it replaces.
2. **Inject the pool rows into the Rust integer B&B node relaxation** (a
   Python→Rust row-append seam), so the existing fast path inherits the cuts.

Both are bounded but real; (1) is the cheaper experiment. Determining the exact
dispatch gate that sends `nvs*` to the Rust path (and whether it can opt into the
LP node path when a cut pool exists) is the first concrete P3 task.

## 8. Success criteria

`nvs17` (and ≥ 6 of the `nvs*` set) certified within 60 s, `incorrect_count == 0`
across smoke + phase1, and the root bound on `nvs17` ≤ −2453 *reaching the global
dual bound of the search* (not just the discarded probe).
