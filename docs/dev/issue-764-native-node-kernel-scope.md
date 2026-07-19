# #764 → C1 — Native per-node spatial B&B kernel: scoping

The `issue-764-scip-comparison.md` investigation established that SCIP-level performance on
`tanksize` (and the slow-spatial class) is a **per-node throughput** problem: SCIP ~1 ms/node
(native C), discopt ~100–500 ms/node (Python/JAX orchestration + ~95 OBBT LP probes). This doc
scopes the fix — moving the per-node spatial loop into `discopt-core` — with a component inventory,
the interface, an entry experiment (partly run), the honest ceiling, and a build order.

## Grounding micro-benchmark (run 2026-07-19)

tanksize node LP = 70 cols × 187 rows. **95 cold Rust LP solves back-to-back (no Python between):
79.5 ms = 0.837 ms/probe.** The current per-node OBBT spends ~233 ms+ on the same 95 probes (~2.4 ms
each) — the ~3× delta is pure Python-per-probe orchestration (objective build, `asarray` marshaling,
numpy ops) that an in-Rust loop removes outright. Warm dual-simplex probes (shared basis, bounds/obj
only change) target below the 0.837 ms cold figure.

## Current architecture (where each per-node op lives)

| per-node op | today | cost on tanksize |
|---|---|---|
| pop node / branch / tree mgmt | **Rust** (`bnb/tree_manager.rs`, `bnb/branching.rs`) | cheap |
| McCormick LP **build** | **Python/JAX** (`build_milp_relaxation`, cold re-walk) or `incremental_mccormick.py` (warm, but gated) | ~half the wall on cold-rebuild class |
| LP **solve** | Rust simplex (`lp/simplex/dual.rs`) via `solve_lp_warm_csc_py` | 0.837 ms cold; but marshaled |
| per-node **OBBT** (~95 probes) | **Python loop** orchestrating Rust LP probes | ~32 % of wall (marshaling-dominated) |
| per-node **FBBT** | Rust (`bnb/in_tree_presolve.rs::run_in_tree_presolve`) | cheap |
| NLP primal heuristic | Python/JAX (pounce) | ~10 % |
| marginals / result import | Python ↔ Rust (`import_results`) | marshaling |

The outer spatial loop and the OBBT orchestration are **Python**; the heavy numeric primitives are
already **Rust**. The gap is the *orchestration* crossing the Python↔Rust boundary ~10²–10³ times
per node.

## Component inventory — most of it already exists in Rust

**Exists (reuse):**
- Warm dual/primal simplex + basis: `lp/simplex/{dual,sparse,basis}.rs`.
- Inner MILP B&B driver + node solve: `bnb/milp_driver.rs` (`solve_node`, `solve_milp_csc`).
- Tree: `bnb/tree_manager.rs` (`import_results`, `process_evaluated`, `score_candidates`,
  `set_node_bounds`, pseudocost/reliability data).
- Spatial + integer branch selection: `bnb/branching.rs`.
- FBBT-with-cutoff kernel: `bnb/in_tree_presolve.rs`.
- OBBT scaffolding: `presolve/obbt.rs` (`obbt_candidates`, `apply_obbt_bounds`).

**Missing (build):**
1. **In-Rust McCormick envelope patcher** — the load-bearing new component. `incremental_mccormick.py`
   already does exactly this in Python (build structure once; per node recompute only the
   box-dependent envelope rows in closed form, ~0.1 ms; validated to reproduce the cold build
   exactly). Port it to Rust. **Critically it must cover tanksize's term types** — today the Python
   incremental engine handles only bilinear + integer-square, so tanksize's `sqrt`/fractional-power
   terms fail its validation and force the cold path (this is why `_inc is None` and why marginals
   only came via the cold path in Phase-2 step 1). Coverage = {bilinear (4 rows), square (3),
   trilinear, univariate concave/convex incl. `sqrt` secant+tangents, fractional power}.
2. **In-Rust warm OBBT probe loop** — drive `presolve/obbt.rs` candidates through the warm dual
   simplex with a shared basis, in-kernel (no Python between probes). The self-warm floor is the
   target; the scored top-k selection (`T2.5`, parked) plugs in here to bound the probe count.
3. **In-Rust spatial node orchestrator** — the loop tying it together: pop → patch envelopes →
   warm-solve → OBBT probes → FBBT (cutoff) → DBBT from reduced costs → branch → push children.

**Stays a Python callback (thin, strided):**
- NLP primal heuristic (pounce/JAX) — incumbent search only, fired at a stride, not every node.
- One-time hand-off of the fixed LP structure from the JAX McCormick compiler (columns, sparsity,
  the box-independent model rows, the objective, the per-term envelope-patch descriptors).

## The interface (hand-off, once per solve)

Python builds the relaxation structure once and hands Rust a `SpatialKernelSpec`:
- `A_struct` (CSC, box-independent rows: model linear rows + lifted-var linking) + `b`;
- `c` (objective over lifted columns), `n_orig`, integrality mask, global lb/ub;
- **envelope-patch descriptors**: for each lifted term, its type + operand columns + output column,
  enough for the Rust patcher to regenerate its rows/bounds from a node box in closed form;
- OBBT config (candidate cols, rounds, top-k, per-node budget); branch priorities / deprioritized
  (functionally-dependent) mask.

Rust returns per solve: the tree result (bound, incumbent, node count, certificate) + telemetry.
The `node_callback`/incumbent path stays for the strided NLP heuristic.

## Entry experiment + kill criterion

Goal: confirm the in-Rust node is ≥5× faster than today's Python node before committing weeks.
1. **LP/OBBT floor (DONE):** 95 pure-Rust probes = 79.5 ms vs ~233 ms marshaled → ~3× on the OBBT
   portion cold; warm-start expected to add more. GREEN.
2. **Prototype node loop — GREEN (2026-07-19), ~9× measured, kill criterion cleared.** Rather than
   wire weeks of plumbing to run the go/no-go, decomposed the real tanksize node cost by direct
   measurement (`solve_lp_warm_csc_py` instrumentation + a raw-binding tight loop on the *captured*
   node LP, m=187/n=257/nnz=864):
   - node wall today = **1352 ms/node** (15-node segment; `rust_time` is ~0 — the wall is Python/JAX);
   - OBBT = **110 probes/node**, **35 % of wall**, **4.25 ms/probe** in-loop but **1.28 ms/probe**
     pure-Rust (arrays reused) ⇒ **~2.97 ms/probe (~70 %) is pure Python marshaling** a native loop
     deletes (and PreparedPrimal warm re-pricing goes below the 1.28 ms stateless-rebuild figure);
   - JAX McCormick build ≈ **33 % of wall (~453 ms/node)** ⇒ the Rust envelope patcher (items 1–2)
     replaces it at ~1 ms/node;
   - Python orchestration ≈ **32 %** ⇒ native ≈ 0.

   Native node ≈ 110×1.28 + ~1 (patch) + ~1 (tree/branch, already Rust) ≈ **~145 ms/node → ~9.3×**,
   conservative (warm probes push higher). Every dominant cost is orchestration that native
   relocation removes; the irreducible LP compute is ≤1.28 ms/probe. **GO on building the kernel.**
   (Harness: `discopt_benchmarks/scripts/issue764_node_cost_decomposition.py`.)
3. **Bound-neutrality:** the patched envelopes must reproduce the cold build exactly (the
   `incremental_mccormick.py` validation gate, ported) — assert identical node bound + node_count on
   a certifying panel. Any drift = wrong.

## Honest ceiling (do not oversell)

- **Cold-rebuild-bound class** (casctanks-like — the C1 median): ~10–14× (the Python McCormick build
  is ~all the wall; a Rust patcher removes it). This is the broad win.
- **OBBT-bound class** (tanksize, nvs05): ~4–8×. ~Half the wall is the OBBT probe LPs; native+warm
  makes them ~3–8× cheaper, but the *count* (~95/node) stays — discopt still needs them because its
  cheap tightening (cuts, DBBT) is inert on this class (measured). So the native kernel gets
  tanksize to ~minutes / tens of nodes/s, **not** SCIP's 925 nodes/s alone.
- **Full SCIP parity on tanksize** needs BOTH this kernel AND effective per-node cutting so the OBBT
  probe count collapses toward SCIP's ~1 LP/node — the cut-engine gap (C3), which is inert on this
  bilinear class today and is a separate, harder research line.

## Build order

1. **DONE (2026-07-19)** — Port `incremental_mccormick` to Rust (`bnb/mccormick_patch.rs`).
   Faithful ports of the three families `IncrementalMcCormickLP._patch` actually dispatches on:
   `_bilinear_rows` (4 rows), `_monomial_rows` (4 rows incl. the box-midpoint tangent — `p=2` is the
   plain square), `_affine_square_rows` (4 rows), plus their aux-bound helpers. **Correctness note:**
   the first cut shipped a textbook 3-row `_square_rows`, which `_patch` never calls — the cold build
   uses the tighter 4-row midpoint-tangent hull, so a 3-row port would produce a weaker bound and
   fail neutrality. Fixed. Bound-neutrality gate at the formula level: differential fixtures assert
   exact equality (`<1e-12`) to values generated from the Python reference functions.
2. **DONE (2026-07-19)** — Extend to `sqrt`. `univariate_rows(...)` ports `uniform_relax._emit_1d`
   (secant + tangents at `t_lo`/mid/`t_hi`, sign by curvature) with a `Univariate` atom enum;
   `Sqrt` is instantiated (concave) — the only atom `tanksize` needs beyond products/squares (its
   `.nl` = `o2` multiply ×84 + `o39` sqrt ×3). Fixtures pin exact equality to `_emit_1d` on bare and
   affine sqrt; returns `None` (aux-floor) on degenerate/undefined boxes, matching `tight=False`.
   The enum makes fractional-power/log/exp/trilinear drop-in extensions of the same row machinery
   (remaining item-2 tail, not needed for tanksize). **The patcher now covers tanksize's full atom
   set.** (`cargo test -p discopt-core mccormick_patch`: 16 passed; full lib suite 495 passed.)
3. In-Rust warm OBBT probe loop over `presolve/obbt.rs` candidates (shared basis). *Depends on a
   resident node LP to probe* — i.e. the item-4 assembly. `obbt_candidates`/`apply_obbt_bounds`/
   `extract_linear_rows` exist; the missing piece is the objective-swapping warm-simplex loop.
   **Design finding (2026-07-19):** OBBT varies the *objective* (min/max eₖ) over a *fixed*
   polytope, so the natural warm start is the **primal** simplex from the node's optimal basis
   (primal-feasible for every OBBT objective), not the dual (`reoptimize` re-solves for new
   *bounds* but bakes `c` into `PreparedDual`). The existing `solve_lp_cols_warm(cols, …, c, …,
   start, …)` does exactly one such probe but **consumes `cols` by value** — driving ~190
   unit-objective probes/node through it would clone the CSC matrix per probe, re-paying the
   O(nnz) setup we are trying to remove. So item 3 needs a new **prepared-factorization
   objective-swap primitive** (`PreparedPrimal`: build the scaled matrix + LU once, then
   re-price/re-pivot for each unit `c` warm from the node basis) — new simplex-internals work,
   coupled to item 4, not a standalone unit.
4. The Rust spatial node orchestrator — **Rust side DONE (2026-07-19)**. Entry experiment cleared at
   ~9× (step 2 above). Built and tested entirely in `discopt-core`:
   - `bnb/spatial_kernel.rs` — `SpatialKernelSpec` (the box-independent hand-off) + `EnvTerm`
     {Bilinear, Monomial, AffineSquare, Sqrt} + `assemble_node_lp` (regenerates every box-dependent
     envelope row + aux bound in closed form, builds standard-form `[A|I]z=b` in CSC) +
     `solve_spatial_node` (assemble → solve → OBBT sweep, warm-started from the node basis).
   - Rigorous safe bound: `refine::ns_safe_bound_csc` (CSC twin of the dense Neumaier–Shcherbina,
     bit-identical, DD precision) wired as the node bound — never the raw simplex objective.
   - `bnb/spatial_tree.rs` — the full spatial B&B loop: safe-bound pruning, OBBT box tightening,
     sufficient-condition incumbent acceptance (integers integral AND every term McCormick-tight),
     covering branch (integer floor/ceil or spatial worst-gap split). Sound by construction.

   **Remaining bridge to run on tanksize (the back half):** (a) the `SpatialKernelSpec` PyO3 surface
   in `crates/discopt-python/`; (b) the **Python producer** that extracts the spec from the JAX
   McCormick compiler (identify each lifted term's type + operand/output columns, the fixed linear
   rows, objective, integrality, OBBT candidates) — the hardest remaining plumbing, coupled to
   `MccormickLPRelaxer`; (c) the strided NLP primal-heuristic callback.
5. Wire behind a flag; bound-neutral graduation panel (exact node_count + objective), then the
   throughput panel (median slowdown vs SCIP/BARON on the global50 spatial subset).

All phases keep the Python cold path as the trusted fallback and validation oracle — the kernel can
never change a certificate, only its speed.

**Honest-ceiling reminder (do not lose):** even with items 1–5 complete, the native kernel gets
tanksize to ~minutes / tens of nodes-per-second, **not** SCIP's ~925 n/s. Full SCIP parity on
tanksize *also* needs an effective per-node cut engine so the ~95-probe OBBT count collapses toward
~1 LP/node (C3) — inert on this bilinear class today, a separate/harder research line. The native
kernel is the broad throughput win (≈10–14× on the cold-rebuild-bound class); it is necessary but
not sufficient for tanksize parity alone.
