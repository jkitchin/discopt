# Parallelizing the Rust MILP engine with Rayon — design write-up

**Status:** implemented behind the `parallel` cargo feature (off by default); benchmarked.
**Scope:** the pure-Rust MILP branch-and-bound driver (`crates/discopt-core/src/bnb/milp_driver.rs`).
**Author:** design note + results.

## 0. Results (measured)

The map→reduce design below is implemented. The batch loop in `solve_milp` was
refactored into a per-node `solve_node` (pure over a shared `NodeCtx` snapshot) plus
a sequential, batch-ordered reduce. Parallelism is gated by the `parallel` feature
(`cargo build --features parallel`); the serial path is unchanged when it's off. The
Python binding (`solve_milp_py`) now copies its numpy inputs and runs the solve under
`py.allow_threads`.

Benchmark: 6 multidimensional 0/1 knapsacks solved to optimality
(`cargo run --release --example par_bench [--features parallel]`), 4-core machine:

| mode | threads | wall time | speedup | total nodes | total pivots |
|------|--------:|----------:|--------:|------------:|-------------:|
| serial | 1 | 55.04 s | 1.00× | 41 782 | 214 101 |
| parallel | 2 | 28.46 s | 1.93× | 41 782 | 214 101 |
| parallel | 4 | 14.96 s | 3.68× | 41 782 | 214 101 |

Near-linear scaling (92% efficiency at 4 cores). Crucially the **node and pivot
counts — and every per-instance objective — are bit-identical** across all three
configurations, confirming the parallel path explores the exact same tree as serial:
correctness and determinism are preserved, only wall-clock changes.

Not yet done (deferred, see §4/§7): parallel *strong branching* and making the
feature default. The size gate `PAR_MIN_BATCH` is set conservatively at 4.

## 1. Summary

The Rust-internal MILP solve (`solve_milp`) processes the B&B tree in **batches** of
up to 64 nodes, solving each node's LP relaxation sequentially. The per-node work in
a batch is independent and CPU-bound, which makes it a strong candidate for data
parallelism with [rayon](https://docs.rs/rayon). This document proposes a
**map → reduce** restructuring of the batch loop that keeps the solver fully
deterministic and preserves every existing correctness invariant, plus a secondary
opportunity inside strong branching. It also flags a GIL issue in the Python binding
that must be fixed for the speedup to reach Python callers.

This is a write-up only; it recommends an implementation plan and how to measure it.

## 2. Where the time goes (and what is parallelizable)

The hot loop is `solve_milp` at `crates/discopt-core/src/bnb/milp_driver.rs:246-422`:

```text
'search: loop {
    let batch = tm.export_batch(64);              // up to 64 open nodes
    for k in 0..batch.node_ids.len() {            // <-- SEQUENTIAL today
        build full_l / full_u for node k
        solve_lp / solve_lp_warm                  // dominant cost
        feasibility check
        optional: rounding heuristic              // try_rounding
        optional: cover separation                // separate_cover
        optional: strong branching                // 2 warm re-solves per candidate
        push NodeResult
        mutate tm: set_node_basis / inject_incumbent / set_branch_hint
    }
    tm.import_results(...); tm.process_evaluated();
    fold pending cuts into the working matrix
}
```

### Why the per-node body is safe to run in parallel

- **The working LP is immutable during a batch.** `a_w`, `b_w`, `c_w`, `slack_l`,
  `slack_u`, `m_w`, `n_w` are only mutated *between* batches (the cut-folding step),
  never inside the loop. So every node reads a stable snapshot.
- **The LP solver is a pure function.** `solve_lp` and `solve_lp_warm`
  (`crates/discopt-core/src/lp/simplex/`) take `&LpView`, `&[f64]`, `&Basis`,
  `&SimplexOptions` and return an owned `LpSolve`. A grep over `lp/` finds no
  `static mut`, `thread_local`, `RefCell`, or interior mutability — the only `unsafe`
  hits are in comments. The feral LU backend operates on owned per-call state. These
  functions are `Send`/`Sync`-clean.
- **`try_rounding`, `separate_cover`, `separate_gomory`, and `strong_branch`** are
  likewise pure given the immutable working LP and a per-node snapshot.

### What blocks naive parallelism

The loop interleaves **mutations of the shared `TreeManager`**:

| call | when | effect |
|------|------|--------|
| `tm.set_node_basis(id, b)` | after every optimal LP | stores warm-start basis on the node |
| `tm.inject_incumbent(x, o)` | rounding heuristic hit | may improve the incumbent |
| `tm.set_branch_hint(id, v)` | strong branching | records preferred branch var |

and **reads** of `tm`: `node_basis(id)`, `incumbent()`, `get_reliability_threshold()`,
`score_candidates(xs)`. All reads are `&self`; all writes are cheap bookkeeping. The
fix is to **defer the writes** out of the parallel region.

## 3. Proposed design: map → reduce per batch

Replace the sequential `for` with a **parallel map** that produces a per-node output
struct, followed by a **sequential reduce** that applies the `tm` mutations *in batch
order*.

```rust
/// Everything one node's parallel work produces; applied later, in order.
struct NodeOutput {
    result: NodeResult,
    basis: Option<Basis>,             // -> set_node_basis
    incumbent: Option<(Vec<f64>, f64)>, // -> inject_incumbent (heuristic)
    branch_hint: Option<usize>,       // -> set_branch_hint (strong branch)
    cuts: Vec<GomoryCut>,             // -> pending_cuts
    iters: usize,                     // -> lp_iters
    unbounded: bool,
    iter_or_numerical: bool,          // -> gap_certified = false
}

// --- snapshot read-only tm state needed during the map ---
let reliability = tm.get_reliability_threshold();
let inc_snapshot = tm.incumbent().map(|(_, v)| v);   // for the SB prunable test
let tm_ref: &TreeManager = &tm;                       // shared, immutable borrow

let outputs: Vec<NodeOutput> = (0..batch.node_ids.len())
    .into_par_iter()
    .map(|k| {
        // build full_l/full_u, solve LP (cold or warm via tm_ref.node_basis(id)),
        // feasibility, heuristic, cover sep, strong branch (uses inc_snapshot,
        // reliability, tm_ref.score_candidates(xs)) — all read-only.
        // returns NodeOutput
    })
    .collect();

// --- sequential reduce: deterministic, batch-ordered ---
for (k, out) in outputs.into_iter().enumerate() {
    let id = batch.node_ids[k];
    lp_iters += out.iters;
    if out.unbounded { unbounded = true; break 'search; }
    if out.iter_or_numerical { gap_certified = false; }
    if let Some(b) = out.basis { tm.set_node_basis(id, Some(b)); }
    if let Some((x, o)) = out.incumbent { tm.inject_incumbent(x, o); }
    if let Some(v) = out.branch_hint { tm.set_branch_hint(id, v); }
    pending_cuts.extend(out.cuts);
    results.push(out.result);
}
tm.import_results(&results);
tm.process_evaluated();
```

`TreeManager` is `Sync` (it holds only `Vec`, `HashMap`, `f64`, `bool`), so the shared
`&tm` borrow inside `into_par_iter` is sound. `Basis`, `NodeResult`, `GomoryCut` are
plain data and `Send`.

### Why determinism is preserved

The existing `test_determinism` (`tree_manager.rs:792`) and the non-negotiable
correctness invariant (`incorrect_count ≤ 0`) require that the search be reproducible.
This design keeps that:

1. **Node selection is unchanged** — `export_batch` still hands out the same 64 nodes
   in the same order.
2. **All `tm` mutations happen in the sequential reduce, iterated in batch index
   order** — identical to today's order. Parallelism only changes *when* an LP is
   solved, never the order in which results are folded into the tree, the incumbent,
   or the pseudocosts.
3. **The strong-branch `prunable` pre-check uses an incumbent snapshot** taken before
   the batch. A slightly stale incumbent can only change *whether we probe* a node
   (an effort decision), never the bound or the chosen child — strong branching is
   already documented as affecting "only the node count," not correctness.
4. Floating-point results are independent of thread scheduling because each LP solve is
   self-contained; there is no parallel reduction over floats.

Net: bit-for-bit identical trees to the serial path. The determinism test should pass
unchanged.

## 4. Secondary opportunity: parallel strong branching

`strong_branch` (`milp_driver.rs:467`) probes up to `sb_max_cands` candidates, each
with **two** warm dual re-solves. These probes are independent and read-only over the
node basis. They can be a nested `par_iter` with a deterministic reduce
(`max_by` on the product score, ties broken by lowest index to stay reproducible).

Caveat: nesting rayon inside the batch-level `par_iter` can oversubscribe. Options:
keep strong branching serial within each (already-parallel) node, or only parallelize
SB when the batch is small (few nodes, many candidates). The batch level captures the
majority of the available parallelism, so I'd ship that first and treat SB parallelism
as a follow-up gated by measurement.

## 5. The GIL issue in the Python binding (must-fix)

`solve_milp_py` (`crates/discopt-python/src/lp_bindings.rs:264-317`) calls
`core_solve_milp` **while holding the GIL** — there is no `py.allow_threads`. Two
consequences:

- The entire (potentially long) Rust solve blocks the Python interpreter today, even
  serially.
- More importantly, rayon worker threads doing the parallel solve would run *while the
  calling thread holds the GIL*. That is fine for the Rust compute itself (it touches
  no Python objects), but it needlessly freezes the interpreter and any concurrent
  Python work for the whole solve.

**Fix:** copy the borrowed numpy inputs into owned `Vec`s, then run the solve inside
`py.allow_threads(|| ...)`:

```rust
// PyReadonlyArray borrows require the GIL, so materialize owned copies first.
let a_owned = a_flat.to_vec();
let b_owned = b.as_slice()?.to_vec();
// ... c, lb, ub, into owned Vecs; build owned MilpOptions ...
let res = py.allow_threads(|| {
    let lp = LpView { a: &a_owned, m, n, c: &c_owned, l: &lb_owned, u: &ub_owned };
    core_solve_milp(&lp, &b_owned, obj_const, &opts)
});
```

This is independently worthwhile (it unblocks the interpreter during solves) and is a
prerequisite for parallelism to benefit real Python callers.

## 6. Expected gains and limits

- **Upper bound:** batch width is 64, so at most ~64× on the LP-solve portion, in
  practice bounded by core count and Amdahl's serial fraction (export/import/process,
  cut folding, basis stores).
- **Best case:** large MILPs whose node LPs take non-trivial pivots — most of the
  wall-clock is in `solve_lp_warm`, which now runs `min(cores, 64)`-wide.
- **Worst case:** tiny LPs where rayon task overhead dominates. Mitigation: gate the
  parallel path on a threshold (e.g. `batch.len() >= 8 && n_w >= some_dim`), falling
  back to the serial loop otherwise. This also keeps small-instance latency flat.
- **Memory:** each parallel node allocates its own `full_l`/`full_u` and `LpSolve`
  (already true today, just concurrently). Peak memory rises by ~`min(cores,64)×` the
  per-node working set — modest.

## 7. Implementation checklist (when greenlit)

1. Add `rayon` to `crates/discopt-core/Cargo.toml`. Pure-Rust, preserves clean-wheel
   builds. (MSRV 1.75 is fine; current rayon supports it.)
2. **Gate behind a cargo feature** (e.g. `parallel`, default off initially) so the
   serial path stays the reference until benchmarks land. Expose a runtime toggle /
   thread-count via `MilpOptions` and a `solve_milp_py` kwarg if desired.
3. Refactor the batch body into a free function `fn solve_node(...) -> NodeOutput`
   taking only `&`-shared inputs — this makes both the serial and parallel paths call
   the same code and keeps the diff reviewable.
4. Swap the batch `for` for `into_par_iter().map(solve_node).collect()` under the
   feature; keep the serial reduce.
5. Fix the GIL: `py.allow_threads` + owned input copies in `solve_milp_py`.
6. Add a **criterion bench** (no bench harness exists yet) over a handful of MILP
   instances of varying size to measure serial vs. parallel and pick the gating
   threshold. Wire `[[bench]]` + `criterion` as a dev-dependency.
7. Verify `test_determinism` and the full `pytest` correctness suite pass identically
   under the feature. Run `cargo test -p discopt-core --features parallel`.
8. Consider `RAYON_NUM_THREADS` / a configured global pool so benchmark and CI runs
   are reproducible and don't oversubscribe shared runners.

## 8. Risks

- **Determinism regressions** if any `tm` mutation leaks into the parallel region —
  mitigated by the strict map/reduce split and the existing determinism test.
- **Oversubscription** from nested SB parallelism — defer SB parallelism.
- **rayon overhead on small instances** — mitigated by the size gate and the
  feature flag.
- **No measured speedup** until the GIL fix lands for Python callers — sequence the
  GIL fix first.

## 9. Recommendation

Proceed in this order: (1) GIL fix in the binding, (2) extract `solve_node` + add the
criterion bench to get a serial baseline, (3) add the feature-gated `par_iter` batch
loop and measure, (4) tune the size gate, (5) only then consider parallel strong
branching. Keep correctness/determinism tests green at every step before making the
parallel path default.
