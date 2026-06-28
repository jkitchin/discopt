# Issue #331 — Step 8: the sparse-cut LP layer — what the per-cut-node cost actually is

Steps 6–7 named the open frontier as "sparse cut representation in the LP" — the
hypothesis being that discopt's separated cuts are *dense rows* that defeat the
#334 sparse-LP engine, so every cut-augmented node pays a dense re-solve. Step 8
**measures** where the per-cut-node wall actually goes, before building. The
hypothesis does not survive the measurement: the LP layer is *already* sparse, the
cut rows are *not* the cost, and the real per-node cost is the dual-simplex pivot
loop. One concrete, sound, universal speedup falls out of the measurement and is
shipped.

## Grounding — per-node warm-solve cost breakdown (instrumented, real driver)

Every B&B node re-solves its LP with the warm dual simplex
(`solve_lp_warm_scaled`). Splitting that call's time on real sparse cut-augmented
instances (`smdk70x20`, `smdk80x25`, prod cuts, zero cold fallbacks):

| stage | what it does | share of warm-solve |
|---|---|---|
| `from_dense` (CSC rebuild) | scan dense `a` → compressed sparse columns | **~2–6.5 %** |
| rest of `prepare` (factorize + dual-feas check) | sparse LU of the basis | ~10–14 % |
| **`reoptimize` (dual pivots)** | ftran/btran + ratio-test pricing per pivot | **~84 %** |

Two facts kill the "dense cut rows" hypothesis:

1. **The matrix is already stored and solved sparsely.** `SparseCols` (CSC) backs
   the pricing, the LU factorize is sparse (`factorize_sparse`, O(nnz)), and the
   reduced-cost / ratio-test dots are sparse (`sp.dot`, O(nnz_j)). A cut row with
   few nonzeros *is* cheap in this layer; a dense GMI row costs only its own
   nonzeros. There is no dense `m×m` anywhere in the per-node path.
2. **`from_dense` — the one genuinely dense O(m·n) scan — is ≤6.5 %.** Even
   eliminating it entirely cannot be the "5× SCIP" closer Step 6 sought. The 84 %
   is the dual pivots, whose cost scales with the LP's *size* (more rows+cols ⇒
   bigger ftran/btran and a longer pricing pass), regardless of whether the cut
   rows are stored sparsely. Cuts cost wall because they make the LP **bigger**,
   not because they are stored **densely**.

So "sparse cut representation" as a *storage* change has a ~6.5 % ceiling. The
real lever for matching SCIP's per-cut-node cost is **incremental simplex updates**
(maintain `x_B`, `y`, and reduced costs across pivots instead of recomputing each
iteration) — a revised-simplex upgrade, not a cut-storage change, and out of scope
for a contained, provably-sound step.

## What was shipped — batch-level CSC reuse (the sound slice of the ceiling)

The measurement did expose one pure-waste redundancy worth removing: **every node
rebuilds the CSC from scratch**, even though the working matrix is *constant across
a whole B&B batch* (cuts fold in only between batches). The batch solves 64 nodes
— plus their strong-branch probes and dive re-solves — against one matrix, each
call doing its own O(m·n) `from_dense`.

Step 8 builds the CSC **once per batch** and shares it across every node solve in
that batch:

- `PreparedDual` now borrows a caller-supplied `&SparseCols` instead of rebuilding
  one; new entry point `solve_lp_warm_scaled_csc(lp, b, start, opts, &sp)`.
- The driver builds `SparseCols::from_dense(sa, …)` once per batch and threads it
  through `NodeCtx` to all four warm-solve sites (node LP, strong-branch prepare,
  strong-branch fallback, dive).

**Sound by construction** — same math, the cache is just the CSC of the same
matrix. Locked by a unit test (`csc_path_matches_rebuild_path`: the cached and
rebuild paths agree bit-for-bit in status/iters/objective) and, end-to-end, by
**identical node counts** on every bench instance (a pure-perf change cannot alter
the tree):

| instance | nodes (new == committed) | wall before | wall after | Δ |
|---|---|---|---|---|
| mdk120x15 | 4713 == 4713 | 0.300s | 0.252s | **−16 %** |
| mdk200x25 | 17935 == 17935 | 2.189s | 1.948s | **−11 %** |
| mdk150x20 | 1797 == 1797 | 0.335s | 0.314s | −6 % |
| mdk90x12 | 197 == 197 | 0.027s | 0.025s | −6 % |
| smdk50x15 | 703 == 703 | 0.136s | 0.115s | −15 % |
| smdk80x25 | 29147 == 29147 | 3.799s | 3.618s | −5 % |

Wall improves (or is at parity within noise) on **every** instance, dense and
sparse — it removes per-node work that exists whether or not cuts are present, so
even cut-free knapsacks benefit. The win exceeds the bare `from_dense` share
because the batch CSC is also shared across the many strong-branch probe solves
(each previously rebuilt the CSC in `PreparedDual::prepare`). This satisfies the
issue's net-wall-time-on-every-instance constraint with no node regression.

## Status

- **Hypothesis corrected by measurement:** the cut-augmented per-node cost is the
  dual pivot loop (~84 %), not dense cut storage (the layer is already sparse;
  `from_dense` ≤6.5 %).
- **Shipped:** batch-level CSC reuse — sound (identical trees, unit-tested),
  faster on every instance (−5 % to −16 % on the larger ones).
- **Remaining frontier (named, scoped, not built):** incremental dual-simplex
  updates to make each cut-enlarged node cheaper — the only thing that closes the
  per-cut-node cost to SCIP's level, and a revised-simplex project rather than a
  cut/data-layer tweak.

### Reproducing

The split was measured with a temporary `wsbench`-gated timer around
`PreparedDual::prepare` / `reoptimize` driving the real `solve_milp` on
`gen_sparse_mdk` instances (removed after measuring). The shipped win is verified
by `cargo test -p discopt-core csc_path_matches_rebuild_path` and the committed
`milp_node_efficiency.{md,json}` (node counts identical to the pre-change run,
wall lower).
