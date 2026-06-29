# Issue #331 — Step 9: incremental dual simplex — closing the per-cut-node cost

STEP8 measured that the per-cut-node wall cost on the sparse frontier is the
**dual-simplex pivot loop** (~84% of each warm re-solve), not cut storage (the LP
layer is already sparse). It named the fix: maintain the simplex state across
pivots instead of recomputing it from scratch each iteration. Step 9 builds that.

## What the loop did before

Every dual iteration in `run_dual` recomputed, from scratch:

1. **basic values** `x_B = B⁻¹(b − Σ_nonbasic A_j x_j)` — an O(nnz) scatter + an
   ftran;
2. **dual variables** `y = B⁻ᵀc_B` — a second btran — and then
3. **reduced costs** `d_j = c_j − yᵀA_j` for every nonbasic column — a second
   O(nnz) pass.

So each pivot paid **2 ftran + 2 btran + 2 full passes over A**, even though a warm
re-solve typically takes only a handful of pivots and the state changes by a
rank-1 update each time.

## What Step 9 changes

`x_B` and the reduced costs `d` are now **maintained incrementally** across pivots
(textbook revised dual simplex):

- **reduced costs**: after a pivot on `(row r, entering q)`, `d_j −= (d_q/α_rq)·α_rj`
  for every nonbasic `j` (the `α_rj = ρ·A_j` already computed for the ratio test),
  the leaving column takes `−d_q/α_rq`, and `d_q := 0`. The per-iteration
  `y = B⁻ᵀc_B` btran and the reduced-cost pass are gone.
- **basic values**: the bound flips' RHS effect (`x_B −= B⁻¹ Σ_flipped A_jΔx_j`,
  one aggregated ftran) plus the entering step (`x_B −= t·α_q`, reusing the `α_q`
  ftran already needed for Devex). The full O(nnz) scatter + recompute ftran is
  gone.

### Soundness by construction

The maintained `x_B`/`d` drive **pricing only** — which row leaves, which column
enters — which the engine already documents as never affecting correctness. The
**returned result is always computed from exact recomputes**:

- `x_B` and `d` are recomputed exactly on a fixed cadence (every 32 iterations)
  and after every LU refactorization;
- optimality is declared only after an **exact** `x_B` shows primal feasibility
  **and** an **exact** `d` passes a dual-feasibility check;
- infeasibility is declared only after confirming the empty candidate set on
  **exact** `d`;
- any near-zero pivot or numerical difficulty still falls back to the robust cold
  primal solve.

So a drift or bug in an increment can only cost extra iterations or trigger the
cold fallback — it can never return a wrong optimum. Guarded by
`warm_random_lps_match_cold` and a new `warm_incremental_matches_cold_under_many_pivots`
stress test (larger LPs, heavy bound-flipping, forced refresh/refactor crossings;
asserts the warm path actually runs and every result matches the cold optimum).

## Result — faster on every instance, sound

Measured vs the pre-change merged engine (same machine; objectives cross-checked
against SCIP — **all match**; node counts essentially unchanged, since pricing
barely shifts):

| instance | nodes | wall before | wall after | Δ |
|---|---|---|---|---|
| **smdk50x15** | 955 | 0.085s | 0.071s | **−16%** |
| **smdk60x20** | 1189 | 0.171s | 0.143s | **−16%** |
| **smdk70x20** | ~6.2k | 0.589s | 0.500s | **−15%** |
| **smdk80x25** | ~40.6k | 4.712s | 4.018s | **−15%** |
| mdk90x12 | 281 | 0.038s | 0.024s | −37% |
| mdk120x15 | 5113 | 0.243s | 0.226s | −7% |
| mdk150x20 | 1725 | 0.173s | 0.158s | −9% |
| mdk200x25 | 18213 | 1.987s | 1.884s | −5% |

The whole **sparse frontier drops ~15% in wall** with identical objectives and
node counts — a direct per-node-cost win, exactly the lever STEP8 isolated. Tiny
sub-30ms instances are at parity within noise.

## Status

- **Per-cut-node cost reduced** (the STEP8 lever): ~15% faster warm re-solves on
  the sparse frontier, ~5–37% on dense, sound (exact-recompute guards + cold
  fallback; objectives match SCIP).
- The remaining sparse-frontier gap is now **node count** (e.g. smdk80x25 ~40k vs
  SCIP ~750) — #340's set-covering-tuned strong-branch budget over-trims on these
  knapsacks. That is a branching/budget-tuning item, separate from the LP-layer
  per-node cost this step closed.

### Reproducing

```bash
python -m discopt_benchmarks.perf.milp_node_efficiency --out discopt_benchmarks/results/issue331
cargo test -p discopt-core warm_incremental_matches_cold_under_many_pivots
```
