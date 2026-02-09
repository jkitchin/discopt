# Sparse Solvers Roadmap for discopt

## Context

The discopt IPM solver is entirely dense — `jnp.linalg.solve()` for KKT systems, `jax.hessian()`/`jax.jacobian()` for derivatives. This limits practical problem size to ~500-1000 variables. Real-world MINLP instances often have 5K-50K+ variables with 1-5% Jacobian density, meaning 95%+ of the KKT matrix is zeros. Adding sparse solvers will unlock these problem sizes incrementally.

**Approach**: Extend the pure-JAX IPM (the default B&B path) with sparse linear algebra. Sparsity patterns extracted from expression DAGs at build time. Python/JAX side only (Rust sparse deferred). No new required dependencies — scipy.sparse is already available.

---

## Phase 1: Sparsity Detection + Sparse Jacobians

**Goal**: Extract sparsity patterns from expression DAGs, compute Jacobians via graph coloring + compressed JVPs. Reduces Jacobian cost from O(n) to O(p) evaluations where p = chromatic number (typically 5-20).

### New files

- `python/discopt/_jax/sparsity.py` (~300 lines)
  - `collect_variable_indices(expr, model) -> set[int]` — recursive DAG walk, collects Variable offsets
  - `SparsityPattern` dataclass — stores `jacobian_sparsity` (scipy CSR, m×n bool), `hessian_sparsity` (CSR, n×n bool), density stats
  - `detect_sparsity_dag(model) -> SparsityPattern` — walks each `model._constraints[i].body` to collect per-constraint variable incidence; Hessian sparsity from variable co-occurrence in product/matmul nodes
  - `compute_coloring(pattern) -> (colors, n_colors)` — greedy graph coloring on column intersection graph
  - `make_seed_matrix(colors, n_colors, n) -> jnp.ndarray` — (n, p) seed matrix for compressed JVPs
  - `should_use_sparse(pattern, threshold=0.15) -> bool` — auto-selection gate

- `python/discopt/_jax/sparse_jacobian.py` (~200 lines)
  - `sparse_jacobian_jvp(fn, x, seed_matrix, pattern) -> scipy.sparse.csc_matrix` — compressed forward-mode evaluation
  - `make_sparse_jac_fn(con_fn, pattern, colors, seed_matrix)` — returns callable `sparse_jac(x) -> csc_matrix`

- `python/tests/test_sparsity.py` (~400 lines)

### Files to modify

- `python/discopt/_jax/nlp_evaluator.py` — add `sparsity_pattern` property, `evaluate_sparse_jacobian(x)` method

### Dependencies

- None new (scipy.sparse already available)
- Optional: `sparsejac` (MIT) as alternative coloring engine

### Definition of done

- Sparse Jacobian matches dense within atol=1e-10 on all `test_correctness.py` problems
- Faster than dense for n >= 200 with density <= 10%
- No regression on small problems (auto-gate prevents sparse path for n < 50)

---

## Phase 2: Sparse KKT Assembly + Direct Solve

**Goal**: Replace dense `jnp.linalg.solve` and `jnp.linalg.eigvalsh` with sparse equivalents for the IPM. Uses Python while-loop (scipy sparse ops are not JAX-traceable).

### New files

- `python/discopt/_jax/sparse_kkt.py` (~350 lines)
  - `assemble_kkt_sparse(H, J, sigma, D, delta_w, delta_c) -> scipy.sparse.csc_matrix`
  - `solve_kkt_direct(kkt_csc, rhs) -> np.ndarray` — via `scipy.sparse.linalg.spsolve`
  - `detect_inertia_sparse(H, n) -> (bool, float)` — via LU pivot signs or `eigsh(k=1, which='SA')`

- `python/discopt/_jax/sparse_ipm.py` (~500 lines)
  - `sparse_ipm_solve(obj_fn, con_fn, x0, bounds, sparsity, options) -> IPMState` — Python while-loop with per-iteration JIT calls for evaluations, scipy for linear algebra
  - Same `IPMState` NamedTuple as dense path for interchangeability

- `python/tests/test_sparse_kkt.py` (~300 lines)

### Files to modify

- `python/discopt/_jax/ipm.py` — add `linear_solver="sparse_direct"` to `IPMOptions`; dispatch to `sparse_ipm_solve` when selected
- `python/discopt/_jax/nlp_evaluator.py` — add `evaluate_sparse_hessian(x, y)` using Phase 1 coloring infrastructure

### Architecture note

The sparse IPM uses a **Python while-loop** (not `jax.lax.while_loop`). Per-iteration JAX work (objective, gradient, constraints, line search) stays JIT-compiled. Only KKT assembly/solve exits JIT. This matches how IPOPT and other production IPM solvers work.

### Dependencies

- `scipy.sparse.linalg` (already available)
- Optional: `klujax` (LGPL-2.0) for JAX-native sparse direct solve

### Definition of done

- Sparse IPM matches dense IPM objectives within atol=1e-4 on correctness suite
- Wall-clock speedup >= 2x at n=1000/5% density, >= 10x at n=5000
- Dense path unchanged (zero regression)

---

## Phase 3: Solver Integration + L-BFGS Option

**Goal**: Wire sparse IPM into `solver.py` for end-to-end `Model.solve()`. Add automatic dense/sparse selection. Add L-BFGS quasi-Newton for memory-limited cases.

### New files

- `python/discopt/_jax/sparse_hessian.py` (~250 lines) — symmetric coloring for Hessian, L-BFGS two-loop recursion (Hv products only, no matrix formed)
- `python/tests/test_sparse_ipm_integration.py` (~400 lines)

### Files to modify

- `python/discopt/solver.py` — in `solve_model()`, call `detect_sparsity_dag(model)` after building evaluator; add `nlp_solver="sparse_ipm"` dispatch; add auto-selection logic:
  ```
  total < 500 or density > 0.15  -> dense
  500 <= total <= 50000          -> sparse_direct
  total > 50000                  -> sparse_iterative
  ```
  Sparsity pattern is computed once and reused for all B&B nodes (same constraint structure).

- `python/discopt/_jax/ipm.py` — add `"lbfgs"` option to `IPMOptions.linear_solver`

### Definition of done

- `Model.solve()` works end-to-end on 5K-50K variable NLPs
- Automatic selection correct on existing test suite (small -> dense, large sparse -> sparse)
- B&B with sparse node solves works on MINLP with 1000+ continuous variables

---

## Phase 4: Very Large Scale + Parallel B&B

**Goal**: Push to 50K+ variables with iterative solvers, warm-start factorizations, and threaded B&B.

### Components

- **Warm-start factorization**: Reuse symbolic factorization across IPM iterations (`scipy.sparse.linalg.splu` symbolic once, numeric refactor each iteration)
- **Iterative KKT solve**: `jax.scipy.sparse.linalg.cg` / `gmres` with incomplete Cholesky preconditioner — matrix-free, O(n) memory
- **Threaded B&B batch**: `concurrent.futures.ThreadPoolExecutor` for parallel per-node sparse IPM (scipy releases GIL during factorization)
- **Matrix-free IPM**: Hessian-vector products only via `jax.jvp(jax.grad(...))`, never forming H

### New files

- `python/discopt/_jax/sparse_batch.py` — thread-pooled sparse IPM for B&B
- `python/discopt/_jax/matrix_free_ipm.py` — fully matrix-free IPM variant

### Optional library integrations (detected at import time)

- `klujax` (LGPL-2.0) — JIT-compatible sparse direct solve
- `lineax` (Apache-2.0) — linear solver abstraction with better preconditioner support

### Definition of done

- 50K-variable NLP in < 5 min, < 8GB RAM
- Matrix-free mode handles 100K+ variables
- Thread-parallel B&B shows near-linear speedup

---

## Data Flow

```
Model._constraints[i].body  (Expression DAG)
  -> sparsity.collect_variable_indices()  [DAG walk, Phase 1]
    -> SparsityPattern  (scipy CSR bool matrices)
      -> compute_coloring()  -> seed_matrix
        -> sparse_jacobian_jvp()  -> scipy.sparse.csc_matrix  [Phase 1]
        -> sparse_hessian_eval()  -> scipy.sparse.csc_matrix  [Phase 2]
          -> assemble_kkt_sparse()  -> scipy.sparse.csc_matrix  [Phase 2]
            -> spsolve() or cg()  -> dx, dy  [Phase 2/4]
```

Sparsity pattern computed **once** at model build time. Reused across all B&B nodes (constraint structure doesn't change, only variable bounds).

## Sparse Format Choices

| Stage | Format | Reason |
|-------|--------|--------|
| Sparsity detection | scipy CSR (bool) | Efficient row slicing per constraint |
| Seed matrix | dense jnp.ndarray (n x p) | Small, JAX-traced for JVPs |
| Jacobian/Hessian | scipy CSC | Required by spsolve/splu |
| KKT matrix | scipy CSC | Required by spsolve/splu |
| Iterative solve | matrix-free matvec | No matrix stored; `jax.scipy.sparse.linalg.cg` |

## EPL-2.0 Compatible Libraries Summary

| Library | License | Role | Required? |
|---------|---------|------|-----------|
| scipy.sparse | BSD-3 | Core sparse storage + direct solve | Yes (already dep) |
| sparsejac | MIT | Sparse Jacobian via coloring | Optional |
| klujax | LGPL-2.0 | JAX-native sparse direct solve | Optional |
| lineax | Apache-2.0 | Linear solver abstraction | Optional (Phase 4) |
| faer-sparse | MIT | Rust sparse LDLT (future) | Deferred |

## Verification

After each phase:
1. Run `pytest python/tests/ -v` — all existing tests must pass
2. Run new sparse-specific tests — equivalence with dense path
3. Run `pytest python/tests/test_correctness.py` — zero incorrect results
4. Performance benchmarks: sparse vs dense at n=500, 1K, 2K, 5K with ~5% density
5. `ruff check python/` and `ruff format --check python/` clean
