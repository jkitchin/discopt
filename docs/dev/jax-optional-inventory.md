# JAX as an optional extra — inventory (#1873)

Status: **done** for the packaging change. `jax`/`jaxlib` moved from `[project]
dependencies` to the `jax` extra (`pip install 'discopt[jax]'`). `dev`, `all`, `gnn` and
`learned` include it. This document is the inventory step #1873 asks for: every path
that still reaches JAX, what it costs a JAX-less install, and what would remove it.

The mechanism is `python/discopt/_optional_jax.py`:

- `require_jax(feature)` / `missing_jax_error(feature)` raise `MissingJaxError`, which
  subclasses `ModuleNotFoundError` with `name="jax"`. Any existing `except ImportError`
  therefore behaves as it did when `import jax` failed on its own. The message names
  the feature and `pip install 'discopt[jax]'`.
- Every module with a module-scope `import jax` checks `jax_available()`
  before that import (37 such modules at first; 15 remain since 22 were retired with the
  McCormick-"nlp" route, §3). `jax_available()` uses `importlib.util.find_spec`, so it
  never imports JAX.
- Function-level JAX users on a public path call `require_jax(...)` before importing it.

Guards (`python/tests/test_1873_jax_optional.py`):

- the whole in-repo `.nl` corpus is solved in a fresh interpreter. `jax` is blocked
  there by a `sys.meta_path` finder that *records the stack of every attempt* before
  refusing it. A caller that catches the `ImportError` and degrades is caught as well
  as one that crashes (CLAUDE.md §7);
- every module-scope importer raises the named error rather than a bare
  `No module named 'jax'`;
- the evaluator fallback and `dm.custom` refuse with the named error.

CI's `Python without JAX` job runs `pytest -m smoke` with JAX uninstalled, plus that
file.

## Measurements

All measured 2026-10-22 on this branch (Linux, Python 3.13, 4 cores).

- **Corpus guard, JAX blocked:** all **66** in-repo `.nl` instances
  (`python/tests/data/minlplib_nl/`) solve at `time_limit=10` with **0** attempts
  to import `jax` and 0 exceptions (`test_the_whole_in_repo_corpus_...`, 275 s).
- **`pytest -m smoke` with jax/jaxlib uninstalled, before the test-side work:**
  118 collection errors, all `No module named 'jax'`: 53 test modules import
  JAX directly and 65 import a JAX-only `discopt` module (the legacy evaluator or
  DAG compiler as an oracle). After `importorskip`: 3418 passed, **82 failed, 14
  errors**. Triage of those 96:
  - **Library paths that used JAX on ordinary, non-JAX models. These were real
    bugs for a JAX-less install, and the corpus guard cannot see them because
    `.nl` solves never reach them:** `discopt.mo` cross-evaluated objectives
    through `dag_compiler` (epsilon-constraint, AUGMECON2, NBI/NNC, weighted sums),
    the GDP simplex lowering's source residuals, and the nonsmooth lift's
    warm-start completion. All three now evaluate on the POUNCE tape
    (`discopt/_point_eval.py`), with the JAX compiler kept only as the fallback for
    an expression the tape cannot lower. Tape and JAX values agree to
    **1.0e-14 relative** over **3066** comparisons, covering every
    objective/constraint body of the 66 corpus instances (up to 40 per instance) at
    3 random in-box points each, with identical finiteness. These are point values
    (reports, warm starts), not bounds.
  - A test fixture that imported JAX only to clear its cache
    (`test_amp_integration.py`), and an mpec test that used `dm.custom` only to
    get an opaque node (it now builds `CustomCall` directly).
  - The rest, 64 node IDs in 47 functions, genuinely exercise JAX features:
    `dm.custom`-based DIRECT/surrogate solves, the JAX evaluator as an oracle,
    `DISCOPT_ANALYTIC_SEPGRAD`, and trainable NN surrogates whose stacked
    matmul has no tape opcode (§1). They are marked `requires_jax`. (The
    trainable-surrogate group was un-marked in the follow-up that lowered
    stacked matmul and array-valued Gauss-Newton residuals; §1.)
- **After those changes:** `pytest -m smoke` without JAX gives **3449 passed, 213
  skipped, 0 failed** (434 s). With JAX installed it gives **3768 passed, 24
  skipped, 0 failed** (466 s).
- **Notebooks:** 7 of 70 in `docs/notebooks/` import JAX directly, and others
  use `dm.custom`, estimation, sensitivity or differentiable layers. The docs
  build installs JAX explicitly (`docs.yml`), so it is unaffected.

## 1. The evaluator fallback

`build_evaluator` falls back to the JAX `NLPEvaluator` when `_nl_expr_compiler`
raises `UnsupportedForTape`. Without JAX, `_tape_nlp_evaluator.jax_fallback_evaluator`
re-derives the tape's reason and refuses with it. For example:

> Evaluating this model needs the legacy JAX NLP evaluator because the POUNCE tape has
> no opcode for part of it (norm2 of a 2-D argument is jnp.linalg.norm's MATRIX norm
> (induced/spectral), which is not a fold over elements); it requires JAX, which is
> not installed. … install it with `pip install 'discopt[jax]'`.

Every `UnsupportedForTape` raise in `_nl_expr_compiler.py`, the modeling-API
construct that reaches it, and what removing the fallback would take:

| Raise (function) | Reached by | Tape work to remove it |
|---|---|---|
| `_Budget.charge` (node cap) | a dense array body over `_MAX_TAPE_NODES` (e.g. a large `A @ x` written as an expression) | none. This is a cost refusal, and the LP/QP callers already read such rows off `_affine_constraint_rows` first (#1743) |
| `_const_array` | a non-numeric constant in an expression | none. This is a malformed model, and JAX would fail on it too |
| `_map` (> 2 operands) | no current caller (defensive) | none |
| `compile_to_nl_expr` (array-valued) | a scalar-only caller given an array body; `TapeNLPEvaluator` flattens bodies, so not on the evaluator path | none |
| `rec`, binary/unary operator | an operator outside `+ - * / ** neg abs`; the modeling layer emits none | none |
| ~~`rec`, empty `SumExpression`~~ | `dm.sum` of a zero-size array | **Done** (second follow-up): `_empty_sum` lowers it to `0.0` constants of numpy's reduced shape, as `jnp.sum` does |
| `rec`, `IndexExpression` not static | an index the compiler cannot resolve to a slot, including out-of-range (numpy raises; `jnp` clamps) | none. Refusing is correct, and with JAX the fallback clamps silently |
| `rec`, `CustomCall` | `dm.custom`, `dm.implicit` (and `dm.argmin` layers); ~~`dm.external`~~ | `dm.external`: **done** (#1903). It supplies numpy value/Jacobian/Hessian, so it is a numpy-evaluated *hole* in the tape (`discopt/_tape_holes.py`), folded back in by the chain rule. A hole nested inside another hole's argument is still refused. `dm.custom`/`dm.implicit`/`dm.argmin`: **impossible by design** -- the body is an opaque JAX callable that JAX differentiates, so these stay `[jax]`-only |
| `rec`, unknown node type | a node class the compiler does not know (none today) | none |
| `_matmul` (scalar operand, stacked, shape mismatch) | `@` with a 0-D operand, >2-D operands, or mismatched shapes | moderate for stacked (batched) matmul; the others are malformed models |
| `_sum_along` (axis out of range, repeated axis) | `dm.sum(x, axis=…)` with a bad axis | none (malformed) |
| `_broadcast_size` | operands that do not broadcast | none (malformed) |
| `_require_fused_xlog` | `entropy`/`centropy` with a pounce build older than #489 | none. Upgrading pounce removes it |
| `_lower_function` (`function {name!r}`) | a `FunctionCall` name with no lowering. Every name `dm.*` emits is lowered today (`exp log log10 log1p log2 sqrt sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh erf abs sign min max atan2 sigmoid softplus entropy centropy signpower prod norm*`), so only a hand-built node reaches it | none |
| `_fold` (empty) ~~/ deeper than `NlExpr.max_depth`~~ | `dm.prod`/`norm` of an empty array; ~~or of more elements than the tape's nesting depth~~ | Depth: **done** (second follow-up), a balanced pairwise tree from the limit up (the left fold below it is unchanged, bit for bit). Empty: none -- `jnp.max` of an empty array raises, so refusing is right |
| `_lower_norm` (order; matrix norm; scalar) | `dm.norm(X, ord)` with 2-D `X` (formerly the CLAUDE.md case: 219 `jax` modules), or a 0-D argument | `ord=1`/`inf` of a matrix: **done** (second follow-up), max column/row absolute sum. `ord=2`: **done** (#1903) -- the spectral norm is a numpy hole (SVD value, `u1 v1^T` gradient, Jordan-Wielandt Hessian), not a POUNCE opcode; the earlier claim here that it *needed* one was wrong (retracted on #1903). It also relaxes and certifies (`uniform_relax._build_spectral_norm`). Matrix norms of other orders, and >2-D arguments, are refused -- `jnp.linalg.norm` refuses them too. `"fro"` is refused at `dm.norm` already (#1369) |
| ~~`TapeNLPEvaluator._build_residual_exprs` → `compile_to_nl_expr` (array-valued)~~ | Gauss-Newton (`gauss_newton=True`, which `discopt.ml.train()` sets) with an **array-valued residual** | **Done** (#1873 follow-up): each residual is lowered with `compile_to_nl_array(r).reshape(-1)` and concatenated, the JAX arm's `residual_vector` order |
| ~~`_matmul` stacked operands, from `TrainableNetwork`~~ | trainable NN layers, e.g. `(20, 1, 6) @ (6, 1)` | **Done** (#1873 follow-up): `_stacked_matmul` follows `np.matmul` (broadcast batch axes, 1-D promotion) |
| `_require` (arity) | a function called with the wrong number of arguments | none (malformed) |

So the fallback a JAX-less user can actually hit, with a correctly formed model, comes
down to **`dm.custom`-family nodes** (permanently `[jax]`), a **`dm.external` nested
inside another one's argument**, and a body over the tape's node budget. Each refuses
with the tape's own reason and the extra. The spectral norm and `dm.external` left
this list in #1903: they are numpy-evaluated holes, checked against the JAX evaluator
to 3e-15 (values, gradients, Jacobians and Lagrangian Hessians, on the dense and the
sparse chain-rule paths; `test_spectral_norm.py`). Matrix 1-/inf-norms,
reductions deeper than `max_depth` and empty sums were lowered in the second
follow-up; tape and JAX agree on them to 1e-14 relative (values and gradients, 28
comparisons plus 4 on a 12,000-factor product).

**Stacked matmul and array-valued Gauss-Newton residuals were lowered in the #1873
follow-up**, so `discopt.ml` trainable surrogates (`train()`) and `discopt.dae.fit`
no longer need JAX. Because that moves those models from the JAX evaluator to the tape
on JAX installs too, it was checked differentially on the four `test_nn_trainable` /
`test_dae_fit` training models: 120 comparisons of objective, gradient, constraints,
Jacobian and Gauss-Newton Lagrangian Hessian at random points agree to 1.0e-15
relative, and `train()` on each backend (3 interleaved repetitions) gives identical
iteration counts and objectives on all four, while running about 10× faster
(tape 0.06–0.36 s vs JAX 0.77–2.05 s per fit).

## 2. Deliberate JAX autodiff users (`[jax]`-only, refusing with the named error)

| Feature | Entry point guarded |
|---|---|
| `dm.custom` | `modeling/core.py:custom`, refuses at construction |
| `dm.argmin` / `argmin_layer` | `modeling/argmin.py` (`_build_layer`, `argmin`) |
| `dm.implicit` / `implicit_full_space` | `modeling/implicit.py` (`_implicit_solver`, `implicit`) |
| `dm.external` | `modeling/external.py:_build_composite` |
| `discopt.estimate` (exact sensitivity Jacobian) | `estimate.py:_response_jacobian_wrt_parameters` |
| `discopt.sensitivity` chain rule (`d()`) | `sensitivity.py:_chain_rule` |
| `pounce_sensitivity` (sIPOPT) | `solvers/sipopt.py` |
| `discopt.parametric` / `_relax/parametric.py` | module guard on `_relax/parametric.py` |
| differentiable solves (`_relax/differentiable*.py`, `pounce_layer.py`) | module guards |
| `_relax/symbolic/{runtime,verification,signed_signomial}` (design-time toolkit / oracle) | module guards |
| `DISCOPT_BLOCK_VECTOR_EVAL` (default OFF) | `_block_eval.py:_make_jac_values_fn` |

`use_learned_relaxations`, the ICNN modules and the rest of `_relax/symbolic` were retired
with the McCormick-"nlp" route (§3).

## 3. Module-scope importers in `_relax/` — resolved

**Retraction.** This section first said that most of these modules had "no importer outside
other JAX modules and tests", and the PR description repeated it as "retirement
*candidates*". That framing was wrong. Most of them were reachable from `solver.py` through
`relaxation_compiler`, behind `mccormick_bounds="nlp"`/`partitions`/`relaxation_arithmetic`/
`use_learned_relaxations` (with `nlp_bb=False`, on convex models). The owner asked for a
definite answer, so it was measured. **`docs/dev/nlp-mccormick-route-retirement.md` is the
record and the per-module table.** In short:

- **Retired (22 JAX modules + `chebyshev_model`, plus `symbolic.domains/__init__`):** the
  McCormick-"nlp" route. Its bound changed nothing in 36 of 36 measured instance/variant
  pairs, and on two instances it turned a certified 2–6 s solve into a 27× time-limit
  overrun and a 13.6 GB OOM kill. Its four options are refused.
- **Kept (15), each with a stated reason:** the JAX evaluator (`nlp_evaluator`,
  `dag_compiler`, `sparse_jacobian`, `sparse_hessian`); the `dm.custom` global engine
  (`mcbox`, `mccormick_subgradient`, `multivariate_mccormick`); the differentiable layers
  (`differentiable*`, `pounce_layer`, `parametric`); and the symbolic toolkit
  (`runtime`, `verification`) with its oracle `signed_signomial`.
