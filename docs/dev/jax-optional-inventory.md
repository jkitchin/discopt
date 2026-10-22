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
- Every module with a module-scope `import jax` (37 of them) checks `jax_available()`
  before that import. `jax_available()` uses `importlib.util.find_spec`, so it never
  imports JAX.
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

MEASUREMENTS_PLACEHOLDER

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
| `rec`, empty `SumExpression` | `dm.sum` of a zero-size array | small: lower to the constant 0 |
| `rec`, `IndexExpression` not static | an index the compiler cannot resolve to a slot, including out-of-range (numpy raises; `jnp` clamps) | none. Refusing is correct, and with JAX the fallback clamps silently |
| `rec`, `CustomCall` | `dm.custom`, `dm.implicit`, `dm.external` (and `dm.argmin` layers) | **impossible by design**. The body is an opaque JAX callable, so these stay `[jax]`-only |
| `rec`, unknown node type | a node class the compiler does not know (none today) | none |
| `_matmul` (scalar operand, stacked, shape mismatch) | `@` with a 0-D operand, >2-D operands, or mismatched shapes | moderate for stacked (batched) matmul; the others are malformed models |
| `_sum_along` (axis out of range, repeated axis) | `dm.sum(x, axis=…)` with a bad axis | none (malformed) |
| `_broadcast_size` | operands that do not broadcast | none (malformed) |
| `_require_fused_xlog` | `entropy`/`centropy` with a pounce build older than #489 | none. Upgrading pounce removes it |
| `_lower_function` (`function {name!r}`) | a `FunctionCall` name with no lowering. Every name `dm.*` emits is lowered today (`exp log log10 log1p log2 sqrt sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh erf abs sign min max atan2 sigmoid softplus entropy centropy signpower prod norm*`), so only a hand-built node reaches it | none |
| `_fold` (empty; deeper than `NlExpr.max_depth`) | `dm.prod`/`dm.min`/`dm.max`/`norm` of an empty array, or of more elements than the tape's nesting depth | moderate: an n-ary tape opcode, or a balanced-tree fold, removes the depth limit |
| `_lower_norm` (order; **matrix norm**; scalar) | `dm.norm(X, ord)` with 2-D `X` (the CLAUDE.md case: 219 `jax` modules), or a 0-D argument | large for `ord=2` (spectral norm needs an SVD opcode with derivatives). `ord=1`/`inf` of a matrix are max-of-column/row-sums, which are cheap folds. `"fro"` is refused at `dm.norm` already (#1369) |
| `_require` (arity) | a function called with the wrong number of arguments | none (malformed) |

So the fallback a JAX-less user can actually hit, with a correctly formed model, comes
down to four cases: **`dm.custom`-family nodes** (permanently `[jax]`), **matrix
norms**, **stacked matmul**, and **reductions deeper than `max_depth`**. The last
three are opcode work in pounce, tracked by this table rather than by a separate
issue.

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
| learned / ICNN relaxations, `_relax/symbolic/*` | module guards (`icnn*`, `learned_relaxations` guard *before* `equinox`) |
| `DISCOPT_BLOCK_VECTOR_EVAL` (default OFF) | `_block_eval.py:_make_jac_values_fn` |

`use_learned_relaxations=True` still catches `ImportError` and warns that it is
falling back to standard McCormick. It did so before #1873 when equinox was missing.
The warning is loud, the fallback is sound (standard McCormick is a valid
relaxation), and the `gnn`/`learned` extras now include JAX.

## 3. Module-scope importers in `_relax/`

Which non-JAX modules import each one, statically:

| Module | Imported from (outside the JAX set and tests) |
|---|---|
| `nlp_evaluator` | `_tape_nlp_evaluator` (the fallback), `solver`, `solvers.{nlp_pounce,nlp_ipopt,oa,gdpopt_loa}`, `_relax.{primal_heuristics,differentiable_solve}`, `_block_eval`. All lazy and reached only through the fallback or `DISCOPT_NLP_EVAL=jax` |
| `dag_compiler` | `_relax.__init__` (lazy `__getattr__`), `uniform_relax` (`DISCOPT_SEPGRAD=jax` or no POUNCE only), `problem_classifier` (`_extract_lp_data_autodiff` / `_qp_terms_jax`, the legacy fallbacks behind the tape), `nonsmooth_lift`, `simplex_lowering`, `mo.utils`, `modeling.argmin`, `solvers.nlp_native`, `cli` |
| `relaxation_compiler`, `batch_evaluator`, `mccormick_nlp` | `solver`, only when `_mc_mode == "nlp"` survives to the node loop. Under the default `mccormick_bounds="auto"` it does not: a convex model starts at `"none"`, and the #120 soundness guard demotes every lp→nlp fallback on a nonconvex model to alphaBB. So this needs an explicit `mccormick_bounds="nlp"` on a convex model |
| `mccormick_subgradient` (+ `mcbox`) | `solver`, the reduced-space engine for `dm.custom` models (already `[jax]`-only) |
| `learned_relaxations` (+ `icnn*`) | `solver`, `use_learned_relaxations=True` |
| `differentiable` | `modeling.core`, `parametric`, `sensitivity` (deliberate users, §2) |
| `parametric` | `discopt.parametric` (§2) |
| `sparse_hessian` | `_block_eval` (flag-gated, default OFF) |
| `symbolic.runtime`, `symbolic.verification` | `symbolic.codegen`, `symbolic.__init__` (design-time toolkit) |
| `alphabb`, `differentiable_lp/qp`, `ellipsoidal_arith`, `envelopes`, `mccormick`, `mccormick_evaluator`, `multivariate_mccormick`, `oa_relax`, `piecewise_mccormick`, `polyhedral_oa`, `pounce_layer`, `sparse_jacobian`, `taylor_model`, `symbolic.{certified_learned,gp_hull,patterns,registry,signed_signomial,domains.*}` | **no importer outside other JAX modules and tests** |

**Retirement (CLAUDE.md §5) is not done here, deliberately.** This issue asked whether
each module is reachable. The answer for the last row is "only from other JAX modules
and tests", which makes them retirement *candidates*, not modules shown dead. Several
are reached through `relaxation_compiler` (the `mccormick_bounds="nlp"` path) or
through the differentiable layers, and both of those are reachable. Deleting a module
together with its tests is a separate decision per module, with its own measurement.
Optional JAX does not depend on it: a JAX-less install never imports any of these on
the default path (measured above), and an opt-in path that does refuses with the
named error.
