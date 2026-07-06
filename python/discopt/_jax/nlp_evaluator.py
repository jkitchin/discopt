"""
NLP Evaluator: JIT-compiled objective, gradient, Hessian, constraint, and Jacobian.

Wraps the DAG compiler output to provide evaluation callbacks suitable for
NLP solvers (cyipopt in Phase 1, Rust Ipopt later).

All evaluate_* methods accept and return numpy arrays for compatibility
with C-based solvers.

Parameters are threaded through the JITed callables as a runtime pytree
argument, so repeated solves that only mutate ``Parameter.value`` hit the
XLA cache instead of forcing a recompile. Combined with evaluator caching
in ``solver._make_evaluator``, this enables amortized per-solve cost for
NMPC-style closed-loop workloads.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from discopt._jax.dag_compiler import (
    _build_param_index,
    compile_constraint_params,
    compile_expression_params,
    compile_objective_params,
)
from discopt.modeling.core import Constraint, Model, ObjectiveSense

# Above this many dense Jacobian entries (m * n), ``evaluate_jacobian`` routes
# through the sparse coloring path instead of compiling the dense
# ``jax.jacfwd``. The dense compile replicates the whole constraint program once
# per input variable, so for a large model its XLA jaxpr explodes during MLIR
# lowering and aborts the *process* with a native SIGBUS/SIGILL (rsyn0810m03hfsg:
# 1935 x 1185 ~ 2.3M entries, yet only ~0.2% nonzero). The cap sits well above
# any small/medium model — which keep the faster dense path with no behavior
# change — and below the crash regime, where the dense compile is not a viable
# option anyway, so sparse is strictly safer.
_DENSE_JACOBIAN_COMPILE_LIMIT = 1_000_000


# --- First-time Lagrangian-Hessian XLA compile-cost model (F4) ------------------
# The first evaluate_hessian_values call forces an uninterruptible first-time XLA
# compile. The root-heuristic budget gate (solver.py) needs an a-priori estimate
# of that cost, from cheap model-size features available before the compile runs.
#
# Measured (docs/dev/perf-followup-plan-2026-07-05.md F4 entry experiment; M-series
# arm64, JAX 0.10.2, pounce 0.7.0). First-compile wall vs model size, per path:
#
#   path    instance         n_vars  hess_nnz   compile_s
#   dense   tls2                 37        16       0.15
#   dense   fac2                 66       972       0.15
#   sparse  heatexch_gen1       112       164       ~1.0
#   sparse  hda                 722      1094       2.5–8.3   (noisy across runs)
#   sparse  casctanks           500       820       3.7–5.0
#   sparse  heatexch_gen3       580      1020      46–49
#   sparse  contvar             296      1168     186
#
# FALSIFICATION (§0.6): the plan hypothesized a clean compile ~ f(n_vars, nnz)
# curve. It does not exist. Regressing log(compile) on n_vars gives R^2 = 0.002
# (essentially zero — contvar at n=296 compiles ~74x slower than hda at n=722),
# and the same instance's compile varies 2.5s->8.3s run to run. The cost is
# governed by the *shape/depth* of the lifted DAG (contvar's deep nested
# log/exp/division chains), not by any cheap size scalar, and is not reliably
# predictable in advance.
#
# Consequently the estimate is deliberately CONSERVATIVE, not a point predictor:
#   * DENSE path: bounded and always cheap in the measured range -> small constant.
#   * SPARSE (compressed-HVP) path with the kernel not yet compiled: the compile
#     is potentially very large (up to ~3x a 60s budget) and unpredictable, so the
#     estimate returns a large floor. The gate then only enters when there is
#     ample budget headroom, and skips (soundly — it is a primal heuristic) when
#     there is not. The conservative floor is what preserves the time_limit
#     contract; a precise number is neither available nor needed, because being
#     wrong high only skips a heuristic (never affects the dual bound).
_HESSIAN_COMPILE_DENSE_S = 0.5
# Floor for the risky first sparse compile. The compile can be arbitrarily large
# (measured 1s->186s cold, R^2~0 vs any cheap size feature) and cannot be polled
# once entered, so this is a *risk headroom*, not a point estimate: the gate only
# starts a first sparse compile when at least this much budget remains. Chosen so
# the in-solve first heuristic still runs on a normal (tens-of-seconds) budget —
# the measured in-solve first compile is a few seconds — while a tight budget
# (e.g. time_limit=5) refuses to gamble the whole contract on an unbounded
# compile. A single policy constant on the whole no-relaxation class, not tuned
# per instance. Over-estimating only skips a primal heuristic (sound).
_HESSIAN_COMPILE_SPARSE_FLOOR_S = 15.0


def estimate_hessian_compile_s(n_vars: int, hessian_nnz: int, use_sparse: bool) -> float:
    """Conservative estimate of the first-time Hessian XLA compile wall (seconds).

    See the module-level comment for the measured basis and why this is a
    conservative floor rather than a point predictor (the compile is not reliably
    predictable from cheap size features; R^2 ~ 0 vs n_vars). Used only to gate
    entry into *primal-heuristic* NLPs, so an over-estimate merely skips a
    heuristic (always sound) and never touches the dual bound.
    """
    if not use_sparse:
        return _HESSIAN_COMPILE_DENSE_S
    # Sparse compressed-HVP path: unpredictable and potentially budget-dwarfing.
    return _HESSIAN_COMPILE_SPARSE_FLOOR_S


def evaluator_fingerprint(model: Model) -> tuple:
    """Structural fingerprint of a model for evaluator-cache validity.

    Captures the object identity of the objective, constraints, variables, and
    parameters, plus the Gauss-Newton flag — but NOT mutable variable bounds or
    ``Parameter.value`` (the evaluator reads those live on each call). Two models
    with the same fingerprint can therefore share one compiled ``NLPEvaluator``
    across bound changes (every B&B node) and parameter re-binds.
    """
    return (
        id(model._objective),
        tuple(id(c) for c in model._constraints),
        tuple(id(v) for v in model._variables),
        tuple(id(p) for p in model._parameters),
        bool(getattr(model, "_gauss_newton_hessian", False)),
    )


def cached_evaluator(model: Model) -> "NLPEvaluator":
    """Return a per-model cached ``NLPEvaluator``, reusing its compiled JAX
    callables across repeated constructions as long as the model's *structure* is
    unchanged.

    Constructing an ``NLPEvaluator`` re-traces and re-compiles the model's
    constraint / objective / Jacobian functions — a real per-call Python cost.
    The B&B loop, primal heuristics, and POUNCE node solves all evaluate the *same*
    model (only bounds and parameter values change, which the evaluator reads
    live), so they can share one evaluator. This centralizes the cache that was
    previously only used by ``solver._make_evaluator``; call sites that built a
    fresh ``NLPEvaluator(model)`` per call (e.g. the diving heuristic, ~110×/solve
    on gear4) re-paid that construction cost on every call. Keyed on
    :func:`evaluator_fingerprint`, so a structurally different model rebuilds.
    """
    fp = evaluator_fingerprint(model)
    cached = getattr(model, "_nlp_evaluator_cache", None)
    if cached is not None:
        ev, cached_fp = cached
        if cached_fp == fp:
            return ev
    ev = NLPEvaluator(model, gauss_newton=bool(getattr(model, "_gauss_newton_hessian", False)))
    model._nlp_evaluator_cache = (ev, fp)  # type: ignore[attr-defined]
    return ev


def validate_sparse_values(evaluator, x: np.ndarray, atol: float = 1e-8) -> bool:
    """Check that sparse COO values agree with dense evaluation.

    Computes both dense and sparse Jacobian/Hessian at x and verifies that
    the sparse values match the dense values at the declared positions.

    Returns True if validation passes, False otherwise.
    """
    import logging

    logger = logging.getLogger("discopt.sparse")
    ok = True

    # Validate Jacobian
    if evaluator.n_constraints > 0:
        jac_dense = evaluator.evaluate_jacobian(x)
        rows, cols = evaluator.jacobian_structure()
        sparse_vals = evaluator.evaluate_jacobian_values(x)
        dense_at_pos = jac_dense[rows, cols]
        if not np.allclose(sparse_vals, dense_at_pos, atol=atol):
            max_err = float(np.max(np.abs(sparse_vals - dense_at_pos)))
            logger.warning("Sparse Jacobian validation failed (max err=%.2e)", max_err)
            ok = False

    # Validate Hessian
    m = evaluator.n_constraints
    lam = np.ones(m, dtype=np.float64)
    hess_dense = evaluator.evaluate_lagrangian_hessian(x, 1.0, lam)
    rows, cols = evaluator.hessian_structure()
    sparse_vals = evaluator.evaluate_hessian_values(x, 1.0, lam)
    dense_at_pos = hess_dense[rows, cols]
    if not np.allclose(sparse_vals, dense_at_pos, atol=atol):
        max_err = float(np.max(np.abs(sparse_vals - dense_at_pos)))
        logger.warning("Sparse Hessian validation failed (max err=%.2e)", max_err)
        ok = False

    return ok


class NLPEvaluator:
    """
    JAX-based NLP evaluation layer providing JIT-compiled callbacks.

    Wraps the DAG compiler output to provide objective, gradient, Hessian,
    constraint, and Jacobian evaluations suitable for NLP solvers.

    All methods return numpy arrays (not JAX arrays) for compatibility
    with cyipopt and other C-based solvers.

    Usage:
        evaluator = NLPEvaluator(model)
        obj = evaluator.evaluate_objective(x)
        grad = evaluator.evaluate_gradient(x)
        hess = evaluator.evaluate_hessian(x)
        cons = evaluator.evaluate_constraints(x)
        jac = evaluator.evaluate_jacobian(x)
    """

    def __init__(self, model: Model, gauss_newton: bool = False) -> None:
        """
        Compile model expressions into JIT-compiled evaluation functions.

        Args:
            model: A Model with objective and constraints set.
            gauss_newton: When True and the objective is a non-negative-weighted
                sum of squares, use the Gauss-Newton objective Hessian
                (``2 Jᵀ J`` of the residuals) instead of the dense
                ``jax.hessian``. This sidesteps the super-linear second-
                derivative compile for least-squares objectives (issue #98) at
                the cost of dropping the ``Σ rᵢ ∇²rᵢ`` curvature term. Silently
                falls back to the exact dense Hessian when the objective is not
                a recognized sum of squares (or the model maximizes).
        """
        if model._objective is None:
            raise ValueError("Model has no objective set.")

        self._model = model
        self._negate = model._objective.sense == ObjectiveSense.MAXIMIZE

        # Variable count (flat).
        self._n_variables = sum(v.size for v in model._variables)

        # Whether the Lagrangian-Hessian XLA kernel has been traced+compiled yet.
        # The first ``evaluate_hessian_values`` call triggers an *uninterruptible*
        # first-time XLA compile (seconds-to-minutes on large flowsheet DAGs);
        # once done, subsequent calls hit the warm XLA cache in microseconds.
        # The root-heuristic budget gate (solver.py) reads this flag to decide
        # whether entering a compile-triggering NLP fits the remaining wall
        # budget — see :meth:`hessian_kernel_compiled`.
        self._hessian_compiled = False

        # Parameter plumbing: capture identity order so snapshots stay aligned
        # even if model._parameters is later extended (caching in solver.py
        # will invalidate on structural changes anyway).
        self._parameters = list(model._parameters)
        self._param_index = _build_param_index(model)

        # Compile objective as fn(x, params). The ``*_jit`` attributes below
        # are the source-of-truth jitted callables; they are JIT-compiled once
        # per evaluator lifetime and key their XLA cache on shapes only, so
        # mutating ``Parameter.value`` between calls hits the cache instead
        # of recompiling.
        raw_obj_fn = compile_objective_params(model, self._param_index)
        if self._negate:

            def obj_fn(x_flat, params):
                return -raw_obj_fn(x_flat, params)
        else:
            obj_fn = raw_obj_fn
        self._obj_fn_jit = jax.jit(obj_fn)
        self._grad_fn_jit = jax.jit(jax.grad(obj_fn, argnums=0))
        # The objective Hessian (``self._hess_fn_jit``) is built below, after the
        # constraints, so the Gauss-Newton path can share the constraint plumbing.

        # Compile constraints. Constraint bodies may be scalar OR array-valued
        # (the latter is what DAEBuilder emits for collocation). We detect
        # each body's output shape once via ``jax.eval_shape`` (no tracing of
        # the real values), record the flat size, and concatenate the ravel'd
        # bodies at call time. This lets DAEBuilder emit one big vector body
        # instead of thousands of scalar closures — the XLA trace shrinks from
        # O(nfe*ncp) scalar ops to O(1) bulk ops.
        self._source_constraints: list[Constraint] = [
            c for c in model._constraints if isinstance(c, Constraint)
        ]
        constraint_fns = [
            compile_constraint_params(c, model, self._param_index) for c in self._source_constraints
        ]
        self._constraint_fns = constraint_fns

        if len(constraint_fns) > 0:
            x_spec = jax.ShapeDtypeStruct((self._n_variables,), jnp.float64)
            param_specs = tuple(
                jax.ShapeDtypeStruct(np.shape(p.value), jnp.float64) for p in self._parameters
            )
            flat_sizes: list[int] = []
            for fn in constraint_fns:
                out = jax.eval_shape(fn, x_spec, param_specs)
                flat_sizes.append(int(np.prod(out.shape)) if out.shape else 1)
            self._constraint_flat_sizes = np.asarray(flat_sizes, dtype=np.intp)
            self._n_constraints = int(self._constraint_flat_sizes.sum())

            def concat_constraints(x_flat, params):
                parts = [jnp.reshape(fn(x_flat, params), (-1,)) for fn in constraint_fns]
                if len(parts) == 1:
                    return parts[0]
                return jnp.concatenate(parts)

            self._cons_fn_jit = jax.jit(concat_constraints)
            # Dense constraint Jacobian via FORWARD-mode AD. ``jax.jacobian``
            # defaults to reverse-mode (jacrev); its XLA codegen for the
            # transpose of certain nonlinear graphs (e.g. fractional powers
            # x**0.75 in MINLPLib's chakra) blows up super-linearly and a single
            # compile can hang for >90s — long enough to starve the solver's
            # own time-limit check, since the compile is an uninterruptible C
            # call. Forward-mode (jacfwd) produces the identical matrix (AD mode
            # never changes the value) and compiles the same case in ~0.2s. The
            # heavy NLP-iteration Jacobian uses the sparse coloring path
            # (evaluate_sparse_jacobian); this dense form backs FBBT linearity
            # detection and the sparse fallback, where forward-mode's robust
            # compile matters far more than its O(n)-vs-O(m) runtime constant.
            self._jac_fn_jit = jax.jit(jax.jacfwd(concat_constraints, argnums=0))
        else:
            self._constraint_flat_sizes = np.zeros(0, dtype=np.intp)
            self._n_constraints = 0
            self._cons_fn_jit = None
            self._jac_fn_jit = None

        # Gauss-Newton objective Hessian (opt-in, issue #98). When requested and
        # the objective is a non-negative-weighted sum of squares, build a
        # residual function r(x, params) and use H_obj ≈ 2 Jᵀ J (J = ∂r/∂x).
        # This avoids compiling the dense second-derivative graph, whose XLA
        # codegen is super-linear in the objective expression size.
        cons_fn_jit = self._cons_fn_jit
        gn_obj_hess_fn = self._build_gauss_newton_obj_hessian(model) if gauss_newton else None
        self._gauss_newton = gn_obj_hess_fn is not None

        # Objective Hessian ∇²f(x). Forward-over-forward for the same compile-
        # robustness reason as the Lagrangian Hessian below: the default
        # ``jax.hessian`` reverse inner pass can blow up XLA codegen on large
        # nonlinear objective graphs. Identical values, faster/robuster compile.
        if gn_obj_hess_fn is not None:
            self._hess_fn_jit = jax.jit(gn_obj_hess_fn)
        else:
            self._hess_fn_jit = jax.jit(jax.jacfwd(jax.jacfwd(obj_fn, argnums=0), argnums=0))

        # Lagrangian Hessian: obj_factor * ∇²f(x) + Σᵢ λᵢ ∇²gᵢ(x).
        if gn_obj_hess_fn is not None:
            # Gauss-Newton objective curvature + the exact constraint curvature.
            # The constraint term keeps its true Hessian (cheap when linear,
            # zero when there are no constraints); only the objective second-
            # derivative graph is sidestepped.
            n_vars = self._n_variables

            def constraint_hessian(x, lam, params):
                if cons_fn_jit is None:
                    return jnp.zeros((n_vars, n_vars), dtype=jnp.float64)
                return jax.hessian(lambda xx: jnp.dot(lam, cons_fn_jit(xx, params)))(x)

            def lagrangian_hess(x, obj_factor, lam, params):
                return obj_factor * gn_obj_hess_fn(x, params) + constraint_hessian(x, lam, params)

            self._lagrangian_hess_fn_jit = jax.jit(lagrangian_hess)
            # The compressed-HVP Hessian path assumes the exact Lagrangian; it
            # is not wired for the Gauss-Newton approximation, so disable it.
            self._lagrangian_hvp_fn_jit = None
        else:

            def lagrangian(x, obj_factor, lam, params):
                L = obj_factor * obj_fn(x, params)
                if cons_fn_jit is not None:
                    L = L + jnp.dot(lam, cons_fn_jit(x, params))
                return L

            # Dense Lagrangian Hessian via FORWARD-over-FORWARD AD. ``jax.hessian``
            # is ``jacfwd(jacrev(...))``; its inner reverse pass hits the same
            # super-linear XLA transpose codegen documented for the constraint
            # Jacobian above — on a large lifted DAG (e.g. MINLPLib's du-opt, 21
            # vars but a heavy constraint graph) that single compile runs ~12s,
            # and because it is an uninterruptible C call it blows straight past
            # the solver's per-node time budget. ``jacfwd(jacfwd(...))`` produces
            # the identical matrix (AD mode never changes the value) and compiles
            # it in ~half the time. This dense form is only taken when the sparse
            # colored-HVP path declines (small ``n``, or a dense Hessian), so the
            # O(n^2)-vs-O(n) forward-mode runtime constant is immaterial while the
            # compile robustness is what matters.
            self._lagrangian_hess_fn_jit = jax.jit(
                jax.jacfwd(jax.jacfwd(lagrangian, argnums=0), argnums=0)
            )

            # Matrix-free Lagrangian Hessian-vector product (forward-over-reverse).
            # Used by the compressed sparse-Hessian path to recover the block-
            # banded Hessian of large DAE/collocation NLPs without ever
            # materializing the dense ``n x n`` matrix (issue #95).
            _lagrangian_grad = jax.grad(lagrangian, argnums=0)

            def lagrangian_hvp(x, obj_factor, lam, params, v):
                _, hv = jax.jvp(
                    lambda xx: _lagrangian_grad(xx, obj_factor, lam, params), (x,), (v,)
                )
                return hv

            self._lagrangian_hvp_fn_jit = jax.jit(lagrangian_hvp)

        # Legacy single-argument wrappers. These close over ``self`` and read
        # the current parameter values at call time, so downstream callers
        # that use the raw ``_obj_fn`` / ``_cons_fn`` attributes (IPM batch,
        # OA, alpha estimation, etc.) see the latest parameter values on
        # every call without any recompile overhead.
        self._obj_fn = self._bind_x_only(self._obj_fn_jit)
        self._grad_fn = self._bind_x_only(self._grad_fn_jit)
        self._hess_fn = self._bind_x_only(self._hess_fn_jit)
        if self._cons_fn_jit is not None:
            self._cons_fn = self._bind_x_only(self._cons_fn_jit)
            self._jac_fn = self._bind_x_only(self._jac_fn_jit)
        else:
            self._cons_fn = None
            self._jac_fn = None
        self._lagrangian_hess_fn = self._bind_x_obj_lam(self._lagrangian_hess_fn_jit)

    def _build_gauss_newton_obj_hessian(self, model: Model):
        """Build ``fn(x, params) -> 2 Jᵀ J`` if the objective is a sum of squares.

        Returns ``None`` (so the caller uses the exact dense ``jax.hessian``)
        when Gauss-Newton does not apply: the model maximizes, or the objective
        is not a recognized non-negative-weighted sum of squares.
        """
        import logging

        logger = logging.getLogger("discopt.nlp")

        if self._negate:
            logger.info(
                "gauss_newton ignored: objective is maximized (not a minimized "
                "sum of squares); using exact dense Hessian."
            )
            return None

        from discopt._jax.least_squares import extract_residuals

        assert model._objective is not None  # guaranteed by __init__
        residual_exprs = extract_residuals(model._objective.expression)
        if not residual_exprs:
            logger.info(
                "gauss_newton requested but the objective is not a recognized "
                "sum of squares; using exact dense Hessian."
            )
            return None

        residual_fns = [
            compile_expression_params(e, model, self._param_index) for e in residual_exprs
        ]

        def residual_vector(x, params):
            parts = [jnp.reshape(fn(x, params), (-1,)) for fn in residual_fns]
            if len(parts) == 1:
                return parts[0]
            return jnp.concatenate(parts)

        def gn_obj_hessian(x, params):
            # J = ∂r/∂x has shape (R, n); the Gauss-Newton Hessian of Σ rᵢ² is
            # 2 Jᵀ J. Only first derivatives of the residuals are compiled.
            jac = jax.jacfwd(residual_vector, argnums=0)(x, params)
            return 2.0 * (jac.T @ jac)

        return gn_obj_hessian

    @property
    def is_gauss_newton(self) -> bool:
        """True when the objective Hessian uses the Gauss-Newton approximation."""
        return getattr(self, "_gauss_newton", False)

    def _bind_x_only(self, fn_xp):
        """Wrap an ``fn(x, params)`` jit into an ``fn(x)`` callable that reads
        current parameter values on each invocation."""
        current_params = self._current_params

        def wrapped(x):
            return fn_xp(x, current_params())

        return wrapped

    def _bind_x_obj_lam(self, fn_xp):
        """Wrap a Lagrangian-Hessian jit into an ``fn(x, obj_factor, lam)``
        callable that reads current parameter values on each invocation."""
        current_params = self._current_params

        def wrapped(x, obj_factor, lam):
            return fn_xp(x, obj_factor, lam, current_params())

        return wrapped

    def _current_params(self) -> tuple:
        """Snapshot current ``Parameter.value`` into a tuple of jax arrays.

        Called on every evaluate_* entry. Shapes are stable across solves, so
        the JIT cache is hit after the first call.
        """
        return tuple(jnp.asarray(p.value, dtype=jnp.float64) for p in self._parameters)

    def evaluate_lagrangian_hessian(
        self, x: np.ndarray, obj_factor: float, lambda_: np.ndarray
    ) -> np.ndarray:
        """Evaluate Hessian of the Lagrangian at x.

        H = obj_factor * ∇²f(x) + Σᵢ λᵢ ∇²gᵢ(x)
        """
        return np.asarray(
            self._lagrangian_hess_fn_jit(x, obj_factor, lambda_, self._current_params())
        )

    def evaluate_objective(self, x: np.ndarray) -> float:
        """Evaluate objective at x. Returns scalar."""
        return float(self._obj_fn_jit(x, self._current_params()))

    def evaluate_gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluate gradient of objective at x. Returns (n,) array."""
        return np.asarray(self._grad_fn_jit(x, self._current_params()))

    def evaluate_hessian(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Hessian of objective at x. Returns (n, n) array."""
        return np.asarray(self._hess_fn_jit(x, self._current_params()))

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all constraint bodies at x. Returns (m,) array."""
        if self._cons_fn_jit is None:
            return np.array([], dtype=np.float64)
        return np.asarray(self._cons_fn_jit(x, self._current_params()))

    def _evaluate_dense_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Raw dense Jacobian via the compiled ``jax.jacfwd``. Callers must size-
        gate this: on a large model the dense compile aborts the process (see
        ``_DENSE_JACOBIAN_COMPILE_LIMIT``). Use :meth:`evaluate_jacobian` for the
        guarded, size-aware entry point."""
        if self._jac_fn_jit is None:
            return np.empty((0, self._n_variables), dtype=np.float64)
        return np.asarray(self._jac_fn_jit(x, self._current_params()))

    def _ensure_sparse_jac_fn(self) -> bool:
        """Lazily build the sparse (coloring-based) Jacobian function. Returns
        True when a sparse evaluator is available for this model. Idempotent."""
        if self._cons_fn_jit is None:
            return False
        if not hasattr(self, "_sparse_jac_fn"):
            self._sparse_jac_fn = None
            try:
                from discopt._jax.sparse_jacobian import make_sparse_jac_fn
                from discopt._jax.sparsity import (
                    compute_coloring,
                    make_seed_matrix,
                    should_use_sparse,
                )

                pattern = self.sparsity_pattern
                if pattern is not None and should_use_sparse(pattern):
                    colors, n_colors = compute_coloring(pattern)
                    seed = make_seed_matrix(colors, n_colors, pattern.n_vars)
                    # make_sparse_jac_fn expects fn(x); the legacy single-arg
                    # wrapper reads current parameter values on each call.
                    self._sparse_jac_fn = make_sparse_jac_fn(self._cons_fn, pattern, colors, seed)
            except Exception:
                pass
        return self._sparse_jac_fn is not None

    def evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Jacobian of constraints at x. Returns a dense (m, n) array.

        For a large model the dense ``jax.jacfwd`` compile explodes XLA — it
        replicates the constraint program once per input variable — and aborts
        the *process* with a native crash that no ``try/except`` can catch. Above
        ``_DENSE_JACOBIAN_COMPILE_LIMIT`` (m * n) this routes through the sparse
        coloring path (O(chromatic-number) JVPs) and densifies, so any caller
        needing a full-model Jacobian at scale stays safe while still receiving
        the same dense (m, n) array. Small/medium models keep the faster dense
        path unchanged. Only triggers when the sparse evaluator is actually
        available; otherwise the dense path is used as before.
        """
        if self._jac_fn_jit is None:
            return np.empty((0, self._n_variables), dtype=np.float64)
        if (
            self._n_constraints * self._n_variables > _DENSE_JACOBIAN_COMPILE_LIMIT
            and self._ensure_sparse_jac_fn()
        ):
            import scipy.sparse as sp

            J = self._sparse_jac_fn(x)
            if sp.issparse(J):
                return np.asarray(J.toarray(), dtype=np.float64)
            return np.asarray(J, dtype=np.float64)
        return self._evaluate_dense_jacobian(x)

    def evaluate_sparse_jacobian(self, x: np.ndarray):
        """Evaluate Jacobian as a sparse CSC matrix using compressed JVPs.

        Uses sparsity detection and graph coloring to evaluate the Jacobian
        in O(p) JVPs where p is the chromatic number (typically 5-20).
        Falls back to dense evaluation if sparsity infrastructure is unavailable.

        Returns:
            scipy.sparse.csc_matrix of shape (m, n), or dense (m, n) ndarray
            if sparse evaluation is not applicable.
        """
        if self._cons_fn_jit is None:
            import scipy.sparse as sp

            return sp.csc_matrix((0, self._n_variables), dtype=np.float64)

        if self._ensure_sparse_jac_fn():
            return self._sparse_jac_fn(x)

        # Fallback to the RAW dense path (not evaluate_jacobian, which would
        # re-enter this method for a large model and recurse).
        return self._evaluate_dense_jacobian(x)

    @property
    def sparsity_pattern(self):
        """Lazily compute and return the sparsity pattern for this model.

        When every source constraint has a scalar body we use the fast DAG
        walker (``detect_sparsity_dag``). When any source constraint has a
        vector-valued body we fall back to a trace-based pattern that
        evaluates the compiled constraint Jacobian at a single interior
        point and takes its nonzero mask. This gives true per-flat-row
        structure (not the conservative per-Constraint union) and keeps
        the Jacobian nonzero count tight for large vectorized NMPC models.
        """
        if not hasattr(self, "_sparse_pattern") or self._sparse_pattern is None:
            try:
                has_vector_body = self._constraint_flat_sizes.size > 0 and bool(
                    np.any(self._constraint_flat_sizes > 1)
                )
                if has_vector_body:
                    self._sparse_pattern = self._detect_sparsity_traced()
                else:
                    from discopt._jax.sparsity import detect_sparsity_dag

                    self._sparse_pattern = detect_sparsity_dag(self._model)
            except Exception:
                self._sparse_pattern = None
        return self._sparse_pattern

    def _detect_sparsity_traced(self):
        """Detect Jacobian sparsity by tracing the compiled vector cons fn.

        Evaluates ``jax.jacfwd(cons)`` at several randomized interior points
        and unions the nonzero masks. A single probe point is unreliable:
        at any specific ``x``, the partial derivative of a product
        ``a * b`` w.r.t. ``b`` is ``a``, which vanishes whenever ``a``
        happens to equal zero at the probe — even though ``b`` is a real
        structural dependency. Using multiple diverse points makes it
        extremely unlikely that all probes hit the zero set of the same
        symbolic coefficient. This is essential for correctness: if the
        detected pattern misses an entry, the downstream graph coloring
        can fuse two columns that actually share a row, and compressed
        forward-mode recovery will then produce wrong values at positions
        that *are* in the declared pattern.

        The Hessian pattern is still obtained from the DAG walker, which
        finds nonlinear variable pairs from expression structure alone and
        therefore remains correct for vector-body constraints.
        """
        import scipy.sparse as sp

        from discopt._jax.sparsity import SparsityPattern, detect_sparsity_dag

        n = self._n_variables
        m = self._n_constraints
        if n == 0 or m == 0 or self._cons_fn_jit is None:
            return None

        dag_pattern = detect_sparsity_dag(self._model)
        if dag_pattern is None:
            return None

        lb, ub = self.variable_bounds
        lb_f = np.where(np.isfinite(lb), lb, -1.0).astype(np.float64)
        ub_f = np.where(np.isfinite(ub), ub, 1.0).astype(np.float64)
        # Keep probes inside the model's declared box. If a bound is tight
        # (equality-fixed init), we still use it and rely on other probes
        # to expose the pattern.
        span = np.maximum(ub_f - lb_f, 1e-6)

        cons_fn_jit = self._cons_fn_jit
        params = self._current_params()

        @jax.jit
        def jac_of_cons(xx):
            return jax.jacfwd(lambda x: cons_fn_jit(x, params))(xx)

        # Deterministic probe sequence. Fixed seed → reproducible sparsity
        # across runs of the same model. 5 probes empirically covers the
        # pathological cases we care about (DAE collocation with coefficients
        # that pass through zero at symmetric interior points).
        rng = np.random.default_rng(0xD15C0)
        mask = np.zeros((m, n), dtype=bool)
        for _ in range(5):
            u = rng.uniform(0.1, 0.9, size=n)
            x_probe = lb_f + u * span
            x_probe = np.clip(x_probe, lb_f, ub_f).astype(np.float64)
            jac_dense = np.asarray(jac_of_cons(jnp.asarray(x_probe)))
            mask |= jac_dense != 0.0

        rows, cols = np.nonzero(mask)
        nnz = len(rows)
        if nnz == 0:
            return None

        jac_csr = sp.csr_matrix(
            (np.ones(nnz, dtype=bool), (rows, cols)),
            shape=(m, n),
        )
        hess_csr = dag_pattern.hessian_sparsity
        hess_nnz = int(hess_csr.nnz) if hess_csr is not None else 0

        return SparsityPattern(
            jacobian_sparsity=jac_csr,
            hessian_sparsity=hess_csr,
            n_vars=n,
            n_cons=m,
            jacobian_nnz=nnz,
            hessian_nnz=hess_nnz,
        )

    def has_sparse_structure(self) -> bool:
        """True when a sparsity pattern is available for structure reporting.

        The pattern-reporting decision is independent of the evaluation-method
        decision: whenever a nonempty sparsity pattern exists, we report the
        true nonzero indices to the NLP solver (smaller factorization graph),
        regardless of whether values are computed densely or via compressed
        forward-mode JVPs. See :meth:`_use_compressed_eval` for the latter.
        """
        if not hasattr(self, "_use_sparse"):
            self._use_sparse = False
            pattern = self.sparsity_pattern
            if pattern is not None and pattern.n_cons > 0 and pattern.n_vars > 0:
                self._use_sparse = True
        return self._use_sparse

    def _use_compressed_eval(self) -> bool:
        """True when compressed-JVP sparse evaluation is expected to pay off.

        Uses ``sparsity.should_use_sparse`` (density + problem-size threshold)
        to decide. When False but :meth:`has_sparse_structure` is True, we
        evaluate the Jacobian densely and project onto the pattern — cheaper
        per iteration than compressed forward-mode for moderately dense
        problems while still giving the solver the true sparse structure.
        """
        if not hasattr(self, "_use_compressed_cache"):
            self._use_compressed_cache = False
            pattern = self.sparsity_pattern
            if pattern is not None:
                try:
                    from discopt._jax.sparsity import should_use_sparse

                    self._use_compressed_cache = should_use_sparse(pattern)
                except Exception:
                    pass
        return self._use_compressed_cache

    def jacobian_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return Jacobian sparsity as COO (rows, cols) arrays.

        When sparse structure is available, returns only nonzero positions.
        Otherwise returns dense (all m*n positions).
        """
        self._ensure_coo_cache()
        return (self._jac_rows, self._jac_cols)

    def hessian_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return lower-triangle Hessian sparsity as COO (rows, cols) arrays.

        When sparse structure is available, returns only nonzero lower-triangle
        positions. Otherwise returns all lower-triangle positions.
        """
        self._ensure_coo_cache()
        return (self._hess_rows, self._hess_cols)

    def evaluate_jacobian_values(self, x: np.ndarray) -> np.ndarray:
        """Return Jacobian values as 1-D array matching jacobian_structure order."""
        self._ensure_coo_cache()
        if (
            self.has_sparse_structure()
            and self._use_compressed_eval()
            and self._cons_fn_jit is not None
        ):
            values_fn = self._ensure_sparse_jac_values_fn()
            if values_fn is not None:
                return values_fn(x, self._current_params())
        # Dense evaluation, projected onto the COO pattern (which may still
        # be the true sparse pattern — that's the common moderately-dense path)
        jac = self.evaluate_jacobian(x)
        return jac[self._jac_rows, self._jac_cols].astype(np.float64)

    def _ensure_sparse_jac_values_fn(self):
        """Lazily build a jit'd fn(x, params) -> COO values for the sparse path."""
        if hasattr(self, "_sparse_jac_values_fn"):
            return self._sparse_jac_values_fn
        self._sparse_jac_values_fn = None
        try:
            from discopt._jax.sparse_jacobian import make_sparse_jac_values_fn
            from discopt._jax.sparsity import (
                compute_coloring,
                make_seed_matrix,
                should_use_sparse,
            )

            pattern = self.sparsity_pattern
            if pattern is not None and should_use_sparse(pattern):
                colors, n_colors = compute_coloring(pattern)
                seed = make_seed_matrix(colors, n_colors, pattern.n_vars)
                self._sparse_jac_values_fn = make_sparse_jac_values_fn(
                    self._cons_fn_jit, pattern, colors, seed
                )
        except Exception:
            pass
        return self._sparse_jac_values_fn

    def evaluate_hessian_values(
        self, x: np.ndarray, obj_factor: float, lambda_: np.ndarray
    ) -> np.ndarray:
        """Return Lagrangian Hessian values as 1-D array matching hessian_structure order.

        When the Hessian is sparse enough, values are recovered via colored
        Hessian-vector products (see :mod:`discopt._jax.sparse_hessian`),
        avoiding the dense ``n x n`` materialization that makes large
        DAE/collocation solves intractable (issue #95). Otherwise the dense
        Lagrangian Hessian is evaluated and projected onto the COO pattern.
        """
        self._ensure_coo_cache()
        values_fn = self._ensure_sparse_hess_values_fn()
        if values_fn is not None:
            x64 = np.asarray(x, dtype=np.float64)
            lam64 = np.asarray(lambda_, dtype=np.float64)
            out = values_fn(x64, float(obj_factor), lam64, self._current_params())
            # The first call above forces the (possibly very slow) first-time XLA
            # compile of the sparse-Hessian kernel; mark it so the budget gate can
            # treat later calls as free.
            self._hessian_compiled = True
            return out
        h = self.evaluate_lagrangian_hessian(x, obj_factor, lambda_)
        self._hessian_compiled = True
        return h[self._hess_rows, self._hess_cols].astype(np.float64)

    def _use_sparse_hessian(self) -> bool:
        """True when colored-HVP Hessian recovery is expected to pay off.

        Gated on a sparse structure being available, the exact Lagrangian HVP
        being built (not the Gauss-Newton path), and the Hessian being both
        large and sparse enough that the dense ``n x n`` assembly dominates.
        """
        if not hasattr(self, "_use_sparse_hess_cache"):
            self._use_sparse_hess_cache = False
            if (
                self.has_sparse_structure()
                and getattr(self, "_lagrangian_hvp_fn_jit", None) is not None
            ):
                pattern = self.sparsity_pattern
                if (
                    pattern is not None
                    and pattern.n_vars >= 50
                    and pattern.hessian_nnz > 0
                    and pattern.hessian_density < 0.15
                ):
                    self._use_sparse_hess_cache = True
        return self._use_sparse_hess_cache

    def _ensure_sparse_hess_values_fn(self):
        """Lazily build a fn(x, obj_factor, lam, params) -> COO Hessian values."""
        if hasattr(self, "_sparse_hess_values_fn"):
            return self._sparse_hess_values_fn
        self._sparse_hess_values_fn = None
        if self._use_sparse_hessian():
            try:
                from discopt._jax.sparse_hessian import (
                    build_hessian_coloring,
                    make_sparse_hess_values_fn,
                )

                self._ensure_coo_cache()
                seed, coo_seed_idx, coo_lookup_row = build_hessian_coloring(
                    self.sparsity_pattern, self._hess_rows, self._hess_cols
                )
                self._sparse_hess_values_fn = make_sparse_hess_values_fn(
                    self._lagrangian_hvp_fn_jit,
                    seed,
                    coo_seed_idx,
                    coo_lookup_row,
                )
            except Exception:
                self._sparse_hess_values_fn = None
        return self._sparse_hess_values_fn

    def _ensure_coo_cache(self) -> None:
        """Lazily compute and cache COO index arrays for Jacobian and Hessian."""
        if hasattr(self, "_jac_rows"):
            return

        n = self._n_variables
        m = self._n_constraints

        if self.has_sparse_structure():
            pattern = self.sparsity_pattern
            # Jacobian: convert CSR boolean to COO indices
            jac_coo = pattern.jacobian_sparsity.tocoo()
            self._jac_rows = jac_coo.row.astype(np.intp)
            self._jac_cols = jac_coo.col.astype(np.intp)
            # Hessian: extract lower triangle from symmetric pattern
            hess_coo = pattern.hessian_sparsity.tocoo()
            lower_mask = hess_coo.row >= hess_coo.col
            self._hess_rows = hess_coo.row[lower_mask].astype(np.intp)
            self._hess_cols = hess_coo.col[lower_mask].astype(np.intp)
        else:
            # Dense fallback
            if m > 0:
                rows, cols = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
                self._jac_rows = rows.flatten().astype(np.intp)
                self._jac_cols = cols.flatten().astype(np.intp)
            else:
                self._jac_rows = np.array([], dtype=np.intp)
                self._jac_cols = np.array([], dtype=np.intp)
            self._hess_rows, self._hess_cols = np.tril_indices(n)

    def hessian_kernel_compiled(self) -> bool:
        """Whether the first-time Lagrangian-Hessian XLA compile has already run.

        The first :meth:`evaluate_hessian_values` call forces an uninterruptible
        first-time XLA compile that, on large flowsheet DAGs, can exceed a whole
        `time_limit` budget (bottleneck-profile-2026-07-05 §4). Once compiled,
        later Hessian evaluations hit the warm XLA cache. The root-heuristic
        budget gate uses this to decide whether entering a compile-triggering NLP
        can still respect the deadline.
        """
        return bool(self._hessian_compiled)

    def hessian_compile_estimate_s(self) -> float:
        """Estimated wall seconds of the *first* Lagrangian-Hessian XLA compile.

        Returns ``0.0`` when the kernel is already compiled (the cost is spent).
        Otherwise returns a model-size estimate from a measured curve fit on
        MINLPLib instances (see :data:`HESSIAN_COMPILE_FIT` and
        docs/dev/perf-followup-plan-2026-07-05.md F4). The estimate is a general
        function of ``n_variables`` and the Hessian nnz — never keyed to an
        instance name — used only to gate *primal-heuristic* entry, so an
        over- or under-estimate can never affect the dual bound or the returned
        optimum (skipping a heuristic is always sound).
        """
        if self._hessian_compiled:
            return 0.0
        pattern = self.sparsity_pattern
        hnnz = int(pattern.hessian_nnz) if pattern is not None else 0
        return estimate_hessian_compile_s(
            n_vars=self._n_variables,
            hessian_nnz=hnnz,
            use_sparse=self._use_sparse_hessian(),
        )

    @property
    def n_variables(self) -> int:
        """Total number of variables (flat)."""
        return self._n_variables

    @property
    def n_constraints(self) -> int:
        """Total number of constraints."""
        return self._n_constraints

    @property
    def variable_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (lb, ub) arrays of shape (n,) for all variables."""
        lbs = []
        ubs = []
        for v in self._model._variables:
            lbs.append(v.lb.flatten())
            ubs.append(v.ub.flatten())
        return np.concatenate(lbs), np.concatenate(ubs)
