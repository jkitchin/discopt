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

        # Objective Hessian ∇²f(x).
        if gn_obj_hess_fn is not None:
            self._hess_fn_jit = jax.jit(gn_obj_hess_fn)
        else:
            self._hess_fn_jit = jax.jit(jax.hessian(obj_fn, argnums=0))

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

            self._lagrangian_hess_fn_jit = jax.jit(jax.hessian(lagrangian, argnums=0))

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

    def evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Jacobian of constraints at x. Returns (m, n) array."""
        if self._jac_fn_jit is None:
            return np.empty((0, self._n_variables), dtype=np.float64)
        return np.asarray(self._jac_fn_jit(x, self._current_params()))

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

        # Lazy-initialize sparse infrastructure
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

        if self._sparse_jac_fn is not None:
            return self._sparse_jac_fn(x)

        # Fallback to dense
        return self.evaluate_jacobian(x)

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
            return values_fn(x64, float(obj_factor), lam64, self._current_params())
        h = self.evaluate_lagrangian_hessian(x, obj_factor, lambda_)
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
