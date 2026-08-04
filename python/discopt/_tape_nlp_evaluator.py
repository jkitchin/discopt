"""A JAX-free NLP evaluator backed by POUNCE's Rust AD tape (issue #75, Stage 3).

``_jax/nlp_evaluator.py`` is the single trigger that imports JAX on *every*
nonlinear solve — measured across 27 operator-diverse corpus instances, the first
jax import is always ``nlp_evaluator.py:22``. This module supplies the same
quantities (``f``, ``grad f``, ``g``, ``J``, and the Lagrangian Hessian, dense and
sparse) from a tape built by :mod:`discopt._nl_expr_compiler`, with no JAX at all.

**Gated, default OFF.** ``DISCOPT_NLP_EVAL=tape`` opts in; anything else keeps the
JAX evaluator. Routing derivatives through a different AD engine is
bound-CHANGING under CLAUDE.md §5 — the B&B is path-dependent, so even last-digit
differences can move the cut sequence — and graduating the default requires the
differential panel, not this module landing.

Entry experiment before any of this was written (CLAUDE.md §4), tape vs the JAX
evaluator over 66 in-repo corpus instances at 5 points each:

    f  5.48e-16   grad f  3.77e-16   g  4.55e-13   J  3.06e-15   Lagrangian H  7.82e-13

against Step 2.2's bars of 1e-10 on grad/J and 1e-8 on the Hessian.

Two things this refuses rather than approximates, both because the alternative is
a silently wrong derivative:

* **Array-valued constraint bodies.** A tape node is a scalar. ``DAEBuilder``
  collocation emits one array-valued body per block, which has no scalar
  lowering.
* **Gauss-Newton mode.** ``H_obj ~ 2 J^T J`` over residuals has no tape analogue;
  the tape computes the exact Hessian, which is a different matrix.

Callers use :func:`try_build`, which returns ``None`` for both, so the JAX path
stays intact underneath.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from discopt._nl_expr_compiler import UnsupportedForTape, compile_to_nl_expr

if TYPE_CHECKING:  # pragma: no cover - typing only
    from discopt.modeling.core import Constraint, Model


def tape_backend_requested() -> bool:
    """True when ``DISCOPT_NLP_EVAL=tape``.

    Read per call rather than cached at import so a test can flip it without
    reloading the module.
    """
    return os.environ.get("DISCOPT_NLP_EVAL", "jax").strip().lower() == "tape"


class TapeNLPEvaluator:
    """Tape-backed drop-in for :class:`discopt._jax.nlp_evaluator.NLPEvaluator`.

    Exposes the same surface the solver, ``nlp_ipopt``, ``oa`` and ``amp``
    consume. Conventions match the JAX evaluator exactly, because the callers
    depend on them:

    * ``evaluate_lagrangian_hessian`` returns a **full dense ``(n, n)``** matrix
      (``nlp_ipopt`` indexes it with ``h[rows, cols]``), while
      ``hessian_structure`` / ``evaluate_hessian_values`` are **lower-triangle**
      COO.
    * a MAXIMIZE objective is negated, so every caller minimizes.
    """

    def __init__(self, model: "Model") -> None:
        import pounce

        from discopt.modeling.core import Constraint, ObjectiveSense

        if model._objective is None:
            raise ValueError("Model has no objective set.")
        if bool(getattr(model, "_gauss_newton_hessian", False)):
            raise UnsupportedForTape(
                "Gauss-Newton objective Hessian has no tape analogue; the tape "
                "computes the exact Hessian, which is a different matrix"
            )

        self._model = model
        self._pounce = pounce
        self._negate = model._objective.sense == ObjectiveSense.MAXIMIZE
        self._n_variables = sum(v.size for v in model._variables)

        # Same source list as the JAX evaluator, including the #840 fast-path
        # builder rows. Reading only ``model._constraints`` both misses those and
        # mis-indexes the ones it does read.
        self._source_constraints: list[Constraint] = [
            c
            for c in (*model._constraints, *model._builder_linear_constraints())
            if isinstance(c, Constraint)
        ]

        # Parameters are baked into the tape as constants (a tape is built for
        # fixed structure). The JAX evaluator instead plumbs them as runtime args
        # and `evaluator_fingerprint` DELIBERATELY excludes `Parameter.value` for
        # that reason. So a tape cached under that fingerprint would serve stale
        # derivatives after a re-bind. Snapshot the values and rebuild when they
        # move -- a tape build is milliseconds, unlike a JAX trace.
        self._parameters = list(model._parameters)
        self._param_snapshot = self._snapshot_params()

        self._build()

    # -- construction -------------------------------------------------------

    def _snapshot_params(self) -> tuple:
        return tuple(np.asarray(p.value, dtype=float).copy() for p in self._parameters)

    def _params_changed(self) -> bool:
        current = self._snapshot_params()
        if len(current) != len(self._param_snapshot):
            return True
        return any(
            a.shape != b.shape or not np.array_equal(a, b)
            for a, b in zip(current, self._param_snapshot)
        )

    def _build(self) -> None:
        """(Re)build this thread's tape and drop every other thread's.

        THREAD AFFINITY, and it is load-bearing. ``pounce.NlProblem`` is a
        ``#[pyclass(unsendable)]``: touching one from a thread other than the one
        that created it raises a Rust ``PanicException``. That is not a normal
        error — ``PanicException`` derives directly from ``BaseException``, so it
        slips past every ``except Exception`` in the solver, and the measured
        result was a **false `infeasible`** on clay0303hfsg (JAX arm: `feasible`,
        obj 26669.1). A wrong certificate is the one outcome that is never
        acceptable (CLAUDE.md §1), so the tape is built per thread rather than
        shared. A build is milliseconds; a JAX trace is seconds, which is why the
        JAX evaluator cannot afford the same trick and does not need it.

        The underlying ``NlTnlp`` IS ``Send`` — pounce keeps the pyclass
        unsendable as a conservative default — so this is a workaround for a
        policy, not a physical limit, and can be dropped if that changes.
        """
        model = self._model
        objective = model._objective
        assert objective is not None  # refused in __init__
        obj = compile_to_nl_expr(objective.expression, model)
        if self._negate:
            obj = -obj

        cons = [compile_to_nl_expr(c.body, model) for c in self._source_constraints]

        # Every body that lowered is scalar by construction: the compiler refuses
        # array variables and every array reduction. So each source constraint is
        # exactly one row, and the row map is the identity -- unlike the JAX path,
        # which must call `jax.eval_shape` because an array body is one Constraint
        # and many rows.
        self._constraint_flat_sizes = np.ones(len(cons), dtype=np.intp)
        self._n_constraints = len(cons)

        lb, ub = self.variable_bounds
        self._lo = [float(v) for v in lb]
        self._hi = [float(v) for v in ub]
        self._obj_expr = obj
        self._con_exprs = cons
        # Per-thread NlProblem, in thread-LOCAL storage rather than a plain dict
        # keyed by thread id. The difference matters: an unsendable pyclass also
        # refuses to be DROPPED on a foreign thread, and a dict outliving its
        # threads hands every entry to whichever thread happens to run the GC ->
        # "unsendable, but is being dropped on another thread" at teardown.
        # CPython clears a threading.local on the owning thread as it exits, so
        # each tape is both built and dropped where it belongs.
        # `_generation` invalidates every thread's copy on a rebuild without
        # needing to enumerate them (threading.local cannot be iterated).
        self._local = threading.local()
        self._generation = getattr(self, "_generation", 0) + 1
        self._jac_struct: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._hess_struct: Optional[tuple[np.ndarray, np.ndarray]] = None

    def _problem_for_this_thread(self) -> Any:
        cached = getattr(self._local, "entry", None)
        if cached is not None and cached[0] == self._generation:
            return cached[1]
        prob = self._pounce.build_nl_problem(
            self._n_variables,
            self._obj_expr,
            constraints=self._con_exprs or None,
            x_l=self._lo,
            x_u=self._hi,
        )
        self._local.entry = (self._generation, prob)
        return prob

    @property
    def _problem(self) -> Any:
        return self._problem_for_this_thread()

    def _ensure_fresh(self) -> None:
        """Rebuild if a ``Parameter.value`` moved since the tape was built."""
        if self._parameters and self._params_changed():
            self._param_snapshot = self._snapshot_params()
            self._build()

    # -- shape / structure --------------------------------------------------

    @property
    def n_variables(self) -> int:
        return self._n_variables

    @property
    def n_constraints(self) -> int:
        return self._n_constraints

    @property
    def is_gauss_newton(self) -> bool:
        # Refused in __init__; a tape is always the exact Hessian.
        return False

    @property
    def variable_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lbs, ubs = [], []
        for v in self._model._variables:
            lbs.append(np.asarray(v.lb).flatten())
            ubs.append(np.asarray(v.ub).flatten())
        if not lbs:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(lbs), np.concatenate(ubs)

    def constraint_row_map(self) -> list[tuple[int, int, "Constraint"]]:
        return [(i, i + 1, c) for i, c in enumerate(self._source_constraints)]

    def jacobian_structure(self) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_fresh()
        if self._jac_struct is None:
            r, c = self._problem.jacobian_structure()
            self._jac_struct = (
                np.asarray(r, dtype=np.int64),
                np.asarray(c, dtype=np.int64),
            )
        return self._jac_struct

    def hessian_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower-triangle COO, matching the JAX evaluator's convention."""
        self._ensure_fresh()
        if self._hess_struct is None:
            r, c = self._problem.hessian_structure()
            self._hess_struct = (
                np.asarray(r, dtype=np.int64),
                np.asarray(c, dtype=np.int64),
            )
        return self._hess_struct

    def has_sparse_structure(self) -> bool:
        """Always: the tape reports exact sparsity with no probing step."""
        return True

    def sparsity_pattern(self):
        return self.jacobian_structure()

    @property
    def hessian_kernel_compiled(self) -> bool:
        """A tape has no compile step, so the budget gate must never wait on one."""
        return True

    def hessian_compile_estimate_s(self) -> float:
        return 0.0

    # -- values -------------------------------------------------------------

    def _x(self, x: np.ndarray) -> list:
        return [float(v) for v in np.asarray(x, dtype=float).ravel()]

    def evaluate_objective(self, x: np.ndarray) -> float:
        self._ensure_fresh()
        return float(self._problem.objective(self._x(x)))

    def evaluate_gradient(self, x: np.ndarray) -> np.ndarray:
        self._ensure_fresh()
        return np.asarray(self._problem.gradient(self._x(x)), dtype=np.float64)

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        self._ensure_fresh()
        if self._n_constraints == 0:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(self._problem.constraints(self._x(x)), dtype=np.float64)

    def evaluate_jacobian_values(self, x: np.ndarray) -> np.ndarray:
        self._ensure_fresh()
        if self._n_constraints == 0:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(self._problem.jacobian(self._x(x)), dtype=np.float64)

    def evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Dense ``(m, n)`` Jacobian, scattered from the tape's COO values."""
        m, n = self._n_constraints, self._n_variables
        if m == 0:
            return np.zeros((0, n), dtype=np.float64)
        rows, cols = self.jacobian_structure()
        dense = np.zeros((m, n), dtype=np.float64)
        np.add.at(dense, (rows, cols), self.evaluate_jacobian_values(x))
        return dense

    def evaluate_hessian_values(
        self, x: np.ndarray, obj_factor: float, lambda_: np.ndarray
    ) -> np.ndarray:
        """Lower-triangle Lagrangian Hessian values, aligned to ``hessian_structure``."""
        self._ensure_fresh()
        lam = np.asarray(lambda_, dtype=float).ravel()
        return np.asarray(
            self._problem.hessian(
                self._x(x), lam=[float(v) for v in lam], obj_factor=float(obj_factor)
            ),
            dtype=np.float64,
        )

    def evaluate_lagrangian_hessian(
        self, x: np.ndarray, obj_factor: float, lambda_: np.ndarray
    ) -> np.ndarray:
        """FULL dense ``(n, n)`` Lagrangian Hessian.

        Full, not lower-triangular: ``nlp_ipopt.hessian`` indexes the result with
        ``h[rows, cols]`` against its own structure, and ``sipopt`` /
        ``benders/_feasibility`` treat it as a symmetric matrix. Returning a bare
        triangle here would silently zero half of every off-diagonal term.
        """
        n = self._n_variables
        rows, cols = self.hessian_structure()
        vals = self.evaluate_hessian_values(x, obj_factor, lambda_)
        lower = np.zeros((n, n), dtype=np.float64)
        np.add.at(lower, (rows, cols), vals)
        full: np.ndarray = lower + lower.T - np.diag(np.diag(lower))
        return full

    def evaluate_hessian(self, x: np.ndarray) -> np.ndarray:
        """Dense objective Hessian: the Lagrangian at ``obj_factor=1, lam=0``."""
        return self.evaluate_lagrangian_hessian(
            x, 1.0, np.zeros(self._n_constraints, dtype=np.float64)
        )

    # -- legacy single-argument wrappers ------------------------------------
    # `_obj_fn` / `_cons_fn` are read directly by the IPM batch path, OA, and
    # alpha estimation. They must exist and must read current parameter values.

    def _obj_fn(self, x: np.ndarray) -> float:
        return self.evaluate_objective(x)

    def _cons_fn(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.evaluate_constraints(x), dtype=np.float64)


def try_build(model: "Model") -> Optional[TapeNLPEvaluator]:
    """Build a tape evaluator, or ``None`` when the model is unrepresentable.

    ``None`` means *representability* only — an array-valued body, an operator
    with no tape lowering, ``dm.custom``, or Gauss-Newton mode. It never means a
    numerical failure, so a caller cannot confuse "no tape" with "bad point".
    """
    try:
        return TapeNLPEvaluator(model)
    except UnsupportedForTape:
        return None


def cached_tape_evaluator(model: "Model") -> Optional[TapeNLPEvaluator]:
    """Per-model LRU-cached tape evaluator, or ``None`` if unrepresentable.

    Shares :mod:`discopt._evaluator_cache` with the JAX backend so the two agree
    on what "still valid" means, on its own cache attribute so a backend switch
    cannot hand one backend's evaluator to the other.

    A ``None`` result is cached too, as a sentinel: representability is a pure
    function of the structural fingerprint, so re-attempting the build at every
    B&B node would re-walk the whole DAG only to fail again.
    """
    from discopt._evaluator_cache import cached_by_fingerprint

    built = cached_by_fingerprint(model, "_tape_evaluator_cache", _build_or_sentinel)
    return None if built is _UNREPRESENTABLE else built


_UNREPRESENTABLE = object()


def _build_or_sentinel(model: "Model") -> Any:
    ev = try_build(model)
    return _UNREPRESENTABLE if ev is None else ev


def build_evaluator(model: "Model", jax_factory: Any) -> Any:
    """The tape evaluator when opted in and representable, else ``jax_factory()``.

    Single decision point, so the fallback is one branch rather than one per call
    site. ``jax_factory`` is a zero-argument callable and is not invoked when the
    tape is used, which is what keeps JAX out of ``sys.modules`` on the tape path.
    """
    if tape_backend_requested():
        ev = cached_tape_evaluator(model)
        if ev is not None:
            return ev
    return jax_factory()
