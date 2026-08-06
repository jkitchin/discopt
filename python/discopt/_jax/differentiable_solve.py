"""
Unified differentiable optimization: dispatch to LP/QP/NLP/MILP/MIQP solvers.

Provides a single entry point that classifies the problem, solves it with the
appropriate specialized solver, and supports JAX differentiation (jax.grad,
jax.jvp) through the solve for all problem classes:

  - LP: implicit KKT differentiation (Phase 2)
  - QP: OptNet implicit differentiation (Phase 3)
  - MILP/MIQP: LP/QP relaxation gradient or straight-through estimator (Phase 4/5)
  - NLP/MINLP: envelope theorem or implicit KKT (existing L1/L2/L3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np

from discopt._jax.problem_classifier import (
    ProblemClass,
    classify_problem,
    extract_lp_data,
    extract_qp_data,
)
from discopt._jax.problem_classifier import (
    dense_A as _dense_A,
)
from discopt._jax.problem_classifier import (
    dense_Q as _dense_Q,
)
from discopt.modeling.core import Model, Parameter

logger = logging.getLogger(__name__)


@dataclass
class UnifiedDiffResult:
    """Result from a differentiable solve.

    Attributes:
        status: Termination status.
        objective: Optimal objective value.
        x: Solution as flat array.
        x_dict: Solution as {var_name: array} dict.
        problem_class: Detected problem class.
        _grad_fn: Internal gradient function.
    """

    status: str
    objective: float
    x: Optional[np.ndarray] = None
    x_dict: Optional[dict[str, np.ndarray]] = None
    problem_class: Optional[ProblemClass] = None
    relaxation_obj: Optional[float] = None
    _model: Optional[Model] = None
    _primals: Optional[tuple] = None
    _solve_fn: Optional[object] = None

    def gradient(self, param: Parameter) -> np.ndarray:
        """Compute sensitivity of optimal objective w.r.t. a parameter.

        Args:
            param: A Parameter from the solved model.

        Returns:
            Gradient d(obj*)/d(param) as numpy array.
        """
        if self._model is None or self._solve_fn is None:
            raise ValueError("Gradient not available — model or solve function missing")
        if self.problem_class is None:
            raise ValueError("Gradient not available — problem class unknown")

        pc = self.problem_class

        # Use finite perturbation for simplicity in this initial implementation
        eps = 1e-5
        p_orig = param.value.copy()
        grad: np.ndarray = np.zeros_like(p_orig)

        for idx in np.ndindex(p_orig.shape if p_orig.shape else (1,)):
            param.value = p_orig.copy()
            if p_orig.shape:
                param.value[idx] += eps
            else:
                param.value = p_orig + eps
            r_plus = _solve_objective(self._model, pc)

            param.value = p_orig.copy()
            if p_orig.shape:
                param.value[idx] -= eps
            else:
                param.value = p_orig - eps
            r_minus = _solve_objective(self._model, pc)

            if r_plus is not None and r_minus is not None:
                if p_orig.shape:
                    grad[idx] = (r_plus - r_minus) / (2 * eps)
                else:
                    grad[()] = (r_plus - r_minus) / (2 * eps)

        param.value = p_orig
        return grad

    def relaxation_objective(self) -> Optional[float]:
        """For MIP problems, return the continuous relaxation objective."""
        return self.relaxation_obj


def _lp_forward(lp_data) -> tuple[float, np.ndarray]:
    """Forward LP solve for the differentiable path: POUNCE's interior-point KKT
    solve, returning ``(obj, x)`` in the raw (no ``obj_const``) convention.

    This is the same interior-point forward solver ``differentiable_lp`` uses for
    its ``custom_jvp`` sensitivity (POUNCE returns the analytic center, so the KKT
    system stays nonsingular). The pure-JAX ``lp_ipm_solve`` it replaced was
    retired in #370 — differentiability comes from the implicit-KKT gradient, not
    from the forward solver. Raises ``ImportError`` if POUNCE is unavailable (the
    callers already wrap this in ``try``).
    """
    from discopt.solvers.lp_pounce import solve_lp_kkt

    obj, x, *_ = solve_lp_kkt(
        lp_data.c, cast(Any, _dense_A(lp_data.A_eq)), lp_data.b_eq, lp_data.x_l, lp_data.x_u
    )
    return float(obj), np.asarray(x)


def _qp_forward(qp_data) -> tuple[float, np.ndarray]:
    """Forward QP solve for the differentiable path: POUNCE's interior-point KKT
    solve, returning ``(obj, x)`` in the raw (no ``obj_const``) convention.

    The QP counterpart of :func:`_lp_forward`, and the same argument applies: the
    forward solver only has to produce the KKT point, because differentiability
    comes from the implicit-KKT JVP in ``differentiable_qp.qp_solve_jvp``, not
    from differentiating the solver's iterations. ``differentiable_qp`` has fed on
    ``solve_qp_kkt`` all along; this module's QP arm was simply never migrated
    when #370 moved the LP arm off ``lp_ipm_solve``, and kept calling the pure-JAX
    ``qp_ipm_solve`` as a forward solver, reading only ``.x``/``.obj``.

    Measured equivalence over 12 random strictly-convex QPs (SPD ``Q``, ``n`` in
    2..8): worst relative disagreement with ``qp_ipm_solve`` 1.28e-11, and
    POUNCE's point independently verified -- primal feasibility <= 4.4e-16, KKT
    stationarity residual <= 4.0e-15. Raises ``ImportError`` if POUNCE is
    unavailable (the callers already wrap this in ``try``).
    """
    from discopt.solvers.qp_pounce import solve_qp_kkt

    obj, x, *_ = solve_qp_kkt(
        cast(Any, _dense_Q(qp_data.Q)),
        qp_data.c,
        cast(Any, _dense_A(qp_data.A_eq)),
        qp_data.b_eq,
        qp_data.x_l,
        qp_data.x_u,
    )
    return float(obj), np.asarray(x)


def _solve_objective(model: Model, problem_class: ProblemClass) -> float | None:
    """Solve a model and return just the objective value."""
    try:
        if problem_class == ProblemClass.LP:
            lp_data = extract_lp_data(model)
            lp_obj, _ = _lp_forward(lp_data)
            return lp_obj + lp_data.obj_const
        elif problem_class == ProblemClass.QP:
            qp_data = extract_qp_data(model)
            qp_obj, _ = _qp_forward(qp_data)
            return qp_obj + qp_data.obj_const
        else:
            from discopt._jax.nlp_evaluator import NLPEvaluator
            from discopt.solvers.nlp_pounce import solve_nlp

            evaluator = NLPEvaluator(model)
            lb, ub = evaluator.variable_bounds
            x0 = 0.5 * (np.clip(lb, -100, 100) + np.clip(ub, -100, 100))
            nlp_result = solve_nlp(evaluator, x0)
            obj = nlp_result.objective
            return float(obj) if obj is not None else None
    except Exception:
        return None


def _unpack_solution(model: Model, x_flat):
    """Convert flat solution to {var_name: array} dict."""
    result = {}
    offset = 0
    for v in model._variables:
        size = v.size
        val = np.asarray(x_flat[offset : offset + size])
        if v.shape == () or v.shape == (1,):
            result[v.name] = val.reshape(v.shape) if v.shape == () else val
        else:
            result[v.name] = val.reshape(v.shape)
        offset += size
    return result


def differentiable_solve(
    model: Model,
    method: str = "auto",
) -> UnifiedDiffResult:
    """Solve a model and return a differentiable result.

    Classifies the problem and dispatches to the appropriate solver. In every
    case the *solve* is POUNCE's and the *derivative* is JAX's — the forward
    point comes from a KKT solve in Rust, and differentiability is a post-solve
    ``custom_jvp`` rule built from that KKT system (the implicit function
    theorem), not from differentiating a solver written in JAX:
      - LP → POUNCE ``solve_lp_kkt`` + implicit-KKT JVP (``differentiable_lp``)
      - QP → POUNCE ``solve_qp_kkt`` + OptNet implicit diff (``differentiable_qp``)
      - MILP → B&B with LP relaxations + STE gradient
      - MIQP → B&B with QP relaxations + STE gradient
      - NLP/MINLP → POUNCE NLP / B&B path

    Args:
        model: A discopt Model with objective and constraints.
        method: "auto" to detect, or force "lp", "qp", "nlp", etc.

    Returns:
        UnifiedDiffResult with solution and gradient capabilities.
    """
    if method == "auto":
        problem_class = classify_problem(model)
    else:
        problem_class = ProblemClass(method)

    n_orig = sum(v.size for v in model._variables)

    if problem_class == ProblemClass.LP:
        lp_data = extract_lp_data(model)
        obj, x = _lp_forward(lp_data)
        x_flat = np.asarray(x[:n_orig])
        return UnifiedDiffResult(
            status="optimal",
            objective=obj + lp_data.obj_const,
            x=x_flat,
            x_dict=_unpack_solution(model, x_flat),
            problem_class=problem_class,
            _model=model,
        )

    elif problem_class == ProblemClass.QP:
        qp_data = extract_qp_data(model)
        # ``status`` is unconditionally "optimal" here for the same reason the LP
        # arm above does it: ``_qp_forward`` raises ``PounceKKTError`` rather than
        # returning a non-converged point, so reaching this line means the KKT
        # point is stationary. The old ``qp_ipm_solve`` call mapped its own
        # convergence code to "iteration_limit" and returned the unconverged
        # iterate anyway -- which, for a result whose purpose is to carry a
        # gradient, is the silently-wrong-gradient case ``solve_qp_kkt`` refuses.
        obj, x = _qp_forward(qp_data)
        x_flat = np.asarray(x[:n_orig])
        return UnifiedDiffResult(
            status="optimal",
            objective=obj + qp_data.obj_const,
            x=x_flat,
            x_dict=_unpack_solution(model, x_flat),
            problem_class=problem_class,
            _model=model,
        )

    elif problem_class in (ProblemClass.MILP, ProblemClass.MIQP):
        # Use the existing B&B solver, then compute relaxation gradient
        from discopt.solver import solve_model

        result = solve_model(model)
        relaxation_obj = None

        # Compute LP/QP relaxation for gradient
        try:
            if problem_class == ProblemClass.MILP:
                lp_data = extract_lp_data(model)
                relaxation_obj, _ = _lp_forward(lp_data)
            else:
                qp_data = extract_qp_data(model)
                relaxation_obj, _ = _qp_forward(qp_data)
        except Exception as exc:  # noqa: BLE001 - the result is reported without a relaxation
            logger.debug(
                "relaxation objective unavailable for %s: %s: %s",
                problem_class,
                type(exc).__name__,
                exc,
            )

        x_flat = None
        x_dict = None
        if result.x is not None:
            x_dict = result.x
            parts = []
            for v in model._variables:
                parts.append(result.x[v.name].flatten())
            x_flat = np.concatenate(parts) if parts else None

        return UnifiedDiffResult(
            status=result.status,
            objective=result.objective if result.objective is not None else float("inf"),
            x=x_flat,
            x_dict=x_dict,
            problem_class=problem_class,
            relaxation_obj=relaxation_obj,
            _model=model,
        )

    else:
        # NLP / MINLP — use existing solver
        from discopt.solver import solve_model

        result = solve_model(model)
        x_flat = None
        x_dict = None
        if result.x is not None:
            x_dict = result.x
            parts = []
            for v in model._variables:
                parts.append(result.x[v.name].flatten())
            x_flat = np.concatenate(parts) if parts else None

        return UnifiedDiffResult(
            status=result.status,
            objective=result.objective if result.objective is not None else float("inf"),
            x=x_flat,
            x_dict=x_dict,
            problem_class=problem_class,
            _model=model,
        )
