"""Shared linear-model extraction for the decomposition solvers.

Both Benders and Lagrangian relaxation operate on the model's linear algebra:
a list of ``<=``-canonical rows and a linear (minimization) objective. Each row
is tagged with the index of the originating ``model._constraints`` entry, so
Lagrangian relaxation can pick out the rows belonging to a coupling constraint
(an equality is split into two ``<=`` rows that share the same source index).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discopt.modeling.core import Constraint, Model, ObjectiveSense


@dataclass
class LinearModel:
    """Linear model in ``<=``-canonical form, internally minimizing."""

    n: int
    rows_coeff: list[np.ndarray]  # each shape (n,), a^T x <= rhs
    rows_rhs: list[float]
    rows_source: list[int]  # index into model._constraints
    c: np.ndarray  # objective coefficients (minimization), shape (n,)
    c_offset: float
    minimize: bool  # False -> objective was maximize (c already negated)


def extract_linear(model: Model) -> LinearModel:
    """Extract ``<=``-canonical linear rows and a linear objective.

    Raises ``NotImplementedError`` for nonlinear constraints/objectives or
    non-algebraic constraints (SOS, indicator, ...).
    """
    from discopt._jax.gdp_reformulate import _extract_body_coeffs, _is_linear

    n = sum(v.size for v in model._variables)

    rows_coeff: list[np.ndarray] = []
    rows_rhs: list[float] = []
    rows_source: list[int] = []

    def _add(vec: np.ndarray, rhs: float, sense: str, src: int) -> None:
        if sense == "<=":
            rows_coeff.append(vec)
            rows_rhs.append(rhs)
            rows_source.append(src)
        elif sense == ">=":
            rows_coeff.append(-vec)
            rows_rhs.append(-rhs)
            rows_source.append(src)
        else:  # "==" -> two inequalities sharing the source index
            rows_coeff.append(vec)
            rows_rhs.append(rhs)
            rows_source.append(src)
            rows_coeff.append(-vec)
            rows_rhs.append(-rhs)
            rows_source.append(src)

    def _coeffs(expr, what: str):
        # ``_extract_body_coeffs`` raises (e.g. a TypeError on multi-dimensional
        # indexed variables) rather than returning None for some unsupported
        # constructs; normalize every failure to a clean NotImplementedError so
        # the solvers report "unsupported", not a stray internal error.
        try:
            res = _extract_body_coeffs(expr, model, n)
        except Exception as exc:
            raise NotImplementedError(
                f"Decomposition v1 could not extract linear coefficients from {what} "
                "(unsupported construct, e.g. a multi-dimensional indexed variable). "
                "Use a 1-D flattened formulation, or Model.solve()."
            ) from exc
        if res is None:
            raise NotImplementedError(
                f"Decomposition v1 requires a linear {what}; could not extract coefficients."
            )
        return res

    for src, c in enumerate(model._constraints):
        if not isinstance(c, Constraint):
            raise NotImplementedError(
                "Decomposition v1 supports only algebraic linear constraints "
                f"(got {type(c).__name__})."
            )
        if not _is_linear(c.body):
            raise NotImplementedError(
                "Decomposition v1 supports linear constraints only; the model "
                "has a nonlinear constraint."
            )
        vec, off = _coeffs(c.body, "a constraint")
        _add(np.asarray(vec, dtype=np.float64), -float(off), c.sense, src)

    obj = model._objective
    if obj is None:
        c_vec = np.zeros(n)
        c_off = 0.0
        minimize = True
    else:
        oc = _coeffs(obj.expression, "the objective")
        c_vec = np.asarray(oc[0], dtype=np.float64)
        c_off = float(oc[1])
        minimize = obj.sense == ObjectiveSense.MINIMIZE
    if not minimize:
        c_vec = -c_vec
        c_off = -c_off

    return LinearModel(n, rows_coeff, rows_rhs, rows_source, c_vec, c_off, minimize)


def relative_gap(ub: float, lb: float) -> float:
    """Relative optimality gap ``(ub - lb) / max(1, |ub|, |lb|)``.

    The ``max(1, ...)`` denominator (rather than ``|ub| + eps``) keeps the gap
    well-scaled when the objective is near zero: a tiny ``|ub|`` would otherwise
    inflate the relative gap and prevent convergence on problems whose optimum is
    close to 0 (T0.5).
    """
    denom = max(1.0, abs(ub), abs(lb))
    return (ub - lb) / denom


def solution_dict(model: Model, x_full: np.ndarray) -> dict[str, np.ndarray]:
    """Split a flat solution vector into a name -> reshaped-array dict."""
    out: dict[str, np.ndarray] = {}
    off = 0
    for v in model._variables:
        vals = x_full[off : off + v.size]
        out[v.name] = np.asarray(vals).reshape(v.shape) if v.shape else vals.reshape(())
        off += v.size
    return out


__all__ = ["LinearModel", "extract_linear", "relative_gap", "solution_dict"]
