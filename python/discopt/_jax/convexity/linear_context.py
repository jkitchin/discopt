"""Linear-constraint context for constraint-aware sign reasoning.

The syntactic SUSPECT-style walker in :mod:`rules` determines the sign
of each subexpression from its algebraic structure and the declared
variable bounds alone. For atoms with restricted domains (``log``,
``sqrt``, ``1/x``, fractional ``x**p``), a sign of the argument that
is merely ``NONNEG`` or ``UNKNOWN`` is not enough to apply the DCP
concavity / convexity rule. But the argument's sign often *is*
provable once linear inequalities and equalities of the model are
taken into account — for example ``log(1 + x1 - x2)`` with the
constraint ``x2 <= x1`` implies the argument is ``>= 1 > 0``.

This module provides a ``LinearContext`` that holds the model's
linear relaxation (variable bounds + linear inequality and equality
constraints) and can answer range queries on affine expressions via
two scipy ``linprog`` calls. The range is a sound enclosure over the
intersection of the box with the linear relaxation, so the resulting
sign label is mathematically valid as a premise of a DCP rule.

Range enclosures for affine expressions are exact; for nonlinear
arguments we fall back to the declared variable box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    IndexExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)

# ──────────────────────────────────────────────────────────────────────
# Affine coefficient extraction
# ──────────────────────────────────────────────────────────────────────


def _compute_var_offset(var: Variable, model: Model) -> int:
    offset = 0
    for v in model._variables[: var._index]:
        offset += v.size
    return offset


def extract_affine(
    expr: Expression, model: Model, n_vars: int
) -> Optional[tuple[np.ndarray, float]]:
    """Return ``(coeffs, const)`` for an affine scalar expression.

    Walks the DAG collecting linear coefficients. Returns ``None``
    when the expression contains any nonlinear operator or shape that
    doesn't reduce to a scalar affine form.
    """

    def walk(node: Expression, scale: float) -> Optional[tuple[dict[int, float], float]]:
        if isinstance(node, Constant):
            val = np.asarray(node.value)
            if val.ndim == 0:
                return {}, scale * float(val)
            return None

        if isinstance(node, Parameter):
            val = np.asarray(node.value)
            if val.ndim == 0:
                return {}, scale * float(val)
            return None

        if isinstance(node, Variable):
            if node.size != 1:
                return None
            return {_compute_var_offset(node, model): scale}, 0.0

        if isinstance(node, IndexExpression) and isinstance(node.base, Variable):
            base_off = _compute_var_offset(node.base, model)
            idx = node.index
            if isinstance(idx, (int, np.integer)):
                return {base_off + int(idx): scale}, 0.0
            if isinstance(idx, tuple) and len(idx) == 1 and isinstance(idx[0], (int, np.integer)):
                return {base_off + int(idx[0]): scale}, 0.0
            try:
                flat = int(np.ravel_multi_index(idx, node.base.shape))
            except (TypeError, ValueError):
                return None
            return {base_off + flat: scale}, 0.0

        if isinstance(node, UnaryOp):
            if node.op == "neg":
                return walk(node.operand, -scale)
            return None

        if isinstance(node, BinaryOp):
            if node.op in ("+", "-"):
                left = walk(node.left, scale)
                right = walk(node.right, scale if node.op == "+" else -scale)
                if left is None or right is None:
                    return None
                merged = dict(left[0])
                for k, v in right[0].items():
                    merged[k] = merged.get(k, 0.0) + v
                return merged, left[1] + right[1]

            if node.op == "*":
                if _is_scalar_const(node.left):
                    return walk(node.right, scale * _scalar_value(node.left))
                if _is_scalar_const(node.right):
                    return walk(node.left, scale * _scalar_value(node.right))
                return None

            if node.op == "/":
                if _is_scalar_const(node.right):
                    divisor = _scalar_value(node.right)
                    if abs(divisor) <= 1e-30:
                        return None
                    return walk(node.left, scale / divisor)
                return None

            return None

        if isinstance(node, SumExpression):
            return walk(node.operand, scale)

        if isinstance(node, SumOverExpression):
            acc: dict[int, float] = {}
            total = 0.0
            for t in node.terms:
                part = walk(t, scale)
                if part is None:
                    return None
                for k, v in part[0].items():
                    acc[k] = acc.get(k, 0.0) + v
                total += part[1]
            return acc, total

        return None

    result = walk(expr, 1.0)
    if result is None:
        return None
    coeffs_dict, const = result
    coeffs = np.zeros(n_vars, dtype=np.float64)
    for idx, v in coeffs_dict.items():
        if 0 <= idx < n_vars:
            coeffs[idx] = v
    return coeffs, const


def _is_scalar_const(expr: Expression) -> bool:
    if isinstance(expr, (Constant, Parameter)):
        val = np.asarray(expr.value)
        return bool(val.ndim == 0)
    return False


def _scalar_value(expr: Expression) -> float:
    return float(np.asarray(expr.value))  # type: ignore[attr-defined]


# ──────────────────────────────────────────────────────────────────────
# Linear context
# ──────────────────────────────────────────────────────────────────────


@dataclass
class LinearContext:
    """Linear relaxation of a model for affine range queries.

    ``A_ub x <= b_ub``, ``A_eq x = b_eq``, ``lb <= x <= ub``. The
    coefficient matrices may be empty arrays (no linear constraints);
    variable bounds are always present. ``n_vars`` is the flattened
    decision-variable dimension.
    """

    n_vars: int
    lb: np.ndarray
    ub: np.ndarray
    A_ub: np.ndarray
    b_ub: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray

    def affine_range(self, coeffs: np.ndarray, const: float) -> tuple[float, float]:
        """Sound enclosure of ``coeffs · x + const`` over the relaxation.

        Uses the declared variable bounds as a free box-only enclosure;
        invokes ``scipy.optimize.linprog`` only when linear constraints
        are present, since the box-only bound is already optimal
        otherwise.
        """
        # Box-only enclosure is optimal when there are no linear rows.
        lo_box, hi_box = _box_range(coeffs, self.lb, self.ub)
        lo_box += const
        hi_box += const
        if self.A_ub.size == 0 and self.A_eq.size == 0:
            return lo_box, hi_box

        from scipy.optimize import linprog

        # Replace ±inf in variable bounds with None for linprog's API.
        bounds = [
            (None if not np.isfinite(lo) else float(lo), None if not np.isfinite(hi) else float(hi))
            for lo, hi in zip(self.lb, self.ub)
        ]

        A_ub = self.A_ub if self.A_ub.size else None
        b_ub = self.b_ub if self.b_ub.size else None
        A_eq = self.A_eq if self.A_eq.size else None
        b_eq = self.b_eq if self.b_eq.size else None

        try:
            lo_res = linprog(
                coeffs,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )
            hi_res = linprog(
                -coeffs,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )
        except (ValueError, RuntimeError):
            return lo_box, hi_box

        lo = float(lo_res.fun) + const if lo_res.success else lo_box
        hi = -float(hi_res.fun) + const if hi_res.success else hi_box
        # Intersect with the box-only enclosure; linprog errors only widen.
        return max(lo, lo_box), min(hi, hi_box)


def _box_range(coeffs: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> tuple[float, float]:
    """Box-only enclosure of ``coeffs · x`` without the linear rows."""
    pos = coeffs > 0
    neg = coeffs < 0
    lo = float(np.sum(coeffs[pos] * lb[pos]) + np.sum(coeffs[neg] * ub[neg]))
    hi = float(np.sum(coeffs[pos] * ub[pos]) + np.sum(coeffs[neg] * lb[neg]))
    return lo, hi


def build_linear_context(model: Model) -> Optional[LinearContext]:
    """Assemble a :class:`LinearContext` from a model's linear rows.

    Returns ``None`` when the model has no variables. Nonlinear
    constraints are silently dropped; only constraints that reduce to
    ``coeffs · x + const  sense  0`` contribute rows.
    """
    if not model._variables:
        return None

    n_vars = sum(v.size for v in model._variables)
    lb = np.empty(n_vars, dtype=np.float64)
    ub = np.empty(n_vars, dtype=np.float64)
    offset = 0
    for v in model._variables:
        vlb = np.asarray(v.lb, dtype=np.float64).reshape(-1)
        vub = np.asarray(v.ub, dtype=np.float64).reshape(-1)
        lb[offset : offset + v.size] = vlb
        ub[offset : offset + v.size] = vub
        offset += v.size

    ub_rows: list[tuple[np.ndarray, float]] = []
    eq_rows: list[tuple[np.ndarray, float]] = []

    for c in model._constraints:
        if not isinstance(c, Constraint):
            continue
        aff = extract_affine(c.body, model, n_vars)
        if aff is None:
            continue
        coeffs, const = aff
        # body sense rhs  →  (coeffs · x + const)  sense  rhs
        adjusted_rhs = float(c.rhs) - const
        if c.sense == "<=":
            ub_rows.append((coeffs, adjusted_rhs))
        elif c.sense == ">=":
            ub_rows.append((-coeffs, -adjusted_rhs))
        elif c.sense == "==":
            eq_rows.append((coeffs, adjusted_rhs))

    A_ub = np.vstack([r[0] for r in ub_rows]) if ub_rows else np.zeros((0, n_vars))
    b_ub = np.array([r[1] for r in ub_rows], dtype=np.float64) if ub_rows else np.zeros(0)
    A_eq = np.vstack([r[0] for r in eq_rows]) if eq_rows else np.zeros((0, n_vars))
    b_eq = np.array([r[1] for r in eq_rows], dtype=np.float64) if eq_rows else np.zeros(0)

    return LinearContext(n_vars=n_vars, lb=lb, ub=ub, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)


__all__ = ["LinearContext", "build_linear_context", "extract_affine"]
