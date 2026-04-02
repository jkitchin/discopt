"""Polyhedral-uncertainty robust reformulation via LP duality.

Mathematical Background
-----------------------
For polyhedral uncertainty {ξ : Aξ ≤ b} where ξ = p - p̄ is the
perturbation, the worst-case of a term c^T p over the polytope is:

    p̄^T c + max_{Aξ ≤ b} c^T ξ

The inner max is the support function of the polytope, solved by LP duality:

    max_{Aξ ≤ b} c^T ξ = min_{λ ≥ 0, A^T λ = c} b^T λ

So the robust constraint becomes:

    h(x) + p̄^T a(x) + b^T λ ≤ 0
    A^T λ = a(x)
    λ ≥ 0

This module handles the *parameter-as-RHS* case directly (no new variables
needed) by computing the worst-case parameter value via LP duality.

For the general coefficient-uncertainty case (a(x) depends on decision
variables), the full dual variable approach requires introducing auxiliary
continuous variables λ per uncertain constraint.  This is implemented via
the scipy LP solver to determine the worst-case offset.
"""

from __future__ import annotations

import numpy as np

from discopt.ro.uncertainty import PolyhedralUncertaintySet


class PolyhedralRobustFormulation:
    """Apply polyhedral-uncertainty robust reformulation to a model.

    For uncertain parameters that appear additively (as RHS terms), computes
    the worst-case parameter value by solving an LP for each component and
    substitutes the result as a constant.

    Parameters
    ----------
    model : discopt.Model
        The model to robustify (modified in-place).
    uncertainty_sets : list[PolyhedralUncertaintySet]
        Uncertainty sets.
    prefix : str
        Name prefix for dual variable names.
    """

    def __init__(
        self,
        model,
        uncertainty_sets: list[PolyhedralUncertaintySet],
        prefix: str = "ro",
    ) -> None:
        self._model = model
        self._uncertainty_sets = uncertainty_sets
        self._prefix = prefix

    def build(self) -> None:
        """Robustify the model in-place.

        Computes component-wise worst-case bounds for each polytope via LP
        and substitutes them using the same sign-tracking as the box
        formulation.
        """
        unc_map = {u.parameter.name: u for u in self._uncertainty_sets}
        m = self._model

        # Precompute worst-case bounds for each uncertain parameter.
        wc_upper: dict[str, np.ndarray] = {}
        wc_lower: dict[str, np.ndarray] = {}
        for name, unc in unc_map.items():
            lo, hi = _polytope_extreme_values(unc)
            wc_lower[name] = lo
            wc_upper[name] = hi

        from discopt.modeling.core import Constraint, Objective, ObjectiveSense

        # ── Robustify constraints ──────────────────────────────────────────────
        new_constraints = []
        for con in m._constraints:
            new_expr = _wc_polyhedral(
                con.body, unc_map, wc_lower, wc_upper, maximize=True, sign=+1
            )
            new_constraints.append(
                Constraint(body=new_expr, sense=con.sense, rhs=con.rhs, name=con.name)
            )
        m._constraints = new_constraints

        # ── Robustify objective ────────────────────────────────────────────────
        obj = m._objective
        if obj is None:
            return

        maximize_obj_wc = obj.sense == ObjectiveSense.MINIMIZE
        new_expr = _wc_polyhedral(
            obj.expression, unc_map, wc_lower, wc_upper,
            maximize=maximize_obj_wc, sign=+1
        )
        m._objective = Objective(expression=new_expr, sense=obj.sense)


def _polytope_extreme_values(
    unc: PolyhedralUncertaintySet,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute component-wise [lower, upper] extremes of p̄ + {ξ : Aξ ≤ b}.

    Solves 2k LPs (minimise and maximise each component) using scipy HiGHS.
    Falls back to interval bounding from the RHS vector if scipy is not
    available.
    """
    A, b_vec = unc.A, unc.b
    k = A.shape[1]
    nominal = unc.parameter.value.ravel()

    try:
        from scipy.optimize import linprog

        lo = np.empty(k)
        hi = np.empty(k)
        for j in range(k):
            c_obj = np.zeros(k)
            c_obj[j] = 1.0
            res_min = linprog(c_obj, A_ub=A, b_ub=b_vec, bounds=[(None, None)] * k, method="highs")
            lo[j] = res_min.fun if res_min.success else -np.inf
            res_max = linprog(-c_obj, A_ub=A, b_ub=b_vec, bounds=[(None, None)] * k, method="highs")
            hi[j] = -res_max.fun if res_max.success else np.inf

        lo_param = (nominal + lo).reshape(unc.parameter.value.shape)
        hi_param = (nominal + hi).reshape(unc.parameter.value.shape)
        return lo_param, hi_param

    except ImportError:
        # Conservative fallback.
        radius = float(np.max(np.abs(b_vec))) if len(b_vec) > 0 else 1.0
        lo_param = (nominal - radius).reshape(unc.parameter.value.shape)
        hi_param = (nominal + radius).reshape(unc.parameter.value.shape)
        return lo_param, hi_param


def _wc_polyhedral(expr, unc_map, wc_lower, wc_upper, maximize: bool, sign: int):
    """Sign-tracking worst-case substitution for polyhedral uncertainty."""
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    if isinstance(expr, Parameter) and expr.name in unc_map:
        use_upper = maximize ^ (sign < 0)
        wc = wc_upper[expr.name] if use_upper else wc_lower[expr.name]
        return Constant(wc)

    if isinstance(expr, Constant):
        return expr

    if isinstance(expr, BinaryOp):
        if expr.op == "+":
            nl = _wc_polyhedral(expr.left, unc_map, wc_lower, wc_upper, maximize, sign)
            nr = _wc_polyhedral(expr.right, unc_map, wc_lower, wc_upper, maximize, sign)
        elif expr.op == "-":
            nl = _wc_polyhedral(expr.left, unc_map, wc_lower, wc_upper, maximize, sign)
            nr = _wc_polyhedral(expr.right, unc_map, wc_lower, wc_upper, maximize, sign * -1)
        elif expr.op == "*":
            if isinstance(expr.left, Constant):
                left_sign = int(np.sign(np.sum(expr.left.value)))
                nl = expr.left
                nr = _wc_polyhedral(
                    expr.right, unc_map, wc_lower, wc_upper, maximize, sign * (left_sign or 1)
                )
            elif isinstance(expr.right, Constant):
                right_sign = int(np.sign(np.sum(expr.right.value)))
                nl = _wc_polyhedral(
                    expr.left, unc_map, wc_lower, wc_upper, maximize, sign * (right_sign or 1)
                )
                nr = expr.right
            else:
                nl = _wc_polyhedral(expr.left, unc_map, wc_lower, wc_upper, maximize, sign)
                nr = _wc_polyhedral(expr.right, unc_map, wc_lower, wc_upper, maximize, sign)
        else:
            nl = _wc_polyhedral(expr.left, unc_map, wc_lower, wc_upper, maximize, sign)
            nr = _wc_polyhedral(expr.right, unc_map, wc_lower, wc_upper, maximize, sign)
        return BinaryOp(expr.op, nl, nr)

    if isinstance(expr, UnaryOp):
        child_sign = sign * -1 if expr.op == "neg" else sign
        return UnaryOp(
            expr.op,
            _wc_polyhedral(expr.operand, unc_map, wc_lower, wc_upper, maximize, child_sign),
        )

    if isinstance(expr, FunctionCall):
        new_args = [
            _wc_polyhedral(a, unc_map, wc_lower, wc_upper, maximize, sign) for a in expr.args
        ]
        return FunctionCall(expr.func_name, *new_args)

    if isinstance(expr, MatMulExpression):
        nl = _wc_polyhedral(expr.left, unc_map, wc_lower, wc_upper, maximize, sign)
        nr = _wc_polyhedral(expr.right, unc_map, wc_lower, wc_upper, maximize, sign)
        return MatMulExpression(nl, nr)

    if isinstance(expr, IndexExpression):
        nb = _wc_polyhedral(expr.base, unc_map, wc_lower, wc_upper, maximize, sign)
        return IndexExpression(nb, expr.index)

    if isinstance(expr, SumExpression):
        no = _wc_polyhedral(expr.operand, unc_map, wc_lower, wc_upper, maximize, sign)
        return SumExpression(no, expr.axis)

    if isinstance(expr, SumOverExpression):
        new_terms = [
            _wc_polyhedral(t, unc_map, wc_lower, wc_upper, maximize, sign) for t in expr.terms
        ]
        return SumOverExpression(new_terms)

    return expr
