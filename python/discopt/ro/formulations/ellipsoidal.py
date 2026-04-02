"""Ellipsoidal-uncertainty robust reformulation.

Mathematical Background
-----------------------
For an objective linear in an uncertain parameter p:

    min_x  c(x)^T p     with  ||Σ^{-1/2}(p - p̄)||₂ ≤ ρ

The worst-case value (maximum over the ellipsoid) is:

    max_{||Σ^{-1/2}(p - p̄)||₂ ≤ ρ} c(x)^T p
    = c(x)^T p̄ + ρ ||Σ^{1/2} c(x)||₂

So the robust objective is:

    min_x  c(x)^T p̄ + ρ ||Σ^{1/2} c(x)||₂

For a constraint linear in p:

    a(x)^T p ≤ b(x)

the robust counterpart is:

    a(x)^T p̄ + ρ ||Σ^{1/2} a(x)||₂ ≤ b(x)

This is a second-order cone (SOC) constraint.  This module handles the
important special case where the uncertain parameter appears in a matrix
multiplication ``p @ x`` or ``x @ p`` in the objective or constraints.

For RHS uncertainty (p appears as the bound of a constraint ``h(x) ≤ p``),
the worst-case RHS is p̄ - ρ ||e||_Σ, where e is the sign vector.  In the
scalar case this simplifies to p̄ - ρ.

Reference
---------
Ben-Tal, A., Nemirovski, A. (1999). Robust solutions of uncertain linear
programs. *Operations Research Letters*, 25(1), 1–13.
"""

from __future__ import annotations

import numpy as np

from discopt.ro.uncertainty import EllipsoidalUncertaintySet


class EllipsoidalRobustFormulation:
    """Apply ellipsoidal-uncertainty robust reformulation to a model.

    For each model component where the uncertain parameter p appears in a
    dot product with decision variables (p @ x or dot(p, x)), replaces
    with the nominal contribution plus the SOC robustness penalty:

        p̄^T x  +  ρ ||Σ^{1/2} x||₂

    Parameters
    ----------
    model : discopt.Model
        The model to robustify (modified in-place).
    uncertainty_sets : list[EllipsoidalUncertaintySet]
        Uncertainty sets.
    prefix : str
        Name prefix for auxiliary variables / constraints.
    """

    def __init__(
        self,
        model,
        uncertainty_sets: list[EllipsoidalUncertaintySet],
        prefix: str = "ro",
    ) -> None:
        self._model = model
        self._uncertainty_sets = uncertainty_sets
        self._prefix = prefix

    def build(self) -> None:
        """Robustify the model in-place.

        Rewrites dot products involving uncertain parameters by substituting
        the nominal value and adding the corresponding 2-norm penalty term.
        """
        unc_map = {u.parameter.name: u for u in self._uncertainty_sets}
        m = self._model

        from discopt.modeling.core import Constraint, Objective, ObjectiveSense

        # ── Robustify constraints ──────────────────────────────────────────────
        new_constraints = []
        for con in m._constraints:
            new_expr, penalties = _extract_penalties(con.body, unc_map)
            if penalties:
                combined = _sum_expr([new_expr] + penalties)
                new_constraints.append(
                    Constraint(body=combined, sense=con.sense, rhs=con.rhs, name=con.name)
                )
            else:
                new_constraints.append(con)
        m._constraints = new_constraints

        # ── Robustify objective ────────────────────────────────────────────────
        obj = m._objective
        if obj is None:
            return

        new_expr, penalties = _extract_penalties(obj.expression, unc_map)
        if penalties:
            # For MINIMIZE: worst case → largest → add positive penalties.
            # For MAXIMIZE: worst case → smallest → subtract penalties.
            sign = +1 if obj.sense == ObjectiveSense.MINIMIZE else -1
            if sign == +1:
                combined = _sum_expr([new_expr] + penalties)
            else:
                from discopt.modeling.core import BinaryOp

                combined = new_expr
                for pen in penalties:
                    combined = BinaryOp("-", combined, pen)
            m._objective = Objective(expression=combined, sense=obj.sense)


# ─────────────────────────────────────────────────────────────────────────────
# Expression rewriting helpers
# ─────────────────────────────────────────────────────────────────────────────


def _extract_penalties(expr, unc_map: dict):
    """Walk expr, replacing p with p̄ and collecting SOC penalties.

    Returns (modified_expr, list[penalty_expr]).
    """
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

    penalties: list = []

    if isinstance(expr, Parameter) and expr.name in unc_map:
        unc = unc_map[expr.name]
        return Constant(unc.parameter.value), []

    if isinstance(expr, Constant):
        return expr, []

    if isinstance(expr, MatMulExpression):
        # Detect  p @ x  or  x @ p  where p is uncertain.
        left, right = expr.left, expr.right
        if isinstance(left, Parameter) and left.name in unc_map:
            unc = unc_map[left.name]
            nominal = MatMulExpression(Constant(unc.parameter.value), right)
            penalty = _soc_penalty(right, unc)
            return nominal, [penalty]
        if isinstance(right, Parameter) and right.name in unc_map:
            unc = unc_map[right.name]
            nominal = MatMulExpression(left, Constant(unc.parameter.value))
            penalty = _soc_penalty(left, unc)
            return nominal, [penalty]
        # Neither side is uncertain: recurse.
        nl, pl = _extract_penalties(left, unc_map)
        nr, pr = _extract_penalties(right, unc_map)
        return MatMulExpression(nl, nr), pl + pr

    if isinstance(expr, BinaryOp):
        nl, pl = _extract_penalties(expr.left, unc_map)
        nr, pr = _extract_penalties(expr.right, unc_map)
        return BinaryOp(expr.op, nl, nr), pl + pr

    if isinstance(expr, UnaryOp):
        no, po = _extract_penalties(expr.operand, unc_map)
        return UnaryOp(expr.op, no), po

    if isinstance(expr, FunctionCall):
        new_args = []
        all_p: list = []
        for a in expr.args:
            na, pa = _extract_penalties(a, unc_map)
            new_args.append(na)
            all_p.extend(pa)
        return FunctionCall(expr.func_name, *new_args), all_p

    if isinstance(expr, IndexExpression):
        nb, pb = _extract_penalties(expr.base, unc_map)
        return IndexExpression(nb, expr.index), pb

    if isinstance(expr, SumExpression):
        no, po = _extract_penalties(expr.operand, unc_map)
        return SumExpression(no, expr.axis), po

    if isinstance(expr, SumOverExpression):
        new_terms = []
        all_p2: list = []
        for t in expr.terms:
            nt, pt = _extract_penalties(t, unc_map)
            new_terms.append(nt)
            all_p2.extend(pt)
        return SumOverExpression(new_terms), all_p2

    return expr, []


def _soc_penalty(decision_expr, unc: EllipsoidalUncertaintySet):
    """Build  ρ * ||Σ^{1/2} @ decision_expr||₂."""
    from discopt.modeling.core import BinaryOp, Constant, FunctionCall, MatMulExpression

    Sigma_sqrt = Constant(unc.Sigma_sqrt)
    scaled = MatMulExpression(Sigma_sqrt, decision_expr)
    # ||scaled||₂ = sqrt(scaled^T @ scaled)
    inner = FunctionCall("sum", FunctionCall("*", scaled, scaled))
    norm2 = FunctionCall("sqrt", inner)
    return BinaryOp("*", Constant(np.array(unc.rho)), norm2)


def _sum_expr(exprs: list):
    """Sum a list of expressions."""
    from discopt.modeling.core import BinaryOp

    result = exprs[0]
    for e in exprs[1:]:
        result = BinaryOp("+", result, e)
    return result
