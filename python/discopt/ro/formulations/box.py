"""Box-uncertainty robust reformulation.

Mathematical Background
-----------------------
For a constraint body expression g(x, p) ≤ 0 with box uncertainty
{p : |p_j - p̄_j| ≤ δ_j}, the robust counterpart requires:

    max_{p∈U} g(x, p) ≤ 0

When g is affine in p, the worst-case maximisation has a closed form that
depends on the sign of ∂g/∂p_j:

* ∂g/∂p_j > 0  →  worst-case p_j = p̄_j + δ_j  (upper bound)
* ∂g/∂p_j < 0  →  worst-case p_j = p̄_j - δ_j  (lower bound)

This module performs *sign-tracking* while traversing the expression tree to
determine the correct bound for each parameter occurrence.  The sign starts
at +1 at the tree root and flips through ``-`` (subtraction) and unary
negation nodes.  Multiplication by a negative constant also flips the sign.

For objectives the sign convention is adapted to the sense:
- MINIMIZE f(p): worst case is the *largest* f, so parameter terms appear
  with their natural sign.
- MAXIMIZE f(p): worst case is the *smallest* f, so signs are flipped.

Only constraints of the form ``body ≤ 0`` are supported (which is the
normalized form used internally).
"""

from __future__ import annotations

import numpy as np

from discopt.ro.uncertainty import BoxUncertaintySet


class BoxRobustFormulation:
    """Apply box-uncertainty robust reformulation to a model.

    Parameters
    ----------
    model : discopt.Model
        The model to robustify (modified in-place).
    uncertainty_sets : list[BoxUncertaintySet]
        Uncertainty sets, one per uncertain parameter.
    prefix : str
        Name prefix for any auxiliary variables / constraints added.
    """

    def __init__(
        self, model, uncertainty_sets: list[BoxUncertaintySet], prefix: str = "ro"
    ) -> None:
        self._model = model
        self._uncertainty_sets = uncertainty_sets
        self._prefix = prefix

    def build(self) -> None:
        """Robustify the model in-place.

        For each constraint replaces uncertain parameters with worst-case
        values determined by sign-tracking through the expression tree.
        For the objective, applies the appropriate worst-case substitution
        based on the optimisation sense.
        """
        unc_map = {u.parameter.name: u for u in self._uncertainty_sets}
        m = self._model

        # ── Robustify constraints ──────────────────────────────────────────────
        # Constraint stored as body ≤ 0; worst-case ≡ maximise body over U.
        new_constraints = []
        for con in m._constraints:
            new_expr = _worst_case(con.body, unc_map, maximize=True, sign=+1)
            from discopt.modeling.core import Constraint

            new_constraints.append(
                Constraint(body=new_expr, sense=con.sense, rhs=con.rhs, name=con.name)
            )
        m._constraints = new_constraints

        # ── Robustify objective ────────────────────────────────────────────────
        obj = m._objective
        if obj is None:
            return
        from discopt.modeling.core import ObjectiveSense, Objective

        # For MINIMIZE: worst case is maximum objective → parameters with positive
        # contribution use upper bound, negative use lower.
        # For MAXIMIZE: worst case is minimum objective → flip convention.
        maximize_obj_wc = obj.sense == ObjectiveSense.MINIMIZE
        new_expr = _worst_case(obj.expression, unc_map, maximize=maximize_obj_wc, sign=+1)
        m._objective = Objective(expression=new_expr, sense=obj.sense)


# ─────────────────────────────────────────────────────────────────────────────
# Sign-tracking worst-case substitution
# ─────────────────────────────────────────────────────────────────────────────


def _worst_case(expr, unc_map: dict, maximize: bool, sign: int):
    """Recursively replace uncertain Parameters with worst-case constants.

    Parameters
    ----------
    expr : Expression
        Sub-expression to rewrite.
    unc_map : dict[str, BoxUncertaintySet]
        Maps parameter names to their uncertainty sets.
    maximize : bool
        Whether we are maximising (True) or minimising (False) the root
        expression.
    sign : int
        Accumulated sign (+1 or -1) of the current sub-expression relative
        to the root.  Flips at subtraction and unary negation.
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

    if isinstance(expr, Parameter) and expr.name in unc_map:
        unc = unc_map[expr.name]
        # Effective direction: if (maximize XOR sign<0), use upper bound.
        use_upper = maximize ^ (sign < 0)
        wc_value = unc.upper if use_upper else unc.lower
        return Constant(wc_value)

    if isinstance(expr, Constant):
        return expr

    if isinstance(expr, BinaryOp):
        if expr.op == "+":
            nl = _worst_case(expr.left, unc_map, maximize, sign)
            nr = _worst_case(expr.right, unc_map, maximize, sign)
        elif expr.op == "-":
            nl = _worst_case(expr.left, unc_map, maximize, sign)
            nr = _worst_case(expr.right, unc_map, maximize, sign * -1)
        elif expr.op == "*":
            # Propagate sign through multiplication only when one side is a
            # non-negative constant (the most common case: scalar * variable).
            if isinstance(expr.left, Constant):
                left_sign = int(np.sign(np.sum(expr.left.value)))
                nl = expr.left
                nr = _worst_case(expr.right, unc_map, maximize, sign * (left_sign or 1))
            elif isinstance(expr.right, Constant):
                right_sign = int(np.sign(np.sum(expr.right.value)))
                nl = _worst_case(expr.left, unc_map, maximize, sign * (right_sign or 1))
                nr = expr.right
            else:
                # Bilinear: cannot determine sign cheaply; use nominally safe
                # direction (positive sign assumption).
                nl = _worst_case(expr.left, unc_map, maximize, sign)
                nr = _worst_case(expr.right, unc_map, maximize, sign)
        else:
            # Division, power — propagate sign through without flip.
            nl = _worst_case(expr.left, unc_map, maximize, sign)
            nr = _worst_case(expr.right, unc_map, maximize, sign)
        return BinaryOp(expr.op, nl, nr)

    if isinstance(expr, UnaryOp):
        child_sign = sign * -1 if expr.op == "neg" else sign
        return UnaryOp(expr.op, _worst_case(expr.operand, unc_map, maximize, child_sign))

    if isinstance(expr, FunctionCall):
        new_args = [_worst_case(a, unc_map, maximize, sign) for a in expr.args]
        return FunctionCall(expr.func_name, *new_args)

    if isinstance(expr, MatMulExpression):
        nl = _worst_case(expr.left, unc_map, maximize, sign)
        nr = _worst_case(expr.right, unc_map, maximize, sign)
        return MatMulExpression(nl, nr)

    if isinstance(expr, IndexExpression):
        nb = _worst_case(expr.base, unc_map, maximize, sign)
        return IndexExpression(nb, expr.index)

    if isinstance(expr, SumExpression):
        new_operand = _worst_case(expr.operand, unc_map, maximize, sign)
        return SumExpression(new_operand, expr.axis)

    if isinstance(expr, SumOverExpression):
        new_terms = [_worst_case(t, unc_map, maximize, sign) for t in expr.terms]
        return SumOverExpression(new_terms)

    # Variable, unknown node: return as-is.
    return expr
