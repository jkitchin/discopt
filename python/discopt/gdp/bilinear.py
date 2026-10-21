"""``gdp.bilinear``: binary-times-variable products as disjunctions (#1775 G2.9).

Pyomo's ``gdp.bilinear``. Each product ``b * x`` of a binary scalar ``b`` and a
bounded non-binary scalar ``x`` in the objective or in a top-level row becomes a
new variable ``v`` with the disjunction

    [b == 1, v == x]  v  [b == 0, v == 0],

and ``v`` in ``[min(0, lb_x), max(0, ub_x)]``. It is exact: at ``b = 1`` the
first disjunct is the only one that holds (``v = x = b * x``), at ``b = 0`` the
second (``v = 0 = b * x``). A GDP lowering must follow; under hull this gives the
convex hull of the product's graph over the box.

A pair ``(b, x)`` that occurs more than once shares one ``v``. Products inside a
disjunct or an indicator row are left as they are, as is any node this pass does
not walk (they stay exact products). Refused, loudly: an ``x`` with an infinite or
defaulted bound (``v`` would have none). Atomic: a refusal leaves the model
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from discopt.constants import DEFAULT_VARIABLE_BOUND
from discopt.modeling.core import (
    BinaryOp,
    Constraint,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    Objective,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
    _DisjunctiveConstraint,
)


@dataclass
class BilinearReport:
    """Products replaced, the variables and disjunctions added."""

    products: int = 0
    new_vars: int = 0


def _leaf(e: Any) -> Optional[tuple[Variable, Expression]]:
    """``(owning variable, leaf)`` when ``e`` names one scalar of a variable."""
    if isinstance(e, Variable) and e.shape == ():
        return e, e
    if isinstance(e, IndexExpression) and isinstance(e.base, Variable):
        if tuple(getattr(e, "shape", ()) or ()) == ():
            return e.base, e
    return None


def _bounds(var: Variable, leaf: Expression) -> tuple[float, float]:
    lb = np.broadcast_to(np.asarray(var.lb, dtype=np.float64), var.shape)
    ub = np.broadcast_to(np.asarray(var.ub, dtype=np.float64), var.shape)
    if leaf is var:
        return float(lb), float(ub)
    assert isinstance(leaf, IndexExpression)
    return float(lb[leaf.index]), float(ub[leaf.index])


class _Plan:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.vars: list[Variable] = []
        self.disjunctions: list[_DisjunctiveConstraint] = []
        self.by_pair: dict[tuple[str, str], Variable] = {}
        self._reserved: set[str] = set()
        self.report = BilinearReport()

    def _fresh_name(self, base: str) -> str:
        name, k = base, 2
        while name in self.model._names or name in self._reserved:
            name, k = f"{base}_{k}", k + 1
        self._reserved.add(name)
        return name

    def product(self, b: Expression, x: Expression, xvar: Variable) -> Variable:
        key = (str(b), str(x))
        self.report.products += 1
        if key in self.by_pair:
            return self.by_pair[key]
        lo, hi = _bounds(xvar, x)
        if not all(np.isfinite(t) and abs(t) < DEFAULT_VARIABLE_BOUND for t in (lo, hi)):
            raise ValueError(
                f"gdp.bilinear: {x} in the product {b} * {x} has bounds [{lo:g}, {hi:g}]; "
                "v = b * x needs a finite box. Bound the variable."
            )
        v = Variable(
            self._fresh_name("_gdp_bilinear_v"),
            VarType.CONTINUOUS,
            (),
            min(0.0, lo),
            max(0.0, hi),
            self.model,
        )
        v._index = len(self.model._variables) + len(self.vars)
        self.vars.append(v)
        self.report.new_vars += 1
        on: list[Any] = [b >= 1, v - x == 0]
        off: list[Any] = [b <= 0, v == 0]
        self.disjunctions.append(
            _DisjunctiveConstraint(disjuncts=[on, off], name=f"{v.name}_d", _model=self.model)
        )
        self.by_pair[key] = v
        return v


def _rewrite(e: Any, plan: _Plan) -> Any:
    if isinstance(e, BinaryOp):
        if e.op == "*":
            lhs, rhs = _leaf(e.left), _leaf(e.right)
            for bin_side, other in ((lhs, rhs), (rhs, lhs)):
                if (
                    bin_side is not None
                    and other is not None
                    and bin_side[0].var_type is VarType.BINARY
                    and other[0].var_type is not VarType.BINARY
                ):
                    return plan.product(bin_side[1], other[1], other[0])
        left, right = _rewrite(e.left, plan), _rewrite(e.right, plan)
        if left is e.left and right is e.right:
            return e
        return BinaryOp(e.op, left, right)
    if isinstance(e, UnaryOp):
        operand = _rewrite(e.operand, plan)
        return e if operand is e.operand else UnaryOp(e.op, operand)
    if isinstance(e, FunctionCall):
        args = [_rewrite(a, plan) for a in e.args]
        return e if all(a is b for a, b in zip(args, e.args)) else FunctionCall(e.func_name, *args)
    if isinstance(e, SumOverExpression):
        terms = [_rewrite(t, plan) for t in e.terms]
        return e if all(a is b for a, b in zip(terms, e.terms)) else SumOverExpression(terms)
    if isinstance(e, SumExpression):
        operand = _rewrite(e.operand, plan)
        return e if operand is e.operand else SumExpression(operand, e.axis)
    if isinstance(e, MatMulExpression):
        left, right = _rewrite(e.left, plan), _rewrite(e.right, plan)
        if left is e.left and right is e.right:
            return e
        return MatMulExpression(left, right)
    return e  # a leaf, or a node this pass does not walk: left exact


def bilinear(model: Model) -> BilinearReport:
    """Apply ``gdp.bilinear`` to ``model`` in place; see the module docstring."""
    plan = _Plan(model)
    # Phase 1: plan.
    rows: list[Any] = []
    for c in model._constraints:
        if type(c) is Constraint:
            body = _rewrite(c.body, plan)
            rows.append(
                c
                if body is c.body
                else Constraint(body=body, sense=c.sense, rhs=c.rhs, name=c.name)
            )
        else:
            rows.append(c)
    objective = model._objective
    if objective is not None:
        expr = _rewrite(objective.expression, plan)
        if expr is not objective.expression:
            objective = Objective(expression=expr, sense=objective.sense)
    # Phase 2: commit.
    for v in plan.vars:
        assert v._index == len(model._variables)
        model._register_generated_variable(v)
    model._constraints = rows + list(plan.disjunctions)
    model._objective = objective
    return plan.report


def _bilinear_in_place(model: Model) -> None:
    """The registry's in-place entry point (``transformations``)."""
    bilinear(model)


__all__ = ["BilinearReport", "bilinear"]
