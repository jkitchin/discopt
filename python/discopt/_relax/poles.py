"""Poles of the objective inside the declared box (#1493).

``min 1/x`` over ``x in [-5, 5]`` has no dual bound on any box that contains
``x = 0``: the objective is unbounded below as ``x -> 0-``. The relaxation layer
correctly refuses to bound it -- the ratio envelope keeps an infinite interval
floor -- so the spatial B&B stops with a feasible point and no bound. That exit
is honest under :mod:`discopt.status` (state 3, "feasible, no bound"; ``unbounded``
means *certified* unboundedness, which a relaxation cannot prove at a pole), but
the diagnostic it produced was not: it blamed "a nonlinear term with no envelope"
and advised an epigraph reformulation, which cannot help at a pole. This module
finds the actual cause so the solve can say it.

Only *detection* lives here. It changes no bound, no box and no status.

A denominator whose range over the variable box holds 0 may still be kept away
from 0 by the constraints -- #1616 A-04: ``n_i / sum(n)`` with ``n >= 0`` and a
balance ``n0 + n2 + n4 == 1``, so ``sum(n) >= 1`` at every feasible point. The
relaxation bounds each term on the box, so it still cannot bound the objective,
but "a pole inside the box" is then the wrong account of why. For an affine
denominator :func:`objective_poles` therefore also reports the range an LP over
the model's linear rows implies, and :func:`describe_poles` says which it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    SumExpression,
    SumOverExpression,
    UnaryOp,
)


@dataclass(frozen=True)
class ObjectivePole:
    """One singular objective subterm: ``term`` divides by ``denominator``."""

    term: str
    denominator: str
    lo: float
    hi: float
    #: Range of an affine ``denominator`` over the LP relaxation of the model's
    #: linear rows and the box (``None`` when it was not computed). Diagnostic
    #: text only -- never used as a bound.
    implied_lo: Optional[float] = None
    implied_hi: Optional[float] = None

    @property
    def excluded_by_constraints(self) -> bool:
        """Whether the linear rows keep the denominator away from 0."""
        tol = 1e-7
        return (self.implied_lo is not None and self.implied_lo > tol) or (
            self.implied_hi is not None and self.implied_hi < -tol
        )


def _children(expr: Expression) -> list[Expression]:
    if isinstance(expr, BinaryOp):
        return [expr.left, expr.right]
    if isinstance(expr, UnaryOp):
        return [expr.operand]
    if isinstance(expr, FunctionCall):
        return list(expr.args)
    if isinstance(expr, IndexExpression):
        return [expr.base]
    if isinstance(expr, SumExpression):
        return [expr.operand]
    if isinstance(expr, SumOverExpression):
        return list(expr.terms)
    if isinstance(expr, MatMulExpression):
        return [expr.left, expr.right]
    return []


def _denominator(expr: Expression) -> Optional[Expression]:
    """The expression whose zero is a pole of ``expr``, or ``None``."""
    if not isinstance(expr, BinaryOp):
        return None
    if expr.op == "/" and not isinstance(expr.right, Constant):
        return expr.right
    if expr.op == "**" and isinstance(expr.right, Constant):
        p = np.asarray(expr.right.value)
        if p.ndim == 0 and float(p) < 0.0 and not isinstance(expr.left, Constant):
            return expr.left
    return None


def _walk(expr: Expression) -> Iterator[tuple[Expression, Expression]]:
    seen: set[int] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        den = _denominator(node)
        if den is not None:
            yield node, den
        stack.extend(_children(node))


#: Above this many rows or flat variables the LP range is not attempted: the
#: diagnostic runs after a solve and must not cost a second one.
_LP_RANGE_MAX_SIZE = 20_000


class _LinearRows:
    """The model's scalar linear rows as ``A_ub x <= b_ub``, ``A_eq x == b_eq``.

    A row that is not linear, or not scalar, is left out. Dropping a row only
    enlarges the polyhedron, so every range computed over it still contains the
    denominator's true range over the feasible set.
    """

    def __init__(self, model: Model):
        from discopt._relax.problem_classifier import (
            _extract_linear_coefficients_sparse,
            _NotLinearError,
        )
        from discopt.modeling.core import Constraint

        lbs, ubs = [], []
        for v in model._variables:
            lbs.append(np.broadcast_to(np.asarray(v.lb, dtype=np.float64), v.shape).ravel())
            ubs.append(np.broadcast_to(np.asarray(v.ub, dtype=np.float64), v.shape).ravel())
        self.lb = np.concatenate(lbs) if lbs else np.zeros(0)
        self.ub = np.concatenate(ubs) if ubs else np.zeros(0)
        self.n = int(self.lb.size)
        ub_rows: list[dict[int, float]] = []
        ub_rhs: list[float] = []
        eq_rows: list[dict[int, float]] = []
        eq_rhs: list[float] = []
        for c in model._constraints:
            if not isinstance(c, Constraint) or c.sense not in ("<=", ">=", "=="):
                continue
            try:
                terms, const = _extract_linear_coefficients_sparse(c.body, model, self.n)
            except _NotLinearError:
                continue
            if c.sense == "==":
                eq_rows.append(terms)
                eq_rhs.append(-const)
            else:
                sgn = 1.0 if c.sense == "<=" else -1.0
                ub_rows.append({i: sgn * a for i, a in terms.items()})
                ub_rhs.append(-sgn * const)
        self.A_ub, self.b_ub = self._csr(ub_rows), np.asarray(ub_rhs, dtype=np.float64)
        self.A_eq, self.b_eq = self._csr(eq_rows), np.asarray(eq_rhs, dtype=np.float64)

    def _csr(self, rows: list[dict[int, float]]):
        import scipy.sparse as sp

        if not rows:
            return None
        data: list[float] = []
        idx: list[int] = []
        ptr = [0]
        for r in rows:
            idx.extend(r.keys())
            data.extend(r.values())
            ptr.append(len(idx))
        return sp.csr_matrix((data, idx, ptr), shape=(len(rows), self.n))

    def range(self, terms: dict[int, float], const: float) -> tuple[float, float]:
        """``(min, max)`` of ``terms . x + const`` over the rows and the box.

        An endpoint the LP does not solve to optimality is reported as the
        corresponding infinity, so the result never claims more than was shown.
        """
        from scipy.optimize import linprog

        c = np.zeros(self.n)
        for i, a in terms.items():
            c[i] = a
        big = 1e20
        bounds = [
            (None if lo <= -big else lo, None if hi >= big else hi)
            for lo, hi in zip(self.lb, self.ub)
        ]
        ends = []
        for sgn in (1.0, -1.0):
            res = linprog(
                sgn * c,
                A_ub=self.A_ub,
                b_ub=self.b_ub if self.A_ub is not None else None,
                A_eq=self.A_eq,
                b_eq=self.b_eq if self.A_eq is not None else None,
                bounds=bounds,
                method="highs",
            )
            ends.append(sgn * float(res.fun) + const if res.status == 0 else -sgn * np.inf)
        return ends[0], ends[1]


def _implied_range(den: Expression, model: Model, rows: list) -> Optional[tuple[float, float]]:
    """LP range of an affine ``den`` (see :class:`_LinearRows`), else ``None``."""
    from discopt._relax.problem_classifier import (
        _extract_linear_coefficients_sparse,
        _NotLinearError,
    )

    if not rows:
        n = sum(int(v.size) for v in model._variables)
        if max(n, len(model._constraints)) > _LP_RANGE_MAX_SIZE:
            return None
        rows.append(_LinearRows(model))
    lin = rows[0]
    try:
        terms, const = _extract_linear_coefficients_sparse(den, model, lin.n)
    except _NotLinearError:
        return None
    lo, hi = lin.range(terms, const)
    return float(lo), float(hi)


def objective_poles(model: Model, limit: int = 3) -> list[ObjectivePole]:
    """Objective subterms whose denominator's range over the DECLARED box holds 0.

    Uses the sound interval enclosure of each denominator; a denominator whose
    enclosure cannot be computed as a scalar is skipped (never guessed). For an
    affine denominator the range implied by the linear rows is attached as
    ``implied_lo``/``implied_hi`` (#1616 A-04).
    """
    from discopt._relax.convexity.interval_eval import evaluate_interval

    obj = getattr(model, "_objective", None)
    if obj is None or getattr(obj, "expression", None) is None:
        return []
    out: list[ObjectivePole] = []
    rows: list = []
    implied: dict[int, Optional[tuple[float, float]]] = {}
    for term, den in _walk(obj.expression):
        enc = evaluate_interval(den, model)
        lo_arr = np.asarray(enc.lo, dtype=np.float64)
        hi_arr = np.asarray(enc.hi, dtype=np.float64)
        if lo_arr.size != 1 or hi_arr.size != 1:
            continue
        lo = float(lo_arr.reshape(()))
        hi = float(hi_arr.reshape(()))
        if lo <= 0.0 <= hi:
            if id(den) not in implied:
                implied[id(den)] = _implied_range(den, model, rows)
            rng = implied[id(den)]
            ilo, ihi = rng if rng is not None else (None, None)
            out.append(ObjectivePole(repr(term), repr(den), lo, hi, ilo, ihi))
            if len(out) >= limit:
                break
    return out


def describe_poles(poles: list[ObjectivePole]) -> str:
    """One human-readable clause per pole, for the no-bound diagnostic."""

    # Terms sharing a denominator (``n_i / sum(n)``) are reported together.
    groups: dict[str, list[ObjectivePole]] = {}
    for p in poles:
        groups.setdefault(p.denominator, []).append(p)

    def one(ps: list[ObjectivePole]) -> str:
        p = ps[0]
        # Distinct terms only: the twelve residuals of a kinetic fit each hold
        # their own ``k1 / (k2 - k1)`` node with the same text (#1680).
        names = list(dict.fromkeys(q.term for q in ps))
        terms = ", ".join(f"`{t}`" for t in names)
        head = (
            f"{terms} {'divide' if len(names) > 1 else 'divides'} by `{p.denominator}`, "
            f"whose range over the variable box [{p.lo:.6g}, {p.hi:.6g}] contains 0"
        )
        if p.excluded_by_constraints:
            return (
                f"{head}, but the linear constraints keep it in "
                f"[{p.implied_lo:.6g}, {p.implied_hi:.6g}]"
            )
        return head

    return "; ".join(one(ps) for ps in groups.values())
