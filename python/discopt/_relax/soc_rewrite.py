"""Squared-form second-order cone rewrite (issue #1933).

MIQCP modeling layers (Gurobi, CPLEX and SCIP conventions; pycse's MIP export,
Pyomo models handed over through ``.nl``) write a Lorentz cone in *squared* form:

    sum_i w_i * a_i(x)**2 + c  <=  k * t(x)**2,      t(x) >= 0

with ``a_i`` arbitrary, ``w_i > 0``, ``c >= 0``, ``k > 0`` and ``t`` affine. The
function ``q - k t**2`` is not convex, so the curvature walker (correctly) cannot
prove the row convex, although the *set* it defines is a cone. The equivalent
norm form is proven convex by the walker's ``sqrt(||A x + b||**2)`` rule.

This pass rewrites the row to

    sqrt(sum_i w_i * a_i(x)**2 + c)  <=  sqrt(k) * t(x)

Exactness
---------
Write ``q = sum_i w_i a_i**2 + c``. Every summand is nonnegative by construction
(a positive weight times a square, or a nonnegative constant), so ``q >= 0``
everywhere, and ``t >= 0`` on the declared box (an outward bound on the affine
form, :func:`~discopt._relax.convexity.patterns._affine_lower_bound`). For
``u, v >= 0``, ``u <= v  <=>  sqrt(u) <= sqrt(v)``; with ``u = q`` and
``v = k t**2`` that is ``q <= k t**2  <=>  sqrt(q) <= sqrt(k) * t``. The feasible
set is therefore unchanged on the box, which the solve enforces as constraints.
Variable bounds are only ever tightened later, so the ``t >= 0`` premise cannot
be invalidated downstream.

Both premises are structural proofs, not tolerances: a row with a linear term, a
negative constant, more than one negatively weighted square, a non-affine ``t`` or
a ``t`` not provably nonnegative is left untouched (the hyperbola
``x**2 - 1 <= s**2`` is one such nonconvex set). The rewrite is adopted only when
the curvature walker then proves the new row convex, so it never replaces a row
with one that is no easier to handle.

At the cone's apex (``q == 0``) the new row is not differentiable. The tape lowers
``sqrt`` of a sum of squares as a norm there, with the valid subgradient ``0``
(``_nl_expr_compiler._is_sum_of_squares``), so a solve that starts or ends at the
apex does not stop on an invalid number.

The rewritten row is a different *function*, so its multiplier is not the one the
caller's row has: ``Model.solve`` rescales the reported constraint duals back
(:func:`convert_squared_soc_duals`).

References
----------
Lobo, Vandenberghe, Boyd, Lebret (1998), "Applications of second-order cone
  programming," Linear Algebra Appl. -- the hyperbolic/squared-to-norm
  equivalence of Lorentz cone constraints.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy as _copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    FunctionCall,
    IndexExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
)

from .convexity.patterns import (
    _affine_lower_bound,
    _expr_struct_eq,
    _summed_vector_square_base,
    _total_scalar_variables,
)

_LOG = logging.getLogger(__name__)

#: Deterministic work allowance for one pass over a model (#1456: this runs
#: before branch and bound arms ``time_limit``). Units: one per expression node
#: the scan visits, plus, for a candidate cone, ``n_total`` per dense affine
#: extraction (its ``t``) and the row's term count per curvature unit charged to
#: its classification. Exhausting it stops the scan; the rows already rewritten
#: are kept, which stays exact because each row's rewrite is independent of the
#: others. A 5000-term cone over 5001 variables spends 6.0e4 units (measured), so the cap
#: leaves the scan of models orders of magnitude larger than any in-repo instance.
_SOC_SCAN_WORK_BUDGET = 5_000_000


class _ScanExhausted(Exception):
    """The pass spent :data:`_SOC_SCAN_WORK_BUDGET`. Private: raised by
    :func:`_sum_terms` / :func:`_charge` and caught only by the pass driver."""


def _charge(work: Any, units: int) -> None:
    if work is not None and not work.spend(units):
        raise _ScanExhausted


def _split_const_factors(expr: Expression) -> tuple[float, Optional[Expression]]:
    """``(k, core)`` with ``expr == k * core``, peeling scalar ``Constant`` factors.

    ``core`` is ``None`` when ``expr`` is a scalar constant. Unlike
    :func:`~discopt._relax.convexity.patterns._split_const_mul` this never folds a
    ``Parameter``: the rewritten row bakes the weights in as numbers, and a
    parameter's value may change between solves. A ``Parameter`` factor stays in
    ``core``, which is then not a square and the row is declined.
    """
    if isinstance(expr, Constant):
        v = np.asarray(expr.value)
        return (float(v), None) if v.ndim == 0 else (1.0, expr)
    if isinstance(expr, BinaryOp) and expr.op == "*":
        kl, cl = _split_const_factors(expr.left)
        kr, cr = _split_const_factors(expr.right)
        if cl is None and cr is None:
            return kl * kr, None
        if cl is None:
            return kl * kr, cr
        if cr is None:
            return kl * kr, cl
        if cl is expr.left and cr is expr.right:
            return kl * kr, expr
        return kl * kr, BinaryOp("*", cl, cr)
    return 1.0, expr


def _square_base(core: Expression) -> Optional[Expression]:
    """``b`` when ``core`` is ``b**2`` or ``b*b`` (structurally equal factors).

    Also accepts ``dm.sum(v**2)`` / ``dm.sum(v*v)`` (a full reduction), returning
    the summed node itself: a sum of squares is nonnegative too. Returns ``None``
    for anything else.
    """
    if isinstance(core, SumExpression):
        return core if _summed_vector_square_base(core) is not None else None
    if not isinstance(core, BinaryOp):
        return None
    if core.op == "**" and isinstance(core.right, Constant):
        p = np.asarray(core.right.value)
        if p.ndim == 0 and float(p) == 2.0:
            return core.left
        return None
    if core.op == "*" and _expr_struct_eq(core.left, core.right):
        return core.left
    return None


def _sum_terms(
    expr: Expression, sign: float, work: Any = None
) -> Iterator[tuple[float, Expression]]:
    """``(scale, term)`` over the ``+``/``-``/``neg``/n-ary-sum tree at ``expr``.

    The same reading as :func:`~discopt._relax.convexity.patterns._flatten_sum_terms`,
    but iterative and lazy: this scan runs on every row of every solve, ``.nl``
    rows carry ``+`` chains thousands deep (#1520), and a row stops being a
    candidate at its first term that is not a square.
    """
    stack = [(expr, sign)]
    while stack:
        node, scale = stack.pop()
        _charge(work, 1)
        if isinstance(node, BinaryOp) and node.op in ("+", "-"):
            stack.append((node.right, scale if node.op == "+" else -scale))
            stack.append((node.left, scale))
        elif isinstance(node, UnaryOp) and node.op == "neg":
            stack.append((node.operand, -scale))
        elif isinstance(node, SumOverExpression):
            stack.extend((t, scale) for t in reversed(node.terms))
        else:
            yield scale, node


def _contains_parameter(expr: Expression) -> bool:
    """Does ``expr`` reference a ``Parameter`` (whose value may change between solves)?"""
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Parameter):
            return True
        if isinstance(node, BinaryOp):
            stack.extend((node.left, node.right))
        elif isinstance(node, UnaryOp):
            stack.append(node.operand)
        elif isinstance(node, IndexExpression):
            stack.append(node.base)
        elif isinstance(node, SumExpression):
            stack.append(node.operand)
        elif isinstance(node, SumOverExpression):
            stack.extend(node.terms)
    return False


@dataclass(frozen=True)
class SquaredSocRow:
    """One rewritten row, with what :func:`convert_squared_soc_duals` needs.

    ``t(x) = t_coeffs @ x_flat + t_const`` over the model's variables in
    declaration order; ``sign`` is ``+1`` for a ``q - k t**2 <= 0`` row and ``-1``
    for ``k t**2 - q >= 0``.
    """

    index: int
    key: str
    sign: float
    sqrt_k: float
    t_coeffs: np.ndarray
    t_const: float


def _squared_soc_row(
    c: Constraint, model: Model, work: Any = None
) -> Optional[tuple[Constraint, float, float, Expression]]:
    """``(row, sign, sqrt_k, t)`` for the norm form of ``c``, or ``None``."""
    if c.sense == "<=":
        sign = 1.0
    elif c.sense == ">=":
        sign = -1.0
    else:
        return None
    # A scalar row only: the dual conversion (and ``t``'s scalar check) read one
    # multiplier per row. A vector row ``x**2 <= t**2`` is not proven convex
    # today, but nothing here should depend on that.
    if getattr(c.body, "shape", ()) not in ((), (1,)):
        return None
    positive: list[tuple[float, Expression]] = []
    const = 0.0
    neg: Optional[tuple[float, Expression]] = None
    for scale, term in _sum_terms(c.body, sign, work):
        k, core = _split_const_factors(term)
        w = scale * k
        if core is None:
            const += w
            continue
        base = _square_base(core)
        if base is None or w == 0.0 or not math.isfinite(w):
            return None  # a linear or otherwise non-square term: not this form
        if w > 0.0:
            positive.append((w, core))
            continue
        # A negatively weighted square: the cone's ``-k t**2``. Exactly one, and
        # its base must be a scalar affine form that is nonnegative on the box,
        # with no parameter whose later value could make it negative.
        if neg is not None or isinstance(core, SumExpression):
            return None
        if getattr(base, "shape", None) != () or _contains_parameter(base):
            return None
        _charge(work, _total_scalar_variables(model))  # dense affine extraction
        lb = _affine_lower_bound(base, model)
        if lb is None or lb < 0.0:
            return None
        neg = (-w, base)
    # The cone needs its ``-k t**2`` and at least one square on the other side
    # (``c <= k t**2`` alone is the linear ``sqrt(c) <= sqrt(k) t``, not a cone).
    if neg is None or not positive or not math.isfinite(const) or const < 0.0:
        return None

    q: Expression = positive[0][1] if positive[0][0] == 1.0 else positive[0][0] * positive[0][1]
    for w, core in positive[1:]:
        q = q + (core if w == 1.0 else w * core)
    if const > 0.0:
        q = q + const
    k, t = neg
    sqrt_k = math.sqrt(k)
    rhs: Expression = t if k == 1.0 else sqrt_k * t
    row = Constraint(body=FunctionCall("sqrt", q) - rhs, sense="<=", rhs=0.0, name=c.name)
    return row, sign, sqrt_k, t


def _term_count(body: Expression) -> int:
    """Terms of the rewritten row: the ``sqrt`` argument's summands, plus ``t``."""
    count = 1
    stack = [body]
    while stack:
        node = stack.pop()
        if isinstance(node, BinaryOp) and node.op in ("+", "-"):
            stack.extend((node.left, node.right))
        elif isinstance(node, FunctionCall):
            stack.extend(node.args)
        else:
            count += 1
    return count


def rewrite_squared_soc_rows(
    model: Model, deadline: Optional[Callable[[], bool]] = None
) -> tuple[Model, list[SquaredSocRow]]:
    """Rewrite squared-form second-order cone rows to their norm form.

    Returns ``(model, rows)``. When no row matches (see the module docstring), the
    caller's model is returned unchanged with ``rows == []``. Otherwise a shallow
    copy is returned with a fresh ``_constraints`` list in which each rewritten
    row is a new ``Constraint`` (same name, same position); the caller's objects
    are never mutated. A row is replaced only when ``t`` depends on a variable
    (otherwise ``q <= k c**2`` is a convex quadratic as written) and the
    rewritten row is proven convex.

    Two bounds (#1456): the deterministic :data:`_SOC_SCAN_WORK_BUDGET`, and
    ``deadline``, the solve's ``PresolveDeadline.abandon_hook`` -- checked once
    per row, the outer loop. When it fires the pass discards everything it
    rewrote and returns ``(model, [])``, the "found nothing" outcome.

    The curvature walks recurse one frame per node, so they run with the same
    size-scaled recursion headroom as :func:`~discopt._relax.convexity.classify_model`.
    """
    from .convexity.rules import _recursion_headroom_need, _run_with_deep_recursion

    return _run_with_deep_recursion(
        lambda: _rewrite_squared_soc_rows_inner(model, deadline),
        depth_need=_recursion_headroom_need(model),
    )


def _rewrite_squared_soc_rows_inner(
    model: Model, deadline: Optional[Callable[[], bool]]
) -> tuple[Model, list[SquaredSocRow]]:
    from .convexity import classify_constraint
    from .objective_epigraph import WorkCounter
    from .problem_classifier import _extract_linear_coefficients

    new_constraints: Optional[list] = None
    rows: list[SquaredSocRow] = []
    cache: dict = {}
    n_total = _total_scalar_variables(model)
    work = WorkCounter(_SOC_SCAN_WORK_BUDGET)
    for i, c in enumerate(model._constraints):
        if deadline is not None and deadline():
            return model, []  # time limit spent: abandon wholesale
        if not isinstance(c, Constraint):
            continue
        try:
            found = _squared_soc_row(c, model, work)
            if found is None:
                continue
            new, sign, sqrt_k, t = found
            # The ``t`` extraction below, then the classification walk.
            _charge(work, n_total)
            _charge(work, 8 * _term_count(new.body))
        except _ScanExhausted:
            _LOG.debug(
                "squared-SOC scan stopped at row %d of %d: work budget spent",
                i,
                len(model._constraints),
            )
            break
        # ``t`` passed ``_affine_lower_bound``, which ran this same extraction.
        coeffs, t_const = _extract_linear_coefficients(t, model, n_total)
        if not np.any(np.asarray(coeffs) != 0.0):
            # A constant ``t``: ``q <= k c**2`` is a convex quadratic row as
            # written (a ball), better left to the quadratic routes. This is the
            # structural stand-in for classifying the original row, which costs
            # an eigendecomposition on the support (9 s on a 5000-term cone).
            continue
        if not classify_constraint(new, model, cache):
            continue
        if new_constraints is None:
            new_constraints = list(model._constraints)
        new_constraints[i] = new
        rows.append(
            SquaredSocRow(
                index=i,
                key=c.name if c.name else f"c{i}",
                sign=sign,
                sqrt_k=sqrt_k,
                t_coeffs=np.asarray(coeffs, dtype=np.float64).ravel(),
                t_const=float(t_const),
            )
        )
    if new_constraints is None:
        return model, []
    new_model = _copy.copy(model)
    new_model._constraints = new_constraints
    # A memoized verdict belongs to the rows it was computed on.
    setattr(new_model, "_convexity_classification_cache", None)
    return new_model, rows


def rewrite_squared_soc(
    model: Model, deadline: Optional[Callable[[], bool]] = None
) -> tuple[Model, bool]:
    """:func:`rewrite_squared_soc_rows`, returning ``(model, changed)``."""
    new_model, rows = rewrite_squared_soc_rows(model, deadline)
    return new_model, bool(rows)


# The rows a solve rewrote, keyed by the model it was called with, so
# ``Model.solve`` can report constraint duals for the rows the caller wrote.
_SOC_ROW_SINK: contextvars.ContextVar[Optional[dict[int, list[SquaredSocRow]]]] = (
    contextvars.ContextVar("discopt_squared_soc_rows", default=None)
)


@contextlib.contextmanager
def collect_squared_soc_rows() -> Iterator[dict[int, list[SquaredSocRow]]]:
    """Within this block, record every solve's rewritten rows by ``id(model)``."""
    sink: dict[int, list[SquaredSocRow]] = {}
    token = _SOC_ROW_SINK.set(sink)
    try:
        yield sink
    finally:
        _SOC_ROW_SINK.reset(token)


def record_squared_soc_rows(model: Model, rows: list[SquaredSocRow]) -> None:
    """Record ``rows`` (rewritten on ``model``) into the active collector, if any."""
    sink = _SOC_ROW_SINK.get()
    if sink is not None and rows:
        sink.setdefault(id(model), []).extend(rows)


def convert_squared_soc_duals(result: Any, model: Model, rows: list[SquaredSocRow]) -> None:
    """Rescale ``result.constraint_duals`` from the norm rows to the rows the caller wrote.

    With ``g = sign * (q - k t**2)`` the caller's body and ``h = sqrt(q) - sqrt(k) t``
    the solved one, ``q - k t**2 = h * (sqrt(q) + sqrt(k) t)``, so where ``h = 0``
    ``grad g = sign * 2 sqrt(k) t * grad h`` and the multiplier that gives the same
    stationarity term is ``lambda_g = sign * lambda_h / (2 sqrt(k) t)``. An inactive
    row has a zero multiplier in both forms. At the cone's apex (``t = 0``) the
    caller's row has ``grad g = 0`` and no finite multiplier: a nonzero ``lambda_h``
    there cannot be translated, so the constraint duals are withheld (set to
    ``None``) rather than reported against the wrong row.
    """
    duals = getattr(result, "constraint_duals", None)
    if not duals or not rows:
        return
    x = getattr(result, "x", None)
    if not x:
        result.constraint_duals = None
        return
    x_flat = np.concatenate(
        [np.asarray(x[v.name], dtype=np.float64).ravel() for v in model._variables]
    )
    for row in rows:
        lam = duals.get(row.key)
        if lam is None:
            continue
        lam_f = float(np.asarray(lam, dtype=np.float64).reshape(()))
        if lam_f == 0.0:
            continue
        t_val = float(row.t_coeffs @ x_flat) + row.t_const
        if not t_val > 0.0:
            _LOG.warning(
                "Duals withheld (#1933): row %r was solved in its norm form and is "
                "active at the cone's apex (t = %g), where the squared form you "
                "wrote has no finite multiplier.",
                row.key,
                t_val,
            )
            result.constraint_duals = None
            return
        duals[row.key] = np.asarray(row.sign * lam_f / (2.0 * row.sqrt_k * t_val)).reshape(
            np.shape(lam)
        )


__all__ = [
    "SquaredSocRow",
    "collect_squared_soc_rows",
    "convert_squared_soc_duals",
    "record_squared_soc_rows",
    "rewrite_squared_soc",
    "rewrite_squared_soc_rows",
]
