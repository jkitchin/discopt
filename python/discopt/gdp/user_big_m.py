"""User big-M values on ``either_or`` disjunctions, and how each is validated (#1775 G2.6).

``Model.either_or(..., big_m=...)`` lets the user supply the M of a row instead of
the computed one (Pyomo's ``bigM``). A user M that is too small turns the big-M
row of an *unselected* disjunct into a real constraint and cuts feasible points:
the solver could then certify a wrong optimum, or a false infeasibility. So no
user M is taken on trust. Each side a row needs (``<=``: ``body <= M``; ``>=``:
``-body <= M``; ``==``: both) gets one of three verdicts:

* **certified** -- ``M`` is at least the interval upper bound of that side over
  the variable box, or at least a *rigorous* LP upper bound (the
  Neumaier-Shcherbina bound of :func:`discopt._relax.gdp_reformulate._safe_lp_max`)
  over the model's linear rows and the box -- conditioned, for a top-level
  disjunction, on each other disjunct's linear rows in turn, since a row of
  disjunct ``k`` is relaxed only when another disjunct is selected. Dropping
  nonlinear rows only enlarges an LP region, so the bound stays valid.
* **refused** (``ValueError``) -- an LP maximiser ``x*`` has the side's value
  above ``M`` and :func:`discopt.validation.feasibility.verify_point` accepts
  ``x*`` for the declared model. Disjunct ``k`` does not hold at ``x*`` (its row is
  violated, as ``M >= 0``), so ``x*`` is feasible with ``k`` unselected, and the
  user's row would cut it: a proof that ``M`` is too small.
* **uncertified** -- neither: a nonlinear body whose interval bound exceeds ``M``,
  or an LP bound above ``M`` without a feasible witness. The value is used as
  given, and :meth:`discopt.Model.solve` withholds the certificate of the solve
  (``optimal`` -> ``feasible``, ``infeasible`` -> ``unknown``, bound dropped).

A user M therefore never produces a silently false certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from discopt.modeling.core import Constraint, Model, _DisjunctiveConstraint

#: Relative and absolute slack a certificate must clear: the interval and LP
#: bounds are float results, and a tie must not certify.
_CERT_REL = 1e-9
_CERT_ABS = 1e-12

#: How far a witness must exceed ``M`` to count as a proof (the feasibility
#: tolerance of ``verify_point`` is of the same order).
_WITNESS_TOL = 1e-6


@dataclass(frozen=True)
class UserBigMVerdict:
    """The validation of one row's user M: ``certified``, and why not if not."""

    M: float
    certified: bool
    reason: str


class _Context:
    """Per-model state: the LP relaxation and the exact LP oracle, built lazily."""

    def __init__(self, model: Model, refuse: bool = True):
        self.model = model
        #: Raise on a witness (the lowering), or report it as a reason (the
        #: post-solve guard, which also runs when another method ignored the M).
        self.refuse = refuse
        self._lp: Any = None
        self._lp_built = False
        self.solve_lp = None

    def lp(self):
        if not self._lp_built:
            from discopt._relax.gdp_reformulate import _lp_box, _precompute_lp_relaxation

            self._lp_built = True
            n_vars = sum(v.size for v in self.model._variables)
            data = _precompute_lp_relaxation(self.model)
            if data is None:
                data = (None, None, None, None, n_vars)
            try:
                from discopt.solvers.lp_backend import get_exact_lp_solver

                self.solve_lp = get_exact_lp_solver()
            except ImportError:
                self.solve_lp = None
            self._lp = (data, _lp_box(self.model, data[4]))
        return self._lp


def _label(dc: _DisjunctiveConstraint, k: int, j: int, con: Constraint) -> str:
    return f"{con.name or f'row {j}'} of disjunct {k} of disjunction {dc.name or '<anon>'!r}"


def _verify_side(
    con: Constraint,
    side: str,
    M: float,
    k: int,
    dc: _DisjunctiveConstraint,
    nested: bool,
    ctx: _Context,
    where: str,
) -> Optional[str]:
    """``None`` when ``M`` is certified for ``side`` ("up": ``body <= M``; "down":
    ``-body <= M``); else the reason it is not. Raises on a feasible witness."""
    from discopt._relax.gdp_reformulate import (
        _BIGM_SENTINEL,
        _bound_expression,
        _disjunct_linear_rows,
        _extract_body_coeffs,
        _is_scalar_linear_row,
        _safe_lp_max,
        _stack_rows,
    )

    def clears(bound: float) -> bool:
        return M >= bound + _CERT_REL * abs(bound) + _CERT_ABS

    lo, hi = _bound_expression(con.body, ctx.model)
    ival = float(hi) if side == "up" else -float(lo)
    if np.isfinite(ival) and abs(ival) < _BIGM_SENTINEL and clears(ival):
        return None

    if not _is_scalar_linear_row(con):
        return f"{where}: its interval bound {ival:g} exceeds M = {M:g} and the row is not linear"
    (A_ub, b_ub, A_eq, b_eq, n_vars), box = ctx.lp()
    coeffs = _extract_body_coeffs(con.body, ctx.model, n_vars)
    if coeffs is None:
        return f"{where}: its interval bound {ival:g} exceeds M = {M:g} and the row is not linear"
    if ctx.solve_lp is None:
        return f"{where}: no exact LP solver to check M = {M:g} against the model's rows"
    c_vec, offset = coeffs
    # max side = -min(f.x) + const, with f = -c (up) or c (down).
    sign = -1.0 if side == "up" else 1.0
    const = offset if side == "up" else -offset
    if nested:
        # Relaxed whenever its own selector is 0 -- also when its parent is not
        # selected -- so only the global rows bound it.
        regions = [([], [], [], [])]
    else:
        regions = [
            _disjunct_linear_rows(d, ctx.model, n_vars)
            for j, d in enumerate(dc.disjuncts)
            if j != k
        ]
    from discopt.solvers import SolveStatus
    from discopt.validation.feasibility import verify_point

    for ub_rows, ub_rhs, eq_rows, eq_rhs in regions:
        Au, bu = _stack_rows(A_ub, b_ub, ub_rows, ub_rhs)
        Ae, be = _stack_rows(A_eq, b_eq, eq_rows, eq_rhs)
        res = ctx.solve_lp(c=sign * c_vec, A_ub=Au, b_ub=bu, A_eq=Ae, b_eq=be, bounds=box)
        if res.status != SolveStatus.OPTIMAL or res.objective is None:
            return f"{where}: the LP that bounds it ended {res.status}, so M = {M:g} is unchecked"
        x = getattr(res, "x", None)
        if x is not None and -float(res.objective) + const > M + _WITNESS_TOL * max(1.0, M):
            x = np.asarray(x, dtype=np.float64)
            if x.size == n_vars and verify_point(ctx.model, x).ok:
                if not ctx.refuse:
                    return f"{where}: M = {M:g} is too small (the point {_point(ctx.model, x)})"
                raise ValueError(
                    f"big_m = {M:g} for {where} is too small: the point "
                    f"{_point(ctx.model, x)} satisfies the whole model, its disjunct is "
                    f"not the one that holds there, and the row's "
                    f"{'body' if side == 'up' else '-body'} is "
                    f"{-float(res.objective) + const:.6g} > M, so the big-M row would cut a "
                    "feasible point (#1775 G2.6). Raise big_m, or leave it unset to use "
                    "the computed M."
                )
        safe = _safe_lp_max(sign * c_vec, res, Au, bu, Ae, be, box)
        if safe is None:
            return f"{where}: no rigorous LP bound to check M = {M:g} against"
        safe += const
        if not clears(safe):
            return (
                f"{where}: M = {M:g} is below the LP bound {safe:g} and no feasible point "
                "proves it too small"
            )
    return None


def _point(model: Model, x: np.ndarray, limit: int = 6) -> str:
    out = []
    off = 0
    for v in model._variables:
        vals = x[off : off + v.size]
        off += v.size
        out.append(f"{v.name}={vals[0]:.6g}" if v.size == 1 else f"{v.name}={np.round(vals, 6)}")
        if len(out) >= limit:
            out.append("...")
            break
    return "{" + ", ".join(out) + "}"


def resolve_user_big_ms(
    dc: _DisjunctiveConstraint,
    model: Model,
    *,
    nested: bool = False,
    ctx: Optional[_Context] = None,
) -> dict[int, UserBigMVerdict]:
    """Validate every user M of ``dc`` and of the disjunctions nested in it.

    Returns ``{id(row): verdict}`` for each row with a user M; raises
    ``ValueError`` when one is proven too small. ``model`` is the *declared*
    model (its rows define feasibility). ``nested`` marks ``dc`` as nested in a
    disjunct or a block, which drops the per-disjunct conditioning.
    """
    from discopt._relax.gdp_reformulate import _domain_gap, _domain_gap_error

    ctx = ctx or _Context(model)
    out: dict[int, UserBigMVerdict] = {}
    for k, disjunct in enumerate(dc.disjuncts):
        for j, item in enumerate(disjunct):
            if isinstance(item, _DisjunctiveConstraint):
                out.update(resolve_user_big_ms(item, model, nested=True, ctx=ctx))
                continue
            M = None if dc.big_m is None else dc.big_m[k][j]
            if M is None:
                continue
            gap = _domain_gap(item.body, model)
            if gap is not None:
                raise _domain_gap_error(item, gap, "big-M")
            where = _label(dc, k, j, item)
            sides = [
                s
                for s, on in (
                    ("up", item.sense in ("<=", "==")),
                    ("down", item.sense in (">=", "==")),
                )
                if on
            ]
            reasons = [
                r
                for s in sides
                if (r := _verify_side(item, s, M, k, dc, nested, ctx, where)) is not None
            ]
            out[id(item)] = UserBigMVerdict(M, not reasons, "; ".join(reasons))
    return out


def uncertified_user_big_ms(model: Model) -> list[str]:
    """Why each user M of ``model`` is not certified (empty when all are).

    Walks every ``either_or`` disjunction (and those nested in them or in
    ``add_disjunction`` blocks) that carries ``big_m``. A value proven too small
    is reported, not raised: the lowering has already refused it if it used it.
    """
    roots = [
        c for c in model._constraints if isinstance(c, _DisjunctiveConstraint) and _has_big_m(c)
    ]
    if not roots:
        return []
    ctx = _Context(model, refuse=False)
    reasons: list[str] = []
    for c in roots:
        nested = c.parent_indicator is not None
        for verdict in resolve_user_big_ms(c, model, nested=nested, ctx=ctx).values():
            if not verdict.certified:
                reasons.append(verdict.reason)
    return reasons


def _has_big_m(dc: _DisjunctiveConstraint) -> bool:
    if dc.big_m is not None:
        return True
    return any(
        isinstance(it, _DisjunctiveConstraint) and _has_big_m(it) for d in dc.disjuncts for it in d
    )


def model_has_user_big_m(model: Model) -> bool:
    """Does any disjunction of ``model`` carry a user big-M (cheap check)?"""
    return any(isinstance(c, _DisjunctiveConstraint) and _has_big_m(c) for c in model._constraints)


__all__ = [
    "UserBigMVerdict",
    "model_has_user_big_m",
    "resolve_user_big_ms",
    "uncertified_user_big_ms",
]
