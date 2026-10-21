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
  over the model's linear rows and the box, conditioned on each region where the
  row is relaxed (:func:`_off_regions`). Dropping nonlinear rows only enlarges an
  LP region, so the bound stays valid.
* **refused** (``ValueError``) -- an LP maximiser ``x*`` has the side's value
  above ``M`` and :func:`discopt.validation.feasibility.verify_point` accepts
  ``x*`` for the declared model. Disjunct ``k`` does not hold at ``x*`` (its row is
  violated, as ``M >= 0``), so ``x*`` is feasible with ``k`` unselected, and the
  user's row would cut it: a proof that ``M`` is too small.
* **uncertified** -- neither: a nonlinear body whose interval bound exceeds ``M``,
  an LP bound above ``M`` without a feasible witness, or an LP that did not end
  optimal (a floating-point ``infeasible`` is not a proof that a region is empty:
  no Farkas certificate is checked, so it leaves the M uncertified rather than
  certifying it). The value is used as given, and :meth:`discopt.Model.solve`
  withholds the certificate of a solve whose lowering used it (``optimal`` ->
  ``feasible``, ``infeasible`` -> ``unknown``, bound dropped).

A user M therefore never produces a silently false certificate.

**Where a row is relaxed.** A row of disjunct ``k`` binds as ``body <= M`` exactly
when ``k``'s selector is 0. For a top-level disjunction some other disjunct ``j``
is then selected and its rows hold. For a disjunction nested in disjunct ``p`` of
a parent, the selector is 0 either when the parent selects ``p`` and the nested
one selects ``j != k`` (``p``'s rows and ``j``'s hold), or when the parent's
``p``-selector is itself 0 (the parent's own relaxed regions, recursively). A
disjunction nested in an ``add_disjunction`` block (``parent_indicator``) has a
parent this module does not see, so only the global rows bound it.

**Use tracking.** :func:`record_use` counts, per disjunction, the lowerings that
read its user Ms (only ``gdp_method="big-m"`` does). ``Model.solve`` snapshots the
counts before it starts and checks only the disjunctions whose count moved: a
user M that ``hull`` or ``mbigm`` ignored cannot have weakened the result, and is
not judged (#1851 review). The key is the disjunction's ``_big_m_key``, which
copies of the model keep, so a lowering of a clone still counts.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from discopt._relax._numeric import FLOAT_EPS
from discopt.modeling.core import Constraint, Model, _DisjunctiveConstraint

#: Position of a row in the model: ``(id(disjunction), disjunct, row)``. Keyed by
#: position, not by the row object, so one ``Constraint`` placed in two
#: disjunctions keeps the M of each place (#1851 review nit).
RowKey = tuple[int, int, int]

#: A region where a row is relaxed: the disjuncts whose linear rows all hold
#: there (the empty tuple: the global rows only).
_Region = tuple[list, ...]

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


def _region_rows(region: _Region, model: Model, n_vars: int):
    """The linear rows of every disjunct of ``region``, ``<=`` form."""
    from discopt._relax.gdp_reformulate import _disjunct_linear_rows

    ub_rows: list = []
    ub_rhs: list = []
    eq_rows: list = []
    eq_rhs: list = []
    for d in region:
        u, ur, e, er = _disjunct_linear_rows(d, model, n_vars)
        ub_rows += u
        ub_rhs += ur
        eq_rows += e
        eq_rhs += er
    return ub_rows, ub_rhs, eq_rows, eq_rhs


def _off_regions(
    dc: _DisjunctiveConstraint, k: int, path: _Region, parent_off: list[_Region]
) -> list[_Region]:
    """The regions covering every point where disjunct ``k``'s selector is 0.

    ``path`` holds the disjuncts that hold whenever ``dc``'s parent selector is 1
    (empty at the top level); ``parent_off`` covers the points where it is 0
    (empty at the top level, where it is always 1). An unconditioned region
    contains every other one, so it replaces them.
    """
    regions = list(parent_off) + [path + (d,) for j, d in enumerate(dc.disjuncts) if j != k]
    return [()] if () in regions else regions


def _verify_side(
    con: Constraint,
    side: str,
    M: float,
    regions: list[_Region],
    ctx: _Context,
    where: str,
) -> Optional[str]:
    """``None`` when ``M`` is certified for ``side`` ("up": ``body <= M``; "down":
    ``-body <= M``) over every region of ``regions``; else the reason it is not.
    Raises on a feasible witness when ``ctx.refuse``."""
    from discopt._relax.gdp_reformulate import (
        _BIGM_SENTINEL,
        _extract_body_coeffs,
        _gdp_body_bounds,
        _is_scalar_linear_row,
        _safe_lp_max,
        _stack_rows,
    )

    lo, hi = _gdp_body_bounds(con.body, ctx.model)
    ival = float(hi) if side == "up" else -float(lo)
    # Compared exactly, with no slack: this is the enclosure the computed M
    # (``_compute_big_m``) rests on, so an M equal to it is as valid as the
    # solver's own. (Its float rounding is O(n eps) of the terms, far below the
    # feasibility tolerance the lowered row is solved to.)
    if np.isfinite(ival) and abs(ival) < _BIGM_SENTINEL and M >= ival:
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
    from discopt.solvers import SolveStatus
    from discopt.validation.feasibility import verify_point

    for region in regions:
        ub_rows, ub_rhs, eq_rows, eq_rhs = _region_rows(region, ctx.model, n_vars)
        Au, bu = _stack_rows(A_ub, b_ub, ub_rows, ub_rhs)
        Ae, be = _stack_rows(A_eq, b_eq, eq_rows, eq_rhs)
        res = ctx.solve_lp(c=sign * c_vec, A_ub=Au, b_ub=bu, A_eq=Ae, b_eq=be, bounds=box)
        if res.status == SolveStatus.INFEASIBLE:
            # Not certified (#1851 review): the region may well be empty, but a
            # float LP's "infeasible" is not a proof, and no Farkas ray is checked.
            return (
                f"{where}: the LP over {_region_label(region)} reports infeasible; that is "
                f"not a proof the region is empty, so M = {M:g} is not certified"
            )
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
        # ``safe`` is already the Neumaier-Shcherbina bound, which carries its own
        # float-evaluation margin, so no relative slack is added on top: only the
        # rounding of the addition below (#1851 review, point 4).
        bound = safe + const
        bound += 4.0 * FLOAT_EPS * (abs(safe) + abs(const))
        if not M >= bound:
            return (
                f"{where}: M = {M:g} is below the LP bound {bound:g} and no feasible point "
                "proves it too small"
            )
    return None


def _region_label(region: _Region) -> str:
    return "the global rows" if not region else f"the region of {len(region)} selected disjunct(s)"


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
) -> dict[RowKey, UserBigMVerdict]:
    """Validate every user M of ``dc`` and of the disjunctions nested in it.

    Returns ``{(id(disjunction), disjunct, row): verdict}`` for each row with a
    user M; raises ``ValueError`` when one is proven too small (unless ``ctx``
    was built with ``refuse=False``). ``model`` is the *declared* model (its rows
    define feasibility). ``nested`` marks ``dc`` as nested in an
    ``add_disjunction`` block, whose parent is not visible here: only the global
    rows bound it.
    """
    ctx = ctx or _Context(model)
    out: dict[RowKey, UserBigMVerdict] = {}
    _resolve(dc, ctx, (), [()] if nested else [], out)
    return out


def _resolve(
    dc: _DisjunctiveConstraint,
    ctx: _Context,
    path: _Region,
    parent_off: list[_Region],
    out: dict[RowKey, UserBigMVerdict],
) -> None:
    from discopt._relax.gdp_reformulate import _domain_gap, _domain_gap_error

    for k, disjunct in enumerate(dc.disjuncts):
        regions = _off_regions(dc, k, path, parent_off)
        for j, item in enumerate(disjunct):
            if isinstance(item, _DisjunctiveConstraint):
                _resolve(item, ctx, path + (disjunct,), regions, out)
                continue
            M = None if dc.big_m is None else dc.big_m[k][j]
            if M is None:
                continue
            where = _label(dc, k, j, item)
            gap = _domain_gap(item.body, ctx.model)
            if gap is not None:
                if ctx.refuse:
                    raise _domain_gap_error(item, gap, "big-M")
                out[(id(dc), k, j)] = UserBigMVerdict(
                    M, False, f"{where}: the row is undefined on part of the box ({gap})"
                )
                continue
            sides = [
                s
                for s, on in (
                    ("up", item.sense in ("<=", "==")),
                    ("down", item.sense in (">=", "==")),
                )
                if on
            ]
            reasons = [
                r for s in sides if (r := _verify_side(item, s, M, regions, ctx, where)) is not None
            ]
            out[(id(dc), k, j)] = UserBigMVerdict(M, not reasons, "; ".join(reasons))


def uncertified_user_big_ms(model: Model, used: Optional[set[int]] = None) -> list[str]:
    """Why each user M of ``model`` is not certified (empty when all are).

    Walks every ``either_or`` disjunction (and those nested in them or in
    ``add_disjunction`` blocks) that carries ``big_m`` -- or, with ``used``, only
    the top-level ones whose ``_big_m_key`` (or a nested one's) is in ``used``
    (:func:`used_since`). A value proven too small, or a row with a domain gap,
    is reported, not raised: a lowering that used it has already refused it.
    """
    roots = [
        c for c in model._constraints if isinstance(c, _DisjunctiveConstraint) and _has_big_m(c)
    ]
    if used is not None:
        roots = [c for c in roots if _keys(c) & used or _keyless(c)]
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


# ── use tracking ─────────────────────────────────────────────────────────────

_USES: dict[int, int] = {}
_USES_LOCK = threading.Lock()


def _keys(dc: _DisjunctiveConstraint) -> set[int]:
    """The ``_big_m_key`` of ``dc`` and of every disjunction nested in it."""
    out = set()
    stack = [dc]
    while stack:
        d = stack.pop()
        if d._big_m_key is not None:
            out.add(d._big_m_key)
        stack.extend(
            it for dd in d.disjuncts for it in dd if isinstance(it, _DisjunctiveConstraint)
        )
    return out


def _keyless(dc: _DisjunctiveConstraint) -> bool:
    """Does ``dc`` (or one nested in it) carry user Ms without a tracking key?
    Such a value cannot be traced to a lowering, so it is always judged."""
    stack = [dc]
    while stack:
        d = stack.pop()
        if d.big_m is not None and d._big_m_key is None:
            return True
        stack.extend(
            it for dd in d.disjuncts for it in dd if isinstance(it, _DisjunctiveConstraint)
        )
    return False


def record_use(dc: _DisjunctiveConstraint) -> None:
    """Count one lowering that reads the user Ms of ``dc`` (and those nested in it)."""
    with _USES_LOCK:
        for key in _keys(dc):
            _USES[key] = _USES.get(key, 0) + 1


def use_counts(model: Model) -> dict[int, int]:
    """Snapshot of the use counts of ``model``'s user-M disjunctions."""
    keys: set[int] = set()
    for c in model._constraints:
        if isinstance(c, _DisjunctiveConstraint):
            keys |= _keys(c)
    with _USES_LOCK:
        return {k: _USES.get(k, 0) for k in keys}


def used_since(snapshot: dict[int, int]) -> set[int]:
    """The keys of ``snapshot`` read by a lowering since it was taken."""
    with _USES_LOCK:
        return {k for k, n in snapshot.items() if _USES.get(k, 0) > n}


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
    "RowKey",
    "UserBigMVerdict",
    "model_has_user_big_m",
    "record_use",
    "resolve_user_big_ms",
    "uncertified_user_big_ms",
    "use_counts",
    "used_since",
]
