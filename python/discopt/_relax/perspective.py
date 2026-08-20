"""Perspective structure for semicontinuous convex squares (#1064).

A convex MIQP whose continuous variables are *switched off* by binaries carries
structure the plain relaxation throws away. On the MINLPLib ``squfl`` family
(separable quadratic uncapacitated facility location) every continuous ``x`` is
tied to a binary ``y`` by a variable-upper-bound row ``x - U*y <= 0``, so
``y = 0`` forces ``x = 0``: ``x`` is *semicontinuous*.

For such an ``x`` the epigraph of ``q*x**2`` may be replaced by the
**perspective** ``q*x**2/y``, the convex hull of
``{(x, s, y) : y in {0,1}, x = 0 if y = 0, s >= q*x**2}``. Its linearizations are
the Frangioni-Gentile perspective cuts: for any reference ``z``,

    s >= 2*q*z*x - q*z**2 * y                                             (P)

valid on the two integral values of ``y``:

* ``y = 1``: the ordinary tangent to a convex function -- a global
  underestimator;
* ``y = 0``: semicontinuity gives ``x = 0``, so the true term is ``0`` and (P)
  reads ``s >= 0``.

and (P) dominates the plain tangent everywhere in ``y <= 1``, strictly wherever
``y < 1`` and ``z != 0``. This is what SCIP does automatically (Bestuzheva,
Gleixner and Vigerske, "A computational study of perspective cuts") and what
MINLPLib's hand-written ``squfl*persp`` variants encode by hand; see
``docs/references.bib``.

This module only *detects* the structure. Applying it is
``discopt.solvers.oa._strengthen_objective_cut_perspective``, which uses (P) to
strengthen the OA master's objective epigraph row.

**Why a cut and not a model rewrite.** Rewriting ``q*x**2`` to ``q*s`` with
``x**2 <= s*y`` was built first and measured: on ``squfl015-060`` /
``squfl020-150`` / ``squfl025-040`` at 60 s the primal gap collapses from
+68.9% / +114.8% / +114.9% to 0.00% / 0.00% / 0.05%, but the added row is
nonconvex-quadratic, so the model leaves the convex relaxation for the spatial
path and **the dual bound gets looser on every one**. Trading a certificate for
an incumbent is a certification regression under CLAUDE.md §5, so the rewrite
was dropped. The cut below gives up nothing: the model is untouched and the row
it strengthens is one the master already carried.

Nothing here is keyed to a problem name or shape (CLAUDE.md §2): the detector
reads the model's own linear constraints and objective Hessian.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from discopt.modeling.core import (
    Constraint,
    Model,
    ObjectiveSense,
    Variable,
    VarType,
)

from .term_classifier import _compute_var_offset

logger = logging.getLogger(__name__)

__all__ = ["perspective_objective_terms", "perspective_oa_cut_enabled"]

#: Coefficients below this are treated as structurally zero when reading the
#: objective Hessian. Matches the classifier's own quadratic-term tolerance.
_ZERO_TOL = 1e-12
#: Refuse an indicator upper bound past this magnitude: a semicontinuous variable
#: with a bound that large is not really switched off by its indicator, and
#: ``q*U**2`` would swamp the row it is added to.
_U_CAP = 1e15


class _Candidate:
    """One ``q*x**2`` term whose ``x`` is semicontinuous."""

    __slots__ = ("var", "elem", "flat", "ind_var", "ind_elem", "u", "q")

    def __init__(self, var, elem, flat, ind_var, ind_elem, u, q=0.0):
        self.var = var
        self.elem = elem
        self.flat = flat
        self.ind_var = ind_var
        self.ind_elem = ind_elem
        self.u = u
        #: the term's coefficient in ``q*x**2``, MINIMIZE convention.
        self.q = float(q)


def _flat_index(var: Variable, elem: int, model: Model) -> int:
    return _compute_var_offset(var, model) + elem


def _bound(var: Variable, elem: int, which: str) -> float:
    return float(np.asarray(getattr(var, which)).flat[elem])


def _is_binary(var: Variable, elem: int) -> bool:
    if var.var_type == VarType.BINARY:
        return True
    if var.var_type != VarType.INTEGER:
        return False
    return _bound(var, elem, "lb") >= -_ZERO_TOL and _bound(var, elem, "ub") <= 1.0 + _ZERO_TOL


def _semicontinuity_rows(model: Model) -> dict[int, tuple[Variable, int, float]]:
    """Map ``flat(x) -> (indicator, elem, U)`` for every row ``a*x - b*y <= 0``.

    Only two-term rows with zero constant qualify, because anything else is not
    the switch-off implication the perspective hull needs. A variable governed by
    two different indicators keeps the *tightest* ``U``: both rows hold, so the
    smaller bound is the one the disjunction may assume.
    """
    from .problem_classifier import _extract_linear_coefficients_sparse, _NotLinearError

    n = sum(v.size for v in model._variables)
    flat_to_ref: dict[int, tuple[Variable, int]] = {}
    for v in model._variables:
        base = _compute_var_offset(v, model)
        for e in range(v.size):
            flat_to_ref[base + e] = (v, e)

    found: dict[int, tuple[Variable, int, float]] = {}
    for con in model._constraints:
        if not isinstance(con, Constraint) or con.sense not in ("<=", ">="):
            continue
        try:
            terms, const = _extract_linear_coefficients_sparse(con.body, model, n)
        except _NotLinearError:
            # The extractor's own "this row is not linear" signal, and the only
            # failure this loop may absorb: a nonlinear body simply is not the
            # ``x <= U*y`` implication we are scanning for, and skipping it costs
            # at most a missed candidate (fewer lifts, never an unsound one).
            # Anything else propagates -- a swallowed failure here would read as
            # "this model has no semicontinuous structure" (CLAUDE.md §7).
            continue
        if len(terms) != 2 or abs(const) > _ZERO_TOL:
            continue
        sign = 1.0 if con.sense == "<=" else -1.0
        (i, ai), (j, aj) = terms.items()
        ai, aj = sign * ai, sign * aj
        # Orient as  ai*x + aj*y <= 0  with ai > 0 > aj, i.e.  x <= (-aj/ai) y.
        for (xi, cx), (yi, cy) in ((((i, ai), (j, aj))), (((j, aj), (i, ai)))):
            if cx <= _ZERO_TOL or cy >= -_ZERO_TOL:
                continue
            xvar, xelem = flat_to_ref[xi]
            yvar, yelem = flat_to_ref[yi]
            if not _is_binary(yvar, yelem):
                continue
            if xvar.var_type == VarType.BINARY:
                continue  # a binary's square is linear; nothing to strengthen
            u = -cy / cx
            if not np.isfinite(u) or u <= _ZERO_TOL or u > _U_CAP:
                continue
            prev = found.get(xi)
            if prev is None or u < prev[2]:
                found[xi] = (yvar, yelem, u)
    return found


def _objective_hessian(model: Model):
    """``(Q, n)`` for the objective in MINIMIZE convention, or ``None``.

    ``Q`` follows the classifier's ``0.5 x'Qx`` convention, so a term ``q*x^2``
    appears as ``Q[j, j] = 2q``.
    """
    from .problem_classifier import _extract_quadratic_coefficients, dense_Q

    if model._objective is None:
        return None
    n = sum(v.size for v in model._variables)
    try:
        Q, _c, _d = _extract_quadratic_coefficients(model._objective.expression, model, n)
    except Exception as exc:
        logger.debug("perspective reform: objective is not quadratic (%s)", exc)
        return None
    Q = dense_Q(Q)
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        Q = -Q
    return Q, n


def find_candidates(model: Model) -> list[_Candidate]:
    """Every ``q*x**2`` in the objective whose ``x`` is semicontinuous.

    Each gate is a soundness condition, not a heuristic:

    * ``Q[j, j] > 0`` -- the term must be *convex* in MINIMIZE convention.
      Minimization then drives the lifted ``s`` down onto ``x^2``; with a
      negative coefficient it would be driven up to its bound instead and the
      rewrite would cut off the true optimum.
    * ``Q[j, k] == 0`` for ``k != j`` -- the term must be *separable*. A
      cross-term means ``x`` also appears in a product this pass does not lift,
      and replacing only the square would not be an identity.
    * ``lb(x) >= 0`` -- the indicator row gives ``x <= U*y``, so ``y = 0`` yields
      ``x <= 0``. Only a nonnegative ``x`` is thereby *pinned to zero*; if ``x``
      could go negative the perspective row ``x^2 <= s*y`` would forbid points
      the original model allows.
    """
    hess = _objective_hessian(model)
    if hess is None:
        return []
    Q, _n = hess
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        return []
    rows = _semicontinuity_rows(model)
    if not rows:
        return []

    out: list[_Candidate] = []
    for v in model._variables:
        if v.var_type == VarType.BINARY:
            continue
        base = _compute_var_offset(v, model)
        for e in range(v.size):
            flat = base + e
            hit = rows.get(flat)
            if hit is None:
                continue
            if flat >= Q.shape[0]:
                continue
            if Q[flat, flat] <= _ZERO_TOL:
                continue
            off = np.abs(Q[flat, :]).sum() - abs(Q[flat, flat])
            if off > _ZERO_TOL:
                continue
            if _bound(v, e, "lb") < -_ZERO_TOL:
                continue
            ind_var, ind_elem, u = hit
            out.append(_Candidate(v, e, flat, ind_var, ind_elem, u, 0.5 * float(Q[flat, flat])))
    return out


def perspective_objective_terms(model: Model) -> list[tuple[int, int, float]]:
    """``(x_col, y_col, q)`` per separable convex square over a semicontinuous ``x``.

    The row this feeds is the OA master's objective epigraph tangent, one
    aggregate row per expansion point ``xbar``::

        eta >= f(xbar) + grad f(xbar)^T (x - xbar)

    For a separable convex square over a semicontinuous ``x`` that term's own
    contribution to the tangent, ``2*q*xbar*x - q*xbar**2``, is dominated by the
    perspective cut ``2*q*xbar*x - q*xbar**2 * y`` (module docstring). Summing
    the perspective form over the qualifying terms and the ordinary tangent over
    the rest keeps the whole row a valid underestimator of ``f``, hence a valid
    master cut, and nothing about a node box enters -- so the row is globally
    valid exactly as the tangent it replaces was.

    In row form the master cut is ``grad^T x - eta <= rhs``; strengthening term
    ``i`` subtracts ``q_i*xbar_i**2`` from **both** the ``y_i`` coefficient and
    the right-hand side. At ``y_i = 1`` the two cancel and the row is unchanged,
    which is the algebraic statement of "identical to the tangent at ``y = 1``".

    Returns ``[]`` when the model has no such structure, so the caller pays one
    Hessian read per solve and nothing else.
    """
    out: list[tuple[int, int, float]] = []
    for cand in find_candidates(model):
        if not (cand.q > _ZERO_TOL) or not np.isfinite(cand.q):
            continue
        y_col = _flat_index(cand.ind_var, cand.ind_elem, model)
        out.append((int(cand.flat), int(y_col), float(cand.q)))
    return out


def perspective_oa_cut_enabled() -> bool:
    """Is the #1064 perspective strengthening of the OA objective cut switched on?

    **Default ON (graduated).** Strengthening a master cut is bound-changing
    under CLAUDE.md §5 regime 2 -- the master's optimum is the OA dual bound, so
    a tighter row moves it by construction. Unlike the *rewrite* it replaces, it
    cannot lose a certificate by changing which relaxation the model gets: the
    model is untouched and the row it strengthens is one the master already
    carried.

    It shipped default-OFF and graduated on the differential panel over the full
    affected population -- every corpus instance that carries the structure --
    which cleared both §5 bars: PANEL_CERT_CLEAN and PANEL_NET_POSITIVE (see the
    commit and ``docs/dev/performance-plan.md`` §19 for the table).

    ``DISCOPT_PERSPECTIVE_OA_CUT=0`` is the opt-out. Read per call, not cached at
    import, so a test can flip it without reloading.
    """
    return os.environ.get("DISCOPT_PERSPECTIVE_OA_CUT", "1").strip() not in ("", "0")
