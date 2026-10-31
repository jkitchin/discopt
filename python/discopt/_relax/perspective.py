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

__all__ = [
    "perspective_objective_terms",
    "perspective_oa_cut_enabled",
    "perspective_disaggregation_enabled",
    "perspective_row_cut_enabled",
    "perspective_row_terms",
    "PerspectiveRowTerm",
]

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
        if len(terms) != 2:
            continue
        sign = 1.0 if con.sense == "<=" else -1.0
        (i, ai), (j, aj) = terms.items()
        ai, aj = sign * ai, sign * aj
        # #1397: ``terms`` and ``const`` are *accumulated* sums
        # (``terms[i] = terms.get(i, 0.0) + v``), so a coefficient that is truly zero
        # can survive cancellation as a small nonzero and a truly nonzero constant can
        # cancel to below 1e-12. Both matter for soundness here: this row is what
        # licenses "y = 0 pins x to 0", and a spurious ``cx > 0`` read off a row where
        # ``x`` does not actually appear would let the perspective row ``x**2 <= s*y``
        # forbid points the original model allows. The residue of cancelling
        # contributions of magnitude ``S`` is ``O(u*S)``, and ``S`` is not observable
        # from the surviving coefficients -- but it is bounded below by the row's own
        # magnitude, so the coherent test is *relative to the row*, not absolute. This
        # only ever rejects more rows than before (the sound direction), and the rows it
        # newly rejects have ``U`` beyond ``_U_CAP`` anyway.
        row_scale = max(1.0, abs(ai), abs(aj))
        if abs(const) > _ZERO_TOL * row_scale:
            continue
        # Orient as  ai*x + aj*y <= 0  with ai > 0 > aj, i.e.  x <= (-aj/ai) y.
        for (xi, cx), (yi, cy) in ((((i, ai), (j, aj))), (((j, aj), (i, ai)))):
            if cx <= _ZERO_TOL * row_scale or cy >= -_ZERO_TOL * row_scale:
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
    from .problem_classifier import (
        _EXTRACTION_DECLINES,
        _extract_quadratic_coefficients,
        dense_Q,
    )

    if model._objective is None:
        return None
    n = sum(v.size for v in model._variables)
    try:
        Q, _c, _d = _extract_quadratic_coefficients(model._objective.expression, model, n)
    except _EXTRACTION_DECLINES as exc:
        # #1520: narrowed to the extractor's documented declines; anything else is
        # a defect and must not read as "not quadratic, no perspective candidates".
        # Both of them: its constant folding raises ``_NotLinearError`` on a
        # non-scalar array coefficient (a gathered objective, ``c[idx] * x[idx]``),
        # which crashed OA's decomposition (#1802; ``patterns.py`` already had it).
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
            # #1397: sum the off-diagonals DIRECTLY. The previous form,
            # ``np.abs(Q[flat, :]).sum() - abs(Q[flat, flat])``, is a difference of two
            # large numbers, so the quantity compared against the absolute 1e-12 was
            # itself scale-dependent: with ``Q[flat, flat] = 1e12`` (ulp 1.2e-4) a
            # genuine cross-term of 1e-5 is absorbed by the row sum and the subtraction
            # returns exactly 0.0 -- a non-separable row read as separable, and the
            # separability gate is a *soundness* condition per this docstring, not a
            # heuristic. Summing only the off-diagonal entries has no cancellation at
            # all (every term is non-negative), so its error is relative to ``off``
            # itself and the absolute comparison is coherent again. The tolerance does
            # not move: this is a yardstick fix, not a tolerance fix (#1397 non-goal).
            row = np.abs(Q[flat, :])
            off = float(np.delete(row, flat).sum())
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


def perspective_disaggregation_enabled() -> bool:
    """Is the #1066 *disaggregated* perspective epigraph switched on?

    The strengthening above puts the perspective correction into the master's
    single aggregate objective epigraph row, so what the master gets each round
    is ``sup_z sum_i g_i(z)`` -- the sum of the per-term perspective cuts taken
    at one *common* reference point. The disaggregated form gives each term its
    own epigraph column, so the master's own LP relaxation combines each term's
    best reference independently and gets ``sum_i sup_z g_i(z)``, which is the
    full perspective closure and never weaker.

    Measured on the real instances before implementing (CLAUDE.md §4), as the
    cutting-plane closure of the master's LP relaxation with the binaries
    relaxed to ``[0, 1]``:

    ==============  ==========  =========  ==========  ============
    instance        aggregate   disagg.    optimum     disagg. % opt
    ==============  ==========  =========  ==========  ============
    squfl015-060      173.401   362.941     366.622        99.0%
    ==============  ==========  =========  ==========  ============

    The aggregate arm was still climbing after 250 cutting-plane rounds and
    51.6 s; the disaggregated arm converged in 41 rounds and 5.9 s.

    **Default ON** since the CLAUDE.md §5 graduation panel of 2026-08-29: 111
    instances (the 104 that reach the convex auto-route, plus the seven rows the
    #1066 report names that the route panel did not already carry), ON vs OFF,
    interleaved arm-by-arm per instance at a 60 s limit and default settings.

    ==========================  ==========  ==========
    metric                      OFF         ON
    ==========================  ==========  ==========
    solved to optimality        98          **99**
    total wall, all instances   1398.5 s    **1348.9 s**
    total nodes                 298 840     **292 065**
    status regressions          --          0
    ==========================  ==========  ==========

    *Cert-clean*: 438 executed checks, 0 violations -- no bound above its
    reference optimum, no incumbent below one, no instance that certified with
    the flag OFF failing to certify with it ON, and no solve error on either arm.
    *Net-positive*: one instance gains a certificate (``squfl025-040``, whose
    dual bound moves 143.83 -> 197.33 against a 197.334 optimum and which
    finishes in 32.5 s instead of exhausting the limit), and the aggregate wall
    and node counts both fall. The only material single-instance slowdown is
    ``slay07h`` (23.9 -> 27.5 s), still optimal.

    ``DISCOPT_PERSPECTIVE_DISAGG=0`` is the opt-out and keeps the aggregate path
    exactly as it was. Read per call, not cached at import, so a test can flip it
    without reloading.
    """
    return os.environ.get("DISCOPT_PERSPECTIVE_DISAGG", "1").strip() not in ("", "0")


def perspective_row_disaggregation_enabled() -> bool:
    """Is the #2021 disaggregated epigraph of a split quadratic row switched on?

    The split cut of :meth:`PerspectiveRowTerm.cut` is ONE row per reference,
    so every coordinate's perspective tangent shares that reference (``r_i =
    w^_i``). At a fractional master point that is the perspective tangent at
    ``z_i = 1``, not at ``z_i``, and a cutting-plane loop over the aggregate row
    approaches the perspective closure very slowly. With this on, the
    ``lp_nlp_bb`` master gets one epigraph column ``s_i >= d_i w_i**2 / z_i`` per
    split coordinate, the remainder tangent ``R-tangent + sum_i s_i <= 0``
    (:meth:`PerspectiveRowTerm.remainder_cut`) and per-coordinate rows
    (:meth:`PerspectiveRowTerm.perspective_references`), IN ADDITION to the
    aggregate cut, which stays in the master unchanged. The master's feasible
    set can therefore only shrink, and only by valid rows.

    Measured before building (performance-plan section 103.12), as the
    cutting-plane closure of portfol_classical050_1's continuous relaxation with
    that solve's own split: aggregate -0.105223 after 80 LPs, -0.097474 after
    400, -0.096991 after 4000; disaggregated -0.097138 after 80. The master gets
    about 127 separation rounds in a 30 s solve.

    **Default ON (graduated with #2021, performance-plan section 103.12).**
    Bound-changing (CLAUDE.md section 5 regime 2). Panel over the 102 routed
    MINLPLib instances (30 s, interleaved, every incumbent verified): 612 checks,
    0 problems, certified unchanged, total wall 427.2 -> 428.9 s; it fires only on
    portfol_classical050_1. On the portfol class the split cut fires on (4
    instances, 4 interleaved reps): portfol_robust050_34 certifies in 16.5-18.6 s
    against uncertified at 30 s (4/4), portfol_classical200_2 finds an incumbent
    (4/4 vs 0/4), portfol_robust100_09 is neutral, and portfol_classical050_1's
    master bound improves (best tree bound -0.09738 -> -0.09685) but its reported
    bound does not (the #1066 guard hands over and the unconfirmed route bound is
    withdrawn in both arms) and its incumbent is 0.4% worse (4/4).
    ``DISCOPT_PERSPECTIVE_ROW_DISAGG=0`` restores the aggregate-only master. Read
    per call, not cached at import, so a test can flip it.
    """
    return os.environ.get("DISCOPT_PERSPECTIVE_ROW_DISAGG", "1").strip() not in ("", "0")


# ── #2021 §3: diagonal-perspective split of a coupled quadratic ROW ───────────


def perspective_row_cut_enabled() -> bool:
    """Is the #2021 diagonal-perspective strengthening of OA row cuts switched on?

    Default ON (graduated with #2021, performance-plan section 103.10).
    Bound-changing under CLAUDE.md §5 regime 2: it replaces the OA tangent of a
    qualifying ``<=`` row with a valid row that is never weaker, and strictly
    stronger wherever an indicator is below 1. ``DISCOPT_PERSPECTIVE_ROW_CUT=0``
    restores the plain tangent. Read per call, not cached at import, so a test can
    flip it.
    """
    return os.environ.get("DISCOPT_PERSPECTIVE_ROW_CUT", "1").strip() not in ("", "0")


#: A row whose quadratic block or whose defining-variable block is wider than
#: this is declined: the split costs dense ``O(k**3)`` eigen-decompositions, and
#: declining only ever forgoes a strengthening.
_ROW_SPLIT_MAX_COLS = 500


class PerspectiveRowTerm:
    """One ``<=`` row ``0.5 x_S' Q_SS x_S + c'x + k <= 0`` and its certified split.

    ``x_S = F w + f0`` holds at every feasible point (each ``x_j``, ``j in S``,
    is defined by one linear ``==`` row of the model). ``d`` is the diagonal
    split off ``M = 0.5 F' Q_SS F`` with ``M - diag(d)`` certified PSD; ``z[i]``
    is the binary indicator of ``w_i`` (``-1`` where ``d[i] == 0``).
    """

    __slots__ = ("row", "S", "QSS", "c_idx", "c_val", "k", "F", "f0", "W", "d", "z", "t", "U")

    def __init__(self, row, S, QSS, c_idx, c_val, k, F, f0, W, d, z, t, U=None):
        self.row = int(row)
        self.S = S
        self.QSS = QSS
        self.c_idx = c_idx
        self.c_val = c_val
        self.k = float(k)
        self.F = F
        self.f0 = f0
        self.W = W
        self.d = d
        self.z = z
        self.t = float(t)
        #: ``U[i]`` of the row ``w_i <= U_i z_i`` that licenses the split of
        #: coordinate ``i`` (``inf`` where ``d[i] == 0``). Only the
        #: disaggregated references use it, to clip ``w_i / z_i``.
        self.U = np.full(len(self.d), np.inf) if U is None else np.asarray(U, dtype=np.float64)

    def plain_tangent(self, x, n_vars: int) -> np.ndarray:
        """The row's ordinary OA tangent coefficients at ``x`` (no split)."""
        coeffs = np.zeros(n_vars, dtype=np.float64)
        np.add.at(coeffs, self.c_idx, self.c_val)
        coeffs[self.S] += self.QSS @ np.asarray(x, dtype=np.float64)[self.S]
        return coeffs

    def cut(self, x, n_vars: int) -> tuple[np.ndarray, float, int]:
        """``(coeffs, rhs, n_shifted)``: the split cut referenced at ``x``.

        The reference is projected through ``F`` first (``w = x[W]``,
        ``x^ = F w + f0``) and the tangent is taken at ``x^``, never at
        ``x[S]``: the validity argument is the tangent of the PSD remainder at
        ``w`` plus each ``d_i w_i**2``'s perspective cut, and that tangent is the
        row's tangent at ``x^`` only.
        """
        wbar = np.asarray(x, dtype=np.float64)[self.W]
        xhat = self.F @ wbar + self.f0
        coeffs = np.zeros(n_vars, dtype=np.float64)
        np.add.at(coeffs, self.c_idx, self.c_val)
        coeffs[self.S] += self.QSS @ xhat
        rhs = -self.k + 0.5 * float(xhat @ self.QSS @ xhat)
        shifted = 0
        for di, wi, zi in zip(self.d, wbar, self.z):
            if di > 0.0 and zi >= 0 and wi > 0.0:
                sh = float(di * wi * wi)
                coeffs[zi] -= sh
                rhs -= sh
                shifted += 1
        return coeffs, rhs, shifted

    def split_positions(self) -> np.ndarray:
        """Positions ``i`` (into ``W``/``d``/``z``) of the split coordinates."""
        return np.flatnonzero((self.d > 0.0) & (self.z >= 0))

    def remainder_cut(self, x, n_vars: int) -> tuple[np.ndarray, float]:
        """``(coeffs, rhs)``: tangent at ``w^ = x[W]`` of the convex remainder.

        ``R(w) = g(F w + f0) - sum_i d_i w_i**2`` over the split positions. The
        row ``coeffs . x + sum_i s_i <= rhs`` (``s_i`` the disaggregated epigraph
        columns of this row) is valid wherever ``s_i >= d_i w_i**2``, because
        ``M - diag(d)`` is PSD (``R`` is convex, so its tangent underestimates it)
        and ``R + sum_i d_i w_i**2 = g <= 0``. The tangent is taken at the
        PROJECTED reference ``x^ = F w^ + f0``, as :meth:`cut` does, so it is a
        tangent of ``R`` on the set ``x_S = F w + f0`` the model enforces.
        """
        wbar = np.asarray(x, dtype=np.float64)[self.W]
        xhat = self.F @ wbar + self.f0
        coeffs = np.zeros(n_vars, dtype=np.float64)
        np.add.at(coeffs, self.c_idx, self.c_val)
        coeffs[self.S] += self.QSS @ xhat
        rhs = -self.k + 0.5 * float(xhat @ self.QSS @ xhat)
        for i in self.split_positions():
            wi = float(wbar[i])
            # The tangent of ``-d_i w_i**2`` at ``w^_i`` is ``-2 d_i w^_i w_i + d_i w^_i**2``.
            coeffs[self.W[i]] -= 2.0 * self.d[i] * wi
            rhs -= self.d[i] * wi * wi
        return coeffs, rhs

    def perspective_references(self, x) -> list[tuple[int, float]]:
        """``(i, r)`` references for the per-coordinate rows at the point ``x``.

        Row ``(i, r)`` is ``2 d_i r w_i - d_i r**2 z_i - s_i <= 0``: the
        Frangioni-Gentile tangent of ``d_i w_i**2 / z_i`` at the ratio ``r``,
        valid for every real ``r`` on both integral values of ``z_i`` (``z_i = 1``:
        ``d_i (w_i - r)**2 >= 0``; ``z_i = 0``: ``w_i = 0`` and ``s_i >= 0``).

        Two references per coordinate, deduplicated: ``r = w_i``, the one the
        aggregate :meth:`cut` uses, so the disaggregated rows together with
        :meth:`remainder_cut` imply that cut at every point; and, where
        ``0 < z_i < 1``, ``r = w_i / z_i`` clipped to ``U_i``, the tangent of the
        perspective AT ``(w_i, z_i)`` -- the one reference the aggregate row cannot
        take, since it must share one reference across all coordinates.
        ``w_i <= 0`` gives ``r = 0``, whose row ``s_i >= 0`` is the column bound,
        so it is not returned.
        """
        xv = np.asarray(x, dtype=np.float64)
        wbar = xv[self.W]
        out: list[tuple[int, float]] = []
        for i in self.split_positions():
            wi = float(wbar[i])
            if not np.isfinite(wi) or wi <= 0.0:
                continue
            out.append((int(i), wi))
            zv = float(xv[self.z[i]])
            if np.isfinite(zv) and 0.0 < zv < 1.0:
                r = min(wi / zv, float(self.U[i]))
                if np.isfinite(r) and r > wi:
                    out.append((int(i), r))
        return out


def _flat_lower_bounds(model: Model, n: int) -> np.ndarray:
    lb = np.empty(n, dtype=np.float64)
    for v in model._variables:
        base = _compute_var_offset(v, model)
        lb[base : base + v.size] = np.ravel(
            np.broadcast_to(np.asarray(v.lb, dtype=np.float64), v.shape)
        )
    return lb


def _certified_split(M: np.ndarray, diag: np.ndarray):
    """Largest ``t`` (to bisection precision, then ``x 0.999``) with
    ``M - t*diag(diag)`` PSD past a margin, plus a Cholesky certificate.

    Returns ``(t, d)`` or ``None``. ``None`` whenever ``M`` itself does not clear
    the margin or the certificate fails: the split is refused, never
    approximated.
    """
    scale = max(1.0, float(np.abs(np.diag(M)).max()))
    margin = 1e-9 * scale
    eye = np.eye(M.shape[0])
    if np.linalg.eigvalsh(M)[0] < margin:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if np.linalg.eigvalsh(M - np.diag(mid * diag))[0] >= margin:
            lo = mid
        else:
            hi = mid
    t = lo * 0.999
    if t <= 0.0:
        return None
    d = t * diag
    try:
        np.linalg.cholesky(M - np.diag(d) - margin * eye)
    except np.linalg.LinAlgError:
        # The certificate is the only thing licensing the split; without it the
        # row keeps its ordinary tangent.
        return None
    return t, d


def perspective_row_terms(model: Model, row_map) -> list[PerspectiveRowTerm]:
    """Every scalar ``<=`` quadratic row the diagonal-perspective split applies to.

    ``row_map`` is the evaluator's ``constraint_row_map()``: ``(start, stop,
    Constraint)`` per source constraint, which is what ties a model constraint to
    the evaluator row an OA cut names. Only one-row (scalar) constraints are
    read, and the returned :attr:`PerspectiveRowTerm.row` is the evaluator row.

    **The algebra.** For a row ``g(x) = 0.5 x_S' Q_SS x_S + c'x + k <= 0`` whose
    quadratic columns are each defined by a linear ``==`` row, ``x_S = F w + f0``
    at every feasible point, so on the feasible set ``g`` is a quadratic in ``w``
    with Hessian ``2M``, ``M = 0.5 F' Q_SS F``. Write
    ``g = [g - sum_i d_i w_i**2] + sum_i d_i w_i**2``. With ``M - diag(d)`` PSD
    the bracket is convex in ``w`` and lies above its tangent at ``w^``; each
    ``d_i w_i**2`` with ``w_i`` semicontinuous (``lb >= 0``, ``w_i <= U z_i``,
    ``z_i`` binary) lies above its Frangioni-Gentile perspective cut
    ``2 d_i w^_i w_i - d_i w^_i**2 z_i``. Summing, every feasible point satisfies
    the row's tangent at ``x^ = F w^ + f0`` plus ``d_i w^_i**2 (1 - z_i)``, i.e.
    the tangent with ``coeffs[z_i] -= d_i w^_i**2`` and ``rhs -= d_i w^_i**2``.
    Validity needs only ``M - diag(d)`` PSD and the exact substitution -- not
    ``Q_SS`` PSD and not the row's convexity in ``x``.

    Every gate is a soundness condition and a doubt declines the row:

    * the row is scalar, ``<=``, and quadratic (degree ``<= 2``);
    * each ``j in S`` is the only quadratic column of exactly one scalar linear
      ``==`` row that defines it (a defining coefficient not tiny relative to the
      row), and the defined columns ``W`` exclude ``S``;
    * a split coordinate has ``lb >= 0`` and an indicator from
      :func:`_semicontinuity_rows` that is not itself a ``W`` column it shifts;
    * ``M`` clears a PSD margin, and ``M - diag(d) - margin*I`` passes Cholesky.
    """
    from .problem_classifier import (
        _EXTRACTION_DECLINES,
        _extract_linear_coefficients_sparse,
        _extract_quadratic_terms,
    )

    n = sum(v.size for v in model._variables)
    if n == 0:
        return []
    scalar = [(start, con) for start, stop, con in row_map if stop - start == 1]
    semi = _semicontinuity_rows(model)
    if not semi:
        return []
    lb = _flat_lower_bounds(model, n)

    eq_rows: list[tuple[dict[int, float], float]] = []
    for _start, con in scalar:
        if con.sense != "==":
            continue
        try:
            terms, const = _extract_linear_coefficients_sparse(con.body, model, n)
        except _EXTRACTION_DECLINES:
            continue
        eq_rows.append((terms, const))
    if not eq_rows:
        return []

    out: list[PerspectiveRowTerm] = []
    for start, con in scalar:
        if con.sense != "<=":
            continue
        try:
            qterms, c, k = _extract_quadratic_terms(con.body, model, n)
        except _EXTRACTION_DECLINES:
            continue
        nz = {(i, j): v for (i, j), v in qterms.items() if v != 0.0}
        if not nz:
            continue
        S = np.array(sorted({i for i, _ in nz} | {j for _, j in nz}), dtype=np.int64)
        if S.size < 2 or S.size > _ROW_SPLIT_MAX_COLS:
            continue
        pos = {int(j): a for a, j in enumerate(S)}
        QSS = np.zeros((S.size, S.size))
        for (i, j), v in nz.items():
            QSS[pos[i], pos[j]] = v
        if not np.allclose(QSS, QSS.T, rtol=0.0, atol=1e-12 * max(1.0, np.abs(QSS).max())):
            continue  # not a symmetric Hessian; the gradient Q x would be wrong
        defs: dict[int, tuple[dict[int, float], float]] = {}
        ambiguous: set[int] = set()
        for terms, const in eq_rows:
            hit = [j for j in terms if j in pos and terms[j] != 0.0]
            if len(hit) != 1:
                continue
            j = hit[0]
            if j in defs:
                ambiguous.add(j)
            defs[j] = (terms, const)
        if ambiguous or len(defs) != S.size:
            continue
        wcols = sorted(
            {i for j in defs for i, v in defs[j][0].items() if i not in pos and v != 0.0}
        )
        if not wcols or len(wcols) > _ROW_SPLIT_MAX_COLS:
            continue
        widx = {i: t for t, i in enumerate(wcols)}
        F = np.zeros((S.size, len(wcols)))
        f0 = np.zeros(S.size)
        ok = True
        for a, j in enumerate(S):
            terms, const = defs[int(j)]
            aj = terms[int(j)]
            rscale = max(abs(v) for v in terms.values())
            if abs(aj) < 1e-6 * rscale:
                ok = False  # an ill-conditioned definition; F would amplify rounding
                break
            for i, v in terms.items():
                if i != j and v != 0.0:
                    F[a, widx[i]] = -v / aj
            f0[a] = -const / aj
        if not ok:
            continue
        zcols = np.full(len(wcols), -1, dtype=np.int64)
        ucols = np.full(len(wcols), np.inf)
        for t_i, i in enumerate(wcols):
            sc = semi.get(i)
            if sc is None or lb[i] < 0.0:
                continue
            ind_var, ind_elem, u = sc
            zc = _compute_var_offset(ind_var, model) + ind_elem
            if zc == i:
                continue
            zcols[t_i] = zc
            ucols[t_i] = float(u)
        if not (zcols >= 0).any():
            continue
        M = 0.5 * F.T @ QSS @ F
        M = 0.5 * (M + M.T)
        diag = np.where(zcols >= 0, np.diag(M), 0.0)
        if not (diag > 0.0).any():
            continue
        split = _certified_split(M, diag)
        if split is None:
            continue
        t, d = split
        zcols = np.where(d > 0.0, zcols, -1)
        c = np.asarray(c, dtype=np.float64)
        c_idx = np.flatnonzero(c)
        out.append(
            PerspectiveRowTerm(
                start,
                S,
                QSS,
                c_idx,
                c[c_idx].copy(),
                float(k),
                F,
                f0,
                np.asarray(wcols, dtype=np.int64),
                d,
                zcols,
                t,
                ucols,
            )
        )
    return out
