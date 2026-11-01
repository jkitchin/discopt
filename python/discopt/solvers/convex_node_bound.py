"""A rigorous lower bound for a convex node NLP from its approximate optimum (#2067).

Why the node objective is not a bound
=====================================
NLP-BB on a convex MINLP solves each node's continuous relaxation and has always
imported the NLP's *primal objective* as the node's lower bound. That value is the
objective of an approximate minimiser: it is above the relaxation's true minimum by
the interior-point method's terminal duality and complementarity residuals, and an
iterate that sits a primal tolerance outside the feasible set can be either side of
it. Measured on QPLIB_8153 (124 vars, 31 integer, convex quadratic rows): the root
NLP objective was ``-1.076351850232`` while the reference optimum, feasible to
2.7e-15, is ``-1.076351999365`` -- the root "bound" sat **1.49e-7 past** a feasible
point, and the certified tree bound published on macOS crossed it by 1.57e-7 (#2067).

The bound
=========
For a convex node, every constraint is either affine, a convex body below a finite
upper side, or a concave body above a finite lower side (that is what certifies the
model convex, :func:`discopt._relax.convexity.classify_model`). The linearisation of
each at **any** point ``x̂`` of the node box is therefore a valid outer approximation
of the node's feasible set, and the linearisation of the convex objective underestimates
it. So, with ``f, ∇f, g, J`` evaluated at ``x̂``, the LP

    min  f(x̂) + ∇f(x̂)ᵀ(x - x̂)
    s.t. cl <= g(x̂) + J(x̂)(x - x̂) <= cu,   node_lb <= x <= node_ub

is a relaxation of the node NLP, and a Neumaier-Shcherbina safe bound on *its* minimum
(:func:`discopt.solvers.lp_milp_highs.certify_lp_dual`, valid for every dual vector) is a
rigorous lower bound on the node. At an approximate KKT point the LP's optimum equals the
NLP's to the NLP's own tolerance, so the bound loses nothing a certificate can afford.

Validity rests on three facts and nothing about the NLP solve itself: the model is
certified convex, ``x̂`` lies in the node box (it is clipped into it), and the
evaluator's ``f, ∇f, g, J`` are accurate to the floating-point slack charged below.
The NLP's multipliers are never read, so their row order and sign convention (which
differ between the POUNCE native, callback and Ipopt node engines) cannot matter.

What "rigorous" means here, exactly
-----------------------------------
The LP's own arithmetic is enclosed (the NS bound is summed at double-double precision,
and forming ``k = g - J x̂`` is charged). The *evaluator's* values are not intervals: the
slack also charges ``ROUNDOFF_OPS * eps`` times each value's magnitude and, over the
finite part of the box, times ``|∇| · width`` -- the error a few-ulp gradient makes
across the region the linearisation is used on. That covers an evaluator whose result
is accurate to a few ulps of the magnitudes involved; a row whose AD evaluation cancels
catastrophically (many large terms summing to a small value) can carry more, and an
open box side gets no width term. So the bound is rigorous up to the evaluator's
accuracy at that scale -- ~1e-15 relative, eight orders below the 1.5e-7 crossings this
module exists to remove -- not an interval-arithmetic proof.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from discopt._relax._numeric import FLOAT_EPS, ROUNDOFF_OPS

#: A node box side at or beyond this magnitude is treated as open (the house
#: ``±9.999e19`` "no bound" box and the ``1e20`` sentinel alike). Opening a side only
#: enlarges the LP's box, so the bound stays valid; it lets the certificate ladder's
#: open-column repairs and FBBT box recover a finite bound where a ``1e19`` side would
#: otherwise multiply a rounding-sized reduced cost into a useless one.
_OPEN_SIDE = 1e15

#: The certificate ladder's exact dual correction is bounded by a deterministic
#: operation count (``lp_milp_highs.EXACT_MAX_WORK``/``EXACT_MAX_TOTAL_WORK``), not by a
#: clock of its own: a wall cap here made a node's bound a function of machine speed
#: (#912). Only the caller's ``deadline`` -- its own ``time_limit`` -- can stop it.


class SafeBoundBudget:
    """Per-solve, deterministic rationing of the node bound's expensive paths.

    Each exact dual correction is capped by an operation count, but nothing capped how
    many nodes ran one: on a badly scaled convex model where every node's cheap rungs
    fall short, every node paid it in turn (#2070 review). Two counts, no clock (#912):

    * ``exact_left`` -- exact corrections the solve may still run; past it a node whose
      cheap rungs do not close keeps the best cheap bound (or abstains).
    * ``suspend_after`` -- consecutive nodes with no bound at all after which the solve
      stops trying and every later node abstains at once. Abstaining is the sound
      answer either way; this only stops paying for a ladder that keeps failing.
    """

    EXACT_CALLS = 64
    SUSPEND_AFTER = 32

    def __init__(self) -> None:
        self.exact_left = self.EXACT_CALLS
        self.consecutive_failures = 0
        self.suspended = False

    def record(self, proved: bool) -> None:
        if proved:
            self.consecutive_failures = 0
            return
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.SUSPEND_AFTER:
            self.suspended = True


def _as_csr(J, m: int, n: int) -> sp.csr_matrix:
    if sp.issparse(J):
        out = sp.csr_matrix(J, dtype=np.float64)
    else:
        out = sp.csr_matrix(np.asarray(J, dtype=np.float64).reshape(m, n))
    if out.shape != (m, n):
        raise ValueError(f"constraint Jacobian has shape {out.shape}, expected {(m, n)}")
    return out


def _solve_row_equilibrated(highspy, new_highs, J, xl, xu, s_lo, s_hi, grad):
    """Solve ``min gradᵀx  s.t.  s_lo <= J x <= s_hi, xl <= x <= xu`` with each row
    divided by its largest coefficient; returns ``(highs, scale, status_name)``.

    A row dual ``y_s`` of the scaled rows is the dual ``y_s / scale`` of the unscaled
    ones, in the sign convention of the slack form ``J x - s = 0`` (a positive dual
    prices the row's lower side in both).
    """
    from discopt.solvers.lp_milp_highs import _status_name
    from discopt.solvers.milp_highs import _to_highs_inf

    Jr = sp.csr_matrix(J, dtype=np.float64)
    m, n = Jr.shape
    scale = np.ones(m)
    if Jr.nnz:
        row_max = np.asarray(abs(Jr).max(axis=1).todense()).ravel()
        scale = np.where(row_max > 0.0, row_max, 1.0)
    Js = sp.csc_matrix(sp.diags(1.0 / scale) @ Jr)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = np.asarray(grad, dtype=np.float64)
    lp.col_lower_ = _to_highs_inf(xl, highspy)
    lp.col_upper_ = _to_highs_inf(xu, highspy)
    lp.row_lower_ = _to_highs_inf(np.where(s_lo <= -1e20, s_lo, s_lo / scale), highspy)
    lp.row_upper_ = _to_highs_inf(np.where(s_hi >= 1e20, s_hi, s_hi / scale), highspy)
    lp.sense_ = highspy.ObjSense.kMinimize
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, m
    lp.a_matrix_.start_ = Js.indptr.astype(np.int32)
    lp.a_matrix_.index_ = Js.indices.astype(np.int32)
    lp.a_matrix_.value_ = Js.data
    h = new_highs(highspy, [("presolve", "off")])
    if h.passModel(lp) == highspy.HighsStatus.kError:
        return h, scale, "refused by HiGHS passModel"
    h.run()
    return h, scale, _status_name(h)


def _simplex_dual(J, xl, xu, s_lo, s_hi, grad):
    """The row dual of the linearised LP from the in-house Rust simplex, in the
    slack-form convention of :func:`_lp_dual`; ``(y, value)`` or ``(None, reason)``.

    ``lp_simplex.solve_lp`` takes ``A_ub x <= b_ub`` / ``A_eq x == b_eq`` and returns
    duals in HiGHS's convention (``c - Aᵀy`` is the reduced cost; measured equal on a
    one-row LP). A row ``lo <= J_i x <= hi`` is split into ``J_i x <= hi`` (dual
    ``y_u``) and ``-J_i x <= -lo`` (dual ``y_l``), whose multiplier on ``J_i x`` is
    ``y_u - y_l``; an equality row keeps its own dual.
    """
    from discopt.solvers import SolveStatus
    from discopt.solvers.lp_simplex import SIMPLEX_AVAILABLE, solve_lp

    if not SIMPLEX_AVAILABLE:
        return None, "the Rust simplex binding is not available"
    Jr = sp.csr_matrix(J, dtype=np.float64)
    m = Jr.shape[0]
    lo_f = s_lo > -1e20
    hi_f = s_hi < 1e20
    eq = lo_f & hi_f & (s_lo == s_hi)
    up = hi_f & ~eq
    dn = lo_f & ~eq
    iu, idn, ie = np.flatnonzero(up), np.flatnonzero(dn), np.flatnonzero(eq)
    A_ub = sp.vstack([Jr[iu], -Jr[idn]], format="csr")
    b_ub = np.r_[s_hi[iu], -s_lo[idn]]
    res = solve_lp(
        np.asarray(grad, dtype=np.float64),
        A_ub=A_ub if b_ub.size else None,
        b_ub=b_ub if b_ub.size else None,
        A_eq=Jr[ie] if ie.size else None,
        b_eq=s_lo[ie] if ie.size else None,
        bounds=list(zip(xl, xu)),
    )
    if res.status != SolveStatus.OPTIMAL or res.objective is None or res.dual_values is None:
        return None, f"Rust simplex {res.status.value}"
    d = np.asarray(res.dual_values, dtype=np.float64).ravel()
    if d.shape != (iu.size + idn.size + ie.size,):
        return None, "Rust simplex dual has the wrong length"
    y = np.zeros(m)
    y[iu] += d[: iu.size]
    y[idn] -= d[iu.size : iu.size + idn.size]
    y[ie] = d[iu.size + idn.size :]
    return y, float(res.objective)


def _lp_dual(sf, J, xl, xu, s_lo, s_hi, grad):
    """A candidate row dual of the linearised node LP: ``(y, value, "")`` or
    ``(None, None, reason)``.

    Any ``y`` is a valid input to the certificate (weak duality), so the sources are
    tried in order of how good a dual they usually give, and none can make the bound
    wrong: HiGHS on the slack form ``sf``; HiGHS on a row-equilibrated copy, because
    HiGHS refuses ``|a_ij| >= 1e15`` outright (#1955's ``1e16`` row left the only node
    of a two-variable MINLP unbounded); then the in-house Rust simplex, which is also
    the only source where ``highspy`` is not installed (the browser build).
    """
    from discopt.solvers.lp_milp_highs import (
        HighsUnavailable,
        _new_highs,
        _pass_model,
        _status_name,
        require_highspy,
    )

    reasons = []
    try:
        highspy = require_highspy()
    except HighsUnavailable:
        highspy = None
        reasons.append("highspy is not installed")
    if highspy is not None:
        h = _new_highs(highspy, [("presolve", "off")])
        st, why = _pass_model(h, highspy, sf, integer=False)
        if st == highspy.HighsStatus.kError:
            reasons.append(f"HiGHS refused the LP ({why})")
        else:
            h.run()
            name = _status_name(h)
            if name == "kOptimal":
                y = np.asarray(h.getSolution().row_dual, dtype=np.float64)
                return y, float(h.getInfo().objective_function_value), ""
            reasons.append(f"HiGHS {name}")
        h, scale, name = _solve_row_equilibrated(highspy, _new_highs, J, xl, xu, s_lo, s_hi, grad)
        if name == "kOptimal":
            y = np.asarray(h.getSolution().row_dual, dtype=np.float64) / scale
            return y, float(h.getInfo().objective_function_value), ""
        reasons.append(f"row-equilibrated HiGHS {name}")
    y, val = _simplex_dual(J, xl, xu, s_lo, s_hi, grad)
    if y is not None:
        return y, val, ""
    reasons.append(str(val))
    return None, None, "linearised node LP: " + "; ".join(reasons)


def convex_node_safe_bound(
    evaluator,
    x: np.ndarray,
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    cl: np.ndarray,
    cu: np.ndarray,
    *,
    deadline: Optional[float] = None,
    budget: Optional[SafeBoundBudget] = None,
) -> tuple[Optional[float], str]:
    """A rigorous lower bound on a **convex** node NLP, from any point of its box.

    ``evaluator`` is the node's NLP evaluator (minimisation sense, the ``evaluate_*``
    interface), ``x`` the node solver's approximate optimum, ``node_lb``/``node_ub``
    the node box and ``cl``/``cu`` the constraint sides. Returns ``(bound, "")`` or
    ``(None, reason)``. ``None`` means no bound was proved; it is never a claim that
    the node is infeasible, and the caller must treat the node as unbounded below.

    The caller asserts that the model is certified convex; on a nonconvex model the
    linearisations are not outer approximations and the returned number means nothing.

    ``budget`` (one per solve) rations the expensive paths; see :class:`SafeBoundBudget`.
    """
    if budget is not None and budget.suspended:
        return None, (f"suspended after {budget.SUSPEND_AFTER} consecutive nodes without a bound")
    bound, why = _safe_bound(evaluator, x, node_lb, node_ub, cl, cu, deadline, budget)
    if budget is not None:
        budget.record(bound is not None)
    return bound, why


def _safe_bound(evaluator, x, node_lb, node_ub, cl, cu, deadline, budget):
    from discopt.solvers.lp_milp_highs import CERT_ABS, CERT_REL, INF, StdForm, certify_lp_dual

    lo = np.asarray(node_lb, dtype=np.float64).ravel()
    hi = np.asarray(node_ub, dtype=np.float64).ravel()
    n = lo.shape[0]
    xh = np.asarray(x, dtype=np.float64).ravel()
    if xh.shape != (n,) or not np.all(np.isfinite(xh)):
        return None, "node point is not a finite vector over the columns"
    # Any point of the box is a valid linearisation point; clipping costs nothing.
    xh = np.minimum(np.maximum(xh, lo), hi)

    f = float(evaluator.evaluate_objective(xh))
    grad = np.asarray(evaluator.evaluate_gradient(xh), dtype=np.float64).ravel()
    if not np.isfinite(f) or grad.shape != (n,) or not np.all(np.isfinite(grad)):
        return None, "objective or its gradient is not finite at the node point"

    cl = np.asarray(cl, dtype=np.float64).ravel()
    cu = np.asarray(cu, dtype=np.float64).ravel()
    m = cl.shape[0]
    if m:
        g = np.asarray(evaluator.evaluate_constraints(xh), dtype=np.float64).ravel()
        if hasattr(evaluator, "evaluate_sparse_jacobian"):
            J = _as_csr(evaluator.evaluate_sparse_jacobian(xh), m, n)
        else:
            J = _as_csr(evaluator.evaluate_jacobian(xh), m, n)
        if g.shape != (m,) or not np.all(np.isfinite(g)) or not np.all(np.isfinite(J.data)):
            return None, "constraints or their Jacobian are not finite at the node point"
    else:
        g = np.zeros(0)
        J = sp.csr_matrix((0, n), dtype=np.float64)

    # Linearised row r_i(x) = g_i + J_i (x - x̂) = J_i x + (g_i - J_i x̂), written as
    # J x - s = 0 with s in [cl - k, cu - k], k = g - J x̂. Each finite side is widened
    # by a slack (vectorised; a per-row Python loop sat on the per-node hot path): the
    # round-off of forming ``k``, the evaluator's few-ulp error in ``g_i`` and ``J_i``,
    # and that gradient error carried across the finite part of the box -- see the
    # module docstring for what this does and does not cover. Widening only enlarges
    # the slack's range, so it can only weaken the bound. Only the operands of the side
    # being widened are charged: an open side is the 1e20 sentinel, and charging its
    # round-off (8 * eps * 1e20 = 1.8e5) measured as an LP that ignored every row.
    xl = np.where(lo <= -_OPEN_SIDE, -INF, lo)
    xu = np.where(hi >= _OPEN_SIDE, INF, hi)
    width = np.where((xl > -INF) & (xu < INF), xu - xl, 0.0)
    Jx = J @ xh
    absJ = abs(J)
    k = g - Jx
    unit = ROUNDOFF_OPS * FLOAT_EPS
    base = unit * (np.abs(g) + absJ @ np.abs(xh) + absJ @ width)
    lo_ok = np.isfinite(cl) & (np.abs(cl) < INF)
    hi_ok = np.isfinite(cu) & (np.abs(cu) < INF)
    s_lo = np.where(lo_ok, cl - k - (base + unit * np.abs(np.where(lo_ok, cl, 0.0))), -INF)
    s_hi = np.where(hi_ok, cu - k + (base + unit * np.abs(np.where(hi_ok, cu, 0.0))), INF)
    obj_const = f - float(grad @ xh)
    obj_const -= unit * (abs(f) + float(np.abs(grad) @ np.abs(xh)) + float(np.abs(grad) @ width))

    A = sp.hstack([J.tocsc(), -sp.identity(m, format="csc")], format="csc")
    sf = StdForm.from_arrays(
        np.r_[grad, np.zeros(m)], A, np.zeros(m), np.r_[xl, s_lo], np.r_[xu, s_hi], obj_const
    )

    y, lp_val, why = _lp_dual(sf, J, xl, xu, s_lo, s_hi, grad)
    if y is None:
        return None, why
    if y.shape != (sf.m,) or not np.all(np.isfinite(y)):
        return None, "linearised node LP dual is not finite"
    lp_obj = lp_val + sf.obj_const
    bound, _prov, reason = certify_lp_dual(y, sf, lp_obj, deadline=deadline, exact=False)
    closed = bound is not None and lp_obj - bound <= CERT_ABS + CERT_REL * abs(lp_obj)
    if not closed and (budget is None or budget.exact_left > 0):
        # The cheap rungs fell short; the exact correction is the expensive one, so it
        # runs only while this solve's ration lasts.
        if budget is not None:
            budget.exact_left -= 1
        exact_bound, _prov, reason = certify_lp_dual(y, sf, lp_obj, deadline=deadline)
        if exact_bound is not None and (bound is None or exact_bound > bound):
            bound = exact_bound
    if bound is None or not np.isfinite(bound):
        return None, reason or "no finite Neumaier-Shcherbina bound from the LP dual"
    return float(bound), ""
