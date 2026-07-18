"""Certified global solver for a MIXED-SIGN signomial box program (issue #114).

Scope (deliberately narrow, sound-by-construction)
--------------------------------------------------
This module globally solves a *single* class:

    minimise  S(x) = sum_k c_k * prod_j x_j^{a_kj}        (some c_k < 0)
    subject to        x_j in [lb_j, ub_j],   0 < lb_j <= ub_j < inf,

i.e. a **mixed-sign signomial minimised over a strictly-positive, bounded box,
with no constraints beyond the variable bounds**. Such a program is non-convex
and has no exact ``y = log x`` convex reformulation (``log`` of a mixed-sign sum
is not convex), so the shipped GP path correctly *abstains* on it
(:func:`discopt.gp.classify_gp` returns ``None`` on any negative coefficient).
This module is the opt-in global scheme that certifies that class.

Method — spatial branch-and-bound on the CERTIFIED log-domain DC envelope
-------------------------------------------------------------------------
Lift ``u_j = log x_j``. On any sub-box ``[u_lb, u_ub]`` the certified
difference-of-convex envelope of
:func:`discopt._jax.symbolic.signed_signomial.signed_signomial_dc_envelope`
gives a **convex** underestimator ``cv(u) <= S(x)`` (``cv = Pplus - SEC[Pminus]``,
``Pplus`` the convex positive posynomial part, ``SEC[Pminus]`` the affine secant
overestimator of the negative part). A rigorous lower bound on ``min_box S`` is
then obtained without trusting the numerical optimiser: at any point ``u0`` the
supporting hyperplane of the convex ``cv``,

    h(u) = cv(u0) + grad cv(u0) . (u - u0)  <=  cv(u)  <=  S(exp(u)),

underestimates ``cv`` on the whole box, so its box minimum (closed form — an
affine function is minimised at a box corner)

    LB = cv(u0) + sum_j min( g_j (u_lb_j - u0_j), g_j (u_ub_j - u0_j) )

is a valid dual bound on ``min_box S`` **for any** ``u0`` (the hyperplane bound
holds regardless of optimiser accuracy; a poor ``u0`` only loosens it, never
invalidates it). Evaluating the *true* ``S`` at ``exp(u0)`` — a point inside the
box — gives a valid feasible upper bound (incumbent). Branching bisects the
widest ``u``-dimension; because ``SEC[Pminus] -> Pminus`` as the box shrinks, the
DC gap closes and the scheme converges.

Soundness contract
------------------
* Every node bound is ``<= min_box S`` (a valid dual bound); every incumbent is
  ``S`` at a genuinely feasible box point (a valid primal bound). The reported
  ``bound`` is the minimum open-node dual bound, so ``bound <= optimum <=
  objective`` always holds.
* ``gap_certified=True`` is returned **only** when the tree closes to within the
  gap tolerance with a finite incumbent — never on a node/time limit. On a limit
  the valid (but not yet tight) bound is returned with ``gap_certified=False``: a
  sound under-claim, never a false certificate.
* :func:`classify_signomial_global` abstains (returns ``None``) on anything
  outside the class above — integer variables, non-positive/unbounded box, extra
  constraints, maximisation, non-signomial or single-sign objective — so the
  default solve path and the GP abstention are untouched.

This is opt-in (``DISCOPT_SGO``; default OFF) and general (keyed only on model
structure, never on instance names), per the repo's bound-changing-flag policy.

References
----------
* Lundell, A. & Westerlund, T. Signomial global optimization (SGO).
* Maranas, C. D. & Floudas, C. A. (1997). Global optimization in generalized
  geometric programming. *Comput. Chem. Eng.* 21(4), 351-369.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from discopt._jax.convexity.signomial import is_signomial, signomial_dc_terms
from discopt.modeling.core import Model, SolveResult, VarType

# Absolute floor on a sub-box width (in log units) below which we stop splitting
# a dimension — protects against infinite subdivision on a degenerate axis.
_MIN_LOG_WIDTH = 1e-7


@dataclass
class SignomialGlobalStructure:
    """A recognised mixed-sign signomial box program, ready for the SGO solve.

    ``terms`` / ``offsets`` are the log-domain ``(sigma, log_c, exps)`` form and
    the ordered flat scalar offsets defining the ``u = log x`` coordinate order
    (as produced by :func:`signomial_dc_terms`). ``u_lb`` / ``u_ub`` are the
    log-box. ``offset_to_var`` maps each participating offset back to its model
    variable for recovering ``x`` in the result.
    """

    terms: list
    offsets: list[int]
    u_lb: np.ndarray
    u_ub: np.ndarray
    offset_to_var: dict[int, tuple[str, int, int]]  # offset -> (name, size, local_idx)


def _offset_layout(model: Model) -> dict[int, tuple[str, int, int, float, float]]:
    """Map each flat scalar offset to ``(name, size, local_idx, lb, ub)``."""
    layout: dict[int, tuple[str, int, int, float, float]] = {}
    offset = 0
    for v in model._variables:
        lb = np.asarray(v.lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(v.ub, dtype=np.float64).reshape(-1)
        for k in range(v.size):
            layout[offset + k] = (v.name, v.size, k, float(lb[k]), float(ub[k]))
        offset += v.size
    return layout


def classify_signomial_global(model: Model) -> Optional[SignomialGlobalStructure]:
    """Recognise a mixed-sign signomial box program, else return ``None``.

    Abstains (``None``) unless *all* hold: a single MINIMISE objective; every
    variable continuous with a strictly-positive, finite ``[lb, ub]``; the
    objective is a signomial (:func:`is_signomial`) that is genuinely mixed-sign;
    and there are **no** constraints beyond the variable bounds. The strict
    positivity and finiteness give a bounded ``u = log x`` box; the box-only
    restriction keeps the returned bound an exact optimum of the stated program
    (extra constraints would make the box bound only a relaxation, so we abstain
    rather than risk over-claiming).
    """
    from discopt.modeling.core import ObjectiveSense

    if model._objective is None:
        return None
    if model._objective.sense != ObjectiveSense.MINIMIZE:
        return None
    if getattr(model, "_constraints", None):
        return None  # box-only class; any extra constraint -> abstain
    for v in model._variables:
        if v.var_type != VarType.CONTINUOUS:
            return None
        lb = np.asarray(v.lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(v.ub, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
            return None
        if np.any(lb <= 0.0):
            return None
        if np.any(ub < lb):
            return None
    form = is_signomial(model._objective.expression, model)
    if form is None or not form.is_mixed_sign:
        return None

    terms, offsets = signomial_dc_terms(form)
    layout = _offset_layout(model)
    u_lb = np.array([math.log(layout[o][3]) for o in offsets], dtype=np.float64)
    u_ub = np.array([math.log(layout[o][4]) for o in offsets], dtype=np.float64)
    offset_to_var = {o: (layout[o][0], layout[o][1], layout[o][2]) for o in offsets}
    return SignomialGlobalStructure(
        terms=terms,
        offsets=offsets,
        u_lb=u_lb,
        u_ub=u_ub,
        offset_to_var=offset_to_var,
    )


# ──────────────────────────────────────────────────────────────────────
# Numeric core (plain numpy; the certified envelope is re-expressed here in
# numpy for speed — it is the SAME formula as
# ``signed_signomial.signed_signomial_dc_envelope(overestimator="secant")``,
# validated equal in the regression test).
# ──────────────────────────────────────────────────────────────────────


def _pack(terms):
    sig = np.array([t[0] for t in terms], dtype=np.float64)
    log_c = np.array([t[1] for t in terms], dtype=np.float64)
    exps = np.array([np.asarray(t[2], dtype=np.float64) for t in terms], dtype=np.float64)
    return sig, log_c, exps


def _true_value(sig, log_c, exps, u):
    """Exact signomial value ``S(x)`` at ``x = exp(u)``."""
    return float(np.sum(sig * np.exp(log_c + exps @ u)))


def _cv_and_grad(sig, log_c, exps, u, u_lb, u_ub):
    """Convex underestimator ``cv(u)`` and its gradient on the box.

    ``cv = Pplus(u) - SEC[Pminus](u)`` with ``SEC`` the per-monomial affine
    secant (chord of ``exp`` over each monomial's affine-argument range on the
    box) — identical to
    :func:`discopt._jax.symbolic.signed_signomial._secant_overestimators`.
    Returns ``(cv_value, grad)`` with ``grad`` w.r.t. ``u``.
    """
    pos = sig > 0.0
    neg = ~pos
    # Pplus(u) and its gradient.
    m = np.exp(log_c + exps @ u)  # per-monomial exp value
    pplus = float(np.sum(m[pos]))
    grad_pplus = exps[pos].T @ m[pos] if np.any(pos) else np.zeros_like(u)
    # SEC[Pminus](u): sum of affine chords over negative monomials.
    sec = 0.0
    grad_sec = np.zeros_like(u)
    if np.any(neg):
        a_neg = exps[neg]
        lc_neg = log_c[neg]
        pos_part = np.where(a_neg >= 0.0, a_neg, 0.0)
        neg_part = np.where(a_neg < 0.0, a_neg, 0.0)
        lin_lo = pos_part @ u_lb + neg_part @ u_ub
        lin_hi = pos_part @ u_ub + neg_part @ u_lb
        xi_lo = lc_neg + lin_lo
        xi_hi = lc_neg + lin_hi
        xi = lc_neg + a_neg @ u
        e_lo = np.exp(xi_lo)
        e_hi = np.exp(xi_hi)
        width = xi_hi - xi_lo
        safe = np.where(width > 1e-15, width, 1.0)
        slope = np.where(width > 1e-15, (e_hi - e_lo) / safe, 0.0)
        chord = e_lo + slope * (xi - xi_lo)
        sec = float(np.sum(chord))
        grad_sec = a_neg.T @ slope
    cv = pplus - sec
    grad = grad_pplus - grad_sec
    return cv, grad


def _node_lower_bound(sig, log_c, exps, u_lb, u_ub):
    """Rigorous dual bound on ``min_box S`` plus an incumbent candidate.

    Minimise the convex ``cv`` over the box (L-BFGS-B); then take the supporting
    hyperplane of ``cv`` at the returned point and minimise *that* affine model
    over the box in closed form. The hyperplane underestimates the convex ``cv``
    for any point, so the result is a valid lower bound irrespective of the
    optimiser's accuracy. Returns ``(lb, u_star, true_value_at_u_star)``.
    """
    from scipy.optimize import minimize

    u0 = 0.5 * (u_lb + u_ub)

    def f(u):
        return _cv_and_grad(sig, log_c, exps, u, u_lb, u_ub)[0]

    def g(u):
        return _cv_and_grad(sig, log_c, exps, u, u_lb, u_ub)[1]

    res = minimize(
        f,
        u0,
        jac=g,
        method="L-BFGS-B",
        bounds=list(zip(u_lb, u_ub)),
        options=dict(maxiter=200, ftol=1e-13, gtol=1e-11),
    )
    u_star = np.clip(res.x, u_lb, u_ub)
    cv_val, grad = _cv_and_grad(sig, log_c, exps, u_star, u_lb, u_ub)
    # min over box of the affine support h(u) = cv_val + grad.(u - u_star)
    corner = np.minimum(grad * (u_lb - u_star), grad * (u_ub - u_star))
    lb = cv_val + float(np.sum(corner))
    true_val = _true_value(sig, log_c, exps, u_star)
    return lb, u_star, true_val


def solve_signomial_global(
    model: Model,
    *,
    time_limit: Optional[float] = None,
    gap_tolerance: float = 1e-6,
    max_nodes: int = 100000,
    **_ignored,
) -> Optional[SolveResult]:
    """Certified global solve of a mixed-sign signomial box program, or ``None``.

    Returns ``None`` (sound abstention) if ``model`` is not in the class
    recognised by :func:`classify_signomial_global`, leaving the default solve
    path untouched. Otherwise runs spatial branch-and-bound on the certified DC
    envelope and returns a :class:`SolveResult`; ``gap_certified=True`` only when
    the tree closes within ``gap_tolerance``.
    """
    struct = classify_signomial_global(model)
    if struct is None:
        return None

    t0 = time.perf_counter()
    sig, log_c, exps = _pack(struct.terms)
    u_lb0, u_ub0 = struct.u_lb, struct.u_ub

    incumbent = math.inf
    inc_u: Optional[np.ndarray] = None

    def register(u_star, true_val):
        nonlocal incumbent, inc_u
        if math.isfinite(true_val) and true_val < incumbent:
            incumbent = true_val
            inc_u = u_star

    root_lb, u_star, true_val = _node_lower_bound(sig, log_c, exps, u_lb0, u_ub0)
    register(u_star, true_val)

    counter = itertools.count()
    heap: list = [(root_lb, next(counter), u_lb0.copy(), u_ub0.copy())]
    nodes = 0
    status = "optimal"
    # Rigorous global dual bound = minimum lower bound over the current B&B
    # *frontier* (leaves). A leaf is either still OPEN (in ``heap``) or has been
    # fathomed by bound (its ``lb`` proven >= incumbent - gap tolerance). We track
    # the smallest fathomed-leaf bound in ``min_fathomed``; combined with the open
    # heap minimum it is the valid global bound. (Reporting the last *popped*
    # node's bound instead understates it — sound but never certifying.)
    min_fathomed = math.inf
    limit_open_lb = math.inf  # bound of a node popped under a limit, not yet branched

    def gap_ok(lb: float) -> bool:
        return incumbent - lb <= gap_tolerance * max(1.0, abs(incumbent)) + gap_tolerance

    while heap:
        lb, _, alb, aub = heapq.heappop(heap)
        if gap_ok(lb):
            # Heap is a min-heap: this is the smallest open bound and it already
            # closes the gap, so every remaining node does too. All become
            # fathomed leaves; the frontier min is this bound.
            min_fathomed = min(min_fathomed, lb)
            heap.clear()
            break
        if time_limit is not None and (time.perf_counter() - t0) > time_limit:
            status = "time_limit"
            limit_open_lb = lb  # this node stays an (unbranched) open leaf
            break
        if nodes >= max_nodes:
            status = "node_limit"
            limit_open_lb = lb
            break
        nodes += 1
        width = aub - alb
        j = int(np.argmax(width))
        if width[j] < _MIN_LOG_WIDTH:
            # Cannot refine further: this box is a point-leaf; its bound stands.
            min_fathomed = min(min_fathomed, lb)
            continue
        mid = 0.5 * (alb[j] + aub[j])
        for take_upper in (False, True):
            clb = alb.copy()
            cub = aub.copy()
            if take_upper:
                clb[j] = mid
            else:
                cub[j] = mid
            nlb, nu, nval = _node_lower_bound(sig, log_c, exps, clb, cub)
            register(nu, nval)
            if gap_ok(nlb):
                min_fathomed = min(min_fathomed, nlb)  # fathomed leaf
            else:
                heapq.heappush(heap, (nlb, next(counter), clb, cub))

    # Global dual bound = minimum over the frontier: open heap leaves, the
    # smallest fathomed leaf, and any limit-interrupted open node.
    open_min = min((h[0] for h in heap), default=math.inf)
    global_lb = min(open_min, min_fathomed, limit_open_lb)
    if not math.isfinite(global_lb):
        global_lb = root_lb

    wall = time.perf_counter() - t0
    finite = math.isfinite(incumbent)
    gap = None
    if finite:
        gap = (incumbent - global_lb) / max(1.0, abs(incumbent))
    # Certified-optimal iff no resource limit was hit and the closed-tree gap is
    # within the SAME tolerance used to fathom nodes (using a stricter check than
    # the prune rule would leave a fully-explored tree reporting "feasible"). A
    # limit interruption never certifies.
    certified = bool(finite and status == "optimal" and gap_ok(global_lb))
    if not certified and finite and status == "optimal":
        # Loop ended without a limit but the gap is not within tolerance
        # (numerical corner case): report feasible, not certified — never
        # over-claim optimality.
        status = "feasible"

    x_values = None
    if inc_u is not None:
        x_by_offset = {off: math.exp(float(inc_u[k])) for k, off in enumerate(struct.offsets)}
        # Assemble full x per variable (participating offsets carry the solution;
        # any variable absent from the objective takes its lower bound).
        layout = _offset_layout(model)
        by_name: dict[str, np.ndarray] = {}
        offset = 0
        for v in model._variables:
            arr = np.zeros(v.size, dtype=np.float64)
            for k in range(v.size):
                off = offset + k
                if off in x_by_offset:
                    arr[k] = x_by_offset[off]
                else:
                    arr[k] = layout[off][3]  # lb (not in objective; any feasible)
            by_name[v.name] = arr
            offset += v.size
        x_values = by_name

    return SolveResult(
        status=status,
        objective=incumbent if finite else None,
        bound=global_lb if finite else None,
        gap=gap if certified else (gap if finite else None),
        x=x_values,
        node_count=nodes,
        wall_time=wall,
        convex_fast_path=False,
        gap_certified=certified,
        _model=model,
    )


__all__ = [
    "SignomialGlobalStructure",
    "classify_signomial_global",
    "solve_signomial_global",
]
