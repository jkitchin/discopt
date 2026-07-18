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
* :func:`classify_signomial_global` also accepts a signomial objective with
  **signomial inequality constraints** over the same positive box (issue #114
  follow-up): each body is normalised to ``s_i(u) <= 0`` and relaxed per node by
  its convex DC underestimator ``cv_i(u) <= s_i(u)`` (a valid outer
  approximation — the relaxed feasible region *contains* the true one), so the
  node bound stays a valid dual bound and a closed tree certifies the
  *constrained* optimum. Incumbents come from a true-problem local solve and are
  accepted only if genuinely feasible.
* :func:`classify_signomial_global` abstains (returns ``None``) on anything
  outside that class — integer variables, non-positive/unbounded box, a
  signomial *equality* or any non-signomial constraint body, a purely
  posynomial/GP program (the exact GP path owns it), maximisation, or a
  non-signomial objective — so the default solve path and the GP abstention are
  untouched.
* Constrained-node tightening (issue #741): every device stacked on the
  constrained relaxation is itself certified, so the node bound stays a valid
  dual bound on the *retained* set (all truly feasible points at least as good
  as the incumbent — a set that always contains the constrained optimum):
  certified log-domain OBBT with the incumbent objective cut
  (:func:`_obbt_tighten`, each coordinate bound proven by the Lagrangian
  corner mechanism, never by trusting the convex subsolver, and backed off by
  a numerical margin before use); a rigorous interval floor on the objective
  and on the fitted Lagrangian (:func:`_interval_min`,
  :func:`_interval_min_weighted`); monotone parent-bound inheritance (a child
  box is a subset of its parent's box, so the parent bound remains valid); and
  certified pruning (interval-positive constraint body, or an OBBT-emptied
  box, whose leaf keeps the objective-cut value as its certified bound). A
  tree whose every leaf is certified relaxed-infeasible with no feasible point
  found returns ``status="infeasible"`` with ``gap_certified=True`` — a
  rigorous infeasibility certificate. Branching and warm starts are pure
  efficiency heuristics (gap-guided coordinate choice, split at the node's
  relaxation point) and never affect validity.

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
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from discopt._jax.convexity.signomial import (
    SignomialForm,
    is_signomial,
)
from discopt.modeling.core import Model, SolveResult, VarType

# Absolute floor on a sub-box width (in log units) below which we stop splitting
# a dimension — protects against infinite subdivision on a degenerate axis.
_MIN_LOG_WIDTH = 1e-7


@dataclass
class SignomialGlobalStructure:
    """A recognised signomial program (box, or box + signomial constraints).

    ``terms`` / ``offsets`` are the objective's log-domain ``(sigma, log_c,
    exps)`` form and the ordered flat scalar offsets defining the ``u = log x``
    coordinate order. ``u_lb`` / ``u_ub`` are the log-box. ``offset_to_var`` maps
    each participating offset back to its model variable for recovering ``x`` in
    the result.

    ``constraint_terms`` is the list of signomial *inequality* constraint bodies,
    each already normalised to ``body(u) <= 0`` and expressed over the SAME
    unified ``offsets`` ordering as the objective (a ``>=`` row is sign-flipped on
    ingest). An empty list is the box-only class (unchanged legacy behaviour).
    ``offsets`` is the union of every offset appearing in the objective or any
    constraint, so ``u`` spans all participating variables.
    """

    terms: list
    offsets: list[int]
    u_lb: np.ndarray
    u_ub: np.ndarray
    offset_to_var: dict[int, tuple[str, int, int]]  # offset -> (name, size, local_idx)
    constraint_terms: list = field(default_factory=list)


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


def _terms_over_offsets(
    form: SignomialForm, index: dict[int, int], n: int, *, flip: bool = False
) -> list[tuple[float, float, np.ndarray]]:
    """Pack a :class:`SignomialForm` into ``(sigma, log_c, exps)`` over a FIXED
    unified offset ordering (``index`` maps flat offset -> column). ``flip=True``
    negates every term's sign (used to turn a ``body >= 0`` row into ``-body <=
    0``). Unlike :func:`signomial_dc_terms` the exponent columns are keyed on the
    shared model-wide ordering so the objective and every constraint share one
    ``u`` coordinate system.
    """
    sgn = -1.0 if flip else 1.0
    terms: list[tuple[float, float, np.ndarray]] = []
    for mono in form.monomials:
        sigma = sgn * (1.0 if mono.coeff > 0.0 else -1.0)
        log_c = math.log(abs(mono.coeff))
        exps = np.zeros(n, dtype=np.float64)
        for off, e in mono.exponents.items():
            if abs(e) > 1e-12:
                exps[index[off]] = e
        terms.append((sigma, log_c, exps))
    return terms


def classify_signomial_global(model: Model) -> Optional[SignomialGlobalStructure]:
    """Recognise a signomial program (box, or box + signomial inequalities).

    Abstains (``None``) unless *all* hold: a single MINIMISE objective; every
    variable continuous with a strictly-positive, finite ``[lb, ub]`` (so the
    ``u = log x`` box is bounded); the objective is a signomial
    (:func:`is_signomial`); **every constraint is a signomial inequality**
    (``<=`` / ``>=`` — an equality signomial or any non-signomial body ->
    abstain); and the program is genuinely mixed-sign *somewhere* — the objective
    is mixed-sign, or some normalised constraint body carries a negative term.
    A program with no negative term anywhere is a posynomial/GP that the exact GP
    path owns, so we abstain and leave it untouched.

    With no constraints this is the original box-only class (bound is the exact
    optimum). With signomial constraints each body is normalised to ``body <= 0``
    (a ``>=`` row is sign-flipped) and relaxed per node by its certified convex DC
    underestimator ``cv(u) <= body(u)`` — a valid *outer* approximation whose
    feasible region contains the true one — so the node bound is a valid dual
    bound and a closed tree certifies the constrained optimum.
    """
    from discopt.modeling.core import ObjectiveSense

    if model._objective is None:
        return None
    if model._objective.sense != ObjectiveSense.MINIMIZE:
        return None
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

    obj_form = is_signomial(model._objective.expression, model)
    if obj_form is None:
        return None

    # Parse every constraint as a signomial inequality normalised to body <= 0.
    con_forms: list[SignomialForm] = []
    # The program genuinely needs the signed-signomial DC treatment iff some body
    # (objective, or a normalised constraint) carries a *negative non-constant*
    # monomial — a real ``Pplus - Pminus`` difference. A negative *constant* is
    # just a shifted right-hand side and keeps the body a convex posynomial, so
    # a posynomial objective with only posynomial ``<=`` rows is a convex GP the
    # exact GP path owns; we abstain and leave it untouched.
    needs_dc = obj_form.is_mixed_sign
    for c in getattr(model, "_constraints", None) or []:
        # Only plain relational Constraints are in-class; anything else
        # (indicator, SOS, logical, ...) is out-of-class -> abstain.
        sense = getattr(c, "sense", None)
        body = getattr(c, "body", None)
        rhs = getattr(c, "rhs", None)
        if sense not in ("<=", ">=") or body is None:
            return None
        # Normalised model constraints keep rhs == 0.0; guard defensively.
        if rhs is not None and abs(float(rhs)) > 1e-15:
            return None
        cf = is_signomial(body, model)
        if cf is None:
            return None
        norm = cf if sense == "<=" else _flip_form(cf)  # normalise to body <= 0
        con_forms.append(norm)
        if _has_negative_nonconstant(norm):
            needs_dc = True

    if not needs_dc:
        return None  # pure posynomial/GP -> exact GP path owns it; abstain.

    # Unified offset ordering across the objective and every constraint.
    offset_set: set[int] = set(obj_form.variable_offsets())
    for cf in con_forms:
        offset_set |= cf.variable_offsets()
    offsets = sorted(offset_set)
    index = {off: j for j, off in enumerate(offsets)}
    n = len(offsets)

    obj_terms = _terms_over_offsets(obj_form, index, n)
    constraint_terms = [_terms_over_offsets(cf, index, n) for cf in con_forms]

    layout = _offset_layout(model)
    u_lb = np.array([math.log(layout[o][3]) for o in offsets], dtype=np.float64)
    u_ub = np.array([math.log(layout[o][4]) for o in offsets], dtype=np.float64)
    offset_to_var = {o: (layout[o][0], layout[o][1], layout[o][2]) for o in offsets}
    return SignomialGlobalStructure(
        terms=obj_terms,
        offsets=offsets,
        u_lb=u_lb,
        u_ub=u_ub,
        offset_to_var=offset_to_var,
        constraint_terms=constraint_terms,
    )


def _flip_form(form: SignomialForm) -> SignomialForm:
    """Return ``-form`` (every coefficient negated) as a new :class:`SignomialForm`.

    Used to normalise a ``body >= 0`` constraint into ``-body <= 0`` while keeping
    the canonical monomial structure.
    """
    from discopt._jax.convexity.posynomial import Monomial

    return SignomialForm([Monomial(-m.coeff, dict(m.exponents)) for m in form.monomials])


def _has_negative_nonconstant(form: SignomialForm) -> bool:
    """True iff ``form`` has a monomial with negative coefficient and a nonzero
    exponent (a genuine negative posynomial term, not a shifted constant)."""
    return any(
        m.coeff < 0.0 and any(abs(e) > 1e-12 for e in m.exponents.values()) for m in form.monomials
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


def _freeze_pack(pack, u_lb, u_ub):
    """Freeze a pack's DC underestimator on a FIXED box for fast evaluation.

    On a fixed ``[u_lb, u_ub]`` the secant part of ``cv = Pplus - SEC[Pminus]``
    is a fixed affine function of ``u``: each negative monomial's chord is
    ``e_lo + slope * (log_c + a . u - xi_lo)``, so the whole ``SEC`` collapses
    to ``sec_c + sec_w . u`` with box-only coefficients. Returns
    ``(lc_pos, a_pos, sec_c, sec_w)`` such that

        cv(u) = sum(exp(lc_pos + a_pos @ u)) - sec_c - sec_w . u

    — the SAME certified formula as :func:`_cv_and_grad`, just precomputed once
    per node instead of on every function evaluation inside the convex solves.
    """
    sig, log_c, exps = pack
    pos = sig > 0.0
    neg = ~pos
    lc_pos = log_c[pos]
    a_pos = exps[pos]
    sec_c = 0.0
    sec_w = np.zeros(u_lb.shape[0])
    if np.any(neg):
        a = exps[neg]
        lc = log_c[neg]
        pos_part = np.where(a >= 0.0, a, 0.0)
        neg_part = np.where(a < 0.0, a, 0.0)
        xi_lo = lc + pos_part @ u_lb + neg_part @ u_ub
        xi_hi = lc + pos_part @ u_ub + neg_part @ u_lb
        e_lo = np.exp(xi_lo)
        width = xi_hi - xi_lo
        safe = np.where(width > 1e-15, width, 1.0)
        slope = np.where(width > 1e-15, (np.exp(xi_hi) - e_lo) / safe, 0.0)
        sec_c = float(np.sum(e_lo + slope * (lc - xi_lo)))
        sec_w = a.T @ slope
    return lc_pos, a_pos, sec_c, sec_w


def _frozen_cached(frozen):
    """Value+gradient evaluator over a frozen pack with a 1-point memo.

    SLSQP evaluates objective/constraint values and jacobians through separate
    callbacks at the same point; the memo computes both once. Returns
    ``fg(u) -> (value, grad)``.
    """
    lc_pos, a_pos, sec_c, sec_w = frozen
    memo = {"key": None, "val": None, "grad": None}

    def fg(u):
        key = u.tobytes()
        if memo["key"] != key:
            m = np.exp(lc_pos + a_pos @ u)
            memo["key"] = key
            memo["val"] = float(np.sum(m)) - sec_c - float(sec_w @ u)
            memo["grad"] = a_pos.T @ m - sec_w
        return memo["val"], memo["grad"]

    return fg


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


# ──────────────────────────────────────────────────────────────────────
# Constrained node bound (signomial inequality constraints)
# ──────────────────────────────────────────────────────────────────────


def _constrained_node_bound(
    obj_pack, con_packs, u_lb, u_ub, *, tighten=False, u_hint=None, u_start=None
):
    """Rigorous dual bound on ``min S(x)  s.t.  s_i(x) <= 0`` over the ``u``-box.

    Each constraint body ``s_i`` and the objective ``S`` are replaced by their
    certified convex DC underestimators ``cv_i(u) <= s_i(u)`` and
    ``f(u) = cv_obj(u) <= S(u)`` (identical secant envelope as the box-only
    core). The relaxed node

        min_u  f(u)   s.t.  cv_i(u) <= 0,  u in box

    is convex (``f`` convex, each ``cv_i`` convex); its feasible region *contains*
    the true feasible region, so its optimum is ``<=`` the true node optimum.

    A **rigorous** bound is obtained without trusting the convex solver via
    Lagrangian duality: for ANY multipliers ``lam_i >= 0``,

        L(u) = f(u) + sum_i lam_i cv_i(u)                         (convex in u)

    satisfies, at the true node optimiser ``u*`` (feasible, ``cv_i(u*) <=
    s_i(u*) <= 0``),  ``L(u*) <= f(u*) <= S(u*)``.  The supporting hyperplane of
    the convex ``L`` at the solver's point, minimised in closed form over the box
    (an affine function is minimised at a corner), therefore under-estimates
    ``L`` on the whole box and is a valid dual bound **for any ``lam >= 0``** — a
    poor ``lam`` only loosens it, never invalidates it. We take the ``lam >= 0``
    that best fits the KKT stationarity ``grad f + sum lam_i grad cv_i ~ 0`` via
    non-negative least squares (a proxy for the true multipliers; still sound
    whatever it returns).

    With ``tighten=True`` (the #741 node path) the certificate is strengthened,
    still without ever trusting a solver: (a) if the midpoint SLSQP solve fails,
    it is retried from ``u_hint`` (e.g. the incumbent's point — feasible for the
    relaxation whenever it lies in the box, since ``cv_i <= s_i <= 0`` there);
    (b) for each candidate point the bound is the MAX of the supporting-
    hyperplane corner bound and the rigorous interval min of the same Lagrangian
    ``L = cv_obj + lam . cv`` (:func:`_interval_min_weighted`) — both valid for
    any ``lam >= 0``, so their max is valid. ``tighten=False`` keeps the
    pre-#741 single-hyperplane bound (the differential-test reference; same
    certified formula, evaluated through the per-node frozen packs).

    Returns ``(lb, u_star)``.
    """
    from scipy.optimize import minimize, nnls

    f_obj = _frozen_cached(_freeze_pack(obj_pack, u_lb, u_ub))
    f_cons = [_frozen_cached(_freeze_pack(p, u_lb, u_ub)) for p in con_packs]
    scipy_cons = [
        {"type": "ineq", "fun": (lambda u, fg=fg: -fg(u)[0]), "jac": (lambda u, fg=fg: -fg(u)[1])}
        for fg in f_cons
    ]

    def solve_from(start):
        try:
            res = minimize(
                lambda u: f_obj(u)[0],
                start,
                jac=lambda u: f_obj(u)[1],
                method="SLSQP",
                bounds=list(zip(u_lb, u_ub)),
                constraints=scipy_cons,
                options=dict(maxiter=80, ftol=1e-10),
            )
            return np.clip(res.x, u_lb, u_ub), bool(res.success)
        except Exception:
            return np.clip(start, u_lb, u_ub), False

    def certify(u_star):
        """Valid dual bound from the hyperplane corner (any ``lam >= 0``)."""
        f_val, g_obj = f_obj(u_star)
        if f_cons:
            evals = [fg(u_star) for fg in f_cons]
            Gc = np.array([g for _v, g in evals])  # (m, n) constraint jacs
            cvals = np.array([v for v, _g in evals])  # cv_i(u*)
            # lam >= 0 fitting grad f + Gc^T lam ~ 0  (KKT stationarity proxy).
            try:
                lam, _ = nnls(Gc.T, -g_obj)
            except Exception:
                lam = np.zeros(Gc.shape[0])
            L_val = f_val + float(lam @ cvals)
            L_grad = g_obj + Gc.T @ lam
        else:
            lam = np.zeros(0)
            L_val = f_val
            L_grad = g_obj
        corner = np.minimum(L_grad * (u_lb - u_star), L_grad * (u_ub - u_star))
        lb = L_val + float(np.sum(corner))
        if tighten:
            weighted = [(obj_pack, 1.0)] + [(p, float(w)) for p, w in zip(con_packs, lam)]
            lb = max(lb, _interval_min_weighted(weighted, u_lb, u_ub))
        return lb

    u0 = 0.5 * (u_lb + u_ub)
    if tighten and u_start is not None:
        # Warm start from the parent node's relaxation point (clipped into this
        # box): the child's active set is usually the parent's, so SLSQP
        # converges in far fewer iterations than from the midpoint. Start
        # choice never affects validity — only the certificate's tightness.
        u0 = np.clip(u_start, u_lb, u_ub)
    u_star, ok = solve_from(u0)
    lb = certify(u_star)
    if tighten and not ok and u_hint is not None:
        # A failed convex solve leaves an arbitrary point whose KKT fit (and so
        # the hyperplane) can be very loose; retry from the hint and keep the
        # better CERTIFIED bound (both are valid, so max is valid).
        u2, _ok2 = solve_from(np.clip(u_hint, u_lb, u_ub))
        lb2 = certify(u2)
        if lb2 > lb:
            lb, u_star = lb2, u2
    return lb, u_star


def _interval_min(sig, log_c, exps, u_lb, u_ub):
    """Rigorous interval lower bound on a signomial over the ``u``-box.

    Each monomial ``sigma * exp(log_c + a . u)`` is monotone in every ``u_j``
    (sign of ``a_j``), so its box extrema are attained at per-coordinate corners
    in closed form: positive terms contribute their box minimum, negative terms
    minus their box maximum. Pure interval arithmetic — no optimiser is trusted,
    so the result is a certified floor usable to (a) catch a failed convex solve
    in the Lagrangian corner bound and (b) prune a node whose constraint body is
    provably positive everywhere on the box.
    """
    pos_e = np.where(exps >= 0.0, exps, 0.0)
    neg_e = np.where(exps < 0.0, exps, 0.0)
    lin_lo = pos_e @ u_lb + neg_e @ u_ub  # min of a.u over the box
    lin_hi = pos_e @ u_ub + neg_e @ u_lb  # max of a.u over the box
    per_term = np.where(sig > 0.0, np.exp(log_c + lin_lo), -np.exp(log_c + lin_hi))
    return float(np.sum(per_term))


def _interval_min_weighted(weighted_packs, u_lb, u_ub):
    """Rigorous interval min over the box of ``L(u) = sum_i w_i * cv_i(u)``.

    ``weighted_packs`` is a list of ``(pack, w)`` with every ``w >= 0`` and
    ``cv_i`` the certified DC underestimator of pack ``i`` (positive exponential
    terms minus affine secant chords, exactly as :func:`_cv_and_grad`). Each
    positive term is minimised independently at its per-coordinate corner
    (closed form, monotone in each ``u_j``); the secant chords are affine, so
    they are accumulated into a single affine function and minimised exactly at
    its corner. The sum of independent minima under-estimates ``min_box L`` —
    pure interval arithmetic, no optimiser trusted. Complements the supporting-
    hyperplane corner bound: when the convex solve stalls at a poor point the
    hyperplane can be arbitrarily loose, while this floor cannot.
    """
    total = 0.0
    aff_c = 0.0
    aff_g = np.zeros_like(u_lb)
    for (sig, log_c, exps), w in weighted_packs:
        if w <= 0.0:
            continue
        pos = sig > 0.0
        neg = ~pos
        if np.any(pos):
            a = exps[pos]
            pos_e = np.where(a >= 0.0, a, 0.0)
            neg_e = np.where(a < 0.0, a, 0.0)
            lin_lo = pos_e @ u_lb + neg_e @ u_ub
            total += w * float(np.sum(np.exp(log_c[pos] + lin_lo)))
        if np.any(neg):
            a = exps[neg]
            lc = log_c[neg]
            pos_e = np.where(a >= 0.0, a, 0.0)
            neg_e = np.where(a < 0.0, a, 0.0)
            xi_lo = lc + pos_e @ u_lb + neg_e @ u_ub
            xi_hi = lc + pos_e @ u_ub + neg_e @ u_lb
            e_lo = np.exp(xi_lo)
            width = xi_hi - xi_lo
            safe = np.where(width > 1e-15, width, 1.0)
            slope = np.where(width > 1e-15, (np.exp(xi_hi) - e_lo) / safe, 0.0)
            # chord(u) = [e_lo + slope*(lc - xi_lo)] + slope * (a . u); the cv
            # carries -chord, weighted by w.
            aff_c -= w * float(np.sum(e_lo + slope * (lc - xi_lo)))
            aff_g -= w * (a.T @ slope)
    corner = np.minimum(aff_g * u_lb, aff_g * u_ub)
    return total + aff_c + float(np.sum(corner))


def _corner_lagrangian_bound(obj_val, obj_grad, con_vals, con_grads, u_lb, u_ub, u_star):
    """Shared rigorous-corner machinery: bound ``min_box L`` for a fitted ``lam >= 0``.

    Given a convex objective piece (value/gradient at ``u_star``) and convex
    constraint pieces ``c_j(u) <= 0`` (values/jacobians at ``u_star``), fit
    ``lam >= 0`` to KKT stationarity by non-negative least squares and return the
    box-corner minimum of the supporting hyperplane of ``L = obj + lam . c`` at
    ``u_star``. Valid for ANY ``lam >= 0`` (a poor fit only loosens the bound),
    exactly as in :func:`_constrained_node_bound`.
    """
    from scipy.optimize import nnls

    if len(con_grads):
        Gc = np.asarray(con_grads)
        try:
            lam, _ = nnls(Gc.T, -obj_grad)
        except Exception:
            lam = np.zeros(Gc.shape[0])  # lam = 0 stays valid, only looser
        L_val = obj_val + float(lam @ np.asarray(con_vals))
        L_grad = obj_grad + Gc.T @ lam
    else:
        L_val = obj_val
        L_grad = obj_grad
    corner = np.minimum(L_grad * (u_lb - u_star), L_grad * (u_ub - u_star))
    return L_val + float(np.sum(corner))


def _cert_min_linear(w, con_specs, u_lb, u_ub):
    """Certified lower bound on ``min w . u`` over the convex relaxed set.

    ``con_specs`` is a list of ``(pack, rhs)`` rows meaning ``cv_pack(u) <= rhs``
    (``rhs = 0`` for a relaxed signomial constraint; ``rhs = UB`` for the
    incumbent objective cut ``cv_obj(u) <= UB``). Solves the convex program with
    SLSQP for a good linearisation point, then certifies via the Lagrangian
    supporting-hyperplane corner bound — sound irrespective of SLSQP's accuracy
    or failure. Used by :func:`_obbt_tighten` with ``w = +/- e_i``.
    """
    from scipy.optimize import minimize

    f_cons = [(_frozen_cached(_freeze_pack(p, u_lb, u_ub)), r) for p, r in con_specs]
    scipy_cons = [
        {
            "type": "ineq",
            "fun": (lambda u, fg=fg, r=r: r - fg(u)[0]),
            "jac": (lambda u, fg=fg: -fg(u)[1]),
        }
        for fg, r in f_cons
    ]
    u0 = 0.5 * (u_lb + u_ub)
    try:
        res = minimize(
            lambda u: float(w @ u),
            u0,
            jac=lambda u: w,
            method="SLSQP",
            bounds=list(zip(u_lb, u_ub)),
            constraints=scipy_cons,
            options=dict(maxiter=15, ftol=1e-8),
        )
        u_star = np.clip(res.x, u_lb, u_ub)
    except Exception:
        u_star = u0  # hyperplane at any point stays valid, only looser
    con_vals = []
    con_grads = []
    for fg, rhs in f_cons:
        v, g = fg(u_star)
        con_vals.append(v - rhs)
        con_grads.append(g)
    return _corner_lagrangian_bound(
        float(w @ u_star), np.asarray(w, dtype=np.float64), con_vals, con_grads, u_lb, u_ub, u_star
    )


# Relative slack retained on every certified OBBT tightening (guards numerics —
# the certified value is backed off by this margin before it is applied).
_OBBT_MARGIN = 1e-9
# Stop iterating OBBT rounds once the best per-dimension width reduction of a
# round falls below this fraction.
_OBBT_MIN_SHRINK = 0.05


def _obbt_tighten(obj_pack, con_packs, u_lb, u_ub, incumbent, *, rounds):
    """Certified optimality/feasibility-based bound tightening of the ``u``-box.

    For each coordinate solves ``min +/- u_i`` over the node's convex DC
    relaxation intersected with the incumbent objective cut ``cv_obj(u) <=
    incumbent`` (valid: ``cv_obj <= S``, so every feasible point at least as
    good as the incumbent satisfies it), certifying each new bound with the
    Lagrangian corner mechanism of :func:`_cert_min_linear` — never trusting the
    convex solver. Because the DC secants tighten as the box shrinks, rounds are
    iterated (up to ``rounds``) while the box keeps contracting.

    Every certified bound is backed off by ``_OBBT_MARGIN`` before use, so the
    retained set — all truly feasible points with ``S <= incumbent`` — is a
    superset of what a tolerance-free tightening would keep. Returns the
    tightened ``(u_lb, u_ub)``, or ``None`` when some coordinate's certified
    range is empty: a rigorous proof that the node holds no feasible point with
    objective ``<= incumbent`` (no feasible point at all when the incumbent is
    infinite), so the node may be pruned.
    """
    lb = u_lb.copy()
    ub = u_ub.copy()
    n = lb.shape[0]
    for _ in range(max(rounds, 0)):
        best_shrink = 0.0
        for i in range(n):
            width0 = ub[i] - lb[i]
            if width0 < 1e-6:
                continue
            con_specs = [(p, 0.0) for p in con_packs]
            if math.isfinite(incumbent):
                cut = incumbent + _OBBT_MARGIN * max(1.0, abs(incumbent))
                con_specs.append((obj_pack, cut))
            for sign in (1.0, -1.0):
                w = np.zeros(n)
                w[i] = sign
                cert = _cert_min_linear(w, con_specs, lb, ub)
                if sign > 0:
                    cand = cert - _OBBT_MARGIN * max(1.0, abs(cert))
                    if cand > lb[i]:
                        lb[i] = cand
                else:
                    cand = -cert + _OBBT_MARGIN * max(1.0, abs(cert))
                    if cand < ub[i]:
                        ub[i] = cand
            if lb[i] > ub[i]:
                if lb[i] - ub[i] > 1e-9:
                    return None  # certified empty: prune the node
                # Sub-tolerance crossing: keep the full crossed sliver (the
                # certified retained set lies inside it) rather than collapsing
                # to a point — a strict superset, so nothing can be cut.
                lb[i], ub[i] = ub[i], lb[i]
            if width0 > 0.0:
                best_shrink = max(best_shrink, (width0 - (ub[i] - lb[i])) / width0)
        if best_shrink < _OBBT_MIN_SHRINK:
            break
    return lb, ub


def _branch_scores(packs, u, u_lb, u_ub):
    """Per-dimension attribution of the DC secant gap at ``u`` (branching guide).

    For every negative monomial in every pack, the node relaxation's only error
    is the secant-vs-exp gap ``chord(xi) - exp(xi)`` on that monomial's affine
    argument ``xi = log_c + a . u``. Each gap is attributed to the coordinates
    by their share ``|a_j| * width_j`` of the argument's box range — branching
    the top-scoring coordinate shrinks the gaps that actually loosen the bound
    (widest-dimension branching wastes splits on dimensions the relaxation is
    already exact in). A pure efficiency heuristic: any branching choice keeps
    every node bound valid.
    """
    n = u.shape[0]
    scores = np.zeros(n)
    width = u_ub - u_lb
    for sig, log_c, exps in packs:
        neg = sig < 0.0
        if not np.any(neg):
            continue
        a = exps[neg]
        lc = log_c[neg]
        pos_part = np.where(a >= 0.0, a, 0.0)
        neg_part = np.where(a < 0.0, a, 0.0)
        xi_lo = lc + pos_part @ u_lb + neg_part @ u_ub
        xi_hi = lc + pos_part @ u_ub + neg_part @ u_lb
        xi = lc + a @ u
        w = xi_hi - xi_lo
        safe = np.where(w > 1e-15, w, 1.0)
        slope = np.where(w > 1e-15, (np.exp(xi_hi) - np.exp(xi_lo)) / safe, 0.0)
        chord = np.exp(xi_lo) + slope * (xi - xi_lo)
        gap = np.maximum(chord - np.exp(xi), 0.0)
        contrib = np.abs(a) * width[None, :]
        denom = contrib.sum(axis=1)
        denom = np.where(denom > 0.0, denom, 1.0)
        scores += (gap / denom) @ contrib
    return scores


def _feasibility_start(con_packs, u_lb, u_ub, u0):
    """Phase-1 restoration: drive the true constraint violation to ~0.

    Minimises ``sum_i max(s_i(u), 0)^2`` (smooth where it matters) over the box
    with L-BFGS-B and returns the resulting point — a good *starting* point for
    the exact local solve in :func:`_true_feasible_value`. Purely heuristic:
    feasibility of any incumbent is still independently verified there, so this
    can only help incumbent recovery, never soundness.
    """
    from scipy.optimize import minimize

    def fg(u):
        tot = 0.0
        grad = np.zeros_like(u)
        for s, lc, ex in con_packs:
            m = s * np.exp(lc + ex @ u)
            v = float(np.sum(m))
            if v > 0.0:
                tot += v * v
                grad += 2.0 * v * (ex.T @ m)
        return tot, grad

    try:
        res = minimize(
            fg,
            u0,
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(u_lb, u_ub)),
            options=dict(maxiter=200),
        )
        return np.clip(res.x, u_lb, u_ub)
    except Exception:
        return u0


def _true_feasible_value(obj_pack, con_packs, u_lb, u_ub, u_init):
    """Find a genuinely feasible incumbent by a local solve of the TRUE problem.

    Runs a local NLP solve (SLSQP) of the *exact* (non-relaxed) signomial program
    in the log domain from ``u_init`` and returns ``(S(x), u)`` **only if the
    returned point satisfies every true constraint** ``s_i(x) <= tol`` — any
    genuinely feasible point is a valid upper bound regardless of whether the
    local solve is globally optimal. Returns ``(inf, None)`` when no feasible
    point is recovered, so a non-convex/infeasible node never fabricates an
    incumbent.
    """
    from scipy.optimize import minimize

    sig_o, lc_o, ex_o = obj_pack

    def make_true_fg(sig, lc, ex, flip):
        """Memoised (value, grad) of the exact signomial (or its negation)."""
        memo = {"key": None, "val": None, "grad": None}
        sgn = -1.0 if flip else 1.0

        def fg(u):
            key = u.tobytes()
            if memo["key"] != key:
                m = sig * np.exp(lc + ex @ u)
                memo["key"] = key
                memo["val"] = sgn * float(np.sum(m))
                memo["grad"] = sgn * (ex.T @ m)
            return memo["val"], memo["grad"]

        return fg

    obj_fg = make_true_fg(sig_o, lc_o, ex_o, flip=False)

    def f(u):
        return obj_fg(u)[0]

    def gf(u):
        return obj_fg(u)[1]

    # true constraint feasibility: -s_i(u) >= 0
    con_fgs = [make_true_fg(*p, flip=True) for p in con_packs]
    scipy_cons = [
        {"type": "ineq", "fun": (lambda u, fg=fg: fg(u)[0]), "jac": (lambda u, fg=fg: fg(u)[1])}
        for fg in con_fgs
    ]

    def attempt(start):
        try:
            res = minimize(
                f,
                start,
                jac=gf,
                method="SLSQP",
                bounds=list(zip(u_lb, u_ub)),
                constraints=scipy_cons,
                options=dict(maxiter=120, ftol=1e-11),
            )
        except Exception:
            return math.inf, None
        u = np.clip(res.x, u_lb, u_ub)
        # Verify TRUE feasibility (never trust the local solver's own report).
        for pack in con_packs:
            if _true_value(*pack, u) > 1e-6:
                return math.inf, None
        return f(u), u

    val, u = attempt(u_init)
    if u is None:
        # Retry from a phase-1 restoration point: on small-feasible-region
        # instances the direct local solve routinely stalls infeasible. The
        # returned incumbent is still independently feasibility-verified above.
        val, u = attempt(_feasibility_start(con_packs, u_lb, u_ub, u_init))
    return val, u


# OBBT effort: the root box is tightened to a fixpoint (secants tighten as the
# box shrinks, so rounds compound); descendant nodes get one refresh round.
_OBBT_ROOT_ROUNDS = 8
_OBBT_NODE_ROUNDS = 1
# Adaptive per-node OBBT (measured on the #741 probes — always-on triples the
# node cost for nothing on tight-box instances like ex3_1_2, while wide-box
# instances like ex7_2_3 need it): a node re-runs OBBT when its parent's round
# usefully shrank, when the incumbent improved (stronger objective cut), or
# when the box's max width fell below this fraction of the width at the last
# OBBT run (the secants have materially changed, so a retry can pay again).
_OBBT_RERUN_WIDTH_FRACTION = 0.5


def solve_signomial_global(
    model: Model,
    *,
    time_limit: Optional[float] = None,
    gap_tolerance: float = 1e-6,
    max_nodes: int = 100000,
    obbt: bool = True,
    **_ignored,
) -> Optional[SolveResult]:
    """Certified global solve of a signomial program (box or box+cons), or ``None``.

    Returns ``None`` (sound abstention) if ``model`` is not in the class
    recognised by :func:`classify_signomial_global`, leaving the default solve
    path untouched. Otherwise runs spatial branch-and-bound in the ``u = log x``
    domain: box-only nodes use the certified DC-envelope bound; nodes with
    signomial inequality constraints relax each body by its convex DC
    underestimator ``cv_i(u) <= 0`` (a valid *outer* approximation) and take a
    rigorous Lagrangian node bound, recovering incumbents by a true-problem local
    solve (only genuinely feasible points are accepted). Returns a
    :class:`SolveResult`; ``gap_certified=True`` only when the tree closes within
    ``gap_tolerance`` with a finite incumbent — never on a resource limit.

    ``obbt=True`` (default; issue #741) additionally tightens every constrained
    node by certified log-domain OBBT with the incumbent objective cut
    (:func:`_obbt_tighten`), floors each node bound by the rigorous interval
    bound of the objective, and inherits the parent's bound (a child box is a
    subset of its parent's, so the parent bound remains valid — bounds become
    monotone down the tree). All three devices are certified tightenings of the
    same valid relaxation: no feasible point at least as good as the incumbent
    is ever cut. ``obbt=False`` preserves the pre-#741 constrained node
    relaxation exactly (the differential-test reference). Box-only programs are
    untouched by the flag either way.
    """
    struct = classify_signomial_global(model)
    if struct is None:
        return None

    t0 = time.perf_counter()
    sig, log_c, exps = _pack(struct.terms)
    obj_pack = (sig, log_c, exps)
    u_lb0, u_ub0 = struct.u_lb, struct.u_ub

    # Box-only (no constraints) keeps the original certified path exactly;
    # signomial constraints switch on the DC-outer-approximation node bound
    # (valid relaxation) + true-problem feasible incumbent recovery.
    constrained = bool(struct.constraint_terms)
    con_packs = [_pack(t) for t in struct.constraint_terms] if constrained else []

    incumbent = math.inf
    inc_u: Optional[np.ndarray] = None

    def register(u_star, true_val):
        nonlocal incumbent, inc_u
        if math.isfinite(true_val) and true_val < incumbent:
            incumbent = true_val
            inc_u = u_star

    def eval_node(alb, aub):
        """Box-only rigorous node bound (:func:`_node_lower_bound`), unchanged;
        the incumbent is ``S`` at the node point, always feasible."""
        lb, u_star, true_val = _node_lower_bound(sig, log_c, exps, alb, aub)
        register(u_star, true_val)
        return lb

    def eval_constrained(alb, aub, parent_lb, do_obbt=True, u_start=None):
        """Constrained node: certified bound + (optional) certified tightening.

        Returns ``(lb, alb, aub, u_star, shrank)`` with the box possibly
        OBBT-tightened (``u_star`` the relaxation's solve point, kept for
        gap-guided branching; ``shrank`` whether OBBT usefully contracted the
        box — descendants of a node whose OBBT was a no-op skip OBBT until the
        incumbent improves, a pure cost saving), or ``(lb, None, None, None,
        False)`` when the node is certified prunable — either relaxed-infeasible
        (``lb = inf``: the box holds no feasible point at all) or emptied by
        the incumbent objective cut (``lb`` = the cut value ``>= incumbent``:
        every feasible point in the box is worse than the incumbent, so the
        leaf's certified bound is the cut, never ``inf``). With ``obbt=False``
        this is exactly the pre-#741 node evaluation.
        """
        shrank = False
        if obbt:
            # Cheap certified infeasibility check on the TRUE constraint bodies
            # (pure interval arithmetic: provably positive on the whole box).
            for pack in con_packs:
                if _interval_min(*pack, alb, aub) > 1e-9:
                    return math.inf, None, None, None, False
            if do_obbt:
                rounds = _OBBT_ROOT_ROUNDS if parent_lb is None else _OBBT_NODE_ROUNDS
                tightened = _obbt_tighten(obj_pack, con_packs, alb, aub, incumbent, rounds=rounds)
                if tightened is None:
                    if math.isfinite(incumbent):
                        cut = incumbent + _OBBT_MARGIN * max(1.0, abs(incumbent))
                        return cut, None, None, None, False
                    return math.inf, None, None, None, False
                old_width = aub - alb
                alb, aub = tightened
                shrank = bool(np.any((aub - alb) < 0.99 * old_width))
        hint = inc_u if (obbt and inc_u is not None) else None
        lb, u_star = _constrained_node_bound(
            obj_pack, con_packs, alb, aub, tighten=obbt, u_hint=hint, u_start=u_start
        )
        if obbt:
            # Certified floor + monotone inheritance (child box ⊆ parent box, so
            # the parent bound stays valid): both guard against a failed convex
            # solve producing an arbitrarily loose Lagrangian corner bound.
            lb = max(lb, _interval_min(sig, log_c, exps, alb, aub))
            if parent_lb is not None:
                lb = max(lb, parent_lb)
        val, u_feas = _true_feasible_value(obj_pack, con_packs, alb, aub, u_star)
        if u_feas is not None:
            register(u_feas, val)
        return lb, alb, aub, u_star, shrank

    counter = itertools.count()
    nodes = 0
    min_fathomed_init = math.inf

    if constrained:
        # Root multistart to seed a feasible incumbent (a good bound is what lets
        # the gap close; a valid dual bound holds regardless of finding one).
        rng = np.random.default_rng(0)
        for _ in range(8):
            start = u_lb0 + rng.random(u_lb0.shape[0]) * (u_ub0 - u_lb0)
            val, u_feas = _true_feasible_value(obj_pack, con_packs, u_lb0, u_ub0, start)
            if u_feas is not None:
                register(u_feas, val)
        root_lb, rlb, rub, rstar, rshrank = eval_constrained(u_lb0, u_ub0, None)
        if rlb is None:
            # Root certified prunable: empty tree, the root's certified bound
            # (the objective cut, or +inf if truly infeasible) is the frontier.
            heap = []
            min_fathomed_init = root_lb
        else:
            state = (rshrank, float(np.max(rub - rlb)), incumbent)
            heap = [(root_lb, next(counter), rlb, rub, rstar, state)]
    else:
        root_lb = eval_node(u_lb0, u_ub0)
        state = (False, math.inf, incumbent)
        heap = [(root_lb, next(counter), u_lb0.copy(), u_ub0.copy(), None, state)]
    status = "optimal"
    # Rigorous global dual bound = minimum lower bound over the current B&B
    # *frontier* (leaves). A leaf is either still OPEN (in ``heap``) or has been
    # fathomed by bound (its ``lb`` proven >= incumbent - gap tolerance). We track
    # the smallest fathomed-leaf bound in ``min_fathomed``; combined with the open
    # heap minimum it is the valid global bound. (Reporting the last *popped*
    # node's bound instead understates it — sound but never certifying.)
    min_fathomed = min_fathomed_init
    limit_open_lb = math.inf  # bound of a node popped under a limit, not yet branched

    def gap_ok(lb: float) -> bool:
        # With no incumbent yet (constrained node found no feasible point) the gap
        # is undefined — never fathom, or an infinite incumbent would collapse the
        # whole tree. Box-only always has a finite root incumbent, so this guard is
        # a no-op there (behaviour unchanged).
        if not math.isfinite(incumbent):
            return False
        return incumbent - lb <= gap_tolerance * max(1.0, abs(incumbent)) + gap_tolerance

    while heap:
        lb, _, alb, aub, ustar, pstate = heapq.heappop(heap)
        pshrank, pobbt_w, pinc = pstate
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
        if constrained and obbt and ustar is not None:
            # Gap-guided branching: split the coordinate carrying the largest
            # DC secant-gap attribution at the node's relaxation point, and
            # split *at* that point (clipped inside the box) so both children
            # exclude it. Falls back to widest-dimension when the relaxation is
            # already gap-free at ``ustar``.
            scores = _branch_scores([obj_pack] + con_packs, ustar, alb, aub)
            usable = width >= _MIN_LOG_WIDTH
            if usable.any() and float(np.max(np.where(usable, scores, -1.0))) > 1e-12:
                j = int(np.argmax(np.where(usable, scores, -1.0)))
        if width[j] < _MIN_LOG_WIDTH:
            # Cannot refine further: this box is a point-leaf; its bound stands.
            min_fathomed = min(min_fathomed, lb)
            continue
        mid = 0.5 * (alb[j] + aub[j])
        if constrained and obbt and ustar is not None:
            mid = float(np.clip(ustar[j], alb[j] + 0.2 * width[j], aub[j] - 0.2 * width[j]))
        for take_upper in (False, True):
            clb = alb.copy()
            cub = aub.copy()
            if take_upper:
                clb[j] = mid
            else:
                cub[j] = mid
            if constrained:
                # Adaptive OBBT: re-run when the parent's round usefully shrank,
                # when the incumbent improved since the parent was evaluated (a
                # stronger objective cut), or when the box has contracted enough
                # since OBBT last ran that the secants materially changed.
                # Skipping never affects soundness, only cost.
                inc_improved = incumbent < pinc - 1e-9 * max(1.0, abs(incumbent))
                narrowed = float(np.max(cub - clb)) <= _OBBT_RERUN_WIDTH_FRACTION * pobbt_w
                do_obbt = pshrank or inc_improved or narrowed
                nlb, clb, cub, cstar, cshrank = eval_constrained(
                    clb, cub, lb, do_obbt=do_obbt, u_start=ustar
                )
                if clb is None:
                    # Certified pruned child: a fathomed leaf whose bound is the
                    # objective cut (finite) or +inf (relaxed-infeasible; drops
                    # out of the frontier minimum).
                    min_fathomed = min(min_fathomed, nlb)
                    continue
                cobbt_w = float(np.max(cub - clb)) if do_obbt else pobbt_w
                cstate = (cshrank, cobbt_w, incumbent)
            else:
                nlb = eval_node(clb, cub)
                cstar = None
                cstate = (False, math.inf, incumbent)
            if gap_ok(nlb):
                min_fathomed = min(min_fathomed, nlb)  # fathomed leaf
            else:
                heapq.heappush(heap, (nlb, next(counter), clb, cub, cstar, cstate))

    # Global dual bound = minimum over the frontier: open heap leaves, the
    # smallest fathomed leaf, and any limit-interrupted open node.
    open_min = min((h[0] for h in heap), default=math.inf)
    global_lb = min(open_min, min_fathomed, limit_open_lb)
    if not math.isfinite(global_lb):
        global_lb = root_lb

    wall = time.perf_counter() - t0
    finite = math.isfinite(incumbent)
    if finite:
        # Structural certificate invariant ``bound <= incumbent``. Unreachable
        # when every certified device is exact; pure numerical armour against a
        # frontier emptied at the objective-cut value (cut = incumbent + margin).
        global_lb = min(global_lb, incumbent)
    infeasible_certified = bool(
        constrained
        and not finite
        and status == "optimal"
        and not heap
        and not math.isfinite(min_fathomed)
        and not math.isfinite(limit_open_lb)
    )
    if infeasible_certified:
        # Every leaf was certified relaxed-infeasible (interval / OBBT proof,
        # never a trusted solver) and no feasible point exists: a rigorous
        # infeasibility certificate (no resource limit was hit).
        return SolveResult(
            status="infeasible",
            objective=None,
            bound=None,
            gap=None,
            x=None,
            node_count=nodes,
            wall_time=wall,
            convex_fast_path=False,
            gap_certified=True,
            _model=model,
        )
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
