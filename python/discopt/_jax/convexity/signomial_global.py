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


def _constrained_node_bound(obj_pack, con_packs, u_lb, u_ub):
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

    Returns ``(lb, u_star)``.
    """
    from scipy.optimize import minimize, nnls

    sig_o, lc_o, ex_o = obj_pack

    def f(u):
        return _cv_and_grad(sig_o, lc_o, ex_o, u, u_lb, u_ub)[0]

    def gf(u):
        return _cv_and_grad(sig_o, lc_o, ex_o, u, u_lb, u_ub)[1]

    def make_con(pack):
        s, lc, ex = pack

        def cf(u):
            return _cv_and_grad(s, lc, ex, u, u_lb, u_ub)[0]

        def cj(u):
            return _cv_and_grad(s, lc, ex, u, u_lb, u_ub)[1]

        return cf, cj

    cons = [make_con(p) for p in con_packs]
    scipy_cons = [
        {"type": "ineq", "fun": (lambda u, cf=cf: -cf(u)), "jac": (lambda u, cj=cj: -cj(u))}
        for cf, cj in cons
    ]
    u0 = 0.5 * (u_lb + u_ub)
    res = minimize(
        f,
        u0,
        jac=gf,
        method="SLSQP",
        bounds=list(zip(u_lb, u_ub)),
        constraints=scipy_cons,
        options=dict(maxiter=200, ftol=1e-12),
    )
    u_star = np.clip(res.x, u_lb, u_ub)

    g_obj = gf(u_star)
    if cons:
        Gc = np.array([cj(u_star) for _cf, cj in cons])  # (m, n) constraint jacs
        cvals = np.array([cf(u_star) for cf, _cj in cons])  # cv_i(u*)
        # lam >= 0 fitting grad f + Gc^T lam ~ 0  (KKT stationarity proxy).
        lam, _ = nnls(Gc.T, -g_obj)
        L_val = f(u_star) + float(lam @ cvals)
        L_grad = g_obj + Gc.T @ lam
    else:
        L_val = f(u_star)
        L_grad = g_obj
    corner = np.minimum(L_grad * (u_lb - u_star), L_grad * (u_ub - u_star))
    lb = L_val + float(np.sum(corner))
    return lb, u_star


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

    def f(u):
        return _true_value(sig_o, lc_o, ex_o, u)

    def gf(u):
        m = np.exp(lc_o + ex_o @ u)
        return ex_o.T @ (sig_o * m)

    def make_con(pack):
        s, lc, ex = pack

        def cf(u):  # true constraint feasibility: -s_i(u) >= 0
            return -_true_value(s, lc, ex, u)

        def cj(u):
            m = np.exp(lc + ex @ u)
            return -(ex.T @ (s * m))

        return cf, cj

    cons = [make_con(p) for p in con_packs]
    scipy_cons = [{"type": "ineq", "fun": cf, "jac": cj} for cf, cj in cons]
    res = minimize(
        f,
        u_init,
        jac=gf,
        method="SLSQP",
        bounds=list(zip(u_lb, u_ub)),
        constraints=scipy_cons,
        options=dict(maxiter=300, ftol=1e-12),
    )
    u = np.clip(res.x, u_lb, u_ub)
    # Verify TRUE feasibility (never trust the local solver's own report).
    for pack in con_packs:
        if _true_value(*pack, u) > 1e-6:
            return math.inf, None
    return f(u), u


def solve_signomial_global(
    model: Model,
    *,
    time_limit: Optional[float] = None,
    gap_tolerance: float = 1e-6,
    max_nodes: int = 100000,
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

    def eval_node(alb, aub, u_init=None):
        """Rigorous node lower bound; also register any feasible incumbent found.

        Box-only nodes reuse :func:`_node_lower_bound` (the incumbent is ``S`` at
        the node point, always feasible). Constrained nodes take the valid
        DC-relaxation bound from :func:`_constrained_node_bound` and recover a
        genuinely feasible incumbent via :func:`_true_feasible_value`.
        """
        if constrained:
            lb, u_star = _constrained_node_bound(obj_pack, con_packs, alb, aub)
            start = u_star if u_init is None else u_init
            val, u_feas = _true_feasible_value(obj_pack, con_packs, alb, aub, start)
            if u_feas is not None:
                register(u_feas, val)
            return lb
        lb, u_star, true_val = _node_lower_bound(sig, log_c, exps, alb, aub)
        register(u_star, true_val)
        return lb

    if constrained:
        # Root multistart to seed a feasible incumbent (a good bound is what lets
        # the gap close; a valid dual bound holds regardless of finding one).
        rng = np.random.default_rng(0)
        for _ in range(8):
            start = u_lb0 + rng.random(u_lb0.shape[0]) * (u_ub0 - u_lb0)
            val, u_feas = _true_feasible_value(obj_pack, con_packs, u_lb0, u_ub0, start)
            if u_feas is not None:
                register(u_feas, val)

    root_lb = eval_node(u_lb0, u_ub0)

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
        # With no incumbent yet (constrained node found no feasible point) the gap
        # is undefined — never fathom, or an infinite incumbent would collapse the
        # whole tree. Box-only always has a finite root incumbent, so this guard is
        # a no-op there (behaviour unchanged).
        if not math.isfinite(incumbent):
            return False
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
            nlb = eval_node(clb, cub)
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
