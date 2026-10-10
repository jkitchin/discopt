"""Sound box-local convexity certificate.

The certificate answers the question "is ``f`` convex on the given
box?" with a proof, leveraging:

1. :mod:`interval_ad` for a sound interval enclosure of the Hessian
   over the box.
2. :mod:`eigenvalue` for a sound lower bound on the minimum
   eigenvalue across every concrete Hessian in that enclosure.

If the lower eigenvalue bound is ≥ 0 on the box, ``f`` is convex
there (second-order sufficient condition, Boyd & Vandenberghe §3.1.4).
Symmetrically, an upper bound ≤ 0 proves concavity. Any other
outcome returns ``None`` — a conservative abstention, not a claim
of nonconvexity.

This routine never loosens a verdict from the syntactic walker
:mod:`rules`. Callers combine the two sources by preferring the
syntactic CONVEX/CONCAVE (cheaper) and only falling back to the
certificate when the syntactic walker says UNKNOWN.

References
----------
Boyd, Vandenberghe (2004), *Convex Optimization*, §3.1.4.
Adjiman, Dallwig, Floudas, Neumaier (1998), "αBB — I. Theoretical
  advances," Comput. Chem. Eng. — the interval-Hessian foundation
  this certificate operationalises.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import time
from typing import Iterator, Optional

import numpy as np

from discopt.modeling.core import Constraint, Expression, Model

from .eigenvalue import (
    gershgorin_certifies_nsd,
    gershgorin_certifies_psd,
    psd_2x2_sufficient,
    psd_decision_slack,
)
from .interval import Interval
from .interval_ad import interval_hessian
from .lattice import Curvature

logger = logging.getLogger(__name__)

# The slack for accepting "λ_min ≥ 0" despite floating-point slop lives in
# ``eigenvalue.psd_decision_slack`` and scales with the Hessian's magnitude
# (#1397). It was an absolute ``_PSD_TOL = 1e-10`` here, on the reasoning that
# "the interval Hessian already outward-rounds, so genuine zero eigenvalues may
# appear as small negatives; a very tight tolerance suffices" — true, except that
# "small" is relative to ‖H‖. The outward rounding widens by O(u·‖H‖), so at
# ‖H‖ ~ 1e12 a genuine zero eigenvalue appears as ~-1e-4 and the certificate was
# silently lost; and at ‖H‖ ~ 1e-2 an absolute 1e-10 admitted a relative
# nonconvexity of 1e-8. Both directions are measured in ``psd_decision_slack``.


def _qp_exact_convexity_enabled() -> bool:
    """Whether the exact QP/MIQP objective-Hessian convexity route is active.

    **Default ON** with a ``DISCOPT_QP_EXACT_CONVEXITY=0`` opt-out, graduated by
    the issue-#936 differential panel (see that issue and
    ``docs/dev/certification-gap-plan.md``).

    The route certifies the objective of a model the *problem classifier* proves
    to be a QP/MIQP (exactly-quadratic objective over a polyhedron) from the
    exact Hessian the classifier's own extractor already produces, instead of
    asking the scalar interval-Hessian walker — which cannot even parse the
    vectorized / indexed-summation modeling API, so no model written that way
    could ever be certified convex.
    """
    return os.environ.get("DISCOPT_QP_EXACT_CONVEXITY", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _convex_charge_enabled() -> bool:
    """Whether a solve may accept a QP objective as "convex up to a charge" (#1682).

    **Default ON** (graduated on introduction; opt-out ``DISCOPT_CONVEX_CHARGE=0``
    restores the strict #1679 refusal). The CLAUDE.md §5 panel is recorded in
    ``docs/dev/flag-retirement-audit.md``: over 299 corpus QP/MIQP instances only 5
    (QPLIB 10046/10048/10050/10056/10066) trigger it; on those plus the #1682
    Gram least-squares example, 2 reps x 60 s interleaved, certified 0/6 -> 3/6,
    every bound tightened and none above its reference optimum, 0 violations.

    When :func:`~.eigenvalue.psd_certified` cannot prove the objective Hessian
    ``Q`` PSD -- the float Gram matrix ``2 K'K`` of a rank-deficient ``K`` is
    indefinite in exact arithmetic at ``lambda_min ~ -1e-14`` (#1679) -- the
    exact-QP route may still accept it, inside a solve only, when
    :func:`~.eigenvalue.rigorous_psd_shift` proves ``lambda_min(Q) >= -delta`` and
    the charge ``delta/2 * D**2`` over the bounded box of the quadratic variables
    is at most :data:`CONVEX_CHARGE_ABS_TOL_FRACTION` of the solve's absolute gap
    tolerance. The solve then subtracts the charge from its published bound
    (``solver._apply_convexity_charge``). See :func:`quadratic_objective_charge`.
    """
    return os.environ.get("DISCOPT_CONVEX_CHARGE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


#: A charge is accepted only when it is at most this fraction of the solve's
#: absolute gap tolerance, so subtracting it from the published bound leaves the
#: certificate room to close (#1682). The published pair is re-judged after the
#: subtraction (``solver._refuse_unclosed_published_pair``) either way.
CONVEX_CHARGE_ABS_TOL_FRACTION = 0.1

#: A bound at or beyond this magnitude is the solver's effective infinity
#: (``DEFAULT_VARIABLE_BOUND = 9.999e19``; the LP layer's ``1e20`` sentinel).
_CHARGE_INF_BOUND = 1e19

#: Relative/absolute widening of every box width in the charge's diameter, so a
#: point the solver returns up to this far outside its bounds is still covered.
_CHARGE_BOX_SLACK = 1e-6


class ConvexityChargeScope:
    """Per-solve record of accepted convexity charges (#1682).

    Created by ``solver._stamp_layer_timing`` for each ``solve_model`` call;
    ``abs_gap_tol`` is filled in by ``solve_model`` once resolved. ``charges`` maps
    ``id(model)`` to the largest charge accepted for that model object -- the same
    model re-classified (a nested solve clears the memo) is not double counted;
    distinct models (reformulations) add up, which can only over-subtract.
    """

    __slots__ = ("abs_gap_tol", "charges")

    def __init__(self) -> None:
        self.abs_gap_tol: Optional[float] = None
        self.charges: dict[int, float] = {}

    def record(self, model: Model, charge: float) -> None:
        key = id(model)
        self.charges[key] = max(self.charges.get(key, 0.0), float(charge))

    @property
    def total(self) -> float:
        return float(sum(self.charges.values()))


#: The active solve's charge scope, or ``None`` outside a solve. A ContextVar so
#: the large-stack worker of ``_scoped_deep_recursion`` (which runs on a COPY of
#: the caller's context) shares the same mutable scope object.
_CHARGE_SCOPE: contextvars.ContextVar[Optional[ConvexityChargeScope]] = contextvars.ContextVar(
    "discopt_convexity_charge_scope", default=None
)


@contextlib.contextmanager
def convexity_charge_scope() -> Iterator[ConvexityChargeScope]:
    """Open a fresh charge scope for one solve; nested solves get their own.

    On exit a nested scope's charges are also merged (per-model maximum) into the
    enclosing scope: the outer solve may reuse a convexity verdict memoized while
    the nested solve ran, and must then still subtract the charge. If the outer
    result was built from the nested (already charged) bound, that subtracts the
    charge twice -- an over-subtraction, which is sound.
    """
    parent = _CHARGE_SCOPE.get()
    scope = ConvexityChargeScope()
    token = _CHARGE_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _CHARGE_SCOPE.reset(token)
        if parent is not None:
            for key, charge in scope.charges.items():
                parent.charges[key] = max(parent.charges.get(key, 0.0), charge)


def set_charge_scope_abs_gap_tol(abs_gap_tol: float) -> None:
    """Tell the active scope (if any) the solve's absolute gap tolerance."""
    scope = _CHARGE_SCOPE.get()
    if scope is not None:
        scope.abs_gap_tol = float(abs_gap_tol)


def quadratic_objective_charge(hessian: np.ndarray, x_l, x_u) -> Optional[float]:
    """Rigorous charge for treating ``1/2 x'Qx + c'x`` as convex on a box (#1682).

    Let ``S`` be the symmetric part of ``hessian`` restricted to its active
    (nonzero) rows, ``lambda_min(S) >= -delta`` proved by
    :func:`~.eigenvalue.rigorous_psd_shift`, and ``F`` any feasible set inside the
    box ``[l, u]`` of those variables. For any ``x_hat`` and ``y`` in the box, the
    quadratic is its own second-order Taylor expansion:

        ``f(y) = f(x_hat) + grad f(x_hat)'(y - x_hat) + 1/2 (y - x_hat)' S (y - x_hat)
               >= f(x_hat) + grad f(x_hat)'(y - x_hat) - delta/2 ||y - x_hat||^2
               >= [f(x_hat) + grad f(x_hat)'(y - x_hat)] - delta/2 D^2``

    with ``D^2 = sum_j (u_j - l_j)^2`` over the active variables (inactive ones do
    not enter the quadratic term). The bracket is exactly the first-order model a
    convex method's lower bound rests on: at an exact KKT point over polyhedral
    ``F`` it is ``>= f(x_hat)`` for every ``y in F``; at the approximate point a
    solver returns it is ``>= f(x_hat) - eps(x_hat)``, ``eps`` the same first-order
    (Frank-Wolfe) residual the convex path already has to account for on a
    genuinely convex objective, and a linearization cut or a Lagrangian
    linearization bound is the same bracket again. So every lower bound the convex
    machinery derives on ``F`` -- or on any node's sub-box, which only shrinks
    ``D`` -- overstates the true one by at most ``delta/2 D^2`` beyond what it
    would on a convex objective, and subtracting that charge from the published
    bound makes it valid again.

    Each width is widened by ``2 * 1e-6 * (1 + |l_j| + |u_j|)`` so a returned point
    up to that far outside its bounds is covered, and the result is doubled as a
    margin for the rounding in computing ``D^2`` itself.

    Returns ``None`` -- no acceptance -- when an active variable is unbounded
    (``|bound| >= 1e19``) or no shift is proved.
    """
    from .eigenvalue import rigorous_psd_shift

    Q = np.asarray(hessian, dtype=np.float64)
    S = 0.5 * (Q + Q.T)
    active = np.flatnonzero(np.any(S != 0.0, axis=1))
    if active.size == 0:
        return 0.0
    lo = np.asarray(x_l, dtype=np.float64).ravel()
    hi = np.asarray(x_u, dtype=np.float64).ravel()
    if lo.size != Q.shape[0] or hi.size != Q.shape[0]:
        return None
    lo, hi = lo[active], hi[active]
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return None
    if np.any(np.abs(lo) >= _CHARGE_INF_BOUND) or np.any(np.abs(hi) >= _CHARGE_INF_BOUND):
        return None
    delta = rigorous_psd_shift(S[np.ix_(active, active)])
    if delta is None:
        return None
    width = np.maximum(hi - lo, 0.0) + 2.0 * _CHARGE_BOX_SLACK * (1.0 + np.abs(lo) + np.abs(hi))
    charge = 2.0 * 0.5 * float(delta) * float(np.sum(width * width))
    return charge if np.isfinite(charge) else None


def _oa_convexity_certificate_enabled() -> bool:
    """Whether OA's cut classifier consults the convexity certificates (#1352).

    Governs :func:`~.rules.classify_oa_cut_convexity` for the callers that take
    its default (OA, LP/NLP-BB/GOA, GDPopt-LOA, GBD, the decomposition advisor
    and IR reformulation). When on, those callers get the same certificate path
    the solver's dispatch classifier (``classify_model(use_certificate=True)``)
    runs: the interval-Hessian certificate plus the exact QP/MIQP
    objective-Hessian route. Without it, a convex quadratic written with
    ``dm.sum(...)`` is routed to OA as certified convex and OA then disables its
    master lower-bound updates and objective cuts on it, fails to certify, and
    falls back to the spatial path.

    It governs the per-row cut mask as well: a constraint the certificate proves
    convex gets OA cuts (``classify_constraint(..., use_certificate=...)``).

    **Default ON** (graduated in #1360 under CLAUDE.md §5); opt out with
    ``DISCOPT_OA_CONVEXITY_CERTIFICATE=0`` (also ``false``/``no``/``off``), which
    restores the legacy syntactic verdict. Proving an objective convex turns on
    OA's objective cuts and master lower-bound updates, so this is
    bound-changing; without it OA cannot certify an objective the syntactic rules miss.

    Retraction: an earlier version of this docstring recorded the flag as "not
    net-positive, re-run once #1355 is fixed". That panel ran before OA's master
    gap window and fixed-NLP tolerance were made scale-aware (#1352); the stall
    it measured was those two defects, not the certificate. The graduation panel
    and the explicit-OA measurement are recorded in docs/dev/performance-plan.md
    §73; the default solve does not route a certificate-only objective to OA
    (``solver._objective_syntactically_convex``), so this changes the callers above
    when invoked explicitly, not the default solve's routing.
    """
    return os.environ.get("DISCOPT_OA_CONVEXITY_CERTIFICATE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


# Size cap for the exact QP convexity route. ``numpy.linalg.eigvalsh`` on a dense
# symmetric matrix measured (this container, load average 0.25, min of 3):
# n=1000 -> 0.052 s, n=2000 -> 0.331 s (sd 0.008), n=4000 -> 2.47 s. Classification
# is a *dispatch* step inside the solver's time budget, so the walk is declined
# above n=2000 rather than spending seconds to prove convexity. Abstaining routes
# to the sound spatial path, so the cap is a performance guard, never a soundness
# one. It also bounds the dense materialisation: 2000**2 float64 is 32 MB, well
# under the extractor's own 256 MB dense-Q budget (#863).
_QP_EXACT_CONVEXITY_MAX_N = 2000


def certify_quadratic_objective_convex(model: Model, *, deadline: Optional[float] = None) -> bool:
    """Prove a QP/MIQP objective convex from its exact Hessian, or return ``False``.

    ``True`` means *proven*: the model's objective, in the solver's minimize
    form, has a positive-semidefinite Hessian, so the objective is convex
    everywhere. ``False`` means "not proven" — an abstention, never a claim of
    nonconvexity — and the caller must keep whatever verdict it already had.

    Why this is rigorous (issue #936). ``classify_problem`` returns ``QP``/``MIQP``
    only when the Rust structure detector reports *every* constraint linear AND
    the objective quadratic. On that branch the objective is *exactly* quadratic,
    so its Hessian is a constant matrix — there is no enclosure and no box
    dependence, and an exact symmetric-eigenvalue test on it is a global
    convexity proof valid on every sub-box. A flag-gated sibling,
    ``_certify_quadratic_psd``, made the identical argument from
    ``quadratic_form.extract_quadratic``; #1388 retired it with
    ``DISCOPT_PSD_QFORM``, and this default-ON route is what remains.
    ``extract_qp_data`` walks the model through the Rust
    repr / algebraic / autodiff ladder and handles the vectorized API, whereas
    ``quadratic_form.extract_quadratic`` (and the scalar interval-Hessian walker
    behind :func:`certify_convex`) does not — which is why a convex QP written
    with ``dm.sum(...)`` over array variables was certified by no route at all
    and fell through to spatial McCormick B&B, where it does not converge.

    ``extract_qp_data`` already returns the *minimize-form* data (it negates
    ``Q``/``c`` for a maximization objective), so a PSD verdict here answers the
    question the caller actually has — "does the objective admit the convex
    path?" — for both senses without a further sign flip.

    Slack columns are handled by the same argument: the extractor appends slack
    variables for inequality rows and pads ``Q`` with a zero block, and a
    principal submatrix of a PSD matrix is PSD, so PSD on the padded Hessian
    implies PSD on the original-variable Hessian. The padding can only make the
    test *harder* to pass, never wrongly pass it.

    Args:
        model: The model to certify.
        deadline: Optional ``time.perf_counter()`` timestamp. Crossing it makes
            this abstain (return ``False``) rather than start the work.
    """
    if not _qp_exact_convexity_enabled():
        return False
    if getattr(model, "_objective", None) is None:
        return False
    if deadline is not None and time.perf_counter() > deadline:
        return False

    # Cheapest gate first: refuse an oversized model before any extraction. The
    # extractor appends one slack column per inequality row and pads ``Q`` to the
    # slacked dimension, so the bound must count the rows too — gating on the
    # variable count alone would let a narrow model with a huge row count through
    # and ask for an (n+m)² dense materialisation.
    n_flat = sum(v.size for v in model._variables)
    if n_flat == 0 or n_flat + len(model._constraints) > _QP_EXACT_CONVEXITY_MAX_N:
        return False

    # #1520: no except. Every step declines by return value: ``classify_problem``
    # falls back to NLP/MINLP itself, ``extract_qp_data``'s rungs absorb their own
    # declines, and the shape/size guards below return False. The old blanket
    # handler answered "not convex" (the sound direction) for any exception, so a
    # defect in the extractor was indistinguishable from a non-convex objective.
    from discopt._relax.problem_classifier import (
        ProblemClass,
        classify_problem,
        dense_Q,
        extract_qp_data,
    )

    problem_class = classify_problem(model)
    if problem_class not in (ProblemClass.QP, ProblemClass.MIQP):
        return False
    if deadline is not None and time.perf_counter() > deadline:
        return False
    qd = extract_qp_data(model)
    quad = qd.Q
    # Re-check the dimension we actually got BEFORE densifying: builder-resident
    # rows are not in ``model._constraints``, so the pre-gate above can still be
    # cleared by a model whose extracted form is far larger. ``.shape`` is
    # available on both dense arrays and scipy sparse matrices.
    shape = getattr(quad, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        return False
    if shape[0] > _QP_EXACT_CONVEXITY_MAX_N:
        return False
    hessian = dense_Q(quad)

    # #1679: a PROOF, the same predicate the ``solver="pounce"`` route certifies a
    # QP with (``convex_ipm_pounce.certify_psd``). ``quadratic_is_psd`` -- used here
    # before -- accepts ``lambda_min >= -slack``, "PSD to within roundoff", which
    # admitted the float Gram matrix ``2 K'K`` of a rank-deficient least-squares
    # model although ``exact_psd`` refutes it (lambda_min ~ -1e-14). The objective
    # is then "not proven convex" here; its structural sum-of-squares spelling is
    # still proven by the DCP walker, which does not go through this matrix.
    from .eigenvalue import psd_certified

    if psd_certified(hessian):
        return True
    # #1682: "convex up to a charge", inside a solve only (the scope's owner
    # subtracts the charge from the published bound; ``Model.convexity()`` and
    # every caller outside a solve keep the strict proof).
    scope = _CHARGE_SCOPE.get()
    if scope is None or scope.abs_gap_tol is None or not _convex_charge_enabled():
        return False
    charge = quadratic_objective_charge(hessian, qd.x_l, qd.x_u)
    if charge is None or charge > CONVEX_CHARGE_ABS_TOL_FRACTION * scope.abs_gap_tol:
        return False
    scope.record(model, charge)
    logger.info(
        "QP objective accepted as convex up to a charge of %.3g (#1682); the solve "
        "subtracts it from the published bound.",
        charge,
    )
    return True


def certify_convex(
    expr: Expression,
    model: Model,
    box: Optional[dict] = None,
) -> Optional[Curvature]:
    """Return a sound convex/concave verdict or ``None``.

    Args:
        expr: A scalar expression.
        model: The model defining the variable layout.
        box: Optional ``{Variable: Interval}`` overriding declared
            bounds — used when the caller has a tighter box from
            FBBT or branching than the model's static declaration.

    Returns:
        * ``Curvature.CONVEX`` if the interval Hessian is provably
          PSD on the box.
        * ``Curvature.CONCAVE`` if the interval Hessian is provably
          NSD on the box.
        * ``None`` if neither test succeeds (indefinite, unsupported
          atoms, or a looseness failure in Gershgorin). Returning
          ``None`` is a deliberate abstention — the caller must treat
          the expression as non-convex.
    """
    # Exact PSD-on-Q fast path (Phase 4 item 3, flag default-OFF). When a
    try:
        ad = interval_hessian(expr, model, box=box)
    except ValueError:
        # Expressions referencing array variables directly are not
        # supported by v1; abstain rather than guess. Also catches
        # ``IntervalHessianTooLarge`` (a ValueError subclass) raised when the
        # body's DAG exceeds the interval-Hessian node budget (#654): abstaining
        # to the caller's spatial/looser path is sound.
        return None

    hess = ad.hess
    if not (np.all(np.isfinite(hess.lo)) and np.all(np.isfinite(hess.hi))):
        return None

    # Structural rank-1 PSD fast path. When the AD walker has attached
    # a ``Rank1Factor`` with nonneg coefficient, the Hessian equals
    # ``c · v vᵀ`` pointwise (sound by construction) and is therefore
    # PSD on the entire box even when the entry-wise interval matrix
    # is too loose for Gershgorin to certify.
    rank1 = ad.rank1_factor
    if rank1 is not None and np.all(np.isfinite(rank1.c.lo)):
        # ``c >= 0`` decided against the arithmetic that produced ``c``, not
        # against an absolute constant (#1397): the Hessian is ``c·v vᵀ``, so a
        # coefficient of magnitude 1e12 carries ~1e-4 of rounding and one of
        # magnitude 1e-12 carries ~1e-28.
        c_slack = psd_decision_slack(
            float(np.max(np.maximum(np.abs(rank1.c.lo), np.abs(rank1.c.hi))))
        )
        if np.all(rank1.c.lo >= -c_slack):
            return Curvature.CONVEX

    # 2×2 sufficient PSD test (Sylvester) — useful when the interval
    # Hessian is tight enough that Gershgorin's row-sum loosening
    # would cross zero but the determinant proof still holds.
    if hess.lo.shape == (2, 2) and psd_2x2_sufficient(hess):
        return Curvature.CONVEX

    # Gershgorin's bounds are rigorous (outward-rounded), so the only slack the
    # verdict needs is the widening that outward rounding itself introduced,
    # which is O(u·‖H_i‖) for row i — scaled by each row's OWN magnitude, not
    # an absolute constant (#1397) and not the whole matrix's norm (#1673 B4:
    # a 5e17 row licensed an exact -2 on another row as PSD).
    if gershgorin_certifies_psd(hess):
        return Curvature.CONVEX

    if gershgorin_certifies_nsd(hess):
        return Curvature.CONCAVE

    return None


def refresh_convex_mask(
    model: Model,
    root_mask: list[bool],
    node_lb: np.ndarray,
    node_ub: np.ndarray,
) -> list[bool]:
    """Re-run the certificate against a B&B node's tightened bounds.

    For every constraint already proven convex at the root, the entry
    stays ``True`` (the node box is a subset of the root box and
    soundness propagates). For every constraint still ``False``, the
    certificate is consulted on the node box; when it proves the body
    convex in the sense implied by the constraint direction, the entry
    flips to ``True``.

    Returns a new list without mutating ``root_mask``. Falls back to
    returning the original mask unchanged if ``model`` or the bounds
    are shape-incompatible — the caller must remain functional even
    when the refresh cannot run.

    This function only ever tightens the mask. It never flips a
    ``True`` entry to ``False``, preserving the soundness invariant
    required by the solver's OA-cut and αBB-skip gates.
    """
    n_vars = sum(v.size for v in model._variables)
    if len(node_lb) != n_vars or len(node_ub) != n_vars:
        return list(root_mask)

    # Skip work when nothing can change — every slot is already True,
    # or there are no constraints at all.
    if not root_mask or all(root_mask):
        return list(root_mask)

    # Build the per-variable box from the node's flat bounds.
    box: dict = {}
    offset = 0
    for v in model._variables:
        size = v.size
        shape = v.shape if v.shape else (1,)
        lb_slice = np.asarray(node_lb[offset : offset + size], dtype=np.float64)
        ub_slice = np.asarray(node_ub[offset : offset + size], dtype=np.float64)
        try:
            box[v] = Interval(lb_slice.reshape(shape), ub_slice.reshape(shape))
        except ValueError:
            # lb > ub somewhere — the node is infeasible; return the
            # root mask unchanged. The caller will discover the
            # infeasibility via its own channels.
            return list(root_mask)
        offset += size

    refreshed = list(root_mask)
    constraint_index = 0
    for c in model._constraints:
        if not isinstance(c, Constraint):
            constraint_index += 1
            continue
        if refreshed[constraint_index]:
            constraint_index += 1
            continue
        # #1520: no except. ``certify_convex`` abstains by returning ``None`` (it
        # absorbs the interval-Hessian's documented ``ValueError`` itself), so an
        # exception here is a defect, not a "not convex" verdict.
        cert = certify_convex(c.body, model, box=box)
        if cert is None:
            constraint_index += 1
            continue
        if c.sense == "<=" and cert == Curvature.CONVEX:
            refreshed[constraint_index] = True
        elif c.sense == ">=" and cert == Curvature.CONCAVE:
            refreshed[constraint_index] = True
        elif c.sense == "==" and cert == Curvature.CONVEX and cert == Curvature.CONCAVE:
            # Equality requires affine; the certificate doesn't return
            # AFFINE, so no tightening is possible here.
            pass
        constraint_index += 1
    return refreshed


__all__ = [
    "ConvexityChargeScope",
    "certify_convex",
    "certify_quadratic_objective_convex",
    "convexity_charge_scope",
    "quadratic_objective_charge",
    "refresh_convex_mask",
    "set_charge_scope_abs_gap_tol",
]
