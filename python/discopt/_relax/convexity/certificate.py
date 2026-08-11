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

import logging
import os
import time
from typing import Optional

import numpy as np

from discopt.modeling.core import Constraint, Expression, Model

from .eigenvalue import gershgorin_lambda_max, gershgorin_lambda_min, psd_2x2_sufficient
from .interval import Interval
from .interval_ad import interval_hessian
from .lattice import Curvature

logger = logging.getLogger(__name__)

# Tolerance for accepting "λ_min ≥ 0" despite floating-point slop. The
# interval Hessian already outward-rounds, so genuine zero eigenvalues
# may appear as small negatives; a very tight tolerance suffices.
_PSD_TOL = 1e-10


def _psd_qform_enabled() -> bool:
    """Whether the exact PSD-on-Q convexity fast path is active.

    **Default OFF** (Phase 4 item 3, bound-changing regime — CLAUDE.md §5).
    The path is a *sound tightening*: on a purely quadratic body the
    Hessian is the constant matrix ``2·Q``, so an exact eigenvalue PSD
    test on ``Q`` certifies convexity rigorously where the conservative
    interval-Hessian + Gershgorin row-sum enclosure would abstain. It can
    therefore prove *more* constraints/objectives convex, which changes
    node relaxations and counts — hence it ships behind a flag,
    default-off until validated on consecutive nightly runs.

    Enable with ``DISCOPT_PSD_QFORM=1`` (also ``true``/``yes``/``on``).
    """
    return os.environ.get("DISCOPT_PSD_QFORM", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


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
    convexity proof valid on every sub-box. This is the identical argument
    :func:`_certify_quadratic_psd` makes; the only difference is where the
    coefficients come from. ``extract_qp_data`` walks the model through the Rust
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

    try:
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
        quad = extract_qp_data(model).Q
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
    except Exception as exc:  # noqa: BLE001 - abstention is the sound direction
        # Capability-disabling, not cosmetic: every failure here is a convex QP
        # that silently keeps routing to spatial B&B, so log the reason rather
        # than let "the fast path didn't trigger" be indistinguishable from "the
        # model isn't convex".
        logger.debug(
            "exact QP objective convexity route abstained: %s: %s", type(exc).__name__, exc
        )
        return False

    from discopt._relax.quadratic_form import quadratic_is_psd

    return quadratic_is_psd(hessian, tol=_PSD_TOL) is True


def _certify_quadratic_psd(expr: Expression, model: Model) -> Optional[Curvature]:
    """Exact convexity verdict for a *purely quadratic* body, or ``None``.

    Returns ``Curvature.CONVEX`` when the body ``xᵀ Q x + cᵀ x + d`` has
    ``Q ⪰ 0`` (Hessian ``2·Q ⪰ 0`` — convex everywhere, box-independent),
    ``Curvature.CONCAVE`` when ``Q ⪯ 0``, and ``None`` in every other
    case: the body is not purely quadratic (extraction abstained), ``Q``
    is indefinite, or ``Q`` is not numerically usable. ``None`` means "I
    cannot certify from Q" — the caller MUST then fall back to the
    existing rigorous interval-Hessian path; it must never assume convex.

    This is sound because extraction is exact-or-abstain: a returned
    ``(Q, c, d)`` reproduces the body identically (verified to 1e-12 on
    random points in ``test_quadratic_form.py``), so ``2·Q`` *is* the
    body's Hessian — no enclosure, no looseness.
    """
    # Local import: keep the convexity package import-light and avoid any
    # cycle with the broader _relax package that quadratic_form pulls in.
    from discopt._relax.quadratic_form import (
        extract_quadratic,
        quadratic_is_nsd,
        quadratic_is_psd,
    )

    n = sum(v.size for v in model._variables)
    res = extract_quadratic(expr, n, model)
    if res is None:
        # Not purely quadratic — abstain; caller uses the rigorous path.
        return None
    Q, _c, _d = res

    is_psd = quadratic_is_psd(Q, tol=_PSD_TOL)
    if is_psd is True:
        return Curvature.CONVEX
    is_nsd = quadratic_is_nsd(Q, tol=_PSD_TOL)
    if is_nsd is True:
        return Curvature.CONCAVE
    # Indefinite, or Q unusable (non-finite) — abstain to the rigorous
    # path rather than claim anything.
    return None


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
    # body is *purely quadratic*, its Hessian is the constant matrix
    # ``2·Q`` and an exact eigenvalue test on ``Q`` certifies convexity
    # rigorously — strictly tighter than (and never looser than) the
    # interval-Hessian + Gershgorin enclosure below. On abstention
    # (non-quadratic body, indefinite Q, or unusable Q) this returns
    # ``None`` and we fall through to the rigorous path unchanged. The
    # verdict is box-independent, so the ``box`` argument does not affect
    # it (a global quadratic convexity proof holds on every sub-box).
    if _psd_qform_enabled():
        q_verdict = _certify_quadratic_psd(expr, model)
        if q_verdict is not None:
            return q_verdict

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
    if rank1 is not None and np.all(np.isfinite(rank1.c.lo)) and np.all(rank1.c.lo >= -_PSD_TOL):
        return Curvature.CONVEX

    # 2×2 sufficient PSD test (Sylvester) — useful when the interval
    # Hessian is tight enough that Gershgorin's row-sum loosening
    # would cross zero but the determinant proof still holds.
    if hess.lo.shape == (2, 2) and psd_2x2_sufficient(hess):
        return Curvature.CONVEX

    lam_min = gershgorin_lambda_min(hess)
    if lam_min >= -_PSD_TOL:
        return Curvature.CONVEX

    lam_max = gershgorin_lambda_max(hess)
    if lam_max <= _PSD_TOL:
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
        try:
            cert = certify_convex(c.body, model, box=box)
        except Exception:
            cert = None
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


__all__ = ["certify_convex", "certify_quadratic_objective_convex", "refresh_convex_mask"]
