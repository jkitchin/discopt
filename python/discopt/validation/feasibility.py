"""The single incumbent-feasibility verifier (#908).

Every solver path that promotes a point to an incumbent — the native kernel's
cutoff seed, the convex kernel's incumbent guard — must agree on what "feasible"
means, and must be right. Before this module there were two independent
implementations and both were wrong in the *wrongly-accept* direction.

What went wrong, and why the fix is shaped this way
---------------------------------------------------

Both verifiers advanced **one row index per** :class:`~discopt.modeling.core.Constraint`
**object** while :class:`~discopt._relax.nlp_evaluator.NLPEvaluator` emits **one row
per flat element**. A constraint body may be array-valued — ``x <= 1`` on a
3-vector is one ``Constraint`` and three rows — so on any model with a vector
constraint the two streams desynchronise and every check from that point on reads
the wrong row. Measured on a purpose-built corpus: a point violating row 2 of a
size-3 vector constraint **by 5.0** was reported feasible by both verifiers, and
4 of 5 constructed-infeasible points were wrongly accepted.

The fix is not to patch the arithmetic. Rows are enumerated from
:meth:`NLPEvaluator.constraint_row_map`, the evaluator's own map, so the
misalignment class is **structurally impossible**: the same
``_source_constraints`` / ``_constraint_flat_sizes`` that build the compiled
concatenation also drive the verification, and they cannot drift.

That row map also closes a second hole for free. The evaluator's row set is
``model._constraints`` **plus** ``model._builder_linear_constraints()`` (#840);
a verifier reading only ``_constraints`` never examined the builder-resident
linear rows at all — so a model built through ``add_linear_constraints`` or the
``constraint(fast=True)`` path had those rows silently unchecked.

Tolerance
---------

The old form was ``abs_tol + rel_tol * |residual|``, which is self-referential:
it scales with the *residual* rather than with the *row*. On an equality row it
collapses to a flat ``1e-6`` no matter whether the row's natural magnitude is 1
or 10,782, so the ``rel_tol`` term is arithmetically dead and the check is
scale-blind. Measured consequence in the other direction: ``nvs22``'s incumbent
was rejected at a *relative* residual of 8.1e-9 (absolute 1.71e-5 against a row
value of 2121.64).

This module keys the tolerance on the **row's** scale::

    violation_i <= abs_tol * max(1, |rhs_i|, max_j |J_ij| * |x_j|)

Two properties of that form are load-bearing and easy to get wrong:

* The relative coefficient is ``abs_tol`` (1e-6), **not** the repo's ``rel_tol``
  (1e-4). Using ``rel_tol`` would loosen every unit-scale row 100x.
* The Jacobian term is a *row-scale estimate*, not a slack. Dropping it makes the
  test **stricter**, never looser — which is why the fallback below is sound.

That form still asks only "is the residual small". #1254 is what the missing
half costs: on a row whose every partial derivative is ~2e-7, a residual of 2e-7
is inside every absolute tolerance in the solver while the nearest point that
actually satisfies the row is a third of a variable's box away — and the point
was accepted, and certified optimal, at the wrong optimum.
:func:`feasible_distance_cap` adds the missing half — the first-order distance
from the point to the row's surface must also stay inside 1e-4 — applied as a
``min`` against the form above. Inert unless the row is nearly flat in every
variable. See that function for the measurement, and for why the row's term
magnitude cannot serve here.

``test_vector_constraint_corpus.py`` carries a control for each naive widening of
this form, showing that each one accepts a bad point this form rejects.

Why the row scale is ``|J_ij| * |x_j|`` and not ``|J_ij| * max(1, |x_j|)`` (#1151)
----------------------------------------------------------------------------------

``|J_ij| * |x_j|`` is the first-order magnitude of the row's *j*-th term. Flooring
``|x_j|`` at 1 does not make it a larger term — it makes it *not a term magnitude
at all*, over-estimating the row's scale by ``1/|x_j|`` on every column whose
value is below 1. The tolerance then grows without bound as a variable shrinks,
which is the exact amplification the reformulation layer works to prevent:
``_clear_divisions`` (``_relax/factorable_reform.py``) multiplies a cleared
quotient row by ``1/dmin`` precisely so that a fixed absolute residual test on
``w*D - N == 0`` bounds the error in ``w``; the floored scale divided that
scaling straight back out, leaving the check exactly as loose as if the row had
never been scaled.

Measured consequence (#1151), ``minimize x/y + y/x`` over ``[1e-3, 1e3]^2`` whose
global minimum is exactly 2 by AM-GM. The reformulation emits
``1000*(w0*y - x) == 0``; at the accepted point ``x = y ~ 1.4e-3`` that row is
violated by **9.28e-4** and was accepted, because ``max_j |J_ij| * max(1,|x_j|)``
read the row's scale as 1000 (the coefficient on ``x``) rather than 1.4 (the
magnitude ``1000*x`` actually attains), licensing a tolerance of 1e-3. The
residual maps to an error of ``residual / (1000*y)`` in ``w0``, so the solver
reported ``objective = 1.9987`` — **below the global minimum**, at
``status=optimal``, a false certificate. With the term-magnitude scale the row's
tolerance is 1.4e-6 and the point is rejected.

The guarantee the term-magnitude form buys, for a defining row
``s*(w*D - N) == 0`` (``s`` any positive scaling; ``1/dmin`` as emitted here).
The row's derivative in ``w`` is ``s*D``, so the term-magnitude scale obeys
``S >= s*|D|*|w|``, and for a monomial denominator every other column's term has
that same magnitude, so ``S ~ s*|D|*|w|``. With ``|Δw| = |w*D - N| / |D|`` and a
residual held to ``abs_tol * max(1, S)``:

* when ``s*|D|*|w| >= 1``: ``|Δw| <= abs_tol * s*|D|*|w| / (s*|D|) = abs_tol*|w|``;
* otherwise: ``|Δw| <= abs_tol / (s*|D|) <= abs_tol``, since the ``1/dmin``
  scaling makes ``s*|D| >= 1`` everywhere in the box.

So ``|Δw| <= abs_tol * max(1, |w|)`` — a bound on the *aux value*, keyed on the
aux's own magnitude and on nothing about the denominator, which is exactly what
the reported objective (linear in ``w``) needs. Under the floored form ``S`` is
instead ``~ s*|D|*max(1,|w|)`` divided by nothing at all — it reads ``s`` itself
when ``s*|D| > s*|D|*|w|`` — and the same algebra leaves ``|Δw|`` proportional to
``1/|D|``, unbounded as the denominator shrinks. That 1/D law is what the issue
measured: ``|Δ objective| x denominator`` flat at ~1.9e-6 across box floors.

The change is narrow by construction: ``|x_j| <= max(1, |x_j|)`` columnwise, so
the new scale never exceeds the old one and the two differ only when the column
attaining the floored maximum carries ``|x_j| < 1``. It moves only in the strict
direction — the accepted set shrinks — so no point this verifier now accepts was
rejected before.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sparse

logger = logging.getLogger(__name__)

#: Absolute feasibility tolerance, and the *relative* coefficient in the
#: scale-keyed form above. Deliberately not ``rel_tol`` — see the module note.
ABS_TOL = 1e-6
#: Integrality tolerance for INTEGER/BINARY variables.
INT_TOL = 1e-5
#: Legacy relative coefficient, used ONLY for the variable-bound test, where it
#: is keyed on the bound (a genuine scale) rather than on the residual.
BOUND_REL_TOL = 1e-4
#: The first-order DISTANCE, in variable space, a point may sit from satisfying a
#: row — the repo's declared ``rel=1e-4``. See :func:`feasible_distance_cap`.
FEASIBLE_DISTANCE_TOL = 1e-4
#: Absolute floor under that cap, so a row that is exactly flat at the point is
#: not held to exact arithmetic. Four orders above double-precision evaluation
#: noise on a unit-scale row.
SMALL_ROW_ABS_FLOOR = 1e-12
#: Per-term coefficient of the cancellation-noise floor under the cap — the
#: ``rtol`` ``_relax/primal_heuristics`` has always used for the same purpose.
CANCELLATION_RTOL = 1e-9


def jacobian_row_gradient_norms(J) -> np.ndarray:
    """``max_j |J_ij|`` per row — the row gradient's sup-norm at the point.

    Distinct from :func:`jacobian_row_scales` (``max_j |J_ij| * |x_j|``, the row's
    term MAGNITUDE) and used for a different question: not "how big is this row"
    but "how far must the point move to satisfy it". Non-finite rows return
    ``inf``, which :func:`feasible_distance_cap` turns into "no cap" — an
    unestimatable gradient must not manufacture a strict test.

    A scipy sparse ``J`` is accepted and never densified (#1619); unstored entries
    are exactly 0, which can neither raise a row's max nor make it non-finite, so
    the answer is the dense one.
    """
    if sparse.issparse(J):
        return _sparse_row_gradient_norms(J)
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2:
        raise ValueError(f"expected a 2-D Jacobian, got shape {J.shape}")
    if J.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        mag = np.abs(J)
    finite = np.isfinite(mag)
    out = np.asarray(np.where(finite, mag, 0.0).max(axis=1), dtype=np.float64)
    out[~finite.all(axis=1)] = np.inf
    return out


_EPS = float(np.finfo(np.float64).eps)


def improving_gradient_norms(J, x, lb, ub, direction, integer_mask=None) -> np.ndarray:
    """Per-row gradient norm restricted to moves that can REDUCE the violation (#1284).

    :func:`jacobian_row_gradient_norms` takes ``max_j |J_ij|`` over every column,
    so one column that cannot help — a variable sitting on the bound that blocks
    its improving direction, or an integer variable — sets the cap on its own.
    Measured (#1284): adding ``+ s`` with ``s in [0, 1]`` at ``s = 0`` to the #1254
    row gave ``||grad||_inf = 1``, the cap became 1e-4, and the point 0.87 away in
    ``y`` was certified again at ``z = -20`` (true optimum -6.699).

    Column ``j`` of row ``i`` counts for what a move of at most
    ``FEASIBLE_DISTANCE_TOL`` in ``x_j``, inside ``[lb_j, ub_j]``, can remove from
    the violation: ``|J_ij| * min(1, room_ij / FEASIBLE_DISTANCE_TOL)``, where
    ``room_ij`` is the distance from ``x_j`` to the bound in the direction that
    reduces the violation. An integer column can move only to an integer, and the
    one integer within that distance is ``round(x_j)``: its room is
    ``|round(x_j) - x_j|`` when that lies in the improving direction, else 0.
    The columns move together (a box of half-width ``FEASIBLE_DISTANCE_TOL``), so
    their contributions add, and ``FEASIBLE_DISTANCE_TOL * result`` is the
    first-order reduction the best such move achieves: the statement
    :func:`feasible_distance_cap` makes, now true of the point's box.

    Summed, not maxed (measured on clay0303hfsg): a big-M row
    ``x - 52.5 y <= 0`` at ``x = 1.0e-8`` (bound 0) and ``y = -1.8e-10`` (binary)
    is violated by 1.9e-8, exactly what rounding ``y`` and moving ``x`` onto its
    bound together remove. A per-column max (or zero weight for ``y``) put the cap
    at 1e-8, rejected that point and every node solution like it, and the solve
    certified 28862 against the recorded optimum 26669.

    ``direction[i]`` is ``+1`` when row ``i``'s body must DECREASE, ``-1`` when it
    must increase, ``0`` when the row is not violated (plain sup-norm; the cap is
    irrelevant there). Clipped to :func:`jacobian_row_gradient_norms`, so it can
    only tighten a gate. Non-finite rows return ``inf`` as there.

    A scipy sparse ``J`` is evaluated on its stored entries only (#1619): an
    unstored ``J_ij`` is exactly 0, so its ``|J_ij| * frac`` term is 0 and it adds
    nothing to the row's sum. The dense form built an ``(m, n)`` array per
    intermediate (sign, room, frac, contrib), which made the post-solve screen
    quadratic in the mesh on DAE models (#1619 B-12a).
    """
    sparse_J = sparse.issparse(J)
    if not sparse_J:
        J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2:
        raise ValueError(f"expected a 2-D Jacobian, got shape {J.shape}")
    plain = jacobian_row_gradient_norms(J)
    if J.shape[0] == 0:
        return plain
    x = np.asarray(x, dtype=np.float64).ravel()
    lb = np.asarray(lb, dtype=np.float64).ravel()
    ub = np.asarray(ub, dtype=np.float64).ravel()
    n = J.shape[1]
    if not (x.size == lb.size == ub.size == n):
        raise ValueError(
            f"box of sizes x={x.size}, lb={lb.size}, ub={ub.size} does not match "
            f"{n} Jacobian columns"
        )
    d = np.sign(np.asarray(direction, dtype=np.float64).ravel())
    if d.size != J.shape[0]:
        raise ValueError(f"direction has {d.size} entries for {J.shape[0]} rows")
    if sparse_J:
        return _sparse_improving_gradient_norms(J, x, lb, ub, d, integer_mask, plain)
    with np.errstate(invalid="ignore"):
        step = -d[:, None] * np.sign(J)  # sign of the improving move in x_j
        # Each room carries its own round-off allowance. A violation caused only
        # by ``x_j`` sitting ``room`` inside its bound is repaired by that one move
        # exactly, so ``viol == |J_ij| * room`` up to the round-off of two
        # different products; without the allowance the gate decides that tie by
        # one ulp (portfol_roundlot: ``x11 - 78000 x2 >= 0`` at ``x2 = 7.2e-11``,
        # ``x11 = 0`` integer, rejected 5.63185816304395e-06 against a cap of
        # 5.631858163043949e-06).
        slack = 16.0 * _EPS
        # The allowance is CAPPED at ``slack * FEASIBLE_DISTANCE_TOL`` (#1397).
        # ``slack * (|bound| + |x|)`` is the honest round-off of ``room`` itself,
        # but ``room`` is then divided by an absolute FEASIBLE_DISTANCE_TOL and
        # multiplied by ``|J_ij|``, so a relative allowance is amplified by up to
        # ``|J_ij| / FEASIBLE_DISTANCE_TOL`` before it reaches the cap. Two
        # different round-offs were being conflated: ``room``'s own (which says
        # room is *uncertain* by that much, not that it is *at least* that much)
        # and the one this allowance exists to absorb, which is the tie in
        # ``viol <= cap``.
        #
        # Sizing it, so the constant is derived and not chosen: adding ``a`` to
        # each room gives ``dfrac_j = a / TOL`` and hence ``dcap = TOL *
        # sum_j |J_ij| * a / TOL = a * sum_j |J_ij|``. Requiring that to stay at
        # round-off level, ``dcap <= K*u*cap = K*u*TOL*grad``, and ``grad <=
        # sum_j |J_ij|`` because every ``frac <= 1``, so ``a <= K*u*TOL``
        # suffices. Both terms are in x-units, so the ``min`` is dimensionally
        # coherent; it is a ``min`` and not a ``max`` because the smaller of
        # "room's own round-off" and "what the amplification can absorb" is the
        # one that keeps the cap honest.
        #
        # Measured: without the cap, a column pinned EXACTLY on the bound that
        # blocks its improving direction -- true room exactly 0 -- is credited
        # with phantom room proportional to |bound|. On a row ``[-1000, +1]``
        # with col 0 pinned at ``ub = B``, the returned norm went 1.0 (the true
        # value) -> 1000.0 (the plain sup-norm) as B swept 1e0 -> 1e11,
        # inflating the acceptance cap from 1e-4 to 1e-1 and making #1284's
        # tightening a silent no-op. The cap leaves the allowance BYTE-IDENTICAL
        # on every case #1284 pins (portfol_roundlot, clay0303hfsg: there
        # ``|bound| + |x| < 1e-7``, so the ``min`` selects the original term) and
        # still breaks portfol_roundlot's tie with 24x margin.
        allow = slack * FEASIBLE_DISTANCE_TOL
        up = np.maximum(ub - x, 0.0) + np.minimum(slack * (np.abs(ub) + np.abs(x)), allow)
        down = np.maximum(x - lb, 0.0) + np.minimum(slack * (np.abs(lb) + np.abs(x)), allow)
        up = up[None, :]
        down = down[None, :]
        room = np.where(step > 0, up, np.where(step < 0, down, 0.0))
        if integer_mask is not None:
            mask = np.asarray(integer_mask, dtype=bool).ravel()
            if mask.size != n:
                raise ValueError(f"integer_mask has {mask.size} entries for {n} columns")
            r = np.clip(np.round(x), np.ceil(lb), np.floor(ub))
            gap = r - x
            # Same cap, same derivation (#1397). The integer arm is if anything
            # more exposed: the ``gap == 0`` branch below credits a column that is
            # ALREADY on its integer with ``to_int``, which without the cap is
            # pure round-off scaled by ``|r| + |x|``.
            to_int = np.abs(gap) + np.minimum(slack * (np.abs(r) + np.abs(x)), allow)
            int_room = np.where(step * np.sign(gap)[None, :] > 0, to_int[None, :], 0.0)
            int_room = np.where((gap == 0.0)[None, :] & (step != 0), to_int[None, :], int_room)
            room = np.where(mask[None, :], int_room, room)
        frac = np.minimum(1.0, room / FEASIBLE_DISTANCE_TOL)
        contrib = np.abs(J) * frac
    finite = np.isfinite(contrib)
    out = np.minimum(np.where(finite, contrib, 0.0).sum(axis=1), plain)
    out = np.where(d == 0.0, plain, out)
    out[~np.isfinite(plain)] = np.inf
    return np.asarray(out, dtype=np.float64)


def _sparse_row_gradient_norms(J) -> np.ndarray:
    """:func:`jacobian_row_gradient_norms` for a scipy sparse ``J`` (#1619)."""
    J = sparse.csr_matrix(J, dtype=np.float64)
    m = J.shape[0]
    if m == 0:
        return np.zeros(0, dtype=np.float64)
    rows = np.repeat(np.arange(m), np.diff(J.indptr))
    with np.errstate(invalid="ignore"):
        mag = np.abs(J.data)
    finite = np.isfinite(mag)
    out = np.zeros(m, dtype=np.float64)
    np.maximum.at(out, rows, np.where(finite, mag, 0.0))
    out[np.unique(rows[~finite])] = np.inf
    return out


def _sparse_improving_gradient_norms(J, x, lb, ub, d, integer_mask, plain) -> np.ndarray:
    """:func:`improving_gradient_norms` over the stored entries of a sparse ``J``.

    Entry-for-entry the dense formula, with ``(i, j)`` running over the stored
    entries instead of the full ``(m, n)`` grid; see that function for every
    constant. ``plain`` is the row sup-norm already computed by the caller.
    """
    J = sparse.csr_matrix(J, dtype=np.float64)
    m, n = J.shape
    rows = np.repeat(np.arange(m), np.diff(J.indptr))
    cols = J.indices
    v = J.data
    with np.errstate(invalid="ignore"):
        step = -d[rows] * np.sign(v)
        slack = 16.0 * _EPS
        allow = slack * FEASIBLE_DISTANCE_TOL
        up = np.maximum(ub - x, 0.0) + np.minimum(slack * (np.abs(ub) + np.abs(x)), allow)
        down = np.maximum(x - lb, 0.0) + np.minimum(slack * (np.abs(lb) + np.abs(x)), allow)
        room = np.where(step > 0, up[cols], np.where(step < 0, down[cols], 0.0))
        if integer_mask is not None:
            mask = np.asarray(integer_mask, dtype=bool).ravel()
            if mask.size != n:
                raise ValueError(f"integer_mask has {mask.size} entries for {n} columns")
            r = np.clip(np.round(x), np.ceil(lb), np.floor(ub))
            gap = r - x
            to_int = np.abs(gap) + np.minimum(slack * (np.abs(r) + np.abs(x)), allow)
            int_room = np.where(step * np.sign(gap)[cols] > 0, to_int[cols], 0.0)
            int_room = np.where((gap == 0.0)[cols] & (step != 0), to_int[cols], int_room)
            room = np.where(mask[cols], int_room, room)
        frac = np.minimum(1.0, room / FEASIBLE_DISTANCE_TOL)
        contrib = np.abs(v) * frac
    finite = np.isfinite(contrib)
    sums = np.bincount(rows, weights=np.where(finite, contrib, 0.0), minlength=m)
    out = np.minimum(sums, plain)
    out = np.where(d == 0.0, plain, out)
    out[~np.isfinite(plain)] = np.inf
    return np.asarray(out, dtype=np.float64)


def model_integer_mask(model) -> np.ndarray:
    """Flat mask of the model's INTEGER/BINARY columns."""
    from discopt.modeling.core import VarType

    parts = [
        np.full(int(v.size), v.var_type in (VarType.BINARY, VarType.INTEGER), dtype=bool)
        for v in model._variables
    ]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=bool)


def evaluator_box(evaluator):
    """``(lb, ub, integer_mask)`` for an evaluator, or ``None`` if it exposes no box.

    Unwraps the cut-augmenting (``_ev``) and bound-override (``_evaluator``) wrappers
    to reach the owning model for integrality. A ``None`` result makes the callers
    fall back to the plain sup-norm, which is exactly the pre-#1284 behaviour.
    """
    try:
        lb, ub = evaluator.variable_bounds
    except AttributeError:
        return None
    inner = evaluator
    model = None
    for _ in range(4):
        model = getattr(inner, "_model", None)
        if model is not None:
            break
        inner = getattr(inner, "_ev", None) or getattr(inner, "_evaluator", None)
        if inner is None:
            break
    mask = None if model is None else model_integer_mask(model)
    lb = np.asarray(lb, dtype=np.float64).ravel()
    if mask is not None and mask.size != lb.size:
        mask = None
    return lb, np.asarray(ub, dtype=np.float64).ravel(), mask


def feasible_distance_cap(grad_norms, term_scale=None) -> np.ndarray:
    """Cap on a row's absolute violation: ``FEASIBLE_DISTANCE_TOL * ||grad g_i||_inf``.

    Every feasibility gate in the solver compares a row's violation against a
    tolerance that is ultimately *absolute* — ``ABS_TOL * max(1, ...)`` here,
    ``tol + rtol*scale`` in ``_relax/primal_heuristics``, a bare ``tol`` in
    ``solver._check_constraint_feasibility``. A residual is not, on its own, a
    statement about the point: what makes an absolute residual tolerable is that
    a point carrying it sits within tolerance of a point that satisfies the row.
    The first-order distance to the row's surface is ``violation /
    ||grad g||_inf``, so requiring that distance to stay inside the repo's
    declared ``rel=1e-4`` is the same statement expressed where the user's bounds
    and tolerances live — in variable space.

    Measured (#1254). ``10**y1 + 10**y2 <= 10**z`` with ``y in [-9,-6]``,
    ``z in [-20,0]``: at ``y1=y2=-7, z=-20`` the row is violated by 2.0e-07 and
    every partial derivative is ~2.3e-07 or smaller, so restoring feasibility
    needs ``Δy ~ 0.87`` — a THIRD of that variable's whole box. Every gate
    accepted the point (2e-7 < 1e-6 < 1e-4), it became the incumbent, it met the
    root relaxation bound, and ``solve`` returned ``status="optimal"``,
    ``gap_certified=True`` at ``z = -20`` where the true optimum is ``-6.69897``.
    A false certificate, silent, from an absolute residual alone.

    Why the distance and not the row's term magnitude, which is the other obvious
    way to read "relative to the row". On the Scholtes-regularized MPEC row
    ``x*y <= t`` at ``x = 1, y = 1.67e-08, t = 1e-08``, the residual 6.7e-09 is
    40 % of the row's magnitude — relatively WORSE than #1254's point — yet
    ``dg/dx = 1``, so the point is 6.7e-09 in ``y`` away from feasible and is
    exactly the kind of converged local point the absolute tolerance exists to
    accept (``test_mpec_source_residuals.py`` asserts it must verify). Term
    magnitude cannot separate the two; distance separates them by nine orders.

    Applied as a ``min`` against whatever tolerance the calling gate already
    computes, so it can only ever REJECT a point a gate would have accepted. It
    is inert unless ``||grad g||_inf < 1e-2``, i.e. on a row that is nearly flat
    in EVERY variable — where no small move fixes the residual and calling the
    point near-feasible is not a statement about anything.

    The noise floor is ADDED to the distance allowance, not ``max``-ed with it
    (#1392). A noise floor and an allowance answer different questions — "how
    small a residual can this arithmetic even resolve" and "how far may the point
    sit from the surface" — and the residual carries both, so combining them with
    ``max`` lets the larger one erase the smaller. Measured on ``synthes3`` via
    ``nlp_bb``: the big-M row ``x0 - 10*y9 <= 0`` at ``x0 = 1.0750640431770233e-12``
    (lower bound 0), ``y9 = 0`` evaluates to ``1.0751399770470016e-12`` — the row
    body and its own linear term disagree by 7.6e-17, ordinary double-precision
    noise on a unit-scale evaluation. ``x0`` has exactly ``1.0750640431770233e-12``
    of room to its bound, so the distance allowance is that same number, 7.6e-17
    BELOW the violation: the cap rejected the point over a shortfall four orders
    of magnitude under its own declared floor, and the whole solve returned
    ``status="error"`` with a correct incumbent (68.00974056776073 against
    minlplib's proven ``=opt=`` 68.00974052) withheld. Adding the floor keeps every
    recorded rejection — #1254's point is 2.0e-7 against an allowance of 2.3e-11,
    #770's violations are 0.4-17.6 — since a 1e-12 addition cannot reach a
    violation nine orders above it.
    """
    g = np.abs(np.asarray(grad_norms, dtype=np.float64))
    # Cancellation noise is not a distance: a row built from terms of magnitude
    # 1e5 carries ~1e-9*1e5 of pure floating-point residual no matter where the
    # point is, and a cap that cut into that would reject points for the
    # arithmetic's rounding rather than for their position.
    # ``test_polish_feasibility_gate_1199`` holds the boundary this protects.
    floor = np.broadcast_to(np.float64(SMALL_ROW_ABS_FLOOR), g.shape)
    if term_scale is not None:
        floor = np.maximum(floor, CANCELLATION_RTOL * np.abs(np.asarray(term_scale, np.float64)))
    cap = floor + FEASIBLE_DISTANCE_TOL * g
    return np.where(np.isfinite(g), cap, np.inf)


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of a feasibility verification.

    ``objective`` is the point's TRUE objective in model units (MAXIMIZE
    un-negated), or ``None`` when it was not requested or could not be computed.
    ``reason`` names the first failing check — for logs, never for control flow.
    """

    ok: bool
    objective: Optional[float] = None
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


def _sense_str(con) -> Optional[str]:
    """``Constraint.sense`` is a ``str`` on some paths and an enum on others."""
    s = con.sense
    if not isinstance(s, str):
        s = getattr(s, "value", None)
    return s if s in ("<=", ">=", "==") else None


def _row_violation(val: float, sense: str) -> float:
    """Signed violation of ``body <sense> 0``, ``<= 0`` meaning satisfied."""
    if sense == "<=":
        return val
    if sense == ">=":
        return -val
    return abs(val)


def check_variable_bounds(model, x_flat: np.ndarray) -> VerifyResult:
    """Variable bounds + integrality against the ORIGINAL declared model.

    Bounds use ``abs_tol + rel_tol * |bound|``. Unlike the old *constraint*
    tolerance this is keyed on the bound — a real scale — not on the residual, so
    it is not self-referential: a local NLP returns a bound-active variable a few
    ULPs off its bound, and on a large-magnitude bound (``tanksize`` x41 lb=536) a
    4e-6 absolute slack is 8e-9 relative, inside the regime the whole solver
    operates in.
    """
    from discopt.modeling.core import VarType

    off = 0
    for v in model._variables:
        size = int(getattr(v, "size", 1))
        vals = x_flat[off : off + size]
        if vals.shape[0] != size:
            return VerifyResult(False, None, f"length mismatch at variable {v.name!r}")
        lb_flat = np.asarray(v.lb, dtype=np.float64).flatten()
        ub_flat = np.asarray(v.ub, dtype=np.float64).flatten()
        lb_tol = ABS_TOL + BOUND_REL_TOL * np.abs(lb_flat)
        ub_tol = ABS_TOL + BOUND_REL_TOL * np.abs(ub_flat)
        if np.any(vals < lb_flat - lb_tol) or np.any(vals > ub_flat + ub_tol):
            return VerifyResult(False, None, f"variable {v.name!r} out of bounds")
        if v.var_type in (VarType.INTEGER, VarType.BINARY):
            if np.any(np.abs(vals - np.round(vals)) > INT_TOL):
                return VerifyResult(False, None, f"variable {v.name!r} not integral")
        off += size
    return VerifyResult(True)


def snap_integer_columns(x: np.ndarray, int_idx) -> np.ndarray:
    """``x`` with the columns in ``int_idx`` rounded to their nearest integer.

    The index-keyed primitive behind :func:`snap_integers`, for the matrix paths
    (``StdForm.int_idx``, the MILP engine's offset list) that have no ``Model``.
    See :func:`snap_integers` for why every feasibility test on a point with
    integer columns has to run on the snapped vector.
    """
    x = np.asarray(x, dtype=np.float64)
    idx = np.asarray(int_idx, dtype=np.int64).ravel()
    if idx.size == 0:
        return x
    out = x.copy()
    out[idx] = np.round(out[idx])
    return out


def snap_integers(model, x_flat: np.ndarray) -> np.ndarray:
    """``x_flat`` with every INTEGER/BINARY column rounded to its nearest integer.

    A point is only ever *claimed* at its integral realisation: a solver that
    reports ``z = 1e-6`` for a binary is claiming ``z = 0``, and the value it
    reports for the objective is the value of the claim, not of the fractional
    point it happened to compute. Verifying the fractional point instead lets a
    column inside :data:`INT_TOL` carry ``M * INT_TOL`` of constraint slack — on a
    big-M row (``x <= 1e7 z``) that is 10 units of violation bought with 1e-6 of
    fractionality, enough to certify an *optimal* value below the true optimum.

    Snapping is a no-op on a genuinely integral point (``round`` of an exact
    integer is that integer, bit for bit), so this only ever rejects points whose
    feasibility rests on fractionality the solver has already declared absent.

    Callers must run :func:`check_variable_bounds` FIRST, which proves every such
    column is within :data:`INT_TOL` of an integer; the snap then moves each by at
    most that much.
    """
    from discopt.modeling.core import VarType

    out = np.array(x_flat, dtype=np.float64, copy=True)
    off = 0
    for v in model._variables:
        size = int(getattr(v, "size", 1))
        if off + size > out.shape[0]:
            break
        if v.var_type in (VarType.INTEGER, VarType.BINARY):
            out[off : off + size] = np.round(out[off : off + size])
        off += size
    return out


def jacobian_row_scales(J: np.ndarray, x_flat: np.ndarray) -> np.ndarray:
    """``max_j |J_ij| * |x_j|`` per row — the row's first-order term magnitude.

    **The one definition of this quantity.** It is not private, because three
    call sites need it and #1151 was in part a consequence of there having been
    three *copies*: this module's incumbent gate, ``validation/examiner.py``'s
    scaled primal-feasibility check, and ``_dual_recovery``'s active-set test all
    carried the same expression written out by hand. Fixing one left the other
    two vouching for exactly the point the fixed one rejects — measured, on the
    #1151 witness: ``verify_point`` rejected the row at 9.276e-04 while the
    examiner reported ``[PASS] primal_con_feas (scaled)`` and ``_dual_recovery``
    admitted the row into its active set. Import this rather than re-deriving it.

    Not floored: callers apply their own floor (``max(1, |rhs_i|, …)``), and
    flooring ``|x_j|`` *inside* the max is precisely the #1151 defect — see the
    module docstring. Returns zeros where every term of a row vanishes; the
    caller's floor is what keeps such a row on the plain absolute tolerance.

    **Non-finite rows return 0.0**, i.e. the caller's floor, i.e. the plain
    absolute tolerance — the STRICTEST answer, and the same direction
    :func:`_row_scales_and_gradients` already takes when the Jacobian is
    unavailable. This is
    not hypothetical: an unbounded derivative at a variable pinned to zero
    (``d/dx log(x)`` at ``x = 0``) makes ``inf * 0`` a NaN, and a NaN scale
    propagates into ``violation / row_scale`` as a NaN that compares False
    against every tolerance. Reported by the second review pass on #1157: the
    guard lived in ``_row_scales_and_gradients`` and did not survive being
    factored out here,
    so the two consumers that call this directly emitted a numpy RuntimeWarning
    and a spurious ``[FAIL] primal_con_feas (scaled)`` with ``scale=nan``. Note
    the pre-#1151 floored form gave that row ``inf`` and so an INFINITE
    tolerance, passing it unconditionally; 0.0 keeps the safe direction while
    dropping the warning and the bogus detail line.
    """
    scales, _all_finite = _jacobian_row_scales_checked(J, x_flat)
    return scales


def _jacobian_row_scales_checked(J: np.ndarray, x_flat: np.ndarray) -> tuple[np.ndarray, bool]:
    """:func:`jacobian_row_scales` plus "was every row finite?".

    Split out so :func:`_row_scales_and_gradients` can keep its **whole-batch**
    fallback: one
    non-finite row there sends *every* suspect row to the Jacobian-free bound.
    Zeroing per row instead would leave the co-occurring rows on their own
    (larger) scales, which is looser than what that function did before this
    helper existed — a relaxation, in the accepting direction, smuggled in by a
    refactor. The public helper's per-row 0.0 is right for callers that have no
    batch to fall back for.
    """
    xw = np.abs(np.asarray(x_flat, dtype=np.float64))
    if sparse.issparse(J):
        # NOT np.asarray(J): on a scipy sparse matrix that returns a 0-d OBJECT
        # array rather than raising, so the shape checks below would report
        # "expected a 2-D Jacobian, got shape ()" for a perfectly good Jacobian.
        return _sparse_row_scales_checked(J, xw)
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2:
        raise ValueError(f"expected a 2-D Jacobian, got shape {J.shape}")
    if J.shape[1] != xw.shape[0]:
        raise ValueError(f"Jacobian has {J.shape[1]} columns, point has {xw.shape[0]}")
    if J.shape[0] == 0:
        return np.zeros(0, dtype=np.float64), True
    # ``inf * 0`` is NaN and numpy warns; the result is discarded either way, so
    # the warning is noise on a path that has already decided what to do.
    with np.errstate(invalid="ignore"):
        terms = np.abs(J) * xw[None, :]
    finite = np.isfinite(terms)
    all_finite = bool(finite.all())
    if not all_finite:
        # Any non-finite term makes the whole row's magnitude unestimatable.
        terms = np.where(finite, terms, 0.0)
        bad_rows = ~finite.all(axis=1)
        scales = np.asarray(terms.max(axis=1), dtype=np.float64)
        scales[bad_rows] = 0.0
        return scales, False
    return np.asarray(terms.max(axis=1), dtype=np.float64), True


def _sparse_row_scales_checked(J, xw: np.ndarray) -> tuple[np.ndarray, bool]:
    """:func:`_jacobian_row_scales_checked` for a sparse ``J``, same answers.

    Kept in the same module and reached from the same entry point rather than
    written out at the call site: #1151 was in part a consequence of this
    quantity existing as three hand-written copies, and a sparse fourth copy
    would be the same mistake in a new representation.

    ``max_j |J_ij| * |x_j|`` is a maximum of NON-NEGATIVE terms, which is what
    makes the sparse form exact rather than an approximation: a structurally
    absent entry contributes ``0``, and zero cannot raise a maximum over
    non-negative values. A row with no stored entries therefore scores 0.0 —
    the same answer the dense path gives for a row of zeros, and the caller's
    floor then holds it to the plain absolute tolerance.
    """
    if J.shape[1] != xw.shape[0]:
        raise ValueError(f"Jacobian has {J.shape[1]} columns, point has {xw.shape[0]}")
    m = int(J.shape[0])
    if m == 0:
        return np.zeros(0, dtype=np.float64), True

    if not np.isfinite(xw).all():
        # The one place where sparsity would LOOSEN the answer, so it is handled
        # before the product rather than discovered after it. A dense row has an
        # entry in every column, so a non-finite ``x_j`` poisons every row: the
        # term is ``inf`` where ``J_ij != 0`` and ``|0| * inf = NaN`` where it is
        # zero, and either way ``~finite.all(axis=1)`` marks the row. Sparsity
        # drops exactly the ``|0| * inf`` half, so a structurally absent entry
        # would quietly leave the row finite and hand the caller a LARGER scale —
        # a more permissive activity test, which is the #1151 failure direction.
        # Reproduce the dense verdict instead: every row unestimatable.
        return np.zeros(m, dtype=np.float64), False

    # ``inf * 0`` is NaN and numpy warns; the result is discarded either way.
    with np.errstate(invalid="ignore"):
        terms = abs(J).multiply(xw[None, :])
    terms = sparse.csr_matrix(terms, dtype=np.float64)

    finite = np.isfinite(terms.data)
    all_finite = bool(finite.all())
    if all_finite:
        scales = terms.max(axis=1).toarray().ravel()
        return np.asarray(scales, dtype=np.float64), True

    # Any non-finite term makes the whole row's magnitude unestimatable. Map each
    # stored entry back to its row through the CSR row pointers -- the sparse
    # equivalent of ``~finite.all(axis=1)``.
    row_of_entry = np.repeat(np.arange(m), np.diff(terms.indptr))
    bad_rows = np.zeros(m, dtype=bool)
    bad_rows[row_of_entry[~finite]] = True
    terms.data = np.where(finite, terms.data, 0.0)
    scales = np.asarray(terms.max(axis=1).toarray().ravel(), dtype=np.float64)
    scales[bad_rows] = 0.0
    return scales, False


def screen_jacobian(evaluator, x_flat: np.ndarray):
    """The constraint Jacobian at ``x_flat``: scipy CSR when the evaluator has one.

    #1619 B-12a: the dense ``evaluate_jacobian`` is an ``(m, n)`` scatter of a
    natively sparse tape, so on a collocation model the post-solve screen was
    quadratic in the mesh (cProfile at nfe=2400: 4.8 s in this screen against 3.3 s
    for the solve itself). Every consumer below is sparse-aware, and the CSR form is
    the same matrix (see ``TapeNLPEvaluator.evaluate_sparse_jacobian``: its
    structure is the tape's exact COO, so it is *exactly* the dense matrix).

    Only the tape evaluator is asked. The legacy JAX evaluator's sparse form can
    take its pattern from a nonzero mask traced at one interior point, which is
    not guaranteed to cover an entry that vanishes there; the screen keeps that
    evaluator on its dense Jacobian rather than inherit the assumption.
    """
    from discopt._tape_nlp_evaluator import TapeNLPEvaluator

    if isinstance(evaluator, TapeNLPEvaluator):
        return evaluator.evaluate_sparse_jacobian(x_flat)
    return np.asarray(evaluator.evaluate_jacobian(x_flat), dtype=np.float64)


def _row_scales_and_gradients(evaluator, x_flat: np.ndarray, rows: np.ndarray, box=None):
    """``(max_j |J_ij| * |x_j|, max_j |J_ij|)`` for the given rows, or ``(None, None)``.

    Both halves of the tolerance from ONE Jacobian evaluation: the term magnitude
    says how big the row is, the gradient sup-norm says how far the point is from
    satisfying it, and the two must be read off the same matrix.

    The first product is the first-order magnitude of the row's *j*-th term; see
    the module docstring (#1151) for why flooring ``|x_j|`` at 1 turns this from a
    row-scale estimate into a ``1/|x_j|`` amplification of the tolerance, and how
    that produced a reported objective below the global minimum. It is floored at
    1 by the caller (``max(anchor, scale)`` with ``anchor >= 1``), so a row all of
    whose terms vanish at the point is held to the plain absolute tolerance rather
    than to zero — and then capped from above by :func:`feasible_distance_cap`,
    which is what keeps that floor from vouching for a point no small move can
    make feasible (#1254).

    ``None`` means the Jacobian was unavailable, and the caller must then fall
    back to the Jacobian-free bound — which is STRICTER, so the fallback can only
    reject a point the full form would have accepted. It can never accept one the
    full form would reject.
    """
    try:
        J = screen_jacobian(evaluator, x_flat)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        logger.debug("feasibility: Jacobian unavailable, using the stricter bound: %s", exc)
        return None, None
    if J.ndim != 2 or J.shape[0] <= int(rows.max()):
        logger.debug("feasibility: Jacobian shape %s cannot cover rows; stricter bound", J.shape)
        return None, None
    sub, all_finite = _jacobian_row_scales_checked(J[rows, :], x_flat)
    if not all_finite:
        logger.debug("feasibility: non-finite Jacobian entry; stricter bound")
        return None, None
    if box is None:
        return sub, jacobian_row_gradient_norms(J[rows, :])
    lb, ub, int_mask, direction = box
    return sub, improving_gradient_norms(J[rows, :], x_flat, lb, ub, direction, int_mask)


def check_constraints(model, x_flat: np.ndarray, evaluator=None) -> VerifyResult:
    """Every constraint row, enumerated from the evaluator's own row map."""
    if evaluator is None:
        # #75: via the dispatcher, so the selected backend is honoured and the
        # jax import stays inside its fallback. A direct `cached_evaluator`
        # import here put JAX on every solve that validates a point.
        from discopt._tape_nlp_evaluator import make_evaluator

        evaluator = make_evaluator(model)
    verdict = _row_verdicts(model, x_flat, evaluator)
    if verdict.error is not None:
        return VerifyResult(False, None, verdict.error)
    if not np.any(verdict.failed):
        return VerifyResult(True)
    bad = np.nonzero(verdict.failed)[0]
    w = int(bad[int(np.argmax(verdict.viol[bad] - verdict.allowed[bad]))])
    if not verdict.scaled:
        return VerifyResult(False, None, f"row {w} violated by {verdict.viol[w]:.3e}")
    return VerifyResult(
        False,
        None,
        f"row {w} violated by {verdict.viol[w]:.3e} (allowed {verdict.allowed[w]:.3e})",
    )


@dataclass(frozen=True)
class _RowVerdicts:
    """Per-row outcome of :func:`check_constraints`'s test.

    ``failed[i]`` is the verdict on flat row ``i``; ``viol`` its violation and
    ``allowed`` the allowance it was judged against (``ABS_TOL * anchor`` for a row
    that never became a suspect). ``error`` is set, and the arrays are empty, when
    the rows could not be judged at all -- a caller must then refuse to vouch.
    ``scaled`` records whether the scale-aware allowance was computed.
    """

    failed: np.ndarray
    viol: np.ndarray
    allowed: np.ndarray
    scaled: bool = False
    error: Optional[str] = None


def _row_verdicts(model, x_flat: np.ndarray, evaluator) -> _RowVerdicts:
    empty = np.zeros(0, dtype=np.float64)

    def refuse(reason: str) -> _RowVerdicts:
        return _RowVerdicts(np.zeros(0, dtype=bool), empty, empty, error=reason)

    if evaluator.n_constraints <= 0:
        return _RowVerdicts(np.zeros(0, dtype=bool), empty, empty)

    g = np.asarray(evaluator.evaluate_constraints(x_flat), dtype=np.float64)
    row_map = evaluator.constraint_row_map()
    n_rows = row_map[-1][1] if row_map else 0
    if g.shape[0] < n_rows:
        # The evaluator produced fewer rows than its own map claims. Refuse to
        # vouch rather than check a prefix.
        return refuse(f"evaluator produced {g.shape[0]} rows, map wants {n_rows}")

    # Pass 1 — residuals under the Jacobian-FREE (stricter) bound. Rows that
    # clear this also clear the scale-aware bound, which is never smaller, so the
    # Jacobian is computed only when some row is actually near or over the line.
    viol = np.zeros(n_rows, dtype=np.float64)
    direction = np.zeros(n_rows, dtype=np.float64)
    anchor = np.ones(n_rows, dtype=np.float64)
    for start, stop, con in row_map:
        sense = _sense_str(con)
        if sense is None:
            return refuse(f"unknown constraint sense {con.sense!r}")
        # Honour Constraint.rhs. Bodies built through the operator API are
        # normalised to rhs == 0, but the field is settable and the evaluator
        # compiles the BODY ONLY, so `body <sense> rhs` must be re-centred here.
        rhs = float(getattr(con, "rhs", 0.0) or 0.0)
        for i in range(start, stop):
            val = float(g[i]) - rhs
            if not math.isfinite(val):
                return refuse(f"non-finite residual in row {i}")
            viol[i] = _row_violation(val, sense)
            if viol[i] > 0.0:
                direction[i] = -1.0 if sense == ">=" else float(np.sign(val))
            anchor[i] = max(1.0, abs(rhs))
    allowed = ABS_TOL * anchor
    failed = np.zeros(n_rows, dtype=bool)

    # Pass 1 selects the rows that could fail EITHER bound. The absolute bound is
    # ``ABS_TOL * anchor``; the small-row cap (#1254) can be as low as
    # ``SMALL_ROW_ABS_FLOOR``, so a row violated by more than that floor is a
    # candidate too — without this term a row the cap rejects would short-circuit
    # to "feasible" here and the cap would be a no-op on the very rows it exists
    # for. The Jacobian is still computed only when some row is actually near or
    # over a line, and an exactly-satisfied row (violation 0) never is.
    suspect = np.nonzero((viol > ABS_TOL * anchor) | (viol > SMALL_ROW_ABS_FLOOR))[0]
    if suspect.size == 0:
        return _RowVerdicts(failed, viol, allowed)

    # Pass 2 — only the suspect rows get the full scale-keyed bound.
    from discopt._relax.model_utils import flat_variable_bounds

    lb, ub = flat_variable_bounds(model)
    box = None
    if lb.size == np.asarray(x_flat).size:
        box = (lb, ub, model_integer_mask(model), direction[suspect])
    scales, grad_norms = _row_scales_and_gradients(evaluator, x_flat, suspect, box)
    if scales is None:
        # No Jacobian: no scale estimate, so no cap either (it would be a cap
        # built on nothing). The rows that fail the plain absolute bound are the
        # rejection, exactly as before the cap existed.
        failed[suspect] = viol[suspect] > allowed[suspect]
        return _RowVerdicts(failed, viol, allowed)
    allowed[suspect] = np.minimum(
        ABS_TOL * np.maximum(anchor[suspect], scales),
        feasible_distance_cap(grad_norms, scales),
    )
    failed[suspect] = viol[suspect] > allowed[suspect]
    return _RowVerdicts(failed, viol, allowed, scaled=True)


def max_constraint_violation(model, x_flat, evaluator=None) -> float:
    """Largest constraint-row violation of ``x_flat``, each row over ``max(1, |rhs|)``.

    A *ranking* measure, not a feasibility test: :func:`verify_point` decides
    whether a point may be an incumbent, and this says how much of the tolerance
    that decision allowed the point to use. Two points that both pass
    ``verify_point`` can differ by orders of magnitude here, and the more violated
    one can buy an objective past every truly feasible point (#1496). Satisfied
    rows count as ``0``; a non-finite residual, or an evaluator that cannot produce
    every row its own map declares, is ``inf`` -- the worst rank, never a pass.
    Variable bounds and integrality are not included; callers that rank points
    place them in the box first.
    """
    if evaluator is None:
        from discopt._tape_nlp_evaluator import make_evaluator

        evaluator = make_evaluator(model)
    if evaluator.n_constraints <= 0:
        return 0.0
    x_flat = np.asarray(x_flat, dtype=np.float64)
    g = np.asarray(evaluator.evaluate_constraints(x_flat), dtype=np.float64)
    row_map = evaluator.constraint_row_map()
    n_rows = row_map[-1][1] if row_map else 0
    if g.shape[0] < n_rows:
        return math.inf
    worst = 0.0
    for start, stop, con in row_map:
        sense = _sense_str(con)
        if sense is None:
            return math.inf
        rhs = float(getattr(con, "rhs", 0.0) or 0.0)
        anchor = max(1.0, abs(rhs))
        for i in range(start, stop):
            val = float(g[i]) - rhs
            if not math.isfinite(val):
                return math.inf
            worst = max(worst, _row_violation(val, sense) / anchor)
    return worst


class DeclaredVariable:
    """A frozen record of one declared variable: name, size, bounds and type.

    What :func:`declared_variables` captures before a solve, so a later
    :func:`verify_point` judges bounds and integrality on the variables the
    caller declared -- not on whatever the solve has since appended (structure
    cuts add auxiliary columns to ``model._variables``, #1561) or tightened in
    place.
    """

    __slots__ = ("name", "size", "shape", "lb", "ub", "var_type")

    def __init__(self, v):
        self.name = v.name
        self.size = int(getattr(v, "size", 1))
        self.shape = tuple(getattr(v, "shape", ()))
        self.lb = np.array(v.lb, dtype=np.float64, copy=True)
        self.ub = np.array(v.ub, dtype=np.float64, copy=True)
        self.var_type = v.var_type


def declared_variables(model) -> tuple:
    """Snapshot ``model._variables`` as :class:`DeclaredVariable` records."""
    return tuple(DeclaredVariable(v) for v in model._variables)


class _DeclaredModelView:
    """``model`` with ``_variables`` replaced by a declared snapshot.

    Every helper in this module reads columns, bounds and integrality from
    ``model._variables``; this view points them at the declared set while every
    other attribute resolves on the real model. It is never handed to an
    evaluator factory: :func:`verify_point` requires an explicit evaluator
    whenever ``variables`` is given.
    """

    def __init__(self, model, variables):
        self._model = model
        self._variables = list(variables)

    def __getattr__(self, name):
        return getattr(self._model, name)


# ── declared non-algebraic constraints (#1659) ──────────────────────────────


def _column_indices(expr, model) -> np.ndarray:
    """Flat columns of a variable or an indexed variable, in its own order.

    Indicators, SOS members and boolean variables are always one of these two
    forms; anything else is refused rather than guessed at.
    """
    from discopt.modeling.core import IndexExpression, Variable

    base, index = (expr.base, expr.index) if isinstance(expr, IndexExpression) else (expr, None)
    if not isinstance(base, Variable):
        raise TypeError(f"expected a variable or an indexed variable, got {type(expr).__name__}")
    offset = 0
    for v in model._variables:
        if v is base:
            break
        offset += int(v.size)
    else:
        raise ValueError(f"variable {base.name!r} is not a column of the model")
    cols = offset + np.arange(int(base.size)).reshape(tuple(base.shape) or ())
    if index is not None:
        cols = cols[index]
    return np.atleast_1d(np.asarray(cols, dtype=np.int64)).ravel()


class _StackedRows:
    """An evaluator whose rows are ``base``'s followed by chosen rows of ``extra``.

    ``extra_entries`` are indices into ``extra.constraint_row_map()``. Both
    evaluators are over the same columns. Everything that is not about rows (the
    objective, its gradient) is ``base``'s.
    """

    def __init__(self, base, extra, extra_entries):
        self._base = base
        self._extra = extra
        emap = extra.constraint_row_map()
        bmap = list(base.constraint_row_map()) if base.n_constraints > 0 else []
        off = bmap[-1][1] if bmap else 0
        self._base_rows = off
        rows: list[int] = []
        out = list(bmap)
        for k in extra_entries:
            start, stop, con = emap[k]
            out.append((off, off + (stop - start), con))
            off += stop - start
            rows.extend(range(start, stop))
        self._rows = np.asarray(rows, dtype=np.int64)
        self._map = out
        self.n_constraints = off

    def constraint_row_map(self):
        return list(self._map)

    def evaluate_constraints(self, x):
        parts = []
        if self._base_rows:
            parts.append(np.asarray(self._base.evaluate_constraints(x), dtype=np.float64))
        if self._rows.size:
            g = np.asarray(self._extra.evaluate_constraints(x), dtype=np.float64)
            parts.append(g[self._rows])
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)

    def evaluate_sparse_jacobian(self, x):
        blocks = []
        if self._base_rows:
            blocks.append(sparse.csr_matrix(_dense_jacobian(self._base, x)))
        if self._rows.size:
            blocks.append(sparse.csr_matrix(_dense_jacobian(self._extra, x))[self._rows])
        return sparse.vstack(blocks, format="csr")

    def evaluate_jacobian(self, x):
        return self.evaluate_sparse_jacobian(x).toarray()

    def evaluate_objective(self, x):
        return self._base.evaluate_objective(x)

    def evaluate_gradient(self, x):
        return self._base.evaluate_gradient(x)


class DeclaredLogic:
    """The declared disjunctive, indicator, SOS and logical constraints (#1659).

    The NLP evaluator compiles algebraic rows only, so on a model declared with
    ``either_or`` / ``if_then`` / ``sos1`` / ``logical`` the rows the caller wrote
    inside those constraints were never judged: ``verify_point`` saw zero rows and
    passed any point in the box. Measured on the #1659 witness, a hull incumbent
    violating its selected disjunct's row ``y >= exp(x) - 1`` by 5.9e-5 -- an
    objective 5.4e-5 below the true optimum -- was certified, and the #1537 E
    repair skipped the model as "non-algebraic".

    Construct once, BEFORE the solve, like the declared evaluator: the algebraic
    rows inside the logic are compiled now, so a later in-place rewrite of their
    expressions cannot change what is judged. Every row is judged by
    :func:`check_constraints`' own per-row test; the logic decides which rows
    must hold at the point's integral realisation:

    * a disjunction holds when at least one disjunct has every item holding
      (``SELECT_ONE`` projects to the union of its disjuncts; nested
      disjunctions recurse);
    * an indicator row must hold when the indicator equals its active value;
    * SOS1 allows one nonzero member, SOS2 two that are adjacent;
    * a logical constraint is evaluated on the booleans' integral values.
    """

    def __init__(self, model):
        from discopt.modeling.core import (
            Constraint,
            Model,
            _DisjunctiveConstraint,
            _IndicatorConstraint,
            _LogicalConstraint,
            _SOSConstraint,
        )

        self._leaves: list = []
        self._nodes: list = []
        self._bool_cols: dict[int, int] = {}

        def leaf(c):
            self._leaves.append(c)
            return ("leaf", len(self._leaves) - 1)

        def disjunction(c):
            arms = []
            for disjunct in c.disjuncts:
                items = []
                for item in disjunct:
                    if type(item) is Constraint:
                        items.append(leaf(item))
                    elif isinstance(item, _DisjunctiveConstraint):
                        items.append(disjunction(item))
                    else:
                        raise TypeError(
                            f"disjunct item of type {type(item).__name__} is not checkable"
                        )
                arms.append(("all", items))
            return ("any", arms, c.name)

        for c in model._constraints:
            if type(c) is Constraint:
                continue
            if isinstance(c, _DisjunctiveConstraint):
                self._nodes.append(disjunction(c))
            elif isinstance(c, _IndicatorConstraint):
                cols = _column_indices(c.indicator, model)
                if cols.size != 1:
                    raise ValueError(f"indicator of {c.name!r} is not a scalar")
                self._nodes.append(
                    ("ind", int(cols[0]), float(c.active_value), leaf(c.constraint), c.name)
                )
            elif isinstance(c, _SOSConstraint):
                cols = np.concatenate([_column_indices(v, model) for v in c.variables])
                self._nodes.append(("sos", int(c.sos_type), cols, c.name))
            elif isinstance(c, _LogicalConstraint):
                self._index_booleans(c.expression, model)
                self._nodes.append(("logic", c.expression, c.name))
            else:
                raise TypeError(f"constraint of type {type(c).__name__} is not checkable")

        self._evaluator = None
        if self._leaves:
            from discopt._tape_nlp_evaluator import make_evaluator
            from discopt.modeling.core import Constant

            sub = Model(f"{model.name}__declared_logic")
            sub._variables = list(model._variables)
            sub._constraints = list(self._leaves)
            sub.minimize(Constant(0.0))
            self._evaluator = make_evaluator(sub)
            entries = self._evaluator.constraint_row_map()
            if len(entries) != len(self._leaves) or any(
                e[2] is not c for e, c in zip(entries, self._leaves)
            ):
                raise RuntimeError("declared-logic row map does not follow its constraints")

    def __bool__(self) -> bool:
        return bool(self._nodes)

    def _index_booleans(self, expr, model) -> None:
        from discopt.modeling.core import (
            BooleanVar,
            LogicalAnd,
            LogicalAtLeast,
            LogicalAtMost,
            LogicalEquivalent,
            LogicalExactly,
            LogicalImplies,
            LogicalNot,
            LogicalOr,
        )

        if isinstance(expr, BooleanVar):
            cols = _column_indices(expr.variable, model)
            if cols.size != 1:
                raise ValueError("a boolean in a logical constraint is not a scalar")
            self._bool_cols[id(expr)] = int(cols[0])
        elif isinstance(expr, (LogicalAnd, LogicalOr, LogicalEquivalent)):
            self._index_booleans(expr.left, model)
            self._index_booleans(expr.right, model)
        elif isinstance(expr, LogicalNot):
            self._index_booleans(expr.operand, model)
        elif isinstance(expr, LogicalImplies):
            self._index_booleans(expr.antecedent, model)
            self._index_booleans(expr.consequent, model)
        elif isinstance(expr, (LogicalAtLeast, LogicalAtMost, LogicalExactly)):
            for op in expr.operands:
                self._index_booleans(op, model)
        else:
            raise TypeError(f"logical node {type(expr).__name__} is not checkable")

    def _truth(self, expr, x) -> bool:
        from discopt.modeling.core import (
            BooleanVar,
            LogicalAnd,
            LogicalAtLeast,
            LogicalAtMost,
            LogicalEquivalent,
            LogicalExactly,
            LogicalImplies,
            LogicalNot,
            LogicalOr,
        )

        if isinstance(expr, BooleanVar):
            return bool(round(float(x[self._bool_cols[id(expr)]])) == 1)
        if isinstance(expr, LogicalAnd):
            return self._truth(expr.left, x) and self._truth(expr.right, x)
        if isinstance(expr, LogicalOr):
            return self._truth(expr.left, x) or self._truth(expr.right, x)
        if isinstance(expr, LogicalEquivalent):
            return self._truth(expr.left, x) == self._truth(expr.right, x)
        if isinstance(expr, LogicalNot):
            return not self._truth(expr.operand, x)
        if isinstance(expr, LogicalImplies):
            return (not self._truth(expr.antecedent, x)) or self._truth(expr.consequent, x)
        n_true = sum(self._truth(op, x) for op in expr.operands)
        if isinstance(expr, LogicalAtLeast):
            return n_true >= expr.k
        if isinstance(expr, LogicalAtMost):
            return n_true <= expr.k
        if isinstance(expr, LogicalExactly):
            return n_true == expr.k
        raise TypeError(f"logical node {type(expr).__name__} is not checkable")

    def _leaf_scores(self, model, x) -> np.ndarray:
        """Per leaf: worst ``violation / allowance`` over its rows (``<= 1`` holds)."""
        if self._evaluator is None:
            return np.zeros(0, dtype=np.float64)
        v = _row_verdicts(model, x, self._evaluator)
        if v.error is not None:
            raise ValueError(f"declared-logic rows: {v.error}")
        # ``allowed`` is positive on every row (``ABS_TOL * anchor`` at least), and a
        # row fails exactly when this ratio exceeds 1.
        ratio = v.viol / v.allowed
        out = np.zeros(len(self._leaves), dtype=np.float64)
        for k, (start, stop, _con) in enumerate(self._evaluator.constraint_row_map()):
            if stop > start:
                out[k] = float(np.max(ratio[start:stop]))
        return out

    def _score(self, node, scores, x) -> float:
        kind = node[0]
        if kind == "leaf":
            return float(scores[node[1]])
        if kind == "all":
            return max((self._score(n, scores, x) for n in node[1]), default=0.0)
        if kind == "any":
            return min(self._score(n, scores, x) for n in node[1])
        if kind == "ind":
            _k, col, active, sub, _name = node
            return self._score(sub, scores, x) if float(x[col]) == active else 0.0
        raise AssertionError(kind)

    def _realise(self, node, scores, x, out: list[int]) -> None:
        kind = node[0]
        if kind == "leaf":
            out.append(node[1])
        elif kind == "all":
            for n in node[1]:
                self._realise(n, scores, x, out)
        elif kind == "any":
            best = min(node[1], key=lambda n: self._score(n, scores, x))
            self._realise(best, scores, x, out)
        elif kind == "ind":
            _k, col, active, sub, _name = node
            if float(x[col]) == active:
                self._realise(sub, scores, x, out)

    def check(self, model, x_snapped) -> VerifyResult:
        """Verdict on the point's integral realisation (see the class docstring)."""
        x = np.asarray(x_snapped, dtype=np.float64)
        scores = self._leaf_scores(model, x)
        for pos, node in enumerate(self._nodes):
            kind = node[0]
            label = node[-1] if node[-1] is not None else f"#{pos}"
            if kind in ("any", "ind"):
                worst = self._score(node, scores, x)
                if worst > 1.0:
                    what = "disjunction" if kind == "any" else "indicator constraint"
                    return VerifyResult(
                        False,
                        None,
                        f"{what} {label!r} not satisfied "
                        f"(closest row violation {worst:.3g}x its allowance)",
                    )
            elif kind == "sos":
                _k, sos_type, cols, _name = node
                nz = np.nonzero(np.abs(x[cols]) > ABS_TOL)[0]
                ok = nz.size <= 1 or (sos_type == 2 and nz.size == 2 and nz[1] - nz[0] == 1)
                if not ok:
                    return VerifyResult(
                        False, None, f"SOS{sos_type} {label!r} has nonzero members {nz.tolist()}"
                    )
            elif kind == "logic":
                if not self._truth(node[1], x):
                    return VerifyResult(False, None, f"logical constraint {label!r} is false")
        return VerifyResult(True)

    def realised_evaluator(self, model, x_snapped, base_evaluator):
        """``base_evaluator``'s rows plus the logic rows that must hold at the point.

        Each disjunction contributes the disjunct closest to holding (by the same
        per-row allowance :meth:`check` uses), each active indicator its row. SOS
        and logical constraints contribute no rows; a move that breaks one is
        caught by re-checking the moved point.
        """
        x = np.asarray(x_snapped, dtype=np.float64)
        scores = self._leaf_scores(model, x)
        chosen: list[int] = []
        for node in self._nodes:
            if node[0] in ("any", "ind"):
                self._realise(node, scores, x, chosen)
        if not chosen:
            return base_evaluator
        return _StackedRows(base_evaluator, self._evaluator, sorted(set(chosen)))


def has_declared_logic(model) -> bool:
    """True when ``model`` declares any non-algebraic constraint."""
    from discopt.modeling.core import Constraint

    return any(type(c) is not Constraint for c in model._constraints)


def verify_point(
    model,
    x_flat,
    *,
    with_objective: bool = False,
    evaluator=None,
    variables=None,
    logic: Optional[DeclaredLogic] = None,
) -> VerifyResult:
    """Verify ``x_flat`` is feasible for ``model``; optionally return its objective.

    The contract is strict, because callers use this to decide whether a value may
    seed an incumbent cutoff and an unverified seed poisons every downstream
    certificate: this returns ``ok=True`` ONLY when the evaluator successfully
    evaluated every constraint row and every residual, bound and integrality
    condition is within tolerance. Any evaluator failure, shape mismatch or
    non-finite value yields ``ok=False`` — never an optimistic pass.

    ``evaluator`` (#1561) supplies the rows instead of compiling them from
    ``model`` now. ``Model.solve`` passes the evaluator it snapshotted BEFORE the
    solve, because presolve may rewrite the constraint DAG in place and root cuts
    may append rows; the rows judged are then the ones the caller declared. Bounds
    and integrality are read from ``model`` unless ``variables`` is given.

    ``variables`` (#1561) is the declared variable set -- a
    :func:`declared_variables` snapshot -- that ``x_flat`` is laid out over and
    that bounds and integrality are judged on. A solve can append auxiliary
    columns to ``model._variables``; judging the declared point against the
    extended list is a length mismatch, not a verdict. It requires ``evaluator``
    (built on the same declared model), since the rows must match the columns.

    ``logic`` (#1659) is a :class:`DeclaredLogic` for the model's disjunctive,
    indicator, SOS and logical constraints, which the evaluator does not compile.
    Without one it is built from ``model`` when the model declares any; with
    ``variables`` it must be passed, built before the solve like the evaluator.
    """
    from discopt.modeling.core import ObjectiveSense

    if variables is not None:
        if evaluator is None:
            raise ValueError("verify_point(variables=...) requires an explicit evaluator")
        model = _DeclaredModelView(model, variables)

    x_flat = np.asarray(x_flat, dtype=np.float64)
    if x_flat.ndim != 1 or not np.all(np.isfinite(x_flat)):
        return VerifyResult(False, None, "point is not a finite 1-D vector")
    if variables is not None:
        n_declared = sum(int(v.size) for v in model._variables)
        if x_flat.shape[0] != n_declared:
            return VerifyResult(
                False, None, f"point has {x_flat.shape[0]} columns, declared {n_declared}"
            )

    res = check_variable_bounds(model, x_flat)
    if not res.ok:
        return res

    # #1380: verify the point the solver is CLAIMING -- its integral realisation
    # -- not the fractional point it computed. ``check_variable_bounds`` above has
    # just proved every integer column is within INT_TOL of an integer, so the
    # snap moves nothing further than that; what it removes is the ability of a
    # column at 1e-6 to buy M*1e-6 of slack on a big-M row.
    x_flat = snap_integers(model, x_flat)
    res = check_variable_bounds(model, x_flat)
    if not res.ok:
        # The integral realisation is out of bounds -- the claim is not feasible.
        return VerifyResult(False, None, f"{res.reason} (at the integral point)")

    try:
        # #75: via the dispatcher, so the selected backend is honoured and the
        # jax import stays inside its fallback. A direct `cached_evaluator`
        # import here put JAX on every solve that validates a point.
        if evaluator is None:
            from discopt._tape_nlp_evaluator import make_evaluator

            evaluator = make_evaluator(model)
        res = check_constraints(model, x_flat, evaluator=evaluator)
        if not res.ok:
            return res
        if logic is None and has_declared_logic(model):
            if variables is not None:
                raise ValueError(
                    "the model declares non-algebraic constraints; verify_point("
                    "variables=...) needs the DeclaredLogic built with the evaluator"
                )
            logic = DeclaredLogic(model)
        if logic:
            res = logic.check(model, x_flat)
            if not res.ok:
                return res
        if not with_objective:
            return VerifyResult(True)
        obj_min = float(evaluator.evaluate_objective(x_flat))
    except Exception as exc:  # noqa: BLE001 - the evaluator could not vouch
        logger.debug("feasibility verification declined (evaluator error): %s", exc)
        return VerifyResult(False, None, f"evaluator error: {exc}")

    if not math.isfinite(obj_min):
        return VerifyResult(False, None, "non-finite objective")
    # ``evaluate_objective`` minimises the negation for a MAXIMIZE model; undo
    # that so the returned value is the objective in model units.
    model_obj = -obj_min if model._objective.sense == ObjectiveSense.MAXIMIZE else obj_min
    return VerifyResult(True, float(model_obj))


# ── incumbent repair (#1537 E) ────────────────────────────────────────────────

#: Most Newton steps :func:`repair_point` takes. Measured on the three #1537 E
#: cells: one step takes each residual from 1e-9..7e-6 to float noise.
REPAIR_MAX_ITER = 8
#: Per-term round-off coefficient of the noise floor a repaired row is driven to:
#: 16 ulps of the row's summed term magnitude, the allowance
#: :func:`improving_gradient_norms` already uses for the same arithmetic.
REPAIR_NOISE_ULPS = 16.0
#: Largest dense ``|active rows| x |free columns|`` system the repair will factor.
#: Above it the repair declines (and says so) rather than stall a solve's exit.
REPAIR_MAX_DENSE = 4_000_000


@dataclass(frozen=True)
class RepairResult:
    """What :func:`repair_point` did. ``x`` is ``None`` when it declined, and
    ``reason`` then says why (for logs and ``solver_stats``, never control flow)."""

    x: Optional[np.ndarray]
    excess_before: float
    excess_after: float
    iterations: int
    reason: str = ""


def _row_senses_and_rhs(evaluator):
    """``(sense, rhs, None)`` per flat row, from the evaluator's own row map.

    A row whose sense is not one of ``<=``/``>=``/``==`` returns ``(None, None,
    reason)`` rather than raising: ``verify_point`` reports such a point not-ok, and
    the repair -- a post-solve publication step -- declines rather than crash the
    solve that produced the point.
    """
    senses: list[str] = []
    rhs: list[float] = []
    for start, stop, con in evaluator.constraint_row_map():
        s = _sense_str(con)
        if s is None:
            return None, None, f"unknown constraint sense {getattr(con, 'sense', None)!r}"
        r = float(getattr(con, "rhs", 0.0) or 0.0)
        senses.extend([s] * (stop - start))
        rhs.extend([r] * (stop - start))
    return np.array(senses, dtype=object), np.array(rhs, dtype=np.float64), None


def _signed_rows(evaluator, x, senses, rhs):
    """``(residual, violation)``: ``body - rhs`` and the per-row violation (> 0 = violated)."""
    g = np.asarray(evaluator.evaluate_constraints(x), dtype=np.float64)
    if g.shape[0] != rhs.shape[0]:
        raise ValueError(f"evaluator produced {g.shape[0]} rows, map wants {rhs.shape[0]}")
    r = g - rhs
    viol = np.where(senses == "<=", r, np.where(senses == ">=", -r, np.abs(r)))
    return r, viol


def _dense_jacobian(evaluator, x):
    sparse_fn = getattr(evaluator, "evaluate_sparse_jacobian", None)
    if sparse_fn is not None:
        return sparse.csr_matrix(sparse_fn(x), dtype=np.float64)
    return sparse.csr_matrix(np.asarray(evaluator.evaluate_jacobian(x), dtype=np.float64))


def _noise_floor(J, x, r, rhs):
    """Per-row float-evaluation noise: ``16 ulp * (sum_j |J_ij x_j| + |body| + |rhs|)``.

    The residual a row can be driven to in float64 at this point, whatever units it
    is written in: it is covariant with a row rescaling (every term scales) and, under
    a translation, grows only with the magnitude the arithmetic really carries.
    """
    term = np.asarray(abs(J) @ np.abs(x), dtype=np.float64).ravel()
    return REPAIR_NOISE_ULPS * _EPS * (term + np.abs(r + rhs) + np.abs(rhs))


def repair_point(model, x_flat, *, evaluator, variables=None) -> RepairResult:
    """Project ``x_flat`` onto its own rows, in the units the model is written in (#1537 E).

    Why. :func:`verify_point` allows a row ``ABS_TOL * max(1, |rhs|, max_j |J_ij x_j|)``,
    and that allowance is not invariant under a change of variables that leaves the
    model identical: the floor ``1`` does not scale with the row, and ``|x_j|``
    grows with the distance of the origin. A solver's own iterates carry residuals
    at its working accuracy (LP feasibility, NLP convergence), and the allowance lets
    them through as the published incumbent. Measured on the in-repo corpus
    (``test_1537_invariance.py``): ``ex14_1_9`` with rows x1e-3 certified ``x1 =
    -9.98e-6`` against a true minimum of ~0 (row violated 9.98e-9 in the solver's
    units = 9.98e-6 in the original's, allowed 1e-6 in each); ``ex1225`` shifted by
    ~1e6 certified 30.9999959 against 31 on an equality residual of 2.1e-6 the
    shifted row's term scale allowed ~1; ``syn05hfsg`` likewise, 6.7e-6. Each is a
    point the solver's tolerance exploits, and the objective it reports is bought
    with that exploitation.

    The fix is not a different tolerance -- no fixed absolute allowance can be
    invariant under row scaling -- but a cleaner point. Newton steps on the rows
    that are violated beyond float noise (plus every equality row, so a repair
    cannot break one) move the continuous columns by the minimum-norm correction,
    with integer columns snapped and frozen and every column clipped into its box.
    The minimum-norm step is itself invariant under both transforms: a row rescale
    scales both sides of ``J dx = -r``, a translation changes neither. The result
    is feasible to float noise in ANY units, so the caller's verdict no longer
    depends on which coordinates the model was written in.

    Accepted only if no row ends more violated than it began (beyond noise) and the
    largest violation-beyond-noise strictly falls. The caller must still run
    :func:`verify_point` on the result and re-evaluate the objective there: this
    function moves the point, it does not vouch for it.

    ``variables`` is a :func:`declared_variables` snapshot, as in
    :func:`verify_point`; the evaluator must be built on the same declared model.
    """
    if variables is not None:
        model = _DeclaredModelView(model, variables)
    x0 = np.asarray(x_flat, dtype=np.float64)
    if x0.ndim != 1 or not np.all(np.isfinite(x0)):
        return RepairResult(None, math.inf, math.inf, 0, "point is not a finite 1-D vector")
    lb = np.concatenate([np.ravel(np.asarray(v.lb, dtype=np.float64)) for v in model._variables])
    ub = np.concatenate([np.ravel(np.asarray(v.ub, dtype=np.float64)) for v in model._variables])
    if lb.size != x0.size:
        return RepairResult(None, math.inf, math.inf, 0, "point does not match the columns")
    if evaluator.n_constraints <= 0:
        return RepairResult(None, 0.0, 0.0, 0, "no rows")

    senses, rhs, sense_problem = _row_senses_and_rhs(evaluator)
    if sense_problem is not None:
        # Excess unknown, not zero: the caller records the decline.
        return RepairResult(None, math.inf, math.inf, 0, sense_problem)
    frozen = model_integer_mask(model) | (lb == ub)
    # E's repair, first half: the point the solver CLAIMS -- integers snapped
    # (#1380) and every column inside its declared box.
    x = np.clip(snap_integers(model, x0), lb, ub)
    r, viol = _signed_rows(evaluator, x, senses, rhs)
    if not np.all(np.isfinite(r)):
        return RepairResult(None, math.inf, math.inf, 0, "non-finite residual")
    J = _dense_jacobian(evaluator, x)
    noise0 = _noise_floor(J, x, r, rhs)
    viol0 = viol.copy()
    excess0 = float(np.max(np.maximum(viol0 - noise0, 0.0), initial=0.0))
    if excess0 <= 0.0:
        return RepairResult(None, 0.0, 0.0, 0, "already feasible to float noise")

    active = (senses == "==") | (viol > noise0)
    best_x, best_excess, it = None, excess0, 0
    for it in range(1, REPAIR_MAX_ITER + 1):
        rows = np.nonzero(active)[0]
        cols = np.nonzero(~frozen)[0]
        if rows.size == 0 or cols.size == 0:
            break
        if rows.size * cols.size > REPAIR_MAX_DENSE:
            return RepairResult(
                None, excess0, excess0, it, f"system {rows.size}x{cols.size} too large"
            )
        A = J[rows][:, cols].toarray()
        if not np.all(np.isfinite(A)):
            break
        step, *_ = np.linalg.lstsq(A, -r[rows], rcond=None)
        xn = x.copy()
        xn[cols] += step
        hit = (xn < lb) | (xn > ub)
        xn = np.clip(xn, lb, ub)
        rn, vn = _signed_rows(evaluator, xn, senses, rhs)
        if not np.all(np.isfinite(rn)):
            break
        Jn = _dense_jacobian(evaluator, xn)
        noise_n = _noise_floor(Jn, xn, rn, rhs)
        x, r, viol, J = xn, rn, vn, Jn
        frozen = frozen | hit
        active = active | (viol > noise_n)
        excess = float(np.max(np.maximum(viol - noise_n, 0.0), initial=0.0))
        # No row may end more violated than it began, beyond either noise floor.
        no_row_worse = bool(np.all(viol <= np.maximum(viol0, np.maximum(noise0, noise_n))))
        if no_row_worse and excess < best_excess:
            best_x, best_excess = x.copy(), excess
        if excess <= 0.0:
            break
    if best_x is None:
        return RepairResult(None, excess0, excess0, it, "no step reduced the violation")
    return RepairResult(best_x, excess0, best_excess, it)


def incumbent_repair_enabled() -> bool:
    """``DISCOPT_INCUMBENT_REPAIR`` (#1537 E): repair the published incumbent.

    See :func:`repair_point`. ``=0`` publishes the solver's point unrepaired.

    Default ON from introduction. The §5 panel (2026-10-02, the 66 in-repo
    MINLPLib instances, 20 s, arms interleaved) was cert-clean: incorrect 0 in
    both arms, certified 49 -> 49, no certificate lost or gained, 16 incumbents
    repaired with an objective shift of at most 2.4e-7. ``=0`` is kept as an
    opt-out for A/B-ing the published point against the solver's raw one, not
    as a parked graduation.
    """
    import os

    return os.environ.get("DISCOPT_INCUMBENT_REPAIR", "1") != "0"
