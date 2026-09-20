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
    """
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
    """
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
        up = np.maximum(ub - x, 0.0) + slack * (np.abs(ub) + np.abs(x))
        down = np.maximum(x - lb, 0.0) + slack * (np.abs(lb) + np.abs(x))
        up = up[None, :]
        down = down[None, :]
        room = np.where(step > 0, up, np.where(step < 0, down, 0.0))
        if integer_mask is not None:
            mask = np.asarray(integer_mask, dtype=bool).ravel()
            if mask.size != n:
                raise ValueError(f"integer_mask has {mask.size} entries for {n} columns")
            r = np.clip(np.round(x), np.ceil(lb), np.floor(ub))
            gap = r - x
            to_int = np.abs(gap) + slack * (np.abs(r) + np.abs(x))
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
    J = np.asarray(J, dtype=np.float64)
    xw = np.abs(np.asarray(x_flat, dtype=np.float64))
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
        J = np.asarray(evaluator.evaluate_jacobian(x_flat), dtype=np.float64)
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

    if evaluator.n_constraints <= 0:
        return VerifyResult(True)

    g = np.asarray(evaluator.evaluate_constraints(x_flat), dtype=np.float64)
    row_map = evaluator.constraint_row_map()
    n_rows = row_map[-1][1] if row_map else 0
    if g.shape[0] < n_rows:
        # The evaluator produced fewer rows than its own map claims. Refuse to
        # vouch rather than check a prefix.
        return VerifyResult(
            False, None, f"evaluator produced {g.shape[0]} rows, map wants {n_rows}"
        )

    # Pass 1 — residuals under the Jacobian-FREE (stricter) bound. Rows that
    # clear this also clear the scale-aware bound, which is never smaller, so the
    # Jacobian is computed only when some row is actually near or over the line.
    viol = np.zeros(n_rows, dtype=np.float64)
    direction = np.zeros(n_rows, dtype=np.float64)
    anchor = np.ones(n_rows, dtype=np.float64)
    for start, stop, con in row_map:
        sense = _sense_str(con)
        if sense is None:
            return VerifyResult(False, None, f"unknown constraint sense {con.sense!r}")
        # Honour Constraint.rhs. Bodies built through the operator API are
        # normalised to rhs == 0, but the field is settable and the evaluator
        # compiles the BODY ONLY, so `body <sense> rhs` must be re-centred here.
        rhs = float(getattr(con, "rhs", 0.0) or 0.0)
        for i in range(start, stop):
            val = float(g[i]) - rhs
            if not math.isfinite(val):
                return VerifyResult(False, None, f"non-finite residual in row {i}")
            viol[i] = _row_violation(val, sense)
            if viol[i] > 0.0:
                direction[i] = -1.0 if sense == ">=" else float(np.sign(val))
            anchor[i] = max(1.0, abs(rhs))

    # Pass 1 selects the rows that could fail EITHER bound. The absolute bound is
    # ``ABS_TOL * anchor``; the small-row cap (#1254) can be as low as
    # ``SMALL_ROW_ABS_FLOOR``, so a row violated by more than that floor is a
    # candidate too — without this term a row the cap rejects would short-circuit
    # to "feasible" here and the cap would be a no-op on the very rows it exists
    # for. The Jacobian is still computed only when some row is actually near or
    # over a line, and an exactly-satisfied row (violation 0) never is.
    suspect = np.nonzero((viol > ABS_TOL * anchor) | (viol > SMALL_ROW_ABS_FLOOR))[0]
    if suspect.size == 0:
        return VerifyResult(True)

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
        hard = suspect[viol[suspect] > ABS_TOL * anchor[suspect]]
        if hard.size == 0:
            return VerifyResult(True)
        worst = int(hard[int(np.argmax(viol[hard]))])
        return VerifyResult(False, None, f"row {worst} violated by {viol[worst]:.3e}")
    allowed = np.minimum(
        ABS_TOL * np.maximum(anchor[suspect], scales),
        feasible_distance_cap(grad_norms, scales),
    )
    over = viol[suspect] > allowed
    if np.any(over):
        k = int(np.argmax(np.where(over, viol[suspect] - allowed, -np.inf)))
        w = int(suspect[k])
        return VerifyResult(
            False, None, f"row {w} violated by {viol[w]:.3e} (allowed {allowed[k]:.3e})"
        )
    return VerifyResult(True)


def verify_point(
    model,
    x_flat,
    *,
    with_objective: bool = False,
) -> VerifyResult:
    """Verify ``x_flat`` is feasible for ``model``; optionally return its objective.

    The contract is strict, because callers use this to decide whether a value may
    seed an incumbent cutoff and an unverified seed poisons every downstream
    certificate: this returns ``ok=True`` ONLY when the evaluator successfully
    evaluated every constraint row and every residual, bound and integrality
    condition is within tolerance. Any evaluator failure, shape mismatch or
    non-finite value yields ``ok=False`` — never an optimistic pass.
    """
    from discopt.modeling.core import ObjectiveSense

    x_flat = np.asarray(x_flat, dtype=np.float64)
    if x_flat.ndim != 1 or not np.all(np.isfinite(x_flat)):
        return VerifyResult(False, None, "point is not a finite 1-D vector")

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
        from discopt._tape_nlp_evaluator import make_evaluator

        evaluator = make_evaluator(model)
        res = check_constraints(model, x_flat, evaluator=evaluator)
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
