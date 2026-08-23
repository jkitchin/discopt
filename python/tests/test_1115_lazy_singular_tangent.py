"""Vertical-tangent recovery placed where the LP binds (issue #1115).

#1111 recovered the facet ``_emit_1d`` drops at a vertical tangent (``sqrt`` at
``t=0``, ``asin``/``acos`` at ``t=±1``) by emitting ONE extra relaxation row at a
fixed geometric anchor near the endpoint, on every node. That is sound but
measurably harmful: on ``tspn08`` the tree grows 135 -> 191 nodes (+41 %) for a
bound gain in the 11th digit, because a row that rarely binds still moves the LP
vertex and therefore the branching choice.

``SolverTuning.singular_tangent_lazy`` (default ON, and consulted only when
``singular_tangent`` is on) moves the facet into separation:
``MccormickLPRelaxer._separate_singular_tangent`` adds the supporting tangent at
the CURRENT LP point, and only when that point is actually violated. Where the LP
vertex sits exactly on the singularity — the case the facet exists for — no
finite-slope tangent is available there, so the touch point falls back to #1111's
conditioning-capped ladder anchor while the violation test still decides whether
the row goes in.

This file locks the properties that gate a bound-CHANGING change (CLAUDE.md §5):

1. **Inert unless asked.** With recovery off no spec is registered and the
   separator appends nothing, so the default polytope is unchanged.
2. **Soundness — no feasible point is cut.** Every row the separator appends
   holds at the exact lifted graph point ``(t, f(t))``, densely sampled over the
   box including the singular endpoint itself.
3. **Differential bound.** The separated bound is ``>=`` the un-separated one
   (structural: the path only ADDS rows) and still ``<=`` the true optimum over
   the same fixed box.
4. **It only fires where it is needed.** An LP vertex already on the graph draws
   no row at all.

Each test carries an executed-comparison count and asserts it is non-zero
(CLAUDE.md §6): a probe that silently traverses nothing reports "0 violations"
and reads as a pass.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver_tuning as solver_tuning
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.mccormick_lp import MccormickLPRelaxer
from discopt._relax.milp_relaxation import DiscretizationState, build_milp_relaxation
from discopt._relax.term_classifier import classify_nonlinear_terms
from scipy.optimize import minimize_scalar

# ``relaxation``: theorem-style property tests over the separated rows, fast, and
# NOT deselected by the default addopts — the soundness and differential-bound
# checks run on every PR.
pytestmark = [pytest.mark.relaxation]

#: ``(label, atom, numpy f, lo, hi, a, wsgn)`` — boxes whose endpoint derivative
#: diverges, paired with an objective ``min a*x + wsgn*y`` whose LP vertex lands on
#: the singular endpoint, i.e. exactly where the dropped facet is missing. Three
#: ``sqrt`` boxes nine orders of magnitude apart pin scale-freeness. Generic
#: atoms and boxes; no named instance (CLAUDE.md §2).
BINDING_CASES = [
    ("sqrt[0,4]", dm.sqrt, np.sqrt, 0.0, 4.0, 2.0, -1.0),
    ("sqrt[0,1e-3]", dm.sqrt, np.sqrt, 0.0, 1e-3, 20.0, -1.0),
    ("sqrt[0,1e6]", dm.sqrt, np.sqrt, 0.0, 1e6, 0.002, -1.0),
]

#: Same boxes, but an objective whose LP vertex sits ON the graph, where the
#: static envelope is already exact. Nothing may be separated here.
SLACK_CASES = [
    ("sqrt[0,4]", dm.sqrt, np.sqrt, 0.0, 4.0, 0.0, -1.0),
    ("sqrt[0,1e6]", dm.sqrt, np.sqrt, 0.0, 1e6, 0.0, -1.0),
]

#: Boxes with no divergent endpoint derivative — no spec may ever be registered.
REGULAR_CASES = [
    ("sqrt[1,4]", dm.sqrt, 1.0, 4.0),
    ("log[0.5,3]", dm.log, 0.5, 3.0),
    ("exp[-2,2]", dm.exp, -2.0, 2.0),
]


def _model(atom, lo, hi, a=0.0, wsgn=-1.0):
    """``y == atom(x)`` over ``x in [lo,hi]``, minimizing ``a*x + wsgn*y``."""
    m = dm.Model()
    span = max(abs(lo), abs(hi), 1.0)
    x = m.continuous("x", lb=lo, ub=hi)
    y = m.continuous("y", lb=-1e3 * span, ub=1e3 * span)
    m.subject_to(y == atom(x))
    m.minimize(a * x + wsgn * y)
    return m


def _tuning(mode: str):
    """``off`` / ``eager`` / ``lazy`` as a pinned :class:`SolverTuning`."""
    return solver_tuning.current().replace(
        singular_tangent=(mode != "off"),
        singular_tangent_lazy=(mode == "lazy"),
    )


def _separate(atom, lo, hi, a, wsgn, mode: str):
    """Build the node relaxation, solve it, run the separator, report the deltas.

    Returns ``(n_specs, n_rows_before, n_rows_after, bound_before, bound_after,
    A_new, b_new, n_orig)`` where ``A_new``/``b_new`` are ONLY the rows the
    separator appended.
    """
    model = _model(atom, lo, hi, a, wsgn)
    span = max(abs(lo), abs(hi), 1.0)
    box = (np.array([lo, -1e3 * span]), np.array([hi, 1e3 * span]))
    token = solver_tuning.set_current(_tuning(mode))
    try:
        milp, varmap = build_milp_relaxation(
            model, classify_nonlinear_terms(model), DiscretizationState(), bound_override=box
        )
        specs = list(varmap.get("singular_tangent_relaxations") or [])
        A0 = sp.csr_matrix(milp._A_ub, dtype=float)
        n0 = A0.shape[0]
        res0 = milp.solve()
        assert res0.status == "optimal", f"pre-separation LP was {res0.status}"
        res1 = MccormickLPRelaxer(model)._separate_singular_tangent(milp, varmap, res0, None)
        A1 = sp.csr_matrix(milp._A_ub, dtype=float).toarray()
        b1 = np.asarray(milp._b_ub, dtype=float).ravel()
        return (
            specs,
            n0,
            A1.shape[0],
            float(res0.objective),
            float(res1.objective),
            A1[n0:],
            b1[n0:],
        )
    finally:
        solver_tuning.reset_current(token)


def _true_optimum(fnp, lo, hi, a, wsgn):
    """``min_{t in [lo,hi]} a*t + wsgn*f(t)`` — a fine grid, then a local refine.

    Only ever used as an UPPER bound on the true optimum, which is the direction
    a valid dual bound must not cross.
    """
    grid = np.linspace(lo, hi, 200001)
    vals = a * grid + wsgn * fnp(grid)
    k = int(np.argmin(vals))
    best = float(vals[k])
    span = (hi - lo) / 200000.0
    ref = minimize_scalar(
        lambda t: a * t + wsgn * float(fnp(t)),
        bounds=(max(lo, grid[k] - 2 * span), min(hi, grid[k] + 2 * span)),
        method="bounded",
    )
    return min(best, float(ref.fun))


def _graph_point(t: float, fnp, aux_col: int, n_total: int) -> np.ndarray:
    """The exact lifted point for ``(t, f(t))``: ``x=t``, ``y=f(t)``, ``aux=f(t)``."""
    fv = float(fnp(t))
    z = np.zeros(n_total, dtype=float)
    z[0] = t
    z[1] = fv
    z[aux_col] = fv
    return z


# --------------------------------------------------------------------------- #
# 1. Inert unless asked
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,atom,fnp,lo,hi,a,wsgn", BINDING_CASES)
def test_recovery_off_registers_no_spec_and_separates_nothing(label, atom, fnp, lo, hi, a, wsgn):
    """With ``singular_tangent`` off the spec list is empty, so the separator is a
    no-op: same row count, bit-identical bound. This is what makes the whole
    mechanism free on the default path."""
    specs, n0, n1, before, after, _A, _b = _separate(atom, lo, hi, a, wsgn, "off")
    assert specs == [], f"{label}: recovery is off but {len(specs)} spec(s) were registered"
    assert n1 == n0, f"{label}: {n0} -> {n1} rows with recovery off"
    assert after == before, f"{label}: bound moved {before!r} -> {after!r} with recovery off"


@pytest.mark.parametrize("label,atom,lo,hi", REGULAR_CASES)
def test_regular_box_registers_no_spec_even_with_lazy_on(label, atom, lo, hi):
    """No endpoint derivative diverges, so nothing is dropped and nothing is
    deferred — the separator never even sees these atoms."""
    specs, n0, n1, before, after, _A, _b = _separate(atom, lo, hi, 1.0, -1.0, "lazy")
    assert specs == [], f"{label}: registered {len(specs)} spec(s) on a regular box"
    assert n1 == n0 and after == before


@pytest.mark.parametrize("label,atom,fnp,lo,hi,a,wsgn", SLACK_CASES)
def test_no_violation_draws_no_row(label, atom, fnp, lo, hi, a, wsgn):
    """The LP vertex already sits on the graph, so the static envelope binds it
    exactly. Lazy placement must add NOTHING — that is the whole difference from
    the eager anchor, which pays a row at every node regardless."""
    specs, n0, n1, before, after, _A, _b = _separate(atom, lo, hi, a, wsgn, "lazy")
    assert len(specs) == 1, f"{label}: expected the facet to be deferred, got {specs}"
    assert n1 == n0, f"{label}: separated {n1 - n0} row(s) at an unviolated LP point"
    assert after == before


# --------------------------------------------------------------------------- #
# 2. Soundness — no feasible point is cut
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,atom,fnp,lo,hi,a,wsgn", BINDING_CASES)
def test_separated_rows_never_cut_a_feasible_point(label, atom, fnp, lo, hi, a, wsgn):
    """Every appended row holds at the exact lifted graph point, sampled densely
    over the box INCLUDING the singular endpoint itself and a geometric refinement
    towards it (the region the facet is about, where a linear grid has no
    resolution)."""
    specs, n0, n1, _before, _after, A_new, b_new = _separate(atom, lo, hi, a, wsgn, "lazy")
    assert len(specs) == 1
    assert n1 > n0, f"{label}: nothing was separated, so this test would prove nothing"
    aux = int(specs[0].aux_col)
    n_total = A_new.shape[1]

    ts = list(np.linspace(lo, hi, 401))
    # Geometric refinement into the singular endpoint.
    width = hi - lo
    edge = int(specs[0].edge)
    ts += [(lo + 8.0**-k * width) if edge < 0 else (hi - 8.0**-k * width) for k in range(1, 18)]
    ts = [t for t in ts if lo <= t <= hi]

    checks = 0
    worst = -np.inf
    for t in ts:
        z = _graph_point(t, fnp, aux, n_total)
        slack = b_new - A_new @ z
        checks += slack.size
        worst = max(worst, float(-np.min(slack)))
    scale = max(1.0, abs(float(np.max(np.abs(b_new)))))
    assert checks > 0, f"{label}: soundness probe made zero comparisons"
    assert worst <= 1e-7 * scale, (
        f"{label}: a separated row cuts the true graph by {worst:.6g} "
        f"(scale {scale:.6g}) over {checks} comparisons"
    )


# --------------------------------------------------------------------------- #
# 3. Differential bound
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,atom,fnp,lo,hi,a,wsgn", BINDING_CASES)
def test_bound_tightens_and_never_crosses_the_true_optimum(label, atom, fnp, lo, hi, a, wsgn):
    """Structural: the separator only ADDS rows, so the LP feasible set shrinks and
    the (minimization) bound can only rise. And it must stay a valid dual bound —
    at or below the true optimum over the same box."""
    _specs, n0, n1, before, after, _A, _b = _separate(atom, lo, hi, a, wsgn, "lazy")
    truth = _true_optimum(fnp, lo, hi, a, wsgn)
    scale = max(1.0, abs(truth))
    assert n1 > n0, f"{label}: nothing separated"
    assert after >= before - 1e-9 * scale, f"{label}: bound LOOSENED {before:.12g} -> {after:.12g}"
    assert after > before + 1e-6 * scale, (
        f"{label}: separation bought nothing ({before:.12g} -> {after:.12g}); the LP point "
        "was violated, so a row that changes no bound means the row is misplaced"
    )
    assert after <= truth + 1e-6 * scale, (
        f"{label}: separated bound {after:.12g} CROSSES the true optimum {truth:.12g} — "
        "the relaxation is unsound"
    )


@pytest.mark.parametrize("label,atom,fnp,lo,hi,a,wsgn", BINDING_CASES)
def test_lazy_placement_beats_the_eager_anchor(label, atom, fnp, lo, hi, a, wsgn):
    """The point of #1115: placing the facet where the LP binds recovers strictly
    more of the gap than #1111's fixed geometric anchor, while emitting no row at
    all on the nodes where the anchor's row would not have bound."""
    _s0, _n0, _n1, off, _off2, _A0, _b0 = _separate(atom, lo, hi, a, wsgn, "off")
    _s1, _e0, _e1, _eb, eager, _A1, _b1 = _separate(atom, lo, hi, a, wsgn, "eager")
    _s2, _l0, _l1, _lb, lazy, _A2, _b2 = _separate(atom, lo, hi, a, wsgn, "lazy")
    truth = _true_optimum(fnp, lo, hi, a, wsgn)
    scale = max(1.0, abs(truth))
    for name, bound in (("eager", eager), ("lazy", lazy)):
        assert bound <= truth + 1e-6 * scale, f"{label}: {name} bound crosses the true optimum"
        assert bound >= off - 1e-9 * scale, f"{label}: {name} bound is below the off bound"
    assert lazy > eager + 1e-6 * scale, (
        f"{label}: lazy {lazy:.12g} did not beat eager {eager:.12g} (off {off:.12g}, "
        f"true {truth:.12g})"
    )


def test_the_singular_point_itself_is_covered_not_skipped():
    """The LP vertex landing exactly ON the singularity is the case the facet
    exists for, and no finite-slope tangent is available there. Pin that the
    fallback touch point fires rather than the separator degrading to a no-op —
    this is the defect that made the first lazy implementation useless."""
    lo, hi, a = 0.0, 4.0, 2.0
    specs, n0, n1, before, after, _A, _b = _separate(dm.sqrt, lo, hi, a, -1.0, "lazy")
    assert len(specs) == 1
    s = specs[0]
    # The pre-separation LP really does sit on the singular endpoint.
    assert not np.isfinite(float(s.fp(lo))), "sqrt'(0) must diverge for this test to mean anything"
    assert n1 > n0, "the separator skipped the singular LP point instead of anchoring the tangent"
    truth = _true_optimum(np.sqrt, lo, hi, a, -1.0)
    # Recovering essentially the whole gap is the observable difference: off is
    # -0.7071, the true optimum -0.125.
    closed = (after - before) / (truth - before)
    assert closed > 0.9, f"lazy closed only {closed:.1%} of the root gap ({before} -> {after})"
