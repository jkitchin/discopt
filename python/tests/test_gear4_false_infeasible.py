"""Soundness lock: division clearing must not fabricate a false infeasibility.

``gear4`` (a *feasible* MINLPLib instance, optimum ``1.64342847``) was reported
``status="infeasible", gap_certified=True`` — a certified wrong answer, the worst
failure class — on ``main``. It is also a member of the smoke suite, so the bug
silently corrupted the smoke correctness gate.

Root cause (issue #145). gear4's single nonlinear constraint is a quotient with
two *unbounded* continuous slacks ``x4, x5`` (``ub = +inf``) used to model
``min |target - (1e6 x0 x1)/(x2 x3)|``::

    -(1e6 x0 x1)/(x2 x3) - x4 + x5 == -144279.32 ,  x4, x5 >= 0 .

Sign-definite denominator clearing multiplies the *whole* constraint through by
``D = x2 x3``, which turns the benign linear slack terms ``-x4 + x5`` into the
trilinear products ``-x4 x2 x3 + x5 x2 x3``. A product with a non-finitely-bounded
factor (``x4``/``x5`` are unbounded above) has no valid finite McCormick envelope:
the cleared relaxation *excludes* feasible points, the root LP is reported
infeasible, the whole tree is fathomed as infeasible, and the leftover
``status != "infeasible"`` exemption in ``SolveResult.__post_init__`` lets it pass
through as ``gap_certified=True``.

The fix (``factorable_reform``): reject denominator clearing for a constraint
when multiplying through introduces a nonlinear product whose interval bound is
non-finite. gear4 then keeps its quotient, which the McCormick-``lp`` path bounds
soundly (feasible incumbent, valid dual bound at the relaxation floor ~0).

These tests are intentionally *unmarked* (CI-visible): the false-infeasible fired
at the root node, so a tiny bounded solve reproduces it, and the regression must
never be invisible to CI the way ``@pytest.mark.correctness`` tests are.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import pytest
from discopt.modeling.core import BinaryOp, Expression, from_nl

_DATA = Path(__file__).parent / "data" / "minlplib"
_GEAR4_OPT = 1.64342847


def _contains_division(expr: Expression) -> bool:
    """True if *expr* still contains an unrewritten ``/`` node."""
    if isinstance(expr, BinaryOp):
        if expr.op == "/":
            return True
        return _contains_division(expr.left) or _contains_division(expr.right)
    for attr in ("operand", "left", "right"):
        child = getattr(expr, attr, None)
        if isinstance(child, Expression) and _contains_division(child):
            return True
    return False


def test_has_unbounded_nonlinear_term_unit():
    """The guard flags a nonlinear product with an unbounded factor, not a bounded one."""
    from discopt._jax.factorable_reform import _has_unbounded_nonlinear_term

    m = dm.Model("unit")
    a = m.continuous("a", lb=1.0, ub=10.0)  # bounded
    b = m.continuous("b", lb=0.0, ub=float("inf"))  # unbounded above (a slack)

    # a * a : both factors bounded -> safe.
    assert not _has_unbounded_nonlinear_term(a * a, m)
    # b alone is linear -> safe (degree 1, never enveloped as a product).
    assert not _has_unbounded_nonlinear_term(3.0 * b, m)
    # a * b : a bounded but b unbounded -> no valid finite envelope -> flagged.
    assert _has_unbounded_nonlinear_term(a * b, m)


def test_clearing_skipped_for_unbounded_slack_product():
    """``factorable_reformulate`` must keep gear4's quotient, not clear it into an
    unbounded trilinear product."""
    from discopt._jax.factorable_reform import (
        _clear_divisions,
        _has_unbounded_nonlinear_term,
        factorable_reformulate,
    )

    m = from_nl(str(_DATA / "gear4.nl"))

    # Sanity: clearing this constraint *would* introduce an unbounded product...
    con = m._constraints[0]
    cleared_body, _sense = _clear_divisions(con.body, con.sense, m)
    assert cleared_body is not con.body, "expected the denominator to be clearable"
    assert _has_unbounded_nonlinear_term(cleared_body, m), (
        "test premise: cleared gear4 body must contain an unbounded nonlinear product"
    )

    # ...so the guard must reject it and keep the quotient intact.
    m2 = factorable_reformulate(m)
    assert any(_contains_division(c.body) for c in m2._constraints), (
        "guard failed: gear4 quotient was cleared into an unbounded-product relaxation"
    )


@pytest.mark.slow
def test_gear4_not_false_infeasible():
    """gear4 must never be reported certified-infeasible; bounds stay sound."""
    r = from_nl(str(_DATA / "gear4.nl")).solve(time_limit=10, max_nodes=200)

    # Headline: the certified-wrong-answer must be gone.
    assert not (r.status == "infeasible" and r.gap_certified), (
        f"gear4 falsely certified infeasible: status={r.status} cert={r.gap_certified}"
    )
    # gear4 is feasible, so it must not be reported infeasible at all.
    assert r.status != "infeasible", f"gear4 is feasible but status={r.status}"

    # Soundness: any reported dual bound never exceeds the true optimum.
    if r.bound is not None:
        assert r.bound <= _GEAR4_OPT + 1e-3, f"unsound dual bound {r.bound} > optimum {_GEAR4_OPT}"
    # Any feasible incumbent is >= the true optimum (it cannot beat it).
    if r.objective is not None:
        assert r.objective >= _GEAR4_OPT - 1e-3, (
            f"incumbent {r.objective} below true optimum {_GEAR4_OPT}"
        )


@pytest.mark.slow
def test_synthetic_unbounded_slack_quotient_not_false_infeasible():
    """Self-contained gear4-class model (issue #145): a large-coefficient quotient
    equality with unbounded continuous slacks must not be certified infeasible."""
    m = dm.Model("synthetic_unbounded_slack_quotient")
    x0 = m.integer("x0", lb=12, ub=60)
    x1 = m.integer("x1", lb=12, ub=60)
    x2 = m.integer("x2", lb=12, ub=60)
    x3 = m.integer("x3", lb=12, ub=60)
    x4 = m.continuous("x4", lb=0.0, ub=float("inf"))
    x5 = m.continuous("x5", lb=0.0, ub=float("inf"))
    m.minimize(x4 + x5)
    # x4 - x5 = target - ratio  =>  the slacks absorb the residual, so every
    # integer assignment is feasible: the model is unconditionally feasible.
    m.subject_to(-(1e6 * x0 * x1) / (x2 * x3) - x4 + x5 == -144279.32477276)

    r = m.solve(time_limit=10, max_nodes=200)

    assert not (r.status == "infeasible" and r.gap_certified), (
        f"synthetic quotient falsely certified infeasible: status={r.status} cert={r.gap_certified}"
    )
    assert r.status != "infeasible", f"model is feasible but status={r.status}"
    # A feasible point exists (e.g. any gears + slacks); the solver must find one.
    assert r.objective is not None, "feasible model returned no incumbent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ── Spatial branching on nonlinear-term integers (issue #194 / PR #202) ──────
# The fix above only removed the *false-infeasible*. The positive win — gear4
# now *certifies* because the spatial tree branches on its integer-valued
# integers (i1..i4) that carry an open McCormick gap — was unguarded: a
# regression in nonlinear-term-integer detection or spatial-integer branching
# would drop gear4 back to feasible-uncertified and the soundness test above
# would still pass. These guard the win.


def test_nonlinear_term_integers_are_detected_for_spatial_branching():
    """A bilinear product of two *integer* variables must expose those integer
    columns as nonlinear columns, so the solver registers them for spatial
    branching (the `_nl_int_cols` step in solve_model)."""

    from discopt._jax.mccormick_lp import MccormickLPRelaxer
    from discopt.solver import _extract_variable_info

    m = dm.Model("int_bilinear")
    i = m.integer("i", lb=1, ub=10)
    j = m.integer("j", lb=1, ub=10)
    m.minimize(i * j)
    m.subject_to(i + j >= 5)

    _, _, _, int_offsets, int_sizes = _extract_variable_info(m)
    int_cols = {c for off, sz in zip(int_offsets, int_sizes) for c in range(off, off + int(sz))}
    relaxer = MccormickLPRelaxer(m)
    assert relaxer.has_relaxable_nonlinearity
    nl_int_cols = int_cols & set(relaxer.nonlinear_columns)
    assert nl_int_cols, (
        "integer variables inside a bilinear product were not flagged as nonlinear "
        "columns; spatial-integer branching (#194) would not engage"
    )


@pytest.mark.slow
@pytest.mark.requires_pounce
def test_gear4_certifies_via_spatial_integer_branching():
    """gear4's integers (i1..i4) are integer-valued at the root but carry an open
    McCormick gap on i1*i2 / i3*i4. Spatial branching on those integer columns
    (#194 / PR #202) closes the gap and certifies the global optimum 1.64342847.
    Without it the node dead-ends and gear4 stays feasible-but-uncertified."""
    r = from_nl(str(_DATA / "gear4.nl")).solve(time_limit=180, gap_tolerance=1e-4)
    assert r.status == "optimal", f"gear4 did not certify (status={r.status})"
    assert r.gap_certified, "gear4 reached optimum but gap was not certified"
    assert r.objective == pytest.approx(_GEAR4_OPT, abs=1e-2)
    # The dual bound must have actually risen to meet the incumbent.
    assert r.bound is not None and r.bound <= _GEAR4_OPT + 1e-3
