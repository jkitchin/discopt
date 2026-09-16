"""``log10(sum(10**y))`` must get a dual bound (#1256).

Three defects, one model. ``minimize log10(sum_i 10**y_i)`` subject to
``sum_i y_i >= -12`` over ``y in [-8,-1]^3`` has its optimum at ``y_i = -4``,
``log10(3e-4) = -3.5229``.

1. With SHAPED variables the incremental McCormick structure died with
   ``cannot reshape array of size 1 into shape (3,)``: the root box was built
   with one entry per ``Variable`` OBJECT while the lifted layout has one column
   per flat ELEMENT.
2. ``10**y`` — a constant base with a variable exponent — canonicalized to an
   OPAQUE node, and an opaque node has no envelope, so the relaxation had no
   valid objective bound and the solve returned ``bound=None`` after one node.
   ``b**e`` for ``b > 0`` is ``exp(ln(b)*e)`` identically, and ``exp`` is in the
   univariate envelope table.
3. ``sum(<array expression>)`` was opaque too unless its operand was a bare
   variable, so the shaped form could not be relaxed even once ``10**y`` could.

The same model written with scalar variables and an epigraph got a tight bound
throughout, which is what made the gap a reporting puzzle rather than a visible
failure — so the fourth test covers the visibility half: a solve that ends with
no dual bound at all now says so above DEBUG.
"""

import logging

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.canonical_expr import canonicalize
from discopt._relax.model_utils import flat_variable_bounds

TRUE_OPTIMUM = float(np.log10(3e-4))


def _shaped():
    m = dm.Model("log10_sum_shaped")
    y = m.continuous("y", lb=-8.0, ub=-1.0, shape=(3,))
    m.subject_to(y[0] + y[1] + y[2] >= -12.0)
    m.minimize(dm.log10(dm.sum(10.0**y)))
    return m


def _scalar():
    m = dm.Model("log10_sum_scalar")
    ys = [m.continuous(f"y{i}", lb=-8.0, ub=-1.0) for i in range(3)]
    m.subject_to(ys[0] + ys[1] + ys[2] >= -12.0)
    m.minimize(dm.log10(10.0 ** ys[0] + 10.0 ** ys[1] + 10.0 ** ys[2]))
    return m


@pytest.mark.unit
@pytest.mark.parametrize("build", [_shaped, _scalar], ids=["shaped", "scalar"])
def test_no_opaque_node_survives_canonicalization(build):
    """Every node of the objective must be something the envelope library knows."""
    dag = canonicalize(build())
    opaque = [n for n in dag.nodes if n.kind == "opaque"]
    assert not opaque, f"opaque nodes left in the DAG: {[str(n.payload) for n in opaque]}"
    assert any(n.kind == "call" and n.payload == "exp" for n in dag.nodes), (
        "10**y must lower to exp(ln(10)*y)"
    )


@pytest.mark.unit
def test_incremental_structure_boxes_are_flat():
    """The root box is one entry per COLUMN, not per ``Variable`` object."""
    from discopt._relax.incremental_mccormick import IncrementalMcCormickLP
    from discopt._relax.term_classifier import classify_nonlinear_terms

    m = _shaped()
    n_cols = sum(int(v.size) for v in m._variables)
    assert n_cols == 3 and len(m._variables) == 1, "the shaped model is the point of this test"

    inc = IncrementalMcCormickLP(m, classify_nonlinear_terms(m), deadline=None)
    lb, ub = inc._root_box()
    assert lb.size == ub.size == n_cols, f"root box has {lb.size} entries for {n_cols} columns"
    flat_lb, flat_ub = flat_variable_bounds(m)
    assert np.allclose(lb, flat_lb) and np.allclose(ub, flat_ub)
    # Whatever it decides, it must not decide it because of a reshape.
    assert "reshape" not in (inc.decline_reason or ""), (
        f"structure declined on a shape bug: {inc.decline_reason}"
    )
    # A caller-supplied box over the flat columns must be accepted, not rejected
    # for having the "wrong" length.
    boxed = IncrementalMcCormickLP(
        m, classify_nonlinear_terms(m), deadline=None, box=(flat_lb, flat_ub)
    )
    assert boxed._box is not None, "a correctly-sized flat box must be accepted"


@pytest.mark.smoke
@pytest.mark.parametrize("build", [_shaped, _scalar], ids=["shaped", "scalar"])
def test_solve_gets_a_finite_bound(build):
    r = build().solve(time_limit=60)
    assert r.bound is not None, "log10 of a positive sum must be bounded"
    assert r.bound <= TRUE_OPTIMUM + 1e-6, (
        f"dual bound {r.bound!r} is above the true optimum {TRUE_OPTIMUM!r}"
    )
    assert r.objective is not None and abs(r.objective - TRUE_OPTIMUM) < 1e-4, (
        f"objective {r.objective!r} is not the true optimum {TRUE_OPTIMUM!r}"
    )


@pytest.mark.smoke
def test_a_solve_with_no_dual_bound_says_so(caplog):
    """``bound=None`` used to be the only trace, and only at DEBUG."""
    m = dm.Model("unbounded_relaxation")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    # ``sign`` is deliberately kept opaque by the canonicalizer: no sound
    # continuous envelope exists, so this model genuinely cannot be bounded.
    m.minimize(dm.sign(x) + x)
    with caplog.at_level(logging.WARNING, logger="discopt.solver"):
        r = m.solve(time_limit=10)
    if r.bound is None:
        assert any("no valid dual bound" in rec.message.lower() for rec in caplog.records), (
            f"a boundless solve must say so at WARNING; got {[r.message for r in caplog.records]}"
        )
