"""#2099: an evaluator cached before a body is rewritten in place is not reused.

``verify_point`` before ``solve`` cached a tape evaluator built from the objective
as written, ``(M*x + x) - M*x`` with ``M = 2**53``, which binary64 reads as ``0``.
The solve door's exact pass (#2090) then replaced the objective's expression *in
place* -- same ``Objective`` object -- and the evaluator fingerprint, which keyed on
that object's identity only, kept serving the stale evaluator: ``feasible``,
objective 0.0 at ``x = 1`` (truth 1), certificate withdrawn. Measured on
``a624e0f``.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._evaluator_cache import evaluator_fingerprint
from discopt.validation.feasibility import verify_point

M = 2.0**53

pytestmark = pytest.mark.smoke


def _build():
    m = dm.Model()
    x = m.continuous("x", lb=1, ub=2)
    m.subject_to(x * x <= 4)
    m.minimize((M * x + x) - M * x)  # exactly x on [1, 2]: optimum 1
    return m


def test_solve_after_verify_point_reports_the_true_objective():
    m = _build()
    verify_point(m, np.array([1.0]), with_objective=True)
    assert getattr(m, "_tape_evaluator_cache", None)  # the probe built one (§6)
    r = m.solve(time_limit=30, threads=1)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.0, abs=1e-9)
    assert r.gap_certified


def test_fingerprint_sees_an_in_place_body_replacement():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    m.subject_to(x <= 1)
    m.minimize(x)
    before = evaluator_fingerprint(m)
    m._constraints[0].body = x - 0.5
    assert evaluator_fingerprint(m) != before
    mid = evaluator_fingerprint(m)
    m._objective.expression = 2 * x
    assert evaluator_fingerprint(m) != mid
    assert evaluator_fingerprint(m) == evaluator_fingerprint(m)


def test_fingerprint_does_not_build_a_lazy_body():
    from discopt.modeling.core import builder_row_of

    m = dm.Model()
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    m.add_linear_constraints(np.ones((2, 3)), x, "<=", np.ones(2), name="b")
    m.minimize(dm.sum(x))
    lazy = m._builder_linear_constraints(lazy=True)
    assert lazy and all(builder_row_of(c) is not None for c in lazy)
    m._constraints.extend(lazy)
    evaluator_fingerprint(m)
    assert all(builder_row_of(c) is not None for c in lazy)  # still unbuilt


def test_tightening_cache_pins_the_bodies_it_keys_on():
    """The bound-tightening structure cache keys on ``id(body)``; a rewritten body
    could be freed and its id recycled onto another row, so each entry holds it."""
    from discopt._relax import nonlinear_bound_tightening as nbt
    from discopt._relax.nonlinear_bound_tightening import _cached_flat_terms

    m = _build()
    body = m._constraints[0].body
    terms = _cached_flat_terms(m, body)
    entry = m._nl_struct_cache["flatten"][id(body)]
    assert entry[0] is body and entry[1] is terms
    assert nbt  # module imported for the cache under test
