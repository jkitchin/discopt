"""#1661: ``centropy(x, y) = x*log(x/y)`` reaches the Rust IR.

``canonicalize_entropy`` (run on every solve) rewrites ``x*log(x/y)`` to
``centropy(x, y)``, which had no ``MathFunc``. ``model_to_repr`` raised, and the
solve ran with no FBBT, no root presolve and no in-tree propagation -- one
warning was the only sign. Seven MINLPLib instances hit it (``ex6_2_5``,
``ex6_2_9`` .. ``ex6_2_14``), as does the #1616 A-04 reformer model.
"""

from __future__ import annotations

import logging
import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.factorable_reform import canonicalize_entropy
from discopt._rust import model_to_repr
from discopt.modeling.core import FunctionCall
from discopt.tightening import fbbt_box

G0 = np.array([19.492, -192.590, -200.275, 0.0, -395.886]) / (8.314e-3 * 1000)
A = np.array([[1, 0, 1, 0, 1], [4, 2, 0, 2, 0], [0, 1, 1, 0, 2]])
B = A @ np.array([1.0, 2, 0, 0, 0])


def _reformer():
    # The #1616 A-04 model with the denominator lifted, so the solve certifies.
    m = dm.Model("reformer")
    n = [m.continuous(f"n{i}", lb=0, ub=10) for i in range(5)]
    tot = m.continuous("N", lb=3.0, ub=50)
    m.subject_to(tot == sum(n))
    m.minimize(sum(n[i] * (G0[i] + dm.log(n[i] / tot)) for i in range(5)))
    for k in range(3):
        m.subject_to(sum(int(A[k, i]) * n[i] for i in range(5)) == B[k])
    return m


def _uses_centropy(model) -> bool:
    return "centropy" in repr(model._objective.expression)


def test_canonicalized_model_builds_a_rust_repr():
    m = canonicalize_entropy(_reformer())
    assert _uses_centropy(m), "the rewrite this issue is about did not fire"
    rep = model_to_repr(m, getattr(m, "_builder", None))
    # Parity with the Python value at an interior point, and the x = 0 limit.
    checks = 0
    for n in ([1.0, 1.5, 0.2, 0.3, 0.4], [0.0, 2.0, 0.5, 0.5, 0.25]):
        x = np.array([*n, sum(n)])
        want = sum(
            n[i] * G0[i] + (n[i] * math.log(n[i] / x[5]) if n[i] > 0 else 0.0) for i in range(5)
        )
        assert rep.evaluate_objective(x) == pytest.approx(want, rel=1e-12, abs=1e-12)
        checks += 1
    assert checks == 2


def test_solve_runs_with_fbbt_and_certifies(caplog):
    with caplog.at_level(logging.WARNING, logger="discopt.solver"):
        r = _reformer().solve(time_limit=60)
    assert not [rec for rec in caplog.records if "Rust model repr unavailable" in rec.getMessage()]
    assert r.status == "optimal" and r.gap_certified
    assert r.bound <= r.objective + 1e-6


def test_wrong_arity_is_refused():
    m = dm.Model("bad")
    x = m.continuous("x", lb=0.1, ub=1)
    m.minimize(FunctionCall("centropy", x))
    with pytest.raises(ValueError, match="centropy"):
        model_to_repr(m, None)


def test_fbbt_tightens_through_centropy():
    """Backward FBBT through both operands: each tightened bound is the exact
    projection, rounded outward, so the feasible boundary point survives."""
    m = dm.Model("fbbt")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=1, ub=2)
    m.subject_to(FunctionCall("centropy", x, y) <= 0)
    m.minimize(-x)
    box = fbbt_box(m)
    assert float(box.ub[0]) == pytest.approx(2.0, abs=1e-6)
    assert float(box.ub[0]) >= 2.0  # the point (2, 2) is feasible

    m = dm.Model("fbbt_y")
    x = m.continuous("x", lb=0.1, ub=1)
    y = m.continuous("y", lb=1e-3, ub=5)
    m.subject_to(FunctionCall("centropy", x, y) <= -0.3)
    m.minimize(y)
    box = fbbt_box(m)
    # y >= min over x of x*exp(0.3/x) = 0.3*e, attained at (0.3, 0.3e).
    assert float(box.lb[1]) == pytest.approx(0.3 * math.e, rel=1e-9)
    assert float(box.lb[1]) <= 0.3 * math.e
