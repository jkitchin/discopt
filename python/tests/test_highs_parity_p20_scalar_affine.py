"""highs-parity P2.0 (ii): scalar linear rows skip ``affine_of``'s per-node sparse algebra.

P2.1 found the post-solve certificate re-check (#1561) spending up to 5 s building its
:class:`~discopt._affine_evaluator.AffineEvaluator` on `.nl`-shaped LPs: one scipy matrix
per expression node (9,461 on ``scrs8``). ``scalar_linear_of`` reads a scalar row's
coefficients in one walk. Measured over the 65 L/M/INF panel instances: every ``J``,
``g0``, ``c``, ``c0`` and row map bit-identical to the ``affine_of`` build, total build
14.2 s -> 0.57 s. These pin the fast path's agreement with ``affine_of``, its declines,
and that an `.nl`-shaped model's rows no longer go through ``affine_of`` at all.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.affine_array import NotAffineError, affine_of, scalar_linear_of


def _model():
    m = dm.Model("p20ii")
    xs = [m.continuous(f"x{i}", lb=-5, ub=5) for i in range(5)]
    p = m.parameter("p", value=2.5)
    return m, xs, p


def _dense(expr, m):
    n = sum(int(v.size) for v in m._variables)
    lin = scalar_linear_of(expr, m._flat_var_offset, n)
    aff = affine_of(expr, m._flat_var_offset, n)
    ref = np.zeros(n) if aff.M is None else aff.M.toarray().ravel()
    return lin, ref, float(aff.b[0]), n


def test_agrees_with_affine_of_on_scalar_trees():
    m, x, p = _model()
    exprs = [
        x[0] - (-(3.0 * x[1]) + x[2] / 4.0) - 7.0,
        2.0 * (x[0] + 0.5 * (x[1] - x[0])) + x[3] ** 1,
        -(x[4]) + p * x[0] - x[0] * 1.5 + 0.0 * x[2],
        dm.sum([x[0], 2.0 * x[1], x[2] - x[3], 4.0]),
        (((x[0] + x[1]) + x[2]) + x[3]) + x[0],  # repeated column: summed
    ]
    checked = 0
    for e in exprs:
        lin, ref, b_ref, n = _dense(e, m)
        assert lin is not None, e
        cols, vals, b = lin
        got = np.zeros(n)
        np.add.at(got, cols, vals)
        assert np.allclose(got, ref, rtol=1e-15, atol=0) and b == pytest.approx(b_ref, abs=1e-15)
        checked += 1
    assert checked == len(exprs)


def test_declines_what_it_cannot_represent():
    m, x, _p = _model()
    y = m.continuous("y", shape=(3,), lb=0, ub=1)
    n = sum(int(v.size) for v in m._variables)
    off = m._flat_var_offset
    for e in [x[0] * x[1], dm.exp(x[0]), x[0] / x[1], x[0] / 0.0, x[0] ** 2, y[1] + x[0],
              dm.sum(y)]:  # fmt: skip
        assert scalar_linear_of(e, off, n) is None, e
    # Declining is not refusing: affine_of still decides, as before.
    with pytest.raises(NotAffineError):
        affine_of(x[0] * x[1], off, n)
    assert affine_of(y[1] + x[0], off, n).size == 1


def test_scalar_lp_rows_never_reach_affine_of(monkeypatch):
    import discopt._relax.affine_array as AA
    from discopt._affine_evaluator import AffineEvaluator

    m, x, _p = _model()
    m.subject_to(x[0] + 2.0 * x[1] - x[2] <= 4.0)
    m.subject_to(-(x[3]) + x[4] >= -1.0)
    m.subject_to(x[0] - x[4] == 0.5)
    m.minimize(3.0 * x[0] + x[1] - 2.0)
    calls = []
    real = AA.affine_of
    monkeypatch.setattr(AA, "affine_of", lambda *a, **k: calls.append(1) or real(*a, **k))
    ev = AffineEvaluator.try_build(m)
    assert ev is not None and calls == []
    pt = np.array([1.0, 2.0, 0.5, -1.0, 0.5])
    g = ev.evaluate_constraints(pt)
    assert g.shape == (3,)
    # Each body is ``lhs - rhs`` in its sense; check the values directly.
    assert np.allclose(g, [1.0 + 4.0 - 0.5 - 4.0, -1.0 - (-(-1.0) + 0.5), 1.0 - 0.5 - 0.5])
    assert ev.evaluate_objective(pt) == pytest.approx(3.0 + 2.0 - 2.0)
