"""#1714: the separable-quadratic rule reads both sides of an affine ``==`` row.

``SeparableQuadraticUpperBoundRule`` accepted ``==`` rows but inferred only from
``body <= 0``, so ``z == 1`` on ``[-10, 10]`` tightened to ``[-10, 1]``. The solve
path starts the NLP at the midpoint of that box, which put the #1711 chain NLP's
fixed end columns at -5 and cost POUNCE ~2x the iterations.

Bound-changing (CLAUDE.md 5), so the mirrored side is behind
``DISCOPT_EQ_MIRROR_BOUNDS`` (default OFF). These tests pin:

* the OFF arm is byte-identical to the legacy one-sided box;
* the ON arm reaches the exact box on fixed / affine-equality columns;
* the differential bound test: ON box is a subset of the OFF box, and every
  sampled feasible point stays inside it;
* rows with a square term are not mirrored (their negation is reverse-convex).
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest
from discopt._relax.nonlinear_bound_tightening import (
    SeparableQuadraticUpperBoundRule,
    tighten_nonlinear_bounds,
)

FLAG = "DISCOPT_EQ_MIRROR_BOUNDS"


def _tighten(model, *, rules=None):
    ev = S._make_evaluator(model)
    lb, ub = ev.variable_bounds
    kwargs = {} if rules is None else {"rules": rules}
    return tighten_nonlinear_bounds(
        model, np.array(lb, dtype=float), np.array(ub, dtype=float), **kwargs
    )


def _box(model, *, rules=None):
    out_lb, out_ub, stats = _tighten(model, rules=rules)
    assert not stats.infeasible, stats.infeasibility_reason
    return out_lb, out_ub


def _fixed():
    m = dm.Model("fixed")
    z = m.continuous("z", lb=-10, ub=10)
    m.subject_to(z == 1)
    m.minimize(z)
    return m


def _scaled():
    m = dm.Model("scaled")
    v = m.continuous("v", lb=-10, ub=10)
    m.subject_to(2 * v + 1 == 3)
    m.minimize(v)
    return m


def _two_var():
    m = dm.Model("two_var")
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=0, ub=1)
    m.subject_to(x + y == 1)
    m.minimize(x)
    return m


@pytest.mark.parametrize(
    "build, off_box, on_box",
    [
        (_fixed, ([-10.0], [1.0]), ([1.0], [1.0])),
        (_scaled, ([-10.0], [1.0]), ([1.0], [1.0])),
        (_two_var, ([-10.0, 0.0], [1.0, 1.0]), ([0.0, 0.0], [1.0, 1.0])),
    ],
)
def test_affine_equality_box(monkeypatch, build, off_box, on_box):
    monkeypatch.delenv(FLAG, raising=False)
    lb, ub = _box(build())
    np.testing.assert_allclose(lb, off_box[0], atol=1e-12)
    np.testing.assert_allclose(ub, off_box[1], atol=1e-12)

    monkeypatch.setenv(FLAG, "1")
    lb, ub = _box(build())
    # The outward float_err widening may leave an O(eps) margin, never an inward one.
    np.testing.assert_allclose(lb, on_box[0], atol=1e-12)
    np.testing.assert_allclose(ub, on_box[1], atol=1e-12)
    assert np.all(lb <= np.asarray(on_box[0]))
    assert np.all(ub >= np.asarray(on_box[1]))


def test_mirror_side_proves_infeasibility(monkeypatch):
    m = dm.Model("infeasible")
    z = m.continuous("z", lb=-10, ub=0.5)
    m.subject_to(z == 1)
    m.minimize(z)

    rules = [SeparableQuadraticUpperBoundRule()]
    monkeypatch.delenv(FLAG, raising=False)
    assert not _tighten(m, rules=rules)[2].infeasible  # legacy: z <= 1 is no news

    monkeypatch.setenv(FLAG, "1")
    stats = _tighten(m, rules=rules)[2]
    assert stats.infeasible
    assert "separable_quadratic_upper_bound" in stats.infeasibility_reason


def test_square_row_is_not_mirrored(monkeypatch):
    def build():
        m = dm.Model("square")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)
        m.subject_to(x**2 + y == 1)
        m.minimize(y)
        return m

    rules = [SeparableQuadraticUpperBoundRule()]
    monkeypatch.delenv(FLAG, raising=False)
    off = _box(build(), rules=rules)
    monkeypatch.setenv(FLAG, "1")
    on = _box(build(), rules=rules)
    np.testing.assert_array_equal(off[0], on[0])
    np.testing.assert_array_equal(off[1], on[1])


def test_chain_end_columns_fixed(monkeypatch):
    """The #1711 chain: the four ``ends`` equalities pin their columns exactly."""
    n = 20
    L = 1.5 / n

    def build():
        m = dm.Model("chain")
        x = m.continuous("x", shape=(n + 1,), lb=-10, ub=10)
        y = m.continuous("y", shape=(n + 1,), lb=-10, ub=10)
        m.subject_to([x[0] == 0, y[0] == 0, x[n] == 1, y[n] == 0], name="ends")
        m.subject_to((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2 <= L**2, name="link")
        m.minimize(0.5 * dm.sum(y[1:] + y[:-1]))
        return m

    monkeypatch.setenv(FLAG, "1")
    lb, ub = _box(build())
    # flat layout: x[0..n], y[0..n]
    ends = {0: 0.0, n: 1.0, n + 1: 0.0, 2 * n + 1: 0.0}
    for idx, val in ends.items():
        assert lb[idx] <= val <= ub[idx]
        assert ub[idx] - lb[idx] <= 1e-12, (idx, lb[idx], ub[idx])
    x0 = 0.5 * (lb + ub)
    assert np.all(np.abs(x0[list(ends)] - np.array(list(ends.values()))) <= 1e-12)


def test_differential_random_affine_rows(monkeypatch):
    """ON box subset of OFF box, and no sampled feasible point is cut off."""
    rng = np.random.default_rng(1714)
    checked_points = 0
    checked_models = 0
    strictly_tighter = 0
    for _ in range(40):
        nv = int(rng.integers(1, 5))
        nrows = int(rng.integers(1, 4))
        lo = rng.uniform(-10, 0, nv)
        hi = lo + rng.uniform(0.5, 20, nv)
        coeffs = rng.normal(size=(nrows, nv))
        coeffs[rng.random(coeffs.shape) < 0.3] = 0.0
        # Anchor point inside the box; every row passes through it, so the
        # model is feasible and the anchor is a known feasible point.
        anchor = rng.uniform(lo, hi)
        rhs = coeffs @ anchor

        def build():
            m = dm.Model("rand")
            xs = [m.continuous(f"x{i}", lb=float(lo[i]), ub=float(hi[i])) for i in range(nv)]
            for r in range(nrows):
                if not np.any(coeffs[r]):
                    continue
                body = sum(float(coeffs[r, i]) * xs[i] for i in range(nv) if coeffs[r, i])
                m.subject_to(body == float(rhs[r]))
            m.minimize(xs[0])
            return m

        monkeypatch.delenv(FLAG, raising=False)
        off_lb, off_ub = _box(build())
        monkeypatch.setenv(FLAG, "1")
        on_lb, on_ub = _box(build())

        # Subset up to the rules' outward round-off widening: ON may stop one
        # float_err short of a bound another rule derived exactly on the OFF arm,
        # which is looser by an ulp, never tighter, and so still sound.
        slack = 1e-12 * (1.0 + np.abs(np.concatenate([off_lb, off_ub])).max())
        assert np.all(on_lb >= off_lb - slack), (on_lb - off_lb, coeffs, lo, hi)
        assert np.all(on_ub <= off_ub + slack), (off_ub - on_ub, coeffs, lo, hi)
        strictly_tighter += int(np.any(on_lb > off_lb + 1e-6) or np.any(on_ub < off_ub - 1e-6))
        # Feasible samples: the anchor plus points along the null space of the rows.
        null = np.linalg.svd(coeffs)[2][np.linalg.matrix_rank(coeffs) :]
        samples = [anchor]
        for _ in range(50):
            p = anchor + (null.T @ rng.normal(size=null.shape[0]) if null.size else 0.0)
            if np.all(p >= lo) and np.all(p <= hi):
                samples.append(p)
        tol = 1e-9 * (1.0 + np.abs(np.concatenate([on_lb, on_ub])).max())
        for p in samples:
            assert np.all(p >= on_lb - tol) and np.all(p <= on_ub + tol), (p, on_lb, on_ub)
            checked_points += 1
        checked_models += 1
    assert checked_models == 40
    assert checked_points >= 40
    # The panel must exercise the mirror, not just agree with OFF everywhere.
    assert strictly_tighter >= 10, strictly_tighter
