"""AMP piecewise 1-d secant rows must not cut a truly feasible point.

``_emit_piecewise_1d`` writes the secant ``w <= ssl*arg + sint`` with the rounded
intercept ``sint = flo - ssl*alo``, which cancels on a wide piece like
``[-5e8, 0.02]``. Checked in exact ``Fraction`` arithmetic, the rows missed the
true ``(x, f(x), lam=1)`` point by up to ~1e3, and the partitioned relaxation of
``max x s.t. (3x)**2 >= 0.0018`` on ``[-1e9, 0.02]`` returned ``x* = 0.0199999716``,
cutting off the true optimum ``0.02``.
"""

from __future__ import annotations

import random
from fractions import Fraction

import discopt._relax.uniform_relax as U
import discopt.modeling as dm
import pytest
from discopt._relax.discretization import initialize_partitions


class _Ctx:
    def __init__(self):
        self.rows = []

    def add_row(self, coeffs, rhs):
        self.rows.append((dict(coeffs), float(rhs)))


def test_secant_rows_hold_exactly_at_true_points():
    W, X, LAM = 0, 1, 2
    rnd = random.Random(7)
    checks = 0
    for p in (2, 4):
        cv = U._curv_const("convex")
        for _ in range(400):
            span = 10 ** rnd.uniform(2, 9 if p == 2 else 4)
            h = 10 ** rnd.uniform(-4, 0)
            coeff = rnd.choice([1.0, 3.0, 0.1, 7.0, -2.0])
            const = rnd.choice([0.0, 0.7, -0.3, 1.0, 0.1])
            a0, a1 = coeff * -span + const, coeff * h + const
            alo, ahi = min(a0, a1), max(a0, a1)
            ctx = _Ctx()
            U._emit_piecewise_1d(
                ctx,
                W,
                X,
                LAM,
                coeff,
                const,
                alo,
                ahi,
                lambda t, p=p: float(t**p),
                lambda t, p=p: float(p * t ** (p - 1)),
                cv(alo, ahi),
            )
            for xv in (Fraction(h), Fraction(h) / 2, Fraction(0)):
                fv = (Fraction(coeff) * xv + Fraction(const)) ** p
                for c, r in ctx.rows:
                    act = (
                        Fraction(c.get(W, 0.0)) * fv
                        + Fraction(c.get(X, 0.0)) * xv
                        + Fraction(c.get(LAM, 0.0))
                    )
                    assert act <= Fraction(r), (p, span, h, coeff, const, float(act - Fraction(r)))
                    checks += 1
    assert checks > 5000, checks


@pytest.mark.parametrize("c", [1.0, 3.0])
@pytest.mark.parametrize("npart", [2, 3])
def test_partitioned_relaxation_keeps_true_optimum(c, npart):
    span, h = 1e9, 0.02
    m = dm.Model()
    x = m.continuous("x", lb=-span, ub=h)
    m.maximize(x)
    m.subject_to((c * x) ** 2 >= 0.5 * (c * h) ** 2)
    ds = initialize_partitions([0], [-span], [h], n_init=npart)
    rel = U.build_uniform_relaxation(m, disc_state=ds)
    res = rel.model.solve()
    assert res.status == "optimal"
    # On main: 0.0199999716 (c=3, npart=2), short by 2.8e-8.
    assert float(res.x[0]) >= h - 1e-9, float(res.x[0])
