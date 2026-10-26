"""Rust LP presolve FBBT must enclose the rounding error of the whole activity sum.

``presolve::fbbt_row`` widened each residual by ``8 eps max|term|``, which covers about
one rounding of the largest term. ``x0 + z + sum_{2000} y_i == 2^40`` with
``y_i <= 2^-14`` loses every ``y_i`` in the float sum, so presolve derived
``x0 >= -0.00195`` and cut off the true optimum ``x0 = -2000 * 2^-14``. On the native
MILP route that certified ``0.998`` as optimal for a model whose optimum is
``0.8779296875``. The budget is now ``(n + 8) eps (sum|terms| + |b|)`` plus the
quotient's rounding (the bound ``lp_milp_highs.fbbt_box`` uses).
"""

from __future__ import annotations

from fractions import Fraction

import discopt
import pytest

M = 2.0**40
D = 2.0**-14
N = 2000
EXACT = Fraction(1) - N * Fraction(D)  # 0.8779296875


def _model():
    m = discopt.Model("presolve_fbbt_rounding")
    x0 = m.continuous("x0", lb=-1e6, ub=1e6)
    z = m.continuous("z", lb=0, ub=M)
    y = m.continuous("y", shape=(N,), lb=0, ub=D)
    k = m.integer("k", lb=0, ub=2)
    m.subject_to(x0 + z + sum(y[i] for i in range(N)) + 0 * k == M)
    m.subject_to(k >= 1)
    m.minimize(x0 + k)
    return m


@pytest.mark.smoke
@pytest.mark.parametrize("backend", ["rust", "highs"])
def test_native_milp_bound_never_exceeds_exact_optimum(backend, monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", backend)
    r = _model().solve(time_limit=60, threads=1)
    assert r.bound is not None
    assert Fraction(r.bound) <= EXACT, (r.status, r.objective, r.bound)
    if r.gap_certified:
        assert abs(Fraction(r.objective) - EXACT) <= Fraction(1, 10**6), (r.objective, r.bound)
