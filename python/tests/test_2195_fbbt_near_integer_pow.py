"""#2195: FBBT treated an exponent within 1e-12 of an integer as that integer.

``x ** (2 + 0.9e-12)`` was enclosed as ``x ** 2``. On ``x <= 1e7`` that drops
1450.9 from the top of the range (``1e7 ** p = 1e14 + 1450.9``), so the feasible
``x ** p >= 1e14 + 725`` was declared infeasible. The backward pass inverted
``x ** 2`` the same way and raised ``lo`` above the feasible ``x = 1e7``. The Rust
unit tests ``test_2195_*`` in ``presolve/fbbt.rs`` pin both passes; this is the
end-to-end reproducer from the issue.
"""

import discopt.modeling as dm
import pytest

P = 2.0 + 0.9e-12


def _model():
    m = dm.Model("fbbt_near_int")
    x = m.continuous("x", lb=1.0, ub=1e7)
    m.subject_to(x**P >= 1e14 + 725.0)
    m.minimize(-x)
    return m


def test_fixture_is_feasible_at_the_upper_bound():
    # Guards the premise: x = 1e7 satisfies the constraint in f64.
    assert 1e7**P >= 1e14 + 725.0


@pytest.mark.parametrize("sign_domain", ["1", "0"])
def test_near_integer_exponent_is_not_declared_infeasible(monkeypatch, sign_domain):
    monkeypatch.setenv("DISCOPT_FBBT_SIGN_DOMAIN", sign_domain)
    r = _model().solve(time_limit=30)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(-1e7, rel=1e-9)
