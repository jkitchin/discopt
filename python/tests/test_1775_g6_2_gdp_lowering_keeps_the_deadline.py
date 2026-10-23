"""#1775 G6.2: the GDP-lowered working model keeps the solve's clocks.

``solve_model`` stamps the absolute ``_solve_deadline`` (#654/#875) and the
``_convexity_time_budget`` on the model it is given; the pre-B&B passes that read
them (nonlinear bound tightening, the convexity classifier) are bounded by them.
The GDP lowering returns a NEW model and copied neither, so on every GDP solve
those passes ran with no deadline. Measured on GDPlib ``biofuel`` (hull,
``time_limit=10``): 398 s, with every sample from 85 s to 355 s inside one
``tighten_nonlinear_bounds`` pass; on ``eight_process`` (hull, 30 s) 159 of 161
tightening calls saw ``deadline=None``. Every other reform (binary-multilinear,
integer-multilinear, integer-bilinear) already carried both.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest
from discopt._relax import nonlinear_bound_tightening as NBT

pytestmark = pytest.mark.smoke


def _gdp():
    m = dm.Model("clocks")
    x = m.continuous("x", lb=0.1, ub=4)
    y = m.continuous("y", lb=0.1, ub=4)
    m.either_or([[x + y <= 2], [x + y >= 3, x * y >= 1.5]], name="d")
    m.subject_to(x**2 + y >= 1)
    m.minimize((x - 1.7) ** 2 + (y - 0.6) ** 2 - x * y)
    return m


@pytest.mark.parametrize("method", ["big-m", "hull"])
def test_every_tightening_pass_on_the_lowered_model_has_the_deadline(method, monkeypatch):
    seen: list[tuple[object, object, object]] = []
    real = NBT.tighten_nonlinear_bounds

    def spy(model, lb, ub, *a, deadline=None, **k):
        seen.append(
            (
                deadline,
                getattr(model, "_solve_deadline", None),
                getattr(model, "_convexity_time_budget", None),
            )
        )
        return real(model, lb, ub, *a, deadline=deadline, **k)

    monkeypatch.setattr(NBT, "tighten_nonlinear_bounds", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _gdp().solve(gdp_method=method, time_limit=30)
    assert r.status == "optimal"
    assert seen, "the tightening pass never ran (rule 6)"
    assert all(d is not None for d, _, _ in seen), seen
    assert all(s is not None and b is not None for _, s, b in seen), seen
