"""#2054 (big-M part): the GDP big-M enclosure, refined by subdividing a repeated variable.

Interval arithmetic treats each occurrence of ``t`` as independent, so a row with
``t`` in a numerator and a denominator (the PFR by-product row) is enclosed hundreds
of times too wide, and the big-M built on it weakens every relaxation. Behind
``DISCOPT_GDP_BIGM_SUBDIVISION``. Differential (new enclosure inside the old one)
plus feasible-point sampling (the new enclosure still holds every sampled value).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import gdp_reformulate as G

FLAG = "DISCOPT_GDP_BIGM_SUBDIVISION"
KAP = 0.25

ROWS = {
    "pfr_byproduct": lambda t, o: 60 * KAP * t / (o.exp((1 - KAP) * t) - 1),
    "cstr2": lambda t, o: 50 * (1 + KAP * t / 2) ** 2 / ((t / 2) * (2 + (1 + KAP) * t / 2)),
    "conversion_ratio": lambda t, o: 55 * (1 - o.exp(-(1 - KAP) * t)) / (1 + KAP * t),
    "nonmonotone": lambda t, o: (t - 2) ** 3 - 3 * t + t * o.exp(-t),
}


def _plain(body, m):
    return G._bound_expression(body, m, G._BoundMemo(array_reductions=True))


@pytest.mark.parametrize("name", sorted(ROWS))
@pytest.mark.parametrize("box", [(0.25, 4.0), (0.5, 1.0), (1.0, 7.0)])
def test_enclosure_holds_every_sample_and_is_never_wider(name, box, monkeypatch):
    f = ROWS[name]
    m = dm.Model("row")
    t = m.continuous("t", lb=box[0], ub=box[1])
    p = m.continuous("P", lb=0, ub=100)
    body = f(t, dm) - p
    monkeypatch.setenv(FLAG, "1")
    lo, hi = G._gdp_body_bounds(body, m)
    plo, phi = _plain(body, m)
    assert plo <= lo <= hi <= phi
    g = np.linspace(*box, 20001)
    y = f(g, np)
    # P in [0, 100] enters once and linearly: the body's range is f's range minus it.
    assert lo <= y.min() - 100 + 1e-12
    assert hi >= y.max() - 1e-12


def test_pfr_byproduct_row_is_nearly_exact(monkeypatch):
    f = ROWS["pfr_byproduct"]
    m = dm.Model("row")
    t = m.continuous("t", lb=0.25, ub=4.0)
    p = m.continuous("P", lb=0, ub=100)
    body = f(t, dm) - p
    true_hi = float(f(np.linspace(0.25, 4.0, 20001), np).max())
    monkeypatch.setenv(FLAG, "0")
    assert G._gdp_body_bounds(body, m)[1] > 10 * true_hi  # the #2054 overestimate
    monkeypatch.setenv(FLAG, "1")
    hi = G._gdp_body_bounds(body, m)[1]
    assert true_hi <= hi <= true_hi * 1.01


def test_big_m_shrinks_with_the_flag(monkeypatch):
    f = ROWS["pfr_byproduct"]
    m = dm.Model("row")
    t = m.continuous("t", lb=0.25, ub=4.0)
    p = m.continuous("P", lb=0, ub=100)
    con = f(t, dm) - p <= 0
    monkeypatch.setenv(FLAG, "0")
    m_off = G._compute_big_m(con, m)
    monkeypatch.setenv(FLAG, "1")
    m_on = G._compute_big_m(con, m)
    assert m_on < m_off / 10
    assert m_on >= float(f(np.linspace(0.25, 4.0, 20001), np).max())


def test_two_repeated_variables(monkeypatch):
    m = dm.Model("two")
    x = m.continuous("x", lb=0.5, ub=3.0)
    y = m.continuous("y", lb=0.5, ub=3.0)
    body = x * y / (x + y)  # each variable twice
    monkeypatch.setenv(FLAG, "1")
    lo, hi = G._gdp_body_bounds(body, m)
    plo, phi = _plain(body, m)
    gx, gy = np.meshgrid(np.linspace(0.5, 3, 401), np.linspace(0.5, 3, 401))
    vals = gx * gy / (gx + gy)
    assert plo <= lo <= vals.min() and vals.max() <= hi <= phi
    assert hi < phi  # it did refine


def test_indexed_element_repeated(monkeypatch):
    m = dm.Model("arr")
    x = m.continuous("x", shape=(3,), lb=0.25, ub=4.0)
    body = x[1] / (dm.exp(x[1]) - 0.5)
    monkeypatch.setenv(FLAG, "1")
    lo, hi = G._gdp_body_bounds(body, m)
    g = np.linspace(0.25, 4.0, 20001)
    vals = g / (np.exp(g) - 0.5)
    assert lo <= vals.min() and vals.max() <= hi
    assert hi < _plain(body, m)[1]


@pytest.mark.parametrize(
    "make",
    [
        # no repeated variable: the plain enclosure is already free of dependency
        lambda m, t, u, v: dm.exp(t) + u,
        # three repeated variables: more than the split budget
        lambda m, t, u, v: t * u * v / (t + u + v),
    ],
    ids=["no_repeat", "three_repeated"],
)
def test_abstains(make):
    m = dm.Model("abst")
    t = m.continuous("t", lb=0.5, ub=2)
    u = m.continuous("u", lb=0.5, ub=2)
    v = m.continuous("v", lb=0.5, ub=2)
    assert G._subdivided_body_bounds(make(m, t, u, v), m) is None


def test_abstains_on_unbounded_repeated_variable():
    m = dm.Model("free")
    t = m.continuous("t", lb=0.25)
    assert G._subdivided_body_bounds(t / (1 + t), m) is None
