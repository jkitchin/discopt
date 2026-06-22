"""Integer-bilinear MINLP support: implied-integer detection + exact reformulation.

When a bilinear term ``x_i*x_j`` has an integer (declared or *implied*) factor,
discopt's single McCormick envelope over the wide integer box is loose — its
integer optimum can sit strictly below the true optimum (the ``ex126x`` trim-loss
family: 19.1 vs 19.6). The reformulation binary-expands the integer factor and
big-M-linearizes the resulting ``binary*var`` products, producing an *equivalent
pure MILP* whose relaxation is exact, which discopt's MILP branch-and-bound then
solves to proven optimality.

These tests pin the two correctness-critical properties:

* **implied-integer soundness** — a variable is marked integer only when an
  integer-defining equality *proves* it integer; never on a structure that does
  not force integrality (marking a free variable integer would cut off the
  optimum — the cardinal violation), and
* **value preservation** — the reformulated model has the same optimum as the
  original on a basket of declared-integer bilinear instances.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402
from discopt._jax.implied_integer import detect_implied_integers  # noqa: E402
from discopt._jax.integer_product_reform import (  # noqa: E402
    has_reformulation_work,
    reformulate_integer_bilinear,
)
from discopt._jax.term_classifier import classify_nonlinear_terms  # noqa: E402

pytestmark = pytest.mark.unit

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib")


# --------------------------------------------------------------------------- #
# implied-integer detection: soundness is everything
# --------------------------------------------------------------------------- #


def test_implied_integer_from_binary_expansion():
    """``x = b0 + 2*b1`` (binaries) forces x integer."""
    m = dm.Model("p")
    b0, b1 = m.binary("b0"), m.binary("b1")
    x = m.continuous("x", lb=0, ub=3)
    m.minimize(x)
    m.subject_to(x == b0 + 2 * b1)
    assert detect_implied_integers(m) == {(x._index, 0)}


def test_implied_integer_general_integer_combination():
    """``x = 3*k - 2`` (k integer) forces x integer — not just binary factors."""
    m = dm.Model("g")
    k = m.integer("k", lb=0, ub=5)
    x = m.continuous("x", lb=-2, ub=13)
    m.minimize(x)
    m.subject_to(x == 3 * k - 2)
    assert (x._index, 0) in detect_implied_integers(m)


def test_implied_integer_fixpoint_chain():
    """``y = x + 1`` with ``x = b0 + b1`` proves both via fixpoint iteration."""
    m = dm.Model("c")
    b0, b1 = m.binary("b0"), m.binary("b1")
    x = m.continuous("x", lb=0, ub=2)
    y = m.continuous("y", lb=0, ub=3)
    m.minimize(y)
    m.subject_to(x == b0 + b1)
    m.subject_to(y == x + 1)
    assert len(detect_implied_integers(m)) == 2


def test_range_link_is_not_implied_integer():
    """``b <= x <= b+4`` does NOT force x integer — must not be detected (sound)."""
    m = dm.Model("r")
    b = m.binary("b")
    x = m.continuous("x", lb=0, ub=5)
    m.minimize(x)
    m.subject_to(x >= b)
    m.subject_to(x <= b + 4)
    assert detect_implied_integers(m) == set()


def test_non_unit_coefficient_not_implied_integer():
    """``2x = k`` ⇒ x = k/2 is NOT always integer — must not be detected (sound)."""
    m = dm.Model("h")
    k = m.integer("k", lb=0, ub=4)
    x = m.continuous("x", lb=0, ub=2)
    m.minimize(x)
    m.subject_to(2 * x == k)
    assert detect_implied_integers(m) == set()


def test_continuous_neighbor_not_implied_integer():
    """``x = y + b`` with y continuous does NOT force x integer (sound)."""
    m = dm.Model("d")
    b = m.binary("b")
    y = m.continuous("y", lb=0, ub=3)
    x = m.continuous("x", lb=0, ub=4)
    m.minimize(x)
    m.subject_to(x == y + b)
    assert detect_implied_integers(m) == set()


# --------------------------------------------------------------------------- #
# reformulation: pure MILP + value preservation
# --------------------------------------------------------------------------- #


@pytest.mark.slow  # does end-to-end solves; kept out of the fast xdist suite
@pytest.mark.parametrize(
    "ux,uy,rhs,coef",
    [(5, 4, 7, 1), (6, 6, 20, 2), (5, 5, 13, 2), (8, 4, 15, 1), (7, 3, 11, 3)],
)
def test_reformulation_value_preserving(ux, uy, rhs, coef):
    """Reformulating an integer-bilinear model preserves its optimum."""
    m = dm.Model("t")
    a = m.integer("a", lb=0, ub=ux)
    b = m.integer("b", lb=0, ub=uy)
    m.minimize(a + coef * b)
    m.subject_to(a * b >= rhs)
    o0 = m.solve(time_limit=15).objective
    rm = reformulate_integer_bilinear(m)
    # the reformulation must leave no bilinear term (pure MILP)
    assert len(classify_nonlinear_terms(rm).bilinear) == 0
    o1 = rm.solve(time_limit=15).objective
    assert o0 is not None and o1 is not None
    assert o1 == pytest.approx(o0, rel=1e-4, abs=1e-4)


@pytest.mark.slow  # does end-to-end solves; kept out of the fast xdist suite
def test_knife_range_needs_enough_bits():
    """A factor whose value exceeds a few bits is expanded exactly (the 'knife=11'
    regression: too few bits truncates the product and over-constrains)."""
    m = dm.Model("k")
    n = m.integer("n", lb=0, ub=15)  # needs 4 bits
    w = m.integer("w", lb=0, ub=3)
    m.minimize(n)
    m.subject_to(n * w >= 33)  # n=11,w=3 -> 33 feasible; <4 bits would block n=11
    o0 = m.solve(time_limit=15).objective
    o1 = reformulate_integer_bilinear(m).solve(time_limit=15).objective
    assert o1 == pytest.approx(o0, rel=1e-4, abs=1e-4)


@pytest.mark.slow  # does end-to-end solves; kept out of the fast xdist suite
@pytest.mark.parametrize("ub", [3, 5, 6, 7])
def test_integer_square_reformulation_value_preserving(ub):
    """An integer square ``x**2`` is exactly linearized (binary expansion + AND
    products) — value-preserving and yields a pure MILP."""
    m = dm.Model("sq")
    x = m.integer("x", lb=0, ub=ub)
    m.minimize(x * x - 4 * x)  # (x-2)^2 - 4, optimum at x=2 -> -4
    o0 = m.solve(time_limit=10).objective
    rm = reformulate_integer_bilinear(m)
    t = classify_nonlinear_terms(rm)
    assert len(t.bilinear) == 0 and len(t.monomial) == 0
    o1 = rm.solve(time_limit=10).objective
    assert o1 == pytest.approx(o0, rel=1e-4, abs=1e-4)


@pytest.mark.slow  # does end-to-end solves; kept out of the fast xdist suite
def test_pow_two_form_is_handled():
    """The ``x ** 2`` (power) form, not just ``x*x``, is linearized."""
    m = dm.Model("p2")
    x = m.integer("x", lb=0, ub=6)
    m.minimize(x**2 - 4 * x)
    rm = reformulate_integer_bilinear(m)
    assert len(classify_nonlinear_terms(rm).monomial) == 0
    assert rm.solve(time_limit=10).objective == pytest.approx(-4.0, abs=1e-3)


def test_noop_on_continuous_bilinear():
    """A bilinear term with no integer factor is left untouched (no-op)."""
    m = dm.Model("c")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.minimize(x + y)
    m.subject_to(x * y >= 4)
    assert has_reformulation_work(m) is False
    assert reformulate_integer_bilinear(m) is m


def test_blowup_guard_falls_back():
    """When ``distribute_products`` would explode (nvs17: 7 vars -> 2751), the
    size guard discards the reformulation and returns the original model, so the
    solver falls back to the normal path instead of building an intractable MILP.
    (nvs14, a small sibling, is still adopted — see the value-preservation tests.)"""
    path = os.path.join(_DATA, "nvs17.nl")
    if not os.path.exists(path):
        pytest.skip("nvs17 instance unavailable")
    m = dm.from_nl(path)
    assert has_reformulation_work(m)  # it does contain integer bilinear/squares
    assert reformulate_integer_bilinear(m) is m  # ...but the reform is too large -> fall back


# --------------------------------------------------------------------------- #
# end-to-end: the ex126x family discopt previously could not solve at all
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.requires_pounce
def test_ex1263_solves_to_optimum_automatically():
    """``from_nl(ex1263).solve()`` reaches the proven optimum 19.6 with no manual
    intervention — implied-integer detection + reformulation run automatically.
    Before this work the solve returned no feasible solution at all."""
    path = os.path.join(_DATA, "ex1263.nl")
    if not os.path.exists(path):
        pytest.skip("ex1263 instance unavailable")
    m = dm.from_nl(path)
    assert has_reformulation_work(m)
    r = m.solve(time_limit=180, gap_tolerance=1e-4)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(19.6, abs=1e-2)
