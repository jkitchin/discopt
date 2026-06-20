"""Tests for log-curvature classification (GP change of variables)."""

import pytest

sp = pytest.importorskip("sympy")  # design-time [sympy] extra; skip if absent

from discopt._jax.symbolic.log_curvature import (  # noqa: E402
    is_monomial,
    is_posynomial,
    log_curvature,
)

pytestmark = pytest.mark.relaxation

x, y, z = sp.symbols("x y z", positive=True)


def test_monomial_is_log_affine():
    expr = 3 * x**2 * y**0.5
    assert log_curvature(expr) == "log_affine"
    is_mono, log_coeff, exps = is_monomial(expr)
    assert is_mono is True
    assert log_coeff == pytest.approx(float(sp.log(3)))
    assert exps[x] == 2
    assert float(exps[y]) == pytest.approx(0.5)


def test_posynomial_is_log_convex():
    expr = x * y + 2 * x**2 + 0.5 * y
    assert is_posynomial(expr) is True
    assert is_monomial(expr)[0] is False
    assert log_curvature(expr) == "log_convex"


def test_reciprocal_posynomial_is_log_concave():
    expr = 1 / (x + y)
    assert log_curvature(expr) == "log_concave"


def test_mixed_sign_is_none():
    expr = x + y - z
    assert is_posynomial(expr) is False
    assert log_curvature(expr) == "none"


def test_transcendental_is_none():
    expr = sp.sin(x)
    assert is_monomial(expr)[0] is False
    assert log_curvature(expr) == "none"


def test_single_monomial_not_log_convex():
    expr = 4 * x**1.5
    assert log_curvature(expr) == "log_affine"
    is_mono, log_coeff, exps = is_monomial(expr)
    assert is_mono is True
    assert log_coeff == pytest.approx(float(sp.log(4)))
    assert float(exps[x]) == pytest.approx(1.5)


def test_monomial_ratio_is_log_affine():
    expr = x / y
    assert log_curvature(expr) == "log_affine"
    is_mono, _, exps = is_monomial(expr)
    assert is_mono is True
    assert exps[x] == 1
    assert exps[y] == -1
