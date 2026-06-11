"""Tests for the posynomial / monomial recogniser (issue #41, phase 1).

Acceptance criterion #1: ``is_posynomial`` passes a suite covering simple
monomials, sums of monomials, nested product expansion, and the three
rejection cases (negative coefficient, zero lower bound, non-constant
exponent).
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt._jax.convexity.posynomial import (
    Monomial,
    PosynomialForm,
    is_monomial,
    is_posynomial,
)
from discopt.modeling.core import Model

# A default strictly-positive box for GP variables.
POS = dict(lb=1e-3, ub=1e3)


def _pos_model(name: str, *names: str) -> tuple[Model, list]:
    m = Model(name)
    vars_ = [m.continuous(n, **POS) for n in names]
    return m, vars_


def _offsets(model: Model, *vars_) -> list[int]:
    """Flat scalar offsets of the named scalar variables, in order."""
    out = []
    for target in vars_:
        offset = 0
        for v in model._variables:
            if v is target:
                out.append(offset)
                break
            offset += v.size
    return out


# ──────────────────────────────────────────────────────────────────────
# Positive recognition
# ──────────────────────────────────────────────────────────────────────


class TestMonomialRecognition:
    def test_bare_variable_is_monomial(self):
        m, (x,) = _pos_model("bare", "x")
        mono = is_monomial(x, m)
        assert mono is not None
        assert mono.coeff == pytest.approx(1.0)
        (ox,) = _offsets(m, x)
        assert mono.exponents == {ox: 1.0}

    def test_scaled_product_monomial(self):
        m, (x, y) = _pos_model("prod", "x", "y")
        mono = is_monomial(2.0 * x * y, m)
        assert mono is not None
        assert mono.coeff == pytest.approx(2.0)
        ox, oy = _offsets(m, x, y)
        assert mono.exponents == {ox: 1.0, oy: 1.0}

    def test_monomial_with_negative_real_exponent(self):
        # 3 * x / sqrt(y) = 3 * x^1 * y^(-1/2)
        m, (x, y) = _pos_model("negexp", "x", "y")
        mono = is_monomial(3.0 * x / dm.sqrt(y), m)
        assert mono is not None
        assert mono.coeff == pytest.approx(3.0)
        ox, oy = _offsets(m, x, y)
        assert mono.exponents[ox] == pytest.approx(1.0)
        assert mono.exponents[oy] == pytest.approx(-0.5)

    def test_power_of_monomial(self):
        # (2 * x * y)^1.5 = 2^1.5 * x^1.5 * y^1.5
        m, (x, y) = _pos_model("pow", "x", "y")
        mono = is_monomial((2.0 * x * y) ** 1.5, m)
        assert mono is not None
        assert mono.coeff == pytest.approx(2.0**1.5)
        ox, oy = _offsets(m, x, y)
        assert mono.exponents[ox] == pytest.approx(1.5)
        assert mono.exponents[oy] == pytest.approx(1.5)

    def test_nested_product_expansion_accumulates_exponents(self):
        # x * x * x -> x^3 (same variable appears repeatedly)
        m, (x,) = _pos_model("nested", "x")
        mono = is_monomial(x * x * x, m)
        assert mono is not None
        (ox,) = _offsets(m, x)
        assert mono.exponents[ox] == pytest.approx(3.0)

    def test_division_of_monomials_subtracts_exponents(self):
        # (x^2 * y) / (x * y^3) = x^1 * y^-2
        m, (x, y) = _pos_model("div", "x", "y")
        mono = is_monomial((x * x * y) / (x * y * y * y), m)
        assert mono is not None
        ox, oy = _offsets(m, x, y)
        assert mono.exponents[ox] == pytest.approx(1.0)
        assert mono.exponents[oy] == pytest.approx(-2.0)


class TestPosynomialRecognition:
    def test_sum_of_two_monomials(self):
        # 2*x*y + 3*x/sqrt(y) (the issue's worked example)
        m, (x, y) = _pos_model("posy", "x", "y")
        form = is_posynomial(2.0 * x * y + 3.0 * x / dm.sqrt(y), m)
        assert form is not None
        assert not form.is_monomial
        assert len(form.monomials) == 2
        coeffs = sorted(mo.coeff for mo in form.monomials)
        assert coeffs == pytest.approx([2.0, 3.0])

    def test_three_term_posynomial(self):
        # x^1.5 + x*y^(-0.5)*z^2 + 5
        m, (x, y, z) = _pos_model("posy3", "x", "y", "z")
        expr = x**1.5 + x * y ** (-0.5) * z**2 + 5.0
        form = is_posynomial(expr, m)
        assert form is not None
        assert len(form.monomials) == 3

    def test_indexed_vector_variable_posynomial(self):
        m = Model("vecposy")
        x = m.continuous("x", shape=(3,), lb=1e-2, ub=10.0)
        form = is_posynomial(2.0 * x[0] * x[1] + x[2] ** 2, m)
        assert form is not None
        assert len(form.monomials) == 2
        # x[0], x[1], x[2] are flat offsets 0, 1, 2.
        assert form.variable_offsets() == {0, 1, 2}

    def test_constant_plus_variable_posynomial(self):
        # 4 + x: a positive constant monomial plus a variable monomial.
        m, (x,) = _pos_model("constplus", "x")
        form = is_posynomial(4.0 + x, m)
        assert form is not None
        assert len(form.monomials) == 2

    def test_variable_offsets_excludes_cancelled_exponents(self):
        # x * y / y = x (y cancels to exponent 0)
        m, (x, y) = _pos_model("cancel", "x", "y")
        form = is_posynomial(x * y / y, m)
        assert form is not None
        (ox,) = _offsets(m, x)
        assert form.variable_offsets() == {ox}


# ──────────────────────────────────────────────────────────────────────
# Rejection cases
# ──────────────────────────────────────────────────────────────────────


class TestPosynomialRejection:
    def test_negative_coefficient_rejected(self):
        # 2*x*y - 3*x is a signomial, not a posynomial.
        m, (x, y) = _pos_model("signomial", "x", "y")
        assert is_posynomial(2.0 * x * y - 3.0 * x, m) is None

    def test_explicit_negative_monomial_rejected(self):
        m, (x,) = _pos_model("negmono", "x")
        assert is_posynomial(-2.0 * x, m) is None
        assert is_monomial(-2.0 * x, m) is None

    def test_zero_lower_bound_variable_rejected(self):
        m = Model("zerolb")
        x = m.continuous("x", lb=0.0, ub=10.0)
        y = m.continuous("y", lb=1e-3, ub=10.0)
        # x has lb == 0, so x*y is not a posynomial on the strictly-positive box.
        assert is_posynomial(x * y, m) is None

    def test_negative_lower_bound_variable_rejected(self):
        m = Model("neglb")
        x = m.continuous("x", lb=-1.0, ub=10.0)
        assert is_posynomial(x * x, m) is None

    def test_non_constant_exponent_rejected(self):
        # x ** y has a variable exponent.
        m, (x, y) = _pos_model("varexp", "x", "y")
        assert is_posynomial(x**y, m) is None

    def test_non_posynomial_atom_rejected(self):
        # exp(x) is not a monomial/posynomial term.
        m, (x,) = _pos_model("expatom", "x")
        assert is_posynomial(dm.exp(x) + x, m) is None

    def test_log_atom_rejected(self):
        m, (x,) = _pos_model("logatom", "x")
        assert is_posynomial(dm.log(x) + 1.0, m) is None

    def test_non_integer_power_of_negative_base_rejected(self):
        # (-x)^0.5 has no real value on x > 0; the negative coefficient
        # base raised to a fractional power must be rejected.
        m, (x,) = _pos_model("negbase", "x")
        assert is_monomial((-1.0 * x) ** 0.5, m) is None


class TestMonomialVsPosynomial:
    def test_is_monomial_false_for_multi_term(self):
        m, (x, y) = _pos_model("multi", "x", "y")
        assert is_monomial(x + y, m) is None
        assert is_posynomial(x + y, m) is not None

    def test_dataclasses_constructable(self):
        # Light structural sanity check on the public dataclasses.
        mono = Monomial(2.0, {0: 1.0, 1: -0.5})
        form = PosynomialForm([mono])
        assert form.is_monomial
        assert form.variable_offsets() == {0, 1}
