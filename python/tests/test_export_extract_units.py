"""Unit tests for the export-layer expression extraction (#87).

``export/_extract`` decomposes objectives/constraints into flat linear and
quadratic coefficient maps for the LP/MPS writers. Every extraction here is
validated by re-evaluating the coefficient form against the original
expression at sampled points, plus the documented loud-refusal on
unsupported (nonlinear / super-quadratic) input.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.export._extract import (
    extract_linear_terms,
    extract_quadratic_terms,
    flatten_variables,
)
from discopt.modeling.core import Model, VarType

pytestmark = pytest.mark.unit


def _model():
    m = Model("ex")
    x = m.continuous("x", lb=0.0, ub=10.0, shape=(2,))
    z = m.continuous("z", lb=-5.0, ub=5.0)
    i = m.integer("i", lb=0, ub=3)
    return m, x, z, i


def test_flatten_variables_names_types_bounds():
    m, x, z, i = _model()
    flat = flatten_variables(m)
    names = [f[0] for f in flat]
    assert names == ["x_0", "x_1", "z", "i"]
    kinds = [f[1] for f in flat]
    assert kinds[:2] == [VarType.CONTINUOUS, VarType.CONTINUOUS]
    assert kinds[3] == VarType.INTEGER
    assert flat[2][3] == -5.0 and flat[2][4] == 5.0  # z bounds
    # 2-D arrays get multi-index suffixes.
    m2 = Model("ex2")
    m2.continuous("w", lb=0.0, ub=1.0, shape=(2, 2))
    names2 = [f[0] for f in flatten_variables(m2)]
    assert names2 == ["w_0_0", "w_0_1", "w_1_0", "w_1_1"]


def _lin_value(coeffs, const, pt):
    return const + sum(c * pt[j] for j, c in coeffs.items())


def test_extract_linear_terms_reproduces_expression():
    m, x, z, i = _model()
    flat = flatten_variables(m)
    expr = 3.0 * x[0] - x[1] / 2.0 + 2.0 * z - i + 4.5
    coeffs, const = extract_linear_terms(expr, flat, m._variables)
    rng = np.random.RandomState(0)
    for _ in range(5):
        pt = rng.uniform(-1.0, 1.0, size=4)
        want = 3.0 * pt[0] - pt[1] / 2.0 + 2.0 * pt[2] - pt[3] + 4.5
        assert _lin_value(coeffs, const, pt) == pytest.approx(want, abs=1e-12)


def test_extract_linear_terms_scalar_sum_and_array_refusal():
    m, x, z, i = _model()
    flat = flatten_variables(m)
    coeffs, const = extract_linear_terms(x[0] + x[1] + 1.0, flat, m._variables)
    assert const == pytest.approx(1.0)
    assert coeffs[0] == pytest.approx(1.0) and coeffs[1] == pytest.approx(1.0)
    # Whole-array reduction is deliberately unsupported for export: refuse
    # loudly rather than guess an element mapping.
    with pytest.raises(ValueError, match="without indexing"):
        extract_linear_terms(dm.sum(x) + 1.0, flat, m._variables)


def test_extract_linear_terms_refuses_nonlinear():
    m, x, z, i = _model()
    flat = flatten_variables(m)
    with pytest.raises(ValueError):
        extract_linear_terms(x[0] * x[1], flat, m._variables)
    with pytest.raises(ValueError):
        extract_linear_terms(dm.exp(z), flat, m._variables)
    with pytest.raises(ValueError):
        extract_linear_terms(z, flat, None)  # model_vars is mandatory


def _quad_value(quad, lin, const, pt):
    total = const + sum(c * pt[j] for j, c in lin.items())
    for (a, b), c in quad.items():
        total += c * pt[a] * pt[b]
    return total


def test_extract_quadratic_terms_reproduces_expression():
    m, x, z, i = _model()
    flat = flatten_variables(m)
    expr = x[0] ** 2 + 2.0 * x[0] * x[1] - (z**2) / 2.0 + 3.0 * z - 1.0
    quad, lin, const = extract_quadratic_terms(expr, flat, m._variables)
    # Canonical i <= j keys.
    assert all(a <= b for a, b in quad)
    rng = np.random.RandomState(1)
    for _ in range(6):
        pt = rng.uniform(-1.0, 1.0, size=4)
        want = pt[0] ** 2 + 2.0 * pt[0] * pt[1] - pt[2] ** 2 / 2.0 + 3.0 * pt[2] - 1.0
        assert _quad_value(quad, lin, const, pt) == pytest.approx(want, abs=1e-12)


def test_extract_quadratic_terms_refuses_cubic_and_transcendental():
    m, x, z, i = _model()
    flat = flatten_variables(m)
    with pytest.raises(ValueError):
        extract_quadratic_terms(x[0] ** 3, flat, m._variables)
    with pytest.raises(ValueError):
        extract_quadratic_terms(x[0] * x[1] * z, flat, m._variables)
    with pytest.raises(ValueError):
        extract_quadratic_terms(dm.log(x[0] + 1.0), flat, m._variables)
