"""Property tests for exact Q-matrix extraction (``extract_quadratic``).

The correctness contract (``docs/dev/certification-gap-plan.md`` §8 Phase 4
item 3, CLAUDE.md §5): extraction is *exact or abstains*. These tests pin

1. **Exactness.** For randomly generated quadratic expressions, the
   reconstructed form ``xᵀ Q x + cᵀ x + d`` reproduces the original
   expression identically on ≥ 200 random points to 1e-12.
2. **Symmetric-split convention.** ``Q[i,j] == Q[j,i] == ½·coeff`` for a
   cross term, ``Q[i,i] == coeff`` for a square.
3. **Non-quadratic rejection.** Degree-3+ terms, transcendentals,
   bilinear-with-transcendental, variable-in-denominator, fractional
   powers, and `abs` all return ``None`` — never a mis-extracted ``Q``.
4. **PSD / NSD helpers** agree with a direct eigenvalue computation.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.dag_compiler import compile_expression
from discopt._jax.quadratic_form import (
    extract_quadratic,
    is_purely_quadratic,
    quadratic_is_nsd,
    quadratic_is_psd,
)

# Number of random evaluation points per expression (contract: >= 200).
_N_POINTS = 250
# Reconstruction tolerance (contract: 1e-12).
_TOL = 1e-12


def _reconstruct(Q: np.ndarray, c: np.ndarray, d: float, x: np.ndarray) -> float:
    return float(x @ Q @ x + c @ x + d)


def _random_quadratic_expr(rng: np.random.Generator, n: int):
    """Build a random quadratic expression over ``n`` scalar variables.

    Returns ``(model, var, expr, Q_true, c_true, d_true)`` where the
    *_true arrays are the reference coefficients the expression encodes,
    independent of ``extract_quadratic`` (so the test does not check the
    function against itself).
    """
    m = dm.Model("q")
    x = m.continuous("x", shape=(n,), lb=-3.0, ub=3.0)

    Q_true = np.zeros((n, n))
    c_true = np.zeros(n)
    d_true = float(rng.uniform(-5, 5))

    # Seed the sum with the first square so ``expr`` is an Expression from
    # the start; fold the constant offset in afterwards (scalars wrap to a
    # Constant via the operator overloads).
    expr = 0.0 * x[0]

    # Squares a_i x_i^2.
    for i in range(n):
        if rng.random() < 0.7:
            a = float(rng.uniform(-4, 4))
            Q_true[i, i] += a
            expr = expr + a * x[i] ** 2

    # Linear c_i x_i.
    for i in range(n):
        if rng.random() < 0.7:
            b = float(rng.uniform(-4, 4))
            c_true[i] += b
            expr = expr + b * x[i]

    # Cross terms b_ij x_i x_j (i < j).
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                b = float(rng.uniform(-4, 4))
                Q_true[i, j] += 0.5 * b
                Q_true[j, i] += 0.5 * b
                expr = expr + b * x[i] * x[j]

    expr = expr + d_true

    return m, x, expr, Q_true, c_true, d_true


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(12))
def test_extract_quadratic_exact_reconstruction(seed):
    """Random quadratics reconstruct to 1e-12 on >= 250 points."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 6))
    m, x, expr, Q_true, c_true, d_true = _random_quadratic_expr(rng, n)

    res = extract_quadratic(expr, n, m)
    assert res is not None, "purely quadratic expression must be recognized"
    Q, c, d = res

    # Q is symmetric by construction.
    assert np.allclose(Q, Q.T, atol=0.0), "Q must be exactly symmetric"

    # Coefficients match the independently-tracked reference.
    assert np.allclose(Q, Q_true, atol=1e-12)
    assert np.allclose(c, c_true, atol=1e-12)
    assert abs(d - d_true) <= 1e-12

    # Reconstruction identity on many random points, against the compiled
    # DAG evaluator (the ground-truth evaluation of the original expr).
    f = compile_expression(expr, m)
    pts = rng.uniform(-3.0, 3.0, size=(_N_POINTS, n))
    for k in range(_N_POINTS):
        xv = pts[k]
        original = float(f(xv))
        recon = _reconstruct(Q, c, d, xv)
        assert abs(original - recon) <= _TOL, f"seed={seed} pt={k}: |{original} - {recon}| > {_TOL}"


@pytest.mark.unit
def test_symmetric_split_convention():
    """Cross term coeff b -> Q[i,j]=Q[j,i]=b/2; square a -> Q[i,i]=a."""
    m = dm.Model("s")
    x = m.continuous("x", shape=(3,), lb=-2, ub=2)
    expr = 5.0 * x[0] * x[1] + 3.0 * x[2] ** 2 - 2.0 * x[0] * x[2]
    Q, c, d = extract_quadratic(expr, 3, m)
    assert Q[0, 1] == pytest.approx(2.5)
    assert Q[1, 0] == pytest.approx(2.5)
    assert Q[2, 2] == pytest.approx(3.0)
    assert Q[0, 2] == pytest.approx(-1.0)
    assert Q[2, 0] == pytest.approx(-1.0)
    assert np.allclose(c, 0.0)
    assert d == pytest.approx(0.0)


@pytest.mark.unit
def test_pure_square_and_bilinear_and_cross():
    m = dm.Model("m")
    x = m.continuous("x", shape=(2,), lb=-1, ub=1)
    # (x0 - x1)^2 = x0^2 - 2 x0 x1 + x1^2
    expr = (x[0] - x[1]) ** 2
    Q, c, d = extract_quadratic(expr, 2, m)
    assert np.allclose(Q, np.array([[1.0, -1.0], [-1.0, 1.0]]))
    assert np.allclose(c, 0.0)
    assert d == pytest.approx(0.0)
    # This Q is PSD (the square is convex).
    assert quadratic_is_psd(Q) is True


@pytest.mark.unit
def test_affine_and_constant_are_quadratic_with_zero_Q():
    m = dm.Model("a")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    aff = 3.0 * x[0] - 2.0 * x[1] + 4.0
    Q, c, d = extract_quadratic(aff, 2, m)
    assert np.allclose(Q, 0.0)
    assert np.allclose(c, [3.0, -2.0])
    assert d == pytest.approx(4.0)
    # A constant offset folded onto a zeroed variable term.
    cst = extract_quadratic(0.0 * x[0] + 7.0, 2, m)
    assert cst is not None
    Qc, cc, dc = cst
    assert np.allclose(Qc, 0.0) and np.allclose(cc, 0.0) and dc == pytest.approx(7.0)


@pytest.mark.unit
def test_non_quadratic_expressions_return_none():
    """Every non-(purely-)quadratic shape must abstain (return None)."""
    m = dm.Model("n")
    x = m.continuous("x", shape=(3,), lb=0.5, ub=3.0)

    cases = {
        "cube": x[0] ** 3,
        "quartic": x[0] ** 4,
        "trilinear": x[0] * x[1] * x[2],
        "square_times_var": x[0] ** 2 * x[1],
        "exp": dm.exp(x[0]),
        "log": dm.log(x[0]),
        "sin": dm.sin(x[0]),
        "sqrt": dm.sqrt(x[0]),
        "bilinear_with_transcendental": x[0] * dm.sin(x[1]),
        "sum_with_transcendental": x[0] ** 2 + dm.exp(x[1]),
        "var_in_denominator": x[0] / x[1],
        "fractional_power": x[0] ** 0.5,
        "abs": abs(x[0]),
        "reciprocal": 1.0 / x[0],
    }
    for name, expr in cases.items():
        assert extract_quadratic(expr, 3, m) is None, f"{name} must abstain"
        assert not is_purely_quadratic(expr, 3, m), f"{name} is_purely_quadratic must be False"


@pytest.mark.unit
def test_non_quadratic_never_misextracted_on_points():
    """A degree-3 expression must NOT be silently reduced to a wrong Q.

    Belt-and-braces: even if some future refactor mistakenly returned a Q,
    it could not reproduce a cubic on random points. We assert the
    abstention directly, which is the sound behavior.
    """
    m = dm.Model("c")
    x = m.continuous("x", shape=(2,), lb=-2, ub=2)
    cubic = x[0] ** 3 + x[0] * x[1]
    assert extract_quadratic(cubic, 2, m) is None


@pytest.mark.unit
def test_out_of_range_index_abstains():
    m = dm.Model("r")
    x = m.continuous("x", shape=(3,), lb=-1, ub=1)
    expr = x[2] ** 2 + x[0] * x[1]
    # n too small to hold index 2 -> must abstain rather than write OOB.
    assert extract_quadratic(expr, 2, m) is None
    # Correct n succeeds.
    assert extract_quadratic(expr, 3, m) is not None


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(20))
def test_psd_nsd_helpers_match_eigvalsh(seed):
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(1, 6))
    A = rng.uniform(-2, 2, size=(n, n))
    Q = 0.5 * (A + A.T)  # symmetric
    lam = np.linalg.eigvalsh(Q)
    expect_psd = bool(lam.min() >= -1e-10)
    expect_nsd = bool(lam.max() <= 1e-10)
    assert quadratic_is_psd(Q) is expect_psd
    assert quadratic_is_nsd(Q) is expect_nsd


@pytest.mark.unit
def test_psd_helper_handles_degenerate_inputs():
    # Non-finite -> None (unusable).
    bad = np.array([[np.nan, 0.0], [0.0, 1.0]])
    assert quadratic_is_psd(bad) is None
    assert quadratic_is_nsd(bad) is None
    # Non-square -> None.
    assert quadratic_is_psd(np.zeros((2, 3))) is None
    # Zero-dimensional form is trivially PSD.
    assert quadratic_is_psd(np.zeros((0, 0))) is True
    # A clearly-PSD identity, a clearly-NSD negative identity.
    assert quadratic_is_psd(np.eye(3)) is True
    assert quadratic_is_nsd(-np.eye(3)) is True
    # Indefinite.
    indef = np.array([[1.0, 0.0], [0.0, -1.0]])
    assert quadratic_is_psd(indef) is False
    assert quadratic_is_nsd(indef) is False
