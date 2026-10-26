"""#1952: implied-integer detection must not mark a variable the model leaves continuous.

Two independent causes, both of which marked a continuous variable integer and let
the integer-bilinear lift certify 2.0 on a model whose optimum is 2.5:

1. **Index resolution.** The linear extractor's ``IndexExpression`` fast path computed
   ``offset + int(idx)``, so ``a[-1]`` named the slot of the variable declared before
   ``a`` (the #941 pattern), and a partial index ``A[1]`` on a 2-D variable named the
   single slot ``A[0, 1]``. The same fast path lived in ``_try_extract_const_var``,
   ``linear_context._affine_terms`` and ``export/_extract.py``.
2. **Tolerance.** The detector accepted coefficients within 1e-9 of an integer and
   dropped coefficients below 1e-9 from the row's support, and because the float
   extractor rounds while it accumulates, even an exact float test proves integrality
   of a row that differs from the stored one (``2**53*x + 0.5*x - 2**53*x + x - y``
   extracts as ``{x: 1, y: -1}``; its exact ``x`` coefficient is 3/2).
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest
from discopt._relax.implied_integer import detect_implied_integers


def _neg_index_model(k):
    m = dm.Model("neg_index")
    b = m.continuous("b", lb=0, ub=2.5)  # declared before a
    a = m.continuous("a", shape=(3,), lb=0, ub=3)
    y = m.integer("y", lb=0, ub=3)
    z = m.continuous("z", lb=0, ub=1)
    m.subject_to(a[k] - y == 0)  # makes a[2] integer, says nothing about b
    m.maximize(b * z)
    return m, b, a


@pytest.mark.smoke
@pytest.mark.parametrize("k", [2, -1])
def test_negative_index_marks_the_indexed_element(k):
    m, b, a = _neg_index_model(k)
    assert detect_implied_integers(m) == {(a._index, 2)}


@pytest.mark.smoke
def test_partial_index_is_not_a_scalar_slot():
    m = dm.Model("partial")
    A = m.continuous("A", shape=(2, 3), lb=0, ub=3)
    y = m.integer("y", lb=0, ub=3)
    m.subject_to(A[1] - y == 0)
    m.minimize(dm.sum(A))
    assert (A._index, 1) not in detect_implied_integers(m)


@pytest.mark.smoke
def test_negative_index_end_to_end_certifies_true_optimum():
    m, b, _ = _neg_index_model(-1)
    r = m.solve(time_limit=20, threads=1)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(2.5, abs=1e-5)
    assert r.bound >= 2.5 - 1e-5  # max sense: a bound below 2.5 is a false certificate


@pytest.mark.smoke
def test_index_sites_resolve_negative_index():
    from discopt._relax.convexity.linear_context import _affine_terms
    from discopt._relax.problem_classifier import (
        _extract_linear_coefficients_sparse,
        _try_extract_const_var,
    )

    m, b, a = _neg_index_model(-1)
    n = sum(v.size for v in m._variables)
    a2 = m._flat_var_offset(a) + 2
    terms, _ = _extract_linear_coefficients_sparse(a[-1], m, n)
    assert terms == {a2: 1.0}
    assert _try_extract_const_var(a[-1], m) == (1.0, a2)
    assert _affine_terms(a[-1], m, n) == ({a2: 1.0}, 0.0)


@pytest.mark.smoke
def test_export_resolves_negative_index_and_refuses_partial():
    from discopt.export._extract import _var_index, _VarOffsets, flatten_variables

    m, b, a = _neg_index_model(-1)
    flat = flatten_variables(m)
    off = _VarOffsets(m._variables)
    assert _var_index(a[-1], flat, off) == off.get(a) + 2
    A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
    flat = flatten_variables(m)
    off = _VarOffsets(m._variables)
    with pytest.raises(ValueError):
        _var_index(A[1], flat, off)  # a whole row, not the single column of A[0, 1]


@pytest.mark.smoke
@pytest.mark.parametrize("coef", [5e-10, 5e-9])
def test_tiny_coefficient_keeps_its_variable_in_the_row(coef):
    m = dm.Model("tiny_coef")
    x = m.continuous("x", lb=0, ub=2.5)
    z = m.continuous("z", lb=0, ub=1)
    y = m.integer("y", lb=0, ub=3)
    w = m.continuous("w", lb=0, ub=1e9)
    m.subject_to(x - y + coef * w == 0)
    m.maximize(x * z)
    assert detect_implied_integers(m) == set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # deliberate bad scaling
        r = m.solve(time_limit=20, threads=1)
    assert r.bound >= 2.5 - 1e-5, (r.status, r.objective, r.bound)


def _rows(build):
    m = dm.Model("rows")
    x = m.continuous("x", lb=0, ub=3)
    y = m.integer("y", lb=0, ub=3)
    m.subject_to(build(x, y) == 0)
    m.minimize(x)
    return m


@pytest.mark.smoke
@pytest.mark.parametrize(
    "build",
    [
        lambda x, y: (1 + 2**-40) * x - y,  # not exactly ±1
        lambda x, y: 0.1 * x + 0.2 * x + 0.7 * x - y,  # sums to 1 only in floats
        lambda x, y: (1 - 2**-27) * ((1 + 2**-27) * x) - y,  # product rounds to 1
        lambda x, y: 2**53 * x + 0.5 * x - 2**53 * x + x - y,  # exact coefficient 3/2
        lambda x, y: x - y + 2**-60,  # non-integer constant hidden by rounding
    ],
    ids=["near_one", "decimal_sum", "product", "cancellation", "constant"],
)
def test_rounded_rows_are_not_integrality_proofs(build):
    assert detect_implied_integers(_rows(build)) == set()


@pytest.mark.smoke
@pytest.mark.parametrize(
    "build",
    [
        lambda x, y: x - y,
        lambda x, y: x - 2 * y + 3,
        lambda x, y: 0.5 * x + 0.5 * x - y,  # exact in binary64
        lambda x, y: (x - y) / 1,
        lambda x, y: -(y - x),
    ],
)
def test_exact_integer_rows_still_detected(build):
    m = _rows(build)
    x = m._variables[0]
    assert detect_implied_integers(m) == {(x._index, 0)}
