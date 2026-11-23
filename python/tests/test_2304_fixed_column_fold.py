"""#2304: a big-M row on a binary pinned by a singleton row lost its #1654 certificate.

Since f83addeba the HiGHS standard form rounds ``z == 1`` into ``z``'s box, ``[1, 1]``.
The #1654 tightening rewrites only free binaries, so ``z``'s big-M stayed in the matrix
and the #1295 guard read it against the row's logical (``1/M`` below ``2**-20``),
declining the tightened re-solve too. :func:`fold_fixed_columns` moves a fixed
column's entries into ``b`` before the re-solve, exactly.
"""

from __future__ import annotations

import warnings
from fractions import Fraction

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solver import _highs_std_form
from discopt.solvers import lp_milp_highs as L


def _two_jobs(big_m: float, fix: bool = True) -> dm.Model:
    m = dm.Model("two_jobs")
    s = m.continuous("s", shape=(2,), lb=0, ub=100)
    ms = m.continuous("MS", lb=0, ub=200)
    z = m.binary("z")
    p = [5.0, 3.0]
    m.subject_to(s[1] >= s[0] + p[0] - big_m * (1 - z), name="seq01")
    m.subject_to(s[0] >= s[1] + p[1] - big_m * z, name="seq10")
    if fix:
        m.subject_to(z == 1, name="sym")
    for j in range(2):
        m.subject_to(ms >= s[j] + p[j], name=f"ms{j}")
    m.minimize(ms)
    return m


@pytest.mark.requires_highs
@pytest.mark.parametrize("big_m", [1e7, 1e8, 1e9])
def test_pinned_binary_big_m_certifies(big_m, monkeypatch):
    """The issue's reproducer: certified ``optimal`` 8.0 at every large M."""
    pytest.importorskip("highspy")
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _two_jobs(big_m).solve()
    assert r.status == "optimal"
    assert r.gap_certified is True
    assert r.objective == pytest.approx(8.0, abs=1e-9)
    # Nothing here is a free binary, so only the fold ran, and the route says so.
    assert "fixed-column-folded form (#2304)" in r.algorithm_route


def test_fold_is_exact_on_the_reproducer():
    """The fold removes every entry of the fixed column, keeps its index and box, and
    every point with ``z`` at its value satisfies ``A x = b`` in both forms alike."""
    _, _, sf = _highs_std_form(_two_jobs(1e8))
    fixed = np.flatnonzero((sf.xl == sf.xu) & ~L._logical_columns(sf))
    assert fixed.size >= 1  # the singleton row pinned z (the #2304 premise)
    assert L.open_column_coefficient_ratio(sf) < L.UNSCALABLE_OPEN_RATIO
    sf_f, n = L.fold_fixed_columns(sf)
    assert sf_f is not None and n >= 2
    assert sf_f.n == sf.n and sf_f.m == sf.m
    assert np.array_equal(sf_f.xl, sf.xl) and np.array_equal(sf_f.xu, sf.xu)
    assert np.array_equal(sf_f.c, sf.c)
    for j in fixed:
        assert sf_f.A.indptr[j + 1] == sf_f.A.indptr[j]
    assert L.open_column_coefficient_ratio(sf_f) >= L.UNSCALABLE_OPEN_RATIO
    # The logical-column mask is unchanged: no column was left with a single entry.
    assert np.array_equal(L._logical_columns(sf_f), L._logical_columns(sf))
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(200):
        x = rng.uniform(np.where(sf.xl > -L.INF, sf.xl, -10), np.where(sf.xu < L.INF, sf.xu, 10))
        x[fixed] = sf.xl[fixed]
        r0 = sf.A @ x - sf.b
        r1 = sf_f.A @ x - sf_f.b
        assert np.allclose(r0, r1, rtol=0, atol=1e-6 * (1 + np.abs(sf.b).max()))
        checked += 1
    assert checked == 200


def test_fold_refuses_an_inexact_entry():
    """An entry whose fold is not representable leaves its whole column in place."""
    a = sp.csc_matrix(np.array([[1.0, 0.1, 1.0], [1.0, 3.0, 0.0]]))
    b = np.array([1.0, 2.0])
    xl = np.array([0.0, 0.3, 0.0])
    xu = np.array([5.0, 0.3, np.inf])
    sf = L.StdForm.from_arrays(np.array([1.0, 0.0, 0.0]), a, b, xl, xu)
    assert Fraction(1.0) - Fraction(0.1) * Fraction(0.3) != Fraction(1.0 - 0.1 * 0.3)
    assert L.fold_fixed_columns(sf) == (None, 0)
    # Exact values fold: 0.5 * 2.0 and 3.0 * 2.0 are representable.
    xl2, xu2 = np.array([0.0, 2.0, 0.0]), np.array([5.0, 2.0, np.inf])
    a2 = sp.csc_matrix(np.array([[1.0, 0.5, 1.0], [1.0, 3.0, 0.0]]))
    sf2 = L.StdForm.from_arrays(np.array([1.0, 0.0, 0.0]), a2, b, xl2, xu2)
    got, n = L.fold_fixed_columns(sf2)
    assert n == 2 and got is not None
    assert np.array_equal(got.b, [0.0, -4.0])
    assert got.A.indptr[2] == got.A.indptr[1]


def test_nothing_fixed_is_a_no_op():
    _, _, sf = _highs_std_form(_two_jobs(1e8, fix=False))
    assert L.fold_fixed_columns(sf) == (None, 0)
