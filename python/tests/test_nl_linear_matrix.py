"""``DISCOPT_NL_LINEAR_MATRIX``: a linear model read from a file takes its rows as
fast-API matrix blocks over several variables (``VarBlock``) instead of one expression
tree per row. The rows, their order and the standard form the solver sees must be the
same as the expression path's; these tests pin that, and the multi-variable block API
the reader uses.
"""

from __future__ import annotations

import copy
import pickle

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.problem_classifier import extract_lp_data
from discopt.modeling.core import VarBlock


def _mixed_model() -> dm.Model:
    """Integer and continuous columns sharing rows, all three senses, in an order
    that interleaves them, plus a constant in a row body."""
    m = dm.Model("mixed")
    x = m.integer("x", lb=0, ub=10)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    z = m.integer("z", lb=0, ub=3)
    m.subject_to(x + 2 * y - z <= 7.5)
    m.subject_to(3 * x - y >= 2.0)
    m.subject_to(x + y + z + 1.0 == 6.0)
    m.subject_to(-x + 4 * z <= 9.0)
    m.subject_to(y - z >= -4.0)
    m.minimize(-2 * x - y + 3 * z)
    return m


def _write_nl(m: dm.Model, tmp_path) -> str:
    p = str(tmp_path / f"{m.name}.nl")
    m.to_nl(p)
    return p


def _std(m: dm.Model):
    lp = extract_lp_data(m)
    a = sp.csr_matrix(lp.A_eq)
    a.sort_indices()
    return a, [np.asarray(getattr(lp, f)) for f in ("b_eq", "c", "x_l", "x_u")], lp.obj_const


@pytest.mark.smoke
def test_reader_rows_are_matrix_blocks_with_an_identical_standard_form(tmp_path, monkeypatch):
    p = _write_nl(_mixed_model(), tmp_path)
    monkeypatch.setenv("DISCOPT_NL_LINEAR_MATRIX", "0")
    m0 = dm.from_nl(p)
    monkeypatch.setenv("DISCOPT_NL_LINEAR_MATRIX", "1")
    m1 = dm.from_nl(p)
    assert len(m0._constraints) == 5 and not m0._builder_linear_blocks
    assert not m1._constraints and m1._builder_linear_blocks, "the matrix path did not engage"
    assert all(isinstance(b[1], VarBlock) for b in m1._builder_linear_blocks)
    (a0, v0, k0), (a1, v1, k1) = _std(m0), _std(m1)
    assert a0.shape == a1.shape
    for u, w in ((a0.indptr, a1.indptr), (a0.indices, a1.indices), (a0.data, a1.data)):
        assert np.array_equal(u, w)
    for u, w in zip(v0, v1):
        assert np.array_equal(u, w)
    assert k0 == k1


@pytest.mark.smoke
def test_reader_models_solve_to_the_same_certified_optimum(tmp_path, monkeypatch):
    p = _write_nl(_mixed_model(), tmp_path)
    out = {}
    for arm in ("0", "1"):
        monkeypatch.setenv("DISCOPT_NL_LINEAR_MATRIX", arm)
        r = dm.from_nl(p).solve(time_limit=30)
        assert r.status == "optimal" and r.gap_certified, (arm, r.status)
        out[arm] = r.objective
    assert out["0"] == pytest.approx(out["1"], abs=1e-9)
    # The expression model itself agrees.
    assert _mixed_model().solve(time_limit=30).objective == pytest.approx(out["1"], abs=1e-9)


@pytest.mark.smoke
def test_one_scalar_variable_keeps_its_row(tmp_path):
    """A scalar variable in a fast-API block used to be referenced as ``x[0]`` and the
    row was lost: the one-variable model solved to 0 instead of 1 (#1561 caught it)."""
    m = dm.Model("tiny")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    m.subject_to(x >= 1.0)
    r = dm.from_nl(_write_nl(m, tmp_path)).solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified and r.objective == pytest.approx(1.0)
    m2 = dm.Model("s")
    y = m2.continuous("y", lb=0, ub=10)
    m2.add_linear_constraints(np.array([[1.0]]), y, ">=", [1.0])
    m2.minimize(y)
    r2 = m2.solve(time_limit=30)
    assert r2.status == "optimal" and r2.objective == pytest.approx(1.0)


@pytest.mark.smoke
def test_multi_variable_block_matches_the_expression_model():
    """``add_linear_constraints(A, [x, y, z], ...)``: the columns are x's, then y's,
    then z's elements; mixed types in one row."""
    m = dm.Model("blk")
    x = m.integer("x", lb=0, ub=10)
    y = m.continuous("y", shape=(2,), lb=-5.0, ub=5.0)
    z = m.integer("z", lb=0, ub=3)
    A = np.array([[1.0, 2.0, 0.0, -1.0], [3.0, -1.0, 1.0, 0.0]])
    m.add_linear_constraints(A, [x, y, z], "<=", [7.5, 9.0])
    m.add_linear_constraints(np.array([[1.0, 1.0, 1.0, 1.0]]), [x, y, z], "==", [6.0])
    m.minimize(-2 * x - y[0] - y[1] + 3 * z)
    e = dm.Model("expr")
    ex = e.integer("x", lb=0, ub=10)
    ey = e.continuous("y", shape=(2,), lb=-5.0, ub=5.0)
    ez = e.integer("z", lb=0, ub=3)
    e.subject_to(ex + 2 * ey[0] - ez <= 7.5)
    e.subject_to(3 * ex - ey[0] + ey[1] <= 9.0)
    e.subject_to(ex + ey[0] + ey[1] + ez == 6.0)
    e.minimize(-2 * ex - ey[0] - ey[1] + 3 * ez)
    rb, re = m.solve(time_limit=30), e.solve(time_limit=30)
    assert rb.status == re.status == "optimal"
    assert rb.objective == pytest.approx(re.objective, abs=1e-9)
    with pytest.raises(ValueError):
        dm.Model("other").add_linear_constraints(A, [x, y, z], "<=", [1.0, 1.0])


@pytest.mark.smoke
def test_var_block_models_copy_pickle_and_serialize(tmp_path, monkeypatch):
    monkeypatch.setenv("DISCOPT_NL_LINEAR_MATRIX", "1")
    m = dm.from_nl(_write_nl(_mixed_model(), tmp_path))
    want = m.solve(time_limit=30).objective
    checked = 0
    for clone in (copy.deepcopy(m), pickle.loads(pickle.dumps(m))):
        assert any(isinstance(b[1], VarBlock) for b in clone._builder_linear_blocks)
        assert clone.solve(time_limit=30).objective == pytest.approx(want, abs=1e-9)
        checked += 1
    from discopt import serialize

    doc = serialize.dumps(m) if hasattr(serialize, "dumps") else None
    if doc is not None:
        back = serialize.loads(doc)
        assert back.solve(time_limit=30).objective == pytest.approx(want, abs=1e-9)
        checked += 1
    assert checked >= 2
