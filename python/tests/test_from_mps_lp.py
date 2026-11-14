"""``discopt.modeling.from_mps`` / ``from_lp``: vector-form models read from MPS / LP.

Oracles, independent of the readers under test:

* a random differential: small mixed-integer models built with the scalar API,
  written with ``Model.to_mps`` / ``to_lp``, read back and solved; the optimum must
  equal the directly built model's;
* every returned point is checked against the RAW file arrays (``read_mps`` /
  ``read_lp``) in file column order, for bounds, rows and integrality, and its
  objective is recomputed from the file's ``c``, offset and Hessian;
* hand-written files whose optimum is worked out by hand.
"""

from __future__ import annotations

import gzip

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._lpfile import read_lp
from discopt._mps import read_mps
from discopt.modeling._linear_files import CONT_NAME, INT_NAME


def _check_against_file(d: dict, model, result, hessian=None) -> float:
    """The point in file order satisfies the file's own arrays; returns its objective."""
    x = model.file_columns.values(result)
    assert x is not None and x.shape == (d["n"],)
    tol = 1e-6
    assert np.all(x >= d["col_lb"] - tol) and np.all(x <= d["col_ub"] + tol)
    integ = np.asarray(d["integrality"]) == 1
    assert np.all(np.abs(x[integ] - np.round(x[integ])) <= 1e-6)
    act = d["A"] @ x
    assert np.all(act >= d["row_lo"] - 1e-6 * (1 + np.abs(d["row_lo"])))
    assert np.all(act <= d["row_hi"] + 1e-6 * (1 + np.abs(d["row_hi"])))
    obj = float(d["c"] @ x + d["offset"])
    if hessian is not None:
        h = hessian.toarray()
        q = np.tril(h) + np.tril(h, -1).T
        obj += 0.5 * float(x @ q @ x)
    return obj


def _random_model(rng, quadratic: bool) -> Model:
    m = Model("r")
    n_int, n_cont = int(rng.integers(1, 4)), int(rng.integers(0, 3))
    xs = [
        m.integer(f"i{k}", lb=float(rng.integers(-3, 1)), ub=float(rng.integers(1, 4)))
        for k in range(n_int)
    ]
    xs += [
        m.continuous(f"c{k}", lb=float(rng.integers(-2, 1)), ub=float(rng.integers(1, 3)))
        for k in range(n_cont)
    ]
    for _ in range(int(rng.integers(1, 4))):
        a = rng.integers(-3, 4, size=len(xs)).astype(float)
        if not a.any():
            a[0] = 1.0
        body = dm.sum([float(a[j]) * xs[j] for j in range(len(xs))])
        kind = int(rng.integers(0, 3))
        rhs = float(rng.integers(-2, 4))
        if kind == 0:
            m.subject_to(body <= rhs)
        elif kind == 1:
            m.subject_to(body >= rhs - 4)
        else:
            m.subject_to(body <= rhs)
            m.subject_to(body >= rhs - 2)
    c = rng.integers(-3, 4, size=len(xs)).astype(float)
    obj = dm.sum([float(c[j]) * xs[j] for j in range(len(xs))]) + 1.5
    if quadratic:
        for j in range(len(xs)):
            obj = obj + float(rng.integers(0, 3)) * xs[j] * xs[j]
        if len(xs) > 1:
            obj = obj + 0.5 * xs[0] * xs[-1] + 0.5 * xs[0] * xs[0]
    if rng.integers(0, 2) and not quadratic:
        m.maximize(obj)
    else:
        m.minimize(obj)
    return m


def _solve(m):
    return m.solve(time_limit=30, gap_tolerance=1e-9)


@pytest.mark.parametrize("seed", range(12))
def test_mps_round_trip_matches_the_direct_model(seed, tmp_path):
    rng = np.random.default_rng(seed)
    direct = _random_model(rng, quadratic=False)
    path = str(tmp_path / "m.mps")
    direct.to_mps(path)
    want = _solve(direct)
    read = dm.from_mps(path)
    got = _solve(read)
    assert got.status == want.status, (got.status, want.status)
    if want.status == "optimal":
        assert abs(got.objective - want.objective) <= 1e-6 * (1 + abs(want.objective))
        val = _check_against_file(read_mps(path), read, got)
        assert abs(val - got.objective) <= 1e-6 * (1 + abs(val))


@pytest.mark.parametrize("seed", range(8))
def test_lp_round_trip_with_a_quadratic_objective(seed, tmp_path):
    rng = np.random.default_rng(100 + seed)
    direct = _random_model(rng, quadratic=True)
    path = str(tmp_path / "m.lp")
    direct.to_lp(path)
    d = read_lp(path)
    want = _solve(direct)
    read = dm.from_lp(path)
    got = _solve(read)
    assert got.status == want.status, (got.status, want.status)
    if want.status == "optimal":
        assert abs(got.objective - want.objective) <= 1e-5 * (1 + abs(want.objective))
        val = _check_against_file(d, read, got, hessian=d["hessian"])
        assert abs(val - got.objective) <= 1e-6 * (1 + abs(val))


MPS = """NAME hand
OBJSENSE
    MAX
ROWS
 N  obj
 L  cap
 E  bal
 G  low
 N  free
COLUMNS
    MARKER  'MARKER'  'INTORG'
    y  obj  3  cap  2
    y  bal  1  free  9
    MARKER  'MARKER'  'INTEND'
    z  obj  1  cap  1
    z  bal  -1  low  1
    z  free  1
RHS
    rhs  cap  7.5  bal  0.5
    rhs  low  -1  obj  -10
RANGES
    rng  cap  6
BOUNDS
 UP bnd y 10
 UP bnd z 10
ENDATA
"""


def test_hand_written_mps(tmp_path):
    """max 3y + z + 10 (an RHS on the objective row is -offset in HiGHS's reading)
    s.t. 1.5 <= 2y + z <= 7.5 (ranged), y - z = 0.5, z >= -1; y integer in [0, 10],
    z in [0, 10]; the N row `free` is dropped. With z = y - 0.5: 3y - 0.5 <= 7.5, so
    y <= 2 (integer), z = 1.5, objective 3*2 + 1.5 + 10 = 17.5."""
    path = str(tmp_path / "hand.mps.gz")
    with gzip.open(path, "wt") as f:
        f.write(MPS)
    d = read_mps(path)
    m = dm.from_mps(path)
    assert [v.name for v in m._variables] == [INT_NAME, CONT_NAME]
    assert m.file_columns.names == ("y", "z")
    r = _solve(m)
    assert r.status == "optimal"
    assert abs(r.objective - 17.5) <= 1e-7, r.objective
    np.testing.assert_allclose(m.file_columns.values(r), [2.0, 1.5], atol=1e-7)
    assert m.file_columns.as_dict(r) == pytest.approx({"y": 2.0, "z": 1.5}, abs=1e-7)
    assert abs(_check_against_file(d, m, r) - 17.5) <= 1e-7
    # The block names carry the rows: one ranged row in both `le` and `ge`.
    names = [blk[4] for blk in m._builder_linear_blocks]
    assert sorted(names) == ["eq", "ge", "le"]


def test_refusals(tmp_path):
    semi = MPS.replace(" UP bnd z 10", " SC bnd z 10")
    p = tmp_path / "semi.mps"
    p.write_text(semi)
    with pytest.raises(NotImplementedError, match="semi-continuous"):
        dm.from_mps(str(p))
    lp = tmp_path / "q.lp"
    lp.write_text("minimize\n obj: x + [ 2 x^2 ] / 2\nsubject to\n c1: x >= 1\nend\n")
    m = dm.from_lp(str(lp))
    r = _solve(m)
    assert r.status == "optimal" and abs(r.objective - 2.0) <= 1e-6


def test_the_differential_compares_optima():
    """CLAUDE.md §6: the round-trip tests above compare objectives only on optimal
    instances; most of their seeds must be optimal or they prove little
    (measured 2026-11-14: 19 of 20, the other infeasible)."""
    optimal = 0
    for seed in range(12):
        optimal += _solve(_random_model(np.random.default_rng(seed), False)).status == "optimal"
    for seed in range(8):
        optimal += _solve(_random_model(np.random.default_rng(100 + seed), True)).status == "optimal"
    assert optimal >= 17, optimal
