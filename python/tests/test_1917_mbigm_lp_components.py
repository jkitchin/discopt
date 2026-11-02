"""#1917 stages B-D: what mbigm's big-M LPs cost, and the two changes that cut it.

Measured on GDPlib ``biofuel`` at 6d3e2f79: mbigm lowering ran past 600 s
(9098 distinct LPs, 388 ms and 1178 pivots each). Two changes:

* ``_lp_box_memo`` (bound-neutral, always on): the LP box was rebuilt per row,
  78 % of the profiled lowering. It is now rebuilt only when a variable object or
  a bound array has been replaced, so every M is byte-identical to the old path.
* ``DISCOPT_MBIGM_LP_COMPONENTS`` (bound-changing, default OFF): solve each LP
  over the connected components of the column-row graph that the body touches.
  Dropping rows relaxes the LP, so an M can only grow; on a separable polytope
  it is unchanged.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import gdp_reformulate as G
from discopt.solver_tuning import SolverTuning


def _separable():
    """Two blocks, x-y and z-w, linked by no linear row."""
    m = dm.Model("sep")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=-3.0, ub=4.0)
    z = m.continuous("z", shape=(2,), lb=[0.0, -1.0], ub=[2.0, 5.0])
    m.subject_to(x + 2.0 * y <= 8.0)
    m.subject_to(3.0 * x - y >= -1.0)
    m.subject_to(z[0] + z[1] == 1.5)
    m.minimize(x + z[0])
    return m, x, y, z


def _linked():
    """The same blocks plus one row that links them: the row tightens x."""
    m, x, y, z = _separable()
    m.subject_to(x + z[0] <= 1.0)
    return m, x, y, z


def _bodies(x, y, z):
    return [
        x <= 1.0,
        x + y >= 0.5,
        2.0 * y - x == 0.0,
        z[0] - z[1] <= 0.0,
        x - z[1] >= -2.0,  # touches both blocks
    ]


def _M(con, m, flag, monkeypatch):
    monkeypatch.setenv("DISCOPT_MBIGM_LP_COMPONENTS", "1" if flag else "0")
    return G._compute_big_m_lp(con, m, G._precompute_lp_relaxation(m))


def _oracle():
    from discopt.solvers.lp_backend import get_exact_lp_solver

    if get_exact_lp_solver() is None:
        pytest.skip("no exact LP oracle")


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("DISCOPT_MBIGM_LP_COMPONENTS", raising=False)
    assert SolverTuning().mbigm_lp_components is False
    assert G._mbigm_lp_components_enabled() is False


def test_box_memo_is_the_box_and_rebuilds_on_a_rebound(monkeypatch):
    m, x, y, z = _separable()
    data = G._precompute_lp_relaxation(m)
    n = data[4]
    box, key = G._lp_box_memo(data, m, n)
    assert box == G._lp_box(m, n)
    assert key == np.asarray(G._lp_box(m, n), dtype=np.float64).tobytes()
    # Cached: the same list object comes back while nothing was rebound.
    again, key2 = G._lp_box_memo(data, m, n)
    assert again is box and key2 == key
    # A rebound bound must be seen, and give the box _lp_box gives.
    y.ub = np.asarray(2.0)
    rebuilt, key3 = G._lp_box_memo(data, m, n)
    assert rebuilt is not box and rebuilt == G._lp_box(m, n)
    assert rebuilt[1] == (-3.0, 2.0) and key3 != key
    # Appended variables (mbigm's selectors) leave the first n columns alone.
    m.binary("sel")
    after, _ = G._lp_box_memo(data, m, n)
    assert after is rebuilt and len(after) == n


def test_box_memo_keeps_every_M_byte_identical(monkeypatch):
    """The bound-neutral claim on its own: memoized vs. a fresh box per call."""
    _oracle()
    monkeypatch.setenv("DISCOPT_MBIGM_LP_COMPONENTS", "0")
    m, x, y, z = _linked()
    memo = G._precompute_lp_relaxation(m)
    with_memo = [G._compute_big_m_lp(c, m, memo) for c in _bodies(x, y, z)]
    real = G._lp_box_memo
    monkeypatch.setattr(
        G,
        "_lp_box_memo",
        lambda d, mm, n: (
            G._lp_box(mm, n),
            np.asarray(G._lp_box(mm, n), dtype=np.float64).tobytes(),
        ),
    )
    fresh = G._precompute_lp_relaxation(m)
    without = [G._compute_big_m_lp(c, m, fresh) for c in _bodies(x, y, z)]
    monkeypatch.setattr(G, "_lp_box_memo", real)
    assert len(with_memo) == 5
    assert [float(a).hex() for a in with_memo] == [float(b).hex() for b in without]


def test_components_partition_the_polytope():
    m, *_ = _separable()
    data = G._precompute_lp_relaxation(m)
    comp = G._LPComponents(data[0], data[2], data[4])
    # Columns x, y | z0, z1.
    assert comp.col[0] == comp.col[1] != comp.col[2] == comp.col[3]
    # Each row sits in the component of its columns.
    assert set(comp.ub) == {comp.col[0]}
    assert set(comp.eq) == {comp.col[2]}
    sub = G._component_sub_lp(data, np.array([1.0, 0.0, 0.0, 0.0]), G._lp_box(m, 4))
    assert sub[0].shape == (2,) and sub[1].shape == (2, 2) and sub[3].shape == (0, 2)
    assert sub[5] == [(0.0, 10.0), (-3.0, 4.0)]
    # A zero objective solves the full LP.
    full = G._component_sub_lp(data, np.zeros(4), G._lp_box(m, 4))
    assert full[1] is data[0] and full[3] is data[2]
    # So does an objective whose components cover every column.
    every = G._component_sub_lp(data, np.array([1.0, 0.0, 1.0, 0.0]), G._lp_box(m, 4))
    assert every[1] is data[0] and every[3] is data[2] and len(every[5]) == 4


@pytest.mark.parametrize("build", [_separable, _linked], ids=["separable", "linked"])
def test_component_M_is_never_smaller(build, monkeypatch):
    """The differential test: ON >= OFF on every body; equal when separable."""
    _oracle()
    checked = 0
    m, x, y, z = build()
    for con in _bodies(x, y, z):
        off = _M(con, m, False, monkeypatch)
        on = _M(con, m, True, monkeypatch)
        assert on >= off * (1 - 1e-9) - 1e-9, (con, on, off)
        if build is _separable:
            assert on == pytest.approx(off, rel=1e-9, abs=1e-9), (con, on, off)
        checked += 1
    assert checked == 5


def test_a_linking_row_is_kept(monkeypatch):
    """On ``_linked`` the x-z row bounds x by 1; it joins x and z in one
    component, so the restricted LP still holds it and the M stays tight."""
    _oracle()
    m, x, y, z = _linked()
    off = _M(x <= 0.0, m, False, monkeypatch)
    on = _M(x <= 0.0, m, True, monkeypatch)
    assert off == pytest.approx(1.01)
    assert on == pytest.approx(1.01)  # x and z share a component through the row
    # A body over y alone touches the x-y-z component too: still equal.
    assert _M(y <= 0.0, m, True, monkeypatch) == pytest.approx(_M(y <= 0.0, m, False, monkeypatch))


def test_memo_key_separates_the_arms(monkeypatch):
    """One lp_data, both arms: the OFF answer is never served to ON, or back."""
    from discopt.solvers import lp_backend

    _oracle()
    m, x, y, z = _separable()
    data = G._precompute_lp_relaxation(m)
    real = lp_backend.get_exact_lp_solver()
    widths = []

    def counting(**kw):
        widths.append(len(kw["c"]))
        return real(**kw)

    monkeypatch.setattr(lp_backend, "get_exact_lp_solver", lambda: counting)
    monkeypatch.setenv("DISCOPT_MBIGM_LP_COMPONENTS", "0")
    G._compute_big_m_lp(x <= 1.0, m, data)
    monkeypatch.setenv("DISCOPT_MBIGM_LP_COMPONENTS", "1")
    G._compute_big_m_lp(x <= 1.0, m, data)
    G._compute_big_m_lp(x <= 3.0, m, data)  # memo hit within the ON arm
    assert widths == [4, 2]


def test_lowered_model_keeps_its_optimum(monkeypatch):
    """Feasible points are not cut: the mbigm optimum is the same with the flag ON."""
    _oracle()
    from discopt.transformations import create_using

    def build():
        m = dm.Model("disj")
        x = m.continuous("x", lb=0.0, ub=10.0)
        y = m.continuous("y", lb=0.0, ub=10.0)
        z = m.continuous("z", lb=0.0, ub=4.0)
        m.subject_to(x + y <= 12.0)
        m.subject_to(z <= 3.0)
        m.either_or([[x <= 2.0, y >= 6.0], [x >= 5.0, y <= 3.0]], name="d")
        m.either_or([[z <= 1.0], [z >= 2.5]], name="e")
        m.minimize((x - 4.0) ** 2 + (y - 4.0) ** 2 + (z - 1.8) ** 2)
        return m

    objs = []
    for flag in ("0", "1"):
        monkeypatch.setenv("DISCOPT_MBIGM_LP_COMPONENTS", flag)
        monkeypatch.setenv("DISCOPT_GDP_BOUND_PRETRANSFORM", "0")
        res = create_using("gdp.mbigm", build()).solve(time_limit=60)
        assert res.status == "optimal"
        objs.append(res.objective)
    # Reference by enumeration of the four disjunct pairs, on the box.
    assert objs[0] == pytest.approx(objs[1], abs=1e-6)
    assert objs[0] == pytest.approx(min((5 - 4) ** 2 + 1, 4 + 4) + 0.49, abs=1e-5)
