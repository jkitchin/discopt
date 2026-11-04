"""highs-parity P0.4: the master-parity panel's own checks (no solve).

``discopt_benchmarks/scripts/master_parity_panel.py`` replays captured master MILPs through
HiGHS and the Rust simplex. These pin what makes its verdicts trustworthy:

* every checked-in master fixture converts to ``solve_milp`` keywords of a consistent shape
  (the three ``.npz`` layouts and the ``.mps.gz``), with each source's row count;
* the script's own point check accepts the fac2 witness ``z_star`` (verified feasible, the
  README's objective) and rejects a perturbed point, an integrality break and a wrong length;
* the cross-check flags a bound above a verified incumbent and two "optimal" answers farther
  apart than the gap, and accepts agreeing engines;
* ``DISCOPT_DECOMP_MASTER`` forces the decomposition master engine and refuses a typo.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "discopt_benchmarks" / "scripts" / "master_parity_panel.py"
_DATA = _ROOT / "python" / "tests" / "data"


@pytest.fixture(scope="module")
def mp():
    pytest.importorskip("highspy")
    spec = importlib.util.spec_from_file_location("master_parity_panel_p04", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: (rows in A_ub, rows in A_eq, columns, integers), read off the converted fixtures; the
#: oa_masters totals equal the README's "rows x cols" (tls2 52 x 37, rsyn0830m 915 x 250,
#: fac2 418 x 67).
SHAPES = {
    "oa_masters/tls2_master0.npz": (46, 6, 37, 33),
    "oa_masters/rsyn0830m_master0.npz": (830, 85, 250, 94),
    "oa_masters/fac2_master0.npz": (397, 21, 67, 12),
    "oa_masters/m6_final_master_2115.npz": (258, 0, 86, 30),
    "issue2069/clay0203_master.mps.gz": (525, 6, 30, 18),
    "milp_default_box_clay0203_master.npz": (537, 0, 30, 18),
    "issue2107/clay0204hfsg_master.npz": (709, 42, 164, 32),
    "issue2107/rsyn0820m_master.npz": (390, 78, 215, 84),
}


def test_every_fixture_is_listed(mp):
    assert set(mp.FIXTURES) == set(SHAPES)


@pytest.mark.parametrize("rel", sorted(SHAPES))
def test_fixture_converts_with_its_shape(mp, rel):
    kw = mp.fixture_kw(_DATA / rel)
    m_ub, m_eq, n, n_int = SHAPES[rel]
    assert len(kw["c"]) == n and len(kw["bounds"]) == n and len(kw["integrality"]) == n
    assert int(np.sum(kw["integrality"])) == n_int
    assert kw["A_ub"].shape == (m_ub, n) and len(kw["b_ub"]) == m_ub
    if m_eq:
        assert kw["A_eq"].shape == (m_eq, n) and len(kw["b_eq"]) == m_eq
    else:
        assert kw.get("A_eq") is None


def test_point_check_accepts_the_fac2_witness_and_rejects_breaks(mp):
    d = np.load(_DATA / "oa_masters" / "fac2_master0.npz")
    kw = mp.fixture_kw(_DATA / "oa_masters" / "fac2_master0.npz")
    z = np.asarray(d["z_star"], float)
    ok, obj, viol = mp.check_point(kw, z)
    assert ok and viol <= mp.FEAS_TOL
    assert obj == pytest.approx(float(d["z_star_objective"]), abs=1e-6)
    j = int(np.flatnonzero(np.asarray(kw["integrality"], bool))[0])
    z_frac = z.copy()
    z_frac[j] += 0.5
    assert not mp.check_point(kw, z_frac)[0]
    assert not mp.check_point(kw, z * 1.5 + 1.0)[0]
    assert not mp.check_point(kw, z[:-1])[0]
    assert not mp.check_point(kw, None)[0]


def _rec(h, s):
    return {"highs": h, "simplex": s}


def test_cross_check(mp):
    agree = _rec(
        {"status": "SolveStatus.OPTIMAL", "obj": 10.0, "bound": 10.0, "verified": True},
        {"status": "SolveStatus.OPTIMAL", "obj": 10.0005, "bound": 9.9995, "verified": True},
    )
    assert mp.cross_check(agree, 1e-4) == []
    false_bound = _rec(
        {"status": "SolveStatus.OPTIMAL", "obj": 10.0, "bound": 10.0, "verified": True},
        {"status": "SolveStatus.TIME_LIMIT", "obj": None, "bound": 10.5, "verified": False},
    )
    assert any("above" in w for w in mp.cross_check(false_bound, 1e-4))
    far = _rec(
        {"status": "SolveStatus.OPTIMAL", "obj": 10.0, "bound": 9.0, "verified": False},
        {"status": "SolveStatus.OPTIMAL", "obj": 11.0, "bound": 9.0, "verified": False},
    )
    assert any("both optimal" in w for w in mp.cross_check(far, 1e-4))
    # One engine proves infeasible while the other returns a point the script verified.
    split = _rec(
        {"status": "SolveStatus.INFEASIBLE", "obj": None, "bound": None, "verified": False},
        {"status": "SolveStatus.OPTIMAL", "obj": 3.0, "bound": 3.0, "verified": True},
    )
    assert any("infeasible" in w for w in mp.cross_check(split, 1e-4))
    both_inf = _rec(
        {"status": "SolveStatus.INFEASIBLE", "obj": None, "bound": None, "verified": False},
        {"status": "SolveStatus.INFEASIBLE", "obj": None, "bound": None, "verified": False},
    )
    assert mp.cross_check(both_inf, 1e-4) == []
    unb = _rec(
        {"status": "SolveStatus.UNBOUNDED", "obj": None, "bound": None, "verified": False},
        {"status": "SolveStatus.OPTIMAL", "obj": 3.0, "bound": 3.0, "verified": True},
    )
    assert any("unbounded" in w for w in mp.cross_check(unb, 1e-4))


def test_decomp_master_env_forces_the_engine(monkeypatch):
    from discopt.solvers import lp_backend

    monkeypatch.setenv("DISCOPT_DECOMP_MASTER", "simplex")
    assert lp_backend.get_decomposition_master_solver()[1] == "the discopt simplex B&B"
    monkeypatch.setenv("DISCOPT_DECOMP_MASTER", "highs")
    assert lp_backend.get_decomposition_master_solver()[1] == "HiGHS"
    monkeypatch.delenv("DISCOPT_DECOMP_MASTER")
    assert lp_backend.get_decomposition_master_solver()[1] == "HiGHS"  # default unchanged
    monkeypatch.setenv("DISCOPT_DECOMP_MASTER", "higs")
    with pytest.raises(ValueError, match="DISCOPT_DECOMP_MASTER"):
        lp_backend.get_decomposition_master_solver()


def test_vendored_gdp_models_resolve_without_gdplib(monkeypatch):
    # P0.4's GDP capture: the vendored Pyomo GDPopt models need no GDPlib, but
    # discover_models imported it unconditionally, so with GDPlib absent the 7 vendored
    # masters could not be captured either. A request needing GDPlib still raises.
    pytest.importorskip("pyomo.environ")
    import sys

    monkeypatch.syspath_prepend(str(_ROOT / "discopt_benchmarks"))
    monkeypatch.setitem(sys.modules, "gdplib", None)  # makes ``import gdplib`` fail
    from benchmarks import gdplib_runner as gr

    specs = gr.discover_models(include=["pyomo.eight_process", "pyomo.jobshop_small"])
    assert [s.name for s in specs] == ["pyomo.eight_process", "pyomo.jobshop_small"]
    with pytest.raises(ImportError):
        gr.discover_models(include=["pyomo.eight_process", "biofuel"])
    with pytest.raises(ImportError):
        gr.discover_models()
