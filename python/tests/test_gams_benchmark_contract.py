"""GAMS link contracts: option forwarding (#1704) and dual-bound export (#1705).

No GAMS installation is needed; GEV/GMO are faked.
"""

import sys
from types import ModuleType, SimpleNamespace

import discopt.modeling as dm
import pytest
from discopt.gams import link


def _fake_gams(monkeypatch, options):
    gev = ModuleType("gams.core.gev")
    gmo = ModuleType("gams.core.gmo")
    gev.gevResLim, gev.gevOptCR, gev.gevOptCA, gev.gevThreads = range(4)
    opts = {
        gev.gevResLim: options[0],
        gev.gevOptCR: options[1],
        gev.gevOptCA: options[2],
        gev.gevThreads: options[3],
    }
    gev.gevGetDblOpt = lambda handle, key: opts[key]
    gev.gevGetIntOpt = lambda handle, key: opts[key]
    gev.new_gevHandle_tp = lambda: object()
    gev.gevCreateD = lambda *a: (True, "")
    gev.gevInitEnvironmentLegacy = lambda *a: None
    gev.gevHandleToPtr = lambda h: h
    gev.gevStatCon = lambda *a: None
    gev.gevStatEOF = lambda *a: None
    gev.gevFree = lambda *a: None
    gev.delete_gevHandle_tp = lambda *a: None
    gmo.new_gmoHandle_tp = lambda: object()
    gmo.gmoCreateD = lambda *a: (True, "")
    gmo.gmoRegisterEnvironment = lambda *a: None
    gmo.gmoLoadDataLegacy = lambda *a: None
    gmo.gmoObjType_Fun = 1
    gmo.gmoObjStyleSet = lambda *a: None
    gmo.gmoIndexBaseSet = lambda *a: None
    gmo.gmoFree = lambda *a: None
    gmo.delete_gmoHandle_tp = lambda *a: None
    package, core = ModuleType("gams"), ModuleType("gams.core")
    package.core, core.gev, core.gmo = core, gev, gmo
    for n, m in (
        ("gams", package),
        ("gams.core", core),
        ("gams.core.gev", gev),
        ("gams.core.gmo", gmo),
    ):
        monkeypatch.setitem(sys.modules, n, m)
    monkeypatch.setattr(link, "_GmoAdapter", lambda *a: object())
    monkeypatch.setattr(link, "_write_solution", lambda *a: None)


def test_control_file_forwards_gams_resource_limit_and_relative_gap(monkeypatch, tmp_path):
    _fake_gams(monkeypatch, (0.25, 1e-6, 0, 1))
    observed = {}

    def solve(view, *, time_limit=1e10, gap=1e-4, **kwargs):
        observed.update(time_limit=time_limit, gap=gap, **kwargs)
        return object(), object()

    monkeypatch.setattr(link, "solve_view", solve)
    assert link.solve_from_control_file("unused", sysdir=str(tmp_path)) == 0
    # optca=0 -> None (discopt's positive default), threads forwarded.
    assert observed == {"time_limit": 0.25, "gap": 1e-6, "abs_gap": None, "threads": 1}


def test_positive_optca_and_threads_forwarded():
    gev = SimpleNamespace(gevResLim=0, gevOptCR=1, gevOptCA=2, gevThreads=3)
    vals = {0: 5.0, 1: 1e-3, 2: 1e-5, 3: 4}
    gev.gevGetDblOpt = lambda h, k: vals[k]
    gev.gevGetIntOpt = lambda h, k: vals[k]
    assert link._read_gams_options(gev, None) == {
        "time_limit": 5.0,
        "gap": 1e-3,
        "abs_gap": 1e-5,
        "threads": 4,
    }


@pytest.mark.parametrize(
    "vals",
    [
        (0.0, 1e-4, 0, 1),
        (10, -1.0, 0, 1),
        (10, 1e-4, -1.0, 1),
        (10, 1e-4, 0, -2),
        (10, float("nan"), 0, 1),
    ],
)
def test_unsupported_options_rejected(vals):
    gev = SimpleNamespace(gevResLim=0, gevOptCR=1, gevOptCA=2, gevThreads=3)
    d = dict(enumerate(vals))
    gev.gevGetDblOpt = lambda h, k: d[k]
    gev.gevGetIntOpt = lambda h, k: d[k]
    with pytest.raises(ValueError):
        link._read_gams_options(gev, None)


def _export(model, result):
    exported = {}
    gmo = SimpleNamespace(
        gmoHobjval="objval",
        gmoHobjest="objest",
        gmoModelStatSet=lambda *a: None,
        gmoSolveStatSet=lambda *a: None,
        gmoSetVarLOne=lambda *a: None,
        gmoSetHeadnTail=lambda h, f, v: exported.__setitem__(f, v),
        gmoUnloadSolutionLegacy=lambda *a: None,
    )
    link._write_solution(object(), gmo, model, result)
    return exported


@pytest.mark.parametrize("sense,incumbent,bound", [("minimize", 5.0, 2.0), ("maximize", 2.0, 5.0)])
def test_valid_global_dual_bound_is_exported_even_without_closed_gap(sense, incumbent, bound):
    model = dm.Model("gams-bound")
    x = model.continuous("x", lb=0, ub=10)
    getattr(model, sense)(x)
    result = dm.SolveResult(
        status="feasible",
        objective=incumbent,
        bound=bound,
        bound_valid=True,
        gap_certified=False,
        x={"x": incumbent},
    )
    exported = _export(model, result)
    assert exported["objval"] == incumbent
    assert exported.get("objest") == bound


def test_invalid_dual_bound_is_not_exported():
    model = dm.Model("gams-untrusted-bound")
    x = model.continuous("x", lb=0, ub=10)
    model.minimize(x)
    result = dm.SolveResult(
        status="feasible",
        objective=5.0,
        bound=2.0,
        bound_valid=False,
        gap_certified=False,
        x={"x": 5.0},
    )
    exported = _export(model, result)
    assert exported["objval"] == 5.0
    assert "objest" not in exported


def test_valid_bound_without_incumbent_is_exported():
    model = dm.Model("gams-nosol")
    x = model.continuous("x", lb=0, ub=10)
    model.minimize(x)
    result = dm.SolveResult(
        status="time_limit",
        objective=None,
        bound=1.5,
        bound_valid=True,
        gap_certified=False,
        x=None,
    )
    exported = _export(model, result)
    assert exported == {"objest": 1.5}
