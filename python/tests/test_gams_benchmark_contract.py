"""GAMS link contracts: option forwarding (#1704) and dual-bound export (#1705).

No GAMS installation is needed; GEV/GMO are faked.  The fakes mirror the real
gamsapi shapes: ``gevResLim``/``gevOptCR``/``gevOptCA`` are option-name strings
read via ``gevGetDblOpt``, while ``gevThreads`` is a *function* of the handle
returning the resolved thread count (a constant-style fake hid exactly that).
"""

import sys
from types import ModuleType, SimpleNamespace

import discopt.modeling as dm
import pytest
from discopt.gams import link


def _fake_gams(monkeypatch, options):
    gev = ModuleType("gams.core.gev")
    gmo = ModuleType("gams.core.gmo")
    _install_gev_options(gev, options)
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
    monkeypatch.setattr(link, "_write_solution", lambda *a, **k: None)


def _install_gev_options(gev, options):
    gev.gevResLim, gev.gevOptCR, gev.gevOptCA = "ResLim", "OptCR", "OptCA"
    dbl = dict(zip(("ResLim", "OptCR", "OptCA"), options[:3]))
    gev.gevGetDblOpt = lambda handle, name: dbl[name]
    gev.gevThreads = lambda handle: options[3]


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
    gev = SimpleNamespace()
    _install_gev_options(gev, (5.0, 1e-3, 1e-5, 4))
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
        (10, 1e-4, 0, 0),
        (10, float("nan"), 0, 1),
    ],
)
def test_unsupported_options_rejected(vals):
    gev = SimpleNamespace()
    _install_gev_options(gev, vals)
    with pytest.raises(ValueError):
        link._read_gams_options(gev, None)


def _export(model, result, **kwargs):
    exported = {}
    gmo = SimpleNamespace(
        gmoHobjval="objval",
        gmoTmipbest="objest",  # the real tail GAMS reports as objest
        gmoHresused="resused",
        gmoModelStatSet=lambda *a: None,
        gmoSolveStatSet=lambda *a: None,
        gmoSetVarLOne=lambda *a: None,
        gmoSetHeadnTail=lambda h, f, v: exported.__setitem__(f, v),
        gmoUnloadSolutionLegacy=lambda *a: None,
    )
    link._write_solution(object(), gmo, model, result, **kwargs)
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
    assert exported == {"objest": 1.5, "resused": 0.0}


@pytest.mark.parametrize("kwargs,expected", [({"resused": 4.25}, 4.25), ({}, 3.5)])
def test_resource_usage_is_exported(kwargs, expected):
    # The link's own timing wins; without it, the solver's wall_time is reported.
    model = dm.Model("gams-resused")
    x = model.continuous("x", lb=0, ub=10)
    model.minimize(x)
    result = dm.SolveResult(status="time_limit", objective=None, x=None, wall_time=3.5)
    assert _export(model, result, **kwargs)["resused"] == expected
