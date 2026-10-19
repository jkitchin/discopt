"""#1775 G0.5: the GDP parity panel -- vendored Pyomo examples, suite, oracle table.

Before G0.5 there was no ``gdp_parity`` suite, Pyomo's ``examples/gdp`` models
(which a pip install of Pyomo does not ship) were not available to any benchmark,
the oracle table covered only part of GDPlib, and ``solve_native`` always ran the
default ``gdp_method``.
"""

from __future__ import annotations

import pytest

from benchmarks import gdplib_runner as gr
from benchmarks import pyomo_gdp_examples as px

pyo = pytest.importorskip("pyomo.environ")


def test_every_panel_instance_has_an_oracle_row():
    """Spec G0.5 done-when: a certified value or an explicit 'unproven' per instance."""
    table = gr.panel_oracle()
    assert set(table) == set(gr.GDP_PARITY)
    for name, rec in table.items():
        if rec.source == "unproven":
            assert rec.value is None and rec.note, name
        else:
            assert rec.source in {"baron+scip", "baron", "scip", "highs"}, name
            assert isinstance(rec.value, float), name
    # The private tables name no instance outside the panel (a stale row would be
    # an oracle nothing checks).
    tables = [set(gr._HIGHS_OPTIMA), set(gr._GAMS_OPTIMA), set(gr._UNPROVEN)]
    tables.append(set(gr.reference_optima()) & set(gr.GDP_PARITY))
    for i, a in enumerate(tables):
        assert a <= set(gr.GDP_PARITY)
        for b in tables[i + 1 :]:
            assert not a & b, a & b
    assert all(src in {"baron+scip", "baron", "scip"} for _, src in gr._GAMS_OPTIMA.values())


def test_parity_suite_is_pr58_plus_pyomo_gdpopt_models():
    assert gr.SUITES["gdp_parity"] == gr.GDP_PARITY
    assert len(gr.GDPLIB_PR58) == 17
    pyomo_names = [n for n in gr.GDP_PARITY if n.startswith(px.PREFIX)]
    assert len(pyomo_names) == 7
    assert all(n[len(px.PREFIX) :] in px.CATALOG for n in pyomo_names)


def test_vendored_copy_carries_its_provenance_and_license():
    lic = (px.DATA_DIR / "LICENSE.md").read_text()
    assert "National Technology and Engineering Solutions of" in lic
    assert "Redistributions of source code must retain the above copyright" in lic
    prov = (px.DATA_DIR / "PROVENANCE.md").read_text()
    assert "6.10.0" in prov and "2ebd0873e63d60f563fe30ba72548b366e4976f9" in prov
    for entry in px.CATALOG.values():
        src = (px.DATA_DIR / entry.module).read_text()
        assert "Pyomo: Python Optimization Modeling Objects" in src, entry.module
        assert "3-clause BSD License" in src, entry.module


@pytest.mark.parametrize("name", px.names())
def test_every_catalog_instance_builds_and_converts(name):
    """Each vendored instance builds, is a GDP, and passes evaluation parity."""
    from discopt.pyomo.gdp import check_evaluation_parity, convert_pyomo_gdp
    from pyomo.gdp import Disjunct, Disjunction

    m = px.builder(name)()
    n = sum(
        1
        for _ in m.component_data_objects(
            Disjunction, active=True, descend_into=(pyo.Block, Disjunct)
        )
    )
    assert n > 0, f"{name} has no active disjunction"
    rep = check_evaluation_parity(convert_pyomo_gdp(m), n_points=2)
    assert rep.row_checks > 0 and rep.objective_checks == 2


def test_builders_return_fresh_models():
    a, b = px.builder("pyomo.jobshop_small")(), px.builder("pyomo.jobshop_small")()
    assert a is not b


def test_pyomo_expected_values_name_catalog_instances_only():
    assert set(px.PYOMO_EXPECTED) <= set(px.CATALOG)


@pytest.mark.skipif(not gr.is_available(), reason="needs gdplib (from source)")
def test_default_sweep_lists_no_pyomo_names():
    """The default sweep (``include=None``) is GDPlib's alone. ``include=[]`` -- what
    this used to call -- selects nothing, so it passed whatever the sweep did."""
    default = {s.name for s in gr.discover_models()}
    assert default, "the default sweep found no GDPlib model; the check would be vacuous"
    assert not any(n.startswith(px.PREFIX) for n in default)


def test_discover_resolves_pyomo_names_only_when_asked():
    specs = gr.discover_models(include=["pyomo.eight_process", "pyomo.no_such_model"])
    assert [s.name for s in specs] == ["pyomo.eight_process"]


@pytest.mark.skipif(not gr.is_available(), reason="needs gdplib (from source)")
def test_suite_resolves_every_panel_name(capsys):
    """--suite gdp_parity finds all 24 names, so it does not refuse as shrunken."""
    found = {s.name for s in gr.discover_models(include=list(gr.GDP_PARITY))}
    assert set(gr.GDP_PARITY) == found


def test_solve_native_passes_the_gdp_method_through(monkeypatch):
    from benchmarks import gdplib_native as gn

    seen = {}

    class _Model:
        def solve(self, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop")

    monkeypatch.setitem(gn.NATIVE_BUILDERS, "probe", lambda: _Model())
    gn.solve_native("probe", time_limit=5, gdp_method="hull", gdp_options={"threads": 2})
    assert seen == {"time_limit": 5, "gdp_method": "hull", "threads": 2}
    seen.clear()
    gn.solve_native("probe", time_limit=5)
    assert seen == {"time_limit": 5}


def test_gams_oracle_script_refuses_without_gams(monkeypatch):
    import importlib.util
    from pathlib import Path

    if pyo.SolverFactory("gams").available(exception_flag=False):
        pytest.skip("GAMS is installed here; the refusal path is not reachable")
    path = Path(__file__).resolve().parents[1] / "scripts" / "gdp_parity_oracle.py"
    spec = importlib.util.spec_from_file_location("gdp_parity_oracle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main([]) == 2
