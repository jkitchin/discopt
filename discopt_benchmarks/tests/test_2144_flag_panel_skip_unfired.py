"""#2144: the faster GDP flag panel (``--skip-unfired``, ``--jobs``) and the
gdplib-free resolution of the vendored panel instances.

``--skip-unfired`` stops a cell after its first ON solve when the flag's
process-wide counter reads 0, because OFF then runs the same code. These tests pin
the two ways that could go wrong: a cell skipped on a solve that did not complete,
and a skipped or incomplete cell being read as a comparison.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
KEY = "gdp_config/feas_sampled_process"


def _panel():
    import sys

    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "_panel2144", SCRIPTS / "issue1917_gdp_flag_panel.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(arm, rep=0, acted=0, **kw):
    r = {
        "phase": "solve",
        "flag": "DISCOPT_GDP_CONFIG_FEAS_SAMPLE",
        "instance": "pyomo.probe",
        "method": "hull",
        "arm": arm,
        "rep": rep,
        "status": "optimal",
        "objective": 1.0,
        "bound": 1.0,
        "wall": 1.0,
        "node_count": 3,
        "verdict": "solved",
        "stats": {KEY: acted},
    }
    r.update(kw)
    return r


@pytest.mark.smoke
def test_flag_reads_the_process_wide_counter():
    p = _panel()
    assert p._FIRE_STATS["DISCOPT_GDP_CONFIG_FEAS_SAMPLE"] == KEY


@pytest.mark.smoke
def test_unfired_needs_a_completed_on_solve_with_a_zero_counter():
    p = _panel()
    assert p._unfired(_row("on"), KEY)
    assert not p._unfired(_row("on", acted=3), KEY)
    assert not p._unfired(_row("off"), KEY)
    assert not p._unfired(_row("on", status="killed"), KEY)
    assert not p._unfired(_row("on", error="boom"), KEY)
    missing = _row("on")
    missing["stats"] = {}
    assert not p._unfired(missing, KEY), "an absent counter is not a zero counter"


def _report(p, tmp_path, rows, monkeypatch):
    out = tmp_path / "rows.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(p, "_ORACLE", {})

    class A:
        flag = "DISCOPT_GDP_CONFIG_FEAS_SAMPLE"
        time_limit = 120.0

    A.out = out
    return p.report(A)


@pytest.mark.smoke
def test_report_lists_a_skipped_cell_without_comparing_it(tmp_path, monkeypatch, capsys):
    p = _panel()
    rows = [_row("on")]  # an unfired cell: one ON solve, counter 0, no OFF arm
    rows += [_row("off", rep=k, instance="pyomo.acts") for k in range(2)]
    rows += [_row("on", rep=k, acted=2, instance="pyomo.acts") for k in range(2)]
    assert _report(p, tmp_path, rows, monkeypatch) == 0
    text = capsys.readouterr().out
    assert "skipped" in text and "('pyomo.probe', 'hull')" in text
    assert "solved-gained=0" in text, "a skipped cell was read as an ON-only gain"


@pytest.mark.smoke
def test_report_fails_on_an_incomplete_cell(tmp_path, monkeypatch, capsys):
    p = _panel()
    # ON acted but no OFF row exists: an interrupted run, not a comparison.
    assert _report(p, tmp_path, [_row("on", acted=5)], monkeypatch) == 1
    assert "INCOMPLETE" in capsys.readouterr().out


@pytest.mark.smoke
def test_vendored_instances_resolve_without_gdplib(monkeypatch):
    pytest.importorskip("pyomo.environ")
    from benchmarks import gdplib_runner as gr

    real_import = builtins.__import__

    def _no_gdplib(name, *a, **kw):
        if name == "gdplib" or name.startswith("gdplib."):
            raise ModuleNotFoundError("No module named 'gdplib'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_gdplib)
    specs = gr.discover_models(include=["pyomo.eight_process", "pyomo.jobshop_small"])
    assert [s.name for s in specs] == ["pyomo.eight_process", "pyomo.jobshop_small"]
    with pytest.raises(ModuleNotFoundError):
        gr.discover_models(include=["batch_processing"])
