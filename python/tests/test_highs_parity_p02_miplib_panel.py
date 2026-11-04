"""highs-parity P0.2: the MIPLIB-38 panel harness and its checked-in manifest.

``discopt_benchmarks/scripts/miplib_panel.py`` rebuilds the 38-instance MIPLIB 2017 panel
of ``milp-competitiveness-plan.md`` §0 (its old harness was never checked in). These pin,
without downloading or solving anything:

* the panel is exactly the 38 recovered names, and the checked-in manifest holds each of
  them once, validated against the ``.solu`` ``=opt=`` value, with a recorded sha256;
* a fetched file whose sha256 differs from the recorded one is refused;
* the screen tells an unvalidated instance (array solve hit its limit) from a real oracle
  mismatch, and accepts a reproduction within the panel tolerance;
* ``.solu`` parsing, and that ``lp_milp_panel run`` accepts ``--h0-subprocess`` (every arm
  in a fresh process).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "discopt_benchmarks" / "scripts"
_PANEL_JSON = _ROOT / "discopt_benchmarks" / "data" / "miplib" / "panel.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_p02", _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mp():
    pytest.importorskip("highspy")
    sys.path.insert(0, str(_SCRIPTS))
    try:
        return _load("miplib_panel")
    finally:
        sys.path.remove(str(_SCRIPTS))


def test_panel_is_the_38_recovered_names(mp):
    assert len(mp.PANEL) == 38
    assert len(set(mp.PANEL)) == 38


def test_manifest_holds_every_panel_instance_once_validated(mp):
    man = json.loads(_PANEL_JSON.read_text())
    names = [m["name"] for m in man["members"]]
    assert sorted(names) == sorted(mp.PANEL)
    assert man["excluded"] == []
    shas = man["source"]["sha256"]
    assert "miplib2017-v31.solu" in shas
    for m in man["members"]:
        assert m["panel"] == mp.PANEL_LABEL
        assert m["path"] == f"ref/miplib2017/{m['name']}.mps.gz"
        assert len(shas[f"{m['name']}.mps.gz"]) == 64
        assert m["oracle_status"] == "Optimal"
        assert m["oracle_source"] == "miplib2017-v31.solu =opt="
        assert m["n_int"] > 0
        ref, h0 = m["oracle_objective"], m["validation_h0_objective"]
        assert abs(h0 - ref) <= 1e-6 * (1 + abs(ref)) + 1e-4 * abs(ref), m["name"]


def test_sha_mismatch_is_refused(mp):
    mp.check_sha("x", None, "a" * 64)  # first fetch records
    mp.check_sha("x", "a" * 64, "a" * 64)
    with pytest.raises(RuntimeError, match="does not match"):
        mp.check_sha("x", "a" * 64, "b" * 64)


def test_screen_separates_unvalidated_from_mismatch(mp):
    assert mp.oracle_mismatch({"status": "Optimal", "objective": 255.0}, 255.0, 1e-4) is None
    assert mp.oracle_mismatch({"status": "Optimal", "objective": 255.02}, 255.0, 1e-4) is None
    why = mp.oracle_mismatch({"status": "Optimal", "objective": 256.0}, 255.0, 1e-4)
    assert why.startswith("array objective")
    why = mp.oracle_mismatch({"status": "Time limit reached", "objective": 260.0}, 255.0, 1e-4)
    assert why.startswith("unvalidated")


def test_parse_solu(mp, tmp_path):
    f = tmp_path / "x.solu"
    f.write_text("=opt=  gr4x6  202.35\n=inf=  bad\n=best= open  -3.5\n")
    assert mp.parse_solu(f) == {
        "gr4x6": ("=opt=", 202.35),
        "bad": ("=inf=", None),
        "open": ("=best=", -3.5),
    }


def test_lp_milp_panel_run_takes_h0_subprocess(mp, monkeypatch):
    seen = {}
    monkeypatch.setattr(mp.P, "cmd_run", lambda a: seen.update(vars(a)) or 0)
    assert mp.P.main(["run", "--manifest", "m.json", "--h0-subprocess"]) == 0
    assert seen["h0_subprocess"] is True
    # The MIPLIB runner always asks for it.
    seen.clear()
    assert mp.main(["run", "--instances", "gr4x6"]) == 0
    assert seen["h0_subprocess"] is True and seen["instances"] == "gr4x6"


def _cell(statuses, certs, walls, nodes, wrong=()):
    return {"status_all": statuses, "certified_all": certs, "wall_med": walls,
            "nodes_all": nodes, "incorrect": list(wrong)}  # fmt: skip


def test_panel_report_summarises_by_panel_and_arm():
    sys.path.insert(0, str(_SCRIPTS))
    try:
        rep = _load("panel_report")
    finally:
        sys.path.remove(str(_SCRIPTS))
    run = {
        "complete": True,
        "args": {"arms": "R,H0"},
        "rows": [
            {"name": "a", "panel": "M", "oracle_status": "Optimal", "wall_valid": True,
             "R": _cell(["optimal"] * 3, [True] * 3, 1.0, [10, 20, 30]),
             "H0": _cell(["optimal"] * 3, [None] * 3, 0.1, [1, 1, 1])},
            {"name": "b", "panel": "M", "oracle_status": "Optimal", "wall_valid": False,
             "R": _cell(["optimal", "time_limit", "optimal"], [True, False, True], 20.0,
                        [5, 9, 7], wrong=["x"]),
             "H0": _cell(["optimal"] * 3, [None] * 3, 0.3, [2, 2, 2])},
            {"name": "c", "panel": "INF", "oracle_status": "Infeasible", "wall_valid": True,
             "R": _cell(["infeasible"] * 3, [True] * 3, 0.5, [0, 0, 0]),
             "H0": _cell(["Infeasible"] * 3, [None] * 3, 0.01, [0, 0, 0])},
        ],
    }  # fmt: skip
    lines, cells = rep.summarise([run])
    assert cells == 6
    by = {(ln["panel"], ln["arm"]): ln for ln in lines}
    r = by[("M", "R")]
    assert (r["n"], r["solved"], r["certified"], r["incorrect"]) == (2, 1, 1, 1)
    assert r["total_wall"] == 21.0 and r["median_solved"] == 1.0
    assert r["nodes"] == 20 + 7 and r["timing_valid"] == 1
    assert by[("M", "H0")]["certified"] is None and by[("M", "H0")]["solved"] == 2
    assert by[("INF", "R")]["solved"] == 1 and by[("INF", "H0")]["solved"] == 1
    with pytest.raises(SystemExit, match="incomplete"):
        rep.summarise([{**run, "complete": False}])
