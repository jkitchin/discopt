"""#2006: the scip_wins screen's verdict must catch primal-side soundness failures.

Before #2006 ``verdict`` returned ``WRONG`` only for a claimed optimum away from
the oracle or a claim on a proven-infeasible instance. A time-limited incumbent
strictly better than the proven optimum -- which must be an infeasible point --
was labelled ``FAIL``, the same as "did not find it", and ``analyze`` exited 0.
Incumbents were never checked for feasibility at all. These tests pin the new
classification and the end-to-end exit code.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

# Loaded by path: this module runs in python-fast beside python/tests, where a
# bare ``scripts`` package name is not guaranteed to resolve to this directory.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "scip_wins_screen.py"
_spec = importlib.util.spec_from_file_location("scip_wins_screen", _SCRIPT)
screen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(screen)
_verify_incumbent, analyze, verdict = screen._verify_incumbent, screen.analyze, screen.verdict

NL_DIR = Path(__file__).resolve().parents[2] / "python" / "tests" / "data" / "minlplib_nl"
OPT = 7.6672  # ex1221, MINLPLib primalbound


def _ok(**kw):
    return {"status": "time_limit", "verified": True, **kw}


@pytest.mark.parametrize(
    "status,obj,sense,opt",
    [
        ("time_limit", 4.0, "min", OPT),  # below a proven minimum
        ("feasible", 4.0, "min", OPT),
        ("time_limit", 9.0, "max", OPT),  # above a proven maximum
    ],
)
@pytest.mark.unit
def test_incumbent_beating_proven_optimum_is_wrong(status, obj, sense, opt):
    # Was FAIL: indistinguishable from "did not find the optimum".
    res = {"status": status, "objective": obj, "verified": True, "verified_obj": obj}
    assert verdict(res, opt, sense) == "WRONG_BEATS_OPT"


@pytest.mark.unit
def test_suboptimal_incumbent_is_still_fail_not_wrong():
    # An honest gap must not be reported as a soundness failure.
    assert verdict(_ok(objective=9.0, verified_obj=9.0), OPT, "min") == "FAIL"
    assert verdict(_ok(objective=4.0, verified_obj=4.0), OPT, "max") == "FAIL"
    assert verdict(_ok(objective=OPT, verified_obj=OPT), OPT, "min") == "FOUND_UNPROVEN"


@pytest.mark.unit
def test_unverified_point_is_wrong_at_any_status():
    for st in ("optimal", "time_limit", "feasible"):
        res = {"status": st, "objective": OPT, "verified": False, "verified_obj": None}
        assert verdict(res, OPT, "min") == "WRONG_INFEASIBLE_POINT"
    assert verdict({"status": "optimal", "verify_error": "KeyError"}, OPT) == "WRONG_UNVERIFIABLE"


@pytest.mark.unit
def test_reported_objective_not_the_points_own_is_wrong():
    res = {"status": "optimal", "objective": OPT, "verified": True, "verified_obj": OPT + 1.0}
    assert verdict(res, OPT, "min") == "WRONG_OBJECTIVE"


@pytest.mark.unit
def test_scip_rows_without_verification_fields_classify_as_before():
    assert verdict({"status": "optimal", "objective": OPT}, OPT, "min") == "OK"
    assert verdict({"status": "gaplimit", "objective": OPT * 1.5}, OPT, "min") == "WRONG"
    assert verdict({"status": "infeasible", "objective": None}, None) == "OK"


def test_verify_incumbent_accepts_a_solved_point_and_rejects_a_perturbed_one():
    import discopt.modeling as dm

    nl = str(NL_DIR / "ex1221.nl")
    r = dm.from_nl(nl).solve(time_limit=30)
    assert r.status == "optimal" and r.x is not None
    good = _verify_incumbent(nl, r.x)
    assert good["verified"] is True, good
    assert abs(good["verified_obj"] - OPT) <= 1e-3

    bad_x = {k: v + 1e3 for k, v in r.x.items()}  # far outside every box
    bad = _verify_incumbent(nl, bad_x)
    assert bad["verified"] is False and bad["verify_reason"]

    missing = _verify_incumbent(nl, {})
    assert "verify_error" in missing


def _panel_row(name="p1", sense="min", opt=OPT):
    return {
        "name": name,
        "probtype": "MBNLP",
        "nvars": 5,
        "ncons": 5,
        "nint": 3,
        "sense": sense,
        "convex": "",
        "opt": opt,
    }


def _write(tmp_path, d, s):
    f = tmp_path / "results.jsonl"
    f.write_text(
        "".join(
            json.dumps({"name": "p1", "solver": solver, "wall": 1.0, "proc_wall": 1.0, **row})
            + "\n"
            for solver, row in (("discopt", d), ("scip", s))
        )
    )
    return f


SCIP_OK = {"status": "optimal", "objective": OPT, "bound": OPT}


@pytest.mark.unit
def test_analyze_fails_on_a_primal_crossing(tmp_path):
    # Dual bound is fine; only the incumbent crosses. Pre-#2006 this exited 0.
    d = {"status": "time_limit", "objective": 4.0, "bound": 3.0, "verified": True}
    d["verified_obj"] = 4.0
    assert analyze([_panel_row()], _write(tmp_path, d, SCIP_OK), 5.0) == 1


@pytest.mark.unit
def test_analyze_fails_on_an_unverified_incumbent(tmp_path):
    # A resumed pre-#2006 results file has objectives but no verification fields.
    d = {"status": "optimal", "objective": OPT, "bound": OPT}
    assert analyze([_panel_row()], _write(tmp_path, d, SCIP_OK), 5.0) == 1


@pytest.mark.unit
def test_analyze_passes_a_clean_row(tmp_path):
    d = {"status": "optimal", "objective": OPT, "bound": OPT, "verified": True}
    d["verified_obj"] = OPT
    assert analyze([_panel_row()], _write(tmp_path, d, SCIP_OK), 5.0) == 0
