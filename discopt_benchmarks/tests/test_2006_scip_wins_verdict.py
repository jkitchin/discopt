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


# --- QPLIB arm (#2006): best-known oracle, upstream .lp for SCIP -------------

QPLIB_DATA = Path(__file__).resolve().parents[2] / "python" / "tests" / "data" / "qplib"
REF = 15.38637379  # QPLIB_0031, qplib.solu =best=


@pytest.mark.unit
def test_best_known_oracle_does_not_call_a_better_incumbent_wrong():
    # A verified point better than a best-known (unproven) value is an
    # improvement, not an infeasible point: no WRONG_BEATS_OPT, and a claimed
    # optimum there is OK.
    better = {"status": "time_limit", "objective": 15.0, "verified": True, "verified_obj": 15.0}
    assert verdict(better, REF, "min", proven=False) == "FOUND_UNPROVEN"
    assert verdict(dict(better, status="optimal"), REF, "min", proven=False) == "OK"
    assert verdict(better, REF, "min") == "WRONG_BEATS_OPT"  # proven: still wrong


@pytest.mark.unit
def test_best_known_oracle_still_catches_a_worse_claim_and_infeasible():
    worse = {"status": "optimal", "objective": 16.0, "verified": True, "verified_obj": 16.0}
    assert verdict(worse, REF, "min", proven=False) == "WRONG"
    assert verdict(dict(worse, status="time_limit"), REF, "min", proven=False) == "FAIL"
    # The reference point is feasible, so an infeasible claim is wrong.
    assert verdict({"status": "infeasible"}, REF, "min", proven=False) == "WRONG"
    # Max sense mirrors.
    assert verdict(dict(worse, objective=14.0, verified_obj=14.0), REF, "max", proven=False) == (
        "WRONG"
    )


def _qplib_row(**kw):
    return {**_panel_row(opt=REF), "proven": False, "files": {}, **kw}


def _write_q(tmp_path, d, s):
    f = tmp_path / "results.jsonl"
    f.write_text(
        "".join(
            json.dumps({"name": "p1", "solver": solver, "wall": 1.0, "proc_wall": 1.0, **row})
            + "\n"
            for solver, row in (("discopt", d), ("scip", s))
        )
    )
    return f


D_OK = {"status": "optimal", "objective": REF, "bound": REF, "verified": True, "verified_obj": REF}


@pytest.mark.unit
def test_analyze_drops_a_scip_row_whose_input_does_not_reproduce_the_reference(tmp_path, capsys):
    # SCIP "solves" the instance, but its .lp evaluated the reference point to
    # another value: it read a different model, so it must not count as a win.
    s = {"status": "optimal", "objective": 1.0, "bound": 1.0, "fidelity_obj": 2.0}
    d = dict(D_OK, status="time_limit", objective=20.0, verified_obj=20.0)
    assert analyze([_qplib_row()], _write_q(tmp_path, d, s), 5.0) == 0
    out = capsys.readouterr().out
    assert "mismatches: 1" in out and "discopt not OK: 0" in out


@pytest.mark.unit
def test_analyze_fails_when_a_qplib_scip_row_was_never_checked(tmp_path):
    s = {"status": "optimal", "objective": REF, "bound": REF}  # no fidelity fields
    assert analyze([_qplib_row()], _write_q(tmp_path, D_OK, s), 5.0) == 1


@pytest.mark.unit
def test_analyze_passes_a_clean_qplib_row(tmp_path, capsys):
    s = {"status": "optimal", "objective": REF, "bound": REF, "fidelity_obj": REF + 1e-9}
    assert analyze([_qplib_row()], _write_q(tmp_path, D_OK, s), 5.0) == 0
    assert "fidelity checks: 1, mismatches: 0" in capsys.readouterr().out


def _manifest(tmp_path, rows):
    cols = "name,probtype,sense,nvars,ncons,n_integral,nonconvex,reference,usable_oracle"
    f = tmp_path / "qplib_manifest.csv"
    f.write_text(cols + "\n" + "".join(",".join(map(str, r)) + "\n" for r in rows))
    return tmp_path


@pytest.mark.unit
def test_qplib_panel_is_gated_on_usable_oracle_and_size(tmp_path):
    q = _manifest(
        tmp_path,
        [
            ("QPLIB_A", "QML", "minimize", 60, 32, 30, "True", 15.4, "True"),
            ("QPLIB_B", "QML", "maximize", 10, 5, 0, "False", 1.0, "True"),
            ("QPLIB_C", "QML", "minimize", 10, 5, 0, "True", 2.0, "False"),  # unusable
        ],
    )
    panel = screen.build_qplib_panel(tmp_path / "out", q, 50, 50, None)
    assert [p["name"] for p in panel] == ["QPLIB_B"]
    p = panel[0]
    assert p["sense"] == "max" and p["proven"] is False and p["convex"] == "True"
    assert p["files"]["discopt"].endswith("qplib/QPLIB_B.qplib")
    assert p["files"]["scip"][0].endswith("/lp/QPLIB_B.lp")
    # A names file restricts but never admits an instance the manifest rejects.
    assert [p["name"] for p in screen.build_qplib_panel(tmp_path, q, 50, 50, {"QPLIB_A"})] == [
        "QPLIB_A"
    ]
    with pytest.raises(SystemExit):
        screen.build_qplib_panel(tmp_path, q, 50, 50, {"QPLIB_C"})


def test_verify_incumbent_reads_a_qplib_instance():
    from discopt.interfaces import qplib

    path = str(QPLIB_DATA / "qplib" / "QPLIB_0031.qplib")
    inst = qplib.read_qplib(path)
    x, _ = qplib.read_solution(str(QPLIB_DATA / "sol" / "QPLIB_0031.sol"), inst)
    point = {inst.variable_name(i): float(x[i]) for i in range(inst.n_vars)}
    good = _verify_incumbent(path, point)
    assert good["verified"] is True, good
    assert abs(good["verified_obj"] - REF) <= 1e-6
    bad = _verify_incumbent(path, {k: v + 1e3 for k, v in point.items()})
    assert bad["verified"] is False


def test_scip_fidelity_reproduces_the_reference_and_detects_a_wrong_point(tmp_path):
    pytest.importorskip("pyscipopt")
    lp = str(QPLIB_DATA / "lp" / "QPLIB_0031.lp")
    sol = QPLIB_DATA / "sol" / "QPLIB_0031.sol"
    good = screen._scip_fidelity(lp, str(sol))
    assert good["fidelity_fixed"] == 60, good  # every declared column, not SCIP's aux
    assert abs(good["fidelity_obj"] - REF) <= 1e-6 * REF
    # Perturb one continuous coordinate: the probe must see a different objective
    # (or an infeasible point), i.e. it measures something.
    lines = sol.read_text().splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.split()[0].startswith("x"))
    name, val = lines[k].split()
    lines[k] = f"{name} {float(val) + 0.25}"
    bad_sol = tmp_path / "bad.sol"
    bad_sol.write_text("\n".join(lines) + "\n")
    bad = screen._scip_fidelity(lp, str(bad_sol))
    assert bad["fidelity_obj"] is None or abs(bad["fidelity_obj"] - REF) > 1e-6 * REF


# --- #2066 review ------------------------------------------------------------


@pytest.mark.unit
def test_unverified_claim_better_than_a_best_known_reference_is_not_ok():
    # SCIP rows carry no verification: "optimal" below a verified feasible
    # reference is unchecked, never a win (it may be a misread model).
    s = {"status": "optimal", "objective": REF - 1.0, "bound": REF - 1.0}
    assert verdict(s, REF, "min", proven=False) == "BEATS_REF_UNVERIFIED"
    assert verdict(dict(s, status="gaplimit"), REF, "min", proven=False) == "BEATS_REF_UNVERIFIED"
    assert verdict(dict(s, objective=REF + 1.0), REF, "max", proven=False) == (
        "BEATS_REF_UNVERIFIED"
    )
    # Verified, it is a genuine improvement; at the reference it is plain OK.
    assert verdict(dict(s, verified=True), REF, "min", proven=False) == "OK"
    assert verdict(dict(s, objective=REF), REF, "min", proven=False) == "OK"


@pytest.mark.unit
def test_analyze_does_not_count_an_unverified_better_claim_as_a_win(tmp_path, capsys):
    s = {"status": "optimal", "objective": REF - 1.0, "bound": REF - 1.0, "fidelity_obj": REF}
    d = dict(D_OK, status="time_limit", objective=20.0, verified_obj=20.0)
    analyze([_qplib_row()], _write_q(tmp_path, d, s), 5.0)
    out = capsys.readouterr().out
    assert "discopt not OK: 0" in out and "not wins): ['p1']" in out


@pytest.mark.unit
def test_analyze_reports_a_timed_out_scip_row_as_fidelity_unchecked(tmp_path, capsys):
    assert analyze([_qplib_row()], _write_q(tmp_path, D_OK, {"status": "HARD_TIMEOUT"}), 5.0) == 0
    assert "unchecked (SCIP job crashed/timed out): [('p1', 'HARD_TIMEOUT')]" in (
        capsys.readouterr().out
    )


@pytest.mark.unit
def test_scip_job_timeout_includes_the_fidelity_solve():
    assert screen._job_timeout(30, ["ref.sol"]) == 30 + 30 + screen.FIDELITY_TIME_LIMIT
    assert screen._job_timeout(30, []) == 60


@pytest.mark.unit
def test_bound_check_is_feasibility_scale_not_gap_scale(tmp_path):
    # A dual bound 5e-5 (relative) past a proven optimum sits inside GAP=1e-4 and
    # was invisible to the old check; it is a certificate violation.
    d = {"status": "optimal", "objective": OPT, "bound": OPT * (1 + 5e-5), "verified": True}
    d["verified_obj"] = OPT
    assert analyze([_panel_row()], _write(tmp_path, d, SCIP_OK), 5.0) == 1
    d["bound"] = OPT * (1 + 1e-7)  # rounding-scale: still clean
    assert analyze([_panel_row()], _write(tmp_path, d, SCIP_OK), 5.0) == 0
