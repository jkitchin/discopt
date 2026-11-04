"""highs-parity P0.1: the LP/MILP panel harness refuses rows it would mis-report.

``discopt_benchmarks/scripts/lp_milp_panel.py`` is the instrument every R1 parity number
(``docs/dev/highs-parity-plan.md`` §1.3) comes from. These pin the four P0.1 additions as
pure functions -- no solve, no HiGHS reference files:

* a subprocess-arm row answered by the other engine's route raises (a HiGHS-arm solve the
  route declined and passed on to the Rust engine would otherwise be reported as a HiGHS
  measurement, CLAUDE.md §8);
* the arm order rotates so each arm leads one round;
* the load gate withholds walls when any sampled load exceeds it;
* the INF panel's verdict check flags a returned point or a contradicting verdict, and
  accepts an honest refusal.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPTS = Path(__file__).resolve().parents[2] / "discopt_benchmarks" / "scripts"


@pytest.fixture(scope="module")
def panel():
    pytest.importorskip("highspy")
    sys.path.insert(0, str(_SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "lp_milp_panel_p01", _SCRIPTS / "lp_milp_panel.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(_SCRIPTS))
    # Load marker: the P0.1 route check must be the code under test.
    assert hasattr(mod, "check_route"), "lp_milp_panel.py predates P0.1"
    return mod


def _row(route: str, env: str) -> dict:
    return {"route": route, "env_backend": env, "algorithm_route": "x", "status": "optimal"}


@pytest.mark.parametrize("arm, route", [("R", "rust"), ("H", "highs")])
def test_matching_route_passes(panel, arm, route):
    panel.check_route("inst", arm, _row(route, route))


@pytest.mark.parametrize("arm", ["R", "H"])
def test_presolve_answered_row_is_kept_as_neither_engine(panel, arm):
    # Root presolve proved infeasibility before the LP/MILP dispatch (gams10am on the
    # INF panel, measured): neither engine ran, so the row is not a wrong-route row.
    panel.check_route("inst", arm, _row("presolve", panel.ARM_BACKEND[arm]))
    with pytest.raises(RuntimeError, match="expects route"):
        panel.check_route("inst", arm, _row("presolve", None))


@pytest.mark.parametrize(
    "arm, route, env",
    [
        ("H", "rust", "highs"),  # the HiGHS route declined; the Rust engine answered
        ("R", "highs", "rust"),  # the opt-out was ignored
        ("H", "highs", "rust"),  # the subprocess ran under the wrong env
        ("R", "rust", None),  # env not propagated
    ],
)
def test_wrong_route_row_raises(panel, arm, route, env):
    with pytest.raises(RuntimeError, match="expects route"):
        panel.check_route("inst", arm, _row(route, env))


def test_arm_order_rotates_so_each_arm_leads_once(panel):
    arms = ["R", "H", "H0"]
    orders = [panel.arm_order(arms, r) for r in range(3)]
    assert orders == [["R", "H", "H0"], ["H", "H0", "R"], ["H0", "R", "H"]]
    assert sorted(o[0] for o in orders) == sorted(arms)
    assert panel.arm_order(arms, 3) == arms


def test_load_gate_withholds_walls_on_either_sample(panel):
    assert panel.load_ok(None, 50.0, 50.0)
    assert panel.load_ok(2.0, 1.0, 2.0)
    assert not panel.load_ok(2.0, 2.5, 1.0)  # start over the gate
    assert not panel.load_ok(2.0, 1.0, 2.5)  # end over the gate


@pytest.mark.parametrize(
    "status, oracle, wrong",
    [
        ("optimal", "Infeasible", True),
        ("feasible", "Infeasible", True),
        ("unbounded", "Infeasible", True),
        ("infeasible", "Unbounded", True),
        ("optimal", "Unbounded", True),
        ("infeasible", "Infeasible", False),
        ("Infeasible", "Infeasible", False),  # H0's HiGHS spelling
        ("unbounded", "Unbounded", False),
        ("error", "Infeasible", False),  # an honest refusal is not a wrong answer
        ("time_limit", "Infeasible", False),
    ],
)
def test_inf_panel_verdicts(panel, status, oracle, wrong):
    assert bool(panel._incorrect_inf({"status": status}, oracle)) is wrong


def test_kind_filter(panel):
    assert panel._kind_reject("lp", 3) == "has integer columns"
    assert panel._kind_reject("milp", 0) == "no integer columns"
    assert panel._kind_reject("lp", 0) is None
    assert panel._kind_reject("milp", 2) is None
    assert panel._kind_reject("inf", 2) is None
    assert panel._kind_reject("any", 0) is None


# P0.3: the run JSON is written after every instance and an interrupted run can resume
# (a container restart killed the first MIPLIB-38 baseline 13/38 in, losing every row).
def _summ(name, panel, executed, wrong, wall_ok, routes=("rust",)):
    return {"name": name, "panel": panel, "executed": executed, "incorrect_solves": wrong,
            "wall_valid": wall_ok, "R": {"routes": list(routes)}}  # fmt: skip


def test_tally_recomputes_totals_from_rows(panel):
    rows = [
        _summ("a", "L", 9, 0, True),
        _summ("b", "L", 9, 1, False),
        _summ("c", "INF", 9, 0, True, routes=("presolve",)),
    ]
    t = panel.tally(rows, ["R", "H", "H0"])
    assert t["executed"] == 27
    assert t["incorrect_count"] == 1
    assert t["executed_by_panel"] == {"L": 18, "INF": 9}
    assert t["walls_gated"] == 1
    assert t["presolve_answered"] == {"R": 1}


def _args(**kw):
    base = {k: None for k in ("manifest", "panel", "instances", "arms", "rounds",
                              "time_limit", "gap", "max_nodes", "h0_subprocess",
                              "load_gate")}  # fmt: skip
    base.update({"arms": "R,H,H0", "rounds": 3, "time_limit": 20.0, **kw})
    return base


def test_resume_rows_appends_only_to_the_same_run(panel, tmp_path):
    f = tmp_path / "run.json"
    rows = [_summ("a", "M", 9, 0, True)]
    panel._write_run(f, {"commit": "abc"}, _args(), rows, ["R"], complete=False)
    assert panel.resume_rows(f, "abc", _args()) == rows
    with pytest.raises(SystemExit, match="one commit"):
        panel.resume_rows(f, "def", _args())
    with pytest.raises(SystemExit, match="arguments differ"):
        panel.resume_rows(f, "abc", _args(time_limit=5.0))
    panel._write_run(f, {"commit": "abc"}, _args(), rows, ["R"], complete=True)
    with pytest.raises(SystemExit, match="complete run"):
        panel.resume_rows(f, "abc", _args())
    with pytest.raises(SystemExit, match="does not exist"):
        panel.resume_rows(tmp_path / "missing.json", "abc", _args())
