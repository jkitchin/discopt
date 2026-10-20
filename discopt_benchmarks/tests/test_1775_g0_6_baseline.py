"""#1775 G0.6: scoring of the open-solver GDP baseline (``scripts/gdp_baseline.py``).

The scorer decides what the gap table calls solved, so it is pinned on the cases
the 300 s run actually produced: a GDPopt-GLOA "optimal" 6e-4 below the certified
optimum of a maximization (scored ``solved`` under the first, 0.1 % tolerance), an
"infeasible" claim on a feasible instance, and a SIGKILL whose worker had printed
warnings first (scored ``error`` instead of a kill).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from benchmarks.gdplib_runner import OracleRecord

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gdp_baseline.py"


@pytest.fixture(scope="module")
def gb():
    spec = importlib.util.spec_from_file_location("gdp_baseline", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MED_TERM = OracleRecord(6797.539701513444, "highs")


def _rec(**kw):
    return {"instance": "i", "arm": "a", "minimize": False, **kw}


def test_optimal_claim_outside_rel_1e4_is_incorrect(gb):
    """GDPopt-GLOA on med_term_purchasing (maximize): 'optimal' at 6793.458."""
    assert gb.score(_rec(status="optimal", objective=6793.4582021), MED_TERM) == "incorrect"


def test_optimal_claim_within_tolerance_is_solved(gb):
    assert gb.score(_rec(status="optimal", objective=6797.5397015134), MED_TERM) == "solved"
    near = 6797.539701513444 * (1 - 0.9e-4)
    assert gb.score(_rec(status="optimal", objective=near), MED_TERM) == "solved"


def test_a_better_than_optimal_incumbent_is_incorrect(gb):
    assert gb.score(_rec(status="feasible", objective=6800.0), MED_TERM) == "incorrect"
    assert gb.score(_rec(status="feasible", objective=6700.0), MED_TERM) == "unsolved"


def test_infeasible_claim_on_a_feasible_instance_is_incorrect(gb):
    assert gb.score(_rec(status="infeasible"), MED_TERM) == "incorrect"
    unproven = OracleRecord(None, "unproven", "gap open")
    assert gb.score(_rec(status="infeasible"), unproven) == "unsolved"
    assert gb.score(_rec(status="infeasible"), unproven, feasible_known=True) == "incorrect"


def test_sigkill_counts_as_a_kill_even_after_output(gb):
    rec = _rec(error="WARNING (W1002): Setting Var 'Ltotal[4,25]' ...", returncode=-9)
    assert gb.score(rec, MED_TERM) == "unsolved"
    assert gb.score(_rec(error="ValueError: boom", returncode=0), MED_TERM) == "error"


def test_a_point_infeasible_in_pyomo_is_incorrect_even_unproven(gb):
    unproven = OracleRecord(None, "unproven", "gap open")
    bad = {"max_rel_violation": 0.02, "bad_counts": 0, "bad_logic": 0}
    assert gb.score(_rec(status="optimal", objective=1.0, in_pyomo=bad), unproven) == "incorrect"


def test_parallel_driver_runs_each_pending_pair_once(gb, tmp_path, monkeypatch):
    """``--jobs 3``: every pending (instance, arm) runs exactly once, recorded runs
    are skipped, and every row is written whole (one JSON object per line)."""
    import argparse
    import json
    import threading

    out = tmp_path / "base.jsonl"
    out.write_text(json.dumps({"instance": "i0", "arm": "a0", "status": "optimal"}) + "\n")
    seen, lock = [], threading.Lock()

    def fake_run(name, arm, args):
        with lock:
            seen.append((name, arm))
        return {"instance": name, "arm": arm, "status": "killed", "load1": 0.0, "jobs": args.jobs}

    monkeypatch.setattr(gb, "_run_one", fake_run)
    monkeypatch.setattr(gb, "preflight", lambda arms: None)
    args = argparse.Namespace(
        include="i0,i1,i2",
        arms="a0,a1",
        out=str(out),
        jobs=3,
        time_limit=1.0,
        grace=1.0,
        max_load=16.0,
    )
    assert gb.drive(args) == 0
    pairs = {(f"i{i}", f"a{a}") for i in range(3) for a in range(2)}
    assert sorted(seen) == sorted(pairs - {("i0", "a0")})
    rows = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert {(r["instance"], r["arm"]) for r in rows} == pairs and len(rows) == 6


# -- direction, tolerance -----------------------------------------------------


ZERO = OracleRecord(0.0, "highs")
MIN_OPT = OracleRecord(100.0, "highs")  # tol = 1e-6 + 1e-4 * 100 = 0.010001


def _min(**kw):
    return _rec(minimize=True, **kw)


def test_minimization_better_than_optimal_is_incorrect(gb):
    """A minimization incumbent below the optimum is a false primal; above it is not."""
    assert gb.score(_min(status="feasible", objective=99.0), MIN_OPT) == "incorrect"
    assert gb.score(_min(status="feasible", objective=101.0), MIN_OPT) == "unsolved"


def test_minimization_tolerance_boundary(gb):
    tol = 1e-6 + 1e-4 * 100.0
    assert gb.score(_min(status="feasible", objective=100.0 - 0.9 * tol), MIN_OPT) == "unsolved"
    assert gb.score(_min(status="feasible", objective=100.0 - 1.1 * tol), MIN_OPT) == "incorrect"
    assert gb.score(_min(status="optimal", objective=100.0 - 0.9 * tol), MIN_OPT) == "solved"
    assert gb.score(_min(status="optimal", objective=100.0 + 0.9 * tol), MIN_OPT) == "solved"
    assert gb.score(_min(status="optimal", objective=100.0 + 1.1 * tol), MIN_OPT) == "incorrect"


def test_absolute_tolerance_is_1e6(gb):
    """At a zero optimum only the absolute tolerance applies: 1e-6, not looser."""
    assert gb.score(_min(status="optimal", objective=5e-7), ZERO) == "solved"
    assert gb.score(_min(status="optimal", objective=1e-4), ZERO) == "incorrect"
    assert gb.score(_rec(status="optimal", objective=-1e-4), ZERO) == "incorrect"


# -- "verified feasible in Pyomo" ---------------------------------------------


def _inp(**kw):
    good = {
        "max_rel_violation": 0.0,
        "bad_counts": 0,
        "bad_logic": 0,
        "checked": 10,
        "max_bound_violation": 0.0,
        "max_integrality_violation": 0.0,
    }
    return {**good, **kw}


def test_verified_feasible_needs_every_check(gb):
    assert gb.verified_feasible(_inp())
    assert not gb.verified_feasible(_inp(checked=0))  # §6: nothing was checked
    assert not gb.verified_feasible(_inp(max_rel_violation=2e-5))
    assert not gb.verified_feasible(_inp(bad_counts=1))
    assert not gb.verified_feasible(_inp(bad_logic=1))
    assert not gb.verified_feasible(_inp(max_bound_violation=2e-5))
    assert not gb.verified_feasible(_inp(max_integrality_violation=2e-5))
    old = _inp()
    del old["max_bound_violation"], old["max_integrality_violation"]
    assert not gb.verified_feasible(old)  # a row from before the domain checks
    assert not gb.verified_feasible(None)


def test_a_point_off_bounds_or_integrality_is_incorrect(gb):
    off = _rec(status="feasible", objective=6700.0, in_pyomo=_inp(max_bound_violation=0.5))
    assert gb.score(off, MED_TERM) == "incorrect"
    frac = _rec(status="feasible", objective=6700.0, in_pyomo=_inp(max_integrality_violation=0.4))
    assert gb.score(frac, MED_TERM) == "incorrect"


UNPROVEN = {"u": OracleRecord(None, "unproven", "gap open")}


def _rows(in_pyomo):
    return [
        {"instance": "u", "arm": "pyomo-lbb", "status": "infeasible", "minimize": True},
        {
            "instance": "u",
            "arm": "discopt-big-m",
            "status": "feasible",
            "objective": 1.0,
            "minimize": True,
            "in_pyomo": in_pyomo,
        },
    ]


def _verdict(gb, rows, arm="pyomo-lbb"):
    return next(r["verdict"] for r in gb.score_rows(rows, UNPROVEN) if r["arm"] == arm)


def test_report_refutes_infeasibility_with_a_verified_point(gb):
    assert _verdict(gb, _rows(_inp())) == "incorrect"


def test_report_ignores_a_point_that_violates_the_rows(gb):
    rows = _rows(_inp(max_rel_violation=0.02))
    assert _verdict(gb, rows) == "unsolved"
    assert _verdict(gb, rows, "discopt-big-m") == "incorrect"


def test_report_ignores_an_unverified_point(gb):
    assert _verdict(gb, _rows(_inp(checked=0))) == "unsolved"
    assert _verdict(gb, _rows(_inp(max_integrality_violation=0.3))) == "unsolved"


def test_report_keeps_the_newest_row_per_pair(gb):
    rows = [
        {"instance": "u", "arm": "a", "error": "ValueError: boom"},
        {"instance": "u", "arm": "a", "status": "optimal", "objective": 1.0},
    ]
    (row,) = gb.score_rows(rows, UNPROVEN)
    assert row["verdict"] == "claimed_unverified"


def test_pyomo_violation_domains_are_opt_in(gb):
    """``domains=True`` sees the bound and integrality violations that loading with
    ``skip_validation`` and rounding hide; the default output is unchanged."""
    import sys
    from types import SimpleNamespace

    import pyomo.environ as pyo
    from pyomo.common.collections import ComponentMap

    sys.path.insert(0, str(_PATH.parent))
    from gdp_convert_parity import _pyomo_violation

    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 1))
    m.n = pyo.Var(domain=pyo.Integers, bounds=(0, 5))
    m.c = pyo.Constraint(expr=m.x + m.n <= 10)
    conv = SimpleNamespace(
        variables=ComponentMap(
            [(m.x, SimpleNamespace(name="x")), (m.n, SimpleNamespace(name="n"))]
        ),
        booleans=ComponentMap(),
        disjunctions=[],
    )
    result = SimpleNamespace(x={"x": 1.5, "n": 2.4})
    plain = _pyomo_violation(m, conv, result)
    assert plain == {"max_rel_violation": 0.0, "bad_counts": 0, "bad_logic": 0, "checked": 1}
    full = _pyomo_violation(m, conv, result, domains=True)
    assert full["max_bound_violation"] == pytest.approx(0.5 / 1.5)
    assert full["max_integrality_violation"] == pytest.approx(0.4)
    assert not gb.verified_feasible(full)
    ok = _pyomo_violation(m, conv, SimpleNamespace(x={"x": 0.5, "n": 2.0}), domains=True)
    assert gb.verified_feasible(ok)


# -- driver: preflight, error retry, scheduling, provenance --------------------


def _args(out, **kw):
    import argparse

    base = {
        "include": "i0,i1",
        "arms": "pyomo-bigm,discopt-big-m",
        "out": str(out),
        "jobs": 1,
        "time_limit": 1.0,
        "grace": 1.0,
        "max_load": 16.0,
        "skip_errors": False,
    }
    return argparse.Namespace(**{**base, **kw})


def test_preflight_refuses_before_any_row(gb, tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "_solver_available", lambda s: s != "scip_direct")
    ran = []
    monkeypatch.setattr(gb, "_run_one", lambda *a: ran.append(a))
    out = tmp_path / "base.jsonl"
    with pytest.raises(RuntimeError, match="pyomo-bigm needs scip_direct"):
        gb.drive(_args(out))
    assert not out.exists() and not ran
    with pytest.raises(ValueError, match="unknown arm"):
        gb.preflight(["pyomo-bogus"])
    monkeypatch.setattr(gb, "_solver_available", lambda s: s != "mcpp")
    with pytest.raises(RuntimeError, match="pyomo-gloa needs mcpp"):
        gb.preflight(["pyomo-gloa"])
    gb.preflight(["pyomo-loa", "discopt-hull"])  # neither needs mcpp


def test_error_rows_are_retried_by_default(gb, tmp_path):
    import json

    out = tmp_path / "base.jsonl"
    rows = [
        {"instance": "i0", "arm": "a", "status": "optimal"},
        {"instance": "i1", "arm": "a", "error": "RuntimeError: dlsym"},
        {"instance": "i2", "arm": "a", "error": "killed (SIGKILL; presumed OOM)", "returncode": -9},
    ]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert gb._done(out) == {("i0", "a"), ("i2", "a")}
    assert gb._done(out, skip_errors=True) == {("i0", "a"), ("i1", "a"), ("i2", "a")}


def test_driver_never_co_schedules_one_instance(gb, tmp_path, monkeypatch):
    import threading
    import time

    lock, running, overlaps, seen = threading.Lock(), [], [], []

    def fake_run(name, arm, args):
        with lock:
            overlaps.append(name in running)
            running.append(name)
            seen.append((name, arm))
        time.sleep(0.02)
        with lock:
            running.remove(name)
        return {"instance": name, "arm": arm, "status": "killed", "load1": 0.0, "jobs": args.jobs}

    monkeypatch.setattr(gb, "_run_one", fake_run)
    monkeypatch.setattr(gb, "preflight", lambda arms: None)
    out = tmp_path / "base.jsonl"
    assert gb.drive(_args(out, include="i0,i1,i2", arms="a0,a1,a2,a3", jobs=3)) == 0
    assert len(seen) == 12 and len(set(seen)) == 12
    assert overlaps and not any(overlaps)


def test_provenance_records_head_and_host(gb):
    import discopt

    prov = gb._provenance(discopt.__file__)
    assert prov["hostname"]
    assert prov["git_head"] is None or len(prov["git_head"]) == 40
    assert (prov["git_head"] is None) == ("git_head_error" in prov)
    nowhere = gb._provenance("/")
    assert nowhere["git_head"] is None and nowhere["git_head_error"]


def test_worker_row_carries_domain_checks_and_provenance(gb):
    """A real discopt run (~2 s): the incumbent is checked with bounds and
    integrality on, so it can count as verified feasible, and the row says where
    it ran."""
    rec = gb.worker("pyomo.jobshop_small", "discopt-big-m", 60.0)
    assert gb.verified_feasible(rec["in_pyomo"])
    assert rec["hostname"] and ("git_head_error" in rec) == (rec["git_head"] is None)
