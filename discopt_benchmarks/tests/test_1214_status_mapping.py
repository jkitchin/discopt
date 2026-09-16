"""#1214: ``unbounded`` and ``error`` survive the shared discopt status table.

``DISCOPT_STATUS_MAP`` omitted both, so the in-process runner and the
subprocess worker recorded a discopt ``"unbounded"`` or ``"error"`` result as
``UNKNOWN``. The fixtures below pin the intended movement between report
buckets, so a later diff can tell a re-bucketed row from a changed solve:

======================  =================  ==================
discopt status          before (#1148)     after (#1214)
======================  =================  ==================
``unbounded``           ``unknown``        ``unknown`` *
``error``               ``unknown``        ``error``
======================  =================  ==================

\\* ``score_result`` has no unbounded bucket; the row now carries
``SolveStatus.UNBOUNDED`` and ``cert_neutrality`` treats it as settled.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import _subprocess_worker
from benchmarks.metrics import (
    DISCOPT_STATUS_MAP,
    SolveResult,
    SolveStatus,
    incorrect_count,
    proved_optimal_count,
)
from benchmarks.runner import BenchmarkConfig, BenchmarkRunner, SolverConfig
from utils.cert_neutrality import _is_wall_limited
from utils.minlplib_data import OUTCOME_ERROR, OUTCOME_UNKNOWN, score_result

pytestmark = pytest.mark.unit

NEW = {"unbounded": SolveStatus.UNBOUNDED, "error": SolveStatus.ERROR}
LOCAL = ("local_optimal", "local_limit", "local_infeasible")

#: The table as #1148 left it, verbatim, for the before/after fixture.
BEFORE_1214 = {
    "optimal": SolveStatus.OPTIMAL,
    "feasible": SolveStatus.FEASIBLE,
    "infeasible": SolveStatus.INFEASIBLE,
    "time_limit": SolveStatus.TIME_LIMIT,
    "node_limit": SolveStatus.TIME_LIMIT,
    "local_optimal": SolveStatus.LOCAL,
    "local_limit": SolveStatus.LOCAL,
    "local_infeasible": SolveStatus.LOCAL,
}


def _lookup(table, status):
    """The call sites' lookup, with their fail-closed default."""
    return table.get(status, SolveStatus.UNKNOWN)


@pytest.mark.parametrize("status, expected", sorted(NEW.items()))
def test_the_table_names_both_statuses(status, expected):
    assert _lookup(DISCOPT_STATUS_MAP, status) is expected


def test_the_table_only_grew_by_the_two_entries():
    assert DISCOPT_STATUS_MAP == BEFORE_1214 | NEW


@pytest.mark.parametrize("status", LOCAL)
def test_local_statuses_stay_local(status):
    assert _lookup(DISCOPT_STATUS_MAP, status) is SolveStatus.LOCAL


@pytest.mark.parametrize("status", ["", "bogus", "OPTIMAL", "numerical_error"])
def test_unrecognised_statuses_still_fail_closed(status):
    assert _lookup(DISCOPT_STATUS_MAP, status) is SolveStatus.UNKNOWN


@pytest.mark.parametrize("status", [*NEW, *LOCAL])
def test_no_new_or_local_row_is_scored_as_a_proved_optimum(status):
    row = SolveResult(
        instance="probe",
        solver="discopt",
        status=_lookup(DISCOPT_STATUS_MAP, status),
        objective=-1e9,
    )
    assert not row.is_solved
    assert incorrect_count([row], {"probe": 1.0}) == 0
    assert proved_optimal_count([row]) == 0
    assert score_result(row, None) != "optimal_proven"


@pytest.mark.parametrize(
    "status, before, after",
    [("unbounded", OUTCOME_UNKNOWN, OUTCOME_UNKNOWN), ("error", OUTCOME_UNKNOWN, OUTCOME_ERROR)],
)
def test_report_bucket_movement(status, before, after):
    def bucket(table):
        return score_result(
            SolveResult(instance="probe", solver="discopt", status=_lookup(table, status)), None
        )

    assert bucket(BEFORE_1214) == before
    assert bucket(DISCOPT_STATUS_MAP) == after


def test_an_unbounded_row_is_now_settled_for_cert_neutrality():
    """A settled row is never explained away as a wall-clock coincidence."""

    def row(table):
        return SolveResult(
            instance="probe",
            solver="discopt",
            status=_lookup(table, "unbounded"),
            wall_time=59.0,
        ).to_dict()

    assert _is_wall_limited(row(BEFORE_1214), budget=60.0) is True
    assert _is_wall_limited(row(DISCOPT_STATUS_MAP), budget=60.0) is False


# --- Through the two producers --------------------------------------------


def _fake_result(status):
    return SimpleNamespace(
        status=status,
        objective=None,
        bound=None,
        wall_time=0.25,
        node_count=0,
        root_gap=None,
        root_time=None,
        rust_time=None,
        jax_time=None,
        python_time=None,
        pounce_time=None,
        algorithm_route=None,
    )


@pytest.fixture
def stub_solve(monkeypatch):
    """Replace ``from_nl`` so each producer sees a solve returning ``status``."""
    import discopt.modeling as dm

    calls = []

    def install(status):
        def fake_from_nl(path):
            calls.append(path)
            return SimpleNamespace(solve=lambda **kw: _fake_result(status))

        monkeypatch.setattr(dm, "from_nl", fake_from_nl)
        return calls

    return install


NL = Path(__file__).resolve().parents[2] / "python" / "tests" / "data" / "minlplib_nl" / "tls2.nl"


@pytest.mark.parametrize("status, expected", sorted(NEW.items()))
def test_in_process_runner(stub_solve, status, expected):
    calls = stub_solve(status)
    runner = BenchmarkRunner(BenchmarkConfig(suite_name="unit", time_limit=5))
    cfg = SolverConfig(name="discopt", command="", solver_type="internal")
    res = runner._run_discopt(cfg, str(NL), 0)
    assert calls, "the stubbed solve never ran"
    assert res.status is expected
    assert SolveResult.from_dict(res.to_dict()).status is expected


@pytest.mark.parametrize("status, expected", sorted(NEW.items()))
def test_subprocess_worker(stub_solve, status, expected):
    calls = stub_solve(status)
    d = _subprocess_worker._solve("probe", str(NL), 5.0, {})
    assert calls, "the stubbed solve never ran"
    assert "_error" not in d, d
    assert d["status"] == expected.value
    assert SolveResult.from_dict(d).status is expected
