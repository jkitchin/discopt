"""Tests for SolveResult serialization and the CLI option/result helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest
from discopt.modeling.core import SolveResult
from discopt.result_io import (
    deserialize_result,
    options_from_payload,
    options_to_payload,
    serialize_result,
    summary_text,
    write_json,
    write_sol,
)
from discopt.solver_tuning import SolverTuning

pytestmark = pytest.mark.unit


def _optimal_result() -> SolveResult:
    return SolveResult(
        status="optimal",
        objective=4.5796,
        bound=4.5796,
        gap=0.0,
        x={"x": np.array(1.5), "y": np.array([0.0, 2.0])},
        wall_time=0.09,
        node_count=3,
    )


def test_serialize_round_trip_optimal():
    r = _optimal_result()
    d = serialize_result(r)
    assert d["schema_version"] == 2  # bumped by #1266 (provenance / options / report)
    assert d["status"] == "optimal" and d["objective"] == pytest.approx(4.5796)
    # ndarray -> list/number
    assert d["x"]["x"] == pytest.approx(1.5)
    assert d["x"]["y"] == [0.0, 2.0]

    r2 = deserialize_result(d)
    assert r2.status == "optimal"
    assert r2.objective == pytest.approx(4.5796)
    np.testing.assert_allclose(r2.x["y"], [0.0, 2.0])


def test_serialize_round_trip_mip_nlp_trace():
    r = _optimal_result()
    r.mip_nlp_trace = {
        "schema_version": 1,
        "solver": "mip-nlp",
        "method": "oa",
        "profile": "shot",
        "iterations": [{"index": 0, "cuts_added": 2}],
        "summary": {"mip_count": 1},
    }

    d = serialize_result(r)
    assert d["mip_nlp_trace"]["profile"] == "shot"

    r2 = deserialize_result(d)
    assert r2.mip_nlp_trace == r.mip_nlp_trace


def test_serialize_infeasible_and_no_solution():
    r = SolveResult(status="infeasible", objective=None, bound=None, gap=None, x=None)
    d = serialize_result(r)
    assert d["status"] == "infeasible"
    assert d["objective"] is None
    assert "x" not in d  # None dict fields are omitted
    r2 = deserialize_result(d)
    assert r2.status == "infeasible" and r2.x is None


def test_non_serializable_fields_are_dropped():
    """``_model`` is a live object graph, not data, so it never travels.

    This used to assert the same of ``infeasibility_certificate``. #1266 carries
    the certificate instead, and a non-certificate value in that field now RAISES
    rather than being dropped -- strictly stricter, and covered by
    ``test_a_non_certificate_in_that_field_is_refused_not_dropped`` below.
    """
    r = _optimal_result()
    r._model = object()  # not JSON-safe
    d = serialize_result(r)

    json.dumps(d)  # must not raise
    assert "_model" not in d


def test_options_to_payload_flattens_tuning_and_drops_callables():
    opts = {
        "time_limit": 60.0,
        "tuning": SolverTuning(rlt_quad=False, node_nlp_stride=8),
        "incumbent_callback": lambda *a: True,  # callbacks cannot cross a socket
    }
    payload = options_to_payload(opts)
    assert payload["time_limit"] == 60.0
    assert isinstance(payload["tuning"], dict)
    assert payload["tuning"]["rlt_quad"] is False
    assert "incumbent_callback" not in payload
    import json

    json.dumps(payload)  # JSON-safe


def test_options_from_payload_rebuilds_tuning():
    payload = {"time_limit": 30.0, "tuning": {"rlt_quad": False, "node_nlp_stride": 8}}
    opts = options_from_payload(payload)
    assert isinstance(opts["tuning"], SolverTuning)
    assert opts["tuning"].rlt_quad is False and opts["tuning"].node_nlp_stride == 8
    # unknown tuning keys are filtered, not crash
    opts2 = options_from_payload({"tuning": {"rlt_quad": True, "bogus": 1}})
    assert opts2["tuning"].rlt_quad is True


def test_write_json_and_sol(tmp_path):
    r = _optimal_result()
    jp = tmp_path / "m.result.json"
    write_json(r, jp)
    import json

    loaded = json.loads(jp.read_text())
    assert loaded["status"] == "optimal"

    sp = tmp_path / "m.sol"
    write_sol(r, ["x", "y"], sp)  # .nl column order
    text = sp.read_text()
    assert text.startswith("discopt optimal")
    # x (1 value) + y (2 values) = 3 primal lines
    assert text.count("\n") >= 4


def test_summary_text_renders():
    s = summary_text(_optimal_result())
    assert "status:" in s and "optimal" in s and "objective:" in s


# ── #1266: provenance, solve options and the validation report ──────────────
#
# Before this, an archived result carried `wall_time` and a node count with
# nothing saying which discopt produced them, with what options, or when -- and
# the Examiner validation report it had passed was dropped outright as
# "non-JSON-safe". The two properties these tests defend hardest are the ones
# that are easy to get subtly wrong:
#
#   * a result that crossed the daemon socket must reach the file with the
#     DAEMON's provenance, not the writing client's
#     (`test_a_carried_provenance_is_never_replaced_by_the_writer`); and
#   * the report must come back as a real ExaminerReport, because the CLI
#     deserializes a daemon reply and then re-serializes it to disk
#     (`test_validation_report_survives_a_daemon_style_round_trip`).


def _report():
    """A small hand-built ExaminerReport (no solve needed)."""
    from discopt.validation.examiner import CheckResult, ExaminerReport

    return ExaminerReport(
        checks=[
            CheckResult(name="primal_feasibility", passed=True, tolerance=1e-6),
            CheckResult(
                name="dual_feasibility",
                passed=False,
                tolerance=1e-6,
                max_violation=3.5e-4,
                norm2_violation=4.1e-4,
                worst_label="c7",
                detail="multiplier sign",
                violators=[("c7", 3.5e-4), ("c2", 1.1e-4)],
            ),
        ],
        merit=1.2e-7,
        n_active_constraints=3,
        n_active_bounds=1,
        duals_recovered=True,
        dual_recovery_residual=2.5e-9,
    )


@pytest.mark.smoke
def test_schema_version_was_bumped_for_the_added_sections():
    assert serialize_result(_optimal_result())["schema_version"] == 2


@pytest.mark.smoke
def test_provenance_is_absent_unless_asked_for():
    """The embedded-in-a-model path must not grow a second, redundant block."""
    assert "provenance" not in serialize_result(_optimal_result())


@pytest.mark.smoke
def test_write_json_always_records_provenance_and_options(tmp_path):
    """The archival path: a file outlives its process, so it must say what made it."""
    import discopt

    p = tmp_path / "r.json"
    write_json(_optimal_result(), p, options={"profile": "fast", "time_limit": 30.0})
    d = json.loads(p.read_text())

    assert d["provenance"]["software"]["version"] == discopt.__version__
    assert d["provenance"]["software"]["rust_core"] is not None
    assert d["provenance"]["created"]
    assert d["solve_options"] == {"profile": "fast", "time_limit": 30.0}


@pytest.mark.smoke
def test_a_carried_provenance_is_never_replaced_by_the_writer(tmp_path):
    """The daemon solved it, so the daemon's identity is the true one.

    Recomputing at the writer would record the CLI's version for numbers another
    process produced -- the precise failure provenance exists to prevent.
    """
    r = _optimal_result()
    r._provenance = {"software": {"version": "9.9.9-daemon", "rust_core": "9.9.9"}}
    r._solve_options = {"profile": "as-solved"}

    p = tmp_path / "r.json"
    write_json(r, p, options={"profile": "as-requested"})
    d = json.loads(p.read_text())

    assert d["provenance"]["software"]["version"] == "9.9.9-daemon"
    assert d["solve_options"] == {"profile": "as-solved"}, (
        "the client's requested options must not overwrite what was actually solved"
    )


@pytest.mark.smoke
def test_validation_report_round_trips_as_a_real_report():
    r = _optimal_result()
    r.validation_report = _report()

    d = serialize_result(r)
    assert d["validation_report"]["passed"] is False
    assert d["validation_report"]["merit"] == pytest.approx(1.2e-7)
    assert [c["name"] for c in d["validation_report"]["checks"]] == [
        "primal_feasibility",
        "dual_feasibility",
    ]

    back = deserialize_result(d).validation_report
    from discopt.validation.examiner import ExaminerReport

    assert isinstance(back, ExaminerReport)
    assert back.passed is False
    assert back.first_failure().name == "dual_feasibility"
    assert back.checks[1].violators == [
        ("c7", pytest.approx(3.5e-4)),
        ("c2", pytest.approx(1.1e-4)),
    ]
    assert back.n_active_constraints == 3 and back.duals_recovered is True


@pytest.mark.smoke
def test_validation_report_survives_a_daemon_style_round_trip(tmp_path):
    """serialize -> wire -> deserialize -> write_json, the real CLI daemon path.

    A report left as a plain dict by the decoder would be refused by the encoder
    on the way out, silently losing it at exactly this step.
    """
    r = _optimal_result()
    r.validation_report = _report()

    wire = json.loads(json.dumps(serialize_result(r, provenance=True)))
    client_side = deserialize_result(wire)

    p = tmp_path / "r.json"
    write_json(client_side, p)
    d = json.loads(p.read_text())
    assert d["validation_report"]["checks"][1]["worst_label"] == "c7"


@pytest.mark.smoke
def test_a_non_examiner_validation_report_is_refused_not_dropped():
    """A report that silently vanishes is the bug this change fixes."""
    r = _optimal_result()
    r.validation_report = {"passed": True}
    with pytest.raises(TypeError, match="no faithful encoding"):
        serialize_result(r)


@pytest.mark.smoke
def test_an_unbounded_objective_no_longer_writes_a_bare_infinity(tmp_path):
    """The reachable case: `objective=-inf` survives `SolveResult.__post_init__`.

    (`bound` and `gap` do not -- the soundness guard there nulls a non-finite
    value -- so `objective` and the timing fields are where this actually bites.)
    Before this, the file held a bare `Infinity`, which is not JSON: every parser
    outside Python rejects it. `write_json` now dumps with `allow_nan=False`, so
    a float that slipped past the encoders raises rather than writing one.
    """
    r = SolveResult(
        status="unbounded",
        objective=float("-inf"),
        bound=None,
        gap=None,
        x={"x": np.array([1.0, float("inf")])},
        wall_time=float("nan"),
    )
    p = tmp_path / "r.json"
    write_json(r, p)

    raw = p.read_text()
    assert "Infinity" not in raw and "NaN" not in raw, "a bare non-JSON token reached the file"
    d = json.loads(raw)
    assert d["objective"] == "-inf" and d["wall_time"] == "nan"
    assert d["x"]["x"] == [1.0, "inf"]

    back = deserialize_result(d)
    assert back.objective == float("-inf")
    assert back.wall_time != back.wall_time  # NaN
    assert np.isinf(back.x["x"][1])
    assert back.x["x"].dtype == float, "a tagged entry left as a string gives an object array"


@pytest.mark.smoke
def test_a_string_scalar_that_reads_like_a_float_tag_is_left_alone():
    """`status`/`algorithm_route` are text; decoding them as floats would corrupt them."""
    d = serialize_result(_optimal_result())
    d["status"] = "inf"
    d["algorithm_route"] = "nan"
    back = deserialize_result(d)
    assert back.status == "inf" and back.algorithm_route == "nan"


@pytest.mark.smoke
def test_infeasibility_certificate_round_trips():
    """An infeasible result is a claim; the witness for it is worth archiving."""
    from discopt.solvers import InfeasibilityCertificate

    r = SolveResult(status="infeasible", objective=None, bound=None, gap=None, x=None)
    r.infeasibility_certificate = InfeasibilityCertificate(
        total_violation=2.5,
        ineq_violations=np.array([0.0, 2.5]),
        eq_violations=np.array([0.0]),
    )

    d = serialize_result(r)
    assert d["infeasibility_certificate"]["total_violation"] == pytest.approx(2.5)
    assert d["infeasibility_certificate"]["ineq_violations"] == [0.0, 2.5]

    back = deserialize_result(json.loads(json.dumps(d, allow_nan=False)))
    cert = back.infeasibility_certificate
    assert isinstance(cert, InfeasibilityCertificate)
    np.testing.assert_allclose(cert.ineq_violations, [0.0, 2.5])
    np.testing.assert_allclose(cert.eq_violations, [0.0])
    assert cert.total_violation == pytest.approx(2.5)


@pytest.mark.smoke
def test_a_non_certificate_in_that_field_is_refused_not_dropped():
    r = SolveResult(status="infeasible", objective=None, bound=None, gap=None, x=None)
    r.infeasibility_certificate = {"total_violation": 1.0}
    with pytest.raises(TypeError, match="no faithful encoding"):
        serialize_result(r)


@pytest.mark.smoke
def test_non_finite_floats_in_the_new_sections_stay_standard_json():
    """`json` writes bare NaN/Infinity, which other languages' parsers reject."""
    from discopt.validation.examiner import CheckResult, ExaminerReport

    r = _optimal_result()
    r.validation_report = ExaminerReport(
        checks=[CheckResult(name="c", passed=False, tolerance=1e-6, max_violation=float("inf"))],
        merit=float("nan"),
    )
    d = serialize_result(r, options={"time_limit": float("inf")})

    json.dumps(d["validation_report"], allow_nan=False)
    json.dumps(d["solve_options"], allow_nan=False)
    assert d["solve_options"]["time_limit"] == "inf"

    back = deserialize_result(d)
    assert back.validation_report.merit != back.validation_report.merit  # NaN
    assert back.validation_report.checks[0].max_violation == float("inf")
    assert back._solve_options["time_limit"] == float("inf")


@pytest.mark.smoke
def test_a_pre_1266_document_still_reads():
    """Version-1 files have none of these keys and must load unchanged."""
    legacy = {
        "schema_version": 1,
        "status": "optimal",
        "objective": 4.5796,
        "bound": 4.5796,
        "gap": 0.0,
        "x": {"x": 1.5},
    }
    r = deserialize_result(legacy)
    assert r.status == "optimal" and r.objective == pytest.approx(4.5796)
    assert r.validation_report is None
    assert getattr(r, "_provenance", None) is None
    assert getattr(r, "_solve_options", None) is None
