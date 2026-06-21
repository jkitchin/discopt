"""Tests for SolveResult serialization and the CLI option/result helpers."""

from __future__ import annotations

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
    assert d["schema_version"] == 1
    assert d["status"] == "optimal" and d["objective"] == pytest.approx(4.5796)
    # ndarray -> list/number
    assert d["x"]["x"] == pytest.approx(1.5)
    assert d["x"]["y"] == [0.0, 2.0]

    r2 = deserialize_result(d)
    assert r2.status == "optimal"
    assert r2.objective == pytest.approx(4.5796)
    np.testing.assert_allclose(r2.x["y"], [0.0, 2.0])


def test_serialize_infeasible_and_no_solution():
    r = SolveResult(status="infeasible", objective=None, bound=None, gap=None, x=None)
    d = serialize_result(r)
    assert d["status"] == "infeasible"
    assert d["objective"] is None
    assert "x" not in d  # None dict fields are omitted
    r2 = deserialize_result(d)
    assert r2.status == "infeasible" and r2.x is None


def test_non_serializable_fields_are_dropped():
    r = _optimal_result()
    r._model = object()  # not JSON-safe
    r.infeasibility_certificate = object()
    d = serialize_result(r)
    import json

    json.dumps(d)  # must not raise
    assert "_model" not in d and "infeasibility_certificate" not in d


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
