"""Archived results must not turn strings reading "nan"/"inf" into floats (#1292).

A non-finite float was written as the bare string ``"nan"``/``"inf"``, and the
reader turned every such string back into a float: a validation report's
``worst_label`` for a variable named ``nan``, and -- through ``Model.save`` --
``error``/``algorithm_route`` as well. ``mip_nlp_trace`` was written untagged, so a
``-inf`` in it made ``write_json`` raise.
"""

import json
import math

import discopt as do
import discopt.result_io as rio
import discopt.serialize as ser
import numpy as np
import pytest
from discopt.validation.examiner import CheckResult, ExaminerReport, examine


def _nan_named_report():
    m = do.Model("v")
    a = m.continuous("nan", lb=0, ub=2)
    m.minimize(a)
    m.subject_to(a <= 1, name="cap")
    r = do.SolveResult(status="feasible", objective=1.9, x={"nan": np.array(1.9)})
    r.validation_report = examine(r, m)
    labels = [c.worst_label for c in r.validation_report.checks]
    assert "nan" in labels
    return m, r


def _labels(r):
    return [c.worst_label for c in r.validation_report.checks]


def test_worst_label_nan_survives_write_json(tmp_path):
    _, r = _nan_named_report()
    path = tmp_path / "r.json"
    rio.write_json(r, path)
    back = rio.deserialize_result(json.loads(path.read_text()))
    assert _labels(back) == _labels(r)
    assert all(isinstance(lbl, str) for lbl in _labels(back))


def test_string_fields_survive_model_save(tmp_path):
    m, r = _nan_named_report()
    r.error = "nan"
    r.algorithm_route = "inf"
    r.node_count = np.int64(7)
    m.save(tmp_path / "m.dopt", result=r)
    back = do.load(tmp_path / "m.dopt").saved_result
    assert _labels(back) == _labels(r)
    assert back.error == "nan" and back.algorithm_route == "inf"
    assert back.node_count == 7


def test_report_float_fields_still_decode():
    r = do.SolveResult(status="feasible")
    r.validation_report = ExaminerReport(
        checks=[
            CheckResult(
                name="c",
                passed=False,
                tolerance=1e-6,
                max_violation=float("inf"),
                worst_label="inf",
                detail="nan",
                violators=[("nan", float("nan")), ("x", 2.0)],
            )
        ],
        merit=float("nan"),
    )
    back = rio.deserialize_result(json.loads(json.dumps(rio.serialize_result(r))))
    c = back.validation_report.checks[0]
    assert c.max_violation == math.inf and c.worst_label == "inf" and c.detail == "nan"
    assert c.violators[0][0] == "nan" and math.isnan(c.violators[0][1])
    assert c.violators[1] == ("x", 2.0)
    assert math.isnan(back.validation_report.merit)


def test_solve_options_strings_and_floats_are_distinct():
    r = do.SolveResult(status="optimal")
    options = {
        "label": "inf",
        "time_limit": float("inf"),
        "nested": {"x": "-inf", "y": float("-inf")},
        "collide": {"__float__": "nan"},
        "collide2": {"__dict__": {"__float__": "inf"}},
    }
    d = json.loads(json.dumps(rio.serialize_result(r, options=options), allow_nan=False))
    back = rio.deserialize_result(d)._solve_options
    assert back["label"] == "inf"
    assert back["time_limit"] == math.inf
    assert back["nested"] == {"x": "-inf", "y": -math.inf}
    assert back["collide"] == {"__float__": "nan"}
    assert back["collide2"] == {"__dict__": {"__float__": "inf"}}


def test_mip_nlp_trace_with_non_finite_values_round_trips(tmp_path):
    r = do.SolveResult(status="feasible")
    r.mip_nlp_trace = {
        "solver": "nan",
        "iterations": [{"lb": float("-inf"), "ub": 3.0, "note": "inf"}],
    }
    path = tmp_path / "r.json"
    rio.write_json(r, path)
    back = rio.deserialize_result(json.loads(path.read_text())).mip_nlp_trace
    assert back["solver"] == "nan"
    assert back["iterations"] == [{"lb": -math.inf, "ub": 3.0, "note": "inf"}]


def test_version_2_documents_still_read():
    legacy = {
        "schema_version": 2,
        "status": "feasible",
        "objective": "inf",
        "error": "nan",
        "solve_options": {"time_limit": "inf"},
        "mip_nlp_trace": {"solver": "mip-nlp", "lb": None},
    }
    back = rio.deserialize_result(legacy)
    assert back.objective == math.inf and back.error == "nan"
    assert back._solve_options == {"time_limit": math.inf}
    assert back.mip_nlp_trace == {"solver": "mip-nlp", "lb": None}


@pytest.mark.parametrize("bad", [{"__float__": "x"}, {"__float__": 1.0}, {"__dict__": [1]}])
def test_malformed_tag_is_refused(bad):
    r = do.SolveResult(status="optimal")
    d = rio.serialize_result(r, options={"k": 1})
    d["solve_options"] = {"k": bad}
    with pytest.raises(ser.SerializationError):
        rio.deserialize_result(d)
