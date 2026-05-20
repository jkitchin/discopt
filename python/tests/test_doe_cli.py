"""Smoke tests for the ``discopt doe`` CLI surface.

Drives every verb through its pure ``do_*`` function (not via the
argparse layer) so the contract a future GUI binds to is what's
under test. Argparse parsing is exercised separately in one focused
test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.smoke


openpyxl = pytest.importorskip("openpyxl")

from discopt.doe.cli import (  # noqa: E402
    DoEError,
    ExtendParams,
    NewParams,
    do_extend,
    do_fit,
    do_new,
    do_status,
    do_templates,
)
from discopt.doe.workbook import Workbook  # noqa: E402

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _fill_response(path: Path, response: str, predictor) -> None:
    """Fill in the response column on every pending row using `predictor`."""
    wb = openpyxl.load_workbook(path)
    runs = wb["runs"]
    headers = [c.value for c in runs[1]]
    resp_idx = headers.index(response) + 1
    for row in runs.iter_rows(min_row=2):
        if row[0].value is None:
            continue
        row_dict = dict(zip(headers, [c.value for c in row]))
        row[resp_idx - 1].value = float(predictor(row_dict))
    wb.save(path)


# ──────────────────────────────────────────────────────────────────
# do_templates
# ──────────────────────────────────────────────────────────────────


def test_templates_lists_all():
    out = do_templates()
    names = [t["name"] for t in out["templates"]]
    assert names == [
        "linear",
        "polynomial-1d",
        "response-surface-2d",
        "response-surface-3d",
        "scheffe-linear",
        "scheffe-quadratic",
        "scheffe-special-cubic",
        "latin-square",
        "graeco-latin",
        "hyper-graeco-latin",
        "factorial-2level",
        "optimize",
    ]


# ──────────────────────────────────────────────────────────────────
# do_new — one test per template
# ──────────────────────────────────────────────────────────────────


def _new_params(tmpdir, *, template, inputs, n, degree=None, response="y", error=1.0, n_starts=3):
    return NewParams(
        output=Path(tmpdir) / f"{template}.xlsx",
        n=n,
        inputs=inputs,
        response_name=response,
        measurement_error=error,
        criterion="determinant",
        seed=0,
        n_starts=n_starts,
        template=template,
        degree=degree,
    )


def test_new_linear(tmp_path):
    out = do_new(_new_params(tmp_path, template="linear", inputs=[("x", 0.0, 10.0)], n=3))
    assert out["template"] == "linear"
    assert out["parameter_names"] == ["b0", "b1"]
    assert len(out["designs"]) == 3
    assert all(0 <= d["x"] <= 10 for d in out["designs"])
    assert out["next_command"].startswith("discopt doe status")


def test_new_polynomial_1d(tmp_path):
    out = do_new(
        _new_params(tmp_path, template="polynomial-1d", inputs=[("x", 0.0, 10.0)], n=4, degree=3)
    )
    assert out["parameter_names"] == ["b0", "b1", "b2", "b3"]
    assert len(out["designs"]) == 4


def test_new_response_surface_2d(tmp_path):
    out = do_new(
        _new_params(
            tmp_path,
            template="response-surface-2d",
            inputs=[("x1", 0.0, 10.0), ("x2", -5.0, 5.0)],
            n=6,
        )
    )
    assert out["parameter_names"] == ["b0", "b1", "b2", "b11", "b22", "b12"]
    assert len(out["designs"]) == 6


@pytest.mark.slow
def test_new_response_surface_3d(tmp_path):
    # Marked slow: a 10-parameter D-optimal search drives ~700 inner QP solves
    # and runs ~40s locally, blowing past the 120s CI fast-job timeout under
    # parallel load. n_starts=1 keeps the full-suite run as short as possible.
    # The response-surface model-based path is already covered by the 2D test.
    out = do_new(
        _new_params(
            tmp_path,
            template="response-surface-3d",
            inputs=[("x1", 0.0, 10.0), ("x2", -5.0, 5.0), ("x3", 0.0, 1.0)],
            n=10,
            n_starts=1,
        )
    )
    assert len(out["parameter_names"]) == 10
    assert len(out["designs"]) == 10


# ──────────────────────────────────────────────────────────────────
# Synthetic-truth round trip: new → fill → fit → recover
# ──────────────────────────────────────────────────────────────────


def test_fit_recovers_known_truth_response_surface(tmp_path):
    inputs = [("x1", 0.0, 10.0), ("x2", -5.0, 5.0)]
    truth = {
        "b0": 1.0,
        "b1": 0.5,
        "b2": -0.3,
        "b11": -0.02,
        "b22": 0.1,
        "b12": 0.05,
    }

    def predict(row):
        x1, x2 = row["x1"], row["x2"]
        return (
            truth["b0"]
            + truth["b1"] * x1
            + truth["b2"] * x2
            + truth["b11"] * x1 * x1
            + truth["b22"] * x2 * x2
            + truth["b12"] * x1 * x2
        )

    # n_starts=1: this test only checks fit recovery of known parameters, not
    # design optimality. The default n_starts=3 triples the inner D-optimal QP
    # solves and pushes the test past the 120s CI timeout under parallel load.
    out = do_new(
        _new_params(
            tmp_path,
            template="response-surface-2d",
            inputs=inputs,
            n=10,
            error=0.01,
            n_starts=1,
        )
    )
    wb_path = Path(out["workbook_path"])
    _fill_response(wb_path, "y", predict)

    fit = do_fit({"workbook": str(wb_path)})
    estimates = {p["name"]: p["estimate"] for p in fit["parameters"]}
    for name, expected in truth.items():
        assert estimates[name] == pytest.approx(expected, abs=1e-3)
    assert fit["n_observations"] == 10
    assert fit["log_det_fim"] > 0


def test_extend_appends_new_batch(tmp_path):
    inputs = [("x", 0.0, 10.0)]

    def predict(row):
        return 2.0 + 3.0 * row["x"]

    out = do_new(_new_params(tmp_path, template="linear", inputs=inputs, n=4, error=0.05))
    wb_path = Path(out["workbook_path"])
    _fill_response(wb_path, "y", predict)
    do_fit({"workbook": str(wb_path)})

    before_status = do_status({"workbook": str(wb_path)})
    assert before_status["n_pending"] == 0

    ext = do_extend(ExtendParams(workbook=wb_path, n=3, n_starts=3))
    assert ext["batch"] == 2
    assert len(ext["new_run_ids"]) == 3
    assert sorted(ext["new_run_ids"]) == ext["new_run_ids"]  # contiguous & sorted

    after_status = do_status({"workbook": str(wb_path)})
    assert after_status["n_completed"] == 4
    assert after_status["n_pending"] == 3
    assert after_status["n_total"] == 7


# ──────────────────────────────────────────────────────────────────
# Module escape hatch
# ──────────────────────────────────────────────────────────────────


_MODULE_EXPERIMENT_SOURCE = """
import discopt.modeling as dm
from discopt.estimate import Experiment, ExperimentModel


class CliTestKExp(Experiment):
    \"\"\"y = k * x for the --module escape-hatch test.\"\"\"

    def create_model(self, **kwargs):
        m = dm.Model("k_exp")
        k = m.continuous("k", lb=0.01, ub=20.0)
        x = m.continuous("x", lb=0.1, ub=10.0)
        return ExperimentModel(
            model=m,
            unknown_parameters={"k": k},
            design_inputs={"x": x},
            responses={"y": k * x},
            measurement_error={"y": 0.1},
        )
"""


@pytest.fixture
def module_experiment(tmp_path, monkeypatch):
    """Write a tiny Experiment module to a tmp dir and put it on sys.path."""
    import sys

    pkg_dir = tmp_path / "mod_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "kexp.py").write_text(_MODULE_EXPERIMENT_SOURCE)
    monkeypatch.syspath_prepend(str(pkg_dir))
    sys.modules.pop("kexp", None)
    yield "kexp:CliTestKExp"
    sys.modules.pop("kexp", None)


def test_module_callable_escape_hatch(tmp_path, module_experiment):
    out_path = tmp_path / "mod.xlsx"
    out = do_new(
        NewParams(
            output=out_path,
            n=2,
            inputs=[("x", 0.1, 10.0)],
            response_name="y",
            measurement_error=0.1,
            criterion="determinant",
            seed=0,
            n_starts=3,
            module_callable=module_experiment,
            param_initial_guess={"k": 1.5},
        )
    )
    assert out["module_callable"] == module_experiment
    assert out["template"] is None
    assert out["parameter_names"] == ["k"]
    assert out["designs"][0]["x"] == pytest.approx(10.0, abs=1e-3)


def test_module_fit_refused(tmp_path, module_experiment):
    out_path = tmp_path / "mod.xlsx"
    do_new(
        NewParams(
            output=out_path,
            n=2,
            inputs=[("x", 0.1, 10.0)],
            response_name="y",
            measurement_error=0.1,
            criterion="determinant",
            seed=0,
            n_starts=3,
            module_callable=module_experiment,
            param_initial_guess={"k": 1.5},
        )
    )
    # Fill responses so fit reaches the template-check guard, not the
    # no-data guard.
    _fill_response(out_path, "y", lambda row: 1.5 * row["x"])
    with pytest.raises(DoEError, match="not yet implemented"):
        do_fit({"workbook": str(out_path)})


# ──────────────────────────────────────────────────────────────────
# Failure modes
# ──────────────────────────────────────────────────────────────────


def test_new_rejects_wrong_input_count(tmp_path):
    with pytest.raises((DoEError, ValueError), match="exactly two inputs"):
        do_new(
            _new_params(
                tmp_path,
                template="response-surface-2d",
                inputs=[("x", 0.0, 10.0)],
                n=4,
            )
        )


def test_new_rejects_both_template_and_module(tmp_path):
    with pytest.raises(DoEError, match="either --template or --module"):
        do_new(
            NewParams(
                output=tmp_path / "x.xlsx",
                n=1,
                inputs=[("x", 0, 1)],
                response_name="y",
                measurement_error=1.0,
                criterion="determinant",
                seed=0,
                n_starts=2,
                template="linear",
                module_callable="some.mod:thing",
            )
        )


def test_fit_no_completed_runs(tmp_path):
    out = do_new(_new_params(tmp_path, template="linear", inputs=[("x", 0, 1)], n=2))
    with pytest.raises(DoEError, match="no completed runs"):
        do_fit({"workbook": out["workbook_path"]})


def test_status_missing_workbook(tmp_path):
    with pytest.raises(FileNotFoundError):
        do_status({"workbook": str(tmp_path / "nope.xlsx")})


# ──────────────────────────────────────────────────────────────────
# Workbook persistence round-trip
# ──────────────────────────────────────────────────────────────────


def test_workbook_metadata_roundtrip(tmp_path):
    out = do_new(
        _new_params(
            tmp_path,
            template="response-surface-2d",
            inputs=[("x1", 0.0, 10.0), ("x2", -5.0, 5.0)],
            n=4,
            error=0.5,
        )
    )
    wb = Workbook.open(out["workbook_path"])
    assert wb.template_name() == "response-surface-2d"
    assert wb.response_name() == "y"
    assert wb.measurement_error() == 0.5
    specs = wb.input_specs()
    assert [(s.name, s.lb, s.ub) for s in specs] == [
        ("x1", 0.0, 10.0),
        ("x2", -5.0, 5.0),
    ]
    _, names = wb.rebuild_experiment()
    assert names == ["b0", "b1", "b2", "b11", "b22", "b12"]


def test_fim_persisted_and_used_by_extend(tmp_path):
    inputs = [("x1", 0.0, 10.0), ("x2", -5.0, 5.0)]
    out = do_new(
        _new_params(tmp_path, template="response-surface-2d", inputs=inputs, n=6, error=0.5)
    )
    wb_path = Path(out["workbook_path"])

    def predict(row):
        return 1 + 0.5 * row["x1"] - 0.3 * row["x2"]

    _fill_response(wb_path, "y", predict)
    do_fit({"workbook": str(wb_path)})

    wb = Workbook.open(wb_path)
    fim_data = wb.read_fim()
    assert fim_data is not None
    fim, names = fim_data
    assert names == ["b0", "b1", "b2", "b11", "b22", "b12"]
    eigvals = np.linalg.eigvalsh(fim)
    assert (eigvals > 0).all()
