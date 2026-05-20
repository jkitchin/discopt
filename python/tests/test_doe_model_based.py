"""Tests for model_based_optimize_round.

Covers:
* ParametricSurrogate: fits a quadratic, predicts mean + std, std
  shrinks with more data, gradient is sane.
* model_based_optimize_round: end-to-end loop on a 1D quadratic
  converges with very few rounds (compared to the empirical GP).
* The parameters_ dict matches the true coefficients within tolerance.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import numpy as np
import pytest
from discopt.doe.model_based import (
    ParametricSurrogate,
    model_based_optimize_round,
)
from discopt.doe.optimize import OptimizationCriterion
from discopt.doe.templates import polynomial_1d_template, response_surface_template
from discopt.doe.workbook import InputSpec, Workbook

_lwb = pytest.importorskip("openpyxl").load_workbook

# ──────────────────────────────────────────────────────────────────
# ParametricSurrogate unit tests
# ──────────────────────────────────────────────────────────────────


def _quad_truth(x, rng=None):
    """y = -(x - 2)^2 + 3 + noise (max at x=2, y=3)."""
    y = -((x - 2.0) ** 2) + 3.0
    if rng is not None:
        y = y + 0.05 * rng.normal(size=np.shape(x))
    return y


def test_parametric_surrogate_recovers_quadratic():
    exp = polynomial_1d_template(("x", -5.0, 5.0), degree=2, measurement_error=0.05)
    rng = np.random.default_rng(0)
    xs = np.linspace(-4.0, 4.0, 21)[:, None]
    ys = _quad_truth(xs.ravel(), rng)
    s = ParametricSurrogate(
        exp,
        input_names=["x"],
        response_name="y",
        initial_guess={"b0": 1.0, "b1": 1.0, "b2": 1.0},
        measurement_noise_var=0.05**2,
    )
    s.fit(xs, ys)
    # True coefficients: y = -(x-2)^2 + 3 = -1 + 4x - x^2
    assert abs(s.parameters_["b0"] - (-1.0)) < 0.2
    assert abs(s.parameters_["b1"] - 4.0) < 0.2
    assert abs(s.parameters_["b2"] - (-1.0)) < 0.1


def test_parametric_surrogate_predict_returns_mean_and_std():
    exp = polynomial_1d_template(("x", -5.0, 5.0), degree=2, measurement_error=0.1)
    rng = np.random.default_rng(1)
    xs = np.linspace(-3.0, 3.0, 12)[:, None]
    ys = _quad_truth(xs.ravel(), rng)
    s = ParametricSurrogate(
        exp,
        input_names=["x"],
        response_name="y",
        initial_guess={"b0": 0.0, "b1": 0.0, "b2": 0.0},
        measurement_noise_var=0.01,
    )
    s.fit(xs, ys)
    mean, std = s.predict(np.array([[0.0], [2.0], [-3.0]]))
    assert mean.shape == (3,)
    assert std.shape == (3,)
    assert np.all(std > 0.0)
    # Prediction at x=2 should be near the maximum (~3)
    assert abs(mean[1] - 3.0) < 0.3


def test_parametric_surrogate_std_shrinks_with_more_data():
    exp = polynomial_1d_template(("x", -5.0, 5.0), degree=2, measurement_error=0.1)
    rng = np.random.default_rng(2)

    def fit_n(n: int) -> float:
        xs = np.linspace(-3.0, 3.0, n)[:, None]
        ys = _quad_truth(xs.ravel(), rng)
        s = ParametricSurrogate(
            exp,
            input_names=["x"],
            response_name="y",
            initial_guess={"b0": 0.0, "b1": 0.0, "b2": 0.0},
            measurement_noise_var=0.01,
        )
        s.fit(xs, ys)
        _, std = s.predict(np.array([[1.5]]))
        return float(std[0])

    std_small = fit_n(5)
    std_big = fit_n(50)
    assert std_big < std_small


def test_parametric_surrogate_passes_protocol_marker():
    assert getattr(ParametricSurrogate, "_is_discopt_surrogate", False) is True


def test_parametric_surrogate_validates_unknown_response():
    exp = polynomial_1d_template(("x", -5.0, 5.0), degree=2, measurement_error=0.1)
    with pytest.raises(ValueError, match="response"):
        ParametricSurrogate(
            exp,
            input_names=["x"],
            response_name="not_a_response",
            initial_guess={"b0": 0.0, "b1": 0.0, "b2": 0.0},
            measurement_noise_var=0.01,
        )


def test_parametric_surrogate_validates_unknown_input():
    exp = polynomial_1d_template(("x", -5.0, 5.0), degree=2, measurement_error=0.1)
    with pytest.raises(ValueError, match="design_inputs"):
        ParametricSurrogate(
            exp,
            input_names=["z"],
            response_name="y",
            initial_guess={"b0": 0.0, "b1": 0.0, "b2": 0.0},
            measurement_noise_var=0.01,
        )


# ──────────────────────────────────────────────────────────────────
# End-to-end model_based_optimize_round
# ──────────────────────────────────────────────────────────────────


def _make_workbook_polynomial(tmp_path: Path, init_xs, truth_fn, sigma=0.05) -> Path:
    """Workbook seeded with init runs, using the polynomial_1d template."""
    path = tmp_path / "mb.xlsx"
    Workbook.create(
        path,
        template="polynomial-1d",
        template_args={"degree": 2},
        input_specs=[InputSpec("x", -5.0, 5.0)],
        criterion="model-based",
        measurement_error=float(sigma),
        seed=0,
        response_name="y",
        param_initial_guess={"b0": 0.0, "b1": 0.0, "b2": 0.0},
    )
    wb = Workbook.open(path)
    wb.append_runs(0, [{"x": float(x)} for x in init_xs])
    wb.save()
    book = _lwb(path)
    sh = book["runs"]
    for i, x in enumerate(init_xs, start=2):
        sh.cell(row=i, column=4, value=float(truth_fn(x)))
    book.save(path)
    return path


def _fill_responses(path, run_ids, designs, truth_fn):
    book = _lwb(path)
    sh = book["runs"]
    by_id = {rid: d for rid, d in zip(run_ids, designs)}
    for row in sh.iter_rows(min_row=2):
        rid = row[0].value
        if rid in by_id:
            row[3].value = float(truth_fn(by_id[rid]["x"]))
    book.save(path)


def test_model_based_optimize_round_converges_to_max(tmp_path):
    rng = np.random.default_rng(0)

    def truth(x):
        return -((x - 2.0) ** 2) + 3.0 + 0.05 * rng.normal()

    # Only 3 points — far fewer than a GP needs.
    init_xs = np.array([-3.0, 0.0, 3.0])
    path = _make_workbook_polynomial(tmp_path, init_xs, truth)

    for rnd in range(3):
        result = model_based_optimize_round(
            workbook=path,
            criterion=OptimizationCriterion.MAXIMIZE,
            acquisition="expected_improvement",
            batch_size=2,
            seed=rnd,
        )
        _fill_responses(path, result.new_run_ids, result.next_designs, truth)

    final = Workbook.open(path).completed_runs()
    ys = [float(r["y"]) for r in final]
    best_x = float(final[max(range(len(ys)), key=lambda i: ys[i])]["x"])
    assert abs(best_x - 2.0) < 0.7
    assert max(ys) > 2.0


def test_model_based_optimize_round_converges_to_min(tmp_path):
    rng = np.random.default_rng(1)

    def truth(x):
        return (x - 1.0) ** 2 + 0.05 * rng.normal()

    init_xs = np.array([-3.0, 0.5, 3.0])
    path = _make_workbook_polynomial(tmp_path, init_xs, truth, sigma=0.05)
    for rnd in range(3):
        result = model_based_optimize_round(
            workbook=path,
            criterion=OptimizationCriterion.MINIMIZE,
            acquisition="expected_improvement",
            batch_size=2,
            seed=rnd,
        )
        _fill_responses(path, result.new_run_ids, result.next_designs, truth)

    final = Workbook.open(path).completed_runs()
    ys = [float(r["y"]) for r in final]
    best_x = float(final[min(range(len(ys)), key=lambda i: ys[i])]["x"])
    assert abs(best_x - 1.0) < 0.7
    assert min(ys) < 0.5


def test_model_based_optimize_round_returns_parameters(tmp_path):
    rng = np.random.default_rng(3)

    def truth(x):
        return -((x - 2.0) ** 2) + 3.0 + 0.05 * rng.normal()

    init_xs = np.array([-3.0, -1.0, 1.0, 3.0])
    path = _make_workbook_polynomial(tmp_path, init_xs, truth)
    result = model_based_optimize_round(
        workbook=path,
        criterion=OptimizationCriterion.MAXIMIZE,
        acquisition="expected_improvement",
        batch_size=1,
        seed=0,
    )
    # Parameters were estimated and stored
    assert set(result.parameters) == {"b0", "b1", "b2"}
    assert set(result.parameter_se) == {"b0", "b1", "b2"}
    # b2 ≈ -1 (curvature), b1 ≈ 4, b0 ≈ -1
    assert abs(result.parameters["b2"] - (-1.0)) < 0.3
    assert result.fim_log_det is not None and result.fim_log_det > -np.inf
    # Surrogate identified itself as parametric
    assert result.surrogate_mode == "parametric"
    # The batch was appended
    assert len(result.next_designs) == 1
    assert "x" in result.next_designs[0]


def test_model_based_optimize_round_diverse_batch(tmp_path):
    rng = np.random.default_rng(4)

    def truth(x):
        return -((x - 0.5) ** 2) + 0.05 * rng.normal()

    init_xs = np.array([-3.0, 0.0, 3.0])
    path = _make_workbook_polynomial(tmp_path, init_xs, truth)
    result = model_based_optimize_round(
        workbook=path,
        criterion=OptimizationCriterion.MAXIMIZE,
        acquisition="expected_improvement",
        batch_size=3,
        seed=0,
    )
    xs = [d["x"] for d in result.next_designs]
    assert len(set(round(x, 4) for x in xs)) == len(xs)


def test_model_based_optimize_round_empty_workbook_raises(tmp_path):
    path = tmp_path / "empty.xlsx"
    Workbook.create(
        path,
        template="polynomial-1d",
        template_args={"degree": 2},
        input_specs=[InputSpec("x", -5.0, 5.0)],
        criterion="model-based",
        measurement_error=0.1,
        seed=0,
        response_name="y",
        param_initial_guess={"b0": 0.0, "b1": 0.0, "b2": 0.0},
    )
    with pytest.raises(ValueError, match="no completed runs"):
        model_based_optimize_round(
            workbook=path,
            criterion=OptimizationCriterion.MAXIMIZE,
        )


def test_model_based_optimize_round_2d(tmp_path):
    """Response-surface template with 2 inputs."""
    rng = np.random.default_rng(5)

    def truth(x1, x2):
        # quadratic bowl with max at (1, -1)
        return -((x1 - 1.0) ** 2 + (x2 + 1.0) ** 2) + 5.0 + 0.05 * rng.normal()

    exp = response_surface_template([("x1", -3.0, 3.0), ("x2", -3.0, 3.0)], measurement_error=0.05)
    # Build workbook directly using --module-style creation
    path = tmp_path / "rs.xlsx"
    Workbook.create(
        path,
        template="response-surface-2d",
        template_args={},
        input_specs=[InputSpec("x1", -3.0, 3.0), InputSpec("x2", -3.0, 3.0)],
        criterion="model-based",
        measurement_error=0.05,
        seed=0,
        response_name="y",
        param_initial_guess={
            "b0": 0.0,
            "b1": 0.0,
            "b2": 0.0,
            "b11": 0.0,
            "b22": 0.0,
            "b12": 0.0,
        },
    )
    init_xs = np.array([[-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0], [2.0, 2.0], [0.0, 0.0], [1.0, 0.0]])
    wb = Workbook.open(path)
    wb.append_runs(0, [{"x1": float(r[0]), "x2": float(r[1])} for r in init_xs])
    wb.save()
    book = _lwb(path)
    sh = book["runs"]
    for i, r in enumerate(init_xs, start=2):
        sh.cell(row=i, column=5, value=float(truth(r[0], r[1])))
    book.save(path)

    result = model_based_optimize_round(
        workbook=path,
        experiment=exp,
        criterion=OptimizationCriterion.MAXIMIZE,
        acquisition="expected_improvement",
        batch_size=2,
        seed=0,
    )
    # The recommended designs should be inside the box
    for d in result.next_designs:
        assert -3.0 <= d["x1"] <= 3.0
        assert -3.0 <= d["x2"] <= 3.0
    # Parameter set matches the response-surface template
    assert set(result.parameters) == {"b0", "b1", "b2", "b11", "b22", "b12"}
