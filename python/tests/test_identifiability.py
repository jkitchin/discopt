"""Tests for diagnose_identifiability (Piece 1 of issue #45).

Covers:
- Backwards compatibility of check_identifiability.
- New Belsley/Gutenkunst fields on identifiable linear model.
- The canonical a*b*x + c*x^2 fixture where a*b and c are identifiable
  but a and b individually are not. This fixture pins down the
  factor-of-2 VDP transpose, the VIF infinity handling, the null-space
  sign convention, and the Gutenkunst spectrum shape all at once.
- The legacy collinear-regressors model with known closed-form VIFs.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import (
    IdentifiabilityDiagnostics,
    IdentifiabilityResult,
    check_identifiability,
    diagnose_identifiability,
)
from discopt.estimate import Experiment, ExperimentModel


class LinearExperiment(Experiment):
    """y_i = a + b*x_i; fully identifiable at any x_data with >= 2 distinct x."""

    def __init__(self, x_data):
        self.x_data = x_data

    def create_model(self, **kwargs):
        m = dm.Model("linear")
        a = m.continuous("a", lb=-20, ub=20)
        b = m.continuous("b", lb=-20, ub=20)
        responses = {f"y_{i}": a + b * xi for i, xi in enumerate(self.x_data)}
        errors = {k: 1.0 for k in responses}
        return ExperimentModel(m, {"a": a, "b": b}, {}, responses, errors)


class AbUnidExperiment(Experiment):
    """y_i = a*b*x_i + c*x_i^2; only (a*b) and c are identifiable."""

    def __init__(self, x_data):
        self.x_data = x_data

    def create_model(self, **kwargs):
        m = dm.Model("ab_unid")
        a = m.continuous("a", lb=0.1, ub=10)
        b = m.continuous("b", lb=0.1, ub=10)
        c = m.continuous("c", lb=-10, ub=10)
        responses = {f"y_{i}": a * b * xi + c * xi * xi for i, xi in enumerate(self.x_data)}
        errors = {k: 1.0 for k in responses}
        return ExperimentModel(m, {"a": a, "b": b, "c": c}, {}, responses, errors)


class CollinearExperiment(Experiment):
    """Two columns with controllable correlation rho.

    y_i = b1*z1_i + b2*z2_i where z2 = rho*z1 + sqrt(1-rho^2)*w,
    with z1 and w orthonormal in the design. The analytic VIF for b1
    and b2 is 1 / (1 - rho^2) exactly, regardless of scale.
    """

    def __init__(self, rho, n_points=10, seed=0):
        self.rho = rho
        rng = np.random.default_rng(seed)
        z1 = rng.standard_normal(n_points)
        z1 = (z1 - z1.mean()) / z1.std()
        w = rng.standard_normal(n_points)
        w = w - (w @ z1) * z1 / (z1 @ z1)
        w = (w - w.mean()) / w.std()
        z2 = rho * z1 + np.sqrt(1.0 - rho**2) * w
        self.z1 = z1
        self.z2 = z2

    def create_model(self, **kwargs):
        m = dm.Model("collinear")
        b1 = m.continuous("b1", lb=-20, ub=20)
        b2 = m.continuous("b2", lb=-20, ub=20)
        responses = {f"y_{i}": b1 * self.z1[i] + b2 * self.z2[i] for i in range(len(self.z1))}
        errors = {k: 1.0 for k in responses}
        return ExperimentModel(m, {"b1": b1, "b2": b2}, {}, responses, errors)


class TestBackwardsCompatibility:
    def test_check_identifiability_still_returns_old_type(self):
        exp = LinearExperiment([1.0, 2.0, 3.0, 4.0])
        res = check_identifiability(exp, {"a": 1.0, "b": 1.0})
        assert isinstance(res, IdentifiabilityResult)
        assert res.is_identifiable is True
        assert res.fim_rank == 2
        assert res.n_parameters == 2
        assert res.problematic_parameters == []
        assert np.isfinite(res.condition_number)

    def test_check_identifiability_unid_flags_problematic(self):
        exp = AbUnidExperiment([1.0, 2.0, 3.0, 4.0])
        res = check_identifiability(exp, {"a": 1.0, "b": 1.0, "c": 1.0})
        assert res.is_identifiable is False
        assert res.fim_rank == 2
        assert res.n_parameters == 3
        assert len(res.problematic_parameters) == 1


class TestDiagnoseIdentifiabilityIdentifiable:
    def test_all_fields_present_and_shaped(self):
        exp = LinearExperiment([1.0, 2.0, 3.0, 4.0])
        diag = diagnose_identifiability(exp, {"a": 1.0, "b": 1.0})
        assert isinstance(diag, IdentifiabilityDiagnostics)
        assert diag.is_identifiable is True
        assert diag.fim_rank == 2
        assert diag.n_parameters == 2
        assert diag.singular_values.shape == (2,)
        assert diag.condition_indices.shape == (2,)
        assert set(diag.vif.keys()) == {"a", "b"}
        assert diag.variance_decomposition.shape == (2, 2)
        assert diag.correlation_matrix.shape == (2, 2)
        assert diag.log_eigenvalue_spectrum.shape == (2,)
        assert diag.normalized_log_spectrum.shape == (2,)
        assert diag.null_space == []
        assert set(diag.standard_errors.keys()) == {"a", "b"}

    def test_vdp_rows_sum_to_one(self):
        """Belsley VDP pi_{jk} should sum to 1 along k for each j."""
        exp = LinearExperiment([1.0, 2.0, 3.0, 4.0])
        diag = diagnose_identifiability(exp, {"a": 1.0, "b": 1.0})
        row_sums = diag.variance_decomposition.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-12)

    def test_normalized_log_spectrum_top_is_zero(self):
        """log10(lambda_max / lambda_max) == 0 by construction."""
        exp = LinearExperiment([1.0, 2.0, 3.0, 4.0])
        diag = diagnose_identifiability(exp, {"a": 1.0, "b": 1.0})
        assert diag.normalized_log_spectrum[0] == pytest.approx(0.0, abs=1e-12)
        assert np.all(diag.normalized_log_spectrum[1:] <= 0.0 + 1e-12)

    def test_condition_index_top_is_one(self):
        exp = LinearExperiment([1.0, 2.0, 3.0, 4.0])
        diag = diagnose_identifiability(exp, {"a": 1.0, "b": 1.0})
        assert diag.condition_indices[0] == pytest.approx(1.0, abs=1e-12)


class TestDiagnoseIdentifiabilityUnidentifiable:
    """The a*b*x + c*x^2 fixture: only a*b and c are identifiable."""

    def _diag(self):
        exp = AbUnidExperiment([1.0, 2.0, 3.0, 4.0])
        return diagnose_identifiability(exp, {"a": 1.0, "b": 1.0, "c": 1.0})

    def test_rank_is_two(self):
        diag = self._diag()
        assert diag.is_identifiable is False
        assert diag.fim_rank == 2
        assert diag.n_parameters == 3

    def test_trailing_singular_value_is_tiny(self):
        diag = self._diag()
        assert diag.singular_values[-1] < 1e-10 * diag.singular_values[0]

    def test_trailing_condition_index_is_huge(self):
        diag = self._diag()
        # Effectively infinite.
        assert diag.condition_indices[-1] > 1e10 or np.isinf(diag.condition_indices[-1])

    def test_vif_of_a_and_b_are_infinite(self):
        diag = self._diag()
        assert np.isinf(diag.vif["a"])
        assert np.isinf(diag.vif["b"])

    def test_vif_of_c_is_finite(self):
        """VIF[c] is finite (c is identifiable) even though moderate
        correlation with the (a, b) direction is expected: for x=[1..4]
        the vectors [x] and [x^2] have correlation ~0.97.
        """
        diag = self._diag()
        assert np.isfinite(diag.vif["c"])

    def test_null_space_is_1d_and_points_along_a_minus_b(self):
        diag = self._diag()
        assert len(diag.null_space) == 1
        direction = diag.null_space[0]
        # Should be roughly (a: +1/sqrt(2), b: -1/sqrt(2), c: 0), or sign-flipped.
        # Sign normalization guarantees the largest-magnitude entry positive.
        assert abs(direction["c"]) < 1e-6
        assert abs(abs(direction["a"]) - 1 / np.sqrt(2)) < 1e-6
        assert abs(abs(direction["b"]) - 1 / np.sqrt(2)) < 1e-6
        # Sign normalization: largest-magnitude entry positive.
        max_entry = max(direction, key=lambda k: abs(direction[k]))
        assert direction[max_entry] > 0

    def test_standard_errors_nan_for_unid_parameters(self):
        diag = self._diag()
        assert np.isnan(diag.standard_errors["a"])
        assert np.isnan(diag.standard_errors["b"])
        assert np.isfinite(diag.standard_errors["c"])

    def test_correlation_matrix_has_nan_on_unid_rows(self):
        diag = self._diag()
        names = ["a", "b", "c"]
        a_idx = names.index("a")
        b_idx = names.index("b")
        c_idx = names.index("c")
        assert np.isnan(diag.correlation_matrix[a_idx, c_idx])
        assert np.isnan(diag.correlation_matrix[b_idx, c_idx])
        # c's self-correlation should still be 1.
        assert diag.correlation_matrix[c_idx, c_idx] == pytest.approx(1.0, abs=1e-12)

    def test_warnings_flag_collinearity_and_vif(self):
        diag = self._diag()
        text = " ".join(diag.warnings).lower()
        assert "collinearity" in text or "condition index" in text
        assert "vif[a]" in text
        assert "vif[b]" in text


class TestDiagnoseIdentifiabilityCollinear:
    """Controlled collinearity: two regressors with correlation rho.

    The Belsley VIF for each regressor should be 1 / (1 - rho^2).
    """

    @pytest.mark.parametrize("rho", [0.3, 0.7, 0.9])
    def test_vif_matches_analytic(self, rho):
        exp = CollinearExperiment(rho=rho, n_points=20, seed=0)
        diag = diagnose_identifiability(exp, {"b1": 1.0, "b2": 1.0})
        expected = 1.0 / (1.0 - rho**2)
        np.testing.assert_allclose(diag.vif["b1"], expected, rtol=1e-6)
        np.testing.assert_allclose(diag.vif["b2"], expected, rtol=1e-6)


class TestSloppyModelSpectrum:
    def test_spectrum_monotone_nonincreasing(self):
        exp = LinearExperiment([1.0, 2.0, 3.0, 4.0, 5.0])
        diag = diagnose_identifiability(exp, {"a": 1.0, "b": 1.0})
        diffs = np.diff(diag.log_eigenvalue_spectrum)
        assert np.all(diffs <= 1e-12)

    def test_normalized_spectrum_decades_below_zero(self):
        exp = LinearExperiment([1.0, 2.0, 3.0, 4.0])
        diag = diagnose_identifiability(exp, {"a": 1.0, "b": 1.0})
        # First entry should be exactly 0 (lambda_max / lambda_max).
        assert diag.normalized_log_spectrum[0] == pytest.approx(0.0, abs=1e-12)
        # Subsequent entries non-positive.
        assert np.all(diag.normalized_log_spectrum <= 1e-12)
