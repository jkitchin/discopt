"""Tests for estimability_rank, collinearity_index, d_optimal_subset
(Piece 2 of issue #45).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import (
    EstimabilityResult,
    collinearity_index,
    d_optimal_subset,
    estimability_rank,
)
from discopt.estimate import Experiment, ExperimentModel


class LinearExperiment(Experiment):
    """y_i = a + b*x_i."""

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
    """y_i = a*b*x_i + c*x_i^2; only a*b and c identifiable."""

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


class OrthogonalExperiment(Experiment):
    """Three orthogonal regressors:

    y_i = p1*z1_i + p2*z2_i + p3*z3_i

    with designed-orthogonal z's so the Gram matrix is diag(1, 1, 1)
    after scaling. Useful for consistency checks between selection
    algorithms.
    """

    def __init__(self):
        # Rows of (z1, z2, z3) -- explicit orthonormal columns.
        rows = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0],
            ]
        )
        self.rows = rows

    def create_model(self, **kwargs):
        m = dm.Model("ortho")
        p1 = m.continuous("p1", lb=-10, ub=10)
        p2 = m.continuous("p2", lb=-10, ub=10)
        p3 = m.continuous("p3", lb=-10, ub=10)
        responses = {}
        errors = {}
        for i, (z1, z2, z3) in enumerate(self.rows):
            responses[f"y_{i}"] = p1 * z1 + p2 * z2 + p3 * z3
            errors[f"y_{i}"] = 1.0
        return ExperimentModel(m, {"p1": p1, "p2": p2, "p3": p3}, {}, responses, errors)


class TestEstimabilityRank:
    def test_linear_fully_identifiable(self):
        exp = LinearExperiment([1.0, 2.0, 3.0, 4.0])
        res = estimability_rank(exp, {"a": 1.0, "b": 1.0})
        assert isinstance(res, EstimabilityResult)
        assert set(res.ranking) == {"a", "b"}
        assert len(res.recommended_subset) == 2
        # Projected norms sorted by selection order: descending.
        assert res.projected_norms[0] >= res.projected_norms[1]

    def test_unid_puts_redundant_last(self):
        """a*b*x + c*x^2: rank must be 2; the third parameter has
        projected norm ~0 and is excluded from the recommended subset.
        """
        exp = AbUnidExperiment([1.0, 2.0, 3.0, 4.0])
        res = estimability_rank(exp, {"a": 1.0, "b": 1.0, "c": 1.0})
        # Last projected norm ~0.
        assert res.projected_norms[-1] < 1e-6 * res.projected_norms[0]
        # Recommended subset has exactly 2 elements.
        assert len(res.recommended_subset) == 2
        # c is in the recommended subset (dominant projected norm).
        assert "c" in res.recommended_subset

    def test_parameter_scales_override(self):
        """Passing explicit parameter_scales should not alter ranking
        for an already-well-scaled problem but should not crash either.
        """
        exp = LinearExperiment([1.0, 2.0, 3.0, 4.0])
        res = estimability_rank(exp, {"a": 1.0, "b": 1.0}, parameter_scales={"a": 2.0, "b": 0.5})
        assert set(res.ranking) == {"a", "b"}


class TestCollinearityIndex:
    def test_orthogonal_design_gamma_is_one(self):
        """For orthonormal regressors (after normalization), gamma_K = 1."""
        exp = OrthogonalExperiment()
        gamma = collinearity_index(exp, {"p1": 1.0, "p2": 1.0, "p3": 1.0}, ["p1", "p2"])
        assert gamma == pytest.approx(1.0, abs=1e-10)

    def test_singular_subset_returns_inf(self):
        """a and b produce the same sensitivity column in a*b*x,
        so {a, b} is perfectly collinear."""
        exp = AbUnidExperiment([1.0, 2.0, 3.0, 4.0])
        gamma = collinearity_index(exp, {"a": 1.0, "b": 1.0, "c": 1.0}, ["a", "b"])
        assert np.isinf(gamma)

    def test_increasing_correlation_increases_gamma(self):
        """For {b, c} in the kinetics fixture, gamma is finite and > 1."""
        exp = AbUnidExperiment([1.0, 2.0, 3.0, 4.0])
        gamma_ac = collinearity_index(exp, {"a": 1.0, "b": 1.0, "c": 1.0}, ["a", "c"])
        assert np.isfinite(gamma_ac) and gamma_ac > 1.0


class TestDOptimalSubset:
    def test_enumerate_and_greedy_agree_on_orthogonal_design(self):
        """With equal column norms and orthogonal columns, any
        two-of-three subset has the same D-criterion. enumerate may
        return a different subset than greedy but both are optimal.
        """
        exp = OrthogonalExperiment()
        pvals = {"p1": 1.0, "p2": 1.0, "p3": 1.0}
        S_enum = d_optimal_subset(exp, pvals, k=2, method="enumerate")
        S_greedy = d_optimal_subset(exp, pvals, k=2, method="greedy")
        assert len(S_enum) == 2 and len(S_greedy) == 2
        # Both subsets should be valid selections of the parameters.
        assert set(S_enum).issubset({"p1", "p2", "p3"})
        assert set(S_greedy).issubset({"p1", "p2", "p3"})

    def test_auto_dispatches_to_enumerate_for_small_p(self):
        """For p=3, auto mode should match enumerate exactly."""
        exp = OrthogonalExperiment()
        pvals = {"p1": 1.0, "p2": 1.0, "p3": 1.0}
        S_auto = d_optimal_subset(exp, pvals, k=2, method="auto")
        S_enum = d_optimal_subset(exp, pvals, k=2, method="enumerate")
        assert set(S_auto) == set(S_enum)

    def test_k_too_large_raises(self):
        exp = OrthogonalExperiment()
        with pytest.raises(ValueError, match="k must be in"):
            d_optimal_subset(exp, {"p1": 1.0, "p2": 1.0, "p3": 1.0}, k=5)

    def test_minlp_method_not_implemented(self):
        exp = OrthogonalExperiment()
        with pytest.raises(NotImplementedError, match="reserved"):
            d_optimal_subset(exp, {"p1": 1.0, "p2": 1.0, "p3": 1.0}, k=2, method="minlp")

    def test_unknown_method_raises(self):
        exp = OrthogonalExperiment()
        with pytest.raises(ValueError, match="Unknown method"):
            d_optimal_subset(exp, {"p1": 1.0, "p2": 1.0, "p3": 1.0}, k=2, method="nope")

    def test_enumerate_excludes_redundant_in_kinetics_fixture(self):
        """For a*b*x + c*x^2 with k=2, the optimal subset must contain c
        (its sensitivity is orthogonal to the a/b ridge) and exactly one
        of (a, b). It must never be {a, b}.
        """
        exp = AbUnidExperiment([1.0, 2.0, 3.0, 4.0])
        S = d_optimal_subset(exp, {"a": 1.0, "b": 1.0, "c": 1.0}, k=2, method="enumerate")
        assert "c" in S
        assert not (set(S) == {"a", "b"})
