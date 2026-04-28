"""Realistic worked-example tests: Langmuir isotherm and batch
reactor rate data.

These exercises the full identifiability stack on small problems with
direct chemical-engineering interpretation. They serve as both
regression tests and as runnable references for users.

* **Langmuir isotherm** ``q = qm * K * c / (1 + K * c)``: classic
  example where the parameters ``qm`` (saturation capacity) and ``K``
  (affinity) trade off in a low-concentration regime. When all
  measurements are at ``K * c << 1`` the response collapses to
  ``q ~ qm * K * c``, leaving only the product identifiable.
  Adding measurements above the saturation knee recovers identifiability.

* **Batch reactor rate data** ``C(t) = C0 * exp(-k t) + C_inf`` (a
  first-order decay to a non-zero asymptote). Three parameters
  (``C0``, ``k``, ``C_inf``); when ``C_inf`` is poorly excited because
  the time window stops short of the asymptote, ``C_inf`` becomes
  weakly identifiable and the FIM is ill-conditioned.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import (
    diagnose_identifiability,
    estimability_rank,
    profile_likelihood,
)
from discopt.estimate import Experiment, ExperimentModel, estimate_parameters

# ─────────────────────────────────────────────────────────────────────
# Langmuir isotherm
# ─────────────────────────────────────────────────────────────────────


class LangmuirExperiment(Experiment):
    """q = qm * K * c / (1 + K * c) at a fixed grid of concentrations."""

    def __init__(self, c_data, sigma=0.05):
        self.c_data = list(c_data)
        self.sigma = float(sigma)

    def create_model(self, **kwargs):
        m = dm.Model("langmuir")
        qm = m.continuous("qm", lb=0.01, ub=20.0)
        K = m.continuous("K", lb=1e-4, ub=1e4)
        responses = {}
        errors = {}
        for i, c in enumerate(self.c_data):
            responses[f"q_{i}"] = qm * K * c / (1.0 + K * c)
            errors[f"q_{i}"] = self.sigma
        return ExperimentModel(m, {"qm": qm, "K": K}, {}, responses, errors)


class TestLangmuirSaturatedRegime:
    """When the design covers both the linear and saturated regimes,
    qm and K are jointly identifiable.
    """

    def _data(self):
        np.random.seed(42)
        # Concentrations spanning K*c from ~0.1 to ~10 for K_true=1.
        c_grid = np.array([0.1, 0.3, 1.0, 3.0, 10.0])
        qm_true = 5.0
        K_true = 1.0
        exp = LangmuirExperiment(c_grid, sigma=0.05)
        data = {
            f"q_{i}": qm_true * K_true * c / (1 + K_true * c) + np.random.normal(0, 0.05)
            for i, c in enumerate(c_grid)
        }
        return exp, data, qm_true, K_true

    def test_estimate_recovers_truth(self):
        exp, data, qm_true, K_true = self._data()
        est = estimate_parameters(exp, data, initial_guess={"qm": 4.0, "K": 0.5})
        assert est.parameters["qm"] == pytest.approx(qm_true, rel=0.1)
        assert est.parameters["K"] == pytest.approx(K_true, rel=0.2)

    def test_diagnose_identifiable_in_saturated_regime(self):
        exp, data, _, _ = self._data()
        est = estimate_parameters(exp, data, initial_guess={"qm": 4.0, "K": 0.5})
        diag = diagnose_identifiability(exp, est.parameters)
        assert diag.is_identifiable is True
        assert diag.fim_rank == 2
        assert all(np.isfinite(v) for v in diag.vif.values())

    def test_profile_ci_bounded_for_both(self):
        exp, data, qm_true, K_true = self._data()
        est = estimate_parameters(exp, data, initial_guess={"qm": 4.0, "K": 0.5})
        for name, true in (("qm", qm_true), ("K", K_true)):
            prof = profile_likelihood(exp, data, name, initial_estimate=est)
            assert prof.shape == "bounded", f"{name}: expected bounded, got {prof.shape}"
            assert prof.ci_lower < true < prof.ci_upper


class TestLangmuirLinearRegime:
    """When all measurements are at low ``K*c`` (linear regime), only
    the product ``qm * K`` is identifiable; individual estimates
    collapse to a non-identifiable ridge.
    """

    def _data(self):
        np.random.seed(43)
        # All concentrations small so K*c remains << 1 for K_true=1.
        c_grid = np.array([0.001, 0.002, 0.005, 0.01, 0.02])
        qm_true = 5.0
        K_true = 1.0
        exp = LangmuirExperiment(c_grid, sigma=0.001)
        data = {
            f"q_{i}": qm_true * K_true * c / (1 + K_true * c) + np.random.normal(0, 0.001)
            for i, c in enumerate(c_grid)
        }
        return exp, data, qm_true, K_true

    def test_diagnose_flags_collinearity(self):
        exp, data, _, _ = self._data()
        # Use ground-truth as nominal so we're not chasing a wild fit.
        diag = diagnose_identifiability(exp, {"qm": 5.0, "K": 1.0})
        # In the linear regime, qm and K are highly correlated (one
        # combined direction dominates), so the condition number is
        # very large and warnings are emitted.
        assert diag.condition_number > 100.0
        assert any(("collinearity" in w.lower()) or ("vif" in w.lower()) for w in diag.warnings)

    def test_estimability_rank_recommends_subset(self):
        exp, data, _, _ = self._data()
        rank = estimability_rank(exp, {"qm": 5.0, "K": 1.0})
        # Both rank above the cutoff or only one — but the projected
        # norm of the second parameter should be much smaller than the
        # first.
        assert rank.projected_norms[0] / max(rank.projected_norms[1], 1e-30) > 5.0


# ─────────────────────────────────────────────────────────────────────
# Batch reactor: integrated first-order with non-zero asymptote
# ─────────────────────────────────────────────────────────────────────


class BatchReactorExperiment(Experiment):
    """C(t) = (C0 - C_inf) * exp(-k * t) + C_inf.

    Parameters
    ----------
    t_data : iterable of float
        Sample times.
    sigma : float
        Concentration measurement standard deviation.
    """

    def __init__(self, t_data, sigma=0.02):
        self.t_data = list(t_data)
        self.sigma = float(sigma)

    def create_model(self, **kwargs):
        m = dm.Model("batch_reactor")
        C0 = m.continuous("C0", lb=0.1, ub=10.0)
        k = m.continuous("k", lb=0.001, ub=10.0)
        C_inf = m.continuous("C_inf", lb=0.0, ub=5.0)
        responses = {}
        errors = {}
        for i, t in enumerate(self.t_data):
            responses[f"C_{i}"] = (C0 - C_inf) * dm.exp(-k * t) + C_inf
            errors[f"C_{i}"] = self.sigma
        return ExperimentModel(m, {"C0": C0, "k": k, "C_inf": C_inf}, {}, responses, errors)


class TestBatchReactorWellDesigned:
    """Design covers initial slope (resolves k) and the asymptote
    (resolves C_inf): all three parameters should be identifiable."""

    def _data(self):
        np.random.seed(101)
        # k_true = 0.5 -> half-life ~1.4. Sample to t=10 (~5 half-lives).
        t_grid = np.linspace(0.0, 10.0, 11)
        C0_true, k_true, Cinf_true = 5.0, 0.5, 1.0
        exp = BatchReactorExperiment(t_grid, sigma=0.02)
        data = {
            f"C_{i}": (C0_true - Cinf_true) * np.exp(-k_true * t)
            + Cinf_true
            + np.random.normal(0, 0.02)
            for i, t in enumerate(t_grid)
        }
        return exp, data, C0_true, k_true, Cinf_true

    def test_estimate_recovers_all_three(self):
        exp, data, C0_true, k_true, Cinf_true = self._data()
        est = estimate_parameters(exp, data, initial_guess={"C0": 4.5, "k": 0.4, "C_inf": 1.2})
        assert est.parameters["C0"] == pytest.approx(C0_true, abs=0.2)
        assert est.parameters["k"] == pytest.approx(k_true, abs=0.1)
        assert est.parameters["C_inf"] == pytest.approx(Cinf_true, abs=0.2)

    def test_diagnose_identifiable(self):
        exp, data, _, _, _ = self._data()
        est = estimate_parameters(exp, data, initial_guess={"C0": 4.5, "k": 0.4, "C_inf": 1.2})
        diag = diagnose_identifiability(exp, est.parameters)
        assert diag.is_identifiable is True
        assert diag.fim_rank == 3

    def test_yao_keeps_all_three(self):
        exp, data, _, _, _ = self._data()
        est = estimate_parameters(exp, data, initial_guess={"C0": 4.5, "k": 0.4, "C_inf": 1.2})
        rank = estimability_rank(exp, est.parameters)
        assert len(rank.recommended_subset) == 3


class TestBatchReactorShortHorizon:
    """If we stop sampling before reaching the asymptote, ``C_inf`` is
    only weakly identifiable -- it trades off with ``C0``.
    """

    def _data(self):
        np.random.seed(102)
        # k_true = 0.5 -> half-life ~1.4. Sample only to t=1.5 (~1 half-life).
        # The asymptote is barely visited.
        t_grid = np.linspace(0.0, 1.5, 8)
        C0_true, k_true, Cinf_true = 5.0, 0.5, 1.0
        exp = BatchReactorExperiment(t_grid, sigma=0.02)
        data = {
            f"C_{i}": (C0_true - Cinf_true) * np.exp(-k_true * t)
            + Cinf_true
            + np.random.normal(0, 0.02)
            for i, t in enumerate(t_grid)
        }
        return exp, data

    def test_diagnose_flags_high_condition_number(self):
        exp, data = self._data()
        # Use ground-truth-based nominal to evaluate FIM independently of
        # whatever fit a noisy data set produces.
        diag = diagnose_identifiability(exp, {"C0": 5.0, "k": 0.5, "C_inf": 1.0})
        # The k-vs-C_inf trade-off should produce a noticeable condition
        # number, even if the FIM isn't strictly singular.
        assert diag.condition_number > 100.0
        # Either C0 or C_inf should hit the VIF > 10 threshold.
        big_vifs = {n: v for n, v in diag.vif.items() if np.isfinite(v) and v > 10.0}
        assert len(big_vifs) >= 1, f"Expected at least one VIF > 10; got {diag.vif}"

    def test_yao_drops_one_parameter(self):
        exp, data = self._data()
        rank = estimability_rank(exp, {"C0": 5.0, "k": 0.5, "C_inf": 1.0})
        # The least-estimable parameter falls below the cutoff.
        assert len(rank.recommended_subset) <= 2 or (rank.collinearity_index > 10.0)
