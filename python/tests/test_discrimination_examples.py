"""Canonical worked-example tests for model discrimination.

* **Arrhenius vs Eyring** — both predict :math:`k(T)` for a chemical
  rate; their predictions agree at one temperature and diverge
  elsewhere. Optimal discrimination design picks the temperature
  furthest from the agreement point that lies inside the design bounds.
* **First-order vs second-order kinetics** — ``-dC/dt = k·C`` versus
  ``-dC/dt = k·C²``; integrated forms diverge most at high
  concentration / early time.

These double as smoke tests on the canonical use case and integration
tests across Layer A + Layer B.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import (
    DiscriminationCriterion,
    discriminate_design,
    model_selection,
)
from discopt.estimate import Experiment, ExperimentModel, estimate_parameters

# ─────────────────────────────────────────────────────────────────────
# Arrhenius vs Eyring
# ─────────────────────────────────────────────────────────────────────


_R = 8.314  # J/(mol·K)
_kB_h = 2.0837e10  # k_B / h ≈ 2.08e10 1/(K·s)


class ArrheniusExp(Experiment):
    """``k(T) = A · exp(−E_a / (R·T))``. Parameters: log_A, E_a."""

    def create_model(self, **kw):
        m = dm.Model("arrhenius")
        log_A = m.continuous("log_A", lb=-5.0, ub=30.0)
        Ea = m.continuous("Ea", lb=0.0, ub=2.0e5)
        T = m.continuous("T", lb=250.0, ub=900.0)
        rate = dm.exp(log_A - Ea / (_R * T))
        return ExperimentModel(
            m,
            unknown_parameters={"log_A": log_A, "Ea": Ea},
            design_inputs={"T": T},
            responses={"k": rate},
            measurement_error={"k": 1e-3},
        )


class EyringExp(Experiment):
    """``k(T) = (k_B/h)·T·exp(ΔS/R)·exp(−ΔH/(R·T))``. Parameters: dS, dH."""

    def create_model(self, **kw):
        m = dm.Model("eyring")
        dS = m.continuous("dS", lb=-200.0, ub=200.0)
        dH = m.continuous("dH", lb=0.0, ub=2.0e5)
        T = m.continuous("T", lb=250.0, ub=900.0)
        rate = _kB_h * T * dm.exp(dS / _R) * dm.exp(-dH / (_R * T))
        return ExperimentModel(
            m,
            unknown_parameters={"dS": dS, "dH": dH},
            design_inputs={"T": T},
            responses={"k": rate},
            measurement_error={"k": 1e-3},
        )


def _matched_at(T_ref: float):
    """Return Arrhenius and Eyring parameter values that produce the
    same ``k(T_ref)`` so the prediction disagreement is purely from
    the temperature pre-factor in Eyring elsewhere.
    """
    Ea = 80_000.0
    A = 1e6
    log_A = float(np.log(A))
    dS = _R * np.log(A / (_kB_h * T_ref))
    dH = Ea
    return {"log_A": log_A, "Ea": Ea}, {"dS": dS, "dH": dH}


class TestArrheniusVsEyring:
    def test_designs_at_temperature_extreme(self):
        """When the models agree at T_ref=500 K, the optimal design
        should sit at one of the temperature bounds, where the Eyring
        ``T`` pre-factor diverges most from Arrhenius.
        """
        ar_pe, ey_pe = _matched_at(500.0)
        result = discriminate_design(
            experiments={"arrhenius": ArrheniusExp(), "eyring": EyringExp()},
            param_estimates={"arrhenius": ar_pe, "eyring": ey_pe},
            design_bounds={"T": (300.0, 800.0)},
            criterion=DiscriminationCriterion.BF,
            n_starts=8,
            seed=0,
        )
        chosen = result.design["T"]
        assert chosen == pytest.approx(300.0, abs=20.0) or chosen == pytest.approx(
            800.0, abs=20.0
        ), f"chosen T {chosen} not at a bound"

    def test_hr_and_bf_agree_qualitatively(self):
        """HR and BF should pick the same end of the temperature range."""
        ar_pe, ey_pe = _matched_at(500.0)
        ex = {"arrhenius": ArrheniusExp(), "eyring": EyringExp()}
        pe = {"arrhenius": ar_pe, "eyring": ey_pe}
        bounds = {"T": (300.0, 800.0)}
        r_hr = discriminate_design(
            ex, pe, bounds, criterion=DiscriminationCriterion.HR, n_starts=8, seed=0
        )
        r_bf = discriminate_design(
            ex, pe, bounds, criterion=DiscriminationCriterion.BF, n_starts=8, seed=0
        )
        # Both at the same bound (within 50 K).
        assert abs(r_hr.design["T"] - r_bf.design["T"]) < 50.0


# ─────────────────────────────────────────────────────────────────────
# First-order vs second-order kinetics — design + estimate at a fixed C0
# ─────────────────────────────────────────────────────────────────────


# C0 is baked in at construction so estimate_parameters only fits k.
class FirstOrderKinetics(Experiment):
    """``C(t) = C0·exp(−k·t)`` at a fixed pre-set ``C0``."""

    def __init__(self, t_grid, C0):
        self.t_grid = list(t_grid)
        self.C0 = float(C0)

    def create_model(self, **kw):
        m = dm.Model("first")
        k = m.continuous("k", lb=0.001, ub=10.0)
        responses = {f"C_{i}": self.C0 * dm.exp(-k * t) for i, t in enumerate(self.t_grid)}
        errors = {kn: 0.05 for kn in responses}
        return ExperimentModel(
            m,
            unknown_parameters={"k": k},
            design_inputs={},
            responses=responses,
            measurement_error=errors,
        )


class SecondOrderKinetics(Experiment):
    """``C(t) = C0/(1+k·C0·t)`` at a fixed pre-set ``C0``."""

    def __init__(self, t_grid, C0):
        self.t_grid = list(t_grid)
        self.C0 = float(C0)

    def create_model(self, **kw):
        m = dm.Model("second")
        k = m.continuous("k", lb=0.001, ub=10.0)
        responses = {f"C_{i}": self.C0 / (1.0 + k * self.C0 * t) for i, t in enumerate(self.t_grid)}
        errors = {kn: 0.05 for kn in responses}
        return ExperimentModel(
            m,
            unknown_parameters={"k": k},
            design_inputs={},
            responses=responses,
            measurement_error=errors,
        )


# Variant for design-time exploration: C0 enters as a design input on a
# single-point response so we can choose C0 by maximising discrimination.
class FirstOrderDesignProbe(Experiment):
    def create_model(self, **kw):
        m = dm.Model("first_probe")
        k = m.continuous("k", lb=0.001, ub=10.0)
        C0 = m.continuous("C0", lb=0.1, ub=10.0)
        # Two probe points at fixed times
        responses = {
            "C_t1": C0 * dm.exp(-k * 1.0),
            "C_t2": C0 * dm.exp(-k * 2.0),
        }
        return ExperimentModel(
            m,
            unknown_parameters={"k": k},
            design_inputs={"C0": C0},
            responses=responses,
            measurement_error={"C_t1": 0.05, "C_t2": 0.05},
        )


class SecondOrderDesignProbe(Experiment):
    def create_model(self, **kw):
        m = dm.Model("second_probe")
        k = m.continuous("k", lb=0.001, ub=10.0)
        C0 = m.continuous("C0", lb=0.1, ub=10.0)
        responses = {
            "C_t1": C0 / (1.0 + k * C0 * 1.0),
            "C_t2": C0 / (1.0 + k * C0 * 2.0),
        }
        return ExperimentModel(
            m,
            unknown_parameters={"k": k},
            design_inputs={"C0": C0},
            responses=responses,
            measurement_error={"C_t1": 0.05, "C_t2": 0.05},
        )


class TestFirstVsSecondOrderKinetics:
    def test_designs_at_high_C0(self):
        """First- and second-order responses agree at low ``C0`` and
        diverge at high ``C0``; the discriminator should push to the
        upper bound.
        """
        result = discriminate_design(
            experiments={"first": FirstOrderDesignProbe(), "second": SecondOrderDesignProbe()},
            param_estimates={"first": {"k": 0.5}, "second": {"k": 0.5}},
            design_bounds={"C0": (0.5, 5.0)},
            criterion=DiscriminationCriterion.BF,
            n_starts=8,
            seed=0,
        )
        assert result.design["C0"] == pytest.approx(5.0, abs=0.05)

    def test_post_experiment_aic_picks_first_order(self):
        """Generate data from first-order; AIC prefers first-order over
        second-order."""
        np.random.seed(0)
        t_grid = np.linspace(0.1, 5.0, 10)
        k_true = 0.5
        C0 = 2.0
        first = FirstOrderKinetics(t_grid, C0=C0)
        second = SecondOrderKinetics(t_grid, C0=C0)
        data = {
            f"C_{i}": float(C0 * np.exp(-k_true * t) + np.random.normal(0, 0.05))
            for i, t in enumerate(t_grid)
        }
        est_first = estimate_parameters(first, data, initial_guess={"k": 0.5})
        est_second = estimate_parameters(second, data, initial_guess={"k": 0.1})
        sel = model_selection({"first": est_first, "second": est_second}, method="aic")
        assert sel.best_model == "first"
        assert sel.weights["first"] > sel.weights["second"]


# ─────────────────────────────────────────────────────────────────────
# Round-trip pipeline consistency (Layer A → Layer B)
# ─────────────────────────────────────────────────────────────────────


class TestPipelineConsistency:
    """The discrimination tool picks a design point; data simulated
    from the true model at that point lets the selection layer
    correctly identify the true model.
    """

    def test_design_then_select(self):
        np.random.seed(1)
        t_grid = np.linspace(0.5, 3.0, 6)
        k_true = 0.5

        # Step 1: pick a discriminating C0 with the probe experiments.
        design = discriminate_design(
            experiments={"first": FirstOrderDesignProbe(), "second": SecondOrderDesignProbe()},
            param_estimates={"first": {"k": k_true}, "second": {"k": k_true}},
            design_bounds={"C0": (0.5, 5.0)},
            criterion=DiscriminationCriterion.BF,
            n_starts=5,
            seed=0,
        )
        C0_use = float(design.design["C0"])

        # Step 2: simulate data from the true (first-order) model.
        first = FirstOrderKinetics(t_grid, C0=C0_use)
        second = SecondOrderKinetics(t_grid, C0=C0_use)
        data = {
            f"C_{i}": float(C0_use * np.exp(-k_true * t) + np.random.normal(0, 0.05))
            for i, t in enumerate(t_grid)
        }

        # Step 3: fit and select.
        est_first = estimate_parameters(first, data, initial_guess={"k": k_true})
        est_second = estimate_parameters(second, data, initial_guess={"k": 0.1})
        sel = model_selection({"first": est_first, "second": est_second}, method="aic")

        assert sel.best_model == "first"
        assert sel.weights["first"] > sel.weights["second"]
