"""Tests for the sequential discrimination loop.

The hard part is that each round adds a fresh observation whose
response key must be present in the experiment's response dict on the
next round's fit. We use stateful ``Experiment`` instances that
accumulate observation keys via ``add_observation``.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.doe import (
    DiscriminationCriterion,
    DiscriminationRound,
    sequential_discrimination,
)
from discopt.estimate import Experiment, ExperimentModel

# ─────────────────────────────────────────────────────────────────────
# Stateful experiments: first-order vs second-order kinetics.
# Both share the design inputs (t, C0) used by the proposed design at
# each round.
# ─────────────────────────────────────────────────────────────────────


class _AccumulatingKinetics(Experiment):
    """Common state-handling: ``observations`` is a list of
    ``(label, t_value, C0_value)`` triples, one per gathered datum.
    """

    def __init__(self):
        self.observations: list[tuple[str, float, float]] = []

    def add_observation(self, label: str, t: float, C0: float) -> None:
        self.observations.append((label, float(t), float(C0)))


class FirstOrderExp(_AccumulatingKinetics):
    """``C(t, C0) = C0 · exp(−k·t)``."""

    def create_model(self, **kw):
        m = dm.Model("first_order")
        k = m.continuous("k", lb=0.01, ub=5.0)
        # Each accumulated observation is a fixed-design response.
        responses = {label: C0 * dm.exp(-k * t) for (label, t, C0) in self.observations}
        # Include a single "next observation" with design inputs t, C0
        # so discriminate_design can vary them.
        t_var = m.continuous("t", lb=0.0, ub=20.0)
        C0_var = m.continuous("C0", lb=0.1, ub=5.0)
        responses[_NEXT_KEY] = C0_var * dm.exp(-k * t_var)
        errors = {name: 0.05 for name in responses}
        return ExperimentModel(
            m,
            unknown_parameters={"k": k},
            design_inputs={"t": t_var, "C0": C0_var},
            responses=responses,
            measurement_error=errors,
        )


class SecondOrderExp(_AccumulatingKinetics):
    """``C(t, C0) = C0 / (1 + k·C0·t)``."""

    def create_model(self, **kw):
        m = dm.Model("second_order")
        k = m.continuous("k", lb=0.01, ub=5.0)
        responses = {label: C0 / (1.0 + k * C0 * t) for (label, t, C0) in self.observations}
        t_var = m.continuous("t", lb=0.0, ub=20.0)
        C0_var = m.continuous("C0", lb=0.1, ub=5.0)
        responses[_NEXT_KEY] = C0_var / (1.0 + k * C0_var * t_var)
        errors = {name: 0.05 for name in responses}
        return ExperimentModel(
            m,
            unknown_parameters={"k": k},
            design_inputs={"t": t_var, "C0": C0_var},
            responses=responses,
            measurement_error=errors,
        )


_NEXT_KEY = "C_next_design_slot"


@pytest.fixture
def kinetics_setup():
    """A pair of stateful kinetics experiments with 4 initial points
    generated from the first-order truth (k=0.5).
    """
    np.random.seed(0)
    k_true = 0.5
    initial_points = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 1.0)]
    initial_data: dict[str, float] = {}

    first = FirstOrderExp()
    second = SecondOrderExp()
    for i, (t, C0) in enumerate(initial_points):
        label = f"C_init_{i}"
        first.add_observation(label, t, C0)
        second.add_observation(label, t, C0)
        initial_data[label] = float(C0 * np.exp(-k_true * t) + np.random.normal(0, 0.05))

    return first, second, initial_data, k_true


def _make_simulator(first: FirstOrderExp, second: SecondOrderExp, k_true: float, rng):
    """Build a ``run_experiment`` callback that:

    1. Generates a synthetic observation from the *true* (first-order)
       model at the proposed design.
    2. Adds the observation to *both* experiments' state under a fresh
       label so the next round's models include it.
    3. Returns the new datum keyed by that label.
    """
    counter = {"i": 0}

    def run(design: dict[str, float]) -> dict[str, float]:
        counter["i"] += 1
        label = f"C_round_{counter['i']}"
        t = float(design["t"])
        C0 = float(design["C0"])
        y_true = C0 * np.exp(-k_true * t) + rng.normal(0, 0.05)
        first.add_observation(label, t, C0)
        second.add_observation(label, t, C0)
        return {label: y_true}

    return run


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────


class TestSequentialDiscrimination:
    def test_runs_to_completion(self, kinetics_setup):
        first, second, initial_data, k_true = kinetics_setup
        rng = np.random.default_rng(1)
        runner = _make_simulator(first, second, k_true, rng)

        rounds = sequential_discrimination(
            experiments={"first": first, "second": second},
            initial_data=initial_data,
            design_bounds={"t": (0.5, 10.0), "C0": (0.5, 5.0)},
            n_rounds=4,
            run_experiment=runner,
            criterion=DiscriminationCriterion.BF,
            stop_when_dominant=2.0,  # never stop early via weight
            n_starts=4,
            seed=0,
        )
        assert len(rounds) == 4
        assert all(isinstance(r, DiscriminationRound) for r in rounds)

    def test_first_order_dominates(self, kinetics_setup):
        """Data is generated from first-order; AIC weight of first-order
        should rise above second-order within a few rounds."""
        first, second, initial_data, k_true = kinetics_setup
        rng = np.random.default_rng(2)
        runner = _make_simulator(first, second, k_true, rng)

        rounds = sequential_discrimination(
            experiments={"first": first, "second": second},
            initial_data=initial_data,
            design_bounds={"t": (0.5, 10.0), "C0": (0.5, 5.0)},
            n_rounds=5,
            run_experiment=runner,
            criterion=DiscriminationCriterion.BF,
            stop_when_dominant=0.95,
            n_starts=4,
            seed=0,
        )
        final = rounds[-1].selection
        assert final.weights["first"] > final.weights["second"], (
            f"first should win; final weights = {final.weights}"
        )

    def test_callback_fires_each_round(self, kinetics_setup):
        first, second, initial_data, k_true = kinetics_setup
        rng = np.random.default_rng(3)
        runner = _make_simulator(first, second, k_true, rng)

        seen: list[int] = []

        def cb(r: DiscriminationRound) -> None:
            seen.append(r.round)

        rounds = sequential_discrimination(
            experiments={"first": first, "second": second},
            initial_data=initial_data,
            design_bounds={"t": (0.5, 10.0), "C0": (0.5, 5.0)},
            n_rounds=3,
            run_experiment=runner,
            stop_when_dominant=2.0,
            callback=cb,
            n_starts=3,
            seed=0,
        )
        assert seen == list(range(len(rounds)))

    def test_caller_driven_mode_returns_one_round_with_design(self, kinetics_setup):
        """When ``run_experiment`` is None, the loop should return after
        proposing the first design (one DiscriminationRound with
        ``design`` populated and ``collected_data=None``)."""
        first, second, initial_data, _ = kinetics_setup
        rounds = sequential_discrimination(
            experiments={"first": first, "second": second},
            initial_data=initial_data,
            design_bounds={"t": (0.5, 10.0), "C0": (0.5, 5.0)},
            n_rounds=5,
            run_experiment=None,
            stop_when_dominant=2.0,
            n_starts=3,
            seed=0,
        )
        assert len(rounds) == 1
        assert rounds[0].design is not None
        assert rounds[0].collected_data is None

    def test_stop_when_dominant(self, kinetics_setup):
        """If the stopping threshold is generous enough, the loop should
        stop early before n_rounds is reached."""
        first, second, initial_data, k_true = kinetics_setup
        rng = np.random.default_rng(4)
        runner = _make_simulator(first, second, k_true, rng)

        rounds = sequential_discrimination(
            experiments={"first": first, "second": second},
            initial_data=initial_data,
            design_bounds={"t": (0.5, 10.0), "C0": (0.5, 5.0)},
            n_rounds=10,
            run_experiment=runner,
            stop_when_dominant=0.6,  # easy to reach
            n_starts=3,
            seed=0,
        )
        assert len(rounds) < 10
        # Stop reason: dominant weight reached. Final round's design should be None.
        assert rounds[-1].design is None

    def test_invalid_inputs(self, kinetics_setup):
        first, _, initial_data, _ = kinetics_setup
        with pytest.raises(ValueError, match="at least 2"):
            sequential_discrimination(
                {"only": first},
                initial_data,
                {"t": (0.5, 10.0), "C0": (0.5, 5.0)},
            )
        with pytest.raises(ValueError, match="n_rounds"):
            sequential_discrimination(
                {"first": first, "second": SecondOrderExp()},
                initial_data,
                {"t": (0.5, 10.0), "C0": (0.5, 5.0)},
                n_rounds=0,
            )
