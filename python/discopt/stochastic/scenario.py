"""Scenarios for stochastic programming.

A :class:`Scenario` binds a probability to a *realization* of the uncertain
data (a mapping ``name -> value``). A :class:`ScenarioSet` is a finite,
probability-weighted collection — given explicitly, or drawn from distributions
by **sample average approximation (SAA)**. The realized values are consumed by a
scenario-creator callback in :mod:`~discopt.stochastic.extensive_form` (the
mpi-sppy / PySP pattern), so no expression-graph surgery is needed.

See ``docs/dev/stochastic-module-plan.md`` §3.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Scenario", "ScenarioSet"]

_PROB_TOL = 1e-9


@dataclass(frozen=True)
class Scenario:
    """One realization of the uncertain data with its probability."""

    probability: float
    data: dict  # name -> realized value (float or ndarray)


class ScenarioSet:
    """A finite, probability-weighted set of scenarios (probabilities sum to 1)."""

    def __init__(
        self, scenarios: list[Scenario], *, seed: int | None = None, n_sampled: int | None = None
    ):
        if not scenarios:
            raise ValueError("ScenarioSet needs at least one scenario")
        probs = np.array([s.probability for s in scenarios], dtype=float)
        if np.any(probs < -_PROB_TOL):
            raise ValueError("scenario probabilities must be nonnegative")
        total = float(probs.sum())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"scenario probabilities must sum to 1, got {total:.6g}. "
                f"(Use ScenarioSet.from_samples for equal-weight SAA scenarios.)"
            )
        self.scenarios: list[Scenario] = list(scenarios)
        self.seed = seed  # provenance for SAA (None if explicit)
        self.n_sampled = n_sampled

    # ── constructors ──────────────────────────────────────────────────

    @classmethod
    def from_list(cls, items: list[tuple[float, dict]]) -> "ScenarioSet":
        """From explicit ``[(probability, data_dict), ...]``."""
        return cls([Scenario(float(p), dict(d)) for p, d in items])

    @classmethod
    def from_samples(cls, samples: dict[str, np.ndarray], probabilities=None) -> "ScenarioSet":
        """From arrays of realizations (SAA). ``samples`` maps each uncertain name to
        an ``n``-length array; scenario ``s`` uses each array's ``s``-th entry. Equal
        weights ``1/n`` unless ``probabilities`` is given."""
        names = list(samples)
        n = len(np.asarray(samples[names[0]]))
        for k in names:
            if len(np.asarray(samples[k])) != n:
                raise ValueError(f"sample array for '{k}' has length != {n}")
        probs = np.full(n, 1.0 / n) if probabilities is None else np.asarray(probabilities, float)
        scen = [
            Scenario(float(probs[s]), {k: np.asarray(samples[k])[s] for k in names})
            for s in range(n)
        ]
        return cls(scen, n_sampled=n)

    @classmethod
    def sample(cls, samplers: dict, n: int, *, seed: int) -> "ScenarioSet":
        """SAA: draw ``n`` equal-weight scenarios. ``samplers`` maps each name to a
        callable ``rng -> value``. The ``seed`` is recorded for reproducibility."""
        rng = np.random.default_rng(seed)
        draws = {k: np.array([fn(rng) for _ in range(n)]) for k, fn in samplers.items()}
        s = cls.from_samples(draws)
        return cls(s.scenarios, seed=seed, n_sampled=n)

    # ── access ────────────────────────────────────────────────────────

    @property
    def probabilities(self) -> np.ndarray:
        return np.array([s.probability for s in self.scenarios], dtype=float)

    def __len__(self) -> int:
        return len(self.scenarios)

    def __iter__(self):
        return iter(self.scenarios)

    def __getitem__(self, i: int) -> Scenario:
        return self.scenarios[i]
