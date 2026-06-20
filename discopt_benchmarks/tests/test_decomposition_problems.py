"""Correctness gate for the registered decomposition benchmark instances.

Runs each block-structured / two-stage instance registered in
``benchmarks.problems.decomposition_problems`` with the **default** solver and
asserts it reaches the known optimum. This is the gate the adversarial review
asked for: a regression in the MILP solver (e.g. the equilibration over-scaling
bug that returned a binary at -1, optimum 16 instead of 17) would fail here, not
just in the decomposition-specific unit tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import discopt.modeling as dm  # noqa: E402

from benchmarks.problems.base import get_problems  # noqa: E402

_DECOMP = [p for p in get_problems("milp", level="full") if "decomposable" in p.tags]


@pytest.mark.correctness
@pytest.mark.parametrize("problem", _DECOMP, ids=lambda p: p.name)
def test_registered_decomposition_instance_hits_known_optimum(problem):
    assert _DECOMP, "no decomposable instances registered"
    model = problem.build_fn()
    assert isinstance(model, dm.Model)
    result = model.solve(time_limit=60)
    assert result.status in ("optimal", "feasible")
    assert result.objective == pytest.approx(problem.known_optimum, abs=1e-3), (
        f"{problem.name}: got {result.objective}, known optimum {problem.known_optimum}"
    )
    # No structural variable may leave its box (the off-bound -1 regression).
    if result.x is not None:
        for var in model._variables:
            import numpy as np

            vals = np.atleast_1d(result.x[var.name])
            lo = np.atleast_1d(var.lb).reshape(-1)
            hi = np.atleast_1d(var.ub).reshape(-1)
            assert np.all(vals >= lo.min() - 1e-5)
            assert np.all(vals <= hi.max() + 1e-5)
