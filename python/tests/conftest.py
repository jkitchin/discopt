"""Configure JAX for testing."""

import os

# Force CPU backend — Metal/GPU backend is experimental and may fail.
os.environ["JAX_PLATFORMS"] = "cpu"
# Enable 64-bit precision for float64 support.
os.environ["JAX_ENABLE_X64"] = "1"

import pytest


@pytest.fixture
def heterogeneous_array_bounds():
    """Shared fixture for the X-2 "variable block treated as a scalar" class (#413).

    Returns a factory ``make(shape=(3,), lb=..., ub=...)`` that builds a fresh
    ``Model`` with a single array variable ``x`` whose elements carry *distinct*
    per-element bounds, plus a trivial linear objective/constraint so the model
    is well-formed for classify/extract/export/reformulate consumers.

    Element 0's bounds are deliberately the *tightest* so that any consumer that
    collapses the block to element 0 (``v.lb.flat[0]`` / ``.first()`` /
    block-as-scalar) is exposed: it will illegally narrow the wider elements.
    Every code site that reads array-variable bounds should be exercised through
    this fixture (`.nl`/MPS/LP/GAMS export, FBBT seeding, big-M, classify).

    Returns
    -------
    callable
        ``make(shape=(3,), lb=[0,2,4], ub=[1,5,9]) -> (model, x, lb, ub)`` where
        ``lb``/``ub`` are the flattened float arrays actually applied.
    """
    import numpy as np
    from discopt import Model

    def make(shape=(3,), lb=(0.0, 2.0, 4.0), ub=(1.0, 5.0, 9.0), name="model"):
        lb_arr = np.asarray(lb, dtype=np.float64).reshape(shape)
        ub_arr = np.asarray(ub, dtype=np.float64).reshape(shape)
        if np.all(lb_arr == lb_arr.flat[0]) and np.all(ub_arr == ub_arr.flat[0]):
            raise ValueError(
                "heterogeneous_array_bounds fixture requires distinct per-element "
                "bounds (homogeneous bounds hide the collapse-to-element-0 bug)."
            )
        m = Model(name)
        x = m.continuous("x", shape=shape, lb=lb_arr, ub=ub_arr)
        first = tuple(0 for _ in shape) if len(shape) > 1 else 0
        m.minimize(x[first])
        m.subject_to(x[first] >= float(lb_arr.flat[0]))
        return m, x, lb_arr.ravel(), ub_arr.ravel()

    return make


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "correctness: Known-optimum correctness validation")
    config.addinivalue_line("markers", "minlptests: MINLPTests.jl standardized NLP/MINLP problems")
    config.addinivalue_line("markers", "integration: solver-dependent integration tests")
    config.addinivalue_line("markers", "amp_benchmark: opt-in AMP benchmark/incidence tests")
    config.addinivalue_line("markers", "requires_cyipopt: requires cyipopt/Ipopt")
    config.addinivalue_line(
        "markers", "relaxation: per-operator relaxation soundness/coverage audit"
    )
