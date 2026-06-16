"""
JaxMINLP Test Configuration

Shared fixtures, markers, and configuration for the test suite.
Aligns with Section 3 (Software Engineering Practices) of the development plan.

Test categories (pytest markers):
  - smoke: Fast tests run on every PR (<5 min total)
  - unit: Component-level unit tests
  - integration: Cross-layer Rust↔JAX tests
  - correctness: Known-optimum validation
  - regression: Performance regression detection
  - property: Property-based tests (proptest/Hypothesis)
  - adversarial: Edge cases and adversarial inputs
  - slow: Tests requiring >60 seconds
  - gpu: Tests requiring GPU hardware
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
import numpy as np


# ─────────────────────────────────────────────────────────────
# MARKERS
# ─────────────────────────────────────────────────────────────


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "smoke: Fast smoke tests for CI")
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Cross-layer integration tests")
    config.addinivalue_line("markers", "correctness: Known-optimum correctness validation")
    config.addinivalue_line("markers", "regression: Performance regression tests")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "adversarial: Adversarial/edge-case tests")
    config.addinivalue_line("markers", "slow: Tests requiring >60 seconds")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def has_gpu() -> bool:
    """Check if GPU is available for JAX."""
    try:
        import jax

        return len(jax.devices("gpu")) > 0
    except Exception:
        return False


@pytest.fixture
def numerical_tolerance():
    """Standard numerical tolerances for the project."""

    @dataclass
    class Tolerances:
        abs_tol: float = 1e-6  # Absolute tolerance
        rel_tol: float = 1e-4  # Relative tolerance
        bound_tol: float = 1e-6  # Bound validity tolerance
        integrality_tol: float = 1e-5  # Integer feasibility
        constraint_tol: float = 1e-6  # Constraint satisfaction
        factorization_tol: float = 1e-12  # Sparse LA accuracy

    return Tolerances()


@pytest.fixture
def timer():
    """Simple wall-clock timer for performance assertions."""

    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None

        def __enter__(self):
            self.start_time = time.monotonic()
            return self

        def __exit__(self, *args):
            self.elapsed = time.monotonic() - self.start_time

    return Timer


@pytest.fixture
def random_lp():
    """Generate random LP instances for property-based testing."""

    def _make(m: int, n: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        c = rng.standard_normal(n)
        A = rng.standard_normal((m, n))
        x_feas = rng.uniform(0, 1, n)
        b = A @ x_feas + rng.uniform(0, 1, m)  # Ensure feasibility
        return {"c": c, "A_ub": A, "b_ub": b, "bounds": [(0, None)] * n}

    return _make


@pytest.fixture
def random_bounds():
    """Generate random variable bounds for relaxation testing."""

    def _make(n: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        lb = rng.uniform(-5, 0, n)
        ub = lb + rng.uniform(0.1, 5, n)
        return lb, ub

    return _make


# ─────────────────────────────────────────────────────────────
# TEST RESULT COLLECTION (for regression tracking)
# ─────────────────────────────────────────────────────────────


class PerformanceCollector:
    """Collect timing data during test runs for regression analysis."""

    def __init__(self):
        self.results: list[dict] = []

    def record(self, test_name: str, metric: str, value: float, unit: str = "seconds"):
        self.results.append(
            {
                "test": test_name,
                "metric": metric,
                "value": value,
                "unit": unit,
                "timestamp": time.time(),
            }
        )

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)


@pytest.fixture(scope="session")
def perf_collector():
    """Session-scoped performance data collector."""
    collector = PerformanceCollector()
    yield collector
    # Save results at end of session
    output = Path("reports") / "test_performance.json"
    collector.save(output)


# ─────────────────────────────────────────────────────────────
# HOOKS
# ─────────────────────────────────────────────────────────────


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests when no GPU available."""
    try:
        import jax

        has_gpu = len(jax.devices("gpu")) > 0
    except Exception:
        has_gpu = False

    if not has_gpu:
        skip_gpu = pytest.mark.skip(reason="No GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
