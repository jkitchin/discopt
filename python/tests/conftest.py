"""Configure JAX for testing."""

import os

# Force CPU backend — Metal/GPU backend is experimental and may fail.
os.environ["JAX_PLATFORMS"] = "cpu"
# Enable 64-bit precision for float64 support.
os.environ["JAX_ENABLE_X64"] = "1"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "correctness: Known-optimum correctness validation")
    config.addinivalue_line("markers", "minlptests: MINLPTests.jl standardized NLP/MINLP problems")
    config.addinivalue_line("markers", "integration: solver-dependent integration tests")
    config.addinivalue_line("markers", "amp_benchmark: opt-in AMP benchmark/incidence tests")
    config.addinivalue_line("markers", "requires_cyipopt: requires cyipopt/Ipopt")
