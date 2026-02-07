"""Configure JAX for testing."""

import os

# Force CPU backend — Metal/GPU backend is experimental and may fail.
os.environ["JAX_PLATFORMS"] = "cpu"
