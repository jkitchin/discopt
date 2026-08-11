"""Python-side presolve package — A3 of the presolve roadmap.

This package implements the Rust↔Python presolve handshake. Python passes
that need to call into JAX (e.g. the convexity detector, the NN-embedded
MINLP presolver, reverse-mode interval AD) live here and conform to the
:class:`PresolvePass` protocol so they can run inside the orchestrator's
fixed-point loop alongside the Rust kernels.

Top-level exports:

- :class:`PresolvePass` — protocol every Python pass implements.
- :class:`PresolveResult`, :class:`PresolveDelta` — Python mirrors of the
  Rust data carriers, used by the orchestrator wrapper to assemble a
  deterministic, byte-stable record.
- :func:`run_orchestrated_presolve` — Python wrapper that interleaves
  Python passes between Rust orchestrator sweeps to a fixed point.
- :func:`make_python_delta` — helper for Python passes to construct a
  delta dict in the same shape Rust passes emit.
"""

from .convex_reform import ConvexReformPass
from .orchestrator import run_orchestrated_presolve
from .protocol import (
    PresolveDelta,
    PresolvePass,
    PresolveResult,
    delta_made_progress,
    make_python_delta,
)
from .reverse_ad import ReverseADPass
from .separability import SeparabilityPass, SeparabilityReport, detect_separability

__all__ = [
    "PresolvePass",
    "PresolveDelta",
    "PresolveResult",
    "make_python_delta",
    "delta_made_progress",
    "run_orchestrated_presolve",
    "ConvexReformPass",
    "ReverseADPass",
    "SeparabilityPass",
    "SeparabilityReport",
    "detect_separability",
]
