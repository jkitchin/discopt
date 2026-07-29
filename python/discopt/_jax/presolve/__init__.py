"""Python-side presolve helpers.

This package used to implement the **A3 Rust↔Python presolve handshake**:
``run_orchestrated_presolve`` interleaved Python passes between Rust orchestrator
sweeps, and three passes (``ConvexReformPass``, ``ReverseADPass``,
``SeparabilityPass``) existed only to be run by it. Phase 1 Card 1b deleted the
handshake — verified unreachable: no production caller ever passed
``python_passes=`` to ``run_root_presolve``, so the whole layer ran only in its own
tests. The live equivalents survive elsewhere: reverse-AD tightening is
``presolve_pipeline.run_reverse_ad_tightening`` (called from ``solver.py``), and
convexity detection lives in :mod:`discopt._jax.convexity`.

What remains here:

- :class:`PresolvePass` / :func:`make_python_delta` / :class:`PresolveDelta` — the
  delta contract, still used by :mod:`discopt.nn.presolve`.
- :func:`detect_separability` / :class:`SeparabilityReport` — standalone
  block-separability analysis (no pass wrapper).
"""

from .protocol import PresolveDelta, PresolvePass, make_python_delta
from .separability import SeparabilityReport, detect_separability

__all__ = [
    "PresolvePass",
    "PresolveDelta",
    "make_python_delta",
    "SeparabilityReport",
    "detect_separability",
]
