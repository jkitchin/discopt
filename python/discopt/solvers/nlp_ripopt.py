"""Deprecated alias for :mod:`discopt.solvers.nlp_pounce`.

The previous Rust ``ripopt`` IPM has been replaced by POUNCE
(https://github.com/jkitchin/pounce), a pure-Rust port of Ipopt with
Python bindings. The Python evaluator protocol is unchanged, so
existing call sites that import ``solve_nlp`` from this module keep
working — they now drive POUNCE instead of the discontinued ripopt
crate.

This module re-exports :func:`discopt.solvers.nlp_pounce.solve_nlp` and
emits a :class:`DeprecationWarning` on import. It will be removed in
a future release; import :func:`discopt.solvers.nlp_pounce.solve_nlp`
directly.
"""

from __future__ import annotations

import warnings

from discopt.solvers.nlp_pounce import solve_nlp

warnings.warn(
    "discopt.solvers.nlp_ripopt is deprecated; use discopt.solvers.nlp_pounce "
    "(POUNCE is a pure-Rust port of Ipopt). The 'ripopt' name will be removed "
    "in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["solve_nlp"]
