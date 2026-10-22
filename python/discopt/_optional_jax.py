"""JAX is an optional dependency: one place that says so, and how to get it (#1873).

``jax``/``jaxlib`` live in the ``discopt[jax]`` extra, not in ``[project]
dependencies``. The default solve path runs on the POUNCE tape, the Rust core and
numpy, and imports no ``jax`` module. JAX is reached only by:

- the legacy NLP evaluator, used when an expression has no tape opcode
  (``dm.custom``, a matrix 2-norm, ...) or when ``DISCOPT_NLP_EVAL=jax`` asks for it;
- the features that use JAX autodiff on purpose (``discopt.estimate``,
  ``discopt.sensitivity``, the differentiable/parametric layers, ``dm.argmin`` /
  implicit / external functions, sIPOPT, the learned/ICNN relaxations, ...);
- the legacy ``_relax`` modules from the ``_jax`` era, kept as opt-outs.

Each of those modules calls :func:`require_jax` before it imports ``jax``, so a
JAX-less install fails at the entry point with a message that names the feature
and the extra, instead of ``No module named 'jax'`` from somewhere inside it.

:class:`MissingJaxError` subclasses ``ModuleNotFoundError`` (with ``name="jax"``),
so code that already handles a missing optional package through ``except
ImportError`` behaves exactly as it did when ``import jax`` itself failed.
"""

from __future__ import annotations

import importlib.util
import sys

JAX_EXTRA = "discopt[jax]"
INSTALL_HINT = f"pip install '{JAX_EXTRA}'"


class MissingJaxError(ModuleNotFoundError):
    """A JAX-only feature was used on an install without the ``discopt[jax]`` extra."""


_available: bool | None = None


def jax_available() -> bool:
    """Whether ``jax`` can be imported, without importing it.

    ``find_spec`` locates the package without executing it, so this check never
    puts JAX in ``sys.modules``. A module already imported counts as available;
    ``sys.modules["jax"] = None`` (the standard way to block an import) does not.
    The answer is cached for the life of the process.
    """
    global _available
    if _available is None:
        if "jax" in sys.modules:
            _available = sys.modules["jax"] is not None
        else:
            _available = importlib.util.find_spec("jax") is not None
    return _available


def missing_jax_error(feature: str) -> MissingJaxError:
    """The error :func:`require_jax` raises, for callers that need to raise it lazily."""
    return MissingJaxError(
        f"{feature} requires JAX, which is not installed. JAX is an optional "
        f"dependency of discopt; install it with `{INSTALL_HINT}`.",
        name="jax",
    )


def require_jax(feature: str) -> None:
    """Raise :class:`MissingJaxError` naming ``feature`` when JAX is not installed."""
    if not jax_available():
        raise missing_jax_error(feature)
