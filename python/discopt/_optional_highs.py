"""HiGHS reach, in one place: whether it is installed, and the error when it is not.

``highspy`` is the optional ``discopt[highs]`` extra (highs-parity P6.3). This module is
the single place that imports it, so every reach can be inventoried
(``docs/dev/highs-optional-inventory.md``) and fails at the entry point with a message
naming the feature and the extra instead of ``No module named 'highspy'`` from somewhere
inside it. No default path needs it: without it the pure-MILP route falls back to the
in-house engine (``solver._lp_milp_backend``); the explicit HiGHS opt-ins raise.

:class:`MissingHighsError` subclasses ``ModuleNotFoundError`` (``name="highspy"``), so
code that already handles a missing optional package through ``except ImportError``
behaves exactly as it did when ``import highspy`` itself failed. The two historical
names, ``lp_milp_highs.HighsUnavailable`` and ``milp_highs.HighsBackendUnavailable``, are
aliases of it.
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

HIGHS_EXTRA = "discopt[highs]"
INSTALL_HINT = f"pip install '{HIGHS_EXTRA}'"


class MissingHighsError(ModuleNotFoundError):
    """A HiGHS-backed feature was used on an install without ``highspy``."""


def highs_available() -> bool:
    """Whether ``highspy`` can be imported, without importing it.

    ``find_spec`` locates the package without executing it. A module already imported
    counts as available; ``sys.modules["highspy"] = None`` (the standard way to block an
    import, used by the tests that simulate a HiGHS-less install) does not. Not cached:
    those tests toggle the block within one process.
    """
    if "highspy" in sys.modules:
        return sys.modules["highspy"] is not None
    try:
        return importlib.util.find_spec("highspy") is not None
    except (ImportError, ValueError):
        return False


def missing_highs_error(feature: str) -> MissingHighsError:
    """The error :func:`require_highspy` raises, for callers that raise it lazily."""
    return MissingHighsError(
        f"{feature} needs HiGHS, which is not installed. Install it with `{INSTALL_HINT}` "
        "(highspy>=1.11,!=1.12.0).",
        name="highspy",
    )


def require_highspy(feature: str = "this HiGHS-backed feature") -> ModuleType:
    """Import and return ``highspy``, or raise :class:`MissingHighsError` naming
    ``feature``."""
    try:
        import highspy
    except ImportError as err:
        raise missing_highs_error(feature) from err
    mod: ModuleType = highspy
    return mod
