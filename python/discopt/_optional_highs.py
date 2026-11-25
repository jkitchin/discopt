"""HiGHS reach, in one place: whether it is installed, and the error when it is not.

``highspy`` is a core dependency again since the 2026-11-17 engine switch (it was the
optional ``discopt[highs]`` extra from highs-parity P6.3), but discopt must still run
without it: Pyodide has no wheel, and there ``DISCOPT_LP_ENGINE=auto`` resolves to the
in-house Rust engine (``solver._lp_engine``). This module is
the single place that imports it, so every reach can be inventoried
(``docs/dev/highs-optional-inventory.md``) and fails at the entry point with a message
naming the feature and the extra instead of ``No module named 'highspy'`` from somewhere
inside it. No default path needs it: without it the pure LP/MILP routes and IIS run on
the in-house engine (``solver._engine_default``); the explicit HiGHS opt-ins raise.

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
from typing import Any, Callable

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
        "(highspy>=1.16).",
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


# ---------------------------------------------------------------------------
# Exceptions raised inside a HiGHS callback (#2221)
# ---------------------------------------------------------------------------
#: Attribute on a ``highspy.Highs`` instance holding the exceptions its wrapped
#: callbacks raised that :func:`run_checked` has not re-raised yet.
_CALLBACK_ERRORS_ATTR = "_discopt_callback_errors"


def wrap_callback(h: Any, fn: Callable[..., Any]) -> Callable[..., Any]:
    """``fn`` as a callback of ``h`` whose exception :func:`run_checked` re-raises.

    Up to highspy 1.15 a Python exception raised inside a callback propagated out of
    ``Highs.run()``. From 1.16 ``Highs::run`` wraps the solve in ``try``/``catch
    (const std::exception&)``; pybind11's ``error_already_set`` is a
    ``std::exception``, so the exception is swallowed and ``run`` returns ``kError``
    with model status ``kSolveError``. Measured (#2221) on a two-variable MILP whose
    ``kCallbackMipImprovingSolution`` callback raises: highspy 1.15.1 raises it out of
    ``run``, 1.16.0 returns ``HighsStatus.kError``. A caller that maps ``kSolveError``
    to a status would then report a broken separator as an ordinary solver error
    (CLAUDE.md §7), so every Python callback discopt hands HiGHS goes through this
    wrapper and every ``run`` that can call one goes through
    :func:`run_checked`. The exception is still raised inside the callback,
    so HiGHS stops the solve as before on either version.
    """
    errors = getattr(h, _CALLBACK_ERRORS_ATTR, None)
    if errors is None:
        errors = []
        setattr(h, _CALLBACK_ERRORS_ATTR, errors)

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except BaseException as exc:
            errors.append(exc)
            raise

    return _wrapped


def run_checked(h: Any) -> Any:
    """``h.run()``, re-raising the first exception a :func:`wrap_callback` callback raised.

    The record is cleared before the run, so an exception from an earlier run of the
    same object (which on highspy <= 1.15 already left that run) is never re-raised
    here. After the run, ANY recorded exception is re-raised, whatever status HiGHS
    ended the run with: highspy 1.16 ends a run whose callback raised in ``kSolveError``
    (measured, #2221), but the guarantee does not rest on that detail, so an exception
    from an interrupt-style callback that a later release ended in, say,
    ``kInterrupt`` is not dropped either. On highspy <= 1.15 the exception leaves
    ``run`` itself and propagates unchanged; the record is cleared on the way out.
    Returns ``run``'s :class:`HighsStatus`.
    """
    errors = getattr(h, _CALLBACK_ERRORS_ATTR, None)
    if errors is not None:
        errors.clear()
    try:
        status = h.run()
    except BaseException:
        if errors is not None:
            errors.clear()
        raise
    if errors:
        exc = errors[0]
        errors.clear()
        raise exc
    return status
