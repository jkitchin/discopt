"""Keep a macOS test worker alive when cyipopt and torch would share it (#2205).

On macOS, ``cyipopt`` and ``torch`` from pip wheels each bundle ``libomp``; the
second one to load calls ``abort()`` (``OMP: Error #15``), which kills the xdist
worker and every test queued on it. Linux CI is unaffected.

This plugin, registered from ``conftest.py`` on macOS only:

- installs a ``sys.meta_path`` finder that runs
  :func:`discopt._openmp_guard.check_openmp_compatible` before ``import cyipopt`` /
  ``import torch`` (including ``pytest.importorskip``), and turns a detected clash
  into a *skip* that names #2205 -- at collection time as a module-level skip;
- turns :class:`~discopt._openmp_guard.OpenMPRuntimeConflictError` raised by
  discopt's own guarded import sites during a test into the same skip.

A skip, not a pass: the test did not run in this process. Run the skipped
tests on their own (``pytest -k torch``, or ``-p no:randomly -n0`` on the file) to
exercise them; the reason string says so.
"""

from __future__ import annotations

import importlib.abc
import sys

import pytest

_GUARDED = ("cyipopt", "torch")
_HINT = (
    " Skipped on macOS so the worker is not aborted (#2205); run this test in a "
    "process that has not loaded the other package, e.g. `pytest <nodeid> -n0`."
)


class _OpenMPGuardFinder(importlib.abc.MetaPathFinder):
    """Check the OpenMP runtimes before a guarded top-level import is resolved."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname in _GUARDED:
            from discopt._openmp_guard import (
                OpenMPRuntimeConflictError,
                check_openmp_compatible,
            )

            try:
                check_openmp_compatible(fullname, feature="a test")
            except OpenMPRuntimeConflictError as exc:
                raise pytest.skip.Exception(str(exc) + _HINT, allow_module_level=True) from exc
        return None  # let the regular finders locate the module


class OpenMPIsolationPlugin:
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        if call.excinfo is None or report.outcome != "failed":
            return
        from discopt._openmp_guard import OpenMPRuntimeConflictError

        if call.excinfo.errisinstance(OpenMPRuntimeConflictError):
            path, lineno, _ = item.reportinfo()
            report.outcome = "skipped"
            report.longrepr = (str(path), (lineno or 0) + 1, str(call.excinfo.value) + _HINT)


def configure(config) -> None:
    """Install the finder and the report hook; a no-op off macOS."""
    if sys.platform != "darwin":
        return
    if not any(isinstance(f, _OpenMPGuardFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _OpenMPGuardFinder())
    config.pluginmanager.register(OpenMPIsolationPlugin(), "discopt-openmp-isolation")
