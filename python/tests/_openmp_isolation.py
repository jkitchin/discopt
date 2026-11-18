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
import threading

import pytest

_GUARDED = ("cyipopt", "torch")
_HINT = (
    " Skipped on macOS so the worker is not aborted (#2205); run this test in a "
    "process that has not loaded the other package, e.g. `pytest <nodeid> -n0`."
)


# Names whose check is running on this thread, shared by every finder instance.
# The check may itself call ``importlib.util.find_spec(name)``
# (``_openmp_guard._package_binaries``'s fallback when the distribution's RECORD
# lists no binaries, which is always the case for an absent package); that walks
# ``sys.meta_path`` back into the finder(s). Without this guard the re-entry
# repeated the check and recursed until ``RecursionError`` (#2243). Module-level,
# not per instance: with two finders on ``sys.meta_path`` (a second
# ``configure()``, or a test inserting its own) a per-instance guard let each
# finder re-run the check before tripping, multiplying the work.
_ACTIVE = threading.local()


class _OpenMPGuardFinder(importlib.abc.MetaPathFinder):
    """Check the OpenMP runtimes before a guarded top-level import is resolved."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _GUARDED:
            return None
        active = getattr(_ACTIVE, "names", None)
        if active is None:
            active = _ACTIVE.names = set()
        if fullname in active:
            return None  # re-entered from our own check: defer to the regular finders
        from discopt._openmp_guard import (
            OpenMPRuntimeConflictError,
            check_openmp_compatible,
        )

        active.add(fullname)
        try:
            check_openmp_compatible(fullname, feature="a test")
        except OpenMPRuntimeConflictError as exc:
            raise pytest.skip.Exception(str(exc) + _HINT, allow_module_level=True) from exc
        finally:
            active.discard(fullname)
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
