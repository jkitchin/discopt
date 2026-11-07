"""Refuse to load a second OpenMP runtime into a macOS process (#2205).

On macOS, ``cyipopt``'s Ipopt and ``torch`` each bring their own copy of the LLVM
OpenMP runtime (``libomp.dylib``). When the second copy initialises it finds the
first already running and calls ``abort()``::

    OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
    Abort trap: 6

That happens in either import order and cannot be caught: the process dies, with
no traceback and nothing written by the caller. In a test run it takes the whole
xdist worker with it.

This module turns that abort into an exception raised *before* the second runtime
is loaded. :func:`check_openmp_compatible` is called at every place discopt
imports ``cyipopt`` or ``torch``:

1. it lists the OpenMP runtimes already mapped into the process (from ``dyld``'s
   image list), and
2. the OpenMP runtimes the package about to be imported ships in its own wheel
   (from the distribution's ``RECORD``, without importing the package);

and raises :class:`OpenMPRuntimeConflictError` when the package ships a runtime
and a *different* file is already loaded. Two installs that resolve to one shared
``libomp`` (a conda environment, where both packages link ``$PREFIX/lib``) ship
none in their wheels, or the same file, and pass.

What it cannot see: a package that links a system ``libomp`` (Homebrew) rather
than shipping one. If that package is imported *second*, its runtime is not
known in advance and the abort still happens. torch's wheels ship their runtime,
so ``cyipopt`` first and ``torch`` second -- the order in the issue's test-suite
crash -- is always detected.

``KMP_DUPLICATE_LIB_OK=TRUE`` is honoured as the user's explicit choice (OpenMP
itself then does not abort) and only produces a :class:`RuntimeWarning`. discopt
never sets it: OpenMP's own message says it "may cause crashes or silently
produce incorrect results", which a solver cannot accept by default.

Off macOS the check returns immediately; Linux wheels of the two packages do not
collide (``libgomp`` / a single ``libomp`` load).
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from collections.abc import Iterable

ISSUE_URL = "https://github.com/jkitchin/discopt/issues/2205"

# The LLVM runtime and the Intel one it is ABI-compatible with; both abort with
# OMP Error #15 when a second copy initialises. GNU ``libgomp`` is a different
# runtime and does not take part in this check.
_OMP_RUNTIME = re.compile(r"(^|/)lib(omp|iomp5)(\.\d+)*\.dylib$")

# The distribution that provides each guarded import name.
_DISTRIBUTIONS = {"cyipopt": "cyipopt", "torch": "torch"}


class OpenMPRuntimeConflictError(RuntimeError):
    """Importing a package would load a second OpenMP runtime and abort the process.

    A ``RuntimeError``, not an ``ImportError``: the package *is* installed, and
    code that treats ``ImportError`` as "backend unavailable, try another" must not
    silently route around this.
    """


def _on_macos() -> bool:
    return sys.platform == "darwin"


def is_omp_runtime(path: str) -> bool:
    """Whether ``path`` names an LLVM/Intel OpenMP runtime library."""
    return bool(_OMP_RUNTIME.search(path))


def _real(path: str) -> str:
    try:
        return os.path.realpath(path)
    except OSError:  # pragma: no cover - realpath of a string does not raise in practice
        return path


def loaded_omp_runtimes() -> set[str]:
    """Real paths of the OpenMP runtimes mapped into this process (macOS only).

    Reads ``dyld``'s image list through ``_dyld_image_count`` /
    ``_dyld_get_image_name``. Returns an empty set on any other platform.
    """
    if not _on_macos():
        return set()
    import ctypes

    libc = ctypes.CDLL(None)
    count = libc._dyld_image_count
    count.restype = ctypes.c_uint32
    name = libc._dyld_get_image_name
    name.restype = ctypes.c_char_p
    name.argtypes = [ctypes.c_uint32]
    paths = set()
    for i in range(count()):
        raw = name(i)
        if raw is None:
            continue
        path = raw.decode("utf-8", "surrogateescape")
        if is_omp_runtime(path):
            paths.add(_real(path))
    return paths


def shipped_omp_runtimes(distribution: str) -> set[str]:
    """Real paths of the OpenMP runtimes ``distribution`` installs, without importing it.

    Read from the distribution's file list (``RECORD``). Empty when the
    distribution is not installed or ships none (it then links an external one).
    """
    from importlib import metadata

    try:
        files = metadata.files(distribution)
    except metadata.PackageNotFoundError:
        return set()
    if not files:
        return set()
    return {_real(str(f.locate())) for f in files if is_omp_runtime(str(f))}


def find_conflict(loaded: Iterable[str], shipped: Iterable[str]) -> set[str]:
    """The already-loaded runtimes that a package shipping ``shipped`` would collide with.

    Empty when the package ships no runtime (nothing new is loaded) or when every
    loaded runtime is one of the package's own files (it is already in use).
    """
    shipped = set(shipped)
    if not shipped:
        return set()
    return set(loaded) - shipped


def check_openmp_compatible(module: str, *, feature: str | None = None) -> None:
    """Raise if importing ``module`` would abort this process on a second ``libomp``.

    ``module`` is ``"cyipopt"`` or ``"torch"``. Call it immediately before the
    import. A no-op off macOS, when ``module`` is already imported, or when no
    conflicting runtime is loaded.
    """
    if not _on_macos() or module in sys.modules:
        return
    shipped = shipped_omp_runtimes(_DISTRIBUTIONS.get(module, module))
    conflict = find_conflict(loaded_omp_runtimes(), shipped)
    if not conflict:
        return
    loaded_by = [name for name in _DISTRIBUTIONS if name != module and name in sys.modules]
    message = (
        f"Importing {module!r}{f' (for {feature})' if feature else ''} would load a second "
        f"OpenMP runtime into this process, and on macOS that aborts it "
        f"('OMP: Error #15'). Already loaded: {', '.join(sorted(conflict))}"
        f"{' (already imported here: ' + ', '.join(loaded_by) + ')' if loaded_by else ''}; "
        f"{module} ships: {', '.join(sorted(shipped))}. "
        f"cyipopt and torch from pip wheels each bundle their own libomp (as do some "
        f"other wheels, e.g. scikit-learn's). Use them in separate processes -- where "
        f"torch is needed, nlp_solver='pounce' (the default) instead of 'ipopt' -- or "
        f"builds of both that link one shared libomp. "
        f"See {ISSUE_URL}."
    )
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").upper() in {"TRUE", "1", "YES"}:
        warnings.warn(
            message + " Proceeding because KMP_DUPLICATE_LIB_OK is set; OpenMP warns this "
            "'may cause crashes or silently produce incorrect results'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    raise OpenMPRuntimeConflictError(message)
