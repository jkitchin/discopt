"""#2205: on macOS a second OpenMP runtime aborts the process; refuse it loudly instead.

cyipopt's Ipopt and torch each ship ``libomp.dylib`` in their macOS wheels. The
second to load aborts the process (``OMP: Error #15``) in either order, with no
Python traceback. ``discopt._openmp_guard.check_openmp_compatible`` runs before
every discopt import of either package and raises
``OpenMPRuntimeConflictError`` when the package to be imported ships a runtime
and a different one is already mapped.

The clash itself only exists on macOS, so these tests drive the decision logic
with a simulated platform and simulated runtime lists; the one test that reads
the real ``dyld`` image list runs on macOS only.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from discopt import _openmp_guard as guard

TORCH_OMP = "/venv/lib/python3.12/site-packages/torch/lib/libomp.dylib"
CYIPOPT_OMP = "/venv/lib/python3.12/site-packages/cyipopt/.dylibs/libomp.dylib"
CONDA_OMP = "/conda/lib/libomp.dylib"


@pytest.fixture
def macos(monkeypatch):
    """Pretend to be macOS with neither guarded package imported or opted out."""
    monkeypatch.setattr(guard, "_on_macos", lambda: True)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "cyipopt", raising=False)
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)

    def set_runtimes(loaded, shipped):
        monkeypatch.setattr(guard, "loaded_omp_runtimes", lambda: set(loaded))
        monkeypatch.setattr(guard, "shipped_omp_runtimes", lambda dist: set(shipped))

    return set_runtimes


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/x/libomp.dylib", True),
        ("/x/libiomp5.dylib", True),
        ("/x/libomp.5.dylib", True),
        ("libomp.dylib", True),
        ("/x/libgomp.1.dylib", False),
        ("/x/libomp.so", False),
        ("/x/libompx.dylib", False),
        ("/x/mylibomp.dylib", False),
    ],
)
def test_runtime_name_recognition(path, expected):
    assert guard.is_omp_runtime(path) is expected


def test_find_conflict():
    assert guard.find_conflict({CYIPOPT_OMP}, {TORCH_OMP}) == {CYIPOPT_OMP}
    # The package ships nothing: nothing new is loaded.
    assert guard.find_conflict({CYIPOPT_OMP}, set()) == set()
    # Nothing loaded yet.
    assert guard.find_conflict(set(), {TORCH_OMP}) == set()
    # The loaded runtime is the package's own file (already in use).
    assert guard.find_conflict({TORCH_OMP}, {TORCH_OMP}) == set()


def test_torch_after_cyipopt_refused(macos):
    macos(loaded={CYIPOPT_OMP}, shipped={TORCH_OMP})
    with pytest.raises(guard.OpenMPRuntimeConflictError) as info:
        guard.check_openmp_compatible("torch", feature="load_torch_sequential")
    msg = str(info.value)
    assert "OMP: Error #15" in msg
    assert CYIPOPT_OMP in msg and TORCH_OMP in msg
    assert "2205" in msg and "load_torch_sequential" in msg


def test_cyipopt_after_torch_refused(macos):
    macos(loaded={TORCH_OMP}, shipped={CYIPOPT_OMP})
    with pytest.raises(guard.OpenMPRuntimeConflictError):
        guard.check_openmp_compatible("cyipopt")


def test_conflict_is_not_an_import_error():
    """`except ImportError: backend unavailable` must not route around the clash."""
    assert issubclass(guard.OpenMPRuntimeConflictError, RuntimeError)
    assert not issubclass(guard.OpenMPRuntimeConflictError, ImportError)


def test_shared_runtime_passes(macos):
    """conda-forge: both link one $PREFIX/lib/libomp.dylib and ship none in a wheel."""
    macos(loaded={CONDA_OMP}, shipped=set())
    guard.check_openmp_compatible("torch")
    macos(loaded=set(), shipped={TORCH_OMP})
    guard.check_openmp_compatible("torch")


def test_noop_off_macos(monkeypatch):
    monkeypatch.setattr(guard, "_on_macos", lambda: False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(guard, "loaded_omp_runtimes", lambda: {CYIPOPT_OMP})
    monkeypatch.setattr(guard, "shipped_omp_runtimes", lambda dist: {TORCH_OMP})
    guard.check_openmp_compatible("torch")


def test_noop_when_already_imported(macos, monkeypatch):
    macos(loaded={CYIPOPT_OMP}, shipped={TORCH_OMP})
    monkeypatch.setitem(sys.modules, "torch", object())
    guard.check_openmp_compatible("torch")


def test_kmp_duplicate_lib_ok_warns_not_raises(macos, monkeypatch):
    macos(loaded={CYIPOPT_OMP}, shipped={TORCH_OMP})
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    with pytest.warns(RuntimeWarning, match="KMP_DUPLICATE_LIB_OK"):
        guard.check_openmp_compatible("torch")


def test_shipped_runtimes_from_record(monkeypatch, tmp_path):
    from importlib import metadata

    class _F:
        def __init__(self, rel):
            self.rel = rel

        def __str__(self):
            return self.rel

        def locate(self):
            return tmp_path / self.rel

    monkeypatch.setattr(
        metadata,
        "files",
        lambda dist: [_F("torch/__init__.py"), _F("torch/lib/libomp.dylib")],
    )
    assert guard.shipped_omp_runtimes("torch") == {str(tmp_path / "torch/lib/libomp.dylib")}


def test_shipped_runtimes_unknown_distribution():
    assert guard.shipped_omp_runtimes("no-such-distribution-2205") == set()


def test_loaded_runtimes_empty_off_macos(monkeypatch):
    monkeypatch.setattr(guard, "_on_macos", lambda: False)
    assert guard.loaded_omp_runtimes() == set()


@pytest.mark.skipif(sys.platform != "darwin", reason="reads the macOS dyld image list")
def test_loaded_runtimes_real_dyld():
    loaded = guard.loaded_omp_runtimes()
    assert isinstance(loaded, set) and all(guard.is_omp_runtime(p) for p in loaded)


# ── discopt's import sites refuse before importing ─────────────────────────


def test_torch_reader_refuses(macos, monkeypatch):
    from discopt.ml.readers.torch_reader import load_torch_sequential

    macos(loaded={CYIPOPT_OMP}, shipped={TORCH_OMP})
    with pytest.raises(guard.OpenMPRuntimeConflictError, match="load_torch_sequential"):
        load_torch_sequential(object())
    assert "torch" not in sys.modules


def test_ipopt_backend_refuses_and_is_not_listed(macos, monkeypatch):
    from discopt.solvers import nlp_backend

    macos(loaded={TORCH_OMP}, shipped={CYIPOPT_OMP})
    with pytest.raises(guard.OpenMPRuntimeConflictError):
        nlp_backend.get_nlp_solver("cyipopt")
    assert "cyipopt" not in nlp_backend.available_backends()
    assert "cyipopt" not in sys.modules


def test_ipopt_solve_nlp_refuses(macos):
    from discopt.solvers import nlp_ipopt

    macos(loaded={TORCH_OMP}, shipped={CYIPOPT_OMP})
    with pytest.raises(guard.OpenMPRuntimeConflictError):
        nlp_ipopt.solve_nlp(None, np.zeros(1))
    with pytest.raises(guard.OpenMPRuntimeConflictError):
        nlp_ipopt._ipopt_version()
    assert "cyipopt" not in sys.modules


def test_model_solve_with_ipopt_refuses(macos):
    """End to end: `nlp_solver="ipopt"` refuses instead of the process aborting."""
    import discopt.modeling as dm

    m = dm.Model("omp2205")
    x = m.continuous("x", lb=-2, ub=2)
    # A smooth nonlinear objective: a pure quadratic is routed to the QP path and
    # never reaches Ipopt, which is what made the first version of this test vacuous.
    m.minimize(dm.exp(x) + (x - 1) ** 2)
    macos(loaded={TORCH_OMP}, shipped={CYIPOPT_OMP})
    with pytest.raises(guard.OpenMPRuntimeConflictError):
        m.solve(nlp_solver="ipopt")
    assert "cyipopt" not in sys.modules


# ── the test-suite plugin skips instead of aborting the worker ─────────────


def test_suite_finder_skips_on_conflict(macos):
    import _openmp_isolation

    macos(loaded={CYIPOPT_OMP}, shipped={TORCH_OMP})
    finder = _openmp_isolation._OpenMPGuardFinder()
    with pytest.raises(pytest.skip.Exception, match="2205"):
        finder.find_spec("torch")
    # Unrelated imports and a compatible state fall through to the real finders.
    assert finder.find_spec("json") is None
    macos(loaded=set(), shipped={TORCH_OMP})
    assert finder.find_spec("torch") is None
