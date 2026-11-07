"""#2205: on macOS a second OpenMP runtime aborts the process; refuse it loudly instead.

torch ships ``libomp.dylib`` in its macOS wheel and cyipopt's Ipopt links
another (on the issue's machine, Homebrew's, which cyipopt does not ship). The
second to load aborts the process (``OMP: Error #15``) in either order, with no
Python traceback. ``discopt._openmp_guard.check_openmp_compatible`` runs before
every discopt import of either package and raises
``OpenMPRuntimeConflictError`` when the package to be imported would load a
runtime -- shipped, or linked through its Mach-O load commands -- and a
different one is already mapped.

The clash itself only exists on macOS, so these tests drive the decision logic
with a simulated platform and simulated runtime lists, and the Mach-O walk with
synthetic binaries written here (the parser reads files and runs anywhere); the
one test that reads the real ``dyld`` image list runs on macOS only.
"""

from __future__ import annotations

import os
import struct
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
        monkeypatch.setattr(guard, "package_omp_runtimes", lambda module: frozenset(shipped))

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
    """End to end: `nlp_solver="ipopt"` refuses instead of the process aborting.

    The model must really reach an Ipopt NLP solve: a pure quadratic is routed to
    the convex-QP path and never imports cyipopt, which made the first version of
    this test vacuous (it raised nothing on macOS). This one is nonconvex, and the
    test asserts the refusal came from an Ipopt NLP entry point.
    """
    import discopt.modeling as dm

    m = dm.Model("omp2205")
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=-1, ub=1)
    m.minimize(dm.sin(3 * x) * y + x**2)
    macos(loaded={TORCH_OMP}, shipped={CYIPOPT_OMP})
    with pytest.raises(guard.OpenMPRuntimeConflictError) as info:
        m.solve(nlp_solver="ipopt", time_limit=60)
    frames = {entry.name for entry in info.traceback}
    assert frames & {"solve_nlp", "_solve_node_nlp_ipopt", "_ipopt_version"}, frames
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


# ── Mach-O dependency walk (the torch-then-cyipopt direction) ──────────────
#
# Measured on the issue's macOS arm64 machine: cyipopt's RECORD ships no libomp;
# its Ipopt loads /opt/homebrew/Cellar/libomp/<v>/lib/libomp.dylib. A RECORD-only
# check therefore saw nothing and torch-then-cyipopt still aborted. These tests
# rebuild that layout from synthetic Mach-O files.

_LC_LOAD_DYLIB = 0xC
_LC_LOAD_WEAK_DYLIB = 0x80000018
_LC_RPATH = 0x8000001C
_ARM64 = 0x0100000C
_X86_64 = 0x01000007


def _pad8(b: bytes) -> bytes:
    b += b"\0"
    return b + b"\0" * (-len(b) % 8)


def _macho(libs=(), rpaths=(), *, weak=(), cputype=_ARM64) -> bytes:
    """A minimal 64-bit little-endian Mach-O dylib with the given load commands."""
    cmds = b""
    for cmd, name in [(_LC_LOAD_DYLIB, n) for n in libs] + [(_LC_LOAD_WEAK_DYLIB, n) for n in weak]:
        body = _pad8(name.encode())
        cmds += struct.pack("<6I", cmd, 24 + len(body), 24, 2, 0x10000, 0x10000) + body
    for path in rpaths:
        body = _pad8(path.encode())
        cmds += struct.pack("<3I", _LC_RPATH, 12 + len(body), 12) + body
    ncmds = len(libs) + len(weak) + len(rpaths)
    header = struct.pack("<8I", 0xFEEDFACF, cputype, 0, 6, ncmds, len(cmds), 0, 0)
    return header + cmds


def _fat(*slices: bytes, cputypes) -> bytes:
    """A universal binary holding ``slices`` (one per entry of ``cputypes``)."""
    out = struct.pack(">II", 0xCAFEBABE, len(slices))
    offset = 4096
    body = b""
    for blob, cpu in zip(slices, cputypes):
        out += struct.pack(">iiIII", cpu, 0, offset, len(blob), 12)
        body += blob + b"\0" * (4096 - len(blob) % 4096)
        offset += len(blob) + (4096 - len(blob) % 4096)
    return out + b"\0" * (4096 - len(out)) + body


def _write(path, data: bytes) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return str(path)


@pytest.fixture
def homebrew_cyipopt(tmp_path):
    """cyipopt's extension -> Homebrew Ipopt -> MUMPS -> Homebrew libomp (via opt/ symlink)."""
    brew = tmp_path / "homebrew"
    omp = _write(brew / "Cellar/libomp/23.1.5/lib/libomp.dylib", _macho())
    os.makedirs(brew / "opt", exist_ok=True)
    os.symlink(brew / "Cellar/libomp/23.1.5", brew / "opt/libomp")
    # MUMPS reaches libomp through an absolute install name under opt/ (a symlink).
    _write(
        brew / "Cellar/mumps/5.7/lib/libdmumps.dylib",
        _macho([str(brew / "opt/libomp/lib/libomp.dylib"), "/usr/lib/libSystem.B.dylib"]),
    )
    # Ipopt reaches MUMPS through @rpath, from an rpath relative to itself.
    _write(
        brew / "Cellar/ipopt/3.14/lib/libipopt.3.dylib",
        _macho(["@rpath/libdmumps.dylib"], rpaths=["@loader_path/../../../mumps/5.7/lib"]),
    )
    site = tmp_path / "site-packages"
    ext = _write(
        site / "cyipopt/cython/ipopt_wrapper.cpython-313-darwin.so",
        _macho([str(brew / "Cellar/ipopt/3.14/lib/libipopt.3.dylib"), "/usr/lib/libc++.1.dylib"]),
    )
    return ext, os.path.realpath(omp)


def test_linked_runtime_found_through_the_dependency_chain(homebrew_cyipopt):
    ext, omp = homebrew_cyipopt
    assert guard.macho_load_commands(ext) is not None
    assert guard.linked_omp_runtimes([ext]) == {omp}


def test_torch_first_then_homebrew_cyipopt_refused(macos, monkeypatch, homebrew_cyipopt):
    """The direction the RECORD-only check missed: cyipopt ships no runtime, links one."""
    ext, omp = homebrew_cyipopt
    uncached = guard.package_omp_runtimes.__wrapped__  # the real computation, no cache
    macos(loaded=set(), shipped=set())  # macOS, neither package imported
    monkeypatch.setattr(guard, "package_omp_runtimes", uncached)
    monkeypatch.setattr(guard, "shipped_omp_runtimes", lambda dist: set())
    monkeypatch.setattr(guard, "_package_binaries", lambda dist, module: [ext])

    monkeypatch.setattr(guard, "loaded_omp_runtimes", lambda: {TORCH_OMP})
    with pytest.raises(guard.OpenMPRuntimeConflictError) as info:
        guard.check_openmp_compatible("cyipopt", feature="nlp_solver='ipopt'")
    assert omp in str(info.value) and TORCH_OMP in str(info.value)

    # The same install with that runtime already loaded (one shared libomp) passes.
    monkeypatch.setattr(guard, "loaded_omp_runtimes", lambda: {omp})
    guard.check_openmp_compatible("cyipopt")


def test_rpath_inherited_from_the_loader(tmp_path):
    """dyld resolves @rpath in a dependency against its loader's LC_RPATHs too."""
    omp = _write(tmp_path / "rt/libomp.dylib", _macho())
    _write(tmp_path / "lib/libdep.dylib", _macho(["@rpath/libomp.dylib"]))
    top = _write(
        tmp_path / "top.so",
        _macho(["@loader_path/lib/libdep.dylib"], rpaths=[str(tmp_path / "rt")]),
    )
    assert guard.linked_omp_runtimes([top]) == {os.path.realpath(omp)}


def test_weak_dylib_counts_and_missing_or_system_deps_are_skipped(tmp_path):
    omp = _write(tmp_path / "libiomp5.dylib", _macho())
    top = _write(
        tmp_path / "top.so",
        _macho(
            ["/usr/lib/libSystem.B.dylib", str(tmp_path / "missing.dylib")],
            weak=[str(tmp_path / "libiomp5.dylib")],
        ),
    )
    assert guard.linked_omp_runtimes([top]) == {os.path.realpath(omp)}


def test_dependency_cycle_terminates(tmp_path):
    a, b = tmp_path / "liba.dylib", tmp_path / "libb.dylib"
    _write(a, _macho([str(b)]))
    _write(b, _macho([str(a)]))
    assert guard.linked_omp_runtimes([str(a)]) == set()


def test_non_macho_inputs_are_ignored(tmp_path):
    elf = _write(tmp_path / "x.so", b"\x7fELF" + b"\0" * 60)
    short = _write(tmp_path / "y.so", b"\xcf\xfa")
    assert guard.macho_load_commands(elf) is None
    assert guard.macho_load_commands(short) is None
    assert guard.macho_load_commands(str(tmp_path / "absent.so")) is None
    assert guard.linked_omp_runtimes([elf, short, str(tmp_path / "absent.so")]) == set()


@pytest.mark.parametrize("machine, expected", [("arm64", "arm"), ("x86_64", "intel")])
def test_fat_binary_reads_this_machines_slice(tmp_path, monkeypatch, machine, expected):
    arm = _write(tmp_path / "arm/libomp.dylib", _macho())
    intel = _write(tmp_path / "intel/libomp.dylib", _macho())
    fat = _write(
        tmp_path / "fat.so",
        _fat(_macho([arm]), _macho([intel], cputype=_X86_64), cputypes=[_ARM64, _X86_64]),
    )
    monkeypatch.setattr(guard.platform, "machine", lambda: machine)
    libs, _ = guard.macho_load_commands(fat)
    assert libs == [arm if expected == "arm" else intel]
    assert guard.linked_omp_runtimes([fat]) == {
        os.path.realpath(arm if expected == "arm" else intel)
    }


def test_package_binaries_fall_back_to_the_package_directory(monkeypatch, tmp_path):
    """No RECORD (some conda installs): the package directory is walked instead."""
    pkg = tmp_path / "fakepkg2205"
    so = _write(pkg / "sub/ext.cpython-313-darwin.so", _macho())
    _write(pkg / "__init__.py", b"")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(guard, "_distribution_files", lambda dist: [])
    assert guard._package_binaries("fakepkg2205", "fakepkg2205") == [so]
    assert "fakepkg2205" not in sys.modules
