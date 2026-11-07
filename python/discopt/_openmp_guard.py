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
2. the OpenMP runtimes the package about to be imported would load, found
   without importing it (:func:`package_omp_runtimes`): the runtimes its wheel
   ships (the distribution's ``RECORD``), plus every runtime its Mach-O binaries
   link, following ``LC_LOAD_DYLIB`` dependencies (``@rpath`` / ``@loader_path``
   resolved as ``dyld`` does) through non-system libraries. The second part is
   what sees a cyipopt whose Ipopt links Homebrew's ``libomp`` and ships none:
   measured on the macOS arm64 machine of the issue, ``RECORD`` alone found
   nothing for cyipopt, and torch-then-cyipopt still aborted;

and raises :class:`OpenMPRuntimeConflictError` when the package would load a
runtime and a *different* file is already loaded. Two installs that resolve to one
shared ``libomp`` (a conda environment, where both packages link ``$PREFIX/lib``)
name the same file and pass.

What it cannot see: a runtime a package ``dlopen``s at run time rather than
linking (nothing in its load commands names it), and a bare ``import torch`` /
``import cyipopt`` in the caller's own code, which no library guard can intercept.

``KMP_DUPLICATE_LIB_OK=TRUE`` is honoured as the user's explicit choice (OpenMP
itself then does not abort) and only produces a :class:`RuntimeWarning`. discopt
never sets it: OpenMP's own message says it "may cause crashes or silently
produce incorrect results", which a solver cannot accept by default.

Off macOS the check returns immediately; Linux wheels of the two packages do not
collide (``libgomp`` / a single ``libomp`` load).
"""

from __future__ import annotations

import functools
import os
import platform
import re
import struct
import sys
import warnings
from collections.abc import Iterable, Iterator

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


def _distribution_files(distribution: str) -> list[str]:
    """Absolute paths of the files ``distribution`` installed (from ``RECORD``)."""
    from importlib import metadata

    try:
        files = metadata.files(distribution)
    except metadata.PackageNotFoundError:
        return []
    return [str(f.locate()) for f in files or ()]


def shipped_omp_runtimes(distribution: str) -> set[str]:
    """Real paths of the OpenMP runtimes ``distribution`` installs, without importing it.

    Read from the distribution's file list (``RECORD``). Empty when the
    distribution is not installed or ships none (it may still *link* an external
    one; see :func:`package_omp_runtimes`).
    """
    return {_real(f) for f in _distribution_files(distribution) if is_omp_runtime(f)}


# ── Mach-O load commands ────────────────────────────────────────────────────
#
# Just enough of <mach-o/loader.h> and <mach-o/fat.h> to list the libraries a
# binary links and the rpaths it declares. Everything is read from the file; the
# parser does not depend on running on macOS, which is what lets the tests drive
# it with synthetic binaries on Linux.

_MH_MAGIC = 0xFEEDFACE
_MH_MAGIC_64 = 0xFEEDFACF
_FAT_MAGIC = 0xCAFEBABE
_FAT_MAGIC_64 = 0xCAFEBABF
_LC_REQ_DYLD = 0x80000000
_LC_LOAD_DYLIB = 0xC
_LC_LAZY_LOAD_DYLIB = 0x20
_LC_LOAD_WEAK_DYLIB = 0x18 | _LC_REQ_DYLD
_LC_REEXPORT_DYLIB = 0x1F | _LC_REQ_DYLD
_LC_LOAD_UPWARD_DYLIB = 0x23 | _LC_REQ_DYLD
_LC_RPATH = 0x1C | _LC_REQ_DYLD
_DYLIB_COMMANDS = {
    _LC_LOAD_DYLIB,
    _LC_LAZY_LOAD_DYLIB,
    _LC_LOAD_WEAK_DYLIB,
    _LC_REEXPORT_DYLIB,
    _LC_LOAD_UPWARD_DYLIB,
}
_CPU_TYPES = {"arm64": 0x0100000C, "x86_64": 0x01000007}

# Deterministic work budget for the dependency walk: the load commands of at most
# this many binaries are read per package (header + load commands only, never
# the whole file). Measured on macOS arm64 (#2207): cyipopt 0.00 s, torch 0.26 s once,
# then cached per process. Recorded in the #1456 pre-solve inventory, since an
# explicit nlp_solver="ipopt" runs this check before branch and bound starts.
_MACHO_WALK_MAX_FILES = 2000

# dyld's shared cache: never a third-party OpenMP runtime, and not files on disk.
_SYSTEM_PREFIXES = ("/usr/lib/", "/System/")


def _cstring(blob: bytes, offset: int) -> str:
    end = blob.find(b"\0", offset)
    return blob[offset : end if end >= 0 else len(blob)].decode("utf-8", "surrogateescape")


def _thin_load_commands(fh, base: int) -> tuple[list[str], list[str]] | None:
    """(linked libraries, rpaths) of the thin Mach-O image at ``base`` in ``fh``."""
    fh.seek(base)
    head = fh.read(28)
    if len(head) < 28:
        return None
    for endian in ("<", ">"):
        magic = struct.unpack(endian + "I", head[:4])[0]
        if magic in (_MH_MAGIC, _MH_MAGIC_64):
            break
    else:
        return None
    ncmds, sizeofcmds = struct.unpack(endian + "II", head[16:24])
    header_size = 32 if magic == _MH_MAGIC_64 else 28
    fh.seek(base + header_size)
    cmds = fh.read(sizeofcmds)
    libs: list[str] = []
    rpaths: list[str] = []
    pos = 0
    for _ in range(ncmds):
        if pos + 12 > len(cmds):
            break
        cmd, cmdsize, name_off = struct.unpack(endian + "III", cmds[pos : pos + 12])
        if cmdsize < 8:
            break
        blob = cmds[pos : pos + cmdsize]
        if cmd in _DYLIB_COMMANDS:
            libs.append(_cstring(blob, name_off))
        elif cmd == _LC_RPATH:
            rpaths.append(_cstring(blob, name_off))
        pos += cmdsize
    return libs, rpaths


def macho_load_commands(path: str) -> tuple[list[str], list[str]] | None:
    """(linked libraries, rpaths) of the Mach-O file ``path``, or ``None`` if it is not one.

    For a universal (fat) binary, the slice for this machine's architecture,
    else the first slice.
    """
    try:
        with open(path, "rb") as fh:
            magic_bytes = fh.read(8)
            if len(magic_bytes) < 8:
                return None
            magic = struct.unpack(">I", magic_bytes[:4])[0]
            if magic not in (_FAT_MAGIC, _FAT_MAGIC_64):
                return _thin_load_commands(fh, 0)
            nfat = struct.unpack(">I", magic_bytes[4:8])[0]
            entry = 32 if magic == _FAT_MAGIC_64 else 20
            fmt = ">iiQQI" if magic == _FAT_MAGIC_64 else ">iiIII"
            slices = []
            for i in range(min(nfat, 64)):
                fh.seek(8 + i * entry)
                raw = fh.read(entry)
                if len(raw) < entry:
                    break
                cputype, _sub, offset, _size, _align = struct.unpack(
                    fmt, raw[: struct.calcsize(fmt)]
                )
                slices.append((cputype, offset))
            if not slices:
                return None
            want = _CPU_TYPES.get(platform.machine())
            offset = next((o for c, o in slices if c == want), slices[0][1])
            return _thin_load_commands(fh, offset)
    except OSError:
        return None


def _expand(name: str, loader: str, rpaths: list[tuple[str, str]]) -> Iterator[str]:
    """Candidate files for the install name ``name`` linked by ``loader``.

    ``rpaths`` are ``(rpath, declaring binary)`` pairs from the loader chain, in
    dyld's search order; an rpath's own ``@loader_path`` is relative to the binary
    that declared it.
    """
    loader_dir = os.path.dirname(loader)
    exe_dir = os.path.dirname(sys.executable)

    def subst(text: str, declaring_dir: str) -> str:
        if text.startswith("@loader_path"):
            return declaring_dir + text[len("@loader_path") :]
        if text.startswith("@executable_path"):
            return exe_dir + text[len("@executable_path") :]
        return text

    if name.startswith("@rpath/"):
        rest = name[len("@rpath/") :]
        for rpath, declared_by in rpaths:
            yield os.path.join(subst(rpath, os.path.dirname(declared_by)), rest)
    else:
        yield subst(name, loader_dir)


def linked_omp_runtimes(
    binaries: Iterable[str], *, max_files: int = _MACHO_WALK_MAX_FILES
) -> set[str]:
    """Real paths of the OpenMP runtimes ``binaries`` load, following their dependencies.

    Walks ``LC_LOAD_DYLIB``-type commands from each Mach-O file in ``binaries``,
    resolving ``@rpath`` / ``@loader_path`` / ``@executable_path`` against the
    loader chain the way ``dyld`` does, and descends into every dependency that
    exists on disk outside the system prefixes. Non-Mach-O inputs are ignored.
    """
    found: set[str] = set()
    seen: set[str] = set()
    stack: list[tuple[str, list[tuple[str, str]]]] = [(_real(b), []) for b in binaries]
    while stack and len(seen) < max_files:
        path, inherited = stack.pop()
        if path in seen:
            continue
        seen.add(path)
        parsed = macho_load_commands(path)
        if parsed is None:
            continue
        libs, own_rpaths = parsed
        # dyld searches the image's own LC_RPATHs, then those of its loaders.
        chain = [(r, path) for r in own_rpaths] + inherited
        for name in libs:
            if name.startswith(_SYSTEM_PREFIXES):
                continue
            for candidate in _expand(name, path, chain):
                if not os.path.isfile(candidate):
                    continue
                real = _real(candidate)
                if is_omp_runtime(real) or is_omp_runtime(candidate):
                    found.add(real)
                stack.append((real, chain))
                break  # dyld takes the first candidate that exists
    return found


def _package_binaries(distribution: str, module: str) -> list[str]:
    """The Mach-O candidates of a package: ``.so``/``.dylib`` files it installed.

    From ``RECORD`` when there is one; otherwise (some conda installs) the files
    under the package's directory, located without importing it.
    """
    files = [f for f in _distribution_files(distribution) if f.endswith((".so", ".dylib"))]
    if files:
        return files
    import importlib.util

    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return []
    if spec is None:
        return []
    roots = list(spec.submodule_search_locations or [])
    if not roots and spec.origin:
        roots = [os.path.dirname(spec.origin)]
    out: list[str] = []
    for root in roots:
        for dirpath, _dirs, names in os.walk(root):
            out.extend(os.path.join(dirpath, n) for n in names if n.endswith((".so", ".dylib")))
    return out


@functools.lru_cache(maxsize=None)
def package_omp_runtimes(module: str) -> frozenset[str]:
    """Real paths of every OpenMP runtime importing ``module`` would load.

    The runtimes its distribution ships plus those its binaries link (see
    :func:`linked_omp_runtimes`). Cached per process: installed files do not
    change under a running interpreter in any way this check needs to follow.
    """
    distribution = _DISTRIBUTIONS.get(module, module)
    shipped = shipped_omp_runtimes(distribution)
    linked = linked_omp_runtimes(_package_binaries(distribution, module))
    return frozenset(shipped | linked)


def find_conflict(loaded: Iterable[str], shipped: Iterable[str]) -> set[str]:
    """The already-loaded runtimes that a package loading ``shipped`` would collide with.

    Empty when the package loads no runtime (nothing new is loaded) or when every
    loaded runtime is one the package would load anyway (it is already in use).
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
    shipped = set(package_omp_runtimes(module))
    conflict = find_conflict(loaded_omp_runtimes(), shipped)
    if not conflict:
        return
    loaded_by = [name for name in _DISTRIBUTIONS if name != module and name in sys.modules]
    message = (
        f"Importing {module!r}{f' (for {feature})' if feature else ''} would load a second "
        f"OpenMP runtime into this process, and on macOS that aborts it "
        f"('OMP: Error #15'). Already loaded: {', '.join(sorted(conflict))}"
        f"{' (already imported here: ' + ', '.join(loaded_by) + ')' if loaded_by else ''}; "
        f"{module} would load: {', '.join(sorted(shipped))}. "
        f"torch's wheels bundle their own libomp, cyipopt's Ipopt often links another "
        f"(e.g. Homebrew's), and some other wheels bundle one too (e.g. scikit-learn). "
        f"Use them in separate processes -- where torch is needed, nlp_solver='pounce' "
        f"(the default) instead of 'ipopt' -- or "
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
