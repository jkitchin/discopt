"""A stale interpreter-specific ``.so`` must never shadow the built extension.

``crates/discopt-python`` builds an **abi3** wheel, so ``maturin develop`` emits
``python/discopt/_rust.abi3.so``. CPython's extension finder prefers the
interpreter-specific name (``_rust.cpython-312-darwin.so``) over the abi3 one,
so a leftover file under that name wins every import -- silently, and forever,
because nothing rebuilds it.

That happened. The ``Makefile`` named ``python/discopt/_rust$(EXT_SUFFIX)`` with
``EXT_SUFFIX`` read from ``sysconfig``, i.e. the interpreter-specific name that
maturin does not produce. ``make build`` therefore ran maturin (which wrote the
abi3 file), found nothing to copy, and ``touch``-ed the *stale* file to satisfy
the rule -- reporting "Rust extension ready" while every ``import discopt._rust``
kept loading three-day-old Rust. Measured 2026-08-19 in the primary checkout:
``solve_milp_lazy_csc_py`` (the native single-tree entry added days earlier) was
absent from the module actually imported, and two probes were run and believed
against it before the shadow was found.

CLAUDE.md §8 says to verify which code you loaded. This is that check, made
permanent: no measurement in this tree can be trusted while a shadow exists.
"""

from __future__ import annotations

from pathlib import Path

import discopt._rust
import pytest

_PKG = Path(discopt.__file__).resolve().parent


@pytest.mark.smoke
def test_no_interpreter_specific_extension_shadows_the_abi3_build():
    shadows = sorted(p.name for p in _PKG.glob("_rust.cpython-*.so"))
    abi3 = _PKG / "_rust.abi3.so"
    if not abi3.exists():
        # An installed (non-editable) tree may ship only one extension; there is
        # nothing to shadow. Assert that is the situation rather than passing
        # vacuously -- exactly one extension must be present.
        assert len(shadows) <= 1, f"several extensions and no abi3 build: {shadows}"
        return
    assert not shadows, (
        f"stale extension(s) {shadows} shadow {abi3.name}; "
        "`import discopt._rust` is loading them instead of the current build. "
        "Remove them and re-run `make build`."
    )


@pytest.mark.smoke
def test_the_imported_extension_is_the_one_on_disk():
    """Guards the above from passing for the wrong reason: with no shadow, the
    module actually imported must be the file the build writes."""
    loaded = Path(discopt._rust.__file__).resolve()
    assert loaded.parent == _PKG, f"imported {loaded}, expected inside {_PKG}"
    assert loaded.suffixes[-2:] in ([".abi3", ".so"], [".so"]), loaded.name
