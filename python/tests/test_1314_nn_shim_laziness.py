"""#1314: the ``discopt.nn`` shim must be no heavier and no quieter than ``discopt.ml``.

Two gaps between what the #1253 shim promised and what it did:

1. ``import discopt.nn`` eagerly walked the whole ``discopt.ml`` tree, pulling in
   ``readers/*``, ``predictor`` and ``formulations.tree_ensemble`` — modules
   ``import discopt.ml`` never touches. That made the *deprecated* path strictly
   less robust than its target: one eager third-party import anywhere under
   ``discopt.ml`` would break ``import discopt.nn`` for every caller while
   leaving ``import discopt.ml`` unaffected. It was also a regression against the
   pre-rename ``discopt.nn``, whose ``__init__`` was exactly as lazy.
2. The ``DeprecationWarning`` is ignored by Python's default filters anywhere but
   ``__main__``, so a downstream *library* module doing ``import discopt.nn`` —
   the dominant real-world pattern — saw nothing at all. ``pytest`` overrides
   those filters globally, which is why the existing shim test cannot see this.

Both checks run in a fresh subprocess: ``discopt.nn``/``discopt.ml`` are already
in ``sys.modules`` by the time this file runs under pytest, and deleting entries
would not un-import them. The subprocesses run with default warning filters
(``-W`` untouched), which is the state under test in part 2.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap

import pytest

_PROBE_IMPORT_SURFACE = textwrap.dedent(
    """
    import json, sys
    before = set(sys.modules)
    import {module}
    after = set(sys.modules)
    print(json.dumps(sorted(m for m in (after - before) if m.startswith("discopt.ml"))))
    """
)


def _import_surface(module: str) -> set[str]:
    """``discopt.ml.*`` modules pulled into ``sys.modules`` by importing ``module``."""
    import json

    out = subprocess.run(
        [sys.executable, "-c", _PROBE_IMPORT_SURFACE.format(module=module)],
        capture_output=True,
        text=True,
        check=True,
    )
    # The shim logs its deprecation notice to stderr; the payload is the last
    # stdout line, so a future banner cannot silently corrupt the parse.
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    assert lines, f"probe printed nothing (stderr: {out.stderr!r})"
    surface = set(json.loads(lines[-1]))
    assert surface, f"probe imported no discopt.ml modules at all (stderr: {out.stderr!r})"
    return surface


@pytest.mark.unit
def test_shim_import_surface_matches_discopt_ml():
    """The deprecated path must not import more of the tree than the real one."""
    ml_surface = _import_surface("discopt.ml")
    nn_surface = _import_surface("discopt.nn")

    extra = nn_surface - ml_surface
    assert not extra, (
        "importing discopt.nn pulled in modules that importing discopt.ml does not: "
        f"{sorted(extra)}"
    )
    assert nn_surface == ml_surface


@pytest.mark.unit
def test_optional_dependency_readers_are_not_imported_by_the_shim():
    """The named regression: the optional readers stayed out of ``import discopt.ml``."""
    nn_surface = _import_surface("discopt.nn")
    readers = {m for m in nn_surface if ".readers" in m}
    assert not readers, f"shim eagerly imported optional readers: {sorted(readers)}"


@pytest.mark.unit
def test_submodules_still_resolve_lazily_and_by_identity():
    """Laziness must not cost the identity guarantee the alias exists for."""
    checked = 0
    for suffix in (
        "network",
        "formulations.base",
        "formulations.tree_ensemble",
        "readers.sklearn_reader",
        "predictor",
        "tree",
    ):
        alias = importlib.import_module(f"discopt.nn.{suffix}")
        real = importlib.import_module(f"discopt.ml.{suffix}")
        assert alias is real, f"discopt.nn.{suffix} is not discopt.ml.{suffix}"
        checked += 1
    assert checked == 6, f"{checked} submodules compared, expected 6"


@pytest.mark.unit
def test_a_missing_submodule_still_raises_module_not_found():
    """The finder must decline names ``discopt.ml`` does not have, not swallow them."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("discopt.nn.definitely_not_a_module")


@pytest.mark.unit
def test_deprecation_is_observable_from_an_ordinary_library_module(tmp_path):
    """Part 2: the realistic case — imported from library code, default filters.

    ``warnings.warn(DeprecationWarning)`` alone is silent here; the shim's second
    channel (a ``logging`` WARNING, which reaches stderr through
    ``logging.lastResort`` when the application configures no logging) is what
    makes the deprecation observable.
    """
    pkg = tmp_path / "downstream_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "util.py").write_text("import discopt.nn\n")

    out = subprocess.run(
        [sys.executable, "-c", "import downstream_pkg.util"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    combined = out.stdout + out.stderr
    assert "discopt.nn is deprecated" in combined, (
        "importing discopt.nn from a library module produced no visible deprecation "
        f"notice under default warning filters (stdout={out.stdout!r}, stderr={out.stderr!r})"
    )
    assert "discopt.ml" in combined


@pytest.mark.unit
def test_deprecation_warning_is_still_a_real_warning():
    """The logging channel is additive: ``-W error::DeprecationWarning`` must still fire."""
    out = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", "import discopt.nn"],
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0, "DeprecationWarning no longer raised under -W error"
    assert "DeprecationWarning" in out.stderr
