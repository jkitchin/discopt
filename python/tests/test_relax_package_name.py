"""``discopt._jax`` is gone; nothing may reintroduce it.

The package was named ``_jax`` when JAX built the relaxations. It no longer
does -- the 128 modules under ``discopt._relax`` are numpy, and JAX does not
enter ``sys.modules`` during a solve at all. A package whose name describes
none of its contents is a standing source of wrong conclusions: a reader
grepping for JAX on the solve path finds 1818 import lines and believes it.

The guard matters more than the rename. Every other branch, and ``main`` until
this merges, still writes ``discopt._jax``; a merge that reintroduces one of
those lines would fail at import, but only on whichever test happens to touch
that module. This fails immediately and says why.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
#: This file's path relative to the repo root, for the git-grep pathspec below.
_SELF = Path(__file__).resolve().relative_to(_REPO).as_posix()


def test_the_old_package_name_is_not_importable():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("discopt._jax")


def test_the_new_package_is_importable_and_populated():
    """Guards the assertion above from passing for the wrong reason: if the
    package were simply deleted, the ``ModuleNotFoundError`` test would still
    pass."""
    pkg = importlib.import_module("discopt._relax")
    modules = list(Path(pkg.__path__[0]).glob("*.py"))
    assert len(modules) > 50, f"only {len(modules)} modules under discopt._relax"


def test_no_source_file_references_the_old_name():
    """A merge from a branch predating the rename is the realistic way this
    breaks, and ``git grep`` over the tracked tree is what catches it.

    ``CHANGELOG.md`` and ``docs/dev/`` are exempt so that a historical entry or
    a dated diagnosis *may* keep the old spelling where it is describing the
    tree as it was. Both happen to be clean right now; the exemption is there so
    that keeping the record honest never has to fight this test.

    This file is excluded too, and not as a convenience: it has to write the old
    name to search for it, so it matches itself. That is also how this test
    first shipped broken -- it passed locally while still untracked, because
    ``git grep`` only searches tracked files, and went red the moment it was
    committed. The exclusion is a pathspec passed to ``git grep`` rather than a
    filter on the output, so a genuine hit inside this file's *other* lines
    cannot hide behind it either way.
    """
    proc = subprocess.run(
        [
            "git",
            "grep",
            "-nE",
            r"discopt\._jax|discopt/_jax",
            "--",
            ".",
            f":!{_SELF}",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    # git grep exits 1 for "no matches", which is the passing case; any other
    # non-zero status is a broken probe and must not read as success.
    if proc.returncode not in (0, 1):
        pytest.fail(f"git grep failed (rc={proc.returncode}): {proc.stderr.strip()}")

    exempt = ("CHANGELOG.md", "docs/dev/")
    hits = [ln for ln in proc.stdout.splitlines() if ln.strip() and not ln.startswith(exempt)]
    assert not hits, "the old package name came back:\n" + "\n".join(hits[:20])
