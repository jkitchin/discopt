"""The ``[dev]`` extra must provide what the GAMS-link tests import.

``python/tests/test_gams_link.py`` opens with "None of this requires a GAMS
installation" and is written to run in a plain developer environment -- yet three
of its tests do a bare ``import yaml`` with no ``importorskip`` guard, so they
*error* rather than skip when PyYAML is absent. PyYAML was declared only in the
``[gams]`` extra, which that file deliberately does not require.

CI never noticed because four of its jobs ``pip install`` pyyaml explicitly on
top of the extras, so the only environment that broke was the documented one:
``pip install -e ".[dev]"``, the command in CLAUDE.md.

This test derives the requirement from the test file itself rather than
hardcoding "pyyaml", so the next unguarded third-party import added to that file
is caught the same way.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_TARGET = Path(__file__).parent / "test_gams_link.py"

# Import name -> distribution name, for the cases where they differ.
_DIST_FOR_MODULE = {
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "PIL": "pillow",
}

_FIRST_PARTY = {"discopt", "support"}


def _unguarded_third_party_imports(path: Path) -> set[str]:
    """Top-level modules imported outside any ``try``/``except``.

    A ``try``-wrapped import is a deliberate optional dependency; a bare one is a
    hard requirement of the file.
    """
    tree = ast.parse(path.read_text())
    guarded: list[tuple[int, int]] = [
        (n.lineno, max(getattr(c, "end_lineno", n.lineno) for c in ast.walk(n)))
        for n in ast.walk(tree)
        if isinstance(n, ast.Try)
    ]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module.split(".")[0]]
        else:
            continue
        if any(lo <= node.lineno <= hi for lo, hi in guarded):
            continue
        for name in names:
            if name in sys.stdlib_module_names or name in _FIRST_PARTY or name.startswith("_"):
                continue
            found.add(name)
    return found


def _declared_distributions() -> set[str]:
    """Distributions a ``pip install -e ".[dev]"`` resolves: core deps + ``[dev]``."""
    tomllib = pytest.importorskip(
        "tomllib", reason="tomllib is stdlib from 3.11; this is a packaging-metadata check"
    )
    cfg = tomllib.loads(_PYPROJECT.read_text())["project"]
    specs = list(cfg.get("dependencies", [])) + list(
        cfg.get("optional-dependencies", {}).get("dev", [])
    )
    out = set()
    for spec in specs:
        # Strip version/marker/extras decoration: "ruff==0.14.6", "pounce[x]>=1".
        name = spec.split(";")[0].strip()
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
            name = name.split(sep)[0]
        out.add(name.strip().lower().replace("_", "-"))
    return out


def test_gams_link_tests_are_runnable_from_the_dev_extra_alone():
    imported = _unguarded_third_party_imports(_TARGET)
    assert imported, f"scanned {_TARGET.name} and found no third-party imports -- probe is dead"

    declared = _declared_distributions()
    missing = {
        mod: _DIST_FOR_MODULE.get(mod, mod)
        for mod in imported
        if _DIST_FOR_MODULE.get(mod, mod).lower().replace("_", "-") not in declared
    }
    assert not missing, (
        f"{_TARGET.name} imports {sorted(missing)} unguarded, but the dev extra does not "
        f"declare {sorted(missing.values())}. Either declare it in [project."
        f"optional-dependencies].dev or guard the import -- do not rely on CI installing it."
    )
    # Prove the check compared something (CLAUDE.md §6).
    assert len(imported) >= 2, f"only {len(imported)} imports examined: {sorted(imported)}"
