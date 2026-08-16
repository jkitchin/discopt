"""No shipped module may need a newer Python than ``requires-python`` promises.

``pyproject.toml`` declares ``requires-python = ">=3.10"`` and, since #1055, the
release publishes a single ``abi3`` wheel per platform that pip will happily
install on 3.10. That makes the floor a real promise for the first time -- and it
was already broken in two places when the wheels were fixed:

* ``discopt/skills/__init__.py`` did ``from importlib.resources.abc import
  Traversable`` at module scope. That submodule is 3.11+, so ``import
  discopt.skills`` -- and therefore ``discopt install-skills`` -- raised
  ``ModuleNotFoundError`` on 3.10.
* ``python/tests/_optima.py`` did a bare ``import tomllib`` (3.11+), which turned
  the *entire correctness tier* into a collection error on 3.10, since that
  module is the reference-optima oracle.

Neither was caught by CI: every job runs 3.12, so nothing ever executed an import
on the floor. This test does it statically instead, on whatever interpreter is
running, and covers the class rather than those two files -- the next 3.11-only
import added anywhere under ``python/`` fails here.

A guarded import is fine: inside ``try``/``except``, inside ``if TYPE_CHECKING``,
or inside a ``sys.version_info`` branch. Only an unconditional module-scope
import is a violation.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_SCAN_ROOTS = (_REPO_ROOT / "python" / "discopt", _REPO_ROOT / "python" / "tests")

# Stdlib module -> the first Python version that has it.
_NEW_MODULES: dict[str, tuple[int, int]] = {
    "tomllib": (3, 11),
    "importlib.resources.abc": (3, 11),
    "asyncio.taskgroups": (3, 11),
    "wsgiref.types": (3, 11),
    "importlib.metadata.diagnose": (3, 13),
    "dbm.sqlite3": (3, 13),
    "compression": (3, 14),
    "annotationlib": (3, 14),
}

# (module, imported name) -> the first Python version that has the name.
_NEW_NAMES: dict[tuple[str, str], tuple[int, int]] = {
    ("typing", "Self"): (3, 11),
    ("typing", "Never"): (3, 11),
    ("typing", "LiteralString"): (3, 11),
    ("typing", "assert_never"): (3, 11),
    ("typing", "assert_type"): (3, 11),
    ("typing", "reveal_type"): (3, 11),
    ("typing", "TypeVarTuple"): (3, 11),
    ("typing", "Unpack"): (3, 11),
    ("typing", "dataclass_transform"): (3, 11),
    ("typing", "get_overloads"): (3, 11),
    ("typing", "override"): (3, 12),
    ("typing", "TypeAliasType"): (3, 12),
    ("typing", "ReadOnly"): (3, 13),
    ("typing", "TypeIs"): (3, 13),
    ("enum", "StrEnum"): (3, 11),
    ("enum", "ReprEnum"): (3, 11),
    ("enum", "EnumCheck"): (3, 11),
    ("enum", "verify"): (3, 11),
    ("enum", "member"): (3, 11),
    ("enum", "nonmember"): (3, 11),
    ("datetime", "UTC"): (3, 11),
    ("asyncio", "TaskGroup"): (3, 11),
    ("asyncio", "Runner"): (3, 11),
    ("asyncio", "timeout"): (3, 11),
    ("asyncio", "timeout_at"): (3, 11),
    ("contextlib", "chdir"): (3, 11),
    ("hashlib", "file_digest"): (3, 11),
    ("itertools", "batched"): (3, 12),
    ("pathlib", "UnsupportedOperation"): (3, 13),
    ("warnings", "deprecated"): (3, 13),
    ("os", "process_cpu_count"): (3, 13),
}


def _floor() -> tuple[int, int]:
    """The ``requires-python`` floor, as a version tuple."""
    text = _PYPROJECT.read_text()
    m = re.search(r'requires-python\s*=\s*"[^"]*?>=\s*(\d+)\.(\d+)', text)
    assert m, "could not read requires-python from pyproject.toml"
    return int(m.group(1)), int(m.group(2))


def _is_guarded(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """True if ``node`` sits under try/except, ``TYPE_CHECKING``, or a version check."""
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, ast.Try):
            return True
        if isinstance(cur, ast.If):
            test = ast.dump(cur.test)
            if "TYPE_CHECKING" in test or "version_info" in test:
                return True
        cur = parents.get(cur)
    return False


def _violations(floor: tuple[int, int]) -> tuple[list[str], int]:
    """Return (violation messages, number of imports actually examined)."""
    found: list[str] = []
    examined = 0

    for root in _SCAN_ROOTS:
        assert root.is_dir(), f"scan root missing: {root}"
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            parents: dict[ast.AST, ast.AST] = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    targets = [(alias.name, None) for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    mod = node.module or ""
                    targets = [(mod, None)] + [(mod, a.name) for a in node.names]
                else:
                    continue

                guarded = _is_guarded(node, parents)
                rel = path.relative_to(_REPO_ROOT)
                for mod, name in targets:
                    examined += 1
                    need = _NEW_MODULES.get(mod) if name is None else _NEW_NAMES.get((mod, name))
                    if need is None or need <= floor or guarded:
                        continue
                    what = f"from {mod} import {name}" if name else f"import {mod}"
                    found.append(
                        f"{rel}:{node.lineno}: {what} needs Python "
                        f"{need[0]}.{need[1]}, but requires-python floor is "
                        f"{floor[0]}.{floor[1]}"
                    )
    return found, examined


@pytest.mark.smoke
def test_no_unguarded_imports_above_requires_python_floor():
    floor = _floor()
    violations, examined = _violations(floor)

    # Prove the probe fired (CLAUDE.md §6). A refactor that moves the package or
    # breaks the AST walk would otherwise report a clean pass having read nothing.
    assert examined > 1000, (
        f"only {examined} imports examined across {[str(r) for r in _SCAN_ROOTS]} -- "
        "the scan is not reaching the source tree"
    )

    assert not violations, (
        f"{len(violations)} import(s) require a newer Python than the "
        f"{floor[0]}.{floor[1]} floor that pyproject.toml promises and the abi3 "
        "wheel delivers:\n  " + "\n  ".join(violations) + "\n"
        "Guard with try/except, `if TYPE_CHECKING:`, or `if sys.version_info >= "
        "(3, N):` -- or raise the floor in pyproject.toml (and the abi3-pyXY "
        "feature in Cargo.toml with it)."
    )


@pytest.mark.smoke
def test_the_floor_scanner_detects_a_known_violation(tmp_path, monkeypatch):
    """The scanner must fail on a real violation, not just pass on clean code.

    Without this, a typo in `_NEW_MODULES` or a broken guard check would make the
    test above vacuous while still reporting green.
    """
    bad = tmp_path / "pkg"
    bad.mkdir()
    (bad / "unguarded.py").write_text("import tomllib\n")
    (bad / "guarded_try.py").write_text("try:\n    import tomllib\nexcept ImportError:\n    pass\n")
    (bad / "guarded_version.py").write_text(
        "import sys\nif sys.version_info >= (3, 11):\n    import tomllib\n"
    )
    (bad / "guarded_typing.py").write_text(
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
        "    from importlib.resources.abc import Traversable\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "_SCAN_ROOTS", (bad,))
    monkeypatch.setattr(sys.modules[__name__], "_REPO_ROOT", tmp_path)

    violations, examined = _violations((3, 10))
    # 6 `import X` / bare-module entries plus 2 from-import names.
    assert examined == 8, f"expected 8 import targets examined, got {examined}"
    assert len(violations) == 1, f"expected exactly the unguarded one, got {violations}"
    assert "unguarded.py" in violations[0]
