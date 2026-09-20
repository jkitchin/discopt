"""Every third-party import in the shipped package is declared or guarded.

The v0.9.0 release audit found three imports in ``python/discopt/`` that were
neither installed by any dependency/extra nor wrapped in a guard, so a
pip-installed user calling the public method behind them got a naked
``ModuleNotFoundError`` that reads as a broken package:

* ``interfaces/cutest.py`` -- ``from benchmarks.metrics import InstanceInfo``.
  ``benchmarks`` is the *harness* package under ``discopt_benchmarks/``, and
  maturin's ``python-source = "python"`` puts only ``python/discopt/**`` in the
  wheel, so that module ships for nobody. (Near-miss worth naming: the wheel
  *does* contain ``discopt.benchmarks.metrics``, which does not define
  ``InstanceInfo`` -- so this was never a missing ``discopt.`` prefix.)
* ``ml/readers/torch_reader.py`` -- ``import torch.nn``; the ``nn`` extra is the
  ONNX toolchain and has never included torch.
* ``mo/pareto.py`` -- ``import matplotlib.pyplot`` in ``ParetoFront.plot()``.

``solvers/gurobi.py`` already had the right shape (try/except ImportError with an
actionable message), which is what the other three were made to match.

This test pins the **class**, not those three instances: any new undeclared,
unguarded third-party import fails here. Two ways to satisfy it -- declare the
distribution in ``[project] dependencies`` / ``[project.optional-dependencies]``,
or wrap the import in ``try: ... except ImportError:`` and raise something that
tells the user what to install.

Deliberately static (``ast`` + ``tomllib``): it imports neither discopt nor any
optional dependency, so it runs on a bare checkout with no build.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
import tomllib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_PKG = _REPO / "python" / "discopt"

# Distribution name -> the module root it actually provides, where they differ.
_DIST_TO_IMPORT_ROOT = {
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
    "pounce-solver": "pounce",
    "gamsapi": "gams",
    "pytest-timeout": "pytest_timeout",
    "pytest-cov": "pytest_cov",
    "pytest-xdist": "xdist",
    "pre-commit": "pre_commit",
}

# Import roots that are ours, or are resolved from inside the repo rather than
# from a distribution.
_FIRST_PARTY = {"discopt"}


def _declared_import_roots() -> set[str]:
    with (_REPO / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    project = pyproject["project"]
    specs = list(project.get("dependencies", []))
    for extra_specs in project.get("optional-dependencies", {}).values():
        specs.extend(extra_specs)

    roots: set[str] = set()
    for spec in specs:
        # "scikit-learn>=1.0", "gamsapi[core]>=45", "discopt[pounce,ipopt]"
        name = re.split(r"[<>=!~\[;\s]", spec.strip(), maxsplit=1)[0].lower()
        if not name or name == "discopt":
            continue
        roots.add(_DIST_TO_IMPORT_ROOT.get(name, name.replace("-", "_")))
    return roots


def _guarded_import_nodes(tree: ast.AST) -> set[int]:
    """ids() of import nodes sitting under a ``try`` that catches ImportError."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        catches_import_error = False
        for handler in node.handlers:
            exc = handler.type
            names = []
            if isinstance(exc, ast.Name):
                names = [exc.id]
            elif isinstance(exc, ast.Tuple):
                names = [e.id for e in exc.elts if isinstance(e, ast.Name)]
            elif exc is None:  # bare except
                names = ["ImportError"]
            if {"ImportError", "ModuleNotFoundError"} & set(names):
                catches_import_error = True
        if not catches_import_error:
            continue
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    guarded.add(id(sub))
    return guarded


def test_every_third_party_import_is_declared_or_guarded():
    declared = _declared_import_roots()
    stdlib = set(sys.stdlib_module_names)
    assert declared, "parsed no dependencies out of pyproject.toml -- parser drifted"

    files = sorted(_PKG.rglob("*.py"))
    assert files, "probe found no package sources"

    imports_checked = 0
    offenders: dict[str, list[str]] = {}

    for path in files:
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - would be caught by ruff first
            continue
        guarded = _guarded_import_nodes(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [(a.name.split(".")[0], node) for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level or not node.module:  # relative import
                    continue
                roots = [(node.module.split(".")[0], node)]
            else:
                continue

            for root, owner in roots:
                imports_checked += 1
                if root in stdlib or root in _FIRST_PARTY or root in declared:
                    continue
                if id(owner) in guarded:
                    continue
                rel = path.relative_to(_REPO)
                offenders.setdefault(root, []).append(f"{rel}:{owner.lineno}")

    assert imports_checked > 0, "probe walked no import statements -- the AST scan is a no-op"

    assert not offenders, (
        "third-party imports that are neither declared in pyproject.toml nor guarded "
        "by try/except ImportError -- a pip-installed user hits a naked "
        f"ModuleNotFoundError here: {offenders}"
    )


@pytest.mark.parametrize(
    ("module_path", "needle"),
    [
        ("interfaces/cutest.py", "discopt_benchmarks && pip install"),
        ("ml/readers/torch_reader.py", "discopt[torch]"),
        ("mo/pareto.py", "discopt[plot]"),
        ("solvers/gurobi.py", "configure a working Gurobi license"),
    ],
)
def test_the_guard_message_tells_the_user_what_to_install(module_path, needle):
    """A guard that raises without naming the fix is only half the repair."""
    text = (_PKG / module_path).read_text()
    assert needle in text, (
        f"{module_path} no longer tells the user how to satisfy the missing import "
        f"(looked for {needle!r})"
    )
