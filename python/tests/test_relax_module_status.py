"""Guard: every ``_relax`` module is production-imported or declared (#1347).

``_relax`` is allowed to hold modules that nothing in the package imports --
optional user-facing entry points, design-time tooling, and tested capabilities
not yet wired in. What it is *not* allowed to do is leave that status implicit.
Issue #1231 tried to settle it with a ``grep`` and got three live modules wrong;
:mod:`discopt._relax.module_status` is the declaration, and this is the check.

Every module under ``discopt._relax`` must be either

1. imported by another package module (directly or transitively), or
2. declared in :data:`~discopt._relax.module_status.MODULE_STATUS`.

Reachability is computed from the real AST import graph rather than a text
search, because a substring match is what produced the wrong inventory in the
first place. Relative imports are resolved per PEP 328: getting that wrong
silently turns ``from .foo import bar`` into a self-edge and makes dead modules
look live, which is exactly how ``monotonicity`` was missed.

Per CLAUDE.md's measurement discipline, the tests here assert on a counted
number of resolved edges: a graph walk that quietly traverses nothing would
otherwise report "no undeclared modules" and read as a pass.
"""

from __future__ import annotations

import ast
import collections
import pathlib

import pytest
from discopt._relax.module_status import MODULE_STATUS, ModuleStatus

# Deliberately unmarked. This is a static guard over the AST import graph, not a
# solve: `smoke` means "one solve per code path, <1 s" and `unit` means "<0.1 s",
# and parsing ~1,100 files costs ~5 s. Unmarked tests run in the default suite and
# in CI's PR-fast lane, which is what gates this.

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1] / "discopt"
SRC_ROOT = PKG_ROOT.parent
TESTS_ROOT = pathlib.Path(__file__).resolve().parent
CATALOG = pathlib.Path(__file__).resolve().parents[2] / "docs/design/relaxation-catalog.md"
RELAX = "discopt._relax"

# The manifest is the declaration mechanism, so it cannot declare itself without
# circularity. Nothing else is exempt.
SELF_EXEMPT = {f"{RELAX}.module_status"}


def _module_name(path: pathlib.Path, root: pathlib.Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _python_files(root: pathlib.Path) -> list[pathlib.Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _anchor(module: str, is_package: bool) -> str:
    """The package a module's ``level=1`` relative imports resolve against."""
    if is_package:
        return module
    return module.rsplit(".", 1)[0] if "." in module else ""


def _import_targets(tree: ast.AST, module: str, is_package: bool) -> list[str]:
    """Absolute dotted names named by every import statement in ``tree``."""
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = _anchor(module, is_package)
                for _ in range(node.level - 1):
                    base = base.rsplit(".", 1)[0] if "." in base else ""
                if not base:  # escaped the package: not a discopt import
                    continue
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module or ""
            targets.append(target)
            # ``from pkg.mod import name`` may name a submodule, not an attribute.
            targets.extend(f"{target}.{alias.name}" for alias in node.names)
    return targets


@pytest.fixture(scope="module")
def graph() -> dict:
    """AST import graph over the package, plus the test suite's importers."""
    modules = {_module_name(p, SRC_ROOT): p for p in _python_files(PKG_ROOT)}
    packages = {m for m, p in modules.items() if p.name == "__init__.py"}

    importers: dict[str, set[str]] = collections.defaultdict(set)
    imports: dict[str, set[str]] = collections.defaultdict(set)
    edges = 0
    relative = 0
    for module, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative += sum(1 for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level)
        for target in _import_targets(tree, module, path.name == "__init__.py"):
            edges += 1
            resolved = target if target in modules else target.rsplit(".", 1)[0]
            if resolved in modules and resolved != module:
                importers[resolved].add(module)
                imports[module].add(resolved)

    test_importers: dict[str, set[str]] = collections.defaultdict(set)
    test_edges = 0
    for path in _python_files(TESTS_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for target in _import_targets(tree, "", False):
            test_edges += 1
            resolved = target if target in modules else target.rsplit(".", 1)[0]
            if resolved in modules:
                test_importers[resolved].add(path.name)

    return {
        "modules": modules,
        "packages": packages,
        "importers": importers,
        "imports": imports,
        "test_importers": test_importers,
        "edges": edges,
        "relative": relative,
        "test_edges": test_edges,
    }


def _production_imported(name: str, graph: dict) -> bool:
    """True if ``name`` is reachable from another package module.

    A package counts as reachable when any of its submodules is: importing
    ``pkg.sub`` executes ``pkg/__init__.py``.
    """
    if graph["importers"].get(name):
        return True
    if name in graph["packages"]:
        prefix = name + "."
        return any(graph["importers"].get(m) for m in graph["modules"] if m.startswith(prefix))
    return False


def _relax_modules(graph: dict) -> list[str]:
    return sorted(m for m in graph["modules"] if m == RELAX or m.startswith(RELAX + "."))


def test_import_graph_actually_traversed(graph):
    """The probe fired: a graph that resolved nothing must not read as a pass."""
    assert graph["edges"] > 1000, f"only {graph['edges']} import edges -- graph walk is broken"
    assert graph["relative"] > 50, (
        f"only {graph['relative']} relative imports seen; the package has ~120. "
        "Mis-resolved relative imports make dead modules look live."
    )
    assert graph["test_edges"] > 1000, f"only {graph['test_edges']} test import edges"
    assert len(_relax_modules(graph)) > 100, "did not find the _relax package"


def test_relative_import_resolution_is_correct(graph):
    """PEP 328 anchoring: a sibling import must not resolve to a self-edge."""
    # ``_relax/factorable_reform.py`` does ``from .term_classifier import ...``.
    assert f"{RELAX}.factorable_reform" in graph["importers"][f"{RELAX}.term_classifier"]
    # ``_relax/monotonicity.py`` does ``from .convexity import interval as iv``.
    assert f"{RELAX}.monotonicity" in graph["importers"][f"{RELAX}.convexity.interval"]
    # A module must never be recorded as importing itself.
    for module, srcs in graph["importers"].items():
        assert module not in srcs, f"{module} recorded as its own importer"


def test_issue_1231_modules_are_live(graph):
    """The three #1231 called dead are reachable from ``polyhedral_oa`` (#1347)."""
    for name in ("chebyshev_model", "taylor_model", "ellipsoidal_arith"):
        full = f"{RELAX}.{name}"
        assert _production_imported(full, graph), f"{name} should be production-imported"
        assert f"{RELAX}.polyhedral_oa" in graph["importers"][full]


def test_every_relax_module_is_imported_or_declared(graph):
    """The rule: production-imported, or declared in MODULE_STATUS."""
    undeclared = []
    checked = 0
    for module in _relax_modules(graph):
        if module in SELF_EXEMPT:
            continue
        checked += 1
        if _production_imported(module, graph):
            continue
        if module[len(RELAX) + 1 :] in MODULE_STATUS:
            continue
        undeclared.append(module)
    assert checked > 100, f"only {checked} modules checked"
    assert not undeclared, (
        "these _relax modules have no production importer and no MODULE_STATUS "
        f"entry -- declare them in discopt/_relax/module_status.py or retire "
        f"them (#1347): {undeclared}"
    )


def test_no_stale_declarations(graph):
    """A declared module that production now imports must lose its entry."""
    stale = [name for name in MODULE_STATUS if _production_imported(f"{RELAX}.{name}", graph)]
    assert not stale, (
        "these modules are declared in MODULE_STATUS but production imports "
        f"them; drop the entry (#1347): {stale}"
    )


def test_declarations_name_a_real_module(graph):
    """No entry may outlive its module."""
    missing = [name for name in MODULE_STATUS if f"{RELAX}.{name}" not in graph["modules"]]
    assert not missing, f"MODULE_STATUS entries with no module: {missing}"


@pytest.mark.parametrize("name", sorted(MODULE_STATUS))
def test_declaration_is_substantive(name):
    """Every entry carries a real reason; ``public`` carries an entry point."""
    entry: ModuleStatus = MODULE_STATUS[name]
    assert entry.status in ("public", "tooling", "incubating")
    assert len(entry.reason.strip()) >= 40, f"{name}: reason is not a real explanation"
    if entry.status == "public":
        assert entry.entry_point.startswith(RELAX + "."), (
            f"{name}: a public module must name its documented entry point"
        )
    else:
        assert not entry.entry_point, f"{name}: entry_point is for public modules only"


@pytest.mark.parametrize(
    "name", sorted(n for n, e in MODULE_STATUS.items() if e.status in ("public", "incubating"))
)
def test_public_and_incubating_modules_are_tested(name, graph):
    """A capability claimed to work is a capability under test.

    ``tooling`` is exempt -- design-time catalogues may have no automated
    consumer, and their entries say so.
    """
    importers = graph["test_importers"].get(f"{RELAX}.{name}", set())
    assert importers, (
        f"{name} is declared {MODULE_STATUS[name].status} but no test imports it; "
        "test it, re-declare it as tooling, or retire it (#1347)"
    )


def test_no_unreachable_module_cluster(graph):
    """No ``_relax`` module survives only by being imported by another dead one.

    ``_production_imported`` asks "does anything point at this?", which a cluster
    of mutually-importing dead modules answers yes to while being collectively
    unreachable. This asks the stronger question: is the module reachable from a
    real entry point? Roots are the package ``__init__`` files outside ``_relax``
    (the public surface) plus the declared modules, since a declared module's
    private helpers are legitimately reachable only through it.

    Reachability follows import edges plus one structural rule, in the one
    direction Python actually has: importing ``pkg.sub`` executes
    ``pkg/__init__`` first, so reaching a module reaches its *ancestor*
    packages -- and therefore whatever those ``__init__`` files import. It must
    not flow the other way. Expanding a package to its descendants makes the
    root ``discopt`` reach every module in the tree, which silently turns this
    into a no-op: the first draft did that and passed a synthetic two-module
    dead cycle. Dropping the ancestor rule is wrong too -- it reports
    ``_relax/presolve/``'s passes as dead when ``presolve/__init__`` really does
    import them on the way to ``presolve.orchestrator``.
    """
    roots = {m for m in graph["packages"] if not m.startswith(RELAX + ".")}
    roots |= {f"{RELAX}.{name}" for name in MODULE_STATUS}
    roots |= SELF_EXEMPT

    seen: set[str] = set()
    stack = list(roots)
    while stack:
        module = stack.pop()
        if module in seen or module not in graph["modules"]:
            continue
        seen.add(module)
        stack.extend(graph["imports"].get(module, ()))
        parent = module.rsplit(".", 1)[0] if "." in module else ""
        if parent in graph["packages"]:
            stack.append(parent)

    assert len(seen) > 200, f"traversal reached only {len(seen)} modules"
    unreachable = sorted(m for m in graph["modules"] if m.startswith(RELAX + ".") and m not in seen)
    assert not unreachable, (
        "these _relax modules are reachable only from other unreachable modules "
        f"-- a dead cluster; retire it or declare its entry point (#1347): {unreachable}"
    )


@pytest.mark.parametrize(
    "name", sorted(n for n, e in MODULE_STATUS.items() if e.status == "public")
)
def test_public_modules_are_documented(name):
    """#1347 requires a ``public`` module to say in the docs that it is one.

    Without this the ``public`` status is a private note: the whole point is that
    a reader can tell an optional user-facing feature from internal machinery
    without re-deriving it.
    """
    assert CATALOG.exists(), f"missing {CATALOG}"
    text = CATALOG.read_text(encoding="utf-8")
    entry_point = MODULE_STATUS[name].entry_point
    assert entry_point in text, (
        f"{name} is declared public but its entry point {entry_point!r} is not "
        f"named in {CATALOG.name}; document it or re-declare the module (#1347)"
    )
