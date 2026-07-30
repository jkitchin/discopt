#!/usr/bin/env python
"""Rewrite ``solve_model`` locals into attributes of an explicit state object.

This is the migration tool for consolidation-plan item 11.  It is *not* a
sed: a textual rename in a 7,600-line function with 63 early returns and 41
nested scopes is exactly the kind of edit that silently half-lands.  So the
rewrite is position-driven (every ``ast.Name`` node with the target id inside
``solve_model``, excluding nested scopes that bind the name themselves) and then
**proved** by comparing ASTs:

1. Parse the original file.
2. Apply the *same* substitution to the original AST with a transformer.
3. Parse the rewritten file.
4. Require ``ast.dump(transformed_original) == ast.dump(rewritten)`` for
   ``solve_model``, and require every *other* top-level definition to be
   AST-identical to its pre-rewrite self.

Step 4 is the whole point.  If the textual edit touched one character it should
not have — a string literal, an f-string, a keyword argument name, a sibling
function — the dumps differ and the tool refuses to write.  This is the same
proof Card 4b used for ``native_kernel.py`` ("7 of the 8 moved functions are
AST-identical"), applied to a rename instead of a move.

It also refuses when a target name collides with a module-level name in the same
file: with no collision, a *missed* site raises ``NameError`` at runtime, which is
loud; with a collision it would silently read the module global instead, which is
the CLAUDE.md §7 failure mode.

Usage::

    python -u discopt_benchmarks/scripts/thread_solve_model_state.py \
        --holder _timers --map rust_time=rust_time --map jax_time=jax_time
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "python" / "discopt" / "solver" / "__init__.py"
FUNCTION_NAME = "solve_model"


def _module_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                for x in ast.walk(t):
                    if isinstance(x, ast.Name):
                        names.add(x.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                names.add(a.asname or a.name.split(".")[0])
    return names


def _binds_locally(scope: ast.AST, name: str) -> bool:
    """Does this nested scope bind ``name`` itself (so it shadows the outer one)?"""
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = scope.args
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs]:
            if arg.arg == name:
                return True
        if a.vararg and a.vararg.arg == name:
            return True
        if a.kwarg and a.kwarg.arg == name:
            return True
    for child in ast.walk(scope):
        if isinstance(child, ast.Nonlocal) and name in child.names:
            return False  # writes through to the outer binding — still ours
        if isinstance(child, ast.Global) and name in child.names:
            return True
        # a Store inside a nested scope shadows unless declared nonlocal
        if (
            isinstance(child, ast.Name)
            and isinstance(child.ctx, ast.Store)
            and child.id == name
            and not any(isinstance(g, ast.Nonlocal) and name in g.names for g in ast.walk(scope))
        ):
            return True
        if isinstance(child, ast.ExceptHandler) and child.name == name:
            return True
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            for al in child.names:
                if (al.asname or al.name.split(".")[0]) == name:
                    return True
    return False


def _target_sites(fn: ast.AST, names: set[str]) -> list[ast.Name]:
    """Every ``Name`` node inside ``fn`` referring to ``fn``'s own binding."""
    shadowed: dict[str, list[ast.AST]] = {n: [] for n in names}
    for node in ast.walk(fn):
        if node is fn:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            for n in names:
                if _binds_locally(node, n):
                    shadowed[n].append(node)

    def _is_shadowed(node: ast.Name) -> bool:
        return any(
            scope.lineno <= node.lineno <= scope.end_lineno  # type: ignore[attr-defined]
            for scope in shadowed[node.id]
        )

    out: list[ast.Name] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id in names and not _is_shadowed(node):
            out.append(node)
    return out


class _Substituter(ast.NodeTransformer):
    """Reference implementation of the intended edit, applied to the AST."""

    def __init__(self, holder: str, mapping: dict[str, str], sites: set[tuple[int, int]]):
        self.holder = holder
        self.mapping = mapping
        self.sites = sites
        self.applied = 0

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.mapping and (node.lineno, node.col_offset) in self.sites:
            self.applied += 1
            return ast.copy_location(
                ast.Attribute(
                    value=ast.copy_location(ast.Name(id=self.holder, ctx=ast.Load()), node),
                    attr=self.mapping[node.id],
                    ctx=node.ctx,
                ),
                node,
            )
        return node


def _defs(tree: ast.Module) -> dict[str, ast.AST]:
    return {
        n.name: n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def rewrite(source: Path, holder: str, mapping: dict[str, str], apply: bool) -> int:
    text = source.read_text()
    tree = ast.parse(text)
    fns = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == FUNCTION_NAME
    ]
    if len(fns) != 1:
        raise SystemExit(f"expected exactly one top-level {FUNCTION_NAME}")
    fn = fns[0]

    collisions = _module_level_names(tree) & set(mapping)
    if collisions:
        raise SystemExit(
            f"REFUSING: {sorted(collisions)} also exist at module level; a missed "
            "rewrite site would silently read the global instead of raising NameError"
        )

    sites = _target_sites(fn, set(mapping))
    if not sites:
        raise SystemExit(
            f"REFUSING: no sites found for {sorted(mapping)} — the probe fired on nothing"
        )

    # ---- textual edit, right-to-left so earlier offsets stay valid ---------- #
    lines = text.splitlines(keepends=True)
    by_line: dict[int, list[ast.Name]] = {}
    for s in sites:
        by_line.setdefault(s.lineno, []).append(s)
    for lineno, nodes in by_line.items():
        line = lines[lineno - 1]
        for node in sorted(nodes, key=lambda n: -n.col_offset):
            start, end = node.col_offset, node.end_col_offset
            if node.end_lineno != node.lineno:
                raise SystemExit(f"multi-line Name at {lineno}; refusing")
            if line[start:end] != node.id:
                raise SystemExit(
                    f"REFUSING: line {lineno} cols {start}:{end} is {line[start:end]!r}, "
                    f"expected {node.id!r} — AST offsets do not match the text"
                )
            line = line[:start] + f"{holder}.{mapping[node.id]}" + line[end:]
        lines[lineno - 1] = line
    new_text = "".join(lines)

    # ---- proof: the textual edit == the intended AST substitution ----------- #
    site_positions = {(s.lineno, s.col_offset) for s in sites}
    ref_tree = ast.parse(text)
    ref_fn = [
        n
        for n in ref_tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == FUNCTION_NAME
    ][0]
    sub = _Substituter(holder, mapping, site_positions)
    sub.visit(ref_fn)
    ast.fix_missing_locations(ref_fn)

    new_tree = ast.parse(new_text)
    new_fn = [
        n
        for n in new_tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == FUNCTION_NAME
    ][0]

    want = ast.dump(ref_fn, include_attributes=False)
    got = ast.dump(new_fn, include_attributes=False)
    if want != got:
        raise SystemExit(
            "REFUSING: the textual rewrite is not the intended AST substitution "
            "(dumps differ). Nothing written."
        )

    old_defs, new_defs = _defs(tree), _defs(new_tree)
    if set(old_defs) != set(new_defs):
        raise SystemExit("REFUSING: top-level definition set changed")
    unchanged = 0
    for name in old_defs:
        if name == FUNCTION_NAME:
            continue
        if ast.dump(old_defs[name]) != ast.dump(new_defs[name]):
            raise SystemExit(f"REFUSING: sibling definition {name} changed")
        unchanged += 1

    print(f"holder            : {holder}")
    print(f"names             : {len(mapping)} -> {sorted(mapping)}")
    print(f"sites rewritten   : {len(sites)}")
    print(f"substitutions     : {sub.applied} (AST reference implementation)")
    print(f"sibling defs proved AST-identical: {unchanged}")
    if sub.applied != len(sites):
        raise SystemExit("REFUSING: reference substitution count != textual site count")

    if apply:
        source.write_text(new_text)
        print(f"WROTE {source}")
    else:
        print("dry run — nothing written (pass --apply)")
    print(f"executed AST comparisons: {unchanged + 1}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--holder", required=True, help="state object local, e.g. _timers")
    ap.add_argument(
        "--map",
        action="append",
        required=True,
        metavar="LOCAL=FIELD",
        help="repeatable; the local to migrate and the field it becomes",
    )
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    mapping: dict[str, str] = {}
    for spec in args.map:
        local, _, fieldname = spec.partition("=")
        if not local or not fieldname:
            raise SystemExit(f"bad --map {spec!r}; want LOCAL=FIELD")
        mapping[local] = fieldname
    return rewrite(args.source, args.holder, mapping, args.apply)


if __name__ == "__main__":
    sys.exit(main())
