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

.. rubric:: Annotated assignments

``x: T = v`` and ``x.f = v`` are *different AST shapes*, not the same shape with a
different target: ``ast.AnnAssign`` carries ``simple=1`` only when its target is a
bare ``Name``, and re-annotating a non-``self`` attribute (``_mc.f: T = v``) is an
error under mypy even though CPython accepts it.  So a naive rename of an
``AnnAssign`` target either fails this tool's own AST proof (``simple`` differs) or
passes it and fails type-checking.  Both were observed: the first refusal was
``_mc_con_relax_fns: list[Callable] | None = None``.

The tool therefore **lowers** an annotated assignment whose target is being
migrated into a plain ``Assign``, dropping the annotation — which is the correct
place for it to go, because the migrated field carries its own annotation on the
dataclass.  Every dropped annotation is printed so it can be transferred, and the
lowering is proved by the same AST comparison as everything else (the reference
transformer performs the identical lowering).  A bare declaration with no value
(``x: T``) is refused rather than lowered: dropping it would delete a statement.

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
        self.lowered = 0

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        """Lower ``x: T = v`` to ``holder.f = v`` when ``x`` is being migrated.

        Renaming the target in place is not an option: ``AnnAssign.simple`` is 1
        only for a bare ``Name`` target, so the shape itself changes, and mypy
        rejects an annotation on a non-``self`` attribute. The annotation's
        information is not lost — it lives on the dataclass field instead.
        """
        tgt = node.target
        if (
            isinstance(tgt, ast.Name)
            and tgt.id in self.mapping
            and (tgt.lineno, tgt.col_offset) in self.sites
        ):
            if node.value is None:
                raise SystemExit(
                    f"REFUSING: {tgt.id} at line {tgt.lineno} is a bare annotation with no "
                    "value; lowering it would delete a statement. Give it a value by hand first."
                )
            self.applied += 1
            self.lowered += 1
            return ast.copy_location(
                ast.Assign(
                    targets=[
                        ast.copy_location(
                            ast.Attribute(
                                value=ast.copy_location(
                                    ast.Name(id=self.holder, ctx=ast.Load()), tgt
                                ),
                                attr=self.mapping[tgt.id],
                                ctx=ast.Store(),
                            ),
                            tgt,
                        )
                    ],
                    value=self.visit(node.value),
                    type_comment=None,
                ),
                node,
            )
        return self.generic_visit(node)

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

    # The holder name must be free: if ``solve_model`` already uses it for
    # anything else, every rewritten site would silently read that other object
    # instead of the state holder — a wrong answer with no NameError to catch it.
    holder_uses = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == holder]
    if holder_uses:
        raise SystemExit(
            f"REFUSING: {holder!r} is already referenced inside {FUNCTION_NAME} at lines "
            f"{sorted(set(holder_uses))[:10]}; pick a free name"
        )

    sites = _target_sites(fn, set(mapping))
    if not sites:
        raise SystemExit(
            f"REFUSING: no sites found for {sorted(mapping)} — the probe fired on nothing"
        )

    site_positions = {(s.lineno, s.col_offset) for s in sites}

    # ---- annotated assignments are LOWERED, not renamed (see module docstring) #
    ann_lowerings: dict[tuple[int, int], ast.AnnAssign] = {}
    dropped_annotations: list[str] = []
    for node in ast.walk(fn):
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
            continue
        tgt = node.target
        pos = (tgt.lineno, tgt.col_offset)
        if tgt.id not in mapping or pos not in site_positions:
            continue
        if node.value is None:
            raise SystemExit(
                f"REFUSING: {tgt.id} at line {tgt.lineno} is a bare annotation with no value; "
                "lowering it would delete a statement. Give it a value by hand first."
            )
        if not (tgt.lineno == node.value.lineno == node.lineno):
            raise SystemExit(
                f"REFUSING: the annotated assignment to {tgt.id} at line {tgt.lineno} spans "
                "lines; lower it by hand."
            )
        ann_lowerings[pos] = node
        dropped_annotations.append(
            f"  line {tgt.lineno}: {tgt.id}: {ast.unparse(node.annotation)}"
            f"  ->  {holder}.{mapping[tgt.id]}  (annotation now lives on the dataclass field)"
        )

    # ---- textual edit, right-to-left so earlier offsets stay valid ---------- #
    lines = text.splitlines(keepends=True)
    # per source line: (start_col, end_col, original name, replacement)
    edits: dict[int, list[tuple[int, int, str, str]]] = {}
    for node in sites:
        if node.end_lineno != node.lineno:
            raise SystemExit(f"multi-line Name at {node.lineno}; refusing")
        ann = ann_lowerings.get((node.lineno, node.col_offset))
        if ann is not None and ann.target is node:
            # `x: T = v`  ->  `holder.f = v`: replace everything up to the value.
            assert ann.value is not None
            span_end = ann.value.col_offset
            repl = f"{holder}.{mapping[node.id]} = "
        else:
            span_end = node.end_col_offset or node.col_offset
            repl = f"{holder}.{mapping[node.id]}"
        edits.setdefault(node.lineno, []).append((node.col_offset, span_end, node.id, repl))

    for lineno, spans in edits.items():
        line = lines[lineno - 1]
        ordered = sorted(spans, key=lambda s: -s[0])
        for (start, _, _, _), (_, prev_end, _, _) in zip(ordered, ordered[1:], strict=False):
            if prev_end > start:
                raise SystemExit(
                    f"REFUSING: overlapping rewrite spans on line {lineno} — one target name "
                    "sits inside another target's span (e.g. inside an annotation)"
                )
        for start, span_end, name, repl in ordered:
            if line[start : start + len(name)] != name:
                raise SystemExit(
                    f"REFUSING: line {lineno} col {start} is "
                    f"{line[start : start + len(name)]!r}, expected {name!r} — "
                    "AST offsets do not match the text"
                )
            line = line[:start] + repl + line[span_end:]
        lines[lineno - 1] = line
    new_text = "".join(lines)

    # ---- proof: the textual edit == the intended AST substitution ----------- #
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
    print(f"AnnAssign lowered : {sub.lowered}")
    if dropped_annotations:
        print("annotations DROPPED (transfer them to the dataclass field):")
        for note in dropped_annotations:
            print(note)
    print(f"sibling defs proved AST-identical: {unchanged}")
    if sub.applied != len(sites):
        raise SystemExit("REFUSING: reference substitution count != textual site count")
    if sub.lowered != len(ann_lowerings):
        raise SystemExit(
            f"REFUSING: reference lowered {sub.lowered} annotated assignments but the textual "
            f"pass found {len(ann_lowerings)}"
        )

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
