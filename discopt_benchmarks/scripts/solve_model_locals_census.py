#!/usr/bin/env python
"""AST census of ``solve_model``'s local variables (consolidation plan, item 11).

Card 4b's entry experiment established that four of the five modules it named are
*inline statement blocks* of ``solve_model`` sharing "a closure of 200+ locals",
and that carving them requires an explicit state object first.  That number was a
hand-wave; this script replaces it with a classification, because the object's
shape has to follow the census rather than intuition (plan §0.3, CLAUDE.md §4).

Every local bound in ``solve_model``'s own scope is classified as:

``CONFIG``
    Bound exactly once, in the pre-loop region, never augmented, never rebound
    anywhere (including nested scopes).  Read-only configuration derived once
    from the arguments.  These are the ones a frozen state object can carry.

``STATE``
    Rebound more than once, or bound inside the spatial loop, or augmented, or
    rebound by a nested closure through ``nonlocal``.  Genuine mutable search
    state threaded through the loop.  These need a mutable state object.

``SINGLE_USE``
    Bound once and read exactly once.  A temporary; carving does not need it to
    cross a boundary as long as its producer and consumer stay together.

``DEAD``
    Bound and never read.  Nothing has to carry it at all.

The two coupling columns are what actually decide whether a block can be carved:

``nested_reads``
    The local is a free variable of a nested ``def``/``lambda`` inside
    ``solve_model`` — the closure capture Card 4b named.

``regions``
    Which of the three regions (pre-loop / spatial loop / post-loop) bind or read
    the name.  A local bound in one region and read in another is exactly what a
    carve has to pass across a function boundary.

Per CLAUDE.md §6 the script prints an executed-classification count and exits
non-zero when it is zero, so a census that traversed nothing cannot read as a
pass.

Usage::

    python -u discopt_benchmarks/scripts/solve_model_locals_census.py \
        --json reports/solve_model_locals_census.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "python" / "discopt" / "solver" / "__init__.py"
FUNCTION_NAME = "solve_model"

#: Region boundaries, named after Card 4b's five modules and anchored on the
#: source's own phase banners rather than line numbers (which move).  Each entry
#: is (region name, banner substring that *starts* it); the first region starts
#: at the function's first statement and ``loop``/``results`` are derived from
#: the single largest inline ``while``.
REGION_ANCHORS: list[tuple[str, str]] = [
    ("setup", ""),  # from the top of the function
    ("reformulate", "# --- GDP reformulation:"),
    ("root", "# --- Build Rust model representation for FBBT ---"),
]


# --------------------------------------------------------------------------- #
# scope walking
# --------------------------------------------------------------------------- #


@dataclass
class Binding:
    """Every binding and reference site for one name in ``solve_model``."""

    name: str
    store_lines: list[int] = field(default_factory=list)
    load_lines: list[int] = field(default_factory=list)
    aug_lines: list[int] = field(default_factory=list)
    del_lines: list[int] = field(default_factory=list)
    nonlocal_lines: list[int] = field(default_factory=list)
    nested_read_lines: list[int] = field(default_factory=list)
    is_param: bool = False
    #: binding forms seen, e.g. {"assign", "for", "with", "except", "walrus"}
    forms: set[str] = field(default_factory=set)


class _ScopeCollector(ast.NodeVisitor):
    """Collect name bindings/loads for one function scope, not descending into
    nested function or class scopes (those are visited separately)."""

    def __init__(self) -> None:
        self.bindings: dict[str, Binding] = {}
        self.nested: list[ast.AST] = []

    def _b(self, name: str) -> Binding:
        return self.bindings.setdefault(name, Binding(name=name))

    # -- binding forms ------------------------------------------------------ #

    def _bind_target(self, target: ast.AST, line: int, form: str) -> None:
        for node in ast.walk(target):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                b = self._b(node.id)
                b.forms.add(form)
                if isinstance(node.ctx, ast.Del):
                    b.del_lines.append(line)
                else:
                    b.store_lines.append(line)
            elif isinstance(node, ast.Name):
                # e.g. `a[i] = x` reads `a` and `i`
                self._b(node.id).load_lines.append(line)

    def visit_Assign(self, node: ast.Assign) -> None:
        for t in node.targets:
            self._bind_target(t, node.lineno, "assign")
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._bind_target(node.target, node.lineno, "assign")
            self.visit(node.value)
        else:
            # bare annotation binds nothing at runtime
            pass

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            b = self._b(node.target.id)
            b.forms.add("augassign")
            b.store_lines.append(node.lineno)
            b.aug_lines.append(node.lineno)
            b.load_lines.append(node.lineno)
        else:
            self._bind_target(node.target, node.lineno, "augassign")
        self.visit(node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._bind_target(node.target, node.lineno, "walrus")
        self.visit(node.value)

    def _visit_for(self, node: ast.For | ast.AsyncFor) -> None:
        self._bind_target(node.target, node.lineno, "for")
        self.visit(node.iter)
        for s in node.body:
            self.visit(s)
        for s in node.orelse:
            self.visit(s)

    visit_For = _visit_for  # type: ignore[assignment]  # noqa: N815
    visit_AsyncFor = _visit_for  # type: ignore[assignment]  # noqa: N815

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._bind_target(item.optional_vars, node.lineno, "with")
        for s in node.body:
            self.visit(s)

    visit_With = _visit_with  # type: ignore[assignment]  # noqa: N815
    visit_AsyncWith = _visit_with  # type: ignore[assignment]  # noqa: N815

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            b = self._b(node.name)
            b.forms.add("except")
            b.store_lines.append(node.lineno)
        for s in node.body:
            self.visit(s)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            b = self._b(name)
            b.forms.add("import")
            b.store_lines.append(node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            b = self._b(name)
            b.forms.add("import")
            b.store_lines.append(node.lineno)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self._b(name).forms.add("global")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            b = self._b(name)
            b.forms.add("nonlocal-decl")
            b.nonlocal_lines.append(node.lineno)

    # -- references --------------------------------------------------------- #

    def visit_Name(self, node: ast.Name) -> None:
        b = self._b(node.id)
        if isinstance(node.ctx, ast.Store):
            b.store_lines.append(node.lineno)
        elif isinstance(node.ctx, ast.Del):
            b.del_lines.append(node.lineno)
        else:
            b.load_lines.append(node.lineno)

    # -- nested scopes ------------------------------------------------------ #

    def _defer(self, node: ast.AST) -> None:
        self.nested.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        b = self._b(node.name)
        b.forms.add("def")
        b.store_lines.append(node.lineno)
        for dec in node.decorator_list:
            self.visit(dec)
        self._defer(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]  # noqa: N815

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._defer(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        b = self._b(node.name)
        b.forms.add("class")
        b.store_lines.append(node.lineno)
        for dec in node.decorator_list:
            self.visit(dec)
        self._defer(node)

    # comprehensions are their own scope in py3
    def _visit_comp(self, node: ast.AST) -> None:
        self._defer(node)

    visit_ListComp = _visit_comp  # type: ignore[assignment]  # noqa: N815
    visit_SetComp = _visit_comp  # type: ignore[assignment]  # noqa: N815
    visit_DictComp = _visit_comp  # type: ignore[assignment]  # noqa: N815
    visit_GeneratorExp = _visit_comp  # type: ignore[assignment]  # noqa: N815


def _bound_names_of_scope(node: ast.AST) -> set[str]:
    """Names bound by a nested scope itself (so they are not free variables)."""
    bound: set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = node.args
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs]:
            bound.add(arg.arg)
        if a.vararg:
            bound.add(a.vararg.arg)
        if a.kwarg:
            bound.add(a.kwarg.arg)
    body: list[ast.AST]
    if isinstance(node, ast.Lambda):
        body = [node.body]
    elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        body = [node.elt, *node.generators]
    elif isinstance(node, ast.DictComp):
        body = [node.key, node.value, *node.generators]
    else:
        body = list(getattr(node, "body", []))
    sub = _ScopeCollector()
    for s in body:
        sub.visit(s)
    for name, b in sub.bindings.items():
        if b.store_lines and "nonlocal-decl" not in b.forms and "global" not in b.forms:
            bound.add(name)
    return bound


def _free_reads(node: ast.AST) -> dict[str, list[int]]:
    """Names read inside a nested scope (transitively) that it does not bind."""
    bound = _bound_names_of_scope(node)
    reads: dict[str, list[int]] = defaultdict(list)
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Name)
            and isinstance(child.ctx, ast.Load)
            and child.id not in bound
        ):
            reads[child.id].append(child.lineno)
    return reads


def _nonlocal_writes(node: ast.AST) -> dict[str, list[int]]:
    writes: dict[str, list[int]] = defaultdict(list)
    for child in ast.walk(node):
        if isinstance(child, ast.Nonlocal):
            for name in child.names:
                writes[name].append(child.lineno)
    return writes


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #

CONFIG = "CONFIG"
STATE = "STATE"
SINGLE_USE = "SINGLE_USE"
DEAD = "DEAD"
CALLABLE_LOCAL = "CALLABLE"


@dataclass
class Region:
    name: str
    start: int
    end: int

    def holds(self, line: int) -> bool:
        return self.start <= line <= self.end


def classify(b: Binding, regions: list[Region], loop: Region) -> str:
    if "def" in b.forms or "class" in b.forms or "import" in b.forms:
        return CALLABLE_LOCAL
    n_store = len(b.store_lines)
    n_load = len(b.load_lines) + len(b.nested_read_lines)
    if n_load == 0 and not b.is_param:
        return DEAD
    rebound_by_closure = bool(b.nonlocal_lines)
    augmented = bool(b.aug_lines)
    bound_in_loop = any(loop.holds(ln) for ln in b.store_lines)
    deleted = bool(b.del_lines)
    if n_store > 1 or augmented or bound_in_loop or rebound_by_closure or deleted:
        return STATE
    if b.is_param:
        return CONFIG
    if n_load == 1:
        return SINGLE_USE
    return CONFIG


def region_set(lines: list[int], regions: list[Region]) -> list[str]:
    out: list[str] = []
    for r in regions:
        if any(r.holds(ln) for ln in lines):
            out.append(r.name)
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def run(source: Path, function_name: str) -> dict[str, Any]:
    src = source.read_text()
    tree = ast.parse(src)
    fns = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function_name
    ]
    if len(fns) != 1:
        raise SystemExit(f"expected exactly one top-level {function_name}, found {len(fns)}")
    fn = fns[0]

    collector = _ScopeCollector()
    for stmt in fn.body:
        collector.visit(stmt)

    # parameters
    a = fn.args
    params = [arg.arg for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs]]
    if a.vararg:
        params.append(a.vararg.arg)
    if a.kwarg:
        params.append(a.kwarg.arg)
    for p in params:
        b = collector.bindings.setdefault(p, Binding(name=p))
        b.is_param = True
        b.forms.add("param")

    # the single largest inline ``while`` is the spatial loop
    loops = [s for s in fn.body if isinstance(s, ast.While)]
    if not loops:
        raise SystemExit("no top-level while loop found in solve_model")
    spatial = max(loops, key=lambda s: s.end_lineno - s.lineno)
    loop = Region("loop", spatial.lineno, spatial.end_lineno)

    lines = src.splitlines()
    starts: list[tuple[str, int]] = []
    for name, anchor in REGION_ANCHORS:
        if not anchor:
            starts.append((name, fn.lineno))
            continue
        hits = [
            i for i in range(fn.lineno, spatial.lineno) if lines[i - 1].strip().startswith(anchor)
        ]
        if len(hits) != 1:
            raise SystemExit(
                f"region anchor {anchor!r} matched {len(hits)} lines in {function_name}; "
                "the banner moved or was duplicated — fix the anchor, do not guess"
            )
        starts.append((name, hits[0]))
    regions = []
    for idx, (name, start) in enumerate(starts):
        end = starts[idx + 1][1] - 1 if idx + 1 < len(starts) else spatial.lineno - 1
        regions.append(Region(name, start, end))
    regions.append(loop)
    regions.append(Region("results", spatial.end_lineno + 1, fn.end_lineno))

    # nested-scope coupling
    nested_scopes = 0
    queue = list(collector.nested)
    while queue:
        node = queue.pop()
        nested_scopes += 1
        for name, lines in _free_reads(node).items():
            if name in collector.bindings:
                collector.bindings[name].nested_read_lines.extend(lines)
        for name, lines in _nonlocal_writes(node).items():
            if name in collector.bindings:
                collector.bindings[name].nonlocal_lines.extend(lines)

    rows: list[dict[str, Any]] = []
    classified = 0
    for name, b in sorted(collector.bindings.items()):
        # names that are only ever read are module-level references, not locals
        if not b.store_lines and not b.is_param:
            continue
        kind = classify(b, regions, loop)
        classified += 1
        store_regions = region_set(b.store_lines, regions)
        load_regions = region_set(b.load_lines + b.nested_read_lines, regions)
        rows.append(
            {
                "name": name,
                "kind": kind,
                "is_param": b.is_param,
                "stores": len(b.store_lines),
                "loads": len(b.load_lines),
                "aug": len(b.aug_lines),
                "dels": len(b.del_lines),
                "nested_reads": len(b.nested_read_lines),
                "nonlocal_writes": len(b.nonlocal_lines),
                "store_regions": store_regions,
                "load_regions": load_regions,
                "crosses_regions": sorted(set(store_regions) | set(load_regions)) != store_regions
                or len(set(store_regions) | set(load_regions)) > 1,
                "forms": sorted(b.forms),
                "first_store": min(b.store_lines) if b.store_lines else None,
                "last_store": max(b.store_lines) if b.store_lines else None,
            }
        )

    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["kind"]] += 1

    return {
        "source": str(source.relative_to(REPO_ROOT)),
        "function": function_name,
        "function_lines": [fn.lineno, fn.end_lineno],
        "function_loc": fn.end_lineno - fn.lineno + 1,
        "top_level_statements": len(fn.body),
        "n_params": len(params),
        "spatial_loop_lines": [loop.start, loop.end],
        "regions": [{"name": r.name, "start": r.start, "end": r.end} for r in regions],
        "nested_scopes": nested_scopes,
        "classified": classified,
        "counts": dict(counts),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--function", default=FUNCTION_NAME)
    ap.add_argument("--json", type=Path, default=None, help="write the full census here")
    ap.add_argument("--top", type=int, default=30, help="how many STATE rows to print")
    args = ap.parse_args(argv)

    census = run(args.source, args.function)

    print(f"source              : {census['source']}")
    print(
        f"function            : {census['function']} "
        f"lines {census['function_lines'][0]}-{census['function_lines'][1]} "
        f"({census['function_loc']} LOC)"
    )
    print(f"top-level statements: {census['top_level_statements']}")
    print(f"parameters          : {census['n_params']}")
    print(
        f"spatial loop        : lines {census['spatial_loop_lines'][0]}-"
        f"{census['spatial_loop_lines'][1]}"
    )
    print(f"nested scopes       : {census['nested_scopes']}")
    print()
    print("regions (Card 4b's five module names, anchored on the source's banners):")
    for r in census["regions"]:
        print(
            f"  {r['name']:<12} {r['start']:>6}-{r['end']:<6} ({r['end'] - r['start'] + 1:>5} LOC)"
        )
    print()
    print("classification:")
    for kind in (CONFIG, STATE, SINGLE_USE, DEAD, CALLABLE_LOCAL):
        print(f"  {kind:<11}: {census['counts'].get(kind, 0)}")
    print()

    rows = census["rows"]
    state_rows = [r for r in rows if r["kind"] == STATE]
    state_rows.sort(key=lambda r: (-r["stores"], -r["loads"], r["name"]))
    print(f"top {min(args.top, len(state_rows))} STATE locals by rebind count:")
    print(f"  {'name':<42} {'st':>4} {'ld':>5} {'aug':>4} {'nest':>5}  regions")
    for r in state_rows[: args.top]:
        regs = "+".join(sorted(set(r["store_regions"]) | set(r["load_regions"])))
        print(
            f"  {r['name']:<42} {r['stores']:>4} {r['loads']:>5} "
            f"{r['aug']:>4} {r['nested_reads']:>5}  {regs}"
        )
    print()

    region_names = [r["name"] for r in census["regions"]]
    crossers = [
        r
        for r in rows
        if r["kind"] in (CONFIG, STATE)
        and len(set(r["store_regions"]) | set(r["load_regions"])) > 1
    ]
    print("bind-region -> read-region crossings (a local bound in the row's region and")
    print("read in the column's; the diagonal is intra-region and needs no boundary):")
    header = "  " + " " * 13 + "".join(f"{c:>13}" for c in region_names)
    print(header)
    for a in region_names:
        cells = []
        for b in region_names:
            n = sum(
                1
                for r in rows
                if r["kind"] in (CONFIG, STATE)
                and a in r["store_regions"]
                and b in r["load_regions"]
            )
            cells.append(f"{n:>13}")
        print(f"  {a:<13}" + "".join(cells))
    print()
    print(f"locals whose bind/read spans more than one region: {len(crossers)}")
    closure = [r for r in rows if r["nested_reads"] > 0 and r["kind"] in (CONFIG, STATE)]
    print(f"locals captured by a nested closure               : {len(closure)}")
    nl = [r for r in rows if r["nonlocal_writes"] > 0]
    print(f"locals rebound by a closure via `nonlocal`         : {len(nl)}")
    dead = [r["name"] for r in rows if r["kind"] == DEAD]
    print(f"never-read bindings                               : {len(dead)}")
    if dead:
        print("   " + ", ".join(sorted(dead)))
    print()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.json}")

    # CLAUDE.md §6: prove the probe fired.
    print(f"executed classifications: {census['classified']}")
    if census["classified"] == 0:
        print("FAIL: the census classified nothing")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
