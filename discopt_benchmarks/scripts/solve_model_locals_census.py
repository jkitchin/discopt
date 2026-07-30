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
    anywhere (including nested scopes) — **and** with no syntactic mutation of its
    value and no method call on it at all.  Only these may enter a frozen holder.

``MUTATED``
    The *name* is never rebound but the *value* is provably mutated in place:
    a subscript or attribute store/delete on it, or a call to a method from the
    curated unambiguous-mutator set.  Never admissible to a frozen holder.

``NEEDS_AUDIT``
    The name is never rebound and shows no syntactic mutation, but a method is
    called on it whose mutating-or-not status the AST cannot decide
    (``tree.export_batch``, ``evaluator.evaluate_objective``, …).  The census
    **refuses to call these CONFIG**: asserting immutability it cannot see is the
    exact failure the 2026-07-30 design review caught.  Resolving them requires
    reading the callee, which this script deliberately does not do — that is
    ``solve_model_config_mutability_audit.py``'s job.

``STATE``
    Rebound more than once, or bound inside the spatial loop, or augmented, or
    rebound by a nested closure through ``nonlocal``.  Genuine mutable search
    state threaded through the loop.  These need a mutable state object.

``SINGLE_USE``
    Bound once and read exactly once.  A temporary; carving does not need it to
    cross a boundary as long as its producer and consumer stay together.

``EXCEPT_TARGET``
    Bound only as an ``except ... as name`` target.  Python deletes such a name at
    handler exit (PEP 3110), so it **cannot** cross a region boundary no matter
    where it is read; counting it as a crosser inflates the carve cost.

``DEAD``
    Bound and never read.  Nothing has to carry it at all.

.. rubric:: Three classifier defects fixed 2026-07-30

The first revision of this script (commit ``ab8235dc``) had three defects that
propagated a wrong premise into the plan document; all three are fixed here and
each is pinned by a test in ``discopt_benchmarks/tests/test_locals_census.py``.
Those tests are written against **synthetic sources with known answers** rather
than against ``solve_model``, because a test that only asserted properties of the
real 7,600-line function would pass just as happily if a visitor stopped firing.

1. **``CONFIG`` meant "the name is never rebound", not "the value is immutable".**
   ``x[k] = v`` walked into ``_bind_target`` and was recorded as a *load* of ``x``;
   method calls were recorded as nothing at all.  So the live B&B ``tree``, the
   per-node deadline dict ``opts`` and the ``**kwargs`` dict all read as read-only
   configuration.  Now tracked as ``mutation_sites`` with per-site evidence and a
   three-tier verdict (``definite`` / ``known_mutator`` / ``unresolved_call``).
2. **``except`` targets were counted as ordinary bindings.**  They are deleted at
   handler exit and cannot cross a region; they now bind ``except_lines`` rather
   than ``store_lines`` and get their own class.
3. **The JSON's ``crosses_regions`` field and the printed crosser count used two
   different predicates** and disagreed by 11 names on the same data (84 vs 73).
   The field's expression compared an alphabetically ``sorted()`` set-union against
   a region-ordered list, so it also flagged never-stored *parameters* as crossing.
   Both now call the single :func:`crosses` helper; parameters are reported in
   their own column instead of being smuggled into the crossing count.

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

#: Method names that mutate their receiver under *every* stdlib/numpy type that
#: defines them.  Deliberately conservative: a name here is proof of mutation, so
#: an ambiguous entry (``pop`` on a dict mutates; ``pop`` exists nowhere read-only)
#: would turn the instrument into a guess.  Anything not listed is *not* assumed
#: read-only either — it becomes ``NEEDS_AUDIT`` (see the module docstring).
KNOWN_MUTATORS: frozenset[str] = frozenset(
    {
        # list / deque
        "append",
        "extend",
        "insert",
        "remove",
        "reverse",
        "sort",
        "appendleft",
        "extendleft",
        "popleft",
        "rotate",
        # dict / set / Counter
        "update",
        "setdefault",
        "popitem",
        "add",
        "discard",
        "difference_update",
        "intersection_update",
        "symmetric_difference_update",
        "subtract",
        # shared by list/dict/set
        "pop",
        "clear",
        # numpy in-place
        "fill",
        "setflags",
        "resize",
        "itemset",
        "put",
        "partition",
        "byteswap",
        # scipy.sparse in-place
        "eliminate_zeros",
        "sum_duplicates",
        "prune",
        "setdiag",
    }
)


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
    #: ``except ... as <name>`` bindings.  Kept *out* of ``store_lines``: Python
    #: deletes the name at handler exit, so such a binding cannot cross a region.
    except_lines: list[int] = field(default_factory=list)
    #: Per-site mutation evidence.  Each entry is
    #: ``{"line": int, "tier": str, "kind": str, "detail": str}`` with tier in
    #: ``{"definite", "known_mutator", "unresolved_call"}``.
    mutation_sites: list[dict[str, Any]] = field(default_factory=list)
    is_param: bool = False
    #: binding forms seen, e.g. {"assign", "for", "with", "except", "walrus"}
    forms: set[str] = field(default_factory=set)

    def proven_mutation_lines(self) -> list[int]:
        """Lines carrying tier-1/2 evidence — a syntactically proven mutation."""
        return [s["line"] for s in self.mutation_sites if s["tier"] != "unresolved_call"]

    def unresolved_call_lines(self) -> list[int]:
        """Lines carrying a method call the AST cannot adjudicate."""
        return [s["line"] for s in self.mutation_sites if s["tier"] == "unresolved_call"]


class _ScopeCollector(ast.NodeVisitor):
    """Collect name bindings/loads for one function scope, not descending into
    nested function or class scopes (those are visited separately)."""

    def __init__(self) -> None:
        self.bindings: dict[str, Binding] = {}
        self.nested: list[ast.AST] = []

    def _b(self, name: str) -> Binding:
        return self.bindings.setdefault(name, Binding(name=name))

    # -- mutation evidence -------------------------------------------------- #

    def _record_mutation(self, name: str, line: int, tier: str, kind: str, detail: str) -> None:
        self._b(name).mutation_sites.append(
            {"line": line, "tier": tier, "kind": kind, "detail": detail}
        )

    @staticmethod
    def _root_name(node: ast.AST) -> str | None:
        """Innermost ``Name`` under a chain of ``Subscript``/``Attribute`` nodes.

        ``a["k"].b[0] = v`` mutates the object bound to ``a``, so the chain has to
        be unwound rather than only its outermost link inspected.
        """
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    def _note_target_mutation(self, target: ast.AST, line: int, verb: str) -> None:
        """Record ``x[k] = v`` / ``x.a = v`` / ``del x[k]`` as a mutation of ``x``.

        Classifier defect 1 (fixed 2026-07-30): these previously reached
        ``_bind_target``'s ``elif isinstance(node, ast.Name)`` arm and were filed as
        *loads*, which is how the live B&B ``tree`` and the per-node deadline dict
        ``opts`` came to be labelled read-only ``CONFIG``.
        """
        if isinstance(target, ast.Name):
            return  # a plain rebinding, not a mutation of the pointed-to value
        name = self._root_name(target)
        if name is None:
            return
        kind = "subscript_store" if isinstance(target, ast.Subscript) else "attribute_store"
        self._record_mutation(name, line, "definite", kind, f"{verb} {ast.unparse(target)}")

    def visit_Call(self, node: ast.Call) -> None:
        """Record ``x.m(...)`` against ``x``: tier 2 when ``m`` is a known mutator,
        tier 3 (``unresolved_call``) otherwise.  Tier 3 is why ``NEEDS_AUDIT``
        exists — the AST cannot tell ``tree.stats()`` from ``tree.initialize()``.
        """
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            recv = node.func.value.id
            attr = node.func.attr
            if attr in KNOWN_MUTATORS:
                self._record_mutation(recv, node.lineno, "known_mutator", "method_call", f".{attr}")
            else:
                self._record_mutation(
                    recv, node.lineno, "unresolved_call", "method_call", f".{attr}"
                )
        self.generic_visit(node)

    # -- binding forms ------------------------------------------------------ #

    def _bind_target(self, target: ast.AST, line: int, form: str) -> None:
        self._note_target_mutation(target, line, "store into")
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
            # `x[k] += v` mutates x; `_bind_target` files the tier-1 evidence.
            self._note_target_mutation(node.target, node.lineno, "augment")
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    self._b(sub.id).load_lines.append(node.lineno)
        self.visit(node.value)

    def visit_Delete(self, node: ast.Delete) -> None:
        """``del x`` unbinds the name; ``del x[k]`` / ``del x.a`` mutates ``x``."""
        for t in node.targets:
            if isinstance(t, ast.Name):
                b = self._b(t.id)
                b.forms.add("del")
                b.del_lines.append(node.lineno)
            else:
                self._note_target_mutation(t, node.lineno, "delete from")
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Name):
                        self._b(sub.id).load_lines.append(node.lineno)

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
            # Classifier defect 2 (fixed 2026-07-30): an `except ... as e` target is
            # implicitly `del`'d at handler exit (PEP 3110), so it can never be live
            # across a region boundary.  Recording it in `store_lines` made 4 such
            # names read as cross-region STATE and inflated the carve cost.
            b = self._b(node.name)
            b.forms.add("except")
            b.except_lines.append(node.lineno)
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
        # An `except ... as e` target *is* a local binding of the nested scope even
        # though it is deleted at handler exit, so it is not a free variable there.
        if (
            (b.store_lines or b.except_lines)
            and "nonlocal-decl" not in b.forms
            and "global" not in b.forms
        ):
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
MUTATED = "MUTATED"
NEEDS_AUDIT = "NEEDS_AUDIT"
STATE = "STATE"
SINGLE_USE = "SINGLE_USE"
EXCEPT_TARGET = "EXCEPT_TARGET"
DEAD = "DEAD"
CALLABLE_LOCAL = "CALLABLE"

#: Print/report order.  ``CONFIG`` + ``MUTATED`` + ``NEEDS_AUDIT`` is the old,
#: defective ``CONFIG`` bucket; keeping the three adjacent makes the split legible
#: against the pre-fix numbers.
KIND_ORDER = (
    CONFIG,
    MUTATED,
    NEEDS_AUDIT,
    STATE,
    SINGLE_USE,
    EXCEPT_TARGET,
    DEAD,
    CALLABLE_LOCAL,
)


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
    # Defect 2: a name bound *only* as an except target is deleted at handler exit.
    if b.except_lines and not b.store_lines and not b.is_param:
        return EXCEPT_TARGET
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
    # Defect 1: the name not being rebound says nothing about the value.  A holder
    # admitting these would be a `frozen=True` dataclass wrapping a live B&B tree.
    if b.proven_mutation_lines():
        return MUTATED
    if b.unresolved_call_lines():
        return NEEDS_AUDIT
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


#: Kinds whose names a carve would have to pass across a function boundary.
CARVE_RELEVANT = (CONFIG, MUTATED, NEEDS_AUDIT, STATE)


def crosses(store_regions: list[str], load_regions: list[str], mut_regions: list[str]) -> bool:
    """Single definition of "this name spans more than one region".

    Defect 3 (fixed 2026-07-30): the JSON field and the printed count previously
    used two different expressions and disagreed by 11 names on identical data.
    Every consumer now calls this.
    """
    return len(set(store_regions) | set(load_regions) | set(mut_regions)) > 1


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
    mutation_evidence_sites = 0
    for name, b in sorted(collector.bindings.items()):
        # names that are only ever read are module-level references, not locals
        if not b.store_lines and not b.except_lines and not b.is_param:
            continue
        kind = classify(b, regions, loop)
        classified += 1
        store_regions = region_set(b.store_lines, regions)
        load_regions = region_set(b.load_lines + b.nested_read_lines, regions)
        mut_regions = region_set(b.proven_mutation_lines(), regions)
        mutation_evidence_sites += len(b.mutation_sites)
        rows.append(
            {
                "name": name,
                "kind": kind,
                "is_param": b.is_param,
                "stores": len(b.store_lines),
                "loads": len(b.load_lines),
                "aug": len(b.aug_lines),
                "dels": len(b.del_lines),
                "excepts": len(b.except_lines),
                "nested_reads": len(b.nested_read_lines),
                "nonlocal_writes": len(b.nonlocal_lines),
                "store_regions": store_regions,
                "load_regions": load_regions,
                "mutation_regions": mut_regions,
                "crosses_regions": crosses(store_regions, load_regions, mut_regions),
                "proven_mutations": len(b.proven_mutation_lines()),
                "unresolved_calls": len(b.unresolved_call_lines()),
                "mutation_sites": b.mutation_sites,
                "called_methods": sorted(
                    {
                        s["detail"].lstrip(".")
                        for s in b.mutation_sites
                        if s["kind"] == "method_call"
                    }
                ),
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
        "mutation_evidence_sites": mutation_evidence_sites,
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
    for kind in KIND_ORDER:
        print(f"  {kind:<14}: {census['counts'].get(kind, 0)}")
    old_config = sum(census["counts"].get(k, 0) for k in (CONFIG, MUTATED, NEEDS_AUDIT))
    print(f"  {'(pre-fix CONFIG = CONFIG+MUTATED+NEEDS_AUDIT)':<14}: {old_config}")
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
    crossers = [r for r in rows if r["kind"] in CARVE_RELEVANT and r["crosses_regions"]]
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
                if r["kind"] in CARVE_RELEVANT
                and a in set(r["store_regions"]) | set(r["mutation_regions"])
                and b in r["load_regions"]
            )
            cells.append(f"{n:>13}")
        print(f"  {a:<13}" + "".join(cells))
    print()
    print(f"locals whose bind/read spans more than one region  : {len(crossers)}")
    for kind in CARVE_RELEVANT:
        n = sum(1 for r in crossers if r["kind"] == kind)
        print(f"    of which {kind:<12}                        : {n}")
    params_x = [r for r in rows if r["is_param"] and r["kind"] in CARVE_RELEVANT]
    print(f"parameters (supplied at the boundary by definition): {len(params_x)}")
    closure = [r for r in rows if r["nested_reads"] > 0 and r["kind"] in CARVE_RELEVANT]
    print(f"locals captured by a nested closure                : {len(closure)}")
    nl = [r for r in rows if r["nonlocal_writes"] > 0]
    print(f"locals rebound by a closure via `nonlocal`          : {len(nl)}")
    exc = [r for r in rows if r["kind"] == EXCEPT_TARGET]
    exc_x = [r for r in exc if crosses(r["store_regions"], r["load_regions"], [])]
    print(f"except targets excluded from the crossing count    : {len(exc)}")
    print(f"   of which the pre-fix classifier called crossers : {len(exc_x)}")
    if exc_x:
        print("   " + ", ".join(sorted(r["name"] for r in exc_x)))
    dead = [r["name"] for r in rows if r["kind"] == DEAD]
    print(f"never-read bindings                                : {len(dead)}")
    if dead:
        print("   " + ", ".join(sorted(dead)))
    print()

    print("mutable names the pre-fix classifier called CONFIG (defect 1):")
    print(f"  {'name':<28} {'kind':<12} {'ld':>4} {'proven':>7} {'unres':>6}  evidence")
    suspect = [r for r in rows if r["kind"] in (MUTATED, NEEDS_AUDIT) and r["crosses_regions"]]
    suspect.sort(key=lambda r: (-r["loads"], r["name"]))
    for r in suspect[: args.top]:
        ev = ", ".join(sorted({s["kind"] for s in r["mutation_sites"]}))
        print(
            f"  {r['name']:<28} {r['kind']:<12} {r['loads']:>4} "
            f"{r['proven_mutations']:>7} {r['unresolved_calls']:>6}  {ev}"
        )
    print(f"  ({len(suspect)} cross-region names total; showing {min(args.top, len(suspect))})")
    print()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.json}")

    # CLAUDE.md §6: prove the probe fired.  Two counters, because after the
    # defect-1 fix a run that classified everything but recorded zero mutation
    # sites would mean the new mutation visitor never fired — and would still have
    # printed a plausible-looking census.
    print(f"executed classifications  : {census['classified']}")
    print(f"executed mutation probes  : {census['mutation_evidence_sites']}")
    if census["classified"] == 0:
        print("FAIL: the census classified nothing")
        return 1
    if census["mutation_evidence_sites"] == 0:
        print("FAIL: the mutation visitor recorded no sites — it did not fire")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
