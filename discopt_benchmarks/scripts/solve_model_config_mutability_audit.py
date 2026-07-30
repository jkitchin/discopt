#!/usr/bin/env python
"""Per-name mutability audit of ``solve_model``'s config-crosser population.

**Why this exists.** The 2026-07-30 design review of consolidation-plan item 11
found that the locals census's ``CONFIG`` label meant "the *name* is never
rebound", not "the *value* is immutable", and that at least seven names carrying
21 % of the group's loads are mutated in place. A ``frozen=True`` dataclass built
on that label would embed the live B&B tree and the per-node deadline dict in a
thing called *frozen config*, and — because ``frozen`` only blocks field rebinding
— the mutation would still work, silently. That is a **false guarantee, worse than
leaving them as locals**.

The census fix (three classifier defects) closes the *syntactic* half: it now sees
``opts["max_wall_time"] = …`` and ``kwargs.pop(…)``. It deliberately does **not**
close the other half, and the review was explicit that nobody had:

    "the review explicitly did NOT check transitive mutation through callees —
    ``cl_list``/``cu_list``/``int_offsets`` looked read-only at every call site but
    no callee bodies were read"

That is this script. A name passed to a function that mutates its parameter is
mutated, however read-only the call site looks. So is a name whose method resolves
to a Rust ``fn m(&mut self, …)``. Both are invisible to the census by construction.

.. rubric:: What the audit decides, and how

For every name in the population it emits one of:

``MUTATED_DIRECT``
    The census found tier-1/2 evidence inside ``solve_model`` itself: a subscript
    or attribute store, or a call to a curated unambiguous mutator.

``MUTATED_TRANSITIVE``
    The name is passed as an argument to a callee that mutates the corresponding
    parameter — directly, or by passing it onward to something that does (depth
    limited, cycle-safe). The evidence names the whole chain.

``MUTATED_METHOD``
    A method is called on the name that resolves to a Rust ``&mut self`` receiver,
    or to a Python method that assigns to ``self``.

``UNRESOLVED``
    A method call or a callee the analysis could not resolve. **This is not a
    clear**: it is an explicit hand-off, listed with what could not be resolved so
    a human adjudicates it rather than a default doing so silently.

``CLEAN``
    No mutation found on any of the four channels. Only these may enter a frozen
    holder, and even then arrays get ``setflags(write=False)`` — a cheap *true*
    guarantee replacing the false one.

.. rubric:: Conservatism, deliberately asymmetric

Every unresolved edge degrades toward "not clean", never toward "clean". An
over-broad ``MUTATED`` costs a name its seat in ``RootConfig``; an over-broad
``CLEAN`` produces exactly the false guarantee this audit exists to prevent. When
the analysis cannot decide, it says ``UNRESOLVED`` and refuses to guess (§0.4,
CLAUDE.md §3).

Per CLAUDE.md §6 the script prints executed counts on every channel — names
audited, call sites resolved, callee bodies read, Rust signatures matched — and
exits non-zero if any channel executed nothing, because a resolver that silently
resolved nothing would print a page of reassuring ``CLEAN`` verdicts.

Usage::

    python -u discopt_benchmarks/scripts/solve_model_config_mutability_audit.py \\
        --json reports/solve_model_config_mutability_audit.json \\
        --markdown reports/solve_model_config_mutability_audit.md
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SOLVER_SOURCE = REPO_ROOT / "python" / "discopt" / "solver" / "__init__.py"
PYTHON_ROOT = REPO_ROOT / "python" / "discopt"
CRATES_ROOT = REPO_ROOT / "crates"
FUNCTION_NAME = "solve_model"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.solve_model_locals_census import (  # noqa: E402
    CONFIG,
    KNOWN_MUTATORS,
    MUTATED,
    NEEDS_AUDIT,
)
from scripts.solve_model_locals_census import run as run_census  # noqa: E402

#: How deep to follow argument passing.  Four levels covers `solve_model` ->
#: helper -> relaxer -> primitive; beyond that the chain is reported UNRESOLVED
#: rather than truncated silently.
MAX_DEPTH = 4

#: Builtins that provably do not mutate an argument.  Passing a name to ``len``,
#: ``sorted`` or ``dict`` is not a mutation, and reporting it as *unresolved*
#: buries the handful of edges a human actually has to adjudicate under hundreds
#: that need no thought.  ``setattr``/``delattr`` are deliberately **absent** —
#: they mutate, and are listed in :data:`MUTATING_CALLABLES` instead.
NON_MUTATING_BUILTINS: frozenset[str] = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "bytes",
        "callable",
        "chr",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "ord",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
    }
)

#: Non-discopt callables that *do* mutate an argument in place.  Kept explicit so
#: that filtering builtins for noise cannot quietly clear a real mutation.
MUTATING_CALLABLES: frozenset[str] = frozenset(
    {
        "setattr",
        "delattr",
        "copyto",
        "place",
        "put",
        "putmask",
        "fill_diagonal",
        "heappush",
        "heappop",
        "heapify",
        "heappushpop",
        "heapreplace",
        "shuffle",
    }
)

#: Methods that are unambiguously read-only on every stdlib / numpy / scipy type
#: that defines them.  Curated to the same standard as ``KNOWN_MUTATORS`` in the
#: census: an entry here is a *clear*, so an ambiguous one would be exactly the
#: false guarantee this audit exists to prevent.
READ_ONLY_METHODS: frozenset[str] = frozenset(
    {
        # mapping / sequence / set reads
        "items",
        "keys",
        "values",
        "get",
        "copy",
        "index",
        "count",
        "union",
        "intersection",
        "difference",
        "symmetric_difference",
        "issubset",
        "issuperset",
        "isdisjoint",
        # str
        "startswith",
        "endswith",
        "split",
        "rsplit",
        "strip",
        "lstrip",
        "rstrip",
        "join",
        "lower",
        "upper",
        "replace",
        "encode",
        "decode",
        "isdigit",
        "format",
        "casefold",
        "removeprefix",
        "removesuffix",
        # numpy / scipy reads and copies
        "tolist",
        "astype",
        "reshape",
        "ravel",
        "flatten",
        "squeeze",
        "transpose",
        "argmin",
        "argmax",
        "nonzero",
        "toarray",
        "todense",
        "tocsr",
        "tocsc",
        "tocoo",
        "getnnz",
        "diagonal",
        "cumsum",
        "prod",
        "std",
        "var",
        "mean",
        "round",
        "clip",
        "dot",
        "conj",
        "view",
        "item",
    }
)

V_DIRECT = "MUTATED_DIRECT"
V_TRANSITIVE = "MUTATED_TRANSITIVE"
V_METHOD = "MUTATED_METHOD"
V_UNRESOLVED = "UNRESOLVED"
V_CLEAN = "CLEAN"

#: Worst-first, so a name with several channels reports its strongest evidence.
VERDICT_RANK = {V_DIRECT: 0, V_METHOD: 1, V_TRANSITIVE: 2, V_UNRESOLVED: 3, V_CLEAN: 4}


# --------------------------------------------------------------------------- #
# a function index over python/discopt
# --------------------------------------------------------------------------- #


@dataclass
class FuncDef:
    """One top-level ``def`` somewhere under ``python/discopt``."""

    name: str
    module: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    path: Path

    def param_names(self) -> list[str]:
        a = self.node.args
        return [arg.arg for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs]]

    def bind(self, call: ast.Call) -> dict[int | str, str]:
        """Map each call argument to the parameter name it lands on.

        Keys are the positional index for ``args`` and the keyword for
        ``keywords``; ``*args``/``**kwargs`` forwarding is not mapped (it becomes
        an unresolved edge rather than a guessed one).
        """
        a = self.node.args
        positional = [arg.arg for arg in [*a.posonlyargs, *a.args]]
        kwonly = {arg.arg for arg in a.kwonlyargs}
        out: dict[int | str, str] = {}
        for i, _ in enumerate(call.args):
            if i < len(positional):
                out[i] = positional[i]
        for kw in call.keywords:
            if kw.arg is None:
                continue
            if kw.arg in positional or kw.arg in kwonly:
                out[kw.arg] = kw.arg
        return out


@dataclass
class FuncIndex:
    by_name: dict[str, list[FuncDef]] = field(default_factory=lambda: defaultdict(list))
    #: every ``class`` name defined under ``python/discopt``.  Indexed separately
    #: from methods because a ``@dataclass`` has no ``__init__`` in the source, so
    #: probing for ``Cls.__init__`` misses exactly the classes this tree favours.
    class_names: set[str] = field(default_factory=set)
    files_read: int = 0

    def add_file(self, path: Path) -> None:
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            # A syntax error in the tree under audit is a real problem; surface it
            # rather than skipping the file into a silently smaller index.
            raise
        self.files_read += 1
        module = str(path.relative_to(REPO_ROOT))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.by_name[node.name].append(FuncDef(node.name, module, node, path))
            elif isinstance(node, ast.ClassDef):
                self.class_names.add(node.name)
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.by_name[f"{node.name}.{sub.name}"].append(
                            FuncDef(f"{node.name}.{sub.name}", module, sub, path)
                        )

    def resolve(self, name: str) -> FuncDef | None:
        """Exactly one definition, or ``None``.

        Ambiguity is *not* resolved by picking the first: two functions with the
        same name in different modules would make the chain a guess, and a guessed
        chain that ends in ``CLEAN`` is the failure mode this audit prevents.
        """
        hits = self.by_name.get(name, [])
        return hits[0] if len(hits) == 1 else None


def build_index() -> FuncIndex:
    idx = FuncIndex()
    for path in sorted(PYTHON_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        idx.add_file(path)
    return idx


# --------------------------------------------------------------------------- #
# Rust receiver mutability
# --------------------------------------------------------------------------- #

_RUST_FN = re.compile(r"\bfn\s+(\w+)\s*(?:<[^>]*>)?\s*\(\s*(&(?:\s*mut\s+)?self|self|mut\s+self)\b")


@dataclass
class RustMethods:
    """``&mut self`` vs ``&self`` receivers across ``crates/``.

    The PyO3 objects (`tree`, the model repr) are the reason the census cannot
    adjudicate their methods from Python source alone.  Rust states the answer in
    the signature, which makes this the strongest evidence channel in the audit:
    ``fn export_batch(&mut self, …)`` is a mutation by the compiler's own rule.
    """

    mut_self: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    ref_self: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    files_read: int = 0
    signatures: int = 0

    def classify(self, method: str) -> str | None:
        """``"mut"``, ``"ref"``, ``None`` (unknown), or ``"both"`` (ambiguous)."""
        m, r = method in self.mut_self, method in self.ref_self
        if m and r:
            return "both"
        if m:
            return "mut"
        if r:
            return "ref"
        return None


def scan_rust() -> RustMethods:
    rm = RustMethods()
    if not CRATES_ROOT.is_dir():
        return rm
    for path in sorted(CRATES_ROOT.rglob("*.rs")):
        text = path.read_text(errors="replace")
        rm.files_read += 1
        for match in _RUST_FN.finditer(text):
            name, recv = match.group(1), match.group(2)
            rm.signatures += 1
            rel = str(path.relative_to(REPO_ROOT))
            if "mut" in recv:
                rm.mut_self[name].append(rel)
            else:
                rm.ref_self[name].append(rel)
    return rm


# --------------------------------------------------------------------------- #
# does a function mutate one of its parameters?
# --------------------------------------------------------------------------- #


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def taint_closure(fn: ast.AST, root: str) -> set[str]:
    """Local names that may alias part of the object bound to ``root``.

    ``HeuristicGovernor.record`` never writes ``self.x`` — it does
    ``st = self._get(source)`` and then ``st.calls += 1``. A probe that only
    looked for ``self.x = …`` called it read-only, which is how the governor came
    out ``CLEAN`` in the first run of this audit. The same shape appears for
    parameters: ``rows = arg.rows; rows[0] = …`` mutates the caller's object.

    The closure **only grows** — a name is never untainted by a later rebinding.
    That over-approximates, and over-approximation here means a name is denied a
    seat in ``RootConfig``, which is the safe direction (see the module docstring
    on asymmetric conservatism). Under-approximating would mint a false guarantee.
    """
    tainted = {root}
    for _ in range(3):  # fixpoint; 3 rounds is ample for straight-line helpers
        grew = False
        for node in ast.walk(fn):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                if value is None:
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if _names_in(value) & tainted:
                    for t in targets:
                        for sub in ast.walk(t):
                            if isinstance(sub, ast.Name) and sub.id not in tainted:
                                tainted.add(sub.id)
                                grew = True
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                if _names_in(node.iter) & tainted:
                    for sub in ast.walk(node.target):
                        if isinstance(sub, ast.Name) and sub.id not in tainted:
                            tainted.add(sub.id)
                            grew = True
            elif isinstance(node, ast.withitem):
                if node.optional_vars is not None and _names_in(node.context_expr) & tainted:
                    for sub in ast.walk(node.optional_vars):
                        if isinstance(sub, ast.Name) and sub.id not in tainted:
                            tainted.add(sub.id)
                            grew = True
        if not grew:
            break
    return tainted


@dataclass
class Mutation:
    """One reason a name is considered mutated, with its provenance chain."""

    channel: str
    where: str
    detail: str
    chain: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "where": self.where,
            "detail": self.detail,
            "chain": self.chain,
        }


@dataclass
class Analyzer:
    index: FuncIndex
    rust: RustMethods
    #: counters — CLAUDE.md §6
    call_sites_examined: int = 0
    callees_resolved: int = 0
    callee_bodies_read: int = 0
    rust_methods_matched: int = 0
    method_calls_examined: int = 0
    receiver_classes_inferred: int = 0
    _memo: dict[tuple[str, str], list[Mutation]] = field(default_factory=dict)
    #: name -> inferred constructor class, for method resolution
    receiver_class: dict[str, str] = field(default_factory=dict)

    def infer_receiver_classes(self, fn: FuncDef) -> None:
        """Infer ``x``'s class from ``x = SomeClass(...)`` inside ``fn``.

        Without this, a method call ``_heuristic_governor.record(...)`` has to be
        resolved by searching *every* class in the tree for a method named
        ``record`` — and answering "none of them assign to self, therefore clean"
        is a guess in the one direction this audit must never guess.  With the
        class known, ``HeuristicGovernor.record`` resolves exactly.
        """
        assigned: dict[str, set[str]] = defaultdict(set)
        for node in ast.walk(fn.node):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Call):
                continue
            cls = self._class_of_call(node.value, depth=3)
            if cls:
                assigned[target.id].add(cls)
        # Only a single unambiguous constructor counts; two different ones on the
        # same name means the class is branch-dependent and must stay unresolved.
        for name, classes in assigned.items():
            if len(classes) == 1:
                self.receiver_class[name] = next(iter(classes))
                self.receiver_classes_inferred += 1

    def _class_of_call(self, call: ast.Call, depth: int) -> str | None:
        """Class produced by ``call``, following one level of factory function.

        ``x = SomeClass(...)`` is the easy case.  ``_heuristic_governor =
        _get_heuristic_governor()`` is the case that matters and the one an
        initial-capital heuristic misses entirely: the factory is lower-case and
        the class it returns lives in another module.  Both are resolved here, and
        an unresolvable factory yields ``None`` (i.e. UNRESOLVED, never CLEAN).
        """
        if depth <= 0:
            return None
        func = call.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if not name:
            return None
        if name in self.index.class_names:
            return name
        callee = self.index.resolve(name)
        if callee is None:
            return None
        # A return annotation is the most reliable signal and the one that resolves
        # `governor() -> HeuristicGovernor`, whose body is `return _GOVERNOR` — a
        # module-level singleton no call-graph walk would reach.
        ann = callee.node.returns
        ann_name = ann.id if isinstance(ann, ast.Name) else getattr(ann, "attr", None)
        if ann_name and ann_name in self.index.class_names:
            return ann_name
        produced: set[str] = set()
        for node in ast.walk(callee.node):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
                cls = self._class_of_call(node.value, depth - 1)
                if cls:
                    produced.add(cls)
        return next(iter(produced)) if len(produced) == 1 else None

    # -- level 1: syntactic mutation of `target` inside one function body ---- #

    def direct_mutations(self, fn: FuncDef, target: str) -> list[Mutation]:
        found: list[Mutation] = []
        # Names that may alias part of `target`, so that `rows = arg.rows;
        # rows[0] = …` counts as mutating `arg` (see `taint_closure`).
        aliases = taint_closure(fn.node, target)
        for node in ast.walk(fn.node):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        if isinstance(node, ast.AugAssign) and t.id == target:
                            # `p += x` rebinds the local name, but for an ndarray or
                            # a list `__iadd__` mutates in place and the caller sees
                            # it.  Conservative by design (see the module docstring).
                            found.append(
                                Mutation(
                                    "augassign_param",
                                    f"{fn.module}:{node.lineno}",
                                    f"{target} += … (in-place for ndarray/list)",
                                )
                            )
                        continue
                    root = _root_name(t)
                    if root in aliases:
                        found.append(
                            Mutation(
                                "store_into" if root == target else "store_into_alias",
                                f"{fn.module}:{node.lineno}",
                                ast.unparse(t)[:70],
                            )
                        )
            elif isinstance(node, ast.Delete):
                for t in node.targets:
                    if not isinstance(t, ast.Name) and _root_name(t) in aliases:
                        found.append(
                            Mutation("delete_from", f"{fn.module}:{node.lineno}", ast.unparse(t))
                        )
            elif isinstance(node, ast.Call):
                # `np.copyto(dst, …)` / any ufunc with `out=<target>`
                for kw in node.keywords:
                    if kw.arg == "out" and isinstance(kw.value, ast.Name) and kw.value.id == target:
                        found.append(
                            Mutation(
                                "numpy_out",
                                f"{fn.module}:{node.lineno}",
                                f"out={target}",
                            )
                        )
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in aliases
                    and node.func.attr in KNOWN_MUTATORS
                ):
                    found.append(
                        Mutation(
                            "known_mutator",
                            f"{fn.module}:{node.lineno}",
                            f".{node.func.attr}()",
                        )
                    )
        return found

    # -- level 2: methods on the name, resolved through Rust / Python -------- #

    def method_mutations(self, fn: FuncDef, target: str) -> tuple[list[Mutation], list[str]]:
        found: list[Mutation] = []
        unresolved: list[str] = []
        seen: set[str] = set()
        for node in ast.walk(fn.node):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == target
            ):
                continue
            attr = node.func.attr
            self.method_calls_examined += 1
            if attr in KNOWN_MUTATORS or attr in seen:
                continue
            seen.add(attr)

            # 1. the receiver's own class, when it is known — the precise answer
            cls = self.receiver_class.get(target)
            if cls is not None:
                fd = self.index.resolve(f"{cls}.{attr}")
                if fd is not None:
                    site = self._method_self_store(fd)
                    if site:
                        found.append(
                            Mutation("python_self_store", site, f".{attr}() assigns to self")
                        )
                    continue

            # 2. a Rust `&mut self` receiver is the compiler's own verdict
            verdict = self.rust.classify(attr)
            if verdict == "mut":
                self.rust_methods_matched += 1
                found.append(
                    Mutation(
                        "rust_mut_self",
                        ", ".join(sorted(set(self.rust.mut_self[attr]))[:2]),
                        f".{attr}() resolves to `fn {attr}(&mut self, …)`",
                    )
                )
                continue
            if verdict == "ref":
                self.rust_methods_matched += 1
                continue
            if verdict == "both":
                unresolved.append(f".{attr}() — Rust has both &self and &mut self overloads")
                continue

            # 3. an unambiguously read-only stdlib/numpy method
            if attr in READ_ONLY_METHODS:
                continue

            # 4. exactly one Python method with that name anywhere in the tree
            candidates = [
                fd
                for key, defs in self.index.by_name.items()
                if key.endswith(f".{attr}")
                for fd in defs
            ]
            if len(candidates) == 1:
                site = self._method_self_store(candidates[0])
                if site:
                    found.append(Mutation("python_self_store", site, f".{attr}() assigns to self"))
                continue
            if any(self._method_self_store(fd) for fd in candidates):
                # Several classes define it and at least one mutates.  Conservative:
                # this is not a clear.  Naming the mutating one keeps it actionable.
                site = next(s for fd in candidates if (s := self._method_self_store(fd)))
                unresolved.append(
                    f".{attr}() — {len(candidates)} definitions, at least one mutates self ({site})"
                )
                continue
            unresolved.append(
                f".{attr}() — receiver class unknown and {len(candidates)} Python definitions"
            )
        return found, unresolved

    @staticmethod
    def _method_self_store(fd: FuncDef) -> str | None:
        """Location of the first store into ``self`` or anything reached from it.

        Not merely ``self.x = …``: ``HeuristicGovernor.record`` mutates through
        ``st = self._get(source); st.calls += 1``, and an audit that missed that
        cleared a governor which throttles primal heuristics per node.
        """
        aliases = taint_closure(fd.node, "self")
        for node in ast.walk(fd.node):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, (ast.Attribute, ast.Subscript)) and _root_name(t) in aliases:
                        return f"{fd.module}:{node.lineno} ({fd.name})"
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in aliases
                and node.func.attr in KNOWN_MUTATORS
            ):
                return f"{fd.module}:{node.lineno} ({fd.name}, .{node.func.attr}())"
        return None

    # -- level 3: the name passed to a callee that mutates its parameter ----- #

    def transitive_mutations(
        self, fn: FuncDef, target: str, depth: int, chain: list[str]
    ) -> tuple[list[Mutation], list[str]]:
        found: list[Mutation] = []
        unresolved: list[str] = []
        if depth <= 0:
            return found, [f"depth limit reached following {' -> '.join(chain)}"]
        for node in ast.walk(fn.node):
            if not isinstance(node, ast.Call):
                continue
            # which argument slots carry `target` bare?  (`f(x)`, `f(k=x)`)
            slots: list[int | str] = []
            for i, arg in enumerate(node.args):
                if isinstance(arg, ast.Name) and arg.id == target:
                    slots.append(i)
            for kw in node.keywords:
                if kw.arg is not None and isinstance(kw.value, ast.Name) and kw.value.id == target:
                    slots.append(kw.arg)
            if not slots:
                continue
            self.call_sites_examined += 1
            callee_name = self._callee_name(node.func)
            if callee_name is None:
                unresolved.append(f"{fn.module}:{node.lineno} call through a non-name callee")
                continue
            if callee_name in MUTATING_CALLABLES:
                found.append(
                    Mutation(
                        "mutating_callable",
                        f"{fn.module}:{node.lineno}",
                        f"{callee_name}(…) mutates the argument in place",
                        chain=list(chain),
                    )
                )
                continue
            if callee_name in NON_MUTATING_BUILTINS:
                # Not a mutation and not an open question — do not report it as
                # unresolved, or the handful of real edges drown in `len()` calls.
                continue
            callee = self.index.resolve(callee_name)
            if callee is None:
                n = len(self.index.by_name.get(callee_name, []))
                unresolved.append(
                    f"{fn.module}:{node.lineno} -> {callee_name}(…) "
                    + (f"ambiguous: {n} definitions" if n else "not defined under python/discopt")
                )
                continue
            self.callees_resolved += 1
            binding = callee.bind(node)
            for slot in slots:
                param = binding.get(slot)
                if param is None:
                    unresolved.append(
                        f"{fn.module}:{node.lineno} -> {callee_name}(…) "
                        f"slot {slot!r} lands on *args/**kwargs"
                    )
                    continue
                sub_chain = [*chain, f"{callee_name}({param})"]
                subs, sub_unres = self._mutations_of_param(callee, param, depth - 1, sub_chain)
                for m in subs:
                    found.append(
                        Mutation(
                            m.channel,
                            m.where,
                            m.detail,
                            chain=[f"{fn.module}:{node.lineno}", *sub_chain],
                        )
                    )
                unresolved.extend(sub_unres)
        return found, unresolved

    def _mutations_of_param(
        self, fn: FuncDef, param: str, depth: int, chain: list[str]
    ) -> tuple[list[Mutation], list[str]]:
        key = (f"{fn.module}::{fn.name}", param)
        if key in self._memo:
            return list(self._memo[key]), []
        self._memo[key] = []  # cycle guard: a recursive chain resolves to "nothing yet"
        self.callee_bodies_read += 1
        found = self.direct_mutations(fn, param)
        meth, unres = self.method_mutations(fn, param)
        found.extend(meth)
        deeper, deeper_unres = self.transitive_mutations(fn, param, depth, chain)
        found.extend(deeper)
        unres.extend(deeper_unres)
        self._memo[key] = found
        return found, unres

    @staticmethod
    def _callee_name(func: ast.AST) -> str | None:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            # `mod.f(...)` — index by the bare name; `resolve` refuses ambiguity.
            return func.attr
        return None


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def audit(source: Path, function_name: str) -> dict[str, Any]:
    census = run_census(source, function_name)
    rows = {r["name"]: r for r in census["rows"]}

    def pre_fix_crosses(r: dict[str, Any]) -> bool:
        """The population the design review called 'the 84'.

        Reproduced with the *pre-fix* JSON predicate on purpose: this audit has to
        answer for every name the review was reasoning about, including the
        never-rebound parameters that predicate swept in.
        """
        sr = r["store_regions"]
        union = sorted(set(sr) | set(r["load_regions"]))
        return union != sr or len(union) > 1

    population = sorted(
        (
            r
            for r in rows.values()
            if r["kind"] in (CONFIG, MUTATED, NEEDS_AUDIT) and pre_fix_crosses(r)
        ),
        key=lambda r: (-r["loads"], r["name"]),
    )

    index = build_index()
    rust = scan_rust()
    analyzer = Analyzer(index=index, rust=rust)

    solve_model = index.resolve(function_name)
    if solve_model is None:
        raise SystemExit(f"could not resolve a unique top-level {function_name}")
    analyzer.infer_receiver_classes(solve_model)

    results: list[dict[str, Any]] = []
    for row in population:
        name = row["name"]
        muts: list[Mutation] = []
        unresolved: list[str] = []

        # channel 1 — what the census already proved, inside solve_model
        for site in row["mutation_sites"]:
            if site["tier"] != "unresolved_call":
                muts.append(
                    Mutation(
                        "census_" + site["kind"],
                        f"{solve_model.module}:{site['line']}",
                        site["detail"],
                    )
                )
        direct = list(muts)

        # channel 2 — methods on the name, resolved through Rust/Python
        meth, meth_unres = analyzer.method_mutations(solve_model, name)
        unresolved.extend(meth_unres)

        # channel 3/4 — passed to a callee that mutates the parameter
        trans, trans_unres = analyzer.transitive_mutations(
            solve_model, name, MAX_DEPTH, [f"{function_name}({name})"]
        )
        unresolved.extend(trans_unres)

        if direct:
            verdict = V_DIRECT
        elif meth:
            verdict = V_METHOD
        elif trans:
            verdict = V_TRANSITIVE
        elif unresolved:
            verdict = V_UNRESOLVED
        else:
            verdict = V_CLEAN

        results.append(
            {
                "name": name,
                "census_kind": row["kind"],
                "is_param": row["is_param"],
                "loads": row["loads"],
                "verdict": verdict,
                "caught_only_transitively": bool(trans) and not direct and not meth,
                "direct": [m.as_dict() for m in direct],
                "method": [m.as_dict() for m in meth],
                "transitive": [m.as_dict() for m in trans][:6],
                "n_transitive": len(trans),
                "unresolved": sorted(set(unresolved))[:6],
                "n_unresolved": len(set(unresolved)),
                "called_methods": row["called_methods"],
            }
        )

    counts: dict[str, int] = defaultdict(int)
    for r in results:
        counts[r["verdict"]] += 1

    return {
        "source": str(source.relative_to(REPO_ROOT)),
        "function": function_name,
        "population": len(results),
        "population_loads": sum(r["loads"] for r in results),
        "max_depth": MAX_DEPTH,
        "counts": dict(counts),
        "caught_only_transitively": sorted(
            r["name"] for r in results if r["caught_only_transitively"]
        ),
        "executed": {
            "names_audited": len(results),
            "python_files_indexed": index.files_read,
            "functions_indexed": sum(len(v) for v in index.by_name.values()),
            "classes_indexed": len(index.class_names),
            "rust_files_scanned": rust.files_read,
            "rust_signatures_indexed": rust.signatures,
            "call_sites_examined": analyzer.call_sites_examined,
            "callees_resolved": analyzer.callees_resolved,
            "callee_bodies_read": analyzer.callee_bodies_read,
            "method_calls_examined": analyzer.method_calls_examined,
            "receiver_classes_inferred": analyzer.receiver_classes_inferred,
            "rust_methods_matched": analyzer.rust_methods_matched,
        },
        "rows": results,
    }


def to_markdown(rep: dict[str, Any]) -> str:
    out: list[str] = []
    out.append("# `solve_model` config-crosser mutability audit\n")
    out.append(
        "Generated by `discopt_benchmarks/scripts/solve_model_config_mutability_audit.py`.\n"
        "Population: the pre-fix `CONFIG` cross-region group — the names the "
        "2026-07-30 design review called *the 84*.\n"
    )
    out.append(
        f"\n**Population** {rep['population']} names, {rep['population_loads']} loads. "
        f"**Depth limit** {rep['max_depth']}.\n"
    )
    out.append("\n| verdict | names |\n|---|---|\n")
    for v in (V_DIRECT, V_METHOD, V_TRANSITIVE, V_UNRESOLVED, V_CLEAN):
        out.append(f"| `{v}` | {rep['counts'].get(v, 0)} |\n")
    out.append(
        f"\n**Caught only by transitive callee analysis** "
        f"({len(rep['caught_only_transitively'])}): "
        + (", ".join(f"`{n}`" for n in rep["caught_only_transitively"]) or "_none_")
        + "\n"
    )
    out.append("\n## Executed counts (CLAUDE.md §6)\n\n| channel | count |\n|---|---|\n")
    for k, v in rep["executed"].items():
        out.append(f"| {k} | {v} |\n")
    out.append("\n## Per-name verdicts\n\n")
    out.append("| name | loads | census | verdict | evidence |\n|---|---|---|---|---|\n")
    for r in rep["rows"]:
        ev = ""
        if r["direct"]:
            d = r["direct"][0]
            ev = f"`{d['detail']}` at {d['where']}" + (
                f" (+{len(r['direct']) - 1} more)" if len(r["direct"]) > 1 else ""
            )
        elif r["method"]:
            d = r["method"][0]
            ev = f"{d['detail']}" + (
                f" (+{len(r['method']) - 1} more)" if len(r["method"]) > 1 else ""
            )
        elif r["transitive"]:
            d = r["transitive"][0]
            ev = f"`{' -> '.join(d['chain'][-2:])}` {d['channel']} at {d['where']}"
            if r["n_transitive"] > 1:
                ev += f" (+{r['n_transitive'] - 1} more)"
        elif r["unresolved"]:
            ev = r["unresolved"][0]
            if r["n_unresolved"] > 1:
                ev += f" (+{r['n_unresolved'] - 1} more)"
        else:
            ev = "no mutation on any channel"
        ev = ev.replace("|", "\\|")
        out.append(
            f"| `{r['name']}` | {r['loads']} | {r['census_kind']} | **{r['verdict']}** | {ev} |\n"
        )
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=SOLVER_SOURCE)
    ap.add_argument("--function", default=FUNCTION_NAME)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args(argv)

    rep = audit(args.source, args.function)

    print(f"population        : {rep['population']} names, {rep['population_loads']} loads")
    print("verdicts:")
    for v in (V_DIRECT, V_METHOD, V_TRANSITIVE, V_UNRESOLVED, V_CLEAN):
        print(f"  {v:<20}: {rep['counts'].get(v, 0)}")
    print(
        f"caught ONLY transitively: {len(rep['caught_only_transitively'])} "
        f"{rep['caught_only_transitively']}"
    )
    print()
    print("executed counts (CLAUDE.md §6):")
    for k, v in rep["executed"].items():
        print(f"  {k:<26}: {v}")
    print()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.json}")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(to_markdown(rep))
        print(f"wrote {args.markdown}")

    # CLAUDE.md §6: every channel must have fired.  A resolver that resolved
    # nothing would otherwise print a page of reassuring CLEAN verdicts.
    failed = [k for k, v in rep["executed"].items() if v == 0]
    if failed:
        print(f"FAIL: these channels executed nothing: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
