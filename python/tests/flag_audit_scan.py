"""Source scan for `DISCOPT_*` environment gates, used by the CLAUDE.md §5 audit.

Not a test module — the enforcement lives in ``test_1345_flag_retirement_audit.py``.
This is the instrument, kept separate so it can be run standalone as a probe.

Issue #1421: the previous scan was a single regex for ``environ.get(F, "0")``. It saw
10 of the 60 flags then live and **none** of the 23 ``_env_flag(..., default=False)``
gates, several of which are bound-changing solver math (the whole ``DISCOPT_RLT``
family, ``SHOR_SDP_ROOT_BOUND``, ``PHASE2_DBBT``). Worse, the audit document's
``<!-- live-solver-math-gates: N -->`` marker was derived from the same narrow
definition, so scan and document agreed with each other while the real population was
several times larger — two instruments, one blind spot, mutual confirmation.

Three rules make this scan honest about what it cannot see:

1. **Polarity is never read off the default literal.** ``environ.get(F, "")`` is
   default-OFF when the predicate is ``in ("1","true",…)`` and default-**ON** when it
   is ``not in ("0","false",…)``; five live flags are the latter. Polarity comes from
   the gate's own comparison, from ``_env_flag``'s ``default=`` keyword, or from
   calling the accessor with the environment cleared.
2. **A non-boolean default is not a gate.** ``environ.get(F, "120")`` /
   ``"tape"`` / ``"lifted"`` are a budget and two selectors; ``0`` in a numeric knob
   is a value, not an off-switch (CLAUDE.md §5 "Out of scope"). These are classified
   ``selector`` and never counted as default-OFF.
3. **Anything undecidable is ``unknown``, never assumed.** The caller must account for
   every ``unknown`` explicitly; silently defaulting them to "ON, not my problem" is
   how the 50 flags stayed invisible in the first place.
"""

from __future__ import annotations

import ast
import importlib
import os
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parents[2]
PKG = REPO / "python" / "discopt"

#: Words a gate accepts as "on" and as "off". A membership test against the first set
#: means the predicate IS the feature; against the second it is the feature's negation.
ON_WORDS = frozenset({"1", "true", "yes", "on"})
OFF_WORDS = frozenset({"0", "false", "no", "off"})

#: Defaults that can appear in a boolean gate at all. Anything else (``"120"``,
#: ``"tape"``, ``"highs"``) makes the read a budget or a selector, not a gate.
BOOLEAN_DEFAULTS = frozenset({"", "0", "1", "true", "false", "yes", "no", "on", "off"})


@dataclass(frozen=True)
class Site:
    """One read of one `DISCOPT_*` variable."""

    flag: str
    form: str  # env_flag | literal | no_default | subscript
    polarity: str  # on | off | selector | unknown:<reason>
    path: str
    line: int
    accessor: str  # enclosing zero-arg function, or "-"

    @property
    def where(self) -> str:
        return f"{self.path}:{self.line}"


# ---------------------------------------------------------------- AST predicates


def _is_env_get(node: ast.Call) -> bool:
    f = node.func
    if isinstance(f, ast.Attribute) and f.attr == "get":
        v = f.value
        return (isinstance(v, ast.Name) and v.id == "environ") or (
            isinstance(v, ast.Attribute) and v.attr == "environ"
        )
    return (isinstance(f, ast.Attribute) and f.attr == "getenv") or (
        isinstance(f, ast.Name) and f.id == "getenv"
    )


def _is_env_flag(node: ast.Call) -> bool:
    f = node.func
    return (isinstance(f, ast.Name) and f.id == "_env_flag") or (
        isinstance(f, ast.Attribute) and f.attr == "_env_flag"
    )


def _is_env_subscript(node: ast.Subscript) -> bool:
    v = node.value
    return (isinstance(v, ast.Name) and v.id == "environ") or (
        isinstance(v, ast.Attribute) and v.attr == "environ"
    )


def _flag_of(node, consts: dict[str, str] | None = None) -> str | None:
    """The flag this read names — directly, or through a module-level constant.

    `_block_eval.py` binds `BLOCK_VECTOR_EVAL_ENV = "DISCOPT_BLOCK_VECTOR_EVAL"` and
    reads `environ.get(BLOCK_VECTOR_EVAL_ENV, "0")`. That indirection is a fifth read
    form, and it hid a documented default-OFF gate from a scan that already had four.
    """
    arg = node.args[0] if isinstance(node, ast.Call) and node.args else None
    if isinstance(node, ast.Subscript):
        arg = node.slice
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value if arg.value.startswith("DISCOPT_") else None
    if isinstance(arg, ast.Name) and consts:
        return consts.get(arg.id)
    return None


def _module_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "DISCOPT_…"`` bindings."""
    out: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant):
            val = stmt.value.value
            if isinstance(val, str) and val.startswith("DISCOPT_"):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        out[t.id] = val
    return out


def _root_of_chain(node, parents):
    """Ascend through ``.strip()``/``.lower()``-style method calls on the read."""
    cur = node
    while True:
        par = parents.get(cur)
        if isinstance(par, ast.Attribute) and par.value is cur:
            cur = par
            continue
        if isinstance(par, ast.Call) and par.func is cur:
            cur = par
            continue
        return cur


# ---------------------------------------------------------------- polarity


def _word_set(cmp_node: ast.Compare) -> frozenset[str] | None:
    """The literal string words an ``in``/``not in`` comparison tests against."""
    if len(cmp_node.ops) != 1 or not isinstance(cmp_node.ops[0], (ast.In, ast.NotIn)):
        if len(cmp_node.ops) == 1 and isinstance(cmp_node.ops[0], (ast.Eq, ast.NotEq)):
            c = cmp_node.comparators[0]
            if isinstance(c, ast.Constant) and isinstance(c.value, str):
                return frozenset({c.value.lower()})
        return None
    c = cmp_node.comparators[0]
    if not isinstance(c, (ast.Tuple, ast.List, ast.Set)):
        return None
    if not all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in c.elts):
        return None
    # Compare on the SET, not on the element count: `("0","false","False")` is three
    # elements and two words, and counting them made a live gate unreadable.
    return frozenset(e.value.lower() for e in c.elts)


def _polarity_from_compare(cmp_node: ast.Compare, default: str) -> str:
    """Is the feature on when the variable is unset?

    The operator's sign is deliberately NOT consulted, and getting that wrong is the
    first thing this function did. What decides the feature's state is only whether
    the default value is an ON-word or an OFF-word:

      ``val in ("1","true")``      default ""  -> not an on-word -> feature OFF
      ``val not in ("0","false")`` default ""  -> not an off-word -> feature ON
      ``val in ("0","false")``     default "1" -> not an off-word -> feature ON
                                   (the ``in`` guards an early ``return None``)

    The last two differ in operator and agree in meaning, which is exactly why the
    operator cannot be part of the rule. A word set that mixes the two vocabularies
    (or names something else entirely) is ``unknown`` and falls through to calling
    the accessor.
    """
    words = _word_set(cmp_node)
    if words is None:
        return "unknown:comparator-not-literal"
    hit = default.strip().lower() in words
    # `""` is listed alongside real off-words by gates that treat an empty variable as
    # off (`not in ("0","","false")`), but a gate whose ONLY comparison is `raw == ""`
    # is the opposite shape — it returns True there (`_convex_stall_abstain_enabled`).
    # So `""` counts for membership but not for deciding which vocabulary this is; a
    # set that is nothing but `{""}` stays undecided and falls through to the accessor.
    core = words - {""}
    if core and core <= ON_WORDS:
        return "on" if hit else "off"
    if core and core <= OFF_WORDS:
        return "off" if hit else "on"
    if core and not (core & (ON_WORDS | OFF_WORDS)):
        # Compared against `"scipy"` / `"nested"` / `"lifted"`: a backend or mode
        # selector that happens to be spelled as a string test, not a boolean gate.
        return "selector"
    return "unknown:mixed-word-set"


def _find_compare(start, parents, tree, fn_scope) -> ast.Compare | None:
    """The comparison that consumes this read, directly or via one local binding."""
    root = _root_of_chain(start, parents)
    par = parents.get(root)
    if isinstance(par, ast.Compare):
        return par
    # `raw = environ.get(...).strip().lower()` then `raw in (...)` / `raw == ""`.
    if (
        isinstance(par, ast.Assign)
        and len(par.targets) == 1
        and isinstance(par.targets[0], ast.Name)
    ):
        name = par.targets[0].id
        scope = fn_scope if fn_scope is not None else tree
        for n in ast.walk(scope):
            if isinstance(n, ast.Compare):
                left = _root_of_chain(n.left, parents) if False else n.left
                probe = left
                while isinstance(probe, (ast.Call, ast.Attribute)):
                    probe = probe.func if isinstance(probe, ast.Call) else probe.value
                if isinstance(probe, ast.Name) and probe.id == name:
                    return n
    return None


def _call_accessor(modname: str, funcname: str) -> str:
    """Call a zero-arg module-level accessor with the environment cleared.

    This is the authoritative measurement where it applies: it runs the same code the
    solver runs. Exceptions are NOT swallowed into a polarity — they become an
    ``unknown`` naming the exception (CLAUDE.md §7).
    """
    try:
        mod = importlib.import_module(modname)
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        return f"unknown:import:{type(exc).__name__}"
    fn = getattr(mod, funcname, None)
    if fn is None or not callable(fn):
        return "unknown:not-module-level"
    try:
        with mock.patch.dict(os.environ, {}, clear=True):
            val = fn()
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        return f"unknown:call:{type(exc).__name__}"
    return "on" if val else "off"


# ---------------------------------------------------------------- the scan


def _module_name(path: pathlib.Path) -> str:
    rel = path.relative_to(PKG).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(["discopt", *parts])


def scan() -> dict[str, list[Site]]:
    """Every `DISCOPT_*` read under `python/discopt`, keyed by flag name."""
    if str(REPO / "python") not in sys.path:
        sys.path.insert(0, str(REPO / "python"))

    by_flag: dict[str, list[Site]] = defaultdict(list)
    n_files = 0
    for path in sorted(PKG.rglob("*.py")):
        n_files += 1
        tree = ast.parse(path.read_text())
        parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        # Enclosing function per node: its name when zero-arg (callable as a probe),
        # and the scope object either way (needed to resolve a local binding).
        accessor: dict[int, str] = {}
        scope: dict[int, ast.AST] = {}
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = fn.args
                zero = not (a.args or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg)
                for sub in ast.walk(fn):
                    scope.setdefault(id(sub), fn)
                    if zero:
                        accessor.setdefault(id(sub), fn.name)
        consts = _module_constants(tree)
        rel = str(path.relative_to(PKG))

        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and _is_env_subscript(node):
                flag = _flag_of(node, consts)
                if flag:
                    by_flag[flag].append(
                        Site(flag, "subscript", "unknown:subscript", rel, node.lineno, "-")
                    )
                continue
            if not isinstance(node, ast.Call):
                continue
            if _is_env_flag(node):
                flag = _flag_of(node, consts)
                if not flag:
                    continue
                d = {k.arg: k.value for k in node.keywords}.get("default")
                pol = (
                    "on"
                    if isinstance(d, ast.Constant) and d.value is True
                    else "off"
                    if isinstance(d, ast.Constant) and d.value is False
                    else "unknown:env_flag-default-not-literal"
                )
                by_flag[flag].append(Site(flag, "env_flag", pol, rel, node.lineno, "-"))
                continue
            if not _is_env_get(node):
                continue
            flag = _flag_of(node, consts)
            if not flag:
                continue

            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                default = node.args[1].value
                form = "literal"
            elif len(node.args) >= 2:
                default, form = None, "nonliteral_default"
            else:
                default, form = None, "no_default"

            fname = accessor.get(id(node), "-")
            # A read wrapped in int()/float() is a numeric knob: `0` is a value (zero
            # rounds, zero offset), not an off-switch. CLAUDE.md §5 puts these out of
            # scope by construction, and `"0"` is otherwise a perfectly boolean default.
            cast = parents.get(_root_of_chain(node, parents))
            if (
                isinstance(cast, ast.Call)
                and isinstance(cast.func, ast.Name)
                and cast.func.id in {"int", "float"}
            ):
                by_flag[flag].append(Site(flag, form, "selector", rel, node.lineno, fname))
                continue
            if form == "literal" and isinstance(default, str):
                if default.strip().lower() not in BOOLEAN_DEFAULTS:
                    # Rule 2: a non-boolean default makes this a budget or a selector.
                    by_flag[flag].append(Site(flag, form, "selector", rel, node.lineno, fname))
                    continue
            elif form == "literal":
                # int/float literal default -> numeric knob.
                by_flag[flag].append(Site(flag, form, "selector", rel, node.lineno, fname))
                continue

            pol = "unknown:unresolved"
            cmp_node = _find_compare(node, parents, tree, scope.get(id(node)))
            if cmp_node is not None:
                pol = _polarity_from_compare(cmp_node, default if isinstance(default, str) else "")
            if pol.startswith("unknown") and fname != "-":
                pol = _call_accessor(_module_name(path), fname)
            by_flag[flag].append(Site(flag, form, pol, rel, node.lineno, fname))

    assert n_files > 100, f"only scanned {n_files} files — the scan measured nothing"
    assert by_flag, "no DISCOPT_* reads found — the scan stopped matching"
    return dict(by_flag)


def polarity_of(sites: list[Site]) -> str:
    """One verdict per flag. Disagreement between read sites is itself a finding."""
    kinds = {s.polarity.split(":")[0] for s in sites}
    if kinds == {"selector"}:
        return "selector"
    kinds -= {"selector"}
    if kinds == {"off"}:
        return "off"
    if kinds == {"on"}:
        return "on"
    if kinds == {"on", "off"}:
        return "mixed"
    return "unknown"


def main() -> int:
    by_flag = scan()
    n_sites = sum(len(v) for v in by_flag.values())
    buckets: dict[str, list[str]] = defaultdict(list)
    for flag, sites in sorted(by_flag.items()):
        buckets[polarity_of(sites)].append(flag)

    print(f"distinct DISCOPT_* flags : {len(by_flag)}")
    print(f"read sites classified    : {n_sites}")
    for kind in ("off", "on", "selector", "mixed", "unknown"):
        print(f"  {kind:10s} {len(buckets[kind]):4d}")
    for kind in ("off", "unknown", "mixed"):
        print(f"\n-- {kind} --")
        for flag in buckets[kind]:
            s = by_flag[flag][0]
            print(f"  {flag:42s} {s.polarity:34s} {s.form:12s} {s.where}")
    print(f"\nEXECUTED CLASSIFICATIONS: {n_sites}")
    if n_sites == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
