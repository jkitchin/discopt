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

Issue #2211: the same blind spot, one language over. The scan above walks
``python/discopt`` only, so every gate the Rust core reads through ``std::env::var``
(usually behind a ``OnceLock``) was invisible to it, and the audit's
``live-solver-math-gates`` marker was a Python-only count. Ten default-OFF gates over
solver math had no row, ``DISCOPT_PRIMAL_EXPAND_RESET`` among them. ``scan_rust``
below reads ``crates/*/src`` and classifies each read by the expression that consumes
it (see ``_rust_polarity``); ``scan()`` merges both languages, so a flag read on both
sides with different polarity surfaces as ``mixed``.
"""

from __future__ import annotations

import ast
import importlib
import os
import pathlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parents[2]
PKG = REPO / "python" / "discopt"
CRATES = REPO / "crates"

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
    for flag, sites in scan_rust().items():
        by_flag[flag].extend(sites)
    return dict(by_flag)


# ---------------------------------------------------------------- the Rust scan
#
# There is no Rust parser in the test environment, so this is a lexer plus a fixed
# set of consumer shapes. That is only honest because of rule 3: a read whose
# consumer matches none of the shapes is `unknown`, and the test fails on an
# unaccounted `unknown`. A new spelling therefore breaks the test instead of
# vanishing from it.
#
# The shapes, each with the value the expression takes when the variable is UNSET
# (which is the default the audit asks about):
#
#   var(F)...map(..).unwrap_or(true|false)      -> that literal
#   var(F)...unwrap_or(<number>)                -> numeric knob ("selector")
#   var(F).is_ok_and(..) / .is_ok() / var_os(F).is_some()  -> false
#   var_os(F).is_none() / var(F).is_err()       -> true
#   match var(F) { Err(_) => true|false, .. }   -> that literal
#   let raw = var(F).unwrap_or_default(); parse_x(&raw)
#                                               -> the arm of parse_x that holds ""
#
# A leading `!` flips the first three. Polarity is the value of the read's own
# expression, which every accessor in the tree names after the feature
# (`*_enabled`, `cse_enabled`) -- the same convention `_call_accessor` relies on
# for Python.

_RUST_READ = re.compile(r"(?:\bstd\s*::\s*)?\benv\s*::\s*var(_os)?\s*\(")
_RUST_LIT = re.compile(r'\s*"([^"\\]*)"\s*\)')
_RUST_CONST = re.compile(
    r'\bconst\s+([A-Z0-9_]+)\s*:\s*&(?:\'static\s+)?str\s*=\s*"(DISCOPT_[A-Z0-9_]+)"'
)
_RUST_RAW_OPEN = re.compile(r'r(#*)"')
_RUST_CHAR = re.compile(r"'(\\.|[^\\'])'")
_RUST_STR = re.compile(r'"([^"\\]*)"')


def _rust_strip(src: str) -> str:
    """Blank comments and test-only code, keeping every offset and line number.

    Comments must go: doc comments quote `env::var("DISCOPT_…")` spellings, and a
    docstring read as code is the "measured the docstring" failure. Test modules must
    go too: they save and restore variables (`let prior = env::var_os(F)`), which is
    not a gate and has no default.
    """
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out[i:j] = " " * (j - i)
            i = j
        elif c == "/" and src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out[i:j] = [ch if ch == "\n" else " " for ch in src[i:j]]
            i = j
        elif (
            c == "r"
            and (raw := _RUST_RAW_OPEN.match(src, i))
            and not (i and (src[i - 1].isalnum() or src[i - 1] == "_"))
        ):
            hashes = raw.group(1)
            end = src.find('"' + hashes, raw.end())
            i = n if end < 0 else end + 1 + len(hashes)
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            i = j + 1
        elif c == "'" and (ch_lit := _RUST_CHAR.match(src, i)):
            i = ch_lit.end()
        else:
            i += 1
    text = "".join(out)
    # `#[cfg(test)]` items: blank from the attribute to the end of the item it marks.
    for m in reversed(list(re.finditer(r"#\[cfg\(test\)\]", text))):
        semi = text.find(";", m.end())
        brace = text.find("{", m.end())
        if brace < 0 or (0 <= semi < brace):
            end = semi + 1
        else:
            end = _rust_block_end(text, brace)
        text = (
            text[: m.start()]
            + "".join(ch if ch == "\n" else " " for ch in text[m.start() : end])
            + text[end:]
        )
    return text


def _rust_block_end(text: str, open_brace: int) -> int:
    """Offset one past the `}` matching the `{` at ``open_brace``."""
    depth = 0
    for k in range(open_brace, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return k + 1
    raise AssertionError(f"unbalanced braces after offset {open_brace}")


def _rust_expr_tail(text: str, start: int) -> str:
    """The consuming expression from ``start``: up to a `;`/`,` at depth 0, or the
    bracket that closes the enclosing one. A `match` block is inside the tail."""
    depth = 0
    for k in range(start, len(text)):
        ch = text[k]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return text[start:k]
        elif ch in ";," and depth == 0:
            return text[start:k]
    return text[start:]


def _rust_words(pattern: str) -> frozenset[str]:
    return frozenset(w.lower() for w in _RUST_STR.findall(pattern))


def _rust_vocab(words: frozenset[str]) -> str | None:
    core = words - {""}
    if core and core <= ON_WORDS:
        return "on"
    if core and core <= OFF_WORDS:
        return "off"
    return None


def _rust_parse_table(text: str, fname: str) -> str:
    """Polarity of ``fname(raw)`` at ``raw == ""``, read from its own match arms."""
    m = re.search(rf"\bfn\s+{re.escape(fname)}\s*\(", text)
    if not m:
        return f"unknown:parse-fn-not-in-file:{fname}"
    sig_end = text.find("{", m.end())
    body = text[sig_end : _rust_block_end(text, sig_end)]
    for arm in re.finditer(r"((?:\"[^\"]*\"\s*\|\s*)*\"[^\"]*\")\s*=>", body):
        words = _rust_words(arm.group(1))
        if "" in words:
            pol = _rust_vocab(words)
            return pol if pol else "unknown:parse-mixed-word-set"
    # No arm names "": the empty string takes an early `return` instead. A numeric
    # return type makes it a budget or count; anything else is undecided.
    ret = text[m.end() : sig_end]
    if re.search(r"is_empty\(\)", body) and re.search(
        r"Result<\s*(usize|u\d+|i\d+|f64|f32)\b", ret
    ):
        return "selector"
    return "unknown:parse-no-empty-arm"


def _rust_polarity(text: str, start: int, end: int, os_form: bool) -> tuple[str, str]:
    """(form, polarity) for the read spanning ``text[start:end]``."""
    tail = _rust_expr_tail(text, end)
    flat = re.sub(r"\s+", "", tail)
    before = text[:start].rstrip()
    negated = before.endswith("!") and not before.endswith("!=")

    def flip(p: str) -> str:
        return {"on": "off", "off": "on"}.get(p, p) if negated else p

    # Presence tests: the value when unset is fixed by the method itself.
    for meth, unset in (
        (".is_ok_and(", "off"),
        (".is_ok()", "off"),
        (".is_some()", "off"),
        (".is_none()", "on"),
        (".is_err()", "on"),
    ):
        if flat.startswith(meth):
            return "rust_presence", flip(unset)
    # `match env::var(F) { Err(_) => <lit>, Ok(v) => … }`
    if flat.startswith("{") and before.endswith("match"):
        m = re.search(r"Err\(_\)=>(true|false)\b", flat)
        if m:
            return "rust_match", "on" if m.group(1) == "true" else "off"
        return "rust_match", "unknown:match-err-arm-not-literal"
    # `.map(..).unwrap_or(<lit>)` chains.
    m = re.search(r"\.unwrap_or\(([^()]*)\)$", flat) or re.search(r"\.unwrap_or\(([^()]*)\)", flat)
    if m and not flat.startswith(".unwrap_or_default()"):
        arg = m.group(1)
        if arg in ("true", "false"):
            return "rust_unwrap_or", flip("on" if arg == "true" else "off")
        if re.fullmatch(r"-?[0-9][0-9_]*(\.[0-9_]*)?([eE][-+]?[0-9]+)?(usize|f64|u\d+|i\d+)?", arg):
            return "rust_unwrap_or", "selector"
        return "rust_unwrap_or", f"unknown:unwrap_or-arg:{arg}"
    # `let raw = env::var(F).unwrap_or_default(); parse_x(&raw)` (maybe `.unwrap_or_else(panic)`).
    if flat.startswith(".unwrap_or_default()"):
        bind = re.search(r"let\s+(\w+)\s*=\s*$", before)
        if not bind:
            return "rust_parse_table", "unknown:unwrap_or_default-not-bound"
        name = bind.group(1)
        call = re.search(rf"\b(\w+)\s*\(\s*&?\s*{name}\b", text[end : end + 400])
        if not call:
            return "rust_parse_table", "unknown:binding-not-parsed"
        return "rust_parse_table", _rust_parse_table(text, call.group(1))
    return "rust_unrecognised", "unknown:rust-consumer-shape"


def scan_rust() -> dict[str, list[Site]]:
    """Every `DISCOPT_*` read under ``crates/*/src`` (test modules excluded)."""
    by_flag: dict[str, list[Site]] = defaultdict(list)
    files = sorted(CRATES.glob("*/src/**/*.rs"))
    n_reads = 0
    for path in files:
        text = _rust_strip(path.read_text())
        consts = dict(_RUST_CONST.findall(text))
        rel = str(path.relative_to(REPO))
        for m in _RUST_READ.finditer(text):
            n_reads += 1
            lit = _RUST_LIT.match(text, m.end())
            if lit:
                flag = lit.group(1)
                end = lit.end()
            else:
                ident = re.match(r"\s*([A-Z0-9_]+)\s*\)", text[m.end() :])
                if not ident or ident.group(1) not in consts:
                    # A non-literal, non-constant argument could name any flag.
                    # Crash rather than skip: a skipped read is the #1421 blind spot.
                    line = text.count("\n", 0, m.start()) + 1
                    raise AssertionError(
                        f"{rel}:{line}: env::var with an argument the scan cannot "
                        "resolve to a name. Teach flag_audit_scan.py this form."
                    )
                flag = consts[ident.group(1)]
                end = m.end() + ident.end()
            if not flag.startswith("DISCOPT_"):
                continue
            form, pol = _rust_polarity(text, m.start(), end, bool(m.group(1)))
            line = text.count("\n", 0, m.start()) + 1
            by_flag[flag].append(Site(flag, form, pol, rel, line, "-"))

    assert len(files) > 50, f"only {len(files)} Rust files under crates/ -- wrong root?"
    assert n_reads > 0, "no env::var reads in crates/ -- the Rust scan stopped matching"
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

    n_rust = sum(1 for v in by_flag.values() for s in v if s.path.startswith("crates/"))
    print(f"distinct DISCOPT_* flags : {len(by_flag)}")
    print(f"read sites classified    : {n_sites}")
    print(f"  of which Rust (crates/): {n_rust}")
    for kind in ("off", "on", "selector", "mixed", "unknown"):
        print(f"  {kind:10s} {len(buckets[kind]):4d}")
    for kind in ("off", "unknown", "mixed"):
        print(f"\n-- {kind} --")
        for flag in buckets[kind]:
            s = by_flag[flag][0]
            print(f"  {flag:42s} {s.polarity:34s} {s.form:12s} {s.where}")
    print(f"\nEXECUTED CLASSIFICATIONS: {n_sites} ({n_rust} Rust)")
    if n_sites == 0 or n_rust == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
