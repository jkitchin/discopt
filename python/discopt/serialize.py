"""Native model serialization: save a model (and its fitted state) and load it back.

Motivation
----------
Before this module the only way to persist a model was to export it to ``.nl`` or
``.gms``. Both lose information that matters when the point is to *come back to the
same model tomorrow*: ``.nl`` renames every variable positionally (``x0..xn-1``),
neither carries the fitted solution, and both refuse the non-algebraic relations
(indicator / SOS / disjunction) outright. This module is the round-trippable native
format.

Format
------
A single JSON document, optionally gzipped (:func:`save` compresses when the path
ends in ``.gz``; :func:`load` sniffs the gzip magic bytes, so it reads either).
JSON was chosen over ``.npz``/pickle deliberately: it is readable by any language,
carries no numpy format version and no Python version, and is greppable and
diffable in git. Float text compresses well, so gzip covers the size concern
without a binary container.

The expression DAG is written as a **flat node table**, not a nested tree.
Expressions share subnodes heavily -- a collocation model with one surrogate reuses
the same subexpression at every collocation point -- and nesting would re-expand
every share. The table's array positions *are* the node ids, so there is no id
bookkeeping to fall out of sync, and sharing round-trips exactly.

Scope
-----
Carried: variables (name, type, shape, element-wise bounds), parameters, the
objective, algebraic constraints, the fast-construction (Rust builder) linear
blocks and builder objective, indicator / SOS / disjunctive relations (with an
``either_or`` disjunction's indicator columns once they were read, from 1.5),
propositional logic constraints (``Model.logical``, from 1.5), complementarity
relations together with this model's lowering marks and the rows each lowering
emitted (from 1.5; earlier documents reload them as untracked), the starting
point (and a GAMS model's declared ``x.l`` start), and -- optionally -- the solve
result. A ``from_nl`` model also carries the fact that it was parsed from a file
(so a certificate guard stays as conservative as on the original) and its source
path together with a SHA-256 of the file's bytes; the solver reads that file only
while it still hashes to the digest, so a document opened on another machine, or
after the file changed, can never hand it a different problem.

Attributes discopt attaches to a model after ``__init__`` are each classified
(carried, deliberately not carried, or refused); per-solve stashes -- budgets, a
``perf_counter`` deadline, the Gauss-Newton switch, a root bound -- are not
carried, exactly as the last solve result is not.

Also carried, as descriptive metadata rather than model content: a ``provenance``
block recording what wrote the document and when (:mod:`discopt.provenance`). It
comes back on ``Model.provenance``, and a version difference between the writer
and the reader raises a :class:`ProvenanceSkewWarning`. Note that the *reason* the
version needs to travel is the paragraph above about JSON: this format carries no
numpy or Python format version precisely so that the document does not depend on
them, which leaves discopt's own version the only thing that fixes the meaning of
what is written, and it is worth nothing if it is never checked.

Index expressions carry integers, slices and tuples of them; a fancy (array/list)
index, ``None`` or ``Ellipsis`` is refused rather than approximated.

Refused loudly (never silently dropped, per the repo's no-silent-approximation
rule), and **permanently** -- these are not formats waiting for a section:

* :class:`~discopt.modeling.core.CustomCall`: it wraps an arbitrary Python
  callable. Writing its *name* would reload into a different model, and writing
  its code would make ``loads`` execute it, which a reader of untrusted input must
  never do. Rewrite it with ``dm.*`` primitives or a registered function.
* the output of a reformulation pass that carries records pointing back at its
  source model (``binary.multilinear``, ``integer.*``, ``nonsmooth.epigraph``): a
  document cannot hold the source model. Lossless remedy: save the source and
  re-apply the (deterministic) transformation after loading.

Also refused: any other private model attribute this version does not classify,
and any expression node this version does not know. On read,
an unknown ``op`` is a hard error -- a reader that skips what it does not
understand silently drops a constraint and returns a model that solves to the
wrong answer.

Reading is strict for the same reason. Every reference into the node table, the
variables, the parameters or the rows must be an in-range integer (a negative
one or a ``true`` used to resolve, through Python indexing, to a different
object), an index must fit its base's shape, and senses, SOS types, indicator
activation values and disjunction methods must be ones the modeling API can
produce -- an ``active_value`` of 7 once dropped its row and certified a false
optimum. A malformed field these checks do not name raises
:class:`SerializationError` too, chained to the low-level error. An unknown
top-level key is refused too: see *Versioning* below.

Versioning
----------
``SCHEMA`` is ``discopt.model/<major>.<minor>``. A reader refuses any other major.
Every key a document may carry at its top level and in its ``state`` section is
listed, with the minor that first wrote it, in ``_TOP_LEVEL_KEYS`` and
``_STATE_KEYS``, and a reader acts on a document's minor as follows:

* **minor at or below the reader's, only listed keys** -- loaded strictly;
* **minor at or below the reader's, an unlisted key** -- refused: no writer of
  that minor wrote it, so the document is malformed or was edited by hand, and
  dropping it could drop model content;
* **newer minor, only listed keys** -- loaded strictly, with a
  :class:`SchemaVersionWarning`: everything the reader understood was decoded, but
  a newer writer may say more than it knows how to check;
* **newer minor, an unlisted key, or a node, row or index kind the reader does not
  know** -- refused, with a message saying to upgrade discopt.

A top-level key starting with ``x-`` is never read and never refused: it is the
one place a tool may annotate a document without forking the format. It does not
survive a load and re-save -- the reader does not keep what it does not read --
but ``discopt upgrade-model`` carries every top-level ``x-`` key through verbatim.

The same policy holds inside every nested record (#1780) -- a variable, node,
row, builder block, relation, the embedded result -- and a key repeated within
one JSON object is refused too (``json`` would keep the last). Node records are
checked against ``_NODE_KEYS`` (per ``op``) and ``_INDEX_KEYS``; everything else
by recording which keys the decoder reads and refusing the rest
(:class:`_Record`), so no hand-kept list can fall behind the decoder. The ``x-``
escape is top-level only: an ``x-`` key inside a record is refused like any other.
A nested key a new minor adds is read only from documents of that minor or later
(``indicators`` on a disjunction row, 1.5), so in an older document it is
refused like any other key no writer of that minor produced. The same holds for
the other two 1.5 additions, a ``"logical"`` row and a three-element ``lowered``
entry: in a document of minor < 5 each is refused as malformed.

``discopt upgrade-model`` rewrites a document at the current ``SCHEMA``. Through
1.1 an unknown top-level key was ignored; that is how ``provenance`` (1.1) stayed
readable by a 1.0 reader, and the reason no newer document can rely on an older
reader to ignore anything.

Examples
--------
>>> import discopt.modeling as dm
>>> m = dm.Model("kinetics")
>>> k = m.continuous("k", shape=(2,), lb=0.0, ub=10.0)
>>> m.minimize(dm.sum((k - 1.0) ** 2))
>>> result = m.solve()                                  # doctest: +SKIP
>>> m.save("kinetics.dopt", result=result)              # doctest: +SKIP
>>> m2 = dm.load("kinetics.dopt")                       # doctest: +SKIP
>>> m2.saved_result.objective                           # doctest: +SKIP
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    BooleanVar,
    Constant,
    Constraint,
    CustomCall,
    Expression,
    FunctionCall,
    IndexExpression,
    LogicalAnd,
    LogicalAtLeast,
    LogicalAtMost,
    LogicalEquivalent,
    LogicalExactly,
    LogicalImplies,
    LogicalNot,
    LogicalOr,
    MatMulExpression,
    Model,
    Objective,
    ObjectiveSense,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
    _DisjunctiveConstraint,
    _IndicatorConstraint,
    _LogicalConstraint,
    _readonly_bound,
    _SOSConstraint,
)
from discopt.provenance import capture as _capture_provenance
from discopt.provenance import skew_warning

logger = logging.getLogger(__name__)

#: Format identifier written into every document. The minor bumped to 1.1 when
#: the provenance block was added: additive, so a 1.0 reader still reads a 1.1
#: document (it ignores the extra key) and this reader still reads a 1.0 one
#: (it records no provenance).
#:
#: 1.2 is NOT additive, and is the first minor that this reader acts on. Two
#: things changed at once: an index gained its own ``{"k": "bool"}`` kind (#1290)
#: and a sum axis gained a list spelling, neither of which a 1.1 reader
#: understands; and the solution subtree stopped going through a blanket
#: ``_enc_tree`` (#1292), which a 1.1 reader would leave tagged. The minor is
#: what tells the two solution encodings apart on read -- see :func:`loads`.
#:
#: A 1.1 reader loading a 1.2 document still passes the major check and then
#: fails on the first bool index or list axis with "unknown index kind" /
#: "invalid sum axis". That is a clean refusal rather than a mis-build, and it is
#: the best that can be done for readers already shipped: they never looked at
#: the minor. Forward compatibility is not claimed.
#:
#: 1.3 added the registered-atom tag (#1248 A). 1.4 adds three "state" keys for
#: attributes attached after ``Model.__init__``: ``nl_source`` (a ``from_nl``
#: model's parse marker and digest-checked source path), ``_gams_initial_values``
#: and ``_recentre_inner``. All additive -- an older reader ignores them -- but,
#: as with 1.3, absence has two meanings that only the minor separates: "not read
#: from a file" (1.4) versus "the writer did not record it" (<= 1.3).
#:
#: 1.5 (#1785 serialize WP-3 together with #1775 G1.1) adds three things, none
#: of them additive for a 1.4 reader:
#:
#: * propositional logic constraints -- a ``"logical"`` row with a ``logic``
#:   table;
#: * the rows each complementarity lowering emitted -- a third element of a
#:   ``lowered`` entry, written only when the rows are tracked;
#: * ``indicators`` on a ``disjunction`` row: the node references of the binary
#:   columns that ``_DisjunctiveConstraint.indicators`` allocated, written only
#:   once they were read. A disjunction whose indicators were never read is
#:   written exactly as at 1.4.
#:
#: All three are read only from a document of minor >= 5
#: (``_SCHEMA_MINOR_LOGIC_LOWERING_INDICATORS``); in an older document each is
#: refused as content no writer of that minor produced. A 1.4 reader refuses
#: the unknown row kind and the three-element entry, and (with #1783's
#: nested-key strictness) the ``indicators`` key as newer content ("upgrade
#: discopt") rather than dropping it -- which would leave the columns in the
#: model with nothing tying them to the disjunction. Readers shipped before
#: #1783 drop ``indicators`` silently; see the fixture manifest's forward cells.
SCHEMA = "discopt.model/1.5"

#: Major version this reader accepts. A document whose major differs is refused
#: rather than best-effort parsed.
SCHEMA_MAJOR = 1

#: Minor at which the blanket ``_enc_tree`` over the solution subtree was dropped
#: (#1292). Documents below it need that encoding undone on read.
_SCHEMA_MINOR_UNTAGGED_SOLUTION = 2

#: Minor that first wrote the three 1.5 additions: a ``"logical"`` row, the
#: tracked rows of a complementarity lowering (a three-element ``lowered`` entry)
#: and ``indicators`` on a disjunction row (#1775 G1.1). Each is read only from a
#: document of this minor or later.
_SCHEMA_MINOR_LOGIC_LOWERING_INDICATORS = 5

#: This reader's minor, the one ``SCHEMA`` names.
SCHEMA_MINOR = int(SCHEMA.rsplit(".", 1)[1])

#: Every top-level key a writer has produced, with the minor that first wrote it.
#: A key absent here is refused (see *Versioning* in the module docstring); a key
#: a new minor adds is added here in the PR that bumps ``SCHEMA``.
_TOP_LEVEL_KEYS: dict[str, int] = {
    "schema": 0,
    "discopt": 0,
    "name": 0,
    "variables": 0,
    "parameters": 0,
    "nodes": 0,
    "objective": 0,
    "builder_objective": 0,
    "rows": 0,
    "builder_blocks": 0,
    "complementarities": 0,
    "state": 0,
    "initial_point": 0,
    "solution": 0,
    "provenance": 1,
}

#: Every key a writer has produced in the "state" section, likewise. Derived from
#: every revision of this module, not only the last writer of each minor: a key
#: written in between and later dropped would still be on disk.
_STATE_KEYS: dict[str, int] = {
    "_aux_counter": 0,
    "_decomp_stages": 0,
    "_decomp_blocks": 0,
    "_zero_spanning_factor_auxes": 0,
    "_coupling_keys": 0,
    "_sets": 0,
    "_simplex_lowerings": 0,
    "_atan2_preconditions": 2,
    "_block_labels_var": 2,
    "_block_labels_con": 2,
    "_implied_integer_auxes": 3,
    "_free_by_construction": 3,
    "_piecewise_domains": 3,
    "_recentre_inner": 4,
    "_gams_initial_values": 4,
    "nl_source": 4,
}

#: A top-level key with this prefix is an annotation: never read, never refused.
_EXTENSION_PREFIX = "x-"

#: ``{key: (deprecated_since_minor, replacement)}``. Nothing is deprecated.
_DEPRECATED_KEYS: dict[str, tuple[int, str]] = {}

_GZIP_MAGIC = b"\x1f\x8b"

__all__ = [
    "SCHEMA",
    "dumps",
    "loads",
    "save",
    "load",
    "SerializationError",
    "SchemaVersionWarning",
    "ProvenanceSkewWarning",
]


class SerializationError(ValueError):
    """Raised when a model cannot be written faithfully, or a document cannot be read."""


class SchemaVersionWarning(UserWarning):
    """A document was written by a newer minor of the format than this reader's.

    It loaded -- every key it carries is one this reader knows -- but a newer writer
    can say things inside those keys that this reader cannot check. ``warnings.
    simplefilter("error", SchemaVersionWarning)`` turns it into a refusal, which is
    what ``discopt upgrade-model`` does: rewriting such a document would downgrade it.
    """


class ProvenanceSkewWarning(UserWarning):
    """A document was written by a different discopt than the one reading it.

    Its own category (rather than a bare ``UserWarning``) so a caller can silence
    or escalate exactly this -- ``warnings.simplefilter("error",
    ProvenanceSkewWarning)`` makes a version mismatch fatal for a pipeline that
    requires an exact-version reload.
    """


# ── scalars ────────────────────────────────────────────────────────────────
#
# JSON has no NaN/Infinity (Python's json emits bare `NaN`/`Infinity` by default,
# which is non-standard and other languages' parsers reject), so non-finite values
# are written as tagged strings. Everything else is written as a JSON number --
# `repr` is the shortest string that round-trips to the same float64, which is why
# `_enc_float` can assert exactness rather than hope for it.
#
# NOTE the `1e20` INF sentinel of the Rust LP layer is an ordinary finite number
# here and is preserved literally. Normalising it to infinity on write (or back on
# read) would change the model.


def _enc_float(x: Any) -> Union[float, str]:
    """Encode one float. Non-finite values become tagged strings."""
    v = float(x)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    # The round-trip guarantee this format rests on, asserted rather than assumed.
    if float(repr(v)) != v:  # pragma: no cover - defensive; repr round-trips on CPython
        raise SerializationError(f"float {v!r} does not round-trip through its repr")
    return v


def _dec_float(x: Any) -> float:
    if isinstance(x, str):
        if x == "nan":
            return math.nan
        if x == "inf":
            return math.inf
        if x == "-inf":
            return -math.inf
        raise SerializationError(f"unknown float token {x!r}")
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        # JSON `true` is a Python bool, which `float()` reads as 1.0.
        raise SerializationError(f"expected a number, got {type(x).__name__} ({x!r})")
    return float(x)


def _dec_index_int(x: Any, what: str) -> int:
    """An integer the document carries, refused unless it is one.

    JSON ``true`` is a Python ``bool`` -- an ``int`` subclass -- and ``1.0`` or
    ``"1"`` would coerce, so a bare ``int(...)`` accepts all three as something
    else. Every reference and index goes through here instead.
    """
    if type(x) is not int:
        raise SerializationError(f"{what} must be an integer, got {type(x).__name__} ({x!r})")
    return x


class _StrictRefs:
    """A list the document refers into, by position, with every reference checked.

    A bare ``items[d["ref"]]`` resolves a negative reference to an element counted
    from the END and a ``true`` to element 1, so a corrupt document silently
    rebuilt rows on the wrong variable or node, and an out-of-range one raised a
    bare ``IndexError`` that does not name the document. For the node table the
    list grows as nodes are decoded, so a forward reference -- and with it any
    cycle -- is out of range by construction.
    """

    def __init__(self, items: list, what: str) -> None:
        self._items = items
        self._what = what

    def __getitem__(self, ref: Any) -> Any:
        i = _dec_index_int(ref, f"a {self._what} reference")
        if not 0 <= i < len(self._items):
            raise SerializationError(
                f"{self._what} reference {i} is out of range (the document has "
                f"{len(self._items)} so far)"
            )
        return self._items[i]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)


#: Senses ``Model.subject_to`` produces; anything else used to load and fail only
#: inside a solve ("Unknown constraint sense").
_KNOWN_SENSES = frozenset({"<=", "==", ">="})

#: Per-disjunction methods the GDP lowering dispatches on. An unknown one fell
#: through to big-M without a word.
_KNOWN_DISJUNCTION_METHODS = frozenset({None, "hull", "big-m", "mbigm", "simplex"})


def _enc_array(arr: np.ndarray) -> Any:
    """Encode an ndarray as nested lists of encoded floats (shape carried separately)."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 0:
        return _enc_float(a)
    return [_enc_array(sub) for sub in a]


def _dec_array(obj: Any, shape: tuple[int, ...]) -> np.ndarray:
    flat = []

    def walk(o):
        if isinstance(o, list):
            for item in o:
                walk(item)
        else:
            flat.append(_dec_float(o))

    walk(obj)
    return np.array(flat, dtype=np.float64).reshape(shape)


def _enc_index(index: Any) -> Any:
    """Encode an ``IndexExpression`` index (int, bool, tuple, slice, or a mix)."""
    # before the int test: ``bool`` subclasses ``int``, and numpy reads ``x[True]``
    # as a new axis, not ``x[1]`` (#1290)
    if isinstance(index, (bool, np.bool_)):
        return {"k": "bool", "v": bool(index)}
    if isinstance(index, (int, np.integer)):
        return {"k": "int", "v": int(index)}
    if isinstance(index, slice):
        return {
            "k": "slice",
            "start": None if index.start is None else int(index.start),
            "stop": None if index.stop is None else int(index.stop),
            "step": None if index.step is None else int(index.step),
        }
    if isinstance(index, tuple):
        return {"k": "tuple", "v": [_enc_index(i) for i in index]}
    raise SerializationError(
        f"cannot serialize index {index!r} of type {type(index).__name__}; "
        "only integers, slices, and tuples of them are supported."
    )


def _dec_index(obj: Any) -> Any:
    kind = obj["k"]
    extra = dict.keys(obj) - _INDEX_KEYS.get(kind, dict.keys(obj))
    if extra:
        raise SerializationError(f"{kind} index has unknown key(s) {sorted(extra)}")
    if kind == "int":
        return _dec_index_int(obj["v"], "an integer index")
    if kind == "bool":
        if not isinstance(obj["v"], bool):
            raise SerializationError(f"bool index holds {obj['v']!r}")
        return obj["v"]
    if kind == "slice":
        parts = [
            None if obj[k] is None else _dec_index_int(obj[k], f"a slice {k}")
            for k in ("start", "stop", "step")
        ]
        if parts[2] == 0:
            raise SerializationError("a slice step cannot be zero")
        return slice(*parts)
    if kind == "tuple":
        return tuple(_dec_index(i) for i in obj["v"])
    raise SerializationError(f"unknown index kind {kind!r}")


def _enc_axis(axis: Any) -> Any:
    """Encode a ``SumExpression`` axis: ``None``, an int, or a tuple of ints (#1290)."""
    if axis is None:
        return None
    if isinstance(axis, (int, np.integer)) and not isinstance(axis, (bool, np.bool_)):
        return int(axis)
    if isinstance(axis, tuple) and all(
        isinstance(a, (int, np.integer)) and not isinstance(a, (bool, np.bool_)) for a in axis
    ):
        return [int(a) for a in axis]
    raise SerializationError(
        f"cannot serialize sum axis {axis!r} of type {type(axis).__name__}; "
        "only None, an integer, or a tuple of integers is supported."
    )


def _dec_axis(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, int) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, list) and all(isinstance(a, int) and not isinstance(a, bool) for a in obj):
        return tuple(obj)
    raise SerializationError(f"invalid sum axis {obj!r}")


# ── expression DAG ─────────────────────────────────────────────────────────


class _NodeTable:
    """Builds the flat node table, preserving DAG sharing by object identity.

    Keyed on ``id(node)`` -- NOT on equality, because ``Expression.__eq__`` builds a
    ``Constraint`` rather than comparing. ``_alive`` holds a strong reference to
    every visited node so CPython cannot recycle an id mid-walk and alias two
    distinct nodes onto one table entry.
    """

    def __init__(self, var_ids: dict[int, int], param_ids: dict[int, int]):
        self.nodes: list[dict] = []
        self._seen: dict[int, int] = {}
        self._alive: list[Any] = []
        self._var_ids = var_ids
        self._param_ids = param_ids

    def add(self, expr: Expression) -> int:
        """Return the table id for *expr*, encoding it (and its children) if new.

        Iterative, not recursive: a long chain of binary ops (a sum built by
        repeated ``+``) is thousands of nodes deep and would blow the recursion
        limit on a plain post-order walk.
        """
        stack: list[tuple[Any, bool]] = [(expr, False)]
        while stack:
            node, children_done = stack.pop()
            key = id(node)
            if key in self._seen:
                continue
            if not children_done:
                kids = _children(node)
                if kids:
                    stack.append((node, True))
                    # Reversed so children are encoded left-to-right, giving a
                    # deterministic table order for a given model.
                    for kid in reversed(kids):
                        if id(kid) not in self._seen:
                            stack.append((kid, False))
                    continue
            self._alive.append(node)
            self._seen[key] = len(self.nodes)
            enc = self._encode(node)
            # Placed BEFORE tagging: `_tag_atom` may add the atom's argument to
            # the table, and the index reserved above is only correct if this
            # node is already at it.
            self.nodes.append(enc)
            self._tag_atom(node, enc)
        return self._seen[id(expr)]

    def _tag_atom(self, node: Any, enc: dict) -> None:
        """Record that *node* is the lowering of a registered atom (#1248 A).

        The tag is a Python attribute (``operators.ATOM_ATTR``) on an ordinary
        expression object, so it does not survive a round-trip through this
        document unless it is written down.  Losing it is *sound* -- the reader
        gets the lowering and relaxes it term by term -- but it is silently
        looser, and the path that loses it is exactly ``solve_batch(workers>1)``,
        which ships every model through ``dumps``/``loads``.  A registered
        composite therefore lost its envelope in every worker process, at no
        error and with nothing to see but a slower solve.

        What is written is the atom's NAME and a reference to its ARGUMENT.  The
        body is this node itself, so it needs no reference; the argument does,
        because a lowering may bury it arbitrarily deep (and, for a lowering
        that ignores it, may not contain it at all -- hence ``add`` rather than a
        lookup).  ``atom_of`` returns ``None`` for a tag whose registration has
        since been dropped or replaced, so a stale tag is never written.
        """
        from discopt.operators import atom_of

        tag = atom_of(node)
        if tag is None:
            return
        name, arg = tag  # `atom_of` returns (name, arg); the registration stays behind
        enc["atom"] = {"n": name, "arg": self.add(arg)}

    def _ref(self, node: Any) -> int:
        idx = self._seen.get(id(node))
        if idx is None:  # pragma: no cover - post-order guarantees children first
            raise SerializationError(f"child {node!r} encoded before its parent")
        return idx

    def _encode(self, node: Any) -> dict:
        t = type(node)
        if t is Constant:
            val = node.value
            if val.ndim == 0:
                return {"op": "const", "v": _enc_float(val)}
            return {"op": "const", "v": _enc_array(val), "shape": list(val.shape)}
        if t is Variable:
            ref = self._var_ids.get(id(node))
            if ref is None:
                raise SerializationError(
                    f"expression references variable {node.name!r}, which is not "
                    "registered on this model; cannot serialize."
                )
            return {"op": "var", "ref": ref}
        if t is Parameter:
            ref = self._param_ids.get(id(node))
            if ref is None:
                raise SerializationError(
                    f"expression references parameter {node.name!r}, which is not "
                    "registered on this model; cannot serialize."
                )
            return {"op": "param", "ref": ref}
        if t is IndexExpression:
            return {"op": "index", "base": self._ref(node.base), "index": _enc_index(node.index)}
        if t is BinaryOp:
            return {
                "op": "binop",
                "o": node.op,
                "a": self._ref(node.left),
                "b": self._ref(node.right),
            }
        if t is UnaryOp:
            return {"op": "unop", "o": node.op, "a": self._ref(node.operand)}
        if t is FunctionCall:
            return {
                "op": "call",
                "f": node.func_name,
                "args": [self._ref(a) for a in node.args],
            }
        if t is MatMulExpression:
            return {"op": "matmul", "a": self._ref(node.left), "b": self._ref(node.right)}
        if t is SumExpression:
            return {
                "op": "sum",
                "a": self._ref(node.operand),
                "axis": _enc_axis(node.axis),
            }
        if t is SumOverExpression:
            return {"op": "sum_over", "args": [self._ref(a) for a in node.terms]}
        if t is CustomCall:
            # Permanent (plan §3.6): the only faithful encoding of a callable is its
            # code, and `loads` must never execute what it reads.
            raise SerializationError(
                f"cannot serialize the custom function {node.name!r}: a CustomCall wraps "
                "an arbitrary Python callable, so there is nothing faithful to write "
                "(writing its name would reload into a different model). Replace it with "
                "algebraic expressions, or keep the callable in the code that rebuilds "
                "the model."
            )
        raise SerializationError(
            f"cannot serialize expression node of type {t.__name__}; this format "
            f"version knows {sorted(_KNOWN_OPS)}."
        )


def _children(node: Any) -> list[Any]:
    """Direct expression children of *node*, in encode order."""
    t = type(node)
    if t is IndexExpression:
        return [node.base]
    if t is BinaryOp:
        return [node.left, node.right]
    if t is UnaryOp:
        return [node.operand]
    if t is FunctionCall:
        return list(node.args)
    if t is MatMulExpression:
        return [node.left, node.right]
    if t is SumExpression:
        return [node.operand]
    if t is SumOverExpression:
        return list(node.terms)
    if t is CustomCall:
        return list(node.args)
    return []


_KNOWN_OPS = frozenset(
    {"const", "var", "param", "index", "binop", "unop", "call", "matmul", "sum", "sum_over"}
)


#: Non-element-wise function names a ``FunctionCall`` can legally carry. The
#: element-wise ones come from ``core._ELEMENTWISE_FUNCS``; these are the rest of
#: what the modeling layer produces (reductions, two-argument and entropy forms).
#: ``test_every_produced_operator_name_is_accepted`` scans the package for every
#: ``FunctionCall`` / ``BinaryOp`` / ``UnaryOp`` construction with a literal name and
#: fails if any produced name is missing here -- a name omitted from these sets
#: would make `loads` refuse a perfectly valid document. (Spelled without the call
#: parentheses on purpose: that scan would otherwise match this very comment.)
_EXTRA_FUNCS = frozenset({"atan2", "centropy", "entropy", "prod", "signpower", "sum"})

_KNOWN_BINARY_OPS = frozenset({"+", "-", "*", "/", "**"})
_KNOWN_UNARY_OPS = frozenset({"neg", "abs"})

#: Keys of a node record, per ``op``; any node may also carry an ``atom`` tag
#: (#1248 A, from 1.3). A node key outside its op's set is refused (#1780). Checked
#: against what ``_decode_nodes`` actually reads by
#: ``test_serialize_versioning::test_node_key_tables_are_what_the_decoder_reads``.
_NODE_KEYS: dict[str, frozenset] = {
    op: frozenset(keys) | {"op", "atom"}
    for op, keys in {
        "const": {"v", "shape"},
        "var": {"ref"},
        "param": {"ref"},
        "index": {"base", "index"},
        "binop": {"o", "a", "b"},
        "unop": {"o", "a"},
        "call": {"f", "args"},
        "matmul": {"a", "b"},
        "sum": {"a", "axis"},
        "sum_over": {"args"},
    }.items()
}

#: Keys of an index object (a node's ``index``), per kind ``k``.
_INDEX_KEYS: dict[str, frozenset] = {
    "int": frozenset({"k", "v"}),
    "bool": frozenset({"k", "v"}),
    "slice": frozenset({"k", "start", "stop", "step"}),
    "tuple": frozenset({"k", "v"}),
}

#: Keys of a node's ``atom`` tag.
_ATOM_KEYS = frozenset({"n", "arg"})


def _known_funcs() -> frozenset:
    """Function names a ``FunctionCall`` node may carry.

    Built from the modeling layer's own set rather than restated here, so it cannot
    drift from what a node can legally hold. ``BinaryOp``/``UnaryOp``/``FunctionCall``
    do not validate their own ``op``/``func_name``, so without a read-time check a
    document naming an operator that does not exist loads without complaint and
    fails much later, during a solve -- where the first symptom is the
    incumbent-verification snapshot failing and the false-primal guard being
    disabled for that solve. That contradicts this module's own promise that an
    unknown op is refused on read.
    """
    from discopt.modeling import core as _core

    return frozenset(_core._ELEMENTWISE_FUNCS) | _EXTRA_FUNCS


def _decode_nodes(table: list[dict], variables: Any, params: list[Parameter]):
    """Rebuild expression objects from the node table.

    A node's children always precede it (the writer emits post-order), so a single
    forward pass suffices and sharing is restored by construction: two parents
    referencing id ``k`` get the *same* object back.
    """
    known_funcs = _known_funcs()
    built: list[Any] = []
    # Children are read through `refs`, which only reaches nodes already decoded.
    refs = _StrictRefs(built, "node")
    var_refs = _StrictRefs(variables, "variable")
    param_refs = _StrictRefs(params, "parameter")
    for i, nd in enumerate(table):
        op = nd.get("op")
        if op not in _KNOWN_OPS:
            raise SerializationError(
                f"node {i} has unknown op {op!r}. This file was written by a newer "
                "discopt than the one reading it; refusing rather than dropping the node."
            )
        # `dict.keys`, not `nd.keys()`: checking a key is not reading it, which
        # matters when the tests decode with tracked nodes to verify the tables.
        allowed = _NODE_KEYS[op]
        atom = dict.get(nd, "atom")
        if not dict.keys(nd) <= allowed or (
            isinstance(atom, dict) and not dict.keys(atom) <= _ATOM_KEYS
        ):
            extra = [f"/nodes[{i}]/{k}" for k in dict.keys(nd) - allowed]
            if isinstance(atom, dict):
                extra += [f"/nodes[{i}]/atom/{k}" for k in dict.keys(atom) - _ATOM_KEYS]
            raise SerializationError(
                f"{op} node has key(s) no writer of that format produces "
                f"({', '.join(sorted(extra))}); refusing rather than dropping them."
            )
        if op == "binop" and nd.get("o") not in _KNOWN_BINARY_OPS:
            raise SerializationError(
                f"node {i} names unknown binary operator {nd.get('o')!r}. Refusing on "
                "read rather than building a node that fails much later, during a solve."
            )
        if op == "unop" and nd.get("o") not in _KNOWN_UNARY_OPS:
            raise SerializationError(
                f"node {i} names unknown unary operator {nd.get('o')!r}. Refusing on "
                "read rather than building a node that fails much later, during a solve."
            )
        if op == "call" and nd.get("f") not in known_funcs:
            raise SerializationError(
                f"node {i} names unknown function {nd.get('f')!r}. Refusing on read "
                "rather than building a node that fails much later, during a solve."
            )
        if op == "const":
            shape = nd.get("shape")
            if shape is None:
                built.append(Constant(_dec_float(nd["v"])))
            else:
                dims = tuple(_dec_index_int(n, f"a dimension of constant node {i}") for n in shape)
                if any(n < 0 for n in dims):
                    raise SerializationError(f"constant node {i} has a negative dimension: {dims}")
                built.append(Constant(_dec_array(nd["v"], dims)))
        elif op == "var":
            built.append(var_refs[nd["ref"]])
        elif op == "param":
            built.append(param_refs[nd["ref"]])
        elif op == "index":
            base, index = refs[nd["base"]], _dec_index(nd["index"])
            _check_index_in_range(base, index, i)
            built.append(IndexExpression(base, index))
        elif op == "binop":
            built.append(BinaryOp(nd["o"], refs[nd["a"]], refs[nd["b"]]))
        elif op == "unop":
            built.append(UnaryOp(nd["o"], refs[nd["a"]]))
        elif op == "call":
            built.append(FunctionCall(nd["f"], *[refs[a] for a in nd["args"]]))
        elif op == "matmul":
            built.append(MatMulExpression(refs[nd["a"]], refs[nd["b"]]))
        elif op == "sum":
            base, axis = refs[nd["a"]], _dec_axis(nd["axis"])
            _check_axis_in_range(base, axis, i)
            built.append(SumExpression(base, axis))
        elif op == "sum_over":
            built.append(SumOverExpression([refs[a] for a in nd["args"]]))
    _retag_atoms(table, built, variables, params)
    return _StrictRefs(built, "node")


def _check_axis_in_range(base: Any, axis: Any, node: int) -> None:
    """Refuse a sum axis the base does not have (``dm.sum`` refuses it at build)."""
    shape = getattr(base, "shape", None)
    if shape is None or axis is None:
        return
    try:
        np.empty(tuple(shape), dtype=np.int8).sum(axis=axis)
    except (ValueError, TypeError) as exc:  # numpy's AxisError is a ValueError
        raise SerializationError(
            f"node {node} sums a {tuple(shape)}-shaped expression over axis {axis!r}: {exc}"
        ) from exc


def _check_index_in_range(base: Any, index: Any, node: int) -> None:
    """Refuse an index the modeling API would refuse.

    ``IndexExpression`` does not check its index against the base's shape -- the
    API's ``__getitem__`` does, before building one -- so a document indexing
    ``x[7]`` on a length-2 ``x`` loaded, and the solve then evaluated whatever
    the out-of-range read produced.
    """
    shape = getattr(base, "shape", None)
    if shape is None:
        return
    try:
        np.empty(tuple(shape), dtype=np.int8)[index]
    except (IndexError, TypeError, ValueError) as exc:
        raise SerializationError(
            f"node {node} indexes a {tuple(shape)}-shaped expression with {index!r}: {exc}"
        ) from exc


def _same_structure(a: Any, b: Any, var_ids: dict, param_ids: dict) -> bool:
    """Do *a* and *b* encode to byte-identical node tables?

    Reuses the writer rather than hand-rolling a second notion of expression
    equality -- ``Expression.__eq__`` builds a ``Constraint`` rather than
    comparing, so there is no operator to lean on, and a bespoke walker here
    would be a second implementation to keep in sync with ``_encode``.  The
    ``atom`` key is dropped before comparing: it is what this comparison is
    being used to decide, so leaving it in would make the test circular.
    """
    try:
        tables = []
        for expr in (a, b):
            t = _NodeTable(var_ids, param_ids)
            t.add(expr)
            tables.append([{k: v for k, v in nd.items() if k != "atom"} for nd in t.nodes])
    except Exception as exc:  # noqa: BLE001 - any failure to encode means "cannot verify"
        logger.debug("atom re-derivation could not encode a body (%s: %s)", type(exc).__name__, exc)
        return False
    return tables[0] == tables[1]


class _IdsOnFirstSight(dict):
    """An id -> index map that numbers each object the first time it is asked for.

    Lets :func:`_same_structure` compare two bodies that reference the same live
    variables and parameters without a model-wide index: equal objects get equal
    numbers because both encodings share this one map.
    """

    def get(self, key, default=None):  # type: ignore[override]
        if key not in self:
            self[key] = len(self)
        return dict.__getitem__(self, key)


def lowering_reproduces(fn: Any, arg: Any, body: Any) -> bool:
    """Does re-lowering registration *fn* on *arg* rebuild *body* exactly?

    The check :func:`_retag_atoms` applies to a loaded document, for a tag whose
    registration cannot be identified by object identity -- one that crossed a
    process boundary by pickle (:func:`discopt.operators.atom_of`).
    """
    try:
        rebuilt = fn.lower(arg)
    except Exception as exc:  # noqa: BLE001 - arbitrary user code; abstain, but say so
        logger.debug(
            "atom %r: re-lowering the current registration raised %s: %s",
            getattr(fn, "name", "?"),
            type(exc).__name__,
            exc,
        )
        return False
    return _same_structure(body, rebuilt, _IdsOnFirstSight(), _IdsOnFirstSight())


def _retag_atoms(table: list[dict], built: list[Any], variables: list, params: list) -> None:
    """Restore the registered-atom tags the document recorded (#1248 A).

    Re-tagging is not a matter of trusting the name.  The document carries the
    body that the registration *at write time* produced, while the relaxer would
    derive its envelope from whatever the registry holds under that name *now*.
    After a ``register_function(..., replace=True)`` those are different
    functions and the new envelope would cut the old body's true points -- the
    false bound ``atom_of``'s identity check exists to prevent (#1248, commit
    ``4725635b``).  A name alone cannot carry that identity across a process, so
    this re-derives it: the tag is restored only if re-lowering the *current*
    registration on the decoded argument reproduces the decoded body exactly.

    Every way of failing that check -- the name is not registered here, the
    lowering raises, the bodies differ -- leaves the node untagged, which is the
    term-by-term relaxation of the body the model actually carries, and sound.
    Sound, but invisible from the outside: the only symptom is a looser bound.
    The two failures that indicate a *caller* defect rather than an absent
    registration therefore log at DEBUG on ``discopt.serialize``.
    """
    if not any("atom" in nd for nd in table):
        return
    from discopt.operators import ATOM_ATTR, get_registered

    var_ids = {id(v): i for i, v in enumerate(variables)}
    param_ids = {id(p): i for i, p in enumerate(params)}

    for i, nd in enumerate(table):
        spec = nd.get("atom")
        if not isinstance(spec, dict):
            continue
        name, arg_ref = spec.get("n"), spec.get("arg")
        if not isinstance(name, str) or not isinstance(arg_ref, int):
            continue
        if not (0 <= arg_ref < len(built)) or i >= len(built):
            continue
        fn = get_registered(name)
        if fn is None:
            continue
        arg, body = built[arg_ref], built[i]
        try:
            rebuilt = fn.lower(arg)
        except Exception as exc:  # noqa: BLE001 - arbitrary user code; abstain, but say so
            # The abstention is the sound outcome, so this must not raise -- but a
            # registration that blows up on its own recorded argument is a defect in
            # the *caller's* code, and the only symptom otherwise is a quietly looser
            # bound.  Leave the trace: `logging.getLogger("discopt.serialize")`.
            logger.debug(
                "atom %r: re-lowering the current registration raised %s: %s; "
                "loading the recorded body term by term instead",
                name,
                type(exc).__name__,
                exc,
            )
            continue
        if not _same_structure(body, rebuilt, var_ids, param_ids):
            logger.debug(
                "atom %r: the current registration lowers to a different body than the "
                "document records; loading the recorded body term by term instead",
                name,
            )
            continue
        try:
            setattr(body, ATOM_ATTR, (name, arg, fn))
        except AttributeError:  # pragma: no cover - a slotted node cannot be tagged
            continue


# ── relations we refuse, by name ───────────────────────────────────────────

#: ``{type name: (what, remedy)}``. Empty since 1.5 encodes propositional logic;
#: kept so a relation type a later release adds before it has an encoding is
#: refused by name rather than by the generic unknown-row error.
_REFUSED_RELATIONS: dict[str, tuple[str, str]] = {}


def _refuse_unsupported(model: Model) -> None:
    """Refuse, by name, anything this version cannot write faithfully."""
    for con in list(model._constraints):
        tname = type(con).__name__
        if tname in _REFUSED_RELATIONS:
            what, remedy = _REFUSED_RELATIONS[tname]
            name = getattr(con, "name", None)
            where = f" named {name!r}" if name else ""
            raise SerializationError(f"cannot serialize: the model carries {what}{where}. {remedy}")
        # #1775 G1.2: nesting in a Disjunct block is carried by a parent link this
        # format does not record. Dropping it would turn a nested disjunction into a
        # top-level one -- a different feasible set -- so refuse instead.
        nested = (type(con) is _IndicatorConstraint and con.parent is not None) or (
            type(con) is _DisjunctiveConstraint and con.parent_indicator is not None
        )
        if nested:
            raise SerializationError(
                "cannot serialize: the model nests a disjunction in a Disjunct block "
                "(Disjunct.add_disjunction / Disjunct.subject_to), which this format "
                "version does not record (#1775 G1.2). Use pickle or Model.clone() "
                "for same-version transport."
            )
    # #1430: a disjunction built by ``Model.disjunction()``/``make_disjunct()`` and
    # never attached is not IN the model, so a document faithfully records a model
    # without it -- and the reloaded model carries no record that anything was
    # dropped, so ``validate``'s guard cannot fire on it either. That reopens the
    # exact silent-drop this document is written to avoid, one layer removed, and
    # ``solve_batch(workers > 1)`` walks straight into it: it serializes each model
    # for a worker, so a forgotten disjunction would be refused at ``workers=1`` and
    # silently dropped at ``workers=3`` -- the worker-count equivalence that module
    # documents, broken by a guard that only ran on one arm.
    model._reject_unattached_gdp_blocks()
    _refuse_unclassified_attributes(model)


def _refuse_unclassified_attributes(model: Model) -> None:
    """Refuse a model carrying a private attribute this format does not classify.

    ``loads`` rebuilds a fresh ``Model``, so anything not written is reset to its
    default with no error. Solve caches are derived and rebuilt on
    demand; public attributes a user hangs on the object are theirs and are never
    read by discopt. Everything else private is discopt's own state, and must be
    in ``_MODEL_STATE`` / ``_LAZY_STATE`` to be saved. (The few caches not named
    ``*_cache`` are listed in ``discopt.modeling.core._MODEL_UNNAMED_CACHES``.)
    """
    from discopt.modeling.core import _is_model_cache

    unknown = sorted(
        k
        for k in vars(model)
        if k.startswith("_")
        and not _is_model_cache(k)
        and k not in _MODEL_STATE
        and k not in _LAZY_STATE
    )
    if not unknown:
        return
    passes = sorted(
        {
            label
            for k in unknown
            for prefix, label in _REFORMULATION_RECORD_PREFIXES.items()
            if k.startswith(prefix)
        }
    )
    if passes:
        raise SerializationError(
            f"cannot serialize: the model carries records of the {', '.join(passes)} "
            f"transformation ({', '.join(unknown)}), which point back at the model it was "
            "built from and cannot be written. Save the source model instead, and re-apply "
            "the transformation after loading it."
        )
    raise SerializationError(
        f"cannot serialize: the model carries attribute(s) {unknown} that this format "
        "does not know how to write, and dropping them would reload a different model. "
        "Classify each in discopt.serialize (_MODEL_STATE or _LAZY_STATE)."
    )


# ── variables / parameters ─────────────────────────────────────────────────


def _enc_variable(v: Variable) -> dict:
    # `lb`/`ub` are the variable's LIVE box, which inside a `fix()` / `Model.fixed(...)`
    # scope is the pinned one. The declared domain then lives at the bottom of
    # `_bound_stack`, and every entry above it is a box some `fix()` replaced.
    #
    # Writing only the live box would persist a *temporary* pin as the model's
    # permanent declared domain: the reloaded model solves a different problem with
    # no warning, `fix_depth` is 0 so the pin cannot be undone, and re-fixing is
    # refused as "outside its declared bounds". So the stack travels with the box,
    # and the reloaded variable is pinned exactly as deeply as the saved one.
    return {
        "name": v.name,
        "type": v.var_type.value,
        "shape": list(v.shape),
        "lb": _enc_array(v.lb),
        "ub": _enc_array(v.ub),
        "bound_stack": [
            [_enc_array(lb), _enc_array(ub)] for lb, ub in getattr(v, "_bound_stack", [])
        ],
    }


def _dec_name(name, kind: str) -> str:
    """The declared name of a *kind* object, refusing anything the API would.

    `Model.continuous(...)` only ever produces `str` names, so a document
    carrying `1`, `None` or a list is corrupt. Without this the non-string ones
    flowed straight into the model: `1` and `None` loaded silently and reappeared
    in every downstream name lookup and export, while a list or dict failed much
    later with a bare `TypeError: unhashable type` from `model._names.add` --
    neither of which says the document is at fault (#1321).
    """
    if not isinstance(name, str):
        raise SerializationError(
            f"{kind} name must be a string, got {type(name).__name__} ({name!r})"
        )
    return name


def _dec_variable(d: dict, model: Model) -> Variable:
    name = _dec_name(d.get("name"), "variable")
    # #1310: a corrupted document (or a future writer bug) can carry two
    # variables sharing a name. `_register_variable` below only ever ADDS to
    # `model._names` -- it never checks it -- so without this, `load()` hands
    # back a `Model` whose `_variables` and `_names` have silently fallen out
    # of sync, with no error until some *later* call (`solve()`, `to_nl()`)
    # happens to hit the mismatch. Checked against `model._names` directly
    # (not the fuller `Model._check_name`, which also refuses the
    # GDP-auxiliary namespace -- a legitimately lowered model's aux variables
    # already carry that prefix and must still round-trip).
    if name in model._names:
        raise SerializationError(f"duplicate variable/parameter name {name!r}")
    shape = tuple(_dec_index_int(n, f"a dimension of variable {name!r}") for n in d["shape"])
    if any(n < 0 for n in shape):
        raise SerializationError(f"variable {name!r} has a negative dimension: {shape}")
    lb = _dec_array(d["lb"], shape)
    ub = _dec_array(d["ub"], shape)
    # Built through the low-level constructor rather than `model.continuous/
    # integer/...`: those apply defaulting rules (and, for INTEGER, a UserWarning
    # plus a [0, 1e6] fallback for an unspecified bound). Reloading a saved model
    # must reproduce the bounds that were saved, exactly, with no re-defaulting.
    var = Variable(name, VarType(d["type"]), shape, lb, ub, model)
    # Restore the fix stack, so `fix_depth`, `unfix()` and the declared-domain check
    # in `fix()` all see what the saved model saw.
    #
    # Frozen on the way in, exactly as `Variable.fix` freezes what it pushes:
    # `_bound_stack[0]` IS the declared domain `fix()` validates against, and
    # `unfix()` reinstalls these arrays as the live box, which
    # `Model.saved_bounds(copy=False)` then aliases. `_dec_array` hands back a
    # writable array, so a loaded model used to give back a writable box on the
    # first `unfix()` -- an in-place write there silently corrupted the
    # snapshot and changed the solve's answer (#1321).
    var._bound_stack = [
        _dec_bound_stack_entry(entry, i, shape, name, var.var_type)
        for i, entry in enumerate(d.get("bound_stack", []))
    ]
    if var._bound_stack:
        # Inside a fix scope the live box is a pin, and `fix()` refuses a pin
        # outside the declared domain (`_bound_stack[0]`). A document with one
        # outside it describes a model the API cannot produce, and `unfix()` would
        # then restore a domain the live box was never inside.
        dlo, dhi = (np.asarray(b) for b in var._bound_stack[0])
        if np.any(np.asarray(var.lb) < dlo) or np.any(np.asarray(var.ub) > dhi):
            raise SerializationError(
                f"variable {name!r}: its box lies outside its declared domain (bound_stack[0])"
            )
    return model._register_variable(var)


def _dec_bound_stack_entry(entry, index: int, shape, name: str, var_type: VarType):
    """One ``(lb, ub)`` frame of a saved fix stack, refused unless it is one.

    #1332 item 4: this was ``entry[0]``/``entry[1]`` with no checks, so a
    corrupt document loaded a stack frame that is not a box and said nothing.
    ``[10, 0]`` installed an INVERTED declared domain -- and
    ``_bound_stack[0]`` is exactly what ``fix()`` validates against and
    ``unfix()`` restores, so an inverted or NaN frame silently widens or empties
    the model's declared domain. A malformed frame raised a bare ``IndexError``
    or a reshape ``ValueError`` from deep inside numpy, neither of which says
    the document is at fault.

    Frozen on the way in, exactly as ``Variable.fix`` freezes what it pushes:
    ``unfix()`` reinstalls these arrays as the live box, which
    ``Model.saved_bounds(copy=False)`` then aliases, so a writable one silently
    corrupted the snapshot and changed the solve's answer (#1321).
    """
    where = f"variable {name!r} bound_stack[{index}]"
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        raise SerializationError(
            f"{where} must be a [lb, ub] pair, got "
            f"{type(entry).__name__} of length {len(entry) if hasattr(entry, '__len__') else '?'}"
        )
    boxes = []
    for side, raw in (("lb", entry[0]), ("ub", entry[1])):
        try:
            boxes.append(
                _readonly_bound(
                    _dec_array(raw, shape),
                    shape,
                    var_type=var_type,
                    what=f"{where} {side}",
                )
            )
        except SerializationError:
            raise
        except (TypeError, ValueError) as exc:
            raise SerializationError(f"{where} {side} is not a valid bound: {exc}") from exc
    lo, hi = boxes
    if np.any(np.asarray(lo) > np.asarray(hi)):
        raise SerializationError(
            f"{where} is inverted (lb > ub): {np.asarray(lo)!r} > {np.asarray(hi)!r}. "
            "bound_stack[0] is the declared domain that fix() validates against."
        )
    return (lo, hi)


def _enc_parameter(p: Parameter) -> dict:
    return {"name": p.name, "shape": list(p.value.shape), "value": _enc_array(p.value)}


def _dec_parameter(d: dict, model: Model) -> Parameter:
    name = _dec_name(d.get("name"), "parameter")
    # #1310: same name-uniqueness gap as `_dec_variable`, and reachable across
    # the variable/parameter boundary too -- `Model.validate()` only checks
    # uniqueness within `_variables`, so a variable/parameter name collision
    # introduced here is not caught at load, solve, or export time either.
    if name in model._names:
        raise SerializationError(f"duplicate variable/parameter name {name!r}")
    value = _dec_array(d["value"], tuple(d["shape"]))
    param = Parameter(name, value, model)
    model._parameters.append(param)
    model._names.add(param.name)
    return param


# ── builder-resident blocks ────────────────────────────────────────────────
#
# The fast-construction API (`add_linear_constraints`, `add_linear_objective`,
# `add_quadratic_objective`) emits rows straight into the Rust builder; they never
# appear in `model._constraints`. A writer that reads only `_constraints` emits a
# strict subset of the model -- the X-1 hazard documented in `export/_common.py`,
# which here would mean a saved model that reloads missing constraints and solves
# to a wrong answer. They are encoded as their own section, in their own CSR form,
# and restored through the same fast API so they stay builder-resident on reload.


def _enc_csr(A) -> dict:
    import scipy.sparse as sp

    csr = sp.csr_matrix(A)
    return {
        "shape": list(csr.shape),
        "indptr": [int(i) for i in csr.indptr],
        "indices": [int(i) for i in csr.indices],
        "data": [_enc_float(x) for x in csr.data],
    }


def _dec_csr(d: dict):
    import scipy.sparse as sp

    return sp.csr_matrix(
        (
            np.array([_dec_float(x) for x in d["data"]], dtype=np.float64),
            np.array(d["indices"], dtype=np.int32),
            np.array(d["indptr"], dtype=np.int32),
        ),
        shape=tuple(d["shape"]),
    )


def _enc_builder_blocks(model: Model, var_ids: dict[int, int]) -> list[dict]:
    out = []
    for A, x, sense, b, name in getattr(model, "_builder_linear_blocks", []) or []:
        ref = var_ids.get(id(x))
        if ref is None:
            raise SerializationError(
                f"fast-API constraint block references variable "
                f"{getattr(x, 'name', x)!r}, which is not registered on this model."
            )
        b_arr = np.broadcast_to(np.asarray(b, dtype=np.float64), (A.shape[0],))
        out.append(
            {
                "A": _enc_csr(A),
                "x": ref,
                "sense": sense,
                "b": _enc_array(b_arr),
                "name": name,
            }
        )
    return out


# ── relations ──────────────────────────────────────────────────────────────


def _enc_rows(model: Model, table: _NodeTable) -> list[dict]:
    """Encode every row of ``model._constraints`` as one ordered, kind-tagged list.

    One list rather than a section per kind, because ``model._constraints`` is a
    *heterogeneous ordered* list: algebraic rows and the indicator / SOS /
    disjunctive relations are interleaved in declaration order. Writing them as
    separate sections and re-appending section by section on load silently
    reorders a model that mixes them, which changes `.nl` row order and any
    downstream row indexing.

    Indicator and SOS members go into the node table rather than being written as
    variable indices. Their dataclass fields are *annotated* ``Variable``, but the
    modeling API accepts an element of an array variable too -- ``m.if_then(y[0],
    ...)`` stores an ``IndexExpression`` -- so a variable-index encoding refuses
    the ordinary indexed form. The node table carries either shape.
    """
    rows: list[dict] = []
    # `_constraints` is annotated `list[Constraint]` but really holds the relation
    # dataclasses too, so iterate an Any-typed view and narrow by exact type.
    declared: list[Any] = list(model._constraints)
    for con in declared:
        t = type(con)
        if t is Constraint:
            rows.append({"kind": "algebraic", **_enc_constraint(con, table)})
        elif t is _IndicatorConstraint:
            rows.append(
                {
                    "kind": "indicator",
                    "indicator": table.add(con.indicator),
                    "constraint": _enc_constraint(con.constraint, table),
                    "active_value": int(con.active_value),
                    "name": con.name,
                    "disjunction": (None if con.disjunction is None else list(con.disjunction)),
                }
            )
        elif t is _SOSConstraint:
            rows.append(
                {
                    "kind": "sos",
                    "sos_type": int(con.sos_type),
                    "variables": [table.add(v) for v in con.variables],
                    "name": con.name,
                }
            )
        elif t is _DisjunctiveConstraint:
            rows.append(
                {
                    "kind": "disjunction",
                    "disjuncts": [[_enc_constraint(c, table) for c in d] for d in con.disjuncts],
                    "name": con.name,
                    "method": con.method,
                    # By member NAME: this enum's `.value` is an (activation,
                    # cardinality) pair of enums, not a string, so `.value` is
                    # neither JSON-safe nor a stable identity to read back.
                    "semantics": con.semantics.name,
                }
            )
            if con._indicator_vars is not None:
                # #1775 G1.1 (schema 1.5): the indicator columns, once read.
                # Dropping them would leave those binaries in the model with
                # nothing tying them to the disjunction. Omitted while unread, so
                # such a row is written exactly as at 1.4.
                rows[-1]["indicators"] = [table.add(v) for v in con._indicator_vars]
        elif t is _LogicalConstraint:
            rows.append({"kind": "logical", "logic": _enc_logic(con, table), "name": con.name})
        else:
            raise SerializationError(
                f"cannot serialize a row of type {t.__name__}; this format version "
                "carries algebraic constraints and indicator / SOS / disjunctive / "
                "logical relations."
            )
    return rows


# ── propositional logic (1.5) ──────────────────────────────────────────────
#
# A `_LogicalConstraint` holds a tree of `Logical*` dataclasses over `BooleanVar`
# leaves. It is written as a flat table, children before parents and the root
# last -- the node table's layout, for the node table's reasons: `land(...)` builds
# a left-deep chain as long as its argument list, which nested JSON would turn
# into nesting that deep, and a subformula used twice stays one entry. A leaf
# points into the node table at the binary variable (or element of one) behind
# the `BooleanVar`; the wrapper itself carries nothing else.
#
# The lowering (`gdp_reformulate._logical_to_linear`) is a function of the tree
# alone -- no identity cache -- so a reloaded tree lowers to the same rows (WP-3
# entry experiment: 6 models, lowered twice and from a rebuild, identical rows;
# identical objective and node count on solve).

_LOGIC_BINARY = {LogicalAnd: "and", LogicalOr: "or", LogicalEquivalent: "iff"}
_LOGIC_CARDINALITY = {LogicalAtLeast: "atleast", LogicalAtMost: "atmost", LogicalExactly: "exactly"}
_LOGIC_CONNECTIVES = frozenset(
    {"var", "not", "implies", *_LOGIC_BINARY.values(), *_LOGIC_CARDINALITY.values()}
)


def _boolean_leaf_problem(expr: Any) -> Optional[str]:
    """Why *expr* cannot be a ``BooleanVar``'s variable, or ``None`` if it can.

    ``Model.boolean`` backs a ``BooleanVar`` with a binary variable of size 1, or
    an element of a binary array; the lowering reads it as a 0/1 term. Anything
    else is not a model the API builds, and is refused both ways.
    """
    base = expr.base if isinstance(expr, IndexExpression) else expr
    if not isinstance(base, Variable):
        return f"{type(expr).__name__} is not a variable or an element of one"
    if base.var_type is not VarType.BINARY:
        return f"variable {base.name!r} is {base.var_type.value}, not binary"
    if isinstance(expr, Variable) and expr.size != 1:
        return f"variable {base.name!r} has {expr.size} elements, not one"
    return None


def _enc_logic(con: Any, table: _NodeTable) -> list[dict]:
    entries: list[dict] = []
    index: dict[int, int] = {}
    stack: list = [(con.expression, False)]
    while stack:
        node, expanded = stack.pop()
        if id(node) in index:
            continue
        t = type(node)
        if t is BooleanVar:
            problem = _boolean_leaf_problem(node.variable)
            if problem is not None:
                raise SerializationError(
                    f"logical constraint {con.name!r}: a BooleanVar's {problem}"
                )
            entry = {"c": "var", "x": table.add(node.variable)}
        else:
            if t in _LOGIC_BINARY:
                kids = [node.left, node.right]
            elif t is LogicalImplies:
                kids = [node.antecedent, node.consequent]
            elif t is LogicalNot:
                kids = [node.operand]
            elif t in _LOGIC_CARDINALITY:
                kids = list(node.operands)
            else:
                raise SerializationError(
                    f"logical constraint {con.name!r} holds a {t.__name__}, which is not "
                    "a propositional connective this format knows"
                )
            if not expanded:
                stack.append((node, True))
                stack.extend((k, False) for k in reversed(kids) if id(k) not in index)
                continue
            refs = [index[id(k)] for k in kids]
            if t in _LOGIC_BINARY:
                entry = {"c": _LOGIC_BINARY[t], "a": refs[0], "b": refs[1]}
            elif t is LogicalImplies:
                entry = {"c": "implies", "a": refs[0], "b": refs[1]}
            elif t is LogicalNot:
                entry = {"c": "not", "a": refs[0]}
            else:
                k = node.k
                if type(k) is not int:
                    raise SerializationError(
                        f"logical constraint {con.name!r}: cardinality k={k!r} is not an int"
                    )
                entry = {"c": _LOGIC_CARDINALITY[t], "k": k, "args": refs}
        index[id(node)] = len(entries)
        entries.append(entry)
    return entries


def _dec_logic(entries: Any, nodes: list, name: Any) -> Any:
    if not isinstance(entries, list) or not entries:
        raise SerializationError(f"logical row {name!r} has no logic table")
    built: list = []
    refs = _StrictRefs(built, f"logic entry of {name!r}")
    for i, e in enumerate(entries):
        c = e["c"]
        if c not in _LOGIC_CONNECTIVES:
            raise SerializationError(f"logical row {name!r}: unknown connective {c!r}")
        if c == "var":
            leaf = nodes[e["x"]]
            problem = _boolean_leaf_problem(leaf)
            if problem is not None:
                raise SerializationError(f"logical row {name!r}, entry {i}: {problem}")
            built.append(BooleanVar(leaf))
        elif c == "not":
            built.append(LogicalNot(refs[e["a"]]))
        elif c == "implies":
            built.append(LogicalImplies(refs[e["a"]], refs[e["b"]]))
        elif c in ("and", "or", "iff"):
            pair = {"and": LogicalAnd, "or": LogicalOr, "iff": LogicalEquivalent}[c]
            built.append(pair(refs[e["a"]], refs[e["b"]]))
        else:
            k = e["k"]
            if type(k) is not int:
                raise SerializationError(f"logical row {name!r}: cardinality k={k!r} is not an int")
            if not isinstance(e["args"], list):
                raise SerializationError(f"logical row {name!r}: args must be a list")
            card = {"atleast": LogicalAtLeast, "atmost": LogicalAtMost, "exactly": LogicalExactly}[
                c
            ]
            built.append(card(k, [refs[a] for a in e["args"]]))
    return built[-1]


def _dec_indicators(d: dict, nodes: list) -> tuple:
    """The ``indicators`` of a 1.5+ disjunction row: one scalar binary per disjunct.

    Checked by kind as well as by reference: the GDP lowering uses these columns
    as the disjunction's selectors (``reformulate_gdp``), so a reference to a
    continuous column or a constant would lower without complaint to a different
    model.
    """
    refs = d["indicators"]
    name = d.get("name")
    if not isinstance(refs, list):
        raise SerializationError(
            f"disjunction {name!r} has indicators {refs!r}; it must be a list of node "
            "references (the key is omitted while the indicators are unread)"
        )
    if len(refs) != len(d["disjuncts"]):
        raise SerializationError(
            f"disjunction {name!r} has {len(refs)} indicators for {len(d['disjuncts'])} disjuncts"
        )
    cols = tuple(nodes[i] for i in refs)
    for col in cols:
        if not (type(col) is Variable and col.var_type == VarType.BINARY and col.shape == ()):
            raise SerializationError(
                f"disjunction {name!r} has an indicator that is not a scalar binary "
                f"variable ({col!r})"
            )
    if len({id(c) for c in cols}) != len(cols):
        raise SerializationError(f"disjunction {name!r} repeats an indicator column")
    return cols


def _dec_rows(docs: list[dict], nodes: list, minor: int) -> list:
    """Rebuild the ordered, heterogeneous ``_constraints`` list."""
    from discopt.modeling.core import DisjunctionSemantics

    out: list = []
    for d in docs:
        kind = d.get("kind")
        if kind == "algebraic":
            out.append(_dec_constraint(d, nodes))
        elif kind == "indicator":
            active_value = d["active_value"]
            if type(active_value) is not int or active_value not in (0, 1):
                # Anything but 0/1 made the lowering activate the row for an
                # indicator value the variable can never take, so the row was
                # silently dropped and a false optimum came back certified.
                raise SerializationError(
                    f"indicator row {d.get('name')!r} has active_value {active_value!r}; "
                    "it must be 0 or 1"
                )
            out.append(
                _IndicatorConstraint(
                    indicator=nodes[d["indicator"]],
                    constraint=_dec_constraint(d["constraint"], nodes),
                    active_value=active_value,
                    name=d["name"],
                    # Absent in files written before #1615 C-10a.
                    disjunction=(
                        None
                        if d.get("disjunction") is None
                        else (
                            _dec_name(d["disjunction"][0], "indicator disjunction"),
                            _dec_index_int(d["disjunction"][1], "an indicator disjunct index"),
                            _dec_index_int(d["disjunction"][2], "an indicator disjunct count"),
                        )
                    ),
                )
            )
        elif kind == "sos":
            if type(d["sos_type"]) is not int or d["sos_type"] not in (1, 2):
                raise SerializationError(
                    f"SOS row {d.get('name')!r} has sos_type {d['sos_type']!r}; it must be 1 or 2"
                )
            out.append(
                _SOSConstraint(
                    sos_type=d["sos_type"],
                    variables=[nodes[i] for i in d["variables"]],
                    name=d["name"],
                )
            )
        elif kind == "disjunction":
            if d["method"] not in _KNOWN_DISJUNCTION_METHODS:
                raise SerializationError(
                    f"disjunction {d.get('name')!r} names unknown method {d['method']!r}"
                )
            out.append(
                _DisjunctiveConstraint(
                    disjuncts=[[_dec_constraint(c, nodes) for c in dd] for dd in d["disjuncts"]],
                    name=d["name"],
                    method=d["method"],
                    semantics=DisjunctionSemantics[d["semantics"]],
                    # Schema 1.5 (#1775 G1.1), and only once read. Not looked at
                    # in an older document, so there it is refused as unread.
                    _indicator_vars=(
                        _dec_indicators(d, nodes)
                        if minor >= _SCHEMA_MINOR_LOGIC_LOWERING_INDICATORS and "indicators" in d
                        else None
                    ),
                )
            )
        elif kind == "logical":
            if minor < _SCHEMA_MINOR_LOGIC_LOWERING_INDICATORS:
                # No writer before 1.5 wrote a logical row (they refused the
                # model), so an older document carrying one is malformed.
                raise SerializationError(
                    f"a schema 1.{minor} document carries a 'logical' row, which no "
                    "writer of that version produced; the document is malformed or "
                    "was edited by hand"
                )
            name = d["name"]
            if name is not None and not isinstance(name, str):
                raise SerializationError(f"logical row name must be a string, got {name!r}")
            out.append(_LogicalConstraint(_dec_logic(d["logic"], nodes, name), name))
        else:
            raise SerializationError(
                f"row has unknown kind {kind!r}. This file was written by a newer "
                "discopt than the one reading it; refusing rather than dropping the row."
            )
    return out


def _enc_constraint(c: Constraint, table: _NodeTable) -> dict:
    # `Constraint` is stored normalised as `body sense 0.0` and `subject_to` /
    # `validate` refuse a non-zero rhs (#909), so carrying `rhs` would encode a
    # value that is required to be 0.0. Asserted rather than assumed.
    if float(c.rhs) != 0.0:
        raise SerializationError(
            f"constraint {c.name or '<unnamed>'} has non-zero rhs {c.rhs!r}; discopt "
            "stores rows normalised with rhs == 0, so this model is malformed."
        )
    return {"body": table.add(c.body), "sense": c.sense, "name": c.name}


def _dec_constraint(d: dict, nodes: list) -> Constraint:
    if d["sense"] not in _KNOWN_SENSES:
        raise SerializationError(
            f"constraint {d.get('name')!r} has unknown sense {d['sense']!r}; "
            f"expected one of {sorted(_KNOWN_SENSES)}"
        )
    return Constraint(body=nodes[d["body"]], sense=d["sense"], rhs=0.0, name=d["name"])


# ── complementarity relations ──────────────────────────────────────────────
#
# A relation is a node of the model's IR that lives OUTSIDE `_constraints`: it is
# recorded on `model._complementarities`, and whether this particular model already
# carries the rows that encode it is recorded separately, on
# `model._lowered_complementarities` (an identity map relation -> LoweringRecord).
#
# Both halves have to travel. Carrying the relations without the marks would
# reload a model whose rows already encode every relation but which reports them
# all as unlowered -- `unlowered_relations` is a solver-boundary refusal, so the
# reloaded model would not solve at all. Carrying the marks without the *method*
# would be worse than that: `LoweringRecord.is_exact` is derived from the method
# alone, and a `scholtes` lowering is a RELAXATION of the relation. A mark that
# lost its method would read as exact, and the solver would certify a relaxation
# as if it were the declared model -- a false certificate.


def _enc_complementarities(model: Model, table: _NodeTable) -> Optional[dict]:
    """Encode the relation set, its declaration order, and this model's lowering marks."""
    declared: list[Any] = list(getattr(model, "_complementarities", []) or [])
    lowered_map: dict = dict(getattr(model, "_lowered_complementarities", {}) or {})
    if not declared and not lowered_map:
        return None

    # Transitive closure over `source`: an element relation points at the vector
    # relation it was scalarized from, and that parent need not itself appear in
    # `_complementarities`. Dropping it would break `describe()`/attribution back
    # to the declared relation.
    order: list[Any] = []
    seen: set[int] = set()

    def visit(rel: Any) -> None:
        if rel is None or id(rel) in seen:
            return
        seen.add(id(rel))
        visit(rel.source)  # parent first, so its index is known when the child refers to it
        order.append(rel)

    for rel in declared:
        visit(rel)
    for rel in lowered_map:
        visit(rel)

    index_of = {id(rel): i for i, rel in enumerate(order)}

    def enc_shape(shape) -> Optional[list[int]]:
        return None if shape is None else [int(v) for v in shape]

    relations = [
        {
            "f": table.add(rel.f),
            "g": table.add(rel.g),
            "name": rel.name,
            "role": rel.role.value,
            "f_bounds": [_enc_float(rel.f_bounds[0]), _enc_float(rel.f_bounds[1])],
            "g_bounds": [_enc_float(rel.g_bounds[0]), _enc_float(rel.g_bounds[1])],
            "scale": None if rel.scale is None else _enc_float(rel.scale),
            "parent": rel.parent,
            "f_shape": enc_shape(rel.f_shape),
            "g_shape": enc_shape(rel.g_shape),
            "index": enc_shape(rel.index),
            "source": None if rel.source is None else index_of[id(rel.source)],
        }
        for rel in order
    ]

    return {
        "relations": relations,
        # `_complementarities` is the durable declared record and its order is part
        # of the model; the closure above may hold relations that are not in it.
        "declared": [index_of[id(rel)] for rel in declared],
        # Since 1.5, with the rows each lowering emitted, by position in
        # `_constraints` -- the one thing original and reload agree on (the rows
        # are this model's objects). `None` stays `None`: "not tracked" is a
        # different answer from `[]`, "emitted no rows" (`rows_in`).
        # Untracked (`None`) keeps the two-element form a 1.4 reader reads; only a
        # tracked lowering, whose rows that reader would drop, carries the third.
        "lowered": [
            [index_of[id(rel)], record.method]
            + ([] if record.rows is None else [_enc_lowering_rows(model, rel, record)])
            for rel, record in lowered_map.items()
        ],
    }


def _enc_lowering_rows(model: Model, rel: Any, record: Any) -> list[int]:
    pos = {id(c): i for i, c in enumerate(model._constraints)}
    out = []
    for row in record.rows:
        if id(row) not in pos:
            raise SerializationError(
                f"the lowering of {rel.describe()} records a row that is no longer in the "
                "model; the record would point at nothing after a reload"
            )
        out.append(pos[id(row)])
    return out


def _dec_complementarities(doc: Optional[dict], model: Model, nodes: list, minor: int) -> None:
    """Rebuild the relation set and re-apply this model's lowering marks."""
    if not doc:
        return
    from discopt.mpec import Complementarity, ComplementarityRole, LoweringRecord

    def dec_shape(v) -> Optional[tuple]:
        return None if v is None else tuple(int(x) for x in v)

    rebuilt: list[Any] = []
    for d in doc["relations"]:
        src_idx = d["source"]
        if src_idx is not None:
            src_idx = _dec_index_int(src_idx, "a complementarity source")
        if src_idx is not None and not 0 <= src_idx < len(rebuilt):
            raise SerializationError(
                f"complementarity relation refers to a source at index {src_idx} that "
                "has not been read yet; the document's relation order is invalid."
            )
        rebuilt.append(
            Complementarity(
                f=nodes[d["f"]],
                g=nodes[d["g"]],
                name=d["name"],
                role=ComplementarityRole(d["role"]),
                f_bounds=(_dec_float(d["f_bounds"][0]), _dec_float(d["f_bounds"][1])),
                g_bounds=(_dec_float(d["g_bounds"][0]), _dec_float(d["g_bounds"][1])),
                scale=None if d["scale"] is None else _dec_float(d["scale"]),
                parent=d["parent"],
                f_shape=dec_shape(d["f_shape"]),
                g_shape=dec_shape(d["g_shape"]),
                index=dec_shape(d["index"]),
                source=None if src_idx is None else rebuilt[src_idx],
            )
        )

    relations = _StrictRefs(rebuilt, "complementarity relation")
    model._complementarities = [relations[i] for i in doc["declared"]]
    rows = _StrictRefs(list(model._constraints), "row")
    lowered: dict = {}
    for entry in doc["lowered"]:
        # Two elements: untracked rows (`None`) -- all a writer before 1.5 could
        # say. Three, from 1.5 only: the tracked rows' positions.
        if not isinstance(entry, list) or len(entry) not in (
            (2, 3) if minor >= _SCHEMA_MINOR_LOGIC_LOWERING_INDICATORS else (2,)
        ):
            raise SerializationError(f"invalid complementarity lowering entry {entry!r}")
        rel, method = relations[entry[0]], entry[1]
        positions = entry[2] if len(entry) == 3 else None
        if len(entry) == 3:
            if not isinstance(positions, list):
                raise SerializationError(f"lowering rows must be a list, got {positions!r}")
            picked = [rows[p] for p in positions]
            if len({id(r) for r in picked}) != len(picked):
                raise SerializationError(f"lowering rows list a row twice: {positions!r}")
            positions = picked
        if rel in lowered:
            raise SerializationError(f"{rel.describe()} is marked lowered twice")
        lowered[rel] = LoweringRecord(method=method, rows=positions)
    model._lowered_complementarities = lowered


# ── the rest of the model's state ──────────────────────────────────────────
#
# `loads` builds a fresh `Model` and restores section by section, so any attribute
# with no section is silently reset to its `__init__` default. For the decomposition
# annotations that is worse than losing a label: `discopt.decomposition` falls back
# to AUTO-DETECTING structure when they are empty, so a user's declared Benders /
# Lagrangian decomposition is quietly replaced by a guess.
#
# `_MODEL_STATE` is the registry of every `Model.__init__` attribute and how this
# format handles it. `test_every_model_attribute_is_accounted_for` compares it
# against a live `Model`, so a field added to `Model` in future cannot slip through
# unhandled -- it fails the test until someone decides which bucket it belongs in.
#
# That test sees only what `__init__` creates. discopt also attaches private
# attributes to a model *later* -- `from_nl`, a solve, a reformulation pass -- and
# those used to be dropped on save with no error. `_LAZY_STATE` classifies the ones
# known today, and `_refuse_unsupported` refuses any other private attribute by
# name, so one added in future fails loudly on its first save rather than quietly
# reloading as a different model.

#: Attributes carried by a dedicated section elsewhere in the document.
_STATE_IN_OWN_SECTION = frozenset(
    {
        "name",
        "_variables",
        "_parameters",
        "_constraints",
        "_objective",
        "_initial_point",
        "_complementarities",
        "_lowered_complementarities",
        "_builder_linear_blocks",
        "_builder_linear_objective",
        "_builder_quadratic_objective",
        "saved_result",
        "provenance",
    }
)

#: Attributes rebuilt as a side effect of restoring the sections above, so writing
#: them would be writing a derived value that could disagree with its source.
_STATE_DERIVED = frozenset({"_names", "_builder", "_flat_var_offsets_cache"})

#: Attributes deliberately NOT carried: session-local state that a reloaded model
#: must start without, rather than inherit from whoever saved the file.
#:
#: ``_last_solve_result`` (#1313) is the point this model object was last solved
#: to, kept so ``sensitivity()`` can start there and cross-check against it. It is
#: a cache of something that happened in *this* process, and the document already
#: has an explicit, opt-in way to travel with a result -- ``Model.save(...,
#: result=...)`` -> ``saved_result``, which is in ``_STATE_IN_OWN_SECTION`` above.
#: Writing this one too would make a reloaded model silently claim a solve nobody
#: in the new process ran, and ``sensitivity()`` would cross-check against it.
#:
#: ``_gdp_factory_disjunctions`` / ``_gdp_factory_disjuncts`` (#1430) are the
#: build-time record of what ``Model.disjunction()`` / ``Model.make_disjunct()``
#: handed the caller, kept so ``validate`` can refuse a block that was never
#: attached. They are not carried because by the time a document exists there is
#: nothing left in them to carry: ``_refuse_unsupported`` runs
#: ``_reject_unattached_gdp_blocks`` first, so every tracked block is already
#: attached and therefore already written out as part of ``_constraints``.
#: Writing the record too would persist an identity-keyed bookkeeping list whose
#: entries the reloaded model no longer owns.
_STATE_NOT_CARRIED = frozenset(
    {"_last_solve_result", "_gdp_factory_disjunctions", "_gdp_factory_disjuncts"}
)

#: Attached after ``__init__`` by a solve, and deliberately not carried for the
#: same reason as ``_last_solve_result``: each records the *last* solve in this
#: process, and every solve rewrites it before anything reads it.
#: ``_gauss_newton_hessian`` is ``solve(gauss_newton=...)``'s argument, set at the
#: top of every solve (and by ``discopt.ml.train`` before its own); the other two
#: are budgets ``solve_model`` stamps at the start of each solve.
#: ``_solve_deadline`` is moreover an absolute ``time.perf_counter()`` reading,
#: whose origin is per-process, so a saved value would mean nothing on reload.
#: ``_disjunctive_config_floor`` is a root bound a solve computed and stashed for
#: its own node LPs (#732 Stage 2); it describes that solve's root box. The solver
#: only ever sets it on the reformulated model a solve builds internally, so it
#: should never reach a saved model -- classified so that, if it ever does, it is
#: left out rather than refused.
_LAZY_NOT_CARRIED = frozenset(
    {
        "_gauss_newton_hessian",
        "_convexity_time_budget",
        "_solve_deadline",
        "_disjunctive_config_floor",
    }
)

#: Attached after ``__init__`` and carried in "state" as a plain value, only when
#: present. ``_recentre_inner`` marks the inner model of a recentred solve, so
#: solving it does not recentre a second time.
_LAZY_PLAIN = ("_recentre_inner",)

#: ``from_nl``'s parse (a Rust ``PyModelRepr``) and the path of the file it came
#: from. Neither travels as itself:
#:
#: * ``_nl_repr`` cannot be written. What is carried is the fact that the model
#:   has one -- a ``from_nl`` model answers "could not tell" in
#:   ``Model._referenced_variable_names`` -- and ``loads`` attaches a
#:   ``_DetachedNlRepr`` marker so the reloaded model answers the same way
#:   instead of switching to the less conservative expression walk.
#: * ``_source_nl_path`` is carried together with a digest of the file's bytes
#:   (``_source_nl_sha256``, taken by ``from_nl``). The native-AD NLP path reads
#:   the file at that path whenever it exists, so a bare path would hand the
#:   solver whatever file sits there on the machine that loads the document --
#:   another machine's file of the same name, or this one edited since. The
#:   solver uses the path only while the digest still matches; otherwise it falls
#:   back to emitting the in-memory model, as for any model not read from a file.
_LAZY_NL_SOURCE = frozenset({"_nl_repr", "_source_nl_path", "_source_nl_sha256"})

#: Records a reformulation pass attaches to the model it returns, keyed by the
#: pass's prefix. The refusal is permanent (plan §3.6), not a missing section: the
#: ``_ipx_*`` and ``_nsl_*`` sets hold the whole pre-pass source model, which a
#: document cannot hold, and ``_ipx_config_*`` feeds the disjunctive configuration
#: *bound*, so a reload without them would search differently from the model that
#: was saved. The remedy is lossless: save the source model and re-apply the
#: (deterministic) transformation after loading.
_REFORMULATION_RECORD_PREFIXES = {
    "_bml_": "binary.multilinear",
    "_ipx_": "integer.bilinear / integer.multilinear",
    "_nsl_": "nonsmooth.epigraph",
}

#: ``_gams_initial_values``: the starting point a GAMS model declares (``x.l``),
#: which ``from_gams`` / the GMO link attach and the solver seeds its primal
#: search from. Model content, so carried (see ``_enc_gams_initial_values``).
_LAZY_BESPOKE = frozenset({"_gams_initial_values"})

_LAZY_STATE = _LAZY_NOT_CARRIED | frozenset(_LAZY_PLAIN) | _LAZY_NL_SOURCE | _LAZY_BESPOKE

#: Attributes carried verbatim in the "state" section (plain JSON-safe values).
_STATE_PLAIN = ("_aux_counter", "_decomp_stages", "_decomp_blocks")

#: Attributes carried in "state" as a sorted list (a set is not JSON).
#: ``_free_by_construction`` (#1618 X-29b) names the variables discopt itself made
#: free (CVaR's ``eta``); carried so a reloaded model does not start blaming the
#: user for bounds they never declared.
_STATE_AS_SORTED_LIST = (
    "_zero_spanning_factor_auxes",
    "_implied_integer_auxes",
    "_free_by_construction",
)

#: Attributes carried in "state" by a bespoke encoder.
_STATE_BESPOKE = (
    "_atan2_preconditions",
    "_piecewise_domains",
    "_coupling_keys",
    "_sets",
    "_simplex_lowerings",
    "_block_labels_var",
    "_block_labels_con",
)

_MODEL_STATE = (
    _STATE_IN_OWN_SECTION
    | _STATE_DERIVED
    | _STATE_NOT_CARRIED
    | frozenset(_STATE_PLAIN)
    | frozenset(_STATE_AS_SORTED_LIST)
    | frozenset(_STATE_BESPOKE)
)


def _enc_sets(model: Model) -> list[dict]:
    """Named index sets (``Model.set``). Members are scalars or tuples."""
    out = []
    for st in getattr(model, "_sets", []) or []:
        members = [list(mem) if isinstance(mem, tuple) else mem for mem in st.members]
        out.append({"name": st.name, "dimen": int(st.dimen), "members": members})
    return out


def _dec_sets(docs: list[dict]) -> list:
    from discopt.modeling.sets import Set

    # `Model.set()` refuses a name already in use; nothing re-checked it here,
    # so a document with two sets of one name reloaded into a model where every
    # `sum_over` on that name silently resolves to whichever copy is found
    # first (#1321).
    seen: set[str] = set()
    out = []
    for d in docs:
        name = _dec_name(d.get("name"), "set")
        if name in seen:
            raise SerializationError(f"duplicate set name {name!r}")
        seen.add(name)
        out.append(
            Set(
                name,
                [tuple(mem) if isinstance(mem, list) else mem for mem in d["members"]],
                dimen=d["dimen"],
            )
        )
    return out


def _enc_coupling_keys(model: Model, rows: list) -> dict:
    """Coupling-constraint marks (``Model.mark_coupling``).

    The set holds a mix: name strings, and ``id(constraint)`` for the object form.
    A raw ``id`` is a process-local address and means nothing after a reload, so it
    is written as the constraint's ROW INDEX and turned back into the rebuilt row's
    ``id`` on load. An id matching no row is refused rather than dropped -- a
    silently missing coupling mark changes which rows get dualized.
    """
    keys: set = getattr(model, "_coupling_keys", set()) or set()
    row_of_id = {id(con): i for i, con in enumerate(rows)}
    names: list[str] = []
    indices: list[int] = []
    for key in keys:
        if isinstance(key, str):
            names.append(key)
        elif isinstance(key, int):
            idx = row_of_id.get(key)
            if idx is None:
                raise SerializationError(
                    "a coupling mark refers to a constraint object that is not among "
                    "this model's rows, so it cannot be written as a stable reference. "
                    "Re-mark the coupling constraints on the current model "
                    "(Model.mark_coupling), or mark them by name."
                )
            indices.append(idx)
        else:
            raise SerializationError(
                f"cannot serialize a coupling mark of type {type(key).__name__}; "
                "Model.mark_coupling records a name string or a constraint object."
            )
    return {"names": sorted(names), "row_indices": sorted(indices)}


def _dec_coupling_keys(d: Optional[dict], rows: list) -> set:
    if not d:
        return set()
    out: set = {_dec_name(n, "coupling mark") for n in d.get("names", [])}
    row_refs = _StrictRefs(rows, "coupling-mark row")
    for idx in d.get("row_indices", []):
        out.add(id(row_refs[idx]))
    return out


def _enc_block_labels_var(model: Model) -> dict:
    """Element-wise variable block labels (``Model.set_block`` with an array, #1370).

    Keyed by variable name, which is stable across a round-trip, so this is a
    plain name -> list-of-ints map. Written for the same reason the coupling
    marks are: a label that vanishes on reload does not merely lose a hint --
    ``block_structure`` would emit a partition the user never declared, or none.
    """
    labels: dict = getattr(model, "_block_labels_var", {}) or {}
    return {
        str(name): [int(v) for v in np.asarray(vals).reshape(-1)] for name, vals in labels.items()
    }


def _dec_block_labels_var(d: Optional[dict]) -> dict:
    if not d:
        return {}
    return {str(name): np.asarray(vals, dtype=np.int64) for name, vals in d.items()}


def _enc_block_labels_con(model: Model, rows: list) -> dict:
    """Per-row constraint block labels (``Model.set_constraint_block``, #1370).

    The store holds both name keys and ``id(constraint)`` keys pointing at the
    same labels (so a lookup succeeds from either handle). ``id`` is a
    process-local address, so it is written as a ROW INDEX -- and an id matching
    no row is refused rather than dropped, exactly as for a coupling mark.
    """
    store: dict = getattr(model, "_block_labels_con", {}) or {}
    row_of_id = {id(con): i for i, con in enumerate(rows)}
    names: dict[str, list[int]] = {}
    indexed: list[dict] = []
    for key, vals in store.items():
        payload = [int(v) for v in np.asarray(vals).reshape(-1)]
        if isinstance(key, str):
            names[key] = payload
        elif isinstance(key, int):
            idx = row_of_id.get(key)
            if idx is None:
                raise SerializationError(
                    "a constraint block label refers to a constraint object that is not "
                    "among this model's rows, so it cannot be written as a stable "
                    "reference. Re-declare it on the current model "
                    "(Model.set_constraint_block), or declare it by name."
                )
            indexed.append({"row": idx, "labels": payload})
        else:
            raise SerializationError(
                f"cannot serialize a constraint block label keyed by {type(key).__name__}; "
                "Model.set_constraint_block records a name string or a constraint object."
            )
    return {"names": names, "rows": sorted(indexed, key=lambda d: d["row"])}


def _dec_block_labels_con(d: Optional[dict], rows: list) -> dict:
    if not d:
        return {}
    out: dict = {}
    for name, vals in (d.get("names") or {}).items():
        out[str(name)] = np.asarray(vals, dtype=np.int64)
    row_refs = _StrictRefs(rows, "block-label row")
    for entry in d.get("rows", []):
        row = row_refs[entry["row"]]
        labels = np.asarray(entry["labels"], dtype=np.int64)
        out[id(row)] = labels
        cname = getattr(row, "name", None)
        if cname:
            out[cname] = labels
    return out


def _enc_simplex_lowerings(model: Model) -> list[dict]:
    out = []
    for rec in getattr(model, "_simplex_lowerings", []) or []:
        sizes = rec.sizes
        out.append(
            {
                "name": rec.name,
                "n_disjuncts": int(rec.n_disjuncts),
                "weight_names": list(rec.weight_names),
                "sizes": {
                    "disjunctions": int(sizes.disjunctions),
                    "cnf_clauses": int(sizes.cnf_clauses),
                    "literal_occurrences": int(sizes.literal_occurrences),
                    "weight_variables": int(sizes.weight_variables),
                    "rows": int(sizes.rows),
                },
            }
        )
    return out


def _dec_simplex_lowerings(docs: list[dict]) -> list:
    from discopt._relax.simplex_lowering import LoweringSizes, SimplexLoweringRecord

    return [
        SimplexLoweringRecord(
            name=d["name"],
            n_disjuncts=d["n_disjuncts"],
            weight_names=list(d["weight_names"]),
            sizes=LoweringSizes(**d["sizes"]),
        )
        for d in docs
    ]


def _enc_atan2_preconditions(model: Model, table) -> list[dict]:
    """The sign assumptions this model's rewritten ``atan2`` calls were built on.

    These must travel with the document. ``dm.atan2`` rewrites into ``atan`` of a
    sign-definite ratio using the bounds in force at build time, and the guard
    that catches a *later* widening of that bound lives here. Dropping it on load
    would leave a reloaded model rewritten but unguarded — and a widening applied
    after loading then differs from ``atan2`` by pi with no error, which is a
    false model, not a loose one. The denominator is written through the shared
    node table, so it aliases the same subexpression the rewrite already carries
    rather than duplicating it.
    """
    return [
        {"denominator": table.add(denominator), "sign": sign, "label": label}
        for denominator, sign, label in getattr(model, "_atan2_preconditions", []) or []
    ]


def _dec_atan2_preconditions(docs: Optional[list], nodes: list) -> list:
    """Rebuild the guard list. A document predating it simply has none."""
    out = []
    for d in docs or []:
        if d["sign"] not in ("positive", "negative"):
            # The guard reads anything but "positive" as "negative", so a corrupt
            # value would re-check the opposite sign of the one the rewrite assumed.
            raise SerializationError(f"atan2 precondition has unknown sign {d['sign']!r}")
        out.append((nodes[d["denominator"]], d["sign"], d["label"]))
    return out


def _enc_piecewise_domains(model: Model, table) -> list[dict]:
    """The input domains this model's ``piecewise`` calls were declared against.

    Same reason as the ``atan2`` guard: the lowered rows force each input into its
    breakpoint span, and the check that catches a bound widened past it after
    declaration lives here. Dropping it on load would let a reloaded model clamp
    a widened input silently (#1482). The input is written through the shared
    node table so it aliases the node the rows already carry.
    """
    return [
        {"input": table.add(expr), "lo": float(lo), "hi": float(hi), "label": label}
        for expr, lo, hi, label in getattr(model, "_piecewise_domains", []) or []
    ]


def _dec_piecewise_domains(docs: Optional[list], nodes: list) -> list:
    """Rebuild the domain guard list. A document predating it simply has none."""
    out = []
    for d in docs or []:
        lo, hi = _dec_float(d["lo"]), _dec_float(d["hi"])
        if not (lo <= hi):
            raise SerializationError(f"piecewise domain [{lo}, {hi}] is empty or NaN")
        out.append((nodes[d["input"]], lo, hi, d["label"]))
    return out


def _enc_state(model: Model, rows: list, table) -> dict:
    state: dict[str, Any] = {name: getattr(model, name) for name in _STATE_PLAIN}
    for name in _STATE_AS_SORTED_LIST:
        state[name] = sorted(getattr(model, name, set()) or set())
    state["_atan2_preconditions"] = _enc_atan2_preconditions(model, table)
    state["_piecewise_domains"] = _enc_piecewise_domains(model, table)
    state["_coupling_keys"] = _enc_coupling_keys(model, rows)
    state["_sets"] = _enc_sets(model)
    state["_simplex_lowerings"] = _enc_simplex_lowerings(model)
    state["_block_labels_var"] = _enc_block_labels_var(model)
    state["_block_labels_con"] = _enc_block_labels_con(model, rows)
    for name in _LAZY_PLAIN:
        if name in vars(model):
            state[name] = getattr(model, name)
    nl_source = _enc_nl_source(model)
    if nl_source is not None:
        state["nl_source"] = nl_source
    gams_iv = getattr(model, "_gams_initial_values", None)
    if gams_iv:
        state["_gams_initial_values"] = _enc_gams_initial_values(gams_iv)
    return state


def _enc_gams_initial_values(iv: dict) -> list:
    """``{name: value}`` (scalar) / ``{name: {flat_idx: value}}`` as JSON rows."""
    out: list = []
    for name in sorted(iv):
        entry = iv[name]
        if isinstance(entry, dict):
            out.append([name, [[int(k), _enc_float(float(v))] for k, v in sorted(entry.items())]])
        else:
            out.append([name, _enc_float(float(entry))])
    return out


def _dec_gams_initial_values(rows: Any, model: Model) -> None:
    if rows is None:
        return
    if not isinstance(rows, list):
        raise SerializationError("the '_gams_initial_values' entry must be a list")
    sizes = {v.name: v.size for v in model._variables}
    iv: dict = {}
    for row in rows:
        if not (isinstance(row, list) and len(row) == 2):
            raise SerializationError(f"invalid _gams_initial_values row: {row!r}")
        name = _dec_name(row[0], "_gams_initial_values")
        if name not in sizes:
            raise SerializationError(
                f"_gams_initial_values names variable {name!r}, which this document "
                "does not declare"
            )
        if name in iv:
            raise SerializationError(f"_gams_initial_values lists {name!r} twice")
        if isinstance(row[1], list):
            elems: dict = {}
            for pair in row[1]:
                if not (isinstance(pair, list) and len(pair) == 2):
                    raise SerializationError(f"invalid _gams_initial_values entry: {pair!r}")
                idx = _dec_index_int(pair[0], f"_gams_initial_values[{name!r}]")
                if not 0 <= idx < sizes[name] or idx in elems:
                    raise SerializationError(
                        f"_gams_initial_values entry {name!r}[{idx}] is out of range or repeated"
                    )
                elems[idx] = _dec_float(pair[1])
            iv[name] = elems
        else:
            iv[name] = _dec_float(row[1])
    setattr(model, "_gams_initial_values", iv)  # noqa: B010 - attached after __init__


def _enc_nl_source(model: Model) -> Optional[dict]:
    """What a ``from_nl`` model carries about its source file (see ``_LAZY_NL_SOURCE``)."""
    has_repr = getattr(model, "_nl_repr", None) is not None
    path = getattr(model, "_source_nl_path", None)
    if not has_repr and path is None:
        return None
    out: dict[str, Any] = {"parsed": has_repr}
    if path is not None:
        digest = getattr(model, "_source_nl_sha256", None)
        if digest is None:
            # A path with no digest cannot be checked on reload, so it is not written:
            # the reloaded model emits its own `.nl` instead, which is always correct.
            return out
        out["path"] = str(path)
        out["sha256"] = str(digest)
    return out


def _dec_nl_source(doc: Any, model: Model) -> None:
    if doc is None:
        return
    if not isinstance(doc, dict) or not isinstance(doc.get("parsed"), bool):
        raise SerializationError(f"invalid nl_source entry: {doc!r}")
    if doc["parsed"]:
        from discopt.modeling.core import _DetachedNlRepr

        setattr(model, "_nl_repr", _DetachedNlRepr())  # noqa: B010 - attached after __init__
    if "path" in doc or "sha256" in doc:
        path, digest = doc.get("path"), doc.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise SerializationError(f"invalid nl_source entry: {doc!r}")
        setattr(model, "_source_nl_path", path)  # noqa: B010 - attached after __init__
        setattr(model, "_source_nl_sha256", digest)  # noqa: B010


def _dec_state(state: Optional[dict], model: Model, rows: list, nodes: list) -> None:
    if not state:
        return
    if "_aux_counter" in state:
        model._aux_counter = _dec_index_int(state["_aux_counter"], "_aux_counter")
        if model._aux_counter < 0:
            raise SerializationError(f"_aux_counter is negative ({model._aux_counter})")
    for name in ("_decomp_stages", "_decomp_blocks"):
        if name in state:
            raw = state[name]
            if not isinstance(raw, dict):
                raise SerializationError(f"{name} must be an object, got {type(raw).__name__}")
            setattr(
                model,
                name,
                {_dec_name(k, name): _dec_index_int(v, f"{name}[{k!r}]") for k, v in raw.items()},
            )
    for name in _STATE_AS_SORTED_LIST:
        if name in state:
            setattr(model, name, set(state[name]))
    model._atan2_preconditions = _dec_atan2_preconditions(state.get("_atan2_preconditions"), nodes)
    model._piecewise_domains = _dec_piecewise_domains(state.get("_piecewise_domains"), nodes)
    model._coupling_keys = _dec_coupling_keys(state.get("_coupling_keys"), rows)
    model._block_labels_var = _dec_block_labels_var(state.get("_block_labels_var"))
    model._block_labels_con = _dec_block_labels_con(state.get("_block_labels_con"), rows)
    model._sets = _dec_sets(state.get("_sets", []))
    model._simplex_lowerings = _dec_simplex_lowerings(state.get("_simplex_lowerings", []))
    for name in _LAZY_PLAIN:
        if name in state:
            setattr(model, name, state[name])
    _dec_nl_source(state.get("nl_source"), model)
    _dec_gams_initial_values(state.get("_gams_initial_values"), model)


# ── JSON tree sanitising (for the embedded result payload) ─────────────────


def _enc_tree(obj: Any) -> Any:
    """Recursively tag non-finite floats so the document is standard JSON."""
    if isinstance(obj, float):
        return _enc_float(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return _enc_tree(obj.item())
    if isinstance(obj, dict):
        return {k: _enc_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_enc_tree(v) for v in obj]
    return obj


_FLOAT_TOKENS = {"nan", "inf", "-inf"}


def _dec_tree(obj: Any) -> Any:
    """Inverse of :func:`_enc_tree` -- ONLY for trees whose leaves are numbers.

    The bare ``"nan"``/``"inf"`` token is indistinguishable from a string of the
    same text, so every such string is read as a float. Use it where the position
    is known to be numeric (solution arrays), and :func:`_dec_json_tree` for a
    tree that can hold arbitrary strings (#1292).
    """
    if isinstance(obj, str) and obj in _FLOAT_TOKENS:
        return _dec_float(obj)
    if isinstance(obj, dict):
        return {k: _dec_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dec_tree(v) for v in obj]
    return obj


#: Unambiguous tagging for free-form trees (#1292): a non-finite float is written
#: as ``{"__float__": "nan"}``, which no string can be mistaken for. A genuine dict
#: whose only key is one of the two tags is wrapped as ``{"__dict__": {...}}`` so
#: it cannot be mistaken for a tag either.
_FLOAT_TAG = "__float__"
_DICT_TAG = "__dict__"


def _enc_json_tree(obj: Any) -> Any:
    """Encode a free-form JSON tree (strings, numbers, bools, None, lists, dicts)."""
    if isinstance(obj, float):
        v = _enc_float(obj)
        return {_FLOAT_TAG: v} if isinstance(v, str) else v
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return _enc_json_tree(obj.item())
    if isinstance(obj, np.ndarray):
        return _enc_json_tree(obj.tolist())
    if isinstance(obj, dict):
        enc = {k: _enc_json_tree(v) for k, v in obj.items()}
        if len(obj) == 1 and next(iter(obj)) in (_FLOAT_TAG, _DICT_TAG):
            return {_DICT_TAG: enc}
        return enc
    if isinstance(obj, (list, tuple)):
        return [_enc_json_tree(v) for v in obj]
    return obj


def _dec_json_tree(obj: Any) -> Any:
    """Inverse of :func:`_enc_json_tree`. Strings are never reinterpreted."""
    if isinstance(obj, dict):
        if len(obj) == 1 and _FLOAT_TAG in obj:
            token = obj[_FLOAT_TAG]
            if not isinstance(token, str) or token not in _FLOAT_TOKENS:
                raise SerializationError(f"invalid float tag {obj!r}")
            return _dec_float(token)
        if len(obj) == 1 and _DICT_TAG in obj:
            inner = obj[_DICT_TAG]
            if not isinstance(inner, dict):
                raise SerializationError(f"invalid dict tag {obj!r}")
            return {k: _dec_json_tree(v) for k, v in inner.items()}
        return {k: _dec_json_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dec_json_tree(v) for v in obj]
    return obj


# ── document ───────────────────────────────────────────────────────────────


def dumps(
    model: Model,
    *,
    result: Any = None,
    indent: Optional[int] = None,
    provenance: bool = True,
    author: Optional[str] = None,
) -> str:
    """Serialize *model* to a JSON string.

    Parameters
    ----------
    model : Model
        The model to write.
    result : SolveResult, optional
        A solve result to embed alongside the model, so the fitted state travels
        with the model it belongs to instead of desynchronising in a second file.
    indent : int, optional
        ``json.dumps`` indent. ``None`` (default) writes compactly; ``2`` is
        readable and diffs well.
    provenance : bool, default True
        Record who/when/with-what wrote the document (see :mod:`discopt.provenance`).
        Pass ``False`` for a byte-reproducible document: the block carries a
        timestamp, so two saves of the same model are otherwise not identical.
        Turning it off is a deliberate loss of provenance, not a default.
    author : str, optional
        Creator of the model, recorded in the provenance block. Never inferred --
        omitted unless passed here or via ``DISCOPT_PROVENANCE_AUTHOR``.

    Raises
    ------
    SerializationError
        If the model carries anything this version cannot write faithfully.
    """
    _refuse_unsupported(model)

    import discopt

    var_ids = {id(v): i for i, v in enumerate(model._variables)}
    param_ids = {id(p): i for i, p in enumerate(model._parameters)}
    table = _NodeTable(var_ids, param_ids)

    rows = _enc_rows(model, table)

    obj = model._objective
    objective = None
    if obj is not None:
        objective = {
            "sense": obj.sense.value,
            "expr": table.add(obj.expression),
            "placeholder": bool(getattr(obj, "_is_placeholder", False)),
        }

    builder_obj = None
    lin_blk = getattr(model, "_builder_linear_objective", None)
    quad_blk = getattr(model, "_builder_quadratic_objective", None)
    if lin_blk is not None:
        c, x, constant, sense = lin_blk
        builder_obj = {
            "kind": "linear",
            "c": _enc_array(np.asarray(c, dtype=np.float64)),
            "x": var_ids[id(x)],
            "constant": _enc_float(constant),
            "sense": sense,
        }
    elif quad_blk is not None:
        Q, c, x, constant, sense = quad_blk
        builder_obj = {
            "kind": "quadratic",
            "Q": _enc_csr(Q),
            "c": _enc_array(np.asarray(c, dtype=np.float64)),
            "x": var_ids[id(x)],
            "constant": _enc_float(constant),
            "sense": sense,
        }

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        # Kept alongside the richer `provenance` block: a 1.0-era reader looks for
        # this key, and dropping it would break documents this writer produces for
        # readers that predate provenance. `provenance.software.version` is the
        # same string; this is its backward-compatible alias, not a second source.
        "discopt": getattr(discopt, "__version__", None),
        "name": model.name,
        "variables": [_enc_variable(v) for v in model._variables],
        "parameters": [_enc_parameter(p) for p in model._parameters],
        "nodes": table.nodes,
        "objective": objective,
        "builder_objective": builder_obj,
        "rows": rows,
        "builder_blocks": _enc_builder_blocks(model, var_ids),
        "complementarities": _enc_complementarities(model, table),
        "state": _enc_state(model, list(model._constraints), table),
        "initial_point": [
            [name, int(elem), _enc_float(val)]
            for (name, elem), val in sorted(getattr(model, "_initial_point", {}).items())
        ],
    }

    if provenance:
        # `model.provenance` is set only by `loads`, so this records the document
        # this model was read from -- not a block this same call just made up.
        doc["provenance"] = _capture_provenance(
            author=author, previous=getattr(model, "provenance", None)
        )

    if result is not None:
        from discopt.result_io import serialize_result

        # Already JSON-safe, and each field is tagged by its own rule. A blanket
        # re-encode/decode here is what turned a string field reading "nan" into
        # a float on reload (#1292). Dropping it changes the format, which is why
        # the schema minor goes to 1.2 -- `loads` needs to know which of the two
        # encodings it is holding.
        doc["solution"] = serialize_result(result)

    # `allow_nan=False`: bare NaN/Infinity is not valid JSON, and every float has
    # already been routed through `_enc_float`, so this asserts that nothing
    # slipped past the encoders rather than silently writing a non-standard token.
    return json.dumps(doc, allow_nan=False, indent=indent)


def _refuse_json_constant(token: str) -> Any:
    raise SerializationError(
        f"the document contains a bare {token} token, which is not valid JSON; "
        "discopt writes non-finite numbers as the strings 'nan', 'inf' and '-inf'."
    )


def _unique_pairs(pairs: list) -> dict:
    """``json`` object hook: a plain dict, refusing a key repeated in one object."""
    d = dict(pairs)
    if len(d) != len(pairs):
        # `json` keeps the last of a repeated key: the others vanish unread.
        counts: dict = {}
        for k, _ in pairs:
            counts[k] = counts.get(k, 0) + 1
        dup = sorted(k for k, n in counts.items() if n > 1)
        raise SerializationError(
            f"the document repeats key(s) {dup} in one object; JSON keeps only the "
            "last, so the others would be dropped without a word."
        )
    return d


class _Record(dict):
    """A JSON object of a document being read, recording which keys were read.

    Every object outside the node table is turned into one of these before
    decoding (:func:`_track`), so afterwards a key the decoder never read is one
    it would have silently dropped (#1780). That is derived from the decoder
    itself -- per row ``kind``, per record -- so it cannot drift from it, and a
    field a new writer adds to a record is refused by an older reader instead of
    vanishing. Reading means ``[]``, ``get``, ``pop`` or iterating; ``in`` alone
    does not count.

    The node table -- most of any document -- is checked against
    ``_NODE_KEYS`` / ``_INDEX_KEYS`` instead: tracking every node cost +130 % on
    ``loads`` of a 24k-node document (measured); the tables cost a set difference
    per node, and ``test_serialize_versioning`` checks them against exactly what
    this decoder reads in tracking mode.
    """

    __slots__ = ("_read",)

    def __init__(self, pairs=()):
        super().__init__(pairs)
        self._read: set = set()

    def __getitem__(self, key):
        self._read.add(key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self._read.add(key)
        return dict.get(self, key, default)

    def pop(self, key, *default):
        self._read.add(key)
        return dict.pop(self, key, *default)

    def _all(self) -> None:
        self._read.update(dict.keys(self))

    def __iter__(self):
        self._all()
        return dict.__iter__(self)

    def keys(self):
        self._all()
        return dict.keys(self)

    def values(self):
        self._all()
        return dict.values(self)

    def items(self):
        self._all()
        return dict.items(self)


def _track(obj: Any) -> Any:
    """*obj* with every JSON object below it turned into a :class:`_Record`."""
    if isinstance(obj, dict):
        return _Record({k: _track(v) if isinstance(v, (dict, list)) else v for k, v in obj.items()})
    if isinstance(obj, list):
        return [_track(v) if isinstance(v, (dict, list)) else v for v in obj]
    return obj


def _plain(obj: Any) -> Any:
    """*obj* as plain dicts and lists, every key counted as read.

    For a subtree the decoder keeps verbatim (provenance): it is carried, so
    nothing in it is dropped, and the caller gets ordinary dicts back.
    """
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def _unread_keys(doc: dict) -> list[str]:
    """Paths of nested keys the decoder never read (top level: ``_check_keys``)."""
    out: list[str] = []
    # The node table is checked by `_NODE_KEYS` while it is decoded.
    stack: list = [(v, f"/{k}") for k, v in dict.items(doc) if isinstance(v, (_Record, list))]
    while stack:
        obj, path = stack.pop()
        if isinstance(obj, _Record):
            for k, v in dict.items(obj):
                if k not in obj._read:
                    out.append(f"{path}/{k}")
                elif isinstance(v, (_Record, list)):
                    stack.append((v, f"{path}/{k}"))
        else:
            stack.extend(
                (v, f"{path}[{i}]") for i, v in enumerate(obj) if isinstance(v, (_Record, list))
            )
    return sorted(out)


def loads(text: Union[str, bytes]) -> Model:
    """Rebuild a :class:`Model` from a JSON string produced by :func:`dumps`.

    The reloaded model carries the embedded solve result (if one was saved) on its
    ``saved_result`` attribute.

    Raises
    ------
    SerializationError
        If the text is not a well-formed discopt document. A malformed field the
        decoders below do not check by name -- a missing key, a value of the
        wrong type -- surfaces as this too, chained to the low-level error, so a
        caller can tell "this file is bad" from a defect in discopt.
    """
    # `dumps` never writes bare NaN/Infinity (non-finite floats are tagged strings),
    # so one here is not a discopt document; refuse it on read rather than accept
    # it and fail the next save (#1291).
    try:
        doc = json.loads(
            text, parse_constant=_refuse_json_constant, object_pairs_hook=_unique_pairs
        )
    except json.JSONDecodeError as exc:
        raise SerializationError(f"not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise SerializationError(f"a discopt document is a JSON object, got {type(doc).__name__}")
    try:
        return _loads_document(doc)
    except SerializationError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        raise SerializationError(
            f"malformed discopt document: {type(exc).__name__}: {exc}"
        ) from exc


def _loads_document(doc: dict) -> Model:
    schema = doc.get("schema")
    if not isinstance(schema, str) or not schema.startswith("discopt.model/"):
        raise SerializationError(f"not a discopt model document (schema={schema!r}).")
    # Split on "." so "discopt.model/1.2" is read as major 1 rather than refused as
    # a major called "1.2".
    version = schema.split("/", 1)[1]
    major, _, minor_text = version.partition(".")
    if major != str(SCHEMA_MAJOR):
        raise SerializationError(
            f"document schema {schema!r} is major version {major}, but this discopt "
            f"reads major version {SCHEMA_MAJOR}. Refusing rather than guessing."
        )
    # The minor decides how the solution subtree is decoded below, so it has to be
    # a number. A bare major ("discopt.model/1") is minor 0. Anything after the
    # minor is ignored rather than refused -- this reader must not be the thing
    # that stops a later "1.3.1" from loading -- but a minor that is not a number
    # is a malformed identifier, and guessing one would pick a decoder at random.
    minor_field = minor_text.partition(".")[0]
    if minor_field == "":
        minor = 0
    elif minor_field.isdigit():
        minor = int(minor_field)
    else:
        raise SerializationError(
            f"document schema {schema!r} has a non-numeric minor version "
            f"{minor_field!r}; refusing rather than guessing which format it is."
        )

    newer = minor > SCHEMA_MINOR
    for key, value in list(dict.items(doc)):
        if key not in ("nodes", "provenance") and not key.startswith(_EXTENSION_PREFIX):
            doc[key] = _track(value)
    _check_keys(doc, _TOP_LEVEL_KEYS, "top-level", schema, newer, extensions=True)
    if isinstance(doc.get("state"), dict):
        _check_keys(doc["state"], _STATE_KEYS, "'state'", schema, newer, extensions=False)

    if not newer:
        model = _decode_document(doc, minor)
        _refuse_unread(doc, schema, newer)
        return model
    try:
        model = _decode_document(doc, minor)
    except (SerializationError, KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        # A newer writer may use a node, row or index kind this reader has never
        # seen; the low-level refusal is right, but the remedy is not in it.
        raise SerializationError(
            f"document schema {schema!r} is newer than this reader ({SCHEMA}) and "
            f"holds content it cannot decode ({type(exc).__name__}: {exc}); upgrade "
            "discopt to read it."
        ) from exc
    _refuse_unread(doc, schema, newer)
    # 3: past this function and `loads`, to the caller's line.
    warnings.warn(
        f"document schema {schema!r} was written by a newer format than this reader's "
        f"({SCHEMA}). Everything in it is a key this reader knows and was loaded, but "
        "a newer writer may say more inside those keys than this reader can check; "
        "upgrade discopt to be sure.",
        SchemaVersionWarning,
        stacklevel=3,
    )
    return model


def _refuse_unread(doc: dict, schema: str, newer: bool) -> None:
    """Refuse a nested key the decoder did not read (#1780; see :class:`_Record`)."""
    unread = _unread_keys(doc)
    if not unread:
        return
    shown = ", ".join(unread[:5]) + (f" and {len(unread) - 5} more" if len(unread) > 5 else "")
    if newer:
        raise SerializationError(
            f"document schema {schema!r} is newer than this reader ({SCHEMA}) and has "
            f"key(s) this reader does not read ({shown}); refusing rather than dropping "
            "them. Upgrade discopt to read it."
        )
    raise SerializationError(
        f"document schema {schema!r} has key(s) no writer of that format produces "
        f"({shown}); the document is malformed or was edited by hand, and ignoring "
        "them could drop part of the model."
    )


def _check_keys(
    section: dict, known: dict, where: str, schema: str, newer: bool, *, extensions: bool
) -> None:
    """Refuse a key no writer produced (see *Versioning* in the module docstring)."""
    unknown = sorted(
        k
        for k in section
        if k not in known and not (extensions and k.startswith(_EXTENSION_PREFIX))
    )
    if not unknown:
        return
    if newer:
        raise SerializationError(
            f"document schema {schema!r} is newer than this reader ({SCHEMA}) and has "
            f"{where} key(s) {unknown} it does not know; refusing rather than dropping "
            "them. Upgrade discopt to read it."
        )
    hint = (
        f" (a key starting with {_EXTENSION_PREFIX!r} is an annotation and is ignored)"
        if extensions
        else ""
    )
    raise SerializationError(
        f"document schema {schema!r} has {where} key(s) {unknown} that no writer of that "
        f"format produces; the document is malformed or was edited by hand, and "
        f"ignoring them could drop part of the model{hint}."
    )


def _decode_document(doc: dict, minor: int) -> Model:
    model = Model(doc["name"])
    variables = _StrictRefs([_dec_variable(d, model) for d in doc["variables"]], "variable")
    params = [_dec_parameter(d, model) for d in doc["parameters"]]
    nodes = _decode_nodes(doc["nodes"], variables, params)

    model._constraints.extend(_dec_rows(doc["rows"], nodes, minor))
    for row in model._constraints:
        if type(row) is _DisjunctiveConstraint:
            row._model = model  # #1775 G1.1: so ``indicators`` can be read

    for d in doc.get("builder_blocks", []):
        model.add_linear_constraints(
            _dec_csr(d["A"]),
            variables[d["x"]],
            d["sense"],
            _dec_array(d["b"], (d["A"]["shape"][0],)),
            name=d["name"],
        )

    bobj = doc.get("builder_objective")
    if bobj is not None:
        # The builder objective lives in the Rust builder; `_objective` holds the
        # zero placeholder its setter installs, which the writer records too. Read
        # it to confirm it is exactly that: a real objective beside a builder one
        # would otherwise be dropped unread (#1780).
        od = doc.get("objective")
        if od is not None:
            if od.get("placeholder") is not True:
                raise SerializationError(
                    "the document has both a builder objective and a non-placeholder "
                    "'objective'; a model has one objective"
                )
            if od.get("sense") != ("minimize" if bobj["sense"] == "minimize" else "maximize"):
                raise SerializationError(
                    f"the placeholder objective's sense {od.get('sense')!r} disagrees with "
                    f"the builder objective's {bobj['sense']!r}"
                )
            _dec_index_int(od.get("expr"), "the placeholder objective's node")
        x = variables[bobj["x"]]
        c = _dec_array(bobj["c"], (np.asarray(bobj["c"]).size,))
        if bobj["kind"] == "linear":
            model.add_linear_objective(
                c, x, constant=_dec_float(bobj["constant"]), sense=bobj["sense"]
            )
        else:
            model.add_quadratic_objective(
                _dec_csr(bobj["Q"]),
                c,
                x,
                constant=_dec_float(bobj["constant"]),
                sense=bobj["sense"],
            )
    elif doc.get("objective") is not None:
        od = doc["objective"]
        model._objective = Objective(nodes[od["expr"]], ObjectiveSense(od["sense"]))
        if od.get("placeholder"):
            # Set through setattr: the flag is attached dynamically by the two
            # builder-objective setters, not declared on the dataclass.
            setattr(model._objective, "_is_placeholder", True)

    _dec_complementarities(doc.get("complementarities"), model, nodes, minor)
    _dec_state(doc.get("state"), model, list(model._constraints), nodes)

    # An entry naming a variable this document does not declare is dead weight
    # at best: `solve()` looks the point up by name, so the value is dropped
    # without a word, and the user's warm start is quietly gone (#1321).
    declared = {v.name: v.size for v in model._variables}
    initial_point = {}
    for name, elem, val in doc.get("initial_point", []):
        name = _dec_name(name, "initial_point")
        if name not in declared:
            raise SerializationError(
                f"initial_point names variable {name!r}, which this document does not declare"
            )
        idx = _dec_index_int(elem, f"initial_point element of {name!r}")
        if not 0 <= idx < declared[name]:
            raise SerializationError(
                f"initial_point entry {name!r}[{idx}] is out of range for a variable "
                f"of size {declared[name]}"
            )
        if (name, idx) in initial_point:
            raise SerializationError(f"initial_point lists {name!r}[{idx}] twice")
        value = _dec_float(val)
        if math.isnan(value):
            # `set_initial_point` refuses NaN; the `.nl` writer would emit it as is.
            raise SerializationError(f"initial_point entry {name!r}[{idx}] is NaN")
        initial_point[(name, idx)] = value
    model._initial_point = initial_point

    if doc.get("solution") is not None:
        from discopt.result_io import deserialize_result

        solution = doc["solution"]
        if minor < _SCHEMA_MINOR_UNTAGGED_SOLUTION:
            # Through 1.1 this subtree was written as `_enc_tree(serialize_result(r))`
            # -- a blanket tag over everything `serialize_result` had already encoded
            # by field. The blanket pass is what made a non-finite float legal under
            # `allow_nan=False` in the fields `serialize_result` left raw, above all
            # `mip_nlp_trace`: a `nan` in a 1.1 document is on disk as the STRING
            # "nan". `deserialize_result` decodes its own tags and has no legacy path
            # for that field, so without this the trace reloads holding the string --
            # and a re-save at 1.2 makes it permanent.
            #
            # This restores 1.1's decoder for 1.1 documents, which also restores its
            # #1292 flaw: a string field whose value really is "nan" comes back as a
            # float. That ambiguity is baked into the format those documents were
            # written in and cannot be resolved from the document alone -- both
            # readings are one string. 1.2 documents have no blanket pass and so no
            # ambiguity; this branch exists only for what is already on disk.
            solution = _dec_tree(solution)
            # `_dec_tree` rebuilds the subtree as plain dicts by iterating it, which
            # counts every key read; re-track it so the unread-key check (#1780)
            # still sees what `deserialize_result` leaves unread below.
            doc["solution"] = solution = _track(solution)
        model.saved_result = deserialize_result(solution)
    else:
        model.saved_result = None

    # Provenance is descriptive metadata: it is surfaced, never acted on. Reading
    # it cannot change a single coefficient of the model rebuilt above -- the same
    # rule the LLM layer follows, for the same reason.
    prov = doc.get("provenance")
    if prov is not None and not isinstance(prov, dict):
        raise SerializationError(
            f"the 'provenance' section must be an object, got {type(prov).__name__}."
        )
    model.provenance = None if prov is None else _plain(prov)

    message = skew_warning(prov)
    if message is not None:
        # 4: past this function, `_loads_document` and `loads`, to the caller's line.
        warnings.warn(message, ProvenanceSkewWarning, stacklevel=4)

    return model


def save(
    model: Model,
    path: Union[str, Path],
    *,
    result: Any = None,
    indent: Optional[int] = None,
    provenance: bool = True,
    author: Optional[str] = None,
) -> None:
    """Write *model* to *path*. Gzips when the path ends in ``.gz``.

    ``provenance`` and ``author`` are passed through to :func:`dumps`.
    """
    p = Path(path)
    text = dumps(model, result=result, indent=indent, provenance=provenance, author=author)
    if p.suffix == ".gz":
        with gzip.open(p, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        p.write_text(text, encoding="utf-8")


def load(path: Union[str, Path]) -> Model:
    """Read a model written by :func:`save`.

    Gzip is detected by magic bytes, not by the file name, so a document stays
    readable after someone renames or decompresses it.
    """
    raw = Path(path).read_bytes()
    if raw[:2] == _GZIP_MAGIC:
        raw = gzip.decompress(raw)
    return loads(raw.decode("utf-8"))
