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
blocks and builder objective, indicator / SOS / disjunctive relations,
complementarity relations together with this model's lowering marks, the starting
point, and -- optionally -- the solve result.

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
rule): :class:`~discopt.modeling.core.CustomCall` (an arbitrary Python callable;
there is nothing faithful to write, and writing its *name* would reload into a
different model), propositional ``_LogicalConstraint``, and any expression node
this version does not know. On read, an unknown ``op`` is a hard error -- a reader
that skips what it does not understand silently drops a constraint and returns a
model that solves to the wrong answer.

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
import math
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    CustomCall,
    Expression,
    FunctionCall,
    IndexExpression,
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
    _readonly_bound,
    _SOSConstraint,
)
from discopt.provenance import capture as _capture_provenance
from discopt.provenance import skew_warning

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
SCHEMA = "discopt.model/1.2"

#: Major version this reader accepts. A document whose major differs is refused
#: rather than best-effort parsed.
SCHEMA_MAJOR = 1

#: Minor at which the blanket ``_enc_tree`` over the solution subtree was dropped
#: (#1292). Documents below it need that encoding undone on read.
_SCHEMA_MINOR_UNTAGGED_SOLUTION = 2

_GZIP_MAGIC = b"\x1f\x8b"

__all__ = [
    "SCHEMA",
    "dumps",
    "loads",
    "save",
    "load",
    "SerializationError",
    "ProvenanceSkewWarning",
]


class SerializationError(ValueError):
    """Raised when a model cannot be written faithfully, or a document cannot be read."""


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
    return float(x)


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
    if kind == "int":
        return int(obj["v"])
    if kind == "bool":
        if not isinstance(obj["v"], bool):
            raise SerializationError(f"bool index holds {obj['v']!r}")
        return obj["v"]
    if kind == "slice":
        return slice(obj["start"], obj["stop"], obj["step"])
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
            self.nodes.append(self._encode(node))
        return self._seen[id(expr)]

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


def _decode_nodes(table: list[dict], variables: list[Variable], params: list[Parameter]):
    """Rebuild expression objects from the node table.

    A node's children always precede it (the writer emits post-order), so a single
    forward pass suffices and sharing is restored by construction: two parents
    referencing id ``k`` get the *same* object back.
    """
    known_funcs = _known_funcs()
    built: list[Any] = []
    for i, nd in enumerate(table):
        op = nd.get("op")
        if op not in _KNOWN_OPS:
            raise SerializationError(
                f"node {i} has unknown op {op!r}. This file was written by a newer "
                "discopt than the one reading it; refusing rather than dropping the node."
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
                built.append(Constant(_dec_array(nd["v"], tuple(shape))))
        elif op == "var":
            built.append(variables[nd["ref"]])
        elif op == "param":
            built.append(params[nd["ref"]])
        elif op == "index":
            built.append(IndexExpression(built[nd["base"]], _dec_index(nd["index"])))
        elif op == "binop":
            built.append(BinaryOp(nd["o"], built[nd["a"]], built[nd["b"]]))
        elif op == "unop":
            built.append(UnaryOp(nd["o"], built[nd["a"]]))
        elif op == "call":
            built.append(FunctionCall(nd["f"], *[built[a] for a in nd["args"]]))
        elif op == "matmul":
            built.append(MatMulExpression(built[nd["a"]], built[nd["b"]]))
        elif op == "sum":
            built.append(SumExpression(built[nd["a"]], _dec_axis(nd["axis"])))
        elif op == "sum_over":
            built.append(SumOverExpression([built[a] for a in nd["args"]]))
    return built


# ── relations we refuse, by name ───────────────────────────────────────────

_REFUSED_RELATIONS = {
    "_LogicalConstraint": (
        "a propositional logic constraint",
        "This format version carries algebraic rows, indicator, SOS and disjunctive "
        "relations. Reformulate the logic into those before saving.",
    ),
}


def _refuse_unsupported(model: Model) -> None:
    """Refuse, by name, anything this version cannot write faithfully."""
    for con in list(model._constraints):
        tname = type(con).__name__
        if tname in _REFUSED_RELATIONS:
            what, remedy = _REFUSED_RELATIONS[tname]
            name = getattr(con, "name", None)
            where = f" named {name!r}" if name else ""
            raise SerializationError(f"cannot serialize: the model carries {what}{where}. {remedy}")
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
    shape = tuple(d["shape"])
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
        else:
            raise SerializationError(
                f"cannot serialize a row of type {t.__name__}; this format version "
                "carries algebraic constraints and indicator / SOS / disjunctive "
                "relations."
            )
    return rows


def _dec_rows(docs: list[dict], nodes: list) -> list:
    """Rebuild the ordered, heterogeneous ``_constraints`` list."""
    from discopt.modeling.core import DisjunctionSemantics

    out: list = []
    for d in docs:
        kind = d.get("kind")
        if kind == "algebraic":
            out.append(_dec_constraint(d, nodes))
        elif kind == "indicator":
            out.append(
                _IndicatorConstraint(
                    indicator=nodes[d["indicator"]],
                    constraint=_dec_constraint(d["constraint"], nodes),
                    active_value=d["active_value"],
                    name=d["name"],
                )
            )
        elif kind == "sos":
            out.append(
                _SOSConstraint(
                    sos_type=d["sos_type"],
                    variables=[nodes[i] for i in d["variables"]],
                    name=d["name"],
                )
            )
        elif kind == "disjunction":
            out.append(
                _DisjunctiveConstraint(
                    disjuncts=[[_dec_constraint(c, nodes) for c in dd] for dd in d["disjuncts"]],
                    name=d["name"],
                    method=d["method"],
                    semantics=DisjunctionSemantics[d["semantics"]],
                )
            )
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
        # `rows` is deliberately not encoded: it holds Constraint objects of THIS
        # model, and `LoweringRecord` documents `None` ("not tracked for this
        # model") as what a rebuilding pass must record, which is what a reload is.
        "lowered": [[index_of[id(rel)], record.method] for rel, record in lowered_map.items()],
    }


def _dec_complementarities(doc: Optional[dict], model: Model, nodes: list) -> None:
    """Rebuild the relation set and re-apply this model's lowering marks."""
    if not doc:
        return
    from discopt.mpec import Complementarity, ComplementarityRole, LoweringRecord

    def dec_shape(v) -> Optional[tuple]:
        return None if v is None else tuple(int(x) for x in v)

    rebuilt: list[Any] = []
    for d in doc["relations"]:
        src_idx = d["source"]
        if src_idx is not None and src_idx >= len(rebuilt):
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

    model._complementarities = [rebuilt[i] for i in doc["declared"]]
    model._lowered_complementarities = {
        rebuilt[i]: LoweringRecord(method=method, rows=None) for i, method in doc["lowered"]
    }


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

#: Attributes carried verbatim in the "state" section (plain JSON-safe values).
_STATE_PLAIN = ("_aux_counter", "_decomp_stages", "_decomp_blocks")

#: Attributes carried in "state" as a sorted list (a set is not JSON).
_STATE_AS_SORTED_LIST = ("_zero_spanning_factor_auxes",)

#: Attributes carried in "state" by a bespoke encoder.
_STATE_BESPOKE = (
    "_atan2_preconditions",
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
    out: set = set(d.get("names", []))
    for idx in d.get("row_indices", []):
        if idx >= len(rows):
            raise SerializationError(
                f"a coupling mark refers to row {idx}, but the model has {len(rows)} rows."
            )
        out.add(id(rows[idx]))
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
    for entry in d.get("rows", []):
        idx = int(entry["row"])
        if idx >= len(rows):
            raise SerializationError(
                f"a constraint block label refers to row {idx}, but the model has {len(rows)} rows."
            )
        labels = np.asarray(entry["labels"], dtype=np.int64)
        out[id(rows[idx])] = labels
        cname = getattr(rows[idx], "name", None)
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
    return [(nodes[d["denominator"]], d["sign"], d["label"]) for d in docs or []]


def _enc_state(model: Model, rows: list, table) -> dict:
    state: dict[str, Any] = {name: getattr(model, name) for name in _STATE_PLAIN}
    for name in _STATE_AS_SORTED_LIST:
        state[name] = sorted(getattr(model, name, set()) or set())
    state["_atan2_preconditions"] = _enc_atan2_preconditions(model, table)
    state["_coupling_keys"] = _enc_coupling_keys(model, rows)
    state["_sets"] = _enc_sets(model)
    state["_simplex_lowerings"] = _enc_simplex_lowerings(model)
    state["_block_labels_var"] = _enc_block_labels_var(model)
    state["_block_labels_con"] = _enc_block_labels_con(model, rows)
    return state


def _dec_state(state: Optional[dict], model: Model, rows: list, nodes: list) -> None:
    if not state:
        return
    for name in _STATE_PLAIN:
        if name in state:
            setattr(model, name, state[name])
    for name in _STATE_AS_SORTED_LIST:
        if name in state:
            setattr(model, name, set(state[name]))
    model._atan2_preconditions = _dec_atan2_preconditions(state.get("_atan2_preconditions"), nodes)
    model._coupling_keys = _dec_coupling_keys(state.get("_coupling_keys"), rows)
    model._block_labels_var = _dec_block_labels_var(state.get("_block_labels_var"))
    model._block_labels_con = _dec_block_labels_con(state.get("_block_labels_con"), rows)
    model._sets = _dec_sets(state.get("_sets", []))
    model._simplex_lowerings = _dec_simplex_lowerings(state.get("_simplex_lowerings", []))


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


def loads(text: Union[str, bytes]) -> Model:
    """Rebuild a :class:`Model` from a JSON string produced by :func:`dumps`.

    The reloaded model carries the embedded solve result (if one was saved) on its
    ``saved_result`` attribute.
    """
    # `dumps` never writes bare NaN/Infinity (non-finite floats are tagged strings),
    # so one here is not a discopt document; refuse it on read rather than accept
    # it and fail the next save (#1291).
    doc = json.loads(text, parse_constant=_refuse_json_constant)

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

    model = Model(doc["name"])
    variables = [_dec_variable(d, model) for d in doc["variables"]]
    params = [_dec_parameter(d, model) for d in doc["parameters"]]
    nodes = _decode_nodes(doc["nodes"], variables, params)

    model._constraints.extend(_dec_rows(doc["rows"], nodes))

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

    _dec_complementarities(doc.get("complementarities"), model, nodes)
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
        idx = int(elem)
        if not 0 <= idx < declared[name]:
            raise SerializationError(
                f"initial_point entry {name!r}[{idx}] is out of range for a variable "
                f"of size {declared[name]}"
            )
        initial_point[(name, idx)] = _dec_float(val)
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
    model.provenance = prov

    message = skew_warning(prov)
    if message is not None:
        warnings.warn(message, ProvenanceSkewWarning, stacklevel=2)

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
