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
    _SOSConstraint,
)

#: Format identifier written into every document.
SCHEMA = "discopt.model/1"

#: Major version this reader accepts. A document whose major differs is refused
#: rather than best-effort parsed.
SCHEMA_MAJOR = 1

_GZIP_MAGIC = b"\x1f\x8b"

__all__ = ["SCHEMA", "dumps", "loads", "save", "load", "SerializationError"]


class SerializationError(ValueError):
    """Raised when a model cannot be written faithfully, or a document cannot be read."""


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
    """Encode an ``IndexExpression`` index (int, tuple, slice, or a mix)."""
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
    if kind == "slice":
        return slice(obj["start"], obj["stop"], obj["step"])
    if kind == "tuple":
        return tuple(_dec_index(i) for i in obj["v"])
    raise SerializationError(f"unknown index kind {kind!r}")


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
                "axis": None if node.axis is None else int(node.axis),
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


def _decode_nodes(table: list[dict], variables: list[Variable], params: list[Parameter]):
    """Rebuild expression objects from the node table.

    A node's children always precede it (the writer emits post-order), so a single
    forward pass suffices and sharing is restored by construction: two parents
    referencing id ``k`` get the *same* object back.
    """
    built: list[Any] = []
    for i, nd in enumerate(table):
        op = nd.get("op")
        if op not in _KNOWN_OPS:
            raise SerializationError(
                f"node {i} has unknown op {op!r}. This file was written by a newer "
                "discopt than the one reading it; refusing rather than dropping the node."
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
            built.append(SumExpression(built[nd["a"]], nd["axis"]))
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


# ── variables / parameters ─────────────────────────────────────────────────


def _enc_variable(v: Variable) -> dict:
    return {
        "name": v.name,
        "type": v.var_type.value,
        "shape": list(v.shape),
        "lb": _enc_array(v.lb),
        "ub": _enc_array(v.ub),
    }


def _dec_variable(d: dict, model: Model) -> Variable:
    shape = tuple(d["shape"])
    lb = _dec_array(d["lb"], shape)
    ub = _dec_array(d["ub"], shape)
    # Built through the low-level constructor rather than `model.continuous/
    # integer/...`: those apply defaulting rules (and, for INTEGER, a UserWarning
    # plus a [0, 1e6] fallback for an unspecified bound). Reloading a saved model
    # must reproduce the bounds that were saved, exactly, with no re-defaulting.
    var = Variable(d["name"], VarType(d["type"]), shape, lb, ub, model)
    return model._register_variable(var)


def _enc_parameter(p: Parameter) -> dict:
    return {"name": p.name, "shape": list(p.value.shape), "value": _enc_array(p.value)}


def _dec_parameter(d: dict, model: Model) -> Parameter:
    value = _dec_array(d["value"], tuple(d["shape"]))
    param = Parameter(d["name"], value, model)
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
    if isinstance(obj, str) and obj in _FLOAT_TOKENS:
        return _dec_float(obj)
    if isinstance(obj, dict):
        return {k: _dec_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dec_tree(v) for v in obj]
    return obj


# ── document ───────────────────────────────────────────────────────────────


def dumps(model: Model, *, result: Any = None, indent: Optional[int] = None) -> str:
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
        "initial_point": [
            [name, int(elem), _enc_float(val)]
            for (name, elem), val in sorted(getattr(model, "_initial_point", {}).items())
        ],
    }

    if result is not None:
        from discopt.result_io import serialize_result

        doc["solution"] = _enc_tree(serialize_result(result))

    # `allow_nan=False`: bare NaN/Infinity is not valid JSON, and every float has
    # already been routed through `_enc_float`, so this asserts that nothing
    # slipped past the encoders rather than silently writing a non-standard token.
    return json.dumps(doc, allow_nan=False, indent=indent)


def loads(text: Union[str, bytes]) -> Model:
    """Rebuild a :class:`Model` from a JSON string produced by :func:`dumps`.

    The reloaded model carries the embedded solve result (if one was saved) on its
    ``saved_result`` attribute.
    """
    doc = json.loads(text)

    schema = doc.get("schema")
    if not isinstance(schema, str) or not schema.startswith("discopt.model/"):
        raise SerializationError(f"not a discopt model document (schema={schema!r}).")
    major = schema.split("/", 1)[1]
    if major != str(SCHEMA_MAJOR):
        raise SerializationError(
            f"document schema {schema!r} is major version {major}, but this discopt "
            f"reads major version {SCHEMA_MAJOR}. Refusing rather than guessing."
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

    model._initial_point = {
        (name, int(elem)): _dec_float(val) for name, elem, val in doc.get("initial_point", [])
    }

    if doc.get("solution") is not None:
        from discopt.result_io import deserialize_result

        model.saved_result = deserialize_result(_dec_tree(doc["solution"]))
    else:
        model.saved_result = None

    return model


def save(
    model: Model,
    path: Union[str, Path],
    *,
    result: Any = None,
    indent: Optional[int] = None,
) -> None:
    """Write *model* to *path*. Gzips when the path ends in ``.gz``."""
    p = Path(path)
    text = dumps(model, result=result, indent=indent)
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
