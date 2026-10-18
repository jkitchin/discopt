"""Hypothesis strategies that generate whole discopt models (WP-4; reusable by WP-6).

Plan: ``docs/dev/serialization-completion-plan.md`` WP-4. :func:`models` draws a
model out of every part of the native format, to property-test the format, not
the solver:

* variables of every ``VarType``, shapes ``()``, ``(n,)`` and ``(n, m)``, with
  finite, infinite and sentinel (``1e20``) bounds; scalar and array parameters;
* expression trees over every binary and unary operator and every function name a
  ``FunctionCall`` may carry (``serialize._known_funcs()``), every index kind
  (int, slice, tuple, bool -- #1290), sums with and without an axis, sums over a
  list, matrix products, constants (scalar and array), and shared subtrees;
* algebraic rows, indicator / SOS / disjunctive relations, complementarity (lowered
  on declaration), propositional logic (schema 1.5), a fast-API linear block, and a
  scoped fix that is open while the model is written.

``models()`` yields :class:`GeneratedModel`; ``fixed`` is the ``(variable, value)``
pair to hold fixed while writing, or ``None``. :func:`ops_of` lists the node ops and
function names a model's document actually contains -- what the coverage test
counts, read from the written document rather than from the generator's intent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import discopt.modeling as dm
import numpy as np
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    FunctionCall,
    MatMulExpression,
    Model,
    SumOverExpression,
    UnaryOp,
    VarType,
)
from hypothesis import strategies as st

#: Function names by arity. Everything in ``serialize._known_funcs()`` must be in
#: one of these (``test_strategy_covers_the_op_table`` checks the union).
UNARY_FUNCS = (
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "cos",
    "cosh",
    "entropy",
    "erf",
    "exp",
    "log",
    "log10",
    "log1p",
    "log2",
    "prod",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "softplus",
    "sqrt",
    "sum",
    "tan",
    "tanh",
)
BINARY_FUNCS = ("atan2", "centropy", "max", "min", "signpower")
BINARY_OPS = ("+", "-", "*", "/", "**")
UNARY_OPS = ("neg", "abs")

_BOUND = st.one_of(
    st.floats(-50.0, 50.0, allow_nan=False, width=64),
    st.sampled_from([0.0, 1.0, -1.0, 1e20, -1e20, 0.1, 2.5e-7]),
)


@dataclass
class GeneratedModel:
    model: Model
    fixed: Optional[tuple]

    def __repr__(self) -> str:  # hypothesis prints this on failure
        from discopt.serialize import dumps

        try:
            return f"GeneratedModel({dumps(self.model, provenance=False)[:2000]})"
        except Exception as exc:  # noqa: BLE001 - a repr must not raise
            return f"GeneratedModel(<undumpable: {exc}>)"


@st.composite
def _bounds(draw, shape: tuple, var_type: VarType):
    if var_type is VarType.BINARY:
        return None, None
    size = int(np.prod(shape)) if shape else 1
    lo = np.array([draw(_BOUND) for _ in range(size)], dtype=float)
    width = np.array([draw(st.floats(0.0, 20.0, allow_nan=False)) for _ in range(size)])
    hi = lo + width
    if draw(st.booleans()):
        lo[draw(st.integers(0, size - 1))] = -np.inf
    if draw(st.booleans()):
        hi[draw(st.integers(0, size - 1))] = np.inf
    if var_type is VarType.INTEGER:
        lo, hi = np.floor(np.clip(lo, -1e6, 1e6)), np.ceil(np.clip(hi, -1e6, 1e6))
    if not shape:
        return float(lo[0]), float(hi[0])
    return lo.reshape(shape), hi.reshape(shape)


@st.composite
def _variable(draw, m: Model, k: int):
    var_type = draw(st.sampled_from([VarType.CONTINUOUS, VarType.INTEGER, VarType.BINARY]))
    shape = draw(st.sampled_from([(), (2,), (3,), (2, 2), (2, 3)]))
    lb, ub = draw(_bounds(shape, var_type))
    name = f"v{k}"
    if var_type is VarType.BINARY:
        return m.binary(name, shape=shape)
    if var_type is VarType.INTEGER:
        return m.integer(name, shape=shape, lb=lb, ub=ub)
    return m.continuous(name, shape=shape, lb=lb, ub=ub)


@st.composite
def _scalar_leaf(draw, variables: list, params: list):
    """A scalar expression drawn from a variable element, a parameter or a constant,
    reaching every index kind on the way."""
    kind = draw(st.sampled_from(["var", "var", "var", "param", "const"]))
    if kind == "const" or (kind == "param" and not params):
        return Constant(draw(st.floats(-10.0, 10.0, allow_nan=False)))
    if kind == "param":
        p = draw(st.sampled_from(params))
        return p if p.shape == () else p[draw(st.integers(0, p.shape[0] - 1))]
    v = draw(st.sampled_from(variables))
    if v.shape == ():
        return v
    if len(v.shape) == 1:
        how = draw(st.sampled_from(["int", "slice", "bool"]))
        if how == "int":
            return v[draw(st.integers(0, v.shape[0] - 1))]
        if how == "slice":
            return dm.sum(v[0 : draw(st.integers(1, v.shape[0]))])
        # numpy semantics: ``v[True]`` prepends an axis of length 1 (#1290).
        return v[True][0][draw(st.integers(0, v.shape[0] - 1))]
    how = draw(st.sampled_from(["tuple", "slice_tuple", "axis"]))
    if how == "tuple":
        return v[draw(st.integers(0, v.shape[0] - 1)), draw(st.integers(0, v.shape[1] - 1))]
    if how == "slice_tuple":
        return dm.sum(v[0:1, :])
    return dm.sum(v, axis=draw(st.sampled_from([0, 1])))[0]


@st.composite
def _expr(draw, variables: list, params: list, shared: list, depth: int):
    if depth <= 0 or draw(st.integers(0, 4)) == 0:
        if shared and draw(st.integers(0, 3)) == 0:
            return draw(st.sampled_from(shared))
        return draw(_scalar_leaf(variables, params))
    # Weighted toward function calls: there are 31 names to reach and one draw each.
    kind = draw(
        st.sampled_from(
            ["binop", "binop", "unop", "func", "func", "func", "func2", "sum_over", "matmul"]
        )
    )
    sub = _expr(variables, params, shared, depth - 1)
    if kind == "binop":
        out: Any = BinaryOp(draw(st.sampled_from(BINARY_OPS)), draw(sub), draw(sub))
    elif kind == "unop":
        out = UnaryOp(draw(st.sampled_from(UNARY_OPS)), draw(sub))
    elif kind == "func":
        out = FunctionCall(draw(st.sampled_from(UNARY_FUNCS)), draw(sub))
    elif kind == "func2":
        out = FunctionCall(draw(st.sampled_from(BINARY_FUNCS)), draw(sub), draw(sub))
    elif kind == "sum_over":
        out = SumOverExpression([draw(sub) for _ in range(draw(st.integers(2, 4)))])
    else:
        vectors = [v for v in variables if len(v.shape) == 1]
        if not vectors:
            return draw(sub)
        v = draw(st.sampled_from(vectors))
        n = v.shape[0]
        A = np.array(
            [[draw(st.floats(-3.0, 3.0, allow_nan=False)) for _ in range(n)] for _ in range(2)]
        )
        out = MatMulExpression(Constant(A), v)[draw(st.integers(0, 1))]
    if draw(st.integers(0, 3)) == 0:
        shared.append(out)
    return out


def _scalar(e: Any) -> Any:
    """*e* reduced to a scalar if broadcasting left it an array."""
    return e if tuple(getattr(e, "shape", ())) == () else dm.sum(e)


@st.composite
def models(draw, *, relations: bool = True) -> GeneratedModel:
    m = dm.Model(f"gen{draw(st.integers(0, 999))}")
    variables = [draw(_variable(m, k)) for k in range(draw(st.integers(1, 4)))]
    params = []
    for k in range(draw(st.integers(0, 2))):
        if draw(st.booleans()):
            params.append(m.parameter(f"p{k}", value=draw(st.floats(-5.0, 5.0, allow_nan=False))))
        else:
            params.append(
                m.parameter(
                    f"p{k}",
                    value=np.array([draw(st.floats(-5.0, 5.0, allow_nan=False)) for _ in range(2)]),
                )
            )
    shared: list = []
    expr = _expr(variables, params, shared, draw(st.integers(1, 3))).map(_scalar)
    if draw(st.booleans()):
        m.minimize(draw(expr))
    else:
        m.maximize(draw(expr))
    for k in range(draw(st.integers(0, 3))):
        sense = draw(st.sampled_from(["<=", ">=", "=="]))
        body, rhs = draw(expr), draw(st.floats(-20.0, 20.0, allow_nan=False))
        row = {"<=": body <= rhs, ">=": body >= rhs, "==": body == rhs}[sense]
        m.subject_to(row, name=draw(st.sampled_from([f"c{k}", None])))

    fixed = None
    if relations:
        binaries = [v for v in variables if v.var_type is VarType.BINARY and v.shape == ()]
        if draw(st.booleans()):
            z = binaries[0] if binaries else m.binary("z_ind")
            m.if_then(z, [draw(expr) <= 5.0], name="ind")
        vectors = [v for v in variables if len(v.shape) == 1]
        if vectors and draw(st.booleans()):
            v = draw(st.sampled_from(vectors))
            m.sos1([v[i] for i in range(v.shape[0])], name="sos")
        if draw(st.booleans()):
            m.either_or([[draw(expr) <= 1.0], [draw(expr) >= 3.0]], name="disj")
        if draw(st.booleans()):
            p = m.continuous("cp", lb=0.0, ub=10.0)
            q = m.continuous("cq", lb=0.0, ub=10.0)
            m.complementarity(p, q, name="pq")
        if draw(st.booleans()):
            Y = m.boolean("Y", shape=(3,))
            formula = draw(
                st.sampled_from(
                    [
                        lambda: Y[0].implies(Y[1] | ~Y[2]),
                        lambda: Y[0].equivalent_to(Y[1] & Y[2]),
                        lambda: dm.atleast(2, [Y[0], Y[1], Y[2]]),
                        lambda: dm.exactly(1, [Y[0], Y[2]]) & ~Y[1],
                    ]
                )
            )
            m.logical(formula(), name="logic")
        if draw(st.integers(0, 3)) == 0:
            import scipy.sparse as sp

            x = m.continuous("blk", shape=(3,), lb=0.0, ub=4.0)
            A = sp.csr_matrix(np.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0]]))
            m.add_linear_constraints(A, x, "<=", np.array([3.0, 1.0]), name="lin")
        continuous = [v for v in m._variables if v.var_type is VarType.CONTINUOUS and v.shape == ()]
        finite = [v for v in continuous if np.isfinite(v.lb) and np.isfinite(v.ub)]
        if finite and draw(st.integers(0, 3)) == 0:
            v = draw(st.sampled_from(finite))
            fixed = (v, float((float(v.lb) + float(v.ub)) / 2.0))
    return GeneratedModel(m, fixed)


def dumps_generated(g: GeneratedModel, **kwargs) -> str:
    """``dumps`` of a generated model, inside its fix scope when it has one."""
    from discopt.serialize import dumps

    kwargs.setdefault("provenance", False)
    if g.fixed is None:
        return dumps(g.model, **kwargs)
    var, value = g.fixed
    with var.fixed(value):
        return dumps(g.model, **kwargs)


def ops_of(text: str) -> set[str]:
    """Node ops, operator names, function names, index kinds and row kinds in a
    written document: ``op:binop``, ``bin:+``, ``fn:exp``, ``ix:slice``, ``row:sos``."""
    doc = json.loads(text)
    seen: set[str] = set()
    for nd in doc["nodes"]:
        seen.add(f"op:{nd['op']}")
        if nd["op"] == "binop":
            seen.add(f"bin:{nd['o']}")
        elif nd["op"] == "unop":
            seen.add(f"un:{nd['o']}")
        elif nd["op"] == "call":
            seen.add(f"fn:{nd['f']}")
        elif nd["op"] == "index":
            stack = [nd["index"]]
            while stack:
                ix = stack.pop()
                seen.add(f"ix:{ix['k']}")
                if ix["k"] == "tuple":
                    stack.extend(ix["v"])
        elif nd["op"] == "sum" and nd.get("axis") is not None:
            seen.add("sum:axis")
    for row in doc["rows"]:
        seen.add(f"row:{row['kind']}")
    if doc.get("complementarities"):
        seen.add("rel:complementarity")
    if doc.get("builder_blocks"):
        seen.add("rel:builder_block")
    return seen
