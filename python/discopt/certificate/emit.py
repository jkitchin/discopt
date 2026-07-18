"""Build a Tier-1 (feasibility) certificate from a solved model.

The emitter walks the modeling-API expression DAG and the ``SolveResult``
incumbent, restating both in the exact-rational schema of :mod:`.schema`. It
reads *only* the public ``Model`` and ``SolveResult`` -- never a solver internal
-- so enabling it cannot change any bound, node count, or objective (it is
bound-neutral by construction).

Design stance (CLAUDE.md #3, "prefer the hard, right fix over the band-aid"):
the walker **refuses loudly** on any node it cannot faithfully and soundly
represent (array-valued algebra, reductions, opaque callables, parameters),
rather than silently emitting something a checker might wrongly accept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    FunctionCall,
    IndexExpression,
    Model,
    SolveResult,
    UnaryOp,
    Variable,
)

from .schema import CERTIFICATE_SCHEMA_VERSION, Rational, to_rational

# Default feasibility / integrality tolerances (mirror ``conftest.py``: abs=1e-6,
# integrality=1e-5). Emitted into the certificate so the checker uses the same
# slack the solver was held to.
DEFAULT_FEAS_TOL = 1e-6
DEFAULT_INT_TOL = 1e-5

_BINOP = {"+": "add", "-": "sub", "*": "mul", "/": "div", "**": "pow"}
_UNOP = {"neg": "neg", "abs": "abs"}
# Sense/type fields may be stored as an enum member or already as the enum's
# string value; normalize on the value string so either form works.
_CSENSE = {"<=": "le", ">=": "ge", "==": "eq"}
_VTYPE = {"continuous": "continuous", "binary": "binary", "integer": "integer"}


def _enum_value(v: object) -> object:
    """The ``.value`` of an Enum member, else ``v`` unchanged."""
    return getattr(v, "value", v)


class CertificateError(Exception):
    """Raised when a model/result cannot be soundly encoded as a certificate."""


# ── variable flattening ──────────────────────────────────────────────────────
def _flatten_columns(model: Model) -> tuple[list[dict], dict[int, int]]:
    """Flatten every variable block into scalar columns, in ``_variables`` order.

    Returns ``(columns, offsets)`` where ``columns[j]`` is
    ``{name, type, lb, ub}`` for flat column ``j`` and ``offsets[id(var)]`` is the
    starting column of that variable block (so ``var`` scalar -> ``offsets[id]``,
    and ``var[i]`` -> ``offsets[id] + i``).
    """
    columns: list[dict] = []
    offsets: dict[int, int] = {}
    for var in model._variables:
        offsets[id(var)] = len(columns)
        lb = np.asarray(var.lb, dtype=np.float64).ravel()
        ub = np.asarray(var.ub, dtype=np.float64).ravel()
        vtype = _VTYPE.get(_enum_value(var.var_type))
        if vtype is None:
            raise CertificateError(f"variable {var.name!r} has unknown type {var.var_type!r}")
        n = var.size
        for k in range(n):
            suffix = "" if n == 1 else f"[{k}]"
            columns.append(
                {
                    "name": f"{var.name}{suffix}",
                    "type": vtype,
                    "lb": to_rational(lb[k]),
                    "ub": to_rational(ub[k]),
                }
            )
    return columns, offsets


def _column_of(node: Union[Variable, IndexExpression], offsets: dict[int, int]) -> int:
    """Resolve a scalar variable reference to its flat column index."""
    if isinstance(node, Variable):
        if node.size != 1:
            raise CertificateError(
                f"variable {node.name!r} of size {node.size} used without indexing; "
                "array-valued expressions are not supported by the Tier-1 emitter"
            )
        return offsets[id(node)]
    # IndexExpression
    base = node.base
    if not isinstance(base, Variable):
        raise CertificateError(
            "only direct indexing of a variable (x[i]) is supported, "
            f"got index into {type(base).__name__}"
        )
    shape = base.shape or (base.size,)
    try:
        flat = int(np.ravel_multi_index(_as_index_tuple(node.index, shape), shape))
    except (ValueError, TypeError) as exc:
        raise CertificateError(
            f"cannot resolve index {node.index!r} into {base.name!r} shape {shape}: {exc}"
        ) from exc
    return offsets[id(base)] + flat


def _as_index_tuple(index: Any, shape: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(index, tuple):
        idx = index
    else:
        idx = (index,)
    if any(not isinstance(i, (int, np.integer)) for i in idx):
        raise CertificateError(f"non-integer index {index!r} (slices/fancy indexing unsupported)")
    if len(idx) != len(shape):
        raise CertificateError(f"index {index!r} rank does not match shape {shape}")
    return tuple(int(i) for i in idx)


# ── expression serialization ─────────────────────────────────────────────────
def _serialize_expr(node: Any, offsets: dict[int, int]) -> dict:
    """Tagged-dict encoding of an expression node (see schema §expression)."""
    if isinstance(node, Constant):
        val = np.asarray(node.value)
        if val.ndim != 0:
            raise CertificateError("array constants are not supported by the Tier-1 emitter")
        r = to_rational(float(val))
        if r is None:
            raise CertificateError("non-finite constant in expression")
        return {"k": "const", "v": r}
    if isinstance(node, Variable):
        return {"k": "var", "i": _column_of(node, offsets)}
    if isinstance(node, IndexExpression):
        return {"k": "var", "i": _column_of(node, offsets)}
    if isinstance(node, BinaryOp):
        kind = _BINOP.get(node.op)
        if kind is None:
            raise CertificateError(f"unsupported binary operator {node.op!r}")
        return {
            "k": kind,
            "l": _serialize_expr(node.left, offsets),
            "r": _serialize_expr(node.right, offsets),
        }
    if isinstance(node, UnaryOp):
        kind = _UNOP.get(node.op)
        if kind is None:
            raise CertificateError(f"unsupported unary operator {node.op!r}")
        return {"k": kind, "x": _serialize_expr(node.operand, offsets)}
    if isinstance(node, FunctionCall):
        # Transcendental MathFunc. Emitted for completeness/generality, but the
        # Tier-1 exact-rational checker refuses to evaluate it (Phase 1 adds
        # interval enclosures over Mathlib reals).
        return {
            "k": "fn",
            "name": node.func_name,
            "args": [_serialize_expr(a, offsets) for a in node.args],
        }
    raise CertificateError(
        f"cannot encode expression node {type(node).__name__} "
        "(matmul / sum / reduction / custom-call / parameter are not supported by "
        "the Tier-1 emitter -- see docs/dev/lean-certificate-plan.md)"
    )


# ── incumbent ────────────────────────────────────────────────────────────────
def _flat_incumbent(model: Model, result: SolveResult, n_cols: int) -> list[Rational]:
    x = result.x or {}
    flat: list[Rational] = []
    for var in model._variables:
        if var.name not in x:
            raise CertificateError(f"incumbent has no value for variable {var.name!r}")
        vals = np.asarray(x[var.name], dtype=np.float64).ravel()
        if vals.size != var.size:
            raise CertificateError(
                f"incumbent for {var.name!r} has {vals.size} entries, expected {var.size}"
            )
        for v in vals:
            r = to_rational(float(v))
            if r is None:
                raise CertificateError(f"non-finite incumbent value for {var.name!r}")
            flat.append(r)
    if len(flat) != n_cols:
        raise CertificateError(f"incumbent has {len(flat)} columns, expected {n_cols}")
    return flat


# ── top-level builder ────────────────────────────────────────────────────────
def build_feasibility_certificate(
    model: Model,
    result: SolveResult,
    *,
    feas_tol: float = DEFAULT_FEAS_TOL,
    int_tol: float = DEFAULT_INT_TOL,
) -> dict:
    """Build a Tier-1 feasibility certificate for *result* on *model*.

    The certificate restates the model (flat columns, constraints, objective) and
    the incumbent in exact rationals, so an external checker can verify the
    incumbent is feasible with the reported objective value. It makes **no**
    global-optimality claim (that is Tier 2/3).

    Raises :class:`CertificateError` if there is no incumbent to certify or the
    model contains a node the Tier-1 encoder cannot faithfully represent.
    """
    if result.x is None or result.objective is None:
        raise CertificateError(
            f"result has no incumbent to certify (status={result.status!r}); "
            "a feasibility certificate requires a feasible point"
        )

    columns, offsets = _flatten_columns(model)

    if model._objective is None:
        raise CertificateError("model has no objective")
    obj_sense = "min" if _enum_value(model._objective.sense) == "minimize" else "max"
    objective = {
        "sense": obj_sense,
        "body": _serialize_expr(model._objective.expression, offsets),
    }

    constraints = []
    for i, c in enumerate(model._constraints):
        body = getattr(c, "body", None)
        sense = getattr(c, "sense", None)
        if body is None or sense is None:
            raise CertificateError(
                f"constraint #{i} ({getattr(c, 'name', None)!r}) is not a plain "
                f"algebraic constraint ({type(c).__name__}); indicator/SOS/logical "
                "constraints are not supported by the Tier-1 emitter"
            )
        csense = _CSENSE.get(_enum_value(sense))
        if csense is None:
            raise CertificateError(f"constraint #{i} has unknown sense {sense!r}")
        rhs = to_rational(float(getattr(c, "rhs", 0.0)))
        if rhs is None:
            raise CertificateError(f"constraint #{i} has non-finite rhs")
        constraints.append(
            {
                "name": getattr(c, "name", None) or f"c{i}",
                "sense": csense,
                "body": _serialize_expr(body, offsets),
                "rhs": rhs,
            }
        )

    incumbent = _flat_incumbent(model, result, len(columns))
    obj_value = to_rational(float(result.objective))
    if obj_value is None:
        raise CertificateError("non-finite objective value")

    return {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "certificate": {
            "tier": "feasibility",
            "model": {
                "n_columns": len(columns),
                "columns": columns,
                "constraints": constraints,
                "objective": objective,
            },
            "incumbent": {"x": incumbent, "objectiveValue": obj_value},
            "tolerances": {
                "feas": to_rational(feas_tol),
                "int": to_rational(int_tol),
            },
            "meta": {
                "status": result.status,
                "gap_certified": bool(getattr(result, "gap_certified", False)),
            },
        },
    }


def write_certificate(cert: dict, path: Union[str, Path]) -> None:
    """Write *cert* to *path* as indented JSON."""
    import json

    Path(path).write_text(json.dumps(cert, indent=2) + "\n")
