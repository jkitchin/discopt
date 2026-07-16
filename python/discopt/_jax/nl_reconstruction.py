"""
DAG Reconstruction: rebuild Python Expression trees from Rust ExprArena.

Walks the arena bottom-up (topological order) and creates native
discopt.modeling.core Expression objects, enabling McCormick relaxations,
problem classification, algebraic LP/QP extraction, and JAX compilation
for models loaded from .nl files.
"""

from __future__ import annotations

from typing import Optional

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    UnaryOp,
    Variable,
)


def _build_cache(nl_repr, variables: list[Variable]) -> list[Optional[Expression]]:
    """Walk the Rust ExprArena bottom-up and materialize one Expression per node.

    Shared by :func:`reconstruct_dag` and :func:`reconstruct_complementarities`
    so both read from the same arena walk without duplicating the opcode
    dispatch.
    """
    n = nl_repr.arena_len()
    cache: list[Optional[Expression]] = [None] * n

    for i in range(n):
        node = nl_repr.get_node(i)
        ntype = node["type"]

        if ntype == "constant":
            cache[i] = Constant(node["value"])

        elif ntype == "constant_array":
            import numpy as np

            cache[i] = Constant(np.array(node["value"]).reshape(node["shape"]))

        elif ntype == "variable":
            idx = node["index"]
            cache[i] = variables[idx]

        elif ntype == "parameter":
            # Parameters become constants (already substituted in .nl)
            import numpy as np

            vals = node["value"]
            shape = node["shape"]
            if len(vals) == 1:
                cache[i] = Constant(vals[0])
            else:
                cache[i] = Constant(np.array(vals).reshape(shape))

        elif ntype == "binary_op":
            left = cache[node["left"]]
            right = cache[node["right"]]
            cache[i] = BinaryOp(node["op"], left, right)

        elif ntype == "unary_op":
            arg = cache[node["arg"]]
            cache[i] = UnaryOp(node["op"], arg)

        elif ntype == "function_call":
            func = node["func"]
            args = [cache[a] for a in node["args"]]
            cache[i] = FunctionCall(func, *args)

        elif ntype == "sum_over":
            terms = [cache[t] for t in node["terms"]]
            # Build a left-associative chain of additions
            result = terms[0]
            for t in terms[1:]:
                result = BinaryOp("+", result, t)
            cache[i] = result

        elif ntype == "sum":
            # Sum reduction — wrap as FunctionCall or inline
            operand = cache[node["operand"]]
            # For scalar expressions, sum is identity
            cache[i] = operand

        elif ntype == "index":
            from discopt.modeling.core import IndexExpression

            base = cache[node["base"]]
            idx = node["index_spec"]
            if isinstance(idx, list):
                idx = tuple(idx)
            cache[i] = IndexExpression(base, idx)

        elif ntype == "matmul":
            from discopt.modeling.core import MatMulExpression

            left = cache[node["left"]]
            right = cache[node["right"]]
            cache[i] = MatMulExpression(left, right)

        else:
            raise ValueError(f"Unknown node type from arena: {ntype!r}")

    return cache


def reconstruct_dag(
    nl_repr,
    variables: list[Variable],
) -> tuple[Expression, list[tuple[Expression, str, float]]]:
    """Walk the Rust ExprArena bottom-up and build Python Expression trees.

    Parameters
    ----------
    nl_repr : PyModelRepr
        The Rust model representation with arena access methods.
    variables : list[Variable]
        Python Variable objects corresponding to the .nl variable indices.

    Returns
    -------
    objective : Expression
        The reconstructed objective expression.
    constraints : list[tuple[Expression, str, float]]
        List of (body_expression, sense, rhs) tuples for each constraint.
        Complementarity (type-5) rows are *not* included here — they are
        consumed by the complementarity relation; see
        :func:`reconstruct_complementarities`.
    """
    cache = _build_cache(nl_repr, variables)

    # Extract objective
    obj_id = nl_repr.objective_id()
    objective = cache[obj_id]

    # Extract constraints
    constraints = []
    n_cons = nl_repr.n_constraints
    for i in range(n_cons):
        expr_id, sense, rhs = nl_repr.constraint_info(i)
        body = cache[expr_id]
        constraints.append((body, sense, rhs))

    return objective, constraints


def reconstruct_complementarities(
    nl_repr,
    variables: list[Variable],
) -> list[tuple[Expression, int, int]]:
    """Rebuild complementarity relations recovered from a ``.nl`` file (#658).

    Returns a list of ``(body_expression, var_index, flag)`` tuples, where
    ``body_expression`` is the constraint body complementary to the variable at
    ``var_index`` (0-based), and ``flag`` is the raw AMPL MP ``ComplInfo`` bound
    flag for the body (bit 0 ⇒ lower bound ``-inf``, bit 1 ⇒ upper bound
    ``+inf``; an unset bit means the corresponding bound is 0).

    Returns an empty list when the representation carries no complementarities
    (the common, non-MPEC case), so callers can cheaply skip the arena walk.
    """
    n_compl = getattr(nl_repr, "n_complementarities", 0)
    if not n_compl:
        return []

    cache = _build_cache(nl_repr, variables)
    pairs: list[tuple[Expression, int, int]] = []
    for i in range(n_compl):
        body_id, var_index, flag = nl_repr.complementarity_info(i)
        pairs.append((cache[body_id], var_index, flag))
    return pairs
