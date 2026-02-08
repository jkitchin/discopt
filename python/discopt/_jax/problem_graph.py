"""
Bipartite graph representation of MINLP for GNN branching policy.

Builds a variable-constraint incidence graph from a Model + solution state,
suitable for message-passing GNN architectures (e.g., Gasse et al. 2019).

Nodes:
  - Variable nodes: [value, lb, ub, is_integer, fractionality, log_n_vars, log_n_cons]
  - Constraint nodes: [lhs_value, slack, sense_encoding, log_n_vars, log_n_cons]

Including problem-size features (log_n_vars, log_n_cons) on every node mitigates
training distribution mismatch when training on small instances and testing on large.

Edges:
  - Variable i <-> Constraint j if variable i appears in constraint j's body.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)

INTEGRALITY_TOL = 1e-5


@dataclass
class ProblemGraph:
    """Bipartite graph representation for GNN branching.

    Attributes:
        var_features: (n_vars, 7) array of variable node features.
            Columns: [value, lb, ub, is_integer, fractionality,
                      log_n_vars, log_n_cons]
        con_features: (n_cons, 5) array of constraint node features.
            Columns: [lhs_value, slack, sense_encoding,
                      log_n_vars, log_n_cons]
        edge_indices: (2, n_edges) int array. Row 0 = variable indices,
            row 1 = constraint indices. Defines the bipartite adjacency.
        n_vars: Number of variable nodes.
        n_cons: Number of constraint nodes.
    """

    var_features: jnp.ndarray  # (n_vars, 7)
    con_features: jnp.ndarray  # (n_cons, 5)
    edge_indices: jnp.ndarray  # (2, n_edges)
    n_vars: int
    n_cons: int


def _collect_variable_indices(expr: Expression, model: Model) -> set[int]:
    """Recursively collect flat variable indices referenced by an expression."""
    if isinstance(expr, Variable):
        offset = _var_offset(expr, model)
        return set(range(offset, offset + expr.size))
    if isinstance(expr, Constant):
        return set()
    if isinstance(expr, Parameter):
        return set()
    if isinstance(expr, IndexExpression):
        if isinstance(expr.base, Variable):
            offset = _var_offset(expr.base, model)
            idx = expr.index
            if isinstance(idx, (int, np.integer)):
                return {offset + int(idx)}
            # For slice or tuple indexing, conservatively include all
            return set(range(offset, offset + expr.base.size))
        return _collect_variable_indices(expr.base, model)
    if isinstance(expr, BinaryOp):
        left = _collect_variable_indices(expr.left, model)
        right = _collect_variable_indices(expr.right, model)
        return left | right
    if isinstance(expr, UnaryOp):
        return _collect_variable_indices(expr.operand, model)
    if isinstance(expr, FunctionCall):
        result: set[int] = set()
        for arg in expr.args:
            result |= _collect_variable_indices(arg, model)
        return result
    if isinstance(expr, MatMulExpression):
        left = _collect_variable_indices(expr.left, model)
        right = _collect_variable_indices(expr.right, model)
        return left | right
    if isinstance(expr, SumExpression):
        return _collect_variable_indices(expr.operand, model)
    if isinstance(expr, SumOverExpression):
        result = set()
        for t in expr.terms:
            result |= _collect_variable_indices(t, model)
        return result
    return set()


def _var_offset(var: Variable, model: Model) -> int:
    """Compute flat offset for a variable in the concatenated vector."""
    offset = 0
    for v in model._variables[: var._index]:
        offset += v.size
    return offset


def _compute_fractionality(value: float) -> float:
    """Compute fractionality: closeness to 0.5 in [0, 0.5]."""
    frac = value - np.floor(value)
    if frac < INTEGRALITY_TOL or frac > 1.0 - INTEGRALITY_TOL:
        return 0.0
    return float(0.5 - abs(frac - 0.5))


def _sense_encoding(sense: str) -> float:
    """Encode constraint sense as a float: <= -> -1, == -> 0, >= -> 1."""
    if sense == "<=":
        return -1.0
    elif sense == "==":
        return 0.0
    elif sense == ">=":
        return 1.0
    return 0.0


def build_graph(
    model: Model,
    solution: np.ndarray,
    node_lb: Optional[np.ndarray] = None,
    node_ub: Optional[np.ndarray] = None,
) -> ProblemGraph:
    """Build a bipartite problem graph from a Model and current solution.

    Args:
        model: The optimization model.
        solution: Flat solution vector (n_vars,) from the current relaxation.
        node_lb: Current variable lower bounds (node-specific). If None,
            uses the model's global lower bounds.
        node_ub: Current variable upper bounds (node-specific). If None,
            uses the model's global upper bounds.

    Returns:
        ProblemGraph with variable features, constraint features, and edges.
    """
    n_vars = sum(v.size for v in model._variables)
    constraints = [c for c in model._constraints if isinstance(c, Constraint)]
    n_cons = len(constraints)

    # --- Variable features ---
    if node_lb is None or node_ub is None:
        lb_parts = []
        ub_parts = []
        for v in model._variables:
            lb_parts.append(v.lb.flatten())
            ub_parts.append(v.ub.flatten())
        global_lb = np.concatenate(lb_parts) if lb_parts else np.array([])
        global_ub = np.concatenate(ub_parts) if ub_parts else np.array([])
        if node_lb is None:
            node_lb = global_lb
        if node_ub is None:
            node_ub = global_ub

    # Build per-variable type map
    is_int = np.zeros(n_vars, dtype=np.float64)
    offset = 0
    for v in model._variables:
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            is_int[offset : offset + v.size] = 1.0
        offset += v.size

    fractionality = np.zeros(n_vars, dtype=np.float64)
    for i in range(n_vars):
        if is_int[i] > 0.5:
            fractionality[i] = _compute_fractionality(float(solution[i]))

    # Problem-size features (appended to every node for size generalization)
    log_nv = np.full(n_vars, np.log1p(n_vars), dtype=np.float64)
    log_nc = np.full(n_vars, np.log1p(n_cons), dtype=np.float64)

    var_feats = np.stack(
        [
            solution.astype(np.float64),
            node_lb.astype(np.float64),
            node_ub.astype(np.float64),
            is_int,
            fractionality,
            log_nv,
            log_nc,
        ],
        axis=1,
    )  # (n_vars, 7)

    # --- Constraint features and edge construction ---
    from discopt._jax.dag_compiler import compile_constraint

    con_feats_list = []
    edge_var_indices = []
    edge_con_indices = []

    for j, con in enumerate(constraints):
        # Compute LHS value
        con_fn = compile_constraint(con, model)
        try:
            lhs_val = float(con_fn(jnp.array(solution, dtype=jnp.float64)))
        except Exception:
            lhs_val = 0.0

        # Slack: for <= constraint (body - rhs <= 0), slack = -lhs_val
        # Positive slack means feasible.
        sense_enc = _sense_encoding(con.sense)
        if con.sense == "<=":
            slack = -lhs_val
        elif con.sense == ">=":
            slack = lhs_val
        else:
            slack = -abs(lhs_val)

        con_feats_list.append([lhs_val, slack, sense_enc])

        # Collect variable indices in this constraint
        var_indices = _collect_variable_indices(con.body, model)
        for vi in var_indices:
            if 0 <= vi < n_vars:
                edge_var_indices.append(vi)
                edge_con_indices.append(j)

    if n_cons > 0:
        base_feats = np.array(con_feats_list, dtype=np.float64)  # (n_cons, 3)
        log_nv_c = np.full((n_cons, 1), np.log1p(n_vars), dtype=np.float64)
        log_nc_c = np.full((n_cons, 1), np.log1p(n_cons), dtype=np.float64)
        con_feats = np.hstack([base_feats, log_nv_c, log_nc_c])  # (n_cons, 5)
    else:
        con_feats = np.zeros((0, 5), dtype=np.float64)

    if edge_var_indices:
        edges = np.array([edge_var_indices, edge_con_indices], dtype=np.int32)
    else:
        edges = np.zeros((2, 0), dtype=np.int32)

    return ProblemGraph(
        var_features=jnp.array(var_feats),
        con_features=jnp.array(con_feats),
        edge_indices=jnp.array(edges),
        n_vars=n_vars,
        n_cons=n_cons,
    )
