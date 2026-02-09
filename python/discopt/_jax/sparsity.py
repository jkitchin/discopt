"""
Sparsity detection from expression DAGs using graph coloring.

Analyzes the variable-constraint incidence structure of a Model to produce
Jacobian and Hessian sparsity patterns. Uses greedy graph coloring on the
column intersection graph to compute a seed matrix for compressed forward-mode
Jacobian evaluation via JVPs.

For problems with density < 15% and n >= 50 variables, sparse Jacobian
evaluation reduces cost from O(n) JVPs to O(p) JVPs where p is the chromatic
number of the column intersection graph (typically 5-20 for sparse problems).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp

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
)


@dataclass
class SparsityPattern:
    """Sparsity patterns for Jacobian and Hessian of a model.

    Attributes:
        jacobian_sparsity: CSR boolean matrix (m x n) where True indicates
            that constraint i depends on variable j.
        hessian_sparsity: CSR boolean matrix (n x n) where True indicates
            that variables i and j co-occur in a nonlinear term.
        n_vars: Number of variables.
        n_cons: Number of constraints.
        jacobian_nnz: Number of nonzeros in Jacobian pattern.
        hessian_nnz: Number of nonzeros in Hessian pattern.
    """

    jacobian_sparsity: sp.csr_matrix
    hessian_sparsity: sp.csr_matrix
    n_vars: int
    n_cons: int
    jacobian_nnz: int
    hessian_nnz: int

    @property
    def jacobian_density(self) -> float:
        """Fraction of nonzero entries in the Jacobian pattern."""
        total = self.n_cons * self.n_vars
        if total == 0:
            return 0.0
        return self.jacobian_nnz / total

    @property
    def hessian_density(self) -> float:
        """Fraction of nonzero entries in the Hessian pattern."""
        total = self.n_vars * self.n_vars
        if total == 0:
            return 0.0
        return self.hessian_nnz / total


def _var_offset(var: Variable, model: Model) -> int:
    """Compute flat offset for a variable in the concatenated vector."""
    offset = 0
    for v in model._variables[: var._index]:
        offset += v.size
    return offset


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


def _collect_nonlinear_pairs(expr: Expression, model: Model) -> set[tuple[int, int]]:
    """Collect pairs of variable indices that interact nonlinearly.

    Detects:
    - BinaryOp("*", ...) where both sides contain variables (bilinear)
    - BinaryOp("**", ...) where base contains a variable (power)
    - FunctionCall where arguments contain multiple variables
    - MatMulExpression where both sides contain variables
    """
    pairs: set[tuple[int, int]] = set()

    if isinstance(expr, BinaryOp):
        left_vars = _collect_variable_indices(expr.left, model)
        right_vars = _collect_variable_indices(expr.right, model)

        if expr.op == "*" and left_vars and right_vars:
            for i in left_vars:
                for j in right_vars:
                    a, b = min(i, j), max(i, j)
                    pairs.add((a, b))

        if expr.op == "**" and left_vars:
            for i in left_vars:
                pairs.add((i, i))

        pairs |= _collect_nonlinear_pairs(expr.left, model)
        pairs |= _collect_nonlinear_pairs(expr.right, model)

    elif isinstance(expr, UnaryOp):
        operand_vars = _collect_variable_indices(expr.operand, model)
        for i in operand_vars:
            pairs.add((i, i))
        pairs |= _collect_nonlinear_pairs(expr.operand, model)

    elif isinstance(expr, FunctionCall):
        all_vars: set[int] = set()
        for arg in expr.args:
            all_vars |= _collect_variable_indices(arg, model)
        var_list = sorted(all_vars)
        for idx, i in enumerate(var_list):
            for j in var_list[idx:]:
                pairs.add((i, j))
        for arg in expr.args:
            pairs |= _collect_nonlinear_pairs(arg, model)

    elif isinstance(expr, MatMulExpression):
        left_vars = _collect_variable_indices(expr.left, model)
        right_vars = _collect_variable_indices(expr.right, model)
        if left_vars and right_vars:
            for i in left_vars:
                for j in right_vars:
                    a, b = min(i, j), max(i, j)
                    pairs.add((a, b))
        pairs |= _collect_nonlinear_pairs(expr.left, model)
        pairs |= _collect_nonlinear_pairs(expr.right, model)

    elif isinstance(expr, SumExpression):
        pairs |= _collect_nonlinear_pairs(expr.operand, model)

    elif isinstance(expr, SumOverExpression):
        for t in expr.terms:
            pairs |= _collect_nonlinear_pairs(t, model)

    elif isinstance(expr, IndexExpression):
        if isinstance(expr.base, Variable):
            pass
        else:
            pairs |= _collect_nonlinear_pairs(expr.base, model)

    return pairs


def detect_sparsity_dag(model: Model) -> SparsityPattern:
    """Detect Jacobian and Hessian sparsity patterns from a Model's expression DAG.

    Walks each constraint body to determine which variables appear in each
    constraint (Jacobian sparsity) and which variable pairs interact
    nonlinearly (Hessian sparsity).

    Args:
        model: A Model with constraints set.

    Returns:
        SparsityPattern with Jacobian and Hessian patterns as scipy CSR matrices.
    """
    n_vars = sum(v.size for v in model._variables)
    constraints = [c for c in model._constraints if isinstance(c, Constraint)]
    n_cons = len(constraints)

    # Build Jacobian sparsity: for each constraint row, find variable columns
    jac_rows: list[int] = []
    jac_cols: list[int] = []

    # Build Hessian sparsity from nonlinear interactions
    hess_pairs: set[tuple[int, int]] = set()

    for row_idx, con in enumerate(constraints):
        var_indices = _collect_variable_indices(con.body, model)
        for col_idx in var_indices:
            if 0 <= col_idx < n_vars:
                jac_rows.append(row_idx)
                jac_cols.append(col_idx)

        pairs = _collect_nonlinear_pairs(con.body, model)
        hess_pairs |= pairs

    # Also include objective nonlinear pairs
    if model._objective is not None:
        obj_pairs = _collect_nonlinear_pairs(model._objective.expression, model)
        hess_pairs |= obj_pairs

    # Build Jacobian CSR
    if jac_rows:
        data = np.ones(len(jac_rows), dtype=bool)
        jac_csr = sp.csr_matrix(
            (data, (np.array(jac_rows), np.array(jac_cols))),
            shape=(n_cons, n_vars),
        )
    else:
        jac_csr = sp.csr_matrix((n_cons, n_vars), dtype=bool)

    # Build Hessian CSR (symmetric)
    if hess_pairs:
        hess_rows = []
        hess_cols = []
        for i, j in hess_pairs:
            if 0 <= i < n_vars and 0 <= j < n_vars:
                hess_rows.append(i)
                hess_cols.append(j)
                if i != j:
                    hess_rows.append(j)
                    hess_cols.append(i)
        data = np.ones(len(hess_rows), dtype=bool)
        hess_csr = sp.csr_matrix(
            (data, (np.array(hess_rows), np.array(hess_cols))),
            shape=(n_vars, n_vars),
        )
    else:
        hess_csr = sp.csr_matrix((n_vars, n_vars), dtype=bool)

    return SparsityPattern(
        jacobian_sparsity=jac_csr,
        hessian_sparsity=hess_csr,
        n_vars=n_vars,
        n_cons=n_cons,
        jacobian_nnz=jac_csr.nnz,
        hessian_nnz=hess_csr.nnz,
    )


def compute_coloring(pattern: SparsityPattern) -> tuple[np.ndarray, int]:
    """Compute a greedy graph coloring on the column intersection graph.

    Two columns share an edge if they both have a nonzero in the same row
    of the Jacobian. The coloring assigns colors to columns such that no two
    adjacent columns share a color, enabling compressed Jacobian evaluation.

    Uses smallest-degree-first ordering for near-optimal colorings.

    Args:
        pattern: SparsityPattern with Jacobian sparsity.

    Returns:
        Tuple of (colors, n_colors) where colors is an (n_vars,) int array
        and n_colors is the total number of colors used.
    """
    n = pattern.n_vars
    if n == 0:
        return np.array([], dtype=np.int32), 0

    jac = pattern.jacobian_sparsity

    # Build column intersection graph: columns i and j are adjacent if they
    # share a row. This is the nonzero pattern of J^T @ J (off-diagonal).
    jtj = (jac.T @ jac).tocsr()

    # Build adjacency lists
    adjacency: list[set[int]] = [set() for _ in range(n)]
    rows_jtj, cols_jtj = jtj.nonzero()
    for r, c in zip(rows_jtj, cols_jtj):
        if r != c:
            adjacency[r].add(c)

    # Smallest-degree-first ordering
    degrees = np.array([len(adj) for adj in adjacency])
    order = np.argsort(degrees)

    colors = np.full(n, -1, dtype=np.int32)
    n_colors = 0

    for col_idx in order:
        # Find colors used by neighbors
        used = set()
        for neighbor in adjacency[col_idx]:
            if colors[neighbor] >= 0:
                used.add(colors[neighbor])

        # Assign smallest available color
        c = 0
        while c in used:
            c += 1
        colors[col_idx] = c
        if c >= n_colors:
            n_colors = c + 1

    return colors, n_colors


def make_seed_matrix(colors: np.ndarray, n_colors: int, n_vars: int) -> np.ndarray:
    """Build a seed matrix for compressed forward-mode Jacobian evaluation.

    The seed matrix S has shape (n_vars, n_colors) where S[j, c] = 1 if
    column j has color c. Each column of S is used as a tangent direction
    in a JVP call to compute multiple Jacobian columns simultaneously.

    Args:
        colors: (n_vars,) int array of column colors.
        n_colors: Total number of colors.
        n_vars: Number of variables.

    Returns:
        (n_vars, n_colors) float64 seed matrix.
    """
    S = np.zeros((n_vars, n_colors), dtype=np.float64)
    for j in range(n_vars):
        S[j, colors[j]] = 1.0
    return S


def should_use_sparse(
    pattern: SparsityPattern,
    density_threshold: float = 0.15,
    min_vars: int = 50,
) -> bool:
    """Decide whether to use sparse or dense Jacobian evaluation.

    Sparse evaluation is beneficial when:
    1. The Jacobian density is below the threshold (default 15%)
    2. The problem has enough variables (default >= 50)

    Small or dense problems are better served by dense evaluation
    which has lower overhead and benefits from JAX's optimized dense ops.

    Args:
        pattern: SparsityPattern from detect_sparsity_dag.
        density_threshold: Maximum density for sparse evaluation.
        min_vars: Minimum number of variables for sparse evaluation.

    Returns:
        True if sparse evaluation is recommended.
    """
    if pattern.n_vars < min_vars:
        return False
    if pattern.n_cons == 0:
        return False
    return bool(pattern.jacobian_density < density_threshold)


def detect_and_color(
    model: Model,
    density_threshold: float = 0.15,
    min_vars: int = 50,
) -> Optional[tuple[SparsityPattern, np.ndarray, int, np.ndarray]]:
    """Convenience: detect sparsity, check threshold, and compute coloring.

    Returns None if sparse evaluation is not recommended (problem too small
    or too dense). Otherwise returns (pattern, colors, n_colors, seed_matrix).

    Args:
        model: A Model with constraints set.
        density_threshold: Maximum density for sparse evaluation.
        min_vars: Minimum number of variables for sparse evaluation.

    Returns:
        None if dense is preferred, or (pattern, colors, n_colors, seed_matrix).
    """
    pattern = detect_sparsity_dag(model)
    if not should_use_sparse(pattern, density_threshold, min_vars):
        return None
    colors, n_colors = compute_coloring(pattern)
    seed = make_seed_matrix(colors, n_colors, pattern.n_vars)
    return pattern, colors, n_colors, seed
