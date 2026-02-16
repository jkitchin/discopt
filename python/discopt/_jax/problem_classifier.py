"""
Problem classification and standard-form extraction for LP, QP, MILP, MIQP, NLP, MINLP.

Uses existing Rust structure detection (is_linear, is_quadratic) via PyO3 bindings
to classify problems, then extracts standard-form data using the JAX DAG compiler.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    IndexExpression,
    MatMulExpression,
    Model,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)


class ProblemClass(Enum):
    """Classification of an optimization problem."""

    LP = "lp"  # linear obj + linear constraints + all continuous
    QP = "qp"  # ≤quadratic obj + linear constraints + all continuous
    MILP = "milp"  # linear obj + linear constraints + has integer/binary
    MIQP = "miqp"  # ≤quadratic obj + linear constraints + has integer/binary
    NLP = "nlp"  # general nonlinear + all continuous
    MINLP = "minlp"  # general nonlinear + has integer/binary


def classify_problem(model: Model) -> ProblemClass:
    """Classify a model into LP, QP, MILP, MIQP, NLP, or MINLP.

    Uses Rust structure detection for degree analysis of the objective
    and constraints. Falls back to NLP/MINLP if Rust bindings unavailable.

    Args:
        model: A discopt Model with objective and constraints.

    Returns:
        ProblemClass enum value.
    """
    has_integer = any(v.var_type in (VarType.BINARY, VarType.INTEGER) for v in model._variables)

    try:
        from discopt._rust import model_to_repr

        _builder = getattr(model, "_builder", None)
        repr = model_to_repr(model, _builder)
        obj_linear = repr.is_objective_linear()
        obj_quadratic = repr.is_objective_quadratic()
        all_constraints_linear = all(
            repr.is_constraint_linear(i) for i in range(repr.n_constraints)
        )
    except (ImportError, Exception):
        # Rust bindings unavailable — fall back to NLP/MINLP
        return ProblemClass.MINLP if has_integer else ProblemClass.NLP

    if all_constraints_linear:
        if obj_linear:
            return ProblemClass.MILP if has_integer else ProblemClass.LP
        if obj_quadratic:
            return ProblemClass.MIQP if has_integer else ProblemClass.QP

    return ProblemClass.MINLP if has_integer else ProblemClass.NLP


class LPData(NamedTuple):
    """Standard-form LP data: min c'x + d s.t. A_eq x = b_eq, x_l <= x <= x_u."""

    c: jnp.ndarray  # (n,) objective coefficients
    A_eq: jnp.ndarray  # (m, n) equality constraint matrix
    b_eq: jnp.ndarray  # (m,) equality RHS
    x_l: jnp.ndarray  # (n,) lower bounds
    x_u: jnp.ndarray  # (n,) upper bounds
    obj_const: float = 0.0  # constant term in objective


class QPData(NamedTuple):
    """Standard-form QP: min 0.5 x'Qx + c'x + d s.t. A_eq x = b_eq, bounds."""

    Q: jnp.ndarray  # (n, n) quadratic objective matrix (symmetric)
    c: jnp.ndarray  # (n,) linear objective coefficients
    A_eq: jnp.ndarray  # (m, n) equality constraint matrix
    b_eq: jnp.ndarray  # (m,) equality RHS
    x_l: jnp.ndarray  # (n,) lower bounds
    x_u: jnp.ndarray  # (n,) upper bounds
    obj_const: float = 0.0  # constant term in objective


def _get_variable_bounds(model: Model):
    """Extract flat lower and upper bounds from model variables.

    Returns numpy arrays to avoid JAX device-transfer overhead during
    extraction.  The caller (lp_ipm_solve / qp_ipm_solve) converts to
    jnp.array at solve time.
    """
    lb_parts = []
    ub_parts = []
    for v in model._variables:
        lb_parts.append(v.lb.flatten())
        ub_parts.append(v.ub.flatten())
    n = sum(v.size for v in model._variables)
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    lb = np.concatenate(lb_parts).astype(np.float64)
    ub = np.concatenate(ub_parts).astype(np.float64)
    return lb, ub


# ---------------------------------------------------------------------------
# Algebraic coefficient extraction (no autodiff)
# ---------------------------------------------------------------------------


def _compute_var_offset(var: Variable, model: Model) -> int:
    """Compute the starting offset of a variable in the flat x vector."""
    offset = 0
    for v in model._variables[: var._index]:
        offset += v.size
    return offset


class _NotLinearError(Exception):
    """Raised when an expression is not linear."""


class _NotQuadraticError(Exception):
    """Raised when an expression is not quadratic (at most degree 2)."""


def _extract_linear_coefficients(expr, model: Model, n: int):
    """Walk an expression tree to extract linear coefficients and constant.

    Returns (coefficients, constant) where:
      - coefficients is a numpy array of shape (n,) with coefficient for each variable slot
      - constant is a float scalar

    Raises _NotLinearError if the expression is not linear.
    """
    c = np.zeros(n, dtype=np.float64)
    const = 0.0

    def _walk(node, scale=1.0):
        nonlocal const

        if isinstance(node, Constant):
            val = node.value
            if val.ndim == 0:
                const += scale * float(val)
            else:
                raise _NotLinearError("Array constant in unexpected position")
            return

        if isinstance(node, Variable):
            offset = _compute_var_offset(node, model)
            if node.size == 1:
                c[offset] += scale
            else:
                # Array variable treated as sum when used as scalar
                for j in range(node.size):
                    c[offset + j] += scale
            return

        if isinstance(node, IndexExpression):
            if isinstance(node.base, Variable):
                var = node.base
                offset = _compute_var_offset(var, model)
                idx = node.index
                if isinstance(idx, (int, np.integer)):
                    c[offset + int(idx)] += scale
                elif isinstance(idx, tuple) and len(idx) == 1:
                    c[offset + int(idx[0])] += scale
                else:
                    # Multi-dimensional index: flatten
                    flat_idx = np.ravel_multi_index(
                        idx if isinstance(idx, tuple) else (idx,), var.shape
                    )
                    c[offset + int(flat_idx)] += scale
                return
            raise _NotLinearError(f"IndexExpression on non-variable: {type(node.base)}")

        if isinstance(node, BinaryOp):
            if node.op == "+":
                _walk(node.left, scale)
                _walk(node.right, scale)
                return
            if node.op == "-":
                _walk(node.left, scale)
                _walk(node.right, -scale)
                return
            if node.op == "*":
                # One side must be constant for linearity
                if _is_const_expr(node.left):
                    cval = _eval_const(node.left)
                    _walk(node.right, scale * cval)
                    return
                if _is_const_expr(node.right):
                    cval = _eval_const(node.right)
                    _walk(node.left, scale * cval)
                    return
                raise _NotLinearError("Product of two variable expressions")
            if node.op == "/":
                if _is_const_expr(node.right):
                    cval = _eval_const(node.right)
                    _walk(node.left, scale / cval)
                    return
                raise _NotLinearError("Division by variable expression")
            raise _NotLinearError(f"Non-linear operator: {node.op}")

        if isinstance(node, UnaryOp):
            if node.op == "neg":
                _walk(node.operand, -scale)
                return
            raise _NotLinearError(f"Non-linear unary op: {node.op}")

        if isinstance(node, SumOverExpression):
            for term in node.terms:
                _walk(term, scale)
            return

        if isinstance(node, SumExpression):
            # Sum of an array variable or expression
            _walk(node.operand, scale)
            return

        if isinstance(node, MatMulExpression):
            # Handle Constant @ Variable or Variable @ Constant
            if isinstance(node.left, Constant) and isinstance(node.right, Variable):
                mat = node.left.value
                var = node.right
                offset = _compute_var_offset(var, model)
                # mat @ var => result is mat @ x[offset:offset+size]
                # For 1-D mat (dot product), coefficients are mat elements
                if mat.ndim == 1:
                    for j in range(var.size):
                        c[offset + j] += scale * float(mat[j])
                elif mat.ndim == 2:
                    # Returns vector; this should be used inside a sum
                    raise _NotLinearError("MatMul returning vector in scalar context")
                return
            if isinstance(node.right, Constant) and isinstance(node.left, Variable):
                mat = node.right.value
                var = node.left
                offset = _compute_var_offset(var, model)
                if mat.ndim == 1:
                    for j in range(var.size):
                        c[offset + j] += scale * float(mat[j])
                    return
                raise _NotLinearError("MatMul returning vector in scalar context")
            raise _NotLinearError("MatMul between non-trivial expressions")

        raise _NotLinearError(f"Unhandled expression type: {type(node).__name__}")

    _walk(expr)
    return c, const


def _extract_quadratic_coefficients(expr, model: Model, n: int):
    """Walk expression tree to extract quadratic and linear coefficients.

    Returns (Q, c, constant) where:
      - Q is (n, n) numpy array (the Hessian: f = 0.5 x'Qx + c'x + const)
      - c is (n,) numpy array of linear coefficients
      - constant is a float scalar

    Raises _NotQuadraticError if the expression has degree > 2.
    """
    Q = np.zeros((n, n), dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    const = 0.0

    def _get_var_index(node):
        """Get the flat variable index for a variable-like node, or None."""
        if isinstance(node, Variable):
            if node.size != 1:
                return None
            return _compute_var_offset(node, model)
        if isinstance(node, IndexExpression) and isinstance(node.base, Variable):
            offset = _compute_var_offset(node.base, model)
            idx = node.index
            if isinstance(idx, (int, np.integer)):
                return offset + int(idx)
            if isinstance(idx, tuple) and len(idx) == 1:
                return offset + int(idx[0])
            flat_idx = np.ravel_multi_index(
                idx if isinstance(idx, tuple) else (idx,), node.base.shape
            )
            return offset + int(flat_idx)
        return None

    def _walk(node, scale=1.0):
        nonlocal const

        if isinstance(node, Constant):
            val = node.value
            if val.ndim == 0:
                const += scale * float(val)
            else:
                raise _NotQuadraticError("Array constant in unexpected position")
            return

        if isinstance(node, (Variable, IndexExpression)):
            idx = _get_var_index(node)
            if idx is not None:
                c[idx] += scale
                return
            if isinstance(node, Variable) and node.size > 1:
                offset = _compute_var_offset(node, model)
                for j in range(node.size):
                    c[offset + j] += scale
                return
            raise _NotQuadraticError(f"Cannot extract index from {node}")

        if isinstance(node, BinaryOp):
            if node.op == "+":
                _walk(node.left, scale)
                _walk(node.right, scale)
                return
            if node.op == "-":
                _walk(node.left, scale)
                _walk(node.right, -scale)
                return
            if node.op == "*":
                # Check: const * expr, expr * const, or var * var
                if _is_const_expr(node.left):
                    cval = _eval_const(node.left)
                    _walk(node.right, scale * cval)
                    return
                if _is_const_expr(node.right):
                    cval = _eval_const(node.right)
                    _walk(node.left, scale * cval)
                    return
                # var * var => quadratic term
                # Q is the Hessian: f = 0.5 x'Qx, so d²(c*xi*xj)/dxi dxj = c,
                # but d²(c*xi²)/dxi² = 2c. We store the Hessian directly.
                idx_l = _get_var_index(node.left)
                idx_r = _get_var_index(node.right)
                if idx_l is not None and idx_r is not None:
                    if idx_l == idx_r:
                        Q[idx_l, idx_r] += 2.0 * scale
                    else:
                        Q[idx_l, idx_r] += scale
                        Q[idx_r, idx_l] += scale
                    return
                # Handle (const * var) * var or var * (const * var):
                # e.g., (Q[i,j] * x[i]) * x[j] from left-to-right evaluation
                cv_l = _try_extract_const_var(node.left, model)
                if cv_l is not None and idx_r is not None:
                    cval, idx_l2 = cv_l
                    if idx_l2 == idx_r:
                        Q[idx_l2, idx_r] += 2.0 * scale * cval
                    else:
                        Q[idx_l2, idx_r] += scale * cval
                        Q[idx_r, idx_l2] += scale * cval
                    return
                cv_r = _try_extract_const_var(node.right, model)
                if cv_r is not None and idx_l is not None:
                    cval, idx_r2 = cv_r
                    if idx_l == idx_r2:
                        Q[idx_l, idx_r2] += 2.0 * scale * cval
                    else:
                        Q[idx_l, idx_r2] += scale * cval
                        Q[idx_r2, idx_l] += scale * cval
                    return
                raise _NotQuadraticError("Product of non-simple variable expressions")
            if node.op == "/":
                if _is_const_expr(node.right):
                    cval = _eval_const(node.right)
                    _walk(node.left, scale / cval)
                    return
                raise _NotQuadraticError("Division by variable expression")
            if node.op == "**":
                # x**2 => quadratic
                if _is_const_expr(node.right):
                    pval = _eval_const(node.right)
                    if abs(pval - 2.0) < 1e-12:
                        idx = _get_var_index(node.left)
                        if idx is not None:
                            Q[idx, idx] += 2.0 * scale  # x^2 = 0.5 * 2 * x^2
                            return
                    if abs(pval - 1.0) < 1e-12:
                        _walk(node.left, scale)
                        return
                    if abs(pval) < 1e-12:
                        const += scale
                        return
                raise _NotQuadraticError(f"Power with exponent {node.right}")
            raise _NotQuadraticError(f"Unknown binary op: {node.op}")

        if isinstance(node, UnaryOp):
            if node.op == "neg":
                _walk(node.operand, -scale)
                return
            raise _NotQuadraticError(f"Non-linear unary op: {node.op}")

        if isinstance(node, SumOverExpression):
            for term in node.terms:
                _walk(term, scale)
            return

        if isinstance(node, SumExpression):
            _walk(node.operand, scale)
            return

        if isinstance(node, MatMulExpression):
            # Handle Constant @ Variable for linear parts of QP constraints
            if isinstance(node.left, Constant) and isinstance(node.right, Variable):
                mat = node.left.value
                var = node.right
                offset = _compute_var_offset(var, model)
                if mat.ndim == 1:
                    for j in range(var.size):
                        c[offset + j] += scale * float(mat[j])
                    return
                raise _NotQuadraticError("MatMul returning vector")
            if isinstance(node.right, Constant) and isinstance(node.left, Variable):
                mat = node.right.value
                var = node.left
                offset = _compute_var_offset(var, model)
                if mat.ndim == 1:
                    for j in range(var.size):
                        c[offset + j] += scale * float(mat[j])
                    return
                raise _NotQuadraticError("MatMul returning vector")
            raise _NotQuadraticError("MatMul between non-trivial expressions")

        raise _NotQuadraticError(f"Unhandled expression type: {type(node).__name__}")

    _walk(expr)
    return Q, c, const


def _try_extract_const_var(expr, model: Model):
    """Try to decompose expr as (constant * variable).

    Returns (constant_value, flat_var_index) if expr is of the form
    Constant * Variable/IndexExpr or Variable/IndexExpr * Constant,
    or just a bare Variable/IndexExpr (constant = 1.0).

    Returns None if the expression is not of this form.
    """
    # Bare variable => coefficient 1.0
    if isinstance(expr, (Variable, IndexExpression)):
        if isinstance(expr, Variable) and expr.size != 1:
            return None
        if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
            offset = _compute_var_offset(expr.base, model)
            idx = expr.index
            if isinstance(idx, (int, np.integer)):
                return (1.0, offset + int(idx))
            if isinstance(idx, tuple) and len(idx) == 1:
                return (1.0, offset + int(idx[0]))
            flat_idx = np.ravel_multi_index(
                idx if isinstance(idx, tuple) else (idx,), expr.base.shape
            )
            return (1.0, offset + int(flat_idx))
        if isinstance(expr, Variable):
            return (1.0, _compute_var_offset(expr, model))
        return None

    # const * var or var * const
    if isinstance(expr, BinaryOp) and expr.op == "*":
        if _is_const_expr(expr.left):
            cval = _eval_const(expr.left)
            inner = _try_extract_const_var(expr.right, model)
            if inner is not None:
                return (cval * inner[0], inner[1])
        if _is_const_expr(expr.right):
            cval = _eval_const(expr.right)
            inner = _try_extract_const_var(expr.left, model)
            if inner is not None:
                return (cval * inner[0], inner[1])

    # neg(var) => -1.0 * var
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        inner = _try_extract_const_var(expr.operand, model)
        if inner is not None:
            return (-inner[0], inner[1])

    return None


def _is_const_expr(expr) -> bool:
    """Check if an expression is a pure constant (no variables)."""
    if isinstance(expr, Constant):
        return True
    if isinstance(expr, (Variable, IndexExpression)):
        return False
    if isinstance(expr, BinaryOp):
        return _is_const_expr(expr.left) and _is_const_expr(expr.right)
    if isinstance(expr, UnaryOp):
        return _is_const_expr(expr.operand)
    if isinstance(expr, SumOverExpression):
        return all(_is_const_expr(t) for t in expr.terms)
    if isinstance(expr, SumExpression):
        return _is_const_expr(expr.operand)
    return False


def _eval_const(expr) -> float:  # type: ignore[return-value]
    """Evaluate a constant expression to a float scalar."""
    if isinstance(expr, Constant):
        v = expr.value
        return float(v) if v.ndim == 0 else float(v.item())
    if isinstance(expr, BinaryOp):
        lv = _eval_const(expr.left)
        r = _eval_const(expr.right)
        if expr.op == "+":
            return lv + r
        if expr.op == "-":
            return lv - r
        if expr.op == "*":
            return lv * r
        if expr.op == "/":
            return lv / r
        if expr.op == "**":
            return float(lv**r)
        raise ValueError(f"Unknown op in const eval: {expr.op}")
    if isinstance(expr, UnaryOp):
        uv = _eval_const(expr.operand)
        if expr.op == "neg":
            return float(-uv)
        if expr.op == "abs":
            return float(abs(uv))
        raise ValueError(f"Unknown unary op in const eval: {expr.op}")
    if isinstance(expr, SumOverExpression):
        return sum(_eval_const(t) for t in expr.terms)
    if isinstance(expr, SumExpression):
        return _eval_const(expr.operand)
    raise ValueError(f"Not a constant expression: {type(expr).__name__}")


def _extract_constraints_algebraic(model: Model, n_orig: int):
    """Extract linear constraint data algebraically (shared by LP and QP paths).

    Returns (A_eq, b_eq, x_l, x_u, n_slack) where slacks are appended for
    inequality constraints.

    Raises _NotLinearError if any constraint is not linear.
    """
    constraints = [con for con in model._constraints if isinstance(con, Constraint)]

    eq_rows = []
    eq_rhs = []
    ineq_rows = []
    ineq_senses = []
    ineq_rhs = []

    for con in constraints:
        a_row, const = _extract_linear_coefficients(con.body, model, n_orig)
        if con.sense == "==":
            eq_rows.append(a_row)
            eq_rhs.append(-const)
        elif con.sense == "<=":
            ineq_rows.append(a_row)
            ineq_senses.append("le")
            ineq_rhs.append(-const)
        elif con.sense == ">=":
            ineq_rows.append(a_row)
            ineq_senses.append("ge")
            ineq_rhs.append(-const)

    n_eq = len(eq_rows)
    n_ineq = len(ineq_rows)
    n_slack = n_ineq
    n_total = n_orig + n_slack

    A_rows = []
    b_vals = []

    for i in range(n_eq):
        row_full = np.zeros(n_total, dtype=np.float64)
        row_full[:n_orig] = eq_rows[i]
        A_rows.append(row_full)
        b_vals.append(eq_rhs[i])

    for i in range(n_ineq):
        row_full = np.zeros(n_total, dtype=np.float64)
        row_full[:n_orig] = ineq_rows[i]
        if ineq_senses[i] == "le":
            row_full[n_orig + i] = 1.0
        else:
            row_full[n_orig + i] = -1.0
        A_rows.append(row_full)
        b_vals.append(ineq_rhs[i])

    m_total = n_eq + n_ineq
    if m_total > 0:
        A_eq = np.stack(A_rows).astype(np.float64)
        b_eq = np.array(b_vals, dtype=np.float64)
    else:
        A_eq = np.zeros((0, n_total), dtype=np.float64)
        b_eq = np.zeros(0, dtype=np.float64)

    x_l_orig, x_u_orig = _get_variable_bounds(model)
    x_l = np.concatenate([x_l_orig, np.zeros(n_slack, dtype=np.float64)])
    x_u = np.concatenate([x_u_orig, np.full(n_slack, 1e20, dtype=np.float64)])

    return A_eq, b_eq, x_l, x_u, n_slack


def extract_lp_data_algebraic(model: Model) -> LPData:
    """Extract LP standard form by walking the expression DAG algebraically.

    Much faster than extract_lp_data() because it avoids JAX tracing/autodiff.
    Returns numpy arrays — the solver converts to jnp at solve time.

    Raises _NotLinearError if the model is not linear.
    """
    from discopt.modeling.core import ObjectiveSense

    n_orig = sum(v.size for v in model._variables)
    assert model._objective is not None
    obj_expr = model._objective.expression

    c, obj_const = _extract_linear_coefficients(obj_expr, model, n_orig)

    A_eq, b_eq, x_l, x_u, n_slack = _extract_constraints_algebraic(model, n_orig)
    c_full = np.concatenate([c, np.zeros(n_slack, dtype=np.float64)])

    # Handle objective sense: negate for maximization
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        c_full = -c_full
        obj_const = -obj_const

    return LPData(
        c=jnp.asarray(c_full),  # type: ignore[arg-type]
        A_eq=A_eq,
        b_eq=b_eq,
        x_l=x_l,
        x_u=x_u,
        obj_const=obj_const,
    )


def extract_qp_data_algebraic(model: Model) -> QPData:
    """Extract QP standard form by walking the expression DAG algebraically.

    Much faster than extract_qp_data() because it avoids jax.hessian tracing.
    Returns numpy arrays — the solver converts to jnp at solve time.

    Raises _NotQuadraticError if the objective is not quadratic.
    """
    from discopt.modeling.core import ObjectiveSense

    n_orig = sum(v.size for v in model._variables)
    assert model._objective is not None
    obj_expr = model._objective.expression

    Q, c_vec, obj_const = _extract_quadratic_coefficients(obj_expr, model, n_orig)

    A_eq, b_eq, x_l, x_u, n_slack = _extract_constraints_algebraic(model, n_orig)

    if n_slack > 0:
        n_total = n_orig + n_slack
        Q_full = np.zeros((n_total, n_total), dtype=np.float64)
        Q_full[:n_orig, :n_orig] = Q
        c_full = np.concatenate([c_vec, np.zeros(n_slack, dtype=np.float64)])
    else:
        Q_full = Q
        c_full = c_vec

    # Handle objective sense: negate for maximization
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        Q_full = -Q_full
        c_full = -c_full
        obj_const = -obj_const

    return QPData(
        Q=jnp.asarray(Q_full),  # type: ignore[arg-type]
        c=jnp.asarray(c_full),  # type: ignore[arg-type]
        A_eq=A_eq,
        b_eq=b_eq,
        x_l=x_l,
        x_u=x_u,
        obj_const=obj_const,
    )


def _extract_lp_data_from_repr(model: Model) -> LPData:
    """Extract LP data by evaluating the Rust ModelRepr at unit vectors.

    For linear functions, c_j = f(e_j) - f(0) and A_ij = g_i(e_j) - g_i(0).
    This works for fast-API models where Python expression trees don't exist.
    """
    from discopt._rust import model_to_repr

    _builder = getattr(model, "_builder", None)
    repr_ = model_to_repr(model, _builder)

    n_orig = repr_.n_vars
    n_con = repr_.n_constraints

    x_zero = np.zeros(n_orig, dtype=np.float64)
    obj_at_zero = repr_.evaluate_objective(x_zero)

    # Extract objective coefficients
    c = np.zeros(n_orig, dtype=np.float64)
    for j in range(n_orig):
        ej = np.zeros(n_orig, dtype=np.float64)
        ej[j] = 1.0
        c[j] = repr_.evaluate_objective(ej) - obj_at_zero

    # Extract constraint data
    eq_rows = []
    eq_rhs = []
    ineq_rows = []
    ineq_senses = []
    ineq_rhs = []

    for i in range(n_con):
        sense = repr_.constraint_sense(i)
        rhs_val = repr_.constraint_rhs(i)
        g_at_zero = repr_.evaluate_constraint(i, x_zero)

        a_row = np.zeros(n_orig, dtype=np.float64)
        for j in range(n_orig):
            ej = np.zeros(n_orig, dtype=np.float64)
            ej[j] = 1.0
            a_row[j] = repr_.evaluate_constraint(i, ej) - g_at_zero

        if sense == "==":
            eq_rows.append(a_row)
            eq_rhs.append(rhs_val - g_at_zero)
        elif sense == "<=":
            ineq_rows.append(a_row)
            ineq_senses.append("le")
            ineq_rhs.append(rhs_val - g_at_zero)
        elif sense == ">=":
            ineq_rows.append(a_row)
            ineq_senses.append("ge")
            ineq_rhs.append(rhs_val - g_at_zero)

    n_eq = len(eq_rows)
    n_ineq = len(ineq_rows)
    n_slack = n_ineq
    n_total = n_orig + n_slack

    A_rows = []
    b_vals = []

    for i in range(n_eq):
        row_full = np.zeros(n_total, dtype=np.float64)
        row_full[:n_orig] = eq_rows[i]
        A_rows.append(row_full)
        b_vals.append(eq_rhs[i])

    for i in range(n_ineq):
        row_full = np.zeros(n_total, dtype=np.float64)
        row_full[:n_orig] = ineq_rows[i]
        if ineq_senses[i] == "le":
            row_full[n_orig + i] = 1.0
        else:
            row_full[n_orig + i] = -1.0
        A_rows.append(row_full)
        b_vals.append(ineq_rhs[i])

    m_total = n_eq + n_ineq
    if m_total > 0:
        A_eq = np.stack(A_rows).astype(np.float64)
        b_eq = np.array(b_vals, dtype=np.float64)
    else:
        A_eq = np.zeros((0, n_total), dtype=np.float64)
        b_eq = np.zeros(0, dtype=np.float64)

    x_l_orig, x_u_orig = _get_variable_bounds(model)
    c_full = np.concatenate([c, np.zeros(n_slack, dtype=np.float64)])
    x_l = np.concatenate([x_l_orig, np.zeros(n_slack, dtype=np.float64)])
    x_u = np.concatenate([x_u_orig, np.full(n_slack, 1e20, dtype=np.float64)])

    obj_sense = repr_.objective_sense
    if obj_sense == "maximize":
        c_full = -c_full
        obj_at_zero = -obj_at_zero

    return LPData(
        c=jnp.asarray(c_full),
        A_eq=jnp.asarray(A_eq),
        b_eq=jnp.asarray(b_eq),
        x_l=jnp.asarray(x_l),
        x_u=jnp.asarray(x_u),
        obj_const=obj_at_zero,
    )


def _extract_qp_data_from_repr(model: Model) -> QPData:
    """Extract QP data by evaluating the Rust ModelRepr numerically.

    For the quadratic objective 0.5 x'Qx + c'x + d:
      - d = f(0)
      - c_j = f(e_j) - d - 0.5*Q[j,j]   but Q[j,j] = f(e_j) + f(-e_j) - 2*d
      - Q[i,j] = f(e_i+e_j) - f(e_i) - f(e_j) + d  (for i != j)

    Constraints are extracted as in the LP case.
    """
    from discopt._rust import model_to_repr

    _builder = getattr(model, "_builder", None)
    repr_ = model_to_repr(model, _builder)

    n_orig = repr_.n_vars
    x_zero = np.zeros(n_orig, dtype=np.float64)
    d = repr_.evaluate_objective(x_zero)

    # Evaluate at all unit vectors
    f_ej = np.zeros(n_orig, dtype=np.float64)
    f_neg_ej = np.zeros(n_orig, dtype=np.float64)
    for j in range(n_orig):
        ej = np.zeros(n_orig, dtype=np.float64)
        ej[j] = 1.0
        f_ej[j] = repr_.evaluate_objective(ej)
        ej[j] = -1.0
        f_neg_ej[j] = repr_.evaluate_objective(ej)

    # Q diagonal: Q[j,j] = f(e_j) + f(-e_j) - 2*d
    Q = np.zeros((n_orig, n_orig), dtype=np.float64)
    for j in range(n_orig):
        Q[j, j] = f_ej[j] + f_neg_ej[j] - 2 * d

    # Q off-diagonal: Q[i,j] = f(e_i+e_j) - f(e_i) - f(e_j) + d
    for i in range(n_orig):
        for j in range(i + 1, n_orig):
            eij = np.zeros(n_orig, dtype=np.float64)
            eij[i] = 1.0
            eij[j] = 1.0
            f_eij = repr_.evaluate_objective(eij)
            qij = f_eij - f_ej[i] - f_ej[j] + d
            Q[i, j] = qij
            Q[j, i] = qij

    # Linear coefficients: c_j = f(e_j) - d - 0.5*Q[j,j]
    c_vec = np.zeros(n_orig, dtype=np.float64)
    for j in range(n_orig):
        c_vec[j] = f_ej[j] - d - 0.5 * Q[j, j]

    # Extract constraints (same as LP)
    lp_data = _extract_lp_data_from_repr(model)
    n_slack = lp_data.c.shape[0] - n_orig

    if n_slack > 0:
        n_total = n_orig + n_slack
        Q_full = np.zeros((n_total, n_total), dtype=np.float64)
        Q_full[:n_orig, :n_orig] = Q
        c_full = np.concatenate([c_vec, np.zeros(n_slack, dtype=np.float64)])
    else:
        Q_full = Q
        c_full = c_vec

    return QPData(
        Q=jnp.asarray(Q_full),
        c=jnp.asarray(c_full),
        A_eq=lp_data.A_eq,
        b_eq=lp_data.b_eq,
        x_l=lp_data.x_l,
        x_u=lp_data.x_u,
        obj_const=d,
    )


def extract_lp_data(model: Model) -> LPData:
    """Extract LP standard form from a model classified as LP.

    Tries Rust repr-based extraction first (for fast-API models), then
    algebraic extraction (for expression-based), then falls back to
    autodiff-based extraction if the DAG walk fails.

    Inequality constraints are converted to equalities with slacks:
      - body <= 0 becomes body + s = 0, s >= 0
      - body >= 0 becomes body - s = 0, s >= 0

    Args:
        model: A Model classified as ProblemClass.LP.

    Returns:
        LPData with c, A_eq, b_eq, x_l, x_u.
    """
    # Try repr-based extraction first (works for fast-API models)
    _builder = getattr(model, "_builder", None)
    if _builder is not None:
        try:
            return _extract_lp_data_from_repr(model)
        except Exception:
            pass

    try:
        return extract_lp_data_algebraic(model)
    except (_NotLinearError, Exception):
        pass

    return _extract_lp_data_autodiff(model)


def _extract_lp_data_autodiff(model: Model) -> LPData:
    """Extract LP standard form using autodiff (original slow path)."""
    from discopt._jax.dag_compiler import compile_constraint, compile_objective

    n_orig = sum(v.size for v in model._variables)
    obj_fn = compile_objective(model)

    # Extract c and constant: obj(x) = c'x + d, so grad(obj)(0) = c, obj(0) = d
    x_zero = jnp.zeros(n_orig, dtype=jnp.float64)
    c = jax.grad(obj_fn)(x_zero)
    obj_const = float(obj_fn(x_zero))

    # Extract constraint coefficients
    constraints = [con for con in model._constraints if isinstance(con, Constraint)]

    # Separate equality and inequality constraints
    eq_fns = []
    eq_rhs = []
    ineq_fns = []  # body <= 0 form
    ineq_senses = []  # "le" or "ge"

    for con in constraints:
        con_fn = compile_constraint(con, model)
        if con.sense == "==":
            eq_fns.append(con_fn)
            # body - rhs == 0 is compiled, so rhs = 0 in compiled form
            eq_rhs.append(0.0)
        elif con.sense == "<=":
            ineq_fns.append(con_fn)
            ineq_senses.append("le")
        elif con.sense == ">=":
            ineq_fns.append(con_fn)
            ineq_senses.append("ge")

    n_eq = len(eq_fns)
    n_ineq = len(ineq_fns)
    n_slack = n_ineq
    n_total = n_orig + n_slack

    # Build equality constraint matrix for original equality constraints
    A_rows = []
    b_vals = []

    for i, fn in enumerate(eq_fns):
        # For linear constraint, A_row = grad(con)(0)
        a_row = jax.grad(fn)(x_zero)
        # Constant part: fn(0) = a'*0 + const = const, so b = -const
        const = fn(x_zero)
        A_rows.append(jnp.concatenate([a_row, jnp.zeros(n_slack)]))
        b_vals.append(-float(const))

    # Convert inequality constraints to equalities with slacks
    for i, fn in enumerate(ineq_fns):
        a_row = jax.grad(fn)(x_zero)
        const = fn(x_zero)
        slack_col = jnp.zeros(n_slack)
        if ineq_senses[i] == "le":
            # body <= 0 → body + s = 0, s >= 0
            slack_col = slack_col.at[i].set(1.0)
        else:
            # body >= 0 → body - s = 0, s >= 0
            slack_col = slack_col.at[i].set(-1.0)
        A_rows.append(jnp.concatenate([a_row, slack_col]))
        b_vals.append(-float(const))

    m_total = n_eq + n_ineq
    if m_total > 0:
        A_eq = jnp.stack(A_rows)
        b_eq = jnp.array(b_vals, dtype=jnp.float64)
    else:
        A_eq = jnp.zeros((0, n_total), dtype=jnp.float64)
        b_eq = jnp.zeros(0, dtype=jnp.float64)

    # Bounds: original vars keep their bounds, slack vars >= 0
    x_l_orig, x_u_orig = _get_variable_bounds(model)
    c_full = jnp.concatenate([c, jnp.zeros(n_slack)])
    x_l = jnp.concatenate([x_l_orig, jnp.zeros(n_slack)])
    x_u = jnp.concatenate([x_u_orig, jnp.full(n_slack, 1e20)])

    return LPData(
        c=c_full,
        A_eq=A_eq,
        b_eq=b_eq,
        x_l=x_l,
        x_u=x_u,
        obj_const=obj_const,
    )


def extract_qp_data(model: Model) -> QPData:
    """Extract QP standard form from a model classified as QP.

    Tries Rust repr-based extraction first (for fast-API models), then
    algebraic extraction (for expression-based), then falls back to
    autodiff-based extraction if the DAG walk fails.

    Args:
        model: A Model classified as ProblemClass.QP.

    Returns:
        QPData with Q, c, A_eq, b_eq, x_l, x_u.
    """
    _builder = getattr(model, "_builder", None)
    if _builder is not None:
        try:
            return _extract_qp_data_from_repr(model)
        except Exception:
            pass

    try:
        return extract_qp_data_algebraic(model)
    except (_NotQuadraticError, _NotLinearError, Exception):
        pass

    return _extract_qp_data_autodiff(model)


def _extract_qp_data_autodiff(model: Model) -> QPData:
    """Extract QP standard form using autodiff (original slow path)."""
    from discopt._jax.dag_compiler import compile_objective

    n_orig = sum(v.size for v in model._variables)
    obj_fn = compile_objective(model)

    x_zero = jnp.zeros(n_orig, dtype=jnp.float64)

    # Q = hessian(obj) — constant for QP
    Q = jax.hessian(obj_fn)(x_zero)

    # c = grad(obj)(0) = Q*0 + c = c (linear part)
    c_vec = jax.grad(obj_fn)(x_zero)

    # Constant term: f(0) = 0.5*0'Q*0 + c'*0 + d = d
    obj_const = float(obj_fn(x_zero))

    # Extract LP data for constraints (they're all linear)
    lp_data = extract_lp_data(model)
    n_slack = lp_data.c.shape[0] - n_orig

    # Extend Q with zeros for slack variables
    if n_slack > 0:
        n_total = n_orig + n_slack
        Q_full = jnp.zeros((n_total, n_total), dtype=jnp.float64)
        Q_full = Q_full.at[:n_orig, :n_orig].set(Q)
        c_full = jnp.concatenate([c_vec, jnp.zeros(n_slack)])
    else:
        Q_full = Q
        c_full = c_vec

    return QPData(
        Q=Q_full,
        c=c_full,
        A_eq=lp_data.A_eq,
        b_eq=lp_data.b_eq,
        x_l=lp_data.x_l,
        x_u=lp_data.x_u,
        obj_const=obj_const,
    )
