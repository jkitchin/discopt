"""
Differentiable optimization: compute gradients through the solve via custom_jvp.

Level 1 implementation uses the envelope theorem:
  For min_x f(x; p) s.t. g(x; p) <= 0,
  d(obj*)/dp = dL/dp |_{x*, lambda*}
             = df/dp |_{x*} + lambda*^T dg/dp |_{x*}

where L is the Lagrangian, x* is the optimal primal, and lambda* are the
optimal dual variables (constraint multipliers) returned by Ipopt.

This avoids solving any linear system -- the duals from the NLP solve
directly provide the sensitivity information we need.
"""

from __future__ import annotations

from typing import Optional

import jax
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
    ObjectiveSense,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)

# ---------------------------------------------------------------------------
# Parametric DAG compiler
#
# Unlike the standard dag_compiler which bakes Parameter.value into the
# compiled function as constants, this compiler produces functions of the
# form f(x_flat, p_flat) where p_flat is a concatenated vector of all
# parameter values.
# ---------------------------------------------------------------------------


def _compute_var_offset(var: Variable, model: Model) -> int:
    """Compute the starting offset of a variable in the flat x vector."""
    offset = 0
    for v in model._variables[: var._index]:
        offset += v.size
    return offset


def _compute_param_offset(param: Parameter, model: Model) -> int:
    """Compute the starting offset of a parameter in the flat p vector."""
    offset = 0
    for p in model._parameters:
        if p is param:
            return offset
        offset += int(np.prod(p.shape)) if p.shape else 1
    raise ValueError(f"Parameter {param.name!r} not found in model")


def _param_total_size(model: Model) -> int:
    """Total number of scalar parameter values."""
    total = 0
    for p in model._parameters:
        total += int(np.prod(p.shape)) if p.shape else 1
    return total


def _compile_parametric_node(expr: Expression, model: Model):
    """Compile an expression node into f(x_flat, p_flat) -> value."""
    if isinstance(expr, Constant):
        val = jnp.array(expr.value)

        def fn(x_flat, p_flat):
            return val

        return fn

    if isinstance(expr, Variable):
        offset = _compute_var_offset(expr, model)
        size = expr.size
        shape = expr.shape
        if shape == () or (len(shape) == 1 and shape[0] == 1 and shape == ()):

            def fn(x_flat, p_flat):
                return x_flat[offset]

            return fn
        else:

            def fn(x_flat, p_flat, _offset=offset, _size=size, _shape=shape):
                return x_flat[_offset : _offset + _size].reshape(_shape)

            return fn

    if isinstance(expr, Parameter):
        p_offset = _compute_param_offset(expr, model)
        p_size = int(np.prod(expr.shape)) if expr.shape else 1
        p_shape = expr.shape

        if p_shape == () or p_size == 1:

            def fn(x_flat, p_flat):
                return p_flat[p_offset]

            return fn
        else:

            def fn(x_flat, p_flat, _off=p_offset, _sz=p_size, _sh=p_shape):
                return p_flat[_off : _off + _sz].reshape(_sh)

            return fn

    if isinstance(expr, BinaryOp):
        left_fn = _compile_parametric_node(expr.left, model)
        right_fn = _compile_parametric_node(expr.right, model)
        op = expr.op
        if op == "+":

            def fn(x_flat, p_flat):
                return left_fn(x_flat, p_flat) + right_fn(x_flat, p_flat)
        elif op == "-":

            def fn(x_flat, p_flat):
                return left_fn(x_flat, p_flat) - right_fn(x_flat, p_flat)
        elif op == "*":

            def fn(x_flat, p_flat):
                return left_fn(x_flat, p_flat) * right_fn(x_flat, p_flat)
        elif op == "/":

            def fn(x_flat, p_flat):
                return left_fn(x_flat, p_flat) / right_fn(x_flat, p_flat)
        elif op == "**":

            def fn(x_flat, p_flat):
                return left_fn(x_flat, p_flat) ** right_fn(x_flat, p_flat)
        else:
            raise ValueError(f"Unknown binary operator: {op!r}")
        return fn

    if isinstance(expr, UnaryOp):
        operand_fn = _compile_parametric_node(expr.operand, model)
        op = expr.op
        if op == "neg":

            def fn(x_flat, p_flat):
                return -operand_fn(x_flat, p_flat)
        elif op == "abs":

            def fn(x_flat, p_flat):
                return jnp.abs(operand_fn(x_flat, p_flat))
        else:
            raise ValueError(f"Unknown unary operator: {op!r}")
        return fn

    if isinstance(expr, FunctionCall):
        arg_fns = [_compile_parametric_node(a, model) for a in expr.args]
        name = expr.func_name

        _unary_funcs = {
            "exp": jnp.exp,
            "log": jnp.log,
            "log2": jnp.log2,
            "log10": jnp.log10,
            "sqrt": jnp.sqrt,
            "sin": jnp.sin,
            "cos": jnp.cos,
            "tan": jnp.tan,
            "abs": jnp.abs,
            "sign": jnp.sign,
        }

        if name in _unary_funcs:
            jax_fn = _unary_funcs[name]
            a_fn = arg_fns[0]

            def fn(x_flat, p_flat, _jax_fn=jax_fn, _a_fn=a_fn):
                return _jax_fn(_a_fn(x_flat, p_flat))

            return fn

        if name == "min":
            a_fn, b_fn = arg_fns[0], arg_fns[1]

            def fn(x_flat, p_flat):
                return jnp.minimum(a_fn(x_flat, p_flat), b_fn(x_flat, p_flat))

            return fn

        if name == "max":
            a_fn, b_fn = arg_fns[0], arg_fns[1]

            def fn(x_flat, p_flat):
                return jnp.maximum(a_fn(x_flat, p_flat), b_fn(x_flat, p_flat))

            return fn

        if name == "prod":
            a_fn = arg_fns[0]

            def fn(x_flat, p_flat):
                return jnp.prod(a_fn(x_flat, p_flat))

            return fn

        if name == "norm2":
            a_fn = arg_fns[0]

            def fn(x_flat, p_flat):
                return jnp.linalg.norm(a_fn(x_flat, p_flat), ord=2)

            return fn

        raise ValueError(f"Unknown function: {name!r}")

    if isinstance(expr, IndexExpression):
        base_fn = _compile_parametric_node(expr.base, model)
        idx = expr.index

        def fn(x_flat, p_flat, _idx=idx):
            return base_fn(x_flat, p_flat)[_idx]

        return fn

    if isinstance(expr, MatMulExpression):
        left_fn = _compile_parametric_node(expr.left, model)
        right_fn = _compile_parametric_node(expr.right, model)

        def fn(x_flat, p_flat):
            return left_fn(x_flat, p_flat) @ right_fn(x_flat, p_flat)

        return fn

    if isinstance(expr, SumExpression):
        operand_fn = _compile_parametric_node(expr.operand, model)
        axis = expr.axis

        def fn(x_flat, p_flat, _axis=axis):
            return jnp.sum(operand_fn(x_flat, p_flat), axis=_axis)

        return fn

    if isinstance(expr, SumOverExpression):
        term_fns = [_compile_parametric_node(t, model) for t in expr.terms]

        def fn(x_flat, p_flat):
            result = term_fns[0](x_flat, p_flat)
            for t_fn in term_fns[1:]:
                result = result + t_fn(x_flat, p_flat)
            return result

        return fn

    raise TypeError(f"Unhandled expression type: {type(expr).__name__}")


def _compile_parametric_objective(model: Model):
    """Compile model objective into f(x_flat, p_flat) -> scalar."""
    if model._objective is None:
        raise ValueError("Model has no objective set.")
    raw_fn = _compile_parametric_node(model._objective.expression, model)
    if model._objective.sense == ObjectiveSense.MAXIMIZE:

        def obj_fn(x_flat, p_flat):
            return -raw_fn(x_flat, p_flat)

        return obj_fn
    return raw_fn


def _compile_parametric_constraint(constraint: Constraint, model: Model):
    """Compile a constraint body into f(x_flat, p_flat) -> scalar."""
    return _compile_parametric_node(constraint.body, model)


def _flatten_params(model: Model) -> jnp.ndarray:
    """Concatenate all parameter values into a flat array."""
    parts = []
    for p in model._parameters:
        parts.append(np.asarray(p.value, dtype=np.float64).ravel())
    if not parts:
        return jnp.zeros(0, dtype=jnp.float64)
    return jnp.array(np.concatenate(parts), dtype=jnp.float64)


def _get_param_slice(param: Parameter, model: Model) -> tuple[int, int]:
    """Get (start, end) indices for a parameter in the flat p vector."""
    offset = _compute_param_offset(param, model)
    size = int(np.prod(param.shape)) if param.shape else 1
    return offset, offset + size


# ---------------------------------------------------------------------------
# Differentiable solve
# ---------------------------------------------------------------------------


def differentiable_solve(
    model: Model,
    ipopt_options: Optional[dict] = None,
) -> "DiffSolveResult":
    """Solve a continuous model and return a result with parameter sensitivities.

    Uses the envelope theorem to compute d(obj*)/dp without solving any
    additional linear systems. The optimal dual variables (Lagrange
    multipliers) from Ipopt directly provide the sensitivity information.

    This function only works for purely continuous models (no integer variables).
    The model must have at least one Parameter for differentiation to be useful.

    Args:
        model: A Model with objective, constraints, and parameters.
        ipopt_options: Options dict passed to cyipopt.

    Returns:
        DiffSolveResult with solution and .gradient(param) method.
    """
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.modeling.core import VarType
    from discopt.solvers import SolveStatus
    from discopt.solvers.nlp_ipopt import solve_nlp

    model.validate()

    # Check all variables are continuous
    for v in model._variables:
        if v.var_type != VarType.CONTINUOUS:
            raise ValueError(
                "differentiable_solve only supports continuous models. "
                f"Variable '{v.name}' is {v.var_type.value}."
            )

    # Solve the NLP
    evaluator = NLPEvaluator(model)
    lb, ub = evaluator.variable_bounds
    lb_clipped = np.clip(lb, -100.0, 100.0)
    ub_clipped = np.clip(ub, -100.0, 100.0)
    x0 = 0.5 * (lb_clipped + ub_clipped)

    opts = dict(ipopt_options) if ipopt_options else {}
    opts.setdefault("print_level", 0)

    nlp_result = solve_nlp(evaluator, x0, options=opts)

    if nlp_result.status != SolveStatus.OPTIMAL:
        raise RuntimeError(f"NLP solve did not converge to optimal: {nlp_result.status.value}")

    x_star = nlp_result.x
    multipliers = nlp_result.multipliers  # lambda* from Ipopt

    # Build the parametric Lagrangian gradient w.r.t. p
    # L(x, lambda, p) = f(x; p) + lambda^T g(x; p)
    # dL/dp = df/dp + lambda^T dg/dp
    #
    # We compile parametric versions of objective and constraints,
    # then use jax.grad w.r.t. p_flat to get the sensitivities.

    obj_fn = _compile_parametric_objective(model)

    # Collect standard constraints
    constraint_fns = []
    for c in model._constraints:
        if isinstance(c, Constraint):
            constraint_fns.append(_compile_parametric_constraint(c, model))

    p_flat = _flatten_params(model)

    # Compute df/dp at (x*, p)
    x_star_jax = jnp.array(x_star, dtype=jnp.float64)

    # Build Lagrangian as a function of p only (x fixed at x*)
    if multipliers is not None and len(constraint_fns) > 0:
        mults = jnp.array(multipliers, dtype=jnp.float64)

        def lagrangian_p(p_flat_arg):
            obj_val = obj_fn(x_star_jax, p_flat_arg)
            # Ipopt convention: multipliers for g(x) in [cl, cu]
            # For <= constraints (body <= 0): mult >= 0 when active
            # The envelope theorem gives dL/dp = df/dp + lambda^T dg/dp
            con_vals = jnp.array([cf(x_star_jax, p_flat_arg) for cf in constraint_fns])
            return obj_val + jnp.dot(mults, con_vals)
    else:

        def lagrangian_p(p_flat_arg):
            return obj_fn(x_star_jax, p_flat_arg)

    grad_lagrangian_p = jax.grad(lagrangian_p)
    sensitivity = np.asarray(grad_lagrangian_p(p_flat))

    # Unpack solution
    x_dict = {}
    offset = 0
    for v in model._variables:
        size = v.size
        val = x_star[offset : offset + size]
        x_dict[v.name] = val.reshape(v.shape) if v.shape != () else val
        offset += size

    return DiffSolveResult(
        status="optimal",
        objective=nlp_result.objective,
        x=x_dict,
        _model=model,
        _sensitivity=sensitivity,
    )


class DiffSolveResult:
    """Result of a differentiable solve with parameter sensitivity support."""

    def __init__(
        self,
        status: str,
        objective: Optional[float],
        x: Optional[dict[str, np.ndarray]],
        _model: Model,
        _sensitivity: np.ndarray,
    ):
        self.status = status
        self.objective = objective
        self.x = x
        self._model = _model
        self._sensitivity = _sensitivity

    def value(self, var: Variable) -> np.ndarray:
        """Get the optimal value of a variable."""
        if self.x is None:
            raise ValueError("No solution available")
        return self.x[var.name]

    def gradient(self, param: Parameter) -> np.ndarray:
        """Get d(obj*)/d(param) via the envelope theorem.

        For a minimization problem min f(x; p) s.t. g(x; p) <= 0,
        the sensitivity of the optimal objective to parameter p is:
            d(obj*)/dp = dL/dp |_{x*, lambda*}

        Args:
            param: A Parameter from the model.

        Returns:
            Array of same shape as param.value containing the sensitivities.
        """
        start, end = _get_param_slice(param, self._model)
        grad_flat = self._sensitivity[start:end]
        if param.shape == () or (end - start) == 1:
            return float(grad_flat[0])
        return grad_flat.reshape(param.shape)

    def __repr__(self) -> str:
        return f"DiffSolveResult(status={self.status!r}, obj={self.objective})"


# ---------------------------------------------------------------------------
# JAX-native differentiable solve via custom_jvp
#
# This allows embedding the solve inside a larger JAX computation and
# differentiating through it with jax.grad.
# ---------------------------------------------------------------------------


def _make_jax_differentiable_solve(model: Model, ipopt_options: Optional[dict] = None):
    """Create a JAX-differentiable function p_flat -> obj* for the model.

    Returns a function that maps parameter values to optimal objective value,
    and is compatible with jax.grad / jax.jvp.

    Args:
        model: A Model with objective, constraints, and parameters.
        ipopt_options: Options dict passed to cyipopt.

    Returns:
        A function solve_fn(p_flat) -> scalar that supports jax.grad.
    """
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.modeling.core import VarType
    from discopt.solvers.nlp_ipopt import solve_nlp

    model.validate()

    for v in model._variables:
        if v.var_type != VarType.CONTINUOUS:
            raise ValueError(
                "JAX differentiable solve only supports continuous models. "
                f"Variable '{v.name}' is {v.var_type.value}."
            )

    evaluator = NLPEvaluator(model)
    lb, ub = evaluator.variable_bounds
    lb_clipped = np.clip(lb, -100.0, 100.0)
    ub_clipped = np.clip(ub, -100.0, 100.0)
    x0_default = 0.5 * (lb_clipped + ub_clipped)

    opts = dict(ipopt_options) if ipopt_options else {}
    opts.setdefault("print_level", 0)

    n_vars = evaluator.n_variables
    n_constraints = evaluator.n_constraints

    # Pre-compile parametric functions
    obj_fn_parametric = _compile_parametric_objective(model)
    constraint_fns_parametric = []
    for c in model._constraints:
        if isinstance(c, Constraint):
            constraint_fns_parametric.append(_compile_parametric_constraint(c, model))

    @jax.custom_jvp
    def solve_fn(p_flat):
        """Solve the NLP for given parameter values and return optimal objective."""
        # Update parameter values in the model
        offset = 0
        for p in model._parameters:
            p_size = int(np.prod(p.shape)) if p.shape else 1
            p.value = np.asarray(p_flat[offset : offset + p_size]).reshape(p.shape)
            offset += p_size

        # Re-compile evaluator with updated parameter values
        ev = NLPEvaluator(model)

        def _solve(p_np):
            result = solve_nlp(ev, x0_default, options=opts)
            x_sol = result.x if result.x is not None else x0_default
            mults = result.multipliers
            obj = result.objective if result.objective is not None else 0.0
            # Pack: [obj, x_star..., multipliers...]
            if mults is not None:
                packed = np.concatenate([np.array([obj]), x_sol, mults]).astype(np.float64)
            else:
                packed = np.concatenate([np.array([obj]), x_sol, np.zeros(n_constraints)]).astype(
                    np.float64
                )
            return packed

        result_shape = jax.ShapeDtypeStruct((1 + n_vars + n_constraints,), jnp.float64)
        packed = jax.pure_callback(_solve, result_shape, p_flat)
        return packed[0]

    @solve_fn.defjvp
    def solve_fn_jvp(primals, tangents):
        (p_flat,) = primals
        (p_dot,) = tangents

        # Forward: solve the problem
        # We need x* and lambda* for the JVP, so we call the callback again
        offset = 0
        for p in model._parameters:
            p_size = int(np.prod(p.shape)) if p.shape else 1
            p.value = np.asarray(p_flat[offset : offset + p_size]).reshape(p.shape)
            offset += p_size

        ev = NLPEvaluator(model)

        def _solve_full(p_np):
            result = solve_nlp(ev, x0_default, options=opts)
            x_sol = result.x if result.x is not None else x0_default
            mults = result.multipliers
            obj = result.objective if result.objective is not None else 0.0
            if mults is not None:
                packed = np.concatenate([np.array([obj]), x_sol, mults]).astype(np.float64)
            else:
                packed = np.concatenate([np.array([obj]), x_sol, np.zeros(n_constraints)]).astype(
                    np.float64
                )
            return packed

        result_shape = jax.ShapeDtypeStruct((1 + n_vars + n_constraints,), jnp.float64)
        packed = jax.pure_callback(_solve_full, result_shape, p_flat)

        primal_out = packed[0]
        x_star = packed[1 : 1 + n_vars]
        mults = packed[1 + n_vars :]

        # JVP via envelope theorem:
        # d(obj*)/dp . dp_dot = (df/dp + lambda^T dg/dp) . dp_dot
        # Compute this by differentiating the Lagrangian w.r.t. p
        if len(constraint_fns_parametric) > 0:

            def lagrangian_p(p_arg):
                obj_val = obj_fn_parametric(x_star, p_arg)
                con_vals = jnp.array([cf(x_star, p_arg) for cf in constraint_fns_parametric])
                return obj_val + jnp.dot(mults, con_vals)
        else:

            def lagrangian_p(p_arg):
                return obj_fn_parametric(x_star, p_arg)

        _, tangent_out = jax.jvp(lagrangian_p, (p_flat,), (p_dot,))

        return primal_out, tangent_out

    return solve_fn
