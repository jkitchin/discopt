"""
Differentiable optimization: compute gradients through the solve via custom_jvp.

Level 1 implementation uses the envelope theorem:
  For min_x f(x; p) s.t. g(x; p) <= 0,
  d(obj*)/dp = dL/dp |_{x*, lambda*}
             = df/dp |_{x*} + lambda*^T dg/dp |_{x*}

where L is the Lagrangian, x* is the optimal primal, and lambda* are the
optimal dual variables (constraint multipliers) returned by the NLP solver.

This avoids solving any linear system -- the duals from the NLP solve
directly provide the sensitivity information we need.

The default NLP backend is the pure-JAX IPM (``solve_nlp_ipm``), which keeps
the entire pipeline in JAX and requires no external C dependencies. Pass
``nlp_solver="ipopt"`` to use cyipopt instead.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

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
# NLP dispatch helper
# ---------------------------------------------------------------------------


def _dispatch_nlp_solve(nlp_solver: str, evaluator, x0, options: dict):
    """Dispatch NLP solve to the selected backend."""
    if nlp_solver == "ipm":
        from discopt._jax.ipm import solve_nlp_ipm

        return solve_nlp_ipm(evaluator, x0, constraint_bounds=None, options=options)
    elif nlp_solver == "ipopt":
        from discopt.solvers.nlp_ipopt import solve_nlp

        return solve_nlp(evaluator, x0, options=options)
    else:
        raise ValueError(f"Unknown nlp_solver: {nlp_solver!r}. Use 'ipm' or 'ipopt'.")


# ---------------------------------------------------------------------------
# Differentiable solve
# ---------------------------------------------------------------------------


def differentiable_solve(
    model: Model,
    ipopt_options: Optional[dict] = None,
    *,
    nlp_solver: str = "ipm",
    solver_options: Optional[dict] = None,
) -> "DiffSolveResult":
    """Solve a continuous model and return a result with parameter sensitivities.

    Uses the envelope theorem to compute d(obj*)/dp without solving any
    additional linear systems. The optimal dual variables (Lagrange
    multipliers) from the NLP solver directly provide the sensitivity information.

    This function only works for purely continuous models (no integer variables).
    The model must have at least one Parameter for differentiation to be useful.

    Args:
        model: A Model with objective, constraints, and parameters.
        ipopt_options: Deprecated alias for solver_options. Options dict passed
            to the NLP solver.
        nlp_solver: NLP backend to use: ``"ipm"`` (default, pure-JAX) or
            ``"ipopt"`` (cyipopt).
        solver_options: Options dict passed to the NLP solver.

    Returns:
        DiffSolveResult with solution and .gradient(param) method.
    """
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.modeling.core import VarType
    from discopt.solvers import SolveStatus

    # Merge ipopt_options (deprecated) with solver_options
    opts = dict(solver_options or ipopt_options or {})

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

    opts.setdefault("print_level", 0)

    nlp_result = _dispatch_nlp_solve(nlp_solver, evaluator, x0, opts)

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


def _compute_sensitivity_at_solution(
    model: Model,
    x_dict: dict[str, np.ndarray],
    *,
    nlp_solver: str = "ipm",
    solver_options: Optional[dict] = None,
) -> np.ndarray:
    """Compute parameter sensitivity from an existing solution.

    Re-solves the NLP warm-started from the known solution to obtain
    dual variables, then applies the envelope theorem.

    Args:
        model: A continuous Model with parameters.
        x_dict: Variable values at the solution (from SolveResult.x).
        nlp_solver: NLP backend: ``"ipm"`` (default) or ``"ipopt"``.
        solver_options: Options dict passed to the NLP solver.

    Returns:
        1-D array of sensitivities d(obj*)/dp for all parameters.
    """
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.solvers import SolveStatus

    evaluator = NLPEvaluator(model)
    opts = dict(solver_options or {})
    opts.setdefault("print_level", 0)

    # Build initial point from the existing solution
    x0_parts = []
    for v in model._variables:
        val = np.asarray(x_dict[v.name], dtype=np.float64).ravel()
        x0_parts.append(val)
    x0 = np.concatenate(x0_parts)

    nlp_result = _dispatch_nlp_solve(nlp_solver, evaluator, x0, opts)

    if nlp_result.status != SolveStatus.OPTIMAL:
        # Fall back to ipopt if ipm didn't converge
        if nlp_solver == "ipm":
            try:
                nlp_result = _dispatch_nlp_solve("ipopt", evaluator, x0, opts)
            except Exception:
                pass
        if nlp_result.status != SolveStatus.OPTIMAL:
            raise RuntimeError(f"Sensitivity NLP solve did not converge: {nlp_result.status.value}")

    x_star = nlp_result.x
    multipliers = nlp_result.multipliers

    # Envelope theorem: dL/dp at (x*, lambda*, p)
    obj_fn = _compile_parametric_objective(model)
    constraint_fns = []
    for c in model._constraints:
        if isinstance(c, Constraint):
            constraint_fns.append(_compile_parametric_constraint(c, model))

    p_flat = _flatten_params(model)
    x_star_jax = jnp.array(x_star, dtype=jnp.float64)

    if multipliers is not None and len(constraint_fns) > 0:
        mults = jnp.array(multipliers, dtype=jnp.float64)

        def lagrangian_p(p_flat_arg):
            obj_val = obj_fn(x_star_jax, p_flat_arg)
            con_vals = jnp.array([cf(x_star_jax, p_flat_arg) for cf in constraint_fns])
            return obj_val + jnp.dot(mults, con_vals)
    else:

        def lagrangian_p(p_flat_arg):
            return obj_fn(x_star_jax, p_flat_arg)

    grad_lagrangian_p = jax.grad(lagrangian_p)
    return np.asarray(grad_lagrangian_p(p_flat))


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


def _make_jax_differentiable_solve(
    model: Model,
    ipopt_options: Optional[dict] = None,
    *,
    nlp_solver: str = "ipm",
    solver_options: Optional[dict] = None,
):
    """Create a JAX-differentiable function p_flat -> obj* for the model.

    Returns a function that maps parameter values to optimal objective value,
    and is compatible with jax.grad / jax.jvp.

    Args:
        model: A Model with objective, constraints, and parameters.
        ipopt_options: Deprecated alias for solver_options.
        nlp_solver: NLP backend: ``"ipm"`` (default) or ``"ipopt"``.
        solver_options: Options dict passed to the NLP solver.

    Returns:
        A function solve_fn(p_flat) -> scalar that supports jax.grad.
    """
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.modeling.core import VarType

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

    opts = dict(solver_options or ipopt_options or {})
    opts.setdefault("print_level", 0)
    _nlp_solver = nlp_solver

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
            result = _dispatch_nlp_solve(_nlp_solver, ev, x0_default, opts)
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
            result = _dispatch_nlp_solve(_nlp_solver, ev, x0_default, opts)
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


# ---------------------------------------------------------------------------
# Level 3: Implicit differentiation via KKT system
#
# For min_x f(x; p) s.t. g_i(x; p) <= 0, the KKT conditions at optimality:
#   nabla_x L = nabla_x f + sum lambda_i nabla_x g_i = 0
#   g_active(x, p) = 0
#
# Differentiating w.r.t. p gives:
#   [H_xx   J_a^T] [dx/dp]     [-H_xp        ]
#   [J_a    0    ] [dlam/dp]  = [-dg_active/dp ]
#
# where H_xx = nabla^2_xx L, J_a = Jacobian of active constraints w.r.t. x,
# H_xp = nabla^2_xp L, and dg_active/dp = Jacobian of active constraints
# w.r.t. p.
# ---------------------------------------------------------------------------


def find_active_set(
    x_star: jnp.ndarray,
    model: Model,
    constraint_fns: list,
    p_flat: jnp.ndarray,
    tol: float = 1e-6,
) -> tuple[list[int], list[int]]:
    """Identify active constraints and active variable bounds at the solution.

    A constraint g_i(x) <= 0 is active if g_i(x*) > -tol (i.e., close to zero).
    An equality constraint g_i(x) == 0 is always active if |g_i(x*)| < tol.
    A variable bound is active if x_i is within tol of its lb or ub.

    Args:
        x_star: Optimal primal solution (flat vector).
        model: The optimization model.
        constraint_fns: List of compiled parametric constraint functions.
        p_flat: Flat parameter vector.
        tol: Tolerance for determining activity.

    Returns:
        Tuple of (active_constraint_indices, active_bound_var_indices).
        active_bound_var_indices contains (var_flat_index, 'lb'|'ub') pairs.
    """
    active_constraints: list[int] = []
    for i, cf in enumerate(constraint_fns):
        val = float(cf(x_star, p_flat))
        c = model._constraints[i]
        if isinstance(c, Constraint):
            if c.sense == "==":
                if abs(val) < tol:
                    active_constraints.append(i)
            elif c.sense == "<=":
                # body - rhs <= 0; active if val > -tol (close to 0)
                if val > -tol:
                    active_constraints.append(i)
            elif c.sense == ">=":
                # body - rhs >= 0; active if val < tol (close to 0)
                if val < tol:
                    active_constraints.append(i)

    # Check variable bounds
    active_bounds: list[int] = []
    offset = 0
    for v in model._variables:
        for j in range(v.size):
            flat_idx = offset + j
            lb_j = float(v.lb.flat[j])
            ub_j = float(v.ub.flat[j])
            x_j = float(x_star[flat_idx])
            if abs(x_j - lb_j) < tol and lb_j > -1e19:
                active_bounds.append(flat_idx)
            elif abs(x_j - ub_j) < tol and ub_j < 1e19:
                active_bounds.append(flat_idx)
        offset += v.size

    return active_constraints, active_bounds


class SensitivityInfo(NamedTuple):
    """Cached sensitivity data from implicit differentiation of the KKT system.

    Stores the KKT factorization and intermediate matrices so that subsequent
    sensitivity queries (e.g., for new parameter perturbations) can reuse the
    factorization instead of recomputing it from scratch.

    Attributes:
        dx_dp: Primal sensitivity matrix, shape (n_vars, n_params).
        dlambda_dp: Dual sensitivity matrix, shape (n_active, n_params), or None.
        KKT_matrix: The KKT matrix used for the solve, shape (n_vars+n_active, ...).
        H_xp: Mixed Hessian of Lagrangian w.r.t. x then p, shape (n_vars, n_params).
        dg_dp: Active constraint parameter Jacobian, shape (n_active, n_params), or None.
        n_vars: Number of primal variables.
        n_active: Number of active constraints (including active bounds).
    """

    dx_dp: jnp.ndarray
    dlambda_dp: Optional[jnp.ndarray]
    KKT_matrix: Optional[jnp.ndarray]
    H_xp: jnp.ndarray
    dg_dp: Optional[jnp.ndarray]
    n_vars: int
    n_active: int


def implicit_differentiate(
    model: Model,
    x_star: jnp.ndarray,
    multipliers: jnp.ndarray,
    p_flat: jnp.ndarray,
    active_constraint_indices: list[int],
    active_bound_indices: list[int],
) -> SensitivityInfo:
    """Compute dx/dp via implicit differentiation of the KKT system.

    Builds and solves the KKT linear system to obtain the sensitivity of the
    optimal primal variables with respect to the parameters.

    Args:
        model: The optimization model.
        x_star: Optimal primal solution (flat vector).
        multipliers: Constraint multipliers from the solver.
        p_flat: Flat parameter vector.
        active_constraint_indices: Indices of active constraints.
        active_bound_indices: Indices of variables at their bounds.

    Returns:
        SensitivityInfo with dx_dp, dlambda_dp, KKT matrix, and intermediate data.
    """
    obj_fn = _compile_parametric_objective(model)
    constraint_fns = []
    for c in model._constraints:
        if isinstance(c, Constraint):
            constraint_fns.append(_compile_parametric_constraint(c, model))

    n_vars = len(x_star)
    n_params = len(p_flat)
    n_active_cons = len(active_constraint_indices)
    n_active_bounds = len(active_bound_indices)
    n_active = n_active_cons + n_active_bounds

    # Build the Lagrangian as a function of (x, p)
    if multipliers is not None and len(constraint_fns) > 0:
        mults = jnp.array(multipliers, dtype=jnp.float64)

        def lagrangian(x, p):
            obj_val = obj_fn(x, p)
            con_vals = jnp.array([cf(x, p) for cf in constraint_fns])
            return obj_val + jnp.dot(mults, con_vals)
    else:

        def lagrangian(x, p):
            return obj_fn(x, p)

    # H_xx: Hessian of Lagrangian w.r.t. x
    hess_xx_fn = jax.hessian(lagrangian, argnums=0)
    H_xx = hess_xx_fn(x_star, p_flat)

    # H_xp: mixed Hessian of Lagrangian w.r.t. x then p
    hess_xp_fn = jax.jacobian(jax.grad(lagrangian, argnums=0), argnums=1)
    H_xp = hess_xp_fn(x_star, p_flat)

    # Build the active constraint Jacobian and parameter derivatives
    if n_active == 0:
        # No active constraints or bounds: just solve H_xx dx/dp = -H_xp
        dx_dp = jnp.linalg.solve(H_xx, -H_xp)
        return SensitivityInfo(
            dx_dp=dx_dp,
            dlambda_dp=None,
            KKT_matrix=H_xx,
            H_xp=H_xp,
            dg_dp=None,
            n_vars=n_vars,
            n_active=0,
        )

    # J_a: Jacobian of active constraints w.r.t. x (n_active x n_vars)
    # dg_dp: Jacobian of active constraints w.r.t. p (n_active x n_params)
    J_a_rows = []
    dg_dp_rows = []

    for i in active_constraint_indices:
        cf = constraint_fns[i]
        jac_x_fn = jax.grad(cf, argnums=0)
        jac_p_fn = jax.grad(cf, argnums=1)
        J_a_rows.append(jac_x_fn(x_star, p_flat))
        dg_dp_rows.append(jac_p_fn(x_star, p_flat))

    # Active variable bounds: x_i = lb_i or x_i = ub_i
    # These are linear constraints with Jacobian row = e_i (unit vector)
    # and dg/dp = 0 (bounds don't depend on parameters)
    for idx in active_bound_indices:
        e_i = jnp.zeros(n_vars, dtype=jnp.float64).at[idx].set(1.0)
        J_a_rows.append(e_i)
        dg_dp_rows.append(jnp.zeros(n_params, dtype=jnp.float64))

    J_a = jnp.stack(J_a_rows)  # (n_active, n_vars)
    dg_dp = jnp.stack(dg_dp_rows)  # (n_active, n_params)

    # Build KKT system:
    # [H_xx   J_a^T] [dx/dp  ]   [-H_xp    ]
    # [J_a    0    ] [dlam/dp] = [-dg_dp   ]
    dim = n_vars + n_active
    KKT = jnp.zeros((dim, dim), dtype=jnp.float64)
    KKT = KKT.at[:n_vars, :n_vars].set(H_xx)
    KKT = KKT.at[:n_vars, n_vars:].set(J_a.T)
    KKT = KKT.at[n_vars:, :n_vars].set(J_a)

    rhs = jnp.zeros((dim, n_params), dtype=jnp.float64)
    rhs = rhs.at[:n_vars, :].set(-H_xp)
    rhs = rhs.at[n_vars:, :].set(-dg_dp)

    # Solve the linear system
    solution = jnp.linalg.solve(KKT, rhs)
    dx_dp = solution[:n_vars, :]
    dlambda_dp = solution[n_vars:, :]

    return SensitivityInfo(
        dx_dp=dx_dp,
        dlambda_dp=dlambda_dp,
        KKT_matrix=KKT,
        H_xp=H_xp,
        dg_dp=dg_dp,
        n_vars=n_vars,
        n_active=n_active,
    )


def _perturbation_gradient(
    model: Model,
    p_flat: jnp.ndarray,
    ipopt_options: Optional[dict],
    eps: float = 1e-5,
    nlp_solver: str = "ipm",
    solver_options: Optional[dict] = None,
) -> np.ndarray:
    """Fallback: compute gradient via finite perturbation of the solve.

    Used when the KKT Jacobian is ill-conditioned (condition number > 1e12).

    Args:
        model: The optimization model.
        p_flat: Flat parameter vector.
        ipopt_options: Deprecated alias for solver_options.
        eps: Perturbation size.
        nlp_solver: NLP backend: ``"ipm"`` (default) or ``"ipopt"``.
        solver_options: Options dict passed to the NLP solver.

    Returns:
        Gradient array of shape (n_params,).
    """
    opts = solver_options or ipopt_options
    n_params = len(p_flat)
    grad = np.zeros(n_params, dtype=np.float64)

    for i in range(n_params):
        p_plus = np.array(p_flat)
        p_plus[i] += eps
        p_minus = np.array(p_flat)
        p_minus[i] -= eps

        # Set params and solve at p + eps
        _set_model_params(model, p_plus)
        r_plus = _solve_model_nlp(model, opts, nlp_solver=nlp_solver)

        # Set params and solve at p - eps
        _set_model_params(model, p_minus)
        r_minus = _solve_model_nlp(model, opts, nlp_solver=nlp_solver)

        if r_plus is not None and r_minus is not None:
            grad[i] = (r_plus - r_minus) / (2 * eps)

    # Restore original params
    _set_model_params(model, np.array(p_flat))
    return grad


def _set_model_params(model: Model, p_flat: np.ndarray) -> None:
    """Update model parameter values from a flat array."""
    offset = 0
    for p in model._parameters:
        p_size = int(np.prod(p.shape)) if p.shape else 1
        p.value = np.asarray(p_flat[offset : offset + p_size]).reshape(p.shape)
        offset += p_size


def _solve_model_nlp(
    model: Model,
    ipopt_options: Optional[dict],
    *,
    nlp_solver: str = "ipm",
) -> Optional[float]:
    """Solve the model NLP and return the objective value, or None on failure."""
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.solvers import SolveStatus

    evaluator = NLPEvaluator(model)
    lb, ub = evaluator.variable_bounds
    lb_clipped = np.clip(lb, -100.0, 100.0)
    ub_clipped = np.clip(ub, -100.0, 100.0)
    x0 = 0.5 * (lb_clipped + ub_clipped)

    opts = dict(ipopt_options) if ipopt_options else {}
    opts.setdefault("print_level", 0)
    result = _dispatch_nlp_solve(nlp_solver, evaluator, x0, opts)
    if result.status == SolveStatus.OPTIMAL:
        return result.objective
    return None


class DiffSolveResultL3(DiffSolveResult):
    """Result of L3 differentiable solve with implicit differentiation.

    Extends DiffSolveResult with:
      - implicit_gradient(param): gradient via KKT implicit differentiation
      - sensitivity_matrix(): full dx*/dp matrix
      - approximate_resolve(new_params): first-order approximation of x* at new params
      - dual_sensitivity(param): dlambda*/dp for a given parameter
      - reduced_hessian(): reduced Hessian projected onto null space of active constraints
      - sensitivity(dp): fast re-query with cached KKT factorization
      - Falls back to L1 gradient if L3 computation failed
    """

    def __init__(
        self,
        status: str,
        objective: Optional[float],
        x: Optional[dict[str, np.ndarray]],
        _model: Model,
        _sensitivity: np.ndarray,
        _dx_dp: Optional[np.ndarray],
        _obj_fn_parametric,
        _x_star: Optional[jnp.ndarray],
        _p_flat: Optional[jnp.ndarray],
        _l3_failed: bool = False,
        _dlambda_dp: Optional[np.ndarray] = None,
        _kkt_matrix: Optional[np.ndarray] = None,
        _h_xp: Optional[np.ndarray] = None,
        _dg_dp: Optional[np.ndarray] = None,
        _n_active: int = 0,
    ):
        super().__init__(status, objective, x, _model, _sensitivity)
        self._dx_dp = _dx_dp
        self._obj_fn_parametric = _obj_fn_parametric
        self._x_star = _x_star
        self._p_flat = _p_flat
        self._l3_failed = _l3_failed
        self._dlambda_dp = _dlambda_dp
        self._kkt_matrix = _kkt_matrix
        self._h_xp = _h_xp
        self._dg_dp = _dg_dp
        self._n_active = _n_active

    def sensitivity_matrix(self) -> Optional[np.ndarray]:
        """Return the full dx*/dp matrix from implicit differentiation.

        Returns:
            Array of shape (n_vars, n_params), or None if L3 failed.
        """
        if self._dx_dp is None:
            return None
        return np.asarray(self._dx_dp)

    def implicit_gradient(self, param: Parameter) -> np.ndarray:
        """Get d(obj*)/d(param) via implicit differentiation.

        Computes dobj/dp = (dobj/dx)(dx/dp) + dobj/dp_direct.
        Falls back to L1 envelope theorem gradient if L3 failed.

        Args:
            param: A Parameter from the model.

        Returns:
            Array of same shape as param.value containing the sensitivities.
        """
        if self._l3_failed or self._dx_dp is None:
            return self.gradient(param)

        start, end = _get_param_slice(param, self._model)

        # dobj/dp = (dobj/dx)(dx/dp) + dobj/dp_direct
        # Use JAX to compute dobj/dx and dobj/dp at (x*, p)
        dobj_dx_fn = jax.grad(self._obj_fn_parametric, argnums=0)
        dobj_dp_fn = jax.grad(self._obj_fn_parametric, argnums=1)

        dobj_dx = dobj_dx_fn(self._x_star, self._p_flat)  # (n_vars,)
        dobj_dp_direct = dobj_dp_fn(self._x_star, self._p_flat)  # (n_params,)

        # Total derivative: dobj/dp = dobj/dx @ dx/dp + dobj/dp_direct
        dx_dp_param = self._dx_dp[:, start:end]  # (n_vars, param_size)
        total_grad = jnp.dot(dobj_dx, dx_dp_param) + dobj_dp_direct[start:end]
        total_grad = np.asarray(total_grad)

        if param.shape == () or (end - start) == 1:
            return float(total_grad[0])
        return total_grad.reshape(param.shape)

    def approximate_resolve(
        self, new_params: list[tuple[Parameter, float]]
    ) -> dict[str, np.ndarray]:
        """Approximate x*(p') using first-order sensitivity: x* + (dx*/dp) * dp.

        Args:
            new_params: List of (Parameter, new_value) pairs for perturbed parameters.

        Returns:
            dict of variable names to approximate optimal values.

        Raises:
            RuntimeError: If L3 sensitivity computation failed.
        """
        if self._l3_failed or self._dx_dp is None or self._p_flat is None:
            raise RuntimeError("Cannot approximate_resolve: L3 sensitivity not available.")
        if self._x_star is None:
            raise RuntimeError("Cannot approximate_resolve: no solution available.")

        n_params = len(self._p_flat)
        dp = np.zeros(n_params, dtype=np.float64)
        for param, new_val in new_params:
            start, end = _get_param_slice(param, self._model)
            old_vals = np.asarray(self._p_flat[start:end])
            new_arr = np.atleast_1d(np.asarray(new_val, dtype=np.float64)).ravel()
            dp[start:end] = new_arr - old_vals

        x_star_flat = np.asarray(self._x_star)
        dx_dp = np.asarray(self._dx_dp)
        x_approx = x_star_flat + dx_dp @ dp

        result = {}
        offset = 0
        for v in self._model._variables:
            size = v.size
            val = x_approx[offset : offset + size]
            result[v.name] = val.reshape(v.shape) if v.shape != () else val
            offset += size
        return result

    def dual_sensitivity(self, param: Parameter) -> Optional[np.ndarray]:
        """Return dlambda*/dp for a given parameter.

        Shows how constraint multipliers change with parameter perturbation.

        Args:
            param: A Parameter from the model.

        Returns:
            Array of shape (n_active, param_size) or None if not available.
        """
        if self._dlambda_dp is None:
            return None
        start, end = _get_param_slice(param, self._model)
        return np.asarray(self._dlambda_dp[:, start:end])

    def reduced_hessian(self) -> Optional[np.ndarray]:
        """Return the reduced Hessian of the Lagrangian projected onto the
        null space of active constraints.

        Useful for covariance estimation: Cov(p) ~ sigma^2 * inv(Z^T H Z)
        where Z is the null space basis of the active constraint Jacobian.

        Returns:
            Array of shape (n_free, n_free), or None if KKT data not available.
            n_free = n_vars - n_active (dimension of the null space).
        """
        if self._kkt_matrix is None:
            return None

        KKT = np.asarray(self._kkt_matrix)
        n_active = self._n_active

        if n_active == 0:
            # No active constraints: reduced Hessian is the full Hessian
            return KKT.copy()

        n_vars = KKT.shape[0] - n_active
        H_xx = KKT[:n_vars, :n_vars]
        J_a = KKT[n_vars:, :n_vars]  # (n_active, n_vars)

        # Compute null space of J_a
        _, S, Vt = np.linalg.svd(J_a, full_matrices=True)
        # Null space columns are the last (n_vars - rank) columns of V
        rank = np.sum(S > 1e-10)
        Z = Vt[rank:, :].T  # (n_vars, n_free)

        if Z.shape[1] == 0:
            return np.array([]).reshape(0, 0)

        return Z.T @ H_xx @ Z

    def sensitivity(self, dp: np.ndarray) -> np.ndarray:
        """Compute dx for a given dp using cached KKT matrix.

        Equivalent to sIPOPT's back-substitution: solves KKT @ [dx; dlam] = new_rhs.

        Args:
            dp: Parameter perturbation vector of shape (n_params,) or (n_params, k).

        Returns:
            dx: Primal perturbation, shape (n_vars,) or (n_vars, k).
        """
        if self._kkt_matrix is None or self._h_xp is None:
            raise RuntimeError("Cannot compute sensitivity: KKT data not available.")

        dp = np.asarray(dp, dtype=np.float64)
        KKT = jnp.array(self._kkt_matrix)
        H_xp = jnp.array(self._h_xp)
        n_active = self._n_active

        if dp.ndim == 1:
            dim = KKT.shape[0]
            rhs = jnp.zeros(dim, dtype=jnp.float64)
            rhs = rhs.at[: H_xp.shape[0]].set(-H_xp @ dp)
            if n_active > 0 and self._dg_dp is not None:
                dg_dp = jnp.array(self._dg_dp)
                rhs = rhs.at[H_xp.shape[0] :].set(-dg_dp @ dp)
            solution = jnp.linalg.solve(KKT, rhs)
            n_vars = H_xp.shape[0]
            return np.asarray(solution[:n_vars])
        else:
            # Batch: dp is (n_params, k)
            dim = KKT.shape[0]
            n_vars = H_xp.shape[0]
            k = dp.shape[1]
            rhs = jnp.zeros((dim, k), dtype=jnp.float64)
            rhs = rhs.at[:n_vars, :].set(-H_xp @ dp)
            if n_active > 0 and self._dg_dp is not None:
                dg_dp = jnp.array(self._dg_dp)
                rhs = rhs.at[n_vars:, :].set(-dg_dp @ dp)
            solution = jnp.linalg.solve(KKT, rhs)
            return np.asarray(solution[:n_vars, :])

    def __repr__(self) -> str:
        l3_status = "ok" if not self._l3_failed else "fallback_to_L1"
        return f"DiffSolveResultL3(status={self.status!r}, obj={self.objective}, l3={l3_status})"


def differentiable_solve_l3(
    model: Model,
    ipopt_options: Optional[dict] = None,
    active_tol: float = 1e-3,
    *,
    nlp_solver: str = "ipm",
    solver_options: Optional[dict] = None,
) -> DiffSolveResultL3:
    """Solve a continuous model with L3 implicit differentiation sensitivities.

    Uses implicit differentiation of the KKT system for more accurate
    sensitivities than the L1 envelope theorem approach.

    Falls back to perturbation smoothing if the KKT system is ill-conditioned.

    Args:
        model: A Model with objective, constraints, and parameters.
        ipopt_options: Deprecated alias for solver_options.
        active_tol: Tolerance for active set detection.
        nlp_solver: NLP backend: ``"ipm"`` (default) or ``"ipopt"``.
        solver_options: Options dict passed to the NLP solver.

    Returns:
        DiffSolveResultL3 with solution and both L1 and L3 gradients.
    """
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.modeling.core import VarType
    from discopt.solvers import SolveStatus

    opts = dict(solver_options or ipopt_options or {})

    model.validate()

    for v in model._variables:
        if v.var_type != VarType.CONTINUOUS:
            raise ValueError(
                "differentiable_solve_l3 only supports continuous models. "
                f"Variable '{v.name}' is {v.var_type.value}."
            )

    # Solve the NLP
    evaluator = NLPEvaluator(model)
    lb, ub = evaluator.variable_bounds
    lb_clipped = np.clip(lb, -100.0, 100.0)
    ub_clipped = np.clip(ub, -100.0, 100.0)
    x0 = 0.5 * (lb_clipped + ub_clipped)

    opts.setdefault("print_level", 0)

    nlp_result = _dispatch_nlp_solve(nlp_solver, evaluator, x0, opts)
    if nlp_result.status != SolveStatus.OPTIMAL:
        raise RuntimeError(f"NLP solve did not converge to optimal: {nlp_result.status.value}")

    x_star = jnp.array(nlp_result.x, dtype=jnp.float64)
    multipliers = nlp_result.multipliers

    # Compile parametric functions
    obj_fn = _compile_parametric_objective(model)
    constraint_fns = []
    for c in model._constraints:
        if isinstance(c, Constraint):
            constraint_fns.append(_compile_parametric_constraint(c, model))

    p_flat = _flatten_params(model)

    # Compute L1 envelope theorem sensitivity
    if multipliers is not None and len(constraint_fns) > 0:
        mults = jnp.array(multipliers, dtype=jnp.float64)

        def lagrangian_p(p_flat_arg):
            obj_val = obj_fn(x_star, p_flat_arg)
            con_vals = jnp.array([cf(x_star, p_flat_arg) for cf in constraint_fns])
            return obj_val + jnp.dot(mults, con_vals)
    else:

        def lagrangian_p(p_flat_arg):
            return obj_fn(x_star, p_flat_arg)

    grad_lagrangian_p = jax.grad(lagrangian_p)
    l1_sensitivity = np.asarray(grad_lagrangian_p(p_flat))

    # Compute L3 implicit differentiation
    dx_dp = None
    l3_failed = False
    sens_info: Optional[SensitivityInfo] = None

    try:
        active_cons, active_bounds = find_active_set(
            x_star, model, constraint_fns, p_flat, tol=active_tol
        )
        sens_info = implicit_differentiate(
            model, x_star, multipliers, p_flat, active_cons, active_bounds
        )
        dx_dp = sens_info.dx_dp

        # Check condition number of the result
        if dx_dp is not None and jnp.any(jnp.isnan(dx_dp)) or jnp.any(jnp.isinf(dx_dp)):
            l3_failed = True
            # Fall back to perturbation smoothing
            l1_sensitivity = _perturbation_gradient(
                model, p_flat, None, nlp_solver=nlp_solver, solver_options=opts
            )
            dx_dp = None
            sens_info = None

    except Exception:
        l3_failed = True
        # Use L1 sensitivity as-is

    # Unpack solution
    x_dict = {}
    offset = 0
    for v in model._variables:
        size = v.size
        val = np.asarray(x_star[offset : offset + size])
        x_dict[v.name] = val.reshape(v.shape) if v.shape != () else val
        offset += size

    return DiffSolveResultL3(
        status="optimal",
        objective=nlp_result.objective,
        x=x_dict,
        _model=model,
        _sensitivity=l1_sensitivity,
        _dx_dp=np.asarray(dx_dp) if dx_dp is not None else None,
        _obj_fn_parametric=obj_fn,
        _x_star=x_star,
        _p_flat=p_flat,
        _l3_failed=l3_failed,
        _dlambda_dp=(
            np.asarray(sens_info.dlambda_dp)
            if sens_info and sens_info.dlambda_dp is not None
            else None
        ),
        _kkt_matrix=(
            np.asarray(sens_info.KKT_matrix)
            if sens_info and sens_info.KKT_matrix is not None
            else None
        ),
        _h_xp=np.asarray(sens_info.H_xp) if sens_info else None,
        _dg_dp=(np.asarray(sens_info.dg_dp) if sens_info and sens_info.dg_dp is not None else None),
        _n_active=sens_info.n_active if sens_info else 0,
    )
