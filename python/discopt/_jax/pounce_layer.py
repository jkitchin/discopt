"""POUNCE-backed differentiable JAX layers.

Wraps the pure-Rust POUNCE solver as a differentiable JAX function: the forward
solve runs on the host via :func:`jax.pure_callback`, and the gradient is given
by a :func:`jax.custom_vjp` rule that implements the **implicit function theorem
at the KKT point** — exactly the sIPOPT sensitivity system of
:func:`discopt.solvers.sipopt.pounce_sensitivity`, transposed for reverse mode.

The point is composability, not acceleration: the solve still runs on the host
(a `pure_callback` is not an XLA kernel and forces a device round-trip under
``jit``), but the result is a first-order-differentiable JAX value that can sit
inside ``jax.grad``/``jax.jit``/``jax.vmap`` pipelines. POUNCE is an interior
point method, so it returns the analytic center of the optimal face — the
complementarity slacks stay positive and the KKT system is nonsingular, which is
what makes the sensitivity well posed (a simplex vertex would be degenerate).

Differentiation is **w.r.t. model parameter values** (``dm.Parameter``), matching
``pounce_sensitivity``. Use :func:`make_nlp_layer` to build a layer for a given
model + parameter list; it returns ``solve(p) -> (obj, x, lam)`` where ``p`` is
the JAX vector of parameter values.

LP and QP differentiable layers already exist in
:mod:`discopt._jax.differentiable_lp` / :mod:`discopt._jax.differentiable_qp`
(``custom_jvp`` with the KKT sensitivity); they are re-exported here so callers
have one import surface.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from discopt._jax.differentiable_lp import lp_solve, lp_solve_grad  # noqa: F401 (re-export)
from discopt._jax.differentiable_qp import qp_solve  # noqa: F401 (re-export)
from discopt._jax.nlp_evaluator import NLPEvaluator

_F64 = jnp.float64
_FD_EPS = 1e-6


def make_nlp_layer(
    model,
    parameters: list,
    options: Optional[dict] = None,
) -> Callable:
    """Build a differentiable NLP layer for ``model`` over ``parameters``.

    Returns a function ``solve(p) -> (obj, x, lam)`` differentiable w.r.t. the
    JAX vector ``p`` of parameter values (one entry per parameter, in order).
    The forward solve uses POUNCE; the VJP uses the KKT-adjoint sensitivity.

    Parameters
    ----------
    model : dm.Model
        The model whose objective/constraints are already set.
    parameters : list of dm.Parameter
        Scalar parameters to differentiate with respect to.
    options : dict, optional
        POUNCE solver options.
    """
    from discopt.solvers.nlp_pounce import solve_nlp

    params = list(parameters)
    n_p = len(params)
    opts = dict(options or {})
    opts.setdefault("print_level", 0)

    # Static shapes from the model (the parameter values do not change them).
    ev0 = NLPEvaluator(model)
    n = ev0.n_variables
    m = ev0.n_constraints

    def _set_params(p_np: np.ndarray) -> None:
        for k, prm in enumerate(params):
            prm.value = np.float64(p_np[k])

    def _host_solve(p_np):
        _set_params(np.asarray(p_np, dtype=np.float64))
        ev = NLPEvaluator(model)
        lb, ub = ev.variable_bounds
        x0 = 0.5 * (np.clip(lb, -1e2, 1e2) + np.clip(ub, -1e2, 1e2))
        res = solve_nlp(ev, x0, options=opts)
        x = np.asarray(res.x, dtype=np.float64) if res.x is not None else np.zeros(n)
        lam = (
            np.asarray(res.multipliers, dtype=np.float64)
            if res.multipliers is not None
            else np.zeros(m)
        )
        if lam.size != m:
            lam = np.zeros(m)
        obj = float(res.objective) if res.objective is not None else 0.0
        return (np.float64(obj), x, lam)

    def _host_vjp(p_np, x_np, lam_np, g_obj, g_x, g_lam):
        p_np = np.asarray(p_np, dtype=np.float64)
        x_np = np.asarray(x_np, dtype=np.float64)
        lam_np = np.asarray(lam_np, dtype=np.float64)
        g_x = np.asarray(g_x, dtype=np.float64)
        g_lam = np.asarray(g_lam, dtype=np.float64)
        g_obj = float(g_obj)
        _set_params(p_np)
        ev = NLPEvaluator(model)

        # KKT at the solution (same assembly as sipopt.pounce_sensitivity).
        W = ev.evaluate_lagrangian_hessian(x_np, 1.0, lam_np)
        if m > 0:
            J = ev.evaluate_jacobian(x_np)
            kkt = np.block([[W, J.T], [J, np.zeros((m, m))]])
        else:
            kkt = np.asarray(W, dtype=np.float64).copy()
        kkt = kkt + 1e-10 * np.eye(n + m)

        # Adjoint solve: KKT u = [g_x + g_obj*grad f ; g_lam].
        grad_f = ev.evaluate_gradient(x_np)
        rhs_x = g_x + g_obj * grad_f
        rhs = np.concatenate([rhs_x, g_lam]) if m > 0 else rhs_x
        u = np.linalg.solve(kkt, rhs)

        # Parameter cotangent: p̄_k = -M[:,k]·u + g_obj * ∂f/∂p_k, with
        # M[:,k] = [∂(∇ₓL)/∂p_k ; ∂g/∂p_k] via central differences (matching
        # sipopt's RHS construction, transposed onto the adjoint u).
        pbar = np.zeros(n_p, dtype=np.float64)
        for k, prm in enumerate(params):
            orig = float(p_np[k])
            prm.value = np.float64(orig + _FD_EPS)
            evp = NLPEvaluator(model)
            lag_p = evp.evaluate_gradient(x_np)
            f_p = float(evp.evaluate_objective(x_np))
            if m > 0:
                lag_p = lag_p + evp.evaluate_jacobian(x_np).T @ lam_np
                cons_p = evp.evaluate_constraints(x_np)
            prm.value = np.float64(orig - _FD_EPS)
            evm = NLPEvaluator(model)
            lag_m = evm.evaluate_gradient(x_np)
            f_m = float(evm.evaluate_objective(x_np))
            if m > 0:
                lag_m = lag_m + evm.evaluate_jacobian(x_np).T @ lam_np
                cons_m = evm.evaluate_constraints(x_np)
            prm.value = np.float64(orig)

            d_lag = (lag_p - lag_m) / (2.0 * _FD_EPS)
            mk = np.concatenate([d_lag, (cons_p - cons_m) / (2.0 * _FD_EPS)]) if m > 0 else d_lag
            df_dp = (f_p - f_m) / (2.0 * _FD_EPS)
            pbar[k] = -float(mk @ u) + g_obj * df_dp
        return pbar

    out_shapes = (
        jax.ShapeDtypeStruct((), _F64),
        jax.ShapeDtypeStruct((n,), _F64),
        jax.ShapeDtypeStruct((m,), _F64),
    )

    @jax.custom_vjp
    def solve(p):
        return jax.pure_callback(_host_solve, out_shapes, p, vmap_method="sequential")

    def solve_fwd(p):
        out = solve(p)
        return out, (p, out[1], out[2])

    def solve_bwd(res, g):
        p, x, lam = res
        g_obj, g_x, g_lam = g
        pbar = jax.pure_callback(
            _host_vjp,
            jax.ShapeDtypeStruct((n_p,), _F64),
            p,
            x,
            lam,
            g_obj,
            g_x,
            g_lam,
            vmap_method="sequential",
        )
        return (pbar,)

    solve.defvjp(solve_fwd, solve_bwd)
    return solve


def make_milp_objective_layer(
    model,
    parameters: list,
    solver_options: Optional[dict] = None,
) -> Callable:
    """Build a differentiable MILP/MIQP objective layer over ``parameters``.

    Returns ``solve(p) -> obj`` differentiable w.r.t. the JAX vector ``p`` of
    parameter values. The forward solve runs the integer branch-and-bound; the
    VJP is the **fix-and-differentiate** sensitivity — the integers are fixed at
    the optimum and the envelope theorem is applied to the continuous restriction
    (see :func:`discopt._jax.differentiable._differentiable_solve_integer`).

    This is the JAX-composable form of ``differentiable_solve(...).gradient(...)``
    for integer models: it slots into ``jax.grad``/``jax.jit``/``jax.vmap``
    pipelines, which is what decision-focused learning needs. The gradient is
    exact wherever the optimal integer assignment is locally stable in ``p``; at
    breakpoints it is the incumbent assignment's one-sided value.

    Scope: objective sensitivity ``d(obj*)/dp`` (the headline use case).
    Solution sensitivity ``dx*/dp`` through the integers is a follow-on.

    Parameters
    ----------
    model : dm.Model
        A model with integer/binary variables, objective, and constraints set.
    parameters : list of dm.Parameter
        Scalar parameters to differentiate with respect to.
    solver_options : dict, optional
        Options forwarded to the continuous sensitivity solve.
    """
    from discopt._jax.differentiable import differentiable_solve

    params = list(parameters)
    n_p = len(params)

    def _set_params(p_np: np.ndarray) -> None:
        for k, prm in enumerate(params):
            prm.value = np.float64(p_np[k])

    def _host_solve(p_np):
        _set_params(np.asarray(p_np, dtype=np.float64))
        res = differentiable_solve(model, solver_options=solver_options)
        obj = float(res.objective) if res.objective is not None else 0.0
        # Per-parameter envelope sensitivity (gradient() applies the correct
        # parameter offset, so this is robust to the layer's parameter ordering).
        sens = np.array([float(res.gradient(prm)) for prm in params], dtype=np.float64)
        return (np.float64(obj), sens)

    @jax.custom_vjp
    def solve(p):
        obj, _ = jax.pure_callback(
            _host_solve,
            (jax.ShapeDtypeStruct((), _F64), jax.ShapeDtypeStruct((n_p,), _F64)),
            p,
            vmap_method="sequential",
        )
        return obj

    def solve_fwd(p):
        obj, sens = jax.pure_callback(
            _host_solve,
            (jax.ShapeDtypeStruct((), _F64), jax.ShapeDtypeStruct((n_p,), _F64)),
            p,
            vmap_method="sequential",
        )
        return obj, sens

    def solve_bwd(sens, g):
        # d(loss)/dp = g * d(obj*)/dp.
        return (g * sens,)

    solve.defvjp(solve_fwd, solve_bwd)
    return solve
