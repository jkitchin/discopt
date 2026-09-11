"""POUNCE-backed differentiable JAX layers.

Wraps the pure-Rust POUNCE solver as a differentiable JAX function: the forward
solve runs on the host via :func:`jax.pure_callback`, and the derivatives come
from a :func:`jax.lax.custom_root` rule over the **KKT system at the solution**
— the sIPOPT sensitivity of :func:`discopt.solvers.sipopt.pounce_sensitivity`,
shared with :func:`discopt.modeling.argmin` through
``discopt.modeling.argmin._build_layer``.

The point is composability, not acceleration: the solve still runs on the host
(a `pure_callback` is not an XLA kernel and forces a device round-trip under
``jit``), but the result is a differentiable JAX value that can sit inside
``jax.grad``/``jax.jit``/``jax.vmap`` pipelines. Unlike the ``custom_vjp`` rule
this carried before (#1216), ``custom_root``'s rule is expressed in ordinary
differentiable operations, so forward mode and second derivatives work too.
POUNCE is an interior point method, so it returns the analytic center of the
optimal face — the complementarity slacks stay positive and the KKT system is
nonsingular, which is what makes the sensitivity well posed (a simplex vertex
would be degenerate).

Differentiation is **w.r.t. model parameter values** (``dm.Parameter``), matching
``pounce_sensitivity``. Use :func:`make_nlp_layer` to build a layer for a given
model + parameter list; it returns ``solve(p) -> (obj, x, lam)`` where ``p`` is
the JAX vector of parameter values.

LP and QP differentiable layers already exist in
:mod:`discopt._relax.differentiable_lp` / :mod:`discopt._relax.differentiable_qp`
(``custom_jvp`` with the KKT sensitivity); they are re-exported here so callers
have one import surface.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from discopt._relax.differentiable_lp import lp_solve, lp_solve_grad  # noqa: F401 (re-export)
from discopt._relax.differentiable_qp import qp_solve  # noqa: F401 (re-export)

_F64 = jnp.float64


def make_nlp_layer(
    model,
    parameters: list,
    options: Optional[dict] = None,
) -> Callable:
    """Build a differentiable NLP layer for ``model`` over ``parameters``.

    Returns a function ``solve(p) -> (obj, x, lam)`` differentiable w.r.t. the
    JAX vector ``p`` of parameter values (one entry per parameter, in order).
    The forward solve uses POUNCE; the derivatives come from the KKT sensitivity
    at the solution, over the **active set** POUNCE lands on.

    The rule is :func:`jax.lax.custom_root` (via
    :func:`discopt.modeling.argmin.argmin_layer`), not ``custom_vjp``, so the
    layer is differentiable in **both** modes and more than once:
    ``jax.hessian(lambda p: layer(p)[0])`` works, where the reverse-only rule
    this used to carry raised *"can't apply forward-mode autodiff (jvp) to a
    custom_vjp function"* (#1216).

    Parameters
    ----------
    model : dm.Model
        The model whose objective/constraints are already set.
    parameters : list of dm.Parameter
        Scalar parameters to differentiate with respect to.
    options : dict, optional
        POUNCE solver options.
    """
    from discopt.modeling.argmin import _build_layer

    params = list(parameters)
    n_p = len(params)
    phi = _build_layer(
        model,
        params,
        options=options,
        verify_minimizer=False,
        require_min=False,
        full=True,
    )
    n = phi.n_variables

    def solve(p):
        p = jnp.reshape(jnp.asarray(p, dtype=_F64), (n_p,))
        z = phi(p)
        x = z[:n]
        # The objective is evaluated at the differentiable x*(p), so d(obj)/dp
        # picks up both the direct and the solution-path dependence -- the
        # envelope theorem falls out rather than being assumed.
        obj = jnp.reshape(phi.objective_fn(x, phi.params_tuple(p)), ())
        return obj, x, z[n:]

    solve.inner_solve_count = phi.inner_solve_count  # type: ignore[attr-defined]
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
    (see :func:`discopt._relax.differentiable._differentiable_solve_integer`).

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
    from discopt._relax.differentiable import differentiable_solve

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
