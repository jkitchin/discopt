"""Implicit-function expression node (issue #379) -- prototype.

A vector ``v`` defined as the solution of a square nonlinear system
``g(u, v) = 0`` is compiled to a differentiable JAX inner solve: the forward is a
Newton iteration; the reverse is the implicit-function-theorem (IFT) VJP

    dv/du = -(dg/dv)^{-1} (dg/du),

so we never differentiate through the Newton iterations.

This rides entirely on :class:`~discopt.modeling.core.CustomCall`: the returned
node is opaque and AD-only, so a model containing an implicit node is solved on
the **local NLP path only** (no global optimality certificate) and the solver
raises if integer/binary variables are present -- exactly the CustomCall
contract.  It is the core-side primitive that lets reduced-space / variable
aggregation eliminate an irreducible *cyclic* block ``g_B(u, v_B) = 0`` instead
of leaving its variables in the reduced model.

Prototype for #379 -- API and scope may still change.
"""

from __future__ import annotations

from typing import Callable, Sequence

from discopt.modeling.core import Expression, custom


def _implicit_solver(
    residual: Callable,
    x0,
    newton_iters: int = 30,
) -> Callable:
    """Build the JAX callable ``phi(u) -> v`` solving ``residual(u, v) = 0``.

    Forward: fixed-iteration Newton from ``x0``.  Derivatives w.r.t. ``u`` come
    from :func:`jax.lax.custom_root`, which differentiates the root through the
    implicit-function theorem (one linear solve with the block Jacobian) *and*
    supports higher-order AD -- so the NLP solver's Hessian (forward-over-reverse
    through this node) works, which a hand-rolled ``custom_vjp`` (reverse only)
    would break.  Exposed separately from :func:`implicit` so the numerics are
    unit-testable without building a full model.
    """
    import jax
    import jax.numpy as jnp

    x0_arr = jnp.asarray(x0, dtype=float)

    def phi(u):
        def f(v):  # root sought: residual(u, v) == 0, with u closed over
            return jnp.asarray(residual(u, v), dtype=float)

        def solve(f, y0):
            def body(_, v):
                return v - jnp.linalg.solve(jax.jacobian(f)(v), f(v))

            return jax.lax.fori_loop(0, newton_iters, body, y0)

        def tangent_solve(g, y):  # g is the (linear) JVP of f; solve J z = y
            jac = jax.jacobian(g)(jnp.zeros_like(y))
            return jnp.linalg.solve(jac, y)

        return jax.lax.custom_root(f, x0_arr, solve, tangent_solve)

    return phi


def implicit(
    residual: Callable,
    u_inputs: Sequence,
    n_unknowns: int,
    x0=None,
    *,
    newton_iters: int = 30,
    name: str = "implicit",
) -> Expression:
    """Define ``v`` (length ``n_unknowns``) implicitly by ``residual(u, v) = 0``.

    Parameters
    ----------
    residual : callable
        ``residual(u, v) -> array`` of length ``n_unknowns``, written with
        ``jax.numpy`` so it is JAX-traceable.  ``u`` is a 1-D array of the
        evaluated ``u_inputs``; ``v`` is the unknown vector.
    u_inputs : sequence of Expression
        The model expressions the block depends on (the ``u``).  Their scalar
        values are stacked into the ``u`` array passed to ``residual``.
    n_unknowns : int
        Number of components of ``v``.
    x0 : array-like, optional
        Initial guess for the Newton solve (default zeros of length
        ``n_unknowns``).
    newton_iters : int
        Fixed Newton iterations for the forward solve.
    name : str
        Display name used in reprs/errors.

    Returns
    -------
    Expression
        A :class:`CustomCall` node evaluating to the solved ``v`` vector; index
        it (``node[i]``) for components.  Local-NLP-only, no global certificate.
    """
    import jax.numpy as jnp

    if x0 is None:
        x0 = jnp.zeros(n_unknowns, dtype=float)
    phi = _implicit_solver(residual, x0, newton_iters=newton_iters)

    def solve_fn(*u_vals):
        u = jnp.stack([jnp.asarray(x, dtype=float) for x in u_vals]) if u_vals else jnp.zeros(0)
        return phi(u)

    return custom(solve_fn, name=name)(*u_inputs)
