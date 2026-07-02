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
    *,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> Callable:
    """Build the JAX callable ``phi(u) -> v`` solving ``residual(u, v) = 0``.

    Forward: convergence-based Newton from ``x0`` with two gates the issue
    requires -- a **nonsingular-Jacobian** gate (a non-finite Newton step means a
    singular/ill-conditioned block) and a **convergence** gate (residual must
    reach ``tol`` within ``max_iter``).  On either failure the solve returns
    ``NaN``: we cannot raise inside a JAX-traced solve, so the failure propagates
    into the objective/constraint value and the NLP solver reports it as a failed
    evaluation rather than silently returning a wrong root.

    Derivatives w.r.t. ``u`` come from :func:`jax.lax.custom_root`, which
    differentiates the root through the implicit-function theorem (one linear
    solve with the block Jacobian) *and* supports higher-order AD -- so the NLP
    solver's Hessian (forward-over-reverse through this node) works, which a
    hand-rolled ``custom_vjp`` (reverse only) would break.  Exposed separately
    from :func:`implicit` so the numerics are unit-testable without a full model.
    """
    import jax
    import jax.numpy as jnp

    x0_arr = jnp.asarray(x0, dtype=float)

    def phi(u):
        def f(v):  # root sought: residual(u, v) == 0, with u closed over
            return jnp.asarray(residual(u, v), dtype=float)

        def solve(f, y0):
            def cond(state):
                _, it, rnorm, ok = state
                return (rnorm > tol) & (it < max_iter) & ok

            def body(state):
                v, it, _, _ = state
                step = jnp.linalg.solve(jax.jacobian(f)(v), f(v))
                ok = jnp.all(jnp.isfinite(step))  # nonsingular-Jacobian gate
                v_new = v - step
                return (v_new, it + 1, jnp.linalg.norm(f(v_new)), ok)

            state0 = (y0, 0, jnp.linalg.norm(f(y0)), jnp.bool_(True))
            v, _, rnorm, ok = jax.lax.while_loop(cond, body, state0)
            converged = ok & jnp.isfinite(rnorm) & (rnorm <= tol)
            return jnp.where(converged, v, jnp.full_like(v, jnp.nan))

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
    tol: float = 1e-10,
    max_iter: int = 50,
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
    tol : float
        Residual tolerance for the forward Newton solve.
    max_iter : int
        Maximum Newton iterations; exceeding it without reaching ``tol`` is a
        non-convergence failure (propagated as ``NaN``).
    name : str
        Display name used in reprs/errors.

    Returns
    -------
    Expression
        A :class:`CustomCall` node evaluating to the solved ``v`` vector; index
        it (``node[i]``) for components.  Local-NLP-only, no global certificate.

    Raises
    ------
    ValueError
        If ``n_unknowns < 1``, ``x0`` has the wrong length, or ``residual``
        probed at a dummy point does not return exactly ``n_unknowns`` entries.
    """
    import jax.numpy as jnp

    if n_unknowns < 1:
        raise ValueError(f"n_unknowns must be >= 1, got {n_unknowns}")
    if x0 is None:
        x0 = jnp.zeros(n_unknowns, dtype=float)
    x0 = jnp.asarray(x0, dtype=float)
    if x0.shape != (n_unknowns,):
        raise ValueError(f"x0 must have shape ({n_unknowns},), got {tuple(x0.shape)}")

    # Best-effort build-time shape check: the residual must return n_unknowns
    # entries.  Probe at a dummy point; if the probe itself errors (residual not
    # defined there) defer to runtime rather than false-failing.
    try:
        probe = jnp.asarray(residual(jnp.zeros(len(u_inputs)), x0), dtype=float)
    except Exception:
        probe = None
    if probe is not None and probe.shape != (n_unknowns,):
        raise ValueError(
            f"residual must return {n_unknowns} entries, got shape {tuple(probe.shape)}"
        )

    phi = _implicit_solver(residual, x0, tol=tol, max_iter=max_iter)

    def solve_fn(*u_vals):
        # Flatten each input (scalar -> length 1, vector -> raveled) and
        # concatenate, so ``u`` is a flat 1-D array of all dependency values.
        if u_vals:
            u = jnp.concatenate(
                [jnp.atleast_1d(jnp.asarray(x, dtype=float)).ravel() for x in u_vals]
            )
        else:
            u = jnp.zeros(0)
        return phi(u)

    node: Expression = custom(solve_fn, name=name)(*u_inputs)
    return node
