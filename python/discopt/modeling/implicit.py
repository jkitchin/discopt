"""Implicit-function expression node (issue #379) -- prototype.

A vector ``v`` defined as the solution of a square nonlinear system
``g(u, v) = 0`` is compiled to a differentiable JAX inner solve: the forward is a
Newton iteration; the reverse is the implicit-function-theorem (IFT) VJP

    dv/du = -(dg/dv)^{-1} (dg/du),

so we never differentiate through the Newton iterations.

This rides entirely on :class:`~discopt.modeling.core.CustomCall`: the returned
node is opaque and AD-only, so a model containing an implicit node is solved on
the **local NLP path only** (no global optimality certificate: ``status``
``"feasible"``, no ``bound``/``gap``) and the solver raises if integer/binary
variables are present -- exactly the CustomCall contract.  It is the core-side
primitive that lets reduced-space / variable aggregation eliminate an irreducible
*cyclic* block ``g_B(u, v_B) = 0`` instead of leaving its variables in the
reduced model.

Prototype for #379 -- API and scope may still change.
"""

from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np

from discopt.modeling.core import Expression, Model, Variable, custom


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


def _flatten_u_inputs(u_inputs: Sequence) -> list:
    """Flatten ``u_inputs`` to a list of SCALAR expressions.

    The ``custom_root`` arm passes ``residual`` a flat ``u`` array built by
    ``concatenate([atleast_1d(x).ravel() for x in u_vals])`` (see
    :func:`implicit`'s ``solve_fn``).  The full-space arm must index ``u``
    identically or the same residual callable would mean two different things
    depending on the formulation, so this reproduces that flattening on the
    symbolic side: element order within an input, inputs in argument order.
    """
    flat: list = []
    for expr in u_inputs:
        shape = tuple(getattr(expr, "shape", ()) or ())
        size = int(np.prod(shape)) if shape else 1
        if not shape:
            flat.append(expr)
            continue
        if len(shape) != 1:
            raise ValueError(
                f"implicit(formulation='full_space') supports scalar and 1-D inputs; "
                f"got an input with shape {shape}"
            )
        flat.extend(expr[i] for i in range(size))
    return flat


def _unique_name(model: Model, base: str) -> str:
    """``base``, suffixed if the model already has a variable of that name.

    Variable names must be unique in a model, and a lowered block is often
    created in a loop (one per cyclic block), so a fixed default name would
    collide on the second block.
    """
    taken = {v.name for v in model._variables}
    if base not in taken:
        return base
    k = 2
    while f"{base}_{k}" in taken:
        k += 1
    return f"{base}_{k}"


def implicit_full_space(
    model: Model,
    residual: Callable,
    u_inputs: Sequence,
    n_unknowns: int,
    x0=None,
    *,
    bounds=None,
    name: str = "implicit",
) -> Variable:
    """Lower ``residual(u, v) = 0`` into the model as variables + equations.

    The alternative to :func:`implicit`'s opaque ``custom_root`` node: instead of
    hiding the block behind a ``CustomCall``, ``v`` becomes a real vector
    :class:`~discopt.modeling.core.Variable` and each residual becomes a real
    ``== 0`` constraint.  ``residual`` is therefore called with **discopt
    expressions**, not JAX arrays, and must be written in discopt's own operators
    (``+ - * / **``, ``dm.exp``, ``dm.log``, ``dm.sin``, ...) -- not ``jnp.*``.

    Why this exists (#379, #75).  The ``custom_root`` node has two costs that are
    both consequences of the equations being invisible to everything downstream:

    1. **It pins JAX to the solve path.** The tape refuses a ``CustomCall``
       outright (``_nl_expr_compiler``: "CustomCall (dm.custom) has no tape
       equivalent"), so any model containing one falls back to the JAX evaluator.
       Lowered, the block is ordinary algebra and the Rust tape handles it.
    2. **It forecloses a certificate.** A convex relaxation of an implicitly
       defined ``v`` needs the defining equations; with only ``v = phi(u)`` there
       is nothing to relax, so ``_custom_call_reduced_admissible`` refuses the
       global path and the solve is local-only (``gap_certified=False``).
       Measured on ``v**2 - u == 0, v <= 1.4, min -u`` over ``u in [1,2]``: the
       opaque node returns ``-1.96`` from ``x0=+1`` and ``-2.0`` from ``x0=-1``
       -- same box, same equations, different answer, because which root Newton
       lands in *is* the definition.  Lowered, both roots are in the box and the
       spatial B&B certifies the real optimum.

    What you give up is the reduced-space size win that motivated #379: ``v``
    stays in the model rather than being eliminated.  That is the trade, and it
    is the caller's to make -- hence a separate entry point rather than a
    silently different lowering of the same one.

    Parameters
    ----------
    model : Model
        The model to add the unknowns and defining equations to.
    residual : callable
        ``residual(u, v) -> sequence`` of length ``n_unknowns``, where ``u`` is a
        flat list of scalar expressions (the flattened ``u_inputs``) and ``v`` is
        the unknown vector variable.  Indexing matches the ``custom_root`` arm,
        so a residual written in plain arithmetic works unchanged in both.
    u_inputs : sequence of Expression
        The expressions the block depends on.  Unlike the opaque node these are
        not DAG edges into a hidden solve -- they simply appear in the emitted
        equations -- but they are still required, so the two arms take the same
        arguments.
    n_unknowns : int
        Length of ``v``.
    x0 : array-like, optional
        **Refused.**  Accepted in the signature only so that switching
        ``formulation`` gives a pointed error instead of a bare ``TypeError``.
        There is no inner Newton solve to start here, and discopt has no
        per-variable start slot, so a recorded ``x0`` would be read by nothing --
        a silent no-op.  Pass a starting point through
        ``solve(initial_solution={v: x0})`` instead, and select a root with
        ``bounds`` rather than with a starting point.
    bounds : tuple of (array-like, array-like), optional
        ``(lb, ub)`` for ``v``.  Default is the model-wide default box.  Pass the
        real bounds when you have them: an unbounded ``v`` has no finite
        McCormick relaxation, which forfeits the certificate this lowering exists
        to make possible.
    name : str
        Base name for the unknown vector; suffixed if already taken.

    Returns
    -------
    Variable
        The vector variable ``v``, indexable as ``v[i]`` exactly like the opaque
        node's return value.

    Raises
    ------
    ValueError
        If ``n_unknowns < 1``, ``bounds`` has the wrong length, ``x0`` is given,
        or ``residual`` does not return exactly ``n_unknowns`` entries.
    """
    if n_unknowns < 1:
        raise ValueError(f"n_unknowns must be >= 1, got {n_unknowns}")

    if x0 is not None:
        # Refuse rather than record-and-ignore.  ``v`` is an ordinary model
        # variable; nothing in discopt reads a per-variable start value, so
        # stashing x0 on it would look like a warm start and be read by nothing.
        raise ValueError(
            "x0= is not supported with formulation='full_space': there is no "
            "inner Newton solve to start, and a lowered v is an ordinary model "
            "variable. Pass the starting point to the solve instead -- "
            "solve(initial_solution={v: x0}) -- and use bounds= to select which "
            "root you want, which unlike x0 is visible to the relaxation."
        )

    lb: Union[float, np.ndarray]
    ub: Union[float, np.ndarray]
    if bounds is None:
        lb = -9.999e19
        ub = 9.999e19
    else:
        lb_a, ub_a = bounds
        lb = np.broadcast_to(np.asarray(lb_a, dtype=float), (n_unknowns,)).copy()
        ub = np.broadcast_to(np.asarray(ub_a, dtype=float), (n_unknowns,)).copy()

    v = model.continuous(_unique_name(model, name), shape=(n_unknowns,), lb=lb, ub=ub)

    u_flat = _flatten_u_inputs(u_inputs)
    rows = residual(u_flat, v)
    try:
        n_rows = len(rows)
    except TypeError as e:  # not a sequence -- a common mistake for n_unknowns=1
        raise ValueError(
            f"residual must return a sequence of {n_unknowns} expressions, "
            f"got {type(rows).__name__}"
        ) from e
    if n_rows != n_unknowns:
        raise ValueError(f"residual must return {n_unknowns} entries, got {n_rows}")

    # The square defining system.  These are ordinary equality constraints: the
    # tape compiles them, FBBT propagates through them, and the spatial B&B
    # relaxes them -- all of which the opaque node blocks.
    for row in rows:
        model.subject_to(row == 0.0)

    return v


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
