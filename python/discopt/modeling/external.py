"""External functions with caller-supplied derivatives -- the "grey box" node.

``dm.custom`` wraps an opaque callable that **discopt differentiates**: the body
must be JAX-traceable, and the gradient/Hessian come from tracing it. That rules
out the case this module exists for -- a compiled simulator, a subprocess, a
legacy Fortran kernel -- where the values and the derivatives both come from
outside Python and nothing is traceable. Measured before this module existed: a
plain-numpy body that branches on a value fails on *every* path, including
``solver="direct"``, with ``TracerArrayConversionError``.

``dm.external`` closes that gap. You supply the values *and* the derivatives;
discopt consumes them. It is the analogue of Pyomo's ``ExternalGreyBoxModel``,
with one deliberate difference recorded under "What you do not get" below.

How it works, and why it is built this way
------------------------------------------
The callables are reached through :func:`jax.pure_callback`, which is the only
way to put non-traceable code inside the traced DAG. A bare ``pure_callback`` is
not differentiable at all (*"Pure callbacks do not support JVP"*), so the
derivatives are attached with **nested** :func:`jax.custom_jvp` rules:

* the value's JVP rule contracts the caller's Jacobian with the tangent, and
* the *Jacobian* is itself a ``custom_jvp`` function whose rule contracts the
  caller's Hessian -- which is what makes the block **twice** differentiable.

That second level is the whole trick. Without it the first derivatives work and
the second derivatives raise, which is exactly the half-working state measured
before this module: the Jacobian was consumed (4 calls) and then POUNCE asked
for a Hessian and the solve ended ``status="error"`` with the incumbent
withheld. The nested form was validated against an analytic Hessian and a
symbolic twin before any of this was written: 9/9 exact agreement, including
under ``jit`` and in the forward-over-reverse HVP pattern POUNCE requests.

Shapes follow one rule, so scalar and vector blocks share a code path::

    fn(x)   -> shape                      # `shape=` declares this
    jac(x)  -> shape + x.shape
    hess(x) -> shape + x.shape + x.shape

The input shape is read off the traced argument, so nothing about the model's
size has to be declared twice.

What you do not get
-------------------
An external block is **opaque to the relaxation layer**, so it inherits the
whole :class:`~discopt.modeling.core.CustomCall` contract, unchanged and
deliberately: no global optimality certificate (``status="feasible"`` with
``bound``/``gap`` ``None``), a hard refusal when integer/binary variables are
present (global B&B would have no valid node relaxation -- sound-or-refuse),
and a refusal from ``.nl`` export and the lifted McCormick compiler. Reusing
``CustomCall`` rather than adding a node type is why all of that applies
automatically instead of being re-derived here.

``hess`` is optional, but it is not free to omit: the NLP path needs second
derivatives, so a model containing a Hessian-less block is **refused with a
message naming the fix** rather than allowed to fail deep inside the solver.
Without a Hessian the block is still usable through
``Model.solve(solver="direct")``, which needs values only.

If your function *is* expressible in ``dm.*`` primitives, use those instead
(:func:`~discopt.modeling.core.udf`) -- that keeps the global certificate. If it
is traceable but you would rather discopt differentiated it, use
:func:`~discopt.modeling.core.custom`. ``dm.external`` is for the case where the
derivatives are yours.

Example
-------
::

    import numpy as np
    import discopt.modeling as dm

    def sim(x):                 # a stand-in for compiled code
        return np.array([x[0] ** 2 * x[1]])

    def sim_jac(x):
        return np.array([[2.0 * x[0] * x[1], x[0] ** 2]])

    def sim_hess(x):
        return np.array([[[2.0 * x[1], 2.0 * x[0]], [2.0 * x[0], 0.0]]])

    block = dm.external(sim, jac=sim_jac, hess=sim_hess, shape=(1,), name="sim")

    m = dm.Model("greybox")
    x = m.continuous("x", shape=(2,), lb=0.2, ub=3.0)
    m.minimize(dm.sum(x))
    m.subject_to(block(x)[0] == 3.0)
    res = m.solve()             # status="feasible"; no certificate, by contract

Note that importing this module does not import JAX; building an external block
does, because the callback machinery is JAX's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from discopt.modeling.core import custom

__all__ = ["ExternalSpec", "external", "external_failure", "external_spec"]


@dataclass(frozen=True)
class ExternalSpec:
    """What the solver needs to know about an external block.

    Attached to the composite callable a :func:`external` builder wraps, so the
    solver can find it by walking the DAG for ``CustomCall`` nodes without a new
    node type existing. :func:`external_spec` is the accessor.
    """

    name: str
    out_shape: tuple[int, ...]
    has_hessian: bool


def external_spec(fn: Any) -> Optional[ExternalSpec]:
    """The :class:`ExternalSpec` behind *fn*, or ``None`` if it is not external.

    *fn* is a ``CustomCall.fn``: either an ordinary ``dm.custom`` body (no spec)
    or the composite built by :func:`external` (which carries one).
    """
    spec = getattr(fn, "_discopt_external", None)
    return spec if isinstance(spec, ExternalSpec) else None


def _as_shape(shape: Union[None, int, Sequence[int]]) -> tuple[int, ...]:
    """Normalize a declared output shape: ``None``/``()`` scalar, ``3`` -> ``(3,)``."""
    if shape is None:
        return ()
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    try:
        return tuple(int(s) for s in shape)
    except TypeError as exc:  # pragma: no cover - defensive
        raise TypeError(
            f"shape= must be an int or a sequence of ints, got {type(shape).__name__}"
        ) from exc


def _checked(
    f: Callable,
    *,
    role: str,
    label: str,
    want: Callable[[tuple[int, ...]], tuple[int, ...]],
    failures: Optional[list] = None,
) -> Callable:
    """Wrap an external callable so a wrong return is a *named* error.

    :func:`jax.pure_callback` does reject a mis-shaped or mis-typed return -- it
    does not silently coerce, so soundness never depended on this -- but it
    reports ``JaxRuntimeError: INTERNAL: CpuCallback error calling callback``,
    which says nothing about which of three callables was wrong or what shape was
    expected. A transposed Jacobian from a hand-written wrapper is exactly the
    mistake this feature invites, so it gets a message that names the callable,
    the role, and both shapes. (The message survives the callback boundary: JAX
    appends it to the ``JaxRuntimeError`` text, verified.)

    Coercion through ``np.asarray(..., dtype=float)`` is deliberate and is the
    one place this wrapper is lenient: external code legitimately returns a
    Python list, a float, or an integer array, all of which a bare callback
    refuses. Shape is *not* coerced -- a reshape would be the silent-garbage
    failure this whole wrapper exists to prevent.

    ``None`` is rejected before that coercion rather than through it, because
    ``np.asarray(None, dtype=float)`` does **not** raise -- it returns
    ``array(nan)`` with shape ``()``. A callable that falls off the end without a
    ``return`` would therefore pass validation whenever the declared shape is
    scalar, and feed NaN to the solver as if it were a value. (Measured; this is
    the one case where the lenient coercion was silently unsound.)

    *failures* is where a raised error is recorded before being re-raised. The
    caller needs it because raising is not enough on its own: POUNCE catches an
    exception thrown from inside a Hessian callback, logs it, and lets the solve
    continue to a withheld-incumbent ``status="error"`` -- so the error never
    reaches the caller of ``Model.solve()``. The recorded exception is what the
    solver re-raises instead (see ``solver.py``'s ``_external_block_failure``).
    """

    def call(x_np):
        try:
            raw = f(x_np)
            if raw is None:
                raise ValueError(
                    f"external function {label!r}: {role} returned None. It must "
                    f"return a float array of shape "
                    f"{want(tuple(np.shape(x_np)))}; a missing return statement is "
                    f"the usual cause. (None is rejected explicitly because numpy "
                    f"would convert it to NaN rather than raise.)"
                )
            try:
                arr = np.asarray(raw, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"external function {label!r}: {role} returned "
                    f"{type(raw).__name__}, which is not convertible to a float array"
                ) from exc
            expected = want(tuple(np.shape(x_np)))
            if arr.shape != expected:
                raise ValueError(
                    f"external function {label!r}: {role} returned shape {arr.shape}, "
                    f"expected {expected}. The rule is fn -> shape, jac -> shape + "
                    f"x.shape, hess -> shape + x.shape + x.shape, where shape= is the "
                    f"declared output shape and x.shape is {tuple(np.shape(x_np))}. A "
                    f"transposed Jacobian is the usual cause."
                )
            return arr
        except BaseException as exc:
            # Record, then re-raise unchanged. This is not an `except: pass` --
            # nothing is swallowed here (CLAUDE.md §7); the record exists because
            # something downstream swallows it.
            if failures is not None and not failures:
                failures.append(exc)
            raise

    call.__name__ = f"{label}_{role}"
    return call


def external(
    fn: Callable,
    *,
    jac: Optional[Callable] = None,
    hess: Optional[Callable] = None,
    shape: Union[None, int, Sequence[int]] = (),
    name: Optional[str] = None,
) -> Callable:
    """Embed an external function whose derivatives *you* supply.

    Use this when the values and derivatives come from outside Python -- a
    compiled simulator, a subprocess, a legacy kernel -- so nothing about the
    body is JAX-traceable and :func:`~discopt.modeling.core.custom` cannot
    differentiate it. See the module docstring for the design and for the
    certificate contract an external block inherits.

    Parameters
    ----------
    fn : callable
        ``fn(x) -> array`` of shape *shape*. Called with a plain
        :class:`numpy.ndarray`; may do anything, including I/O.
    jac : callable
        ``jac(x) -> array`` of shape ``shape + x.shape``. **Required**: without
        derivatives there is no route through the NLP path, and the honest tool
        for a pure black box is ``dm.custom`` with ``solve(solver="direct")``.
    hess : callable, optional
        ``hess(x) -> array`` of shape ``shape + x.shape + x.shape``. Needed for
        the NLP path, which uses second derivatives; omitting it restricts the
        model to ``solve(solver="direct")`` and the solver says so rather than
        failing deep inside POUNCE.
    shape : int or tuple of int, default ``()``
        The output shape of *fn*. ``()`` is a scalar (an objective term), ``(m,)``
        a vector of *m* residuals (a constraint body). It cannot be inferred
        without calling *fn*, which may be expensive or undefined at the point
        discopt would pick.
    name : str, optional
        Display name used in reprs and in every error message above. Defaults to
        ``fn.__name__``.

    Returns
    -------
    callable
        A builder; call it with an expression argument to produce the DAG node,
        exactly like :func:`~discopt.modeling.core.custom`.

    Raises
    ------
    TypeError
        If *fn*/*jac*/*hess* are not callable, or *jac* is missing.

    Examples
    --------
    >>> import numpy as np, discopt.modeling as dm
    >>> f = lambda x: np.array([x[0] ** 2])          # doctest: +SKIP
    >>> J = lambda x: np.array([[2.0 * x[0]]])       # doctest: +SKIP
    >>> H = lambda x: np.array([[[2.0]]])            # doctest: +SKIP
    >>> block = dm.external(f, jac=J, hess=H, shape=(1,), name="sq")  # doctest: +SKIP
    """
    label = name or str(getattr(fn, "__name__", "external"))
    if not callable(fn):
        raise TypeError(f"external() expects a callable, got {type(fn).__name__}")
    if jac is None:
        raise TypeError(
            f"external({label!r}) requires jac=...: an external function with no "
            "derivatives has no route through the NLP path. For a genuine black box "
            "use dm.custom(...) with Model.solve(solver='direct'), a derivative-free "
            "global search that needs values only."
        )
    if not callable(jac):
        raise TypeError(f"external(jac=...) expects a callable, got {type(jac).__name__}")
    if hess is not None and not callable(hess):
        raise TypeError(f"external(hess=...) expects a callable, got {type(hess).__name__}")

    out_shape = _as_shape(shape)

    # One list per block, shared by its three wrappers: the first error any of
    # them raises lands here so the solver can re-raise it if POUNCE swallowed it.
    failures: list = []

    f_call = _checked(fn, role="fn", label=label, want=lambda xs: out_shape, failures=failures)
    j_call = _checked(
        jac, role="jac", label=label, want=lambda xs: out_shape + xs, failures=failures
    )
    h_call = (
        None
        if hess is None
        else _checked(
            hess,
            role="hess",
            label=label,
            want=lambda xs: out_shape + xs + xs,
            failures=failures,
        )
    )

    composite = _build_composite(label, out_shape, f_call, j_call, h_call)
    spec = ExternalSpec(name=label, out_shape=out_shape, has_hessian=h_call is not None)

    # The spec rides on a plain wrapper rather than on the ``custom_jvp`` object
    # itself: ``CustomCall.fn`` is what the solver inspects, and a plain function
    # is guaranteed to accept an attribute.
    def body(x):
        return composite(x)

    body.__name__ = label
    body._discopt_external = spec  # type: ignore[attr-defined]
    body._discopt_external_failures = failures  # type: ignore[attr-defined]

    builder = custom(body, name=label)
    builder._discopt_external = spec  # type: ignore[attr-defined]
    builder._discopt_external_failures = failures  # type: ignore[attr-defined]
    return builder


def external_failure(fn: Any) -> Optional[BaseException]:
    """The first error *fn*'s external callables raised, if any.

    ``None`` both for a non-external ``fn`` and for an external one that has not
    failed, because the caller does the same thing in either case.
    """
    failures = getattr(fn, "_discopt_external_failures", None)
    if isinstance(failures, list) and failures:
        exc = failures[0]
        return exc if isinstance(exc, BaseException) else None
    return None


def _build_composite(
    label: str,
    out_shape: tuple[int, ...],
    f_call: Callable,
    j_call: Callable,
    h_call: Optional[Callable],
) -> Callable:
    """The nested ``custom_jvp`` composite described in the module docstring.

    JAX is imported here, not at module scope: an ordinary discopt solve imports
    zero ``jax`` modules and merely importing ``discopt.modeling`` must not
    change that. Building an external block does import JAX, because
    ``pure_callback`` is the mechanism.
    """
    import jax
    import jax.numpy as jnp

    def sds(shape: tuple[int, ...]):
        return jax.ShapeDtypeStruct(shape, jnp.float64)

    def contract(tensor, dx):
        """Contract *tensor*'s trailing input axes with the tangent *dx*.

        One expression for every input rank. A scalar input needs the explicit
        multiply: ``tensordot`` with no axes is an outer product, which would be
        silently wrong rather than an error.
        """
        nd = jnp.ndim(dx)
        if nd == 0:
            return tensor * dx
        axes = (tuple(range(-nd, 0)), tuple(range(nd)))
        return jnp.tensordot(tensor, dx, axes=axes)

    @jax.custom_jvp
    def jacobian(x):
        return jax.pure_callback(j_call, sds(out_shape + x.shape), x)

    if h_call is not None:

        @jacobian.defjvp
        def _jacobian_jvp(primals, tangents):
            (x,), (dx,) = primals, tangents
            hessian = jax.pure_callback(h_call, sds(out_shape + x.shape + x.shape), x)
            return jacobian(x), contract(hessian, dx)

    else:

        @jacobian.defjvp
        def _jacobian_jvp(primals, tangents):
            # Defence in depth. ``solve_model`` refuses a Hessian-less external
            # block up front with a fuller message; this fires if some other route
            # differentiates the Jacobian anyway, and replaces JAX's
            # "Pure callbacks do not support JVP" with something actionable.
            raise ValueError(
                f"external function {label!r} supplies no Hessian, and something is "
                "asking for its second derivatives. Pass hess=... to dm.external, or "
                "solve with Model.solve(solver='direct'), which needs values only."
            )

    @jax.custom_jvp
    def value(x):
        return jax.pure_callback(f_call, sds(out_shape), x)

    @value.defjvp
    def _value_jvp(primals, tangents):
        (x,), (dx,) = primals, tangents
        return value(x), contract(jacobian(x), dx)

    return value
