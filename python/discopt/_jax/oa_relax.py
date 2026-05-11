"""JAX-traceable (cv, cc) shim for polyhedral outer approximations.

The standalone ``polyhedral_oa.outer_approximation`` (M11 of issue #51)
returns a list of affine cuts that sandwich a univariate function ``f``
on a static reference box. The relaxation compiler, by contrast, needs a
``(mid, cv_a, cc_a) -> (cv, cc)`` pair that is jit/vmap compatible and
plays the same role as the existing McCormick ``relax_*`` functions.

This module performs the conversion at *compile time*: cuts are
materialised once into two stacks of ``(slope, intercept)`` pairs (one
for lower-bound cuts, one for upper-bound cuts). At runtime the cv and
cc are recovered as the elementwise max / min of all affine cuts
evaluated at the linearisation point ``mid``. The result is sound by
construction because each cut globally encloses ``f`` on the reference
box (see ``polyhedral_oa.py`` docstring).

Reference-box choice
--------------------

The reference box used to build the OA is the *static* interval
enclosure of the inner expression, computed via
``evaluate_interval`` from ``discopt._jax.convexity``. Using the static
declared bounds means the OA is built once at compile time and stays
fixed across B&B nodes. As nodes shrink, the OA may become looser than
the McCormick envelope evaluated on the node-local interval — but it is
always sound. Future work: rebuild OAs per node when the gain warrants
the recompile cost.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np

from discopt._jax import polyhedral_oa

_NAME_TO_JAX_FN: dict[str, Callable] = {
    "exp": jnp.exp,
    "log": jnp.log,
    "log2": jnp.log2,
    "log10": jnp.log10,
    "sqrt": jnp.sqrt,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "tan": jnp.tan,
    "atan": jnp.arctan,
    "asin": jnp.arcsin,
    "acos": jnp.arccos,
    "sinh": jnp.sinh,
    "cosh": jnp.cosh,
    "tanh": jnp.tanh,
}


def is_supported(name: str) -> bool:
    """Whether ``polyhedral_oa`` can build an OA for the given op."""
    return name in _NAME_TO_JAX_FN


def make_oa_relax(
    name: str,
    ref_box: tuple[float, float],
    arithmetic: str,
    *,
    degree: int = 8,
    n_slopes: int = 16,
) -> Callable:
    """Build a ``(mid, cv_a, cc_a) -> (cv, cc)`` JAX function.

    Args:
        name: univariate operator name (must be a key of ``_NAME_TO_JAX_FN``).
        ref_box: ``(a, b)`` reference box used to fit the OA. Must satisfy
            ``a < b`` and be inside the natural domain of ``f``.
        arithmetic: one of ``"chebyshev"``, ``"taylor"``, ``"mccormick"``.
        degree: polynomial degree for the bound provider.
        n_slopes: number of candidate slopes (yields up to ``2 * n_slopes``
            cuts after deduplication).

    Returns:
        A jit/vmap-friendly function with the same signature as the
        McCormick ``relax_*`` shims used by ``relaxation_compiler.py``.

    Raises:
        ValueError: when ``name`` is not in ``_NAME_TO_JAX_FN`` or
            ``ref_box`` is degenerate / outside the operator's domain.
    """
    if name not in _NAME_TO_JAX_FN:
        raise ValueError(f"OA shim does not support operator {name!r}")
    a, b = float(ref_box[0]), float(ref_box[1])
    if not (a < b):
        raise ValueError(f"ref_box must have a < b, got {(a, b)}")
    f = _NAME_TO_JAX_FN[name]
    oa = polyhedral_oa.outer_approximation(f, (a, b), arithmetic, degree=degree, n_slopes=n_slopes)

    lower_slopes: list[float] = []
    lower_intercepts: list[float] = []
    upper_slopes: list[float] = []
    upper_intercepts: list[float] = []
    for cut in oa.cuts:
        sx = float(cut.coeffs[0])  # = -s
        sy = float(cut.coeffs[1])  # = +1
        if sy <= 0.0:
            continue
        slope = -sx / sy
        intercept = float(cut.rhs) / sy
        if cut.sense == ">=":
            lower_slopes.append(slope)
            lower_intercepts.append(intercept)
        elif cut.sense == "<=":
            upper_slopes.append(slope)
            upper_intercepts.append(intercept)
    if not lower_slopes or not upper_slopes:
        # OA degenerate; fall back to a trivial function that defers to
        # the natural-domain image. We materialise constant arrays so the
        # caller's downstream jit path stays unbroken.
        lo_arr_lb = jnp.asarray([float("-inf")], dtype=jnp.float64)
        lo_arr_int = jnp.asarray([0.0], dtype=jnp.float64)
        hi_arr_lb = jnp.asarray([0.0], dtype=jnp.float64)
        hi_arr_int = jnp.asarray([float("inf")], dtype=jnp.float64)
    else:
        lo_arr_lb = jnp.asarray(lower_slopes, dtype=jnp.float64)
        lo_arr_int = jnp.asarray(lower_intercepts, dtype=jnp.float64)
        hi_arr_lb = jnp.asarray(upper_slopes, dtype=jnp.float64)
        hi_arr_int = jnp.asarray(upper_intercepts, dtype=jnp.float64)

    def relax_fn(
        mid,
        cv_a,
        cc_a,
        _lo_s=lo_arr_lb,
        _lo_i=lo_arr_int,
        _hi_s=hi_arr_lb,
        _hi_i=hi_arr_int,
    ):
        cv_vals = _lo_s * mid + _lo_i
        cc_vals = _hi_s * mid + _hi_i
        return jnp.max(cv_vals), jnp.min(cc_vals)

    return relax_fn


def _safe_box_for_op(name: str, lo: float, hi: float) -> tuple[float, float]:
    """Clip ``[lo, hi]`` to the natural domain of the named operator.

    Returns a strictly nondegenerate box; if the natural-domain
    intersection is empty or degenerate, returns ``(lo, hi)`` unchanged
    (the caller will surface the eventual error).
    """
    eps = 1e-9
    if name in {"log", "log2", "log10"}:
        lo = max(lo, eps)
        hi = max(hi, lo + eps)
    elif name == "sqrt":
        lo = max(lo, 0.0)
        hi = max(hi, lo + eps)
    elif name in {"asin", "acos"}:
        lo = max(lo, -1.0 + eps)
        hi = min(hi, 1.0 - eps)
        if hi <= lo:
            hi = lo + eps
    return float(lo), float(hi)


def static_box_for_arg(arg, model) -> tuple[float, float] | None:
    """Compute a static interval bound for an inner expression.

    Returns ``(lo, hi)`` or ``None`` if the bound cannot be inferred
    (e.g. the expression contains an unsupported operator).
    """
    try:
        from discopt._jax.convexity.interval_eval import evaluate_interval
    except Exception:
        return None
    try:
        iv = evaluate_interval(arg, model)
    except Exception:
        return None
    try:
        lo = float(np.asarray(iv.lo).reshape(()))
        hi = float(np.asarray(iv.hi).reshape(()))
    except Exception:
        return None
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None
    return lo, hi


__all__ = ["is_supported", "make_oa_relax", "static_box_for_arg", "_safe_box_for_op"]
