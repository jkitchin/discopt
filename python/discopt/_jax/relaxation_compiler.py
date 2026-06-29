"""
Relaxation Compiler: Expression tree -> McCormick relaxation function.

Walks the Expression DAG and produces a pure jax.numpy function that computes
compositional McCormick relaxations (convex underestimator cv, concave
overestimator cc). The returned functions are compatible with jax.jit and
jax.vmap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from discopt._jax.learned_relaxations import LearnedRelaxationRegistry

from discopt._jax.mccormick import (
    relax_abs,
    relax_acos,
    relax_add,
    relax_asin,
    relax_atan,
    relax_bilinear,
    relax_cos,
    relax_cosh,
    relax_div,
    relax_entropy,
    relax_exp,
    relax_log,
    relax_log2,
    relax_log10,
    relax_neg,
    relax_pow,
    relax_sigmoid,
    relax_sign,
    relax_sin,
    relax_sinh,
    relax_softplus,
    relax_sqrt,
    relax_sub,
    relax_tan,
    relax_tanh,
)
from discopt._jax.multivariate_mccormick import get_composition_rule
from discopt._jax.piecewise_mccormick import (
    piecewise_mccormick_bilinear,
    piecewise_relax_cos,
    piecewise_relax_exp,
    piecewise_relax_log,
    piecewise_relax_sin,
    piecewise_relax_sqrt,
)

# Import expression types from the modeling API
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    CustomCall,
    Expression,
    FunctionCall,
    IndexExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)
from discopt.solver_tuning import current as _tuning


def _compute_var_offset(var: Variable, model: Model) -> int:
    """Compute the starting offset of a variable in the flat x vector."""
    offset = 0
    for v in model._variables[: var._index]:
        offset += v.size
    return offset


def _is_constant_expr(expr: Expression) -> bool:
    """Check if an expression is a Constant."""
    return isinstance(expr, Constant)


def _get_constant_value(expr: Expression):
    """Get the numeric value from a Constant expression."""
    return jnp.array(expr.value)


def _resolve_scalar_var_offset(expr: Expression, model: Model) -> int | None:
    """Resolve a scalar Variable or scalar-IndexExpression to its flat offset.

    Returns the flat offset into the x vector if ``expr`` is a scalar
    ``Variable`` or an ``IndexExpression`` over a ``Variable`` that resolves
    to a single scalar component. Returns ``None`` otherwise.
    """
    if isinstance(expr, Variable) and expr.size == 1:
        return _compute_var_offset(expr, model)
    if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
        base_off = _compute_var_offset(expr.base, model)
        idx = expr.index
        if isinstance(idx, int):
            return base_off + idx
        if isinstance(idx, tuple) and len(idx) == 1 and isinstance(idx[0], int):
            return base_off + idx[0]
    return None


def _try_extract_trilinear_chain(expr: Expression, model: Model) -> tuple[int, int, int] | None:
    """Detect a 3-distinct-variable trilinear chain ``v_a * v_b * v_c``.

    Recognizes both left- and right-associative parsings:
      * ``Mul(Mul(v_a, v_b), v_c)``
      * ``Mul(v_a, Mul(v_b, v_c))``

    Each leaf must be a scalar ``Variable`` (or scalar ``IndexExpression``).
    The three resolved flat offsets must be pairwise distinct (so squared or
    cubed terms fall through to other handlers).

    Returns ``(off_a, off_b, off_c)`` on match, else ``None``.
    """
    if not (isinstance(expr, BinaryOp) and expr.op == "*"):
        return None

    inner, outer = None, None
    if isinstance(expr.left, BinaryOp) and expr.left.op == "*":
        inner, outer = expr.left, expr.right
    elif isinstance(expr.right, BinaryOp) and expr.right.op == "*":
        inner, outer = expr.right, expr.left
    else:
        return None

    a = _resolve_scalar_var_offset(inner.left, model)
    b = _resolve_scalar_var_offset(inner.right, model)
    c = _resolve_scalar_var_offset(outer, model)
    if a is None or b is None or c is None:
        return None
    # Pairwise distinct — repeated-variable cases (x*x*y, etc.) belong to
    # the signomial / power handlers.
    if a == b or a == c or b == c:
        return None
    return (a, b, c)


def _try_extract_signed_abs_product(expr: Expression, model: Model) -> int | None:
    """Detect the Weymouth term ``f * |f|`` over a single scalar variable.

    Matches ``Mul(v, Abs(v))`` and ``Mul(Abs(v), v)`` where both factors resolve
    to the *same* scalar variable offset. Returns that flat offset on match, else
    ``None``. Routes to the tight single-inflection envelope
    (:func:`discopt._jax.symbolic.domains.gas.weymouth_relax`) instead of the
    loose bilinear product of ``f`` and ``|f|``.
    """
    if not (isinstance(expr, BinaryOp) and expr.op == "*"):
        return None
    for factor, other in ((expr.left, expr.right), (expr.right, expr.left)):
        if isinstance(other, UnaryOp) and other.op == "abs":
            off = _resolve_scalar_var_offset(factor, model)
            off_abs = _resolve_scalar_var_offset(other.operand, model)
            if off is not None and off == off_abs:
                return off
    return None


def _try_extract_signed_power(expr: Expression, model: Model) -> tuple[int, float] | None:
    """Detect the signed-power flow term ``f * |f|**(beta-1)`` over one scalar var.

    Matches ``Mul(v, Pow(Abs(v), p))`` / ``Mul(Pow(Abs(v), p), v)`` with ``p`` a
    positive constant and both factors the *same* scalar variable. Returns
    ``(offset, beta)`` with ``beta = p + 1`` on match, else ``None``. This is the
    Panhandle generalization of the Weymouth term (``beta == 2``, handled by
    :func:`_try_extract_signed_abs_product`); it routes to
    :func:`discopt._jax.symbolic.domains.gas.signed_power_relax`, which is a tight
    single-inflection envelope valid over any box (no domain restriction).
    """
    if not (isinstance(expr, BinaryOp) and expr.op == "*"):
        return None
    for factor, other in ((expr.left, expr.right), (expr.right, expr.left)):
        if (
            isinstance(other, BinaryOp)
            and other.op == "**"
            and isinstance(other.right, Constant)
            and isinstance(other.left, UnaryOp)
            and other.left.op == "abs"
        ):
            try:
                p = float(other.right.value)
            except (TypeError, ValueError):
                continue
            if p <= 0.0:
                continue
            off = _resolve_scalar_var_offset(factor, model)
            off_abs = _resolve_scalar_var_offset(other.left.operand, model)
            if off is not None and off == off_abs:
                return (off, p + 1.0)
    return None


def _try_extract_xlogx(expr: Expression, model: Model) -> int | None:
    """Detect the entropy/mixing term ``x * log(x)`` over a single scalar variable.

    Matches ``Mul(v, log(v))`` and ``Mul(log(v), v)`` where the bare variable factor
    and the ``log`` argument resolve to the *same* scalar offset. Returns that flat
    offset on match, else ``None``. Routes to the dedicated convex envelope of
    ``x ln x`` (:func:`discopt._jax.symbolic.domains.chemeng.xlogx_relax`) — tighter
    than the generic bilinear product of ``v`` and ``log(v)``. The envelope is valid
    only for ``v > 0`` (the term's natural domain), so the runtime dispatch uses it
    only when the node's lower bound is positive and otherwise falls back to the
    bilinear.
    """
    if not (isinstance(expr, BinaryOp) and expr.op == "*"):
        return None
    for factor, other in ((expr.left, expr.right), (expr.right, expr.left)):
        if isinstance(other, FunctionCall) and other.func_name == "log" and len(other.args) == 1:
            off = _resolve_scalar_var_offset(factor, model)
            off_log = _resolve_scalar_var_offset(other.args[0], model)
            if off is not None and off == off_log:
                return off
    return None


def _try_extract_monod(expr: Expression, model: Model) -> tuple[int, float] | None:
    """Detect the Monod / saturating term ``x/(K+x)`` over a single scalar var.

    Matches ``Div(v, Add(K, v))`` / ``Div(v, Add(v, K))`` with ``K`` a positive
    constant and the numerator and the additive variable the SAME scalar var.
    Returns ``(offset, K)`` on match, else ``None``, routing to the dedicated
    concave envelope (:func:`...chemeng.saturating_relax`).

    The envelope is valid only for ``v >= 0``. That is enforced at COMPILE time by
    requiring the variable's *declared* lower bound to be ``>= 0``: spatial B&B only
    tightens bounds, so every node box then keeps ``v >= 0`` and no per-node guard
    is needed. When the declared lower bound can be negative the term is left to the
    generic division relaxation.
    """
    if not (isinstance(expr, BinaryOp) and expr.op == "/"):
        return None
    num_off = _resolve_scalar_var_offset(expr.left, model)
    if num_off is None:
        return None
    den = expr.right
    if not (isinstance(den, BinaryOp) and den.op == "+"):
        return None
    for kside, vside in ((den.left, den.right), (den.right, den.left)):
        if not _is_constant_expr(kside):
            continue
        try:
            kval = float(_get_constant_value(kside))
        except (TypeError, ValueError):
            continue
        v_off = _resolve_scalar_var_offset(vside, model)
        if kval > 0.0 and v_off is not None and v_off == num_off:
            from discopt._jax.model_utils import flat_variable_bounds

            lb0, _ = flat_variable_bounds(model)
            if float(lb0[num_off]) >= 0.0:
                return (num_off, kval)
    return None


def _try_extract_arrhenius(expr: Expression, model: Model) -> tuple[int, float] | None:
    """Detect the Arrhenius rate term ``exp(-c/T)`` over a single scalar variable.

    Matches ``exp(Div(Constant(neg), v))`` with ``neg < 0`` (so the term is
    ``exp(-c/T)`` with ``c = -neg > 0``) and ``v`` a scalar variable. Returns
    ``(offset, c)`` on match, else ``None``, routing to the dedicated
    single-inflection envelope (:func:`...chemeng.arrhenius_relax`).

    The term (and its envelope) require ``T > 0`` (a pole at ``T = 0``). That is
    enforced at COMPILE time by requiring the variable's declared lower bound to be
    strictly positive; spatial B&B only tightens bounds, so every node box keeps
    ``T > 0`` and no per-node guard is needed.
    """
    if not (isinstance(expr, FunctionCall) and expr.func_name == "exp" and len(expr.args) == 1):
        return None
    arg = expr.args[0]
    if not (isinstance(arg, BinaryOp) and arg.op == "/" and _is_constant_expr(arg.left)):
        return None
    try:
        neg = float(_get_constant_value(arg.left))
    except (TypeError, ValueError):
        return None
    if neg >= 0.0:  # need exp(-c/T) with c > 0
        return None
    v_off = _resolve_scalar_var_offset(arg.right, model)
    if v_off is None:
        return None
    from discopt._jax.model_utils import flat_variable_bounds

    lb0, _ = flat_variable_bounds(model)
    if float(lb0[v_off]) > 0.0:
        return (v_off, -neg)
    return None


def _try_extract_signomial_factors(
    expr: Expression, model: Model
) -> list[tuple[int, float]] | None:
    """Try to decompose a multiplication tree into signomial factors.

    Walks a tree of BinaryOp("*") nodes and collects (var_offset, exponent)
    pairs where each leaf is Variable^Constant (or just Variable, i.e. ^1).

    Returns None if the tree contains non-signomial terms (e.g., general
    expressions, constants, or non-variable bases).
    """
    factors: list[tuple[int, float]] = []

    def _collect(e: Expression) -> bool:
        if isinstance(e, BinaryOp) and e.op == "*":
            return _collect(e.left) and _collect(e.right)
        if isinstance(e, BinaryOp) and e.op == "**":
            if isinstance(e.right, Constant):
                exp_val = float(e.right.value)
                base = e.left
                if isinstance(base, Variable) and base.size == 1:
                    offset = _compute_var_offset(base, model)
                    factors.append((offset, exp_val))
                    return True
                if isinstance(base, IndexExpression) and isinstance(base.base, Variable):
                    base_off = _compute_var_offset(base.base, model)
                    idx = base.index
                    flat_idx = (
                        base_off + idx
                        if isinstance(idx, int)
                        else base_off + idx[0]
                        if isinstance(idx, tuple) and len(idx) == 1
                        else None
                    )
                    if flat_idx is not None:
                        factors.append((flat_idx, exp_val))
                        return True
            return False
        if isinstance(e, Variable) and e.size == 1:
            offset = _compute_var_offset(e, model)
            factors.append((offset, 1.0))
            return True
        if isinstance(e, IndexExpression) and isinstance(e.base, Variable):
            base_off = _compute_var_offset(e.base, model)
            idx = e.index
            flat_idx = (
                base_off + idx
                if isinstance(idx, int)
                else base_off + idx[0]
                if isinstance(idx, tuple) and len(idx) == 1
                else None
            )
            if flat_idx is not None:
                factors.append((flat_idx, 1.0))
                return True
        return False

    if _collect(expr):
        return factors if len(factors) >= 2 else None
    return None


def _compile_relax_node(
    expr: Expression,
    model: Model,
    partitions: int = 0,
    mode: str = "standard",
    learned_registry: Optional["LearnedRelaxationRegistry"] = None,
    arithmetic: str = "mccormick",
) -> Callable:
    """
    Recursively compile an Expression node into a relaxation function.

    Each returned function takes (x_cv, x_cc, lb, ub) and returns (cv, cc)
    where cv is a convex underestimator and cc is a concave overestimator.

    Args:
        expr: Expression node to compile.
        model: Model containing variable definitions.
        partitions: If > 0, use piecewise McCormick relaxations with this
            many partitions for supported operations (bilinear, exp, log,
            sqrt, sin, cos). If 0, use standard McCormick.
        mode: Relaxation mode — ``"standard"`` (default McCormick),
            ``"piecewise"`` (piecewise McCormick), or ``"learned"``
            (ICNN-based learned relaxations with McCormick fallback).
        learned_registry: Registry of trained learned relaxations.
            Required when ``mode="learned"``.
        arithmetic: Univariate-envelope provider — ``"mccormick"``
            (default; existing analytic envelopes), ``"chebyshev"``, or
            ``"taylor"``. The latter two route supported univariate ops
            through ``oa_relax.make_oa_relax`` (M2 / M3 of issue #51).
            Falls back to McCormick for unsupported operators or when the
            inner expression's static box cannot be inferred.
    """

    if isinstance(expr, Constant):
        val = jnp.array(expr.value)

        def fn(x_cv, x_cc, lb, ub):
            return val, val

        return fn

    if isinstance(expr, Variable):
        offset = _compute_var_offset(expr, model)
        size = expr.size
        shape = expr.shape
        if shape == () or (len(shape) == 1 and shape[0] == 1 and shape == ()):

            def fn(x_cv, x_cc, lb, ub):
                return x_cv[offset], x_cc[offset]

            return fn
        else:

            def fn(x_cv, x_cc, lb, ub, _offset=offset, _size=size, _shape=shape):
                return (
                    x_cv[_offset : _offset + _size].reshape(_shape),
                    x_cc[_offset : _offset + _size].reshape(_shape),
                )

            return fn

    if isinstance(expr, Parameter):
        val = jnp.array(expr.value)

        def fn(x_cv, x_cc, lb, ub):
            return val, val

        return fn

    if isinstance(expr, BinaryOp):
        left_fn = _compile_relax_node(
            expr.left, model, partitions, mode, learned_registry, arithmetic
        )
        right_fn = _compile_relax_node(
            expr.right, model, partitions, mode, learned_registry, arithmetic
        )
        op = expr.op

        if op == "+":

            def fn(x_cv, x_cc, lb, ub):
                cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                return relax_add(cv_l, cc_l, cv_r, cc_r)

            return fn

        if op == "-":

            def fn(x_cv, x_cc, lb, ub):
                cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                return relax_sub(cv_l, cc_l, cv_r, cc_r)

            return fn

        if op == "*":
            # Optimize constant * expr and expr * constant
            if _is_constant_expr(expr.left):
                c = _get_constant_value(expr.left)

                def fn(x_cv, x_cc, lb, ub, _c=c):
                    cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                    pos = _c >= 0
                    new_cv = jnp.where(pos, _c * cv_r, _c * cc_r)
                    new_cc = jnp.where(pos, _c * cc_r, _c * cv_r)
                    return new_cv, new_cc

                return fn

            if _is_constant_expr(expr.right):
                c = _get_constant_value(expr.right)

                def fn(x_cv, x_cc, lb, ub, _c=c):
                    cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                    pos = _c >= 0
                    new_cv = jnp.where(pos, _c * cv_l, _c * cc_l)
                    new_cc = jnp.where(pos, _c * cc_l, _c * cv_l)
                    return new_cv, new_cc

                return fn

            # Weymouth pattern detection: f * |f| over a single scalar variable.
            # Routes to the tight single-inflection envelope (gas networks) when
            # the symbolic-derived JAX closure is importable; otherwise falls
            # through to the generic bilinear product of f and |f|.
            gas_off = _try_extract_signed_abs_product(expr, model)
            if gas_off is not None:
                try:
                    from discopt._jax.symbolic.domains.gas import weymouth_relax
                except ImportError:
                    weymouth_relax = None  # type: ignore[assignment]
                if weymouth_relax is not None:

                    def fn(x_cv, x_cc, lb, ub, _off=gas_off, _wr=weymouth_relax):
                        return _wr(x_cv[_off], lb[_off], ub[_off])

                    return fn

            # Signed-power (Panhandle) flow term f*|f|**(beta-1) over a single
            # scalar variable — the general-exponent sibling of Weymouth. Like
            # Weymouth it is a single-inflection function valid over any box, so the
            # tight envelope needs no domain guard. Falls through to the generic
            # bilinear when the optional [sympy] domain pack is absent.
            sp_match = _try_extract_signed_power(expr, model)
            if sp_match is not None:
                _sp_off, _sp_beta = sp_match
                try:
                    from discopt._jax.symbolic.domains.gas import signed_power_relax
                except ImportError:
                    signed_power_relax = None  # type: ignore[assignment]
                if signed_power_relax is not None:
                    _sp_fn = signed_power_relax(_sp_beta)

                    def fn(x_cv, x_cc, lb, ub, _off=_sp_off, _wr=_sp_fn):
                        return _wr(x_cv[_off], lb[_off], ub[_off])

                    return fn

            # x*log(x) entropy/mixing term over a single scalar variable. Routes to
            # the dedicated convex envelope (tighter than the bilinear of x and
            # log(x)) when x is provably positive at this node; otherwise — and when
            # the optional [sympy]-derived domain pack is absent — falls back to the
            # generic bilinear. Sound by construction: the tight branch is taken only
            # where the envelope is valid (lb > _XLOGX_POS_FLOOR), so the positivity
            # clamp below is always a no-op there and never shrinks the actual box.
            xlogx_off = _try_extract_xlogx(expr, model)
            if xlogx_off is not None:
                try:
                    from discopt._jax.symbolic.domains.chemeng import xlogx_relax
                except ImportError:
                    xlogx_relax = None  # type: ignore[assignment]
                if xlogx_relax is not None:
                    _XLOGX_POS_FLOOR = 1e-9

                    def fn(
                        x_cv,
                        x_cc,
                        lb,
                        ub,
                        _off=xlogx_off,
                        _xl=xlogx_relax,
                        _flo=_XLOGX_POS_FLOOR,
                    ):
                        # Tight branch is used only where lb > _flo, so flooring the
                        # inputs at _flo is a no-op exactly when this value is kept;
                        # it exists solely to keep the jnp.where-discarded branch from
                        # evaluating log of a non-positive number and NaN-poisoning the
                        # gradient on an lb<=0 node. The box is never shrunk where used.
                        _x = jnp.maximum(x_cv[_off], _flo)
                        _l = jnp.maximum(lb[_off], _flo)
                        _u = jnp.maximum(ub[_off], _l)
                        cv_t, cc_t = _xl(_x, _l, _u)
                        # Generic bilinear fallback (the status-quo relaxation of
                        # x*log(x)) for non-positive / undefined-domain nodes.
                        cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                        cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                        mid_l = 0.5 * (cv_l + cc_l)
                        mid_r = 0.5 * (cv_r + cc_r)
                        cv_bl, cc_bl = relax_bilinear(mid_l, mid_r, cv_l, cc_l, cv_r, cc_r)
                        pos = lb[_off] > _flo
                        return jnp.where(pos, cv_t, cv_bl), jnp.where(pos, cc_t, cc_bl)

                    return fn

            # Trilinear pattern detection: x*y*z over 3 distinct scalar
            # Variables. Routes to the Meyer-Floudas/Rikun convex-hull
            # envelope by default, with debug selectors for the older
            # best-of-three and nested-bilinear paths.
            #
            # Opt-out: setting DISCOPT_TRILINEAR=nested in the environment
            # skips this dispatch and uses the original nested-bilinear
            # path. Setting DISCOPT_TRILINEAR=exact selects the historical
            # permutation-symmetric nested McCormick path.
            tuning = _tuning()
            tri_offsets = (
                None if tuning.trilinear_nested else _try_extract_trilinear_chain(expr, model)
            )
            if tri_offsets is not None:
                if tuning.trilinear_exact:
                    from discopt._jax.envelopes import relax_trilinear_exact as _relax_trilinear
                else:
                    from discopt._jax.envelopes import (
                        relax_trilinear_meyer_floudas as _relax_trilinear,
                    )

                _ti, _tj, _tk = tri_offsets

                def fn(x_cv, x_cc, lb, ub, _i=_ti, _j=_tj, _k=_tk):
                    # Use compositional bounds (x_cv / x_cc) the same way
                    # the bilinear path does. For Variable leaves these
                    # collapse to the point value at feasible-point
                    # evaluation, and to the box [lb, ub] when computing
                    # the LP relaxation at a B&B node.
                    x_lb_, x_ub_ = x_cv[_i], x_cc[_i]
                    y_lb_, y_ub_ = x_cv[_j], x_cc[_j]
                    z_lb_, z_ub_ = x_cv[_k], x_cc[_k]
                    xv = 0.5 * (x_lb_ + x_ub_)
                    yv = 0.5 * (y_lb_ + y_ub_)
                    zv = 0.5 * (z_lb_ + z_ub_)
                    return _relax_trilinear(
                        xv,
                        yv,
                        zv,
                        x_lb_,
                        x_ub_,
                        y_lb_,
                        y_ub_,
                        z_lb_,
                        z_ub_,
                    )

                return fn

            # Signomial pattern detection: product of Variable^Constant terms
            # When all factors are x_i^{a_i} with positive lower bounds,
            # dispatch to relax_signomial_multi for tighter relaxation.
            sig_factors = _try_extract_signomial_factors(expr, model)
            if sig_factors is not None:
                from discopt._jax.envelopes import relax_signomial_multi

                _offsets = np.array([f[0] for f in sig_factors])
                _exps = np.array([f[1] for f in sig_factors], dtype=np.float64)

                def fn(x_cv, x_cc, lb, ub, _offs=_offsets, _exps=_exps):
                    xs = x_cv[_offs]
                    var_lbs = lb[_offs]
                    var_ubs = ub[_offs]
                    # Only use signomial when all lower bounds are positive
                    all_pos = jnp.all(var_lbs > 0)
                    cv_sig, cc_sig = relax_signomial_multi(xs, var_lbs, var_ubs, jnp.array(_exps))
                    # Fallback: bilinear
                    cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                    cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                    mid_l = 0.5 * (cv_l + cc_l)
                    mid_r = 0.5 * (cv_r + cc_r)
                    cv_bl, cc_bl = relax_bilinear(mid_l, mid_r, cv_l, cc_l, cv_r, cc_r)
                    return (
                        jnp.where(all_pos, cv_sig, cv_bl),
                        jnp.where(all_pos, cc_sig, cc_bl),
                    )

                return fn

            # General bilinear: use cv/cc as bounds for McCormick envelopes,
            # but also try to propagate tighter variable-level bounds.

            # Learned relaxation for bilinear
            if mode == "learned" and learned_registry is not None:
                lr_bilinear = learned_registry.get("bilinear")
                if lr_bilinear is not None:

                    def fn(x_cv, x_cc, lb, ub, _lr=lr_bilinear):
                        cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                        cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                        mid_l = 0.5 * (cv_l + cc_l)
                        mid_r = 0.5 * (cv_r + cc_r)
                        true_val = mid_l * mid_r
                        xy = jnp.stack([mid_l, mid_r])
                        xy_lb = jnp.stack([cv_l, cv_r])
                        xy_ub = jnp.stack([cc_l, cc_r])
                        return _lr(xy, xy_lb, xy_ub, true_val)

                    return fn

            if partitions > 0:
                _k = partitions

                def fn(x_cv, x_cc, lb, ub, _pw_k=_k):
                    cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                    cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                    mid_l = 0.5 * (cv_l + cc_l)
                    mid_r = 0.5 * (cv_r + cc_r)
                    return piecewise_mccormick_bilinear(
                        mid_l, mid_r, cv_l, cc_l, cv_r, cc_r, k=_pw_k
                    )

                return fn

            def fn(x_cv, x_cc, lb, ub):
                cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                # Use the midpoint of cv/cc as the "x" value for bilinear
                mid_l = 0.5 * (cv_l + cc_l)
                mid_r = 0.5 * (cv_r + cc_r)
                return relax_bilinear(mid_l, mid_r, cv_l, cc_l, cv_r, cc_r)

            return fn

        if op == "/":
            if _is_constant_expr(expr.right):
                c = _get_constant_value(expr.right)
                inv_c = 1.0 / c

                def fn(x_cv, x_cc, lb, ub, _inv_c=inv_c):
                    cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                    pos = _inv_c >= 0
                    new_cv = jnp.where(pos, _inv_c * cv_l, _inv_c * cc_l)
                    new_cc = jnp.where(pos, _inv_c * cc_l, _inv_c * cv_l)
                    return new_cv, new_cc

                return fn

            # Monod / saturating term x/(K+x) over a single scalar variable with a
            # nonneg declared domain. Routes to the dedicated concave envelope
            # (tighter than the generic division relaxation); falls through to the
            # generic path when the optional [sympy] domain pack is absent.
            monod = _try_extract_monod(expr, model)
            if monod is not None:
                _mo_off, _mo_k = monod
                try:
                    from discopt._jax.symbolic.domains.chemeng import saturating_relax
                except ImportError:
                    saturating_relax = None  # type: ignore[assignment]
                if saturating_relax is not None:
                    _mo_fn = saturating_relax(_mo_k)

                    def fn(x_cv, x_cc, lb, ub, _off=_mo_off, _wr=_mo_fn):
                        return _wr(x_cv[_off], lb[_off], ub[_off])

                    return fn

            def fn(x_cv, x_cc, lb, ub):
                cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                mid_l = 0.5 * (cv_l + cc_l)
                mid_r = 0.5 * (cv_r + cc_r)
                return relax_div(mid_l, mid_r, cv_l, cc_l, cv_r, cc_r)

            return fn

        if op == "**":
            # Integer power: use tight envelope when base is a plain variable
            if _is_constant_expr(expr.right):
                n_val = expr.right.value
                n_int = int(n_val)
                if np.isclose(float(n_val), float(n_int)):
                    # Check if the base is a plain variable for tight bounds
                    if isinstance(expr.left, (Variable, IndexExpression)):
                        from discopt._jax.envelopes import relax_power_int

                        if isinstance(expr.left, Variable) and expr.left.size == 1:
                            vi = _compute_var_offset(expr.left, model)

                            def fn(x_cv, x_cc, lb, ub, _n=n_int, _vi=vi):
                                return relax_power_int(x_cv[_vi], lb[_vi], ub[_vi], _n)

                            return fn
                        elif isinstance(expr.left, IndexExpression):
                            if isinstance(expr.left.base, Variable):
                                base_off = _compute_var_offset(expr.left.base, model)
                                idx = expr.left.index
                                flat_idx = (
                                    base_off + idx
                                    if isinstance(idx, int)
                                    else base_off + idx[0]
                                    if isinstance(idx, tuple) and len(idx) == 1
                                    else None
                                )
                                if flat_idx is not None:

                                    def fn(x_cv, x_cc, lb, ub, _n=n_int, _fi=flat_idx):
                                        return relax_power_int(x_cv[_fi], lb[_fi], ub[_fi], _n)

                                    return fn

                    # Fallback: compositional McCormick
                    def fn(x_cv, x_cc, lb, ub, _n=n_int):
                        cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                        mid = 0.5 * (cv_l + cc_l)
                        return relax_pow(mid, cv_l, cc_l, _n)

                    return fn

            # Constant fractional exponent with tight envelope
            if _is_constant_expr(expr.right):
                alpha = float(expr.right.value)
                if 0.0 < alpha < 1.0:
                    # x^alpha is CONCAVE for 0 < alpha < 1, x > 0.
                    # Concave envelope = x^alpha (the function itself).
                    # Convex envelope = secant: f(a) + (f(b)-f(a))/(b-a)*(x-a).
                    _alpha = alpha

                    def fn(x_cv, x_cc, lb, ub, _a=_alpha):
                        cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                        # Clamp to positive to avoid NaN in power
                        cv_l = jnp.maximum(cv_l, 1e-30)
                        cc_l = jnp.maximum(cc_l, 1e-30)
                        # Concave overestimator: f(x) = x^alpha (tight)
                        cc_out = cc_l**_a
                        # Convex underestimator: secant line between bounds
                        fa = cv_l**_a
                        fb = cc_l**_a
                        slope = (fb - fa) / jnp.maximum(cc_l - cv_l, 1e-30)
                        mid = 0.5 * (cv_l + cc_l)
                        cv_out = fa + slope * (mid - cv_l)
                        return cv_out, cc_out

                    return fn
                elif alpha > 1.0 and not np.isclose(alpha, round(alpha)):
                    # x^alpha is CONVEX for alpha > 1, x > 0.
                    # Convex envelope = x^alpha (the function itself).
                    # Concave envelope = secant line.
                    _alpha = alpha

                    def fn(x_cv, x_cc, lb, ub, _a=_alpha):
                        cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                        cv_l = jnp.maximum(cv_l, 1e-30)
                        cc_l = jnp.maximum(cc_l, 1e-30)
                        # Convex underestimator: f(x) = x^alpha (tight)
                        cv_out = cv_l**_a
                        # Concave overestimator: secant line
                        fa = cv_l**_a
                        fb = cc_l**_a
                        slope = (fb - fa) / jnp.maximum(cc_l - cv_l, 1e-30)
                        mid = 0.5 * (cv_l + cc_l)
                        cc_out = fa + slope * (mid - cv_l)
                        return cv_out, cc_out

                    return fn

            # General case: x^y = exp(y * log(x))
            # Use piecewise operations when partitions > 0.
            if partitions > 0:
                _k = partitions

                def fn(x_cv, x_cc, lb, ub, _pw_k=_k):
                    cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                    cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                    mid_l = 0.5 * (cv_l + cc_l)
                    mid_r = 0.5 * (cv_r + cc_r)
                    log_cv, log_cc = piecewise_relax_log(mid_l, cv_l, cc_l, k=_pw_k)
                    mid_log = 0.5 * (log_cv + log_cc)
                    prod_cv, prod_cc = piecewise_mccormick_bilinear(
                        mid_r, mid_log, cv_r, cc_r, log_cv, log_cc, k=_pw_k
                    )
                    mid_prod = 0.5 * (prod_cv + prod_cc)
                    return piecewise_relax_exp(mid_prod, prod_cv, prod_cc, k=_pw_k)

                return fn

            def fn(x_cv, x_cc, lb, ub):
                cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                # Use midpoints for evaluation
                mid_l = 0.5 * (cv_l + cc_l)
                mid_r = 0.5 * (cv_r + cc_r)
                # Relaxation of log(x)
                log_cv, log_cc = relax_log(mid_l, cv_l, cc_l)
                # Relaxation of y * log(x) via bilinear
                mid_log = 0.5 * (log_cv + log_cc)
                prod_cv, prod_cc = relax_bilinear(mid_r, mid_log, cv_r, cc_r, log_cv, log_cc)
                # Relaxation of exp(product)
                mid_prod = 0.5 * (prod_cv + prod_cc)
                return relax_exp(mid_prod, prod_cv, prod_cc)

            return fn

        raise ValueError(f"Unknown binary operator: {op!r}")

    if isinstance(expr, UnaryOp):
        operand_fn = _compile_relax_node(
            expr.operand, model, partitions, mode, learned_registry, arithmetic
        )
        op = expr.op

        if op == "neg":

            def fn(x_cv, x_cc, lb, ub):
                cv_child, cc_child = operand_fn(x_cv, x_cc, lb, ub)
                return relax_neg(cv_child, cc_child)

            return fn

        if op == "abs":

            def fn(x_cv, x_cc, lb, ub):
                cv_child, cc_child = operand_fn(x_cv, x_cc, lb, ub)
                mid = 0.5 * (cv_child + cc_child)
                return relax_abs(mid, cv_child, cc_child)

            return fn

        raise ValueError(f"Unknown unary operator: {op!r}")

    if isinstance(expr, FunctionCall):
        arg_fns = [
            _compile_relax_node(a, model, partitions, mode, learned_registry, arithmetic)
            for a in expr.args
        ]
        name = expr.func_name

        # Learned relaxation dispatch: use ICNN-based relaxations when available
        _learned_univariate_ops = {"exp", "log", "sqrt", "sin"}
        if mode == "learned" and learned_registry is not None and name in _learned_univariate_ops:
            lr_model = learned_registry.get(name)
            if lr_model is not None:
                a_fn = arg_fns[0]
                _true_fns = {
                    "exp": jnp.exp,
                    "log": jnp.log,
                    "sqrt": jnp.sqrt,
                    "sin": jnp.sin,
                }
                _tfn = _true_fns[name]

                def fn(x_cv, x_cc, lb, ub, _lr=lr_model, _af=a_fn, _tf=_tfn):
                    cv_child, cc_child = _af(x_cv, x_cc, lb, ub)
                    mid = 0.5 * (cv_child + cc_child)
                    true_val = _tf(mid)
                    return _lr(mid, cv_child, cc_child, true_val)

                return fn

        # Tight sin/cos dispatch when argument is a plain variable
        if name in ("sin", "cos") and len(expr.args) == 1:
            arg = expr.args[0]
            if isinstance(arg, (Variable, IndexExpression)):
                from discopt._jax.envelopes import relax_cos_tight, relax_sin_tight

                _tight_fn = relax_sin_tight if name == "sin" else relax_cos_tight

                if isinstance(arg, Variable) and arg.size == 1:
                    vi = _compute_var_offset(arg, model)

                    def fn(x_cv, x_cc, lb, ub, _vi=vi, _tf=_tight_fn):
                        return _tf(x_cv[_vi], lb[_vi], ub[_vi])

                    return fn
                elif isinstance(arg, IndexExpression) and isinstance(arg.base, Variable):
                    base_off = _compute_var_offset(arg.base, model)
                    idx = arg.index
                    flat_idx = (
                        base_off + idx
                        if isinstance(idx, int)
                        else base_off + idx[0]
                        if isinstance(idx, tuple) and len(idx) == 1
                        else None
                    )
                    if flat_idx is not None:

                        def fn(x_cv, x_cc, lb, ub, _fi=flat_idx, _tf=_tight_fn):
                            return _tf(x_cv[_fi], lb[_fi], ub[_fi])

                        return fn

        # Arrhenius rate term exp(-c/T) over a single positive-domain scalar var.
        # Routes the whole exp(-c/T) node to the dedicated single-inflection
        # envelope (tighter than composing relax_exp with the relaxed reciprocal);
        # falls through to the generic exp path when the optional [sympy] domain
        # pack is absent.
        if name == "exp" and len(expr.args) == 1:
            arr = _try_extract_arrhenius(expr, model)
            if arr is not None:
                _ar_off, _ar_c = arr
                try:
                    from discopt._jax.symbolic.domains.chemeng import arrhenius_relax
                except ImportError:
                    arrhenius_relax = None  # type: ignore[assignment]
                if arrhenius_relax is not None:
                    _ar_fn = arrhenius_relax(_ar_c)

                    def fn(x_cv, x_cc, lb, ub, _off=_ar_off, _wr=_ar_fn):
                        return _wr(x_cv[_off], lb[_off], ub[_off])

                    return fn

        # Piecewise-capable univariate operations
        _piecewise_relax = {
            "exp": piecewise_relax_exp,
            "log": piecewise_relax_log,
            "sqrt": piecewise_relax_sqrt,
            "sin": piecewise_relax_sin,
            "cos": piecewise_relax_cos,
        }

        _univariate_relax = {
            "exp": relax_exp,
            "log": relax_log,
            "log2": relax_log2,
            "log10": relax_log10,
            "sqrt": relax_sqrt,
            "sin": relax_sin,
            "cos": relax_cos,
            "tan": relax_tan,
            "atan": relax_atan,
            "sinh": relax_sinh,
            "cosh": relax_cosh,
            "asin": relax_asin,
            "acos": relax_acos,
            "tanh": relax_tanh,
            "sigmoid": relax_sigmoid,
            "softplus": relax_softplus,
            "abs": relax_abs,
            "entropy": relax_entropy,
        }

        if partitions > 0 and name in _piecewise_relax:
            pw_fn = _piecewise_relax[name]
            a_fn = arg_fns[0]
            _k = partitions

            def fn(x_cv, x_cc, lb, ub, _pw_fn=pw_fn, _a_fn=a_fn, _pw_k=_k):
                cv_a, cc_a = _a_fn(x_cv, x_cc, lb, ub)
                mid = 0.5 * (cv_a + cc_a)
                return _pw_fn(mid, cv_a, cc_a, k=_pw_k)

            return fn

        if name in _univariate_relax:
            a_fn = arg_fns[0]

            # Chebyshev / Taylor / ellipsoidal dispatch (M2 / M3 of issue
            # #51; M7 of issue #81): build a pre-compiled outer-approximation
            # shim if (a) the operator is supported, (b) we can infer a finite
            # static box for the inner expression. Otherwise fall through to
            # the existing McCormick path so the choice is non-blocking.
            if arithmetic in ("chebyshev", "taylor", "ellipsoidal"):
                from discopt._jax import oa_relax

                if oa_relax.is_supported(name):
                    box = oa_relax.static_box_for_arg(expr.args[0], model)
                    if box is not None:
                        try:
                            safe_box = oa_relax._safe_box_for_op(name, box[0], box[1])
                            oa_fn = oa_relax.make_oa_relax(name, safe_box, arithmetic)

                            def fn(x_cv, x_cc, lb, ub, _oa_fn=oa_fn, _a_fn=a_fn):
                                cv_a, cc_a = _a_fn(x_cv, x_cc, lb, ub)
                                mid = 0.5 * (cv_a + cc_a)
                                return _oa_fn(mid, cv_a, cc_a)

                            return fn
                        except Exception:
                            # Fall through to McCormick on any OA failure.
                            pass

            # Prefer the TM2014 (multivariate McCormick) composition rule when
            # available — it is sound at non-degenerate inner intervals, where
            # the legacy midpoint composition can violate the bounds. See
            # python/discopt/_jax/multivariate_mccormick.py and issue #51 (M1).
            tm_rule = get_composition_rule(name)
            if tm_rule is not None:

                def fn(x_cv, x_cc, lb, ub, _tm=tm_rule, _a_fn=a_fn):
                    cv_a, cc_a = _a_fn(x_cv, x_cc, lb, ub)
                    # Use [cv_a, cc_a] as the (tightest known pointwise) bounds
                    # on the inner expression g over which we build f's envelope.
                    return _tm(cv_a, cc_a, cv_a, cc_a)

                return fn

            relax_fn = _univariate_relax[name]

            def fn(x_cv, x_cc, lb, ub, _relax_fn=relax_fn, _a_fn=a_fn):
                cv_a, cc_a = _a_fn(x_cv, x_cc, lb, ub)
                mid = 0.5 * (cv_a + cc_a)
                return _relax_fn(mid, cv_a, cc_a)

            return fn

        # Envelope-based relaxations (use actual variable bounds when possible)
        _envelope_relax = {"asinh", "acosh", "atanh", "erf", "log1p"}
        if name in _envelope_relax and len(expr.args) == 1:
            arg = expr.args[0]
            if isinstance(arg, (Variable, IndexExpression)):
                from discopt._jax.envelopes import (
                    relax_acosh,
                    relax_asinh,
                    relax_atanh,
                    relax_erf,
                    relax_log1p,
                )

                _env_fns = {
                    "asinh": relax_asinh,
                    "acosh": relax_acosh,
                    "atanh": relax_atanh,
                    "erf": relax_erf,
                    "log1p": relax_log1p,
                }
                _env_fn = _env_fns[name]

                if isinstance(arg, Variable) and arg.size == 1:
                    vi = _compute_var_offset(arg, model)

                    def fn(x_cv, x_cc, lb, ub, _vi=vi, _ef=_env_fn):
                        return _ef(x_cv[_vi], lb[_vi], ub[_vi])

                    return fn
                elif isinstance(arg, IndexExpression) and isinstance(arg.base, Variable):
                    base_off = _compute_var_offset(arg.base, model)
                    idx = arg.index
                    flat_idx = (
                        base_off + idx
                        if isinstance(idx, int)
                        else base_off + idx[0]
                        if isinstance(idx, tuple) and len(idx) == 1
                        else None
                    )
                    if flat_idx is not None:

                        def fn(x_cv, x_cc, lb, ub, _fi=flat_idx, _ef=_env_fn):
                            return _ef(x_cv[_fi], lb[_fi], ub[_fi])

                        return fn

            # Fallback: use envelopes with propagated bounds
            from discopt._jax.envelopes import (
                relax_acosh,
                relax_asinh,
                relax_atanh,
                relax_erf,
                relax_log1p,
            )

            _env_fns_fb = {
                "asinh": relax_asinh,
                "acosh": relax_acosh,
                "atanh": relax_atanh,
                "erf": relax_erf,
                "log1p": relax_log1p,
            }
            _env_fn_fb = _env_fns_fb[name]
            a_fn = arg_fns[0]

            def fn(x_cv, x_cc, lb, ub, _ef=_env_fn_fb, _a_fn=a_fn):
                cv_a, cc_a = _a_fn(x_cv, x_cc, lb, ub)
                mid = 0.5 * (cv_a + cc_a)
                return _ef(mid, cv_a, cc_a)

            return fn

        if name == "sign":
            a_fn = arg_fns[0]

            def fn(x_cv, x_cc, lb, ub):
                cv_a, cc_a = a_fn(x_cv, x_cc, lb, ub)
                mid = 0.5 * (cv_a + cc_a)
                return relax_sign(mid, cv_a, cc_a)

            return fn

        if name == "min":
            a_fn, b_fn = arg_fns[0], arg_fns[1]

            def fn(x_cv, x_cc, lb, ub):
                cv_a, cc_a = a_fn(x_cv, x_cc, lb, ub)
                cv_b, cc_b = b_fn(x_cv, x_cc, lb, ub)
                from discopt._jax.mccormick import relax_min

                mid_a = 0.5 * (cv_a + cc_a)
                mid_b = 0.5 * (cv_b + cc_b)
                return relax_min(mid_a, mid_b, cv_a, cc_a, cv_b, cc_b)

            return fn

        if name == "max":
            a_fn, b_fn = arg_fns[0], arg_fns[1]

            def fn(x_cv, x_cc, lb, ub):
                cv_a, cc_a = a_fn(x_cv, x_cc, lb, ub)
                cv_b, cc_b = b_fn(x_cv, x_cc, lb, ub)
                from discopt._jax.mccormick import relax_max

                mid_a = 0.5 * (cv_a + cc_a)
                mid_b = 0.5 * (cv_b + cc_b)
                return relax_max(mid_a, mid_b, cv_a, cc_a, cv_b, cc_b)

            return fn

        if name == "prod":
            # Multilinear product prod_i x_i over an array argument, relaxed by
            # the recursive McCormick fold of envelopes._nested_trilinear_one_order
            # generalized to n factors: maintain the accumulated product's value
            # envelope (w_cv, w_cc) AND its exact interval [w_lb, w_ub] (via
            # corner products), folding one factor at a time through the bilinear
            # envelope. A valid relaxation for every factor sign.
            #
            # NOTE: this is not the literal convex hull of the multilinear
            # monomial (Rikun 1997 / Meyer & Floudas 2004). The exact hull is
            # given by RLT bound-factor products over *lifted* bilinear variables,
            # which this compositional value-evaluator does not carry. Merging
            # association orderings does NOT help here: at the midpoint/box-center
            # linearization point this evaluator uses, recursive McCormick is
            # order-invariant (verified: relax_trilinear_exact's three orderings
            # coincide at the midpoint). Tightening to the true hull needs the
            # lifted-variable LP path and remains future work.
            a_fn = arg_fns[0]
            n = getattr(expr.args[0], "size", None)
            if n is None or n < 1:
                raise ValueError(
                    "prod relaxation requires a fixed-size array argument; "
                    f"got {type(expr.args[0]).__name__}"
                )

            def fn(x_cv, x_cc, lb, ub, _a_fn=a_fn, _n=int(n)):
                cv_a, cc_a = _a_fn(x_cv, x_cc, lb, ub)
                cv_a = jnp.reshape(cv_a, (-1,))
                cc_a = jnp.reshape(cc_a, (-1,))
                if _n == 1:
                    return cv_a[0], cc_a[0]
                # Seed the fold with the first two factors' bilinear envelope.
                w_cv, w_cc = relax_bilinear(
                    0.5 * (cv_a[0] + cc_a[0]),
                    0.5 * (cv_a[1] + cc_a[1]),
                    cv_a[0],
                    cc_a[0],
                    cv_a[1],
                    cc_a[1],
                )
                corners = jnp.stack(
                    [cv_a[0] * cv_a[1], cv_a[0] * cc_a[1], cc_a[0] * cv_a[1], cc_a[0] * cc_a[1]]
                )
                w_lb, w_ub = jnp.min(corners), jnp.max(corners)
                for k in range(2, _n):
                    c_lb, c_ub = cv_a[k], cc_a[k]
                    c_mid = 0.5 * (c_lb + c_ub)
                    cv1, cc1 = relax_bilinear(w_cv, c_mid, w_lb, w_ub, c_lb, c_ub)
                    cv2, cc2 = relax_bilinear(w_cc, c_mid, w_lb, w_ub, c_lb, c_ub)
                    new_cv = jnp.minimum(cv1, cv2)
                    new_cc = jnp.maximum(cc1, cc2)
                    corners = jnp.stack([w_lb * c_lb, w_lb * c_ub, w_ub * c_lb, w_ub * c_ub])
                    w_lb, w_ub = jnp.min(corners), jnp.max(corners)
                    w_cv, w_cc = new_cv, new_cc
                return w_cv, w_cc

            return fn

        if name.startswith("norm"):
            # p-norm ||x||_p over an array argument (p >= 1, e.g. norm1/norm2).
            # Valid bounds for every p >= 1 from norm equivalence:
            #   ||x||_inf <= ||x||_p <= ||x||_1
            # giving a convex underestimator max_i |x_i| and a concave
            # overestimator sum_i |x_i|, each built from the per-component
            # |.| envelope. A scalar argument reduces to |x|.
            a_fn = arg_fns[0]

            def fn(x_cv, x_cc, lb, ub, _a_fn=a_fn):
                cv_a, cc_a = _a_fn(x_cv, x_cc, lb, ub)
                cv_a = jnp.reshape(cv_a, (-1,))
                cc_a = jnp.reshape(cc_a, (-1,))
                mid = 0.5 * (cv_a + cc_a)
                cv_abs, cc_abs = relax_abs(mid, cv_a, cc_a)
                # ||x||_p >= ||x||_inf = max_i |x_i|  (convex underestimator)
                cv = jnp.max(cv_abs)
                # ||x||_p <= ||x||_1 = sum_i |x_i|    (concave overestimator)
                cc = jnp.sum(cc_abs)
                return cv, cc

            return fn

        raise ValueError(f"Unknown function: {name!r}")

    if isinstance(expr, IndexExpression):
        base_fn = _compile_relax_node(
            expr.base, model, partitions, mode, learned_registry, arithmetic
        )
        idx = expr.index

        def fn(x_cv, x_cc, lb, ub, _idx=idx):
            cv_base, cc_base = base_fn(x_cv, x_cc, lb, ub)
            return cv_base[_idx], cc_base[_idx]

        return fn

    if isinstance(expr, SumExpression):
        operand_fn = _compile_relax_node(
            expr.operand, model, partitions, mode, learned_registry, arithmetic
        )
        axis = expr.axis

        def fn(x_cv, x_cc, lb, ub, _axis=axis):
            cv_op, cc_op = operand_fn(x_cv, x_cc, lb, ub)
            return jnp.sum(cv_op, axis=_axis), jnp.sum(cc_op, axis=_axis)

        return fn

    if isinstance(expr, SumOverExpression):
        term_fns = [
            _compile_relax_node(t, model, partitions, mode, learned_registry, arithmetic)
            for t in expr.terms
        ]

        def fn(x_cv, x_cc, lb, ub):
            cv_acc, cc_acc = term_fns[0](x_cv, x_cc, lb, ub)
            for t_fn in term_fns[1:]:
                cv_t, cc_t = t_fn(x_cv, x_cc, lb, ub)
                cv_acc = cv_acc + cv_t
                cc_acc = cc_acc + cc_t
            return cv_acc, cc_acc

        return fn

    if isinstance(expr, CustomCall):
        raise NotImplementedError(
            f"Cannot build a McCormick relaxation for the opaque AD-only user "
            f"function {expr.name!r} (dm.custom). Global / spatial branch-and-bound "
            f"needs rigorous convex/concave relaxations, which an opaque callable "
            f"cannot provide. Solve on the local NLP path (pure-continuous models "
            f"route there automatically), or rebuild the function from dm.* "
            f"primitives and use dm.udf."
        )

    raise TypeError(f"Unhandled expression type: {type(expr).__name__}")


def compile_relaxation(
    expr: Expression,
    model: Model,
    partitions: int = 0,
    mode: str = "standard",
    learned_registry: Optional["LearnedRelaxationRegistry"] = None,
    arithmetic: str = "mccormick",
) -> Callable:
    """
    Compile an Expression into a McCormick relaxation function.

    Args:
        expr: Expression to relax
        model: Model containing variable definitions
        partitions: If > 0, use piecewise McCormick relaxations with this
            many partitions for supported operations (bilinear, exp, log,
            sqrt, sin, cos). If 0 (default), use standard McCormick.
        mode: Relaxation mode — ``"standard"`` (default), ``"piecewise"``,
            or ``"learned"`` (ICNN-based with McCormick fallback).
        learned_registry: Registry of trained learned relaxations.
            Required when ``mode="learned"``.

    Returns:
        A function f(x_cv, x_cc, lb, ub) -> (cv, cc) where:
          - x_cv: convex relaxation values for variables (1D flat array)
          - x_cc: concave relaxation values for variables (1D flat array)
          - lb: lower bounds for all variables (1D flat array)
          - ub: upper bounds for all variables (1D flat array)
          - cv: convex underestimator of expr
          - cc: concave overestimator of expr

        The function is compatible with jax.jit and jax.vmap.

        When ``arithmetic="alphabb"``, the whole expression is relaxed with a
        rigorous (sound interval-Hessian) alphaBB underestimator/overestimator
        instead of compositional McCormick. For the default McCormick path, any
        node with no McCormick rule triggers an automatic alphaBB fallback so a
        valid relaxation is always produced when the interval Hessian is finite.
    """
    if arithmetic == "alphabb":
        from discopt._jax.alphabb import compile_alphabb_relaxation

        return compile_alphabb_relaxation(expr, model)

    try:
        return _compile_relax_node(expr, model, partitions, mode, learned_registry, arithmetic)
    except (ValueError, NotImplementedError) as exc:
        # No McCormick rule for some node — fall back to a rigorous alphaBB
        # relaxation of the whole expression. This keeps the solver supplied
        # with a valid relaxation (gap D). If alphaBB also abstains (unbounded
        # interval Hessian) re-raise the original McCormick error.
        try:
            from discopt._jax.alphabb import compile_alphabb_relaxation

            return compile_alphabb_relaxation(expr, model)
        except (ValueError, NotImplementedError):
            raise exc


def compile_objective_relaxation(
    model: Model,
    partitions: int = 0,
    mode: str = "standard",
    learned_registry: Optional["LearnedRelaxationRegistry"] = None,
    arithmetic: str = "mccormick",
) -> Callable:
    """Compile relaxation of the model's objective."""
    if model._objective is None:
        raise ValueError("Model has no objective set.")
    return compile_relaxation(
        model._objective.expression,
        model,
        partitions,
        mode,
        learned_registry,
        arithmetic,
    )


def compile_constraint_relaxation(
    constraint: Constraint,
    model: Model,
    partitions: int = 0,
    mode: str = "standard",
    learned_registry: Optional["LearnedRelaxationRegistry"] = None,
    arithmetic: str = "mccormick",
) -> Callable:
    """Compile relaxation of a constraint body."""
    return compile_relaxation(
        constraint.body, model, partitions, mode, learned_registry, arithmetic
    )
