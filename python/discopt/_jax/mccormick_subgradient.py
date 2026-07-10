"""Reduced-space McCormick relaxation + Kelley-cut LP bound, on the MCBox type.

MAiNGO-parity plan P0.3: this reimplements the #572 reduced-space evaluator on top of
:mod:`discopt._jax.mcbox`. Where the original walked the model AST into hand-rolled
``(cv, cc)`` closures (affine / QP scope, ``jax.grad`` subgradients), this interprets
the AST into :class:`~discopt._jax.mcbox.MCBox` operations — so the relaxation and its
subgradients propagate by rule, and the scope widens to everything MCBox supports
(general products, transcendental composition), still sound-or-refuse.

Public API is unchanged from #572: :class:`UnsupportedRelaxation`,
:func:`build_reduced_relaxation`, :func:`reduced_mccormick_lp_bound`,
:class:`ReducedRelaxation`, :class:`ReducedBound`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from discopt._jax import mcbox as _mcbox
from discopt._jax.mcbox import UnsupportedMcboxOp, mcbox_leaves
from discopt._jax.relaxation_compiler import _resolve_scalar_var_offset
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    FunctionCall,
    ObjectiveSense,
    Parameter,
    UnaryOp,
    Variable,
)


class UnsupportedRelaxation(Exception):
    """Model is outside the sound reduced-space scope — caller falls back (α-BB/lifted)."""


def _scalar_const(expr) -> float:
    v = np.asarray(expr.value, dtype=float)
    if v.size != 1:
        raise UnsupportedRelaxation("non-scalar constant/parameter")
    return float(v.reshape(()))


def _to_mcbox(expr, leaves, model):
    """Interpret a scalar model expression into an MCBox over the current leaves."""
    if isinstance(expr, (Constant, Parameter)):
        return _mcbox._const(_scalar_const(expr), len(leaves))

    off = _resolve_scalar_var_offset(expr, model)
    if off is not None:
        return leaves[off]
    if isinstance(expr, Variable):
        raise UnsupportedRelaxation("non-scalar variable leaf in scalar expression")

    if isinstance(expr, UnaryOp):
        a = _to_mcbox(expr.operand, leaves, model)
        if expr.op == "neg":
            return -a
        raise UnsupportedRelaxation(f"unary op '{expr.op}'")

    if isinstance(expr, BinaryOp):
        op = expr.op
        if op == "+":
            return _to_mcbox(expr.left, leaves, model) + _to_mcbox(expr.right, leaves, model)
        if op == "-":
            return _to_mcbox(expr.left, leaves, model) - _to_mcbox(expr.right, leaves, model)
        if op == "*":
            return _to_mcbox(expr.left, leaves, model) * _to_mcbox(expr.right, leaves, model)
        if op == "/":
            if isinstance(expr.right, (Constant, Parameter)):
                c = _scalar_const(expr.right)
                if c == 0.0:
                    raise UnsupportedRelaxation("division by zero constant")
                return _to_mcbox(expr.left, leaves, model) * (1.0 / c)
            # variable division via the sign-definite reciprocal (P1.2)
            return _to_mcbox(expr.left, leaves, model) / _to_mcbox(expr.right, leaves, model)
        if op == "**":
            if not isinstance(expr.right, (Constant, Parameter)):
                raise UnsupportedRelaxation("non-constant exponent")
            p = _scalar_const(expr.right)
            try:
                return _to_mcbox(expr.left, leaves, model) ** p
            except UnsupportedMcboxOp as e:
                raise UnsupportedRelaxation(str(e)) from e
        raise UnsupportedRelaxation(f"binary op '{op}'")

    if isinstance(expr, FunctionCall):
        if len(expr.args) != 1:
            raise UnsupportedRelaxation(f"multi-arg function '{expr.func_name}'")
        a = _to_mcbox(expr.args[0], leaves, model)
        fn = getattr(_mcbox, expr.func_name, None)
        if fn is None or not callable(fn):
            raise UnsupportedRelaxation(f"function '{expr.func_name}'")
        try:
            return fn(a)
        except UnsupportedMcboxOp as e:
            raise UnsupportedRelaxation(str(e)) from e

    raise UnsupportedRelaxation(f"node type {type(expr).__name__}")


@dataclass
class ReducedRelaxation:
    obj_under: Callable  # convex underestimator of the objective (minimize form)
    con_feas: list  # convex h_i(x) with h_i<=0 the relaxed feasibility
    negate: bool
    n: int


def build_reduced_relaxation(model, node_lb, node_ub) -> ReducedRelaxation:
    """Build the sound reduced-space McCormick relaxation (MCBox), or raise.

    Raises :class:`UnsupportedRelaxation` if any part is outside MCBox scope. The
    objective/constraint closures interpret the AST into MCBox at each point (jit-able);
    a concrete trial build at the box midpoint surfaces unsupported structure eagerly.
    """
    lb = np.asarray(node_lb, dtype=float)
    ub = np.asarray(node_ub, dtype=float)
    n = int(lb.size)
    negate = model._objective.sense == ObjectiveSense.MAXIMIZE
    obj_expr = model._objective.expression

    def obj_under(x):
        z = _to_mcbox(obj_expr, mcbox_leaves(x, lb, ub), model)
        return -z.cc if negate else z.cv

    con_feas = []
    for c in model._constraints:
        if not isinstance(c, Constraint):
            raise UnsupportedRelaxation(f"non-Constraint {type(c).__name__}")
        body, sense = c.body, c.sense

        if sense in ("<=", "=="):
            con_feas.append(lambda x, b=body: _to_mcbox(b, mcbox_leaves(x, lb, ub), model).cv)
        if sense in (">=", "=="):
            con_feas.append(lambda x, b=body: -_to_mcbox(b, mcbox_leaves(x, lb, ub), model).cc)

    # eager buildability check (concrete midpoint) so the caller can fall back
    mid = jnp.asarray(0.5 * (lb + ub))
    _ = float(obj_under(mid))
    for h in con_feas:
        _ = float(h(mid))
    return ReducedRelaxation(obj_under=obj_under, con_feas=con_feas, negate=negate, n=n)


def _build_jit_evaluator(model, lb, ub):
    """A single jitted fn ``x -> (u, gu, h, gh)`` for the reduced relaxation, using
    MCBox's **explicit** rule-based subgradients (not ``jax.grad`` over the whole
    construction — ~4 orders of magnitude faster, P2.1). ``u``/``gu`` are the objective
    underestimator (minimize form) value/subgradient; ``h`` (m,) / ``gh`` (m, n) are the
    per-constraint feasibility values/subgradients (``h_i <= 0`` relaxed). Raises
    :class:`UnsupportedRelaxation` on first call (compile) if out of MCBox scope."""
    lbj, ubj = jnp.asarray(lb), jnp.asarray(ub)
    n = int(lb.size)
    negate = model._objective.sense == ObjectiveSense.MAXIMIZE
    obj_expr = model._objective.expression
    cons = []
    for c in model._constraints:
        if not isinstance(c, Constraint):
            raise UnsupportedRelaxation(f"non-Constraint {type(c).__name__}")
        cons.append((c.body, c.sense))

    def ev(x):
        leaves = mcbox_leaves(x, lbj, ubj)
        zo = _to_mcbox(obj_expr, leaves, model)
        u = -zo.cc if negate else zo.cv
        gu = -zo.sub_cc if negate else zo.sub_cv
        hs, ghs = [], []
        for body, sense in cons:
            zb = _to_mcbox(body, leaves, model)
            if sense in ("<=", "=="):
                hs.append(zb.cv)
                ghs.append(zb.sub_cv)
            if sense in (">=", "=="):
                hs.append(-zb.cc)
                ghs.append(-zb.sub_cc)
        h = jnp.stack(hs) if hs else jnp.zeros(0)
        gh = jnp.stack(ghs) if ghs else jnp.zeros((0, n))
        return u, gu, h, gh

    return jax.jit(ev), negate, n


@dataclass
class ReducedBound:
    bound: float
    status: str  # optimal | infeasible | unbounded | unsupported
    rounds: int
    history: list


def reduced_mccormick_lp_bound(
    model, node_lb, node_ub, *, max_rounds: int = 40, tol: float = 1e-7
) -> ReducedBound:
    """MAiNGO-style reduced-space lower bound: Kelley cutting planes on the MCBox
    relaxation, solved as an LP over the original variables (no auxiliary columns).
    The per-round value+subgradient evaluation is a single jitted MCBox pass with
    explicit subgradients (P2.1). scipy/HiGHS scaffold for the inner LP; P2.2 wires the
    in-house simplex."""
    import scipy.optimize as sopt

    lb = np.asarray(node_lb, dtype=float)
    ub = np.asarray(node_ub, dtype=float)
    try:
        evj, negate, n = _build_jit_evaluator(model, lb, ub)
        _u, _gu, _h, _gh = evj(jnp.asarray(0.5 * (lb + ub)))  # compile + eager scope check
    except UnsupportedRelaxation:
        return ReducedBound(bound=-np.inf, status="unsupported", rounds=0, history=[])

    c = np.zeros(n + 1)
    c[n] = 1.0
    bounds = [(float(lb[i]), float(ub[i])) for i in range(n)] + [(None, None)]
    A: list = []
    rhs: list = []
    xk = 0.5 * (lb + ub)
    prev, bound = -np.inf, -np.inf
    history: list = []

    r = 0
    for r in range(1, max_rounds + 1):
        u_j, gu_j, h_j, gh_j = evj(jnp.asarray(xk))
        u0 = float(u_j)
        g = np.asarray(gu_j, dtype=float)
        if np.all(np.isfinite(g)) and np.isfinite(u0):
            row = np.zeros(n + 1)
            row[:n] = g
            row[n] = -1.0
            A.append(row)
            rhs.append(float(g @ xk - u0))
        h_np = np.asarray(h_j, dtype=float)
        gh_np = np.asarray(gh_j, dtype=float)
        for h0, gg in zip(h_np, gh_np):
            if np.all(np.isfinite(gg)) and np.isfinite(h0):
                row = np.zeros(n + 1)
                row[:n] = gg
                A.append(row)
                rhs.append(float(gg @ xk - h0))

        res = sopt.linprog(
            c, A_ub=np.asarray(A), b_ub=np.asarray(rhs), bounds=bounds, method="highs"
        )
        if not res.success:
            msg = (res.message or "").lower()
            if "infeasible" in msg:
                return ReducedBound(bound=np.inf, status="infeasible", rounds=r, history=history)
            if "unbounded" in msg:
                return ReducedBound(bound=-np.inf, status="unbounded", rounds=r, history=history)
            break
        bound = float(res.fun)
        history.append(bound)
        xk = np.asarray(res.x[:n], dtype=float)
        if bound - prev < tol * (abs(bound) + 1.0):
            break
        prev = bound

    dual = -bound if negate else bound
    return ReducedBound(bound=dual, status="optimal", rounds=r, history=history)
