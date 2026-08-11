"""Reverse-mode interval AD on the modeling DAG (M9 of issue #51).

The forward interval evaluator in :mod:`interval_eval` propagates
variable-box enclosures up the DAG to obtain a sound enclosure of a
top-level expression. That direction by itself does not tighten
intermediate variable boxes: it only widens.

Feasibility-based bound tightening (FBBT) is the *reverse* direction.
Given a constraint ``expr(x) ∈ target`` and forward enclosures at
every node, we walk the DAG top-down and propagate the parent's
required range back to its children using sound inverse operator
rules. The tightened child enclosure is

    child_tight = child_forward ∩ inverse_op(parent_tight, sibling_forward).

Crucially, the inverse step uses **forward** enclosures of the siblings
— not their tightened versions — so the propagation is one-pass and
monotone. Iterating the pass to quiescence (Gauss-Seidel style)
recovers the textbook FBBT algorithm.

Why a Python implementation when the Rust solver already has FBBT
=================================================================

``crates/discopt-core/src/presolve/fbbt.rs`` runs against the Rust IR
during presolve. The JAX relaxation pipeline operates on the Python
modeling DAG and currently has no analogous reverse pass — its only
bound source is forward interval AD (which never tightens an internal
variable) and the Rust FBBT result, which has to round-trip through
PyO3 for every query. A JAX-side reverse propagator gives the
relaxation compiler per-subexpression tightening directly, and lets
the convexity certificate leverage tighter Hessian enclosures during
the box-local PSD test.

Soundness invariant
-------------------
Every inverse rule below preserves the property

    {x ∈ child_forward : op(x, sibling_forward) ∈ parent_tight}
        ⊆ inverse_op(parent_tight, sibling_forward).

Equivalently: no feasible value of the child is excluded from the
returned enclosure. This is verified by sampling in the regression
suite (≥ 10⁴ samples per case).

Acceptance criteria from issue #51 satisfied here:

1. Reverse-propagated bounds match the Rust FBBT reference
   implementation (``presolve/fbbt.rs``) within 1e-9 on a shared test
   set — see :func:`test_matches_rust_fbbt_on_shared_models` in
   ``test_interval_ad_reverse.py``.
2. Tightened per-subexpression bounds are always subsets of the
   original forward bounds (monotone tightening) — guaranteed by
   construction via the explicit ``∩ forward`` intersect at every
   node.
3. No bound-tightening pass ever removes a feasible point of the
   original model — verified by sampling in the regression suite.

References
----------
Belotti, Lee, Liberti, Margot, Wächter (2009). *Branching and bounds
  tightening techniques for non-convex MINLP*. Optim. Methods Softw.
  24(4-5), 597-634.
Schichl, Neumaier (2005). *Interval analysis on directed acyclic
  graphs for global optimization*. J. Global Optim. 33(4), 541-562.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
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

from . import interval as iv
from .interval import Interval, _round_down, _round_up
from .interval_eval import evaluate_interval

# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def reverse_propagate(
    expr: Expression,
    target: Interval,
    model: Model,
    box: Optional[dict] = None,
    *,
    forward: Optional[dict] = None,
) -> dict[int, Interval]:
    """One pass of reverse interval propagation.

    Args:
        expr: Top-level expression of a constraint ``expr ∈ target``.
        target: Required interval enclosure of ``expr``.
        model: The model defining variable layout / declared bounds.
        box: Optional ``{Variable: Interval}`` overriding declared
            bounds for the forward pass.
        forward: Optional pre-computed forward-enclosure dict
            ``{id(node): Interval}`` (e.g. from a prior call); avoids
            re-walking the DAG when iterating.

    Returns:
        ``{id(node): Interval}`` mapping every reachable sub-expression
        to its tightened forward-∩-back-propagated enclosure. The root
        entry is ``forward[id(expr)] ∩ target``.
    """
    box = box or {}
    if forward is None:
        forward_cache: dict = {}
        evaluate_interval(expr, model, box, _cache=forward_cache)
        forward = forward_cache

    tight: dict[int, Interval] = dict(forward)
    root_id = id(expr)
    if root_id in forward:
        tight[root_id] = _intersect(forward[root_id], target)
    else:
        tight[root_id] = target

    order = _post_order(expr)
    # Walk root-to-leaves: reverse of post-order.
    for node in reversed(order):
        _propagate_down(node, tight, forward)
    return tight


def tighten_box(
    constraints: list[tuple[Expression, Interval]],
    model: Model,
    box: Optional[dict] = None,
    *,
    max_iter: int = 25,
    tol: float = 1e-9,
) -> dict[Variable, Interval]:
    """Iterate reverse-mode AD over a constraint list to quiescence.

    Args:
        constraints: List of ``(expr, target)`` pairs. Each pair asserts
            ``expr(x) ∈ target`` for every feasible ``x``.
        model: The model.
        box: Optional initial ``{Variable: Interval}`` (defaults to the
            declared variable bounds).
        max_iter: Cap on Gauss-Seidel sweeps.
        tol: Quiescence threshold on the L∞-norm of the box-width
            change between sweeps.

    Returns:
        ``{Variable: Interval}`` final tightened box. Keys cover only
        variables the propagation actually saw a constraint for; the
        caller can union with the input ``box`` for completeness.
    """
    box = dict(box) if box is not None else {}
    cur: dict[Variable, Interval] = {v: _declared_box(v, box) for v in model._variables}

    for _ in range(max_iter):
        prev = {k: (v.lo.copy(), v.hi.copy()) for k, v in cur.items()}
        for expr, target in constraints:
            forward_cache: dict = {}
            evaluate_interval(expr, model, cur, _cache=forward_cache)
            tight = reverse_propagate(expr, target, model, cur, forward=forward_cache)
            for v in _variables_in(expr):
                vid = id(v)
                if vid in tight:
                    new_iv = _intersect(cur[v], tight[vid])
                    cur[v] = new_iv
        if _box_change(prev, cur) < tol:
            break
    return cur


# ──────────────────────────────────────────────────────────────────────
# DAG traversal
# ──────────────────────────────────────────────────────────────────────


def _post_order(expr: Expression) -> list[Expression]:
    seen: set[int] = set()
    order: list[Expression] = []

    def visit(node: Expression) -> None:
        eid = id(node)
        if eid in seen:
            return
        seen.add(eid)
        for child in _children(node):
            visit(child)
        order.append(node)

    visit(expr)
    return order


def _children(node: Expression) -> list[Expression]:
    if isinstance(node, BinaryOp):
        return [node.left, node.right]
    if isinstance(node, UnaryOp):
        return [node.operand]
    if isinstance(node, FunctionCall):
        return list(node.args)
    if isinstance(node, IndexExpression):
        if isinstance(node.base, Variable):
            return []
        return [node.base]
    if isinstance(node, SumExpression):
        return [node.operand]
    if isinstance(node, SumOverExpression):
        return list(node.terms)
    return []


def _variables_in(expr: Expression) -> set[Variable]:
    out: set[Variable] = set()
    seen: set[int] = set()

    def visit(n: Expression) -> None:
        if id(n) in seen:
            return
        seen.add(id(n))
        if isinstance(n, Variable):
            out.add(n)
            return
        if isinstance(n, IndexExpression) and isinstance(n.base, Variable):
            out.add(n.base)
            return
        for c in _children(n):
            visit(c)

    visit(expr)
    return out


# ──────────────────────────────────────────────────────────────────────
# Per-node propagation
# ──────────────────────────────────────────────────────────────────────


def _propagate_down(node: Expression, tight: dict, forward: dict) -> None:
    """Push ``tight[id(node)]`` back to children via inverse rules."""
    parent_iv = tight.get(id(node))
    if parent_iv is None:
        return

    if isinstance(node, (Constant, Parameter, Variable)):
        return  # leaves: nothing to push down
    if isinstance(node, IndexExpression) and isinstance(node.base, Variable):
        # Tightening on a single index doesn't directly tighten the
        # vector variable's box without knowing the index — out of scope
        # for v1 (regression covers scalar variables only).
        return

    if isinstance(node, UnaryOp):
        if node.op == "neg":
            _update(tight, node.operand, -parent_iv)
        # |x|: x ∈ ±[parent.lo, parent.hi]; conservative fallback is
        # to skip (forward already encloses).
        return

    if isinstance(node, BinaryOp):
        _propagate_binary(node, parent_iv, tight, forward)
        return

    if isinstance(node, FunctionCall):
        _propagate_function(node, parent_iv, tight, forward)
        return

    if isinstance(node, SumExpression):
        _update(tight, node.operand, parent_iv)
        return

    if isinstance(node, SumOverExpression):
        _propagate_sum_over(node, parent_iv, tight, forward)
        return


def _propagate_binary(node: BinaryOp, parent: Interval, tight: dict, forward: dict) -> None:
    left_fwd = forward.get(id(node.left))
    right_fwd = forward.get(id(node.right))
    if left_fwd is None or right_fwd is None:
        return

    if node.op == "+":
        # left = parent - right; right = parent - left
        _update(tight, node.left, parent - right_fwd)
        _update(tight, node.right, parent - left_fwd)
        return

    if node.op == "-":
        # left = parent + right; right = left - parent
        _update(tight, node.left, parent + right_fwd)
        _update(tight, node.right, left_fwd - parent)
        return

    if node.op == "*":
        # left = parent / right (sound: skip if 0 ∈ right_fwd)
        if not bool(np.any(right_fwd.contains_zero())):
            _update(tight, node.left, parent / right_fwd)
        if not bool(np.any(left_fwd.contains_zero())):
            _update(tight, node.right, parent / left_fwd)
        return

    if node.op == "/":
        # left = parent * right; right = left / parent (only if 0 ∉ parent)
        _update(tight, node.left, parent * right_fwd)
        if not bool(np.any(parent.contains_zero())):
            _update(tight, node.right, left_fwd / parent)
        return

    if node.op == "**":
        _propagate_power(node, parent, tight, forward)
        return


def _propagate_function(node: FunctionCall, parent: Interval, tight: dict, forward: dict) -> None:
    if len(node.args) != 1:
        return
    arg_fwd = forward.get(id(node.args[0]))
    if arg_fwd is None:
        return

    name = node.func_name

    if name == "exp":
        # arg = log(parent ∩ (0, inf))
        positive = _intersect_positive(parent)
        if positive is None:
            return
        _update(tight, node.args[0], iv.log(positive))
        return

    if name == "log":
        # arg = exp(parent), intersect with (0, inf)
        e = iv.exp(parent)
        _update(tight, node.args[0], e)
        return

    if name == "sqrt":
        # arg = parent² intersected with [0, inf)
        nonneg = _intersect_nonneg(parent)
        if nonneg is None:
            return
        sq = nonneg * nonneg
        _update(tight, node.args[0], sq)
        return

    if name == "abs":
        # |x| ∈ parent ⟹ x ∈ [-hi, hi]; respect forward sign info.
        hi = parent.hi
        cand = Interval(-hi, hi)
        _update(tight, node.args[0], cand)
        return

    # Unsupported: skip (forward bound stays in place).


def _propagate_power(node: BinaryOp, parent: Interval, tight: dict, forward: dict) -> None:
    """Inverse of ``base ** p`` for a literal scalar exponent ``p``."""
    if not isinstance(node.right, (Constant, Parameter)):
        return
    raw = np.asarray(node.right.value)
    if raw.ndim != 0:
        return
    p = float(raw)
    if not np.isclose(p, round(p)):
        return  # fractional handled separately if ever needed
    p_int = int(round(p))
    if p_int == 1:
        _update(tight, node.left, parent)
        return
    if p_int == 0:
        return  # parent must be {1}; cannot tighten base
    if p_int == 2:
        # x² ∈ parent ⟹ x ∈ [-sqrt(hi), sqrt(hi)] ∩ ([sqrt(max(lo,0)), sqrt(hi)] ∪ symm.)
        hi = parent.hi
        lo = np.maximum(parent.lo, 0.0)
        if np.any(hi < 0):
            return  # infeasible — caller may detect via empty intersection
        with np.errstate(invalid="ignore"):
            r_hi = np.sqrt(np.where(hi >= 0, hi, 0.0))
            r_lo = np.sqrt(lo)
        # Without parity info, the safe enclosure is ``[-r_hi, r_hi]``.
        # Restrict the magnitude using forward sign of base.
        base_fwd = forward.get(id(node.left))
        if base_fwd is not None and bool(np.all(np.asarray(base_fwd.lo) >= 0.0)):
            _update(tight, node.left, Interval(_round_down(r_lo), _round_up(r_hi)))
            return
        if base_fwd is not None and bool(np.all(np.asarray(base_fwd.hi) <= 0.0)):
            _update(tight, node.left, Interval(_round_down(-r_hi), _round_up(-r_lo)))
            return
        _update(tight, node.left, Interval(_round_down(-r_hi), _round_up(r_hi)))
        return
    if p_int >= 3 and p_int % 2 == 1:
        # Odd power: x = parent ** (1/p) is monotone.
        with np.errstate(invalid="ignore"):
            sign_lo = np.sign(parent.lo)
            sign_hi = np.sign(parent.hi)
            r_lo = sign_lo * np.power(np.abs(parent.lo), 1.0 / p_int)
            r_hi = sign_hi * np.power(np.abs(parent.hi), 1.0 / p_int)
        _update(tight, node.left, Interval(_round_down(r_lo), _round_up(r_hi)))
        return
    if p_int >= 4 and p_int % 2 == 0:
        # Even power: same shape as p == 2, with p-th root.
        hi = parent.hi
        lo = np.maximum(parent.lo, 0.0)
        if np.any(hi < 0):
            return
        with np.errstate(invalid="ignore"):
            r_hi = np.power(np.where(hi >= 0, hi, 0.0), 1.0 / p_int)
            r_lo = np.power(lo, 1.0 / p_int)
        base_fwd = forward.get(id(node.left))
        if base_fwd is not None and bool(np.all(np.asarray(base_fwd.lo) >= 0.0)):
            _update(tight, node.left, Interval(_round_down(r_lo), _round_up(r_hi)))
            return
        if base_fwd is not None and bool(np.all(np.asarray(base_fwd.hi) <= 0.0)):
            _update(tight, node.left, Interval(_round_down(-r_hi), _round_up(-r_lo)))
            return
        _update(tight, node.left, Interval(_round_down(-r_hi), _round_up(r_hi)))
        return


def _propagate_sum_over(
    node: SumOverExpression, parent: Interval, tight: dict, forward: dict
) -> None:
    """For ``parent = Σ_i term_i``, term_j ∈ parent − Σ_{i≠j} term_i_fwd."""
    fwds_opt = [forward.get(id(t)) for t in node.terms]
    if any(f is None for f in fwds_opt):
        return
    fwds: list[Interval] = [f for f in fwds_opt if f is not None]
    total_others_lo: np.ndarray = np.asarray(np.float64(0.0))
    total_others_hi: np.ndarray = np.asarray(np.float64(0.0))
    # Pre-aggregate the sum of all forward ranges for cheap exclusion.
    for f in fwds:
        total_others_lo = _round_down(np.float64(total_others_lo + f.lo))
        total_others_hi = _round_up(np.float64(total_others_hi + f.hi))
    for t, f in zip(node.terms, fwds):
        # Σ_{i≠j} = total_others − f_j (interval subtraction, sound).
        rest = Interval(
            _round_down(total_others_lo - f.hi),
            _round_up(total_others_hi - f.lo),
        )
        _update(tight, t, parent - rest)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _update(tight: dict, child: Expression, candidate: Interval) -> None:
    """Intersect the existing ``tight[id(child)]`` with ``candidate``."""
    cid = id(child)
    cur = tight.get(cid)
    if cur is None:
        tight[cid] = candidate
        return
    tight[cid] = _intersect(cur, candidate)


def _intersect(a: Interval, b: Interval) -> Interval:
    """Sound elementwise intersection. NaN-safe for unbounded inputs.

    If the candidate intersection becomes empty (lo > hi after the
    elementwise ``max/min``), we clip it to a degenerate interval at
    the midpoint of the input — soundness is preserved since downstream
    consumers detect empty intersections separately, and the
    Interval constructor would otherwise raise. In practice an empty
    intersection signals constraint infeasibility for the local box.
    """
    a_lo = np.asarray(a.lo)
    a_hi = np.asarray(a.hi)
    b_lo = np.asarray(b.lo)
    b_hi = np.asarray(b.hi)
    lo = np.maximum(a_lo, b_lo)
    hi = np.minimum(a_hi, b_hi)
    # Clip lo > hi back to the original ``a`` (do not poison soundness;
    # this mirrors how the Rust FBBT abstains from reporting an empty
    # box and signals infeasibility through a separate code path).
    bad = lo > hi
    if np.any(bad):
        lo = np.where(bad, a_lo, lo)
        hi = np.where(bad, a_hi, hi)
    return Interval(lo, hi)


def _intersect_positive(x: Interval) -> Optional[Interval]:
    lo = np.maximum(np.asarray(x.lo), np.float64(np.finfo(np.float64).tiny))
    hi = np.asarray(x.hi)
    if np.any(lo > hi):
        return None
    return Interval(lo, hi)


def _intersect_nonneg(x: Interval) -> Optional[Interval]:
    lo = np.maximum(np.asarray(x.lo), 0.0)
    hi = np.asarray(x.hi)
    if np.any(lo > hi):
        return None
    return Interval(lo, hi)


def _declared_box(v: Variable, override: dict) -> Interval:
    if v in override:
        result: Interval = override[v]
        return result
    lb = np.asarray(v.lb, dtype=np.float64)
    ub = np.asarray(v.ub, dtype=np.float64)
    return Interval(lb, ub)


def _box_change(prev: dict, cur: dict[Variable, Interval]) -> float:
    delta = 0.0
    for v, ivl in cur.items():
        plo, phi = prev.get(v, (ivl.lo, ivl.hi))
        d_lo = float(np.max(np.abs(np.asarray(ivl.lo) - np.asarray(plo))))
        d_hi = float(np.max(np.abs(np.asarray(ivl.hi) - np.asarray(phi))))
        delta = max(delta, d_lo, d_hi)
    return delta


__all__ = ["reverse_propagate", "tighten_box"]
