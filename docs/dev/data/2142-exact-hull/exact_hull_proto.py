"""#2142 exact-hull prototype: three candidate lowerings of a NONLINEAR disjunct row that keep
``1/yhat`` out of every product. Script-only (CLAUDE.md: no solver-code change in this step).

``install(arm)`` monkeypatches ``discopt._relax.gdp_reformulate`` so that every nonlinear
disjunct row is routed through the hull's existing origin-free hook (the #1917 eps-free lifted
perspective, ``_origin_free_hull_row``), which is replaced by the arm's lowering. Everything
else the hull emits (selectors, copies, aggregation, bound links, linear rows) is unchanged.

Arms, for a row ``g(x) = sum_m c_m prod_{i in m} x_i  <sense>  rhs`` of disjunct k, with the
disjunct's copies ``v_i`` (pinned to 0 when ``y = 0`` by the bound links) and ``t = v / y``:

* ``ratio``  -- the hook as shipped: ``y * g(t) <sense> y * rhs`` with ``v_i - y * t_i = 0``,
  ``t_i`` boxed to the disjunct's own box. Exact at y in {0, 1}; the perspective at
  fractional y.
* ``homog``  -- the homogenized perspective (candidate (a)): with d = max monomial degree,
  ``sum_m c_m prod v_i * y^(d - |m|) <sense> rhs * y^d``. No aux, no division. Exact at
  y in {0, 1}; at fractional y it is ``y^d g(t)``, the perspective's solution set.
* ``lifted`` -- lift-then-hull (candidate (b) + (a) for the aux definitions): each monomial
  of degree >= 2 is factored left to right into auxes ``w_1 = v_i1 * v_i2``, ``w_j = w_{j-1}
  * v_ij`` on the ratio scale (``w_j = y * prod t``), each defined by the homogenized
  bilinear equality ``w_j * y = w_{j-1} * v_ij`` and tightened by the perspective of its
  McCormick envelope on the disjunct's box (the Balas hull of the per-disjunct McCormick
  relaxation, linear rows). The row itself becomes linear: ``sum_m c_m w_m - rhs * y``.

Every arm declines (raises) on a row that is not a polynomial in the disjunct's variables;
a shipped implementation would fall back to the eps-perspective there.
"""
from __future__ import annotations

import itertools

import numpy as np

import discopt._relax.gdp_reformulate as G
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    UnaryOp,
    Variable,
    VarType,
    _wrap,
)

STATS = {"rows": 0, "auxes": 0, "hull_mc_rows": 0, "arm": None}
_ORIG_BODY_AT_ZERO = G._body_at_zero
_ORIG_HOOK = G._origin_free_hull_row


class NotPolynomial(ValueError):
    pass


def monomials(expr) -> dict[tuple[str, ...], float]:
    """``expr`` as {sorted variable-name tuple: coefficient}; raises NotPolynomial."""
    if isinstance(expr, Variable):
        if expr.shape != ():
            raise NotPolynomial(f"array variable {expr.name}")
        return {(expr.name,): 1.0}
    if isinstance(expr, Constant):
        v = np.asarray(expr.value, dtype=float)
        if v.shape != () and v.size != 1:
            raise NotPolynomial("array constant")
        return {(): float(v.ravel()[0])}
    if isinstance(expr, UnaryOp) and expr.op in ("neg", "-"):
        return {k: -c for k, c in monomials(expr.operand).items()}
    if isinstance(expr, BinaryOp):
        if expr.op in ("+", "-"):
            out = dict(monomials(expr.left))
            sign = 1.0 if expr.op == "+" else -1.0
            for k, c in monomials(expr.right).items():
                out[k] = out.get(k, 0.0) + sign * c
            return {k: c for k, c in out.items() if c != 0.0}
        if expr.op == "*":
            out: dict[tuple[str, ...], float] = {}
            for (ka, ca), (kb, cb) in itertools.product(
                monomials(expr.left).items(), monomials(expr.right).items()
            ):
                k = tuple(sorted(ka + kb))
                out[k] = out.get(k, 0.0) + ca * cb
            return {k: c for k, c in out.items() if c != 0.0}
        raise NotPolynomial(f"operator {expr.op!r}")
    terms = G._sumover_terms(expr)
    if terms is not None:
        out = {}
        for t in terms:
            for k, c in monomials(t).items():
                out[k] = out.get(k, 0.0) + c
        return {k: c for k, c in out.items() if c != 0.0}
    raise NotPolynomial(type(expr).__name__)


def _imul(a, b):
    lo = min(a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    hi = max(a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (lo, hi)


def _ypow(y, p: int):
    out = _wrap(1.0)
    for _ in range(p):
        out = out * y
    return out


def _prod(factors):
    out = factors[0]
    for f in factors[1:]:
        out = out * f
    return out


def _new_var(model, name, lo, hi, new_vars):
    w = Variable(model._fresh_name(name), VarType.CONTINUOUS, (), lo, hi, model)
    model._register_generated_variable(w)
    new_vars.append(w)
    STATS["auxes"] += 1
    return w


def _mc_hull_rows(w, a, b, box_a, box_b, y, stem, new_cons) -> None:
    """The perspective of McCormick(a, b) on boxes ``box_a``, ``box_b`` for ``w = y * ta * tb``
    with ``a = y * ta``, ``b = y * tb``: each envelope row ``ta tb >= L_b ta + L_a tb - L_a L_b``
    times ``y >= 0`` gives ``w >= L_b a + L_a b - L_a L_b y``. Exact when ``a`` and ``b`` are the
    same column (then the secant/tangent rows of the square)."""
    La, Ua = box_a
    Lb, Ub = box_b
    if a is b:
        # square: w >= 2c a - c^2 y (tangents), w <= (L+U) a - L U y (secant)
        for c in (La, 0.5 * (La + Ua), Ua):
            new_cons.append(Constraint(body=w - _wrap(2.0 * c) * a + _wrap(c * c) * y,
                                       sense=">=", rhs=0.0, name=f"{stem}_sqtan"))
        new_cons.append(Constraint(body=w - _wrap(La + Ua) * a + _wrap(La * Ua) * y,
                                   sense="<=", rhs=0.0, name=f"{stem}_sqsec"))
        STATS["hull_mc_rows"] += 4
        return
    rows = [
        (">=", Lb, La, La * Lb),
        (">=", Ub, Ua, Ua * Ub),
        ("<=", Ub, La, La * Ub),
        ("<=", Lb, Ua, Ua * Lb),
    ]
    for sense, ca, cb, cy in rows:
        new_cons.append(Constraint(body=w - _wrap(ca) * a - _wrap(cb) * b + _wrap(cy) * y,
                                   sense=sense, rhs=0.0, name=f"{stem}_mc"))
    STATS["hull_mc_rows"] += 4


def _hook(con, y_k, owned, bounds_k, cache, model, stem, new_vars, new_cons, origin_exc):
    arm = STATS["arm"]
    if arm == "ratio":
        return _ORIG_HOOK(con, y_k, owned, bounds_k, cache, model, stem, new_vars, new_cons,
                          origin_exc)
    copies = dict(owned)  # name -> disjunct k's copy
    mono = monomials(con.body)
    rhs = float(np.max(con.rhs))
    STATS["rows"] += 1
    if arm == "homog":
        d = max((len(k) for k in mono), default=0)
        body = _wrap(0.0)
        for k, c in mono.items():
            body = body + _wrap(c) * _prod([copies[n] for n in k] + [_ypow(y_k, d - len(k))])
        return body, _wrap(rhs) * _ypow(y_k, d)
    assert arm == "lifted", arm
    box = {n: G._lifted_bounds(n, v, bounds_k) for n, v in owned}  # t's box = disjunct box
    body = _wrap(0.0)
    for k, c in mono.items():
        if len(k) == 0:
            body = body + _wrap(c) * y_k
            continue
        if len(k) == 1:
            body = body + _wrap(c) * copies[k[0]]
            continue
        # chain w_1 = y t_1 t_2, w_j = w_{j-1} t_j (ratio scale), cached per disjunct by monomial
        key = "mono:" + "*".join(k)
        prev, prev_box = copies[k[0]], box[k[0]]
        for j in range(1, len(k)):
            ckey = key if j == len(k) - 1 else "mono:" + "*".join(k[: j + 1])
            w = cache.get(ckey)
            nb = _imul(prev_box, box[k[j]])
            if w is None:
                w = _new_var(model, f"_hull_{stem}_w_{'_'.join(k[: j + 1])}",
                             min(nb[0], 0.0), max(nb[1], 0.0), new_vars)
                cache[ckey] = w
                vj = copies[k[j]]
                # exact, homogenized: w * y = prev * v_j  (w = y prod t, prev * v_j = y^2 prod t)
                new_cons.append(Constraint(body=w * y_k - prev * vj, sense="==", rhs=0.0,
                                           name=f"_hull_{stem}_wdef_{'_'.join(k[: j + 1])}"))
                # bound links (implied by the hull rows below; cheap for propagation)
                new_cons.append(Constraint(body=w - _wrap(nb[1]) * y_k, sense="<=", rhs=0.0,
                                           name=f"_hull_{stem}_wub"))
                new_cons.append(Constraint(body=w - _wrap(nb[0]) * y_k, sense=">=", rhs=0.0,
                                           name=f"_hull_{stem}_wlb"))
                _mc_hull_rows(w, prev, vj, prev_box, box[k[j]], y_k,
                              f"_hull_{stem}_wmc_{'_'.join(k[: j + 1])}", new_cons)
            prev, prev_box = w, nb
        body = body + _wrap(c) * prev
    return body, _wrap(rhs) * y_k


def _body_at_zero(expr, all_vars):
    if G._is_linear(expr):
        return _ORIG_BODY_AT_ZERO(expr, all_vars)
    raise G.HullPerspectiveOriginError("#2142 prototype: nonlinear row routed to the hook")


def install(arm: str) -> None:
    assert arm in ("ratio", "homog", "lifted"), arm
    STATS.update(rows=0, auxes=0, hull_mc_rows=0, arm=arm)
    G._body_at_zero = _body_at_zero
    G._origin_free_hull_row = _hook


def uninstall() -> None:
    G._body_at_zero = _ORIG_BODY_AT_ZERO
    G._origin_free_hull_row = _ORIG_HOOK
    STATS["arm"] = None
