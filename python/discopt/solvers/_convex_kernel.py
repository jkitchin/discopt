"""Convex LP-OA branch-and-cut kernel — production producer + routing gate (#798).

The native Rust kernel (`discopt._rust.solve_convex_tree_py`) certifies convex
MINLPs of the `rsyn*`/`syn*` family far faster than the NLP-BB path (measured:
all 4 panel instances certified in ~24 s vs NLP-BB timing out uncertified at
120 s each). This module is the analyze-once producer + the **soundness gate**
that decides whether a model may be routed to it.

## Soundness gate (do NOT relax)

The kernel outer-approximates every nonlinear constraint by first-order tangents,
which is a VALID relaxation only for CONVEX `≤` rows. A model is routed here ONLY
when ALL of these hold; otherwise `build_convex_spec` returns ``None`` and the
caller keeps the (always-correct) NLP-BB path:

* the objective is LINEAR (its gradient is constant);
* every nonlinear constraint decomposes into composite-of-affine form
  ``g(x) = a·x + b + Σ_t coeff_t·func_t(p_t·x + q_t)``;
* each such term is CONVEX in the constraint's ``≤`` normal form — a convex
  ``func`` (exp) with ``coeff ≥ 0``, or a concave ``func`` (log/sqrt/log1p) with
  ``coeff ≤ 0`` (a ``≥`` row is negated to ``≤`` first, flipping every sign);
* nonlinear EQUALITY constraints are never routed (a nonlinear equality is not a
  convex feasible set).

Routing an unproven-convex model would give an unsound (too-tight) dual bound and
a possible false ``optimal`` — so the gate is conservative by construction: any
unrecognized function, non-affine argument, bilinear term, or wrong-curvature
term makes the whole model fall back.

## Perspective terms (#865)

Hull-reformulated (``*hfsg``) models — ``syn*hfsg``, ``rsyn*hfsg``, and the rest of
the smoothed-hull family — write their disjunctive nonlinearities as the
**perspective** ``s·f(a/s)``, e.g. ``syn05hfsg``'s

    (x2/ε − log(x0/ε + 1)) · ε ≤ 0 ,    ε = 0.001 + 0.999·y ,  y binary

which distributes to ``x2 − ε·log(x0/ε + 1) ≤ 0``. Syntactically this is a product
of two non-constant subexpressions, so the plain gate rejected it as a "bilinear
product"; mathematically it is nothing of the sort — the perspective of a convex
``f`` is JOINTLY CONVEX in ``(a, s)`` on ``s > 0``, and ``a``/``s`` are affine in
``x``, so ``s·f(a/s)`` is convex in ``x``. Admitting it is recognising convexity the
syntactic gate missed, not loosening anything.

A perspective term is accepted only when ALL of these hold (else the model falls
back exactly as before):

* the same curvature rule as a plain term — ``sign(coeff)·curvature(func) ≥ 0``;
* ``s > 0`` PROVEN by interval arithmetic over the variable box (this is the
  convexity precondition — the perspective is convex only on the open half-space
  ``s > 0``; the smoothing floor ``0.001`` is exactly what makes it hold here);
* ``a/s`` lies inside ``func``'s domain over the box, so value and tangent are
  finite everywhere the kernel can evaluate them.

Bounds enter the gate here, so an unbounded/undetermined box is a refusal, not a
guess.

## Quadratic inner function (#879)

The inner ``func`` may also be a ``** 2`` (see ``_pow_as_sqr``), whose perspective
``s·(a/s)² = a²/s`` is quadratic-over-linear — the ``clay*hfsg`` hull shape. Only
the exponent 2 is admitted; every other power (odd, fractional, negative, or
variable) is nonconvex, domain-restricted, or signomial, and keeps falling back,
as does a non-affine base such as ``(log x)²``.

This term class was withdrawn once (the evidence is recorded in #879): routing
``clay0303hfsg`` reported
``optimal`` at three mutually inconsistent objectives (28351.42 / 36397.83 /
55092.52), each worse than a point the default path attains, which read as a dual
bound sitting above the true optimum. It was not. Those three numbers were
**incumbents**, published as certified by the tree bug #871 fixed — a subtree
silently discarded on a `numerical` node LP, after which the reported ``bound``
fell back to the incumbent's own objective. Re-measured with that fix in place,
the ``a²/s`` relaxation is sound at every separation setting (root safe bound
``0.0`` vs the optimum ``26669.11``, i.e. valid and merely weak), and
``clay0303hfsg`` now certifies ``26669.1096`` against its MINLPLib reference.

The lesson that *does* stand: exactness and convexity of the marshaled rows are
not sufficient evidence to admit a term class. A routed instance's certified
objective must be checked against a known optimum —
``test_convex_kernel_perspective_865.py`` does that here, with the reference
value in ``python/tests/data/known_optima.toml``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from discopt.modeling.core import SolveResult

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    FunctionCall,
    IndexExpression,
    UnaryOp,
    Variable,
)

# func -> (numpy value, curvature) where curvature is +1 convex, -1 concave.
# A term coeff*func(affine) is convex iff sign(coeff) * curvature >= 0.
_FUNC = {
    "log": (np.log, -1),
    "log1p": (np.log1p, -1),
    "sqrt": (np.sqrt, -1),
    "exp": (np.exp, +1),
    # `sqr` has no FunctionCall spelling — it is how a `** 2` node is admitted
    # (see `_pow_as_sqr`). Its perspective `s·(a/s)² = a²/s` is
    # quadratic-over-linear, the `clay*hfsg` hull shape.
    "sqr": (np.square, +1),
}
# Rust term_func codes (must match ConvexFunc in convex_kernel.rs).
_FUNC_CODE = {"log": 0, "exp": 1, "sqrt": 2, "log1p": 3, "sqr": 4}


logger = logging.getLogger(__name__)


class NotConvexKernel(Exception):
    """The model cannot be soundly routed to the convex kernel (→ NLP-BB)."""


def _flat_offsets(model) -> dict[int, int]:
    off, cur = {}, 0
    for v in model._variables:
        off[v._index] = cur
        cur += v.size
    return off


def _col_of(node, offsets: dict[int, int]) -> int:
    if isinstance(node, Variable):
        if node.size != 1:
            raise NotConvexKernel("array variable used as scalar")
        return offsets[node._index]
    if isinstance(node, IndexExpression) and isinstance(node.base, Variable):
        base, idx = node.base, node.index
        flat = int(np.ravel_multi_index(idx, base.shape)) if isinstance(idx, tuple) else int(idx)
        return offsets[base._index] + flat
    raise NotConvexKernel("non-variable leaf")


class _Decomp:
    """Affine part + composite terms.

    A term is ``{coeff, func, arg_aff, arg_const, sc_aff, sc_const}``. With
    ``sc_aff is None`` it denotes ``coeff·func(arg)``; otherwise it denotes the
    perspective ``coeff·s·func(arg/s)`` with ``s = sc_aff·x + sc_const``.
    """

    __slots__ = ("aff", "const", "terms")

    def __init__(self):
        self.aff: dict[int, float] = {}
        self.const: float = 0.0
        self.terms: list[dict] = []

    def scale(self, k: float) -> _Decomp:
        self.const *= k
        for c in list(self.aff):
            self.aff[c] *= k
        for t in self.terms:
            t["coeff"] *= k
        return self

    def add(self, other: _Decomp) -> _Decomp:
        self.const += other.const
        for c, v in other.aff.items():
            self.aff[c] = self.aff.get(c, 0.0) + v
        self.terms.extend(other.terms)
        return self


def _as_const(node) -> Optional[float]:
    if isinstance(node, Constant) and node.value.ndim == 0:
        return float(node.value)
    return None


def _decompose(node, offsets) -> _Decomp:
    """Decompose into affine + composite-univariate terms, or raise."""
    d = _Decomp()
    c = _as_const(node)
    if c is not None:
        d.const = c
        return d
    if isinstance(node, (Variable, IndexExpression)):
        d.aff[_col_of(node, offsets)] = 1.0
        return d
    if isinstance(node, UnaryOp):
        if node.op == "neg":
            return _decompose(node.operand, offsets).scale(-1.0)
        raise NotConvexKernel(f"unary {node.op}")
    if isinstance(node, BinaryOp):
        if node.op == "+":
            return _decompose(node.left, offsets).add(_decompose(node.right, offsets))
        if node.op == "-":
            return _decompose(node.left, offsets).add(_decompose(node.right, offsets).scale(-1.0))
        if node.op == "*":
            lc, rc = _as_const(node.left), _as_const(node.right)
            if lc is not None:
                return _decompose(node.right, offsets).scale(lc)
            if rc is not None:
                return _decompose(node.left, offsets).scale(rc)
            # Neither factor is constant. Before declaring it bilinear, try the
            # PERSPECTIVE shape `s · h(·/s)` (#865): if one factor is an affine `s`
            # and the other is built from `·/s` ratios, the product is a sum of
            # affine terms and perspectives — convex, not bilinear.
            return _try_perspective(node, offsets)
        if node.op == "/":
            rc = _as_const(node.right)
            if rc is not None and rc != 0.0:
                return _decompose(node.left, offsets).scale(1.0 / rc)
            raise NotConvexKernel("division by non-constant")
        if node.op == "**":
            return _pow_as_sqr(node, _decompose, offsets)
        raise NotConvexKernel(f"binary {node.op}")
    if isinstance(node, FunctionCall):
        if node.func_name not in _FUNC:
            raise NotConvexKernel(f"unsupported func {node.func_name}")
        if len(node.args) != 1:
            raise NotConvexKernel(f"multi-arg func {node.func_name}")
        arg = _decompose(node.args[0], offsets)
        if arg.terms:
            raise NotConvexKernel("non-affine function argument")
        d.terms.append(_term(1.0, node.func_name, arg.aff, arg.const))
        return d
    raise NotConvexKernel(f"node {type(node).__name__}")


def _pow_as_sqr(node, decompose_fn, offsets) -> _Decomp:
    """``base ** 2`` → a convex ``sqr`` term over an affine base; else refuse.

    Only the exponent 2 is admitted. Every other power (odd, fractional, negative,
    or variable) is either nonconvex, domain-restricted, or a signomial — none of
    which this gate may wave through, so they keep falling back. ``decompose_fn``
    is the caller's decomposer, so this works unchanged in ratio space: the base of
    ``(x/s)**2`` decomposes to the ratio ``x/s``, and the surrounding lift turns the
    term into the perspective ``s·(a/s)² = a²/s``.
    """
    e = _as_const(node.right)
    if e is None:
        raise NotConvexKernel("variable exponent")
    if e != 2.0:
        raise NotConvexKernel(f"power {e:g}")
    base = decompose_fn(node.left, offsets)
    if base.terms:
        raise NotConvexKernel("non-affine power base")
    d = _Decomp()
    d.terms.append(_term(1.0, "sqr", base.aff, base.const))
    return d


def _term(coeff, func, arg_aff, arg_const, sc_aff=None, sc_const=0.0) -> dict:
    return {
        "coeff": coeff,
        "func": func,
        "arg_aff": arg_aff,
        "arg_const": arg_const,
        "sc_aff": sc_aff,
        "sc_const": sc_const,
    }


# ── perspective recognition (#865) ────────────────────────────────────────────


def _affine_of(node, offsets) -> _Decomp:
    """Decompose `node`, requiring it to be purely affine (no composite terms)."""
    d = _decompose(node, offsets)
    if d.terms:
        raise NotConvexKernel("non-affine factor")
    return d


def _same_affine(a: _Decomp, b: _Decomp, tol: float = 1e-12) -> bool:
    """True iff two affine forms are the same expression (coefficient-wise)."""
    if abs(a.const - b.const) > tol:
        return False
    for col in set(a.aff) | set(b.aff):
        if abs(a.aff.get(col, 0.0) - b.aff.get(col, 0.0)) > tol:
            return False
    return True


def _try_perspective(node, offsets) -> _Decomp:
    """Decompose `L * R` as a perspective, or raise ``bilinear product``.

    One factor must be an affine `s`; the other must decompose in "ratio space"
    (every occurrence of a variable divided by that same `s`). Multiplying the
    ratio-space form back by `s` clears every division, leaving affine terms plus
    perspective terms — see the module docstring.
    """
    for scale_node, inner_node in ((node.left, node.right), (node.right, node.left)):
        try:
            s = _affine_of(scale_node, offsets)
        except NotConvexKernel:
            continue
        if not s.aff:  # a constant `s` is the plain scaling case, handled above
            continue
        try:
            return _lift_by_scale(_decompose_over(inner_node, s, offsets), s)
        except NotConvexKernel:
            continue
    raise NotConvexKernel("bilinear product")


def _decompose_over(node, s: _Decomp, offsets) -> _Decomp:
    """Decompose `node` in *ratio space* relative to the affine scale `s`.

    The returned ``_Decomp`` is read with its ``aff`` coefficients applying to the
    ratios ``x_col / s`` rather than to ``x_col`` (``const`` and the term arguments'
    affine parts follow the same convention). A bare variable — anything NOT under a
    ``/ s`` — is a genuine bilinear factor and raises.
    """
    d = _Decomp()
    c = _as_const(node)
    if c is not None:
        d.const = c
        return d
    if isinstance(node, UnaryOp):
        if node.op == "neg":
            return _decompose_over(node.operand, s, offsets).scale(-1.0)
        raise NotConvexKernel(f"unary {node.op}")
    if isinstance(node, BinaryOp):
        if node.op == "+":
            return _decompose_over(node.left, s, offsets).add(
                _decompose_over(node.right, s, offsets)
            )
        if node.op == "-":
            return _decompose_over(node.left, s, offsets).add(
                _decompose_over(node.right, s, offsets).scale(-1.0)
            )
        if node.op == "*":
            lc, rc = _as_const(node.left), _as_const(node.right)
            if lc is not None:
                return _decompose_over(node.right, s, offsets).scale(lc)
            if rc is not None:
                return _decompose_over(node.left, s, offsets).scale(rc)
            raise NotConvexKernel("bilinear product")
        if node.op == "/":
            rc = _as_const(node.right)
            if rc is not None and rc != 0.0:
                return _decompose_over(node.left, s, offsets).scale(1.0 / rc)
            den = _affine_of(node.right, offsets)
            if not _same_affine(den, s):
                raise NotConvexKernel("division by non-constant")
            num = _affine_of(node.left, offsets)
            # `(a·x + b)/s` would contribute `b/s`, which is not a ratio of the
            # admitted form; only a constant-free numerator is representable.
            if abs(num.const) > 1e-12:
                raise NotConvexKernel("perspective numerator has a constant")
            d.aff = dict(num.aff)
            return d
        if node.op == "**":
            return _pow_as_sqr(node, lambda nd, off: _decompose_over(nd, s, off), offsets)
        raise NotConvexKernel(f"binary {node.op}")
    if isinstance(node, FunctionCall):
        if node.func_name not in _FUNC:
            raise NotConvexKernel(f"unsupported func {node.func_name}")
        if len(node.args) != 1:
            raise NotConvexKernel(f"multi-arg func {node.func_name}")
        arg = _decompose_over(node.args[0], s, offsets)
        if arg.terms:
            raise NotConvexKernel("non-affine function argument")
        d.terms.append(_term(1.0, node.func_name, arg.aff, arg.const))
        return d
    # A `Variable`/`IndexExpression` reached here is NOT under a `/s`, so the
    # product really is bilinear.
    raise NotConvexKernel("bilinear product")


def _lift_by_scale(r: _Decomp, s: _Decomp) -> _Decomp:
    """Multiply a ratio-space decomposition `r` by its scale `s`.

    * ``Σ k·(x_c/s) · s`` → the plain affine ``Σ k·x_c``;
    * ``b · s`` → the affine ``b·s``;
    * ``k·f(Σ a_j (x_j/s) + q) · s`` → the perspective ``k·s·f(A/s)`` with the
      affine numerator ``A = Σ a_j x_j + q·s``, since ``Σ a_j x_j/s + q = A/s``.
    """
    d = _Decomp()
    d.aff = dict(r.aff)
    if r.const:
        for col, k in s.aff.items():
            d.aff[col] = d.aff.get(col, 0.0) + r.const * k
        d.const += r.const * s.const
    for t in r.terms:
        q = t["arg_const"]
        arg_aff = dict(t["arg_aff"])
        for col, k in s.aff.items():
            arg_aff[col] = arg_aff.get(col, 0.0) + q * k
        d.terms.append(_term(t["coeff"], t["func"], arg_aff, q * s.const, dict(s.aff), s.const))
    return d


# Below this the perspective `s·f(a/s)` is numerically meaningless (and `1/s`
# unusable), so a box that cannot prove `s` above it is a refusal, not a guess.
_MIN_SCALE = 1e-9


def _aff_interval(aff: dict[int, float], const: float, lb, ub) -> tuple[float, float]:
    """Interval of `aff·x + const` over the box `[lb, ub]` (may be ±inf)."""
    lo = hi = const
    for col, k in aff.items():
        if k == 0.0:
            continue
        if k > 0.0:
            lo += k * lb[col]
            hi += k * ub[col]
        else:
            lo += k * ub[col]
            hi += k * lb[col]
    return lo, hi


def _assert_convex_le(d: _Decomp, lb, ub) -> None:
    """Every term of a `≤`-normal-form g must be provably convex, else raise."""
    for t in d.terms:
        _val, curv = _FUNC[t["func"]]
        # convex iff sign(coeff)*curvature >= 0 (coeff==0 term is trivially fine).
        # This rule is the same for a plain term and for its perspective: the
        # perspective of a convex function is jointly convex on `s > 0`.
        if t["coeff"] * curv < -1e-15:
            raise NotConvexKernel(
                f"nonconvex term coeff={t['coeff']:+.3g} func={t['func']} (curv={curv:+d})"
            )
        if t["sc_aff"] is None:
            continue
        # `s > 0` is the perspective's convexity precondition — PROVE it on the box.
        s_lo, _s_hi = _aff_interval(t["sc_aff"], t["sc_const"], lb, ub)
        if not (s_lo >= _MIN_SCALE):
            raise NotConvexKernel(f"perspective scale not provably positive (lo={s_lo:.3g})")
        # With `s > 0`, `sign(a/s) == sign(a)`, so the domain of `f(a/s)` reduces to
        # a condition on the numerator's interval (log1p needs `a/s > −1` ⟺ `a+s > 0`).
        a_lo, _a_hi = _aff_interval(t["arg_aff"], t["arg_const"], lb, ub)
        func = t["func"]
        if func == "log" and not (a_lo > 0.0):
            raise NotConvexKernel(f"perspective log argument not provably positive ({a_lo:.3g})")
        if func == "sqrt" and not (a_lo >= 0.0):
            raise NotConvexKernel(f"perspective sqrt argument not provably >= 0 ({a_lo:.3g})")
        if func == "log1p":
            sum_aff = dict(t["arg_aff"])
            for col, k in t["sc_aff"].items():
                sum_aff[col] = sum_aff.get(col, 0.0) + k
            lo, _hi = _aff_interval(sum_aff, t["arg_const"] + t["sc_const"], lb, ub)
            if not (lo > 0.0):
                raise NotConvexKernel(f"perspective log1p argument out of domain ({lo:.3g})")


def build_convex_spec(model, bounds=None) -> Optional[dict]:
    """Marshal `model` into the flat arrays for `solve_convex_tree_py`, or `None`.

    Returns ``None`` (→ keep the NLP-BB path) whenever the model is not provably a
    convex composite-of-affine MINLP per the soundness gate in the module docstring.
    """
    try:
        return _build(model, bounds)
    except NotConvexKernel:
        return None


def _build(model, bounds) -> dict:
    from discopt._relax.gdp_reformulate import reformulate_gdp
    from discopt._relax.model_utils import flat_variable_bounds
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.modeling.core import VarType

    m = reformulate_gdp(model, method="big-m")
    lb, ub = flat_variable_bounds(m)
    n = len(lb)
    lb = lb.astype(float)
    ub = ub.astype(float)

    is_int = np.zeros(n, bool)
    k = 0
    for v in m._variables:
        for _ in range(v.size):
            if v.var_type in (VarType.BINARY, VarType.INTEGER):
                is_int[k] = True
            k += 1

    ev = make_evaluator(m)  # #1063: canonical funnel, not the JAX ctor
    senses = [c.sense if isinstance(c.sense, str) else c.sense.value for c in m._constraints]
    if m._objective is None or m._objective.sense.name not in ("MAXIMIZE", "MINIMIZE"):
        raise NotConvexKernel("no usable objective")
    sense_max = m._objective.sense.name == "MAXIMIZE"

    # Objective must be LINEAR (constant gradient) to be an LP objective.
    rng = np.random.default_rng(0)
    lo = np.where(np.isfinite(lb), lb, 0.0)
    hi = np.where(np.isfinite(ub), ub, lo + 5.0)
    xa = lo + rng.random(n) * (hi - lo)
    xb = lo + rng.random(n) * (hi - lo)
    ga = np.asarray(ev.evaluate_gradient(xa), float)
    gb = np.asarray(ev.evaluate_gradient(xb), float)
    if not np.allclose(ga, gb, atol=1e-9):
        raise NotConvexKernel("nonlinear objective")
    negate = bool(getattr(ev, "_negate", sense_max))
    c = (-ga if negate else ga).astype(float)
    # ... and the objective's CONSTANT term. ``c`` is the gradient, so the
    # kernel optimises ``c @ x`` while the model's objective is ``c @ x + f(0)``.
    # Dropping it does not merely weaken a bound -- the kernel's incumbent IS
    # the reported ``objective``, so a model with a constant came back with a
    # CERTIFIED WRONG optimal value. Measured on ``min -3x + 4y + K`` s.t.
    # ``exp(x) <= 20y`` with ``K = -4321.5``: reported objective
    # -4.987196820759 at a point whose true objective is -4326.487196821, with
    # ``bound`` -4.987196820767 (above the true optimum) and
    # ``gap_certified=True``. Same defect as the one fixed in
    # ``solvers/_root_cuts.py``; found by auditing that fix's class.
    #
    # Equal constants at BOTH sampled points is the affine check the gradient
    # comparison alone cannot make: two gradients can coincide on a nonlinear
    # objective, and then ``c @ x + c0`` is not the objective at all.
    _c0a = float(ev.evaluate_objective(xa)) - float(ga @ xa)
    _c0b = float(ev.evaluate_objective(xb)) - float(gb @ xb)
    if not (np.isfinite(_c0a) and np.isfinite(_c0b)) or abs(_c0a - _c0b) > 1e-9 * max(
        1.0, abs(_c0a), abs(_c0b)
    ):
        raise NotConvexKernel("objective is not affine")
    obj_const = float(-_c0a if negate else _c0a)

    # Classify rows linear (constant Jacobian) vs nonlinear.
    ja = ev.evaluate_jacobian(xa)
    jb = ev.evaluate_jacobian(xb)
    lin_rows = np.all(np.isclose(ja, jb, atol=1e-9), axis=1)
    g0 = np.asarray(ev.evaluate_constraints(xa), float)
    const = g0 - ja @ xa
    offsets = _flat_offsets(m)

    le_rows, eq_rows = [], []  # each: (cols, coeffs, rhs)
    nl_specs = []  # convex rows: (lin_aff, lin_const, terms, rhs=0)
    for i in range(ja.shape[0]):
        s = senses[i]
        if lin_rows[i]:
            a = np.asarray(ja[i], float)
            ci = float(const[i])
            if s == "<=":
                le_rows.append((a, -ci))
            elif s == ">=":
                le_rows.append((-a, ci))
            else:
                eq_rows.append((a, -ci))
            continue
        # Nonlinear row: decompose g_i (constraint g_i {sense} 0) and gate convexity.
        if s not in ("<=", ">="):
            raise NotConvexKernel("nonlinear equality constraint")
        expr = _constraint_expr(m, i)
        d = _decompose(expr, offsets)
        if s == ">=":  # normalize g ≥ 0 → (−g) ≤ 0
            d.scale(-1.0)
        _assert_convex_le(d, lb, ub)
        if not d.terms:
            # The row's nonlinearity cancelled under decomposition (a perspective
            # whose every term was a bare ratio, e.g. `ε·(x/ε) ≤ 0`). It is exactly
            # linear now — emit it as a linear row rather than a term-less "convex"
            # one, so the kernel sees it in its natural form.
            a = np.zeros(n)
            # `coef`, not `k`: `k` is the integer column counter above, and reusing
            # it for a float coefficient is a type error mypy (correctly) flags.
            for col, coef in d.aff.items():
                a[col] = coef
            le_rows.append((a, -d.const))
            continue
        nl_specs.append(d)

    if not nl_specs:
        # #1346: a model with NO nonlinear row is an LP or a MILP, and it belongs to
        # the routes built for it -- `lp_milp_highs.py` (HiGHS with discopt-verified
        # certificates, `docs/dev/lp-milp-highs-routing-plan.md`) and the Rust MILP
        # engine -- not to an outer-approximation tree that has nothing to
        # outer-approximate. It would "qualify" trivially: a linear objective and
        # zero nonlinear rows pass every clause of the convexity gate, so before this
        # refusal the gate claimed every pure LP and MILP in the library.
        #
        # This was invisible while the flag was opt-in and became a routing hijack
        # the moment it graduated: 17 smoke tests failed, almost all of them MILP or
        # HiGHS-route tests, because `Model.solve()` consulted the kernel first. The
        # §5 panel did NOT catch it -- the in-repo `.nl` corpus classified only 3
        # instances eligible and contains no pure LP/MILP member, so a corpus-wide
        # panel can pass while the change breaks a route the corpus never exercises.
        # The corpus bounds what a panel can see; it is not a proof of safety.
        raise NotConvexKernel("no nonlinear row: an LP/MILP belongs to the LP/MILP route")
    if not is_int.any():
        # #1346, the same defect from the other side: a model with no integer
        # variable is a continuous convex NLP, not a MINLP. A branch-and-cut tree
        # has nothing to branch on, and -- the part that bites -- the kernel's
        # ``SolveResult`` carries **no duals**, while the NLP path returns them.
        #
        # Routing one therefore silently drops `constraint_duals` and
        # `bound_duals_upper` from a result that used to carry them, which is
        # exactly the regression #1037 was opened to fix. Caught by the HiGHS-route
        # CI lane: 8 failures in `test_solver_duals.py`, all "duals were withheld",
        # on the ``minimize -x s.t. x^2 + y^2 <= r^2`` model -- linear objective, one
        # convex row, zero integers, claimed by every other clause of the gate.
        #
        # NOT fixable by handing back the tree's LP duals: those are multipliers of
        # the OUTER APPROXIMATION, not of the model's own nonlinear row. Reporting
        # them would be "duals of a different problem" -- the precise thing #1037's
        # withholding message exists to prevent -- so the sound fix is to leave the
        # model on the path that can produce real ones.
        raise NotConvexKernel("no integer variable: a continuous convex NLP keeps the NLP path")
    return _marshal(n, c, sense_max, is_int, lb, ub, le_rows, eq_rows, nl_specs, obj_const)


def _constraint_expr(model, row_idx):
    con = model._constraints[row_idx]
    for attr in ("expr", "body", "lhs"):
        e = getattr(con, attr, None)
        if e is not None:
            return e
    raise NotConvexKernel("cannot locate constraint expression")


def _csr_from_rows(rows, n):
    ptr, cols, vals, rhs = [0], [], [], []
    for a, r in rows:
        a = np.asarray(a, float)
        nz = np.where(np.abs(a) > 1e-13)[0]
        cols.extend(nz.tolist())
        vals.extend(a[nz].tolist())
        ptr.append(len(cols))
        rhs.append(float(r))
    return (
        np.asarray(ptr, np.int64),
        np.asarray(cols, np.int64),
        np.asarray(vals, float),
        np.asarray(rhs, float),
    )


def _affine_csr(items):
    cs = sorted(items)
    return np.asarray(cs, np.int64), np.asarray([items[c] for c in cs], float)


def _marshal(n, c, sense_max, is_int, lb, ub, le_rows, eq_rows, nl_specs, obj_const=0.0) -> dict:
    le_ptr, le_cols, le_coeffs, le_rhs = _csr_from_rows(le_rows, n)
    eq_ptr, eq_cols, eq_coeffs, eq_rhs = _csr_from_rows(eq_rows, n)

    nl_rhs, nl_lin_const = [], []
    nl_lin_ptr, nl_lin_cols, nl_lin_coeffs = [0], [], []
    nl_term_ptr = [0]
    term_coeff, term_func, term_arg_const = [], [], []
    term_arg_ptr, term_arg_cols, term_arg_coeffs = [0], [], []
    # Perspective scale per term; an empty CSR row with a zero constant marks a
    # plain composite term (#865).
    term_scale_const = []
    term_scale_ptr, term_scale_cols, term_scale_coeffs = [0], [], []
    for d in nl_specs:
        lc, lk = _affine_csr(d.aff)
        nl_lin_cols.extend(lc.tolist())
        nl_lin_coeffs.extend(lk.tolist())
        nl_lin_ptr.append(len(nl_lin_cols))
        nl_lin_const.append(d.const)
        nl_rhs.append(0.0)
        for t in d.terms:
            term_coeff.append(t["coeff"])
            term_func.append(_FUNC_CODE[t["func"]])
            term_arg_const.append(t["arg_const"])
            ac, ak = _affine_csr(t["arg_aff"])
            term_arg_cols.extend(ac.tolist())
            term_arg_coeffs.extend(ak.tolist())
            term_arg_ptr.append(len(term_arg_cols))
            sc_aff = t["sc_aff"] or {}
            sc, sk = _affine_csr(sc_aff)
            term_scale_cols.extend(sc.tolist())
            term_scale_coeffs.extend(sk.tolist())
            term_scale_ptr.append(len(term_scale_cols))
            term_scale_const.append(t["sc_const"] if t["sc_aff"] is not None else 0.0)
        nl_term_ptr.append(len(term_coeff))

    return dict(
        n=n,
        c=np.asarray(c, float),
        # NOT a kernel input: ``solve_convex_tree`` pops it and adds it back to
        # the kernel's incumbent/bound (see ``_build``).
        obj_const=float(obj_const),
        integrality=np.asarray(is_int, np.int64),
        lo=np.asarray(lb, float),
        hi=np.asarray(ub, float),
        sense_max=bool(sense_max),
        le_row_ptr=le_ptr,
        le_cols=le_cols,
        le_coeffs=le_coeffs,
        le_rhs=le_rhs,
        eq_row_ptr=eq_ptr,
        eq_cols=eq_cols,
        eq_coeffs=eq_coeffs,
        eq_rhs=eq_rhs,
        nl_rhs=np.asarray(nl_rhs, float),
        nl_lin_const=np.asarray(nl_lin_const, float),
        nl_lin_ptr=np.asarray(nl_lin_ptr, np.int64),
        nl_lin_cols=np.asarray(nl_lin_cols, np.int64),
        nl_lin_coeffs=np.asarray(nl_lin_coeffs, float),
        nl_term_ptr=np.asarray(nl_term_ptr, np.int64),
        term_coeff=np.asarray(term_coeff, float),
        term_func=np.asarray(term_func, np.int64),
        term_arg_const=np.asarray(term_arg_const, float),
        term_arg_ptr=np.asarray(term_arg_ptr, np.int64),
        term_arg_cols=np.asarray(term_arg_cols, np.int64),
        term_arg_coeffs=np.asarray(term_arg_coeffs, float),
        term_scale_const=np.asarray(term_scale_const, float),
        term_scale_ptr=np.asarray(term_scale_ptr, np.int64),
        term_scale_cols=np.asarray(term_scale_cols, np.int64),
        term_scale_coeffs=np.asarray(term_scale_coeffs, float),
    )


def convex_kernel_enabled() -> bool:
    """`DISCOPT_CONVEX_KERNEL` opt-out (default-ON since #1346).

    Graduated under the CLAUDE.md §5 Regime-2 gate by
    ``discopt_benchmarks/scripts/issue1346_convex_kernel_graduation_panel.py`` over
    the 66-instance in-repo corpus, arms interleaved within each instance and the
    arm order alternated by index, ``deterministic=True``. Gate 1 cert-clean PASS
    (0 unsound bounds, 0 certification regressions, 0 objective drift, 0 errors);
    gate 2 net-positive PASS on the routed class::

        clay0303hfsg   off  feasible/UNCERTIFIED 29911.20 (12.2% above opt)  90.2 s
                       on   optimal/CERTIFIED    26669.1096   149 nodes      21.2 s
        syn05hfsg      off  optimal  277 nodes  23.8 s -> on  optimal  2 nodes  0.01 s

    Node counts, not wall, carry that result: 277 -> 2 is structural, and both
    outcomes reproduced across four independent runs. Full numbers, and the
    concentration caveat (only 3 of 66 in-repo instances are eligible at all), in
    ``docs/dev/convex-kernel-plan.md``.

    **No counter-case guard ships with this, and that is a measured decision.** A
    two-stage probe guard was built for the ``watercontamination0202`` case that
    ``sota-parity-analysis-2026-07-27.md`` G-C records at 2001 s with no bound. Run
    against the actual instance it turns out to be refused by this gate's *existing*
    ``nonlinear objective`` clause, in 3.8 s, with or without the guard -- G-C's
    "convex/MIQP route" is the problem classifier's route, not this kernel. The
    guard was defending against a threat that cannot reach here, cost +2.6 s on
    ``clay0303hfsg``, and bought nothing measurable anywhere, so it was deleted
    rather than shipped (§4: no fix ships on a hypothesis; the
    ``DISCOPT_CUT_INHERIT`` lesson: sound is not the same as helpful).

    **Why this was default-OFF for so long, since the history misleads.** It was not
    a failed panel. ``#798`` proved both §5 bars on the convex family and the
    66-instance Regime-2 panel came back cert-clean; ``#800``'s close-out then
    deferred graduation to ``#807`` (native-warm-LP *SCIP wall parity*, ~2 s vs the
    kernel's 7-80 s). That is a **stretch goal strictly above** what §5 asks for --
    §5 scores ON against OFF, not against SCIP -- so the flag sat in the
    "indefinitely parked" state that the §5 retirement clause (#1345) exists to
    stop. #1346 re-ran the gate and acted on it. ``#807`` remains open as a
    performance issue; it is no longer a graduation gate.
    """
    return os.environ.get("DISCOPT_CONVEX_KERNEL", "1") not in ("0", "", "false", "False")


def keep_declined_bound_enabled() -> bool:
    """`DISCOPT_CONVEX_KERNEL_KEEP_BOUND` opt-out (default-ON, #1422).

    **§5 graduation panel, run 2026-09-22 before this shipped**
    (``scratchpad/ck_panel1422.py``): ON vs OFF, interleaved within each instance,
    over the 39 convex-kernel-eligible instances of the MINLPLib snapshot at an 8 s
    budget, **73 executed comparisons**.

    * *cert-clean* — **0 violations of every kind**: no bound above its reference
      optimum, no certification regression, no status change, no objective drift,
      no bound made looser.
    * *net-positive* — **16 of 39 gain a dual bound where the OFF arm has none**
      (0 lost, 0 loosened). Total wall 190.7 s both arms, **-0.0%**: the mechanism
      reads a number the tree already computed, so it cannot cost time.

    Graduated on introduction on that panel, per the ``DISCOPT_FARKAS_RAY_CLEANUP``
    / ``DISCOPT_TREE_SENTINEL_PRUNE_GUARD`` precedent. The ``=0`` opt-out and the
    legacy path are kept intact, as §5 requires.

    Gates whether ``Model.solve`` adopts the rigorous dual bound a **declined**
    convex-kernel attempt already proved (:func:`last_declined_bound`) when the
    default path finishes with a weaker bound or none at all.

    **What it fixes.** The attempt is granted ``min(time_limit,
    DISCOPT_CONVEX_KERNEL_BUDGET)`` -- the *whole* budget for any ``time_limit <=
    120``. On a model it then declines, the caller gets ~0 s of default path, its
    #654 deadline short-circuit, and ``bound=None`` -- while the kernel had in fact
    proved a bound and thrown it away. Measured over the 39 convex-kernel-eligible
    instances of the MINLPLib snapshot at an 8 s budget, **82% of all kernel wall
    (152 s of 185 s) is spent on models it declines**, and 15 of the 20 decliners
    carry a finite discarded bound.

    **Why this and not the two obvious alternatives**, both eliminated by
    measurement rather than taste:

    * *Cap the attempt to a fraction of the budget* -- built and rejected under
      #911: the kernel needs ~0.8 of a tight budget exactly where it wins.
    * *Abandon an attempt holding no incumbent* -- falsified for #1422. The kernel
      finds its first incumbent essentially **at convergence**: 10 of 19 certifiers
      at >=97% of their own wall, 16 of 19 past 50%. Simulated at a 50%-of-budget
      threshold it destroys certifications in the tight-budget regime (3 of 16 at
      ``tl=4 s``). "No incumbent yet" does not predict "will not certify".

    Adopting the bound takes **no time from the attempt**, so unlike both of those
    it cannot cost a certification. It does not fix the allocation -- that remains
    open on #1422 -- it stops the allocation being paid for nothing.

    **Soundness.** A declined attempt has no verified incumbent, so #779's
    cross-check is unavailable and this is a genuine trust step, taken on evidence:
    every declined bound on those 39 instances was checked against ``minlplib.solu``
    (``scratchpad/ck_soundness.py``) -- **12 oracle-backed comparisons, 0 bounds
    above their reference optimum**, 19/19 certifiers matching. Two guards ship with
    it regardless, in ``Model.solve``: the bound is adopted only when it is *better*
    than what the default path proved, and it is **rejected loudly** (logged at
    ERROR, result unchanged) if it crosses an incumbent the solve actually found,
    which is a live contradiction test on every solve that finds one. It is never
    used to prune, so it cannot cut off an optimum even if wrong.

    **The bounds are valid but of varying quality, and that is measured, not
    assumed.** They range from nearly worthless -- ``p_ball_20b_5p_2d_m`` proves
    ~0 against a true optimum of 2.437 -- to substantial: ``clay0303hfsg`` proves
    19188.0 against an optimum of 26669.1, and ``ball_mk2_30`` proves -28.8789
    where the solve reports ``bound=None`` today. The case for adopting them rests
    on their being **free**, not on their being tight, and an earlier claim on
    #1422 that the ``p_ball_*`` bounds were "essentially the exact bound" was
    retracted there: it assumed those optima were 0 without consulting the oracle.

    Note what this is NOT a substitute for. Disabling the kernel outright
    (``DISCOPT_CONVEX_KERNEL=0``) hands ``ball_mk2_30``'s whole 8 s budget to the
    default path, which then proves **-27.8836** -- tighter than the -28.8789
    recovered here. The comparison that matters is against the *same*
    configuration, where the attempt runs and declines and the answer is ``None``;
    the allocation problem that makes the default path's better bound unreachable
    is #1422's item 4 and remains open.
    """
    return os.environ.get("DISCOPT_CONVEX_KERNEL_KEEP_BOUND", "1") not in (
        "0",
        "",
        "false",
        "False",
    )


def dominated_cols_enabled() -> bool:
    """`DISCOPT_CVX_DOMINATED_COLS` opt-out (default-ON inside the kernel, #879).

    Gates the dominated-cost-column upper bound. Unlike FBBT this is an
    OPTIMALITY-based reduction — it keeps an optimal solution, not every feasible
    point (see ``ConvexKernelSpec::tighten_dominated_columns``) — so it keeps its
    own switch on top of the kernel's default-off gate. It is ON by default because
    an infinite structural upper bound is what makes a node LP break down and its
    Neumaier–Shcherbina safe bound decline: measured on `clay0303hfsg`, turning it
    off takes the instance from `optimal` back to `exhausted`, and it is
    bit-identical (no-op) on every other in-repo instance the kernel routes.
    """
    return os.environ.get("DISCOPT_CVX_DOMINATED_COLS", "1") not in ("0", "", "false", "False")


def solve_convex_tree(spec: dict, *, time_limit_s: Optional[float] = None, **cfg) -> dict:
    """Run the native convex kernel on a marshaled `spec` (from build_convex_spec).

    The single chokepoint for the objective's constant term. ``spec["c"]`` is the
    objective GRADIENT, so the kernel optimises ``c @ x`` while the model's
    objective is ``c @ x + obj_const``; the constant is popped here (it is not a
    kernel input) and added back to the incumbent and the bound on the way out.
    Both must be corrected: the incumbent IS the reported ``objective``, and a
    bound left on the shifted scale is a false dual bound. Doing it here rather
    than at the call site means a new caller cannot forget it.

    ``initial_incumbent`` is given in the MODEL's objective values, so it is
    shifted the other way before it goes in.
    """
    import discopt._rust as _rust

    spec = dict(spec)
    obj_const = float(spec.pop("obj_const", 0.0))
    _init_inc = cfg.get("initial_incumbent", None)
    if _init_inc is not None:
        _init_inc = float(_init_inc) - obj_const

    params = dict(
        max_nodes=cfg.get("max_nodes", 100000),
        gap_tol=cfg.get("gap_tol", 1e-4),
        int_tol=cfg.get("int_tol", 1e-5),
        oa_tol=cfg.get("oa_tol", 1e-6),
        max_oa_rounds=cfg.get("max_oa_rounds", 60),
        max_sep_rounds=cfg.get("max_sep_rounds", 12),
        fbbt_rounds=cfg.get("fbbt_rounds", 20),
        dominated_cols=cfg.get("dominated_cols", dominated_cols_enabled()),
        initial_incumbent=_init_inc,
        time_limit_s=time_limit_s,
    )
    result: dict = dict(_rust.solve_convex_tree_py(**spec, **params))
    if obj_const:
        for _key in ("incumbent", "bound"):
            _v = result.get(_key)
            if _v is not None:
                result[_key] = float(_v) + obj_const
    return result


# Wall spent on the LAST convex-kernel attempt on this thread, whether or not the
# attempt was adopted (#911). ``Model.solve`` deducts it from the budget it hands the
# default path, so a DECLINED attempt can no longer make a ``time_limit=T`` solve run
# for ~2T. Thread-local because ``Model.solve`` may run concurrently on several
# threads and a process-global counter would let one solve deduct another's attempt.
class _AttemptClock(threading.local):
    def __init__(self) -> None:
        self.seconds = 0.0
        # The native-tree share of ``seconds`` (#1422). ``Model.solve`` bills the
        # whole attempt to ``SolveResult.wall_time`` and needs this split to keep
        # the documented ``rust_time + python_time == wall_time`` partition true.
        self.rust_seconds = 0.0
        # Rigorous dual bound proved by a DECLINED attempt, in the MODEL's objective
        # values, or None (#1422). See :func:`last_declined_bound`.
        self.declined_bound: Optional[float] = None


_ATTEMPT = _AttemptClock()


def last_attempt_seconds() -> float:
    """Wall of the last convex-kernel attempt on this thread, in seconds.

    **Exactly 0.0 when the flag is off**, which is load-bearing rather than cosmetic:
    ``Model.solve`` subtracts this from the budget it passes to ``solve_model``, so a
    nonzero reading on the default path would perturb every deadline-sensitive
    decision in the solver. :func:`try_convex_solve` therefore resets it to 0.0 on
    entry and starts the clock only *after* the flag check, so a flag-off solve
    subtracts a literal zero and stays bit-identical.

    A thread that has never run an attempt reads 0.0: ``threading.local`` re-runs a
    subclass's ``__init__`` the first time the object is touched on each thread.
    """
    return float(_ATTEMPT.seconds)


def last_declined_bound() -> Optional[float]:
    """Dual bound proved by the last DECLINED attempt on this thread, or ``None``.

    #1422. The kernel's result is adopted only when it fully certifies optimality
    (:func:`try_convex_solve`), so on every other outcome the tree's rigorous dual
    bound was computed and then thrown away — while the attempt had already spent
    the caller's whole budget, leaving the default path ~0 s and no bound of its
    own. ``ball_mk2_30`` is the issue's own instance: the kernel proves ``-28.8789``
    and ``Model.solve`` reports ``bound=None``.

    **Published only for ``time_limit`` / ``node_limit``** — the two outcomes whose
    ``bound`` is the running minimum over *open* nodes, i.e. a genuine dual bound on
    a tree that simply ran out of budget. ``exhausted`` (numerical non-closure) and a
    certified-but-unverifiable incumbent are deliberately excluded: in both the
    kernel is already telling us something is off, and a bound is not worth taking
    from a run we have just decided not to trust.

    Soundness evidence (entry experiment, ``scratchpad/ck_soundness.py``): every
    declined bound on the 39 convex-kernel-eligible instances of the MINLPLib
    snapshot, checked against ``minlplib.solu``. **12 oracle-backed comparisons, 0
    bounds above their reference optimum**, plus 19/19 certifiers matching the
    oracle. Their quality varies and the range is measured, not assumed: from
    nearly worthless (``p_ball_20b_5p_2d_m`` proves ~0 against a true optimum of
    2.437) to substantial (``clay0303hfsg`` proves 19188.0 against 26669.1). See
    :func:`keep_declined_bound_enabled` for why that does not weaken the case, and
    for the measurement showing this is no substitute for fixing the allocation —
    with the kernel disabled entirely, ``ball_mk2_30``'s default path proves
    **-27.8836**, tighter than the -28.8789 recovered here.

    In the MODEL's objective values (``solve_convex_tree`` adds the objective
    constant back to ``bound`` at its single chokepoint), so no caller has to
    re-shift it. Reset to ``None`` on entry to every attempt, so a stale reading
    from a previous solve on this thread can never be adopted.
    """
    return _ATTEMPT.declined_bound


def last_attempt_rust_seconds() -> float:
    """Native-tree share of :func:`last_attempt_seconds`, in seconds (#1422).

    The attempt is spec build + convexity classification (Python/JAX), the native
    convex tree (Rust), and the #779 incumbent verification (Python/JAX). ``Model.solve``
    bills the whole attempt to ``SolveResult.wall_time``; without this split it would
    have to guess which side of the documented ``rust_time``/``python_time`` partition
    to charge, and the partition is documented as exact. Always ``<=``
    :func:`last_attempt_seconds`, and exactly 0.0 when no tree ran (flag off, or a
    model the spec builder declined).
    """
    return float(_ATTEMPT.rust_seconds)


def try_convex_solve(
    model, *, time_limit: float = 3600.0, gap_tolerance: float = 1e-4
) -> Optional[SolveResult]:
    """Route `model` to the native convex kernel, or return ``None`` to fall back.

    Scoped to the smaller/quickly-certifiable convex MINLPs (#798): the kernel gets
    a BOUNDED attempt (``min(time_limit, DISCOPT_CONVEX_KERNEL_BUDGET)``, default
    120 s) and its result is used ONLY when it fully **certifies optimality** and
    the incumbent is verified feasible against the pristine model (#779). Everything
    else — flag off, non-convex, not-certified-within-budget, no incumbent, or an
    unverifiable incumbent — returns ``None`` so the caller keeps the (always-correct)
    default path. Proven-infeasible roots are surfaced. Never reports an unsound or
    uncertified result.

    **Budget accounting (#911).** The attempt is bounded, but until this was fixed it
    was *additive*: ``Model.solve`` afterwards called ``solve_model`` with the
    caller's FULL ``time_limit``, so an eligible-but-uncertifiable model paid the
    attempt on top of its whole default budget. Measured OFF-vs-ON, interleaved, 2
    replicates (``issue911_convex_kernel_budget_entry.py``), medians:

    ==================  ======  ==========  ==========  =========
    instance            budget  OFF wall    ON before   ON after
    ==================  ======  ==========  ==========  =========
    clay0304hfsg          10 s   10.55 s     22.01 s     12.94 s
    clay0305hfsg          10 s   11.89 s     23.88 s     12.89 s
    clay0305hfsg          30 s   31.35 s     63.33 s     33.19 s
    clay0304hfsg          30 s   31.13 s    269.17 s     37.06 s
    ==================  ======  ==========  ==========  =========

    :func:`last_attempt_seconds` publishes the attempt wall and ``Model.solve``
    subtracts it. The spec build is inside the clock deliberately: it is the
    convexity classification, and on the instances this hazard bites it is itself
    ~1 s of wall. The ``clay0304hfsg`` 30 s cell needed a second fix as well — the
    native tree polled its own deadline only *between* nodes (see
    ``ConvexKernelSpec::solve_node_cut_until``); deduction alone would have moved it
    from 269 s to ~237 s, still 7.9x the stated limit.

    Every instance the kernel *certifies* is bit-unchanged by this: same objective and
    same node count on all four certifying cells of the panel.

    **Why the attempt is NOT capped to a fraction of the budget** (the first design,
    falsified before it was built): the kernel needs the large majority of a tight
    budget on exactly the instances where it wins. Measured here, ``clay0303hfsg``
    certifies in ~8 s and turns a 10 s OFF-arm ``time_limit`` (no incumbent) into a
    certified optimum; any fractional cap below ~0.8 of the budget gives that back.
    The fraction was dropped rather than shipped as a dead knob.
    """
    _ATTEMPT.seconds = 0.0
    _ATTEMPT.rust_seconds = 0.0
    # #1422: cleared BEFORE the flag check, so a flag-off solve (and any later solve
    # on this thread) can never read a bound left behind by an earlier attempt.
    _ATTEMPT.declined_bound = None
    if not convex_kernel_enabled():
        return None
    # Clock starts HERE, after the flag check, so a flag-off solve reads exactly 0.0
    # (see ``last_attempt_seconds``). The ``finally`` covers EVERY exit — including
    # the decline paths below and an exception mid-attempt, which still consumed
    # wall the caller has to pay for.
    _attempt_t0 = time.perf_counter()
    try:
        return _attempt_convex_solve(model, time_limit=time_limit, gap_tolerance=gap_tolerance)
    finally:
        _ATTEMPT.seconds = time.perf_counter() - _attempt_t0


def _attempt_convex_solve(
    model, *, time_limit: float, gap_tolerance: float
) -> Optional[SolveResult]:
    """The attempt itself; :func:`try_convex_solve` wraps it to clock every exit."""
    import os

    import numpy as np

    from discopt.modeling.core import SolveResult

    spec = build_convex_spec(model)
    if spec is None:
        return None

    budget = min(time_limit, float(os.environ.get("DISCOPT_CONVEX_KERNEL_BUDGET", "120")))
    t0 = time.perf_counter()
    try:
        r = solve_convex_tree(
            spec,
            time_limit_s=budget,
            gap_tol=gap_tolerance,
            initial_incumbent=None,
        )
    finally:
        # Publish the native tree's wall even when it raised: the caller still paid
        # for it and still has to bill it (#1422). Records only -- nothing is
        # swallowed, the exception propagates to ``try_convex_solve``'s own
        # ``finally`` exactly as before.
        _ATTEMPT.rust_seconds = time.perf_counter() - t0
    wall = time.perf_counter() - t0

    incumbent = r["incumbent"]
    inc_x = np.asarray(r["incumbent_x"], float)

    if r["status"] == "infeasible":
        return SolveResult(status="infeasible", bound=r["bound"], wall_time=wall, nlp_bb=False)
    # Use the kernel result ONLY when it CERTIFIED optimality within budget; any
    # limit / feasible-only / no-incumbent outcome defers to the default path, which
    # then gets the caller's budget MINUS what this attempt just spent (#911).
    if r["status"] != "optimal" or incumbent is None or inc_x.size == 0:
        # #1422: the tree's rigorous dual bound used to die here with the attempt,
        # after it had already spent the caller's whole budget. Publish it for
        # ``Model.solve`` to adopt (behind DISCOPT_CONVEX_KERNEL_KEEP_BOUND) -- but
        # ONLY from the two outcomes whose ``bound`` is a genuine minimum over open
        # nodes. See :func:`last_declined_bound` for the soundness evidence and for
        # why ``exhausted`` is excluded.
        if r["status"] in ("time_limit", "node_limit"):
            from discopt.solvers._gap import BOUND_INF

            _b = r.get("bound")
            # ``BOUND_INF`` is the solvers' shared "no bound yet" sentinel (1e19).
            # The kernel reports the sentinel rather than ``inf``, so a plain
            # ``isfinite`` check would adopt "no bound" as if it were one.
            if _b is not None and np.isfinite(_b) and abs(float(_b)) < BOUND_INF:
                _ATTEMPT.declined_bound = float(_b)
        return None
    status = "optimal"

    # Map the flat structural point back onto the ORIGINAL model's variables
    # (reformulation appends aux columns, so the original vars are a prefix), and
    # VERIFY feasibility against the pristine model — the #779 guard. Any violation
    # beyond tolerance ⇒ fall back rather than report an unsound incumbent.
    x_dict, x_flat = _unflatten(model, inc_x)
    if not _incumbent_is_feasible(model, x_flat):
        return None

    gap = None
    if incumbent not in (None, 0.0):
        gap = abs(incumbent - r["bound"]) / max(1.0, abs(incumbent))
    # The kernel's own termination test is relative, and it ran on the objective
    # BEFORE the constant term was added back (see ``solve_convex_tree``). A
    # shift leaves the absolute gap alone but changes the denominator, so a
    # certificate earned on the shifted scale is not automatically one on the
    # model's. Re-test it here on the values actually reported, and defer to the
    # default path rather than certify a gap the caller did not ask for.
    if gap is not None and gap > gap_tolerance:
        logger.debug(
            "convex kernel: relative gap %.3g on the model's objective scale exceeds "
            "the requested %.3g (the kernel converged on the pre-constant scale); "
            "deferring to the default path",
            gap,
            gap_tolerance,
        )
        return None
    return SolveResult(
        status=status,
        objective=float(incumbent),
        bound=float(r["bound"]),
        gap=gap,
        x=x_dict,
        wall_time=wall,
        node_count=int(r["node_count"]),
        gap_certified=(status == "optimal"),
        nlp_bb=False,
    )


def _unflatten(model, inc_x):
    """(dict name→array, flat original-var vector) from the kernel's structural x."""
    import numpy as np

    x_dict = {}
    flat = []
    off = 0
    for v in model._variables:
        vals = np.asarray(inc_x[off : off + v.size], float)
        off += v.size
        flat.extend(vals.tolist())
        x_dict[v.name] = vals.reshape(v.shape) if v.shape else vals.reshape(())
    return x_dict, np.asarray(flat, float)


def _incumbent_is_feasible(model, x_flat, tol: float = 1e-5) -> bool:
    """#779: evaluate the PRISTINE model's constraints at `x_flat`; True iff feasible.

    #908: this used to ``zip(g, model._constraints)`` — pairing per-ROW evaluator
    values against per-CONSTRAINT-OBJECT entries. On any vector constraint the two
    desynchronise (and ``zip`` silently truncates to the shorter), so it vouched for
    points violating a row outright; it also checked neither variable bounds nor
    integrality. It now delegates to the single verifier, which enumerates rows from
    the evaluator's own row map.

    ``tol`` is accepted for call-compatibility and ignored: the verifier keys its
    tolerance on each row's own scale rather than on a flat constant, which is what
    fixes the scale-blindness in both directions.
    """
    from discopt.validation.feasibility import verify_point

    return bool(verify_point(model, x_flat).ok)
