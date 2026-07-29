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

import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from discopt.modeling.core import SolveResult

import numpy as np

from discopt._env import env_bool, env_float
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
    from discopt._jax.gdp_reformulate import reformulate_gdp
    from discopt._jax.model_utils import flat_variable_bounds
    from discopt._jax.nlp_evaluator import NLPEvaluator
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

    ev = NLPEvaluator(m)
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

    return _marshal(n, c, sense_max, is_int, lb, ub, le_rows, eq_rows, nl_specs)


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


def _marshal(n, c, sense_max, is_int, lb, ub, le_rows, eq_rows, nl_specs) -> dict:
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
    """`DISCOPT_CONVEX_KERNEL` opt-in (default-OFF)."""
    return env_bool("DISCOPT_CONVEX_KERNEL", False)


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
    return env_bool("DISCOPT_CVX_DOMINATED_COLS", True)


def solve_convex_tree(spec: dict, *, time_limit_s: Optional[float] = None, **cfg) -> dict:
    """Run the native convex kernel on a marshaled `spec` (from build_convex_spec)."""
    import discopt._rust as _rust

    params = dict(
        max_nodes=cfg.get("max_nodes", 100000),
        gap_tol=cfg.get("gap_tol", 1e-4),
        int_tol=cfg.get("int_tol", 1e-5),
        oa_tol=cfg.get("oa_tol", 1e-6),
        max_oa_rounds=cfg.get("max_oa_rounds", 60),
        max_sep_rounds=cfg.get("max_sep_rounds", 12),
        fbbt_rounds=cfg.get("fbbt_rounds", 20),
        dominated_cols=cfg.get("dominated_cols", dominated_cols_enabled()),
        initial_incumbent=cfg.get("initial_incumbent", None),
        time_limit_s=time_limit_s,
    )
    result: dict = dict(_rust.solve_convex_tree_py(**spec, **params))
    return result


# Wall spent on the LAST convex-kernel attempt in this thread, whether or not the
# attempt was adopted (consolidation plan Phase 5.4). ``Model.solve`` deducts it
# from the budget it hands the default path, so a declined attempt can no longer
# make a ``time_limit=T`` solve run for ~2T.
class _AttemptClock(threading.local):
    def __init__(self) -> None:  # pragma: no cover - trivial
        self.seconds = 0.0


_ATTEMPT = _AttemptClock()


def last_attempt_seconds() -> float:
    """Wall of the last convex-kernel attempt on this thread, in seconds.

    **Exactly 0.0 when the flag is off**, and that is load-bearing rather than
    cosmetic: ``Model.solve`` subtracts this from the budget it passes to
    ``solve_model``, so a nonzero reading on the default path would perturb every
    deadline-sensitive decision in a Regime-N-visible way. :func:`try_convex_solve`
    therefore resets it to 0.0 on entry and only starts the clock *after* the
    flag check, so a flag-off solve subtracts a literal zero.
    """
    st = _ATTEMPT
    if not hasattr(st, "seconds"):  # pragma: no cover - fresh thread
        st.__init__()
    return float(st.seconds)


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
    default path. Proven-infeasible roots are surfaced.

    **Budget accounting (consolidation plan Phase 5.4).** The attempt is bounded,
    but until this was fixed it was *additive*: ``Model.solve`` called
    ``solve_model`` afterwards with the caller's FULL ``time_limit``, so an
    eligible-but-uncertifiable model paid the attempt on top of its whole default
    budget. Measured in-repo, ``clay0303hfsg`` at a **10 s** budget:
    ON 25.2 s (sd 0.12) vs OFF 13.5 s (sd 0.05), reproduced in both replicates —
    2.5x the stated limit. That is the mechanism behind the
    ``watercontamination0202`` counter-case Phase 5.4 names as the graduation
    blocker. :func:`last_attempt_seconds` publishes the attempt wall (spec build
    included — it is 2.34 s on ``clay0303hfsg`` and 1.16 s on
    ``cvxnonsep_psig40r``, i.e. not negligible) and ``Model.solve`` subtracts it.

    **Why the attempt is NOT capped to a fraction of the budget**, which was this
    card's first design: measured attempt costs on the four in-repo eligible
    instances are ``clay0303hfsg`` 41.9 s (spec 2.34 + tree 39.55, certifies),
    ``cvxnonsep_psig40r`` 1.16 s (declines at the root), ``syn05hfsg`` 0.93 s,
    ``syn05m`` 0.77 s. ``clay0303hfsg`` therefore needs ~93 % of a 45 s budget to
    certify, so ANY fractional cap below that turns the corpus's only
    certification win (OFF ``feasible`` -> ON ``optimal``) back into ``feasible``.
    The fraction was dropped rather than shipped as a dead knob; see the plan's §6.
    """
    import time

    import numpy as np

    from discopt.modeling.core import SolveResult

    _ATTEMPT.seconds = 0.0
    if not convex_kernel_enabled():
        return None
    # Clock starts HERE, after the flag check, so a flag-off solve reads exactly
    # 0.0 (see ``last_attempt_seconds``). It covers the spec build deliberately:
    # ``build_convex_spec`` is the convexity classification, and on the
    # counter-case class that classification is itself seconds of wall.
    _attempt_t0 = time.perf_counter()
    try:
        spec = build_convex_spec(model)
    finally:
        _ATTEMPT.seconds = time.perf_counter() - _attempt_t0
    if spec is None:
        return None

    budget = min(time_limit, env_float("DISCOPT_CONVEX_KERNEL_BUDGET", 120.0))
    t0 = time.perf_counter()
    r = solve_convex_tree(
        spec,
        time_limit_s=budget,
        gap_tol=gap_tolerance,
        initial_incumbent=None,
    )
    wall = time.perf_counter() - t0
    _ATTEMPT.seconds = time.perf_counter() - _attempt_t0

    incumbent = r["incumbent"]
    inc_x = np.asarray(r["incumbent_x"], float)

    if r["status"] == "infeasible":
        return SolveResult(status="infeasible", bound=r["bound"], wall_time=wall, nlp_bb=False)
    # Use the kernel result ONLY when it CERTIFIED optimality within budget; any
    # limit / feasible-only / no-incumbent outcome defers to the default path,
    # which then gets the caller's full time budget.
    if r["status"] != "optimal" or incumbent is None or inc_x.size == 0:
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
    """#779: evaluate the PRISTINE model at `x_flat`; True iff the point is feasible.

    Delegates to :func:`discopt.validation.feasibility.verify_point`, the repo's
    single verifier. The hand-rolled loop this replaced had four measured defects
    (transcripts in that module's header): it zipped flat evaluator rows against
    ``model._constraints`` — so it ACCEPTED a point violating row 2 of a size-3
    vector constraint by 5.0, raised ``AttributeError`` mid-loop on a model carrying
    an SOS constraint, ignored ``Constraint.rhs``, and never checked variable bounds
    or integrality at all. Its tolerance was also a flat absolute ``tol`` with no
    row-scale term, which rejects ``nvs22``'s certified optimum.

    ``tol`` keeps its meaning as the absolute floor (unchanged at 1e-5 for this
    caller) and is also the relative coefficient on the row scale, so an ordinary
    unit-scale row is judged exactly as before.
    """
    from discopt.validation.feasibility import verify_point

    return bool(verify_point(model, x_flat, abs_tol=tol).ok)
