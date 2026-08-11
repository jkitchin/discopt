"""Pure-binary multilinear exact linearization (Fortet/Glover) — issue #187.

A polynomial over binary-valued variables is *multilinear* on its feasible
points (``b**2 == b`` for ``b in {0,1}``), so every monomial
``coef * prod_i b_i`` can be replaced exactly by an auxiliary ``z`` with the
Fortet/Glover linearization

    z <= b_i          (for every factor i)
    z >= sum_i b_i - (n - 1)
    z >= 0

which forces ``z == prod_i b_i`` at every binary vertex (Fortet 1960; Glover &
Woolsey 1974; pattern P10 in ``design/relaxation-patterns.md``). Applying it to
every monomial of degree >= 2 turns the model into an **equivalent pure MILP**
whose optimum equals the original optimum — no spatial relaxation and no JAX
Jacobian is ever needed.

This is the architectural fix for the ``autocorr_bern*`` wall (issue #187): the
Bernasconi autocorrelation objective ``sum_k (sum_i s_i s_{i+k})**2`` over
``{0,1}``-valued variables expands to a ~3000-monomial quartic whose full-DAG
JAX Jacobian compile and sparsity walk each exceed a 30 s budget on their own.
All work here happens directly on the *factored* DAG with plain dict
arithmetic over monomials (never materializing an expanded expression DAG and
never touching sympy/JAX), and the reformulated model routes to the MILP
branch-and-bound.

Two exact encodings are combined:

1. **Squared integer-valued forms** (the autocorr structure). An objective
   addend ``c * E**2`` where ``E`` expands to a multilinear polynomial with
   *integer* coefficients over *integer-valued* variables takes only integer
   values, so ``E**2`` is described exactly at integer points by the secants of
   the parabola between consecutive integers:

       y == E_linearized,   t >= (2j+1)*y - j*(j+1)   for j = lo .. hi-1,

   where ``[lo, hi]`` is an interval bound on ``E``. At every attainable
   integer ``y`` the max of the secants equals ``y**2`` exactly, so every
   original solution stays feasible at its original objective value (validity)
   and the reformulated optimum is exact, *provided the optimization pressure
   pushes ``t`` down onto the secant envelope* — guaranteed when the addend
   sits in the objective with a min-effective positive coefficient (or the
   equivalent condition in the objective variable's defining row, see below).
   Between attainable points the chords of the convex parabola lie above it,
   which only *tightens* the continuous relaxation. Compared to flat expansion
   this needs O(range) rows per square instead of O(#monomials) rows total:
   autocorr_bern25-25 drops from ~15k rows to ~1.2k.

2. **Flat monomial linearization** (the general case). Everything not
   expressible as (1) — constraint bodies, squares with the wrong sign, or an
   objective already delivered in expanded form — is expanded to its
   multilinear normal form and every degree>=2 monomial is Fortet-linearized
   per the scheme above. Budgeted: a blow-up aborts and falls back.

The `.nl` "objvar" convention (objective = a bare free variable ``tau`` whose
value is pinned by one defining row ``a*tau + g(b) <sense> rhs``) is
recognized so encoding (1) applies inside that row exactly when the sign/sense
conditions make the row transmit the objective's minimization pressure to the
squares (``mu * (-c/a) > 0`` with ``mu = +1`` for MIN / ``-1`` for MAX, and
the row either an equality or the inequality that bounds ``tau`` on its
objective-improving side, with ``tau`` free, scalar, continuous, and appearing
nowhere else). Everything else falls back to (2).

Scope and soundness rules:

- ``{0,1}``-bounded ``INTEGER`` variables are recognized as binary-valued
  (``from_nl`` types MINLPLib's 0/1 columns as ``INTEGER``, issue #187
  correction 1) — the linearization only needs *values* in ``{0,1}``, which
  integrality plus the bounds guarantees.
- Every monomial of degree >= 2 must consist exclusively of binary-valued
  factors; a continuous or general-integer variable may appear **linearly**
  only. Anything else (transcendentals, fractional/negative powers, divisions
  by non-constants, vector expressions) aborts the pass, which then returns
  the model **unchanged** — the solver keeps its existing paths, so the pass
  never regresses a model it cannot handle exactly.
- The rewrite fires only when a term of total degree >= ``_MIN_DEGREE`` (3) is
  present. Degree-2-only binary models are already handled exactly per-term by
  McCormick on the existing paths; rerouting them wholesale is out of scope
  here (and would change behavior on the whole binary-quadratic class).
- Expansion and the emitted MILP are budgeted (``_MAX_MONOMIALS``,
  ``_MAX_PRODUCT_OPS``, ``_MAX_ROWS``): a blow-up aborts and falls back
  rather than emitting a MILP too large for the (dense-marshaling) in-house
  MILP engines.

The solver adopts the reformulation only under the same guard as the
integer-bilinear pass (a genuinely pure MILP per both ``classify_problem`` and
the DAG-walking ``classify_nonlinear_terms``), and routes it to the MILP
engine.

All DAG walks here are **iterative**: ``from_nl`` builds n-ary sums as
left-associative ``BinaryOp`` chains, so a ~3000-term objective is a
~3000-deep tree and any recursive walk would hit Python's recursion limit
(silently disabling the pass on exactly the models it targets). For the same
reason the rebuilt linear bodies use *balanced* sum trees, keeping downstream
recursive walkers (term classifier, LP extractor) at O(log n) depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    IndexExpression,
    Model,
    Objective,
    ObjectiveSense,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)

# A monomial is a frozenset of scalar-variable keys ``(var._index, elem)``;
# the empty frozenset is the constant term. Binary factors collapse via set
# union (``b**2 == b``); the expansion aborts if a *non*-binary factor would
# collapse (that would silently rewrite ``x**2 -> x``) or appears in a
# degree>=2 monomial (not multilinear-reducible).
_Mono = frozenset
_EMPTY: frozenset = frozenset()

# Fire only when a genuine degree>=3 term exists: n=2 products are already
# exact per-term under McCormick on the existing paths (see module docstring),
# and degree >= 3 is where the nested-bilinear relaxation is loose.
_MIN_DEGREE = 3
# Budget caps: abort (fall back to the existing paths) rather than emit an
# intractable MILP. The row cap is calibrated to the in-house MILP engines'
# dense LP marshaling (extract_lp_data materializes an (m, n+m) float64
# matrix): autocorr_bern25-25 needs ~1.6k rows under the squared-form encoding
# and fits comfortably; a 15k-row flat expansion of the same model would OOM.
_MAX_MONOMIALS = 200_000
_MAX_PRODUCT_OPS = 10_000_000
_MAX_ROWS = 6_000
# Largest constant integer exponent expanded by repeated multiplication.
_MAX_POW = 16
# Syntactic degree values in the cheap gate saturate here (avoids overflow on
# deep power towers; anything at the cap is "large enough" for the witness).
_DEG_CAP = 64
# A variable bound at/above this magnitude counts as free (matches discopt's
# effective-infinity sentinel).
_FREE_BOUND = 1e19


class _Unsupported(Exception):
    """Structure this pass cannot linearize exactly — abort and fall back."""


def _scalar_constant(node: Constant) -> float:
    v = node.value
    if v.ndim != 0 and v.size != 1:
        raise _Unsupported("non-scalar constant")
    return float(v.reshape(()))


def _binary_like(var: Variable, elem: int) -> bool:
    """True if ``var[elem]`` can only take values in ``{0, 1}``: declared
    BINARY, or INTEGER with bounds inside ``(-1, 2)`` (so the only integers in
    the box are 0/1). This is the {0,1}-integer recognition of issue #187."""
    if var.var_type == VarType.BINARY:
        return True
    if var.var_type != VarType.INTEGER:
        return False
    lo = float(np.asarray(var.lb).flat[elem])
    hi = float(np.asarray(var.ub).flat[elem])
    return lo > -1.0 and hi < 2.0


def _integer_valued(var: Variable) -> bool:
    return var.var_type in (VarType.BINARY, VarType.INTEGER)


def _leaf_ref(node: Expression) -> Optional[tuple[Variable, int, Expression]]:
    """If *node* references one scalar variable element, return
    ``(variable, flat_element, reference_expression)``, else ``None``."""
    if isinstance(node, Variable):
        if node.size == 1 and node.shape == ():
            return node, 0, node
        return None
    if isinstance(node, IndexExpression):
        base = node.base
        if not isinstance(base, Variable):
            return None
        idx = node.index
        if isinstance(idx, (int, np.integer)):
            flat = int(idx)
            if 0 <= flat < base.size and len(base.shape) <= 1:
                return base, flat, node
            return None
        if isinstance(idx, tuple) and all(isinstance(i, (int, np.integer)) for i in idx):
            try:
                flat = int(np.ravel_multi_index(tuple(int(i) for i in idx), base.shape))
            except ValueError:
                return None
            return base, flat, node
        return None
    return None


# ---------------------------------------------------------------------------
# Cheap syntactic gate
# ---------------------------------------------------------------------------


def has_binary_multilinear_work(model: Model) -> bool:
    """True if the objective or a constraint contains a product/power subterm
    of syntactic degree >= ``_MIN_DEGREE`` whose variables are all
    binary-valued — the witness that the exact Fortet/Glover linearization has
    something to gain. Cheap (one iterative pass over each DAG, O(1) per node)
    and safe: ``False`` means the pass is skipped, never that a model is
    mis-rewritten. Returns ``False`` on any error."""
    try:
        if model._objective is None:
            return False
        bodies = [model._objective.expression]
        for c in model._constraints:
            if isinstance(c, Constraint):
                bodies.append(c.body)
        return any(_witness_scan(b) for b in bodies)
    except Exception:
        return False


def _gate_children(node: Expression) -> Optional[list[Expression]]:
    if isinstance(node, BinaryOp):
        return [node.left, node.right]
    if isinstance(node, UnaryOp):
        return [node.operand]
    if isinstance(node, SumExpression):
        return [node.operand]
    if isinstance(node, SumOverExpression):
        return list(node.terms)
    if isinstance(node, (Constant, Variable, IndexExpression, Parameter)):
        return []
    return None  # unsupported node type


def _witness_scan(root: Expression) -> bool:
    """Iterative post-order scan computing, per node, ``(degree, all_binary,
    supported)`` — degree saturating at ``_DEG_CAP`` — and reporting whether
    any ``*``/``**`` node reaches degree >= ``_MIN_DEGREE`` with all-binary
    variables and fully supported structure underneath."""
    # memo: id(node) -> (deg, all_bin, ok)
    memo: dict[int, tuple[int, bool, bool]] = {}
    stack: list[Expression] = [root]
    while stack:
        node = stack[-1]
        if id(node) in memo:
            stack.pop()
            continue
        children = _gate_children(node)
        if children is None:
            memo[id(node)] = (0, True, False)
            stack.pop()
            continue
        pending = [c for c in children if id(c) not in memo]
        if pending:
            stack.extend(pending)
            continue
        stack.pop()
        memo[id(node)] = _gate_eval(node, memo)
        deg, all_bin, ok = memo[id(node)]
        if (
            ok
            and all_bin
            and deg >= _MIN_DEGREE
            and isinstance(node, BinaryOp)
            and node.op in ("*", "**")
        ):
            return True
    return False


def _gate_eval(node: Expression, memo: dict[int, tuple[int, bool, bool]]) -> tuple[int, bool, bool]:
    if isinstance(node, (Constant, Parameter)):
        return (0, True, True)
    ref = _leaf_ref(node)
    if ref is not None:
        return (1, _binary_like(ref[0], ref[1]), True)
    if isinstance(node, (IndexExpression, Variable)):
        return (0, True, False)  # vector/opaque reference
    if isinstance(node, UnaryOp):
        d, b, ok = memo[id(node.operand)]
        return (d, b, ok and node.op == "neg")
    if isinstance(node, (SumExpression, SumOverExpression)):
        children = _gate_children(node) or []
        deg, all_bin, ok = 0, True, True
        for c in children:
            d, b, o = memo[id(c)]
            deg, all_bin, ok = max(deg, d), all_bin and b, ok and o
        return (deg, all_bin, ok)
    if isinstance(node, BinaryOp):
        dl, bl, okl = memo[id(node.left)]
        dr, br, okr = memo[id(node.right)]
        ok = okl and okr
        all_bin = bl and br
        if node.op in ("+", "-"):
            return (max(dl, dr), all_bin, ok)
        if node.op == "*":
            return (min(dl + dr, _DEG_CAP), all_bin, ok)
        if node.op == "/":
            return (dl, all_bin, ok and dr == 0)
        if node.op == "**":
            if not (isinstance(node.right, Constant) and okl):
                return (0, True, False)
            try:
                k = _scalar_constant(node.right)
            except _Unsupported:
                return (0, True, False)
            if k != int(k) or not (0 <= k <= _MAX_POW):
                return (0, True, False)
            return (min(dl * int(k), _DEG_CAP), bl, okl)
        return (0, True, False)
    return (0, True, False)


# ---------------------------------------------------------------------------
# Exact multilinear expansion (dict arithmetic over monomials)
# ---------------------------------------------------------------------------


class _ExpandCtx:
    """Shared state of one reformulation attempt: the (var, elem) -> reference
    registry, the binary/integer-likeness caches, and the product-op budget."""

    def __init__(self) -> None:
        self.refs: dict[tuple[int, int], Expression] = {}
        self.is_bin: dict[tuple[int, int], bool] = {}
        self.is_int: dict[tuple[int, int], bool] = {}
        self.ops = 0

    def register(self, var: Variable, elem: int, ref: Expression) -> tuple[int, int]:
        key = (var._index, elem)
        if key not in self.refs:
            self.refs[key] = ref
            self.is_bin[key] = _binary_like(var, elem)
            self.is_int[key] = _integer_valued(var)
        return key


def _poly_add(p: dict, q: dict, sign: float = 1.0) -> dict:
    out = dict(p)
    for m, c in q.items():
        nc = out.get(m, 0.0) + sign * c
        if nc == 0.0:
            out.pop(m, None)
        else:
            out[m] = nc
    if len(out) > _MAX_MONOMIALS:
        raise _Unsupported("monomial budget exceeded")
    return out


def _poly_scale(p: dict, c: float) -> dict:
    if c == 0.0:
        return {}
    return {m: v * c for m, v in p.items()}


def _poly_mul(p: dict, q: dict, ctx: _ExpandCtx) -> dict:
    ctx.ops += len(p) * len(q)
    if ctx.ops > _MAX_PRODUCT_OPS:
        raise _Unsupported("product-op budget exceeded")
    out: dict = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            dup = m1 & m2
            if dup:
                # Collapsing a repeated factor is ``b**2 == b`` — valid only
                # for binary-valued factors. A repeated non-binary factor
                # would be silently *rewritten* (x**2 -> x): abort instead.
                for key in dup:
                    if not ctx.is_bin[key]:
                        raise _Unsupported("repeated non-binary factor")
            m = m1 | m2
            nc = out.get(m, 0.0) + c1 * c2
            if nc == 0.0:
                out.pop(m, None)
            else:
                out[m] = nc
    if len(out) > _MAX_MONOMIALS:
        raise _Unsupported("monomial budget exceeded")
    return out


def _poly_pow(p: dict, k: int, ctx: _ExpandCtx) -> dict:
    if k == 0:
        return {_EMPTY: 1.0}
    out = p
    for _ in range(k - 1):
        out = _poly_mul(out, p, ctx)
    return out


def _expand_children(node: Expression) -> list[Expression]:
    if isinstance(node, BinaryOp):
        return [node.left, node.right]
    if isinstance(node, UnaryOp):
        if node.op != "neg":
            raise _Unsupported(f"unary op {node.op!r}")
        return [node.operand]
    if isinstance(node, SumExpression):
        return [node.operand]
    if isinstance(node, SumOverExpression):
        return list(node.terms)
    if isinstance(node, (Constant, Parameter)):
        return []
    if _leaf_ref(node) is not None:
        return []
    raise _Unsupported(f"unsupported node {type(node).__name__}")


def _expand_to_multilinear(root: Expression, ctx: _ExpandCtx) -> dict:
    """Expand *root* into ``{monomial: coefficient}`` with binary squares
    collapsed (``b**2 == b``), via an iterative post-order walk (see module
    docstring for why recursion is forbidden here). Raises ``_Unsupported`` on
    any structure the pass cannot linearize exactly."""
    memo: dict[int, dict] = {}
    stack: list[Expression] = [root]
    while stack:
        node = stack[-1]
        if id(node) in memo:
            stack.pop()
            continue
        children = _expand_children(node)
        pending = [c for c in children if id(c) not in memo]
        if pending:
            stack.extend(pending)
            continue
        stack.pop()
        memo[id(node)] = _expand_eval(node, memo, ctx)
    return memo[id(root)]


def _expand_eval(node: Expression, memo: dict[int, dict], ctx: _ExpandCtx) -> dict:
    if isinstance(node, Constant):
        v = _scalar_constant(node)
        return {_EMPTY: v} if v != 0.0 else {}
    if isinstance(node, Parameter):
        pv = np.asarray(node.value)
        if pv.ndim != 0 and pv.size != 1:
            raise _Unsupported("non-scalar parameter")
        v = float(pv.reshape(()))
        return {_EMPTY: v} if v != 0.0 else {}
    ref = _leaf_ref(node)
    if ref is not None:
        var, elem, ref_expr = ref
        key = ctx.register(var, elem, ref_expr)
        return {_Mono((key,)): 1.0}
    if isinstance(node, UnaryOp):  # only "neg" reaches here
        return _poly_scale(memo[id(node.operand)], -1.0)
    if isinstance(node, SumExpression):
        return memo[id(node.operand)]
    if isinstance(node, SumOverExpression):
        out: dict = {}
        for t in node.terms:
            out = _poly_add(out, memo[id(t)])
        return out
    if isinstance(node, BinaryOp):
        left = memo[id(node.left)]
        right = memo[id(node.right)]
        if node.op == "+":
            return _poly_add(left, right)
        if node.op == "-":
            return _poly_add(left, right, sign=-1.0)
        if node.op == "*":
            return _poly_mul(left, right, ctx)
        if node.op == "/":
            if set(right.keys()) - {_EMPTY}:
                raise _Unsupported("division by a non-constant")
            denom = right.get(_EMPTY, 0.0)
            if denom == 0.0:
                raise _Unsupported("division by zero")
            return _poly_scale(left, 1.0 / denom)
        if node.op == "**":
            if not isinstance(node.right, Constant):
                raise _Unsupported("non-constant exponent")
            k = _scalar_constant(node.right)
            if k != int(k) or not (0 <= k <= _MAX_POW):
                raise _Unsupported(f"exponent {k!r} not a small non-negative integer")
            return _poly_pow(left, int(k), ctx)
        raise _Unsupported(f"binary op {node.op!r}")
    raise _Unsupported(f"unsupported node {type(node).__name__}")


# ---------------------------------------------------------------------------
# Addend collection and squared-form detection
# ---------------------------------------------------------------------------


def _collect_addends(root: Expression) -> tuple[float, list[tuple[float, Expression]]]:
    """Split *root* into ``(constant, [(coefficient, node), ...])`` by walking
    the +/- spine iteratively, folding constant factors/divisors and unary
    negations into the coefficients. Nodes below a non-additive operation are
    returned whole (their internal structure is handled by expansion)."""
    const = 0.0
    addends: list[tuple[float, Expression]] = []
    stack: list[tuple[float, Expression]] = [(1.0, root)]
    while stack:
        coef, node = stack.pop()
        if isinstance(node, Constant):
            const += coef * _scalar_constant(node)
            continue
        if isinstance(node, UnaryOp) and node.op == "neg":
            stack.append((-coef, node.operand))
            continue
        if isinstance(node, SumExpression):
            stack.append((coef, node.operand))
            continue
        if isinstance(node, SumOverExpression):
            stack.extend((coef, t) for t in reversed(node.terms))
            continue
        if isinstance(node, BinaryOp):
            if node.op == "+":
                stack.append((coef, node.right))
                stack.append((coef, node.left))
                continue
            if node.op == "-":
                stack.append((-coef, node.right))
                stack.append((coef, node.left))
                continue
            if node.op == "*":
                if isinstance(node.left, Constant):
                    stack.append((coef * _scalar_constant(node.left), node.right))
                    continue
                if isinstance(node.right, Constant):
                    stack.append((coef * _scalar_constant(node.right), node.left))
                    continue
            if node.op == "/" and isinstance(node.right, Constant):
                denom = _scalar_constant(node.right)
                if denom == 0.0:
                    raise _Unsupported("division by zero")
                stack.append((coef / denom, node.left))
                continue
        addends.append((coef, node))
    # The stack pops left-most last for '+' chains; addends were appended in
    # traversal order already (left pushed last, popped first).
    return const, addends


def _square_base(node: Expression) -> Optional[Expression]:
    """Return ``E`` when *node* is syntactically ``E**2`` or ``E*E`` (the same
    object on both sides, as DAG-shared squares are), else ``None``."""
    if isinstance(node, BinaryOp):
        if node.op == "**" and isinstance(node.right, Constant):
            try:
                if _scalar_constant(node.right) == 2.0:
                    return node.left
            except _Unsupported:
                return None
        if node.op == "*" and node.left is node.right:
            return node.left
    return None


@dataclass
class _SquareTerm:
    """One objective-pressure-encodable ``coef * E**2`` addend: ``poly`` is the
    multilinear expansion of ``E`` (integer coefficients over integer-valued
    variables), ``[lo, hi]`` its interval range snapped to the attainable
    value grid, and ``step`` the grid spacing: when every non-constant
    coefficient of ``E`` is divisible by ``g``, ``E`` only attains values
    congruent to its constant term mod ``g``, so secants are only needed
    between consecutive attainable points (autocorr's ``C_k`` has ``g = 2``,
    halving the secant rows)."""

    coef: float
    poly: dict
    lo: int
    hi: int
    step: int = 1

    @property
    def n_secants(self) -> int:
        return (self.hi - self.lo) // self.step


@dataclass
class _ProcessedBody:
    """A body split into a flat multilinear polynomial plus secant-encodable
    squared terms (empty for plain constraint bodies)."""

    flat: dict
    squares: list[_SquareTerm] = field(default_factory=list)


def _poly_int_range(poly: dict, ctx: _ExpandCtx) -> Optional[tuple[int, int]]:
    """Integer interval bound of a multilinear *poly* whose every monomial is a
    product of {0,1}-valued factors, when all coefficients (and the constant)
    are integers and all participating variables are integer-valued. Returns
    ``None`` when those conditions fail — the caller then falls back to flat
    expansion of the square."""
    lo = hi = 0.0
    for m, c in poly.items():
        if not float(c).is_integer():
            return None
        if not m:
            lo += c
            hi += c
            continue
        for key in m:
            if not ctx.is_int[key]:
                return None
        if len(m) >= 2 and not all(ctx.is_bin[key] for key in m):
            return None
        if len(m) == 1:
            key = next(iter(m))
            vlo, vhi = (0.0, 1.0) if ctx.is_bin[key] else _var_bounds(ctx.refs[key])
            if not (np.isfinite(vlo) and np.isfinite(vhi)):
                return None
            lo += min(c * vlo, c * vhi)
            hi += max(c * vlo, c * vhi)
        else:
            lo += min(c, 0.0)
            hi += max(c, 0.0)
    if abs(lo) > 2**31 or abs(hi) > 2**31:
        return None
    return int(np.floor(lo)), int(np.ceil(hi))


def _var_bounds(ref: Expression) -> tuple[float, float]:
    leaf = _leaf_ref(ref)
    assert leaf is not None
    var, elem, _ = leaf
    return (
        float(np.asarray(var.lb).flat[elem]),
        float(np.asarray(var.ub).flat[elem]),
    )


def _interval_of_dag(root: Expression) -> tuple[float, float]:
    """Sound interval enclosure of *root* over the variable box, computed on
    the *factored* DAG. On sum-of-products structure this is much tighter than
    the per-monomial interval of the expanded polynomial (autocorr's
    ``C_k = sum_i s_i s_{i+k}`` with ``s in [-1, 1]`` encloses to exactly
    ``[-(n-k), n-k]``, ~4x tighter than the expanded-poly interval), which
    directly cuts the number of secant rows. Unsupported structure yields
    ``(-inf, inf)`` — the caller intersects with the expanded-poly interval,
    so this is only ever a refinement."""
    inf = float("inf")
    memo: dict[int, tuple[float, float]] = {}
    stack: list[Expression] = [root]
    while stack:
        node = stack[-1]
        if id(node) in memo:
            stack.pop()
            continue
        children = _gate_children(node)
        if children is None:
            memo[id(node)] = (-inf, inf)
            stack.pop()
            continue
        pending = [c for c in children if id(c) not in memo]
        if pending:
            stack.extend(pending)
            continue
        stack.pop()
        memo[id(node)] = _interval_eval(node, memo)
    return memo[id(root)]


def _interval_mul(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    products = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    products = [p for p in products if not np.isnan(p)]
    if not products:
        return (-float("inf"), float("inf"))
    return (min(products), max(products))


def _interval_eval(node: Expression, memo: dict[int, tuple[float, float]]) -> tuple[float, float]:
    inf = float("inf")
    top = (-inf, inf)
    if isinstance(node, Constant):
        try:
            v = _scalar_constant(node)
        except _Unsupported:
            return top
        return (v, v)
    if isinstance(node, Parameter):
        pv = np.asarray(node.value)
        if pv.ndim != 0 and pv.size != 1:
            return top
        v = float(pv.reshape(()))
        return (v, v)
    ref = _leaf_ref(node)
    if ref is not None:
        var, elem, _ = ref
        lo = float(np.asarray(var.lb).flat[elem])
        hi = float(np.asarray(var.ub).flat[elem])
        return (lo, hi)
    if isinstance(node, UnaryOp):
        lo, hi = memo[id(node.operand)]
        return (-hi, -lo) if node.op == "neg" else top
    if isinstance(node, SumExpression):
        return memo[id(node.operand)]
    if isinstance(node, SumOverExpression):
        lo = hi = 0.0
        for t in node.terms:
            tl, th = memo[id(t)]
            lo, hi = lo + tl, hi + th
        return (lo, hi)
    if isinstance(node, BinaryOp):
        left = memo[id(node.left)]
        right = memo[id(node.right)]
        if node.op == "+":
            return (left[0] + right[0], left[1] + right[1])
        if node.op == "-":
            return (left[0] - right[1], left[1] - right[0])
        if node.op == "*":
            return _interval_mul(left, right)
        if node.op == "/":
            if right[0] == right[1] and right[0] not in (0.0,):
                d = right[0]
                lo, hi = left[0] / d, left[1] / d
                return (min(lo, hi), max(lo, hi))
            return top
        if node.op == "**":
            if not isinstance(node.right, Constant):
                return top
            try:
                k = _scalar_constant(node.right)
            except _Unsupported:
                return top
            if k != int(k) or not (0 <= k <= _MAX_POW):
                return top
            out = (1.0, 1.0)
            for _ in range(int(k)):
                out = _interval_mul(out, left)
            # Even powers are nonnegative regardless of the sign box.
            if int(k) % 2 == 0:
                out = (max(out[0], 0.0), out[1])
            return out
        return top
    return top


def _attainable_grid(poly: dict, lo: int, hi: int) -> Optional[tuple[int, int, int]]:
    """Snap ``[lo, hi]`` to the value grid of the integer-coefficient form
    *poly*: with ``g = gcd`` of the non-constant coefficients, every value is
    congruent to the constant term mod ``g``, so the bounds tighten to the
    outermost attainable points and secants only need spacing ``g``. Returns
    ``(lo_a, hi_a, g)`` or ``None`` when the snapped interval is empty (a
    numerically defensive impossibility — the true value is attainable and
    inside every sound interval; the caller then falls back to flat)."""
    c0 = int(round(poly.get(_EMPTY, 0.0)))
    g = 0
    for m, c in poly.items():
        if m:
            g = int(np.gcd(g, int(round(abs(c)))))
    if g <= 1:
        return (lo, hi, 1) if lo <= hi else None
    lo_a = lo + (c0 - lo) % g
    hi_a = hi - (hi - c0) % g
    if lo_a > hi_a:
        return None
    return (lo_a, hi_a, g)


def _process_body(root: Expression, ctx: _ExpandCtx, eff: Optional[float]) -> _ProcessedBody:
    """Expand *root*, peeling off addends ``coef * E**2`` for the secant
    encoding when ``eff`` (the objective-pressure multiplier of this body) is
    set and ``eff * coef > 0`` and ``E`` is an integer-valued integer-
    coefficient multilinear form. Everything else lands in the flat poly."""
    const, addends = _collect_addends(root)
    flat: dict = {_EMPTY: const} if const != 0.0 else {}
    squares: list[_SquareTerm] = []
    for coef, node in addends:
        if coef == 0.0:
            continue
        base = _square_base(node)
        if base is not None and eff is not None and eff * coef > 0.0:
            inner = _expand_to_multilinear(base, ctx)
            rng = _poly_int_range(inner, ctx)
            if rng is not None:
                # Refine with the factored-DAG interval (both are sound, so
                # their intersection is): fewer secant rows, smaller MILP.
                dlo, dhi = _interval_of_dag(base)
                lo = max(rng[0], int(np.ceil(dlo)) if np.isfinite(dlo) else rng[0])
                hi = min(rng[1], int(np.floor(dhi)) if np.isfinite(dhi) else rng[1])
                grid = _attainable_grid(inner, lo, hi)
            else:
                grid = None
            # A constant or affine-in-one-binary square is cheaper flat.
            if grid is not None and grid[1] - grid[0] >= 2 and len(inner) >= 2:
                squares.append(_SquareTerm(coef, inner, grid[0], grid[1], grid[2]))
                continue
        poly = _expand_to_multilinear(node, ctx)
        flat = _poly_add(flat, _poly_scale(poly, coef))
    return _ProcessedBody(flat=flat, squares=squares)


# ---------------------------------------------------------------------------
# Objective-variable ("objvar") defining-row detection
# ---------------------------------------------------------------------------


def _contains_node(root: Expression, target: Expression) -> bool:
    """Iterative scan: does *root* reference *target* (by identity)?"""
    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if node is target:
            return True
        for attr in ("left", "right", "operand", "base"):
            child = getattr(node, attr, None)
            if isinstance(child, Expression):
                stack.append(child)
        for attr in ("args", "terms"):
            seq = getattr(node, attr, None)
            if isinstance(seq, (list, tuple)):
                stack.extend(c for c in seq if isinstance(c, Expression))
    return False


def _detect_objvar_row(model: Model, mu: float) -> Optional[tuple[int, float]]:
    """Detect the `.nl` objvar convention: the objective is a bare free scalar
    continuous variable ``tau`` pinned by exactly one row ``a*tau + g(x)
    <sense> rhs`` that transmits the objective pressure to ``g`` (equality, or
    the inequality bounding ``tau`` on its objective-improving side). Returns
    ``(row_index, a)`` or ``None``. ``tau`` must be genuinely free: finite
    bounds could clip the surrogate below the true objective value and break
    exactness (see module docstring)."""
    obj = model._objective
    if obj is None or not isinstance(obj.expression, Variable):
        return None
    tau = obj.expression
    if tau.var_type != VarType.CONTINUOUS or tau.size != 1 or tau.shape != ():
        return None
    lb = float(np.asarray(tau.lb).reshape(()))
    ub = float(np.asarray(tau.ub).reshape(()))
    if lb > -_FREE_BOUND or ub < _FREE_BOUND:
        return None
    rows = [i for i, c in enumerate(model._constraints) if _contains_node(c.body, tau)]
    if len(rows) != 1:
        return None
    row = model._constraints[rows[0]]
    # tau must enter the row as a plain linear addend with constant
    # coefficient a, and appear in no other addend.
    try:
        _, addends = _collect_addends(row.body)
    except _Unsupported:
        return None
    a = 0.0
    for coef, node in addends:
        if node is tau:
            if a != 0.0:
                return None
            a = coef
        elif _contains_node(node, tau):
            return None
    if a == 0.0:
        return None
    # Sense condition: '==' always transmits; an inequality only when it
    # bounds tau on its objective-improving side (min: from below; max: from
    # above). Constraint bodies are normalized against ``rhs``:
    #   '>=' means body >= rhs  ->  a*tau >= rhs - g  (lower-bounds tau iff a>0)
    #   '<=' means body <= rhs  ->  upper-bounds tau iff a>0.
    if row.sense == "==":
        return rows[0], a
    if row.sense == ">=" and mu * a > 0:
        return rows[0], a
    if row.sense == "<=" and mu * a < 0:
        return rows[0], a
    return None


# ---------------------------------------------------------------------------
# Rebuild: Fortet aux variables, secant epigraphs, linear bodies
# ---------------------------------------------------------------------------


def _mono_order(poly: dict) -> list[frozenset]:
    """Non-constant monomials of *poly* in the deterministic canonical order."""
    return sorted((m for m in poly if m), key=lambda m: tuple(sorted(m)))


def _balanced_sum(terms: list[Expression]) -> Expression:
    """Combine *terms* into a balanced ``+`` tree (depth O(log n)) so that
    downstream recursive DAG walkers never see a linear-depth chain."""
    if not terms:
        return Constant(0.0)
    while len(terms) > 1:
        terms = [
            BinaryOp("+", terms[i], terms[i + 1]) if i + 1 < len(terms) else terms[i]
            for i in range(0, len(terms), 2)
        ]
    return terms[0]


def _poly_terms(poly: dict, ctx: _ExpandCtx, aux: dict[frozenset, Variable]) -> list[Expression]:
    terms: list[Expression] = []
    const = poly.get(_EMPTY, 0.0)
    for mono in sorted((m for m in poly if m), key=lambda m: tuple(sorted(m))):
        c = poly[mono]
        base: Expression = ctx.refs[next(iter(mono))] if len(mono) == 1 else aux[mono]
        terms.append(base if c == 1.0 else BinaryOp("*", Constant(float(c)), base))
    if const != 0.0 or not terms:
        terms.append(Constant(float(const)))
    return terms


def _model_carries_unsupported_state(model: Model) -> bool:
    """True when the model holds structure this pass does not copy into the
    rebuilt model (SOS/indicator objects in ``_constraints``, complementarity
    conditions, builder-side linear blocks/objectives). Firing on such a model
    would silently drop that structure, so the pass abstains instead."""
    if any(not isinstance(c, Constraint) for c in model._constraints):
        return True
    if getattr(model, "_complementarities", None):
        return True
    if getattr(model, "_builder_linear_blocks", None):
        return True
    if getattr(model, "_builder_linear_objective", None) is not None:
        return True
    if getattr(model, "_builder_quadratic_objective", None) is not None:
        return True
    return False


def reformulate_binary_multilinear(model: Model) -> Model:
    """Return an equivalent pure-MILP model with every binary multilinear
    monomial exactly Fortet/Glover-linearized (and objective squares of
    integer-valued forms secant-encoded), or *model* unchanged when the
    structure is out of scope, a budget is exceeded, or anything errs — the
    pass never regresses a model it cannot handle exactly."""
    try:
        return _reformulate(model)
    except _Unsupported:
        return model
    except Exception:  # pragma: no cover - defensive
        return model


def _reformulate(model: Model) -> Model:
    obj = model._objective
    if obj is None:
        return model
    if _model_carries_unsupported_state(model):
        return model

    mu = 1.0 if obj.sense == ObjectiveSense.MINIMIZE else -1.0
    objvar_row = _detect_objvar_row(model, mu)

    ctx = _ExpandCtx()
    if objvar_row is None:
        obj_body = _process_body(obj.expression, ctx, eff=mu)
    else:
        obj_body = _process_body(obj.expression, ctx, eff=None)
    con_bodies: list[_ProcessedBody] = []
    for i, c in enumerate(model._constraints):
        if objvar_row is not None and i == objvar_row[0]:
            # The defining row transmits the objective pressure to its squares
            # with multiplier -mu/a (min tau with a*tau + c*E**2 + ... pins
            # tau to -(c/a)*E**2 + ..., so the min-effective coefficient of
            # E**2 is -mu*c/a).
            con_bodies.append(_process_body(c.body, ctx, eff=-mu / objvar_row[1]))
        else:
            con_bodies.append(_process_body(c.body, ctx, eff=None))

    all_bodies = [obj_body, *con_bodies]

    # Structural verdict: every degree>=2 flat monomial must be purely binary
    # (a continuous / general-integer variable may only appear linearly), and
    # a genuine degree>=_MIN_DEGREE term must exist for the rewrite to pay off
    # (squares count as twice their inner degree).
    max_degree = 0
    monos: set = set()
    n_secant_rows = 0
    n_squares = 0
    for pbody in all_bodies:
        for m in pbody.flat:
            if len(m) >= 2:
                if not all(ctx.is_bin[k] for k in m):
                    return model
                monos.add(m)
                max_degree = max(max_degree, len(m))
        for sq in pbody.squares:
            for m in sq.poly:
                if len(m) >= 2:
                    monos.add(m)
            inner_deg = max((len(m) for m in sq.poly), default=0)
            max_degree = max(max_degree, 2 * inner_deg)
            n_secant_rows += sq.n_secants
            n_squares += 1
    if max_degree < _MIN_DEGREE:
        return model
    n_rows = (
        len(model._constraints)
        + sum(len(m) + 1 for m in monos)  # Fortet rows per monomial
        + n_squares  # y == L linking rows
        + n_secant_rows
    )
    if n_rows > _MAX_ROWS:
        return model

    new_model = Model(model.name)
    new_model._variables = list(model._variables)
    new_model._parameters = list(model._parameters)
    new_model._rebuild_name_index()

    # Flat positions of the original variables (the aux columns are appended
    # after them, so the original flat layout is a prefix of the reformed one).
    # Recorded per aux variable — in exact append order — so a warm start /
    # heuristic point over the ORIGINAL variables can be extended to a feasible
    # point of the reformed MILP (see ``extend_initial_point``).
    _flat_off: dict[int, int] = {}
    _off = 0
    for v in model._variables:
        _flat_off[v._index] = _off
        _off += v.size
    n_orig_flat = _off

    def _flat(key: tuple[int, int]) -> int:
        return _flat_off[key[0]] + key[1]

    aux_spec: list[tuple] = []

    # One aux + Fortet rows per distinct monomial, in deterministic order.
    aux: dict[frozenset, Variable] = {}
    aux_rows: list[Constraint] = []
    for i, mono in enumerate(sorted(monos, key=lambda m: tuple(sorted(m)))):
        z = Variable(f"_bml_z{i}", VarType.CONTINUOUS, (), 0.0, 1.0, new_model)
        new_model._variables.append(z)
        aux[mono] = z
        aux_spec.append(("z", tuple(_flat(k) for k in sorted(mono))))
        refs = [ctx.refs[k] for k in sorted(mono)]
        # z <= b_i for every factor (z >= 0 is the variable's lower bound).
        for r in refs:
            aux_rows.append(Constraint(BinaryOp("-", z, r), "<=", 0.0))
        # z >= sum_i b_i - (n-1)  <=>  z - sum_i b_i + (n-1) >= 0. RHS kept at
        # 0 with the constant folded into the body (the LP extractor folds the
        # body against a zero RHS; see integer_product_reform's note).
        lower: Expression = z
        for r in refs:
            lower = BinaryOp("-", lower, r)
        lower = BinaryOp("+", lower, Constant(float(len(refs) - 1)))
        aux_rows.append(Constraint(lower, ">=", 0.0))

    # Secant epigraphs: per encodable square, y == linearized(E) plus the
    # integer-point secants t >= (2j+1)*y - j*(j+1). At every integer y their
    # max equals y**2 exactly; the objective pressure (eff*coef > 0, checked
    # at collection) pins t to the envelope at the optimum.
    sq_counter = 0

    def _encode_square(sq: _SquareTerm) -> Variable:
        nonlocal sq_counter
        y = Variable(
            f"_bml_y{sq_counter}",
            VarType.CONTINUOUS,
            (),
            float(sq.lo),
            float(sq.hi),
            new_model,
        )
        new_model._variables.append(y)
        t_lo = 0.0 if sq.lo <= 0 <= sq.hi else float(min(sq.lo**2, sq.hi**2))
        t_hi = float(max(sq.lo**2, sq.hi**2))
        t = Variable(f"_bml_t{sq_counter}", VarType.CONTINUOUS, (), t_lo, t_hi, new_model)
        new_model._variables.append(t)
        sq_counter += 1
        y_terms = tuple(
            (sq.poly[m], tuple(_flat(k) for k in sorted(m))) for m in _mono_order(sq.poly)
        )
        aux_spec.append(("y", y_terms, float(sq.poly.get(_EMPTY, 0.0))))
        aux_spec.append(("t",))
        link = _balanced_sum([y, *(_negate(term) for term in _poly_terms(sq.poly, ctx, aux))])
        aux_rows.append(Constraint(link, "==", 0.0))
        for u in range(sq.lo, sq.hi, sq.step):
            # Secant of the parabola through (u, u^2) and (u+g, (u+g)^2):
            # t >= (2u+g)*y - u*(u+g). Exact at both endpoints, so the max of
            # all grid secants equals y**2 at every attainable y.
            v = u + sq.step
            row: Expression = BinaryOp("-", t, BinaryOp("*", Constant(float(u + v)), y))
            row = BinaryOp("+", row, Constant(float(u * v)))
            aux_rows.append(Constraint(row, ">=", 0.0))
        return t

    def _rebuild(pb: _ProcessedBody) -> Expression:
        terms = _poly_terms(pb.flat, ctx, aux)
        for sq in pb.squares:
            t = _encode_square(sq)
            terms.append(t if sq.coef == 1.0 else BinaryOp("*", Constant(float(sq.coef)), t))
        return _balanced_sum(terms)

    new_obj_expr = _rebuild(obj_body)
    rebuilt: list[Constraint] = []
    for c, pbody in zip(model._constraints, con_bodies):
        # Rebuild EVERY body from its expansion, not only those that gained
        # aux terms: a body like ``b*b + x`` expands to the *linear* ``b + x``
        # (no aux, so no "work"), but keeping the original nonlinear
        # expression would make the pure-MILP adoption guard reject the whole
        # reformulation. The rebuilt form is algebraically identical on the
        # feasible set.
        rebuilt.append(Constraint(_rebuild(pbody), c.sense, c.rhs, c.name))

    new_model._constraints = rebuilt + aux_rows
    new_model._objective = Objective(expression=new_obj_expr, sense=obj.sense)

    # --- Warm-start / heuristic metadata (incumbent seeding) ---
    # Dynamic attributes (Model does not declare them): consumed only through
    # getattr by extend_initial_point / heuristic_incumbent.
    setattr(new_model, "_bml_aux_spec", aux_spec)  # noqa: B010
    setattr(new_model, "_bml_n_orig_flat", n_orig_flat)  # noqa: B010
    setattr(  # noqa: B010
        new_model,
        "_bml_binary_flat",
        frozenset(_flat(k) for k, is_b in ctx.is_bin.items() if is_b),
    )
    setattr(  # noqa: B010
        new_model,
        "_bml_heuristic",
        _heuristic_meta(model, ctx, mu, objvar_row, obj_body, con_bodies, _flat_off, n_orig_flat),
    )
    return new_model


def _heuristic_meta(
    model: Model,
    ctx: _ExpandCtx,
    mu: float,
    objvar_row: Optional[tuple[int, float]],
    obj_body: _ProcessedBody,
    con_bodies: list[_ProcessedBody],
    flat_off: dict[int, int],
    n_orig_flat: int,
) -> Optional[dict]:
    """Build the primal local-search description of the model, or ``None`` when
    the class-level feasibility argument does not apply. The heuristic needs
    "every 0/1 assignment of the binary variables is feasible", which holds
    exactly when the only constraint is the objective variable's defining row
    (or there are none) and every variable is binary-valued — except the free
    objective variable itself, whose value is *determined* by the bits."""
    n_extra_rows = 1 if objvar_row is not None else 0
    if len(model._constraints) != n_extra_rows:
        return None

    def _flat(key: tuple[int, int]) -> int:
        return flat_off[key[0]] + key[1]

    tau_key: Optional[tuple[int, int]] = None
    if objvar_row is not None:
        assert model._objective is not None
        tau = model._objective.expression
        assert isinstance(tau, Variable)
        tau_key = (tau._index, 0)

    bits: list[int] = []
    fixed: list[tuple[int, float]] = []
    for v in model._variables:
        for elem in range(v.size):
            key = (v._index, elem)
            if tau_key is not None and key == tau_key:
                continue
            if not _binary_like(v, elem):
                return None
            lo = float(np.asarray(v.lb).flat[elem])
            hi = float(np.asarray(v.ub).flat[elem])
            flo, fhi = max(0.0, np.ceil(lo)), min(1.0, np.floor(hi))
            if flo > fhi:
                return None
            if flo == fhi:
                fixed.append((_flat(key), flo))
            else:
                bits.append(_flat(key))

    def _poly_terms_flat(poly: dict, scale: float, skip: Optional[tuple[int, int]] = None):
        terms = []
        for m in _mono_order(poly):
            if skip is not None and m == frozenset((skip,)):
                continue
            terms.append((scale * poly[m], tuple(_flat(k) for k in sorted(m))))
        return terms, scale * poly.get(_EMPTY, 0.0)

    squares: list[tuple[float, list, float]] = []
    if objvar_row is None:
        body = obj_body
        flat_terms, const = _poly_terms_flat(body.flat, 1.0)
        for sq in body.squares:
            iterms, iconst = _poly_terms_flat(sq.poly, 1.0)
            squares.append((sq.coef, iterms, iconst))
        tau_spec = None
    else:
        # Row a*tau + R(b) + sum c_q*E_q**2 <sense> rhs pins the objective to
        # tau*(b) = (rhs - R(b) - sum c_q E_q(b)**2) / a on its binding side.
        row_idx, a = objvar_row
        c = model._constraints[row_idx]
        body = con_bodies[row_idx]
        assert tau_key is not None
        flat_terms, const = _poly_terms_flat(body.flat, -1.0 / a, skip=tau_key)
        const += float(c.rhs) / a
        for sq in body.squares:
            iterms, iconst = _poly_terms_flat(sq.poly, 1.0)
            squares.append((-sq.coef / a, iterms, iconst))
        tau_spec = _flat(tau_key)

    return {
        "bits": bits,
        "fixed": fixed,
        "tau": tau_spec,
        "flat_terms": flat_terms,
        "flat_const": const,
        "squares": squares,
        "sense_mul": mu,
        "n_orig_flat": n_orig_flat,
    }


def _negate(term: Expression) -> Expression:
    if isinstance(term, Constant):
        return Constant(-float(term.value.reshape(())))
    return UnaryOp("neg", term)


# ---------------------------------------------------------------------------
# Incumbent seeding: warm-start mapping + primal local search
# ---------------------------------------------------------------------------


def extend_initial_point(reformed: Model, x0) -> Optional[np.ndarray]:
    """Extend a point over the ORIGINAL variables (flat, length
    ``_bml_n_orig_flat``) to a point over the reformed model's full variable
    vector by evaluating each auxiliary from its recorded definition
    (``z = prod b``, ``y = E(b)``, ``t = y**2``). Binary-valued entries are
    snapped to their nearest integer first so the aux values satisfy the
    Fortet/secant rows exactly. Returns ``None`` when the model carries no
    reformulation metadata or the point has the wrong length — the caller then
    simply does not seed (the MILP driver re-validates whatever is passed, so
    this mapping is an optimization, never a soundness lever)."""
    spec = getattr(reformed, "_bml_aux_spec", None)
    n0 = getattr(reformed, "_bml_n_orig_flat", None)
    if spec is None or n0 is None:
        return None
    x = np.asarray(x0, dtype=np.float64).ravel()
    if x.size != n0 or not np.all(np.isfinite(x)):
        return None
    x = x.copy()
    for j in getattr(reformed, "_bml_binary_flat", frozenset()):
        r = round(float(x[j]))
        if abs(x[j] - r) > 1e-6:
            return None
        x[j] = float(r)
    out = list(x)
    last_y = 0.0
    for entry in spec:
        kind = entry[0]
        if kind == "z":
            v = 1.0
            for j in entry[1]:
                v *= x[j]
        elif kind == "y":
            v = entry[2]
            for coef, idxs in entry[1]:
                term = coef
                for j in idxs:
                    term *= x[j]
                v += term
            last_y = v
        else:  # "t"
            v = last_y * last_y
        out.append(float(v))
    return np.asarray(out, dtype=np.float64)


def _heuristic_value(meta: dict, xb: dict[int, float]) -> float:
    """Objective value (model sense's raw value, not min-normalized) of the
    bit assignment *xb* (flat index -> 0/1) under the recorded description."""
    val = meta["flat_const"]
    for coef, idxs in meta["flat_terms"]:
        term = coef
        for j in idxs:
            term *= xb[j]
        val += term
    for coef, iterms, iconst in meta["squares"]:
        e = iconst
        for c2, idxs in iterms:
            term = c2
            for j in idxs:
                term *= xb[j]
            e += term
        val += coef * e * e
    return float(val)


class _FlipSearch:
    """1-flip steepest-descent machinery over the recorded objective
    description, with incremental deltas: flipping bit ``j`` changes each
    multilinear part by ``(1 - 2*b_j) * (sum of its j-monomials evaluated on
    the other factors)``, and each square by ``coef*(2*E*dE + dE**2)`` — so a
    flip probe costs O(terms containing j), not O(all terms)."""

    def __init__(self, meta: dict):
        self.meta = meta
        self.bits: list[int] = meta["bits"]
        # Per bit: flat-part monomials containing it, and per-square monomials
        # containing it (square index, coefficient, remaining factors).
        self.flat_by_bit: dict[int, list] = {j: [] for j in self.bits}
        self.sq_by_bit: dict[int, list] = {j: [] for j in self.bits}
        bitset = set(self.bits)
        for coef, idxs in meta["flat_terms"]:
            for j in set(idxs) & bitset:
                others = tuple(k for k in idxs if k != j)
                self.flat_by_bit[j].append((coef, others))
        for q, (_, iterms, _) in enumerate(meta["squares"]):
            for c2, idxs in iterms:
                for j in set(idxs) & bitset:
                    others = tuple(k for k in idxs if k != j)
                    self.sq_by_bit[j].append((q, c2, others))

    def descend(self, xb: dict[int, float], budget: list) -> float:
        """Steepest-descend *xb* in place; returns the raw objective value.
        ``budget`` is a single-element mutable flip-probe countdown shared
        across restarts (deterministic work cap)."""
        meta = self.meta
        e_vals = []
        for _, iterms, iconst in meta["squares"]:
            e = iconst
            for c2, idxs in iterms:
                term = c2
                for j in idxs:
                    term *= xb[j]
                e += term
            e_vals.append(e)
        mu = meta["sense_mul"]
        while budget[0] > 0:
            best_j, best_d, best_des = None, -1e-9, None
            for j in self.bits:
                budget[0] -= 1
                f = 1.0 - 2.0 * xb[j]
                d_flat = 0.0
                for coef, others in self.flat_by_bit[j]:
                    term = coef
                    for k in others:
                        term *= xb[k]
                    d_flat += f * term
                des: dict[int, float] = {}
                d_val = d_flat
                for q, c2, others in self.sq_by_bit[j]:
                    term = c2
                    for k in others:
                        term *= xb[k]
                    des[q] = des.get(q, 0.0) + f * term
                for q, de in des.items():
                    d_val += meta["squares"][q][0] * (2.0 * e_vals[q] * de + de * de)
                if mu * d_val < best_d:
                    best_j, best_d, best_des = j, mu * d_val, des
            if best_j is None:
                break
            xb[best_j] = 1.0 - xb[best_j]
            for q, de in (best_des or {}).items():
                e_vals[q] += de
        # Exact value of the terminal point (the incremental E_q caches only
        # steer the descent; the reported value is recomputed from scratch).
        return _heuristic_value(meta, xb)

    def run(self, restarts: int, max_flip_probes: int, seed: int):
        import random

        rng = random.Random(seed)
        mu = self.meta["sense_mul"]
        starts: list[dict[int, float]] = [
            {j: 0.0 for j in self.bits},
            {j: 1.0 for j in self.bits},
        ]
        while len(starts) < restarts:
            starts.append({j: float(rng.getrandbits(1)) for j in self.bits})
        budget = [max_flip_probes]
        best_val, best_bits = None, None
        for xb in starts:
            if budget[0] <= 0:
                break
            for jf, vf in self.meta["fixed"]:
                xb[jf] = vf
            v = self.descend(xb, budget)
            if best_val is None or mu * v < mu * best_val:
                best_val, best_bits = v, dict(xb)
        return best_val, best_bits


def heuristic_incumbent(
    reformed: Model, *, restarts: int = 512, max_flip_probes: int = 2_000_000, seed: int = 0
) -> Optional[np.ndarray]:
    """Deterministic 1-flip steepest-descent local search over the binary
    variables of a reformed model whose recorded structure proves EVERY bit
    assignment feasible (no constraints beyond the objvar defining row; all
    non-objvar variables binary-valued). Returns the best point found,
    extended over the reformed variable vector, or ``None`` when the class
    argument does not apply. Purely a primal seed: the MILP driver re-validates
    it, and a better incumbent only prunes — it never changes the certified
    optimum. Fixed ``seed``/``restarts``/``max_flip_probes`` keep the search
    deterministic (``deterministic=True`` default)."""
    meta = getattr(reformed, "_bml_heuristic", None)
    if not meta:
        return None
    bits: list[int] = meta["bits"]
    if not bits or len(bits) > 1024:
        return None
    best_val, best_bits = _FlipSearch(meta).run(restarts, max_flip_probes, seed)
    if best_bits is None:
        return None
    x0 = np.zeros(meta["n_orig_flat"], dtype=np.float64)
    for j, v in best_bits.items():
        x0[j] = v
    if meta["tau"] is not None:
        x0[meta["tau"]] = _heuristic_value(meta, best_bits)
    return extend_initial_point(reformed, x0)
