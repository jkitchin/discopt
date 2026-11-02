"""Exact collection of like terms in a scalar expression at model-build time (#2090).

The problem
-----------
A row such as ``((t + x) - t) - x + x + y == 3`` with ``t = 2**53 * x`` is, in exact
arithmetic, ``x + y == 3``. Every consumer that collects the row's coefficients in
binary64 -- ``_extract_linear_coefficients_sparse``, the Rust polynomial
``canonicalise``, the arena's ``quadratic_form`` walk, and pointwise evaluation and
forward-mode derivatives of the tree -- adds the contributions ``2**53, 1, -2**53,
-1, 1`` one at a time, and ``2**53 + 1`` rounds to ``2**53``. They all read the row
as ``y == 3``, solve that model, and certify its optimum as the optimum of the model
that was written (``optimal``, ``gap_certified=True``, wrong bound). Fixing one
collector does not help: the others, and the evaluator, still see the rounded row.

The fix
-------
Every scalar row body and objective is collected **exactly** (``fractions.Fraction``)
into a polynomial and is replaced by its canonical form
``sum(round(c_m) * m) + round(c_0)`` -- one term per monomial, so no consumer has
like terms left to add -- **only** when some monomial's contributions cancel
catastrophically: when, under *some* order of floating-point summation, the
coefficient could lose more than half its significant digits
(:func:`_sum_is_benign`). Bodies that pass are returned unchanged (the same object),
so ordinary models are not touched.

When: at the two solve doors, ``Model.validate`` and the ``solve_model`` preamble
(:func:`canonicalize_model`), before anything reads a body, and cached per row. Not
at ``subject_to``: measured interleaved over 8 x 40 000-row builds, a per-row pass
there cost +6.7 us/row on an 8.1 us/row build (GC off; +9 us with GC on), and the
solve path walks every row several times anyway. The two build-time readers of a
body get the per-item form: ``Model.constraint``'s fast path, which hands
``affine_form``'s float coefficients to the Rust builder (only rows where it merged
like terms), and ``Model.if_else``'s branch values, which derive a box at once.

Most bodies have no like terms. :func:`_fast_clean` proves that structurally,
without exact arithmetic; only the rest pay for the exact pass.

A sum that is exact in every order is always benign. The order-independent
exactness test (:func:`_sum_is_order_exact`): the
contributions ``q_k`` of one monomial are all dyadic (``n / 2**d``); let ``2**e`` be
the largest power of two dividing every one of them. Every partial sum, in any
order, is then an integer multiple of ``2**e`` of magnitude at most ``S = sum|q_k|``,
so if ``S < 2**(53 + e)`` every partial sum is exactly representable and no
collector can round. The same bound covers a factored evaluation of a product of
sums (the products' common power of two is the product of the factors'). A single
contribution is never a sum, so it is not tested: its rounding is a relative error
of a few ulps in that one coefficient, not a cancellation.

What is collected (the "polynomial skeleton"): ``+``, ``-``, unary ``neg``,
``SumOverExpression``, scalar ``Constant``, scalar variable references (a scalar
``Variable`` or an ``IndexExpression`` resolving to one entry), ``*`` (expanded),
``/`` by a constant, ``**`` by a small non-negative integer constant, the scalar dot
product ``c @ x`` of a 1-D constant and a 1-D variable, and the full reduction
``dm.sum(x)`` / ``dm.sum(C * x)``. Anything else is an opaque *atom* -- keyed by
object identity, rebuilt unchanged except that the arguments of a ``FunctionCall``,
a non-``neg`` ``UnaryOp`` or a non-polynomial ``BinaryOp`` are canonicalised
recursively (``exp(t + x - t)`` has the same defect inside the argument). A
``Parameter`` is an atom, so its value stays symbolic.

Declining is always sound here: the expression is then left exactly as written,
which is the behaviour before this module existed. It declines for a body whose
static shape is not ``()``, an expansion over :data:`_MAX_CONTRIBUTIONS`
contributions, or a non-finite constant.

Residual exposure (not handled; recorded so it is not mistaken for covered): an
opaque node that a downstream collector nevertheless expands -- e.g. a
``dm.sum`` of a nonlinear array, or ``x @ Q @ x`` -- can still share a monomial with
the visible terms of its row, and two *distinct* but structurally equal atoms are
not merged here.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    _known_shape,
    _wrap,
    affine_array_sum,
)

#: Expansion budget. A body whose exact expansion needs more contributions than this
#: is left as written (sound: that is the pre-#2090 behaviour), so a product of large
#: sums cannot make model construction quadratic.
_MAX_CONTRIBUTIONS = 20_000

#: Largest integer exponent ``**`` is expanded for.
_MAX_POWER = 4

_ONE = Fraction(1)

# A polynomial: monomial key -> list of exact contributions. A monomial key is a
# sorted tuple of factor keys (with repetition); ``()`` is the constant monomial.
# A factor key is ``("v", id(var), flat)`` for a variable entry or ``("a", id(node))``
# for an atom.
_Poly = dict


class _Decline(Exception):
    """The body is outside what this module collects; leave it as written."""


def _flat_slot(var: Variable, index) -> Optional[int]:
    from discopt._flat_index import flat_index_in_shape

    shape = tuple(var.shape or ())
    if not shape:
        return None
    return flat_index_in_shape(index, shape)


def _sum_is_order_exact(contribs: list) -> bool:
    """True iff every order of float summation of ``contribs`` is exact (module doc)."""
    if len(contribs) <= 1:
        return True
    e_min = None
    total = Fraction(0)
    for q in contribs:
        if q == 0:
            continue
        d = q.denominator
        if d & (d - 1):  # not a power of two: no float holds q exactly
            return False
        n = abs(q.numerator)
        e = ((n & -n).bit_length() - 1) - (d.bit_length() - 1)
        e_min = e if e_min is None else min(e_min, e)
        total += abs(q)
    if e_min is None:
        return True
    # total / 2**e_min < 2**53, in exact integer arithmetic.
    scaled = total / Fraction(2) ** e_min
    return bool(scaled < 2**53)


class _Collector:
    """One exact collection of a scalar expression (see module docstring)."""

    def __init__(self, memo: dict):
        # ``memo`` maps id(node) -> canonicalised node for atom-argument rewrites,
        # shared across a model-level pass so a shared subexpression is done once.
        self.memo = memo
        self.factor_nodes: dict = {}  # factor key -> node used when rebuilding
        self.n_contribs = 0
        self.dirty = False  # an atom changed, or a divisor/exponent was rounded

    # -- polynomial algebra over exact contributions --------------------------

    def _count(self, k: int) -> None:
        self.n_contribs += k
        if self.n_contribs > _MAX_CONTRIBUTIONS:
            raise _Decline("expansion budget exceeded")

    def _const(self, value: float) -> _Poly:
        v = float(value)
        if not np.isfinite(v):
            raise _Decline("non-finite constant")
        self._count(1)
        return {(): [Fraction(v)]}

    def _factor(self, fkey, node) -> _Poly:
        self.factor_nodes.setdefault(fkey, node)
        self._count(1)
        return {(fkey,): [_ONE]}

    @staticmethod
    def _add(p: _Poly, q: _Poly) -> _Poly:
        out = {k: list(v) for k, v in p.items()}
        for k, v in q.items():
            out.setdefault(k, []).extend(v)
        return out

    def _scale(self, p: _Poly, s: Fraction) -> _Poly:
        return {k: [c * s for c in v] for k, v in p.items()}

    def _mul(self, p: _Poly, q: _Poly) -> _Poly:
        self._count(sum(map(len, p.values())) * sum(map(len, q.values())))
        out: _Poly = {}
        for k1, v1 in p.items():
            for k2, v2 in q.items():
                key = tuple(sorted(k1 + k2))
                out.setdefault(key, []).extend(a * b for a in v1 for b in v2)
        return out

    def _exact_constant(self, p: _Poly) -> Optional[Fraction]:
        """Exact value of a constant polynomial, or ``None`` if it has a variable."""
        if any(k != () for k in p):
            return None
        contribs = p.get((), [])
        if not _sum_is_benign(contribs):
            # A float collector computes this constant with a cancellation error, so
            # whatever it scales reads differently in floats: rewrite the body.
            self.dirty = True
        return sum(contribs, Fraction(0))

    # -- the walk --------------------------------------------------------------

    def _atom(self, node: Expression) -> _Poly:
        new = canonicalize_expression(node, memo=self.memo, _atom_children_only=True)
        if new is not node:
            self.dirty = True
        return self._factor(("a", id(new)), new)

    def collect(self, root: Expression) -> _Poly:
        """Exact polynomial of ``root``. Iterative post-order (deep ``+`` chains)."""
        results: dict = {}
        stack: list = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            nid = id(node)
            if nid in results and not expanded:
                continue
            children = self._children(node)
            if children is None:  # leaf of the skeleton
                results[nid] = self._leaf(node)
                continue
            if not expanded:
                stack.append((node, True))
                for ch in reversed(children):
                    if id(ch) not in results:
                        stack.append((ch, False))
                continue
            results[nid] = self._combine(node, [results[id(ch)] for ch in children])
        out: _Poly = results[id(root)]
        return out

    def _children(self, node) -> Optional[list]:
        """Skeleton children of ``node``, or ``None`` if it is collected as a leaf."""
        if isinstance(node, BinaryOp):
            if node.op in ("+", "-", "*"):
                return [node.left, node.right]
            if node.op == "/" and isinstance(node.right, Constant):
                return [node.left, node.right]
            if node.op == "**" and self._small_power(node.right) is not None:
                return [node.left]
            return None
        if isinstance(node, UnaryOp) and node.op == "neg":
            return [node.operand]
        if isinstance(node, SumOverExpression):
            return list(node.terms)
        return None

    @staticmethod
    def _small_power(expo) -> Optional[int]:
        if not isinstance(expo, Constant) or expo.value.ndim != 0:
            return None
        v = float(expo.value)
        if v.is_integer() and 0 <= v <= _MAX_POWER:
            return int(v)
        return None

    def _leaf(self, node) -> _Poly:
        if isinstance(node, Constant):
            if node.value.ndim != 0:
                raise _Decline("array constant in scalar position")
            return self._const(node.value)
        if isinstance(node, Variable):
            if node.size != 1 or tuple(node.shape or ()) != ():
                raise _Decline("non-scalar variable in scalar position")
            return self._factor(("v", id(node), 0), node)
        if isinstance(node, IndexExpression) and isinstance(node.base, Variable):
            flat = _flat_slot(node.base, node.index)
            if flat is None:
                return self._atom(node)
            return self._factor(("v", id(node.base), flat), node)
        if isinstance(node, MatMulExpression):
            return self._matmul(node)
        if isinstance(node, SumExpression):
            aff = affine_array_sum(node)
            if aff is not None:
                var, coefs = aff
                return self._linear_array(var, coefs)
            return self._atom(node)
        return self._atom(node)

    def _matmul(self, node) -> _Poly:
        if isinstance(node.left, Constant) and isinstance(node.right, Variable):
            mat, var = node.left.value, node.right
        elif isinstance(node.right, Constant) and isinstance(node.left, Variable):
            mat, var = node.right.value, node.left
        else:
            return self._atom(node)
        if mat.ndim != 1 or len(var.shape) != 1 or mat.shape[0] != var.size:
            return self._atom(node)
        return self._linear_array(var, np.asarray(mat, dtype=np.float64))

    def _linear_array(self, var: Variable, coefs: np.ndarray) -> _Poly:
        shape = tuple(var.shape)
        self._count(int(coefs.size))
        if not np.all(np.isfinite(coefs)):
            raise _Decline("non-finite coefficient")
        out: _Poly = {}
        for j in range(int(coefs.size)):
            c = float(coefs[j])
            if c == 0.0:
                continue
            fkey = ("v", id(var), j)
            if fkey not in self.factor_nodes:
                idx = j if len(shape) == 1 else tuple(int(i) for i in np.unravel_index(j, shape))
                self.factor_nodes[fkey] = var[idx]
            out.setdefault((fkey,), []).append(Fraction(c))
        return out

    def _combine(self, node, parts: list) -> _Poly:
        if isinstance(node, SumOverExpression):
            acc: _Poly = {}
            for p in parts:
                acc = self._add(acc, p)
            return acc
        if isinstance(node, UnaryOp):  # neg
            return self._scale(parts[0], Fraction(-1))
        op = node.op
        if op == "+":
            return self._add(parts[0], parts[1])
        if op == "-":
            return self._add(parts[0], self._scale(parts[1], Fraction(-1)))
        if op == "*":
            return self._mul(parts[0], parts[1])
        if op == "/":
            d = self._exact_constant(parts[1])
            if d is None or d == 0:
                raise _Decline("division by zero or by a non-constant")
            return self._scale(parts[0], 1 / d)
        if op == "**":
            k = self._small_power(node.right)
            assert k is not None
            out: _Poly = {(): [_ONE]}
            for _ in range(k):
                out = self._mul(out, parts[0])
            return out
        raise AssertionError(f"unexpected skeleton op {op!r}")  # pragma: no cover


def _floats_sum_order_exact(values: list) -> bool:
    """:func:`_sum_is_order_exact` for plain floats, without building Fractions."""
    if len(values) <= 1:
        return True
    parts = []
    for v in values:
        if v == 0.0:
            continue
        n, d = v.as_integer_ratio()  # d is a power of two for every finite float
        n = abs(n)
        parts.append((n, d.bit_length() - 1, ((n & -n).bit_length() - 1) - (d.bit_length() - 1)))
    if not parts:
        return True
    e_min = min(p[2] for p in parts)
    # sum(n / 2**k) / 2**e_min, every term an integer since k + e_min <= v2(n).
    scaled = sum(n >> (k + e_min) if k + e_min >= 0 else n << -(k + e_min) for n, k, _ in parts)
    return bool(scaled < 2**53)


class _NotClean(Exception):
    """The fast pre-pass cannot prove the body needs no rewrite."""


def _fast_clean(root: Expression, memo: dict) -> bool:
    """True when ``root`` provably needs no rewrite, decided without exact arithmetic.

    The common row has no like terms at all. This walks the additive skeleton once,
    reading each term as a product of leaves (a constant factor is a scale, not a
    term) and checks that every monomial occurs once, that the additive constants
    sit under no scale and sum exactly in every order, and that no atom's arguments
    change. Anything else -- a repeated monomial, a product of sums, a power of a
    non-leaf, ``c @ x`` / ``dm.sum`` sharing a variable with another term -- returns
    ``False`` and the exact pass decides. ``False`` is never wrong, only slower.

    Dispatch is on exact ``type`` (none of these node classes is subclassed); this
    runs once per row at build time, so it is kept to a few microseconds (#1215).
    """
    seen: set = set()
    scalar_vars: set = set()
    block_vars: set = set()
    consts: list = []
    try:
        stack: list = [(root, False)]  # (node, under a scale)
        pop, push = stack.pop, stack.append
        while stack:
            node, scaled = pop()
            t = type(node)
            if t is BinaryOp:
                op = node.op
                if op == "+" or op == "-":
                    push((node.left, scaled))
                    push((node.right, scaled))
                    continue
                if op == "*":
                    if type(node.left) is Constant:
                        push((node.right, True))
                        continue
                    if type(node.right) is Constant:
                        push((node.left, True))
                        continue
                elif op == "/" and type(node.right) is Constant:
                    push((node.left, True))
                    continue
            elif t is UnaryOp:
                if node.op == "neg":
                    push((node.operand, scaled))
                    continue
            elif t is SumOverExpression:
                stack.extend([(term, scaled) for term in node.terms])
                continue
            elif t is Constant:
                if scaled or node.value.ndim != 0:
                    raise _NotClean
                consts.append(float(node.value))
                continue
            elif t is MatMulExpression or t is SumExpression:
                var = _array_linear_var(node)
                if var is None:
                    key: tuple = (_fast_atom_key(node, memo),)
                    if key in seen:
                        raise _NotClean
                    seen.add(key)
                    continue
                if id(var) in block_vars:
                    raise _NotClean
                block_vars.add(id(var))
                continue
            # A term: a product of leaves.
            if t is IndexExpression or t is Variable:
                mono: tuple = (_fast_leaf_key(node, memo, scalar_vars),)
            else:
                mono = tuple(sorted(_fast_monomial(node, memo, scalar_vars)))
            if mono in seen:
                raise _NotClean
            seen.add(mono)
        if block_vars and not scalar_vars.isdisjoint(block_vars):
            raise _NotClean
    except _NotClean:
        return False
    return len(consts) <= 1 or _floats_sum_order_exact(consts)


def _array_linear_var(node) -> Optional[Variable]:
    if type(node) is SumExpression:
        aff = affine_array_sum(node)
        return None if aff is None else aff[0]
    left, right = node.left, node.right
    if type(left) is Constant and type(right) is Variable:
        mat, var = left.value, right
    elif type(right) is Constant and type(left) is Variable:
        mat, var = right.value, left
    else:
        return None
    if mat.ndim != 1 or len(var.shape) != 1 or mat.shape[0] != var.size:
        return None
    return var


def _fast_atom_key(node, memo: dict) -> tuple:
    new = canonicalize_expression(node, memo=memo, _atom_children_only=True)
    if new is not node:
        raise _NotClean
    return ("a", id(node))


def _fast_monomial(node, memo: dict, scalar_vars: set) -> list:
    """Factor keys of a product-of-leaves term, or raise :class:`_NotClean`."""
    factors: list = []
    stack = [node]
    while stack:
        n = stack.pop()
        t = type(n)
        if t is BinaryOp:
            op = n.op
            if op == "*":
                stack.append(n.left)
                stack.append(n.right)
                continue
            if op == "/" and type(n.right) is Constant:
                stack.append(n.left)
                continue
            if op == "**":
                k = _Collector._small_power(n.right)
                if k is not None:
                    base = n.left
                    if type(base) is not Variable and type(base) is not IndexExpression:
                        raise _NotClean
                    factors.extend([_fast_leaf_key(base, memo, scalar_vars)] * k)
                    continue
            elif op == "+" or op == "-":
                raise _NotClean  # a product of sums: the exact pass expands it
        elif t is Constant:
            if n.value.ndim != 0:
                raise _NotClean
            continue
        elif t is SumOverExpression or t is MatMulExpression or t is SumExpression:
            raise _NotClean
        elif t is UnaryOp and n.op == "neg":
            raise _NotClean
        factors.append(_fast_leaf_key(n, memo, scalar_vars))
    return factors


def _fast_leaf_key(n, memo: dict, scalar_vars: set) -> tuple:
    t = type(n)
    if t is IndexExpression:
        base = n.base
        if type(base) is Variable:
            idx = n.index
            shape = base._shape  # a Variable always caches its declared shape tuple
            flat: Optional[int]
            if type(idx) is int and len(shape) == 1:
                flat = idx + shape[0] if idx < 0 else idx
                if not 0 <= flat < shape[0]:
                    flat = None
            else:
                flat = _flat_slot(base, idx)
            if flat is not None:
                scalar_vars.add(id(base))
                return ("v", id(base), flat)
    elif t is Variable:
        if n.size != 1 or tuple(n.shape or ()) != ():
            raise _NotClean
        scalar_vars.add(id(n))
        return ("v", id(n), 0)
    return _fast_atom_key(n, memo)


#: ``kappa * (n - 1)`` above which a sum is rewritten; see :func:`_sum_is_benign`.
_CANCELLATION_LIMIT = 2**27


def _sum_is_benign(contribs: list) -> bool:
    """True unless float summation of ``contribs`` can lose over half its digits.

    Any-order binary64 summation of ``n`` terms has relative error at most
    ``(n - 1) * u * kappa`` (``u = 2**-53``, ``kappa = sum|q| / |sum q|``, the
    condition number of the sum). The #2090 defect is catastrophic cancellation:
    ``kappa ~ 2**53``, so the coefficient is wrong in its leading digit. Ordinary
    data rounds at the ulp level instead -- measured on the in-repo corpus before
    this test existed, the "exact in every order" test alone rewrote rows of
    ``dispatch``/``ex8_5_3`` (symmetric ``x*y + y*x`` halves, ``kappa = 1``),
    ``ex1222`` (constants ``1.25 + 0.8``, ``kappa = 1``) and ``nvs04`` (expanded
    squares, ``kappa = 35``), and the rewrite changed seven smoke results. Those
    rows' coefficients are already right to ~1e-14; rewriting them is churn, not a
    fix. So a sum is rewritten only when its error bound exceeds ``2**-26`` (half
    the significand): ``kappa * (n - 1) > 2**27``, which leaves ~24 binary orders
    of magnitude between the defect and the corpus on either side. A sum that is
    exact in every order (:func:`_sum_is_order_exact`) is benign whatever its
    ``kappa`` -- that covers a full cancellation ``M - M`` to an exact zero.
    """
    if _sum_is_order_exact(contribs):
        return True
    total = sum(contribs, Fraction(0))
    if total == 0:
        return False  # inexact terms cancelling to zero: any float order may not
    n = sum(1 for q in contribs if q != 0)
    abs_sum = sum((abs(q) for q in contribs), Fraction(0))
    return bool(abs_sum * (n - 1) <= _CANCELLATION_LIMIT * abs(total))


def _needs_rewrite(poly: _Poly) -> bool:
    return any(not _sum_is_benign(v) for v in poly.values())


def _rebuild(poly: _Poly, factor_nodes: dict) -> Expression:
    terms: list = []
    const_term = None
    for key, contribs in poly.items():
        exact = sum(contribs, Fraction(0))
        try:
            c = float(exact)  # correctly rounded
        except OverflowError:
            raise _Decline("coefficient outside the binary64 range") from None
        if c == 0.0:
            continue
        if key == ():
            const_term = _wrap(c)
            continue
        mono: Optional[Expression] = None
        i = 0
        while i < len(key):
            j = i
            while j < len(key) and key[j] == key[i]:
                j += 1
            node = factor_nodes[key[i]]
            factor = node if j - i == 1 else BinaryOp("**", node, _wrap(float(j - i)))
            mono = factor if mono is None else BinaryOp("*", mono, factor)
            i = j
        assert mono is not None
        terms.append(mono if c == 1.0 else BinaryOp("*", _wrap(c), mono))
    if const_term is not None:
        terms.append(const_term)
    if not terms:
        return _wrap(0.0)
    if len(terms) == 1:
        only: Expression = terms[0]
        return only
    return SumOverExpression(terms)


def _rebuild_atom_children(node: Expression, memo: dict) -> Expression:
    """``node`` with its arguments canonicalised, or ``node`` itself if none changed."""
    if isinstance(node, FunctionCall):
        new_args = tuple(canonicalize_expression(a, memo=memo) for a in node.args)
        if all(a is b for a, b in zip(new_args, node.args)):
            return node
        return FunctionCall(node.func_name, *new_args)
    if isinstance(node, UnaryOp) and node.op != "neg":
        new = canonicalize_expression(node.operand, memo=memo)
        return node if new is node.operand else UnaryOp(node.op, new)
    if isinstance(node, BinaryOp):
        left = canonicalize_expression(node.left, memo=memo)
        right = canonicalize_expression(node.right, memo=memo)
        if left is node.left and right is node.right:
            return node
        return BinaryOp(node.op, left, right)
    return node


def canonicalize_expression(
    expr: Expression, *, memo: Optional[dict] = None, _atom_children_only: bool = False
) -> Expression:
    """Return ``expr``, or its exact canonical form if float collection could round.

    The returned object *is* ``expr`` whenever no rewrite is needed. See the module
    docstring for what is collected and when a rewrite happens.
    """
    if memo is None:
        memo = {}
    key = (id(expr), _atom_children_only)
    hit = memo.get(key)
    if hit is not None:
        cached: Expression = hit[1]
        return cached
    if _atom_children_only:
        out = _rebuild_atom_children(expr, memo)
    elif not isinstance(expr, Expression) or _known_shape(expr) != ():
        out = expr
    elif _fast_clean(expr, memo):
        out = expr
    else:
        coll = _Collector(memo)
        try:
            poly = coll.collect(expr)
            out = (
                _rebuild(poly, coll.factor_nodes) if (coll.dirty or _needs_rewrite(poly)) else expr
            )
        except _Decline:
            out = expr
    # Keep ``expr`` alive in the memo so its id cannot be reused within this pass.
    memo[key] = (expr, out)
    return out


def canonicalize_constraint(constraint, memo: Optional[dict] = None) -> None:
    """Replace ``constraint.body`` in place by its canonical form when needed.

    In place so the caller's object stays the model's row (identity-keyed APIs).
    The checked body is remembered on the row (``_exact_body``), so a row is
    walked once however many solves it goes through; assigning a new ``body``
    invalidates that, since the check is by identity.
    """
    body = constraint.body
    if getattr(constraint, "_exact_body", None) is body:
        return
    new = canonicalize_expression(body, memo=memo)
    if new is not body:
        constraint.body = new
    constraint._exact_body = new


def canonicalize_model(model) -> None:
    """:func:`canonicalize_rows_in` over every row of ``model``, and its objective."""
    memo: dict = {}
    for item in model._constraints:
        canonicalize_rows_in(item, memo)
    obj = model._objective
    if obj is not None and getattr(obj, "_exact_expression", None) is not obj.expression:
        new = canonicalize_expression(obj.expression, memo=memo)
        if new is not obj.expression:
            obj.expression = new
        obj._exact_expression = new


def canonicalize_rows_in(item, memo: Optional[dict] = None) -> None:
    """:func:`canonicalize_constraint` on every row nested in ``item``.

    Mirrors ``core._reject_unnormalized_rhs_in``: a ``Constraint``, an
    ``_IndicatorConstraint``, a ``_DisjunctiveConstraint`` (through its disjuncts),
    or a list/tuple of those.
    """
    from discopt.modeling.core import (
        Constraint,
        _DisjunctiveConstraint,
        _IndicatorConstraint,
    )

    if isinstance(item, Constraint):
        canonicalize_constraint(item, memo)
    elif isinstance(item, _IndicatorConstraint):
        canonicalize_rows_in(item.constraint, memo)
    elif isinstance(item, _DisjunctiveConstraint):
        canonicalize_rows_in(item.disjuncts, memo)
    elif isinstance(item, (list, tuple)):
        for sub in item:
            canonicalize_rows_in(sub, memo)
