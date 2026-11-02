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
recursively (``exp(t + x - t)`` has the same defect inside the argument).

Parameters. A scalar ``Parameter`` (or a scalar entry of one) is a symbolic factor
of its monomial, so a rewrite keeps it symbolic and ``p*z - p*z`` cancels exactly
for every value. But every downstream collector substitutes the value, so the
*decision* groups terms by their variable monomial with the current values
substituted -- ``p*z + 0.5*z - p*z`` at ``p = 2**53`` is the #2090 defect even
though no two of its monomials are equal symbolically (review of #2094). When even
the symbolic canonical form still cancels at the current values (``p*z - q*z`` with
``p == q``), no body that keeps the Parameters symbolic is exact, so
:class:`ParameterCancellationError` is raised rather than freezing the values.
Because the decision depends on values, the per-row cache also snapshots them, and
a changed value re-runs the check at the next solve.

A rewritten body is the **fully expanded** sum of monomials: once one monomial
triggers a rewrite, ``(x - 1)**2`` elsewhere in the same body becomes
``x**2 - 2*x + 1`` and ``x / 3`` becomes ``0.333... * x``. That is equal in exact
arithmetic and sound, but a relaxation built from the expanded form can be weaker
than one built from the factored form. It only happens to rewritten bodies (none in
the in-repo corpus).

Declining is always sound here: the expression is then left exactly as written,
which is the behaviour before this module existed. It declines for a body whose
static shape is not ``()``, an expansion over :data:`_MAX_CONTRIBUTIONS`
contributions, or a non-finite constant.

Residual exposure (not handled; recorded so it is not mistaken for covered): an
opaque node that a downstream collector nevertheless expands -- e.g. a
``dm.sum`` of a nonlinear array, or ``x @ Q @ x`` -- can still share a monomial with
the visible terms of its row, and two *distinct* but structurally equal atoms are
not merged here. A coefficient whose exact value is **zero** has no relative
error, so its sum is judged against the row's largest exact coefficient instead
(:func:`_sum_is_benign`): in a row dominated by a much larger coefficient, a
zero coefficient may keep a float residue below ``2**-26`` of that scale --
below any row-relative feasibility tolerance, but weaker than the per-coefficient
test every nonzero coefficient gets.
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
    Parameter,
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


def _flat_slot(var: Variable | Parameter, index) -> Optional[int]:
    from discopt._flat_index import flat_index_in_shape

    shape = tuple(var.shape or ())
    if not shape:
        return None
    return flat_index_in_shape(index, shape)


def _sum_is_order_exact(contribs: list) -> bool:
    """True if every order of float summation of ``contribs`` is provably exact.

    Sufficient, not necessary: the power-of-two bound in the module docstring,
    plus the exact answer for two float terms (:func:`_two_floats_add_exactly`).
    """
    if len(contribs) <= 1:
        return True
    nonzero = [q for q in contribs if q != 0]
    if len(nonzero) == 2 and all(_is_binary64(q) for q in nonzero):
        return _two_floats_add_exactly(float(nonzero[0]), float(nonzero[1]))
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


def _is_binary64(q: Fraction) -> bool:
    try:
        return Fraction(float(q)) == q
    except OverflowError:
        return False


def _two_floats_add_exactly(a: float, b: float) -> bool:
    """Whether ``a + b`` is exact in binary64. Two terms have one order, so this
    is exact rather than a bound: it holds for ``a - a`` and, by Sterbenz, for any
    difference of floats within a factor of two -- both measured as false
    rewrites under the bound alone (``0.50001 - 0.50001`` in
    ``test_nn_equivalence``, ``0.20000000004654214 - 0.2`` in ``test_mo_nbi_fast``).
    """
    s = a + b
    if s in (float("inf"), float("-inf")) or s != s:
        return False
    return Fraction(s) == Fraction(a) + Fraction(b)


class ParameterCancellationError(ValueError):
    """A row's coefficient cancels catastrophically at the current Parameter values.

    Raised when even the exact symbolic canonical form (like terms collected per
    monomial *including* its Parameter factors, so ``p*z - p*z`` is gone) still
    leaves terms that downstream collectors add in binary64 with a catastrophic
    cancellation once the Parameters' values are substituted -- e.g. ``p*z - q*z``
    with ``p == q == 2**53``. No body that keeps the Parameters symbolic is exact
    for those values, and substituting the values would silently freeze them
    (breaking re-solves and sensitivities), so this refuses (CLAUDE.md §3).
    """


class _Collector:
    """One exact collection of a scalar expression (see module docstring).

    Additive chains (``+``, ``-``, ``neg``, ``SumOverExpression``, and scaling by a
    constant) are flattened into signed terms and merged **once** into a single
    accumulator, so a row costs time linear in its size. (The first version
    combined child polynomials pairwise and copied the accumulator at every ``+``:
    measured 0.70 / 2.5 / 11.0 / 44.1 s for a 1k / 2k / 4k / 8k-term row with one
    repeated variable -- review of #2094.)
    """

    def __init__(self, memo: dict, params: list):
        # ``memo`` maps id(node) -> canonicalised node for atom-argument rewrites,
        # shared across a model-level pass so a shared subexpression is done once.
        self.memo = memo
        self.params = params  # every Parameter read, for the caller's value cache
        self.factor_nodes: dict = {}  # factor key -> node used when rebuilding
        self.param_values: dict = {}  # ("p", ...) factor key -> exact current value
        self.terms: dict = {}  # id(node) -> poly of a non-additive term (DAG reuse)
        self.n_contribs = 0
        self.dirty = False  # an atom's arguments were rewritten

    # -- polynomial algebra over exact contributions --------------------------

    def _count(self, k: int) -> None:
        self.n_contribs += k
        if self.n_contribs > _MAX_CONTRIBUTIONS:
            raise _Decline("expansion budget exceeded")

    @staticmethod
    def _frac(node) -> Fraction:
        v = float(node.value)
        if not np.isfinite(v):
            raise _Decline("non-finite constant")
        return Fraction(v)

    def _factor(self, fkey, node) -> _Poly:
        self.factor_nodes.setdefault(fkey, node)
        return {(fkey,): [_ONE]}

    def _mul(self, p: _Poly, q: _Poly) -> _Poly:
        self._count(sum(map(len, p.values())) * sum(map(len, q.values())))
        out: _Poly = {}
        for k1, v1 in p.items():
            for k2, v2 in q.items():
                key = tuple(sorted(k1 + k2))
                out.setdefault(key, []).extend(a * b for a in v1 for b in v2)
        return out

    @staticmethod
    def _small_power(expo) -> Optional[int]:
        if type(expo) is not Constant or expo.value.ndim != 0:
            return None
        v = float(expo.value)
        if v.is_integer() and 0 <= v <= _MAX_POWER:
            return int(v)
        return None

    # -- the walk --------------------------------------------------------------

    def collect(self, root: Expression) -> _Poly:
        return self._sum_poly(root)

    def _sum_poly(self, root) -> _Poly:
        """Exact polynomial of ``root``: flatten its additive chain, merge once."""
        acc: _Poly = {}
        stack: list = [(root, _ONE)]
        pop, push = stack.pop, stack.append
        while stack:
            node, s = pop()
            t = type(node)
            if t is BinaryOp:
                op = node.op
                if op == "+":
                    push((node.right, s))
                    push((node.left, s))
                    continue
                if op == "-":
                    push((node.right, -s))
                    push((node.left, s))
                    continue
                if op == "*":
                    if type(node.left) is Constant and node.left.value.ndim == 0:
                        push((node.right, s * self._frac(node.left)))
                        continue
                    if type(node.right) is Constant and node.right.value.ndim == 0:
                        push((node.left, s * self._frac(node.right)))
                        continue
                elif op == "/" and type(node.right) is Constant and node.right.value.ndim == 0:
                    d = self._frac(node.right)
                    if d == 0:
                        raise _Decline("division by zero")
                    push((node.left, s / d))
                    continue
            elif t is UnaryOp and node.op == "neg":
                push((node.operand, -s))
                continue
            elif t is SumOverExpression:
                stack.extend([(term, s) for term in reversed(node.terms)])
                continue
            elif t is Constant:
                if node.value.ndim != 0:
                    raise _Decline("array constant in scalar position")
                self._count(1)
                acc.setdefault((), []).append(s * self._frac(node))
                continue
            term = self.terms.get(id(node))
            if term is None:
                term = self._term(node)
                self.terms[id(node)] = term
            for k, v in term.items():
                self._count(len(v))
                lst = acc.get(k)
                if lst is None:
                    acc[k] = lst = []
                lst.extend(v if s == 1 else [q * s for q in v])
        return acc

    def _term(self, node) -> _Poly:
        """Polynomial of a non-additive node: a product, a small power or a leaf."""
        t = type(node)
        if t is BinaryOp:
            op = node.op
            if op == "*":
                factors: list = []
                scale = _ONE
                stack = [node]
                while stack:
                    n = stack.pop()
                    if type(n) is BinaryOp and n.op == "*":
                        stack.append(n.right)
                        stack.append(n.left)
                    elif type(n) is Constant and n.value.ndim == 0:
                        scale *= self._frac(n)
                    else:
                        factors.append(n)
                out: _Poly = {(): [scale]}
                for f in factors:
                    out = self._mul(out, self._sum_poly(f))
                return out
            if op == "**":
                k = self._small_power(node.right)
                if k is not None:
                    base = self._sum_poly(node.left)
                    out = {(): [_ONE]}
                    for _ in range(k):
                        out = self._mul(out, base)
                    return out
            return self._atom(node)
        if t is Variable:
            if node.size != 1 or tuple(node.shape or ()) != ():
                raise _Decline("non-scalar variable in scalar position")
            return self._factor(("v", id(node), 0), node)
        if t is Parameter:
            if tuple(node.shape or ()) != ():
                raise _Decline("non-scalar parameter in scalar position")
            return self._param(node, node, 0)
        if t is IndexExpression:
            base = node.base
            if type(base) is Variable:
                flat = _flat_slot(base, node.index)
                if flat is not None:
                    return self._factor(("v", id(base), flat), node)
            elif type(base) is Parameter:
                flat = _flat_slot(base, node.index)
                if flat is not None:
                    return self._param(base, node, flat)
            return self._atom(node)
        if t is MatMulExpression:
            var = _array_linear_var(node)
            if var is None:
                return self._atom(node)
            mat = node.left.value if type(node.left) is Constant else node.right.value
            return self._linear_array(var, np.asarray(mat, dtype=np.float64))
        if t is SumExpression:
            aff = affine_array_sum(node)
            if aff is not None:
                return self._linear_array(*aff)
            return self._atom(node)
        return self._atom(node)

    def _param(self, base, node, flat: int) -> _Poly:
        v = float(np.asarray(base.value, dtype=np.float64).reshape(-1)[flat])
        if not np.isfinite(v):
            raise _Decline("non-finite parameter value")
        fkey = ("p", id(base), flat)
        self.param_values[fkey] = Fraction(v)
        self.params.append(base)
        return self._factor(fkey, node)

    def _atom(self, node: Expression) -> _Poly:
        new = canonicalize_expression(
            node, memo=self.memo, _atom_children_only=True, _params=self.params
        )
        if new is not node:
            self.dirty = True
        return self._factor(("a", id(new)), new)

    def _linear_array(self, var: Variable, coefs: np.ndarray) -> _Poly:
        shape = tuple(var.shape)
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
    if len(parts) == 2:
        a, b = (v for v in values if v != 0.0)
        return _two_floats_add_exactly(a, b)
    e_min = min(p[2] for p in parts)
    # sum(n / 2**k) / 2**e_min, every term an integer since k + e_min <= v2(n).
    scaled = sum(n >> (k + e_min) if k + e_min >= 0 else n << -(k + e_min) for n, k, _ in parts)
    return bool(scaled < 2**53)


class _NotClean(Exception):
    """The fast pre-pass cannot prove the body needs no rewrite."""


def _fast_clean(root: Expression, memo: dict, found: list) -> bool:
    """True when ``root`` provably needs no rewrite, decided without exact arithmetic.

    The common row has no like terms at all. This walks the additive skeleton once,
    reading each term as a product of leaves (a constant factor is a scale, not a
    term; so is a Parameter, since every downstream collector substitutes its
    value) and checks that every variable monomial occurs once, that the additive
    constants sit under no scale and sum exactly in every order, and that no atom's
    arguments change. Anything else -- a repeated monomial, a product of sums, a
    power of a non-leaf, ``c @ x`` / ``dm.sum`` sharing a variable with another
    term -- returns ``False`` and the exact pass decides. ``False`` is never wrong,
    only slower. A ``True`` holds for *every* Parameter value: each variable
    monomial then has one contribution, so there is no sum to cancel.

    The constants are tested with their signs dropped (``x + 0.3 - 0.1`` is tested
    as ``0.3, 0.1``). That is deliberate and safe: with two terms the result is
    the same rounding either way, and with more the test uses only magnitudes.

    Dispatch is on exact ``type`` (none of these node classes is subclassed); this
    runs once per row, so it is kept to a few microseconds (#1215).
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
                    key: tuple = (_fast_atom_key(node, memo, found),)
                    if key in seen:
                        raise _NotClean
                    seen.add(key)
                    continue
                if id(var) in block_vars:
                    raise _NotClean
                block_vars.add(id(var))
                continue
            # A term: a product of leaves (Parameter factors dropped, as scales).
            if t is IndexExpression or t is Variable:
                leaf = _fast_leaf_key(node, memo, scalar_vars, found)
                mono: tuple = () if leaf is None else (leaf,)
            else:
                mono = tuple(sorted(_fast_monomial(node, memo, scalar_vars, found)))
            if mono in seen:
                raise _NotClean
            seen.add(mono)
        if block_vars and not scalar_vars.isdisjoint(block_vars):
            raise _NotClean
        if consts and () in seen:  # a Parameter-only term next to a constant term
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


def _fast_atom_key(node, memo: dict, found: list) -> tuple:
    new = canonicalize_expression(node, memo=memo, _atom_children_only=True, _params=found)
    if new is not node:
        raise _NotClean
    return ("a", id(node))


def _fast_monomial(node, memo: dict, scalar_vars: set, found: list) -> list:
    """Variable factor keys of a product-of-leaves term, or raise :class:`_NotClean`."""
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
                    tb = type(base)
                    if tb is not Variable and tb is not IndexExpression and tb is not Parameter:
                        raise _NotClean
                    leaf = _fast_leaf_key(base, memo, scalar_vars, found)
                    if leaf is not None:
                        factors.extend([leaf] * k)
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
        leaf = _fast_leaf_key(n, memo, scalar_vars, found)
        if leaf is not None:
            factors.append(leaf)
    return factors


def _fast_leaf_key(n, memo: dict, scalar_vars: set, found: list) -> Optional[tuple]:
    """Factor key of a leaf; ``None`` for a scalar Parameter (a scale, not a factor)."""
    t = type(n)
    if t is IndexExpression:
        base = n.base
        tb = type(base)
        if tb is Variable:
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
        elif tb is Parameter and _flat_slot(base, n.index) is not None:
            found.append(base)
            return None
    elif t is Variable:
        if n.size != 1 or tuple(n.shape or ()) != ():
            raise _NotClean
        scalar_vars.add(id(n))
        return ("v", id(n), 0)
    elif t is Parameter:
        if tuple(n.shape or ()) != ():
            raise _NotClean
        found.append(n)
        return None
    return _fast_atom_key(n, memo, found)


#: ``kappa * (n - 1)`` above which a sum is rewritten; see :func:`_sum_is_benign`.
_CANCELLATION_LIMIT = 2**27


def _sum_is_benign(contribs: list, scale: Optional[Fraction] = None) -> bool:
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
    n = sum(1 for q in contribs if q != 0)
    abs_sum = sum((abs(q) for q in contribs), Fraction(0))
    if total == 0:
        # A coefficient that is exactly 0 has no relative error to bound, so the
        # same error bound is measured against the row's ``scale`` (its largest
        # exact coefficient) instead. Measured: the hull perspective rows of
        # ``test_1846`` carry symbolically matched pairs ``[1e-8, -1e-8, -4e-8,
        # 4e-8]`` -- residue <= ~1e-23 next to O(1) coefficients, not a defect --
        # while the issue's ``(t + a) - t - a`` (``[M, 1, -M, -1]``, nothing else
        # in the row) can read as ``-a``: an O(1) error in a row of zeros.
        if scale is None or scale == 0:
            return False
        return bool(abs_sum * (n - 1) <= _CANCELLATION_LIMIT * scale)
    return bool(abs_sum * (n - 1) <= _CANCELLATION_LIMIT * abs(total))


def _groups_benign(groups: list) -> list:
    """The groups of one body that are *not* benign, judged against its scale."""
    scale = max((abs(sum(g, Fraction(0))) for g in groups), default=Fraction(0))
    return [g for g in groups if not _sum_is_benign(g, scale)]


def _by_variable_monomial(poly: _Poly, param_values: dict) -> list:
    """The sums downstream collectors form: Parameter values substituted, so terms
    are grouped by their *variable* monomial alone."""
    if not param_values:
        return list(poly.values())
    groups: dict = {}
    for key, contribs in poly.items():
        pv = _ONE
        vkey = []
        for f in key:
            if f[0] == "p":
                pv *= param_values[f]
            else:
                vkey.append(f)
        lst = groups.setdefault(tuple(vkey), [])
        lst.extend(contribs if pv == 1 else [q * pv for q in contribs])
    return list(groups.values())


def _needs_rewrite(poly: _Poly, param_values: dict) -> bool:
    return bool(_groups_benign(_by_variable_monomial(poly, param_values)))


def _check_rewrite_is_exact(poly: _Poly, param_values: dict) -> None:
    """Refuse when the symbolic canonical form still cancels at these values."""
    if not param_values:
        return  # one term per variable monomial: nothing is left to add
    rounded: _Poly = {}
    for key, contribs in poly.items():
        c = Fraction(float(sum(contribs, Fraction(0))))
        if c != 0:
            rounded[key] = [c]
    for group in _groups_benign(_by_variable_monomial(rounded, param_values)):
        if group:
            raise ParameterCancellationError(
                "terms of one variable monomial, with Parameter coefficients, cancel "
                f"catastrophically at the current Parameter values (contributions "
                f"{[float(q) for q in group][:6]}, exact sum "
                f"{float(sum(group, Fraction(0)))!r}); binary64 collection would read a "
                "different coefficient, and no body that keeps the Parameters symbolic "
                "is exact for these values. Combine those Parameters into one before "
                "multiplying the variable (e.g. write (p - q) * z, not p*z - q*z), or "
                "use a constant (#2090)."
            )


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


def _rebuild_atom_children(node: Expression, memo: dict, found: list) -> Expression:
    """``node`` with its arguments canonicalised, or ``node`` itself if none changed."""
    if isinstance(node, FunctionCall):
        new_args = tuple(canonicalize_expression(a, memo=memo, _params=found) for a in node.args)
        if all(a is b for a, b in zip(new_args, node.args)):
            return node
        return FunctionCall(node.func_name, *new_args)
    if isinstance(node, UnaryOp) and node.op != "neg":
        new = canonicalize_expression(node.operand, memo=memo, _params=found)
        return node if new is node.operand else UnaryOp(node.op, new)
    if isinstance(node, BinaryOp):
        left = canonicalize_expression(node.left, memo=memo, _params=found)
        right = canonicalize_expression(node.right, memo=memo, _params=found)
        if left is node.left and right is node.right:
            return node
        return BinaryOp(node.op, left, right)
    return node


def canonicalize_expression(
    expr: Expression,
    *,
    memo: Optional[dict] = None,
    _atom_children_only: bool = False,
    _params: Optional[list] = None,
) -> Expression:
    """Return ``expr``, or its exact canonical form if float collection could round.

    The returned object *is* ``expr`` whenever no rewrite is needed. See the module
    docstring for what is collected and when a rewrite happens. ``_params``, when
    given, receives every Parameter the decision read (for a value-keyed cache).

    Raises :class:`ParameterCancellationError` for a body no symbolic rewrite can
    make exact at the current Parameter values.
    """
    if memo is None:
        memo = {}
    key = (id(expr), _atom_children_only)
    hit = memo.get(key)
    if hit is not None:
        if _params is not None:
            _params.extend(hit[2])
        cached: Expression = hit[1]
        return cached
    found: list = []
    if _atom_children_only:
        out = _rebuild_atom_children(expr, memo, found)
    elif not isinstance(expr, Expression) or _known_shape(expr) != ():
        out = expr
    elif _fast_clean(expr, memo, found):
        out = expr
    else:
        found.clear()
        coll = _Collector(memo, found)
        try:
            poly = coll.collect(expr)
            if coll.dirty or _needs_rewrite(poly, coll.param_values):
                out = _rebuild(poly, coll.factor_nodes)
                _check_rewrite_is_exact(poly, coll.param_values)
            else:
                out = expr
        except _Decline:
            out = expr
    # Keep ``expr`` alive in the memo so its id cannot be reused within this pass.
    memo[key] = (expr, out, tuple(found))
    if _params is not None:
        _params.extend(found)
    return out


def _snapshot(params: list) -> tuple:
    seen: dict = {}
    for p in params:
        seen.setdefault(id(p), p)
    return tuple((p, np.array(p.value, dtype=np.float64, copy=True)) for p in seen.values())


def _snapshot_current(snap) -> bool:
    return snap is not None and all(np.array_equal(p.value, v) for p, v in snap)


def _canonical_cached(holder, attr: str, memo: Optional[dict], what: str):
    """Canonical form of ``getattr(holder, attr)``, cached on ``holder``.

    The cache is the checked expression (by identity) *and* a snapshot of every
    Parameter value the check read: the decision depends on those values, so a
    changed value re-runs it at the next solve.
    """
    expr = getattr(holder, attr)
    if getattr(holder, "_exact_body", None) is expr and _snapshot_current(
        getattr(holder, "_exact_params", None)
    ):
        return expr
    found: list = []
    try:
        new = canonicalize_expression(expr, memo=memo, _params=found)
    except ParameterCancellationError as e:
        raise ParameterCancellationError(f"{what}: {e}") from None
    holder._exact_body = new
    holder._exact_params = _snapshot(found)
    return new


def canonicalize_constraint(constraint, memo: Optional[dict] = None) -> None:
    """Replace ``constraint.body`` in place by its canonical form when needed.

    In place so the caller's object stays the model's row (identity-keyed APIs).
    Cached on the row by body identity and Parameter values (see
    :func:`_canonical_cached`), so a row is walked once per distinct set of values.
    """
    name = getattr(constraint, "name", None)
    new = _canonical_cached(constraint, "body", memo, f"constraint {name!r}")
    if new is not constraint.body:
        constraint.body = new


def canonicalize_model(model) -> None:
    """:func:`canonicalize_rows_in` over every row of ``model``, and its objective."""
    memo: dict = {}
    for item in model._constraints:
        canonicalize_rows_in(item, memo)
    obj = model._objective
    if obj is not None:
        new = _canonical_cached(obj, "expression", memo, "objective")
        if new is not obj.expression:
            obj.expression = new
    # ``if_else`` branch values derive the auxiliary's box at every solve
    # (``Model._refresh_if_else_bounds``, which runs after this). ``_IfElseAux`` has
    # ``__slots__``, so they are not cached; there are a handful per model.
    # ``getattr``, as ``Model._refresh_if_else_bounds`` does: a model unpickled from
    # before the attribute existed has no ``if_else`` auxiliaries at all.
    for aux in getattr(model, "_if_else_auxes", ()):
        for attr in ("then_expr", "else_expr"):
            expr = getattr(aux, attr)
            try:
                new = canonicalize_expression(expr, memo=memo)
            except ParameterCancellationError as e:
                raise ParameterCancellationError(f"if_else {attr}: {e}") from None
            if new is not expr:
                setattr(aux, attr, new)


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
