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

Arrays (#2097). An array expression is collected as a numpy object array of
polynomials, one per element, with numpy broadcasting and C-order flat slots (the
slot every downstream reader uses): elementwise ``+ - * / **``, basic and fancy
indexing of a Variable/Parameter, ``dm.sum`` over all or one axis, and ``@`` of 1-D
and 2-D operands. A scalar ``dm.sum``/``@`` (``dm.sum(t + x - t)``, ``x @ Q @ x``,
``dm.sum(V * x[R] * x[C])``) is expanded into its entries and merged with the row's
visible terms, because that is what the downstream collectors do. An array *body*
is one ``Constraint`` and many rows: each element is judged as its own row, and a
rewrite is a sum of aligned terms ``C * x[a:b:c] * ...``, one per monomial slot
every element shares, using basic indices only (a gather disables the Rust model's
FBBT). When the elements do not line up that way,
:class:`ArrayRowCancellationError` refuses the row rather than certifying its
rounded form.

Declining is sound only in the sense that the expression is then left exactly as
written, which is the behaviour before this module existed. It declines for an
expansion over :data:`_MAX_CONTRIBUTIONS` contributions, a non-finite constant,
or an array node of unknown shape; a reduction that declines stays an opaque atom
(its budget is refunded so the rest of its row is still collected).

Shared sums (#2100). Every walk here follows a DAG, not a tree: the fast passes
stop at the first sum they reach twice (a shared sum repeats its monomials), and
an additive chain whose tree expansion is over the budget but whose DAG is not
(``r = r + r`` repeated: ``d + 1`` nodes, ``2**d`` occurrences) is collected by
exact multiplicity over the DAG and always rewritten, to ``2**d * x``. Downstream
walkers then never see the sharing. When that coefficient has no binary64
(``d >= 1024``), :class:`SharedSumOverflowError` refuses the row.

Atoms are keyed **structurally** (:func:`_structural_id`), so two distinct but
identical subtrees -- two separate ``dm.exp(z)`` objects -- are one monomial. Before
#2101 they were keyed by object identity and summed in floats, which certified wrong
optima. A node is merged only when its fields fully determine its value; tagged nodes,
``CustomCall`` and unusual index forms keep identity keys and are simply never merged.

Residual exposure (not handled; recorded so it is not mistaken for covered): a
reduction over :data:`_MAX_CONTRIBUTIONS` entries, and an index into an expression
(``(A @ x)[i]``) are atoms -- their inner array is collected exactly once per model
pass, but its entries are not merged with the row's visible terms.

A coefficient whose exact value is **zero** is rewritten whenever its sum is not
exact in every order (#2098): there is no relative error to bound it by, and a
residue next to a large coefficient or constant is a coefficient the written row
does not have -- small against the row for checking a point, but enough to fix a
variable the row leaves free and certify a bound that excludes feasible points.
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


class _Overflow(_Decline):
    """A collected coefficient is outside the binary64 range."""


def _size(shape: tuple) -> int:
    return int(np.prod(shape, dtype=np.int64))


def _scalar_array(poly: _Poly) -> np.ndarray:
    out = np.empty((), dtype=object)
    out[()] = poly
    return out


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
        self.arrays: dict = {}  # id(node) -> object array of polys (#2097, DAG reuse)
        self.tree_sizes: dict = {}  # id(node) -> leaf occurrences of its additive tree
        self.shared = False  # a shared sum was collected through its DAG (#2100)
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

    def _additive_children(self, node) -> Optional[list]:
        """``[(child, exact factor)]`` of an additive node, ``None`` for a term.

        The same edges :meth:`_sum_poly` follows: ``+``, ``-``, ``neg``,
        ``SumOverExpression``, and ``*`` / ``/`` by a scalar constant.
        """
        t = type(node)
        if t is BinaryOp:
            op = node.op
            if op == "+":
                return [(node.left, _ONE), (node.right, _ONE)]
            if op == "-":
                return [(node.left, _ONE), (node.right, -_ONE)]
            if op == "*":
                if type(node.left) is Constant and node.left.value.ndim == 0:
                    return [(node.right, self._frac(node.left))]
                if type(node.right) is Constant and node.right.value.ndim == 0:
                    return [(node.left, self._frac(node.right))]
            elif op == "/" and type(node.right) is Constant and node.right.value.ndim == 0:
                d = self._frac(node.right)
                if d == 0:
                    raise _Decline("division by zero")
                return [(node.left, _ONE / d)]
            return None
        if t is UnaryOp and node.op == "neg":
            return [(node.operand, -_ONE)]
        if t is SumOverExpression:
            return [(term, _ONE) for term in node.terms]
        return None

    def _tree_terms(self, root) -> int:
        """Leaf occurrences of ``root``'s additive chain *as a tree* (#2100).

        Linear in the DAG (memoised by node). A shared sum ``r = r + r`` nested
        ``d`` deep is ``d + 1`` nodes and ``2**d`` occurrences -- what the walk in
        :meth:`_sum_poly` would visit.
        """
        memo = self.tree_sizes
        if id(root) in memo:
            size: int = memo[id(root)]
            return size
        stack: list = [(root, None)]
        while stack:
            node, kids = stack.pop()
            if id(node) in memo:
                continue
            if kids is None:
                kids = self._additive_children(node)
                if kids is None:
                    memo[id(node)] = 1
                    continue
                stack.append((node, kids))
                stack.extend((k, None) for k, _ in kids if id(k) not in memo)
                continue
            memo[id(node)] = sum(memo[id(k)] for k, _ in kids)
        size = memo[id(root)]
        return size

    def _dag_leaf_edges(self, root) -> int:
        """Term references in ``root``'s additive chain counted once per distinct
        sum node -- what :meth:`_sum_poly_dag` reads; linear in the DAG."""
        seen: set = set()
        edges = 0
        stack = [root]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            kids = self._additive_children(node)
            if kids is None:
                continue
            for k, _ in kids:
                if self._additive_children(k) is None:
                    edges += 1
                elif id(k) not in seen:
                    stack.append(k)
        return edges

    def _sum_poly_dag(self, root) -> _Poly:
        """Exact polynomial of a *shared* additive DAG, linear in its size (#2100).

        Each node's coefficient is the exact sum, over every path from ``root``, of
        the product of the factors on that path (parents before children, in
        reverse post-order), so ``r = r + r`` ``d`` deep gives ``x`` the single
        coefficient ``2**d``. These aggregated contributions are not the per-path
        sums a downstream collector would form, so they cannot decide whether the
        row needs a rewrite: the row is always rewritten (``dirty``), which is
        sound and leaves every downstream walk a row with no sharing to expand.
        """
        self.dirty = True
        self.shared = True
        order: list = []
        seen: set = set()
        kids_of: dict = {}
        stack: list = [(root, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            kids = self._additive_children(node)
            if kids is None:
                continue
            kids_of[id(node)] = kids
            stack.append((node, True))
            stack.extend((k, False) for k, _ in reversed(kids) if id(k) not in seen)
        coef: dict = {id(root): _ONE}
        leaves: dict = {}  # id(leaf) -> [leaf, exact coefficient]
        for node in reversed(order):  # reverse post-order: every parent first
            c = coef.pop(id(node), None)
            if c is None:
                continue
            for k, f in kids_of[id(node)]:
                if id(k) in kids_of:
                    coef[id(k)] = coef.get(id(k), Fraction(0)) + c * f
                else:
                    hit = leaves.get(id(k))
                    if hit is None:
                        leaves[id(k)] = [k, c * f]
                    else:
                        hit[1] += c * f
        if not kids_of:  # ``root`` is itself a term
            leaves[id(root)] = [root, _ONE]
        acc: _Poly = {}
        for leaf, c in leaves.values():
            if c == 0:
                continue
            if type(leaf) is Constant:
                if leaf.value.ndim != 0:
                    raise _Decline("array constant in scalar position")
                self._count(1)
                acc.setdefault((), []).append(c * self._frac(leaf))
                continue
            term = self.terms.get(id(leaf))
            if term is None:
                term = self._term(leaf)
                self.terms[id(leaf)] = term
            for k, v in term.items():
                self._count(len(v))
                acc.setdefault(k, []).extend(q * c for q in v)
        return acc

    def _sum_poly(self, root) -> _Poly:
        """Exact polynomial of ``root``: flatten its additive chain, merge once."""
        if (
            self._tree_terms(root) > _MAX_CONTRIBUTIONS
            and self._dag_leaf_edges(root) <= _MAX_CONTRIBUTIONS
        ):
            # Over budget as a tree, within it through its sharing (#2100). A chain
            # that is long without shared sums still declines, on the budget,
            # exactly as before.
            return self._sum_poly_dag(root)
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
            if var is not None:
                mat = node.left.value if type(node.left) is Constant else node.right.value
                return self._linear_array(var, np.asarray(mat, dtype=np.float64))
            return self._reduction(node)
        if t is SumExpression:
            aff = affine_array_sum(node)
            if aff is not None:
                return self._linear_array(*aff)
            return self._reduction(node)
        return self._atom(node)

    def _reduction(self, node) -> _Poly:
        """A scalar ``dm.sum`` / ``@`` of an array, expanded per flat slot (#2097).

        Downstream collectors expand ``dm.sum(x**2)``, ``x @ Q @ x`` and
        ``dm.sum(V * x[R] * x[C])`` into their entries and merge them with the
        row's other terms, so they are expanded here the same way. Falls back to
        an opaque atom -- the pre-#2097 behaviour -- when the operand is outside the
        array skeleton, the expansion is over budget, or the result holds an
        element of an opaque array (rebuilding it would need an index into an
        opaque node). The budget spent on a failed attempt is refunded, so a large
        ``x @ Q @ x`` does not stop the rest of its row from being collected.
        """
        saved = self.n_contribs
        try:
            if type(node) is SumExpression:
                if node.axis is not None:
                    raise _Decline("axis reduction in scalar position")
                arr = _scalar_array(self._add_all(self._array(node.operand).reshape(-1).tolist()))
            else:
                arr = self._matmul(self._array(node.left), self._array(node.right))
        except _Decline:
            self.n_contribs = saved
            return self._atom(node)
        if arr.shape != ():
            self.n_contribs = saved
            return self._atom(node)
        poly = arr[()]
        if any(f[0] == "e" for key in poly for f in key):
            self.n_contribs = saved
            return self._atom(node)
        return dict(poly)

    # -- the array walk (#2097) --------------------------------------------------
    #
    # An array expression is collected as a numpy object array of polynomials, one
    # per element, with numpy's broadcasting and C-order flat slots: element ``j``
    # of a Variable ``x`` is the factor ``("v", id(x), j)``, exactly the slot the
    # scalar path gives ``x[np.unravel_index(j, x.shape)]`` and the one every
    # downstream reader uses (the #941 rule: a wrong slot is a false certificate).
    # Polynomials are never mutated once built, so elements may be shared.

    def _array(self, node) -> np.ndarray:
        hit = self.arrays.get(id(node))
        if hit is None:
            hit = self._array_uncached(node)
            known = _known_shape(node)
            if known is not None and tuple(known) != hit.shape:
                raise _Decline("array walk disagrees with the static shape")
            self.arrays[id(node)] = hit
        return hit

    def _array_uncached(self, node) -> np.ndarray:
        t = type(node)
        if t is Constant:
            vals = np.asarray(node.value, dtype=np.float64)
            if not np.all(np.isfinite(vals)):
                raise _Decline("non-finite constant")
            self._count(int(vals.size))
            out = np.empty(vals.shape, dtype=object)
            for j, v in enumerate(vals.reshape(-1).tolist()):
                # A zero entry keeps its (zero) contribution, so ``C * y`` with a
                # zero in ``C`` still holds ``y``'s monomial in that element and the
                # elements stay aligned for :func:`_rebuild_array`.
                out.reshape(-1)[j] = {(): [Fraction(v)]}
            return out
        if t is Variable or t is Parameter:
            if tuple(node.shape or ()) == ():
                return _scalar_array(self._term(node))
            shape = tuple(node.shape)
            return self._leaf_entries(node, np.arange(_size(shape)).reshape(shape))
        if t is IndexExpression:
            base = node.base
            if type(base) is Variable or type(base) is Parameter:
                shape = tuple(base.shape or ())
                if shape == ():
                    raise _Decline("index into a scalar leaf")
                try:
                    flat = np.arange(_size(shape)).reshape(shape)[node.index]
                except (IndexError, TypeError, ValueError):
                    raise _Decline("index numpy cannot resolve") from None
                return self._leaf_entries(base, np.asarray(flat))
            base_arr = self._array(base)
            try:
                sub = base_arr[node.index]
            except (IndexError, TypeError, ValueError):
                raise _Decline("index numpy cannot resolve") from None
            return sub if isinstance(sub, np.ndarray) else _scalar_array(sub)
        if t is BinaryOp:
            op = node.op
            if op == "+" or op == "-" or op == "*":
                a = self._array(node.left)
                b = self._array(node.right)
                return self._elementwise(op, a, b)
            if op == "/" and type(node.right) is Constant:
                d = np.asarray(node.right.value, dtype=np.float64)
                if not np.all(np.isfinite(d)) or np.any(d == 0.0):
                    raise _Decline("division by zero or a non-finite constant")
                inv = np.empty(d.shape, dtype=object)
                for j, v in enumerate(d.reshape(-1).tolist()):
                    inv.reshape(-1)[j] = {(): [_ONE / Fraction(v)]}
                return self._elementwise("*", self._array(node.left), inv)
            if op == "**":
                k = self._small_power(node.right)
                if k is not None:
                    base_arr = self._array(node.left)
                    out = np.empty(base_arr.shape, dtype=object)
                    for j, p in enumerate(base_arr.reshape(-1).tolist()):
                        acc: _Poly = {(): [_ONE]}
                        for _ in range(k):
                            acc = self._mul(acc, p)
                        out.reshape(-1)[j] = acc
                    return out
        elif t is UnaryOp and node.op == "neg":
            a = self._array(node.operand)
            out = np.empty(a.shape, dtype=object)
            for j, p in enumerate(a.reshape(-1).tolist()):
                out.reshape(-1)[j] = self._scaled(p, -_ONE)
            return out
        elif t is SumOverExpression:
            acc_arr: Optional[np.ndarray] = None
            for term in node.terms:
                a = self._array(term)
                acc_arr = a if acc_arr is None else self._elementwise("+", acc_arr, a)
            if acc_arr is None:
                return _scalar_array({})
            return acc_arr
        elif t is SumExpression:
            if node.axis is None:
                return _scalar_array(self._term(node))
            a = self._array(node.operand)
            try:
                moved = np.moveaxis(a, node.axis, 0)
            except (np.exceptions.AxisError, TypeError, ValueError):
                raise _Decline("reduction axis numpy cannot resolve") from None
            out = np.empty(moved.shape[1:], dtype=object)
            flat_out = out.reshape(-1)
            cols = moved.reshape(moved.shape[0], -1)
            for j in range(flat_out.size):
                flat_out[j] = self._add_all(cols[:, j].tolist())
            return out
        elif t is MatMulExpression:
            if _array_linear_var(node) is not None:
                return _scalar_array(self._term(node))
            arr = self._matmul(self._array(node.left), self._array(node.right))
            if arr.shape == () and any(f[0] == "e" for key in arr[()] for f in key):
                return _scalar_array(self._atom(node))
            return arr
        static = _known_shape(node)
        if static == ():
            return _scalar_array(self._sum_poly(node))
        if static is None:
            raise _Decline("array node of unknown shape")
        return self._array_atom(node, tuple(static))

    def _leaf_entries(self, base, flat: np.ndarray) -> np.ndarray:
        """Polynomials of the entries ``flat`` (C-order slots) of a shaped leaf."""
        self._count(int(flat.size))
        out = np.empty(flat.shape, dtype=object)
        dst = out.reshape(-1)
        is_var = type(base) is Variable
        for k, j in enumerate(flat.reshape(-1).tolist()):
            if is_var:
                fkey = ("v", id(base), j)
                self.factor_nodes.setdefault(fkey, (base, j))
                dst[k] = {(fkey,): [_ONE]}
            else:
                dst[k] = self._param(base, (base, j), j)
        return out

    def _array_atom(self, node, shape: tuple) -> np.ndarray:
        """An opaque array node: element ``j`` is the factor ``("e", sid, j)``, with
        the node keyed structurally (#2101) like a scalar atom."""
        new = canonicalize_expression(
            node, memo=self.memo, _atom_children_only=True, _params=self.params
        )
        if new is not node:
            self.dirty = True
        size = int(np.prod(shape, dtype=np.int64))
        self._count(size)
        out = np.empty(shape, dtype=object)
        dst = out.reshape(-1)
        for j in range(size):
            fkey = ("e", _structural_id(new, self.memo), j)
            self.factor_nodes.setdefault(fkey, (new, j))
            dst[j] = {(fkey,): [_ONE]}
        return out

    def _scaled(self, p: _Poly, s: Fraction) -> _Poly:
        return {k: [q * s for q in v] for k, v in p.items()}

    def _add_all(self, polys: list) -> _Poly:
        out: _Poly = {}
        for p in polys:
            for k, v in p.items():
                self._count(len(v))
                lst = out.get(k)
                if lst is None:
                    out[k] = list(v)
                else:
                    lst.extend(v)
        return out

    def _elementwise(self, op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        try:
            a2, b2 = np.broadcast_arrays(a, b)
        except ValueError:
            raise _Decline("operands do not broadcast") from None
        out = np.empty(a2.shape, dtype=object)
        dst = out.reshape(-1)
        for j, (p, q) in enumerate(zip(a2.reshape(-1).tolist(), b2.reshape(-1).tolist())):
            if op == "*":
                dst[j] = self._mul(p, q)
            elif op == "+":
                dst[j] = self._add_all([p, q])
            else:
                dst[j] = self._add_all([p, self._scaled(q, -_ONE)])
        return out

    def _matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """numpy ``@`` for 1-D and 2-D operands, expanded exactly."""
        if a.ndim == 0 or b.ndim == 0 or a.ndim > 2 or b.ndim > 2:
            raise _Decline("matmul outside 1-D / 2-D")
        a2 = a if a.ndim == 2 else a.reshape(1, -1)
        b2 = b if b.ndim == 2 else b.reshape(-1, 1)
        if a2.shape[1] != b2.shape[0]:
            raise _Decline("matmul inner dimensions differ")
        out = np.empty((a2.shape[0], b2.shape[1]), dtype=object)
        for i in range(a2.shape[0]):
            for k in range(b2.shape[1]):
                out[i, k] = self._add_all(
                    [self._mul(a2[i, j], b2[j, k]) for j in range(a2.shape[1])]
                )
        if a.ndim == 1 and b.ndim == 1:
            return _scalar_array(out[0, 0])
        if a.ndim == 1:
            return out.reshape(-1)
        if b.ndim == 1:
            return out.reshape(-1)
        return out

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
        return self._factor(("a", _structural_id(new, self.memo)), new)

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


#: The instance attributes each node class carries when it is exactly what its
#: fields say. A node with any *other* attribute -- ``register_function``'s
#: ``_discopt_atom`` tag on an ordinary ``BinaryOp``, or anything added later -- is
#: keyed by identity instead, so it is never merged with an untagged twin.
_STRUCTURAL_ATTRS: dict = {
    Constant: frozenset({"value"}),
    BinaryOp: frozenset({"_shape", "left", "op", "right"}),
    UnaryOp: frozenset({"_shape", "op", "operand"}),
    FunctionCall: frozenset({"_shape", "args", "func_name"}),
    IndexExpression: frozenset({"_shape", "base", "index"}),
    SumOverExpression: frozenset({"terms"}),
    SumExpression: frozenset({"axis", "operand"}),
    MatMulExpression: frozenset({"left", "right"}),
}

_STRUCTURAL_TABLE = "__structural_ids__"


def _plain_index(index) -> Optional[tuple]:
    """``index`` as a tuple of Python ints, or ``None`` for any other index form."""
    idx = index if isinstance(index, tuple) else (index,)
    out = []
    for i in idx:
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)):
            return None
        out.append(int(i))
    return tuple(out)


def _structural_parts(node) -> Optional[tuple]:
    """``(head, children)`` of a node whose value its fields fully determine.

    ``None`` means "key by identity": a Variable or Parameter (a distinct object
    *is* a distinct quantity), a ``CustomCall`` (an arbitrary callable), a node
    carrying any attribute beyond :data:`_STRUCTURAL_ATTRS`, an unusual index,
    and every node type not listed.
    """
    t = type(node)
    allowed = _STRUCTURAL_ATTRS.get(t)
    # ``keys() <=`` compares the instance dict's key view without building a set.
    if allowed is None or not node.__dict__.keys() <= allowed:
        return None
    if t is Constant:
        v = node.value
        return ("c", v.shape, v.dtype.str, v.tobytes()), ()
    if t is BinaryOp:
        return ("b", node.op), (node.left, node.right)
    if t is UnaryOp:
        return ("u", node.op), (node.operand,)
    if t is FunctionCall:
        return ("f", node.func_name, len(node.args)), tuple(node.args)
    if t is IndexExpression:
        idx = _plain_index(node.index)
        if idx is None:
            return None
        return ("i", idx), (node.base,)
    if t is SumOverExpression:
        return ("so", len(node.terms)), tuple(node.terms)
    if t is SumExpression:
        return ("s", node.axis), (node.operand,)
    return ("mm",), (node.left, node.right)  # MatMulExpression


def _structural_id(root, memo: dict) -> int:
    """An integer equal for two nodes iff they are the same expression tree (#2101).

    Hash-consing: each node's key is its head plus its children's ids, so keys stay
    flat however deep the tree, and the walk is iterative. Two *distinct* but
    identical ``dm.exp(z)`` objects get one id, so their like terms are collected
    instead of being summed in floats with a catastrophic cancellation. Equal ids
    imply equal values: only pure nodes whose fields say everything get a
    structural key (:func:`_structural_parts`); the rest are keyed by identity,
    which can only fail to merge, never merge wrongly. Constants compare by their
    exact bytes, so ``0.0`` and ``-0.0`` stay apart.
    """
    table = memo.get(_STRUCTURAL_TABLE)
    if table is None:
        fresh: tuple[dict[int, tuple[object, int]], dict[tuple, int]] = ({}, {})
        table = memo[_STRUCTURAL_TABLE] = fresh
    by_node: dict[int, tuple[object, int]] = table[0]  # id(node) -> (node, sid)
    by_key: dict[tuple, int] = table[1]  # key -> sid
    hit = by_node.get(id(root))
    if hit is not None:
        sid: int = hit[1]
        return sid
    stack: list = [(root, False)]
    pop, push = stack.pop, stack.append
    while stack:
        node, expanded = pop()
        nid = id(node)
        if nid in by_node:
            continue
        t = type(node)
        if t is Variable or t is Parameter:  # a distinct object is a distinct quantity
            key0 = ("id", nid)
            sid0 = by_key.get(key0)
            if sid0 is None:
                sid0 = by_key[key0] = len(by_key)
            by_node[nid] = (node, sid0)
            continue
        parts = _structural_parts(node)
        if parts is not None and parts[1] and not expanded:
            push((node, True))
            stack.extend([(c, False) for c in parts[1] if id(c) not in by_node])
            continue
        if parts is None:
            key: tuple = ("id", nid)
        else:
            head, children = parts
            key = (head, *(by_node[id(c)][1] for c in children))
        got = by_key.get(key)
        if got is None:
            got = by_key[key] = len(by_key)
        # The node is kept alive in the table, so its id cannot be reused this pass.
        by_node[nid] = (node, got)
    out: int = by_node[id(root)][1]
    return out


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
    sums: set = set()  # additive nodes already walked (#2100)
    atoms: list = []
    try:
        stack: list = [(root, False)]  # (node, under a scale)
        pop, push = stack.pop, stack.append
        while stack:
            node, scaled = pop()
            t = type(node)
            if t is BinaryOp:
                op = node.op
                if op == "+" or op == "-":
                    # A sum reached twice repeats every monomial under it, so it
                    # is not clean -- and stopping here keeps this walk linear in
                    # the DAG instead of exponential in its sharing depth (#2100).
                    if id(node) in sums:
                        raise _NotClean
                    sums.add(id(node))
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
                if id(node) in sums:
                    raise _NotClean
                sums.add(id(node))
                stack.extend([(term, scaled) for term in node.terms])
                continue
            elif t is Constant:
                if scaled or node.value.ndim != 0:
                    raise _NotClean
                consts.append(float(node.value))
                continue
            elif t is MatMulExpression or t is SumExpression:
                var = _array_linear_var(node)
                if var is not None:
                    bases = {id(var)}
                elif t is SumExpression and _opaque_head(node.operand):
                    key: tuple = (_fast_atom_key(node, memo, found, atoms),)
                    if key in seen:
                        raise _NotClean
                    seen.add(key)
                    continue
                else:
                    # #2097: downstream expands the reduction into its entries and
                    # merges them with the row's other terms. A product block (one
                    # monomial per element, no two elements alike) is clean when it
                    # shares no variable with anything else; the rest go exact.
                    bases = _fast_product_block(node, memo, found)
                if not block_vars.isdisjoint(bases):
                    raise _NotClean
                block_vars.update(bases)
                continue
            # A term: a product of leaves (Parameter factors dropped, as scales).
            if t is IndexExpression or t is Variable:
                leaf = _fast_leaf_key(node, memo, scalar_vars, found, atoms)
                mono: tuple = () if leaf is None else (leaf,)
            else:
                mono = tuple(sorted(_fast_monomial(node, memo, scalar_vars, found, atoms)))
            if mono in seen:
                raise _NotClean
            seen.add(mono)
        if block_vars and not scalar_vars.isdisjoint(block_vars):
            raise _NotClean
        if consts and () in seen:  # a Parameter-only term next to a constant term
            raise _NotClean
    except _NotClean:
        return False
    if not _fast_atoms_distinct(atoms, memo):
        return False
    return len(consts) <= 1 or _floats_sum_order_exact(consts)


def _opaque_head(node) -> bool:
    """True when ``node`` is not part of the polynomial skeleton at its top."""
    t = type(node)
    if t is BinaryOp:
        return node.op not in ("+", "-", "*", "/", "**")
    if t is UnaryOp:
        return bool(node.op != "neg")
    return t is FunctionCall


def _fast_product_block(node, memo: dict, found: list) -> set:
    """Variable ids of ``dm.sum(P)`` when ``P`` is one elementwise product of whole
    leaves; raises :class:`_NotClean` otherwise.

    Every Variable factor must have exactly ``P``'s shape (no broadcasting), so
    element ``j`` is one monomial over slot ``j`` of each Variable and no two
    elements share a monomial. Constants and Parameters are scales.
    """
    if node.axis is not None:
        raise _NotClean
    op = node.operand
    shape = _known_shape(op)
    if not shape:
        raise _NotClean
    bases: set = set()
    stack = [op]
    while stack:
        n = stack.pop()
        t = type(n)
        if t is BinaryOp:
            if n.op == "*":
                stack.append(n.left)
                stack.append(n.right)
                continue
            if n.op == "/" and type(n.right) is Constant:
                stack.append(n.left)
                continue
            if n.op == "**" and _Collector._small_power(n.right) is not None:
                stack.append(n.left)
                continue
            raise _NotClean
        if t is Constant:
            continue
        if t is Parameter:
            found.append(n)
            continue
        if t is Variable and tuple(n.shape) == shape:
            bases.add(id(n))
            continue
        raise _NotClean
    if not bases:
        raise _NotClean
    return bases


def _fast_clean_array(root: Expression, memo: dict, found: list) -> bool:
    """:func:`_fast_clean` for an array body or one of unknown static shape (#2097).

    Each element is its own row, so two terms are like terms only where they hold
    the same monomial *in the same element*. A term over one Variable leaf (``c *
    x[1:]``, ``x**2``) is checked per element: the flat slots it reads are compared,
    with numpy broadcasting, against those of every other such term over the same
    Variable and power -- so ``x[1:] - x[:-1]`` is clean and ``M*x + x - M*x`` is
    not. Everything coarser (a product of two Variables, an atom, a reduction, a
    matmul) is keyed by the leaves it reads, and any overlap with another term's
    leaves sends the body to the exact pass. ``False`` is never wrong.
    """
    single: dict = {}  # (var id, power) -> list of flat-slot arrays
    multi: set = set()  # leaf keys of products of several leaves, and of atoms
    coarse: set = set()  # Variable keys a reduction / matmul expands
    sums: set = set()  # additive nodes already walked (#2100, as in _fast_clean)
    one_var_products: set = set()  # Variables with a product-of-leaves term over them alone
    atoms: list = []  # every atom, for the #2101 structural comparison
    n_const = 0
    try:
        stack: list = [(root, False)]
        pop, push = stack.pop, stack.append
        while stack:
            node, scaled = pop()
            t = type(node)
            if t is BinaryOp:
                op = node.op
                if op == "+" or op == "-":
                    if id(node) in sums:
                        raise _NotClean
                    sums.add(id(node))
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
            elif t is UnaryOp and node.op == "neg":
                push((node.operand, scaled))
                continue
            elif t is SumOverExpression:
                if id(node) in sums:
                    raise _NotClean
                sums.add(id(node))
                stack.extend([(term, scaled) for term in node.terms])
                continue
            elif t is Constant:
                n_const += 1
                if scaled or n_const > 1:
                    raise _NotClean
                continue
            elif t is MatMulExpression or t is SumExpression:
                if t is MatMulExpression:
                    var = _matmul_linear_var(node)
                    if var is None:
                        raise _NotClean
                    keys: set = {("V", id(var))}
                elif _opaque_head(node.operand):
                    keys = {_fast_atom_key(node, memo, found, atoms)}
                else:
                    keys = {("V", b) for b in _fast_product_block_any_axis(node, memo, found)}
                if not coarse.isdisjoint(keys) or not multi.isdisjoint(keys):
                    raise _NotClean
                if any(k[0] == "V" and (k[1], p) in single for k in keys for p in _POWERS):
                    raise _NotClean
                coarse |= keys
                continue
            keys, one = _fast_array_term_keys(node, memo, found, atoms)
            if not keys:
                n_const += 1  # a Parameter-only term: a constant once substituted
                if n_const > 1:
                    raise _NotClean
                continue
            if not coarse.isdisjoint(keys):
                raise _NotClean
            if one is None:
                if not multi.isdisjoint(keys):
                    raise _NotClean
                multi |= keys
                # A product of leaves of ONE Variable (``x*x``, ``x[1:]*x[:-1]``)
                # can hold a single-leaf power's monomial (``x**2``); a product
                # over two Variables or an atom never can.
                if len(keys) == 1:
                    (only,) = keys
                    if only[0] == "V":
                        if any((only[1], p) in single for p in _POWERS):
                            raise _NotClean
                        one_var_products.add(only[1])
                continue
            skey, flat = one
            if skey[0] in one_var_products:
                raise _NotClean
            for other in single.get(skey, ()):
                try:
                    a, b = np.broadcast_arrays(flat, other)
                except ValueError:
                    raise _NotClean from None
                if np.any(a == b):
                    raise _NotClean
            single.setdefault(skey, []).append(flat)
    except _NotClean:
        return False
    # Atoms are keyed by identity above; two separate but identical ones (#2101)
    # would read as two monomials, so they are compared structurally here.
    return _fast_atoms_distinct(atoms, memo)


#: Powers a single-leaf term may carry (``x``, ``x**2``, ... ``x**_MAX_POWER``).
_POWERS = tuple(range(1, _MAX_POWER + 1))


def _fast_product_block_any_axis(node, memo: dict, found: list) -> set:
    """:func:`_fast_product_block` for a reduction over any axis: each entry of a
    full-shape Variable lands in exactly one element, once."""
    if node.axis is None:
        return _fast_product_block(node, memo, found)
    full = SumExpression(node.operand)
    return _fast_product_block(full, memo, found)


def _matmul_linear_var(node) -> Optional[Variable]:
    """The Variable of ``C @ x`` / ``x @ C`` (constant ``C`` of any rank), else None."""
    left, right = node.left, node.right
    if type(left) is Constant and type(right) is Variable:
        return right
    if type(right) is Constant and type(left) is Variable:
        return left
    return None


def _fast_array_term_keys(node, memo: dict, found: list, atoms: list) -> tuple:
    """``(leaf keys, single)`` of one elementwise product term, or raise
    :class:`_NotClean`. ``single`` is ``((var id, power), flat slots)`` when the term
    reads exactly one Variable leaf once (possibly raised to a power), else None."""
    keys: set = set()
    leaves: list = []  # (Variable, index or None, power)
    has_atom = False
    stack = [(node, 1)]
    while stack:
        n, power = stack.pop()
        t = type(n)
        if t is BinaryOp:
            op = n.op
            if op == "*":
                stack.append((n.left, power))
                stack.append((n.right, power))
                continue
            if op == "/" and type(n.right) is Constant:
                stack.append((n.left, power))
                continue
            if op == "**":
                k = _Collector._small_power(n.right)
                if k is not None:
                    tb = type(n.left)
                    if tb is not Variable and tb is not IndexExpression and tb is not Parameter:
                        raise _NotClean
                    stack.append((n.left, power * k))
                    continue
            if op == "+" or op == "-":
                raise _NotClean
        elif t is Constant:
            continue
        elif t is Parameter:
            found.append(n)
            continue
        elif t is Variable:
            keys.add(("V", id(n)))
            leaves.append((n, None, power))
            continue
        elif t is IndexExpression:
            tb = type(n.base)
            if tb is Variable:
                keys.add(("V", id(n.base)))
                leaves.append((n.base, n.index, power))
                continue
            if tb is Parameter:
                found.append(n.base)
                continue
            raise _NotClean  # an index into an expression: the exact pass reads it
        elif t is UnaryOp and n.op == "neg":
            raise _NotClean
        elif t is SumOverExpression or t is MatMulExpression or t is SumExpression:
            raise _NotClean
        keys.add(_fast_atom_key(n, memo, found, atoms))
        has_atom = True
    if has_atom or len(leaves) != 1 or leaves[0][2] == 0:
        return keys, None
    var, index, power = leaves[0]
    shape = tuple(var.shape or ())
    flat = np.arange(_size(shape)).reshape(shape)
    if index is not None:
        try:
            flat = np.asarray(flat[index])
        except (IndexError, TypeError, ValueError):
            raise _NotClean from None
    return keys, ((id(var), power), flat)


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


def _fast_atom_key(node, memo: dict, found: list, atoms: list) -> tuple:
    """Identity key of an atom whose arguments need no rewrite; the atom is recorded
    in ``atoms`` so :func:`_fast_atoms_distinct` can compare it structurally."""
    new = canonicalize_expression(node, memo=memo, _atom_children_only=True, _params=found)
    if new is not node:
        raise _NotClean
    atoms.append(node)
    return ("a", id(node))


def _atom_head(node) -> tuple:
    """The top of an atom: two structurally equal atoms always have equal heads."""
    return (type(node), getattr(node, "op", None), getattr(node, "func_name", None))


def _fast_atoms_distinct(atoms: list, memo: dict) -> bool:
    """False if two of a row's atoms may be the same expression (#2101).

    The fast pass keys atoms by identity, so two separate but identical
    ``dm.exp(z)`` objects would look like two monomials. Structural ids
    (:func:`_structural_id`) decide, but they cost a walk of each atom's subtree, so
    they are computed only among atoms that share a head -- equal trees always do.
    A row with one atom, or with atoms of different functions/ops, pays nothing.
    Any structural match (even across different monomials) sends the row to the
    exact pass, which keys every atom structurally: conservative, never wrong.
    """
    if len(atoms) < 2:
        return True
    groups: dict = {}
    for a in atoms:
        groups.setdefault(_atom_head(a), []).append(a)
    for group in groups.values():
        if len(group) < 2:
            continue
        seen: set = set()
        for a in group:
            sid = _structural_id(a, memo)
            if sid in seen:
                return False
            seen.add(sid)
    return True


def _fast_monomial(node, memo: dict, scalar_vars: set, found: list, atoms: list) -> list:
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
                    leaf = _fast_leaf_key(base, memo, scalar_vars, found, atoms)
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
        leaf = _fast_leaf_key(n, memo, scalar_vars, found, atoms)
        if leaf is not None:
            factors.append(leaf)
    return factors


def _fast_leaf_key(n, memo: dict, scalar_vars: set, found: list, atoms: list) -> Optional[tuple]:
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
    return _fast_atom_key(n, memo, found, atoms)


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

    A sum whose exact value is **zero** but which is *not* exact in every order is
    never benign (#2098): its ``kappa`` is infinite, and any float residue is a
    coefficient on a variable the written row does not hold. An earlier version
    measured that residue against the row's largest coefficient and kept it when
    it was below ``2**-26`` of that scale. That is safe for checking a point and
    wrong for a bound: in ``((t + x) - t) - x + y == 1e13`` the residue ``-1`` on
    ``x`` read the row as ``-x + y == 1e13``, which with ``y <= 1e13`` forces
    ``x == 0``, and the certified bound excluded every feasible ``x > 0``.
    Rewriting a zero coefficient only *drops* its terms, so it is always sound.
    """
    if _sum_is_order_exact(contribs):
        return True
    total = sum(contribs, Fraction(0))
    if total == 0:
        return False
    n = sum(1 for q in contribs if q != 0)
    abs_sum = sum((abs(q) for q in contribs), Fraction(0))
    return bool(abs_sum * (n - 1) <= _CANCELLATION_LIMIT * abs(total))


def _groups_benign(groups: list) -> list:
    """The groups of one body that are *not* benign."""
    return [g for g in groups if not _sum_is_benign(g)]


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


def _entry_index(shape: tuple, j: int):
    """The index of C-order flat slot ``j`` in ``shape``, as the scalar path writes it."""
    if len(shape) == 1:
        return int(j)
    return tuple(int(i) for i in np.unravel_index(j, shape))


def _resolve(factor_nodes: dict, fkey) -> Expression:
    """The node of a factor; array entries are stored lazily as ``(base, flat)``."""
    node = factor_nodes[fkey]
    if type(node) is tuple:
        base, j = node
        node = base[_entry_index(tuple(_known_shape(base) or base.shape), j)]
        factor_nodes[fkey] = node
    out: Expression = node
    return out


def _rebuild(poly: _Poly, factor_nodes: dict) -> Expression:
    terms: list = []
    const_term = None
    for key, contribs in poly.items():
        exact = sum(contribs, Fraction(0))
        try:
            c = float(exact)  # correctly rounded
        except OverflowError:
            raise _Overflow("coefficient outside the binary64 range") from None
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
            node = _resolve(factor_nodes, key[i])
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


class SharedSumOverflowError(ValueError):
    """A shared sum's collected coefficient is outside the binary64 range (#2100).

    ``r = r + r`` nested ``d`` deep is ``2**d * x``; for ``d >= 1024`` no float holds
    that coefficient. The row cannot be rewritten, and left as written its DAG is
    expanded as a tree, in time ``2**d``, by every consumer that walks it -- the
    solve would never return, whatever its time limit. So it is refused.
    """


class ArrayRowCancellationError(ValueError):
    """An array-valued row needs an exact rewrite that no slice-only form can hold.

    An array body (``x <= 1`` on a vector is one ``Constraint`` and many rows) whose
    like terms cancel catastrophically is rewritten as a sum of *aligned* terms,
    ``C * x[a:b:c] * y[...]``, one per monomial slot shared by every element. When
    the elements' monomials do not line up that way -- different variables per
    element, or an entry pattern no basic slice selects -- the exact form would need
    a gather (``x[np.array([...])]``), which the Rust model cannot represent (it
    then disables FBBT/presolve for the whole model). Leaving the row as written
    certifies the rounded row (#2097), so this refuses instead (CLAUDE.md §3).
    """


def _basic_index(flat: np.ndarray, base_shape: tuple):
    """A basic (int/slice) index of ``base_shape`` selecting exactly ``flat``, or None.

    ``flat`` holds C-order slots in the result's shape. Returns ``Ellipsis`` for the
    whole leaf in its own shape.
    """
    size = int(np.prod(base_shape, dtype=np.int64))
    if flat.shape == tuple(base_shape) and np.array_equal(flat.reshape(-1), np.arange(size)):
        return Ellipsis
    grid = np.unravel_index(flat.reshape(-1), base_shape)
    idx: list = []
    for g in grid:
        u = np.unique(g)
        if u.size == 1:
            idx.append(int(u[0]))
            continue
        step = int(u[1] - u[0])
        if not np.array_equal(u, np.arange(int(u[0]), int(u[-1]) + 1, step)):
            return None
        idx.append(slice(int(u[0]), int(u[-1]) + 1, step))
    index = idx[0] if len(idx) == 1 else tuple(idx)
    cand = np.arange(size).reshape(base_shape)[index]
    if np.shape(cand) != flat.shape or not np.array_equal(cand, flat):
        return None
    return index


def _factor_array(fkeys: list, shape: tuple, factor_nodes: dict) -> Expression:
    """One factor position of an aligned slot: a node whose element ``j`` is
    ``fkeys[j]`` (a single node when every element shares the factor)."""
    first = fkeys[0]
    if all(k == first for k in fkeys):
        return _resolve(factor_nodes, first)
    kind = first[0]
    if kind in ("v", "p", "e") and all(k[0] == kind and k[1] == first[1] for k in fkeys):
        lazy = factor_nodes[first]
        base = lazy[0] if type(lazy) is tuple else None
        if base is None:  # an entry already resolved; recover its base
            n = lazy
            base = n.base if type(n) is IndexExpression else None
        if base is not None:
            flat = np.asarray([k[2] for k in fkeys], dtype=np.int64).reshape(shape)
            base_shape = tuple(_known_shape(base) or ())
            index = _basic_index(flat, base_shape) if base_shape else None
            if index is Ellipsis:
                return base
            if index is not None and kind != "e":
                out: Expression = base[index]
                if _known_shape(out) == tuple(shape):
                    return out
    raise ArrayRowCancellationError(
        "an array-valued row's like terms cancel catastrophically in binary64, and "
        "its exact form does not line up element by element into slices (#2097). "
        "Write the row with its like terms already collected, or as scalar rows."
    )


def _rebuild_array(arr: np.ndarray, factor_nodes: dict) -> Expression:
    """Exact canonical form of an array body: aligned monomial slots (see
    :class:`ArrayRowCancellationError`) plus one constant array."""
    shape = arr.shape
    n = int(arr.size)
    const = np.zeros(n)
    elems: list = []
    for j, poly in enumerate(arr.reshape(-1).tolist()):
        items = []
        for key, contribs in poly.items():
            exact = sum(contribs, Fraction(0))
            try:
                c = float(exact)
            except OverflowError:
                raise _Overflow("coefficient outside the binary64 range") from None
            if key == ():
                const[j] = c
                continue
            sig = tuple(f[:2] for f in key)
            items.append((sig, key, c))
        items.sort(key=lambda it: (it[0], it[1]))
        elems.append(items)
    sigs = [tuple(it[0] for it in e) for e in elems]
    if any(sg != sigs[0] for sg in sigs):
        raise ArrayRowCancellationError(
            "an array-valued row's like terms cancel catastrophically in binary64, and "
            "its elements do not share one monomial structure, so no slice-aligned "
            "exact form exists (#2097). Write the row with its like terms already "
            "collected, or as scalar rows."
        )
    terms: list = []
    for slot in range(len(sigs[0])):
        coefs = np.array([elems[j][slot][2] for j in range(n)], dtype=np.float64)
        if not np.any(coefs):
            continue  # cancelled to exactly zero in every element
        keys = [elems[j][slot][1] for j in range(n)]
        mono: Optional[Expression] = None
        prev = None
        power = 0
        factors: list = []
        for pos in range(len(keys[0])):
            f = _factor_array([k[pos] for k in keys], shape, factor_nodes)
            if f is prev:
                power += 1
                continue
            if prev is not None:
                factors.append((prev, power))
            prev, power = f, 1
        if prev is not None:
            factors.append((prev, power))
        for f, k in factors:
            fac = f if k == 1 else BinaryOp("**", f, _wrap(float(k)))
            mono = fac if mono is None else BinaryOp("*", mono, fac)
        assert mono is not None
        if np.all(coefs == coefs[0]):
            c0 = float(coefs[0])
            terms.append(mono if c0 == 1.0 else BinaryOp("*", _wrap(c0), mono))
        else:
            terms.append(BinaryOp("*", Constant(coefs.reshape(shape)), mono))
    if np.any(const):
        terms.append(Constant(const.reshape(shape)))
    body: Optional[Expression] = None
    for term in terms:
        body = term if body is None else BinaryOp("+", body, term)
    if body is None or _known_shape(body) != shape:
        zeros = Constant(np.zeros(shape))
        body = zeros if body is None else BinaryOp("+", zeros, body)
    return body


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
    # #2097: an opaque reduction or index keeps its place in the row, but the array
    # it reads is collected exactly (once per model pass, through ``memo``).
    if type(node) is SumExpression:
        new = canonicalize_expression(node.operand, memo=memo, _params=found)
        return node if new is node.operand else SumExpression(new, axis=node.axis)
    if type(node) is MatMulExpression:
        left = canonicalize_expression(node.left, memo=memo, _params=found)
        right = canonicalize_expression(node.right, memo=memo, _params=found)
        if left is node.left and right is node.right:
            return node
        return MatMulExpression(left, right)
    if type(node) is IndexExpression and not isinstance(node.base, (Variable, Parameter)):
        new = canonicalize_expression(node.base, memo=memo, _params=found)
        return node if new is node.base else IndexExpression(new, node.index)
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
    shape = _known_shape(expr) if isinstance(expr, Expression) else None
    if _atom_children_only:
        out = _rebuild_atom_children(expr, memo, found)
    elif not isinstance(expr, Expression):
        out = expr
    elif shape == ():
        out = expr if _fast_clean(expr, memo, found) else _exact_scalar(expr, memo, found)
    elif _fast_clean_array(expr, memo, found):
        out = expr
    else:
        out = _exact_array(expr, memo, found)
    # Keep ``expr`` alive in the memo so its id cannot be reused within this pass.
    memo[key] = (expr, out, tuple(found))
    if _params is not None:
        _params.extend(found)
    return out


def _shared_overflow() -> SharedSumOverflowError:
    return SharedSumOverflowError(
        "a sum shared through its own terms (e.g. r = r + r, repeated) collects to a "
        "coefficient outside the binary64 range; no float row can hold it, and the "
        "shared form costs time exponential in its sharing depth to read (#2100)."
    )


def _exact_scalar(expr: Expression, memo: dict, found: list) -> Expression:
    found.clear()
    coll = _Collector(memo, found)
    try:
        poly = coll.collect(expr)
        if coll.dirty or _needs_rewrite(poly, coll.param_values):
            out = _rebuild(poly, coll.factor_nodes)
            _check_rewrite_is_exact(poly, coll.param_values)
            return out
    except _Overflow:
        if coll.shared:
            raise _shared_overflow() from None
    except _Decline:
        pass
    return expr


def _exact_array(expr: Expression, memo: dict, found: list) -> Expression:
    """The exact pass for a body whose static shape is not ``()`` (#2097): an array
    body, or one whose shape the cheap inference does not know (``dm.sum(...) - 6``).
    Each element is a row of its own and is judged as one."""
    found.clear()
    coll = _Collector(memo, found)
    try:
        arr = coll._array(expr)
        if arr.shape == ():
            poly: _Poly = arr.item()  # the 0-d object array's polynomial
            if any(f[0] == "e" for key in poly for f in key):
                return expr  # an element of an opaque array: no scalar form to emit
            if coll.dirty or _needs_rewrite(poly, coll.param_values):
                out = _rebuild(poly, coll.factor_nodes)
                _check_rewrite_is_exact(poly, coll.param_values)
                return out
            return expr
        polys = arr.reshape(-1).tolist()
        if coll.dirty or any(_needs_rewrite(p, coll.param_values) for p in polys):
            out = _rebuild_array(arr, coll.factor_nodes)
            for p in polys:
                _check_rewrite_is_exact(p, coll.param_values)
            return out
    except _Overflow:
        if coll.shared:
            raise _shared_overflow() from None
    except _Decline:
        pass
    return expr


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
    except (
        ParameterCancellationError,
        ArrayRowCancellationError,
        SharedSumOverflowError,
    ) as e:
        raise type(e)(f"{what}: {e}") from None
    holder._exact_body = new
    holder._exact_params = _snapshot(found)
    return new


def _row_label(constraint, where: Optional[str]) -> str:
    """How an error names a row: its name, else its position in the model.

    No body excerpt: ``repr`` of an expression recurses, so on a deep ``+`` chain it
    raises ``RecursionError`` -- measured on #2094 (``test_sweep_rust_presolve_
    fbbt_rounding``), where an eager excerpt broke every solve of such a row.
    """
    name = getattr(constraint, "name", None)
    if name is not None:
        return f"constraint {name!r}"
    place = f" at {where}" if where else ""
    return f"unnamed constraint{place} ('{constraint.sense} 0' row)"


def canonicalize_constraint(
    constraint, memo: Optional[dict] = None, where: Optional[str] = None
) -> None:
    """Replace ``constraint.body`` in place by its canonical form when needed.

    In place so the caller's object stays the model's row (identity-keyed APIs).
    Cached on the row by body identity and Parameter values (see
    :func:`_canonical_cached`), so a row is walked once per distinct set of values.
    ``where`` locates an unnamed row in an error message (e.g. ``_constraints[3]``).
    """
    new = _canonical_cached(constraint, "body", memo, _row_label(constraint, where))
    if new is not constraint.body:
        constraint.body = new


def canonicalize_model(model) -> None:
    """:func:`canonicalize_rows_in` over every row of ``model``, and its objective."""
    memo: dict = {}
    for i, item in enumerate(model._constraints):
        canonicalize_rows_in(item, memo, where=f"_constraints[{i}]")
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
            except (
                ParameterCancellationError,
                ArrayRowCancellationError,
                SharedSumOverflowError,
            ) as e:
                raise type(e)(f"if_else {attr}: {e}") from None
            if new is not expr:
                setattr(aux, attr, new)


def canonicalize_rows_in(item, memo: Optional[dict] = None, where: Optional[str] = None) -> None:
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
        canonicalize_constraint(item, memo, where)
    elif isinstance(item, _IndicatorConstraint):
        canonicalize_rows_in(item.constraint, memo, where)
    elif isinstance(item, _DisjunctiveConstraint):
        canonicalize_rows_in(item.disjuncts, memo, where)
    elif isinstance(item, (list, tuple)):
        for k, sub in enumerate(item):
            canonicalize_rows_in(sub, memo, f"{where}[{k}]" if where else None)
