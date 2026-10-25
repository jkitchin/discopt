"""Canonical factorable normal form for the claim dispatch (issue #632, R1.1).

This module hash-conses a model's objective + constraint expression DAGs into a
content-addressed canonical form. It underpins the uniform factorable relaxation
engine (``uniform_relax.build_uniform_relaxation``) that, as of the #632 cutover,
replaced the federated defer-list of claim predicates with one uniform "one owner
per atom" dispatch (the BARON/Couenne/SCIP AVM shape).

What "canonical" buys:

- **Content addressing.** Structurally identical subexpressions intern to the
  *same* :class:`CNode` object, so a claim keyed on a ``CNode`` survives the
  ``distribute_products`` tree rebuild by construction (killing the id()-orphaning
  and keep-alive-pinning fragility of the current federation) and shares one aux
  column across occurrences (CSE).
- **A total grammar.** Every node is a var / const / sum / prod / pow / call /
  callN / opaque, normalized deterministically, so the downstream atomizer can
  assign exactly one owner per node with no arbitration.

Design decision (refinement over the plan's §2.1 draft — recorded in the plan
§9): canonicalization is **fully box-independent**. Division ``a/b`` is
represented structurally as ``a · b^-1`` (a negative-exponent product factor);
the *soundness* gate that a reciprocal needs a sign-definite denominator is a
**dispatch** concern (checked per node box, where a zero-spanning denominator
routes the ratio atom to the composed fallback), NOT a canonicalization concern.
This keeps one canonical DAG valid across every B&B node while per-node dispatch
stays box-aware. Repeated-factor merging is therefore restricted to
positive-integer exponents so that ``x · x^-1`` is never collapsed to ``1`` (which
would change the value at ``x = 0``).

Semantic faithfulness is the load-bearing property and is tested directly
(``test_canonical_expr.py``): :func:`reconstruct` rebuilds an ``Expression`` from
a ``CNode`` and it must evaluate equal to the original at random points; anything
the grammar cannot represent soundly becomes an ``opaque`` node that reconstructs
to the *original* subexpression verbatim, so equivalence holds trivially there and
the relaxation layer falls back to its existing path for that node.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import operator
import time
from typing import Any, Callable, Optional

import numpy as np

from discopt._relax.scalarize import (
    scalar_elements,
    scalar_matmul_contraction,
    static_shape,
    sum_is_full_reduction,
)
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    maximum,
)

__all__ = [
    "CNode",
    "CanonicalDAG",
    "canonicalize",
    "reconstruct",
    "UnsupportedCanonicalization",
    "RelaxationBuildAborted",
    "Atom",
    "AtomPartition",
    "atomize",
    "var_support",
    "is_affine",
]


class RelaxationBuildAborted(Exception):
    """A relaxation build (or the canonicalization under it) ran out of time.

    Raised only for a caller that passed an abort deadline -- i.e. one with a
    sound way to go without this relaxation (decline a structure, report a node
    as bound-less). Nothing built so far is returned or cached: an aborted build
    is discarded whole, never used truncated, so a bound cannot come from it.
    Defined here, the lowest layer that polls; re-exported by ``uniform_relax``.
    """


#: Recursive steps between deadline polls under an abort deadline. One poll is a
#: ``time.perf_counter()`` call; one step of a product costs ~70-170 us (QPLIB_3709),
#: so a 256-call stride bounds the overshoot to tens of milliseconds.
ABORT_POLL_STRIDE = 256


class UnsupportedCanonicalization(Exception):
    """Raised internally when a node cannot be canonicalized soundly.

    Never escapes to a caller: :func:`canonicalize` catches it and produces an
    ``opaque`` :class:`CNode` wrapping the original subexpression, so the
    relaxation layer falls back to its existing path for exactly that node.
    """


# --------------------------------------------------------------------------- #
# CNode — an interned canonical node
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True, eq=False)
class CNode:
    """One node of the canonical DAG. Interned by :attr:`key` within a build.

    Two ``CNode`` objects are the same iff they are the *same object* (identity);
    interning guarantees structural equality ⇒ identity. :attr:`key` is a fully
    primitive nested tuple (str/int/float), so it is hashable, orderable by
    ``repr``, and content-addressed (deterministic across construction orders).

    kind      one of: "var", "const", "sum", "prod", "pow", "call", "callN",
              "opaque".
    key       the canonical content key (hashable, deterministic).
    children  the CNode operands (empty for leaves/opaque).
    payload   kind-specific data: var→flat index; const→float; sum→(coeffs,const);
              prod→(exponents,); pow→exponent; call/callN→func name; opaque→the
              original Expression + a unique token.
    """

    kind: str
    key: tuple
    children: tuple["CNode", ...]
    payload: Any

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return f"CNode({self.kind}, {self.key!r})"

    @property
    def is_opaque(self) -> bool:
        return self.kind == "opaque"


def _sort_key(node: "CNode"):
    """Total, deterministic ordering of CNodes (by their primitive key repr)."""
    return repr(node.key)


# Numpy-backed evaluators for constant-argument intrinsic calls. Deliberately the
# numpy analogues of discopt's own (jnp-based) evaluator, which agree to ~1e-15 on
# these functions — well inside the semantic-equivalence tolerance. Intrinsics
# whose evaluator is not a plain numpy ufunc (erf/sigmoid/softplus/sign) and the
# n-ary ops beyond min/max are intentionally absent: those are left unfolded (a
# call node), which the atomizer's zero-support guard still keeps out of the
# univariate bucket.
_UNARY_CONST_FN = {
    "exp": np.exp, "log": np.log, "log2": np.log2, "log10": np.log10, "sqrt": np.sqrt,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "atan": np.arctan, "asin": np.arcsin,
    "acos": np.arccos, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh, "log1p": np.log1p,
    "abs": np.abs,
}  # fmt: skip
_NARY_CONST_FN = {"min": min, "max": max}


def _fold_const_call(name: str, values: tuple[float, ...]) -> Optional[float]:
    """Evaluate an intrinsic on constant argument(s), or ``None`` to not fold.

    A call whose arguments are all constant (e.g. ``sin(2.0)``) is a constant, not
    a nonlinear atom; folding it keeps it out of the atomizer's univariate bucket
    (the #636 Finding-2 trap of a zero-variable "univariate" atom). Returns
    ``None`` (leave the call unfolded) for unsupported intrinsics or when the
    result is not a finite real (out-of-domain arg, e.g. ``log(-1)`` → nan) — never
    a nan/complex constant in the canonical form.
    """
    try:
        with np.errstate(all="ignore"):  # out-of-domain (log(-1)) -> nan, handled below
            if len(values) == 1 and name in _UNARY_CONST_FN:
                out = float(_UNARY_CONST_FN[name](float(values[0])))
            elif name in _NARY_CONST_FN:
                out = float(_NARY_CONST_FN[name](*(float(v) for v in values)))
            else:
                return None
    except (ArithmeticError, ValueError, TypeError):
        return None
    return out if math.isfinite(out) else None


# --------------------------------------------------------------------------- #
# Canonicalizer
# --------------------------------------------------------------------------- #
class _Canonicalizer:
    """Builds interned CNodes for one model (one build's lifetime)."""

    def __init__(self, model: Model, abort_deadline: Optional[float] = None):
        self.model = model
        self.n_vars = sum(v.size for v in model._variables)
        self._abort_deadline = abort_deadline
        self._abort_polls = 0
        self._intern: dict[tuple, CNode] = {}
        # id(expr) -> (expr, CNode). Each entry PINS its source Expression, so an
        # id() in the memo can never be recycled by a different expression while
        # the entry exists. Pinning by the model alone is not enough: the sum
        # branch canonicalizes temporary element expressions from
        # ``scalar_elements``, which are freed on return, and a later temporary
        # reusing the address got the stale CNode of a different term — a false
        # certified bound (#1283).
        self._memo: dict[int, tuple[Expression, CNode]] = {}
        self._opaque_counter = itertools.count()
        # #1944: distribute a product of two affine forms (see
        # ``SolverTuning.distribute_affine_products``). Read once per build.
        from discopt.solver_tuning import current as _tuning

        _tun = _tuning()
        self._distribute_max = (
            int(_tun.distribute_affine_products_max) if _tun.distribute_affine_products else 0
        )

    # -- interning ---------------------------------------------------------- #
    def _mk(self, kind: str, key: tuple, children: tuple[CNode, ...], payload: Any) -> CNode:
        existing = self._intern.get(key)
        if existing is not None:
            return existing
        node = CNode(kind=kind, key=key, children=children, payload=payload)
        self._intern[key] = node
        return node

    def _var(self, flat: int) -> CNode:
        return self._mk("var", ("var", int(flat)), (), int(flat))

    def _const(self, value: float) -> CNode:
        v = float(value)
        return self._mk("const", ("const", v), (), v)

    def _opaque(self, expr: Expression) -> CNode:
        # Distinct opaque nodes must not collapse (they may be different subexprs);
        # a per-node token keeps their keys unique while still interning the exact
        # same object if visited twice.
        token = next(self._opaque_counter)
        return self._mk("opaque", ("opaque", token), (), expr)

    def _pow(self, base: CNode, p: float) -> CNode:
        if p == 1.0:
            return base
        if p == 0.0:
            return self._const(1.0)
        return self._mk("pow", ("pow", base.key, float(p)), (base,), float(p))

    def _call(self, name: str, arg: CNode) -> CNode:
        if arg.kind == "const":
            folded = _fold_const_call(name, (arg.payload,))
            if folded is not None:
                return self._const(folded)
        return self._mk("call", ("call", name, arg.key), (arg,), name)

    def _callN(self, name: str, args: tuple[CNode, ...]) -> CNode:
        if all(a.kind == "const" for a in args):
            folded = _fold_const_call(name, tuple(a.payload for a in args))
            if folded is not None:
                return self._const(folded)
        key = ("callN", name, tuple(a.key for a in args))
        return self._mk("callN", key, args, name)

    # -- sum / prod builders (normalizing) ---------------------------------- #
    def _sum(self, terms: list[tuple[float, CNode]], const: float) -> CNode:
        """Build a normalized additive node from (coef, child) terms + constant.

        Flattens nested sums, folds child constants into ``const``, merges equal
        children by summing coefficients, drops zero-coefficient terms, and orders
        terms deterministically. Collapses to a bare child/const when trivial.
        """
        acc: dict[tuple, list] = {}  # child.key -> [coef, child]
        const_acc = float(const)
        stack = list(terms)
        while stack:
            coef, child = stack.pop()
            coef = float(coef)
            if coef == 0.0:
                continue
            if child.kind == "const":
                const_acc += coef * child.payload
                continue
            if child.kind == "sum":
                child_coeffs, child_const = child.payload
                const_acc += coef * child_const
                for cc, ch in zip(child_coeffs, child.children):
                    stack.append((coef * cc, ch))
                continue
            slot = acc.get(child.key)
            if slot is None:
                acc[child.key] = [coef, child]
            else:
                slot[0] += coef
        ordered = sorted(
            ((c, ch) for c, ch in acc.values() if c != 0.0),
            key=lambda t: _sort_key(t[1]),
        )
        if not ordered:
            return self._const(const_acc)
        if len(ordered) == 1 and const_acc == 0.0 and ordered[0][0] == 1.0:
            only_child: CNode = ordered[0][1]
            return only_child
        coeffs = tuple(float(c) for c, _ in ordered)
        children = tuple(ch for _, ch in ordered)
        key = ("sum", tuple((c, ch.key) for c, ch in zip(coeffs, children)), const_acc)
        return self._mk("sum", key, children, (coeffs, const_acc))

    def _prod(self, factors: list[tuple[CNode, float]], coef: float = 1.0) -> CNode:
        """Build a normalized product from (base, exponent) factors × a scalar coef.

        Flattens nested products, folds constant bases into ``coef``, and merges
        equal bases by **summing exponents ONLY when both are positive** (so a
        genuine reciprocal ``x^-1`` never cancels ``x`` into ``x^0`` — that would
        change the value at ``x = 0``). A leading scalar ``coef`` is emitted as a
        surrounding sum (``coef · prod``) so products stay monic.
        """
        merged: dict[tuple, list] = {}  # base.key -> [exp_sum, base] (mergeable set)
        unmerged: list[tuple[CNode, float]] = []  # non-mergeable factors kept as-is
        coef_acc = float(coef)
        stack = list(factors)
        # Preserve coupled univariate atoms: when a factor is a transcendental
        # ``call`` (log/exp/…), a sibling scaled factor ``c·g`` must stay INTACT so
        # the atom recognizers see the shared affine form — e.g. ``(2x)·log(2x)`` is
        # ``t·log(t)`` with ``t = 2x``, but extracting the ``2`` into ``coef_acc``
        # would leave ``2·(x·log(2x))`` and break the ``_entropy_prod_var`` /
        # ``_xexp_prod_var`` match (issue #640 Bucket 4 regression). The
        # scalar-extraction CSE only targets pure variable products (``x0·x1``),
        # which carry no ``call`` factor, so gate it off when one is present.
        _has_call_factor = any(getattr(b, "kind", None) == "call" for b, _ in factors)
        while stack:
            base, exp = stack.pop()
            exp = float(exp)
            if exp == 0.0:
                continue
            if base.kind == "const":
                # const**exp is a constant multiplier. It is not total over the
                # reals: 0**(negative) raises (e.g. division by a zero-valued
                # parameter) and (negative)**(fractional) yields a complex. Both
                # mean this node has no sound real canonical form, so route it to
                # opaque rather than letting the exception escape canonicalize()
                # (the module's "never raises to the caller" contract).
                try:
                    factor = float(base.payload) ** exp
                except ArithmeticError as exc:
                    raise UnsupportedCanonicalization("undefined constant power") from exc
                if isinstance(factor, complex):
                    raise UnsupportedCanonicalization("non-real constant power")
                coef_acc *= factor
                continue
            if base.kind == "prod":
                inner_exps = base.payload[0]
                inner_coef = base.payload[1] if len(base.payload) > 1 else 1.0
                coef_acc *= float(inner_coef) ** exp
                for ie, ich in zip(inner_exps, base.children):
                    stack.append((ich, ie * exp))
                continue
            if base.kind == "pow":
                # (b^q)^exp -> b^(q*exp), keep folding.
                stack.append((base.children[0], base.payload * exp))
                continue
            if base.kind == "sum":
                # A monic scaled single term ``c·g`` (one child, zero const) inside a
                # product: ``(c·g)^exp = c^exp · g^exp``. Extract the scalar into
                # ``coef_acc`` and keep folding ``g`` so the product stays MONIC. This
                # canonicalizes ``(-2·x0)·x1`` and ``(-2·x1)·x0`` to the same
                # ``-2·prod(x0,x1)``, so commuted scaled products share one interned
                # node / lifted column (CSE) instead of two (issue #640 Bucket 4).
                # Guarded to a real ``c^exp`` (integer exp, or c>0 for a fractional
                # one) so no complex/undefined power slips through.
                scoeffs, sconst = base.payload
                if (
                    not _has_call_factor
                    and float(sconst) == 0.0
                    and len(base.children) == 1
                    and len(scoeffs) == 1
                    and float(scoeffs[0]) != 0.0
                    and (float(exp).is_integer() or float(scoeffs[0]) > 0.0)
                ):
                    try:
                        factor = float(scoeffs[0]) ** exp
                    except ArithmeticError as exc:
                        raise UnsupportedCanonicalization("undefined scaled-factor power") from exc
                    if isinstance(factor, complex):
                        raise UnsupportedCanonicalization("non-real scaled-factor power")
                    coef_acc *= factor
                    stack.append((base.children[0], exp))
                    continue
            # Mergeable only when positive (see docstring).
            if exp > 0:
                slot = merged.get(base.key)
                if slot is None:
                    merged[base.key] = [exp, base]
                else:
                    slot[0] += exp
            else:
                unmerged.append((base, exp))

        factor_list: list[tuple[CNode, float]] = [
            (base, e) for e, base in merged.values() if e != 0.0
        ]
        factor_list.extend(unmerged)
        if coef_acc == 0.0:
            return self._const(0.0)
        if not factor_list:
            return self._const(coef_acc)
        factor_list.sort(key=lambda t: _sort_key(t[0]))
        # A single-factor product is never a ``prod`` node: b^1 is just ``b`` and
        # b^e is ``pow(b, e)`` — with the scalar coef applied by a surrounding sum.
        # (Collapsing this matters once constant-folding can turn a 2-factor
        # product like ``sin(2.0)·z`` into a single ``z`` factor; leaving it as
        # ``prod(z^1)`` would spuriously mark the affine result nonlinear — #636.)
        if len(factor_list) == 1:
            base_node, e = factor_list[0]
            single = base_node if e == 1.0 else self._pow(base_node, e)
            if coef_acc == 1.0:
                return single
            return self._sum([(coef_acc, single)], 0.0)
        exps = tuple(float(e) for _, e in factor_list)
        children = tuple(b for b, _ in factor_list)
        prod_key = ("prod", tuple((ch.key, e) for ch, e in zip(children, exps)))
        prod_node = self._mk("prod", prod_key, children, (exps,))
        if coef_acc == 1.0:
            return prod_node
        return self._sum([(coef_acc, prod_node)], 0.0)

    @staticmethod
    def _affine_in_vars(node: CNode) -> Optional[tuple[list[tuple[float, CNode]], float]]:
        """``(terms, const)`` when ``node`` is ``const + sum_i c_i x_i`` over bare
        variables, else ``None``."""
        if node.kind == "var":
            return [(1.0, node)], 0.0
        if node.kind == "sum" and all(ch.kind == "var" for ch in node.children):
            coeffs, const = node.payload
            return [(float(c), ch) for c, ch in zip(coeffs, node.children)], float(const)
        return None

    def _distribute_affine_pair(self, left: CNode, right: CNode) -> Optional[CNode]:
        """``(c + a'x) * (d + b'y)`` as ``cd + c b'y + d a'x + sum a_i b_j x_i y_j``.

        ``None`` (keep the product) unless both factors are affine in bare
        variables, their supports are disjoint, at least one has more than one
        part, and the pair count is within the cap. Value-identical either way;
        why the distributed form never relaxes looser is in
        ``SolverTuning.distribute_affine_products`` (#1944).
        """
        fa = self._affine_in_vars(left)
        fb = self._affine_in_vars(right)
        if fa is None or fb is None:
            return None
        (ta, ca), (tb, cb) = fa, fb
        if len(ta) + (ca != 0.0) < 2 and len(tb) + (cb != 0.0) < 2:
            return None  # a bare bilinear x*y: nothing to distribute
        if {v.key for _, v in ta} & {v.key for _, v in tb}:
            return None
        if len(ta) * len(tb) > self._distribute_max:
            return None
        terms: list[tuple[float, CNode]] = []
        for ai, xi in ta:
            for bj, yj in tb:
                terms.append((ai * bj, self._prod([(xi, 1.0), (yj, 1.0)])))
        if cb != 0.0:
            terms.extend((cb * ai, xi) for ai, xi in ta)
        if ca != 0.0:
            terms.extend((ca * bj, yj) for bj, yj in tb)
        return self._sum(terms, ca * cb)

    # -- the recursive walk ------------------------------------------------- #
    def canon(self, expr: Expression) -> CNode:
        cached = self._memo.get(id(expr))
        if cached is not None:
            assert cached[0] is expr, "canonical memo id() collision (#1283)"
            return cached[1]
        if self._abort_deadline is not None:
            self._abort_polls += 1
            if (
                self._abort_polls % ABORT_POLL_STRIDE == 0
                and time.perf_counter() >= self._abort_deadline
            ):
                raise RelaxationBuildAborted(
                    f"canonicalization passed its abort deadline after "
                    f"{self._abort_polls} expressions"
                )
        try:
            node = self._canon_dispatch(expr)
        except UnsupportedCanonicalization:
            node = self._opaque(expr)
        self._memo[id(expr)] = (expr, node)
        return node

    def _canon_dispatch(self, expr: Expression) -> CNode:
        from discopt._relax.term_classifier import _get_flat_index
        from discopt.operators import atom_of

        # A registered domain operator (#1248 A): the model carries the LOWERING —
        # so evaluation, export and the Rust core see an ordinary expression — and
        # the lowering carries a tag naming the atom. Emitting the named ``call``
        # node here is what lets the relaxer envelope the composite whole instead
        # of term by term, which is the entire point of registering it.
        atom = atom_of(expr)
        if atom is not None:
            name, arg = atom
            return self._call(name, self.canon(arg))

        if isinstance(expr, Constant):
            if expr.value.ndim == 0:
                return self._const(float(expr.value))
            raise UnsupportedCanonicalization("array constant")

        if isinstance(expr, Parameter):
            val = np.asarray(expr.value)
            if val.ndim == 0:
                return self._const(float(val))
            raise UnsupportedCanonicalization("array parameter")

        if isinstance(expr, Variable):
            if expr.size == 1:
                flat = _get_flat_index(expr, self.model)
                if flat is not None:
                    return self._var(flat)
            raise UnsupportedCanonicalization("array variable")

        if isinstance(expr, IndexExpression):
            flat = _get_flat_index(expr, self.model)
            if flat is not None:
                return self._var(flat)
            raise UnsupportedCanonicalization("non-scalar index")

        if isinstance(expr, UnaryOp):
            if expr.op == "neg":
                return self._sum([(-1.0, self.canon(expr.operand))], 0.0)
            if expr.op == "abs":
                return self._call("abs", self.canon(expr.operand))
            raise UnsupportedCanonicalization(f"unary {expr.op}")

        if isinstance(expr, BinaryOp):
            return self._canon_binary(expr)

        if isinstance(expr, FunctionCall):
            return self._canon_call(expr)

        if isinstance(expr, SumOverExpression):
            return self._sum([(1.0, self.canon(t)) for t in expr.terms], 0.0)

        if isinstance(expr, SumExpression):
            # sum(array_variable) over its elements is affine; anything else is opaque.
            # Only a FULL reduction collapses to one affine form: `sum(A, axis=1)`
            # is one row per row of `A`, and folding its elements into a single
            # canonical sum is the axis-collapse of #1160.
            if not sum_is_full_reduction(expr):
                raise UnsupportedCanonicalization("axis-reduced sum (array-valued)")
            op = expr.operand
            if isinstance(op, Variable):
                from discopt._relax.term_classifier import _compute_var_offset

                offset = _compute_var_offset(op, self.model)
                terms = [(1.0, self._var(offset + k)) for k in range(op.size)]
                return self._sum(terms, 0.0)
            # A reduction over anything else — ``sum(10**y)``, ``sum(x*z)`` — is a
            # sum of its scalar ELEMENTS, which :mod:`_relax.scalarize` already
            # derives for the constraint path (it is value-preserving by
            # construction). Without this the whole reduction was one opaque node,
            # so ``log10(sum(10**y))`` had no envelope and the solve returned
            # ``bound=None`` (#1256) — while the same model written with scalar
            # variables relaxed fine. ``None`` means the elements cannot be derived
            # statically; that is the previous behaviour, unchanged.
            elems = scalar_elements(op)
            if elems is None:
                raise UnsupportedCanonicalization("sum reduction")
            return self._sum([(1.0, self.canon(e)) for e in elems], 0.0)

        if isinstance(expr, MatMulExpression):
            # A scalar-valued ``c @ x`` is the affine contraction ``sum_k c_k x_k``.
            # Left opaque, it had no envelope and an unbounded aux column, so a
            # trivial integer MILP written ``minimize(c @ x)`` got no dual bound
            # from AMP (#1502) while the same model with scalar terms certified.
            # ``None`` (array-valued, unknown shape, over the cap) keeps it opaque.
            contraction = scalar_matmul_contraction(expr)
            if contraction is None:
                raise UnsupportedCanonicalization("matmul")
            return self.canon(contraction)

        raise UnsupportedCanonicalization(type(expr).__name__)

    def _canon_binary(self, expr: BinaryOp) -> CNode:
        op = expr.op
        if op == "+":
            return self._sum([(1.0, self.canon(expr.left)), (1.0, self.canon(expr.right))], 0.0)
        if op == "-":
            return self._sum([(1.0, self.canon(expr.left)), (-1.0, self.canon(expr.right))], 0.0)
        if op == "*":
            left, right = expr.left, expr.right
            if isinstance(left, Constant) and left.value.ndim == 0:
                return self._sum([(float(left.value), self.canon(right))], 0.0)
            if isinstance(right, Constant) and right.value.ndim == 0:
                return self._sum([(float(right.value), self.canon(left))], 0.0)
            cl, cr = self.canon(left), self.canon(right)
            if self._distribute_max:
                dist = self._distribute_affine_pair(cl, cr)
                if dist is not None:
                    return dist
            return self._prod([(cl, 1.0), (cr, 1.0)])
        if op == "/":
            left, right = expr.left, expr.right
            if isinstance(right, Constant) and right.value.ndim == 0:
                denom = float(right.value)
                if denom == 0.0:
                    raise UnsupportedCanonicalization("division by zero constant")
                return self._sum([(1.0 / denom, self.canon(left))], 0.0)
            # a / b -> a · b^-1 (structural; sign-definiteness is a dispatch gate).
            return self._prod([(self.canon(left), 1.0), (self.canon(right), -1.0)])
        if op == "**":
            base, rexp = expr.left, expr.right
            if isinstance(rexp, Constant) and rexp.value.ndim == 0:
                return self._pow(self.canon(base), float(rexp.value))
            if isinstance(base, Constant) and base.value.ndim == 0:
                # ``b**e`` with a POSITIVE constant base is ``exp(ln(b) * e)`` —
                # an identity, not an approximation, for every real ``e``. Without
                # it the node is opaque, and an opaque node has no envelope: the
                # relaxation then has no valid objective bound at all and the solve
                # returns ``bound=None`` after one node (#1256, ``10**y`` written
                # for a log-concentration). ``exp`` is in the univariate table with
                # a certified-convex verdict on every box, so the rewrite hands the
                # engine a term it already relaxes.
                b = float(base.value)
                if b > 0.0:
                    return self._call("exp", self._sum([(math.log(b), self.canon(rexp))], 0.0))
                # ``b <= 0`` has no real-valued continuous extension in ``e``
                # (``(-2)**0.5``), so it stays unsupported rather than being given
                # an envelope that is wrong off the integers.
                raise UnsupportedCanonicalization("non-positive constant base")
            raise UnsupportedCanonicalization("non-constant exponent")
        raise UnsupportedCanonicalization(f"binary {op}")

    def _canon_call(self, expr: FunctionCall) -> CNode:
        name = expr.func_name
        args = expr.args
        if name.startswith("norm") and len(args) == 1:
            # ``dm.norm`` of a VECTOR is a reduction over its elements, spelled out
            # here exactly as the POUNCE tape lowers it (``_nl_expr_compiler.
            # _lower_norm``): ``norm1 = sum|e_i|``, ``norminf = max|e_i|``,
            # ``norm2 = sqrt(sum e_i*e_i)``, ``normp = (sum |e_i|**p)**(1/p)``. The
            # canonical node is then the one the user's own scalar spelling of the
            # same norm produces, so it relaxes through the existing abs / max /
            # square / sqrt envelopes and no new envelope is introduced (#1493).
            # Before, the vector argument became an array-valued opaque child and
            # the relaxation build raised ``TypeError`` on it, so no norm of a
            # vector ever had a dual bound.
            spelled = _norm_scalar_spelling(name, args[0])
            if spelled is not None:
                return self.canon(spelled)
            # #1903: the SPECTRAL norm of a matrix (largest singular value) is no
            # fold over elements, but it is a convex function of them with a
            # dedicated envelope (`uniform_relax._build_spectral_norm`). It
            # becomes a multivariable node over the row-major elements, the shape
            # carried in its name so the builder and the interval rule can rebuild
            # the matrix.
            shape = static_shape(args[0])
            if _norm_order(name) == 2.0 and shape is not None and len(shape) == 2:
                if 0 not in shape:
                    elems = scalar_elements(args[0])
                    if elems is not None:
                        return self._callN(
                            f"specnorm:{shape[0]}x{shape[1]}",
                            tuple(self.canon(e) for e in elems),
                        )
        if name == "prod" and len(args) == 1:
            # ``prod`` of an array is the product of its ELEMENTS (``jnp.prod``;
            # the tape's ``*`` fold in ``_nl_expr_compiler``), so it is spelled
            # out exactly as the user's ``x[0] * x[1] * ...`` would be and relaxes
            # through the existing bilinear/multilinear envelopes (#1582). Before,
            # the call stayed opaque with an array-shaped enclosure and the
            # relaxation build refused it, so no ``dm.prod`` model had a bound.
            spelled = _prod_scalar_spelling(args[0])
            if spelled is not None:
                return self.canon(spelled)
        if any(_array_valued(a) for a in args):
            # A call over an array-valued argument has no scalar canonical form
            # (a 2-D ``norm`` is jnp's MATRIX norm, not a fold over elements).
            # Keep the WHOLE call opaque -- a scalar node with its interval
            # enclosure -- rather than handing the builder an array-valued child
            # (#1493: that raised ``TypeError`` deep in the relaxation build).
            raise UnsupportedCanonicalization(f"call {name} of an array-valued argument")
        if len(args) == 1:
            return self._call(name, self.canon(args[0]))
        if len(args) >= 2:
            return self._callN(name, tuple(self.canon(a) for a in args))
        raise UnsupportedCanonicalization(f"call {name}/{len(args)}")


def _array_valued(expr: Expression) -> bool:
    """True when ``expr`` is statically known to be non-scalar."""
    shape = static_shape(expr)
    return shape is not None and shape != ()


def _norm_order(name: str) -> Optional[float]:
    """The ``p`` of a ``norm{p}`` call name, or ``None`` if it is not one."""
    suffix = name[len("norm") :]
    if suffix == "inf":
        return math.inf
    if suffix == "":
        return 2.0
    try:
        p = float(suffix)
    except ValueError:
        return None
    return p if p >= 1.0 else None


def _prod_scalar_spelling(arg: Expression) -> Optional[Expression]:
    """Value-preserving scalar spelling of a one-argument ``prod(arg)`` (#1582).

    The left ``*`` fold over ``arg``'s C-order elements, matching
    ``_nl_expr_compiler``'s lowering. An empty argument is the empty product
    ``1``. ``None`` (unknown shape, or too large to scalarize) keeps the call
    opaque; its interval enclosure is then the scalar ``_prod_interval``.
    """
    shape = static_shape(arg)
    if shape is None:
        return None
    if shape == ():
        return arg
    if int(np.prod(shape)) == 0:
        return Constant(1.0)
    elems = scalar_elements(arg)
    if elems is None:
        return None
    out: Expression = elems[0]
    for e in elems[1:]:
        out = BinaryOp("*", out, e)
    return out


def _norm_scalar_spelling(name: str, arg: Expression) -> Optional[Expression]:
    """Value-preserving scalar spelling of ``norm{p}(arg)``.

    Mirrors ``_nl_expr_compiler._lower_norm`` term for term, including what it
    refuses: a 0-d argument (``jnp.linalg.norm`` raises on it) and a 2-D one
    other than ``ord`` 1 / inf, and anything of more dimensions
    (``jnp.linalg.norm``'s spectral or undefined matrix norms are not folds over
    elements). Those return ``None`` and the caller keeps the call opaque.

    The 2-D ``ord=1`` / ``ord=inf`` cases are the largest absolute COLUMN / ROW
    sum, spelled out as ``max`` of ``sum`` of ``abs`` -- exactly what the tape
    evaluates, so they relax through the existing abs/max envelopes.
    """
    p = _norm_order(name)
    if p is None:
        return None
    shape = static_shape(arg)
    if shape is None or 0 in shape:
        return None
    if len(shape) == 2 and p in (1.0, math.inf):
        mat = scalar_elements(arg)
        if mat is None:
            return None
        rows, cols = shape
        cell: list[list[Expression]] = [
            [UnaryOp("abs", mat[i * cols + j]) for j in range(cols)] for i in range(rows)
        ]
        if p == 1.0:
            lines = [[cell[i][j] for i in range(rows)] for j in range(cols)]
        else:
            lines = cell
        sums: list[Expression] = [
            line[0] if len(line) == 1 else SumOverExpression(line) for line in lines
        ]
        return sums[0] if len(sums) == 1 else maximum(*sums)
    if len(shape) != 1:
        return None
    elems = scalar_elements(arg)
    if elems is None:
        return None
    absolutes: list[Expression] = [UnaryOp("abs", e) for e in elems]
    if p == math.inf:
        if len(absolutes) == 1:
            return absolutes[0]
        # ``dm.maximum``'s balanced binary fold: the tape's ``max`` is binary.
        return maximum(*absolutes)
    if p == 1.0:
        return SumOverExpression(absolutes)
    if p == 2.0:
        return FunctionCall("sqrt", SumOverExpression([e * e for e in elems]))
    powered: list[Expression] = [BinaryOp("**", a, Constant(float(p))) for a in absolutes]
    return BinaryOp("**", SumOverExpression(powered), Constant(1.0 / p))


@dataclasses.dataclass
class CanonicalDAG:
    """The canonical form of one model's objective + constraint bodies."""

    model: Model
    objective: Optional[CNode]
    #: One node per SCALAR constraint row, not per :class:`Constraint` object: an
    #: array-valued body (``A @ x == b`` on a 3-vector) contributes one node per
    #: flat element, matching the NLP evaluator's row convention (#981). Use
    #: :attr:`constraint_index` to get back to the owning ``Constraint``.
    constraints: tuple[CNode, ...]
    #: ``constraint_index[r]`` is the position in ``model._constraints`` of the
    #: ``Constraint`` that row ``r`` came from. Rows of one array-valued
    #: constraint are contiguous and in row-major (``np.ravel``) order.
    constraint_index: tuple[int, ...]
    #: The scalar body expression behind each row, so callers can recover the
    #: row's expression without re-scalarizing.
    constraint_exprs: tuple[Expression, ...]
    #: ``id(expr) -> (expr, CNode)``; every entry pins its key expression (#1283).
    _memo: dict[int, tuple[Expression, CNode]]
    _intern: dict[tuple, CNode]

    def cnode_of(self, expr: Expression) -> CNode:
        """Return the canonical node for ``expr`` (memoized within this build).

        Valid for the pinned trees this DAG was built from and any subexpression
        thereof. Content-addressed: two structurally equal expressions return the
        same interned ``CNode`` object.
        """
        cached = self._memo.get(id(expr))
        if cached is not None:
            return cached[1]
        # A node not seen during the original walk (e.g. a freshly distributed
        # tree): canonicalize it now against the same intern table so identical
        # structure still maps to the same CNode.
        canon = _Canonicalizer(self.model)
        canon._intern = self._intern
        canon._memo = self._memo
        # Seed the opaque counter past every ("opaque", k) token already interned.
        # A fresh _Canonicalizer resets its counter to 0, so an opaque node in the
        # re-canonicalized subtree would otherwise request key ("opaque", 0) — which
        # may already be interned wrapping a DIFFERENT original expression, and _mk
        # would return that stale node (wrong reconstruct, broken CSE identity).
        # Distinct opaque subtrees must get distinct tokens across cnode_of calls
        # too (#636 Finding 3 — latent until distribute_products calls cnode_of).
        opaque_tokens = [k[1] for k in self._intern if k[0] == "opaque"]
        canon._opaque_counter = itertools.count(max(opaque_tokens) + 1 if opaque_tokens else 0)
        return canon.canon(expr)

    @property
    def nodes(self) -> tuple[CNode, ...]:
        return tuple(self._intern.values())


def canonicalize(model: Model, abort_deadline: Optional[float] = None) -> CanonicalDAG:
    """Hash-cons ``model``'s objective + constraint bodies into a canonical DAG.

    Pure and box-independent. Unsupported nodes become ``opaque`` CNodes wrapping
    the original subexpression (never raises to the caller).

    Array-valued constraint bodies are expanded element-wise first (#981), so the
    returned DAG carries one node per SCALAR row. Before this, a vectorized body
    such as ``(-k) * X[:, 1:] == 0`` reached the relaxation builder as a single
    array-valued node whose interval enclosure is an array — which crashed the
    build (``only 0-dimensional arrays can be converted to Python scalars``) and
    cost the model its entire relaxation, collapsing the dual bound to the trivial
    objective floor. An element that cannot be scalarized statically keeps the
    whole constraint unexpanded, exactly as before.

    ``abort_deadline`` (absolute ``time.perf_counter()``): past it, raise
    :class:`RelaxationBuildAborted` instead of finishing. On a dense quadratic this
    walk alone is ~12 s (QPLIB_3709, 165,300 products) and runs before any of the
    relaxation builder's own polls.
    """
    canon = _Canonicalizer(model, abort_deadline=abort_deadline)
    obj_node: Optional[CNode] = None
    if model._objective is not None:
        obj_node = canon.canon(model._objective.expression)

    con_nodes: list[CNode] = []
    con_index: list[int] = []
    con_exprs: list[Expression] = []
    for i, c in enumerate(model._constraints):
        rows = scalar_elements(c.body)
        if rows is None:
            rows = [c.body]  # not statically scalarizable — unchanged behaviour
        for body in rows:
            con_nodes.append(canon.canon(body))
            con_index.append(i)
            con_exprs.append(body)

    return CanonicalDAG(
        model=model,
        objective=obj_node,
        constraints=tuple(con_nodes),
        constraint_index=tuple(con_index),
        constraint_exprs=tuple(con_exprs),
        _memo=canon._memo,
        _intern=canon._intern,
    )


# --------------------------------------------------------------------------- #
# Reconstruction (for the semantic-equivalence gate)
# --------------------------------------------------------------------------- #
def _flat_var_expr(model: Model) -> list[Expression]:
    """Map each flat variable index to a scalar Expression (Variable or index)."""
    out: list[Expression] = []
    for v in model._variables:
        if v.size == 1:
            out.append(v)
        else:
            for k in range(v.size):
                if len(v.shape) > 1:
                    index: Any = tuple(int(i) for i in np.unravel_index(k, v.shape))
                else:
                    index = k
                out.append(v[index])
    return out


# A canonical ``sum``/``prod`` node is n-ary, but ``Expression`` is binary, so
# reconstruct has to fold it. A left fold is ``n`` deep, and every consumer of the
# reconstructed tree (``evaluate_interval``, ``classify_expr``,
# ``compile_expression``) recurses a frame or two per level. A model whose own
# expressions are shallow can still canonicalize to a very wide node -- the
# product of ten shifted binomials distributes to a 1024-term sum -- and its left
# fold then overran CPython's recursion limit inside the relaxation build
# (#1544). Wide nodes are therefore folded as a balanced tree (``log2 n`` deep).
# Up to this width the legacy left fold is kept, so every node that already
# reconstructed keeps its exact operator tree and floating-point evaluation order
# (interval arithmetic is outward-rounded, so either order is sound).
_RECONSTRUCT_FOLD_WIDTH = 128


def _balanced_fold(
    items: list[Expression], op: Callable[[Expression, Expression], Expression]
) -> Expression:
    """Fold ``items`` with the binary ``op`` as a balanced tree, ``ceil(log2 n)``
    deep. Iterative (pairwise rounds), so it cannot itself recurse deeply."""
    level = list(items)
    while len(level) > 1:
        nxt = [op(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def reconstruct(node: CNode, model: Model, _flat: Optional[list[Expression]] = None) -> Expression:
    """Rebuild an :class:`Expression` from a canonical node.

    The round-trip ``reconstruct(canonicalize(...))`` must be value-equivalent to
    the original everywhere it is defined (the property test). ``opaque`` nodes
    reconstruct to their stored original subexpression verbatim.
    """
    flat = _flat if _flat is not None else _flat_var_expr(model)

    def rec(n: CNode) -> Expression:
        if n.kind == "var":
            expr: Expression = flat[n.payload]
            return expr
        if n.kind == "const":
            return Constant(n.payload)
        if n.kind == "opaque":
            orig: Expression = n.payload  # the original Expression
            return orig
        if n.kind == "call":
            return FunctionCall(n.payload, rec(n.children[0]))
        if n.kind == "callN":
            return FunctionCall(n.payload, *[rec(c) for c in n.children])
        if n.kind == "pow":
            powed: Expression = rec(n.children[0]) ** n.payload
            return powed
        if n.kind == "sum":
            coeffs, const = n.payload
            terms = [
                rec(child) if coef == 1.0 else Constant(coef) * rec(child)
                for coef, child in zip(coeffs, n.children)
            ]
            if len(terms) > _RECONSTRUCT_FOLD_WIDTH:
                wide_sum: Expression = Constant(const) + _balanced_fold(terms, operator.add)
                return wide_sum
            sum_acc: Expression = Constant(const)
            for term in terms:
                sum_acc = sum_acc + term
            return sum_acc
        if n.kind == "prod":
            (exps,) = n.payload
            factors = [
                rec(child) if exp == 1.0 else rec(child) ** exp
                for exp, child in zip(exps, n.children)
            ]
            assert factors
            if len(factors) > _RECONSTRUCT_FOLD_WIDTH:
                return _balanced_fold(factors, operator.mul)
            prod_acc = factors[0]
            for factor in factors[1:]:
                prod_acc = prod_acc * factor
            return prod_acc
        raise ValueError(f"cannot reconstruct CNode kind {n.kind!r}")

    return rec(node)


# --------------------------------------------------------------------------- #
# Atomizer — partition the canonical DAG into maximal nonlinear atoms
# --------------------------------------------------------------------------- #
# Each maximal nonlinear canonical node is one *atom* with one coarse kind. The
# kind fixes the owner family for the structurally-keyed side (product/ratio/
# multivar) directly; single-variable atoms carry kind "univariate" and their
# *envelope* owner is chosen per node box by the dominance dispatcher (rule 1..4
# of the plan §2.4) — a bare ``x**p`` univariate atom routes to the same
# monomial-secant kernel as today, ``sin(x)**2`` over a small integer domain to
# the exact table, a non-convex single-variable composite to the 1-D hull. The
# dominance dispatch (which needs the node box + curvature/finite-domain
# predicates + the concrete envelope builders) is threaded in with the R1.2
# wiring; atomization here is box-independent and pure.

_ATOM_OPAQUE = "opaque"
_ATOM_UNIVARIATE = "univariate"
# multivariable product of positive powers (bilinear / multilinear / monomial):
_ATOM_PRODUCT = "product"
_ATOM_RATIO = "ratio"  # multivariable product with a negative exponent (division)
_ATOM_MULTIVAR = "multivar"  # multivariable call/pow/centropy (curvature-certified owner)


def _support_map(nodes) -> dict[int, frozenset[int]]:
    """Memoized variable-support (set of flat var indices) per CNode, by id."""
    memo: dict[int, frozenset[int]] = {}

    def supp(n: CNode) -> frozenset[int]:
        cached = memo.get(id(n))
        if cached is not None:
            return cached
        if n.kind == "var":
            s = frozenset({n.payload})
        elif n.kind in ("const", "opaque"):
            s = frozenset()
        else:
            acc: set[int] = set()
            for c in n.children:
                acc |= supp(c)
            s = frozenset(acc)
        memo[id(n)] = s
        return s

    for node in nodes:
        supp(node)
    return memo


def var_support(node: CNode) -> frozenset[int]:
    """Flat variable indices referenced by ``node`` (opaque contributes none)."""
    return _support_map([node])[id(node)]


def is_affine(node: CNode) -> bool:
    """True if ``node`` is an affine combination of variables (no nonlinear atom).

    ``var``/``const`` are affine; a ``sum`` is affine iff every child is affine;
    everything else (prod/pow/call/callN/opaque) is a nonlinear atom.
    """
    if node.kind in ("var", "const"):
        return True
    if node.kind == "sum":
        return all(is_affine(c) for c in node.children)
    return False


@dataclasses.dataclass(frozen=True)
class Atom:
    """One maximal nonlinear canonical node and its coarse owner kind."""

    node: CNode
    kind: str
    support: frozenset[int]

    @property
    def is_univariate(self) -> bool:
        return self.kind == _ATOM_UNIVARIATE


@dataclasses.dataclass
class AtomPartition:
    """The set of maximal nonlinear atoms of a canonical DAG (one owner each)."""

    atoms: tuple[Atom, ...]

    def by_kind(self, kind: str) -> tuple[Atom, ...]:
        return tuple(a for a in self.atoms if a.kind == kind)

    @property
    def kinds(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.atoms:
            out[a.kind] = out.get(a.kind, 0) + 1
        return out


def _classify_atom(node: CNode, support: frozenset[int]) -> str:
    if node.is_opaque:
        return _ATOM_OPAQUE
    if len(support) <= 1:
        # A single-variable nonlinear node (monomial, power, univariate call, or a
        # single-variable additive/multiplicative composite). Owner chosen by the
        # dominance dispatcher per node box.
        return _ATOM_UNIVARIATE
    if node.kind == "prod":
        (exps,) = node.payload
        if any(e < 0 for e in exps):
            return _ATOM_RATIO
        return _ATOM_PRODUCT
    # multivariable pow / call / callN (centropy, norm, ...) -> curvature-certified
    # multivar owner (or opaque fallback if uncertifiable, decided at dispatch).
    return _ATOM_MULTIVAR


def atomize(dag: CanonicalDAG) -> AtomPartition:
    """Partition ``dag``'s objective + constraint bodies into maximal atoms.

    Walks each root; affine combinations contribute no atom (they stay linear
    rows). A ``sum`` that is itself nonlinear but single-variable is ONE
    univariate atom (e.g. nvs09's ``(ln(x-2))**2 + (ln(10-x))**2``); a
    multivariable ``sum`` is split into the atoms of its nonlinear children.
    Every non-``sum`` nonlinear node is an atom of its classified kind. Atoms are
    deduplicated by CNode identity (CSE: one atom per shared subexpression).
    """
    roots: list[CNode] = []
    if dag.objective is not None:
        roots.append(dag.objective)
    roots.extend(dag.constraints)

    supp = _support_map(roots)
    seen: set[int] = set()
    atoms: list[Atom] = []

    _opaque_memo: dict[int, bool] = {}

    def has_opaque(n: CNode) -> bool:
        cached = _opaque_memo.get(id(n))
        if cached is not None:
            return cached
        r = n.is_opaque or any(has_opaque(c) for c in n.children)
        _opaque_memo[id(n)] = r
        return r

    def visit(n: CNode) -> None:
        if is_affine(n):
            return
        s = supp[id(n)]
        # A sum is one univariate atom ONLY when it is genuinely single-variable
        # AND contains no opaque descendant (an opaque node has empty variable
        # support, so it would otherwise hide inside a single-var sum). Otherwise
        # descend so each nonlinear/opaque child becomes its own atom and the
        # affine part stays linear.
        if n.kind == "sum" and (len(s) > 1 or has_opaque(n)):
            for c in n.children:
                visit(c)
            return
        # A non-opaque node with zero variable support is constant-valued (e.g.
        # a ``sin(2.0)`` that did not constant-fold, or a call on parameters). It
        # has no variable to build a 1-D envelope over, so it is NOT a univariate
        # atom; the evaluator/linear path handles the constant. Opaque nodes also
        # have empty support but must still surface as their own fallback atom —
        # hence the ``not is_opaque`` (the #636 Finding-2 zero-variable trap).
        if not s and not n.is_opaque:
            return
        # Otherwise this node is a maximal nonlinear atom.
        if id(n) in seen:
            return
        seen.add(id(n))
        atoms.append(Atom(node=n, kind=_classify_atom(n, s), support=s))

    for r in roots:
        visit(r)

    # Deterministic order for reproducible plans/tests.
    atoms.sort(key=lambda a: _sort_key(a.node))
    return AtomPartition(atoms=tuple(atoms))
