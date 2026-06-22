"""Integer-factor bilinear reformulation: relax ``x_i * x_j`` *exactly* when a
factor is an integer variable.

A single continuous McCormick envelope of a bilinear term ``x_i * x_j`` over a
wide box is loose — its integer optimum can sit strictly below the true optimum
(e.g. the ``ex126x`` trim-loss family: discopt's relaxation caps at 19.1 while
the true optimum is 19.6). When one factor is an *integer* variable
``x_i in [lo, hi]`` it can be binary-expanded,

    x_i = lo + sum_k 2^k e_k        (e_k binary,  k = 0 .. ceil(log2(hi-lo+1))-1)

so the product becomes

    x_i * x_j = lo * x_j + sum_k 2^k (e_k * x_j),

and each ``binary x variable`` product ``e_k * x_j`` is lifted to an auxiliary
``v_k`` with its **exact big-M linearization** (``v_k <= U*e_k``,
``v_k <= x_j``, ``v_k >= x_j - U*(1-e_k)``, ``v_k >= 0``). The result is a purely
**linear** model — every bilinear term is gone, so the bilinear MINLP becomes an
equivalent pure MILP that discopt's MILP branch-and-bound solves directly. The
big-M is exact at ``e_k in {0,1}`` (no McCormick gap), so the MILP optimum equals
the true MINLP optimum.

The rewrite is value-preserving: ``x_i = lo + sum 2^k e_k`` reproduces every
integer value of ``x_i`` over ``[lo, hi]`` (combinations exceeding ``hi`` are
ruled out by ``x_i``'s own upper bound), and the product expansion is an
algebraic identity. Only the *relaxation* changes (loose -> exact). The pass is a
no-op when no integer-factor bilinear term exists, so it never regresses a model.

Note: this triggers on *declared* integer factors. Models whose factors are
implied-integer (declared continuous but forced integer by the constraints, as
in ``ex1263``) need the implied-integer detection pass to mark them integer
first; this module is that detector's downstream consumer.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    IndexExpression,
    Model,
    Variable,
    VarType,
)

from .factorable_reform import _collect_mul_factors
from .term_classifier import distribute_products

# Skip an integer factor whose range needs more than this many bits — the
# expansion adds one binary and one aux product per bit, so a huge range would
# blow up the model for no practical gain (and such a variable is better left to
# spatial branching). 12 bits covers ranges up to 4095.
_MAX_BITS = 12


def _scalar_var_ref(expr: Expression) -> Optional[tuple[Variable, int]]:
    """Return ``(variable, flat_element)`` if *expr* is a scalar variable
    reference (a scalar ``Variable`` or an ``IndexExpression`` selecting one
    scalar element), else ``None``."""
    if isinstance(expr, Variable):
        if expr.size == 1:
            return expr, 0
        return None
    if isinstance(expr, IndexExpression):
        var = expr.variable if hasattr(expr, "variable") else getattr(expr, "var", None)
        if not isinstance(var, Variable):
            return None
        idx = getattr(expr, "index", None)
        if isinstance(idx, int):
            return var, idx
        # Multi-dim or non-scalar index: not handled here.
        return None
    return None


def _int_factor_range(
    var: Variable, elem: int, implied: "frozenset[tuple[int, int]]" = frozenset()
) -> Optional[tuple[int, int]]:
    """Return integer ``(lo, hi)`` if ``var[elem]`` is integer-valued (declared
    ``INTEGER``, or *implied-integer* per the ``implied`` set) with finite bounds
    spanning a range expressible in ``_MAX_BITS`` bits, else ``None``. Binary
    variables are already exact under McCormick, so they are excluded."""
    if var.var_type != VarType.INTEGER and (var._index, elem) not in implied:
        return None
    lo = float(np.asarray(var.lb).flat[elem])
    hi = float(np.asarray(var.ub).flat[elem])
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo:
        return None
    lo_i, hi_i = int(math.floor(lo)), int(math.ceil(hi))
    span = hi_i - lo_i
    if span <= 0 or span > (1 << _MAX_BITS) - 1:
        return None
    return lo_i, hi_i


class _Expander:
    """Creates and caches the binary expansion of integer factor variables and
    accumulates the linking ``x_i == lo + sum 2^k e_k`` constraints."""

    def __init__(self, model: Model, implied=frozenset(), participation=None):
        self.model = model
        self.implied = implied
        # (var._index, elem) -> number of bilinear products the factor appears in;
        # the more-shared factor is expanded so its bits are reused (fewer vars).
        self.participation = participation or {}
        self.aux_constraints: list[Constraint] = []
        # (var._index, elem) -> (lo, [(coef, e_k Variable), ...])
        self._cache: dict[tuple[int, int], tuple[int, list[tuple[int, Variable]]]] = {}
        self._counter = 0

    def expansion(self, var: Variable, elem: int, lo: int, hi: int):
        key = (var._index, elem)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        nbits = max(1, math.ceil(math.log2(hi - lo + 1)))
        bits: list[tuple[int, Variable]] = []
        ref = var if var.size == 1 else IndexExpression(var, elem)
        # x_i - lo - sum 2^k e_k == 0
        link = ref if lo == 0 else BinaryOp("-", ref, Constant(float(lo)))
        for k in range(nbits):
            e = Variable(f"_ipx_e{self._counter}", VarType.BINARY, (), 0.0, 1.0, self.model)
            self._counter += 1
            self.model._variables.append(e)
            bits.append((1 << k, e))
            term = e if k == 0 else BinaryOp("*", Constant(float(1 << k)), e)
            link = BinaryOp("-", link, term)
        self.aux_constraints.append(Constraint(link, "==", 0.0))
        self._cache[key] = (lo, bits)
        return lo, bits

    def bigm_product(self, e: Variable, other: Expression, lo_o: float, hi_o: float) -> Variable:
        """Return an aux ``v == e * other`` (``e`` binary, ``other in [lo_o, hi_o]``)
        via the **exact** big-M linearization

            v <= hi_o * e,   v >= lo_o * e,
            v <= other - lo_o*(1-e),   v >= other - hi_o*(1-e).

        At ``e in {0,1}`` these force ``v = 0`` (e=0) or ``v = other`` (e=1), so the
        product is reproduced exactly — no McCormick gap, and no bilinear term
        remains in the model."""
        v = Variable(
            f"_ipx_v{self._counter}",
            VarType.CONTINUOUS,
            (),
            min(0.0, lo_o),
            max(0.0, hi_o),
            self.model,
        )
        self._counter += 1
        self.model._variables.append(v)
        ac = self.aux_constraints
        # discopt's LP extractor folds the RHS into the body (``body sense 0``);
        # every row below is therefore normalized to rhs == 0 with the constant
        # carried inside the body (a nonzero ``Constraint`` rhs is silently
        # dropped by ``extract_lp_data`` — see _extract_constraints_algebraic).
        # v - hi_o*e <= 0
        ac.append(Constraint(BinaryOp("-", v, BinaryOp("*", Constant(hi_o), e)), "<=", 0.0))
        # v - lo_o*e >= 0
        ac.append(Constraint(BinaryOp("-", v, BinaryOp("*", Constant(lo_o), e)), ">=", 0.0))
        # v - other - lo_o*e + lo_o <= 0   (i.e. v <= other - lo_o*(1-e))
        body_u = BinaryOp("-", BinaryOp("-", v, other), BinaryOp("*", Constant(lo_o), e))
        ac.append(Constraint(BinaryOp("+", body_u, Constant(lo_o)), "<=", 0.0))
        # v - other - hi_o*e + hi_o >= 0   (i.e. v >= other - hi_o*(1-e))
        body_l = BinaryOp("-", BinaryOp("-", v, other), BinaryOp("*", Constant(hi_o), e))
        ac.append(Constraint(BinaryOp("+", body_l, Constant(hi_o)), ">=", 0.0))
        return v


def _expand_product(lo: int, bits, other: Expression, lo_o: float, hi_o: float, exp: "_Expander"):
    """Build ``lo*other + sum_k 2^k v_k`` with ``v_k == e_k*other`` big-M-lifted —
    a purely *linear* expression (no bilinear term survives)."""
    out: Optional[Expression] = None
    if lo != 0:
        out = BinaryOp("*", Constant(float(lo)), other)
    for coef, e in bits:
        v = exp.bigm_product(e, other, lo_o, hi_o)
        term = v if coef == 1 else BinaryOp("*", Constant(float(coef)), v)
        out = term if out is None else BinaryOp("+", out, term)
    return out if out is not None else Constant(0.0)


def _try_expand_mul(node: BinaryOp, model: Model, exp: _Expander) -> Optional[Expression]:
    """If *node* is a product whose only variable factors are two distinct scalar
    variable references, one of them an integer, return the exact expansion
    (carrying any constant coefficient factors), else ``None``."""
    factors = _collect_mul_factors(node)
    const = 1.0
    var_refs: list[tuple[Expression, Variable, int]] = []
    for f in factors:
        if isinstance(f, Constant):
            const *= float(f.value)
            continue
        ref = _scalar_var_ref(f)
        if ref is None:
            return None  # a non-scalar / non-constant factor — leave to McCormick
        var_refs.append((f, ref[0], ref[1]))
    if len(var_refs) != 2:
        return None
    (e0, v0, el0), (e1, v1, el1) = var_refs
    if v0._index == v1._index and el0 == el1:
        return None  # square term (x^2), handled by the monomial lift
    # Each expanded bit adds one big-M aux per product, so expanding the
    # smaller-range factor (fewer bits) minimizes added variables; use product
    # sharing only as a tiebreaker (a cached factor avoids re-adding its bits).
    cands = []
    for (ei, vi, eli), (eo, vo, elo) in (
        ((e0, v0, el0), (e1, v1, el1)),
        ((e1, v1, el1), (e0, v0, el0)),
    ):
        rng = _int_factor_range(vi, eli, exp.implied)
        if rng is not None:
            share = exp.participation.get((vi._index, eli), 0)
            cands.append((rng[1] - rng[0], -share, (vi, eli, rng), (eo, vo, elo)))
    if not cands:
        return None
    cands.sort(key=lambda t: (t[0], t[1]))
    _, _, (vi, eli, (lo, hi)), (other_e, vo, elo) = cands[0]
    base_lo, bits = exp.expansion(vi, eli, lo, hi)
    lo_o = float(np.asarray(vo.lb).flat[elo])
    hi_o = float(np.asarray(vo.ub).flat[elo])
    expanded = _expand_product(base_lo, bits, other_e, lo_o, hi_o, exp)
    if const != 1.0:
        expanded = BinaryOp("*", Constant(const), expanded)
    return expanded


def _rewrite(expr: Expression, model: Model, exp: _Expander) -> Expression:
    """Recursively rewrite integer-factor bilinear products in *expr*."""
    if isinstance(expr, BinaryOp):
        if expr.op == "*":
            replaced = _try_expand_mul(expr, model, exp)
            if replaced is not None:
                return replaced
        left = _rewrite(expr.left, model, exp)
        right = _rewrite(expr.right, model, exp)
        if left is expr.left and right is expr.right:
            return expr
        return BinaryOp(expr.op, left, right)
    # Other node types: rebuild children generically via known attributes.
    for attr in ("operand",):
        child = getattr(expr, attr, None)
        if isinstance(child, Expression):
            new = _rewrite(child, model, exp)
            if new is not child:
                import copy

                clone = copy.copy(expr)
                setattr(clone, attr, new)
                return clone
    for attr in ("args", "terms"):
        seq = getattr(expr, attr, None)
        if isinstance(seq, (list, tuple)):
            new_seq = [_rewrite(c, model, exp) if isinstance(c, Expression) else c for c in seq]
            if any(n is not o for n, o in zip(new_seq, seq)):
                import copy

                clone = copy.copy(expr)
                setattr(clone, attr, type(seq)(new_seq))
                return clone
    return expr


def _for_each_int_bilinear(expr: Expression, implied, fn) -> None:
    """Call ``fn(int_factor_refs)`` for each integer-factor bilinear product in
    *expr*, where ``int_factor_refs`` is the list of ``(var, elem)`` factors that
    are integer-valued (one or both factors)."""
    if isinstance(expr, BinaryOp):
        if expr.op == "*":
            factors = _collect_mul_factors(expr)
            refs = [_scalar_var_ref(f) for f in factors if not isinstance(f, Constant)]
            if all(r is not None for r in refs) and len(refs) == 2:
                (v0, e0), (v1, e1) = refs  # type: ignore[misc]
                if not (v0._index == v1._index and e0 == e1):
                    ints = [
                        (vi, eli)
                        for (vi, eli) in ((v0, e0), (v1, e1))
                        if _int_factor_range(vi, eli, implied)
                    ]
                    if ints:
                        fn(ints)
        _for_each_int_bilinear(expr.left, implied, fn)
        _for_each_int_bilinear(expr.right, implied, fn)
        return
    c = getattr(expr, "operand", None)
    if isinstance(c, Expression):
        _for_each_int_bilinear(c, implied, fn)
    for attr in ("args", "terms"):
        seq = getattr(expr, attr, None)
        if isinstance(seq, (list, tuple)):
            for c in seq:
                if isinstance(c, Expression):
                    _for_each_int_bilinear(c, implied, fn)


def _bodies(model: Model):
    for c in model._constraints:
        if isinstance(c, Constraint):
            yield distribute_products(c.body)
    if model._objective is not None:
        yield distribute_products(model._objective.expression)


def has_integer_product_work(model: Model, implied=frozenset()) -> bool:
    """True if any constraint/objective has an integer-factor bilinear product
    (with integer or *implied*-integer factor) this pass can exactly linearize."""
    found = []
    try:
        for body in _bodies(model):
            _for_each_int_bilinear(body, implied, lambda ints: found.append(True))
            if found:
                return True
    except Exception:
        return False
    return False


def _participation(model: Model, implied) -> dict:
    """Count, per integer factor ``(var._index, elem)``, the number of bilinear
    products it appears in — used to pick the most-shared factor to expand."""
    counts: dict[tuple[int, int], int] = {}

    def tally(ints):
        for vi, eli in ints:
            counts[(vi._index, eli)] = counts.get((vi._index, eli), 0) + 1

    for body in _bodies(model):
        _for_each_int_bilinear(body, implied, tally)
    return counts


def expand_integer_products(model: Model, implied=frozenset()) -> Model:
    """Return a model equivalent to *model* with integer-factor bilinear products
    binary-expanded + big-M-linearized to their exact (pure-MILP) form. ``implied``
    is an optional set of ``(var._index, elem)`` treated as integer-valued in
    addition to declared integers. Returns *model* unchanged when no such product
    exists or on any unexpected error (never regresses)."""
    try:
        if not has_integer_product_work(model, implied):
            return model
        new_model = Model(model.name)
        new_model._variables = list(model._variables)
        new_model._parameters = list(model._parameters)
        new_model._objective = model._objective
        exp = _Expander(new_model, implied=implied, participation=_participation(model, implied))

        rebuilt: list[Constraint] = []
        for c in model._constraints:
            if not isinstance(c, Constraint):
                rebuilt.append(c)
                continue
            body = _rewrite(distribute_products(c.body), new_model, exp)
            rebuilt.append(c if body is c.body else Constraint(body, c.sense, c.rhs, c.name))

        if new_model._objective is not None:
            obj = new_model._objective
            new_expr = _rewrite(distribute_products(obj.expression), new_model, exp)
            if new_expr is not obj.expression:
                import copy

                new_obj = copy.copy(obj)
                new_obj.expression = new_expr
                new_model._objective = new_obj

        new_model._constraints = rebuilt + exp.aux_constraints
        return new_model
    except Exception:
        return model


def has_reformulation_work(model: Model) -> bool:
    """True if the model has any integer-factor bilinear product — counting both
    declared-integer and *implied*-integer factors — that this pass can linearize."""
    try:
        from .implied_integer import detect_implied_integers

        implied = frozenset(detect_implied_integers(model))
        return has_integer_product_work(model, implied)
    except Exception:
        return False


def reformulate_integer_bilinear(model: Model) -> Model:
    """End-to-end pass: detect implied-integer factor variables, then exactly
    linearize every integer-factor bilinear product into pure-MILP form. This is
    the entry the solver calls; it is a no-op (returns *model*) when nothing
    applies and never mutates the input model's variable types."""
    try:
        from .implied_integer import detect_implied_integers

        implied = frozenset(detect_implied_integers(model))
        return expand_integer_products(model, implied)
    except Exception:
        return model
