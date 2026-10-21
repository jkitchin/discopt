"""Ellipsoidal-uncertainty robust reformulation.

Mathematical Background
-----------------------
The uncertainty set is ``{p̄ + Σ^{1/2} u : ||u||₂ ≤ ρ}`` with ``Σ = Σ^{1/2} Σ^{1/2}ᵀ``
(equivalently ``||Σ^{-1/2}(p - p̄)||₂ ≤ ρ``). For a scalar row affine in p,

    g(x, p) = g(x, p̄) + a(x)ᵀ (p - p̄),     a(x) = ∂g/∂p,

the worst case over the ellipsoid is

    max_p g(x, p) = g(x, p̄) + ρ ||Σ^{1/2}ᵀ a(x)||₂ = g(x, p̄) + ρ sqrt(a(x)ᵀ Σ a(x)),

so the robust counterpart of ``g(x, p) ≤ 0`` is the second-order cone row
``g(x, p̄) + ρ ||Σ^{1/2}ᵀ a(x)||₂ ≤ 0``. For an objective, ``min f`` gets
``+ ρ||·||`` and ``max f`` gets ``- ρ||·||``.

``a(x)`` is the coefficient of p aggregated over the *whole* row (#1835): every
occurrence of p, with every outer scaling, sum, index, negation or matmul around
it applied. It is computed symbolically by a reverse (cotangent) sweep from the
row root to the parameter leaves, so it may depend on x (``x * p``). The former
implementation instead attached an unscaled ``||Σ^{1/2} M||`` penalty to each
``M @ p`` node it found, regardless of what surrounded it: ``x + 5 * (c @ p)``
got a penalty 5x too small and certified non-robust points, and
``x + M @ p`` (split per row) got the full-matrix Frobenius norm on every row.

:class:`~discopt.ro.counterpart.RobustCounterpart` has already split vector rows
per element and oriented every uncertain row to ``body <= rhs`` (``>=`` negated,
``==`` split into two rows), so each body handled here is a scalar worst-case
maximum. A row (or objective) that is not affine in an uncertain parameter --
a product of two uncertain terms, p in a divisor, power or function -- is
refused with ``NotImplementedError``: no SOC counterpart is exact for it.

Distinct uncertain parameters carry independent ellipsoids, so their worst cases
add: one penalty per parameter per row, exact for the product set.

Reference
---------
Ben-Tal, A., Nemirovski, A. (1999). Robust solutions of uncertain linear
programs. *Operations Research Letters*, 25(1), 1–13.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from discopt.ro.formulations._common import (
    _contains_uncertain_param,
    _shape_of,
    eval_constant_expr,
    substitute_param,
)
from discopt.ro.uncertainty import EllipsoidalUncertaintySet


class EllipsoidalRobustFormulation:
    """Apply ellipsoidal-uncertainty robust reformulation to a model.

    Every scalar row (and the objective) that contains an uncertain parameter
    p is replaced by its nominal value at p̄ plus the SOC worst-case penalty

        ρ ||Σ^{1/2}ᵀ a(x)||₂,     a(x) = ∂ row / ∂ p,

    with ``a(x)`` aggregated over every occurrence of p in the row.

    Parameters
    ----------
    model : discopt.Model
        The model to robustify (modified in-place).
    uncertainty_sets : list[EllipsoidalUncertaintySet]
        Uncertainty sets.
    prefix : str
        Name prefix for auxiliary variables / constraints.
    """

    def __init__(
        self,
        model,
        uncertainty_sets: list[EllipsoidalUncertaintySet],
        prefix: str = "ro",
    ) -> None:
        self._model = model
        self._uncertainty_sets = uncertainty_sets
        self._prefix = prefix

    def build(self) -> None:
        """Robustify the model in-place.

        Substitutes the nominal value of every uncertain parameter and adds,
        per parameter, the 2-norm penalty of the row's aggregated coefficient.
        """
        unc_map = {u.parameter.name: u for u in self._uncertainty_sets}
        m = self._model

        from discopt.modeling.core import BinaryOp, Constraint, Objective, ObjectiveSense

        # ── Robustify constraints ──────────────────────────────────────────────
        # Only plain Constraint objects carry .body / .sense / .rhs; other
        # types (_IndicatorConstraint, _DisjunctiveConstraint, _SOSConstraint,
        # _LogicalConstraint) are passed through unchanged.
        new_constraints = []
        for con in m._constraints:
            if not isinstance(con, Constraint) or not _contains_uncertain_param(
                con.body, set(unc_map)
            ):
                new_constraints.append(con)
                continue
            if con.sense != "<=":  # pragma: no cover - RobustCounterpart orients rows
                raise ValueError(
                    "Ellipsoidal robust counterpart: expected an oriented '<=' row, got "
                    f"{con.sense!r} on {con.name!r}."
                )
            where = f"constraint {con.name!r}" if con.name is not None else "a constraint"
            new_expr, penalties = _nominal_and_penalties(con.body, unc_map, where)
            new_constraints.append(
                Constraint(
                    body=_sum_expr([new_expr] + penalties),
                    sense=con.sense,
                    rhs=con.rhs,
                    name=con.name,
                )
            )
        m._constraints = new_constraints

        # ── Robustify objective ────────────────────────────────────────────────
        obj = m._objective
        if obj is None or not _contains_uncertain_param(obj.expression, set(unc_map)):
            return

        new_expr, penalties = _nominal_and_penalties(obj.expression, unc_map, "the objective")
        # For MINIMIZE: worst case → largest → add positive penalties.
        # For MAXIMIZE: worst case → smallest → subtract penalties.
        if obj.sense == ObjectiveSense.MINIMIZE:
            combined = _sum_expr([new_expr] + penalties)
        else:
            combined = new_expr
            for pen in penalties:
                combined = BinaryOp("-", combined, pen)
        m._objective = Objective(expression=combined, sense=obj.sense)


# ─────────────────────────────────────────────────────────────────────────────
# Row coefficient a(x) = ∂ row / ∂ p by a reverse (cotangent) sweep
# ─────────────────────────────────────────────────────────────────────────────


class _NotAffine(NotImplementedError):
    """The row is not affine in an uncertain parameter, or has an unsupported shape."""


class _Cot:
    """A cotangent: the coefficient a node's value carries into the row.

    Exactly one of ``const`` (a numpy array, folded eagerly) or ``expr`` (an
    expression, possibly depending on decision variables) is set; ``shape`` is
    the shape of that value, always the shape of the node it is attached to.
    """

    __slots__ = ("const", "expr", "shape")

    def __init__(self, *, const=None, expr=None, shape=None):
        self.const = None if const is None else np.asarray(const, dtype=np.float64)
        self.expr = expr
        self.shape = tuple(self.const.shape) if const is not None else tuple(shape)

    def as_expr(self):
        from discopt.modeling.core import Constant

        return Constant(self.const) if self.const is not None else self.expr


def _nominal_and_penalties(expr, unc_map: dict, where: str):
    """``(g(x, p̄), [ρ_p ||Σ_p^{1/2}ᵀ a_p(x)||  for each uncertain p in the row])``.

    Refuses (``NotImplementedError``) a row that is not affine in an uncertain
    parameter or whose coefficient cannot be formed exactly.
    """
    from discopt.modeling.core import Constant

    try:
        coeffs = _row_coefficient(expr, unc_map)
    except _NotAffine as exc:
        raise NotImplementedError(
            f"Ellipsoidal robust counterpart: {where} is not supported -- {exc} The "
            "ellipsoidal (SOC) counterpart g(x, p_bar) + rho*||Sigma^1/2' a(x)|| is exact "
            "only for a row affine in the uncertain parameter, with coefficient "
            "a(x) = d row / d p. Refusing rather than returning an approximate (possibly "
            "non-robust) model."
        ) from None

    nominal = expr
    for name, unc in unc_map.items():
        nominal = substitute_param(nominal, name, Constant(unc.parameter.value))
    # Substituting p_bar leaves variable-free subtrees such as ``Constant @ Constant``
    # (a 2-D matrix parameter: ``sum(B * (P @ A))``); fold each to its exact numpy
    # value so the solver never evaluates a 2-D matmul of constants (#1845).
    nominal = _fold_constants(nominal)[0]

    penalties = []
    for name, unc in unc_map.items():
        if name in coeffs:
            pen = _soc_penalty(coeffs[name], unc)
            if pen is not None:
                penalties.append(_fold_constants(pen)[0])
    for e in [nominal] + penalties:
        _refuse_2d_matmul(e, where)
    return nominal, penalties


def _fold_constants(expr):
    """``(expr', numeric)``: *expr* with every maximal Constant-only subtree folded.

    A subtree with no Variable and no Parameter is replaced by ``Constant`` of
    its exact numpy value (:func:`eval_constant_expr`); ``numeric`` says whether
    all of *expr* was such a subtree. A node the evaluator does not know is kept
    as it is, with its children folded.
    """
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    if isinstance(expr, Constant):
        return expr, True
    if isinstance(expr, BinaryOp):
        (lf, ln), (rf, rn) = _fold_constants(expr.left), _fold_constants(expr.right)
        new, numeric = BinaryOp(expr.op, lf, rf), ln and rn
    elif isinstance(expr, MatMulExpression):
        (lf, ln), (rf, rn) = _fold_constants(expr.left), _fold_constants(expr.right)
        new, numeric = MatMulExpression(lf, rf), ln and rn
    elif isinstance(expr, UnaryOp):
        of, numeric = _fold_constants(expr.operand)
        new = UnaryOp(expr.op, of)
    elif isinstance(expr, IndexExpression):
        bf, numeric = _fold_constants(expr.base)
        new = IndexExpression(bf, expr.index)
    elif isinstance(expr, SumExpression):
        of, numeric = _fold_constants(expr.operand)
        new = SumExpression(of, axis=expr.axis)
    elif isinstance(expr, SumOverExpression):
        parts = [_fold_constants(t) for t in expr.terms]
        new, numeric = SumOverExpression([t for t, _ in parts]), all(n for _, n in parts)
    elif isinstance(expr, FunctionCall):
        parts = [_fold_constants(a) for a in expr.args]
        new = FunctionCall(expr.func_name, *[a for a, _ in parts])
        numeric = all(n for _, n in parts)
    else:
        return expr, False
    if not numeric:
        return new, False
    try:
        return Constant(eval_constant_expr(new)), True
    except NotImplementedError:
        return new, False


def _refuse_2d_matmul(expr, where: str) -> None:
    """Refuse a counterpart that still contains a 2-D @ 2-D matmul (#1845).

    The solver mis-evaluates a matrix-matrix product inside an element-wise
    product or a sum (``sum(B * (A @ X))`` certifies a false optimum, #1845).
    After :func:`_fold_constants` such a node can only involve a decision
    variable (or a certain Parameter), e.g. the coefficient ``X @ Aᵀ`` of ``P`` in
    ``sum(X * (P @ A))``; refuse it rather than hand it to the solver.
    """
    from discopt.modeling.core import (
        BinaryOp,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    if isinstance(expr, MatMulExpression):
        sl, sr = _shape_of(expr.left), _shape_of(expr.right)
        if sl is not None and sr is not None and len(sl) == 2 and len(sr) == 2:
            raise NotImplementedError(
                f"Ellipsoidal robust counterpart: {where} is not supported -- its robust "
                f"counterpart contains the matrix-matrix product {expr!r} (shapes {sl} @ "
                f"{sr}), which the solver currently mis-evaluates "
                "(#1845: sum(B * (A @ X)) with a 2-D X certifies a false optimum). "
                "Refusing rather than returning a possibly wrong certified answer."
            )
        children = [expr.left, expr.right]
    elif isinstance(expr, BinaryOp):
        children = [expr.left, expr.right]
    elif isinstance(expr, (UnaryOp, SumExpression)):
        children = [expr.operand]
    elif isinstance(expr, IndexExpression):
        children = [expr.base]
    elif isinstance(expr, SumOverExpression):
        children = list(expr.terms)
    elif isinstance(expr, FunctionCall):
        children = list(expr.args)
    else:
        return
    for c in children:
        _refuse_2d_matmul(c, where)


def _row_coefficient(expr, unc_map: dict) -> dict[str, _Cot]:
    """``{name: a_name}`` with ``a_name = ∂ expr / ∂ p_name``, shaped like ``p_name``.

    *expr* must be a single scalar (size-1) value. Each occurrence of a
    parameter contributes the cotangent reaching its leaf; the contributions are
    summed, so ``p @ x + 2 * (p @ x)`` has coefficient ``3 x``.
    """
    from discopt.modeling.core import UnaryOp

    shape = _shape_of(expr)
    if shape is None:
        raise _NotAffine("its shape is not statically known.")
    if int(np.prod(shape)) != 1:
        raise _NotAffine(f"it is vector-valued (shape {shape}), not one scalar row.")

    names = set(unc_map)
    acc: dict[str, list[_Cot]] = {}
    _push(expr, _Cot(const=np.ones(shape)), names, acc)

    out: dict[str, _Cot] = {}
    for name, parts in acc.items():
        pshape = tuple(np.shape(unc_map[name].parameter.value))
        consts = [c.const for c in parts if c.const is not None]
        exprs = [c.expr for c in parts if c.const is None]
        if not exprs:
            out[name] = _Cot(const=np.sum(consts, axis=0).reshape(pshape))
            continue
        if (
            len(exprs) == 1
            and not consts
            and isinstance(exprs[0], UnaryOp)
            and exprs[0].op == "neg"
        ):
            # ||Σ^{1/2}ᵀ(-a)|| == ||Σ^{1/2}ᵀ a||: drop a lone sign, so ``-(p @ x)``
            # gets the same ``||Σ^{1/2}ᵀ x||`` node as ``p @ x``.
            exprs = [exprs[0].operand]
        terms = list(exprs)
        if consts:
            terms.append(_Cot(const=np.sum(consts, axis=0)).as_expr())
        out[name] = _Cot(expr=_sum_expr(terms), shape=pshape)
    return out


def _shape(expr) -> tuple[int, ...]:
    shape = _shape_of(expr)
    if shape is None:
        raise _NotAffine(f"the shape of {expr!r} is not statically known.")
    return shape


def _is_numeric(expr) -> bool:
    """True when *expr* is built from Constants only (no Variable, no Parameter)."""
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    if isinstance(expr, Constant):
        return True
    if isinstance(expr, (BinaryOp, MatMulExpression)):
        return _is_numeric(expr.left) and _is_numeric(expr.right)
    if isinstance(expr, (UnaryOp,)):
        return _is_numeric(expr.operand)
    if isinstance(expr, (IndexExpression,)):
        return _is_numeric(expr.base)
    if isinstance(expr, SumExpression):
        return _is_numeric(expr.operand)
    if isinstance(expr, SumOverExpression):
        return all(_is_numeric(t) for t in expr.terms)
    if isinstance(expr, FunctionCall):
        return all(_is_numeric(a) for a in expr.args)
    return False


def _numeric_value(expr) -> Optional[np.ndarray]:
    """The value of a Constant-only expression, else ``None``."""
    if not _is_numeric(expr):
        return None
    try:
        return np.asarray(eval_constant_expr(expr), dtype=np.float64)
    except NotImplementedError:
        return None


def _neg(w: _Cot) -> _Cot:
    from discopt.modeling.core import UnaryOp

    if w.const is not None:
        return _Cot(const=-w.const)
    if isinstance(w.expr, UnaryOp) and w.expr.op == "neg":
        return _Cot(expr=w.expr.operand, shape=w.shape)
    return _Cot(expr=UnaryOp("neg", w.expr), shape=w.shape)


def _mul(w: _Cot, f) -> _Cot:
    """Cotangent ``w * f`` for a parameter-free factor *f* (element-wise)."""
    from discopt.modeling.core import BinaryOp, UnaryOp

    fshape = _shape(f)
    out_shape = tuple(int(d) for d in np.broadcast_shapes(w.shape, fshape))
    fv = _numeric_value(f)
    if w.const is not None:
        if fv is not None:
            return _Cot(const=w.const * fv)
        if out_shape == fshape and np.all(w.const == 1.0):
            return _Cot(expr=f, shape=fshape)
        if out_shape == fshape and np.all(w.const == -1.0):
            return _Cot(expr=UnaryOp("neg", f), shape=fshape)
    return _Cot(expr=BinaryOp("*", w.as_expr(), f), shape=out_shape)


def _div(w: _Cot, d) -> _Cot:
    """Cotangent ``w / d`` for a parameter-free divisor *d*."""
    from discopt.modeling.core import BinaryOp

    dshape = _shape(d)
    out_shape = tuple(int(s) for s in np.broadcast_shapes(w.shape, dshape))
    dv = _numeric_value(d)
    if w.const is not None and dv is not None:
        return _Cot(const=w.const / dv)
    return _Cot(expr=BinaryOp("/", w.as_expr(), d), shape=out_shape)


def _fit(w: _Cot, target: tuple[int, ...]) -> _Cot:
    """Bring a cotangent to the shape of the child it flows into.

    Up: a child the parent broadcast receives the summed cotangent (reduction
    over the broadcast axes). Down: a cotangent narrower than its node (from a
    scalar seed) is broadcast up to it.
    """
    from discopt.modeling.core import BinaryOp, Constant, SumExpression

    target = tuple(target)
    if w.shape == target:
        return w
    if w.const is not None:
        v = w.const
        try:
            if tuple(np.broadcast_shapes(v.shape, target)) == target:
                return _Cot(const=np.broadcast_to(v, target).copy())
        except ValueError:
            pass
        while v.ndim > len(target):
            v = v.sum(axis=0)
        for ax, d in enumerate(target):
            if d == 1 and v.shape[ax] != 1:
                v = v.sum(axis=ax, keepdims=True)
        if tuple(v.shape) != target:
            raise _NotAffine(f"cannot fit a coefficient of shape {w.shape} to {target}.")
        return _Cot(const=v)
    try:
        up = tuple(np.broadcast_shapes(w.shape, target)) == target
    except ValueError:
        up = False
    if up:
        return _Cot(expr=BinaryOp("*", w.expr, Constant(np.ones(target))), shape=target)
    if target == ():
        return _Cot(expr=SumExpression(w.expr, axis=None), shape=())
    if len(w.shape) == len(target) + 1 and w.shape[1:] == target:
        return _Cot(expr=SumExpression(w.expr, axis=0), shape=target)
    raise _NotAffine(f"cannot fit a coefficient of shape {w.shape} to {target}.")


def _push(expr, w: _Cot, names: set[str], acc: dict) -> None:
    """Propagate the cotangent *w* of *expr* down to the uncertain parameter leaves."""
    from discopt.modeling.core import (
        BinaryOp,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    if not _contains_uncertain_param(expr, names):
        return

    if isinstance(expr, Parameter):
        acc.setdefault(expr.name, []).append(_fit(w, _shape(expr)))
        return

    if isinstance(expr, BinaryOp):
        lhas = _contains_uncertain_param(expr.left, names)
        rhas = _contains_uncertain_param(expr.right, names)
        if expr.op in ("+", "-"):
            if lhas:
                _push(expr.left, _fit(w, _shape(expr.left)), names, acc)
            if rhas:
                wr = w if expr.op == "+" else _neg(w)
                _push(expr.right, _fit(wr, _shape(expr.right)), names, acc)
            return
        if expr.op == "*":
            if lhas and rhas:
                raise _NotAffine(
                    f"{expr!r} multiplies two terms that both contain an uncertain parameter."
                )
            e, f = (expr.left, expr.right) if lhas else (expr.right, expr.left)
            _push(e, _fit(_mul(w, f), _shape(e)), names, acc)
            return
        if expr.op == "/" and not rhas:
            _push(expr.left, _fit(_div(w, expr.right), _shape(expr.left)), names, acc)
            return
        raise _NotAffine(f"an uncertain parameter enters {expr!r} non-affinely.")

    if isinstance(expr, UnaryOp):
        if expr.op == "neg":
            _push(expr.operand, _neg(w), names, acc)
            return
        raise _NotAffine(f"an uncertain parameter enters {expr.op}(...) non-affinely.")

    if isinstance(expr, MatMulExpression):
        _push_matmul(expr, w, names, acc)
        return

    if isinstance(expr, IndexExpression):
        _push(expr.base, _scatter(w, _shape(expr.base), expr.index), names, acc)
        return

    if isinstance(expr, SumExpression):
        so = _shape(expr.operand)
        axis = expr.axis
        if axis is None or len(so) <= 1:
            _push(expr.operand, _fit(w, so), names, acc)
            return
        axis = axis % len(so)
        if w.const is not None:
            v = np.broadcast_to(np.expand_dims(w.const, axis), so).copy()
            _push(expr.operand, _Cot(const=v), names, acc)
            return
        if axis == 0:
            _push(expr.operand, _fit(w, so), names, acc)
            return
        raise _NotAffine(
            f"sum over axis {expr.axis} of a shape-{so} term with a variable coefficient."
        )

    if isinstance(expr, SumOverExpression):
        for t in expr.terms:
            if _contains_uncertain_param(t, names):
                _push(t, _fit(w, _shape(t)), names, acc)
        return

    raise _NotAffine(f"an uncertain parameter enters {type(expr).__name__} {expr!r}.")


def _push_matmul(expr, w: _Cot, names: set[str], acc: dict) -> None:
    """Cotangent of the uncertain side of ``L @ R`` (1-D/2-D operands)."""
    from discopt.modeling.core import BinaryOp, Constant, MatMulExpression

    lhas = _contains_uncertain_param(expr.left, names)
    rhas = _contains_uncertain_param(expr.right, names)
    if lhas and rhas:
        raise _NotAffine(f"{expr!r} multiplies two terms that both contain an uncertain parameter.")
    sl, sr = _shape(expr.left), _shape(expr.right)
    if not (1 <= len(sl) <= 2 and 1 <= len(sr) <= 2):
        raise _NotAffine(f"matmul of shapes {sl} @ {sr} is not supported.")
    e, f = (expr.right, expr.left) if rhas else (expr.left, expr.right)
    se, sf = (sr, sl) if rhas else (sl, sr)
    fv = _numeric_value(f)

    if len(sl) == 1 and len(sr) == 1:
        # Dot product: scalar w times the other vector.
        cot = _mul(w, f)
    elif len(se) == 1 and len(sf) == 2:
        # v = f @ e (e on the right, f (m,k)) or v = e @ f (e on the left, f (k,n)).
        if w.const is not None and fv is not None:
            cot = _Cot(const=(w.const @ fv) if rhas else (fv @ w.const))
        else:
            mm = MatMulExpression(w.as_expr(), f) if rhas else MatMulExpression(f, w.as_expr())
            cot = _Cot(expr=mm, shape=se)
    elif len(se) == 2 and len(sf) == 1:
        # Outer product: e (m,k) @ f (k,) -> w ⊗ f;  f (k,) @ e (k,n) -> f ⊗ w.
        if w.const is not None and fv is not None:
            cot = _Cot(const=np.outer(fv, w.const) if rhas else np.outer(w.const, fv))
        elif rhas and fv is not None:
            cot = _Cot(expr=BinaryOp("*", Constant(fv[:, None]), w.expr), shape=se)
        elif not rhas and w.const is not None:
            cot = _Cot(expr=BinaryOp("*", Constant(w.const[:, None]), f), shape=se)
        else:
            raise _NotAffine(f"{expr!r}: outer-product coefficient of two variable vectors.")
    else:
        # Both 2-D: e (m,k) @ f (k,n) -> w @ fᵀ;  f (m,k) @ e (k,n) -> fᵀ @ w.
        if fv is None:
            raise _NotAffine(f"{expr!r}: matrix-matrix product with a variable factor.")
        if w.const is not None:
            cot = _Cot(const=(fv.T @ w.const) if rhas else (w.const @ fv.T))
        else:
            mm = (
                MatMulExpression(Constant(fv.T), w.expr)
                if rhas
                else MatMulExpression(w.expr, Constant(fv.T))
            )
            cot = _Cot(expr=mm, shape=se)
    _push(e, _fit(cot, se), names, acc)


def _scatter(w: _Cot, base_shape: tuple[int, ...], index) -> _Cot:
    """Cotangent of the base of ``base[index]``: *w* scattered back to its positions."""
    from discopt.modeling.core import BinaryOp, Constant, MatMulExpression

    if w.const is not None:
        z = np.zeros(base_shape)
        np.add.at(z, index, w.const)
        return _Cot(const=z)
    n = int(np.prod(base_shape))
    ids = np.asarray(np.arange(n).reshape(base_shape)[index])
    if ids.ndim == 0:
        onehot = np.zeros(n)
        onehot[int(ids)] = 1.0
        return _Cot(
            expr=BinaryOp("*", w.expr, Constant(onehot.reshape(base_shape))), shape=base_shape
        )
    if ids.ndim == 1 and len(base_shape) == 1:
        sel = np.zeros((ids.size, n))
        sel[np.arange(ids.size), ids] = 1.0
        return _Cot(expr=MatMulExpression(w.expr, Constant(sel)), shape=base_shape)
    raise _NotAffine(
        f"indexing a shape-{base_shape} term with a variable coefficient of shape {w.shape}."
    )


def _soc_penalty(a: _Cot, unc: EllipsoidalUncertaintySet):
    """``ρ ||Σ^{1/2}ᵀ a||₂ = ρ sqrt(aᵀ Σ a)`` for the row coefficient *a* of ``unc``.

    ``None`` for an identically zero constant coefficient. ``Σ^{1/2}`` is the
    factor with ``Σ = Σ^{1/2} Σ^{1/2}ᵀ`` (Cholesky), so the norm is of its
    *transpose* times ``a``; the former ``||Σ^{1/2} a||`` equals it only when
    ``Σ`` is diagonal.
    """
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        Expression,
        FunctionCall,
        MatMulExpression,
        SumExpression,
    )

    pshape = tuple(np.shape(unc.parameter.value))
    S_T = np.asarray(unc.Sigma_sqrt, dtype=np.float64).T
    rho = Constant(np.array(unc.rho))
    if a.const is not None:
        st_a = S_T @ a.const.reshape(-1)
        val = unc.rho * float(np.sqrt(st_a @ st_a))
        return None if val == 0.0 else Constant(np.array(val))

    scaled: Expression
    if len(pshape) == 0:
        scaled = BinaryOp("*", Constant(np.array(S_T[0, 0])), a.expr)
    elif len(pshape) == 1:
        scaled = MatMulExpression(Constant(S_T), a.expr)
    elif np.count_nonzero(S_T - np.diag(np.diag(S_T))) == 0:
        scaled = BinaryOp("*", Constant(np.diag(S_T).reshape(pshape)), a.expr)
    else:
        # n-D parameter, correlated Σ: (Σ^{1/2}ᵀ vec(a))_r = sum(T_r * a), T_r = row r reshaped.
        sq = [
            SumExpression(BinaryOp("*", Constant(S_T[r].reshape(pshape)), a.expr), axis=None)
            for r in range(S_T.shape[0])
        ]
        inner = _sum_expr([BinaryOp("*", t, t) for t in sq])
        return BinaryOp("*", rho, FunctionCall("sqrt", inner))
    # ||scaled||₂ = sqrt(sum(scaled * scaled)); BinaryOp("*") is element-wise.
    inner = SumExpression(BinaryOp("*", scaled, scaled), axis=None)
    return BinaryOp("*", rho, FunctionCall("sqrt", inner))


def _sum_expr(exprs: list):
    """Sum a list of expressions."""
    from discopt.modeling.core import BinaryOp

    result = exprs[0]
    for e in exprs[1:]:
        result = BinaryOp("+", result, e)
    return result
