"""RobustCounterpart: main entry point for robust optimization.

This module provides the :class:`RobustCounterpart` class, which takes a
nominal discopt model with uncertain parameters and rewrites it into an
equivalent deterministic model whose solution is feasible (and optimal in
the minimax sense) for *every* realization of the uncertainty within the
specified set.

Supported uncertainty sets and the corresponding reformulation strategies:

+----------------------------+------------------------------------------+
| Uncertainty set            | Reformulation                            |
+============================+==========================================+
| :class:`BoxUncertaintySet` | Component-wise worst-case substitution   |
|                            | (1-norm penalty for objective terms)     |
+----------------------------+------------------------------------------+
| :class:`EllipsoidalUncert  | 2-norm (SOCP) penalty for objective /    |
| aintySet`                  | constraint uncertainty                   |
+----------------------------+------------------------------------------+
| :class:`PolyhedralUncertai | LP-dual auxiliary variables per          |
| ntySet`                    | uncertain constraint                     |
+----------------------------+------------------------------------------+

The class follows the same builder pattern as
:class:`~discopt.ml.formulations.base.NNFormulation`: construct, then call
:meth:`formulate` to mutate the model.

Example
-------
>>> import discopt.modeling as dm
>>> from discopt.ro import BoxUncertaintySet, RobustCounterpart
>>>
>>> m = dm.Model("production")
>>> x = m.continuous("x", shape=(3,), lb=0)
>>> cost = m.parameter("cost", value=[10.0, 15.0, 8.0])
>>> demand = m.parameter("demand", value=100.0)
>>>
>>> m.minimize(dm.sum(cost * x))
>>> m.subject_to(dm.sum(x) >= demand, name="meet_demand")
>>> m.subject_to(x[0] + 2 * x[1] <= 80, name="resource")
>>>
>>> # Declare cost uncertain: each component ± 10 %
>>> unc_cost = BoxUncertaintySet(cost, delta=0.10 * cost.value)
>>> # Declare demand uncertain: ± 5 %
>>> unc_demand = BoxUncertaintySet(demand, delta=0.05 * demand.value)
>>>
>>> rc = RobustCounterpart(m, [unc_cost, unc_demand])
>>> rc.formulate()   # rewrites m in-place
>>> result = m.solve()
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from discopt.ro.uncertainty import (
    BoxUncertaintySet,
    EllipsoidalUncertaintySet,
    PolyhedralUncertaintySet,
    UncertaintySet,
)

AnyUncertaintySet = Union[BoxUncertaintySet, EllipsoidalUncertaintySet, PolyhedralUncertaintySet]


class RobustCounterpart:
    """Convert a nominal model into its deterministic robust counterpart.

    Parameters
    ----------
    model : discopt.Model
        The nominal model.  The model is modified **in-place** by
        :meth:`formulate`.
    uncertainty_sets : UncertaintySet or list[UncertaintySet]
        One or more uncertainty sets.  All sets in a single call must be of
        the *same type* (i.e., all box, all ellipsoidal, or all polyhedral).
        Mixed uncertainty types require separate :class:`RobustCounterpart`
        instances applied sequentially.
    prefix : str
        Name prefix for any auxiliary variables / constraints introduced by
        the reformulation.

    Raises
    ------
    ValueError
        If the uncertainty sets are of mixed types or an unsupported type.
    RuntimeError
        If :meth:`formulate` is called more than once.

    Examples
    --------
    Box uncertainty on cost parameters::

        unc = BoxUncertaintySet(cost, delta=0.1 * cost.value)
        rc = RobustCounterpart(m, unc)
        rc.formulate()

    Ellipsoidal uncertainty on return vector::

        unc = EllipsoidalUncertaintySet(returns, rho=2.0)
        rc = RobustCounterpart(m, unc)
        rc.formulate()

    Multiple uncertain parameters (same uncertainty type)::

        rc = RobustCounterpart(m, [unc_cost, unc_demand])
        rc.formulate()

    Two-stage adjustable robust optimization (apply ADR first)::

        from discopt.ro import AffineDecisionRule, BoxUncertaintySet, RobustCounterpart

        adr = AffineDecisionRule(y, uncertain_params=xi)
        adr.apply()   # substitutes y -> y0 + Y0*xi; model still contains xi

        rc = RobustCounterpart(m, BoxUncertaintySet(xi, delta=0.1))
        rc.formulate()  # eliminates xi with worst-case substitution
    """

    def __init__(
        self,
        model,
        uncertainty_sets: Union[AnyUncertaintySet, list[AnyUncertaintySet]],
        prefix: str = "ro",
    ) -> None:
        if isinstance(uncertainty_sets, UncertaintySet):
            uncertainty_sets = [uncertainty_sets]

        if not uncertainty_sets:
            raise ValueError("uncertainty_sets must not be empty")

        # Validate uniform type.
        kinds = {u.kind for u in uncertainty_sets}
        if len(kinds) > 1:
            raise ValueError(
                f"All uncertainty sets must be of the same type; got {kinds}. "
                "Apply RobustCounterpart twice for mixed uncertainty."
            )

        self._model = model
        self._uncertainty_sets = list(uncertainty_sets)
        self._prefix = prefix
        self._formulated = False

    @property
    def kind(self) -> str:
        """The uncertainty set type: ``'box'``, ``'ellipsoidal'``, or ``'polyhedral'``."""
        return self._uncertainty_sets[0].kind

    def formulate(self) -> None:
        """Rewrite the model as its deterministic robust counterpart.

        This method can only be called once per :class:`RobustCounterpart`
        instance.  It modifies the underlying model in-place.
        """
        if self._formulated:
            raise RuntimeError("formulate() has already been called")

        self._refuse_opaque_uncertain_calls()
        self._orient_uncertain_rows()
        strategy = self._build_strategy()
        strategy.build()

        # Universal soundness guard (RO-2): a correct counterpart eliminates every
        # uncertain parameter. If any survived, the pattern was silently left at
        # nominal — refuse loudly rather than return a non-robust model.
        from discopt.ro.formulations._common import assert_no_uncertain_params_remain

        assert_no_uncertain_params_remain(
            self._model,
            {u.parameter.name for u in self._uncertainty_sets},
            kind=self.kind,
        )
        self._formulated = True

    def _refuse_opaque_uncertain_calls(self) -> None:
        """Refuse an uncertain parameter passed into an opaque ``dm.custom`` call.

        No counterpart can see inside a user function, so it can neither take
        the worst case of ``f(p)`` nor substitute ``p`` in it; before #1833 such
        a row was solved at the nominal ``p``. Refuse every set type up front,
        for constraints and objective alike, with the reason rather than a
        downstream shape or affinity error.
        """
        from discopt.modeling.core import Constraint

        names = {u.parameter.name for u in self._uncertainty_sets}
        m = self._model
        sites = [(c.name, c.body) for c in m._constraints if isinstance(c, Constraint)]
        if m._objective is not None:
            sites.append(("<objective>", m._objective.expression))
        for where, expr in sites:
            call = _opaque_uncertain_call(expr, names)
            if call is not None:
                raise NotImplementedError(
                    f"RobustCounterpart: uncertain parameter(s) {sorted(names)} are "
                    f"passed into the opaque custom function {call!r} in {where!r}. "
                    "A robust counterpart cannot take the worst case over a function "
                    "it cannot see into; write the dependence on the uncertain "
                    "parameter algebraically."
                )

    def _orient_uncertain_rows(self) -> None:
        """Rewrite every uncertain row as ``body <= rhs`` before robustifying.

        Every formulation robustifies a constraint body by its *worst-case
        maximum*, which is the counterpart of ``body <= rhs`` only. An
        uncertain ``body >= rhs`` row must instead hold at the worst-case
        minimum, and an uncertain equality ``body(x, xi) == rhs`` must hold for
        *every* ``xi`` -- i.e. both ``max_xi body <= rhs`` and
        ``min_xi body >= rhs``. Applying the one-sided maximum to such rows
        produced counterparts that are not robust (#1611 X-30a). So each
        uncertain ``>=`` row is negated into ``<=`` form, and each uncertain
        equality is split into the two one-sided rows ``body <= rhs`` and
        ``-body <= -rhs``; the formulations then see only ``<=`` rows. Rows that
        contain no uncertain parameter are left untouched.
        """
        from discopt.modeling.core import BinaryOp, Constant, Constraint
        from discopt.ro.formulations._common import _contains_uncertain_param

        names = {u.parameter.name for u in self._uncertainty_sets}
        m = self._model
        oriented = []
        for con in _split_vector_uncertain_rows(m._constraints, names):
            if (
                not isinstance(con, Constraint)
                or con.sense == "<="
                or not _contains_uncertain_param(con.body, names)
            ):
                oriented.append(con)
                continue
            neg_body = BinaryOp("*", Constant(np.array(-1.0)), con.body)
            neg_rhs = 0.0 - float(con.rhs)
            if con.sense == ">=":
                oriented.append(Constraint(body=neg_body, sense="<=", rhs=neg_rhs, name=con.name))
            elif con.sense == "==":
                base = con.name
                le_name = f"{base}_ro_le" if base is not None else None
                ge_name = f"{base}_ro_ge" if base is not None else None
                oriented.append(Constraint(body=con.body, sense="<=", rhs=con.rhs, name=le_name))
                oriented.append(Constraint(body=neg_body, sense="<=", rhs=neg_rhs, name=ge_name))
            else:
                raise ValueError(
                    f"RobustCounterpart: unsupported constraint sense {con.sense!r} "
                    f"on uncertain row {con.name!r}"
                )
        m._constraints = oriented

    def _build_strategy(self):
        if self.kind == "box":
            from discopt.ro.formulations.box import BoxRobustFormulation

            return BoxRobustFormulation(
                self._model,
                self._uncertainty_sets,  # type: ignore[arg-type]
                self._prefix,
            )
        if self.kind == "ellipsoidal":
            from discopt.ro.formulations.ellipsoidal import EllipsoidalRobustFormulation

            return EllipsoidalRobustFormulation(
                self._model,
                self._uncertainty_sets,  # type: ignore[arg-type]
                self._prefix,
            )
        if self.kind == "polyhedral":
            from discopt.ro.formulations.polyhedral import PolyhedralRobustFormulation

            return PolyhedralRobustFormulation(
                self._model,
                self._uncertainty_sets,  # type: ignore[arg-type]
                self._prefix,
            )
        raise ValueError(f"Unsupported uncertainty set kind: {self.kind!r}")


def _opaque_uncertain_call(expr, names: set[str]):
    """Return the first ``CustomCall`` in *expr* whose arguments hold an uncertain param."""
    from discopt.modeling.core import (
        BinaryOp,
        CustomCall,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )
    from discopt.ro.formulations._common import _contains_uncertain_param

    if isinstance(expr, CustomCall):
        if any(_contains_uncertain_param(a, names) for a in expr.args):
            return expr
        children = list(expr.args)
    elif isinstance(expr, (BinaryOp, MatMulExpression)):
        children = [expr.left, expr.right]
    elif isinstance(expr, (UnaryOp, SumExpression)):
        children = [expr.operand]
    elif isinstance(expr, IndexExpression):
        children = [expr.base]
    elif isinstance(expr, FunctionCall):
        children = list(expr.args)
    elif isinstance(expr, SumOverExpression):
        children = list(expr.terms)
    else:
        return None
    for child in children:
        found = _opaque_uncertain_call(child, names)
        if found is not None:
            return found
    return None


def _split_vector_uncertain_rows(constraints, names: set[str]) -> list:
    """Expand every vector-valued uncertain row into one scalar row per element.

    A vector constraint ``body(x, xi) <= 0`` of shape ``s`` is the conjunction
    of its ``prod(s)`` scalar rows, and each one must hold for every ``xi``
    *independently*: the adversary picks a different worst case per row. The
    formulations robustify one constraint body at a time, so a vector body was
    treated as a single row -- the polyhedral (budget) counterpart then gave
    the whole vector ONE set of dual multipliers, forcing ``A^T lam`` to equal
    every element's coefficient vector at once. That is exact only when all
    elements share a coefficient vector, and otherwise over-conservative: the
    bound rows ``y0 + Y xi <= ub`` that :class:`AffineDecisionRule` adds for a
    1-D recourse variable pinned every policy column ``Y_j`` to a common
    value, so the affine rule could not adapt and returned the static value
    for every budget (#1611 X-30c). Splitting is exact for every set kind.

    The row's shape is its *evaluated* shape (:func:`_evaluated_shape`), the
    broadcast of the body with the right-hand side -- not a static inference.
    The static one returned ``None`` for ``exp(M @ p)`` and every other node
    built over a matmul that it did not special-case, and ``None`` was read as
    "scalar": the row passed through whole and the polyhedral counterpart
    reported a feasible model infeasible (#1831, #1833). An uncertain row whose
    shape cannot be evaluated is refused, never assumed scalar. Rows with no
    uncertain parameter pass through unchanged.
    """
    from discopt.modeling.core import Constraint, IndexExpression
    from discopt.ro.formulations._common import _contains_uncertain_param

    out = []
    for con in constraints:
        if not isinstance(con, Constraint) or not _contains_uncertain_param(con.body, names):
            out.append(con)
            continue
        body_shape = _evaluated_shape(con.body)
        if body_shape is None:
            raise NotImplementedError(
                f"RobustCounterpart: cannot determine the shape of the uncertain row "
                f"{con.name!r} ({con.body!r}), so it cannot be split into the scalar "
                "rows whose worst cases are taken independently. Refusing rather than "
                "robustifying it as one row. Index it into scalar rows explicitly (#1833)."
            )
        rhs = np.asarray(con.rhs, dtype=np.float64)
        shape = tuple(int(d) for d in np.broadcast_shapes(body_shape, rhs.shape))
        if int(np.prod(shape)) <= 1:
            out.append(con)
            continue
        rhs_full = np.broadcast_to(rhs, shape)
        offset = len(shape) - len(body_shape)
        for idx in np.ndindex(*shape):
            # The element of the body that broadcasts to ``idx``: trailing axes
            # align, and a length-1 body axis is stretched.
            body_idx = tuple(
                0 if body_shape[d] == 1 else idx[d + offset] for d in range(len(body_shape))
            )
            if not body_idx:
                elem_body = con.body
            else:
                elem_body = IndexExpression(
                    con.body, body_idx[0] if len(body_idx) == 1 else body_idx
                )
            name = f"{con.name}[{','.join(map(str, idx))}]" if con.name is not None else None
            out.append(
                Constraint(
                    body=elem_body,
                    sense=con.sense,
                    rhs=float(rhs_full[idx]),
                    name=name,
                )
            )
    return out


class _ShapeUnknown(Exception):
    """Raised inside :func:`_evaluated_shape` for a node it cannot evaluate."""


def _evaluated_shape(expr) -> Optional[tuple[int, ...]]:
    """The numpy shape *expr* evaluates to, or ``None`` when it cannot be evaluated.

    Every leaf is replaced by zeros of its declared shape and the tree is
    evaluated with numpy's own semantics (broadcasting, ``@``, ``axis`` sums,
    indexing), so the result is the shape the solver's evaluators produce,
    whatever composition the body is built from -- including the ones a static
    shape inference does not special-case (an element-wise function over a
    matmul, an indexed sum of vector terms). A node with no numpy rule here (a
    :class:`CustomCall`, an unknown function name) or an operation numpy rejects
    gives ``None``; the caller must refuse on it, never read it as a scalar.
    """
    try:
        with np.errstate(all="ignore"):
            return tuple(int(d) for d in np.shape(_eval_zeros(expr)))
    except _ShapeUnknown:
        return None


def _eval_zeros(expr) -> np.ndarray:
    from discopt.modeling.core import (
        _ELEMENTWISE_FUNCS,
        BinaryOp,
        Constant,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
        Variable,
    )

    try:
        if isinstance(expr, (Constant, Parameter)):
            return np.zeros(np.shape(expr.value))
        if isinstance(expr, Variable):
            return np.zeros(expr.shape)
        if isinstance(expr, BinaryOp):
            lv, rv = _eval_zeros(expr.left), _eval_zeros(expr.right)
            if expr.op in ("+", "-", "*", "/", "**"):
                return np.zeros(np.broadcast_shapes(lv.shape, rv.shape))
        elif isinstance(expr, UnaryOp):
            if expr.op in ("neg", "abs"):
                return _eval_zeros(expr.operand)
        elif isinstance(expr, MatMulExpression):
            return np.asarray(np.matmul(_eval_zeros(expr.left), _eval_zeros(expr.right)))
        elif isinstance(expr, SumExpression):
            return np.asarray(np.sum(_eval_zeros(expr.operand), axis=expr.axis))
        elif isinstance(expr, SumOverExpression):
            shapes = [_eval_zeros(t).shape for t in expr.terms]
            return np.zeros(np.broadcast_shapes(*shapes))
        elif isinstance(expr, IndexExpression):
            return np.asarray(_eval_zeros(expr.base)[expr.index])
        elif isinstance(expr, FunctionCall):
            args = [_eval_zeros(a) for a in expr.args]
            if len(args) == 1 and (expr.func_name == "prod" or expr.func_name.startswith("norm")):
                return np.zeros(())
            if expr.func_name in _ELEMENTWISE_FUNCS and args:
                return np.zeros(np.broadcast_shapes(*(a.shape for a in args)))
    except (ValueError, IndexError, TypeError) as exc:
        # numpy refused the operation (a shape clash, an out-of-range index):
        # there is no shape to report.
        raise _ShapeUnknown(str(exc)) from exc
    raise _ShapeUnknown(f"no shape rule for {type(expr).__name__} {expr!r}")
