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

from typing import Union

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

        # An earlier pass may have read a worst-case direction from the sign of a
        # parameter this pass makes uncertain; that direction stays a worst case
        # only if the sign holds over this pass's set (#1839 review).
        from discopt.ro.formulations._common import (
            check_sign_assumptions,
            uncertainty_bounding_box,
        )

        check_sign_assumptions(
            self._model,
            ranged={u.parameter.name: uncertainty_bounding_box(u) for u in self._uncertainty_sets},
        )
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

    Rows whose body is scalar, and rows with no uncertain parameter, pass
    through unchanged. A body shaped by a matmul or a reduction is split too
    (#1831): ``x + M @ p <= b`` used to pass through whole, and the polyhedral
    counterpart gave its rows one set of duals and reported a feasible model
    infeasible. A body whose shape is not statically known is split by its
    *evaluated* shape, and refused when that cannot be computed (#1833) -- it
    used to pass through whole, with the same false infeasibility.
    """
    from discopt.modeling.core import Constraint, IndexExpression
    from discopt.ro.formulations._common import _contains_uncertain_param, _shape_of

    out = []
    for con in constraints:
        if not isinstance(con, Constraint) or not _contains_uncertain_param(con.body, names):
            out.append(con)
            continue
        shape = _shape_of(con.body)
        if shape is None:
            # #1833: a row whose shape is not statically known used to pass
            # through whole, and a vector row then got ONE set of polyhedral
            # duals (a false infeasibility) -- split by the evaluated shape.
            shape = _evaluated_shape(con)
        if int(np.prod(shape)) <= 1:
            out.append(con)
            continue
        rhs = np.asarray(con.rhs, dtype=np.float64)
        for idx in np.ndindex(*shape):
            index = idx[0] if len(idx) == 1 else idx
            elem_rhs = float(rhs[idx]) if rhs.ndim > 0 else float(rhs)
            name = f"{con.name}[{','.join(map(str, idx))}]" if con.name is not None else None
            out.append(
                Constraint(
                    body=IndexExpression(con.body, index),
                    sense=con.sense,
                    rhs=elem_rhs,
                    name=name,
                )
            )
    return out


def _evaluated_shape(con) -> tuple[int, ...]:
    """Shape of ``con.body`` from a numeric evaluation, or ``NotImplementedError``.

    Every variable is replaced by ones of its shape and every parameter by its
    value; only the shape of the result is used. A body the numeric evaluator
    does not model is refused: a vector row robustified whole is wrong (#1833).
    """
    from discopt.ro.formulations._common import eval_constant_expr, substitute_variables

    try:
        with np.errstate(all="ignore"):
            value = eval_constant_expr(substitute_variables(con.body, lambda v: np.ones(v.shape)))
    except NotImplementedError as exc:
        raise NotImplementedError(
            f"RobustCounterpart: the shape of uncertain row {con.name!r} cannot be "
            "determined, so it cannot be split into scalar rows; robustifying a "
            "vector row whole gives every element one worst case and is wrong for "
            "a coupled uncertainty set. Index the row into scalar rows. Refusing "
            "(#1833)."
        ) from exc
    return tuple(int(d) for d in np.shape(value))
