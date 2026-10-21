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

        from discopt.ro.formulations._common import (
            assert_no_uncertain_params_remain,
            merge_commitment,
            recording_commitments,
        )

        names = {u.parameter.name for u in self._uncertainty_sets}
        self._check_prior_commitments()
        self._orient_uncertain_rows()
        strategy = self._build_strategy()
        try:
            with recording_commitments() as made:
                strategy.build()
        finally:
            # Merged even when the build raises part-way: the model may already
            # carry rewritten rows whose validity rests on these commitments.
            store = dict(getattr(self._model, "_ro_commitments", None) or {})
            for name in names:
                # This pass eliminated the parameter; a sign commitment on it
                # was checked against the whole set by _check_prior_commitments.
                store.pop(name, None)
            for name, entry in made.items():
                if name not in names:
                    merge_commitment(store, name, entry)
            # Attached only while non-empty, so a model with no commitment is
            # exactly as before (``discopt.serialize`` refuses one that has any).
            if store:
                self._model._ro_commitments = store
            elif hasattr(self._model, "_ro_commitments"):
                del self._model._ro_commitments

        # Universal soundness guard (RO-2): a correct counterpart eliminates every
        # uncertain parameter. If any survived, the pattern was silently left at
        # nominal — refuse loudly rather than return a non-robust model.
        assert_no_uncertain_params_remain(self._model, names, kind=self.kind)
        self._formulated = True

    def _check_prior_commitments(self) -> None:
        """Refuse a set whose parameter an earlier counterpart committed (#1839 review).

        An earlier ``formulate()`` on this model may have decided a worst case
        from the value of a then-certain Parameter that this pass now treats as
        uncertain. A *value* commitment (the value was baked into the
        counterpart) cannot hold for a whole set and is refused. A *sign*
        commitment (the Parameter stayed live and only its sign was used) is
        valid exactly when every point of the new set has that sign, which is
        checked on the set's component-wise range; the product of two uncertain
        parameters is then robustified exactly, one pass at a time.
        """
        commitments = getattr(self._model, "_ro_commitments", None) or {}
        for unc in self._uncertainty_sets:
            name = unc.parameter.name
            entry = commitments.get(name)
            if entry is None:
                continue
            if entry["kind"] == "value":
                raise ValueError(
                    f"RobustCounterpart: an earlier RobustCounterpart.formulate() on "
                    f"this model baked the value of Parameter {name!r} into its "
                    "counterpart (a mixed-sign or cancelling coefficient, or a "
                    "polyhedral support coefficient), so it cannot now be made "
                    "uncertain: the earlier worst case is not the worst case over "
                    f"the new set. Pass the uncertainty set of {name!r} to the "
                    "same RobustCounterpart as the others, or rebuild the model."
                )
            lo, hi = _component_range(unc)
            sign = int(entry["sign"])
            if not np.all((lo >= 0) if sign > 0 else (hi <= 0)):
                raise ValueError(
                    f"RobustCounterpart: an earlier RobustCounterpart.formulate() on "
                    f"this model chose its worst case from the sign of Parameter "
                    f"{name!r} ({'non-negative' if sign > 0 else 'non-positive'}), "
                    f"but the new uncertainty set lets it range over "
                    f"[{lo.tolist()!r}, {hi.tolist()!r}], which crosses zero; the "
                    "earlier worst case is not the worst case over that set. Pass "
                    f"the uncertainty set of {name!r} to the same RobustCounterpart "
                    "as the others, or rebuild the model."
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

    Rows whose body is scalar, or whose shape cannot be determined statically,
    and rows with no uncertain parameter, pass through unchanged. A body shaped
    by a matmul or a reduction is split too (#1831): ``x + M @ p <= b`` used to
    pass through whole, and the polyhedral counterpart gave its rows one set of
    duals and reported a feasible model infeasible.
    """
    from discopt.modeling.core import Constraint, IndexExpression
    from discopt.ro.formulations._common import _contains_uncertain_param, _shape_of

    out = []
    for con in constraints:
        if not isinstance(con, Constraint) or not _contains_uncertain_param(con.body, names):
            out.append(con)
            continue
        shape = _shape_of(con.body) or ()
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


def _component_range(unc) -> tuple[np.ndarray, np.ndarray]:
    """Component-wise ``(lower, upper)`` of *unc*'s parameter over the whole set.

    Exact for a box and an ellipsoid (``p_bar_i +- rho * ||Sigma^{1/2}[i, :]||``);
    for a polytope, the support LPs of ``+-e_i`` (each widened outward by the
    LP's conservative margin), which raise when the set is unbounded.
    """
    nominal = np.asarray(unc.parameter.value, dtype=np.float64)
    if unc.kind == "box":
        return np.asarray(unc.lower), np.asarray(unc.upper)
    if unc.kind == "ellipsoidal":
        half = unc.rho * np.linalg.norm(unc.Sigma_sqrt, axis=1).reshape(nominal.shape)
        return nominal - half, nominal + half
    if unc.kind == "polyhedral":
        from discopt.ro.formulations.polyhedral import _support_function_lp

        flat = nominal.reshape(-1)
        lo = np.empty_like(flat)
        hi = np.empty_like(flat)
        for i in range(flat.size):
            e = np.zeros(unc.A.shape[1])
            e[i] = 1.0
            lo[i] = flat[i] + _support_function_lp(e, unc.A, unc.b, maximize=False)
            hi[i] = flat[i] + _support_function_lp(e, unc.A, unc.b, maximize=True)
        return lo.reshape(nominal.shape), hi.reshape(nominal.shape)
    raise NotImplementedError(
        f"RobustCounterpart: no component range for an uncertainty set of kind "
        f"{unc.kind!r}; cannot check an earlier sign commitment against it."
    )
