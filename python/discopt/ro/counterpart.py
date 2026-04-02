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
:class:`~discopt.nn.formulations.base.NNFormulation`: construct, then call
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

        strategy = self._build_strategy()
        strategy.build()
        self._formulated = True

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
