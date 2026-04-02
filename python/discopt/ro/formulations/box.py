"""Box-uncertainty robust reformulation.

Mathematical Background
-----------------------
For a constraint body expression g(x, p) <= 0 with box uncertainty
{p : |p_j - p_bar_j| <= delta_j}, the robust counterpart requires:

    max_{p in U} g(x, p) <= 0

When g is affine in p, the worst-case maximisation has a closed form that
depends on the sign of dg/dp_j:

* dg/dp_j > 0  ->  worst-case p_j = p_bar_j + delta_j  (upper bound)
* dg/dp_j < 0  ->  worst-case p_j = p_bar_j - delta_j  (lower bound)

This module performs *sign-tracking* while traversing the expression tree to
determine the correct bound for each parameter occurrence.  The sign starts
at +1 at the tree root and flips through ``-`` (subtraction) and unary
negation nodes.  Multiplication by a negative constant also flips the sign.

For objectives the sign convention is adapted to the sense:
- MINIMIZE f(p): worst case is the *largest* f, so parameter terms appear
  with their natural sign.
- MAXIMIZE f(p): worst case is the *smallest* f, so signs are flipped.

Only constraints of the form ``body <= 0`` are supported (which is the
normalized form used internally).
"""

from __future__ import annotations

from discopt.ro.formulations._common import sign_tracking_substitute
from discopt.ro.uncertainty import BoxUncertaintySet


class BoxRobustFormulation:
    """Apply box-uncertainty robust reformulation to a model.

    Parameters
    ----------
    model : discopt.Model
        The model to robustify (modified in-place).
    uncertainty_sets : list[BoxUncertaintySet]
        Uncertainty sets, one per uncertain parameter.
    prefix : str
        Name prefix for any auxiliary variables / constraints added.
    """

    def __init__(
        self, model, uncertainty_sets: list[BoxUncertaintySet], prefix: str = "ro"
    ) -> None:
        self._model = model
        self._uncertainty_sets = uncertainty_sets
        self._prefix = prefix

    def build(self) -> None:
        """Robustify the model in-place.

        For each constraint replaces uncertain parameters with worst-case
        values determined by sign-tracking through the expression tree.
        For the objective, applies the appropriate worst-case substitution
        based on the optimisation sense.
        """
        # Precompute worst-case bounds from box sets.
        param_names = {u.parameter.name for u in self._uncertainty_sets}
        wc_lower = {u.parameter.name: u.lower for u in self._uncertainty_sets}
        wc_upper = {u.parameter.name: u.upper for u in self._uncertainty_sets}
        m = self._model

        # -- Robustify constraints ------------------------------------------------
        # Constraint stored as body <= 0; worst-case = maximise body over U.
        # Only plain Constraint objects carry .body / .sense / .rhs; other
        # types (_IndicatorConstraint, _DisjunctiveConstraint, _SOSConstraint,
        # _LogicalConstraint) are passed through unchanged.
        from discopt.modeling.core import Constraint

        new_constraints = []
        for con in m._constraints:
            if not isinstance(con, Constraint):
                new_constraints.append(con)
                continue
            new_expr = sign_tracking_substitute(
                con.body, wc_lower, wc_upper, param_names, maximize=True, sign=+1
            )
            new_constraints.append(
                Constraint(body=new_expr, sense=con.sense, rhs=con.rhs, name=con.name)
            )
        m._constraints = new_constraints

        # -- Robustify objective --------------------------------------------------
        obj = m._objective
        if obj is None:
            return
        from discopt.modeling.core import Objective, ObjectiveSense

        # For MINIMIZE: worst case is maximum objective -> parameters with positive
        # contribution use upper bound, negative use lower.
        # For MAXIMIZE: worst case is minimum objective -> flip convention.
        maximize_obj_wc = obj.sense == ObjectiveSense.MINIMIZE
        new_expr = sign_tracking_substitute(
            obj.expression,
            wc_lower,
            wc_upper,
            param_names,
            maximize=maximize_obj_wc,
            sign=+1,
        )
        m._objective = Objective(expression=new_expr, sense=obj.sense)
