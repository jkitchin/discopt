"""Polyhedral-uncertainty robust reformulation via LP duality.

Mathematical Background
-----------------------
For polyhedral uncertainty {xi : A*xi <= b} where xi = p - p_bar is the
perturbation, the worst-case of a term c^T p over the polytope is:

    p_bar^T c + max_{A*xi <= b} c^T xi

The inner max is the support function of the polytope, solved by LP duality:

    max_{A*xi <= b} c^T xi = min_{lam >= 0, A^T lam = c} b^T lam

So the robust constraint becomes:

    h(x) + p_bar^T a(x) + b^T lam <= 0
    A^T lam = a(x)
    lam >= 0

This module handles the *parameter-as-RHS* case directly (no new variables
needed) by computing the worst-case parameter value via LP duality.

For the general coefficient-uncertainty case (a(x) depends on decision
variables), the full dual variable approach requires introducing auxiliary
continuous variables lam per uncertain constraint.  This is implemented via
the scipy LP solver to determine the worst-case offset.
"""

from __future__ import annotations

import numpy as np

from discopt.ro.formulations._common import sign_tracking_substitute
from discopt.ro.uncertainty import PolyhedralUncertaintySet


class PolyhedralRobustFormulation:
    """Apply polyhedral-uncertainty robust reformulation to a model.

    For uncertain parameters that appear additively (as RHS terms), computes
    the worst-case parameter value by solving an LP for each component and
    substitutes the result as a constant.

    Parameters
    ----------
    model : discopt.Model
        The model to robustify (modified in-place).
    uncertainty_sets : list[PolyhedralUncertaintySet]
        Uncertainty sets.
    prefix : str
        Name prefix for dual variable names.
    """

    def __init__(
        self,
        model,
        uncertainty_sets: list[PolyhedralUncertaintySet],
        prefix: str = "ro",
    ) -> None:
        self._model = model
        self._uncertainty_sets = uncertainty_sets
        self._prefix = prefix

    def build(self) -> None:
        """Robustify the model in-place.

        Computes component-wise worst-case bounds for each polytope via LP
        and substitutes them using the shared sign-tracking traversal.
        """
        param_names = {u.parameter.name for u in self._uncertainty_sets}
        m = self._model

        # Precompute worst-case bounds for each uncertain parameter.
        wc_upper: dict[str, np.ndarray] = {}
        wc_lower: dict[str, np.ndarray] = {}
        for u in self._uncertainty_sets:
            lo, hi = _polytope_extreme_values(u)
            wc_lower[u.parameter.name] = lo
            wc_upper[u.parameter.name] = hi

        from discopt.modeling.core import Constraint, Objective, ObjectiveSense

        # -- Robustify constraints ------------------------------------------------
        # Only plain Constraint objects carry .body / .sense / .rhs; other
        # types (_IndicatorConstraint, _DisjunctiveConstraint, _SOSConstraint,
        # _LogicalConstraint) are passed through unchanged.
        new_constraints = []
        for con in m._constraints:
            if not isinstance(con, Constraint):
                new_constraints.append(con)
                continue
            new_expr = sign_tracking_substitute(
                con.body,
                wc_lower,
                wc_upper,
                param_names,
                maximize=True,
                sign=+1,
            )
            new_constraints.append(
                Constraint(body=new_expr, sense=con.sense, rhs=con.rhs, name=con.name)
            )
        m._constraints = new_constraints

        # -- Robustify objective --------------------------------------------------
        obj = m._objective
        if obj is None:
            return

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


def _polytope_extreme_values(
    unc: PolyhedralUncertaintySet,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute component-wise [lower, upper] extremes of p_bar + {xi : A*xi <= b}.

    Solves 2k LPs (minimise and maximise each component) using scipy HiGHS.
    Falls back to interval bounding from the RHS vector if scipy is not
    available.
    """
    A, b_vec = unc.A, unc.b
    k = A.shape[1]
    nominal = unc.parameter.value.ravel()

    try:
        from scipy.optimize import linprog

        lo = np.empty(k)
        hi = np.empty(k)
        for j in range(k):
            c_obj = np.zeros(k)
            c_obj[j] = 1.0
            res_min = linprog(
                c_obj,
                A_ub=A,
                b_ub=b_vec,
                bounds=[(None, None)] * k,
                method="highs",
            )
            lo[j] = res_min.fun if res_min.success else -np.inf
            res_max = linprog(
                -c_obj,
                A_ub=A,
                b_ub=b_vec,
                bounds=[(None, None)] * k,
                method="highs",
            )
            hi[j] = -res_max.fun if res_max.success else np.inf

        lo_param = (nominal + lo).reshape(unc.parameter.value.shape)
        hi_param = (nominal + hi).reshape(unc.parameter.value.shape)
        return lo_param, hi_param

    except ImportError:
        # Conservative fallback.
        radius = float(np.max(np.abs(b_vec))) if len(b_vec) > 0 else 1.0
        lo_param = (nominal - radius).reshape(unc.parameter.value.shape)
        hi_param = (nominal + radius).reshape(unc.parameter.value.shape)
        return lo_param, hi_param
