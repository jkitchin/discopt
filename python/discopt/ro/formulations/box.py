"""Box-uncertainty robust reformulation.

Mathematical Background
-----------------------
For a constraint body expression g(x, p) <= 0 with box uncertainty
{p : |p_j - p_bar_j| <= delta_j}, the robust counterpart requires:

    max_{p in U} g(x, p) <= 0

When g is affine in p and the coefficient of p is a known constant, the
worst-case value depends on the sign of the coefficient (sign-tracking).

When g contains *bilinear* terms (decision variable * uncertain parameter),
as arises from affine decision rules, the coefficient of p involves decision
variables whose sign is not known a priori.  In this case the correct
reformulation uses absolute-value linearization:

    max_{|xi|<=delta} coeff(x) * xi = delta * |coeff(x)|

The absolute value is linearized via auxiliary variables:
    |coeff| <= t,  with  coeff <= t  and  -coeff <= t,  t >= 0.
"""

from __future__ import annotations

import numpy as np

from discopt.ro.formulations._common import (
    _contains_uncertain_param,
    sign_tracking_substitute,
    substitute_param,
)
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
        """Robustify the model in-place."""
        m = self._model
        param_names = {u.parameter.name for u in self._uncertainty_sets}

        # Check whether any expression has bilinear terms (variable * parameter).
        has_bilinear = self._detect_bilinear(m, param_names)

        if has_bilinear:
            self._build_with_linearization(m, param_names)
        else:
            self._build_sign_tracking(m, param_names)

    # ------------------------------------------------------------------
    # Fast path: no bilinear terms, use sign-tracking substitution
    # ------------------------------------------------------------------

    def _build_sign_tracking(self, m, param_names):
        """Robustify via sign-tracking (original approach, no bilinear terms)."""
        wc_lower = {u.parameter.name: u.lower for u in self._uncertainty_sets}
        wc_upper = {u.parameter.name: u.upper for u in self._uncertainty_sets}

        from discopt.modeling.core import Constraint, Objective, ObjectiveSense

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

    # ------------------------------------------------------------------
    # Bilinear path: coefficient extraction + absolute-value linearization
    # ------------------------------------------------------------------

    def _build_with_linearization(self, m, param_names):
        """Robustify with absolute-value linearization for bilinear terms.

        For each uncertain parameter p with box half-width delta, and each
        expression that is affine in p:

            expr = f(x) + coeff(x) * (p - p_bar)

        the worst case over |p - p_bar| <= delta is:

            f(x) + delta * |coeff(x)|

        We linearize |coeff| by introducing t >= 0 with coeff <= t, -coeff <= t.
        """
        from discopt.modeling.core import (
            BinaryOp,
            Constant,
            Constraint,
            Objective,
            ObjectiveSense,
        )

        unc_map = {u.parameter.name: u for u in self._uncertainty_sets}
        aux_idx = 0

        def _robustify_expr(expr, maximize: bool):
            """Replace uncertain parameters in expr with worst-case penalty.

            Returns (new_expr, list_of_new_constraints).

            For each uncertain parameter p_k, extracts the linear coefficient
            of p_k in the expression (evaluated with all OTHER parameters at
            nominal).  If the coefficient is constant, uses sign-tracking.
            If it involves decision variables, uses absolute-value linearization.
            """
            nonlocal aux_idx
            new_constraints = []

            # Step 1: substitute ALL params with nominal to get the base expression
            all_nominal = expr
            for pname, unc in unc_map.items():
                all_nominal = substitute_param(all_nominal, pname, Constant(unc.parameter.value))

            # Step 2: for each param, extract coefficient and add penalty
            result = all_nominal
            for pname, unc in unc_map.items():
                nominal = unc.parameter.value
                delta = unc.delta

                # Compute coefficient of p_k by substituting all params to nominal
                # except p_k, then evaluating at p_k = p_bar and p_k = p_bar + 1.
                expr_at_nom = expr
                for other_name, other_unc in unc_map.items():
                    if other_name != pname:
                        expr_at_nom = substitute_param(
                            expr_at_nom, other_name, Constant(other_unc.parameter.value)
                        )

                base = substitute_param(expr_at_nom, pname, Constant(nominal))
                unit = substitute_param(
                    expr_at_nom, pname, Constant(nominal + np.ones_like(nominal))
                )
                coeff_expr = BinaryOp("-", unit, base)

                if not _contains_variable(coeff_expr):
                    # Non-bilinear: use sign-tracking for this param only.
                    # Apply to result which already has all params at nominal.
                    # Re-derive the non-bilinear worst-case contribution.
                    wc_lower = {pname: unc.lower}
                    wc_upper = {pname: unc.upper}
                    # Worst case of constant_coeff * p over [p_bar-delta, p_bar+delta].
                    # Contribution = coeff * (p_wc - p_bar) where p_wc depends on sign.
                    wc_expr = sign_tracking_substitute(
                        expr_at_nom,
                        wc_lower,
                        wc_upper,
                        {pname},
                        maximize=maximize,
                        sign=+1,
                    )
                    # Penalty = (worst-case expression) - (nominal expression)
                    wc_penalty = BinaryOp("-", wc_expr, base)
                    result = BinaryOp("+", result, wc_penalty)
                else:
                    # Bilinear: absolute-value linearization.
                    t_ub = _estimate_coeff_bound(m)
                    t_var = m.continuous(
                        f"{self._prefix}_abs{aux_idx}",
                        lb=0,
                        ub=t_ub,
                    )
                    aux_idx += 1

                    # |coeff| <= t:  coeff - t <= 0  and  -coeff - t <= 0
                    new_constraints.append(
                        Constraint(
                            body=BinaryOp("-", coeff_expr, t_var),
                            sense="<=",
                            rhs=0.0,
                            name=f"{self._prefix}_abs_pos{aux_idx - 1}",
                        )
                    )
                    new_constraints.append(
                        Constraint(
                            body=BinaryOp(
                                "-", BinaryOp("*", Constant(np.array(-1.0)), coeff_expr), t_var
                            ),
                            sense="<=",
                            rhs=0.0,
                            name=f"{self._prefix}_abs_neg{aux_idx - 1}",
                        )
                    )

                    # Penalty: delta * t
                    penalty = BinaryOp("*", Constant(delta), t_var)
                    if maximize:
                        result = BinaryOp("+", result, penalty)
                    else:
                        result = BinaryOp("-", result, penalty)

            return result, new_constraints

        # -- Robustify constraints ----------------------------------------
        new_constraints = []
        for con in m._constraints:
            if not isinstance(con, Constraint):
                new_constraints.append(con)
                continue
            new_body, aux_cons = _robustify_expr(con.body, maximize=True)
            new_constraints.append(
                Constraint(body=new_body, sense=con.sense, rhs=con.rhs, name=con.name)
            )
            new_constraints.extend(aux_cons)
        m._constraints = new_constraints

        # -- Robustify objective ------------------------------------------
        obj = m._objective
        if obj is None:
            return
        maximize_obj_wc = obj.sense == ObjectiveSense.MINIMIZE
        new_expr, aux_cons = _robustify_expr(obj.expression, maximize=maximize_obj_wc)
        m._objective = Objective(expression=new_expr, sense=obj.sense)
        m._constraints.extend(aux_cons)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_bilinear(m, param_names):
        """Return True if any constraint or objective has bilinear var*param terms."""
        from discopt.modeling.core import Constraint

        exprs = []
        for con in m._constraints:
            if isinstance(con, Constraint):
                exprs.append(con.body)
        if m._objective is not None:
            exprs.append(m._objective.expression)

        for expr in exprs:
            if _has_bilinear_param_var(expr, param_names):
                return True
        return False


def _estimate_coeff_bound(m) -> float:
    """Estimate an upper bound on the magnitude of any linear coefficient.

    Uses the maximum absolute variable bound in the model as a conservative
    estimate.  This is used to set finite bounds on absolute-value auxiliary
    variables so the IPM solver converges reliably.
    """
    max_bound = 1.0
    for v in m._variables:
        if v.ub is not None:
            max_bound = max(max_bound, float(np.max(np.abs(v.ub))))
        if v.lb is not None:
            max_bound = max(max_bound, float(np.max(np.abs(v.lb))))
    # Cap at a reasonable value to avoid numerical issues
    return min(max_bound + 1.0, 1e8)


def _contains_variable(expr) -> bool:
    """Check whether expr contains any Variable node."""
    from discopt.modeling.core import (
        BinaryOp,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        SumExpression,
        SumOverExpression,
        UnaryOp,
        Variable,
    )

    if isinstance(expr, Variable):
        return True
    if isinstance(expr, (BinaryOp, MatMulExpression)):
        return _contains_variable(expr.left) or _contains_variable(expr.right)
    if isinstance(expr, UnaryOp):
        return _contains_variable(expr.operand)
    if isinstance(expr, FunctionCall):
        return any(_contains_variable(a) for a in expr.args)
    if isinstance(expr, IndexExpression):
        return _contains_variable(expr.base)
    if isinstance(expr, SumExpression):
        return _contains_variable(expr.operand)
    if isinstance(expr, SumOverExpression):
        return any(_contains_variable(t) for t in expr.terms)
    return False


def _has_bilinear_param_var(expr, param_names: set[str]) -> bool:
    """Check whether expr has a true bilinear product: Variable * f(Parameter).

    A "true bilinear" term is one where a decision-variable expression
    multiplies an expression containing an uncertain parameter, AND that
    expression is not just the parameter itself (which sign-tracking
    handles correctly).  The canonical case is ``Y * (p - p_bar)`` from
    an affine decision rule.

    Note: ``Parameter * Variable`` (e.g. ``c * x``) is NOT bilinear in this
    sense because the parameter IS the coefficient and sign-tracking can
    determine its worst-case value directly.
    """
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    if isinstance(expr, BinaryOp):
        if expr.op == "*":
            if not isinstance(expr.left, Constant) and not isinstance(expr.right, Constant):
                # Check for the specific pattern: one side is a Variable (or
                # expression with variables but no params), the other side
                # is an expression containing BOTH variables AND parameters.
                # This catches Y * (p - p_bar) but not Parameter * Variable.
                l_param = _contains_uncertain_param(expr.left, param_names)
                r_param = _contains_uncertain_param(expr.right, param_names)
                l_var = _contains_variable(expr.left)
                r_var = _contains_variable(expr.right)
                # True bilinear: both sides have at least one of {var, param}
                # AND they're mixed (not just param*var at the leaf level).
                if l_var and r_param and r_var:
                    return True  # left has var, right has both var and param
                if r_var and l_param and l_var:
                    return True  # right has var, left has both var and param
                # Also: variable-only * param-containing (but not a bare Parameter)
                if l_var and not l_param and r_param and not isinstance(expr.right, Parameter):
                    return True
                if r_var and not r_param and l_param and not isinstance(expr.left, Parameter):
                    return True
        return _has_bilinear_param_var(expr.left, param_names) or _has_bilinear_param_var(
            expr.right, param_names
        )
    if isinstance(expr, (MatMulExpression,)):
        return _has_bilinear_param_var(expr.left, param_names) or _has_bilinear_param_var(
            expr.right, param_names
        )
    if isinstance(expr, UnaryOp):
        return _has_bilinear_param_var(expr.operand, param_names)
    if isinstance(expr, FunctionCall):
        return any(_has_bilinear_param_var(a, param_names) for a in expr.args)
    if isinstance(expr, IndexExpression):
        return _has_bilinear_param_var(expr.base, param_names)
    if isinstance(expr, SumExpression):
        return _has_bilinear_param_var(expr.operand, param_names)
    if isinstance(expr, SumOverExpression):
        return any(_has_bilinear_param_var(t, param_names) for t in expr.terms)
    return False
