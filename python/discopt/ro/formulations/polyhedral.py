"""Polyhedral-uncertainty robust reformulation via LP duality.

Mathematical Background
-----------------------
For polyhedral uncertainty {xi : A*xi <= b} where xi = p - p_bar is the
perturbation, the worst-case of an expression affine in xi is:

    max_{A*xi <= b} coeff(x)^T xi = min_{lam >= 0, A^T lam = coeff(x)} b^T lam

where coeff(x) is the gradient of the expression w.r.t. xi (which may
involve decision variables).  The dual variables lam become new decision
variables in the reformulated model, and the duality constraints
A^T lam = coeff(x) preserve the coupling between uncertainty components.

This is the correct approach for polyhedral sets including the
budget-of-uncertainty (Bertsimas & Sim 2004), where the budget constraint
couples the components and component-wise bounds alone are insufficient.
"""

from __future__ import annotations

import numpy as np

from discopt.ro.formulations._common import (
    _contains_uncertain_param,
    substitute_param,
)
from discopt.ro.uncertainty import PolyhedralUncertaintySet


class PolyhedralRobustFormulation:
    """Apply polyhedral-uncertainty robust reformulation to a model.

    Uses LP duality to introduce auxiliary dual variables that preserve
    the coupling between uncertainty components.

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
        """Robustify the model in-place via LP duality."""
        from discopt.modeling.core import (
            BinaryOp,
            Constant,
            Constraint,
            Objective,
            ObjectiveSense,
        )

        m = self._model
        unc_map = {u.parameter.name: u for u in self._uncertainty_sets}
        dual_idx = 0

        def _robustify_expr(expr, maximize: bool):
            """Replace uncertain parameters with LP-dual worst-case penalty.

            For each uncertain parameter p with polytope {xi : A*xi <= b}:
            1. Extract the nominal expression (all params at p_bar).
            2. Extract per-component coefficients of xi in the expression.
            3. Introduce dual variables lam >= 0, one per polytope row.
            4. Add equality constraints: A^T lam = coeff_vector(x).
            5. Add penalty: b^T lam (for maximize) or -b^T lam (for minimize).

            Returns (new_expr, list_of_new_constraints).
            """
            nonlocal dual_idx
            new_constraints = []

            # Substitute ALL params with nominal to get the base expression.
            all_nominal = expr
            for pname, unc in unc_map.items():
                all_nominal = substitute_param(all_nominal, pname, Constant(unc.parameter.value))

            result = all_nominal

            for pname, unc in unc_map.items():
                nominal = unc.parameter.value
                nominal_flat = nominal.ravel()
                k = len(nominal_flat)
                A_poly = unc.A  # (n_rows, k)
                b_poly = unc.b  # (n_rows,)
                n_rows = A_poly.shape[0]

                # Isolate this parameter: substitute all OTHER params to nominal.
                expr_isolated = expr
                for other_name, other_unc in unc_map.items():
                    if other_name != pname:
                        expr_isolated = substitute_param(
                            expr_isolated,
                            other_name,
                            Constant(other_unc.parameter.value),
                        )

                # Extract per-component coefficients of p in the expression.
                # coeff_j = (expr at p_j = p_bar_j + 1) - (expr at p_j = p_bar_j)
                base = substitute_param(expr_isolated, pname, Constant(nominal))
                coeff_exprs = []
                for j in range(k):
                    p_plus = nominal_flat.copy()
                    p_plus[j] += 1.0
                    p_plus_val = p_plus.reshape(nominal.shape)
                    unit_j = substitute_param(expr_isolated, pname, Constant(p_plus_val))
                    coeff_j = BinaryOp("-", unit_j, base)
                    coeff_exprs.append(coeff_j)

                # Check if ANY coefficient involves decision variables.
                from discopt.ro.formulations.box import _contains_variable

                has_var_coeff = any(_contains_variable(c) for c in coeff_exprs)

                if not has_var_coeff and not _contains_uncertain_param(expr_isolated, {pname}):
                    # Parameter doesn't appear in this expression.
                    continue

                if not has_var_coeff:
                    # All coefficients are constants: compute the worst-case
                    # value numerically via LP (the old approach, but correct
                    # here since the coefficients don't depend on x).
                    coeff_vals = np.array([_eval_constant_expr(c) for c in coeff_exprs])
                    wc_offset = _support_function_lp(coeff_vals, A_poly, b_poly, maximize)
                    result = BinaryOp("+", result, Constant(np.array(wc_offset)))
                else:
                    # Coefficients involve decision variables: use LP duality.
                    # Introduce lam >= 0 of size n_rows.
                    lam_vars = []
                    lam_ub = float(np.sum(np.abs(b_poly))) + 100.0
                    for i in range(n_rows):
                        lv = m.continuous(
                            f"{self._prefix}_lam{dual_idx}_{i}",
                            lb=0,
                            ub=lam_ub,
                        )
                        lam_vars.append(lv)

                    # Duality constraints: A^T lam = coeff(x)
                    # For each j in 0..k-1: sum_i A[i,j] * lam[i] = coeff_j(x)
                    for j in range(k):
                        # Build sum_i A[i,j] * lam[i]
                        dual_sum = _build_weighted_sum(lam_vars, A_poly[:, j])
                        if maximize:
                            # A^T lam = coeff  =>  coeff - A^T lam == 0
                            eq_body = BinaryOp("-", coeff_exprs[j], dual_sum)
                        else:
                            # Minimizing: worst case is min coeff^T xi,
                            # dual: max_{lam>=0, A^T lam = -coeff} -b^T lam
                            # equivalently: A^T lam = -coeff
                            eq_body = BinaryOp("+", coeff_exprs[j], dual_sum)
                        new_constraints.append(
                            Constraint(
                                body=eq_body,
                                sense="==",
                                rhs=0.0,
                                name=f"{self._prefix}_dual_eq{dual_idx}_{j}",
                            )
                        )

                    # Penalty: b^T lam
                    penalty = _build_weighted_sum(lam_vars, b_poly)
                    if maximize:
                        result = BinaryOp("+", result, penalty)
                    else:
                        result = BinaryOp("-", result, penalty)

                    dual_idx += 1

            return result, new_constraints

        # -- Robustify constraints ----------------------------------------
        new_constraints = []
        for con in m._constraints:
            if not isinstance(con, Constraint):
                new_constraints.append(con)
                continue
            new_body, aux_cons = _robustify_expr(con.body, maximize=True)
            new_constraints.append(
                Constraint(
                    body=new_body,
                    sense=con.sense,
                    rhs=con.rhs,
                    name=con.name,
                )
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


def _eval_constant_expr(expr) -> float:
    """Evaluate a constant expression (no variables) to a float."""
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        UnaryOp,
    )

    if isinstance(expr, Constant):
        return float(np.sum(expr.value))
    if isinstance(expr, BinaryOp):
        lv = _eval_constant_expr(expr.left)
        rv = _eval_constant_expr(expr.right)
        if expr.op == "+":
            return lv + rv
        if expr.op == "-":
            return lv - rv
        if expr.op == "*":
            return lv * rv
        if expr.op == "/":
            return lv / rv
    if isinstance(expr, UnaryOp):
        v = _eval_constant_expr(expr.operand)
        if expr.op == "neg":
            return -v
    return 0.0


def _support_function_lp(coeff: np.ndarray, A: np.ndarray, b: np.ndarray, maximize: bool) -> float:
    """Compute max (or min) coeff^T xi subject to A*xi <= b via LP."""
    try:
        from scipy.optimize import linprog

        k = len(coeff)
        c_obj = -coeff if maximize else coeff
        res = linprog(
            c_obj,
            A_ub=A,
            b_ub=b,
            bounds=[(None, None)] * k,
            method="highs",
        )
        if res.success:
            return float(-res.fun if maximize else res.fun)
    except ImportError:
        pass
    # Conservative fallback: use component-wise bounds.
    return float(np.sum(np.abs(coeff) * np.max(np.abs(b))))


def _build_weighted_sum(variables: list, weights: np.ndarray):
    """Build the expression sum_i weights[i] * variables[i]."""
    from discopt.modeling.core import BinaryOp, Constant

    result = None
    for i, (v, w) in enumerate(zip(variables, weights)):
        if abs(w) < 1e-15:
            continue
        term = BinaryOp("*", Constant(np.array(w)), v)
        if result is None:
            result = term
        else:
            result = BinaryOp("+", result, term)
    if result is None:
        return Constant(np.array(0.0))
    return result
