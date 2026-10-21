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
    eval_constant_expr,
    refuse_non_affine,
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
                k = len(nominal_flat)  # true parameter dimension
                A_poly = unc.A  # (n_rows, n_lifted)
                b_poly = unc.b  # (n_rows,)
                n_rows = A_poly.shape[0]
                # The polytope may live in a lifted space z = (ξ, aux): its first
                # `k` columns are the parameter perturbation ξ (which carry the real
                # coefficients coeff_j(x)); any remaining columns are auxiliary lift
                # variables with zero objective coefficient. Dualizing over the full
                # lifted A is exact — this is how the compact Bertsimas–Sim budget
                # set (uncertainty.budget_uncertainty_set) is represented (RO-3).
                n_lifted = A_poly.shape[1]
                if n_lifted < k:  # pragma: no cover - guarded at construction
                    raise ValueError(f"polytope for {pname!r} has {n_lifted} columns < {k} params")

                # Isolate this parameter: substitute all OTHER params to nominal.
                expr_isolated = expr
                for other_name, other_unc in unc_map.items():
                    if other_name != pname:
                        expr_isolated = substitute_param(
                            expr_isolated,
                            other_name,
                            Constant(other_unc.parameter.value),
                        )

                # A row that does not contain this parameter has no uncertain
                # data: it gets no counterpart. Testing the *coefficients* for
                # variables instead is wrong — coeff_j is built as the
                # syntactic difference ``unit_j - base``, which mentions every
                # variable of the row even when they cancel, so a parameter-free
                # row was given a spurious dual penalty -b^T lam. On an equality
                # row that penalty is a free slack (``sum(w) - 1 + e^T lam = 0``
                # admits w = 0), producing a "robust" optimum below the nominal
                # one (#1611 X-30a).
                if not _contains_uncertain_param(expr_isolated, {pname}):
                    continue

                # The finite difference below is the coefficient of xi_j only for
                # a row affine in p; for ``exp(p)`` it is a secant (#1841).
                refuse_non_affine(expr_isolated, {pname}, kind="polyhedral")

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

                if not has_var_coeff:
                    # All coefficients are constants: compute the worst-case
                    # value numerically via LP (the old approach, but correct
                    # here since the coefficients don't depend on x).
                    coeff_vals = np.zeros(n_lifted)
                    coeff_vals[:k] = [_eval_constant_expr(c) for c in coeff_exprs]
                    wc_offset = _support_function_lp(coeff_vals, A_poly, b_poly, maximize)
                    result = BinaryOp("+", result, Constant(np.array(wc_offset)))
                else:
                    # Coefficients involve decision variables: use LP duality.
                    # Introduce lam >= 0 of size n_rows.
                    # The duals are left unbounded above. A dual of the
                    # polytope equals (a combination of) the row's coefficients
                    # coeff_j(x) -- decision-variable expressions in the
                    # model's units -- so no constant cap is valid: the former
                    # ``sum|b| + 100`` cap cut off the true counterpart and made
                    # the same model infeasible once rescaled (#1611 X-30b).
                    # lam enters only linearly, so an unbounded dual is no
                    # obstacle to the solve.
                    lam_vars = []
                    for i in range(n_rows):
                        lv = m.continuous(
                            f"{self._prefix}_lam{dual_idx}_{i}",
                            lb=0,
                        )
                        # #1755: unbounded above by construction (see above), so
                        # the large-bounds warning -- whose remedy is "declare
                        # tighter bounds" -- does not name it (#1618 X-29b).
                        if not hasattr(m, "_free_by_construction"):
                            m._free_by_construction = set()
                        m._free_by_construction.add(lv.name)
                        lam_vars.append(lv)

                    # Duality constraints: A^T lam = coeff(z), over the full
                    # lifted space z=(ξ,aux). Columns 0..k-1 carry coeff_j(x);
                    # lifted columns k..n_lifted-1 have zero objective coefficient
                    # (aux variables), so their duality RHS is 0.
                    for j in range(n_lifted):
                        coeff_j_expr = coeff_exprs[j] if j < k else Constant(np.array(0.0))
                        # Build sum_i A[i,j] * lam[i]
                        dual_sum = _build_weighted_sum(lam_vars, A_poly[:, j])
                        if maximize:
                            # A^T lam = coeff  =>  coeff - A^T lam == 0
                            eq_body = BinaryOp("-", coeff_j_expr, dual_sum)
                        else:
                            # Minimizing: worst case is min coeff^T xi,
                            # dual: max_{lam>=0, A^T lam = -coeff} -b^T lam
                            # equivalently: A^T lam = -coeff
                            eq_body = BinaryOp("+", coeff_j_expr, dual_sum)
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
    """Evaluate the constant coefficient of one parameter component to a float.

    The row has already been split into scalar rows, so the coefficient is a
    scalar. A non-scalar value means the row is vector-valued and has no single
    coefficient; refuse instead of folding it. The former evaluator returned
    ``float(np.sum(value))`` for a vector constant and ``0.0`` for every node it
    did not recognise (``SumExpression``, ``IndexExpression``, ...), so
    ``sum(c * p) <= b`` got a zero coefficient and was never robustified
    (#1831 audit).
    """
    value = eval_constant_expr(expr)
    if value.size != 1:
        raise NotImplementedError(
            "Polyhedral robust counterpart: the coefficient of an uncertain "
            f"parameter component has shape {value.shape}, not a scalar -- the row "
            "is vector-valued with no statically known shape. Index it into scalar "
            "rows. Refusing to fold the coefficient vector into one number."
        )
    return float(value.reshape(()))


def _support_function_lp(coeff: np.ndarray, A: np.ndarray, b: np.ndarray, maximize: bool) -> float:
    """Compute max (or min) coeff^T xi subject to A*xi <= b via the POUNCE LP.

    Pure-Rust (issue #356 — no SciPy/HiGHS). POUNCE is an interior-point method,
    so a small magnitude-scaled margin is applied in the *conservative* direction
    (larger support on a max, smaller on a min) to keep the robust-optimization
    constraint it feeds an outer (never optimistic) approximation.
    """
    from discopt.solvers import SolveStatus
    from discopt.solvers.lp_pounce import solve_lp

    k = len(coeff)
    c_obj = -coeff if maximize else coeff
    # Free decision variables (the uncertainty ξ), bounded only by A ξ <= b.
    res = solve_lp(c_obj, A_ub=A, b_ub=b, bounds=[(-np.inf, np.inf)] * k)
    if res.status == SolveStatus.OPTIMAL and res.objective is not None:
        val = float(-res.objective if maximize else res.objective)
        margin = 1e-7 * (1.0 + abs(val))
        return val + margin if maximize else val - margin
    # #1831 audit: this used to fall back to ``sum(|coeff|) * max(|b|)``, which
    # bounds the support only when the polytope lies inside the box
    # ``|xi_j| <= max|b|`` -- false for a scaled or unbounded polytope -- and so
    # returned a finite, optimistic worst case exactly when the LP said the
    # support was not a finite optimum. Refuse instead.
    raise ValueError(
        "Polyhedral robust counterpart: the support LP "
        f"{'max' if maximize else 'min'} coeff^T xi over {{A xi <= b}} returned "
        f"status {res.status!r}, not a finite optimum. If the uncertainty polytope "
        "is unbounded in this direction the robust row has no finite worst case. "
        "Refusing to substitute a guessed bound."
    )


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
